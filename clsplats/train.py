"""
CL-Splats Training Entry Point.

This script provides the main entry point for training CL-Splats models
using Hydra for configuration management.
"""

import os
import re
import shutil
import sys
import datetime
import json
import random
from argparse import Namespace

import hydra
from loguru import logger
import omegaconf
import wandb
import numpy as np
import torch

from clsplats.eval import evaluate as run_eval
from clsplats.models.model_factory import create_trainer
from clsplats.dataset import CLSplatsDataset


def set_random_seed(seed: int = 0) -> None:
    """Set global random seeds for reproducible training runs."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _sanitize_filename_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return token if token else "unknown"


def _derive_dataset_name(cfg: omegaconf.DictConfig) -> str:
    dataset_cfg = cfg.get("dataset", {})
    raw_path = dataset_cfg.get("path", "dataset")
    normalized = os.path.normpath(str(raw_path))
    name = os.path.basename(normalized)
    if not name:
        name = os.path.basename(os.path.dirname(normalized))
    return _sanitize_filename_token(name or "dataset")


def _derive_model_name(cfg: omegaconf.DictConfig) -> str:
    model_cfg = cfg.get("model", {})
    method_name = model_cfg.get("name", "model")
    representation_name = model_cfg.get("representation", None)
    if representation_name:
        return _sanitize_filename_token(f"{method_name}_{representation_name}")
    return _sanitize_filename_token(str(method_name))


def _resolve_experiment_tag(cfg: omegaconf.DictConfig) -> str:
    runtime_cfg = cfg.get("runtime", {})
    existing_tag = runtime_cfg.get("experiment_tag", None)
    if existing_tag:
        return _sanitize_filename_token(str(existing_tag))

    generated_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with omegaconf.open_dict(cfg):
        runtime_cfg = cfg.get("runtime", {})
        runtime_cfg["experiment_tag"] = generated_tag
        cfg["runtime"] = runtime_cfg
    return generated_tag


def _build_experiment_log_stem(cfg: omegaconf.DictConfig) -> str:
    model_name = _derive_model_name(cfg)
    dataset_name = _derive_dataset_name(cfg)
    experiment_tag = _resolve_experiment_tag(cfg)
    return f"{model_name}_{dataset_name}_{experiment_tag}"


def _build_training_log_path(cfg: omegaconf.DictConfig) -> str:
    output_dir = cfg.get("output_dir", "outputs")
    training_log_dir = os.path.join(output_dir, "training_logs")
    log_stem = _build_experiment_log_stem(cfg)
    return os.path.join(training_log_dir, f"{log_stem}.log")


def _build_eval_log_path(cfg: omegaconf.DictConfig) -> str:
    output_dir = cfg.get("output_dir", "outputs")
    eval_log_dir = os.path.join(output_dir, "eval")
    log_stem = _build_experiment_log_stem(cfg)
    return os.path.join(eval_log_dir, f"{log_stem}.log")


def _append_eval_log_record(eval_log_path: str, record: dict) -> None:
    try:
        eval_log_dir = os.path.dirname(eval_log_path)
        if eval_log_dir:
            os.makedirs(eval_log_dir, exist_ok=True)

        payload = dict(record)
        payload["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")

        with open(eval_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        logger.warning(f"Failed to append eval log record to {eval_log_path}")


def _cleanup_temp_eval_checkpoint(checkpoint_path: str, eval_log_dir: str) -> None:
    checkpoint_abs = os.path.abspath(checkpoint_path)
    checkpoint_parent_abs = os.path.abspath(os.path.dirname(checkpoint_abs))
    eval_log_dir_abs = os.path.abspath(eval_log_dir)

    if os.path.normcase(checkpoint_parent_abs) != os.path.normcase(eval_log_dir_abs):
        return

    for path in (checkpoint_abs, checkpoint_abs.replace(".pt", ".ply")):
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                logger.warning(f"Failed to remove temporary eval artifact: {path}")


def _run_eval_up_to_timestep(
    cfg: omegaconf.DictConfig,
    train_timestep: int,
    checkpoint_path: str,
    eval_log_path: str,
    force_random_split: bool = False,
) -> dict:
    dataset_cfg = cfg.get("dataset", {})
    data_path = dataset_cfg.get("path", None)

    if data_path is None:
        logger.warning(
            f"[Eval] Skip evaluation for train_timestep={train_timestep}: "
            "dataset.path is not set."
        )
        _append_eval_log_record(
            eval_log_path,
            {
                "status": "skipped",
                "train_timestep": train_timestep,
                "eval_timestep": None,
                "checkpoint_path": os.path.abspath(checkpoint_path),
                "reason": "dataset.path is not set",
            },
        )
        return {
            "status": "skipped",
            "train_timestep": int(train_timestep),
            "avg_metrics": None,
            "per_timestep_results": [],
            "num_eval_timesteps_succeeded": 0,
            "num_eval_timesteps_failed": 0,
        }

    resolved_dataset_path = os.path.abspath(os.path.expanduser(str(data_path)))
    allow_resize_mismatch = bool(dataset_cfg.get("allow_resize_mismatch", False))
    prefer_undist = bool(dataset_cfg.get("prefer_undist", True))
    white_background = bool(
        cfg.get("white_background", dataset_cfg.get("white_background", False))
    )

    eval_max_timestep = max(int(train_timestep), 0)
    logger.info(
        f"[Eval] Train timestep {train_timestep}: evaluate timesteps 0..{eval_max_timestep}"
    )

    success_count = 0
    failed_count = 0
    l1_values: list[float] = []
    ssim_values: list[float] = []
    psnr_values: list[float] = []
    per_timestep_results: list[dict] = []

    for eval_timestep in range(eval_max_timestep + 1):
        eval_args = Namespace(
            dataset_path=resolved_dataset_path,
            checkpoint=os.path.abspath(checkpoint_path),
            timestep=eval_timestep,
            split_seed=dataset_cfg.get("split_seed", 0),
            force_random_split=bool(force_random_split),
            resolution_scale=dataset_cfg.get("resolution", 1.0),
            white_background=white_background,
            prefer_undist=prefer_undist,
            prefer_dist=not prefer_undist,
            device="cuda" if torch.cuda.is_available() else "cpu",
            allow_resize_mismatch=allow_resize_mismatch,
            output_json=None,
            verbose=False,
            enable_legacy_idea3_hash_residual=False,
            eval_train_split=bool(cfg.get("eval_train_split", False)),
        )

        try:
            result = run_eval(eval_args)
            metrics = result.get("metrics", {})
            l1_value = float(metrics.get("l1", 0.0))
            ssim_value = float(metrics.get("ssim", 0.0))
            psnr_value = float(metrics.get("psnr", 0.0))
            temporal_source = str(result.get("temporal_source", "unknown"))
            temporal_visible = result.get("temporal_visible_gaussians", None)
            temporal_total = result.get("temporal_total_gaussians", None)
            train_split_metrics = result.get("train_split_metrics", None)
            residual_diag = {
                "color_residual_cnn_enabled": bool(
                    result.get("color_residual_cnn_enabled", False)
                ),
                "color_residual_cnn_mode": str(
                    result.get("color_residual_cnn_mode", "none")
                ),
                "color_residual_cnn_reason": str(
                    result.get("color_residual_cnn_reason", "not_found")
                ),
                "color_residual_cnn_update_steps": int(
                    result.get("color_residual_cnn_update_steps", 0) or 0
                ),
                "global_hash_state_key": result.get("global_hash_state_key", None),
                "global_hash_cfg_resolved": bool(
                    result.get("global_hash_cfg_resolved", False)
                ),
                "global_hash_params_ckpt_numel": result.get(
                    "global_hash_params_ckpt_numel", None
                ),
                "global_hash_params_runtime_numel": result.get(
                    "global_hash_params_runtime_numel", None
                ),
            }
            train_split_terms = ""
            if isinstance(train_split_metrics, dict) and "l1" in train_split_metrics:
                train_split_terms = f", train_split_l1={float(train_split_metrics['l1']):.6f}"

            logger.info(
                f"[Eval] Train t={train_timestep}, Eval t={eval_timestep}: "
                f"L1={l1_value:.6f}, SSIM={ssim_value:.6f}, PSNR={psnr_value:.4f}, "
                f"source={temporal_source}, visible={temporal_visible}/{temporal_total}"
                f"{train_split_terms}"
            )
            logger.info(
                f"[EvalDiag] Train t={train_timestep}, Eval t={eval_timestep}: "
                f"color_residual_cnn_enabled={residual_diag['color_residual_cnn_enabled']}, "
                f"color_residual_mode={residual_diag['color_residual_cnn_mode']}, "
                f"color_residual_reason={residual_diag['color_residual_cnn_reason']}, "
                f"color_residual_steps={residual_diag['color_residual_cnn_update_steps']}, "
                f"global_hash_state_key={residual_diag['global_hash_state_key']}, "
                f"global_hash_cfg_resolved={residual_diag['global_hash_cfg_resolved']}, "
                f"global_hash_ckpt_numel={residual_diag['global_hash_params_ckpt_numel']}, "
                f"global_hash_runtime_numel={residual_diag['global_hash_params_runtime_numel']}"
            )
            _append_eval_log_record(
                eval_log_path,
                {
                    "status": "success",
                    "train_timestep": train_timestep,
                    "eval_timestep": eval_timestep,
                    "checkpoint_path": result.get(
                        "checkpoint_path", os.path.abspath(checkpoint_path)
                    ),
                    "num_test_cameras": result.get("num_test_cameras"),
                    "num_test_images_loaded": result.get("num_test_images_loaded"),
                    "num_evaluated_views": result.get("num_evaluated_views"),
                    "num_skipped_shape_mismatch": result.get(
                        "num_skipped_shape_mismatch"
                    ),
                    "temporal_visible_gaussians": temporal_visible,
                    "temporal_total_gaussians": temporal_total,
                    **residual_diag,
                    "color_residual_shared_hash_state_key": result.get(
                        "color_residual_shared_hash_state_key", None
                    ),
                    "color_residual_extra_hash_state_key": result.get(
                        "color_residual_extra_hash_state_key", None
                    ),
                    "color_residual_uses_extra_hash": bool(
                        result.get("color_residual_uses_extra_hash", False)
                    ),
                    "color_residual_uses_adapter": bool(
                        result.get("color_residual_uses_adapter", False)
                    ),
                    "metrics": {
                        "l1": l1_value,
                        "ssim": ssim_value,
                        "psnr": psnr_value,
                    },
                    "train_split_metrics": train_split_metrics,
                },
            )
            success_count += 1
            l1_values.append(l1_value)
            ssim_values.append(ssim_value)
            psnr_values.append(psnr_value)
            per_timestep_results.append(
                {
                    "status": "success",
                    "eval_timestep": int(eval_timestep),
                    **residual_diag,
                    "metrics": {
                        "l1": l1_value,
                        "ssim": ssim_value,
                        "psnr": psnr_value,
                    },
                }
            )
        except Exception as exc:
            logger.exception(
                f"[Eval] Train t={train_timestep}, Eval t={eval_timestep} failed: {exc}"
            )
            _append_eval_log_record(
                eval_log_path,
                {
                    "status": "failed",
                    "train_timestep": train_timestep,
                    "eval_timestep": eval_timestep,
                    "checkpoint_path": os.path.abspath(checkpoint_path),
                    "error": str(exc),
                },
            )
            failed_count += 1
            per_timestep_results.append(
                {
                    "status": "failed",
                    "eval_timestep": int(eval_timestep),
                    "error": str(exc),
                }
            )

    summary_metrics = None
    if success_count > 0:
        summary_metrics = {
            "l1": sum(l1_values) / success_count,
            "ssim": sum(ssim_values) / success_count,
            "psnr": sum(psnr_values) / success_count,
        }
        logger.info(
            f"[Eval] Train t={train_timestep}, Avg over t=0..{eval_max_timestep}: "
            f"L1={summary_metrics['l1']:.6f}, "
            f"SSIM={summary_metrics['ssim']:.6f}, "
            f"PSNR={summary_metrics['psnr']:.4f}"
        )
    else:
        logger.warning(
            f"[Eval] Train t={train_timestep}: no successful eval results to average."
        )

    _append_eval_log_record(
        eval_log_path,
        {
            "status": "summary",
            "train_timestep": train_timestep,
            "eval_timestep": f"0..{eval_max_timestep}",
            "num_eval_timesteps_total": eval_max_timestep + 1,
            "num_eval_timesteps_succeeded": success_count,
            "num_eval_timesteps_failed": failed_count,
            "avg_metrics": summary_metrics,
        },
    )

    logger.info(
        f"[Eval] Train timestep {train_timestep} finished: "
        f"success={success_count}, failed={failed_count}"
    )
    return {
        "status": "summary",
        "train_timestep": int(train_timestep),
        "avg_metrics": summary_metrics,
        "per_timestep_results": per_timestep_results,
        "num_eval_timesteps_succeeded": int(success_count),
        "num_eval_timesteps_failed": int(failed_count),
    }


def _cleanup_dated_output_dirs(output_dir: str) -> list[str]:
    removed_dirs: list[str] = []
    if not os.path.isdir(output_dir):
        return removed_dirs

    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and pattern.match(name):
            try:
                shutil.rmtree(path)
                removed_dirs.append(path)
            except Exception:
                # Keep training robust even if cleanup fails.
                pass
    return removed_dirs


def setup_logging(cfg: omegaconf.DictConfig) -> tuple[str, list[str]]:
    """Configure logging based on config."""
    log_level = cfg.get("log_level", "INFO")
    output_dir = cfg.get("output_dir", "outputs")
    training_log_path = _build_training_log_path(cfg)
    training_log_dir = os.path.dirname(training_log_path)
    removed_dated_dirs = _cleanup_dated_output_dirs(output_dir)

    os.makedirs(training_log_dir, exist_ok=True)

    with omegaconf.open_dict(cfg):
        runtime_cfg = cfg.get("runtime", {})
        runtime_cfg["training_log_file"] = training_log_path
        runtime_cfg["training_log_dir"] = training_log_dir
        cfg["runtime"] = runtime_cfg

    logger.remove()
    logger.add(sys.stderr, level=log_level)
    if cfg.get("log_to_file", True):
        logger.add(training_log_path, level=log_level)
    return training_log_path, removed_dated_dirs


def setup_wandb(cfg: omegaconf.DictConfig) -> None:
    """Initialize Weights & Biases logging."""
    wandb_mode = cfg.get("wandb_mode", "disabled")
    
    if wandb_mode == "disabled":
        return
    
    wandb.init(
        project=cfg.get("wandb_project", "cl-splats"),
        name=cfg.get("wandb_run_name", None),
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
        mode=wandb_mode,
    )
    logger.info(f"Initialized wandb in {wandb_mode} mode")


def load_data(cfg: omegaconf.DictConfig) -> CLSplatsDataset:
    """Load dataset based on configuration."""
    dataset_cfg = cfg.get("dataset", {})
    data_path = dataset_cfg.get("path", None)
    
    if data_path is None:
        logger.error("No dataset path specified! Set dataset.path in config or command line.")
        return None
    
    configured_path = str(data_path)
    resolved_path = os.path.abspath(os.path.expanduser(configured_path))
    logger.info(f"Dataset path (config): {configured_path}")
    logger.info(f"Dataset path (resolved): {resolved_path}")
    
    if not os.path.exists(resolved_path):
        logger.error(f"Dataset path does not exist: {resolved_path}")
        return None
    
    logger.info(f"Loading dataset from {resolved_path}")
    
    dataset = CLSplatsDataset(
        path=resolved_path,
        resolution_scale=dataset_cfg.get("resolution", 1.0),
        white_background=dataset_cfg.get("white_background", False),
        eval_mode=dataset_cfg.get("eval", False),
        split_seed=dataset_cfg.get("split_seed", 0),
        prefer_undist=dataset_cfg.get("prefer_undist", True),
    )
    force_random_split = bool(dataset_cfg.get("force_random_split", False))
    if hasattr(dataset, "configure_split_policy"):
        split_policy = dataset.configure_split_policy(
            force_random_split=force_random_split,
            split_seed=dataset_cfg.get("split_seed", 0),
            eval_mode=dataset_cfg.get("eval", False),
            clear_cache=True,
        )
        logger.info(
            "Dataset split policy: "
            f"force_random_split={str(split_policy.get('force_random_split', False)).lower()}, "
            f"has_predefined_split={str(split_policy.get('has_predefined_split', False)).lower()}, "
            f"split_seed={int(split_policy.get('split_seed', dataset_cfg.get('split_seed', 0)))}"
        )
    elif force_random_split:
        if hasattr(dataset, "_predefined_split"):
            dataset._predefined_split = None
        if hasattr(dataset, "_scene_info_cache") and isinstance(dataset._scene_info_cache, dict):
            dataset._scene_info_cache.clear()
        if hasattr(dataset, "_cameras_cache") and isinstance(dataset._cameras_cache, dict):
            dataset._cameras_cache.clear()
        logger.info(
            "Dataset split: force_random_split=true, ignore predefined split file and use random split."
        )
    
    logger.info(f"Dataset loaded: {dataset.get_num_timesteps()} timesteps, type={dataset.dataset_type}")
    if hasattr(dataset, "get_dataset_debug_info"):
        dataset_info = dataset.get_dataset_debug_info()
        logger.info(
            "Dataset summary: "
            f"layout={dataset_info['layout']}, "
            f"images_root={dataset_info['images_root']}, "
            f"sparse_root={dataset_info['sparse_root']}, "
            f"timesteps={dataset_info['timesteps']}"
        )
    return dataset


def _resolve_force_random_split(dataset: CLSplatsDataset, cfg: omegaconf.DictConfig) -> bool:
    """
    Resolve whether evaluation should force random split to match training.
    """
    dataset_cfg = cfg.get("dataset", {})
    force_from_cfg = bool(dataset_cfg.get("force_random_split", False))
    dataset_eval_mode = bool(getattr(dataset, "eval_mode", False))
    if hasattr(dataset, "has_predefined_split"):
        dataset_has_predefined = bool(dataset.has_predefined_split())
    else:
        dataset_has_predefined = getattr(dataset, "_predefined_split", None) is not None
    if hasattr(dataset, "is_force_random_split_enabled"):
        force_from_dataset = bool(dataset.is_force_random_split_enabled())
    else:
        force_from_dataset = False
    return bool(
        dataset_eval_mode
        and (force_from_cfg or force_from_dataset or (not dataset_has_predefined))
    )


@hydra.main(version_base=None, config_path="../configs", config_name="cl-splats")
def main(cfg: omegaconf.DictConfig) -> None:
    """Main training function."""
    # Setup
    training_log_path, removed_dated_dirs = setup_logging(cfg)
    set_random_seed(0)
    runtime_cfg = cfg.get("runtime", {})
    experiment_tag = _sanitize_filename_token(
        str(runtime_cfg.get("experiment_tag", _resolve_experiment_tag(cfg)))
    )
    setup_wandb(cfg)
    if removed_dated_dirs:
        logger.info(
            "Removed dated output directories: "
            + ", ".join(removed_dated_dirs)
        )
    logger.info(f"Training log file: {training_log_path}")
    logger.info(f"Experiment tag: {experiment_tag}")
    
    logger.info("=" * 60)
    logger.info("CL-Splats Training")
    logger.info("=" * 60)
    logger.info("Global random seed fixed to 0")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type != "cuda":
        logger.warning("CUDA not available! Training will be very slow or may not work.")
    
    # Load dataset
    dataset = load_data(cfg)
    if dataset is None:
        return
    
    # Initialize trainer/method from registry
    trainer = create_trainer(cfg)
    trainer_supports_builtin_eval = bool(
        getattr(trainer, "supports_builtin_eval", True)
    )
    trainer_eval_after_training_only = bool(
        getattr(trainer, "eval_after_training_only", False)
    )
    if hasattr(trainer, "_dataset_for_test_render"):
        trainer._dataset_for_test_render = dataset
    if hasattr(trainer, "_force_random_split_for_dataset"):
        try:
            trainer._force_random_split_for_dataset(dataset)
        except Exception:
            logger.exception(
                "trainer._force_random_split_for_dataset failed; continue with dataset default split behavior."
            )
    output_dir = cfg.get("output_dir", "outputs")
    eval_log_path = _build_eval_log_path(cfg)
    eval_log_dir = os.path.dirname(eval_log_path)
    os.makedirs(eval_log_dir, exist_ok=True)
    with omegaconf.open_dict(cfg):
        runtime_cfg = cfg.get("runtime", {})
        runtime_cfg["eval_log_file"] = eval_log_path
        runtime_cfg["eval_log_dir"] = eval_log_dir
        cfg["runtime"] = runtime_cfg
    logger.info(f"Eval log file: {eval_log_path}")
    if not trainer_supports_builtin_eval:
        logger.warning(
            "Built-in eval is disabled for the selected trainer; "
            "train.py will skip automatic eval checkpoints for this run."
        )
    
    # Load checkpoint if resuming
    checkpoint_path = cfg.get("resume_from", None)
    resumed_from_checkpoint = False
    resume_checkpoint_path = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        resume_checkpoint_path = os.path.abspath(str(checkpoint_path))
        trainer.load_checkpoint(resume_checkpoint_path)
        resumed_from_checkpoint = True
        logger.info(f"Resumed from checkpoint: {resume_checkpoint_path}")

        # Run an initial eval immediately after resume, before any new training.
        resume_train_timestep = int(getattr(trainer, "timestep", 0))
        if trainer_supports_builtin_eval and not trainer_eval_after_training_only:
            logger.info(
                f"[Eval] Resume init: evaluate checkpoint at train timestep {resume_train_timestep}"
            )
            resume_eval_summary = _run_eval_up_to_timestep(
                cfg,
                resume_train_timestep,
                resume_checkpoint_path,
                eval_log_path,
                force_random_split=_resolve_force_random_split(dataset, cfg),
            )
            if hasattr(trainer, "on_post_eval"):
                try:
                    trainer.on_post_eval(
                        train_timestep=resume_train_timestep,
                        eval_summary=resume_eval_summary,
                    )
                except Exception:
                    logger.exception(
                        "[EvalHook] trainer.on_post_eval failed during resume init eval"
                    )
    else:
        # Initialize Gaussians from first timestep point cloud
        scene_info = dataset.get_scene_info(0)
        if scene_info.point_cloud is not None:
            cameras = dataset.get_cameras(0)
            spatial_lr_scale = scene_info.nerf_normalization["radius"]
            trainer.initialize_from_point_cloud(
                scene_info.point_cloud, 
                cameras, 
                spatial_lr_scale
            )
        else:
            logger.error("No point cloud found in dataset!")
            return
    
    # Training loop over timesteps
    train_cfg = cfg.get("train", {})
    start_time = train_cfg.get("start_time", 0)
    num_times = min(train_cfg.get("num_times", 10), dataset.get_num_timesteps())
    
    # If resumed, start from next timestep
    if resumed_from_checkpoint:
        start_time = max(start_time, trainer.timestep + 1)
    
    save_interval = cfg.get("save_interval", 1)
    if save_interval <= 0:
        logger.warning(f"Invalid save_interval={save_interval}, fallback to 1.")
        save_interval = 1

    for timestep in range(start_time, num_times):
        logger.info(f"\n{'='*60}")
        logger.info(f"Timestep {timestep}/{num_times-1}")
        logger.info(f"{'='*60}")
        
        # Prepare data for this timestep
        trainer.prepare_timestep(timestep, dataset)
        
        # Train
        stats = trainer.train()
        
        # Log history
        trainer.log_history()
        
        # Save checkpoint
        os.makedirs(output_dir, exist_ok=True)

        checkpoint_is_temporary = False
        if (timestep + 1) % save_interval == 0:
            ckpt_path = os.path.join(output_dir, f"checkpoint_t{timestep}.pt")
        else:
            ckpt_path = os.path.join(
                eval_log_dir, f"_tmp_{experiment_tag}_eval_t{timestep}.pt"
            )
            checkpoint_is_temporary = True

        trainer.save_checkpoint(ckpt_path, save_external_ply=False)
        if trainer_supports_builtin_eval and not trainer_eval_after_training_only:
            eval_summary = _run_eval_up_to_timestep(
                cfg,
                timestep,
                ckpt_path,
                eval_log_path,
                force_random_split=_resolve_force_random_split(dataset, cfg),
            )
            if hasattr(trainer, "on_post_eval"):
                try:
                    trainer.on_post_eval(
                        train_timestep=timestep,
                        eval_summary=eval_summary,
                    )
                except Exception:
                    logger.exception(
                        f"[EvalHook] trainer.on_post_eval failed at train timestep {timestep}"
                    )

        if checkpoint_is_temporary:
            _cleanup_temp_eval_checkpoint(ckpt_path, eval_log_dir)

        logger.info(f"Timestep {timestep} complete: loss={stats['avg_loss']:.6f}, gaussians={stats['num_gaussians']}")
    
    # Save final checkpoint and PLY
    final_ply_path = os.path.join(output_dir, "point_cloud_final.ply")
    if hasattr(trainer, "save_current_ply"):
        trainer.save_current_ply(final_ply_path)
    elif hasattr(trainer, "_save_working_set_ply"):
        trainer._save_working_set_ply(final_ply_path)
    else:
        trainer.gaussians.save_ply(final_ply_path)
    logger.info(f"Saved final point cloud to {final_ply_path}")

    final_ckpt_path = os.path.join(output_dir, "checkpoint_final.pt")
    try:
        trainer.save_checkpoint(
            final_ckpt_path,
            ply_path_override=final_ply_path,
            save_external_ply=False,
            compact_for_eval=True,
        )
    except TypeError:
        trainer.save_checkpoint(
            final_ckpt_path,
            ply_path_override=final_ply_path,
            save_external_ply=False,
        )

    if trainer_supports_builtin_eval and trainer_eval_after_training_only:
        final_eval_timestep = int(getattr(trainer, "timestep", max(num_times - 1, 0)))
        logger.info(
            f"[Eval] Final-only trainer: evaluate final checkpoint at train timestep {final_eval_timestep}"
        )
        final_eval_summary = _run_eval_up_to_timestep(
            cfg,
            final_eval_timestep,
            final_ckpt_path,
            eval_log_path,
            force_random_split=_resolve_force_random_split(dataset, cfg),
        )
        if hasattr(trainer, "on_post_eval"):
            try:
                trainer.on_post_eval(
                    train_timestep=final_eval_timestep,
                    eval_summary=final_eval_summary,
                )
            except Exception:
                logger.exception(
                    "[EvalHook] trainer.on_post_eval failed during final-only eval"
                )
    
    # Cleanup
    if wandb.run is not None:
        wandb.finish()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
