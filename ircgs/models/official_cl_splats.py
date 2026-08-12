"""Bridge the public ``cl-splats-main`` implementation into this workspace.

The training algorithm is intentionally kept in the sibling public repository.
This module only adapts the local multi-timestep COLMAP layout and wraps the
official per-timestep PLY files in the checkpoint contract used by this repo.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import omegaconf
import torch
from loguru import logger

from ircgs.dataset.colmap_reader import read_extrinsics_binary


_BRIDGE_MARKER = "generated_by_clsplats_official_bridge"


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    value = cfg.get(key, default)
    return default if value is None else value


def _as_plain(value: Any) -> Any:
    if omegaconf.OmegaConf.is_config(value):
        return omegaconf.OmegaConf.to_container(value, resolve=True)
    return value


def _as_list(value: Any) -> list[str]:
    value = _as_plain(value)
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in value.split() if part]
    if isinstance(value, Iterable):
        return [str(part) for part in value]
    return [str(value)]


def _hydra_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_official_root(model_cfg: Any) -> Path:
    """Resolve the sibling public repository on local machines and AutoDL."""
    official_cfg = _cfg_get(model_cfg, "official_cl_splats", {})
    configured = _cfg_get(official_cfg, "root", None)
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(str(configured)).expanduser())
    environment_root = os.environ.get("CL_SPLATS_MAIN_DIR")
    if environment_root:
        candidates.append(Path(environment_root).expanduser())

    root = _workspace_root()
    candidates.extend((root / "cl-splats-main", root.parent / "cl-splats-main"))
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "clsplats" / "train.py").is_file() and (
            candidate / "configs" / "cl-splats.yaml"
        ).is_file():
            return candidate

    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "The public cl-splats-main repository was not found. Set "
        "model.official_cl_splats.root or CL_SPLATS_MAIN_DIR. Checked: "
        f"{rendered}"
    )


def _resolve_python(model_cfg: Any) -> str:
    official_cfg = _cfg_get(model_cfg, "official_cl_splats", {})
    configured = _cfg_get(official_cfg, "python", None)
    return str(configured or sys.executable)


def _basename(name: str) -> str:
    return str(name).replace("\\", "/").rsplit("/", 1)[-1]


def _safe_reset_generated(path: Path, marker_name: str) -> None:
    """Remove only a directory previously created by this bridge."""
    if not path.exists():
        return
    marker = path / marker_name
    if not marker.is_file():
        raise RuntimeError(
            f"Refusing to remove unrecognised directory {path}. "
            f"Move it aside or add the generated marker {marker.name}."
        )
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Invalid generated-data marker: {marker}") from exc
    if payload.get("marker") != _BRIDGE_MARKER:
        raise RuntimeError(f"Refusing to remove directory with an unknown marker: {path}")
    shutil.rmtree(path)


def _link_or_copy(source: Path, destination: Path, prefer_hardlinks: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            if os.path.samefile(source, destination):
                return
        except OSError:
            pass
        destination.unlink()
    if prefer_hardlinks:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def _read_binary_image_name(fid) -> tuple[int, str, bytes, bytes]:
    """Read one COLMAP image record, retaining its binary payload verbatim."""
    header = fid.read(64)
    if len(header) != 64:
        raise ValueError("Truncated COLMAP images.bin image header")
    image_id = struct.unpack_from("<i", header, 0)[0]
    name_bytes = bytearray()
    while True:
        char = fid.read(1)
        if not char:
            raise ValueError("Truncated COLMAP images.bin image name")
        if char == b"\x00":
            break
        name_bytes.extend(char)
    count_raw = fid.read(8)
    if len(count_raw) != 8:
        raise ValueError("Truncated COLMAP images.bin point count")
    count = struct.unpack("<Q", count_raw)[0]
    points = fid.read(24 * count)
    if len(points) != 24 * count:
        raise ValueError("Truncated COLMAP images.bin point observations")
    return image_id, name_bytes.decode("utf-8"), header, count_raw + points


def rewrite_images_binary(source: Path, destination: Path, renamed_names: Dict[int, str]) -> int:
    """Copy only selected COLMAP images while changing their file names."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    written_ids: set[int] = set()
    with source.open("rb") as src, destination.open("wb") as dst:
        count_raw = src.read(8)
        if len(count_raw) != 8:
            raise ValueError(f"Invalid COLMAP images.bin: {source}")
        total = struct.unpack("<Q", count_raw)[0]
        dst.write(struct.pack("<Q", len(renamed_names)))
        for _ in range(total):
            image_id, _old_name, header, payload = _read_binary_image_name(src)
            if image_id not in renamed_names:
                continue
            new_name = str(renamed_names[image_id]).replace("\\", "/")
            dst.write(header)
            dst.write(new_name.encode("utf-8"))
            dst.write(b"\x00")
            dst.write(payload)
            written_ids.add(image_id)
    kept = len(written_ids)
    if kept != len(renamed_names):
        missing = sorted(set(renamed_names) - written_ids)
        raise ValueError(
            f"COLMAP images.bin contains {kept} selected records, expected "
            f"{len(renamed_names)}; source={source}, missing ids={missing[:8]}"
        )
    return kept


def count_ply_vertices(path: Path) -> int:
    """Read the vertex count from an ASCII or binary PLY header."""
    vertex_count: Optional[int] = None
    with path.open("rb") as fid:
        if fid.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        while True:
            line = fid.readline()
            if not line:
                raise ValueError(f"PLY header has no end_header marker: {path}")
            stripped = line.strip()
            if stripped.startswith(b"element vertex "):
                try:
                    vertex_count = int(stripped.split()[2])
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"Invalid PLY vertex declaration in {path}") from exc
            if stripped == b"end_header":
                break
    if vertex_count is None:
        raise ValueError(f"PLY header has no vertex element: {path}")
    return vertex_count


def _stage_image_name(timestep: int, original_name: str) -> str:
    return f"day_{int(timestep)}_{_basename(original_name)}"


def _find_sparse_model(dataset: Any, timestep: int) -> Path:
    source = dataset.get_timestep_data_source(timestep)
    raw = Path(str(source["sparse_model_path"])).expanduser()
    if raw.is_dir() and (raw / "images.bin").is_file():
        return raw
    candidates = [
        raw,
        raw.parent,
        Path(str(dataset.path)) / "sparse_undist" / "0",
        Path(str(dataset.path)) / "sparse" / "0",
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "images.bin").is_file():
            return candidate
    raise FileNotFoundError(
        "Official CL-Splats staging requires a binary COLMAP model with images.bin. "
        f"Checked {[str(path) for path in candidates]}"
    )


def materialize_training_dataset(
    dataset: Any,
    stage_dir: Path,
    num_times: int,
    prefer_hardlinks: bool = True,
) -> Dict[str, Any]:
    """Build an official-layout COLMAP workspace from the local train split."""
    _safe_reset_generated(stage_dir, "bridge_manifest.json")
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "bridge_manifest.json").write_text(
        json.dumps({"marker": _BRIDGE_MARKER, "status": "staging"}, indent=2),
        encoding="utf-8",
    )
    sparse_dir = stage_dir / "sparse" / "0"
    images_dir = stage_dir / "images"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    sparse_source = _find_sparse_model(dataset, 0)
    source_extrinsics = read_extrinsics_binary(str(sparse_source / "images.bin"))
    by_basename: Dict[str, list[tuple[int, Any]]] = {}
    for image_id, record in source_extrinsics.items():
        by_basename.setdefault(_basename(record.name), []).append((image_id, record))

    selected: Dict[int, Dict[str, Any]] = {}
    for timestep in range(num_times):
        scene = dataset.get_scene_info(timestep)
        for camera in scene.train_cameras:
            basename = _basename(camera.image_name)
            matches = by_basename.get(basename, [])
            if len(matches) != 1:
                raise ValueError(
                    f"Could not uniquely match train image {camera.image_name!r} "
                    f"to COLMAP images.bin ({len(matches)} matches)."
                )
            image_id, record = matches[0]
            if image_id in selected:
                raise ValueError(f"COLMAP image id {image_id} was selected twice")
            source_image = Path(str(camera.image_path)).expanduser()
            if not source_image.is_file():
                raise FileNotFoundError(
                    f"Train image is missing: {source_image}. "
                    "Check the dataset path and images/images_undist layout."
                )
            staged_name = _stage_image_name(timestep, record.name)
            selected[image_id] = {
                "timestep": int(timestep),
                "source_name": record.name,
                "source_image": str(source_image.resolve()),
                "staged_name": staged_name,
            }
            _link_or_copy(source_image, images_dir / staged_name, prefer_hardlinks)

    for filename in ("cameras.bin", "points3D.bin", "points3D.ply", "project.ini"):
        source_file = sparse_source / filename
        if source_file.is_file():
            _link_or_copy(source_file, sparse_dir / filename, prefer_hardlinks)

    rewrite_images_binary(
        sparse_source / "images.bin",
        sparse_dir / "images.bin",
        {image_id: entry["staged_name"] for image_id, entry in selected.items()},
    )

    manifest = {
        "marker": _BRIDGE_MARKER,
        "source_dataset": str(Path(str(dataset.path)).expanduser().resolve()),
        "source_sparse_model": str(sparse_source.resolve()),
        "num_times": int(num_times),
        "num_train_images": len(selected),
        "images": selected,
    }
    (stage_dir / "bridge_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return manifest


class OfficialCLSplatsTrainer:
    """Run the public CL-Splats trainer while preserving this repo's interface.

    The public trainer exports one PLY for every timestep in a single external
    run. Evaluation is therefore handled by ``run_cl_splats.sh`` afterward so
    that timestep ``t`` is always evaluated with its matching PLY snapshot.
    """

    supports_builtin_eval = False
    eval_after_training_only = False
    supports_exact_history = False
    scene_history_enabled = False

    def __init__(self, cfg: omegaconf.DictConfig):
        self.cfg = cfg
        self.timestep = 0
        self._dataset = None
        self._trained = False
        self._stage_dir: Optional[Path] = None
        self._run_dir: Optional[Path] = None
        self._official_root: Optional[Path] = None
        self._command: list[str] = []
        self._ply_by_timestep: Dict[int, Path] = {}
        self.history: Dict[str, list] = {"losses": [], "num_gaussians": [], "num_pruned": []}

        model_cfg = cfg.get("model", {}) or {}
        self.model_cfg = model_cfg
        self.official_cfg = _cfg_get(model_cfg, "official_cl_splats", {}) or {}
        self.output_dir = Path(str(cfg.get("output_dir", "outputs"))).expanduser().resolve()
        self.bridge_root = self.output_dir / "_official_cl_splats"
        self._stage_dir = self.bridge_root / "dataset"
        self._run_dir = self.bridge_root / "run"

    def initialize_from_point_cloud(self, *args, **kwargs) -> None:
        """The external trainer creates its own model from the staged PLY."""
        del args, kwargs

    def prepare_timestep(self, timestep: int, dataset: Any) -> None:
        self.timestep = int(timestep)
        if self._dataset is None:
            self._dataset = dataset

    def _resolve_num_times(self) -> int:
        train_cfg = self.cfg.get("train", {}) or {}
        requested = int(_cfg_get(train_cfg, "num_times", self._dataset.get_num_timesteps()))
        available = int(self._dataset.get_num_timesteps())
        if requested > available:
            logger.warning(
                "Official CL-Splats requested more timesteps than available; "
                f"requested={requested}, available={available}."
            )
        return max(1, min(requested, available))

    def _reset_generated_paths(self) -> None:
        _safe_reset_generated(self._stage_dir, "bridge_manifest.json")
        _safe_reset_generated(self._run_dir, "bridge_run.json")
        self.bridge_root.mkdir(parents=True, exist_ok=True)

    def _official_overrides(self, num_times: int) -> list[str]:
        official_cfg = self.official_cfg
        iters = int(_cfg_get(official_cfg, "iters_per_timestep", 100))
        sh_degree = int(_cfg_get(official_cfg, "sh_degree", 0))
        overrides = [
            f"train.num_times={num_times}",
            "train.start_time=0",
            f"train.iters_per_timestep={iters}",
            f"model.sh_degree={sh_degree}",
            "history.log_history=true",
            "wandb_mode=disabled",
        ]

        # These values are optional bridge controls. With no values supplied,
        # the public repository's own defaults are used unchanged.
        optional_values = (
            ("change_threshold", "change.threshold"),
            ("dilate_mask", "change.dilate_mask"),
            ("dilate_kernel_size", "change.dilate_kernel_size"),
            ("upsample", "change.upsample"),
            ("lambda_bound", "constraints.lambda_bound"),
            ("prune_every", "constraints.prune_every"),
            ("prune_dist_thresh", "constraints.prune_dist_thresh"),
            ("prune_consecutive", "constraints.prune_consecutive"),
        )
        for source_key, target_key in optional_values:
            value = official_cfg.get(source_key, None)
            if value is not None:
                overrides.append(f"{target_key}={_hydra_scalar(value)}")

        overrides.extend(_as_list(official_cfg.get("overrides", [])))
        return overrides

    def _build_command(self, num_times: int, stage_dir: Path) -> list[str]:
        assert self._official_root is not None
        python_bin = _resolve_python(self.model_cfg)
        command = [
            python_bin,
            "-m",
            "clsplats.train",
            "--data-path",
            str(stage_dir),
            "--images",
            "images",
        ]
        if bool(_cfg_get(self.official_cfg, "offline", False)):
            command.append("--offline")
        command.extend(self._official_overrides(num_times))
        return command

    def _official_environment(self) -> Dict[str, str]:
        assert self._official_root is not None
        env = os.environ.copy()
        repo_root = str(_workspace_root().resolve())
        old_paths = [part for part in env.get("PYTHONPATH", "").split(os.pathsep) if part]
        filtered = [part for part in old_paths if Path(part).resolve() != Path(repo_root)]
        env["PYTHONPATH"] = os.pathsep.join([str(self._official_root), *filtered])
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("WANDB_MODE", "disabled")

        if bool(_cfg_get(self.official_cfg, "offline", False)):
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            env["HF_DATASETS_OFFLINE"] = "1"

            home_torch = Path.home() / ".cache" / "torch"
            torch_candidates = [
                Path(env["TORCH_HOME"]).expanduser()
                if env.get("TORCH_HOME")
                else None,
                home_torch,
                Path("/workdir/torch_cache"),
            ]
            torch_home = next(
                (
                    candidate
                    for candidate in torch_candidates
                    if candidate is not None
                    and (
                        candidate
                        / "hub"
                        / "facebookresearch_dinov2_main"
                        / "hubconf.py"
                    ).is_file()
                    and (
                        candidate
                        / "hub"
                        / "checkpoints"
                        / "dinov2_vitb14_pretrain.pth"
                    ).is_file()
                ),
                home_torch,
            )
            env["TORCH_HOME"] = str(torch_home.resolve())

            home_hf = Path.home() / ".cache" / "huggingface"
            hf_candidates = [
                Path(env["HF_HOME"]).expanduser() if env.get("HF_HOME") else None,
                home_hf,
                Path("/workdir/hf_cache"),
            ]
            depth_cache_dirname = (
                "models--depth-anything--Depth-Anything-V2-Small-hf"
            )
            hf_home = next(
                (
                    candidate
                    for candidate in hf_candidates
                    if candidate is not None
                    and (candidate / "hub" / depth_cache_dirname).is_dir()
                ),
                next(
                    (
                        candidate
                        for candidate in hf_candidates
                        if candidate is not None and candidate.is_dir()
                    ),
                    home_hf,
                ),
            )
            env["HF_HOME"] = str(hf_home.resolve())
            logger.info(
                "Official offline cache roots: TORCH_HOME={}, HF_HOME={}",
                env["TORCH_HOME"],
                env["HF_HOME"],
            )
        return env

    def _expected_ply(self, timestep: int) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "outputs" / f"gaussians_time_{int(timestep):04d}.ply"

    def _missing_official_timesteps(self, num_times: int) -> list[int]:
        missing: list[int] = []
        for timestep in range(int(num_times)):
            ply = self._ply_by_timestep.get(timestep, self._expected_ply(timestep))
            if ply.is_file():
                self._ply_by_timestep[timestep] = ply
            else:
                missing.append(timestep)
        return missing

    def _run_official_training(self) -> None:
        if self._dataset is None:
            raise RuntimeError("Official CL-Splats trainer has no dataset")
        num_times = self._resolve_num_times()
        self._official_root = resolve_official_root(self.model_cfg)
        self._reset_generated_paths()
        assert self._stage_dir is not None and self._run_dir is not None
        prefer_hardlinks = bool(_cfg_get(self.official_cfg, "prefer_hardlinks", True))
        materialize_training_dataset(
            self._dataset,
            self._stage_dir,
            num_times=num_times,
            prefer_hardlinks=prefer_hardlinks,
        )

        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "bridge_run.json").write_text(
            json.dumps(
                {
                    "marker": _BRIDGE_MARKER,
                    "official_root": str(self._official_root),
                    "num_times": num_times,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._command = self._build_command(num_times, self._stage_dir)
        logger.info("Launching public CL-Splats implementation:")
        logger.info("  cwd={}", self._run_dir)
        logger.info("  command={}", " ".join(self._command))
        subprocess.run(
            self._command,
            cwd=str(self._run_dir),
            env=self._official_environment(),
            check=True,
        )

        self._ply_by_timestep = {
            timestep: self._expected_ply(timestep) for timestep in range(num_times)
        }
        missing = [str(path) for path in self._ply_by_timestep.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Public CL-Splats finished without exporting every timestep PLY: "
                + ", ".join(missing)
            )
        self._trained = True

    def train(self) -> Dict[str, Any]:
        if self._trained and self._dataset is not None:
            num_times = self._resolve_num_times()
            missing_timesteps = self._missing_official_timesteps(num_times)
            if missing_timesteps:
                logger.warning(
                    "Resumed official CL-Splats artifacts do not cover the requested "
                    "sequence (num_times={}, missing={}). The public trainer has no "
                    "checkpoint-resume contract, so the full sequence will be rebuilt.",
                    num_times,
                    missing_timesteps,
                )
                self._trained = False
        if not self._trained:
            self._run_official_training()
        ply = self._ply_by_timestep.get(self.timestep, self._expected_ply(self.timestep))
        if not ply.is_file():
            raise FileNotFoundError(f"Official CL-Splats PLY not found for t={self.timestep}: {ply}")
        num_gaussians = count_ply_vertices(ply)
        self.history["losses"].append(0.0)
        self.history["num_gaussians"].append(num_gaussians)
        self.history["num_pruned"].append(0)
        return {
            "avg_loss": 0.0,
            "num_gaussians": num_gaussians,
            "timestep": self.timestep,
            "official_ply": str(ply),
        }

    def _evaluation_config(self) -> Dict[str, Any]:
        payload = _as_plain(self.cfg)
        if not isinstance(payload, dict):
            payload = {}
        payload = dict(payload)
        payload["model"] = {
            "name": "cl-splats",
            "representation": "gaussian",
            "sh_degree": int(_cfg_get(self.official_cfg, "sh_degree", 0)),
            "optimizer_type": "default",
        }
        payload["trainer_type"] = "official_cl_splats"
        return payload

    def save_checkpoint(
        self,
        path: str,
        ply_path_override: Optional[str] = None,
        save_external_ply: bool = False,
        compact_for_eval: bool = False,
    ) -> None:
        del save_external_ply, compact_for_eval
        source_ply = self._ply_by_timestep.get(
            self.timestep, self._expected_ply(self.timestep)
        )
        if not source_ply.is_file():
            raise FileNotFoundError(f"Cannot checkpoint missing official PLY: {source_ply}")
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_ply = (
            Path(ply_path_override)
            if ply_path_override
            else checkpoint_path.with_suffix(".ply")
        )
        checkpoint_ply.parent.mkdir(parents=True, exist_ok=True)
        try:
            same_ply = os.path.samefile(source_ply, checkpoint_ply)
        except (FileNotFoundError, OSError):
            same_ply = False
        if not same_ply:
            shutil.copy2(source_ply, checkpoint_ply)
        payload = {
            "trainer_type": "official_cl_splats",
            "config": self._evaluation_config(),
            "bridge_config": _as_plain(self.cfg),
            "timestep": int(self.timestep),
            "trained": bool(self._trained),
            "history": self.history,
            "official_root": str(self._official_root) if self._official_root else None,
            "run_dir": str(self._run_dir) if self._run_dir else None,
            "stage_dir": str(self._stage_dir) if self._stage_dir else None,
            "command": list(self._command),
            "ply_path": str(checkpoint_ply.resolve()),
            "ply_by_timestep": {str(k): str(v.resolve()) for k, v in self._ply_by_timestep.items()},
            "supports_builtin_eval": False,
        }
        torch.save(payload, checkpoint_path)
        logger.info("Saved official CL-Splats bridge checkpoint to {}", checkpoint_path)

    def load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if str(payload.get("trainer_type", "")).strip().lower() != "official_cl_splats":
            raise ValueError(f"Not an official CL-Splats bridge checkpoint: {path}")
        self.timestep = int(payload.get("timestep", 0))
        self._trained = bool(payload.get("trained", False))
        self.history = payload.get("history", self.history) or self.history
        self._official_root = Path(payload["official_root"]) if payload.get("official_root") else None
        self._run_dir = Path(payload["run_dir"]) if payload.get("run_dir") else self._run_dir
        self._stage_dir = Path(payload["stage_dir"]) if payload.get("stage_dir") else self._stage_dir
        self._command = list(payload.get("command", []))
        self._ply_by_timestep = {
            int(key): Path(value) for key, value in (payload.get("ply_by_timestep", {}) or {}).items()
        }
        if not self._ply_by_timestep and payload.get("ply_path"):
            self._ply_by_timestep[self.timestep] = Path(str(payload["ply_path"]))
        if self._trained:
            missing_paths = [
                path for path in self._ply_by_timestep.values() if not path.is_file()
            ]
            if missing_paths:
                logger.warning(
                    "Official bridge checkpoint refers to missing PLY files ({}). "
                    "The official sequence will be rebuilt on the next train call.",
                    ", ".join(str(path) for path in missing_paths),
                )
                self._trained = False

    def log_history(self) -> None:
        logger.info(
            "Official CL-Splats history is stored under {}",
            self._run_dir / "outputs" / "history" if self._run_dir else "<not-run>",
        )

    def save_current_ply(self, path: str) -> None:
        source = self._ply_by_timestep.get(self.timestep, self._expected_ply(self.timestep))
        if not source.is_file():
            raise FileNotFoundError(f"Cannot export missing official PLY: {source}")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
