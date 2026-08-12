import argparse
import gc
import shutil
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image, ImageDraw

from ircgs.dataset.dataset_reader import CLSplatsDataset
from ircgs.eval import (
    _build_sfgs_eval_model,
    _is_sfgs_model,
    _normalize_model_name,
    _render_sfgs_view,
    build_representation_from_checkpoint,
    render,
    resolve_device,
    resolve_ply_path,
)
from ircgs.utils.prepare_clsplats_object_masks import (
    _build_sam_config,
    _ensure_parent,
    _load_automatic_sam_api,
    _resize_mask_to_shape,
    _save_color_overlay,
    _save_gray_mask,
    resolve_deva_root,
)
from ircgs.utils.camera_utils import Camera


def _clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _tensor_to_uint8_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = (
        image_tensor.detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return image


def _save_png(image_np: np.ndarray, output_path: Path) -> None:
    _ensure_parent(output_path)
    Image.fromarray(image_np).save(output_path)


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _segment_image_np(
    image_np: np.ndarray,
    *,
    auto_segment,
    sam_model,
    sam_config: dict,
    suppress_small_objects: bool,
) -> torch.Tensor:
    mask, _, _ = auto_segment(
        sam_config,
        sam_model,
        image_np,
        forward_mask=None,
        min_side=int(sam_config["size"]),
        suppress_small_mask=bool(suppress_small_objects),
    )
    return _resize_mask_to_shape(mask, image_np.shape[:2])


def _concat_h(images: List[np.ndarray], pad: int = 8) -> np.ndarray:
    if not images:
        raise ValueError("No images to concatenate.")
    h = max(img.shape[0] for img in images)
    padded = []
    for idx, img in enumerate(images):
        if img.shape[0] != h:
            canvas = np.zeros((h, img.shape[1], 3), dtype=np.uint8)
            canvas[: img.shape[0], : img.shape[1]] = img
            img = canvas
        padded.append(img)
        if idx != len(images) - 1:
            padded.append(np.full((h, pad, 3), 255, dtype=np.uint8))
    return np.concatenate(padded, axis=1)


def _draw_header(image_np: np.ndarray, title: str) -> np.ndarray:
    banner_h = 32
    banner = np.full((banner_h, image_np.shape[1], 3), 245, dtype=np.uint8)
    out = np.concatenate([banner, image_np], axis=0)
    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    draw.text((10, 8), title, fill=(0, 0, 0))
    return np.array(pil)


def _build_render_model(
    checkpoint_path: Path,
    checkpoint: dict,
    model_cfg: dict,
    device: str,
    timestep: int,
):
    model_name = _normalize_model_name(model_cfg.get("name", None))
    is_sfgs = _is_sfgs_model(model_name, checkpoint, checkpoint_path)
    if is_sfgs:
        render_model, _, _ = _build_sfgs_eval_model(
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            model_cfg=model_cfg,
            device=device,
            timestep=int(timestep),
        )
        return "ours", render_model

    ply_path = resolve_ply_path(checkpoint_path, checkpoint)
    model = build_representation_from_checkpoint(checkpoint)
    model.load_ply(str(ply_path))
    return "default", model


def _run_render_stage(
    *,
    dataset_path: Path,
    checkpoint_path: Path,
    run_root: Path,
    timestep: int,
    device: str,
    resolution_scale: float,
    split_seed: int,
    prefer_dist: bool,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model_cfg = ckpt_cfg.get("model", {}) if isinstance(ckpt_cfg, dict) else {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}

    dataset = CLSplatsDataset(
        path=str(dataset_path),
        resolution_scale=float(resolution_scale),
        white_background=False,
        eval_mode=True,
        device="cpu",
        split_seed=int(split_seed),
        prefer_undist=not bool(prefer_dist),
    )
    render_kind, render_model = _build_render_model(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        model_cfg=model_cfg,
        device=device,
        timestep=timestep,
    )

    pred_img_dir = run_root / "pred_images"
    gt_img_dir = run_root / "gt_images"
    _clear_dir(pred_img_dir)
    _clear_dir(gt_img_dir)

    scene_info = dataset.get_scene_info(timestep)
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    rendered_count = 0
    with torch.no_grad():
        for idx, cam_info in enumerate(scene_info.train_cameras):
            rendered = None
            render_result = None
            render_camera, gt_np = _prepare_single_render_camera(
                dataset=dataset,
                cam_info=cam_info,
                device=device,
            )
            try:
                if render_kind == "ours":
                    rendered = _render_sfgs_view(render_camera, render_model, bg_color)
                else:
                    render_result = render(render_camera, render_model, bg_color)
                    rendered = render_result["render"]

                pred_np = _tensor_to_uint8_image(rendered)
                image_name = str(getattr(render_camera, "image_name", f"{idx:06d}"))
                stem = Path(image_name).stem
                _save_png(pred_np, pred_img_dir / f"{stem}.png")
                _save_png(gt_np, gt_img_dir / f"{stem}.png")
                rendered_count += 1
            finally:
                del render_camera
                del gt_np
                if rendered is not None:
                    del rendered
                if render_result is not None:
                    del render_result
                _release_memory()

    del render_model
    del dataset
    del scene_info
    del checkpoint
    _release_memory()

    return rendered_count


def _run_sam_stage(
    *,
    run_root: Path,
    deva_root_arg: str | None,
    size: int,
    sam_pred_iou_threshold: float,
    sam_variant: str,
    sam_num_points_per_side: int,
    sam_num_points_per_batch: int,
    suppress_small_objects: bool,
) -> int:
    pred_img_dir = run_root / "pred_images"
    gt_img_dir = run_root / "gt_images"
    if not pred_img_dir.is_dir() or not gt_img_dir.is_dir():
        raise FileNotFoundError(
            "Missing rendered images. Run with `--stage render` first, or use `--stage all`."
        )

    pred_gray_dir = run_root / "pred_mask_gray"
    pred_color_dir = run_root / "pred_mask_color"
    gt_gray_dir = run_root / "gt_mask_gray"
    gt_color_dir = run_root / "gt_mask_color"
    compare_dir = run_root / "compare"
    for path in (
        pred_gray_dir,
        pred_color_dir,
        gt_gray_dir,
        gt_color_dir,
        compare_dir,
    ):
        _clear_dir(path)

    deva_root = resolve_deva_root(deva_root_arg)
    get_sam_model, auto_segment = _load_automatic_sam_api(deva_root)
    sam_config = _build_sam_config(
        deva_root,
        size=int(size),
        sam_pred_iou_threshold=float(sam_pred_iou_threshold),
        sam_variant=str(sam_variant),
        sam_num_points_per_side=int(sam_num_points_per_side),
        sam_num_points_per_batch=int(sam_num_points_per_batch),
        suppress_small_objects=bool(suppress_small_objects),
    )
    sam_model = get_sam_model(sam_config, "cuda")

    pred_paths = sorted(pred_img_dir.glob("*.png"))
    processed = 0
    for pred_img_path in pred_paths:
        stem = pred_img_path.stem
        gt_img_path = gt_img_dir / f"{stem}.png"
        if not gt_img_path.exists():
            continue

        pred_np = np.array(Image.open(pred_img_path).convert("RGB"))
        gt_np = np.array(Image.open(gt_img_path).convert("RGB"))

        pred_mask = _segment_image_np(
            pred_np,
            auto_segment=auto_segment,
            sam_model=sam_model,
            sam_config=sam_config,
            suppress_small_objects=bool(suppress_small_objects),
        )
        gt_mask = _segment_image_np(
            gt_np,
            auto_segment=auto_segment,
            sam_model=sam_model,
            sam_config=sam_config,
            suppress_small_objects=bool(suppress_small_objects),
        )

        pred_gray_path = pred_gray_dir / f"{stem}.png"
        pred_color_path = pred_color_dir / f"{stem}.jpg"
        gt_gray_path = gt_gray_dir / f"{stem}.png"
        gt_color_path = gt_color_dir / f"{stem}.jpg"
        _save_gray_mask(pred_mask, pred_gray_path)
        _save_gray_mask(gt_mask, gt_gray_path)
        _save_color_overlay(pred_np, pred_mask, pred_color_path)
        _save_color_overlay(gt_np, gt_mask, gt_color_path)

        pred_color_np = np.array(Image.open(pred_color_path).convert("RGB"))
        gt_color_np = np.array(Image.open(gt_color_path).convert("RGB"))
        compare_np = _concat_h([pred_np, pred_color_np, gt_np, gt_color_np], pad=10)
        compare_np = _draw_header(
            compare_np,
            f"{stem} | pred image | pred SAM | gt image | gt SAM",
        )
        _save_png(compare_np, compare_dir / f"{stem}.png")
        del pred_np
        del gt_np
        del pred_mask
        del gt_mask
        del pred_color_np
        del gt_color_np
        del compare_np
        _release_memory()
        processed += 1

    del sam_model
    _release_memory()

    return processed


def _prepare_single_render_camera(
    *,
    dataset: CLSplatsDataset,
    cam_info,
    device: str,
) -> tuple[Camera, np.ndarray]:
    camera_cpu = dataset._load_camera(cam_info)
    if camera_cpu.original_image is None:
        raise RuntimeError(f"Failed to load ground-truth image for {cam_info.image_name}")

    gt_np = _tensor_to_uint8_image(camera_cpu.original_image)
    render_camera = Camera(
        uid=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image_width=int(camera_cpu.image_width),
        image_height=int(camera_cpu.image_height),
        image_name=cam_info.image_name,
        image=None,
        object_mask=None,
        device=device,
    )
    del camera_cpu
    return render_camera, gt_np


def run(args: argparse.Namespace) -> Path:
    dataset_path = Path(args.dataset_root).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    device = resolve_device(args.device)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    timestep = int(args.timestep)
    run_root = output_root / f"compare_t{timestep}_train_sam_from_{checkpoint_path.stem}"
    num_rendered = None
    num_segmented = None

    if args.stage in {"render", "all"}:
        num_rendered = _run_render_stage(
            dataset_path=dataset_path,
            checkpoint_path=checkpoint_path,
            run_root=run_root,
            timestep=timestep,
            device=device,
            resolution_scale=float(args.resolution_scale),
            split_seed=int(args.split_seed),
            prefer_dist=bool(args.prefer_dist),
        )

    if args.stage in {"sam", "all"}:
        num_segmented = _run_sam_stage(
            run_root=run_root,
            deva_root_arg=args.deva_root,
            size=int(args.size),
            sam_pred_iou_threshold=float(args.sam_pred_iou_threshold),
            sam_variant=str(args.sam_variant),
            sam_num_points_per_side=int(args.sam_num_points_per_side),
            sam_num_points_per_batch=int(args.sam_num_points_per_batch),
            suppress_small_objects=bool(args.suppress_small_objects),
        )

    summary_lines = [
        f"dataset_root={dataset_path}",
        f"checkpoint={checkpoint_path}",
        f"timestep=t{timestep}",
        f"device={device}",
        f"stage={args.stage}",
    ]
    if num_rendered is not None:
        summary_lines.append(f"num_rendered_views={int(num_rendered)}")
    if num_segmented is not None:
        summary_lines.append(f"num_segmented_views={int(num_segmented)}")

    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "README.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return run_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render one timestep's training views from a checkpoint, run SAM on both rendered "
            "images and ground-truth training images, and save side-by-side comparisons."
        )
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/root/autodl-tmp/cl-splats-reproduction-main/data/cl-splats/WAT/car_resized",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/root/autodl-tmp/cl-splats-reproduction-main/outputs/checkpoint_t0.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/cl-splats-reproduction-main/outputs",
    )
    parser.add_argument("--timestep", type=int, default=1)
    parser.add_argument("--stage", type=str, default="all", choices=["all", "render", "sam"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resolution_scale", type=float, default=1.0)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--prefer_dist", action="store_true")
    parser.add_argument("--deva_root", type=str, default=None)
    parser.add_argument("--size", type=int, default=480)
    parser.add_argument("--sam_pred_iou_threshold", type=float, default=0.7)
    parser.add_argument("--sam_variant", type=str, default="original")
    parser.add_argument("--sam_num_points_per_side", type=int, default=32)
    parser.add_argument("--sam_num_points_per_batch", type=int, default=64)
    parser.add_argument("--suppress_small_objects", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    out_dir = run(args)
    print("=" * 80)
    print("Finished compare_t1_train_sam_from_checkpoint")
    print(f"Saved outputs to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
