import argparse
import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TimestepEntry:
    timestep: int
    images_dir: Path
    gray_output_dir: Path
    color_output_dir: Path


def _parse_timestep_name(name: str) -> Optional[int]:
    if not name.startswith("t"):
        return None
    suffix = name[1:]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _list_timestep_children(root: Path) -> List[Tuple[int, Path]]:
    items: List[Tuple[int, Path]] = []
    if not root.is_dir():
        return items
    for child in root.iterdir():
        timestep = _parse_timestep_name(child.name)
        if child.is_dir() and timestep is not None:
            items.append((timestep, child))
    items.sort(key=lambda x: x[0])
    return items


def _discover_timestep_entries(dataset_root: Path) -> List[TimestepEntry]:
    image_root = dataset_root / "images_undist"
    if not image_root.is_dir():
        return []

    entries: List[TimestepEntry] = []
    for timestep, images_dir in _list_timestep_children(image_root):
        entries.append(
            TimestepEntry(
                timestep=timestep,
                images_dir=images_dir,
                gray_output_dir=dataset_root / "object_mask" / f"t{timestep}",
                color_output_dir=dataset_root / "object_mask_color" / f"t{timestep}",
            )
        )
    return entries


def _validate_images_dir(images_dir: Path) -> None:
    image_files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix in VALID_IMAGE_SUFFIXES]
    if not image_files:
        raise RuntimeError(f"No valid image files found in {images_dir}")


def _is_deva_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "deva").is_dir()
        and (path / "deva" / "ext" / "automatic_sam.py").exists()
    )


def _is_dataset_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        return len(_discover_timestep_entries(path)) > 0
    except Exception:
        return False


def _dedupe_paths(paths: Sequence[Path]) -> List[Path]:
    result: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(resolved)
    return result


def resolve_deva_root(deva_root_arg: Optional[str]) -> Path:
    if deva_root_arg:
        candidate = Path(deva_root_arg).expanduser().resolve()
        if _is_deva_root(candidate):
            return candidate
        raise FileNotFoundError(
            "deva_root is invalid. Expected Tracking-Anything-with-DEVA root containing "
            f"deva/ext/automatic_sam.py: {candidate}"
        )

    env_value = os.environ.get("DEVA_ROOT") or os.environ.get("CLSPLATS_DEVA_ROOT")
    candidates: List[Path] = []
    if env_value:
        candidates.append(Path(env_value))

    candidates.extend(
        [
            REPO_ROOT / "gaussian-grouping-main" / "Tracking-Anything-with-DEVA",
            REPO_ROOT
            / "ablation_backups"
            / "lora_delete_ad"
            / "gaussian-grouping-main"
            / "Tracking-Anything-with-DEVA",
            REPO_ROOT / "Tracking-Anything-with-DEVA",
            Path.cwd() / "Tracking-Anything-with-DEVA",
            Path.cwd().parent / "Tracking-Anything-with-DEVA",
        ]
    )

    for candidate in _dedupe_paths(candidates):
        if _is_deva_root(candidate):
            return candidate

    checked = ", ".join(str(p) for p in _dedupe_paths(candidates))
    raise FileNotFoundError(
        "Could not auto-locate Tracking-Anything-with-DEVA. "
        "Pass --deva_root explicitly or set DEVA_ROOT / CLSPLATS_DEVA_ROOT. "
        f"Checked: {checked}"
    )


def resolve_dataset_root(dataset_root_arg: Optional[str], scene_arg: Optional[str]) -> Path:
    if dataset_root_arg:
        candidate = Path(dataset_root_arg).expanduser().resolve()
        if _is_dataset_root(candidate):
            return candidate
        raise FileNotFoundError(
            "dataset_root is invalid. Expected a scene directory containing "
            f"images_undist/t0, images_undist/t1, ... : {candidate}"
        )

    env_value = (
        os.environ.get("CLSPLATS_DATASET_ROOT")
        or os.environ.get("DATA_ROOT")
        or os.environ.get("DATASET_ROOT")
    )
    scene_name = scene_arg or os.environ.get("SCENE")

    candidates: List[Path] = []
    if env_value:
        env_root = Path(env_value)
        candidates.append(env_root)
        if scene_name:
            candidates.append(env_root / scene_name)

    if scene_name:
        candidates.extend(
            [
                REPO_ROOT / "data" / "cl-splats" / "WAT" / scene_name,
                REPO_ROOT / "data" / scene_name,
            ]
        )

    for candidate in _dedupe_paths(candidates):
        if _is_dataset_root(candidate):
            return candidate

    checked = ", ".join(str(p) for p in _dedupe_paths(candidates))
    raise FileNotFoundError(
        "Could not auto-locate a scene directory with images_undist/t0... layout. "
        "Pass --dataset_root explicitly or set CLSPLATS_DATASET_ROOT / DATA_ROOT. "
        f"Checked: {checked}"
    )


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_automatic_sam_api(deva_root: Path):
    auto_sam_module = _load_module_from_file(
        "_clsplats_deva_automatic_sam",
        deva_root / "deva" / "ext" / "automatic_sam.py",
    )
    return auto_sam_module.get_sam_model, auto_sam_module.auto_segment


def _read_image_rgb(image_path: Path) -> np.ndarray:
    return np.array(Image.open(image_path).convert("RGB"))


def _save_gray_mask(mask: torch.Tensor, output_path: Path) -> None:
    mask_np = mask.detach().cpu().numpy().astype(np.uint16)
    Image.fromarray(mask_np).save(output_path)


def _resize_mask_to_shape(mask: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if tuple(mask.shape[-2:]) == (target_h, target_w):
        return mask

    resized = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0).float(),
        size=(target_h, target_w),
        mode="nearest",
    )[0, 0]
    return resized.to(dtype=mask.dtype)


def _colorize_mask(mask: torch.Tensor) -> np.ndarray:
    mask_np = mask.detach().cpu().numpy().astype(np.int64)
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    unique_ids = np.unique(mask_np)
    for obj_id in unique_ids:
        if int(obj_id) <= 0:
            continue
        # Deterministic pseudo-coloring for easy visual inspection.
        color = np.array(
            [
                (37 * int(obj_id) + 17) % 255,
                (67 * int(obj_id) + 29) % 255,
                (97 * int(obj_id) + 71) % 255,
            ],
            dtype=np.uint8,
        )
        rgb[mask_np == obj_id] = color
    return rgb


def _save_color_overlay(image_np: np.ndarray, mask: torch.Tensor, output_path: Path) -> None:
    rgb_mask = _colorize_mask(mask)
    alpha = (rgb_mask.sum(axis=2, keepdims=True) > 0).astype(np.float32) * 0.45
    blend = (image_np.astype(np.float32) * (1.0 - alpha) + rgb_mask.astype(np.float32) * alpha).astype(np.uint8)
    Image.fromarray(blend).save(output_path)


def _build_sam_config(
    deva_root: Path,
    *,
    size: int,
    sam_pred_iou_threshold: float,
    sam_variant: str,
    sam_num_points_per_side: int,
    sam_num_points_per_batch: int,
    suppress_small_objects: bool,
) -> dict:
    return {
        "SAM_ENCODER_VERSION": "vit_h",
        "SAM_CHECKPOINT_PATH": str(deva_root / "saves" / "sam_vit_h_4b8939.pth"),
        "MOBILE_SAM_CHECKPOINT_PATH": str(deva_root / "saves" / "mobile_sam.pt"),
        "SAM_NUM_POINTS_PER_SIDE": int(sam_num_points_per_side),
        "SAM_NUM_POINTS_PER_BATCH": int(sam_num_points_per_batch),
        "SAM_PRED_IOU_THRESHOLD": float(sam_pred_iou_threshold),
        "SAM_OVERLAP_THRESHOLD": 0.8,
        "sam_variant": str(sam_variant),
        "size": int(size),
        "suppress_small_objects": bool(suppress_small_objects),
    }


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def prepare_masks_for_dataset(
    dataset_root: Path,
    *,
    deva_root: Path,
    size: int,
    sam_pred_iou_threshold: float,
    sam_variant: str,
    sam_num_points_per_side: int,
    sam_num_points_per_batch: int,
    suppress_small_objects: bool,
) -> None:
    timestep_entries = _discover_timestep_entries(dataset_root)
    if not timestep_entries:
        raise RuntimeError(
            "No timestep image folders found. Expected:\n"
            f"  {dataset_root}/images_undist/t0\n"
            f"  {dataset_root}/images_undist/t1\n"
            f"  ..."
        )

    get_sam_model, auto_segment = _load_automatic_sam_api(deva_root)
    config = _build_sam_config(
        deva_root,
        size=size,
        sam_pred_iou_threshold=sam_pred_iou_threshold,
        sam_variant=sam_variant,
        sam_num_points_per_side=sam_num_points_per_side,
        sam_num_points_per_batch=sam_num_points_per_batch,
        suppress_small_objects=suppress_small_objects,
    )

    print("=" * 80)
    print("[prepare_clsplats_object_masks] Pure SAM mode")
    print(f"[prepare_clsplats_object_masks] SAM variant            : {sam_variant}")
    print(f"[prepare_clsplats_object_masks] SAM points per side    : {sam_num_points_per_side}")
    print(f"[prepare_clsplats_object_masks] SAM points per batch   : {sam_num_points_per_batch}")
    print(f"[prepare_clsplats_object_masks] SAM pred IoU threshold : {sam_pred_iou_threshold}")
    print("=" * 80)

    sam_model = get_sam_model(config, "cuda")

    for entry in timestep_entries:
        timestep = entry.timestep
        images_dir = entry.images_dir
        _validate_images_dir(images_dir)

        if entry.gray_output_dir.exists():
            shutil.rmtree(entry.gray_output_dir)
        if entry.color_output_dir.exists():
            shutil.rmtree(entry.color_output_dir)
        entry.gray_output_dir.mkdir(parents=True, exist_ok=True)
        entry.color_output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            [p for p in images_dir.iterdir() if p.is_file() and p.suffix in VALID_IMAGE_SUFFIXES]
        )

        print("=" * 80)
        print(f"[prepare_clsplats_object_masks] Processing timestep t{timestep}")
        print(f"[prepare_clsplats_object_masks] Images dir : {images_dir}")
        print("=" * 80)

        for image_path in image_paths:
            image_np = _read_image_rgb(image_path)
            mask, _, _ = auto_segment(
                config,
                sam_model,
                image_np,
                forward_mask=None,
                min_side=size,
                suppress_small_mask=suppress_small_objects,
            )
            mask = _resize_mask_to_shape(mask, image_np.shape[:2])

            gray_output_path = entry.gray_output_dir / f"{image_path.stem}.png"
            color_output_path = entry.color_output_dir / f"{image_path.stem}.jpg"
            _ensure_parent(gray_output_path)
            _ensure_parent(color_output_path)
            _save_gray_mask(mask, gray_output_path)
            _save_color_overlay(image_np, mask, color_output_path)

        print(
            f"[prepare_clsplats_object_masks] Saved timestep t{timestep} masks to:\n"
            f"  color -> {entry.color_output_dir}\n"
            f"  gray  -> {entry.gray_output_dir}"
        )

    print("=" * 80)
    print("[prepare_clsplats_object_masks] All timesteps finished.")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare per-timestep object masks for CL-Splats using pure Automatic SAM only."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help=(
            "Path to a scene root containing images_undist/t0, images_undist/t1, ... . "
            "If omitted, auto-resolve from CLSPLATS_DATASET_ROOT / DATA_ROOT and repo-local candidates."
        ),
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Optional scene name used for dataset-root auto resolution.",
    )
    parser.add_argument(
        "--deva_root",
        type=str,
        default=None,
        help=(
            "Path to Tracking-Anything-with-DEVA root. "
            "Used here only to reuse its SAM loading code and checkpoints."
        ),
    )
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--sam_pred_iou_threshold", type=float, default=0.7)
    parser.add_argument("--sam_variant", type=str, default="original")
    parser.add_argument("--sam_num_points_per_side", type=int, default=32)
    parser.add_argument("--sam_num_points_per_batch", type=int, default=8)
    parser.add_argument("--suppress_small_objects", action="store_true")
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.dataset_root, args.scene)
    deva_root = resolve_deva_root(args.deva_root)

    print("=" * 80)
    print("[prepare_clsplats_object_masks] Resolved paths")
    print(f"[prepare_clsplats_object_masks] Repo root    : {REPO_ROOT}")
    print(f"[prepare_clsplats_object_masks] Dataset root : {dataset_root}")
    print(f"[prepare_clsplats_object_masks] DEVA root    : {deva_root}")
    print("=" * 80)

    prepare_masks_for_dataset(
        dataset_root,
        deva_root=deva_root,
        size=args.size,
        sam_pred_iou_threshold=args.sam_pred_iou_threshold,
        sam_variant=args.sam_variant,
        sam_num_points_per_side=args.sam_num_points_per_side,
        sam_num_points_per_batch=args.sam_num_points_per_batch,
        suppress_small_objects=bool(args.suppress_small_objects),
    )


if __name__ == "__main__":
    main()
