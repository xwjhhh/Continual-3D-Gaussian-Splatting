import argparse
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clsplats.utils.prepare_clsplats_object_masks import (  # noqa: E402
    VALID_IMAGE_SUFFIXES,
    _build_sam_config,
    _load_automatic_sam_api,
    _read_image_rgb,
    _resize_mask_to_shape,
    _save_color_overlay,
    _save_gray_mask,
    resolve_deva_root,
)


def _collect_image_paths(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix not in VALID_IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image suffix: {input_path.suffix}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    image_paths = sorted(
        [p for p in input_path.iterdir() if p.is_file() and p.suffix in VALID_IMAGE_SUFFIXES]
    )
    if not image_paths:
        raise RuntimeError(f"No valid images found in {input_path}")
    return image_paths


def _default_output_root(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.parent / f"{input_path.stem}_sam_outputs"
    return input_path.parent / f"{input_path.name}_sam_outputs"


def run_sam_on_images(
    image_paths: List[Path],
    *,
    output_root: Path,
    deva_root: Path,
    size: int,
    sam_pred_iou_threshold: float,
    sam_variant: str,
    sam_num_points_per_side: int,
    sam_num_points_per_batch: int,
    suppress_small_objects: bool,
) -> None:
    output_gray_dir = output_root / "object_mask"
    output_color_dir = output_root / "object_mask_color"
    output_gray_dir.mkdir(parents=True, exist_ok=True)
    output_color_dir.mkdir(parents=True, exist_ok=True)

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
    sam_model = get_sam_model(config, "cuda")

    print("=" * 80)
    print("[run_sam_on_folder] Pure SAM folder mode")
    print(f"[run_sam_on_folder] Output root            : {output_root}")
    print(f"[run_sam_on_folder] SAM variant            : {sam_variant}")
    print(f"[run_sam_on_folder] SAM points per side    : {sam_num_points_per_side}")
    print(f"[run_sam_on_folder] SAM points per batch   : {sam_num_points_per_batch}")
    print(f"[run_sam_on_folder] SAM pred IoU threshold : {sam_pred_iou_threshold}")
    print(f"[run_sam_on_folder] Num images             : {len(image_paths)}")
    print("=" * 80)

    for image_path in image_paths:
        print(f"[run_sam_on_folder] Processing: {image_path}")
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

        gray_output_path = output_gray_dir / f"{image_path.stem}.png"
        color_output_path = output_color_dir / f"{image_path.stem}.jpg"
        _save_gray_mask(mask, gray_output_path)
        _save_color_overlay(image_np, mask, color_output_path)

    print("=" * 80)
    print("[run_sam_on_folder] Finished.")
    print(f"[run_sam_on_folder] Gray masks : {output_gray_dir}")
    print(f"[run_sam_on_folder] Visuals    : {output_color_dir}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pure Automatic SAM on one image or one image folder."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to one image file or one folder containing images.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Output folder. Default: create a sibling folder named *_sam_outputs.",
    )
    parser.add_argument(
        "--deva_root",
        type=str,
        default=None,
        help="Tracking-Anything-with-DEVA root, used to reuse SAM loading code/checkpoints.",
    )
    parser.add_argument("--size", type=int, default=480)
    parser.add_argument("--sam_pred_iou_threshold", type=float, default=0.7)
    parser.add_argument("--sam_variant", type=str, default="original")
    parser.add_argument("--sam_num_points_per_side", type=int, default=64)
    parser.add_argument("--sam_num_points_per_batch", type=int, default=8)
    parser.add_argument("--suppress_small_objects", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root is not None
        else _default_output_root(input_path).resolve()
    )
    deva_root = resolve_deva_root(args.deva_root)
    image_paths = _collect_image_paths(input_path)

    run_sam_on_images(
        image_paths,
        output_root=output_root,
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
