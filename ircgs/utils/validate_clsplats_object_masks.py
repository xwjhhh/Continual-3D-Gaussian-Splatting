import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def _parse_timestep_name(name: str) -> Optional[int]:
    if not name.startswith("t"):
        return None
    suffix = name[1:]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _list_timestep_children(root: Path) -> List[Tuple[int, Path]]:
    result: List[Tuple[int, Path]] = []
    if not root.is_dir():
        return result
    for child in root.iterdir():
        timestep = _parse_timestep_name(child.name)
        if child.is_dir() and timestep is not None:
            result.append((timestep, child))
    result.sort(key=lambda x: x[0])
    return result


def _discover_layout(dataset_root: Path) -> List[Tuple[int, Path, Path, Path]]:
    image_root = dataset_root / "images_undist"
    if not image_root.is_dir():
        return []

    entries: List[Tuple[int, Path, Path, Path]] = []
    for timestep, images_dir in _list_timestep_children(image_root):
        entries.append(
            (
                timestep,
                images_dir,
                dataset_root / "object_mask" / f"t{timestep}",
                dataset_root / "object_mask_color" / f"t{timestep}",
            )
        )
    return entries


def _basename_set(folder: Path, valid_suffixes: set[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if not folder.exists():
        return mapping
    for item in folder.iterdir():
        if item.is_file() and item.suffix in valid_suffixes:
            mapping[item.stem] = item
    return mapping


def validate_dataset_masks(dataset_root: Path) -> int:
    entries = _discover_layout(dataset_root)
    if not entries:
        print(f"[validate_clsplats_object_masks] No images_undist/timestep layout found under {dataset_root}")
        return 1

    total_missing_masks = 0
    total_extra_masks = 0
    total_timesteps_without_masks = 0

    print("=" * 80)
    print(f"[validate_clsplats_object_masks] Dataset root: {dataset_root}")
    print("[validate_clsplats_object_masks] Layout      : images_undist/t0 + object_mask/t0")
    print("=" * 80)

    for timestep, images_dir, object_mask_dir, object_mask_color_dir in entries:
        image_map = _basename_set(images_dir, VALID_IMAGE_SUFFIXES)
        mask_map = _basename_set(object_mask_dir, {".png", ".PNG"})
        color_map = _basename_set(object_mask_color_dir, VALID_IMAGE_SUFFIXES | {".jpg", ".JPG"})

        image_names = set(image_map.keys())
        mask_names = set(mask_map.keys())
        color_names = set(color_map.keys())

        missing_masks = sorted(image_names - mask_names)
        extra_masks = sorted(mask_names - image_names)
        missing_color = sorted(image_names - color_names)

        if not object_mask_dir.exists():
            total_timesteps_without_masks += 1

        total_missing_masks += len(missing_masks)
        total_extra_masks += len(extra_masks)

        print(f"[t{timestep}]")
        print(f"  images dir           : {images_dir}")
        print(f"  object_mask dir      : {object_mask_dir}")
        print(f"  object_mask_color dir: {object_mask_color_dir}")
        print(f"  images               : {len(image_names)}")
        print(f"  object_mask          : {len(mask_names)}")
        print(f"  object_mask_color    : {len(color_names)}")
        print(f"  missing gray masks   : {len(missing_masks)}")
        print(f"  extra gray masks     : {len(extra_masks)}")
        print(f"  missing color masks  : {len(missing_color)}")

        if missing_masks[:5]:
            print(f"  examples missing gray: {missing_masks[:5]}")
        if extra_masks[:5]:
            print(f"  examples extra gray  : {extra_masks[:5]}")
        if missing_color[:5]:
            print(f"  examples missing vis : {missing_color[:5]}")

    print("=" * 80)
    print("[validate_clsplats_object_masks] Summary")
    print(f"  timesteps without object_mask dir : {total_timesteps_without_masks}")
    print(f"  total missing gray masks          : {total_missing_masks}")
    print(f"  total extra gray masks            : {total_extra_masks}")
    print("=" * 80)

    if total_timesteps_without_masks > 0 or total_missing_masks > 0:
        return 2
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate per-timestep object_mask outputs for a CL-Splats dataset."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to scene root containing images_undist/t0, images_undist/t1, ...",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    raise SystemExit(validate_dataset_masks(dataset_root))


if __name__ == "__main__":
    main()
