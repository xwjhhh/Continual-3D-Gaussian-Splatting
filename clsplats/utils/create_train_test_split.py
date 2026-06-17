#!/usr/bin/env python3
"""
Create a persistent train/test split file for CL-Splats datasets.

The split is saved under:
  <dataset_path>/train_test_split.json

This file is consumed by `CLSplatsDataset` when `eval_mode=True`, so train and
eval will use the same deterministic split instead of re-sampling each run.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from clsplats.dataset import CLSplatsDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create persistent train/test split file for a CL-Splats dataset."
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Dataset root path (e.g. /root/.../data/cl-splats/WAT/breville).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.125,
        help="Test split ratio in [0,1]. Default: 0.125 (1/8).",
    )
    parser.add_argument(
        "--min-test",
        type=int,
        default=1,
        help="Minimum number of test images per timestep when possible. Default: 1.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed used for split generation (per timestep uses seed+timestep).",
    )
    parser.add_argument(
        "--prefer-undist",
        action="store_true",
        help="Prefer images_undist/sparse_undist (default behavior).",
    )
    parser.add_argument(
        "--prefer-dist",
        action="store_true",
        help="Prefer images/sparse instead of undistorted folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split file.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional custom output json path. Default: <dataset_path>/train_test_split.json",
    )
    return parser


def _compute_num_test(num_items: int, test_ratio: float, min_test: int) -> int:
    if num_items <= 1 or test_ratio <= 0.0:
        return 0

    num_test = int(num_items * test_ratio)
    num_test = max(min_test, num_test)
    # Keep at least one training sample.
    num_test = min(num_test, num_items - 1)
    return num_test


def main() -> None:
    args = build_parser().parse_args()

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    if args.test_ratio < 0.0 or args.test_ratio > 1.0:
        raise ValueError("--test-ratio must be in [0, 1].")

    if args.min_test < 0:
        raise ValueError("--min-test must be >= 0.")

    if args.prefer_dist and args.prefer_undist:
        raise ValueError("--prefer-dist and --prefer-undist cannot be used together.")
    prefer_undist = not args.prefer_dist

    output_file = (
        Path(args.output_file).expanduser().resolve()
        if args.output_file
        else dataset_path / "train_test_split.json"
    )
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"Split file already exists: {output_file}. Use --overwrite to replace it."
        )

    dataset = CLSplatsDataset(
        path=str(dataset_path),
        resolution_scale=1.0,
        white_background=False,
        eval_mode=False,
        device="cpu",
        split_seed=args.split_seed,
        prefer_undist=prefer_undist,
    )

    split_payload = {
        "version": 1,
        "dataset_path": str(dataset_path),
        "dataset_type": dataset.dataset_type,
        "split_seed": args.split_seed,
        "test_ratio": args.test_ratio,
        "timesteps": {},
    }

    print("=" * 72)
    print("Create Train/Test Split")
    print("=" * 72)
    print(f"Dataset path : {dataset_path}")
    print(f"Output file  : {output_file}")
    print(f"Timesteps    : {dataset.get_num_timesteps()}")
    print(f"Test ratio   : {args.test_ratio}")
    print(f"Min test     : {args.min_test}")
    print(f"Split seed   : {args.split_seed}")
    print("-" * 72)

    total_train = 0
    total_test = 0

    for timestep in range(dataset.get_num_timesteps()):
        scene_info = dataset.get_scene_info(timestep)
        cam_infos = list(scene_info.train_cameras)

        num_total = len(cam_infos)
        num_test = _compute_num_test(num_total, args.test_ratio, args.min_test)

        indices = list(range(num_total))
        rng = random.Random(args.split_seed + timestep)
        rng.shuffle(indices)
        test_indices = set(indices[:num_test])

        train_names = [
            cam_infos[idx].image_name for idx in range(num_total) if idx not in test_indices
        ]
        test_names = [
            cam_infos[idx].image_name for idx in range(num_total) if idx in test_indices
        ]

        split_payload["timesteps"][str(timestep)] = {
            "train": train_names,
            "test": test_names,
        }

        total_train += len(train_names)
        total_test += len(test_names)
        print(
            f"t{timestep}: total={num_total}, train={len(train_names)}, test={len(test_names)}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2, ensure_ascii=True)

    print("-" * 72)
    print(f"Saved split file: {output_file}")
    print(f"Global counts   : train={total_train}, test={total_test}")
    print("=" * 72)


if __name__ == "__main__":
    main()

