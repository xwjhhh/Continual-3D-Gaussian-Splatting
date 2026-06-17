#!/usr/bin/env python3
"""
Undistort a CL-Splats scene using official 3DGS convert.py workflow.

Input scene layout (distorted):
  scene/
    images/...
    sparse/0/...
    database.db (optional)

Output layout (undistorted, written into same scene directory):
  scene/
    images_undist/...
    sparse_undist/0/...
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run official 3DGS convert.py --skip_matching and export images_undist/sparse_undist."
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Scene root path, e.g. data/cl-splats/WAT/breville",
    )
    parser.add_argument(
        "--convert-script",
        required=True,
        help="Path to official gaussian-splatting convert.py",
    )
    parser.add_argument(
        "--python-exec",
        default=sys.executable,
        help="Python executable used to run convert.py (default: current python).",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Optional workspace directory for convert.py. Default: <dataset>/.undistort_3dgs_workspace",
    )
    parser.add_argument(
        "--input-mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to place input images into workspace/input (default: symlink).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing images_undist/sparse_undist if they already exist.",
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Keep temporary workspace after successful conversion.",
    )
    return parser


def _safe_remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _ensure_dist_inputs(dataset_path: Path) -> tuple[Path, Path]:
    images_dir = dataset_path / "images"
    sparse0_dir = dataset_path / "sparse" / "0"
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing distorted images directory: {images_dir}")
    if not sparse0_dir.exists():
        raise FileNotFoundError(f"Missing distorted sparse model directory: {sparse0_dir}")
    return images_dir, sparse0_dir


def _prepare_input_images(src_images: Path, workspace_input: Path, mode: str) -> None:
    if workspace_input.exists():
        _safe_remove(workspace_input)

    if mode == "symlink":
        try:
            workspace_input.symlink_to(src_images, target_is_directory=True)
            return
        except OSError:
            print("Warning: symlink failed, fallback to copy mode.")

    shutil.copytree(src_images, workspace_input)


def _prepare_workspace(dataset_path: Path, workspace: Path, input_mode: str) -> None:
    src_images, src_sparse0 = _ensure_dist_inputs(dataset_path)

    if workspace.exists():
        _safe_remove(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    # 3DGS convert.py expects:
    #   workspace/input
    #   workspace/distorted/sparse/0
    workspace_input = workspace / "input"
    _prepare_input_images(src_images, workspace_input, mode=input_mode)

    distorted_sparse0 = workspace / "distorted" / "sparse" / "0"
    distorted_sparse0.mkdir(parents=True, exist_ok=True)
    for file in src_sparse0.iterdir():
        if file.is_file():
            shutil.copy2(file, distorted_sparse0 / file.name)

    src_db = dataset_path / "database.db"
    if src_db.exists():
        (workspace / "distorted").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_db, workspace / "distorted" / "database.db")


def _run_convert(workspace: Path, convert_script: Path, python_exec: str) -> None:
    cmd = [
        python_exec,
        str(convert_script),
        "-s",
        str(workspace),
        "--skip_matching",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _export_outputs(dataset_path: Path, workspace: Path, overwrite: bool) -> None:
    src_images = workspace / "images"
    src_sparse = workspace / "sparse"
    src_sparse0 = src_sparse / "0"

    if not src_images.exists():
        raise FileNotFoundError(f"convert.py output missing: {src_images}")
    if not src_sparse0.exists():
        raise FileNotFoundError(f"convert.py output missing: {src_sparse0}")

    out_images = dataset_path / "images_undist"
    out_sparse = dataset_path / "sparse_undist"

    if out_images.exists() or out_sparse.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {out_images} or {out_sparse}. "
                "Use --overwrite to replace."
            )
        _safe_remove(out_images)
        _safe_remove(out_sparse)

    shutil.copytree(src_images, out_images)
    shutil.copytree(src_sparse, out_sparse)


def main() -> None:
    args = build_parser().parse_args()

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    convert_script = Path(args.convert_script).expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if not convert_script.exists():
        raise FileNotFoundError(f"convert.py not found: {convert_script}")

    workspace = (
        Path(args.workspace).expanduser().resolve()
        if args.workspace
        else dataset_path / ".undistort_3dgs_workspace"
    )

    print("=" * 72)
    print("3DGS Undistortion (convert.py --skip_matching)")
    print("=" * 72)
    print(f"Dataset path : {dataset_path}")
    print(f"convert.py   : {convert_script}")
    print(f"Workspace    : {workspace}")
    print(f"Input mode   : {args.input_mode}")
    print("-" * 72)

    try:
        _prepare_workspace(dataset_path, workspace, args.input_mode)
        _run_convert(workspace, convert_script, args.python_exec)
        _export_outputs(dataset_path, workspace, args.overwrite)
    finally:
        if not args.keep_workspace and workspace.exists():
            _safe_remove(workspace)

    print("Done.")
    print(f"Undistorted images: {dataset_path / 'images_undist'}")
    print(f"Undistorted sparse: {dataset_path / 'sparse_undist' / '0'}")


if __name__ == "__main__":
    main()

