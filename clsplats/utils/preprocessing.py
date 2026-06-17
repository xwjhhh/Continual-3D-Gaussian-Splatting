import os
import re
import subprocess
import shutil
from pathlib import Path
import argparse


def run_colmap(args):
    """Run colmap command with shell=True for Windows compatibility."""
    cmd = " ".join(f'"{a}"' if " " in str(a) else str(a) for a in args)
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def check_and_run_colmap(input_dir):
    """
    Process multi-timestep dataset for CL-Splats training.
    
    Input structure:
        input_dir/
        ├── t1/  (first timestep images)
        ├── t2/  (second timestep images)
        └── ...
    
    Output structure:
        input_dir/
        └── output/
            ├── t0/
            │   ├── images/
            │   └── sparse/0/
            ├── t1/
            │   ├── images/
            │   └── sparse/0/
            └── ...
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"{input_dir} is not a valid directory.")

    # Find subfolders named t* (t followed by a number)
    t_dirs = [d for d in input_dir.iterdir() if d.is_dir() and re.match(r"t\d+$", d.name)]
    if not t_dirs:
        raise ValueError("No subdirectories matching 't*' found.")

    # Sort t_dirs by number
    t_dirs.sort(key=lambda d: int(d.name[1:]))
    print(f"Found {len(t_dirs)} timesteps: {[d.name for d in t_dirs]}")

    # Check each t* subdir contains only images (jpg/png)
    for t_dir in t_dirs:
        files = list(t_dir.iterdir())
        if not files:
            raise ValueError(f"{t_dir} is empty.")
        for f in files:
            if not (f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]):
                raise ValueError(f"Non-image file found in {t_dir}: {f.name}")

    # Prepare workspace
    workspace = input_dir / "colmap_workspace"
    all_images_dir = workspace / "all_images"
    sparse_dir = workspace / "sparse"
    database_path = workspace / "database.db"
    workspace.mkdir(exist_ok=True)
    all_images_dir.mkdir(exist_ok=True)
    sparse_dir.mkdir(exist_ok=True)

    # Track which images belong to which timestep
    timestep_images = {}
    
    # ========== Step 1: Process t0 (base reconstruction) ==========
    print("\n" + "="*60)
    print("Step 1: Processing t0 (base reconstruction)")
    print("="*60)
    
    t0_dir = t_dirs[0]
    timestep_images[0] = []
    
    for img in t0_dir.iterdir():
        target = all_images_dir / img.name
        if not target.exists():
            shutil.copy2(img, target)
        timestep_images[0].append(img.name)

    # Feature extraction
    run_colmap([
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(all_images_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "OPENCV",
    ])

    # Exhaustive matcher
    run_colmap([
        "colmap", "exhaustive_matcher",
        "--database_path", str(database_path),
    ])

    # Mapper
    run_colmap([
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(all_images_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.ba_global_function_tolerance", "0.000001"
    ])

    mapper_output = sparse_dir / "0"
    if not mapper_output.exists():
        raise RuntimeError("No sparse reconstruction found.")

    # ========== Step 2: Register additional timesteps ==========
    for t_idx, t_dir in enumerate(t_dirs[1:], start=1):
        print("\n" + "="*60)
        print(f"Step 2.{t_idx}: Registering {t_dir.name}")
        print("="*60)
        
        timestep_images[t_idx] = []
        
        # Copy new images
        for img in t_dir.iterdir():
            target = all_images_dir / img.name
            if not target.exists():
                shutil.copy2(img, target)
            timestep_images[t_idx].append(img.name)

        # Feature extraction for new images
        run_colmap([
            "colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(all_images_dir),
            "--ImageReader.single_camera", "1"
        ])

        # Exhaustive matcher
        run_colmap([
            "colmap", "exhaustive_matcher",
            "--database_path", str(database_path),
        ])

        # Register new images
        run_colmap([
            "colmap", "image_registrator",
            "--database_path", str(database_path),
            "--input_path", str(mapper_output),
            "--output_path", str(mapper_output)
        ])

        # Bundle adjustment
        run_colmap([
            "colmap", "bundle_adjuster",
            "--input_path", str(mapper_output),
            "--output_path", str(mapper_output)
        ])

    # ========== Step 3: Undistort all images ==========
    print("\n" + "="*60)
    print("Step 3: Undistorting all images")
    print("="*60)
    
    undistorted_dir = workspace / "undistorted_all"
    run_colmap([
        "colmap", "image_undistorter",
        "--image_path", str(all_images_dir),
        "--input_path", str(mapper_output),
        "--output_path", str(undistorted_dir),
        "--output_type", "COLMAP"
    ])

    # ========== Step 4: Create per-timestep output structure ==========
    print("\n" + "="*60)
    print("Step 4: Creating per-timestep output structure")
    print("="*60)
    
    output_dir = input_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Read the undistorted sparse model
    undistorted_sparse = undistorted_dir / "sparse"
    
    for t_idx in range(len(t_dirs)):
        print(f"Creating t{t_idx} structure...")
        
        t_output = output_dir / f"t{t_idx}"
        t_images = t_output / "images"
        t_sparse = t_output / "sparse" / "0"
        
        t_output.mkdir(exist_ok=True)
        t_images.mkdir(exist_ok=True)
        t_sparse.mkdir(parents=True, exist_ok=True)
        
        # Copy images for this timestep
        for img_name in timestep_images[t_idx]:
            src = undistorted_dir / "images" / img_name
            dst = t_images / img_name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        
        # Copy sparse model (cameras.bin, points3D.bin)
        # For images.bin, we need to filter to only include this timestep's images
        for fname in ["cameras.bin", "points3D.bin"]:
            src = undistorted_sparse / fname
            dst = t_sparse / fname
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        
        # For images.bin, copy the full one (filtering would require parsing binary format)
        # The training code will only use images that exist in the images folder
        src = undistorted_sparse / "images.bin"
        dst = t_sparse / "images.bin"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print(f"Output directory: {output_dir}")
    print(f"Timesteps created: {len(t_dirs)}")
    print("="*60)
    print("\nTo train, run:")
    print(f"  python -m clsplats.train dataset.path={output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Process multi-timestep dataset for CL-Splats training."
    )
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True,
        help="Input directory containing t1, t2, ... subfolders with images."
    )
    args = parser.parse_args()
    check_and_run_colmap(args.input_dir)


if __name__ == "__main__":
    main()
