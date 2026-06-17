# Open-Source Release Package

This directory is a cleaned release package for three methods:

- `ours`: our Scaffold-GS based continual update method.
- `pure-3dgs`: sequential vanilla 3D Gaussian Splatting baseline.
- `4dgs`: 4D Gaussian Splatting all-timestep baseline.

The package includes the local CL-Splats/3DGS code, the 3DGS CUDA extension
submodules, `Scaffold-GS-main/`, and `4DGaussians-master/`. Experiment outputs,
datasets, checkpoints, caches, archives, and paper notes were left out.

## Layout

```text
source-code/
  clsplats/
    models/
      ours.py
      pure_3dgs.py
      4dgs.py
      model_factory.py
  configs/
  submodules/
    diff-gaussian-rasterization/
    simple-knn/
  Scaffold-GS-main/
  4DGaussians-master/
  LICENSE
  3DGS_LICENSE.md
  THIRD_PARTY.md
```

## Installation

```bash
conda env create -f environment_windows.yml
conda activate cl-splats-dev
```

or install Python dependencies with pip:

```bash
pip install -r requirements.txt
```

Then compile the 3DGS CUDA extensions:

```bash
pip install -e submodules/diff-gaussian-rasterization --no-build-isolation
pip install -e submodules/simple-knn --no-build-isolation
```

## Running

```bash
python -m clsplats.train dataset.path=/path/to/processed_colmap_scene model.name=ours
python -m clsplats.train dataset.path=/path/to/processed_colmap_scene model.name=pure-3dgs
python -m clsplats.train dataset.path=/path/to/processed_colmap_scene model.name=4dgs
```

`sfgs`, `scaffold-gs`, and `scaffold_gs` are kept as backward-compatible aliases
for `ours`.

The default config is `configs/cl-splats.yaml`. Hydra overrides can be appended
on the command line.

## License Notes

Keep `LICENSE`, `3DGS_LICENSE.md`, `THIRD_PARTY.md`, and all upstream license
files inside the bundled third-party projects.
