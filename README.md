<div align="center">

# IRC-GS

### Inheritance, Revision, and Consolidation for Continual 3D Gaussian Splatting

[Results](#results) | [Installation](#installation) | [Data](#data) | [Training](#training) | [License](#license)

</div>

IRC-GS is a continual 3D Gaussian Splatting framework for long-lived scene
reconstruction. It treats the rendering residual as the common signal for
three operations: inheriting historical anchors, revising locally unexplained
structure through multi-view residual voting, and consolidating only useful
newborn anchors into scene memory.

On ten World Across Time scenes, IRC-GS reaches **28.14 dB PSNR** and
**0.896 SSIM** with **241.6 MB** average model storage.

## Results

The comparison videos use the same camera trajectory for all displayed
methods. MP4 files are stored with Git LFS; click a scene name to open the
full-resolution result.

| Scene | Video | Scene | Video |
|:--|:--:|:--|:--:|
| Breville | [MP4](videos/breville.mp4) | Car | [MP4](videos/car.mp4) |
| Community | [Watch online](https://htmlpreview.github.io/?https://github.com/xwjhhh/Continual-3D-Gaussian-Splatting/blob/main/docs/index.html) / [MP4](videos/community.mp4) | Grill | [MP4](videos/grill.mp4) |
| Living room | [MP4](videos/living_room.mp4) | Street | [MP4](videos/street.mp4) |

## Supported Methods

The public interface contains IRC-GS and four reproducibility baselines only.

| Method | Command value | Role |
|:--|:--|:--|
| IRC-GS | `irc-gs` | Proposed continual method |
| CL-Splats | `cl-splats` | Continual 3DGS baseline |
| Scaffold-GS | `scaffold-gs` | Independent anchor-based reconstruction |
| 3DGS | `3dgs` | Original Gaussian representation baseline |
| 4DGS | `4dgs` | Spatiotemporal Gaussian baseline |

## Repository Layout

```text
.
|-- clsplats/                 # IRC-GS training and evaluation package
|-- configs/irc-gs.yaml       # Default Hydra experiment configuration
|-- cl-splats-main/           # CL-Splats baseline
|-- Scaffold-GS-main/         # Scaffold-GS baseline and renderer
|-- 4DGaussians-master/       # 4DGS baseline
|-- submodules/               # 3DGS CUDA rasterizer and simple-knn
|-- videos/                   # Qualitative comparisons (Git LFS)
|-- run.sh                    # Unified experiment entry point
|-- train.sh                  # IRC-GS/3DGS/4DGS/Scaffold-GS WAT runner
`-- run_cl_splats.sh          # CL-Splats WAT runner
```

Datasets, checkpoints, experiment outputs, paper drafts, review responses, and
local notes are intentionally excluded from the repository.

## Installation

The tested setup uses Linux, Python 3.11, PyTorch 2.7, CUDA 12, and an NVIDIA
GPU. Create the environment and compile the bundled CUDA extensions:

```bash
conda env create -f environment.yml
conda activate irc-gs

pip install -e submodules/diff-gaussian-rasterization --no-build-isolation
pip install -e submodules/simple-knn --no-build-isolation
pip install -e Scaffold-GS-main/submodules/diff-gaussian-rasterization --no-build-isolation
```

The 4DGS baseline has additional CUDA modules documented in
[`4DGaussians-master/README.md`](4DGaussians-master/README.md).

## Data

Set `DATA_ROOT` to the World Across Time root. Each scene is expected to contain
the timestep image folders and COLMAP reconstruction consumed by the dataset
reader. Data is never tracked by Git.

```text
/path/to/WAT/
|-- breville/
|-- kitchen/
|-- living_room/
`-- ...
```

## Training

Run one method on one or more scenes:

```bash
DATA_ROOT=/path/to/WAT METHOD=irc-gs ONLY_SCENES=breville bash run.sh
DATA_ROOT=/path/to/WAT METHOD=cl-splats ONLY_SCENES=breville bash run.sh
DATA_ROOT=/path/to/WAT METHOD=scaffold-gs ONLY_SCENES=breville bash run.sh
DATA_ROOT=/path/to/WAT METHOD=3dgs ONLY_SCENES=breville bash run.sh
DATA_ROOT=/path/to/WAT METHOD=4dgs ONLY_SCENES=breville bash run.sh
```

Use `METHOD=all` for the complete comparison. Multiple scenes are passed as a
comma-separated list, for example `ONLY_SCENES=breville,kitchen,living_room`.
The main controls can be overridden with environment variables:

```bash
DATA_ROOT=/path/to/WAT \
METHOD=irc-gs \
ONLY_SCENES=breville \
BASE_ITERS=30000 \
INC_ITERS=30000 \
bash run.sh
```

Checkpoints, rendered images, logs, and metric summaries are written below
`outputs/<method>/<scene>/` and remain ignored by Git.

## Git LFS

Install Git LFS before cloning if you need the comparison videos:

```bash
git lfs install
git clone https://github.com/xwjhhh/Continual-3D-Gaussian-Splatting.git
```

A normal source clone without the video payload can use
`GIT_LFS_SKIP_SMUDGE=1`.

## License

Keep [`LICENSE`](LICENSE), [`3DGS_LICENSE.md`](3DGS_LICENSE.md),
[`THIRD_PARTY.md`](THIRD_PARTY.md), and every upstream license with any copy or
redistribution. The bundled projects retain their original terms.
