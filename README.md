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
methods. Use **Watch online** for browser playback through the project result
viewer, or **MP4** to download the full-resolution video tracked with Git LFS.

| Scene | Video | Scene | Video |
|:--|:--:|:--|:--:|
| Breville | [Watch online][watch-breville] / [MP4][mp4-breville] | Car | [Watch online][watch-car] / [MP4][mp4-car] |
| Community | [Watch online][watch-community] / [MP4][mp4-community] | Grill | [Watch online][watch-grill] / [MP4][mp4-grill] |
| Living room | [Watch online][watch-living-room] / [MP4][mp4-living-room] | Street | [Watch online][watch-street] / [MP4][mp4-street] |

[watch-breville]: https://htmlpreview.github.io/?https://github.com/xwjhhh/IRC-GS/blob/main/docs/index.html#breville
[watch-car]: https://htmlpreview.github.io/?https://github.com/xwjhhh/IRC-GS/blob/main/docs/index.html#car
[watch-community]: https://htmlpreview.github.io/?https://github.com/xwjhhh/IRC-GS/blob/main/docs/index.html#community
[watch-grill]: https://htmlpreview.github.io/?https://github.com/xwjhhh/IRC-GS/blob/main/docs/index.html#grill
[watch-living-room]: https://htmlpreview.github.io/?https://github.com/xwjhhh/IRC-GS/blob/main/docs/index.html#living-room
[watch-street]: https://htmlpreview.github.io/?https://github.com/xwjhhh/IRC-GS/blob/main/docs/index.html#street

[mp4-breville]: https://media.githubusercontent.com/media/xwjhhh/IRC-GS/main/videos/breville.mp4
[mp4-car]: https://media.githubusercontent.com/media/xwjhhh/IRC-GS/main/videos/car.mp4
[mp4-community]: https://media.githubusercontent.com/media/xwjhhh/IRC-GS/main/videos/community.mp4
[mp4-grill]: https://media.githubusercontent.com/media/xwjhhh/IRC-GS/main/videos/grill.mp4
[mp4-living-room]: https://media.githubusercontent.com/media/xwjhhh/IRC-GS/main/videos/living_room.mp4
[mp4-street]: https://media.githubusercontent.com/media/xwjhhh/IRC-GS/main/videos/street.mp4

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
|-- ircgs/                    # IRC-GS training and evaluation package
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
pip install -r requirements.txt
pip install einops
```

`Scaffold-GS-main` also imports `torch-scatter`. Install a wheel matching the
PyTorch and CUDA versions in your environment; for example, for PyTorch 2.7.0
with CUDA 12.6:

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
```

Install the two `diff_gaussian_rasterization` extensions in the order shown
above. The Scaffold-GS copy must be installed last because it provides the
custom `exact_importance` interface used by IRC-GS. `laspy` is optional and is
only needed for LAS point-cloud inputs.

The 4DGS baseline has additional CUDA modules documented in
[`4DGaussians-master/README.md`](4DGaussians-master/README.md).

## Data

Set `DATA_ROOT` to the World Across Time (WAT) root. Data is never tracked by
Git. The preferred layout is one COLMAP reconstruction per timestep:

```text
/path/to/WAT/
`-- breville/
    |-- t0/
    |   |-- images_undist/
    |   `-- sparse_undist/0/
    |       |-- cameras.bin
    |       |-- images.bin
    |       `-- points3D.bin
    |-- t1/
    |   |-- images_undist/
    |   `-- sparse_undist/0/
    `-- ...
```

The reader also accepts `images/t0`, `images/t1`, ... with a shared root
`sparse/0`, or a single-timestep `images/` + `sparse/0` dataset. When both
variants exist, `images_undist` and `sparse_undist` are preferred. A typical
AutoDL path is `/root/autodl-tmp/Continual-3D-Gaussian-Splatting-main/data/cl-splats/WAT`.

## Training

Run one method on one or more scenes:

```bash
DATA_ROOT=/path/to/WAT METHOD=irc-gs ONLY_SCENES=breville bash run.sh
DATA_ROOT=/path/to/WAT METHOD=cl-splats ONLY_SCENES=breville bash run.sh
DATA_ROOT=/path/to/WAT METHOD=scaffold-gs ONLY_SCENES=breville bash run.sh
DATA_ROOT=/path/to/WAT METHOD=3dgs ONLY_SCENES=breville bash run.sh
DATA_ROOT=/path/to/WAT METHOD=4dgs ONLY_SCENES=breville bash run.sh
```

Use `METHOD=all` for the complete comparison. This runs IRC-GS, CL-Splats,
Scaffold-GS, 3DGS, and 4DGS sequentially, so all baseline-specific CUDA
extensions and dependencies must be installed first. In particular, 4DGS has
additional modules documented in [`4DGaussians-master/README.md`](4DGaussians-master/README.md),
and the official CL-Splats code under `cl-splats-main/` may require its own
environment adjustments. For a first check, run `METHOD=irc-gs` alone, then
add baselines one at a time. Multiple scenes are passed as a comma-separated
list, for example `ONLY_SCENES=breville,kitchen,living_room`.
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
`outputs/<method_slug>/<scene>/`. The runner replaces `-` with `_`, so the
default examples are `outputs/irc_gs/breville/`,
`outputs/scaffold_gs/breville/`, and `outputs/cl_splats_official_30k/breville/`.
These generated artifacts remain ignored by Git. For IRC-GS, timestep 0 trains
through iteration `25000`, applies the importance correction at iteration
`25000`, and continues adaptation from `25001` through `30000`. The total
timestep-0 budget remains `30000` iterations; later timesteps use `INC_ITERS`.

## Git LFS

Install Git LFS before cloning if you need the comparison videos:

```bash
git lfs install
git clone https://github.com/xwjhhh/IRC-GS.git
```

A normal source clone without the video payload can use
`GIT_LFS_SKIP_SMUDGE=1`.

## GitHub Pages

The browser video viewer in `docs/` is deployed by
[`.github/workflows/pages.yml`](.github/workflows/pages.yml). The first time
you use it, enable Pages once in the repository settings:

1. Open **Settings -> Pages** for `xwjhhh/IRC-GS`.
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.
3. Return to **Actions**, select **Deploy Community video page**, and choose
   **Run workflow** (or push a change under `docs/`).

The `configure-pages` error `Get Pages site failed (404)` means this setting
has not been enabled yet. It is a repository configuration issue, not a code
or Git LFS upload failure. After deployment, the viewer is available at
`https://xwjhhh.github.io/IRC-GS/` (the README links use the HTML preview URL
as a fallback).

## License

Keep [`LICENSE`](LICENSE), [`3DGS_LICENSE.md`](3DGS_LICENSE.md),
[`THIRD_PARTY.md`](THIRD_PARTY.md), and every upstream license with any copy or
redistribution. The bundled projects retain their original terms.
