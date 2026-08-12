# Third-Party Code

This release bundles several upstream research codebases so the package is
closer to runnable out of the box:

- `submodules/diff-gaussian-rasterization/`
- `submodules/simple-knn/`
- `cl-splats-main/`
- `Scaffold-GS-main/`
- `4DGaussians-master/`

The 3DGS baseline is implemented through `ircgs/models/pure_3dgs.py`
plus the local 3DGS representation/rendering code and the bundled CUDA
extensions above.

Keep each upstream project's original license file and attribution intact.
Also keep this package's `LICENSE` and `3DGS_LICENSE.md`; some 3DGS-derived
components are for non-commercial research/evaluation use under the upstream
3DGS terms.
