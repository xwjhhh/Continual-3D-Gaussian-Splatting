from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from clsplats.utils.hexplane_utils import grid_sample_wrapper, normalize_aabb


class SpatialTriPlaneField(nn.Module):
    """4DGS-style spatial tri-plane encoding using xy/xz/yz planes only."""

    def __init__(
        self,
        bounds: float,
        feat_dim: int,
        resolution: int,
        multires: Tuple[int, ...],
    ) -> None:
        super().__init__()
        aabb = torch.tensor(
            [[bounds, bounds, bounds], [-bounds, -bounds, -bounds]],
            dtype=torch.float32,
        )
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.feat_dim_per_level = int(feat_dim)
        self.multires = tuple(int(v) for v in multires)
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res_mul in self.multires:
            res = int(resolution) * int(res_mul)
            planes = nn.ParameterList()
            for _ in range(3):
                plane = nn.Parameter(torch.empty((1, int(feat_dim), res, res), dtype=torch.float32))
                nn.init.uniform_(plane, a=0.1, b=0.5)
                planes.append(plane)
            self.grids.append(planes)
            self.feat_dim += int(feat_dim)

    @property
    def get_aabb(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.aabb[0], self.aabb[1]

    def set_aabb(self, xyz_max, xyz_min) -> None:
        xyz_max_t = torch.as_tensor(xyz_max, dtype=torch.float32).reshape(-1)
        xyz_min_t = torch.as_tensor(xyz_min, dtype=torch.float32).reshape(-1)
        aabb = torch.stack([xyz_max_t, xyz_min_t], dim=0)
        self.aabb = nn.Parameter(aabb, requires_grad=False)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        pts = normalize_aabb(pts, self.aabb).reshape(-1, 3)
        outputs: List[torch.Tensor] = []
        coord_pairs = ((0, 1), (0, 2), (1, 2))
        for planes in self.grids:
            interp = 1.0
            for plane, coord_pair in zip(planes, coord_pairs):
                feat = grid_sample_wrapper(plane, pts[:, coord_pair]).view(-1, int(plane.shape[1]))
                interp = interp * feat
            outputs.append(interp)
        if outputs:
            return torch.cat(outputs, dim=-1)
        return torch.zeros((pts.shape[0], 0), device=pts.device, dtype=torch.float32)


class Deformation(nn.Module):
    """4DGS-style deformation head specialized for SH color residuals."""

    def __init__(
        self,
        bounds: float,
        plane_feat_dim: int,
        plane_resolution: int,
        multires: Tuple[int, ...],
        hidden_dim: int,
        depth: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.grid = SpatialTriPlaneField(
            bounds=float(bounds),
            feat_dim=int(plane_feat_dim),
            resolution=int(plane_resolution),
            multires=tuple(int(v) for v in multires),
        )
        self.W = int(hidden_dim)
        self.D = int(depth)
        self.output_dim = int(output_dim)
        self.create_net()
        self.apply(self._initialize_weights)

    @property
    def feat_dim(self) -> int:
        return int(self.grid.feat_dim)

    @property
    def get_aabb(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.grid.get_aabb

    def set_aabb(self, xyz_max, xyz_min) -> None:
        self.grid.set_aabb(xyz_max, xyz_min)

    def create_net(self) -> None:
        layers: List[nn.Module] = [nn.Linear(int(self.grid.feat_dim), int(self.W))]
        for _ in range(max(0, int(self.D) - 1)):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(int(self.W), int(self.W)))
        self.feature_out = nn.Sequential(*layers)
        self.shs_deform = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(int(self.W), int(self.W)),
            nn.ReLU(inplace=True),
            nn.Linear(int(self.W), int(self.output_dim)),
        )

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def query_features(self, pts: torch.Tensor) -> torch.Tensor:
        return self.grid(pts)

    def forward(self, pts: torch.Tensor, shs: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = self.feature_out(self.query_features(pts))
        delta = self.shs_deform(hidden)
        if shs is None:
            return delta
        shs_shape = tuple(shs.shape[1:])
        if shs_shape != (16, 3):
            raise RuntimeError(
                f"Deformation.forward expected SH tensor shape[1:]=(16, 3), got {shs_shape}."
            )
        return shs + delta.reshape(-1, 16, 3)
