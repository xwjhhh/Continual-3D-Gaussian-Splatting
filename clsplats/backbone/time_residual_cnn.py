import torch
from torch import nn


class SimpleResidualCNN(nn.Module):
    """Small fully-convolutional residual head.

    Default architecture:
    Conv(13, 32, 1) -> ReLU ->
    Conv(32, 32, 3) -> ReLU ->
    Conv(32, 32, 3) -> ReLU ->
    Conv(32, 32, 3) -> ReLU ->
    Conv(32, 3, 1)
    """

    def __init__(self, in_channels: int = 13, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 3, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeResidualRefiner(nn.Module):
    """Per-time residual predictor: I_pred = C_base + DeltaC."""

    def __init__(self, in_channels: int = 13, hidden_dim: int = 32):
        super().__init__()
        self.residual_net = SimpleResidualCNN(in_channels=in_channels, hidden_dim=hidden_dim)

    @staticmethod
    def build_input(
        base_rgb: torch.Tensor,
        xyz_map: torch.Tensor,
        normal: torch.Tensor,
        ray_dir: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        expected_channels = (3, 3, 3, 3, 1)
        tensors = (base_rgb, xyz_map, normal, ray_dir, alpha)
        for tensor, channels in zip(tensors, expected_channels):
            if tensor.dim() != 4 or tensor.size(1) != channels:
                raise ValueError(
                    f"Expected tensor shape [B, {channels}, H, W], got {tuple(tensor.shape)}"
                )
        return torch.cat([base_rgb, xyz_map, normal, ray_dir, alpha], dim=1)

    def forward(
        self,
        base_rgb: torch.Tensor,
        xyz_map: torch.Tensor,
        normal: torch.Tensor,
        ray_dir: torch.Tensor,
        alpha: torch.Tensor,
        clamp_output: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.build_input(base_rgb, xyz_map, normal, ray_dir, alpha)
        delta_c = self.residual_net(x)
        pred = base_rgb + delta_c
        if clamp_output:
            pred = pred.clamp(0.0, 1.0)
        return pred, delta_c


def fuse_base_with_residual(
    base_rgb: torch.Tensor,
    residual: torch.Tensor,
    clamp_output: bool = False,
) -> torch.Tensor:
    pred = base_rgb + residual
    if clamp_output:
        pred = pred.clamp(0.0, 1.0)
    return pred
