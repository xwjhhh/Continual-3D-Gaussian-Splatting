import os
from typing import Optional

import torch
from torch import nn


class TemporalLatentState(nn.Module):
    def __init__(
        self,
        num_anchors: int,
        latent_dim: int,
        *,
        num_times: int = 1,
        chunk_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_anchors = int(num_anchors)
        self.latent_dim = int(latent_dim)
        self.num_times = max(int(num_times), 1)
        if chunk_dim is None:
            chunk_dim = self.latent_dim if self.num_times <= 1 else max(self.latent_dim // self.num_times, 1)
        self.chunk_dim = max(int(chunk_dim), 1)
        self.active_time_step = max(0, self.num_times - 1)
        self.latent = nn.Parameter(torch.zeros(self.num_anchors, self.latent_dim))

    def set_layout(self, *, num_times: Optional[int] = None, chunk_dim: Optional[int] = None):
        if num_times is not None:
            self.num_times = max(int(num_times), 1)
        if chunk_dim is not None:
            self.chunk_dim = max(int(chunk_dim), 1)
        self.active_time_step = min(max(int(self.active_time_step), 0), self.num_times - 1)

    def set_active_time_step(self, time_step: int):
        self.active_time_step = min(max(int(time_step), 0), self.num_times - 1)

    def get_active_dim(self, time_step: Optional[int] = None) -> int:
        target_step = self.active_time_step if time_step is None else int(time_step)
        return min(self.latent_dim, max(0, target_step + 1) * self.chunk_dim)

    def mask_future(self, latent: torch.Tensor, time_step: Optional[int] = None) -> torch.Tensor:
        if latent is None:
            return latent
        active_dim = int(self.get_active_dim(time_step))
        if active_dim >= int(latent.shape[-1]):
            return latent
        masked = latent.clone()
        masked[..., active_dim:] = 0
        return masked

    def current_block_slice(self, time_step: Optional[int] = None) -> tuple[int, int]:
        target_step = self.active_time_step if time_step is None else int(time_step)
        start = max(0, target_step) * self.chunk_dim
        end = min(self.latent_dim, start + self.chunk_dim)
        return int(start), int(end)

    def forward(self, visible_mask=None, time_step: Optional[int] = None):
        latent = self.latent if visible_mask is None else self.latent[visible_mask]
        return self.mask_future(latent, time_step=time_step)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "num_anchors": self.num_anchors,
                "latent_dim": self.latent_dim,
                "num_times": self.num_times,
                "chunk_dim": self.chunk_dim,
                "active_time_step": self.active_time_step,
                "latent": self.latent.detach().cpu(),
            },
            path,
        )

    def load(self, path: str, device="cuda"):
        payload = torch.load(path, map_location=device)
        self.num_anchors = int(payload["num_anchors"])
        self.latent_dim = int(payload["latent_dim"])
        self.num_times = max(int(payload.get("num_times", 1)), 1)
        self.chunk_dim = max(int(payload.get("chunk_dim", self.latent_dim)), 1)
        self.active_time_step = int(payload.get("active_time_step", self.num_times - 1))
        self.active_time_step = min(max(self.active_time_step, 0), self.num_times - 1)
        self.latent = nn.Parameter(payload["latent"].to(device))
