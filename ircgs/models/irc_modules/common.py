"""Shared temporal constants and tensor conversions for IRC-GS."""

from __future__ import annotations

from typing import Any

import torch


TEMPORAL_TIMESTEP_DTYPE = torch.int8
TEMPORAL_ALIVE_SENTINEL = 127

DEFAULT_TEMPORAL_STAGE1_UNTIL = 10_000
DEFAULT_TEMPORAL_STAGE2_UNTIL = 25_000
DEFAULT_TEMPORAL_CLONE_MAX_VISIBLE_CANDIDATES = 100_000


def to_temporal_timestep_tensor(
    value: Any,
    device: torch.device | str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=torch.long)
    tensor = torch.where(
        tensor >= int(TEMPORAL_ALIVE_SENTINEL),
        torch.full_like(tensor, int(TEMPORAL_ALIVE_SENTINEL)),
        tensor,
    )
    tensor = tensor.clamp(
        min=int(torch.iinfo(TEMPORAL_TIMESTEP_DTYPE).min),
        max=int(torch.iinfo(TEMPORAL_TIMESTEP_DTYPE).max),
    )
    return tensor.to(dtype=TEMPORAL_TIMESTEP_DTYPE)

