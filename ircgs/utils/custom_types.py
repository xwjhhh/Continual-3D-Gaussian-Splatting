"""Custom type definitions for CL-Splats."""

import torch
from typing import TypeAlias

# Image tensor type: [H, W, 3] with float32
# Note: torchtyping is incompatible with PyTorch 2.x, using standard type alias instead
Image: TypeAlias = torch.Tensor