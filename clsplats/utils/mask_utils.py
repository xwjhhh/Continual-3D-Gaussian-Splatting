from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from loguru import logger


def _load_mask_array(mask_path: Path) -> np.ndarray:
    try:
        pil_image = Image.open(mask_path)
        return np.array(pil_image)
    except (UnidentifiedImageError, OSError):
        try:
            import cv2  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                f"Failed to decode mask image at {mask_path} with PIL, and OpenCV is unavailable."
            ) from exc

        mask_np = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask_np is None:
            raise RuntimeError(
                f"Failed to decode mask image at {mask_path} with both PIL and OpenCV."
            )
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        return mask_np


def load_mask_as_tensor(
    mask_path: str | Path,
    *,
    device: str = "cuda",
    resolution_scale: float = 1.0,
) -> Optional[torch.Tensor]:
    path = Path(mask_path)
    if not path.exists():
        return None

    try:
        mask_np = _load_mask_array(path)
    except Exception as exc:
        logger.warning(
            f"Skipping unreadable object mask: {path} ({type(exc).__name__}: {exc})"
        )
        return None
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    if resolution_scale != 1.0:
        try:
            import cv2  # type: ignore
        except Exception:
            pil_image = Image.fromarray(mask_np)
            width, height = pil_image.size
            new_w = max(1, int(round(float(width) / float(resolution_scale))))
            new_h = max(1, int(round(float(height) / float(resolution_scale))))
            mask_np = np.array(pil_image.resize((new_w, new_h), Image.NEAREST))
        else:
            height, width = mask_np.shape[:2]
            new_w = max(1, int(round(float(width) / float(resolution_scale))))
            new_h = max(1, int(round(float(height) / float(resolution_scale))))
            mask_np = cv2.resize(mask_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    return torch.from_numpy(mask_np).long().to(device)
