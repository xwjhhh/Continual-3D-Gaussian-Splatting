from __future__ import annotations

import gc
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from clsplats.utils.prepare_clsplats_object_masks import (
    _build_sam_config,
    _load_automatic_sam_api,
    _resize_mask_to_shape,
    resolve_deva_root,
)


def clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def remove_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tensor_image_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    return (
        image_tensor.detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


def load_uint16_mask(mask_path: Path) -> torch.Tensor:
    mask_np = np.array(Image.open(mask_path))
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    return torch.from_numpy(mask_np.astype(np.int64))


def save_uint16_mask(mask_tensor: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_np = mask_tensor.detach().cpu().numpy().astype(np.uint16)
    Image.fromarray(mask_np).save(output_path)


def save_float_map_as_uint16(map_tensor: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_cpu = map_tensor.detach().to(dtype=torch.float32).cpu()
    if map_cpu.numel() <= 0:
        map_np = np.zeros((1, 1), dtype=np.uint16)
    else:
        max_value = float(map_cpu.max().item())
        if max_value <= 0.0:
            map_np = torch.zeros_like(map_cpu, dtype=torch.uint16).numpy()
        else:
            scaled = (map_cpu / max_value).clamp(0.0, 1.0).mul(65535.0).round().to(torch.uint16)
            map_np = scaled.numpy()
    Image.fromarray(map_np).save(output_path)


def save_rgb_image(image_np: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_np).save(output_path)


class SamMaskGenerator:
    def __init__(
        self,
        *,
        deva_root_arg: Optional[str],
        size: int,
        sam_pred_iou_threshold: float,
        sam_variant: str,
        sam_num_points_per_side: int,
        sam_num_points_per_batch: int,
        suppress_small_objects: bool,
    ) -> None:
        deva_root = resolve_deva_root(deva_root_arg)
        get_sam_model, auto_segment = _load_automatic_sam_api(deva_root)
        self.auto_segment = auto_segment
        self.sam_config = _build_sam_config(
            deva_root,
            size=int(size),
            sam_pred_iou_threshold=float(sam_pred_iou_threshold),
            sam_variant=str(sam_variant),
            sam_num_points_per_side=int(sam_num_points_per_side),
            sam_num_points_per_batch=int(sam_num_points_per_batch),
            suppress_small_objects=bool(suppress_small_objects),
        )
        self.sam_model = get_sam_model(self.sam_config, "cuda")
        self.suppress_small_objects = bool(suppress_small_objects)

    def segment_image_np(self, image_np: np.ndarray) -> torch.Tensor:
        mask, _, _ = self.auto_segment(
            self.sam_config,
            self.sam_model,
            image_np,
            forward_mask=None,
            min_side=int(self.sam_config["size"]),
            suppress_small_mask=self.suppress_small_objects,
        )
        return _resize_mask_to_shape(mask, image_np.shape[:2]).to(dtype=torch.long)

    def close(self) -> None:
        if getattr(self, "sam_model", None) is not None:
            del self.sam_model
            self.sam_model = None
        release_memory()


def compute_change_mask_from_two_instance_masks(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
) -> torch.Tensor:
    pred_mask = pred_mask.to(dtype=torch.long)
    gt_mask = gt_mask.to(device=pred_mask.device, dtype=torch.long)
    pred_fg = pred_mask > 0
    gt_fg = gt_mask > 0
    return torch.logical_xor(pred_fg, gt_fg)


def compute_instance_iou_weighted_vote_map(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    *,
    match_iou_threshold: float = 0.3,
    unmatched_vote_weight: float = 1.0,
) -> torch.Tensor:
    pred_mask = pred_mask.to(dtype=torch.long)
    gt_mask = gt_mask.to(device=pred_mask.device, dtype=torch.long)
    device = pred_mask.device

    vote_map = torch.zeros_like(pred_mask, dtype=torch.float32, device=device)
    pred_ids = torch.unique(pred_mask)
    gt_ids = torch.unique(gt_mask)
    pred_ids = pred_ids[pred_ids > 0]
    gt_ids = gt_ids[gt_ids > 0]

    if pred_ids.numel() == 0 and gt_ids.numel() == 0:
        return vote_map

    pred_regions = {int(pid.item()): (pred_mask == int(pid.item())) for pid in pred_ids}
    gt_regions = {int(gid.item()): (gt_mask == int(gid.item())) for gid in gt_ids}
    pair_scores = []
    for pred_id, pred_region in pred_regions.items():
        pred_area = float(pred_region.sum().item())
        if pred_area <= 0.0:
            continue
        for gt_id, gt_region in gt_regions.items():
            inter = float((pred_region & gt_region).sum().item())
            if inter <= 0.0:
                continue
            union = pred_area + float(gt_region.sum().item()) - inter
            if union <= 0.0:
                continue
            iou = inter / union
            if iou >= float(match_iou_threshold):
                pair_scores.append((iou, pred_id, gt_id))

    pair_scores.sort(key=lambda item: item[0], reverse=True)
    matched_pred_ids = set()
    matched_gt_ids = set()
    for iou, pred_id, gt_id in pair_scores:
        if pred_id in matched_pred_ids or gt_id in matched_gt_ids:
            continue
        matched_pred_ids.add(pred_id)
        matched_gt_ids.add(gt_id)
        pred_region = pred_regions[pred_id]
        gt_region = gt_regions[gt_id]
        residual_region = torch.logical_xor(pred_region, gt_region)
        vote_weight = max(0.0, 1.0 - float(iou))
        if vote_weight > 0.0:
            vote_map[residual_region] = torch.maximum(
                vote_map[residual_region],
                torch.full_like(vote_map[residual_region], float(vote_weight)),
            )

    unmatched_weight = float(max(unmatched_vote_weight, 0.0))
    if unmatched_weight > 0.0:
        for pred_id, pred_region in pred_regions.items():
            if pred_id not in matched_pred_ids:
                vote_map[pred_region] = torch.maximum(
                    vote_map[pred_region],
                    torch.full_like(vote_map[pred_region], unmatched_weight),
                )
        for gt_id, gt_region in gt_regions.items():
            if gt_id not in matched_gt_ids:
                vote_map[gt_region] = torch.maximum(
                    vote_map[gt_region],
                    torch.full_like(vote_map[gt_region], unmatched_weight),
                )

    return vote_map


def compute_current_instance_instability_map(
    current_mask: torch.Tensor,
    prev_mask: torch.Tensor,
    *,
    stable_iou_threshold: float = 0.7,
    unmatched_vote_weight: float = 1.0,
) -> torch.Tensor:
    current_mask = current_mask.to(dtype=torch.long)
    prev_mask = prev_mask.to(device=current_mask.device, dtype=torch.long)
    device = current_mask.device

    vote_map = torch.zeros_like(current_mask, dtype=torch.float32, device=device)
    current_ids = torch.unique(current_mask)
    prev_ids = torch.unique(prev_mask)
    current_ids = current_ids[current_ids > 0]
    prev_ids = prev_ids[prev_ids > 0]

    if current_ids.numel() == 0 or prev_ids.numel() == 0:
        if current_ids.numel() == 0:
            return vote_map
        unmatched_weight = float(max(unmatched_vote_weight, 0.0))
        if unmatched_weight <= 0.0:
            return vote_map
        for current_id in current_ids:
            current_region = current_mask == int(current_id.item())
            vote_map[current_region] = unmatched_weight
        return vote_map

    prev_regions = {}
    prev_areas = {}
    for prev_id in prev_ids:
        pid = int(prev_id.item())
        region = prev_mask == pid
        prev_regions[pid] = region
        prev_areas[pid] = float(region.sum().item())

    unmatched_weight = float(max(unmatched_vote_weight, 0.0))
    stable_threshold = float(max(stable_iou_threshold, 0.0))

    for current_id in current_ids:
        cid = int(current_id.item())
        current_region = current_mask == cid
        current_area = float(current_region.sum().item())
        if current_area <= 0.0:
            continue

        best_iou = 0.0
        for prev_id, prev_region in prev_regions.items():
            inter = float((current_region & prev_region).sum().item())
            if inter <= 0.0:
                continue
            union = current_area + prev_areas[prev_id] - inter
            if union <= 0.0:
                continue
            best_iou = max(best_iou, inter / union)

        if best_iou >= stable_threshold:
            continue

        if best_iou <= 0.0:
            vote_weight = unmatched_weight
        else:
            vote_weight = max(0.0, 1.0 - float(best_iou))
        if vote_weight > 0.0:
            vote_map[current_region] = float(vote_weight)

    return vote_map


def build_change_mask_cache_entry(
    *,
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    tmp_root: Path,
    image_stem: str,
    use_instance_iou_vote: bool = False,
    use_current_instability_vote: bool = False,
    stable_iou_threshold: float = 0.7,
    instance_match_iou_threshold: float = 0.3,
    unmatched_vote_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, str]]:
    pred_mask_path = tmp_root / "pred_mask_gray" / f"{image_stem}.png"
    change_mask_path = tmp_root / "change_mask" / f"{image_stem}.png"
    save_uint16_mask(pred_mask, pred_mask_path)
    if use_current_instability_vote:
        change_mask = compute_current_instance_instability_map(
            gt_mask,
            pred_mask,
            stable_iou_threshold=float(stable_iou_threshold),
            unmatched_vote_weight=float(unmatched_vote_weight),
        )
        save_float_map_as_uint16(change_mask, change_mask_path)
    elif use_instance_iou_vote:
        change_mask = compute_instance_iou_weighted_vote_map(
            pred_mask,
            gt_mask,
            match_iou_threshold=float(instance_match_iou_threshold),
            unmatched_vote_weight=float(unmatched_vote_weight),
        )
        save_float_map_as_uint16(change_mask, change_mask_path)
    else:
        change_mask = compute_change_mask_from_two_instance_masks(pred_mask, gt_mask).to(dtype=torch.float32)
        save_uint16_mask(change_mask.to(dtype=torch.uint16), change_mask_path)
    meta = {
        "pred_mask_path": str(pred_mask_path),
        "change_mask_path": str(change_mask_path),
        "vote_mode": (
            "current_instability"
            if use_current_instability_vote
            else ("instance_iou" if use_instance_iou_vote else "binary_xor")
        ),
    }
    return change_mask, meta
