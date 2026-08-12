"""Importance-based scene-memory consolidation for IRC-GS."""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger
import torch

from gaussian_renderer import prefilter_voxel as scaffold_prefilter_voxel  # type: ignore
from gaussian_renderer import render as scaffold_render  # type: ignore


class ConsolidationMixin:
    """Score, prune, and retain useful revised anchors."""
    @torch.no_grad()
    def _compute_t0_anchor_importance_scores(self) -> tuple[torch.Tensor, torch.Tensor]:
        n_anchors = int(self.gaussians.get_anchor.shape[0])
        n_offsets = int(getattr(self.gaussians, "n_offsets", 1))
        if n_anchors <= 0:
            return (
                torch.empty((0,), device=self.device, dtype=torch.float32),
                torch.empty((0,), device=self.device, dtype=torch.float32),
            )

        offset_score = torch.zeros((n_anchors * n_offsets,), device=self.device, dtype=torch.float32)
        offset_count = torch.zeros_like(offset_score)
        fallback_score = torch.zeros((n_anchors,), device=self.device, dtype=torch.float32)
        fallback_count = torch.zeros_like(fallback_score)
        was_training = bool(self.gaussians.get_color_mlp.training)
        self.gaussians.train()
        for camera in list(self.current_cameras):
            gt_image = self._lookup_gt_image(camera)
            voxel_visible_mask = scaffold_prefilter_voxel(
                camera,
                self.gaussians,
                self._pipe,
                self.bg_color,
            )
            render_pkg = scaffold_render(
                camera,
                self.gaussians,
                self._pipe,
                self.bg_color,
                visible_mask=voxel_visible_mask,
                retain_grad=False,
                exact_importance_gt=gt_image,
            )
            rendered_image = render_pkg.get("render", None)
            rendered_anchor_ids = render_pkg.get("rendered_anchor_ids", None)
            rendered_slot_ids = render_pkg.get("rendered_offset_slot_ids", None)
            visibility_filter = render_pkg.get("visibility_filter", None)
            if rendered_anchor_ids is None or visibility_filter is None:
                continue
            rendered_anchor_ids = rendered_anchor_ids.to(device=self.device, dtype=torch.long).reshape(-1)
            visibility_filter = visibility_filter.to(device=self.device, dtype=torch.bool).reshape(-1)
            if int(rendered_anchor_ids.shape[0]) != int(visibility_filter.shape[0]):
                continue
            valid_anchor_ids = rendered_anchor_ids[visibility_filter]
            valid_anchor_ids = valid_anchor_ids[
                (valid_anchor_ids >= 0) & (valid_anchor_ids < n_anchors)
            ]
            if valid_anchor_ids.numel() == 0:
                continue

            weight = torch.ones((valid_anchor_ids.numel(),), device=self.device, dtype=torch.float32)
            fallback_score.scatter_add_(0, valid_anchor_ids, weight)
            fallback_count.scatter_add_(0, valid_anchor_ids, weight)

            exact_score = render_pkg.get("exact_importance_score", None)
            exact_count = render_pkg.get("exact_importance_count", None)
            if any(value is None for value in (rendered_image, rendered_slot_ids, exact_score, exact_count)):
                continue
            if rendered_image.shape != gt_image.shape:
                continue

            rendered_slot_ids = rendered_slot_ids.to(device=self.device, dtype=torch.long).reshape(-1)
            exact_score = exact_score.to(device=self.device, dtype=torch.float32).reshape(-1)
            exact_count = exact_count.to(device=self.device, dtype=torch.float32).reshape(-1)
            if not (
                int(rendered_slot_ids.shape[0])
                == int(exact_score.shape[0])
                == int(exact_count.shape[0])
                == int(rendered_anchor_ids.shape[0])
            ):
                continue

            valid_filter = visibility_filter.clone()
            valid_filter &= rendered_anchor_ids >= 0
            valid_filter &= rendered_anchor_ids < n_anchors
            valid_filter &= rendered_slot_ids >= 0
            valid_filter &= rendered_slot_ids < n_offsets
            valid_filter &= torch.isfinite(exact_score)
            valid_filter &= torch.isfinite(exact_count)
            valid_filter &= exact_count > 0
            if not bool(valid_filter.any().item()):
                continue

            anchor_ids = rendered_anchor_ids[valid_filter]
            slot_ids = rendered_slot_ids[valid_filter]
            flat_offset_ids = anchor_ids * n_offsets + slot_ids
            contribution = exact_score[valid_filter]
            finite = torch.isfinite(contribution)
            if not bool(finite.any().item()):
                continue

            contribution = contribution[finite]
            flat_offset_ids = flat_offset_ids[finite]
            contribution_count = exact_count[valid_filter][finite]
            offset_score.scatter_add_(0, flat_offset_ids, contribution)
            offset_count.scatter_add_(0, flat_offset_ids, contribution_count)

        if not was_training:
            self.gaussians.eval()
        if bool((offset_count > 0).any().item()):
            anchor_score = offset_score.view(n_anchors, n_offsets).sum(dim=1)
            anchor_count = offset_count.view(n_anchors, n_offsets).sum(dim=1)
            return anchor_score, anchor_count
        return fallback_score, fallback_count

    def _make_anchor_importance(
        self,
        raw_score: torch.Tensor,
        count_score: torch.Tensor,
        score_type: str,
    ) -> torch.Tensor:
        if score_type == "count":
            importance = count_score
        elif score_type == "opacity":
            importance = self.gaussians.get_opacity.detach().reshape(-1).to(device=self.device, dtype=torch.float32)
        elif score_type == "offset_contribution":
            importance = raw_score
        elif score_type == "v_important_score":
            volume = torch.prod(self.gaussians.get_scaling.detach(), dim=1).to(device=self.device, dtype=torch.float32)
            if int(volume.shape[0]) == int(raw_score.shape[0]) and volume.numel() > 0:
                index = min(max(int(volume.numel() * 0.9), 0), int(volume.numel()) - 1)
                sorted_volume, _ = torch.sort(volume, descending=True)
                kth_volume = torch.clamp(sorted_volume[index], min=1e-8)
                volume_weight = torch.pow(
                    volume / kth_volume,
                    float(getattr(self.training_args, "t0_importance_v_pow", 0.1)),
                )
                importance = raw_score * volume_weight
            else:
                importance = raw_score
        else:
            importance = raw_score
        finite_mask = torch.isfinite(importance)
        return torch.where(finite_mask, importance, torch.zeros_like(importance))

    def _prune_anchor_rows_with_stats(self, prune_mask: torch.Tensor) -> None:
        n_before = int(self.gaussians.get_anchor.shape[0])
        prune_mask = prune_mask.to(device=self.device, dtype=torch.bool).reshape(-1)
        if int(prune_mask.shape[0]) != n_before or not bool(prune_mask.any().item()):
            return
        keep_mask = ~prune_mask
        self._prune_temporal_adapter_payload_rows(keep_mask)
        self.gaussians.prune_anchor(prune_mask)

        for attr_name in ("opacity_accum", "anchor_demon"):
            value = getattr(self.gaussians, attr_name, None)
            if isinstance(value, torch.Tensor) and int(value.shape[0]) == n_before:
                setattr(self.gaussians, attr_name, value[keep_mask])
        for attr_name in ("offset_gradient_accum", "offset_denom"):
            value = getattr(self.gaussians, attr_name, None)
            if isinstance(value, torch.Tensor) and int(value.shape[0]) == n_before * int(self.gaussians.n_offsets):
                keep_offsets = keep_mask.unsqueeze(1).repeat(1, int(self.gaussians.n_offsets)).reshape(-1)
                setattr(self.gaussians, attr_name, value[keep_offsets])
        if hasattr(self.gaussians, "max_radii2D"):
            self.gaussians.max_radii2D = torch.zeros((int(self.gaussians.get_anchor.shape[0]),), device=self.device)
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _apply_t0_anchor_importance_prune(self, iteration: int) -> Dict[str, Any]:
        if self._t0_importance_pruned or int(self.timestep) != 0 or self.training_args is None:
            return {}
        if not bool(getattr(self.training_args, "t0_importance_prune_enabled", True)):
            return {}

        ratio = min(max(float(getattr(self.training_args, "t0_importance_prune_ratio", 0.0)), 0.0), 1.0)
        if ratio <= 0.0:
            self._t0_importance_pruned = True
            return {}

        n_before = int(self.gaussians.get_anchor.shape[0])
        if n_before <= 1:
            self._t0_importance_pruned = True
            return {}

        raw_score, count_score = self._compute_t0_anchor_importance_scores()
        if int(raw_score.shape[0]) != n_before:
            logger.warning(
                f"[T0ImportancePrune] skipped: score length {int(raw_score.shape[0])} "
                f"does not match anchor count {n_before}."
            )
            return {}

        score_type = str(getattr(self.training_args, "t0_importance_score_type", "offset_contribution"))
        importance = self._make_anchor_importance(raw_score, count_score, score_type)

        finite_mask = torch.isfinite(importance)
        if not bool(finite_mask.any()):
            logger.warning("[T0ImportancePrune] skipped: no finite anchor importance scores.")
            return {}
        importance = torch.where(finite_mask, importance, torch.zeros_like(importance))

        prune_count = min(max(int(round(ratio * n_before)), 0), n_before - 1)
        if prune_count <= 0:
            self._t0_importance_pruned = True
            return {}

        _, prune_ids = torch.topk(importance, k=prune_count, largest=False)
        prune_mask = torch.zeros((n_before,), device=self.device, dtype=torch.bool)
        prune_mask[prune_ids] = True
        self._prune_anchor_rows_with_stats(prune_mask)

        n_after = int(self.gaussians.get_anchor.shape[0])
        stats = {
            "iteration": int(iteration),
            "score_type": score_type,
            "ratio": float(ratio),
            "before": int(n_before),
            "pruned": int(prune_count),
            "after": int(n_after),
            "score_min": float(importance.min().item()),
            "score_max": float(importance.max().item()),
            "score_mean": float(importance.mean().item()),
        }
        self._t0_importance_pruned = True
        self._t0_importance_prune_stats = stats
        logger.info(
            "[T0ImportancePrune] "
            f"iter={stats['iteration']}, score_type={stats['score_type']}, "
            f"ratio={stats['ratio']:.4f}, anchors {stats['before']} -> {stats['after']} "
            f"(pruned={stats['pruned']})."
        )
        return stats

    @torch.no_grad()
    def _apply_temporal_new_anchor_importance_prune(self, iteration: int) -> Dict[str, Any]:
        if (
            self._temporal_importance_pruned
            or not self._temporal_enabled()
            or int(self.timestep) <= 0
            or self.training_args is None
        ):
            return {}
        enabled = bool(getattr(self.training_args, "temporal_importance_prune_enabled", True))
        if not enabled:
            self._temporal_importance_pruned = True
            return {}

        ratio = min(max(float(getattr(self.training_args, "temporal_importance_prune_ratio", 0.0)), 0.0), 1.0)
        if ratio <= 0.0:
            self._temporal_importance_pruned = True
            return {}

        n_before = int(self.gaussians.get_anchor.shape[0])
        if n_before <= 1:
            self._temporal_importance_pruned = True
            return {}

        local_mask = self.gaussians.get_temporal_local_mask().to(device=self.device, dtype=torch.bool).reshape(n_before)
        candidate_mask = local_mask
        birth_timestep = getattr(self.gaussians, "temporal_anchor_birth_timestep", None)
        if isinstance(birth_timestep, torch.Tensor) and int(birth_timestep.shape[0]) == n_before:
            current_timestep_mask = birth_timestep.to(device=self.device, dtype=torch.long).reshape(-1) == int(self.timestep)
            candidate_mask = torch.logical_and(candidate_mask, current_timestep_mask)
        death_timestep = getattr(self.gaussians, "temporal_anchor_death_timestep", None)
        if isinstance(death_timestep, torch.Tensor) and int(death_timestep.shape[0]) == n_before:
            alive_mask = death_timestep.to(device=self.device, dtype=torch.long).reshape(-1) > int(self.timestep)
            candidate_mask = torch.logical_and(candidate_mask, alive_mask)

        candidate_count = int(candidate_mask.sum().item())
        if candidate_count <= 1:
            self._temporal_importance_pruned = True
            self._temporal_importance_prune_stats = {
                "iteration": int(iteration),
                "ratio": float(ratio),
                "before": int(n_before),
                "candidates": int(candidate_count),
                "pruned": 0,
                "after": int(n_before),
                "skipped": "not_enough_new_anchors",
            }
            logger.info(
                "[TemporalImportancePrune] skipped: "
                f"current_local_anchor_candidates={candidate_count}, timestep={int(self.timestep)}."
            )
            return self._temporal_importance_prune_stats

        raw_score, count_score = self._compute_t0_anchor_importance_scores()
        if int(raw_score.shape[0]) != n_before:
            logger.warning(
                f"[TemporalImportancePrune] skipped: score length {int(raw_score.shape[0])} "
                f"does not match anchor count {n_before}."
            )
            return {}

        score_type = str(getattr(self.training_args, "t0_importance_score_type", "offset_contribution"))
        importance = self._make_anchor_importance(raw_score, count_score, score_type)
        candidate_ids = torch.where(candidate_mask)[0]
        candidate_importance = importance[candidate_ids]
        prune_count = min(max(int(round(ratio * candidate_count)), 0), candidate_count - 1)
        if prune_count <= 0:
            self._temporal_importance_pruned = True
            return {}

        _, local_prune_indices = torch.topk(candidate_importance, k=prune_count, largest=False)
        prune_ids = candidate_ids[local_prune_indices]
        prune_mask = torch.zeros((n_before,), device=self.device, dtype=torch.bool)
        prune_mask[prune_ids] = True
        self._prune_anchor_rows_with_stats(prune_mask)

        n_after = int(self.gaussians.get_anchor.shape[0])
        stats = {
            "iteration": int(iteration),
            "score_type": score_type,
            "ratio": float(ratio),
            "before": int(n_before),
            "candidates": int(candidate_count),
            "pruned": int(prune_count),
            "after": int(n_after),
            "score_min": float(candidate_importance.min().item()),
            "score_max": float(candidate_importance.max().item()),
            "score_mean": float(candidate_importance.mean().item()),
        }
        self._temporal_importance_pruned = True
        self._temporal_importance_prune_stats = stats
        logger.info(
            "[TemporalImportancePrune] "
            f"timestep={int(self.timestep)}, iter={stats['iteration']}, "
            f"score_type={stats['score_type']}, ratio={stats['ratio']:.4f}, "
            f"current_local_anchor_candidates={stats['candidates']}, anchors {stats['before']} -> {stats['after']} "
            f"(pruned={stats['pruned']})."
        )
        return stats
