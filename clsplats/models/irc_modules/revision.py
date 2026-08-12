"""Residual-driven local structure revision for IRC-GS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import random

from loguru import logger
import torch

try:
    import faiss  # type: ignore
    try:
        import faiss.contrib.torch_utils  # type: ignore  # noqa: F401
    except Exception:
        pass
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None
    _FAISS_AVAILABLE = False

from gaussian_renderer import prefilter_voxel as scaffold_prefilter_voxel  # type: ignore
from gaussian_renderer import render as scaffold_render  # type: ignore

from clsplats.models.irc_modules.common import (
    DEFAULT_TEMPORAL_CLONE_MAX_VISIBLE_CANDIDATES,
    DEFAULT_TEMPORAL_STAGE1_UNTIL,
    DEFAULT_TEMPORAL_STAGE2_UNTIL,
    TEMPORAL_ALIVE_SENTINEL,
)


class RevisionMixin:
    """Detect residual anomalies and revise local anchor structure."""
    def _get_temporal_bootstrap_cache_path(self) -> Path:
        scene_dir_name = str(self._current_scene_key).strip() if self._current_scene_key is not None else ""
        if not scene_dir_name:
            scene_dir_name = "unknown_scene"
        timestep_dir_name = f"t{int(self.timestep)}"
        return self._output_dir_path() / "tmp" / scene_dir_name / timestep_dir_name / "bootstrap_vote_cache.pt"

    def _get_temporal_bootstrap_cache_candidates(self) -> List[Path]:
        primary_path = self._get_temporal_bootstrap_cache_path()
        return [
            primary_path,
            primary_path.with_name("vote_cache.pt"),
        ]

    def _reset_temporal_event_logging_state(self) -> None:
        self._last_temporal_event_counters = {
            "split_events": 0,
            "death_events": 0,
        }

    @torch.no_grad()
    def _get_temporal_status_counts(self) -> Dict[str, int]:
        counts = {
            "anomaly_anchors": 0,
            "dead_anchors": 0,
            "local_anchors": 0,
        }
        if not (self._temporal_enabled() and int(self.timestep) > 0):
            return counts
        anomaly_mask = self.gaussians.get_temporal_anomaly_candidate_mask()
        local_mask = self.gaussians.get_temporal_local_mask()
        death_timestep = getattr(self.gaussians, "temporal_anchor_death_timestep", None)
        if anomaly_mask is not None:
            counts["anomaly_anchors"] = int(anomaly_mask.sum().item())
        if local_mask is not None:
            counts["local_anchors"] = int(local_mask.sum().item())
        if death_timestep is not None:
            death_timestep = death_timestep.to(device=self.device, dtype=torch.long).reshape(-1)
            counts["dead_anchors"] = int((death_timestep < int(TEMPORAL_ALIVE_SENTINEL)).sum().item())
        return counts

    @torch.no_grad()
    def _log_temporal_event_summary(self, iteration: int, total_iterations: int) -> None:
        if not (self._temporal_enabled() and int(self.timestep) > 0):
            return
        current_counters = (
            self.gaussians.get_temporal_event_counters()
            if hasattr(self.gaussians, "get_temporal_event_counters")
            else {}
        )
        split_total = int(current_counters.get("split_events", 0))
        death_total = int(current_counters.get("death_events", 0))
        split_delta = split_total - int(self._last_temporal_event_counters.get("split_events", 0))
        death_delta = death_total - int(self._last_temporal_event_counters.get("death_events", 0))
        self._last_temporal_event_counters = {
            "split_events": split_total,
            "death_events": death_total,
        }
        status_counts = self._get_temporal_status_counts()
        bootstrap_stats = self._temporal_bootstrap_vote_stats or {}
        self._log_to_file_only(
            " | ".join(
                [
                    f"[T{int(self.timestep)}] TemporalEvents iter {int(iteration)}/{int(total_iterations)}",
                    f"bootstrap_vote_positive={int(bootstrap_stats.get('vote_positive_anchors', 0))}",
                    f"bootstrap_structure_eligible={int(bootstrap_stats.get('eligible_structure_anchors', 0))}",
                    f"anomaly={int(status_counts['anomaly_anchors'])}",
                    f"split_total={split_total}",
                    f"split_recent={split_delta}",
                    f"death_total={death_total}",
                    f"death_recent={death_delta}",
                    f"dead={int(status_counts['dead_anchors'])}",
                    f"local={int(status_counts['local_anchors'])}",
                ]
            )
        )

    @torch.no_grad()
    def _accumulate_temporal_anomaly_votes(
        self,
        aggregated_votes: torch.Tensor,
        vote_threshold: Optional[float] = None,
    ) -> torch.Tensor:
        if aggregated_votes is None:
            return torch.zeros((0,), device=self.device, dtype=torch.bool)
        aggregated_votes = aggregated_votes.to(device=self.device, dtype=torch.float32).reshape(-1)
        if aggregated_votes.numel() <= 0:
            return torch.zeros((0,), device=self.device, dtype=torch.bool)
        self.gaussians.update_temporal_anomaly_votes(
            aggregated_votes,
            vote_threshold=float(self.training_args.temporal_clone_vote_threshold if vote_threshold is None else vote_threshold),
        )
        return self.gaussians.get_temporal_anomaly_candidate_mask().to(device=self.device, dtype=torch.bool).reshape(-1)

    @torch.no_grad()
    def _set_temporal_bootstrap_selected_anchors(self, selected_anchor_ids: torch.Tensor, total_anchors: int) -> None:
        total_anchors = max(int(total_anchors), 0)
        device = self.device
        if selected_anchor_ids is None:
            selected_anchor_ids = torch.empty((0,), device=device, dtype=torch.long)
        else:
            selected_anchor_ids = selected_anchor_ids.to(device=device, dtype=torch.long).reshape(-1)
        if total_anchors <= 0:
            self._temporal_bootstrap_selected_anchor_ids = torch.empty((0,), device=device, dtype=torch.long)
            self._temporal_bootstrap_selected_anchor_mask = torch.empty((0,), device=device, dtype=torch.bool)
            return
        selected_anchor_ids = torch.unique(selected_anchor_ids)
        selected_anchor_ids = selected_anchor_ids[
            (selected_anchor_ids >= 0) & (selected_anchor_ids < total_anchors)
        ]
        selected_anchor_mask = torch.zeros((total_anchors,), device=device, dtype=torch.bool)
        if selected_anchor_ids.numel() > 0:
            selected_anchor_mask[selected_anchor_ids] = True
        self._temporal_bootstrap_selected_anchor_ids = selected_anchor_ids
        self._temporal_bootstrap_selected_anchor_mask = selected_anchor_mask

    @torch.no_grad()
    def _set_temporal_bootstrap_vote_components(
        self,
        votes_current_to_prev: torch.Tensor,
        votes_prev_to_current: torch.Tensor,
        votes_total: torch.Tensor,
    ) -> None:
        self._temporal_votes_current_to_prev = votes_current_to_prev.detach().to(device=self.device, dtype=torch.float32).reshape(-1)
        self._temporal_votes_prev_to_current = votes_prev_to_current.detach().to(device=self.device, dtype=torch.float32).reshape(-1)
        self._temporal_votes_total = votes_total.detach().to(device=self.device, dtype=torch.float32).reshape(-1)

    def _save_temporal_bootstrap_cache(
        self,
        *,
        selected_anchor_ids: torch.Tensor,
        votes_current_to_prev: torch.Tensor,
        votes_prev_to_current: torch.Tensor,
        votes_total: torch.Tensor,
        vote_stats: Dict[str, int],
        anomaly_candidate_mask: Optional[torch.Tensor],
    ) -> None:
        if not (self._temporal_enabled() and int(self.timestep) > 0):
            return
        cache_path = self._get_temporal_bootstrap_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_num_anchors = int(votes_total.numel())
        if cache_num_anchors <= 0:
            cache_num_anchors = int(self.gaussians.get_anchor.shape[0])
        payload: Dict[str, Any] = {
            "version": 4,
            "scene_key": str(self._current_scene_key),
            "timestep": int(self.timestep),
            "num_anchors": int(cache_num_anchors),
            "temporal_clone_vote_threshold": float(self.training_args.temporal_clone_vote_threshold),
            "temporal_clone_depth_gate_enabled": bool(self.training_args.temporal_clone_depth_gate_enabled),
            "selected_anchor_ids": selected_anchor_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1),
            "votes_current_to_prev": votes_current_to_prev.detach().to(device="cpu", dtype=torch.float32).reshape(-1),
            "votes_prev_to_current": votes_prev_to_current.detach().to(device="cpu", dtype=torch.float32).reshape(-1),
            "votes_total": votes_total.detach().to(device="cpu", dtype=torch.float32).reshape(-1),
            "vote_stats": dict(vote_stats or {}),
        }
        if anomaly_candidate_mask is not None:
            payload["anomaly_candidate_mask"] = anomaly_candidate_mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
        torch.save(payload, cache_path)

    def _try_load_temporal_bootstrap_cache(self, total_anchors: int) -> bool:
        if not (self._temporal_enabled() and int(self.timestep) > 0):
            return False
        cache_candidates = self._get_temporal_bootstrap_cache_candidates()
        cache_path = next((path for path in cache_candidates if path.is_file()), None)
        if cache_path is None:
            logger.info(
                f"Timestep {int(self.timestep)}: no temporal bootstrap cache found "
                f"(checked={', '.join(str(path) for path in cache_candidates)}), recomputing votes."
            )
            return False
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception:
            logger.warning(
                f"Timestep {int(self.timestep)}: failed to load temporal bootstrap cache from {cache_path}, recomputing votes."
            )
            return False
        if not isinstance(payload, dict):
            logger.info(
                f"Timestep {int(self.timestep)}: temporal bootstrap cache {cache_path} "
                "is not a dict payload, recomputing votes."
            )
            return False
        if int(payload.get("timestep", -1)) != int(self.timestep):
            logger.info(
                f"Timestep {int(self.timestep)}: bootstrap cache timestep mismatch "
                f"(cache={int(payload.get('timestep', -1))}, current={int(self.timestep)}), recomputing votes."
            )
            return False
        if str(payload.get("scene_key", "")) != str(self._current_scene_key):
            logger.info(
                f"Timestep {int(self.timestep)}: bootstrap cache scene mismatch "
                f"(cache={str(payload.get('scene_key', ''))}, current={str(self._current_scene_key)}), recomputing votes."
            )
            return False
        if int(payload.get("num_anchors", -1)) != int(total_anchors):
            logger.info(
                f"Timestep {int(self.timestep)}: bootstrap cache anchor count mismatch "
                f"(cache={int(payload.get('num_anchors', -1))}, current={int(total_anchors)}), recomputing votes."
            )
            return False
        cache_version = int(payload.get("version", 0))
        cache_depth_gate_enabled = bool(payload.get("temporal_clone_depth_gate_enabled", True))
        current_depth_gate_enabled = bool(self.training_args.temporal_clone_depth_gate_enabled)
        if (
            cache_version < 3
            or (cache_version < 4 and not current_depth_gate_enabled)
            or abs(float(payload.get("temporal_clone_vote_threshold", float("nan"))) - float(self.training_args.temporal_clone_vote_threshold)) > 1e-6
            or cache_depth_gate_enabled != current_depth_gate_enabled
        ):
            logger.info(
                f"Timestep {int(self.timestep)}: bootstrap cache voting config mismatch, recomputing votes."
            )
            return False

        selected_anchor_ids = payload.get("selected_anchor_ids", None)
        votes_current_to_prev = payload.get("votes_current_to_prev", None)
        votes_prev_to_current = payload.get("votes_prev_to_current", None)
        votes_total = payload.get("votes_total", None)
        if any(not isinstance(v, torch.Tensor) for v in [selected_anchor_ids, votes_current_to_prev, votes_prev_to_current, votes_total]):
            logger.info(
                f"Timestep {int(self.timestep)}: bootstrap cache {cache_path} "
                "is missing vote tensors, recomputing votes."
            )
            return False
        if (
            int(votes_current_to_prev.numel()) != int(total_anchors)
            or int(votes_prev_to_current.numel()) != int(total_anchors)
            or int(votes_total.numel()) != int(total_anchors)
        ):
            logger.info(
                f"Timestep {int(self.timestep)}: bootstrap cache vote tensor length mismatch "
                f"(current_to_prev={int(votes_current_to_prev.numel())}, "
                f"prev_to_current={int(votes_prev_to_current.numel())}, "
                f"total={int(votes_total.numel())}, anchors={int(total_anchors)}), recomputing votes."
            )
            return False

        self._set_temporal_bootstrap_selected_anchors(selected_anchor_ids, total_anchors)
        self._set_temporal_bootstrap_vote_components(
            votes_current_to_prev,
            votes_prev_to_current,
            votes_total,
        )
        vote_stats = payload.get("vote_stats", {})
        self._temporal_bootstrap_vote_stats = dict(vote_stats) if isinstance(vote_stats, dict) else {}

        anomaly_candidate_mask = payload.get("anomaly_candidate_mask", None)
        if (
            isinstance(anomaly_candidate_mask, torch.Tensor)
            and hasattr(self.gaussians, "temporal_anomaly_candidate_mask")
            and getattr(self.gaussians, "temporal_anomaly_candidate_mask", None) is not None
            and int(anomaly_candidate_mask.numel()) == int(total_anchors)
        ):
            self.gaussians.temporal_anomaly_candidate_mask = anomaly_candidate_mask.to(
                device=self.device,
                dtype=torch.bool,
            ).reshape(-1)

        logger.info(
            f"Timestep {int(self.timestep)}: loaded cached temporal bootstrap votes from {cache_path} "
            f"(selected_parents={int(self._temporal_bootstrap_selected_anchor_ids.numel())})."
        )
        return True

    def _has_valid_temporal_bootstrap_cache(self, total_anchors: int) -> bool:
        if not (self._temporal_enabled() and int(self.timestep) > 0):
            return False
        cache_candidates = self._get_temporal_bootstrap_cache_candidates()
        cache_path = next((path for path in cache_candidates if path.is_file()), None)
        if cache_path is None:
            return False
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        if int(payload.get("timestep", -1)) != int(self.timestep):
            return False
        if str(payload.get("scene_key", "")) != str(self._current_scene_key):
            return False
        if int(payload.get("num_anchors", -1)) != int(total_anchors):
            return False
        cache_version = int(payload.get("version", 0))
        cache_depth_gate_enabled = bool(payload.get("temporal_clone_depth_gate_enabled", True))
        current_depth_gate_enabled = bool(self.training_args.temporal_clone_depth_gate_enabled)
        if (
            cache_version < 3
            or (cache_version < 4 and not current_depth_gate_enabled)
            or abs(float(payload.get("temporal_clone_vote_threshold", float("nan"))) - float(self.training_args.temporal_clone_vote_threshold)) > 1e-6
            or cache_depth_gate_enabled != current_depth_gate_enabled
        ):
            return False
        selected_anchor_ids = payload.get("selected_anchor_ids", None)
        votes_current_to_prev = payload.get("votes_current_to_prev", None)
        votes_prev_to_current = payload.get("votes_prev_to_current", None)
        votes_total = payload.get("votes_total", None)
        if any(not isinstance(v, torch.Tensor) for v in [selected_anchor_ids, votes_current_to_prev, votes_prev_to_current, votes_total]):
            return False
        return (
            int(votes_current_to_prev.numel()) == int(total_anchors)
            and int(votes_prev_to_current.numel()) == int(total_anchors)
            and int(votes_total.numel()) == int(total_anchors)
        )

    @torch.no_grad()
    def _vote_anchors_from_weight_map(
        self,
        camera,
        render_pkg: Dict[str, Any],
        weight_map: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_anchors = int(self.gaussians.get_anchor.shape[0])
        empty_votes = torch.zeros((n_anchors,), device=self.device, dtype=torch.float32)
        empty_mask = torch.zeros((n_anchors,), device=self.device, dtype=torch.bool)
        if weight_map is None:
            return empty_votes, empty_mask, empty_mask
        if tuple(weight_map.shape) != (int(camera.image_height), int(camera.image_width)):
            return empty_votes, empty_mask, empty_mask

        rendered_anchor_ids = render_pkg.get("rendered_anchor_ids", None)
        rendered_slot_ids = render_pkg.get("rendered_offset_slot_ids", None)
        rendered_xy = render_pkg.get("rendered_xy", None)
        rendered_depth = render_pkg.get("rendered_depth", None)
        depth_map = render_pkg.get("depth", None)
        radii = render_pkg.get("radii", None)
        visibility_filter = render_pkg.get("visibility_filter", None)
        if any(v is None for v in [rendered_anchor_ids, rendered_slot_ids, rendered_xy, rendered_depth, depth_map, radii, visibility_filter]):
            return empty_votes, empty_mask, empty_mask

        visibility_filter = visibility_filter.to(device=self.device, dtype=torch.bool).reshape(-1)
        rendered_anchor_ids = rendered_anchor_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        rendered_slot_ids = rendered_slot_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        rendered_xy = rendered_xy.to(device=self.device, dtype=torch.float32).reshape(-1, 2)
        rendered_depth = rendered_depth.to(device=self.device, dtype=torch.float32).reshape(-1)
        depth_map = depth_map.to(device=self.device, dtype=torch.float32)
        radii = radii.to(device=self.device, dtype=torch.float32).reshape(-1)

        if int(visibility_filter.shape[0]) != int(rendered_anchor_ids.shape[0]):
            return empty_votes, empty_mask, empty_mask

        local_mask = self.gaussians.get_temporal_local_mask().to(device=self.device, dtype=torch.bool).reshape(-1)
        render_mask = None
        if hasattr(self.gaussians, "get_temporal_render_mask"):
            render_mask = self.gaussians.get_temporal_render_mask().to(device=self.device, dtype=torch.bool).reshape(-1)

        visible_filter = visibility_filter.clone()
        visible_filter &= torch.isfinite(rendered_xy).all(dim=-1)
        visible_filter &= torch.isfinite(rendered_depth)
        visible_filter &= rendered_depth > 0
        visible_filter &= radii > 0
        if not bool(visible_filter.any()):
            return empty_votes, empty_mask, empty_mask

        candidate_filter = visible_filter.clone()
        candidate_filter &= ~local_mask[rendered_anchor_ids]
        if render_mask is not None and int(render_mask.shape[0]) == n_anchors:
            candidate_filter &= render_mask[rendered_anchor_ids]
        visible_anchor_mask = torch.zeros((n_anchors,), device=self.device, dtype=torch.bool)
        visible_anchor_mask[rendered_anchor_ids[candidate_filter]] = True
        if not bool(candidate_filter.any()):
            return empty_votes, visible_anchor_mask, empty_mask

        cand_anchor_ids = rendered_anchor_ids[candidate_filter]
        cand_slot_ids = rendered_slot_ids[candidate_filter]
        cand_xy = rendered_xy[candidate_filter].to(device=self.device, dtype=torch.float32)
        cand_depth = rendered_depth[candidate_filter].to(device=self.device, dtype=torch.float32)
        if cand_anchor_ids.numel() == 0:
            return empty_votes, visible_anchor_mask, empty_mask
        flat_ids = cand_anchor_ids * int(self.gaussians.n_offsets) + cand_slot_ids

        offset_votes = torch.zeros(
            (n_anchors * int(self.gaussians.n_offsets),),
            device=self.device,
            dtype=torch.float32,
        )
        use_depth_gate = bool(self.training_args.temporal_clone_depth_gate_enabled)
        depth_tol = float(self.training_args.temporal_clone_depth_tolerance)

        pixel_x = torch.round(cand_xy[:, 0]).to(dtype=torch.long)
        pixel_y = torch.round(cand_xy[:, 1]).to(dtype=torch.long)
        in_bounds = (
            (pixel_x >= 0)
            & (pixel_x < int(camera.image_width))
            & (pixel_y >= 0)
            & (pixel_y < int(camera.image_height))
        )
        if not bool(in_bounds.any()):
            return empty_votes, visible_anchor_mask, empty_mask

        pixel_x = pixel_x[in_bounds]
        pixel_y = pixel_y[in_bounds]
        cand_anchor_ids = cand_anchor_ids[in_bounds]
        cand_depth = cand_depth[in_bounds]
        flat_ids = flat_ids[in_bounds]

        pixel_vote_weights = weight_map[pixel_y, pixel_x].to(device=self.device, dtype=torch.float32)
        mask_hits = pixel_vote_weights > 0
        if not bool(mask_hits.any()):
            return empty_votes, visible_anchor_mask, empty_mask

        valid_hits = mask_hits
        if use_depth_gate:
            surface_depths = depth_map[0, pixel_y, pixel_x].to(device=self.device, dtype=torch.float32)
            valid_surface = torch.isfinite(surface_depths) & (surface_depths > 0)
            on_surface = torch.abs(cand_depth - surface_depths) <= depth_tol
            valid_hits = valid_hits & valid_surface & on_surface
        if not bool(valid_hits.any()):
            return empty_votes, visible_anchor_mask, empty_mask

        hit_anchor_ids = cand_anchor_ids[valid_hits]
        hit_offset_ids = flat_ids[valid_hits]
        hit_contributions = pixel_vote_weights[valid_hits]
        offset_votes.scatter_add_(0, hit_offset_ids, hit_contributions)
        anchor_votes = offset_votes.view(n_anchors, int(self.gaussians.n_offsets)).sum(dim=1)
        anchor_votes = torch.where(~local_mask, anchor_votes, torch.zeros_like(anchor_votes))
        if render_mask is not None and int(render_mask.shape[0]) == n_anchors:
            anchor_votes = torch.where(render_mask, anchor_votes, torch.zeros_like(anchor_votes))
        hit_anchor_mask = torch.zeros((n_anchors,), device=self.device, dtype=torch.bool)
        hit_anchor_mask[hit_anchor_ids] = True
        return anchor_votes, visible_anchor_mask, hit_anchor_mask

    def _ensure_temporal_bootstrap_for_stage(self, stage_id: int) -> None:
        if not (self._temporal_enabled() and int(self.timestep) > 0 and int(stage_id) == 2):
            return
        if self._temporal_clone_bootstrap_done:
            return
        logger.info(
            f"Timestep {int(self.timestep)}: entering temporal stage 2; "
            "use residual voting refresh every "
            f"{int(self.training_args.temporal_clone_refresh_interval)} iterations."
        )
        self._temporal_clone_bootstrap_done = True

    def _ensure_temporal_local_training_mode(self, stage_id: int) -> bool:
        if not self._temporal_enabled() or int(self.timestep) <= 0:
            return False
        if int(stage_id) == 1:
            return False
        if self._temporal_clone_phase_active:
            return True
        self._temporal_clone_phase_active = True
        self.gaussians.enter_temporal_clone_phase(0)
        local_mask = self.gaussians.get_temporal_local_mask()
        local_count = int(local_mask.sum().item()) if local_mask is not None else 0
        logger.info(
            f"Timestep {int(self.timestep)}: enabled temporal local training mode "
            f"(local_rows={local_count})."
        )
        return True

    def _get_temporal_stage_id(self, iteration: int) -> int:
        if not self._temporal_enabled() or int(self.timestep) <= 0:
            return 1
        stage1_until = int(getattr(self.training_args, "temporal_stage1_until", DEFAULT_TEMPORAL_STAGE1_UNTIL))
        stage2_until = int(getattr(self.training_args, "temporal_stage2_until", DEFAULT_TEMPORAL_STAGE2_UNTIL))
        if int(iteration) <= stage1_until:
            return 1
        if int(iteration) <= stage2_until:
            return 2
        return 3

    def _get_stage1_local_anchor_mask(self) -> torch.Tensor:
        n_anchors = int(self.gaussians.get_anchor.shape[0])
        device = self.device
        return self.gaussians.get_temporal_local_mask().to(device=device, dtype=torch.bool).reshape(n_anchors)

    def _get_stage1_structure_eligible_anchor_mask(self) -> torch.Tensor:
        n_anchors = int(self.gaussians.get_anchor.shape[0])
        device = self.device
        local_mask = self.gaussians.get_temporal_local_mask().to(device=device, dtype=torch.bool).reshape(n_anchors)
        if hasattr(self.gaussians, "get_temporal_render_mask"):
            render_mask = self.gaussians.get_temporal_render_mask().to(device=device, dtype=torch.bool).reshape(-1)
            if int(render_mask.shape[0]) == n_anchors:
                local_mask = torch.logical_and(local_mask, render_mask)
        return local_mask

    def _get_stage2_attribute_anchor_mask(self) -> torch.Tensor:
        n_anchors = int(self.gaussians.get_anchor.shape[0])
        return self.gaussians.get_temporal_local_mask().to(device=self.device, dtype=torch.bool).reshape(n_anchors)

    def append_temporal_local_clones(self, parent_row_ids: torch.Tensor, suppress_parents: bool = True) -> int:
        old_row_count = int(self.gaussians.get_anchor.shape[0])
        n_new = int(self.gaussians.append_temporal_local_clones(parent_row_ids, suppress_parents=suppress_parents))
        if n_new > 0:
            self._hide_new_rows_in_past_temporal_payloads(old_row_count, n_new)
            if suppress_parents:
                parent_row_ids = parent_row_ids.to(device=self.device, dtype=torch.long).reshape(-1)
                parent_row_ids = torch.unique(parent_row_ids)
                parent_row_ids = parent_row_ids[
                    (parent_row_ids >= 0) & (parent_row_ids < old_row_count)
                ]
                for payload_key, payload in self.temporal_adapter_payloads.items():
                    if not isinstance(payload, dict):
                        continue
                    payload_time = int(payload.get("time_step", payload_key))
                    if payload_time < int(self.timestep):
                        continue
                    death_timestep = payload.get("death_timestep", None)
                    if isinstance(death_timestep, torch.Tensor) and int(death_timestep.shape[0]) >= old_row_count:
                        payload_parent_ids = parent_row_ids.to(device=death_timestep.device, dtype=torch.long)
                        death_timestep[payload_parent_ids] = torch.minimum(
                            death_timestep[payload_parent_ids],
                            torch.full_like(death_timestep[payload_parent_ids], int(self.timestep)),
                        )
            total_local = int(self.gaussians.get_temporal_local_mask().sum().item())
            logger.info(
                f"Timestep {int(self.timestep)}: appended {n_new} temporal local clone rows "
                f"(total_local={total_local}, suppress_parents={'on' if suppress_parents else 'off'})."
            )
        return n_new

    def _project_world_to_pixel_xy(
        self,
        points_3d: torch.Tensor,
        camera,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if points_3d.numel() == 0:
            empty_xy = torch.empty((0, 2), device=self.device, dtype=torch.float32)
            empty_valid = torch.empty((0,), device=self.device, dtype=torch.bool)
            return empty_xy, empty_valid

        pts = points_3d.to(device=self.device, dtype=torch.float32)
        ones = torch.ones((pts.shape[0], 1), device=self.device, dtype=pts.dtype)
        pts_h = torch.cat([pts, ones], dim=1)
        full_proj = camera.full_proj_transform.to(device=self.device, dtype=pts.dtype)
        pts_clip = pts_h @ full_proj
        w = pts_clip[:, 3]
        valid = torch.isfinite(w) & (w > 1e-6)

        ndc_xy = pts_clip[:, :2] / (w.unsqueeze(-1) + 1e-7)
        x = (ndc_xy[:, 0] + 1.0) * 0.5 * float(max(int(camera.image_width) - 1, 1))
        y = (1.0 - (ndc_xy[:, 1] + 1.0) * 0.5) * float(max(int(camera.image_height) - 1, 1))
        xy = torch.stack([x, y], dim=-1)
        valid = valid & torch.isfinite(xy).all(dim=-1)
        return xy, valid

    def _project_world_to_pixel_xy_depth(
        self,
        points_3d: torch.Tensor,
        camera,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if points_3d.numel() == 0:
            empty_xy = torch.empty((0, 2), device=self.device, dtype=torch.float32)
            empty_depth = torch.empty((0,), device=self.device, dtype=torch.float32)
            empty_valid = torch.empty((0,), device=self.device, dtype=torch.bool)
            return empty_xy, empty_depth, empty_valid

        pts = points_3d.to(device=self.device, dtype=torch.float32)
        ones = torch.ones((pts.shape[0], 1), device=self.device, dtype=pts.dtype)
        pts_h = torch.cat([pts, ones], dim=1)

        full_proj = camera.full_proj_transform.to(device=self.device, dtype=pts.dtype)
        view_transform = camera.world_view_transform.to(device=self.device, dtype=pts.dtype)

        pts_clip = pts_h @ full_proj
        pts_view = pts_h @ view_transform
        w = pts_clip[:, 3]
        depth = pts_view[:, 2]
        valid = torch.isfinite(w) & (w > 1e-6) & torch.isfinite(depth) & (depth > 0)

        ndc_xy = pts_clip[:, :2] / (w.unsqueeze(-1) + 1e-7)
        x = (ndc_xy[:, 0] + 1.0) * 0.5 * float(max(int(camera.image_width) - 1, 1))
        y = (1.0 - (ndc_xy[:, 1] + 1.0) * 0.5) * float(max(int(camera.image_height) - 1, 1))
        xy = torch.stack([x, y], dim=-1)
        valid = valid & torch.isfinite(xy).all(dim=-1)
        return xy, depth.to(dtype=torch.float32), valid

    def _topk_nearest_chunked(
        self,
        query_xy: torch.Tensor,
        candidate_xy: torch.Tensor,
        k: int,
        *,
        chunk_size: int = 4096,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_xy.numel() == 0 or candidate_xy.numel() == 0 or int(k) <= 0:
            empty_d = torch.empty((int(query_xy.shape[0]), 0), device=self.device, dtype=torch.float32)
            empty_i = torch.empty((int(query_xy.shape[0]), 0), device=self.device, dtype=torch.long)
            return empty_d, empty_i

        faiss_result = self._topk_nearest_faiss(query_xy, candidate_xy, k)
        if faiss_result is not None:
            return faiss_result

        k = int(min(int(k), int(candidate_xy.shape[0])))
        query_xy = query_xy.to(device=self.device, dtype=torch.float32)
        candidate_xy = candidate_xy.to(device=self.device, dtype=torch.float32)
        m = int(query_xy.shape[0])
        best_d = torch.full((m, k), float("inf"), device=self.device, dtype=torch.float32)
        best_i = torch.full((m, k), -1, device=self.device, dtype=torch.long)
        chunk = max(1024, int(chunk_size))
        total = int(candidate_xy.shape[0])

        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            cand_chunk = candidate_xy[start:end]
            d = torch.cdist(query_xy, cand_chunk)
            idx = torch.arange(start, end, device=self.device, dtype=torch.long).unsqueeze(0).expand(m, -1)
            merged_d = torch.cat([best_d, d], dim=1)
            merged_i = torch.cat([best_i, idx], dim=1)
            order = torch.argsort(merged_d, dim=1)[:, :k]
            best_d = torch.gather(merged_d, 1, order)
            best_i = torch.gather(merged_i, 1, order)

        return best_d, best_i

    def _topk_nearest_faiss(
        self,
        query_xy: torch.Tensor,
        candidate_xy: torch.Tensor,
        k: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if not _FAISS_AVAILABLE:
            return None
        if query_xy.device.type != "cuda" or candidate_xy.device.type != "cuda":
            return None
        try:
            if self._faiss_gpu_resources is None:
                self._faiss_gpu_resources = faiss.StandardGpuResources()
            device_index = int(query_xy.device.index or 0)
            cpu_index = faiss.IndexFlatL2(2)
            gpu_index = faiss.index_cpu_to_gpu(self._faiss_gpu_resources, device_index, cpu_index)
            query_xy = query_xy.contiguous().to(device=query_xy.device, dtype=torch.float32)
            candidate_xy = candidate_xy.contiguous().to(device=query_xy.device, dtype=torch.float32)
            gpu_index.add(candidate_xy)
            distances_sq, indices = gpu_index.search(query_xy, int(k))

            if isinstance(distances_sq, torch.Tensor):
                distances_sq = distances_sq.to(device=query_xy.device, dtype=torch.float32)
            else:
                distances_sq = torch.as_tensor(distances_sq, device=query_xy.device, dtype=torch.float32)
            if isinstance(indices, torch.Tensor):
                indices = indices.to(device=query_xy.device, dtype=torch.long)
            else:
                indices = torch.as_tensor(indices, device=query_xy.device, dtype=torch.long)

            if not self._faiss_knn_info_emitted:
                logger.info("Faiss GPU KNN active for temporal clone candidate search.")
                self._faiss_knn_info_emitted = True

            return distances_sq.clamp_min(0.0).sqrt(), indices
        except Exception as exc:
            if not self._faiss_knn_warning_emitted:
                logger.warning(
                    f"Faiss GPU KNN unavailable at runtime ({type(exc).__name__}: {exc}). "
                    "Falling back to chunked torch.cdist search."
                )
                self._faiss_knn_warning_emitted = True
            return None

    def _sample_temporal_clone_views(self, primary_camera) -> List[Any]:
        if not self.current_cameras:
            return []
        num_views = min(max(int(self.training_args.temporal_clone_num_views), 1), len(self.current_cameras))
        views: List[Any] = [primary_camera]
        remaining = [cam for cam in self.current_cameras if cam is not primary_camera]
        if remaining and num_views > 1:
            sampled = random.sample(remaining, k=min(num_views - 1, len(remaining)))
            views.extend(sampled)
        return views

    @torch.no_grad()
    def _mine_temporal_clone_candidates_for_view(
        self,
        camera,
        gt_image: torch.Tensor,
    ) -> torch.Tensor:
        n_anchors = int(self.gaussians.get_anchor.shape[0])
        if n_anchors <= 0:
            return torch.empty((0,), device=self.device, dtype=torch.float32)

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
            return_depth=True,
        )
        rendered = render_pkg["render"]
        if rendered.shape != gt_image.shape:
            return torch.zeros((n_anchors,), device=self.device, dtype=torch.float32)

        mse_map = ((rendered - gt_image) ** 2).mean(dim=0)
        psnr_map = -10.0 * torch.log10(mse_map.clamp_min(1e-12))
        flat_psnr = psnr_map.reshape(-1)
        if flat_psnr.numel() == 0:
            return torch.zeros((n_anchors,), device=self.device, dtype=torch.float32)

        finite_psnr = torch.isfinite(flat_psnr)
        if not bool(finite_psnr.any()):
            return torch.zeros((n_anchors,), device=self.device, dtype=torch.float32)
        top_percent = min(max(float(self.training_args.temporal_clone_top_low_psnr_percent), 1e-6), 1.0)
        top_pixels = max(1, int(int(finite_psnr.sum().item()) * top_percent))
        top_pixels = min(top_pixels, int(finite_psnr.sum().item()), int(self.training_args.temporal_clone_max_pixels_per_view))
        psnr_thresh = torch.topk(flat_psnr[finite_psnr], k=top_pixels, largest=False).values[-1]
        psnr_limit = min(float(psnr_thresh.item()), float(self.training_args.temporal_clone_max_psnr))
        bad_mask = torch.isfinite(psnr_map) & (psnr_map <= psnr_limit)
        ys, xs = torch.nonzero(bad_mask, as_tuple=True)
        if ys.numel() == 0:
            return torch.zeros((n_anchors,), device=self.device, dtype=torch.float32)
        if ys.numel() > int(self.training_args.temporal_clone_max_pixels_per_view):
            perm = torch.randperm(ys.numel(), device=self.device)[: int(self.training_args.temporal_clone_max_pixels_per_view)]
            ys = ys[perm]
            xs = xs[perm]

        rendered_anchor_ids = render_pkg.get("rendered_anchor_ids", None)
        rendered_slot_ids = render_pkg.get("rendered_offset_slot_ids", None)
        rendered_xy = render_pkg.get("rendered_xy", None)
        rendered_depth = render_pkg.get("rendered_depth", None)
        depth_map = render_pkg.get("depth", None)
        radii = render_pkg.get("radii", None)
        visibility_filter = render_pkg.get("visibility_filter", None)
        if any(v is None for v in [rendered_anchor_ids, rendered_slot_ids, rendered_xy, rendered_depth, depth_map, radii, visibility_filter]):
            return torch.zeros((n_anchors,), device=self.device, dtype=torch.float32)

        visibility_filter = visibility_filter.to(device=self.device, dtype=torch.bool).reshape(-1)
        rendered_anchor_ids = rendered_anchor_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        rendered_slot_ids = rendered_slot_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        rendered_xy = rendered_xy.to(device=self.device, dtype=torch.float32).reshape(-1, 2)
        rendered_depth = rendered_depth.to(device=self.device, dtype=torch.float32).reshape(-1)
        depth_map = depth_map.to(device=self.device, dtype=torch.float32)
        radii = radii.to(device=self.device, dtype=torch.float32).reshape(-1)

        if int(visibility_filter.shape[0]) != int(rendered_anchor_ids.shape[0]):
            return torch.zeros((n_anchors,), device=self.device, dtype=torch.float32)

        local_mask = self.gaussians.get_temporal_local_mask().to(device=self.device, dtype=torch.bool).reshape(-1)
        render_mask = None
        if hasattr(self.gaussians, "get_temporal_render_mask"):
            render_mask = self.gaussians.get_temporal_render_mask().to(device=self.device, dtype=torch.bool).reshape(-1)

        visible_filter = visibility_filter.clone()
        visible_filter &= torch.isfinite(rendered_xy).all(dim=-1)
        visible_filter &= torch.isfinite(rendered_depth)
        visible_filter &= rendered_depth > 0
        visible_filter &= radii > 0
        if not bool(visible_filter.any()):
            return torch.zeros((n_anchors,), device=self.device, dtype=torch.float32)

        candidate_filter = visible_filter.clone()
        candidate_filter &= ~local_mask[rendered_anchor_ids]
        if render_mask is not None and int(render_mask.shape[0]) == n_anchors:
            candidate_filter &= render_mask[rendered_anchor_ids]
        if not bool(candidate_filter.any()):
            return torch.zeros((n_anchors,), device=self.device, dtype=torch.float32)

        cand_anchor_ids = rendered_anchor_ids[candidate_filter]
        cand_slot_ids = rendered_slot_ids[candidate_filter]
        cand_xy = rendered_xy[candidate_filter]
        cand_depth = rendered_depth[candidate_filter]
        cand_r = radii[candidate_filter].clamp_min(1.0)
        max_visible_candidates = max(
            1,
            int(getattr(self.training_args, "temporal_clone_max_visible_candidates", DEFAULT_TEMPORAL_CLONE_MAX_VISIBLE_CANDIDATES)),
        )
        if int(cand_anchor_ids.numel()) > max_visible_candidates:
            perm = torch.randperm(int(cand_anchor_ids.numel()), device=self.device)[:max_visible_candidates]
            cand_anchor_ids = cand_anchor_ids[perm]
            cand_slot_ids = cand_slot_ids[perm]
            cand_xy = cand_xy[perm]
            cand_depth = cand_depth[perm]
            cand_r = cand_r[perm]
        if cand_anchor_ids.numel() == 0:
            return torch.zeros((n_anchors,), device=self.device, dtype=torch.float32)
        flat_ids = cand_anchor_ids * int(self.gaussians.n_offsets) + cand_slot_ids

        pixel_xy = torch.stack([xs.to(dtype=torch.float32), ys.to(dtype=torch.float32)], dim=-1)
        knn = min(int(self.training_args.temporal_clone_knn), int(cand_xy.shape[0]))
        if knn <= 0:
            return torch.zeros((n_anchors,), device=self.device, dtype=torch.float32)
        knn_d, knn_idx = self._topk_nearest_chunked(pixel_xy, cand_xy, k=knn)
        if knn_idx.numel() == 0:
            return torch.zeros((n_anchors,), device=self.device, dtype=torch.float32)

        offset_votes = torch.zeros(
            (n_anchors * int(self.gaussians.n_offsets),),
            device=self.device,
            dtype=torch.float32,
        )
        max_screen_radius = float(self.training_args.temporal_clone_screen_radius)
        use_depth_gate = bool(self.training_args.temporal_clone_depth_gate_enabled)
        depth_tol = float(self.training_args.temporal_clone_depth_tolerance)
        pixel_errors = torch.sqrt(mse_map[ys, xs].clamp_min(1e-12)).to(dtype=torch.float32)

        safe_knn_idx = knn_idx.clamp_min(0)
        gathered_r = cand_r[safe_knn_idx].clamp_min(1.0)
        gathered_depth = cand_depth[safe_knn_idx]
        gathered_flat_ids = flat_ids[safe_knn_idx]

        valid_nn = knn_idx >= 0
        dist_limit = torch.minimum(
            torch.full_like(knn_d, max_screen_radius),
            3.0 * gathered_r,
        )
        dist_mask = knn_d <= dist_limit
        valid_mask = valid_nn & dist_mask
        if use_depth_gate:
            surface_depths = depth_map[0, ys, xs].to(dtype=torch.float32)
            valid_surface = torch.isfinite(surface_depths) & (surface_depths > 0)
            if not bool(valid_surface.any()):
                return torch.zeros((n_anchors,), device=self.device, dtype=torch.float32)
            depth_mask = gathered_depth <= (surface_depths.unsqueeze(1) + depth_tol)
            valid_mask = valid_mask & valid_surface.unsqueeze(1) & depth_mask
        if not bool(valid_mask.any()):
            return torch.zeros((n_anchors,), device=self.device, dtype=torch.float32)

        weights = torch.exp(-0.5 * (knn_d / gathered_r) ** 2)
        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        contributions = pixel_errors.unsqueeze(1) * weights

        flat_valid = valid_mask.reshape(-1)
        if not bool(flat_valid.any()):
            return torch.zeros((n_anchors,), device=self.device, dtype=torch.float32)
        flat_offset_ids = gathered_flat_ids.reshape(-1)[flat_valid]
        flat_contributions = contributions.reshape(-1)[flat_valid]
        offset_votes.scatter_add_(0, flat_offset_ids, flat_contributions)

        anchor_votes = offset_votes.view(n_anchors, int(self.gaussians.n_offsets)).sum(dim=1)
        anchor_votes = torch.where(~local_mask, anchor_votes, torch.zeros_like(anchor_votes))
        if render_mask is not None and int(render_mask.shape[0]) == n_anchors:
            anchor_votes = torch.where(render_mask, anchor_votes, torch.zeros_like(anchor_votes))
        return anchor_votes

    @torch.no_grad()
    def _bootstrap_temporal_local_clones(self) -> int:
        logger.info(
            f"Timestep {int(self.timestep)}: stage2 uses periodic residual voting refresh."
        )
        return 0

    @torch.no_grad()
    def _refresh_temporal_local_clones(
        self,
        iteration: int,
        primary_camera,
        primary_gt_image: torch.Tensor,
    ) -> int:
        if not self.current_cameras:
            return 0

        n_anchors = int(self.gaussians.get_anchor.shape[0])
        if n_anchors <= 0:
            return 0

        aggregated_votes = torch.zeros((n_anchors,), device=self.device, dtype=torch.float32)
        for view_idx, cam in enumerate(self._sample_temporal_clone_views(primary_camera)):
            if view_idx == 0:
                gt_image = primary_gt_image
            else:
                gt_image = self._lookup_gt_image(cam)
            view_votes = self._mine_temporal_clone_candidates_for_view(cam, gt_image)
            if int(view_votes.shape[0]) == n_anchors:
                aggregated_votes += view_votes

        selected_anchor_ids = torch.where(
            aggregated_votes > float(self.training_args.temporal_clone_vote_threshold)
        )[0]
        if selected_anchor_ids.numel() == 0:
            return 0

        max_new = min(int(self.training_args.temporal_clone_max_new_anchors), int(selected_anchor_ids.numel()))
        if int(selected_anchor_ids.numel()) > max_new:
            _, top_idx = torch.topk(aggregated_votes[selected_anchor_ids], k=max_new, largest=True)
            selected_anchor_ids = selected_anchor_ids[top_idx]

        self._set_temporal_bootstrap_selected_anchors(selected_anchor_ids, n_anchors)
        self._set_temporal_bootstrap_vote_components(
            torch.zeros_like(aggregated_votes),
            torch.zeros_like(aggregated_votes),
            aggregated_votes,
        )
        self._temporal_bootstrap_vote_stats = {
            "total_anchors": int(n_anchors),
            "vote_positive_anchors": int(selected_anchor_ids.numel()),
            "eligible_structure_anchors": int(selected_anchor_ids.numel()),
            "selected_clone_parents": int(selected_anchor_ids.numel()),
            "new_local_anchors": 0,
            "vote_threshold": float(self.training_args.temporal_clone_vote_threshold),
            "refresh_interval": int(self.training_args.temporal_clone_refresh_interval),
        }

        n_new = self.append_temporal_local_clones(selected_anchor_ids, suppress_parents=True)
        self._temporal_bootstrap_vote_stats["new_local_anchors"] = int(n_new)
        if n_new > 0:
            logger.info(
                f"Timestep {int(self.timestep)} iter {int(iteration)}: residual voting refresh "
                f"selected {int(selected_anchor_ids.numel())} parents, cloned +{int(n_new)} local rows."
            )
        return int(n_new)
