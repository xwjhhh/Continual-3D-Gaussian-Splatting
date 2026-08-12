"""Historical state inheritance for IRC-GS."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from loguru import logger
import torch

from ircgs.models.irc_modules.common import (
    TEMPORAL_ALIVE_SENTINEL,
    to_temporal_timestep_tensor as _to_temporal_timestep_tensor,
)

if TYPE_CHECKING:
    from ircgs.models.irc_gs import ScaffoldTrainingArgs


class InheritanceMixin:
    """Restore historical anchors, adapters, and lifetime state."""
    @staticmethod
    def _pad_or_trim_temporal_payload_tensor(
        tensor: torch.Tensor,
        target_rows: int,
        fill_value,
    ) -> torch.Tensor:
        target_rows = max(int(target_rows), 0)
        current_rows = int(tensor.shape[0])
        if current_rows == target_rows:
            return tensor
        if current_rows > target_rows:
            return tensor[:target_rows]
        pad_shape = (target_rows - current_rows, *tensor.shape[1:])
        pad_tensor = torch.full(
            pad_shape,
            fill_value=fill_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat((tensor, pad_tensor), dim=0)

    def _sync_temporal_adapter_payloads(self) -> None:
        if not self._temporal_enabled() or not self.temporal_adapter_payloads:
            return
        target_rows = int(self.gaussians.get_anchor.shape[0])
        current_timestep = int(self.timestep)
        synced_payloads: Dict[int, Dict[str, Any]] = {}
        for payload_key, payload in self.temporal_adapter_payloads.items():
            if not isinstance(payload, dict):
                continue
            synced = dict(payload)
            payload_time = int(synced.get("time_step", payload_key))
            for tensor_key, fill_value in (
                ("latent", 0.0),
                ("latent_block", 0.0),
                ("last_opacity_max", 0.0),
                ("birth_timestep", current_timestep),
                ("death_timestep", TEMPORAL_ALIVE_SENTINEL),
                ("local_mask", False),
                ("local_parent_ids", -1),
                ("anomaly_score", 0.0),
                ("anomaly_count", 0),
                ("anomaly_candidate_mask", False),
                ("split_suppression_mask", False),
            ):
                tensor = synced.get(tensor_key, None)
                if isinstance(tensor, torch.Tensor):
                    synced_tensor = self._pad_or_trim_temporal_payload_tensor(
                        tensor,
                        target_rows,
                        fill_value,
                    )
                    if tensor_key in ("birth_timestep", "death_timestep"):
                        synced_tensor = _to_temporal_timestep_tensor(synced_tensor, synced_tensor.device)
                    synced[tensor_key] = synced_tensor
            synced_payloads[payload_time] = synced
        self.temporal_adapter_payloads = synced_payloads

    def _hide_new_rows_in_past_temporal_payloads(self, old_row_count: int, new_row_count: int) -> None:
        if (
            not self._temporal_enabled()
            or not self.temporal_adapter_payloads
            or int(new_row_count) <= 0
        ):
            return
        self._sync_temporal_adapter_payloads()
        start = max(int(old_row_count), 0)
        end = max(start + int(new_row_count), start)
        current_timestep = int(self.timestep)
        for payload_key, payload in self.temporal_adapter_payloads.items():
            if not isinstance(payload, dict):
                continue
            payload_time = int(payload.get("time_step", payload_key))
            if payload_time >= current_timestep:
                continue
            local_mask = payload.get("local_mask", None)
            if isinstance(local_mask, torch.Tensor) and int(local_mask.shape[0]) >= end:
                local_mask[start:end] = False
            local_parent_ids = payload.get("local_parent_ids", None)
            if isinstance(local_parent_ids, torch.Tensor) and int(local_parent_ids.shape[0]) >= end:
                local_parent_ids[start:end] = -1
            birth_timestep = payload.get("birth_timestep", None)
            if isinstance(birth_timestep, torch.Tensor) and int(birth_timestep.shape[0]) >= end:
                birth_timestep[start:end] = current_timestep
            death_timestep = payload.get("death_timestep", None)
            if isinstance(death_timestep, torch.Tensor) and int(death_timestep.shape[0]) >= end:
                death_timestep[start:end] = current_timestep
            latent = payload.get("latent", None)
            if isinstance(latent, torch.Tensor) and int(latent.shape[0]) >= end:
                latent[start:end] = 0
            latent_block = payload.get("latent_block", None)
            if isinstance(latent_block, torch.Tensor) and int(latent_block.shape[0]) >= end:
                latent_block[start:end] = 0
            birth_timestep = payload.get("birth_timestep", None)
            if isinstance(birth_timestep, torch.Tensor) and int(birth_timestep.shape[0]) >= end:
                birth_timestep[start:end] = current_timestep

    def _prune_temporal_adapter_payload_rows(self, keep_mask: torch.Tensor) -> None:
        if not self._temporal_enabled() or not self.temporal_adapter_payloads:
            return
        keep_mask = keep_mask.detach().cpu().to(dtype=torch.bool).reshape(-1)
        if keep_mask.numel() == 0:
            return
        tensor_keys = (
            "latent",
            "latent_block",
            "last_opacity_max",
            "birth_timestep",
            "death_timestep",
            "local_mask",
            "local_parent_ids",
            "anomaly_score",
            "anomaly_count",
            "anomaly_candidate_mask",
            "split_suppression_mask",
        )
        pruned_payloads: Dict[int, Dict[str, Any]] = {}
        for payload_key, payload in self.temporal_adapter_payloads.items():
            if not isinstance(payload, dict):
                continue
            pruned = dict(payload)
            payload_time = int(pruned.get("time_step", payload_key))
            for tensor_key in tensor_keys:
                tensor = pruned.get(tensor_key, None)
                if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1 and int(tensor.shape[0]) == int(keep_mask.shape[0]):
                    pruned[tensor_key] = tensor[keep_mask.to(device=tensor.device)]
            pruned_payloads[payload_time] = pruned
        self.temporal_adapter_payloads = pruned_payloads

    def _restore_lifetime_from_temporal_payload(self, payload_time: int) -> None:
        if not self._temporal_enabled() or not self.temporal_adapter_payloads:
            return
        payload = self.temporal_adapter_payloads.get(int(payload_time), None)
        if payload is None:
            payload = self.temporal_adapter_payloads.get(str(int(payload_time)), None)
        if not isinstance(payload, dict):
            return

        n_anchors = int(self.gaussians.get_anchor.shape[0])
        restored = []
        birth_timestep = payload.get("birth_timestep", None)
        if birth_timestep is not None:
            birth_timestep = _to_temporal_timestep_tensor(birth_timestep, self.device).reshape(-1)
            if int(birth_timestep.shape[0]) == n_anchors:
                self.gaussians.temporal_anchor_birth_timestep = birth_timestep
                restored.append("birth")
        death_timestep = payload.get("death_timestep", None)
        if death_timestep is not None:
            death_timestep = _to_temporal_timestep_tensor(death_timestep, self.device).reshape(-1)
            if int(death_timestep.shape[0]) == n_anchors:
                self.gaussians.temporal_anchor_death_timestep = death_timestep
                restored.append("death")
        if restored:
            logger.info(
                f"Timestep {int(self.timestep)}: restored temporal lifetime metadata "
                f"from payload t={int(payload_time)} ({'/'.join(restored)}, anchors={n_anchors})."
            )

    @torch.no_grad()
    def _log_temporal_lifetime_distribution(self, context: str) -> None:
        if not self._temporal_enabled():
            return
        birth_timestep = getattr(self.gaussians, "temporal_anchor_birth_timestep", None)
        death_timestep = getattr(self.gaussians, "temporal_anchor_death_timestep", None)
        if not isinstance(birth_timestep, torch.Tensor):
            return
        birth_timestep = birth_timestep.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        unique_birth, birth_counts = torch.unique(birth_timestep, return_counts=True)
        birth_summary = ", ".join(
            f"{int(step)}:{int(count)}"
            for step, count in zip(unique_birth.tolist(), birth_counts.tolist())
        )
        dead_count = 0
        if isinstance(death_timestep, torch.Tensor):
            death_timestep = death_timestep.detach().to(device="cpu", dtype=torch.long).reshape(-1)
            dead_count = int((death_timestep < int(TEMPORAL_ALIVE_SENTINEL)).sum().item())
        logger.info(
            f"[TemporalLifetime] {context}: anchors={int(birth_timestep.numel())}, "
            f"birth_counts={{{birth_summary}}}, dead={dead_count}"
        )

    def setup_temporal_training(self, iterations: int) -> ScaffoldTrainingArgs:
        if hasattr(self.gaussians, "current_time_step"):
            self.gaussians.current_time_step = int(self.timestep)
        training_args = self.setup_training(iterations=iterations)
        if int(self.timestep) > 0:
            self._restore_lifetime_from_temporal_payload(int(self.timestep) - 1)
        self._temporal_clone_phase_active = False
        logger.info(f"IRC-GS inheritance module source: {Path(__file__).resolve()}")
        if hasattr(training_args, "lambda_active_offsets"):
            legacy_weight = float(getattr(training_args, "lambda_active_offsets", 0.0))
            if legacy_weight != 0.0:
                setattr(training_args, "lambda_active_offsets", 0.0)
                logger.warning(
                    "Disabled legacy active-offset regularization for temporal training "
                    f"(previous lambda_active_offsets={legacy_weight:.8g})."
                )
        self.gaussians.setup_temporal_adaptation(training_args, time_step=int(self.timestep))
        self._log_temporal_lifetime_distribution(
            f"after setup_temporal_adaptation t={int(self.timestep)}"
        )
        logger.info(
            "Temporal birth-time-gated adaptation active for timestep {}: freeze base MLP trunk and train the current latent block "
            "(block_dim={}, temporal_input_dim={}, appearance_dim={}).".format(
                int(self.timestep),
                int(getattr(self.gaussians, "temporal_block_dim", 0)),
                int(getattr(self.gaussians, "temporal_input_dim", 0)),
                int(getattr(self.gaussians, "appearance_dim", 0)),
            )
        )
        self.gaussians.train()
        return training_args
