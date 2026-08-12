"""Compact CL-Splats history recording and exact scene recovery.

Each continual update stores the pre-update rows selected by the active mask,
plus masks describing their row positions before and after the update. Frozen
rows keep both their values and relative order. Applying records in reverse
therefore reconstructs any earlier scene exactly at the raw-parameter level.

Command-line example::

    python -m ircgs.history \
        --ply outputs/point_cloud_final.ply \
        --history-dir outputs/history \
        --time 0 \
        --out outputs/recovered_t0.ply
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch


FORMAT_VERSION = 1
PARAM_KEYS = (
    "xyz",
    "features_dc",
    "features_rest",
    "opacity",
    "scaling",
    "rotation",
)


@dataclasses.dataclass
class TimestepRecord:
    """Delta required to revert one continual-learning timestep."""

    timestep: int
    active_start_mask: torch.Tensor
    pre_params: Dict[str, torch.Tensor]
    active_end_mask: Optional[torch.Tensor] = None
    format_version: int = FORMAT_VERSION


def _validate_params(params: Dict[str, torch.Tensor]) -> int:
    missing = [key for key in PARAM_KEYS if key not in params]
    if missing:
        raise KeyError(f"Missing Gaussian parameter keys: {missing}")

    row_counts = {key: int(params[key].shape[0]) for key in PARAM_KEYS}
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"Gaussian parameter row counts do not match: {row_counts}")
    return next(iter(row_counts.values()))


def gaussian_params_from_model(gaussians) -> Dict[str, torch.Tensor]:
    """Return a legacy 3DGS model's raw, pre-activation tensors."""
    return {
        "xyz": gaussians._xyz,
        "features_dc": gaussians._features_dc,
        "features_rest": gaussians._features_rest,
        "opacity": gaussians._opacity,
        "scaling": gaussians._scaling,
        "rotation": gaussians._rotation,
    }


class HistoryRecorder:
    """Capture and persist compact per-timestep scene deltas."""

    def __init__(self, records: Optional[Iterable[TimestepRecord]] = None) -> None:
        self.records = list(records or [])

    @property
    def open_record(self) -> Optional[TimestepRecord]:
        if self.records and self.records[-1].active_end_mask is None:
            return self.records[-1]
        return None

    def begin_timestep(
        self,
        timestep: int,
        active_mask: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> None:
        """Snapshot the old rows selected by ``active_mask`` before update."""
        if self.open_record is not None:
            raise RuntimeError("Cannot begin a timestep while another record is open")
        if any(record.timestep == int(timestep) for record in self.records):
            raise ValueError(f"History already contains timestep {timestep}")

        n_rows = _validate_params(params)
        mask = active_mask.detach().bool().reshape(-1)
        if int(mask.shape[0]) != n_rows:
            raise ValueError(
                f"active_start_mask has {mask.shape[0]} rows, expected {n_rows}"
            )

        pre = {
            key: params[key].detach()[mask].cpu().clone()
            for key in PARAM_KEYS
        }
        self.records.append(
            TimestepRecord(
                timestep=int(timestep),
                active_start_mask=mask.cpu().clone(),
                pre_params=pre,
            )
        )

    def end_timestep(self, active_mask: torch.Tensor) -> TimestepRecord:
        """Finalize the open record with its end-of-update active lineage."""
        record = self.open_record
        if record is None:
            raise RuntimeError("end_timestep called without an open record")
        record.active_end_mask = active_mask.detach().bool().reshape(-1).cpu().clone()
        return record

    def save_record(self, out_dir: str | Path, record: TimestepRecord) -> Path:
        """Atomically save one finalized record."""
        if record.active_end_mask is None:
            raise ValueError(f"History record t={record.timestep} is not finalized")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"t{record.timestep:04d}.pt"
        temp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(dataclasses.asdict(record), temp_path)
        os.replace(temp_path, path)
        return path

    def save(self, out_dir: str | Path) -> List[Path]:
        return [self.save_record(out_dir, record) for record in self.records]

    @staticmethod
    def load(history_dir: str | Path) -> List[TimestepRecord]:
        history_dir = Path(history_dir)
        records: List[TimestepRecord] = []
        for path in sorted(history_dir.glob("t*.pt")):
            try:
                data = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                data = torch.load(path, map_location="cpu")
            record = TimestepRecord(**data)
            if int(record.format_version) != FORMAT_VERSION:
                raise ValueError(
                    f"Unsupported history format {record.format_version} in {path}"
                )
            records.append(record)

        timesteps = [record.timestep for record in records]
        if len(timesteps) != len(set(timesteps)):
            raise ValueError(f"Duplicate history timesteps in {history_dir}")
        for previous, current in zip(timesteps, timesteps[1:]):
            if current != previous + 1:
                raise ValueError(
                    f"History is not contiguous: t{previous} is followed by t{current}"
                )
        return records


def revert_timestep(
    params: Dict[str, torch.Tensor],
    record: TimestepRecord,
) -> Dict[str, torch.Tensor]:
    """Reconstruct the state immediately before ``record.timestep``."""
    n_current = _validate_params(params)
    if record.active_end_mask is None:
        raise ValueError(f"History record t={record.timestep} is not finalized")

    end_mask = record.active_end_mask.bool().reshape(-1)
    start_mask = record.active_start_mask.bool().reshape(-1)
    if int(end_mask.shape[0]) != n_current:
        raise ValueError(
            f"t={record.timestep} end mask has {end_mask.shape[0]} rows, "
            f"but current state has {n_current}"
        )

    inactive_now = ~end_mask
    expected_inactive = int((~start_mask).sum().item())
    actual_inactive = int(inactive_now.sum().item())
    if actual_inactive != expected_inactive:
        raise ValueError(
            "Frozen-scene invariant was violated for "
            f"t={record.timestep}: expected {expected_inactive} inactive survivor rows, "
            f"found {actual_inactive}"
        )

    n_previous = int(start_mask.shape[0])
    n_active_start = int(start_mask.sum().item())
    out: Dict[str, torch.Tensor] = {}
    for key in PARAM_KEYS:
        current = params[key].detach().cpu()
        saved = record.pre_params[key].detach().cpu()
        if int(saved.shape[0]) != n_active_start:
            raise ValueError(
                f"t={record.timestep} saved {key} has {saved.shape[0]} rows, "
                f"expected {n_active_start}"
            )
        if tuple(saved.shape[1:]) != tuple(current.shape[1:]):
            raise ValueError(
                f"t={record.timestep} {key} shape mismatch: "
                f"saved={tuple(saved.shape)}, current={tuple(current.shape)}"
            )

        previous = torch.empty(
            (n_previous,) + tuple(current.shape[1:]),
            dtype=current.dtype,
        )
        previous[~start_mask] = current[inactive_now]
        previous[start_mask] = saved.to(dtype=current.dtype)
        out[key] = previous
    return out


def recover_state(
    final_params: Dict[str, torch.Tensor],
    records: List[TimestepRecord],
    target_time: int,
) -> Dict[str, torch.Tensor]:
    """Recursively recover raw Gaussian parameters at ``target_time``."""
    _validate_params(final_params)
    if not records:
        raise ValueError("No history records were provided")

    ordered = sorted(records, key=lambda record: record.timestep)
    earliest = ordered[0].timestep - 1
    latest = ordered[-1].timestep
    if not earliest <= int(target_time) <= latest:
        raise ValueError(
            f"target_time={target_time} outside recoverable range [{earliest}, {latest}]"
        )

    state = {key: value.detach().cpu().clone() for key, value in final_params.items()}
    for record in reversed(ordered):
        if record.timestep <= int(target_time):
            break
        state = revert_timestep(state, record)
    return state


def load_raw_ply(path: str | Path) -> Dict[str, torch.Tensor]:
    """Load raw legacy-3DGS parameters without activation round-trips."""
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"]
    xyz = np.stack([vertex[axis] for axis in ("x", "y", "z")], axis=-1).astype(np.float32)
    features_dc = np.stack(
        [vertex[f"f_dc_{index}"] for index in range(3)], axis=-1
    ).astype(np.float32)[:, None, :]

    rest_names = sorted(
        [prop.name for prop in vertex.properties if prop.name.startswith("f_rest_")],
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if len(rest_names) % 3 != 0:
        raise ValueError(f"Invalid f_rest property count: {len(rest_names)}")
    if rest_names:
        rest_flat = np.stack([vertex[name] for name in rest_names], axis=-1).astype(np.float32)
        features_rest = rest_flat.reshape(xyz.shape[0], 3, -1).transpose(0, 2, 1)
    else:
        features_rest = np.zeros((xyz.shape[0], 0, 3), dtype=np.float32)

    opacity = np.asarray(vertex["opacity"], dtype=np.float32)[:, None]
    scaling_names = sorted(
        [prop.name for prop in vertex.properties if prop.name.startswith("scale_")],
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    rotation_names = sorted(
        [prop.name for prop in vertex.properties if prop.name.startswith("rot_")],
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    scaling = np.stack([vertex[name] for name in scaling_names], axis=-1).astype(np.float32)
    rotation = np.stack([vertex[name] for name in rotation_names], axis=-1).astype(np.float32)

    return {
        "xyz": torch.from_numpy(xyz),
        "features_dc": torch.from_numpy(features_dc),
        "features_rest": torch.from_numpy(features_rest.copy()),
        "opacity": torch.from_numpy(opacity),
        "scaling": torch.from_numpy(scaling),
        "rotation": torch.from_numpy(rotation),
    }


def export_raw_ply(params: Dict[str, torch.Tensor], path: str | Path) -> None:
    """Write raw parameters using the legacy 3DGS PLY property layout."""
    from plyfile import PlyData, PlyElement

    n_rows = _validate_params(params)
    arrays = {
        key: params[key].detach().cpu().numpy().astype(np.float32)
        for key in PARAM_KEYS
    }
    xyz = arrays["xyz"]
    normals = np.zeros_like(xyz)
    features_dc = arrays["features_dc"].transpose(0, 2, 1).reshape(n_rows, -1)
    features_rest = arrays["features_rest"].transpose(0, 2, 1).reshape(n_rows, -1)
    opacity = arrays["opacity"].reshape(n_rows, -1)
    scaling = arrays["scaling"].reshape(n_rows, -1)
    rotation = arrays["rotation"].reshape(n_rows, -1)

    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{index}" for index in range(features_dc.shape[1])]
    names += [f"f_rest_{index}" for index in range(features_rest.shape[1])]
    names += ["opacity"]
    names += [f"scale_{index}" for index in range(scaling.shape[1])]
    names += [f"rot_{index}" for index in range(rotation.shape[1])]

    values = np.concatenate(
        [xyz, normals, features_dc, features_rest, opacity, scaling, rotation],
        axis=1,
    )
    dtype = [(name, "f4") for name in names]
    elements = np.empty(n_rows, dtype=dtype)
    for column, name in enumerate(names):
        elements[name] = values[:, column]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover an exact historical CL-Splats scene from final PLY and deltas."
    )
    parser.add_argument("--ply", "-p", required=True, help="Final/latest PLY checkpoint.")
    parser.add_argument(
        "--history-dir",
        default="outputs/history",
        help="Directory containing tNNNN.pt history records.",
    )
    parser.add_argument("--time", "-t", required=True, type=int, help="Target timestep.")
    parser.add_argument("--out", "-o", required=True, help="Recovered PLY output path.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    records = HistoryRecorder.load(args.history_dir)
    if not records:
        raise SystemExit(f"No history records found in {args.history_dir}")

    final_params = load_raw_ply(args.ply)
    recovered = recover_state(final_params, records, args.time)
    export_raw_ply(recovered, args.out)
    print(
        f"Recovered timestep {args.time}: {recovered['xyz'].shape[0]} Gaussians -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
