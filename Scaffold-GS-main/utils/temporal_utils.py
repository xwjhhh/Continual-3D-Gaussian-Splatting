import torch


def resolve_min_observed_views(
    requirement: int | str,
    num_available_views: int,
) -> int:
    num_available_views = int(num_available_views)
    if num_available_views < 1:
        raise ValueError("num_available_views must be at least 1")

    if isinstance(requirement, str) and requirement.strip().lower() == "all":
        return num_available_views

    try:
        min_observed_views = int(requirement)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "min observed views must be 'all' or a positive integer"
        ) from exc
    if min_observed_views < 1:
        raise ValueError("min observed views must be at least 1")
    if min_observed_views > num_available_views:
        raise ValueError(
            "min observed views exceeds the number of available views: "
            f"{min_observed_views} > {num_available_views}"
        )
    return min_observed_views


def build_temporal_training_mask(
    visibility_counts: torch.Tensor,
    min_observed_views: int,
) -> torch.Tensor:
    if int(min_observed_views) < 1:
        raise ValueError("min_observed_views must be at least 1")
    if visibility_counts.ndim != 1:
        raise ValueError("visibility_counts must be a one-dimensional tensor")
    return visibility_counts >= int(min_observed_views)


def blend_observed_rows(
    temporal_values: torch.Tensor,
    base_values: torch.Tensor,
    observed_rows: torch.Tensor | None,
) -> torch.Tensor:
    if observed_rows is None:
        return temporal_values
    if temporal_values.shape != base_values.shape:
        raise ValueError(
            "Temporal and base values must have identical shapes: "
            f"{tuple(temporal_values.shape)} != {tuple(base_values.shape)}"
        )
    observed_rows = observed_rows.to(
        device=temporal_values.device,
        dtype=torch.bool,
    ).reshape(-1)
    if int(observed_rows.shape[0]) != int(temporal_values.shape[0]):
        raise ValueError(
            "Observation mask row count must match property output rows: "
            f"{int(observed_rows.shape[0])} != {int(temporal_values.shape[0])}"
        )
    row_mask = observed_rows.reshape(
        observed_rows.shape[0],
        *([1] * (temporal_values.ndim - 1)),
    )
    return torch.where(row_mask, temporal_values, base_values.detach())
