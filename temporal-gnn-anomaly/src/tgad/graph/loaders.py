"""Load temporal graph splits and group their events by time step."""

from collections.abc import Iterator
from pathlib import Path

import torch
from torch_geometric.data import TemporalData


def load_temporal_splits(processed_dir: Path):
    """
    Load Module 1 temporal event splits.

    Returns:
        train_data, val_data, test_data
    """
    processed_dir = Path(processed_dir)

    train_data = torch.load(
        processed_dir / "train_data.pt",
        weights_only=False
    )

    val_data = torch.load(
        processed_dir / "val_data.pt",
        weights_only=False
    )

    test_data = torch.load(
        processed_dir / "test_data.pt",
        weights_only=False
    )

    return train_data, val_data, test_data


def iter_time_steps(data: TemporalData) -> Iterator[tuple[int, TemporalData]]:
    """Yield one group per time step from chronologically sorted events."""
    if not isinstance(data, TemporalData):
        raise TypeError("Expected a TemporalData object")
    timestamps = getattr(data, "t", None)
    if timestamps is None:
        raise ValueError("TemporalData must contain event timestamps")

    steps, counts = torch.unique_consecutive(timestamps, return_counts=True)
    if torch.any(steps[1:] <= steps[:-1]):
        raise ValueError("Events must be sorted by time step")

    start = 0
    for step, count in zip(steps.tolist(), counts.tolist()):
        end = start + count
        yield step, data[start:end]
        start = end


def iter_time_step_batches(
    data: TemporalData, batch_size: int = 200
) -> Iterator[tuple[int, TemporalData]]:
    """Yield chronological mini-batches, each from a single time step."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    for step, group in iter_time_steps(data):
        num_events = group.num_events
        for start in range(0, num_events, batch_size):
            end = min(start + batch_size, num_events)
            yield step, group[start:end]
