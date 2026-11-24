"""Load temporal graph splits and group their events by time step."""

from collections.abc import Iterator
from pathlib import Path

import torch
from torch_geometric.data import TemporalData
from torch_geometric.nn.models.tgn import LastNeighborLoader


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


class TemporalNeighborHistory:
    """Maintain recent neighbours from completed time steps only."""

    def __init__(self, num_nodes: int, k: int = 10):
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if k <= 0:
            raise ValueError("k must be positive")

        self.num_nodes = num_nodes
        self.k = k

        self.loader = LastNeighborLoader(
            num_nodes=num_nodes,
            size=k,
            device="cpu"
        )

        self.last_committed_step = -1

    def reset(self) -> None:
        """Clear all historical interactions."""
        self.loader.reset_state()
        self.last_committed_step = -1

    def query(self, node_ids: torch.Tensor, step: int):
        """Retrieve neighbours from strictly earlier steps."""
        if step <= self.last_committed_step:
            raise ValueError(
                "Cannot query a time step already committed "
                "to the historical graph"
            )

        node_ids = node_ids.to(
            device="cpu", dtype=torch.long
        )

        if node_ids.numel() > 0:
            if (
                node_ids.min().item() < 0
                or node_ids.max().item() >= self.num_nodes
            ):
                raise ValueError("Node ID out of range")

        return self.loader(node_ids)

    def commit_step(
        self,
        step: int,
        events: TemporalData
    ) -> None:
        """Add one completed time step to history."""
        if step <= self.last_committed_step:
            raise ValueError(
                "Time steps must be committed in increasing order"
            )

        if events.num_events == 0:
            raise ValueError("Cannot commit an empty time step")

        if not bool(torch.all(events.t == step)):
            raise ValueError(
                "Events contain timestamps different from step"
            )

        src = events.src.to(device="cpu", dtype=torch.long)
        dst = events.dst.to(device="cpu", dtype=torch.long)

        if (
            src.min().item() < 0
            or dst.min().item() < 0
            or src.max().item() >= self.num_nodes
            or dst.max().item() >= self.num_nodes
        ):
            raise ValueError("Event contains invalid node IDs")

        self.loader.insert(src, dst)
        self.last_committed_step = step
