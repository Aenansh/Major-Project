"""Small recurring-node stream for testing temporal neighbour history.

Unlike Elliptic transaction nodes, these synthetic nodes recur at multiple
time steps. This is a structural fixture, not a labelled fraud benchmark.
"""
import torch
from torch_geometric.data import TemporalData


def recurring_node_events() -> TemporalData:
    """Return 5 timestamped interactions across four recurring nodes."""
    return TemporalData(
        src=torch.tensor([0, 0, 1, 0, 2], dtype=torch.long),
        dst=torch.tensor([1, 2, 2, 3, 3], dtype=torch.long),
        t=torch.tensor([1, 2, 2, 3, 4], dtype=torch.long),
        msg=torch.empty((5, 0), dtype=torch.float32),
    )
