
import torch
from torch_geometric.data import TemporalData

from tgad.graph.negative_sampling import (
    sample_negative_destinations
)

# Four nodes, all appearing at time step 1.
node_time_step = torch.tensor([1, 1, 1, 1])

# Observed positive interactions:
# 0 -> 1
# 0 -> 2
events = TemporalData(
    src=torch.tensor([0, 0]),
    dst=torch.tensor([1, 2]),
    t=torch.tensor([1, 1]),
    msg=torch.empty((2, 0)),
)

negative_dst = sample_negative_destinations(
    events=events,
    node_time_step=node_time_step,
    seed=42,
)

print("Positive sources:", events.src.tolist())
print("Positive destinations:", events.dst.tolist())
print("Negative destinations:", negative_dst.tolist())

# Node 3 is the only valid negative for source 0.
assert torch.equal(
    negative_dst,
    torch.tensor([3, 3])
)

# Ensure sampled pairs do not match positive edges.
positive_edges = set(
    zip(events.src.tolist(), events.dst.tolist())
)

for u, v in zip(
    events.src.tolist(),
    negative_dst.tolist()
):
    assert u != v
    assert (u, v) not in positive_edges

print("Negative sampling test passed!")
