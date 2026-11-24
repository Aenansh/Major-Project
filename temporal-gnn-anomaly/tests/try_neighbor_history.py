
import torch
from torch_geometric.data import TemporalData

from tgad.graph.loaders import (
    TemporalNeighborHistory,
    iter_time_steps,
)

# Create a small temporal graph
data = TemporalData(
    src=torch.tensor([0, 0, 1], dtype=torch.long),
    dst=torch.tensor([1, 2, 3], dtype=torch.long),
    t=torch.tensor([1, 2, 3], dtype=torch.long),
    msg=torch.empty((3, 0)),
)

history = TemporalNeighborHistory(
    num_nodes=4,
    k=10,
)

# Process time steps in chronological order
for step, events in iter_time_steps(data):
    print(f"\nTime step: {step}")

    node_ids = torch.unique(
        torch.cat([events.src, events.dst])
    )

    n_id, edge_index, e_id = history.query(
        node_ids, step
    )

    print("Historical nodes:", n_id.tolist())
    print("Historical edge IDs:", e_id.tolist())
    print("Local edge index:", edge_index)

    # Update history only after querying
    history.commit_step(step, events)

print("\nSUCCESS: Neighbour history test completed!")

history.reset()

history.commit_step(1, data[:1])

try:
    history.query(torch.tensor([0]), step=1)
except ValueError as error:
    print("Leakage check passed:", error)
else:
    raise AssertionError("Same-step query should have failed")