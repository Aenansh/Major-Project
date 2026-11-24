"""Reserved for negative sampling of temporal interactions."""

"""Negative sampling for temporal graph interactions."""

from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import TemporalData


def sample_negative_destinations(
    events: TemporalData,
    node_time_step: torch.Tensor,
    seed: int = 42,
) -> torch.Tensor:
    """
    Sample one negative destination for each positive event.

    Rules:
        - Candidate destination belongs to the event's time step.
        - Candidate cannot equal the source node.
        - Candidate cannot form an observed positive edge
          from the same source at this time step.

    Returns:
        torch.LongTensor of shape [num_events].
    """
    if events.num_events == 0:
        return torch.empty(0, dtype=torch.long)

    if node_time_step.ndim != 1:
        raise ValueError("node_time_step must be 1D")

    timestamps = torch.unique(events.t)

    if timestamps.numel() != 1:
        raise ValueError(
            "Expected events from exactly one time step"
        )

    step = int(timestamps.item())

    src = events.src.cpu().long().tolist()
    dst = events.dst.cpu().long().tolist()

    node_steps = node_time_step.cpu().long()

    num_nodes = node_steps.numel()

    if any(
        u < 0 or v < 0 or u >= num_nodes or v >= num_nodes
        for u, v in zip(src, dst)
    ):
        raise ValueError("Event contains an invalid node ID")

    candidate_nodes = torch.where(
        node_steps == step
    )[0].tolist()

    if not candidate_nodes:
        raise ValueError(
            f"No candidate nodes at time step {step}"
        )

    # Collect all observed outgoing destinations.
    positive_destinations = defaultdict(set)

    for u, v in zip(src, dst):
        positive_destinations[u].add(v)

    rng = np.random.default_rng(seed)

    negative_dst = []
    valid_cache = {}

    for u in src:
        if u not in valid_cache:
            forbidden = positive_destinations[u] | {u}

            valid = [
                v for v in candidate_nodes
                if v not in forbidden
            ]

            if not valid:
                raise ValueError(
                    f"No valid negative destination "
                    f"for source {u} at step {step}"
                )

            valid_cache[u] = valid

        sampled = rng.choice(valid_cache[u])
        negative_dst.append(int(sampled))

    return torch.tensor(
        negative_dst,
        dtype=torch.long
    )
