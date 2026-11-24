
"""Build and validate the temporal graph processing pipeline."""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tgad.graph.loaders import (
    TemporalNeighborHistory,
    iter_time_steps,
    iter_time_step_batches,
    load_temporal_splits,
)
from tgad.graph.negative_sampling import (
    sample_negative_destinations,
)


def process_split(
    name,
    data,
    history,
    node_time_step,
    batch_size,
):
    """Process a split without leaking same-step events."""

    total_events = 0
    total_batches = 0
    total_steps = 0
    total_historical_edges_returned = 0
    historical_edges_seen = history.loader.cur_e_id

    for step, group in iter_time_steps(data):

        # Sample using ALL positives from this time step.
        negative_dst = sample_negative_destinations(
            events=group,
            node_time_step=node_time_step,
            seed=42 + step,
        )

        assert negative_dst.shape == group.dst.shape
        assert torch.all(
            node_time_step[negative_dst] == step
        )

        offset = 0
        step_batches = 0

        for _, batch in iter_time_step_batches(
            group, batch_size=batch_size
        ):
            n = batch.num_events

            batch_neg_dst = negative_dst[
                offset:offset + n
            ]

            offset += n

            # All nodes needed to evaluate this batch.
            node_ids = torch.unique(
                torch.cat([
                    batch.src,
                    batch.dst,
                    batch_neg_dst,
                ])
            )

            # History contains earlier time steps only.
            n_id, edge_index, e_id = history.query(
                node_ids, step
            )
            total_historical_edges_returned += e_id.numel()

            assert torch.all(batch.t == step)
            assert batch.num_events <= batch_size

            # Historical event IDs must refer only
            # to previously committed interactions.
            if e_id.numel() > 0:
                assert int(e_id.max()) < historical_edges_seen

            # Validate returned local graph indexing.
            assert edge_index.shape[0] == 2

            if edge_index.numel() > 0:
                assert int(edge_index.min()) >= 0
                assert int(edge_index.max()) < len(n_id)

            total_batches += 1
            step_batches += 1
            total_events += n

        assert offset == group.num_events

        # Commit only AFTER every batch in this step.
        history.commit_step(step, group)

        historical_edges_seen += group.num_events
        total_steps += 1

        print(
            f"[{name}] Step {step:2d} | "
            f"Events: {group.num_events:5d} | "
            f"Batches: {step_batches:3d} | "
            f"History: {historical_edges_seen}"
        )

    assert total_events == data.num_events

    return {
        "events": total_events,
        "batches": total_batches,
        "time_steps": total_steps,
        "historical_edges_returned": total_historical_edges_returned,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")

    # Load Module 1 artifacts.
    train_data, val_data, test_data = (
        load_temporal_splits(args.processed_dir)
    )

    node_time_step = torch.load(
        args.processed_dir / "node_time_step.pt",
        weights_only=True,
        map_location="cpu",
    )

    history = TemporalNeighborHistory(
        num_nodes=len(node_time_step),
        k=args.neighbors,
    )

    history.reset()
    results = {}

    # Maintain history continuously across splits.
    for name, data in [
        ("train", train_data),
        ("validation", val_data),
        ("test", test_data),
    ]:
        results[name] = process_split(
            name=name,
            data=data,
            history=history,
            node_time_step=node_time_step,
            batch_size=args.batch_size,
        )

    summary_path = (
        PROJECT_ROOT
        / "outputs"
        / "metrics"
        / "module2_graph_summary.json"
    )

    summary_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    summary_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nModule 2 integration completed!")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
