"""Acceptance tests for Module 2 temporal graph construction.

Run with: python -m unittest discover -s tests -v
"""
from pathlib import Path
import importlib.util
import sys
import unittest

import torch
from torch_geometric.data import TemporalData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tgad.graph.loaders import (
    TemporalNeighborHistory,
    iter_time_steps,
    iter_time_step_batches,
)
from tgad.graph.negative_sampling import sample_negative_destinations
from tgad.data.synthetic import recurring_node_events


def stream(src, dst, t):
    return TemporalData(
        src=torch.tensor(src, dtype=torch.long),
        dst=torch.tensor(dst, dtype=torch.long),
        t=torch.tensor(t, dtype=torch.long),
        msg=torch.empty((len(t), 0), dtype=torch.float32),
    )


class TimeStepLoaderTests(unittest.TestCase):
    def test_groups_are_chronological_and_complete(self):
        events = stream([0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 1, 2, 2, 4])
        groups = list(iter_time_steps(events))
        self.assertEqual([step for step, _ in groups], [1, 2, 4])
        self.assertEqual([group.num_events for _, group in groups], [2, 2, 1])
        self.assertEqual(sum(group.num_events for _, group in groups), events.num_events)
        for step, group in groups:
            self.assertTrue(bool(torch.all(group.t == step)))

    def test_batches_do_not_mix_time_steps(self):
        events = stream(list(range(8)), list(range(1, 9)), [1] * 5 + [2] * 3)
        batches = list(iter_time_step_batches(events, batch_size=2))
        self.assertEqual([(step, batch.num_events) for step, batch in batches],
                         [(1, 2), (1, 2), (1, 1), (2, 2), (2, 1)])
        self.assertEqual(sum(batch.num_events for _, batch in batches), 8)
        self.assertTrue(all(bool(torch.all(batch.t == step)) for step, batch in batches))

    def test_rejects_unsorted_timestamps(self):
        events = stream([0, 1, 2], [1, 2, 3], [1, 3, 2])
        with self.assertRaisesRegex(ValueError, "sorted"):
            list(iter_time_steps(events))

    def test_rejects_invalid_batch_size(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            list(iter_time_step_batches(stream([0], [1], [1]), batch_size=0))


class NeighborHistoryTests(unittest.TestCase):
    def test_recurring_nodes_retrieve_earlier_interactions(self):
        events = recurring_node_events()
        history = TemporalNeighborHistory(num_nodes=4, k=10)
        for step, group in iter_time_steps(events):
            if step == 3:
                n_id, edge_index, e_id = history.query(
                    torch.tensor([0], dtype=torch.long), step
                )
                self.assertIn(0, n_id.tolist())
                self.assertEqual(edge_index.shape[0], 2)
                # Node 0 appeared as source at steps 1 and 2.
                self.assertEqual(set(e_id.tolist()), {0, 1})
            history.commit_step(step, group)
        self.assertEqual(history.loader.cur_e_id, events.num_events)

    def test_current_step_not_visible_until_commit(self):
        events = stream([0, 0], [1, 2], [1, 1])
        history = TemporalNeighborHistory(num_nodes=3)
        _, _, before = history.query(torch.tensor([0]), step=1)
        self.assertEqual(before.numel(), 0)
        # Only complete steps are committed, not individual mini-batches.
        history.commit_step(1, events)
        with self.assertRaisesRegex(ValueError, "already committed"):
            history.query(torch.tensor([0]), step=1)
        _, _, after = history.query(torch.tensor([0]), step=2)
        self.assertEqual(set(after.tolist()), {0, 1})

    def test_history_persists_between_splits(self):
        history = TemporalNeighborHistory(num_nodes=4)
        history.commit_step(1, stream([0], [1], [1]))
        _, _, past = history.query(torch.tensor([0]), step=2)
        self.assertEqual(past.tolist(), [0])
        history.commit_step(2, stream([0], [2], [2]))
        self.assertEqual(history.loader.cur_e_id, 2)
        _, _, later = history.query(torch.tensor([0]), step=3)
        self.assertEqual(set(later.tolist()), {0, 1})

    def test_elliptic_style_new_nodes_have_no_prior_neighbors(self):
        events = stream([0, 2, 4], [1, 3, 5], [1, 2, 3])
        history = TemporalNeighborHistory(num_nodes=6)
        for step, group in iter_time_steps(events):
            nodes = torch.unique(torch.cat((group.src, group.dst)))
            _, _, past = history.query(nodes, step)
            self.assertEqual(past.numel(), 0)
            history.commit_step(step, group)


class NegativeSamplingTests(unittest.TestCase):
    def test_never_samples_self_or_known_positive(self):
        events = stream([0, 0, 1], [1, 2, 2], [1, 1, 1])
        node_steps = torch.ones(5, dtype=torch.long)
        samples = sample_negative_destinations(events, node_steps, seed=17)
        self.assertEqual(samples.shape, events.dst.shape)
        positives = set(zip(events.src.tolist(), events.dst.tolist()))
        for u, v in zip(events.src.tolist(), samples.tolist()):
            self.assertNotEqual(u, v)
            self.assertNotIn((u, v), positives)
            self.assertEqual(node_steps[v].item(), 1)

    def test_reproducible(self):
        events = stream([0] * 12, [1] * 12, [1] * 12)
        node_steps = torch.ones(6, dtype=torch.long)
        a = sample_negative_destinations(events, node_steps, seed=17)
        b = sample_negative_destinations(events, node_steps, seed=17)
        torch.testing.assert_close(a, b)

    def test_rejects_all_candidates_observed(self):
        events = stream([0, 0, 0], [1, 2, 3], [1, 1, 1])
        with self.assertRaisesRegex(ValueError, "No valid negative"):
            sample_negative_destinations(events, torch.ones(4, dtype=torch.long))

    def test_rejects_foreign_step_endpoint(self):
        events = stream([0], [2], [1])
        with self.assertRaisesRegex(ValueError, "endpoints must belong"):
            sample_negative_destinations(events, torch.tensor([1, 1, 2]))

    def test_empty_group(self):
        events = stream([], [], [])
        out = sample_negative_destinations(events, torch.ones(3, dtype=torch.long))
        self.assertEqual(out.shape, (0,))


class IntegrationTests(unittest.TestCase):
    def test_integrated_processing_and_cumulative_history(self):
        # Elliptic-style test: each node exists in exactly one time step.
        # Each time step has 4 candidate nodes and 1 observed edge.
        events = stream([0, 4, 8], [1, 5, 9], [1, 2, 3])
        node_steps = torch.tensor([1] * 4 + [2] * 4 + [3] * 4)

        module_path = ROOT / "scripts" / "02_build_graph.py"
        spec = importlib.util.spec_from_file_location("module2_build_graph", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        history = TemporalNeighborHistory(num_nodes=12, k=3)
        train = module.process_split("train", events[:1], history, node_steps, 1)
        val = module.process_split("validation", events[1:2], history, node_steps, 1)
        test = module.process_split("test", events[2:], history, node_steps, 1)

        self.assertEqual([train["events"], val["events"], test["events"]], [1, 1, 1])
        self.assertEqual(history.loader.cur_e_id, 3)
        self.assertEqual([train["historical_edges_returned"],
                          val["historical_edges_returned"],
                          test["historical_edges_returned"]], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
