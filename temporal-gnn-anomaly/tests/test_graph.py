"""Tests for negative sampling on temporal graphs."""

from pathlib import Path
import sys
import unittest

import torch
from torch_geometric.data import TemporalData


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tgad.graph.negative_sampling import sample_negative_destinations


class NegativeSamplingTest(unittest.TestCase):
    def test_rejects_positive_endpoint_from_another_time_step(self) -> None:
        node_steps = torch.tensor([1, 1, 2])
        for source, destination in ((2, 1), (0, 2)):
            with self.subTest(source=source, destination=destination):
                events = TemporalData(
                    src=torch.tensor([source]), dst=torch.tensor([destination]),
                    t=torch.tensor([1]),
                )
                with self.assertRaisesRegex(
                    ValueError, "Positive event endpoints must belong to the event time step"
                ):
                    sample_negative_destinations(events, node_steps)

    def test_unobserved_reverse_edge_can_be_a_negative(self) -> None:
        events = TemporalData(
            src=torch.tensor([0, 1]), dst=torch.tensor([1, 2]), t=torch.tensor([1, 1]),
        )
        negatives = sample_negative_destinations(events, torch.ones(3, dtype=torch.long))
        torch.testing.assert_close(negatives, torch.tensor([2, 0]))

    def test_seed_is_reproducible_and_varies_by_time_step(self) -> None:
        node_steps = torch.tensor([1] * 8 + [2] * 8)
        events = TemporalData(
            src=torch.full((32,), 0), dst=torch.full((32,), 1), t=torch.full((32,), 1),
        )
        first = sample_negative_destinations(events, node_steps)
        torch.testing.assert_close(first, sample_negative_destinations(events, node_steps))

        later_events = TemporalData(src=events.src + 8, dst=events.dst + 8, t=events.t + 1)
        later = sample_negative_destinations(later_events, node_steps)
        self.assertFalse(torch.equal(first, later - 8))
