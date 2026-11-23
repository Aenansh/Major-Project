"""Smoke test for the ingestion command in the source checkout."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class IngestionTest(unittest.TestCase):
    def test_ingestion_command_exports_chronological_splits(self) -> None:
        script = Path(__file__).resolve().parents[1] / 'scripts' / '01_ingestion_pipeline.py'

        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / 'raw'
            raw.mkdir()
            processed = Path(directory) / 'processed'
            with (
                (raw / 'elliptic_txs_features.csv').open('w', newline='') as features,
                (raw / 'elliptic_txs_classes.csv').open('w', newline='') as classes,
                (raw / 'elliptic_txs_edgelist.csv').open('w', newline='') as edges,
            ):
                feature_rows = csv.writer(features)
                class_rows = csv.writer(classes)
                edge_rows = csv.writer(edges)
                class_rows.writerow(['txId', 'class'])
                edge_rows.writerow(['txId1', 'txId2'])
                for step in range(1, 50):
                    source, destination = 2 * step, 2 * step + 1
                    for transaction in (source, destination):
                        feature_rows.writerow([transaction, step, *[transaction / 10] * 165])
                    class_rows.writerows([[source, '2'], [destination, '1']])
                    edge_rows.writerow([source, destination])

            result = subprocess.run(
                [sys.executable, str(script), '--data-dir', str(raw),
                 '--processed-dir', str(processed), '--allow-nonstandard-counts'],
                cwd=directory, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads((processed / 'metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['event_counts'], {'train': 29, 'val': 5, 'test': 15})
            self.assertEqual(metadata['num_nodes'], 98)
