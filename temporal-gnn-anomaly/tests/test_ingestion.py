"""Test the ingestion command and Elliptic CSV validation."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from tgad.data.ingestion import prepare


class IngestionTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.raw = self.directory / 'raw'
        self.raw.mkdir()
        self.processed = self.directory / 'processed'
        self.feature_rows = []
        self.class_rows = [['txId', 'class']]
        self.edge_rows = [['txId1', 'txId2']]
        for step in range(1, 50):
            source, destination = 2 * step, 2 * step + 1
            for transaction in (source, destination):
                self.feature_rows.append([transaction, step, *[transaction / 10] * 165])
            self.class_rows.extend([[source, '2'], [destination, '1']])
            self.edge_rows.append([source, destination])

    def write_csvs(self) -> None:
        for name, rows in (
            ('features', self.feature_rows),
            ('classes', self.class_rows),
            ('edgelist', self.edge_rows),
        ):
            with (self.raw / f'elliptic_txs_{name}.csv').open('w', newline='') as file:
                csv.writer(file).writerows(rows)

    def assert_ingestion_rejected(self, message: str) -> None:
        self.write_csvs()
        with self.assertRaisesRegex(ValueError, message):
            prepare(self.raw, self.processed, message_mode='empty', strict_counts=False)
        self.assertFalse(self.processed.exists())

    def test_ingestion_command_exports_chronological_splits(self) -> None:
        self.class_rows[1][1] = 'unknown'
        self.write_csvs()

        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / 'scripts' / '01_ingestion_pipeline.py'),
             '--data-dir', str(self.raw), '--processed-dir', str(self.processed),
             '--allow-nonstandard-counts'],
            cwd=self.directory, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((self.processed / 'metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['event_counts'], {'train': 29, 'val': 5, 'test': 15})
        self.assertEqual(metadata['num_nodes'], 98)
        self.assertEqual(metadata['label_counts'], {'-1': 1, '0': 48, '1': 49})

    def test_rejects_edge_referencing_missing_transaction(self) -> None:
        self.edge_rows[1][1] = 999
        self.assert_ingestion_rejected('Edges reference missing transaction nodes')

    def test_rejects_blank_feature_transaction_id(self) -> None:
        self.feature_rows[0][0] = ''
        self.write_csvs()
        with self.assertRaises(ValueError):
            prepare(self.raw, self.processed, message_mode='empty', strict_counts=False)
        self.assertFalse(self.processed.exists())

    def test_rejects_feature_transaction_without_class_row(self) -> None:
        del self.class_rows[1]
        self.assert_ingestion_rejected('Some feature nodes have no matching class')

    def test_rejects_duplicate_feature_transaction_ids(self) -> None:
        self.feature_rows[1][0] = self.feature_rows[0][0]
        self.assert_ingestion_rejected('Duplicate feature transaction IDs')

    def test_rejects_duplicate_class_transaction_ids(self) -> None:
        self.class_rows[2][0] = self.class_rows[1][0]
        self.assert_ingestion_rejected('Duplicate class transaction IDs')

    def test_rejects_edges_across_time_steps(self) -> None:
        self.edge_rows[1][1] = self.edge_rows[2][1]
        self.assert_ingestion_rejected('1 cross-step edges')

    def test_rejects_unmapped_label(self) -> None:
        self.class_rows[1][1] = 'unexpected'
        self.assert_ingestion_rejected('Unmapped fraud class values')

    def test_rejects_missing_feature_column(self) -> None:
        for row in self.feature_rows:
            row.pop()
        self.assert_ingestion_rejected('Expected 167 feature CSV columns')

    def test_rejects_missing_feature_value(self) -> None:
        self.feature_rows[0][2] = ''
        self.assert_ingestion_rejected('Raw features contain NaN or infinite values')
