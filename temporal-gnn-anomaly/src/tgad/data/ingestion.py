"""Elliptic temporal-graph ingestion and reproducible exports.

The default empty message is a deliberate data-interface baseline, not proof
that the downstream model is leakage-free. Elliptic's per-step event order is
unknown and must be handled explicitly during temporal training.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import TemporalData

EXPECTED_NODES = 203_769
EXPECTED_EDGES = 234_355
N_FEATURES = 165
TRAIN_END, VAL_END, TEST_END = 29, 34, 49
FEATURE_COLS = [f"feature_{i}" for i in range(1, N_FEATURES + 1)]

def require(condition: bool, message: str) -> None:
    """Explicit validations remain active even if Python is run with -O."""
    if not condition:
        raise ValueError(message)


def prepare(data_dir: Path, processed_dir: Path, message_mode: str, strict_counts: bool) -> None:
    paths = {
        'features': data_dir / 'elliptic_txs_features.csv',
        'classes': data_dir / 'elliptic_txs_classes.csv',
        'edges': data_dir / 'elliptic_txs_edgelist.csv',
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    require(not missing, f'Missing dataset file(s): {missing}')

    features = pd.read_csv(paths['features'], header=None)
    classes = pd.read_csv(paths['classes'], dtype={'txId': 'int64', 'class': 'string'})
    edges = pd.read_csv(paths['edges'], dtype={'txId1': 'int64', 'txId2': 'int64'})
    require(features.shape[1] == N_FEATURES + 2,
            f'Expected {N_FEATURES + 2} feature CSV columns; found {features.shape[1]}')
    require({'txId', 'class'}.issubset(classes.columns), 'Class CSV missing columns')
    require({'txId1', 'txId2'}.issubset(edges.columns), 'Edge CSV missing columns')
    if strict_counts:
        require(len(features) == EXPECTED_NODES, f'Expected {EXPECTED_NODES} nodes; found {len(features)}')
        require(len(edges) == EXPECTED_EDGES, f'Expected {EXPECTED_EDGES} edges; found {len(edges)}')
        require(len(classes) == EXPECTED_NODES, 'Class CSV row count differs from expected')
    require(len(features) > 0 and len(edges) > 0, 'Dataset cannot be empty')

    features.columns = ['txId', 'time_step', *FEATURE_COLS]
    features['txId'] = pd.to_numeric(features['txId'], errors='raise').astype('int64')
    features['time_step'] = pd.to_numeric(features['time_step'], errors='raise').astype('int64')
    for col in FEATURE_COLS:
        features[col] = pd.to_numeric(features[col], errors='raise')
    require(features['txId'].is_unique, 'Duplicate feature transaction IDs')
    require(classes['txId'].is_unique, 'Duplicate class transaction IDs')
    require(features['time_step'].between(1, TEST_END).all(), 'Time steps outside 1–49')
    require(set(features['time_step'].unique()) == set(range(1, TEST_END + 1)),
            'Expected all 49 time steps')
    require(np.isfinite(features[FEATURE_COLS].to_numpy(dtype=np.float64)).all(),
            'Raw features contain NaN or infinite values')

    features = features.sort_values(['time_step', 'txId'], kind='mergesort').reset_index(drop=True)
    features['node_id'] = np.arange(len(features), dtype=np.int64)
    tx_to_node = pd.Series(features['node_id'].to_numpy(), index=features['txId'])

    edges['edge_id'] = np.arange(len(edges), dtype=np.int64)
    edges['src'] = edges['txId1'].map(tx_to_node)
    edges['dst'] = edges['txId2'].map(tx_to_node)
    require(edges[['src', 'dst']].notna().all().all(), 'Edges reference missing transaction nodes')
    edges[['src', 'dst']] = edges[['src', 'dst']].astype('int64')
    require(edges[['src', 'dst']].ge(0).all().all(), 'Negative node IDs in edges')
    require(edges[['src', 'dst']].lt(len(features)).all().all(), 'Out-of-range node IDs in edges')

    # Elliptic's edges are expected to join two transactions in the same step.
    node_steps = features['time_step'].to_numpy(dtype=np.int64)
    src_steps = node_steps[edges['src'].to_numpy()]
    dst_steps = node_steps[edges['dst'].to_numpy()]
    cross_step = int(np.count_nonzero(src_steps != dst_steps))
    require(cross_step == 0,
            f'{cross_step} cross-step edges: cannot infer event time safely from this dataset')
    edges['time_step'] = src_steps
    edges = edges.sort_values(['time_step', 'src', 'dst', 'edge_id'],
                              kind='mergesort').reset_index(drop=True)

    mapping = {'unknown': -1, '2': 0, '1': 1}
    classes['label'] = classes['class'].astype('string').str.strip().map(mapping)
    require(classes['label'].notna().all(), 'Unmapped fraud class values')
    classes['label'] = classes['label'].astype('int8')
    features = features.merge(classes[['txId', 'label']], on='txId',
                              how='left', validate='one_to_one', sort=False)
    require(features['label'].notna().all(), 'Some feature nodes have no matching class')
    require(features['label'].isin([-1, 0, 1]).all(), 'Unexpected encoded labels')
    require(np.array_equal(features['node_id'].to_numpy(), np.arange(len(features))),
            'Node IDs not aligned to feature row positions')

    def split_masks(steps: pd.Series):
        return [steps.between(1, TRAIN_END),
                steps.between(TRAIN_END + 1, VAL_END),
                steps.between(VAL_END + 1, TEST_END)]

    node_masks = split_masks(features['time_step'])
    edge_masks = split_masks(edges['time_step'])
    require(np.logical_and.reduce([np.asarray(m) for m in node_masks]).sum() == 0,
            'Invalid overlapping node splits')
    require(sum(int(m.sum()) for m in node_masks) == len(features),
            'Node splits do not partition all nodes')
    require(sum(int(m.sum()) for m in edge_masks) == len(edges),
            'Edge splits do not partition all edges')
    train_nodes, val_nodes, test_nodes = [features.loc[m].copy() for m in node_masks]
    train_edges, val_edges, test_edges = [edges.loc[m].copy() for m in edge_masks]
    require(train_nodes['time_step'].max() < val_nodes['time_step'].min()
            < test_nodes['time_step'].min(), 'Node split chronology failed')
    require(train_edges['time_step'].max() < val_edges['time_step'].min()
            < test_edges['time_step'].min(), 'Edge split chronology failed')
    for a, b in ((train_edges, val_edges), (train_edges, test_edges), (val_edges, test_edges)):
        require(set(a['edge_id']).isdisjoint(b['edge_id']), 'Overlapping edge splits')

    # Fit on training nodes only, then transform every node with those statistics.
    scaler = StandardScaler()
    scaler.fit(train_nodes[FEATURE_COLS])
    normalized_nodes = features.copy()
    normalized_nodes.loc[:, FEATURE_COLS] = scaler.transform(features[FEATURE_COLS])
    normalized_features = normalized_nodes[FEATURE_COLS].to_numpy(dtype=np.float32)
    require(np.isfinite(normalized_features).all(), 'Scaled features contain NaN/inf')
    require(np.allclose(scaler.transform(train_nodes[FEATURE_COLS]).mean(axis=0), 0, atol=1e-5),
            'Training feature means are not approximately zero')
    node_features = torch.from_numpy(np.ascontiguousarray(normalized_features))
    node_labels = torch.tensor(normalized_nodes['label'].to_numpy(dtype=np.int64), dtype=torch.long)
    node_time_step = torch.tensor(node_steps, dtype=torch.long)

    src = torch.tensor(edges['src'].to_numpy(dtype=np.int64), dtype=torch.long)
    dst = torch.tensor(edges['dst'].to_numpy(dtype=np.int64), dtype=torch.long)
    t = torch.tensor(edges['time_step'].to_numpy(dtype=np.int64), dtype=torch.long)
    require(torch.equal(node_time_step[src], t) and torch.equal(node_time_step[dst], t),
            'Event time differs from endpoint node time')
    require(bool(torch.all(t[1:] >= t[:-1])), 'Events are not in chronological order')

    if message_mode == 'empty':
        # Zero-dimensional event message: model can learn from graph structure and time.
        msg = torch.empty((len(edges), 0), dtype=torch.float32)
    elif message_mode == 'source':
        # EXPERIMENTAL: transaction attributes may reveal contemporaneous context.
        # Do not call results leakage-free without auditing feature availability.
        msg = node_features[src].clone()
    else:
        raise ValueError(f'Unsupported message_mode: {message_mode}')

    temporal_data = TemporalData(src=src, dst=dst, t=t, msg=msg)
    event_masks = [t <= TRAIN_END, (t > TRAIN_END) & (t <= VAL_END), t > VAL_END]
    train_data, val_data, test_data = [temporal_data[m] for m in event_masks]
    require(sum(int(m.sum()) for m in event_masks) == len(edges), 'Event split partition failure')
    require(len(train_data.src) == len(train_edges), 'Train event count mismatch')
    require(len(val_data.src) == len(val_edges), 'Validation event count mismatch')
    require(len(test_data.src) == len(test_edges), 'Test event count mismatch')
    require(int(train_data.t.max()) < int(val_data.t.min()) < int(test_data.t.min()),
            'TemporalData chronology failed')

    processed_dir.mkdir(parents=True, exist_ok=True)
    normalized_nodes.to_parquet(processed_dir / 'normalized_nodes.parquet', index=False)
    edges.to_parquet(processed_dir / 'edges.parquet', index=False)
    torch.save(node_features, processed_dir / 'node_features.pt')
    torch.save(node_labels, processed_dir / 'node_labels.pt')
    torch.save(node_time_step, processed_dir / 'node_time_step.pt')
    torch.save(temporal_data, processed_dir / 'temporal_data.pt')
    torch.save(train_data, processed_dir / 'train_data.pt')
    torch.save(val_data, processed_dir / 'val_data.pt')
    torch.save(test_data, processed_dir / 'test_data.pt')
    joblib.dump(scaler, processed_dir / 'scaler.joblib')

    metadata = {
        'num_nodes': len(features),
        'num_edges': len(edges),
        'num_node_features': N_FEATURES,
        'event_message_mode': message_mode,
        'event_message_dim': int(msg.size(1)),
        'time_steps': int(features['time_step'].nunique()),
        'train_end': TRAIN_END,
        'validation_end': VAL_END,
        'test_end': TEST_END,
        'node_counts': {'train': len(train_nodes), 'val': len(val_nodes), 'test': len(test_nodes)},
        'event_counts': {'train': len(train_edges), 'val': len(val_edges), 'test': len(test_edges)},
        'label_counts': {str(k): int(v) for k, v in features['label'].value_counts().sort_index().items()},
        'notes': [
            'Labels are reserved for evaluation, not input to self-supervised training.',
            'All Elliptic timestamps are coarse steps, with no known ordering within a step.',
            'All interactions at the same step should be scored before that step is inserted into history.',
            'Fitting the scaler on train nodes prevents statistical preprocessing leakage.',
            'Node IDs are stable indices into normalized_nodes.parquet and node_features.pt.',
            'source message mode is experimental and requires a feature-availability audit.',
        ],
    }
    (processed_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    # Reload checks cover all splits and whole saved objects' tensor attributes.
    pd.testing.assert_frame_equal(normalized_nodes, pd.read_parquet(processed_dir / 'normalized_nodes.parquet'))
    pd.testing.assert_frame_equal(edges, pd.read_parquet(processed_dir / 'edges.parquet'))
    for file_name, expected in [('node_features.pt', node_features),
                                ('node_labels.pt', node_labels),
                                ('node_time_step.pt', node_time_step)]:
        require(torch.equal(expected, torch.load(processed_dir / file_name, weights_only=True)),
                f'Reloaded artifact changed: {file_name}')
    for file_name, expected in [('temporal_data.pt', temporal_data),
                                ('train_data.pt', train_data),
                                ('val_data.pt', val_data),
                                ('test_data.pt', test_data)]:
        # PyG TemporalData is a trusted local file in this pipeline.
        loaded = torch.load(processed_dir / file_name, weights_only=False)
        for key in ('src', 'dst', 't', 'msg'):
            require(torch.equal(getattr(loaded, key), getattr(expected, key)),
                    f'Reloaded {file_name}.{key} differs')
    loaded_scaler = joblib.load(processed_dir / 'scaler.joblib')
    require(np.array_equal(loaded_scaler.mean_, scaler.mean_) and
            np.array_equal(loaded_scaler.scale_, scaler.scale_), 'Scaler round-trip failed')

    print('\nModule 1 successful.')
    print(json.dumps({k: metadata[k] for k in ('num_nodes', 'num_edges', 'node_counts', 'event_counts', 'label_counts', 'event_message_mode')}, indent=2))
    print('Artifacts saved and verified in:', processed_dir)


