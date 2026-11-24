# Temporal GNN Anomaly Detection

A major-project implementation for temporal graph anomaly detection on the Elliptic Bitcoin transaction dataset. The repository is being developed incrementally, with data ingestion and temporal graph construction implemented so far.

## Current progress

| Area | Status | Notes |
|---|---|---|
| Elliptic data ingestion | Complete | Loads features, labels, and edgelist files; exports chronological graph splits and node metadata. |
| Temporal graph construction | Implemented; final verification pending | Adds chronological neighbor history, batched time-step processing, and negative-destination sampling. |
| Graph integration script | Implemented; full-dataset run reported | `02_build_graph.py` validates train/validation/test splits and writes a graph summary. |
| Tests | Expanded; execution pending | Ingestion tests and Module 2 grouping, history, negative sampling and integration tests. |
| TGN model | Not implemented | The model, embedding, and decoder modules are currently reserved for implementation. |
| Anomaly scoring | Not implemented | Scoring currently contains package scaffolding only. |
| Training pipeline | Not implemented | `03_train_tgn.py` is reserved; no end-to-end training loop is present yet. |
| Evaluation | Not implemented | Baseline, ablation, and metrics modules exist as scaffolding; the evaluation command is reserved. |
| Experiments | Not implemented | The experiment runner is reserved and no experiment results are checked in. |
| Dashboard | Not implemented | `dashboard/app.py` is reserved for the future visualization interface. |
| Reports and notebooks | Pending | Directories are prepared, but no project report or research notebook is currently present. |

### Implemented modules

- `src/tgad/data/` — dataset ingestion and synthetic data support
- `src/tgad/graph/loaders.py` — temporal split loading and neighbor-history construction
- `src/tgad/graph/negative_sampling.py` — negative destination sampling
- `scripts/01_ingestion_pipeline.py` — raw Elliptic dataset preparation
- `scripts/02_build_graph.py` — temporal graph processing and integration validation
- `tests/` — graph, ingestion, model, and scoring tests

## Dataset setup

Place the three Elliptic input files in `temporal-gnn-anomaly/data/raw/`:

- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

Raw and generated datasets are ignored by Git. The processed output is written to `temporal-gnn-anomaly/data/processed/`.

## Installation

Python 3.10 or newer is required. From the repository root:

```sh
cd temporal-gnn-anomaly
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Running the implemented stages

Run all commands from `temporal-gnn-anomaly/`.

### 1. Prepare data

```sh
python scripts/01_ingestion_pipeline.py
```

Useful options:

```sh
python scripts/01_ingestion_pipeline.py --help
python scripts/01_ingestion_pipeline.py --message-mode source
python scripts/01_ingestion_pipeline.py --allow-nonstandard-counts
```

### 2. Build and validate the temporal graph

```sh
python scripts/02_build_graph.py --batch-size 200 --neighbors 10
```

This writes a summary to `outputs/metrics/module2_graph_summary.json`.

### Run tests

```sh
python -m unittest discover -s tests
```

## Project layout

```text
temporal-gnn-anomaly/
├── configs/default.yaml
├── dashboard/              # Reserved for anomaly visualization
├── data/
│   ├── raw/                # Elliptic CSV inputs
│   ├── processed/          # Generated chronological splits
│   └── synthetic/          # Synthetic test data
├── outputs/
│   ├── metrics/
│   ├── plots/
│   └── predictions/
├── scripts/
│   ├── 01_ingestion_pipeline.py
│   ├── 02_build_graph.py
│   ├── 03_train_tgn.py     # Reserved
│   ├── 04_score_anomalies.py
│   ├── 05_evaluate.py
│   └── 06_run_experiments.py
├── src/tgad/
│   ├── data/
│   ├── graph/
│   ├── model/              # Reserved TGN implementation
│   ├── anomaly/            # Reserved anomaly scoring
│   └── evaluation/         # Reserved metrics and baselines
├── tests/
├── checkpoints/
└── reports/
```

## Remaining work

1. Implement the Temporal Graph Network encoder and memory modules.
2. Implement the link decoder and end-to-end TGN training loop.
3. Add temporal edge batching and training with chronological validation.
4. Implement anomaly scores from reconstruction/link-prediction errors.
5. Complete evaluation for precision, recall, F1, ROC-AUC, PR-AUC, and baselines.
6. Add ablation experiments and uncertainty estimation.
7. Generate plots, metrics, checkpoints, and prediction outputs.
8. Build the dashboard for suspicious transaction exploration.
9. Add experiment reproducibility instructions and final project/report documentation.
10. Expand test coverage and add automated CI checks.


## Semester deliverable — Modules 1 and 2

This semester's scope is data ingestion and temporal graph construction only. Modules 3–7 (TGN model, anomaly scoring, baselines, extensions and dashboard) are deferred. The current implementation **does not detect anomalies or train a neural network**.

### Module 2 files

- src/tgad/graph/loaders.py — chronological grouping, step-aware batches, and history.
- src/tgad/graph/negative_sampling.py — seeded directed candidate-negative sampling.
- scripts/02_build_graph.py — end-to-end integration, time-step-safe history updates, and JSON summary.
- src/tgad/data/synthetic.py — recurring-node fixture to test real historical retrieval.
- tests/test_graph.py — unit and integration acceptance tests.

### Run and validate locally

Run from the temporal-gnn-anomaly directory in VS Code:

    python -m pip install -e .
    python -m unittest discover -s tests -v
    python scripts/01_ingestion_pipeline.py
    python scripts/02_build_graph.py --batch-size 200 --neighbors 10

Inspect outputs/metrics/module2_graph_summary.json and retain console output for the project report.

Acceptance criteria: all tests pass, the real Elliptic ingestion completes, all 49 steps and 234,355 interactions are accounted for, history persists across train/validation/test, and same-step/future event history cannot enter historical queries. These are acceptance **requirements**, not a claim the expanded tests have already run on your computer.

### Methodological limitations

Elliptic nodes are transactions, not persistent accounts; ingested edges link nodes within the same coarse time step. With strict earlier-step history, query nodes generally have no previous-step neighbours and historical_edges_returned may be zero. This is expected, not necessarily a bug. The synthetic recurring-node fixture tests the opposite case.

The history loader stores edges bidirectionally while negative sampling preserves direction. We exclude known same-step outgoing positives, but sampled negatives are not verified absent edges. Empty event messages are the initial default; source-feature messages require an availability/leakage audit. Within-step interaction ordering is unknown: score the whole step before committing it. The integration currently validates batches and histories but does not yet export ready-made TGN training subgraphs.
