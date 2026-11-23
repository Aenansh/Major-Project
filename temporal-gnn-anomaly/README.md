# Temporal GNN Anomaly Detection

The ingestion pipeline prepares the Elliptic dataset for temporal graph anomaly detection. It exports chronological graph splits, normalized node features, labels, and preprocessing metadata.

## Run ingestion

Place `elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, and `elliptic_txs_edgelist.csv` in `data/raw/`. Run both commands from the project root.

```sh
python -m pip install -r requirements.txt
python scripts/01_ingestion_pipeline.py
```

The pipeline writes to `data/processed/`. Run the script with `--help` to see the available overrides. To import `tgad` from outside the checkout, install the package with `python -m pip install -e .`.

## Project layout

- `src/tgad/` contains the ingestion implementation and package scaffolding for later modules.
- `scripts/` contains the ingestion command and reserved entry points for later pipeline stages.
- `configs/default.yaml` records the ingestion defaults for reference. The ingestion command accepts overrides through its CLI flags.
- `dashboard/`, `notebooks/`, and `reports/` provide locations for the dashboard, notebooks, and project writing.
- `data/`, `checkpoints/`, and `outputs/` hold local data and generated artifacts.

Run the ingestion smoke test with `python -m unittest discover -s tests`.
