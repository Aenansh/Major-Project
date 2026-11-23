"""Prepare the Elliptic dataset and export temporal-graph artifacts."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data' / 'raw')
    parser.add_argument('--processed-dir', type=Path, default=PROJECT_ROOT / 'data' / 'processed')
    parser.add_argument('--message-mode', choices=['empty', 'source'], default='empty',
                        help='empty is the baseline; source is an experimental alternative')
    parser.add_argument('--allow-nonstandard-counts', action='store_true',
                        help='for small synthetic datasets with all 49 time steps')
    args = parser.parse_args()

    # Run directly from a source checkout after installing the requirements.
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
    from tgad.data.ingestion import prepare

    print(f'Project root: {PROJECT_ROOT}')
    print(f'Reading raw data from: {args.data_dir.resolve()}')
    print(f'Saving processed data to: {args.processed_dir.resolve()}')
    prepare(args.data_dir, args.processed_dir, args.message_mode,
            strict_counts=not args.allow_nonstandard_counts)


if __name__ == '__main__':
    main()
