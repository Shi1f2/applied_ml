# Generates a small hand-label oracle on the val split so Stage A can be
# scored on val too (the heuristic oracle in step 01 only covered train).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import paths
from src.data_io import load_val


SEED = 42
ORACLE_SIZE = 100


def main():
    val = load_val()
    oracle_path = paths.OUTPUTS / 'val_spam_oracle.csv'
    if oracle_path.exists():
        existing = pd.read_csv(oracle_path)
        print(f'oracle already exists: {oracle_path} ({len(existing)} rows, {existing["is_spam"].notna().sum()} labelled)')
        return

    oracle = (
        val.sample(ORACLE_SIZE, random_state=SEED)[['text', 'label']]
        .reset_index()
        .rename(columns={'index': 'val_index'})
    )
    oracle['is_spam'] = ''
    oracle.to_csv(oracle_path, index=False)
    print(f'wrote {oracle_path}')
    print(f'open in a spreadsheet, fill is_spam (1=email/spam, 0=movie review)')


if __name__ == '__main__':
    main()
