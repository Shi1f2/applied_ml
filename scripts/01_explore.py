import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns

from src import paths
from src.data_io import load_all
from src.spam_signals import signals_dataframe


SEED = 42
N_SAMPLES_PER_LABEL = 20
ORACLE_SIZE = 100
SAMPLE_TRUNCATE = 1200


def main():
    train, val, test = load_all()

    paths.OUTPUTS.mkdir(parents=True, exist_ok=True)
    paths.FIGURES.mkdir(parents=True, exist_ok=True)

    print(f'shapes  train={train.shape} val={val.shape} test={test.shape}')
    print(f'train labels:\n{train["label"].value_counts(dropna=False)}')
    print(f'val labels:\n{val["label"].value_counts(dropna=False)}')

    samples_path = paths.OUTPUTS / 'random_samples.txt'
    with samples_path.open('w', encoding='utf-8') as fh:
        for label, name in [(1, 'POS'), (0, 'NEG')]:
            picks = train[train['label'] == label].sample(N_SAMPLES_PER_LABEL, random_state=SEED)
            for i, text in enumerate(picks['text']):
                fh.write(f'--- {name} #{i} ---\n')
                fh.write(text[:SAMPLE_TRUNCATE])
                fh.write('\n\n')
    print(f'wrote {samples_path}')

    train = train.assign(
        char_len=train['text'].str.len(),
        word_len=train['text'].str.split().str.len(),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, ['char_len', 'word_len']):
        sns.histplot(data=train, x=col, hue='label', bins=80, ax=ax,
                     log_scale=(False, True), element='step')
        ax.set_xlabel(col)
        ax.set_ylabel('count (log)')
    fig.tight_layout()
    length_fig = paths.FIGURES / 'train_length_by_label.png'
    fig.savefig(length_fig, dpi=150)
    plt.close(fig)
    print(f'wrote {length_fig}')

    signals = signals_dataframe(train['text'])
    signals['label'] = train['label'].values
    pattern_table = signals.groupby('label').mean().T
    pattern_path = paths.OUTPUTS / 'spam_pattern_means_by_label.csv'
    pattern_table.to_csv(pattern_path)
    print(f'wrote {pattern_path}')
    print(pattern_table)

    oracle_path = paths.OUTPUTS / 'spam_oracle.csv'
    if oracle_path.exists():
        print(f'oracle already exists at {oracle_path}, skipping')
    else:
        oracle = (
            train.sample(ORACLE_SIZE, random_state=SEED)[['text', 'label']]
            .reset_index()
            .rename(columns={'index': 'train_index'})
        )
        oracle['is_spam'] = ''
        oracle.to_csv(oracle_path, index=False)
        print(f'wrote {oracle_path} — open in a spreadsheet and fill is_spam (1=spam, 0=review)')


if __name__ == '__main__':
    main()
