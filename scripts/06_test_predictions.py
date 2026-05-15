import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import paths
from src.data_io import load_test, load_train, load_val
from src.save_csv import save_as_csv
from src.stage_a import heuristic_predict
from src.stage_b_lstm import BiLSTMSentiment


SPAM_DUMMY_LABEL = -1


def main():
    train = load_train()
    val = load_val()
    test = load_test()

    train_reviews = train[heuristic_predict(train['text']) == 0].reset_index(drop=True)
    val_reviews = val[heuristic_predict(val['text']) == 0].reset_index(drop=True)
    print(f'train (non-spam): {len(train_reviews)} rows')
    print(f'val (non-spam):   {len(val_reviews)} rows')

    model = BiLSTMSentiment(
        embed_dim=100, hidden_dim=128, dropout=0.5,
        max_len=400, batch_size=32, epochs=15, lr=1e-3,
        use_glove=True, freeze_embeddings=False, random_state=42,
        patience=2,
    ).fit(
        train_reviews['text'].tolist(),
        train_reviews['label'].values,
        val_texts=val_reviews['text'].tolist(),
        val_labels=val_reviews['label'].values,
    )

    test_spam_mask = heuristic_predict(test['text']) == 1
    print(f'test: {len(test)} rows, Stage A flagged {test_spam_mask.sum()} as spam')

    predictions = np.empty((len(test),), dtype=np.int64)
    predictions[test_spam_mask] = SPAM_DUMMY_LABEL

    review_indices = np.where(~test_spam_mask)[0]
    review_texts = test.loc[review_indices, 'text'].tolist()
    review_predictions = model.predict(review_texts)
    predictions[review_indices] = review_predictions

    out = predictions.reshape(-1, 1).astype(np.float64)
    print(f'output shape: {out.shape}')
    print(f'value counts: spam={SPAM_DUMMY_LABEL}: {(out == SPAM_DUMMY_LABEL).sum()}, '
          f'neg=0: {(out == 0).sum()}, pos=1: {(out == 1).sum()}')

    paths.OUTPUTS.mkdir(parents=True, exist_ok=True)
    save_as_csv(out, str(paths.OUTPUTS))
    csv_path = paths.OUTPUTS / 'results_task1.csv'
    print(f'wrote {csv_path}')

    sanity = np.loadtxt(csv_path, delimiter=',')
    print(f'sanity-check: shape on disk = {sanity.shape}, first 5 = {sanity[:5].tolist()}')


if __name__ == '__main__':
    main()
