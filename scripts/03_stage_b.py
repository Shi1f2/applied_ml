import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import paths
from src.data_io import load_all
from src.eval import binary_metrics, format_metrics
from src.stage_a import heuristic_predict
from src.stage_b import EmbeddingSentiment, PretrainedEmbeddingSentiment, TfidfSentiment


def filter_reviews(df):
    spam_mask = heuristic_predict(df['text']) == 1
    return df[~spam_mask].reset_index(drop=True)


def main():
    train, val, _ = load_all()

    train_reviews = filter_reviews(train)
    val_reviews = filter_reviews(val)

    print(f'train reviews: {len(train_reviews)} (filtered out {len(train) - len(train_reviews)} spam)')
    print(f'val reviews:   {len(val_reviews)} (filtered out {len(val) - len(val_reviews)} spam)')
    print(f'train label balance: {train_reviews["label"].value_counts().to_dict()}')
    print(f'val label balance:   {val_reviews["label"].value_counts().to_dict()}')
    print()

    results = {}

    print('B1: TF-IDF (1-2grams) + LogReg')
    t0 = time.perf_counter()
    b1 = TfidfSentiment().fit(train_reviews['text'].tolist(), train_reviews['label'].values)
    b1_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    b1_val_pred = b1.predict(val_reviews['text'].tolist())
    b1_predict_time = time.perf_counter() - t0
    b1_metrics = binary_metrics(val_reviews['label'].values, b1_val_pred)
    print(f'  train time:   {b1_train_time:.2f}s')
    print(f'  predict time: {b1_predict_time:.2f}s')
    print(format_metrics(b1_metrics))
    results['B1'] = (b1_metrics, b1_train_time)

    print()
    print('B2: Word2Vec (200d) + LogReg')
    t0 = time.perf_counter()
    b2 = EmbeddingSentiment().fit(train_reviews['text'].tolist(), train_reviews['label'].values)
    b2_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    b2_val_pred = b2.predict(val_reviews['text'].tolist())
    b2_predict_time = time.perf_counter() - t0
    b2_metrics = binary_metrics(val_reviews['label'].values, b2_val_pred)
    print(f'  train time:   {b2_train_time:.2f}s')
    print(f'  predict time: {b2_predict_time:.2f}s')
    print(format_metrics(b2_metrics))
    results['B2'] = (b2_metrics, b2_train_time)

    print()
    print('B3: pre-trained GloVe (glove-wiki-gigaword-100) + LogReg')
    t0 = time.perf_counter()
    b3 = PretrainedEmbeddingSentiment().fit(train_reviews['text'].tolist(), train_reviews['label'].values)
    b3_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    b3_val_pred = b3.predict(val_reviews['text'].tolist())
    b3_predict_time = time.perf_counter() - t0
    b3_metrics = binary_metrics(val_reviews['label'].values, b3_val_pred)
    print(f'  train time:   {b3_train_time:.2f}s (includes one-time GloVe download/load)')
    print(f'  predict time: {b3_predict_time:.2f}s')
    print(format_metrics(b3_metrics))
    results['B3'] = (b3_metrics, b3_train_time)

    val_pred_df = val_reviews[['label']].copy()
    val_pred_df['b1_pred'] = b1_val_pred
    val_pred_df['b1_proba'] = b1.predict_proba(val_reviews['text'].tolist())
    val_pred_df['b2_pred'] = b2_val_pred
    val_pred_df['b2_proba'] = b2.predict_proba(val_reviews['text'].tolist())
    val_pred_df['b3_pred'] = b3_val_pred
    val_pred_df['b3_proba'] = b3.predict_proba(val_reviews['text'].tolist())
    val_pred_df['text'] = val_reviews['text'].values
    val_pred_df.to_csv(paths.OUTPUTS / 'val_stage_b_predictions.csv', index=False)

    print()
    print('Summary:')
    print(f'  B1 (TF-IDF):     acc={results["B1"][0]["accuracy"]:.4f}  f1={results["B1"][0]["f1"]:.4f}  train={results["B1"][1]:.2f}s')
    print(f'  B2 (W2V):        acc={results["B2"][0]["accuracy"]:.4f}  f1={results["B2"][0]["f1"]:.4f}  train={results["B2"][1]:.2f}s')
    print(f'  B3 (GloVe-100d): acc={results["B3"][0]["accuracy"]:.4f}  f1={results["B3"][0]["f1"]:.4f}  train={results["B3"][1]:.2f}s')


if __name__ == '__main__':
    main()
