import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import numpy as np
from nltk.corpus import movie_reviews

from src import paths
from src.data_io import load_train
from src.eval import binary_metrics, format_metrics
from src.stage_a import heuristic_predict
from src.stage_b import TfidfSentiment


def load_nltk_corpus():
    texts, labels = [], []
    for fileid in movie_reviews.fileids('neg'):
        texts.append(movie_reviews.raw(fileid))
        labels.append(0)
    for fileid in movie_reviews.fileids('pos'):
        texts.append(movie_reviews.raw(fileid))
        labels.append(1)
    return texts, np.array(labels)


def main():
    train = load_train()
    spam_mask = heuristic_predict(train['text']) == 1
    train_reviews = train[~spam_mask].reset_index(drop=True)

    print(f'train (non-spam): {len(train_reviews)} rows')

    nltk_texts, nltk_labels = load_nltk_corpus()
    print(f'nltk movie_reviews: {len(nltk_texts)} rows ({(nltk_labels == 1).sum()} pos / {(nltk_labels == 0).sum()} neg)')

    print()
    print('B1 (TF-IDF + LogReg, trained on filtered AML training reviews) on NLTK movie_reviews')
    t0 = time.perf_counter()
    model = TfidfSentiment().fit(train_reviews['text'].tolist(), train_reviews['label'].values)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    nltk_pred = model.predict(nltk_texts)
    predict_time = time.perf_counter() - t0
    metrics = binary_metrics(nltk_labels, nltk_pred)
    print(f'  train time:   {train_time:.2f}s')
    print(f'  predict time: {predict_time:.2f}s')
    print(format_metrics(metrics))

    spam_on_nltk = heuristic_predict(nltk_texts).sum()
    print(f'\nStage A on NLTK: {spam_on_nltk}/{len(nltk_texts)} flagged as spam (expected: 0 -- NLTK has no spam)')


if __name__ == '__main__':
    main()
