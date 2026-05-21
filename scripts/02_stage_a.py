# Stage A — train and evaluate the spam filter, then save per-row spam
# predictions for train/val/test so Stage B can drop spam rows from training
# and the final 3-class output can re-insert the dummy spam label.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import paths
from src.data_io import load_all
from src.eval import binary_metrics, format_metrics
from src.stage_a import TfidfSpamFilter, heuristic_predict


def main():
    train, val, test = load_all()

    # Hand-labelled oracle from step 01 — used only as ground truth to score Stage A.
    oracle = pd.read_csv(paths.OUTPUTS / 'spam_oracle.csv').dropna(subset=['is_spam'])
    oracle['is_spam'] = oracle['is_spam'].astype(int)

    print('Approach 1: heuristic (Subject: header at line start)')
    heuristic_pred = heuristic_predict(oracle['text'])
    heuristic_metrics = binary_metrics(oracle['is_spam'], heuristic_pred)
    print(format_metrics(heuristic_metrics))

    print()
    print('Approach 2: TF-IDF (1-2grams) + LogReg, weak-labelled by heuristic on train')
    spam_filter = TfidfSpamFilter().fit(train['text'].tolist())
    tfidf_pred = spam_filter.predict(oracle['text'].tolist())
    tfidf_metrics = binary_metrics(oracle['is_spam'], tfidf_pred)
    print(format_metrics(tfidf_metrics))

    train_pred = spam_filter.predict(train['text'].tolist())
    val_pred = spam_filter.predict(val['text'].tolist())
    test_pred = spam_filter.predict(test['text'].tolist())

    pd.Series(train_pred, name='is_spam_pred').to_csv(paths.OUTPUTS / 'train_spam_pred.csv', index=False)
    pd.Series(val_pred, name='is_spam_pred').to_csv(paths.OUTPUTS / 'val_spam_pred.csv', index=False)
    pd.Series(test_pred, name='is_spam_pred').to_csv(paths.OUTPUTS / 'test_spam_pred.csv', index=False)

    print()
    print('Predicted spam fraction across full splits:')
    print(f'  train: {train_pred.mean():.3f}  ({int(train_pred.sum())}/{len(train_pred)})')
    print(f'  val:   {val_pred.mean():.3f}  ({int(val_pred.sum())}/{len(val_pred)})')
    print(f'  test:  {test_pred.mean():.3f}  ({int(test_pred.sum())}/{len(test_pred)})')


if __name__ == '__main__':
    main()
