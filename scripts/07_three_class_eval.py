# End-to-end 3-class evaluation (spam / neg / pos) on the hand-labelled val
# oracle — produces the required 3-class confusion matrix for the report.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src import paths
from src.data_io import load_train, load_val
from src.stage_a import heuristic_predict
from src.stage_b import TfidfSentiment


SPAM_DUMMY = -1


def main():
    train = load_train()
    val = load_val()
    oracle = pd.read_csv(paths.OUTPUTS / 'val_spam_oracle.csv').dropna(subset=['is_spam'])
    oracle['is_spam'] = oracle['is_spam'].astype(int)

    train_reviews = train[heuristic_predict(train['text']) == 0].reset_index(drop=True)
    model = TfidfSentiment().fit(train_reviews['text'].tolist(), train_reviews['label'].values)

    oracle_texts = oracle['text'].tolist()
    spam_pred = heuristic_predict(oracle_texts)
    sentiment_pred = model.predict(oracle_texts)
    # Spam overrides sentiment in the final 3-class label.
    end_to_end = np.where(spam_pred == 1, SPAM_DUMMY, sentiment_pred)

    true_class = np.where(oracle['is_spam'].values == 1, SPAM_DUMMY, oracle['label'].values).astype(int)

    classes = [SPAM_DUMMY, 0, 1]
    class_names = ['spam', 'neg(0)', 'pos(1)']
    cm = confusion_matrix(true_class, end_to_end, labels=classes)

    print('3-class confusion matrix (rows=true, cols=predicted):')
    print(f'           {class_names[0]:>8} {class_names[1]:>8} {class_names[2]:>8}')
    for i, name in enumerate(class_names):
        row = cm[i]
        print(f'  {name:>6}  {row[0]:>8} {row[1]:>8} {row[2]:>8}')

    accuracy = (end_to_end == true_class).mean()
    print(f'\noverall 3-class accuracy on val oracle: {accuracy:.3f}')

    spam_correct = ((true_class == SPAM_DUMMY) & (end_to_end == SPAM_DUMMY)).sum()
    spam_total = (true_class == SPAM_DUMMY).sum()
    print(f'spam recall:    {spam_correct}/{spam_total}  ({spam_correct/max(spam_total,1):.3f})')

    review_mask = true_class != SPAM_DUMMY
    rev_correct = ((end_to_end == true_class) & review_mask).sum()
    rev_total = review_mask.sum()
    print(f'review accuracy: {rev_correct}/{rev_total}  ({rev_correct/max(rev_total,1):.3f})')

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        paths.OUTPUTS / 'val_oracle_3class_confusion.csv'
    )


if __name__ == '__main__':
    main()
