import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src import paths
from src.data_io import load_train, load_val
from src.eval import binary_metrics, format_metrics
from src.stage_a import heuristic_predict
from src.stage_b_lstm import BiLSTMSentiment


def main():
    print(f'device: {"cuda" if torch.cuda.is_available() else "cpu"}')
    if torch.cuda.is_available():
        print(f'gpu: {torch.cuda.get_device_name(0)}')
    print()

    train = load_train()
    val = load_val()

    train_reviews = train[heuristic_predict(train['text']) == 0].reset_index(drop=True)
    val_reviews = val[heuristic_predict(val['text']) == 0].reset_index(drop=True)

    print(f'train reviews: {len(train_reviews)}, val reviews: {len(val_reviews)}')
    print()

    print('B4: BiLSTM (GloVe-100 init, hidden=128, dropout=0.5, epochs=8)')
    t0 = time.perf_counter()
    model = BiLSTMSentiment(
        embed_dim=100, hidden_dim=128, dropout=0.5,
        max_len=400, batch_size=32, epochs=8, lr=1e-3,
        use_glove=True, freeze_embeddings=False, random_state=42,
    ).fit(
        train_reviews['text'].tolist(),
        train_reviews['label'].values,
        val_texts=val_reviews['text'].tolist(),
        val_labels=val_reviews['label'].values,
    )
    train_time = time.perf_counter() - t0

    val_pred = model.predict(val_reviews['text'].tolist())
    metrics = binary_metrics(val_reviews['label'].values, val_pred)
    print()
    print(f'  total train time: {train_time:.1f}s')
    print(format_metrics(metrics))

    history_path = paths.OUTPUTS / 'bilstm_history.csv'
    pd.DataFrame(model.history).to_csv(history_path, index=False)
    print(f'wrote {history_path}')

    pred_path = paths.OUTPUTS / 'val_bilstm_predictions.csv'
    pd.DataFrame({
        'label': val_reviews['label'].values,
        'pred': val_pred,
        'proba': model.predict_proba(val_reviews['text'].tolist()),
        'text': val_reviews['text'].values,
    }).to_csv(pred_path, index=False)
    print(f'wrote {pred_path}')


if __name__ == '__main__':
    main()
