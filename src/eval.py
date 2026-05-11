import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    accuracy = (y_true == y_pred).mean()
    return {
        'confusion': cm,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
    }


def format_metrics(metrics):
    cm = metrics['confusion']
    lines = [
        f'  accuracy:  {metrics["accuracy"]:.3f}',
        f'  precision: {metrics["precision"]:.3f}',
        f'  recall:    {metrics["recall"]:.3f}',
        f'  f1:        {metrics["f1"]:.3f}',
        '  confusion (rows=true, cols=pred, labels [0, 1]):',
        f'    {cm[0]}',
        f'    {cm[1]}',
    ]
    return '\n'.join(lines)
