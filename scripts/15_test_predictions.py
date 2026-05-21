# Generates the final Task 2 test CSV from the trained CNN checkpoint.
# Predictions are made at 128x128 then rescaled to 256x256 before saving so
# the (554, 5, 2) output matches the original image resolution.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src import paths
from src.data_io_task2 import load_train, load_test
from src.cnn_task2 import LandmarkCNN
from src.preprocess_task2 import (
    eval_transform, predictions_to_original, INPUT_SIZE, ORIGINAL_SIZE, to_chw,
)
from src.save_csv_task2 import save_as_csv


SEED = 42
BATCH = 64


def load_cnn(device):
    images_train, points_train = load_train()
    mean_shape_norm = torch.from_numpy(points_train.mean(axis=0).astype(np.float32) / ORIGINAL_SIZE)
    model = LandmarkCNN(mean_shape_norm=mean_shape_norm).to(device)
    ckpt = torch.load(paths.OUTPUTS / 'task2_cnn_best.pt', map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def predict(model, images, device):
    n = len(images)
    out = np.zeros((n, 5, 2), dtype=np.float32)
    for s in range(0, n, BATCH):
        chunk = images[s:s + BATCH]
        tensors = []
        for img in chunk:
            x, _ = eval_transform(img, None)
            tensors.append(torch.from_numpy(to_chw(x.astype(np.float32))))
        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            pred = model(batch_tensor).cpu().numpy()
        out[s:s + BATCH] = pred * INPUT_SIZE
    return predictions_to_original(out)


def main():
    paths.OUTPUTS.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    model = load_cnn(device)
    images_test = load_test()
    print(f'test shape: {images_test.shape}')

    pred = predict(model, images_test, device)
    print(f'pred shape: {pred.shape}  dtype={pred.dtype}')
    print(f'pred min={pred.min():.2f}  max={pred.max():.2f}  mean={pred.mean():.2f}')

    npz_path = paths.OUTPUTS / 'task2_test_predictions.npz'
    np.savez(npz_path, pred=pred)
    print(f'wrote {npz_path}')

    save_as_csv(pred, location=str(paths.OUTPUTS))
    csv_path = paths.OUTPUTS / 'results_task2.csv'
    print(f'wrote {csv_path}')

    written = np.loadtxt(csv_path, delimiter=',')
    assert written.shape == (554, 10), f'unexpected csv shape {written.shape}'
    print(f'csv shape verified: {written.shape}')

    print('first 3 rows:')
    for i in range(3):
        print(f'  test[{i}] points = {pred[i].tolist()}')


if __name__ == '__main__':
    main()
