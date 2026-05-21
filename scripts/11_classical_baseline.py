# Classical Task 2 baseline: HOG patches at the mean shape + ridge regression.
# Trains, predicts on val, scales predictions back to original resolution and
# saves per-image NME for later comparison with the CNN.
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import paths
from src.data_io_task2 import load_train, load_val
from src.models_task2 import HogPatchSDM, preprocess_for_classical
from src.eval_task2 import per_image_nme, summarise
from src.preprocess_task2 import predictions_to_original


SEED = 42
ALPHA = 1.0


def main():
    paths.OUTPUTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    images_train, points_train = load_train()
    images_val, points_val = load_val()

    t0 = time.perf_counter()
    grays_train, pts_train_resized = preprocess_for_classical(images_train, points_train)
    grays_val, pts_val_resized = preprocess_for_classical(images_val, points_val)
    t_prep = time.perf_counter() - t0

    print(f'preprocess: {t_prep:.1f}s  '
          f'train_gray={grays_train.shape}  val_gray={grays_val.shape}')

    model = HogPatchSDM(alpha=ALPHA)
    t0 = time.perf_counter()
    model.fit(grays_train, pts_train_resized)
    t_fit = time.perf_counter() - t0
    print(f'fit:        {t_fit:.1f}s  feature_dim_per_patch={model.feature_dim_per_patch}')

    t0 = time.perf_counter()
    pred_val_128 = model.predict(grays_val)
    t_pred = time.perf_counter() - t0
    print(f'predict:    {t_pred:.1f}s  pred_val={pred_val_128.shape}')

    # NME is computed at original resolution so it's comparable across methods.
    pred_val_orig = predictions_to_original(pred_val_128)
    nme_val = per_image_nme(pred_val_orig, points_val)
    metrics = summarise(nme_val)
    print('val NME summary:')
    for k, v in metrics.items():
        print(f'  {k:>14s} = {v:.4f}')

    out_path = paths.OUTPUTS / 'task2_val_predictions_classical.npz'
    np.savez(out_path,
             pred_128=pred_val_128,
             pred_original=pred_val_orig,
             gt_original=points_val,
             nme=nme_val)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
