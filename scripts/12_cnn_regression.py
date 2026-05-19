import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src import paths
from src.data_io_task2 import load_train, load_val
from src.cnn_task2 import LandmarkCNN, TrainDataset, EvalDataset
from src.preprocess_task2 import INPUT_SIZE, predictions_to_original
from src.eval_task2 import per_image_nme, summarise


SEED = 42
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 150
PATIENCE = 20
HUBER_BETA = 0.05
NUM_WORKERS = 0


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x).cpu().numpy()
            preds.append(out)
            targets.append(y.numpy())
    preds = np.concatenate(preds, axis=0) * INPUT_SIZE
    targets = np.concatenate(targets, axis=0) * INPUT_SIZE
    return preds, targets


def train_one_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimiser.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


def main():
    set_seed(SEED)
    paths.OUTPUTS.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    images_train, points_train = load_train()
    images_val, points_val = load_val()

    train_ds = TrainDataset(images_train, points_train, seed=SEED)
    val_ds = EvalDataset(images_val, points_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)

    mean_shape_orig = points_train.mean(axis=0).astype(np.float32)
    mean_shape_norm = torch.from_numpy((mean_shape_orig / 256.0))
    model = LandmarkCNN(mean_shape_norm=mean_shape_norm).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'params: {n_params:,}')

    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=MAX_EPOCHS)
    criterion = nn.SmoothL1Loss(beta=HUBER_BETA)

    history = []
    best_nme = float('inf')
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0

    t_start = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimiser, criterion, device)
        scheduler.step()
        pred_128, gt_128 = evaluate(model, val_loader, device)
        pred_orig = predictions_to_original(pred_128)
        nme = per_image_nme(pred_orig, points_val).mean()
        dt = time.perf_counter() - t0
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_nme': float(nme), 'time_s': dt})
        improved = nme < best_nme - 1e-5
        marker = '*' if improved else ' '
        print(f'epoch {epoch:3d} {marker} train_loss={train_loss:.4f}  val_nme={nme:.4f}  ({dt:.1f}s)')
        if improved:
            best_nme = float(nme)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f'early stop at epoch {epoch}')
                break

    t_total = time.perf_counter() - t_start
    print(f'best val_nme={best_nme:.4f} at epoch {best_epoch}  total {t_total:.1f}s')

    model.load_state_dict(best_state)
    pred_128, _ = evaluate(model, val_loader, device)
    pred_orig = predictions_to_original(pred_128)
    nme_val = per_image_nme(pred_orig, points_val)
    metrics = summarise(nme_val)
    print('val NME summary (best epoch):')
    for k, v in metrics.items():
        print(f'  {k:>14s} = {v:.4f}')

    ckpt_path = paths.OUTPUTS / 'task2_cnn_best.pt'
    torch.save({'state_dict': best_state,
                'best_epoch': best_epoch,
                'best_nme': best_nme,
                'history': history}, ckpt_path)
    print(f'wrote {ckpt_path}')

    out_path = paths.OUTPUTS / 'task2_val_predictions_cnn.npz'
    np.savez(out_path,
             pred_128=pred_128,
             pred_original=pred_orig,
             gt_original=points_val,
             nme=nme_val)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
