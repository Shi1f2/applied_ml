# Robustness analysis (10-mark requirement).
# Sweeps the val set through noise / blur / brightness / occlusion / rotation /
# scale at increasing intensities and records NME for both models.
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from src import paths
from src.data_io_task2 import load_train, load_val
from src.models_task2 import HogPatchSDM, preprocess_for_classical
from src.cnn_task2 import LandmarkCNN
from src.preprocess_task2 import (
    eval_transform, predictions_to_original, INPUT_SIZE, ORIGINAL_SIZE,
    affine_matrix, apply_affine, to_chw, to_unit_range,
)
from src.eval_task2 import per_image_nme


SEED = 42
ALPHA = 1.0
COLOURS = {'classical': 'tab:blue', 'cnn': 'tab:orange'}


def fit_classical():
    images_train, points_train = load_train()
    grays_train, pts_train_resized = preprocess_for_classical(images_train, points_train)
    model = HogPatchSDM(alpha=ALPHA)
    model.fit(grays_train, pts_train_resized)
    return model


def load_cnn(device):
    images_train, points_train = load_train()
    mean_shape_norm = torch.from_numpy(points_train.mean(axis=0).astype(np.float32) / ORIGINAL_SIZE)
    model = LandmarkCNN(mean_shape_norm=mean_shape_norm).to(device)
    ckpt = torch.load(paths.OUTPUTS / 'task2_cnn_best.pt', map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def predict_classical(model, images):
    grays = preprocess_for_classical(images)
    pred_128 = model.predict(grays)
    return predictions_to_original(pred_128)


def predict_cnn(model, images, device, batch=64):
    n = len(images)
    out = np.zeros((n, 5, 2), dtype=np.float32)
    for s in range(0, n, batch):
        chunk = images[s:s + batch]
        tensors = []
        for img in chunk:
            x, _ = eval_transform(img, None)
            tensors.append(torch.from_numpy(to_chw(x.astype(np.float32))))
        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            pred = model(batch_tensor).cpu().numpy()
        out[s:s + batch] = pred * INPUT_SIZE
    return predictions_to_original(out)


def perturb_noise(images, points, sigma):
    if sigma <= 0:
        return images, points
    rng = np.random.default_rng(SEED)
    out = []
    for img in images:
        noise = rng.normal(0.0, sigma, size=img.shape)
        out.append(np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8))
    return np.stack(out), points


def perturb_blur(images, points, sigma):
    if sigma <= 0:
        return images, points
    ksize = int(2 * round(3 * sigma) + 1)
    out = np.stack([cv2.GaussianBlur(img, (ksize, ksize), sigma) for img in images])
    return out, points


def perturb_brightness(images, points, shift):
    if shift == 0:
        return images, points
    out = np.clip(images.astype(np.float32) + shift, 0, 255).astype(np.uint8)
    return out, points


def perturb_occlusion(images, points, box_frac):
    if box_frac <= 0:
        return images, points
    rng = np.random.default_rng(SEED)
    h, w = images.shape[1:3]
    side = int(round(box_frac * h))
    out = images.copy()
    for img in out:
        y0 = rng.integers(0, h - side + 1)
        x0 = rng.integers(0, w - side + 1)
        img[y0:y0 + side, x0:x0 + side] = 0
    return out, points


def perturb_affine(images, points, angle_deg=0.0, scale=1.0):
    # Rotate/scale both image and points together so NME stays meaningful.
    if angle_deg == 0.0 and scale == 1.0:
        return images, points
    h, w = images.shape[1:3]
    centre = (w / 2.0, h / 2.0)
    out_imgs = []
    out_pts = []
    for img, pts in zip(images, points):
        M = affine_matrix(angle_deg, scale, 0.0, 0.0, centre)
        new_img, new_pts = apply_affine(img, pts.astype(np.float32), M)
        out_imgs.append(new_img)
        out_pts.append(new_pts)
    return np.stack(out_imgs), np.stack(out_pts)


def evaluate(model_classical, model_cnn, device, images, points):
    pred_c = predict_classical(model_classical, images)
    pred_n = predict_cnn(model_cnn, images, device)
    return float(per_image_nme(pred_c, points).mean()), float(per_image_nme(pred_n, points).mean())


def sweep(name, levels, perturb_fn, model_classical, model_cnn, device, images, points):
    rows = {'level': [], 'classical': [], 'cnn': []}
    print(f'== {name} ==')
    for lvl in levels:
        imgs_p, pts_p = perturb_fn(images, points, lvl)
        c, n = evaluate(model_classical, model_cnn, device, imgs_p, pts_p)
        rows['level'].append(lvl)
        rows['classical'].append(c)
        rows['cnn'].append(n)
        print(f'  {name}={lvl:.3f}  classical NME={c:.4f}  cnn NME={n:.4f}')
    return rows


def plot_sweep(name, rows, xlabel, save_path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(rows['level'], rows['classical'], 'o-', color=COLOURS['classical'], label='classical (HOG+ridge)')
    ax.plot(rows['level'], rows['cnn'], 's-', color=COLOURS['cnn'], label='cnn (4-block)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('mean NME')
    ax.axhline(0.10, color='grey', linestyle='--', linewidth=0.8)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=8)
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    paths.OUTPUTS.mkdir(parents=True, exist_ok=True)
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    t0 = time.perf_counter()
    model_classical = fit_classical()
    model_cnn = load_cnn(device)
    print(f'models ready ({time.perf_counter() - t0:.1f}s)')

    images_val, points_val = load_val()

    sweeps = []

    sweeps.append(('Gaussian noise', 'sigma (0-255 scale)',
                   sweep('gaussian_noise', [0, 5, 10, 20, 40, 60], perturb_noise,
                         model_classical, model_cnn, device, images_val, points_val),
                   'task2_robust_noise.png'))

    sweeps.append(('Gaussian blur', 'sigma (px)',
                   sweep('gaussian_blur', [0.0, 0.5, 1.0, 2.0, 4.0, 6.0], perturb_blur,
                         model_classical, model_cnn, device, images_val, points_val),
                   'task2_robust_blur.png'))

    sweeps.append(('Brightness shift', 'shift (grey levels)',
                   sweep('brightness', [-60, -40, -20, 0, 20, 40, 60], perturb_brightness,
                         model_classical, model_cnn, device, images_val, points_val),
                   'task2_robust_brightness.png'))

    sweeps.append(('Occlusion box', 'box side / image side',
                   sweep('occlusion', [0.0, 0.1, 0.2, 0.3, 0.4], perturb_occlusion,
                         model_classical, model_cnn, device, images_val, points_val),
                   'task2_robust_occlusion.png'))

    rot_levels = [-30, -20, -10, 0, 10, 20, 30]
    rot_rows = {'level': [], 'classical': [], 'cnn': []}
    print('== rotation ==')
    for ang in rot_levels:
        imgs_p, pts_p = perturb_affine(images_val, points_val, angle_deg=ang, scale=1.0)
        c, n = evaluate(model_classical, model_cnn, device, imgs_p, pts_p)
        rot_rows['level'].append(ang)
        rot_rows['classical'].append(c)
        rot_rows['cnn'].append(n)
        print(f'  rotation={ang:+d}deg  classical NME={c:.4f}  cnn NME={n:.4f}')
    sweeps.append(('Rotation', 'angle (degrees)', rot_rows, 'task2_robust_rotation.png'))

    scale_levels = [0.7, 0.85, 1.0, 1.15, 1.3]
    sc_rows = {'level': [], 'classical': [], 'cnn': []}
    print('== scale ==')
    for sc in scale_levels:
        imgs_p, pts_p = perturb_affine(images_val, points_val, angle_deg=0.0, scale=sc)
        c, n = evaluate(model_classical, model_cnn, device, imgs_p, pts_p)
        sc_rows['level'].append(sc)
        sc_rows['classical'].append(c)
        sc_rows['cnn'].append(n)
        print(f'  scale={sc:.2f}  classical NME={c:.4f}  cnn NME={n:.4f}')
    sweeps.append(('Scale', 'scale factor', sc_rows, 'task2_robust_scale.png'))

    summary_path = paths.OUTPUTS / 'task2_robustness.csv'
    with summary_path.open('w', encoding='utf-8') as fh:
        fh.write('perturbation,level,classical_nme,cnn_nme\n')
        for name, _, rows, _ in sweeps:
            for lvl, c, n in zip(rows['level'], rows['classical'], rows['cnn']):
                fh.write(f'{name},{lvl},{c:.6f},{n:.6f}\n')
    print(f'wrote {summary_path}')

    for name, xlabel, rows, fname in sweeps:
        plot_sweep(name, rows, xlabel, paths.FIGURES / fname)
        print(f'wrote {paths.FIGURES / fname}')


if __name__ == '__main__':
    main()
