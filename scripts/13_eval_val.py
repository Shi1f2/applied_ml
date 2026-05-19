import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src import paths
from src.data_io_task2 import load_val
from src.eval_task2 import per_image_nme, summarise, ced_curve, per_point_error


METHOD_FILES = {
    'classical (HOG + ridge)': paths.OUTPUTS / 'task2_val_predictions_classical.npz',
    'cnn (4-block)': paths.OUTPUTS / 'task2_val_predictions_cnn.npz',
}
COLOURS = {'classical (HOG + ridge)': 'tab:blue', 'cnn (4-block)': 'tab:orange'}
POINT_COLOURS = ['red', 'blue', 'lime', 'magenta', 'cyan']
POINT_NAMES = ['L eye', 'R eye', 'nose', 'L mouth', 'R mouth']


def load_method(name, path):
    d = np.load(path)
    return {
        'name': name,
        'pred': d['pred_original'],
        'gt': d['gt_original'],
        'nme': d['nme'],
    }


def write_summary_csv(methods, out_path):
    rows = []
    for m in methods:
        s = summarise(m['nme'])
        s = {'method': m['name'], **s}
        rows.append(s)
    fieldnames = list(rows[0].keys())
    with out_path.open('w', encoding='utf-8') as fh:
        fh.write(','.join(fieldnames) + '\n')
        for r in rows:
            fh.write(','.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in r.values()) + '\n')


def plot_ced(methods, out_path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    thresholds = np.linspace(0.0, 0.20, 401)
    for m in methods:
        x, y = ced_curve(m['nme'], thresholds)
        ax.plot(x, y, label=m['name'], color=COLOURS[m['name']], linewidth=2)
    ax.axvline(0.10, color='grey', linestyle='--', linewidth=0.8, label='NME = 0.10')
    ax.set_xlabel('NME (mean Euclidean error / inter-ocular distance)')
    ax.set_ylabel('fraction of images with NME ≤ threshold')
    ax.set_xlim(0, 0.20)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_title('Cumulative error distribution on validation set')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_boxplot(methods, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    data = [m['nme'] for m in methods]
    labels = [m['name'] for m in methods]
    bp = ax.boxplot(data, labels=labels, showfliers=True, patch_artist=True, widths=0.5)
    for patch, name in zip(bp['boxes'], labels):
        patch.set_facecolor(COLOURS[name])
        patch.set_alpha(0.5)
    ax.set_ylabel('per-image NME')
    ax.axhline(0.10, color='grey', linestyle='--', linewidth=0.8)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax.set_title('Per-image NME on validation set')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_point_error(methods, out_path, gt):
    iod = np.linalg.norm(gt[:, 0, :] - gt[:, 1, :], axis=-1)
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.4
    xs = np.arange(len(POINT_NAMES))
    for i, m in enumerate(methods):
        per_pt = per_point_error(m['pred'], m['gt']) / iod[:, None]
        means = per_pt.mean(axis=0)
        stds = per_pt.std(axis=0)
        ax.bar(xs + (i - 0.5) * width, means, width=width,
               yerr=stds, capsize=3, label=m['name'],
               color=COLOURS[m['name']], alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(POINT_NAMES)
    ax.set_ylabel('mean per-point error / IOD')
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax.legend(fontsize=9)
    ax.set_title('Per-landmark error on validation set')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_qualitative(method, images_val, out_path):
    nme = method['nme']
    pred = method['pred']
    gt = method['gt']
    n = len(nme)
    order = np.argsort(nme)
    best = order[:6]
    mid = order[n // 2 - 3:n // 2 + 3]
    worst = order[-6:][::-1]

    fig, axes = plt.subplots(3, 6, figsize=(13, 7))
    for row, (label, idxs) in enumerate([('best', best), ('median', mid), ('worst', worst)]):
        for col, idx in enumerate(idxs):
            ax = axes[row, col]
            ax.imshow(images_val[idx])
            for k in range(5):
                ax.plot(gt[idx, k, 0], gt[idx, k, 1], 'o',
                        color='white', markersize=8, markeredgecolor='black', markeredgewidth=1.0)
                ax.plot(pred[idx, k, 0], pred[idx, k, 1], 'x',
                        color=POINT_COLOURS[k], markersize=8, markeredgewidth=2)
            ax.set_title(f'{label} idx={idx} NME={nme[idx]:.3f}', fontsize=8)
            ax.axis('off')
    fig.suptitle(f'{method["name"]} — best (top), median (mid), worst (bottom). White ◯ = GT, coloured × = pred',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    paths.OUTPUTS.mkdir(parents=True, exist_ok=True)
    paths.FIGURES.mkdir(parents=True, exist_ok=True)

    images_val, points_val = load_val()

    methods = []
    for name, path in METHOD_FILES.items():
        m = load_method(name, path)
        methods.append(m)
        s = summarise(m['nme'])
        print(f'{name}:')
        for k, v in s.items():
            print(f'  {k:>14s} = {v:.4f}')

    summary_path = paths.OUTPUTS / 'task2_val_summary.csv'
    write_summary_csv(methods, summary_path)
    print(f'wrote {summary_path}')

    ced_path = paths.FIGURES / 'task2_ced.png'
    plot_ced(methods, ced_path)
    print(f'wrote {ced_path}')

    box_path = paths.FIGURES / 'task2_boxplot.png'
    plot_boxplot(methods, box_path)
    print(f'wrote {box_path}')

    pp_path = paths.FIGURES / 'task2_per_point.png'
    plot_per_point_error(methods, pp_path, methods[0]['gt'])
    print(f'wrote {pp_path}')

    for m in methods:
        slug = m['name'].split()[0]
        qual_path = paths.FIGURES / f'task2_qualitative_{slug}.png'
        plot_qualitative(m, images_val, qual_path)
        print(f'wrote {qual_path}')


if __name__ == '__main__':
    main()
