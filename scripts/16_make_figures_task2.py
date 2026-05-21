# Builds the two Task 2 report figures: preprocessing/augmentation pipeline
# and the CNN architecture diagram. Pure matplotlib for portable A4 rendering.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from src import paths


def add_box(ax, xy, w, h, label, fill='#E8F0FE', edge='#1A73E8', font=11, weight='bold'):
    box = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle='round,pad=0.02',
        linewidth=1.4, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, label, ha='center', va='center', fontsize=font, weight=weight, wrap=True)


def add_arrow(ax, p0, p1, label=None, font=10):
    ax.annotate('', xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle='->', lw=1.6, color='#444'))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx + 0.1, my + 0.1, label, fontsize=font, color='#444')


def make_pipeline_figure(out_path):
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    add_box(ax, (0.2, 6.5), 2.5, 1.2,
            'raw image\n256×256×3 uint8\n+ 5 GT points',
            fill='#FFF5E6', edge='#F29900', font=10)

    add_arrow(ax, (2.75, 7.1), (3.6, 7.1))

    add_box(ax, (3.6, 6.5), 2.7, 1.2,
            'resize 256→128\n(scale points ×0.5)',
            fill='#E8F0FE', edge='#1A73E8', font=10)

    add_arrow(ax, (6.35, 7.1), (7.4, 7.1))

    add_box(ax, (7.4, 6.5), 1.6, 1.2, 'split:',
            fill='#F1F3F4', edge='#5F6368', font=11)

    add_arrow(ax, (9.05, 7.4), (10.05, 7.95), label='training')
    add_arrow(ax, (9.05, 6.8), (10.05, 6.25), label='eval/test')

    train_x = 10.05
    train_y = 4.45
    train_w = 3.7
    train_h = 3.55
    add_box(ax, (train_x, train_y), train_w, train_h, '',
            fill='#FEF7E0', edge='#F9AB00', font=12)
    ax.text(train_x + train_w / 2, train_y + train_h - 0.3,
            'training augmentation',
            ha='center', va='center', fontsize=11, weight='bold')

    aug_steps = [
        'hflip p=0.5  (swap 0↔1, 3↔4)',
        'rotate ±20°,  scale ±10%,  trans ±5%',
        'divide by 255  → [0, 1]',
        'brightness/contrast ±20%',
        'Gaussian noise  σ ≤ 0.03',
    ]
    sub_h = 0.45
    sub_pad = 0.08
    for i, lbl in enumerate(aug_steps):
        y = train_y + train_h - 0.85 - i * (sub_h + sub_pad)
        add_box(ax, (train_x + 0.2, y - sub_h / 2), train_w - 0.4, sub_h, lbl,
                fill='#FFFFFF', edge='#F9AB00', font=9, weight='normal')

    eval_x = 10.05
    eval_y = 1.15
    eval_w = 3.7
    eval_h = 1.6
    add_box(ax, (eval_x, eval_y), eval_w, eval_h, '',
            fill='#E6F4EA', edge='#0F9D58', font=12)
    ax.text(eval_x + eval_w / 2, eval_y + eval_h - 0.3,
            'eval / test transform',
            ha='center', va='center', fontsize=11, weight='bold')
    add_box(ax, (eval_x + 0.2, eval_y + 0.25), eval_w - 0.4, 0.55,
            'divide by 255  → [0, 1]   (no augmentation)',
            fill='#FFFFFF', edge='#0F9D58', font=9, weight='normal')

    add_box(ax, (3.5, 3.6), 5.0, 1.0,
            'CNN input  (3, 128, 128)',
            fill='#FCE8E6', edge='#D93025', font=11)

    add_arrow(ax, (10.05, 4.45), (8.5, 4.1))
    add_arrow(ax, (10.05, 1.95), (8.5, 4.1))

    add_box(ax, (3.5, 1.7), 5.0, 1.0,
            'predicted points\n×128, ×2  →  256-frame',
            fill='#FCE8E6', edge='#D93025', font=10)

    add_arrow(ax, (6.0, 3.55), (6.0, 2.75), label='CNN inference')

    fig.suptitle('Figure 1: Task 2 preprocessing and augmentation pipeline. Training augmentation '
                 'is the key robustness mechanism; eval/test paths share only the deterministic resize + scaling.',
                 fontsize=10, y=0.02, wrap=True)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def make_cnn_figure(out_path):
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    levels = [
        ('input image  (3, 128, 128)', '#FFF5E6', '#F29900'),
        ('Conv block 1:  Conv3×3 (32) → BN → ReLU → Conv3×3 (32) → BN → ReLU → MaxPool 2×\n→ (32, 64, 64)',
         '#E8F0FE', '#1A73E8'),
        ('Conv block 2:  same pattern, 64 channels  → (64, 32, 32)', '#E8F0FE', '#1A73E8'),
        ('Conv block 3:  same pattern, 128 channels → (128, 16, 16)', '#E8F0FE', '#1A73E8'),
        ('Conv block 4:  same pattern, 256 channels → (256, 8, 8)', '#E8F0FE', '#1A73E8'),
        ('Global average pool  → (256,)', '#F1F3F4', '#5F6368'),
        ('Linear 256 → 128  → ReLU  → Dropout p=0.2', '#F1F3F4', '#5F6368'),
        ('Linear 128 → 10  (Δx, Δy per landmark)\nweights & bias initialised to 0', '#FEF7E0', '#F9AB00'),
        ('reshape (5, 2)  +  mean training shape  →  predicted (x, y)', '#E6F4EA', '#0F9D58'),
    ]
    box_h = 0.78
    pad = 0.15
    y_start = 9.2
    box_w = 8.0
    box_x = 1.5

    coords = []
    for i, (label, fill, edge) in enumerate(levels):
        y = y_start - i * (box_h + pad)
        add_box(ax, (box_x, y - box_h / 2), box_w, box_h, label,
                fill=fill, edge=edge, font=10, weight='normal')
        coords.append((box_x + box_w / 2, y - box_h / 2, box_x + box_w / 2, y + box_h / 2))

    for i in range(len(coords) - 1):
        bottom_x, bottom_y, _, _ = coords[i]
        top_x, _, _, top_y = coords[i + 1]
        add_arrow(ax, (bottom_x, bottom_y), (top_x, top_y))

    side_text = (
        'Loss:  SmoothL1 (Huber, β = 0.05)\n'
        'Optimiser:  Adam (lr 1e-3, wd 1e-4)\n'
        'Cosine LR schedule (T_max = 150)\n'
        'Gradient clip:  5.0\n'
        'Batch size:  32\n'
        'Max epochs:  150\n'
        'Early stop:  patience 20 on val NME\n'
        'Trainable params:  1.21 M\n'
        'Targets: points / 128 in normalised\n'
        'image space; predictions are residual\n'
        'from the mean training shape.'
    )
    ax.text(10.0, 5.0, side_text, ha='left', va='center', fontsize=9,
            bbox=dict(facecolor='#FAFAFA', edgecolor='#5F6368', boxstyle='round,pad=0.4'))

    fig.suptitle('Figure 2: Task 2 CNN architecture. Mean-shape residual head shifts the optimisation '
                 'problem from absolute coordinate regression to local correction.',
                 fontsize=10, y=0.02, wrap=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    pipeline_path = paths.FIGURES / 'task2_pipeline.png'
    cnn_path = paths.FIGURES / 'task2_cnn_architecture.png'
    make_pipeline_figure(pipeline_path)
    make_cnn_figure(cnn_path)
    print(f'wrote {pipeline_path}')
    print(f'wrote {cnn_path}')


if __name__ == '__main__':
    main()
