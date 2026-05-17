import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src import paths
from src.data_io_task2 import load_train
from src.preprocess_task2 import train_transform, eval_transform, INPUT_SIZE


SEED = 42
N_IMAGES = 4
N_AUGS_PER_IMAGE = 5
POINT_COLOURS = ['red', 'blue', 'lime', 'magenta', 'cyan']


def draw(ax, img, pts, title):
    ax.imshow(np.clip(img, 0, 1))
    for k, (x, y) in enumerate(pts):
        ax.plot(x, y, 'o', color=POINT_COLOURS[k % len(POINT_COLOURS)],
                markersize=5, markeredgecolor='black', markeredgewidth=0.4)
    ax.set_title(title, fontsize=8)
    ax.set_xlim(-2, INPUT_SIZE + 2)
    ax.set_ylim(INPUT_SIZE + 2, -2)
    ax.axis('off')


def main():
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    images, points = load_train()

    pick = rng.choice(len(images), size=N_IMAGES, replace=False)

    cols = N_AUGS_PER_IMAGE + 1
    fig, axes = plt.subplots(N_IMAGES, cols, figsize=(2.0 * cols, 2.0 * N_IMAGES))
    if N_IMAGES == 1:
        axes = axes[None, :]

    for row, idx in enumerate(pick):
        img_eval, pts_eval = eval_transform(images[idx], points[idx])
        draw(axes[row, 0], img_eval, pts_eval, f'idx={idx} (eval)')
        for col in range(N_AUGS_PER_IMAGE):
            img_aug, pts_aug = train_transform(images[idx], points[idx], rng)
            draw(axes[row, col + 1], img_aug, pts_aug, f'aug {col + 1}')

    fig.suptitle('eval_transform vs. train_transform — points must stay on the eyes/nose/mouth', fontsize=10)
    fig.tight_layout()
    out = paths.FIGURES / 'task2_augmentation_check.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
