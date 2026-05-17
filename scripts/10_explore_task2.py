import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src import paths
from src.data_io_task2 import load_train, load_val, load_test


SEED = 42
N_SAMPLE_GRID = 9
POINT_COLOURS = ['red', 'blue', 'lime', 'magenta', 'cyan']


def describe_array(name, arr):
    if arr.dtype == object:
        shapes = {x.shape for x in arr}
        dtypes = {x.dtype for x in arr}
        print(f'{name}: object array len={len(arr)} unique-shapes={shapes} unique-dtypes={dtypes}')
    else:
        print(f'{name}: shape={arr.shape} dtype={arr.dtype} min={arr.min()} max={arr.max()} mean={arr.mean():.2f}')


def stack_if_uniform(images):
    if images.dtype != object:
        return images
    shapes = {x.shape for x in images}
    if len(shapes) == 1:
        return np.stack(list(images))
    return None


def image_size_stats(images):
    if images.dtype != object:
        return np.array([images.shape[1:3]] * len(images))
    return np.array([x.shape[:2] for x in images])


def points_summary(pts):
    print(f'points: shape={pts.shape} dtype={pts.dtype}')
    print(f'  per-axis min={pts.reshape(-1, 2).min(axis=0)}  max={pts.reshape(-1, 2).max(axis=0)}')
    print(f'  per-point mean (x, y):')
    for i, mu in enumerate(pts.mean(axis=0)):
        print(f'    [{i}] = ({mu[0]:.1f}, {mu[1]:.1f})')


def plot_sample_grid(images, points, save_path, rng):
    n = len(images)
    idxs = rng.choice(n, size=N_SAMPLE_GRID, replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, idx in zip(axes.flat, idxs):
        img = images[idx]
        pts = points[idx]
        if img.ndim == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        for k, (x, y) in enumerate(pts):
            ax.plot(x, y, 'o', color=POINT_COLOURS[k % len(POINT_COLOURS)], markersize=6, markeredgecolor='black', markeredgewidth=0.5)
            ax.annotate(str(k), (x, y), color='white', fontsize=9,
                        xytext=(4, 4), textcoords='offset points',
                        path_effects=None)
        ax.set_title(f'idx={idx}')
        ax.axis('off')
    fig.suptitle('training samples with point indices', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_mean_face(images, points, save_path):
    stacked = stack_if_uniform(images)
    if stacked is None:
        return False
    if stacked.ndim == 4:
        mean_img = stacked.astype(np.float32).mean(axis=0) / 255.0
    else:
        mean_img = stacked.astype(np.float32).mean(axis=0)
        mean_img = mean_img / mean_img.max()
    mean_pts = points.mean(axis=0)
    fig, ax = plt.subplots(figsize=(5, 5))
    if mean_img.ndim == 2:
        ax.imshow(mean_img, cmap='gray')
    else:
        ax.imshow(np.clip(mean_img, 0, 1))
    for k, (x, y) in enumerate(mean_pts):
        ax.plot(x, y, 'o', color=POINT_COLOURS[k % len(POINT_COLOURS)], markersize=8, markeredgecolor='black')
        ax.annotate(str(k), (x, y), color='white', fontsize=10, xytext=(5, 5), textcoords='offset points')
    ax.set_title('mean face + mean landmarks')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return True


def plot_point_spread(points, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for k in range(points.shape[1]):
        ax.scatter(points[:, k, 0], points[:, k, 1],
                   color=POINT_COLOURS[k % len(POINT_COLOURS)],
                   s=4, alpha=0.3, label=f'pt {k}')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')
    ax.set_title('per-point spatial spread across training set')
    ax.legend(loc='best', fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    paths.OUTPUTS.mkdir(parents=True, exist_ok=True)
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    images_train, points_train = load_train()
    images_val, points_val = load_val()
    images_test = load_test()

    print('=== arrays ===')
    describe_array('images_train', images_train)
    describe_array('points_train', points_train)
    describe_array('images_val', images_val)
    describe_array('points_val', points_val)
    describe_array('images_test', images_test)

    print('\n=== points (train) ===')
    points_summary(points_train)

    sizes = image_size_stats(images_train)
    print(f'\n=== image sizes (train) ===')
    print(f'  unique sizes count: {len({tuple(s) for s in sizes})}')
    print(f'  H min/median/max: {sizes[:, 0].min()} / {int(np.median(sizes[:, 0]))} / {sizes[:, 0].max()}')
    print(f'  W min/median/max: {sizes[:, 1].min()} / {int(np.median(sizes[:, 1]))} / {sizes[:, 1].max()}')

    sample_path = paths.FIGURES / 'task2_train_samples.png'
    plot_sample_grid(images_train, points_train, sample_path, rng)
    print(f'wrote {sample_path}')

    spread_path = paths.FIGURES / 'task2_point_spread.png'
    plot_point_spread(points_train, spread_path)
    print(f'wrote {spread_path}')

    mean_path = paths.FIGURES / 'task2_mean_face.png'
    if plot_mean_face(images_train, points_train, mean_path):
        print(f'wrote {mean_path}')
    else:
        print(f'skipped {mean_path} (heterogeneous image sizes)')


if __name__ == '__main__':
    main()
