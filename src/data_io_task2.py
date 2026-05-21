# Task 2 .npz loaders. Test split has no points (held out by the markers).
import numpy as np
import matplotlib.pyplot as plt

from .paths import TASK2_TRAIN, TASK2_VAL, TASK2_TEST


def load_train():
    data = np.load(TASK2_TRAIN, allow_pickle=True)
    return data['images'], data['points']


def load_val():
    data = np.load(TASK2_VAL, allow_pickle=True)
    return data['images'], data['points']


def load_test():
    data = np.load(TASK2_TEST, allow_pickle=True)
    return data['images']


def visualise_pts(img, pts, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img)
    ax.plot(pts[:, 0], pts[:, 1], '+r')
    return ax
