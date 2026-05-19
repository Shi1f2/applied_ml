import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.linear_model import Ridge

from .preprocess_task2 import eval_transform, INPUT_SIZE


def to_grayscale_uint8(image_unit_rgb):
    gray = rgb2gray(image_unit_rgb)
    return (gray * 255.0).astype(np.uint8)


def preprocess_for_classical(images, points=None):
    grays = []
    new_points = []
    for i, img in enumerate(images):
        pts = points[i] if points is not None else None
        img_unit, pts_resized = eval_transform(img, pts)
        grays.append(to_grayscale_uint8(img_unit))
        if points is not None:
            new_points.append(pts_resized)
    grays = np.stack(grays)
    if points is not None:
        return grays, np.stack(new_points)
    return grays


class HogPatchSDM:
    def __init__(self, patch_size=32, pixels_per_cell=8, cells_per_block=2,
                 orientations=9, alpha=1.0):
        self.patch_size = patch_size
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.orientations = orientations
        self.alpha = alpha
        self.mean_shape = None
        self.regressor = None
        self.feature_dim_per_patch = None

    def _hog_patch(self, image_gray, x, y):
        h, w = image_gray.shape
        half = self.patch_size // 2
        x0 = int(round(x)) - half
        y0 = int(round(y)) - half
        x1 = x0 + self.patch_size
        y1 = y0 + self.patch_size
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(w, x1), min(h, y1)
        patch = np.zeros((self.patch_size, self.patch_size), dtype=image_gray.dtype)
        if x1c > x0c and y1c > y0c:
            patch[y0c - y0:y1c - y0, x0c - x0:x1c - x0] = image_gray[y0c:y1c, x0c:x1c]
        return hog(patch,
                   orientations=self.orientations,
                   pixels_per_cell=(self.pixels_per_cell, self.pixels_per_cell),
                   cells_per_block=(self.cells_per_block, self.cells_per_block),
                   feature_vector=True,
                   channel_axis=None)

    def _features(self, image_gray, init_pts):
        return np.concatenate([self._hog_patch(image_gray, x, y) for x, y in init_pts])

    def _features_batch(self, images_gray, init_pts_per_image):
        return np.stack([self._features(img, pts) for img, pts in zip(images_gray, init_pts_per_image)])

    def fit(self, images_gray, points):
        self.mean_shape = points.mean(axis=0).astype(np.float32)
        init = np.broadcast_to(self.mean_shape, (len(images_gray),) + self.mean_shape.shape).copy()
        X = self._features_batch(images_gray, init)
        self.feature_dim_per_patch = X.shape[1] // points.shape[1]
        Y = (points - init).reshape(len(points), -1)
        self.regressor = Ridge(alpha=self.alpha)
        self.regressor.fit(X, Y)
        return self

    def predict(self, images_gray):
        init = np.broadcast_to(self.mean_shape, (len(images_gray),) + self.mean_shape.shape).copy()
        X = self._features_batch(images_gray, init)
        deltas = self.regressor.predict(X).reshape(-1, self.mean_shape.shape[0], 2)
        return (init + deltas).astype(np.float32)
