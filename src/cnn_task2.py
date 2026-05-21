# CNN landmark regressor for Task 2.
# 4-block VGG-style backbone -> GAP -> MLP head producing point-wise offsets
# added to a fixed mean shape (residual formulation, easier to train).
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .preprocess_task2 import train_transform, eval_transform, INPUT_SIZE, to_chw


class LandmarkCNN(nn.Module):
    def __init__(self, n_points=5, dropout=0.2, mean_shape_norm=None):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 128)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.delta_head = nn.Linear(128, n_points * 2)
        # Zero-init the offset head so the model starts at the mean shape exactly.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        self.n_points = n_points
        if mean_shape_norm is None:
            mean_shape_norm = torch.full((n_points, 2), 0.5)
        # mean_shape is stored as a non-trainable buffer in [0,1] coordinates.
        self.register_buffer('mean_shape', mean_shape_norm.to(torch.float32))

    @staticmethod
    def _block(in_ch, out_ch):
        # Conv-BN-ReLU x2 then 2x2 max pool — standard VGG block.
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        feat = self.features(x)
        feat = self.gap(feat).flatten(1)
        feat = self.dropout(self.act(self.fc1(feat)))
        delta = self.delta_head(feat).view(-1, self.n_points, 2)
        # Predict offsets from mean shape rather than absolute coords.
        return self.mean_shape.unsqueeze(0) + delta


class TrainDataset(Dataset):
    # Applies fresh random augmentation per sample per epoch.
    # Points are returned in normalised [0,1] coordinates to match the model output.
    def __init__(self, images, points, seed=0):
        self.images = images
        self.points = points
        self.seed = seed

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        rng = np.random.default_rng()
        img, pts = train_transform(self.images[idx], self.points[idx], rng)
        img = to_chw(img.astype(np.float32))
        pts_norm = (pts.astype(np.float32) / INPUT_SIZE)
        return torch.from_numpy(img), torch.from_numpy(pts_norm)


class EvalDataset(Dataset):
    # Deterministic eval-time dataset; supports unlabelled test images too.
    def __init__(self, images, points=None):
        self.images = images
        self.points = points

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        pts = self.points[idx] if self.points is not None else None
        img, pts_resized = eval_transform(self.images[idx], pts)
        img = to_chw(img.astype(np.float32))
        pts_norm = pts_resized.astype(np.float32) / INPUT_SIZE
        return torch.from_numpy(img), torch.from_numpy(pts_norm)
