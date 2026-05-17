# Task 2 — Face Alignment

## 1. Problem and design

The five 0-indexed landmarks are `0` left eye, `1` right eye, `2` nose, `3` left mouth corner, `4` right mouth corner. The brief warns of "variability and transformations that distort the images", so the system has to provide **robustness to unwanted variability** rather than chase pixel accuracy on a canonical pose. The pipeline (Figure 1) downsamples to 128×128 for tractable training [1], applies a deterministic eval transform shared by validation and test, and adds a stochastic augmentation transform on the training stream. Predictions are upscaled back to the 256×256 frame before output, satisfying the brief's "predicted points at original resolution" rule.

![Figure 1](../figures/task2_pipeline.png)

*Figure 1: Task 2 preprocessing and augmentation pipeline. Training is the only source of variability; eval and test share a deterministic resize → unit-range path.*

Two approaches are compared (§4): **A1** Supervised Descent regression [2] (HOG patches [3], ridge [4]); **A2** a from-scratch CNN with a mean-shape residual head and Huber loss [5]. A2 is used for the test submission.

## 2. Data inspection

Splits are 2,600 / 211 / 554 (train / val / test); images are 256×256×3 `uint8`. Mean per-point coordinates `[(79.9, 104.3), (176.2, 103.1), (128.7, 142.6), (98.6, 177.1), (159.6, 176.2)]` confirmed the index-to-feature mapping by overlay, including nose at index 2. Mean inter-ocular distance (IOD) is 96.3 px. The mean-face image is sharp and per-point clusters are tight on the eyes, looser on the mouth — i.e. faces are roughly aligned and centred, with **rotation/yaw and scale jitter as the dominant within-set variability**. This motivated the augmentation choices in §3.

## 3. Pre-processing and augmentation

Deterministic (train/val/test): bilinear resize 256→128 (points × 0.5), divide by 255. Training adds, in order: horizontal flip `p=0.5` with the index swap `(0↔1, 3↔4)`; a single 2×3 affine matrix [6] combining rotation `±20°`, scale `±10%` and translation `±5%` applied to image (`cv2.warpAffine`, reflection padding) and points (multiply by the same `M`); brightness/contrast jitter `±20%`; Gaussian noise `σ ≤ 0.03`. Predictions are multiplied by `2.0` to return to the 256-frame before evaluation or CSV export.