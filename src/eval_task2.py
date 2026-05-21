# Task 2 evaluation metrics.
# Primary metric is the inter-ocular-normalised mean Euclidean error (NME),
# the standard for face-alignment benchmarks. CED + AUC + failure rates
# all derive from per-image NME.
import numpy as np


# Eye points used to normalise — see preprocess_task2.HFLIP_SWAP_PAIRS.
EYE_INDICES = (0, 1)


def euclid_dist(pred_pts, gt_pts):
    pred_pts = np.reshape(pred_pts, (-1, 2))
    gt_pts = np.reshape(gt_pts, (-1, 2))
    return np.sqrt(np.sum(np.square(pred_pts - gt_pts), axis=-1))


def inter_ocular_distance(points, eye_indices=EYE_INDICES):
    a, b = eye_indices
    if points.ndim == 2:
        return np.linalg.norm(points[a] - points[b])
    return np.linalg.norm(points[:, a, :] - points[:, b, :], axis=-1)


def per_image_nme(pred, gt, eye_indices=EYE_INDICES):
    # Mean per-point Euclidean error divided by inter-ocular distance.
    # Makes errors comparable across images at different scales.
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    iod = inter_ocular_distance(gt, eye_indices)
    per_pt = np.linalg.norm(pred - gt, axis=-1)
    return per_pt.mean(axis=-1) / iod


def per_point_error(pred, gt):
    return np.linalg.norm(pred - gt, axis=-1)


def failure_rate(nme_values, threshold=0.10):
    nme_values = np.asarray(nme_values)
    return float((nme_values > threshold).mean())


def ced_curve(nme_values, thresholds=None):
    # Cumulative error distribution — fraction of images with NME <= threshold.
    nme_values = np.asarray(nme_values)
    if thresholds is None:
        thresholds = np.linspace(0.0, 0.30, 301)
    fractions = (nme_values[None, :] <= thresholds[:, None]).mean(axis=1)
    return thresholds, fractions


def auc_under_ced(nme_values, max_threshold=0.10):
    thresholds, fractions = ced_curve(nme_values, np.linspace(0.0, max_threshold, 201))
    return float(np.trapezoid(fractions, thresholds) / max_threshold)


def summarise(nme_values, thresholds=(0.05, 0.08, 0.10)):
    nme_values = np.asarray(nme_values)
    out = {
        'mean_nme': float(nme_values.mean()),
        'median_nme': float(np.median(nme_values)),
        'std_nme': float(nme_values.std()),
        'auc@0.10': auc_under_ced(nme_values, 0.10),
    }
    for t in thresholds:
        out[f'failure@{t:.2f}'] = failure_rate(nme_values, t)
    return out
