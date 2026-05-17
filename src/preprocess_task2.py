import numpy as np
import cv2


ORIGINAL_SIZE = 256
INPUT_SIZE = 128
COORD_SCALE = INPUT_SIZE / ORIGINAL_SIZE
HFLIP_SWAP_PAIRS = [(0, 1), (3, 4)]


def to_input_size(image, points):
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    points = points.astype(np.float32) * COORD_SCALE
    return image, points


def predictions_to_original(points):
    return points.astype(np.float32) / COORD_SCALE


def to_unit_range(image):
    return image.astype(np.float32) / 255.0


def to_chw(image):
    return np.transpose(image, (2, 0, 1)).copy()


def affine_matrix(angle_deg, scale, tx, ty, centre):
    M = cv2.getRotationMatrix2D(centre, angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return M


def apply_affine(image, points, M):
    h, w = image.shape[:2]
    image = cv2.warpAffine(image, M, (w, h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    points = (pts_h @ M.T).astype(np.float32)
    return image, points


def hflip(image, points):
    image = np.ascontiguousarray(image[:, ::-1, :])
    w = image.shape[1]
    points = points.copy()
    points[:, 0] = (w - 1) - points[:, 0]
    for a, b in HFLIP_SWAP_PAIRS:
        points[[a, b]] = points[[b, a]]
    return image, points


def random_brightness_contrast(image, rng, brightness, contrast):
    b = 1.0 + rng.uniform(-brightness, brightness)
    c = 1.0 + rng.uniform(-contrast, contrast)
    return np.clip((image - 0.5) * c + 0.5 * b, 0.0, 1.0)


def random_gaussian_noise(image, rng, sigma_max):
    sigma = rng.uniform(0.0, sigma_max)
    noise = rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0)


def train_transform(image, points, rng,
                    rot_range=20.0, scale_range=0.10, trans_range=0.05,
                    hflip_prob=0.5, brightness=0.2, contrast=0.2, noise_sigma=0.03):
    image, points = to_input_size(image, points)
    h, w = image.shape[:2]
    if rng.random() < hflip_prob:
        image, points = hflip(image, points)
    angle = float(rng.uniform(-rot_range, rot_range))
    scale = 1.0 + float(rng.uniform(-scale_range, scale_range))
    tx = float(rng.uniform(-trans_range, trans_range)) * w
    ty = float(rng.uniform(-trans_range, trans_range)) * h
    M = affine_matrix(angle, scale, tx, ty, (w / 2.0, h / 2.0))
    image, points = apply_affine(image, points, M)
    image = to_unit_range(image)
    image = random_brightness_contrast(image, rng, brightness, contrast)
    image = random_gaussian_noise(image, rng, sigma_max=noise_sigma)
    return image, points


def eval_transform(image, points=None):
    if points is None:
        points = np.zeros((5, 2), dtype=np.float32)
    image, points = to_input_size(image, points)
    image = to_unit_range(image)
    return image, points
