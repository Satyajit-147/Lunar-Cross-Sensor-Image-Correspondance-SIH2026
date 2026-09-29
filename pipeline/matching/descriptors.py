"""
matching/descriptors.py
Stage 3: Feature extraction using SIFT and RIFT2.

SIFT  — Scale & rotation invariant; standard baseline.
RIFT2 — Phase-congruency-based, illumination-invariant descriptor.
        Uses Gabor-filter Maximum Index Maps on Phase Congruency maps
        with log-polar histogram descriptors. Immune to shadow reversals.

Each extractor returns (keypoints, descriptors) compatible with cv2.BFMatcher.
"""
import cv2
import numpy as np
import phasepack.phasecongmono as phasepack_pc


# ---------------------------------------------------------------------------
# 1. SIFT
# ---------------------------------------------------------------------------
def extract_sift(img: np.ndarray) -> tuple:
    """Standard SIFT — scale & rotation invariant."""
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.03, edgeThreshold=10)
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


# ---------------------------------------------------------------------------
# 2. RIFT2 (Phase-congruency based, illumination-invariant)
# ---------------------------------------------------------------------------
def _compute_phase_congruency(img: np.ndarray) -> np.ndarray:
    """Compute Phase Congruency map (illumination-invariant structural feature)."""
    result = phasepack_pc(img, nscale=4, minWaveLength=3, mult=2.1)
    PC = result[0]
    return cv2.normalize(PC, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _maximum_index_map(img: np.ndarray, n_orientations: int = 6) -> np.ndarray:
    """
    Build a Maximum Index Map (MIM): for each pixel, store which orientation
    filter produced the strongest response. This is the core of RIFT2's descriptor.
    """
    h, w = img.shape
    mim = np.zeros((h, w), dtype=np.uint8)
    max_resp = np.zeros((h, w), dtype=np.float32)

    for i in range(n_orientations):
        angle = i * 180.0 / n_orientations
        kernel = cv2.getGaborKernel((15, 15), 3.0, np.deg2rad(angle),
                                    2 * np.pi / 4, 0.5, 0, ktype=cv2.CV_32F)
        resp = np.abs(cv2.filter2D(img.astype(np.float32), cv2.CV_32F, kernel))
        mask = resp > max_resp
        mim[mask] = i
        max_resp[mask] = resp[mask]

    return mim


def extract_rift2(img: np.ndarray,
                   n_orientations: int = 6,
                   log_polar_bins: int = 6) -> tuple:
    """
    Simplified RIFT2 (Python approximation):
    1. Compute Phase Congruency map
    2. Build Maximum Index Map from oriented Gabor filters
    3. Detect keypoints on PC map (FAST corners for speed)
    4. Build log-polar histograms of orientation indices around each keypoint
    Returns float32 descriptors compatible with BFMatcher (NORM_L2).
    """
    pc_map = _compute_phase_congruency(img)
    mim    = _maximum_index_map(pc_map, n_orientations)

    # Keypoint detection on phase congruency map
    fast = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=True)
    kp = fast.detect(pc_map, None)
    if not kp:
        return [], None

    h, w = img.shape
    patch = 32          # half-size of descriptor region
    desc_dim = log_polar_bins * n_orientations

    descriptors = []
    valid_kp = []

    for k in kp:
        x, y = int(k.pt[0]), int(k.pt[1])
        if x < patch or y < patch or x + patch >= w or y + patch >= h:
            continue

        # Crop MIM patch around keypoint
        region = mim[y - patch:y + patch, x - patch:x + patch].astype(np.float32)

        # Log-polar sampling: accumulate orientation histograms per ring
        hist = np.zeros((log_polar_bins, n_orientations), dtype=np.float32)
        max_r = patch * np.sqrt(2)
        Y, X = np.mgrid[-patch:patch, -patch:patch]
        R = np.sqrt(X ** 2 + Y ** 2) + 1e-6
        log_r = np.log(R)
        log_min = np.log(1.0)
        log_max = np.log(max_r)
        bin_idx = np.clip(
            ((log_r - log_min) / (log_max - log_min + 1e-9) * log_polar_bins).astype(int),
            0, log_polar_bins - 1
        )
        for b in range(log_polar_bins):
            mask = (bin_idx == b)
            for o in range(n_orientations):
                hist[b, o] = np.sum((region[mask] == o).astype(np.float32))

        desc = hist.ravel()
        norm = np.linalg.norm(desc) + 1e-9
        descriptors.append(desc / norm)
        valid_kp.append(k)

    if not descriptors:
        return [], None

    return valid_kp, np.array(descriptors, dtype=np.float32)


# ---------------------------------------------------------------------------
# Registry (SIFT + RIFT2 only)
# ---------------------------------------------------------------------------
DESCRIPTORS = {
    "sift":  extract_sift,
    "rift2": extract_rift2,
}


def extract(name: str, img: np.ndarray) -> tuple:
    """Unified entry point: extract(descriptor_name, preprocessed_image)."""
    fn = DESCRIPTORS.get(name.lower())
    if fn is None:
        raise ValueError(f"Unknown descriptor: {name}. Choose from {list(DESCRIPTORS)}")
    return fn(img)
