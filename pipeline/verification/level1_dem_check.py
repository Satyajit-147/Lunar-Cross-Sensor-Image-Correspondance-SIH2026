"""
verification/level1_dem_check.py
Level-1 Physics Gate: Cross-verification of the OHRC image against the LOLA DEM.

This is the "fallback DEM check" — before matching OHRC against any NAC image,
we verify that the OHRC's geometry is physically consistent with the known
3D terrain by:
  1. Loading the OHRC source image
  2. Loading the pre-computed Lambertian synthetic render (from LOLA DEM)
  3. Running Phase Congruency on both
  4. SIFT keypoints + kNN ratio matching
  5. MAGSAC++ inlier count

If inlier count >= INLIER_THRESHOLD → OHRC is geometrically consistent.
                                       Proceed to Level-2 (OHRC × NAC matching).
If below threshold                  → Reject. OHRC frame failed terrain sanity check.
"""
import os
import sys
import cv2
import numpy as np
import tifffile

# Project root is two levels up from this file
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import phasepack.phasecongmono as phasepack_pc

# Paths to pre-computed Approach 1 products
SYNTH_RENDER_PATH = os.path.join(
    PROJECT_ROOT, "data", "synthetic",
    "ch2_ohr_ncp_20260102T2017444613_d_img_d18_synthetic.tiff"
)
SOURCE_PYRAMID_L0 = os.path.join(
    PROJECT_ROOT, "data", "ohrc",
    "ohrc_source.tif"
)

# Also check legacy paths as fallback
SYNTH_RENDER_LEGACY = os.path.join(PROJECT_ROOT, "_archive", "src", "data", "synthetic",
    "ch2_ohr_ncp_20260102T2017444613_d_img_d18_synthetic.tiff")

# Gate threshold
INLIER_THRESHOLD = 3


def _compute_phase_congruency(img: np.ndarray) -> np.ndarray:
    """Phase Congruency map — identical to structure/phase_congruency.py."""
    result = phasepack_pc(img, nscale=4, minWaveLength=3, mult=2.1)
    PC = result[0]
    return cv2.normalize(PC, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _knn_ratio_match(des_src, des_ref, ratio=0.90):
    """kNN (k=2) + Lowe's ratio test."""
    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_src, des_ref, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]
    return good


def _find_file(primary, fallback=None):
    """Return first existing path, or raise FileNotFoundError."""
    if os.path.exists(primary):
        return primary
    if fallback and os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"Required file not found: {primary}")


def level1_gate(verbose: bool = True) -> dict:
    """
    Level-1 cross-verification gate.

    Returns
    -------
    dict with keys: pass, inlier_count, threshold, n_kp_src, n_kp_synth, n_raw_matches.
    """
    if verbose:
        print("  [Level-1] Cross-verification via Approach 1 (OHRC ↔ LOLA DEM render)")

    # Load images
    source_path = _find_file(SOURCE_PYRAMID_L0)
    synth_path  = _find_file(SYNTH_RENDER_PATH, SYNTH_RENDER_LEGACY)

    src_img   = tifffile.imread(source_path)
    synth_img = tifffile.imread(synth_path)

    # Normalise to uint8
    def to_uint8(img):
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    src_img   = to_uint8(src_img)
    synth_img = to_uint8(synth_img)

    # Match shapes and downsample to avoid OOM on 6000x6000 images
    MAX_DIM = 1500
    h, w = src_img.shape
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        src_img = cv2.resize(src_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if synth_img.shape != src_img.shape:
        synth_img = cv2.resize(synth_img, (src_img.shape[1], src_img.shape[0]),
                               interpolation=cv2.INTER_AREA)

    if verbose:
        print(f"  [Level-1]   Processing shape:    {src_img.shape}")

    # Phase Congruency
    if verbose:
        print("  [Level-1]   Computing Phase Congruency on OHRC source...")
    src_pc = _compute_phase_congruency(src_img)

    if verbose:
        print("  [Level-1]   Computing Phase Congruency on LOLA synthetic render...")
    synth_pc = _compute_phase_congruency(synth_img)

    # SIFT detection
    sift = cv2.SIFT_create()
    kp_src,   des_src   = sift.detectAndCompute(src_pc,   None)
    kp_synth, des_synth = sift.detectAndCompute(synth_pc, None)

    if verbose:
        print(f"  [Level-1]   SIFT keypoints: {len(kp_src)} (OHRC PC)  "
              f"{len(kp_synth)} (Synthetic PC)")

    if des_src is None or des_synth is None or len(kp_src) < 4 or len(kp_synth) < 4:
        if verbose:
            print(f"  [Level-1]   ✗ Too few keypoints — gate FAIL")
        return {"pass": False, "inlier_count": 0, "threshold": INLIER_THRESHOLD,
                "n_kp_src": len(kp_src), "n_kp_synth": len(kp_synth), "n_raw_matches": 0}

    # kNN + ratio test
    good = _knn_ratio_match(des_src, des_synth, ratio=0.90)

    if verbose:
        print(f"  [Level-1]   Raw matches (ratio test): {len(good)}")

    if len(good) < 4:
        if verbose:
            print(f"  [Level-1]   ✗ Too few matches — gate FAIL")
        return {"pass": False, "inlier_count": 0, "threshold": INLIER_THRESHOLD,
                "n_kp_src": len(kp_src), "n_kp_synth": len(kp_synth),
                "n_raw_matches": len(good)}

    # MAGSAC++
    src_pts   = np.float32([kp_src[m.queryIdx].pt   for m in good]).reshape(-1, 1, 2)
    synth_pts = np.float32([kp_synth[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    try:
        _, mask = cv2.findHomography(src_pts, synth_pts, cv2.USAC_MAGSAC, 5.0,
                                     maxIters=2000, confidence=0.995)
        inliers = int(mask.sum()) if mask is not None else 0
    except Exception:
        inliers = 0

    passed = inliers >= INLIER_THRESHOLD
    tick   = "✓" if passed else "✗"

    if verbose:
        print(f"  [Level-1]   MAGSAC++ inliers: {inliers}  "
              f"(threshold ≥ {INLIER_THRESHOLD})  [{tick}]")
        if passed:
            print("  [Level-1]   → OHRC ↔ LOLA DEM confirmed. Proceeding to Level-2.")
        else:
            print("  [Level-1]   → OHRC frame failed DEM sanity check. Skipping Level-2.")

    return {
        "pass":          passed,
        "inlier_count":  inliers,
        "threshold":     INLIER_THRESHOLD,
        "n_kp_src":      len(kp_src),
        "n_kp_synth":    len(kp_synth),
        "n_raw_matches": len(good),
    }
