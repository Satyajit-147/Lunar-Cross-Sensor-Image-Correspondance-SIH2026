"""
matching/magsac_ours.py
Phase 2 (Ours), Level 2: identical kNN + Lowe's ratio, but cv2.USAC_MAGSAC
replaces cv2.RANSAC as the outlier rejection method.
MAGSAC++ marginalizes over noise scale — no fixed inlier threshold.
"""
import time
import cv2
import numpy as np
from .ransac_baseline import _knn_ratio_match, _compute_rmse

LOWE_RATIO    = 0.75
MAGSAC_THRESH = 5.0    # Upper bound on reprojection error (soft, unlike RANSAC)


def match_magsac(kp_src, des_src,
                 kp_ref, des_ref,
                 is_binary: bool = False) -> dict:
    """
    Phase 2, Level 2:
    1. kNN (k=2) + Lowe's ratio — IDENTICAL to RANSAC baseline
    2. cv2.USAC_MAGSAC — adaptive noise-scale estimation replaces fixed threshold
    """
    t0 = time.perf_counter()

    if des_src is None or des_ref is None or len(kp_src) < 4 or len(kp_ref) < 4:
        return {
            "phase": 2,
            "method": "MAGSAC++",
            "n_raw_matches": 0,
            "n_inliers": 0,
            "rmse_x": float("inf"),
            "rmse_y": float("inf"),
            "H": None,
            "inlier_matches": [],
            "exec_time_s": 0.0,
            "status": "FAIL_NO_KP",
        }

    good = _knn_ratio_match(des_src, des_ref, is_binary=is_binary)
    n_raw = len(good)

    if n_raw < 4:
        return {
            "phase": 2,
            "method": "MAGSAC++",
            "n_raw_matches": n_raw,
            "n_inliers": 0,
            "rmse_x": float("inf"),
            "rmse_y": float("inf"),
            "H": None,
            "inlier_matches": [],
            "exec_time_s": time.perf_counter() - t0,
            "status": "FAIL_FEW_MATCHES",
        }

    src_pts = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ref_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # The only change from Phase 1: method=cv2.USAC_MAGSAC
    H, mask = cv2.findHomography(
        src_pts, ref_pts,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=MAGSAC_THRESH,
        maxIters=2000,
        confidence=0.995,
    )

    if H is None or mask is None:
        return {
            "phase": 2,
            "method": "MAGSAC++",
            "n_raw_matches": n_raw,
            "n_inliers": 0,
            "rmse_x": float("inf"),
            "rmse_y": float("inf"),
            "H": None,
            "inlier_matches": [],
            "exec_time_s": time.perf_counter() - t0,
            "status": "FAIL_MAGSAC",
        }

    inlier_mask    = mask.ravel().tolist()
    inlier_matches = [m for m, ok in zip(good, inlier_mask) if ok]
    n_inliers      = len(inlier_matches)

    rmse_x, rmse_y = _compute_rmse(kp_src, kp_ref, inlier_matches, H)
    exec_time = time.perf_counter() - t0

    status = "PASS" if n_inliers >= 5 else ("MARGINAL" if n_inliers >= 3 else "FAIL")

    return {
        "phase": 2,
        "method": "MAGSAC++",
        "n_raw_matches": n_raw,
        "n_inliers": n_inliers,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "H": H,
        "inlier_matches": inlier_matches,
        "exec_time_s": exec_time,
        "status": status,
    }
