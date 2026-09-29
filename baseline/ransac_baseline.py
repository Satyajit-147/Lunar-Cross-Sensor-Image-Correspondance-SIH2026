"""
matching/ransac_baseline.py
Phase 1 (Benchmark): kNN matching + Lowe's ratio test + cv2.RANSAC.
Exactly reproduces the geometric verification from Makharia et al. 2025.
"""
import time
import cv2
import numpy as np


LOWE_RATIO = 0.75      # As used in the paper
RANSAC_THRESH = 5.0    # Reprojection threshold in pixels


def _knn_ratio_match(des_src: np.ndarray,
                      des_ref: np.ndarray,
                      is_binary: bool = False) -> list:
    """kNN (k=2) matching + Lowe's ratio test."""
    if is_binary:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    matches_knn = matcher.knnMatch(des_src, des_ref, k=2)
    good = []
    for pair in matches_knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)
    return good


def _compute_rmse(kp_src, kp_ref, matches, H):
    """RMSE_X and RMSE_Y between transformed source pts and reference pts."""
    if H is None or len(matches) < 4:
        return float("inf"), float("inf")

    src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    ref_pts = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(src_pts, H)
    err = projected.squeeze() - ref_pts.squeeze()

    rmse_x = np.sqrt(np.mean(err[:, 0] ** 2))
    rmse_y = np.sqrt(np.mean(err[:, 1] ** 2))
    return float(rmse_x), float(rmse_y)


def match_ransac(kp_src, des_src,
                 kp_ref, des_ref,
                 is_binary: bool = False) -> dict:
    """
    Full Phase 1 matching:
    1. kNN matching + Lowe's ratio
    2. cv2.RANSAC homography estimation (fixed 5 px threshold)
    Returns a results dict.
    """
    t0 = time.perf_counter()

    if des_src is None or des_ref is None or len(kp_src) < 4 or len(kp_ref) < 4:
        return {
            "phase": 1,
            "method": "RANSAC",
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
            "phase": 1,
            "method": "RANSAC",
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

    H, mask = cv2.findHomography(
        src_pts, ref_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESH,
        maxIters=2000,
        confidence=0.995,
    )

    if H is None or mask is None:
        return {
            "phase": 1,
            "method": "RANSAC",
            "n_raw_matches": n_raw,
            "n_inliers": 0,
            "rmse_x": float("inf"),
            "rmse_y": float("inf"),
            "H": None,
            "inlier_matches": [],
            "exec_time_s": time.perf_counter() - t0,
            "status": "FAIL_RANSAC",
        }

    inlier_mask = mask.ravel().tolist()
    inlier_matches = [m for m, ok in zip(good, inlier_mask) if ok]
    n_inliers = len(inlier_matches)

    rmse_x, rmse_y = _compute_rmse(kp_src, kp_ref, inlier_matches, H)
    exec_time = time.perf_counter() - t0

    status = "PASS" if n_inliers >= 5 else ("MARGINAL" if n_inliers >= 3 else "FAIL")

    return {
        "phase": 1,
        "method": "RANSAC",
        "n_raw_matches": n_raw,
        "n_inliers": n_inliers,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "H": H,
        "inlier_matches": inlier_matches,
        "exec_time_s": exec_time,
        "status": status,
    }
