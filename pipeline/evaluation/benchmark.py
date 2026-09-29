"""
evaluation/benchmark.py
Runs both Phase 1 (RANSAC) and Phase 2 (Level-1 gate + MAGSAC++) across
all 5 test cases and all 4 descriptors. Emits results/benchmark_table.csv
and prints the Section 6 comparison table from the spec.
"""
import os
import sys
import json
import time
import csv
import traceback
import numpy as np
import cv2
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow imports from lunar_registration/ root
PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(PIPELINE_ROOT)
sys.path.insert(0, PIPELINE_ROOT)

from preprocessing.enhance             import preprocess_pair
from matching.descriptors              import extract, DESCRIPTORS
from matching.ransac_baseline          import match_ransac
from matching.magsac_ours              import match_magsac
from verification.level1_dem_check     import level1_gate, INLIER_THRESHOLD
from product.warp_and_overlay          import generate_products

BINARY_DESCRIPTORS = {"akaze"}   # keep uint8, use NORM_HAMMING

RESULTS_DIR = os.path.join(PROJECT_ROOT, "final_run")


def load_image(path):
    img = tifffile.imread(path)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def load_meta(path):
    with open(path) as f:
        return json.load(f)


def fmt(v, decimals=2):
    if v == float("inf") or v != v:
        return "∞"
    return f"{v:.{decimals}f}"


def run_all(tc_ids=None, descriptors=None, phases=None, config_path=None):
    """
    tc_ids:      list like ["TC-00","TC-01"] or None (all)
    descriptors: list like ["sift","akaze"] or None (all)
    phases:      list like [1,2] or None (both)
    """
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "test_cases.json")
    with open(config_path) as f:
        config = json.load(f)

    all_tcs  = [tc["id"] for tc in config["test_cases"]]
    all_desc = list(DESCRIPTORS.keys())
    all_ph   = [1, 2]

    tcs_to_run  = tc_ids      if tc_ids      else all_tcs
    descs_to_run = descriptors if descriptors else all_desc
    phases_to_run = phases     if phases      else all_ph

    # Load source once
    src_path = os.path.join(PROJECT_ROOT, config["source"]["image"])
    src_meta_path = os.path.join(PROJECT_ROOT, config["source"]["meta"])
    src_img_raw   = load_image(src_path)
    src_meta      = load_meta(src_meta_path)

    # Load DEM once
    dem_path = os.path.join(PROJECT_ROOT, config["dem"])
    dem      = tifffile.imread(dem_path).astype(np.float32)

    rows = []   # accumulated result rows

    for tc_cfg in config["test_cases"]:
        tc_id = tc_cfg["id"]
        if tc_id not in tcs_to_run:
            continue

        ref_path      = os.path.join(PROJECT_ROOT, tc_cfg["image"])
        ref_meta_path = os.path.join(PROJECT_ROOT, tc_cfg["meta"])
        ref_img_raw   = load_image(ref_path)
        ref_meta      = load_meta(ref_meta_path)

        print(f"\n{'='*70}")
        print(f"TEST CASE: {tc_id}  ({ref_meta.get('label', '')})")
        print(f"  Expected Phase1={ref_meta.get('expected_phase1')}  "
              f"Phase2={ref_meta.get('expected_phase2')}")

        # ---- Phase 2, Level 1 DEM gate (run once — same OHRC for all TCs) -------
        l1_result = None
        if 2 in phases_to_run:
            print("\n  --- Level-1 Cross-Verification (Approach 1: OHRC ↔ LOLA DEM render) ---")
            l1_result = level1_gate(verbose=True)

        # ---- Preprocessing (shared for both phases) ----------------------
        print("\n  --- Preprocessing (Makharia et al. 2025 chain) ---")
        src_pp, ref_pp = preprocess_pair(src_img_raw, ref_img_raw,
                                          src_meta, ref_meta, verbose=True)

        for desc_name in descs_to_run:
            print(f"\n  --- Descriptor: {desc_name.upper()} ---")
            try:
                kp_src, des_src = extract(desc_name, src_pp)
                kp_ref, des_ref = extract(desc_name, ref_pp)
                print(f"    Keypoints: src={len(kp_src)}  ref={len(kp_ref)}")
            except Exception as e:
                print(f"    [ERROR] Descriptor extraction failed: {e}")
                traceback.print_exc()
                continue

            is_bin = desc_name.lower() in BINARY_DESCRIPTORS

            for phase in phases_to_run:
                out_dir = os.path.join(RESULTS_DIR, tc_id, desc_name, f"phase{phase}")
                os.makedirs(out_dir, exist_ok=True)

                try:
                    if phase == 1:
                        result = match_ransac(kp_src, des_src, kp_ref, des_ref,
                                              is_binary=is_bin)
                    else:
                        if l1_result and not l1_result["pass"]:
                            # Level-1 gated — skip matching
                            result = {
                                "phase": 2, "method": "MAGSAC++",
                                "n_raw_matches": 0, "n_inliers": 0,
                                "rmse_x": float("inf"), "rmse_y": float("inf"),
                                "H": None, "inlier_matches": [],
                                "exec_time_s": 0.0,
                                "status": "GATED_BY_LEVEL1",
                            }
                        else:
                            result = match_magsac(kp_src, des_src, kp_ref, des_ref,
                                                  is_binary=is_bin)
                except Exception as e:
                    print(f"    [ERROR] Phase {phase} matching failed: {e}")
                    traceback.print_exc()
                    continue

                print(f"    Phase {phase} | {result['method']:10s} | "
                      f"raw={result['n_raw_matches']:5d}  "
                      f"inliers={result['n_inliers']:4d}  "
                      f"RMSE=({fmt(result['rmse_x'])}, {fmt(result['rmse_y'])}) px  "
                      f"t={result['exec_time_s']:.2f}s  "
                      f"→ {result['status']}")

                # Generate products
                try:
                    generate_products(src_pp, ref_pp, kp_src, kp_ref,
                                      result, tc_id, desc_name, out_dir)
                except Exception as e:
                    print(f"    [WARN] Product generation failed: {e}")

                row = {
                    "tc_id":          tc_id,
                    "label":          ref_meta.get("label", ""),
                    "descriptor":     desc_name,
                    "phase":          phase,
                    "method":         result["method"],
                    "n_kp_src":       len(kp_src),
                    "n_kp_ref":       len(kp_ref),
                    "n_raw_matches":  result["n_raw_matches"],
                    "n_inliers":      result["n_inliers"],
                    "rmse_x":         result["rmse_x"],
                    "rmse_y":         result["rmse_y"],
                    "exec_time_s":    result["exec_time_s"],
                    "status":         result["status"],
                    "expected_phase1": ref_meta.get("expected_phase1", ""),
                    "expected_phase2": ref_meta.get("expected_phase2", ""),
                    "level1_inliers":   l1_result["inlier_count"] if l1_result else None,
                    "level1_threshold": l1_result["threshold"]    if l1_result else None,
                    "level1_pass":      l1_result["pass"]         if l1_result else None,
                }
                rows.append(row)

    # ---- Write CSV -------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "benchmark_table.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved benchmark CSV → {csv_path}")

    # ---- Print comparison table (spec Section 6 format) ------------------
    _print_comparison_table(rows)
    _save_summary_chart(rows)
    return rows


def _print_comparison_table(rows):
    if not rows:
        return
    print("\n" + "=" * 120)
    print("SECTION 6 COMPARISON TABLE — Phase 1 (RANSAC) vs Phase 2 (MAGSAC++)")
    print("=" * 120)
    # Level-1 result is the same for all TCs (same OHRC frame)
    l1_inliers = next((r["level1_inliers"] for r in rows if r["level1_inliers"] is not None), None)
    l1_pass    = next((r["level1_pass"]    for r in rows if r["level1_pass"]    is not None), None)
    l1_thresh  = next((r["level1_threshold"] for r in rows if r["level1_threshold"] is not None), None)
    l1_str = (f"L1 Gate: OHRC↔DEM inliers={l1_inliers} ≥ {l1_thresh}? "
              f"{'✓ PASS' if l1_pass else '✗ FAIL'}")
    print(f"\n  {l1_str}")

    header = (f"{'TC':8s} {'Descriptor':8s} | {'Phase':6s} {'Method':12s} | "
              f"{'L1Gate':8s} | "
              f"{'RawM':6s} {'Inlrs':6s} {'RMSE_X':7s} {'RMSE_Y':7s} {'Time':6s} | Status")
    print(header)
    print("-" * 110)

    last_tc = ""
    for r in rows:
        sep = "  " if r["tc_id"] == last_tc else "\n"
        last_tc = r["tc_id"]
        l1_col = (f"{l1_inliers}inl" if r["phase"] == 2 and l1_inliers is not None
                  else "  —  ")
        print(
            f"{sep}"
            f"{r['tc_id']:8s} {r['descriptor'].upper():8s} | "
            f"P{r['phase']}    {r['method']:12s} | "
            f"{l1_col:8s} | "
            f"{r['n_raw_matches']:6d} {r['n_inliers']:6d} "
            f"{fmt(r['rmse_x']):7s} {fmt(r['rmse_y']):7s} "
            f"{r['exec_time_s']:6.2f}s | "
            f"{r['status']}"
        )
    print("=" * 120)

    # Compute Lunar Image Registration %
    phase2_rows = [r for r in rows if r["phase"] == 2]
    if phase2_rows:
        total_p2 = len(phase2_rows)
        passed_p2 = sum(1 for r in phase2_rows if r["status"] == "PASS")
        registration_pct = (passed_p2 / total_p2) * 100
        print(f"\n[FINAL METRIC] Lunar Image Registration Success (Phase 2): {registration_pct:.1f}% ({passed_p2}/{total_p2} configurations)")
        print("=" * 120)


def _save_summary_chart(rows):
    """Bar chart comparing RANSAC vs MAGSAC++ inlier counts per TC×descriptor."""
    if not rows:
        return
    import pandas as pd
    try:
        df = pd.DataFrame(rows)
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
    metrics = [("n_inliers", "Inlier Count"),
               ("rmse_x",    "RMSE_X (px)")]

    for ax, (metric, ylabel) in zip(axes, metrics):
        tcs   = df["tc_id"].unique()
        descs = df["descriptor"].unique()
        x = np.arange(len(tcs))
        width = 0.4 / len(descs)
        colors_p1 = plt.cm.Blues(np.linspace(0.4, 0.9, len(descs)))
        colors_p2 = plt.cm.Oranges(np.linspace(0.4, 0.9, len(descs)))

        for j, desc in enumerate(descs):
            p1_vals = []
            p2_vals = []
            for tc in tcs:
                sub = df[(df["tc_id"] == tc) & (df["descriptor"] == desc)]
                v1 = sub[sub["phase"] == 1][metric].values
                v2 = sub[sub["phase"] == 2][metric].values
                p1_vals.append(float(v1[0]) if len(v1) and v1[0] != float("inf") else 0)
                p2_vals.append(float(v2[0]) if len(v2) and v2[0] != float("inf") else 0)
            offset = (j - len(descs) / 2) * width + width / 2
            ax.bar(x + offset - width / 2, p1_vals, width,
                   label=f"{desc.upper()} P1", color=colors_p1[j], alpha=0.85)
            ax.bar(x + offset + width / 2, p2_vals, width,
                   label=f"{desc.upper()} P2", color=colors_p2[j], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(tcs, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel} — Phase 1 vs Phase 2", fontsize=12)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Benchmark: RANSAC (Phase 1) vs MAGSAC++ (Phase 2)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "benchmark_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved summary chart → {chart_path}")
