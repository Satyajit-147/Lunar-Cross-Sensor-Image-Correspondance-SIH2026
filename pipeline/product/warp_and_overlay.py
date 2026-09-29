"""
product/warp_and_overlay.py
Warp source into reference frame and generate overlay visualisations.
Also saves a CSV match-point table.
"""
import os
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch


def warp_source(src_img: np.ndarray,
                H: np.ndarray,
                ref_shape: tuple) -> np.ndarray:
    """Warp source image into reference frame using homography H."""
    h, w = ref_shape[:2]
    return cv2.warpPerspective(src_img, H, (w, h))


def save_overlay(src_img, ref_img, warped_src, out_path, tc_id, descriptor, phase):
    """Three-panel overlay: source | reference | warped+blend."""
    h, w = ref_img.shape[:2]
    warped_r = cv2.resize(warped_src, (w, h)) if warped_src.shape[:2] != (h, w) else warped_src
    src_r    = cv2.resize(src_img,    (w, h))

    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    axs[0].imshow(src_r, cmap="gray")
    axs[0].set_title(f"OHRC Source\n[{tc_id}]", fontsize=12)
    axs[0].axis("off")

    axs[1].imshow(ref_img, cmap="gray")
    axs[1].set_title(f"LRO NAC Reference\n[{tc_id}]", fontsize=12)
    axs[1].axis("off")

    # False-colour overlay: red=NAC, green=warped OHRC
    blend = np.zeros((h, w, 3), dtype=np.uint8)
    blend[:, :, 0] = ref_img
    blend[:, :, 1] = warped_r
    blend[:, :, 2] = warped_r
    axs[2].imshow(blend)
    axs[2].set_title(f"Overlay (R=NAC, G=OHRC warped)\nPhase {phase} | {descriptor}", fontsize=12)
    axs[2].axis("off")

    plt.suptitle(f"Lunar Registration — {tc_id} | {descriptor} | Phase {phase}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_match_visualization(src_img, ref_img, kp_src, kp_ref, inlier_matches, out_path):
    """Side-by-side image with inlier match lines drawn via ConnectionPatch."""
    h1, w1 = src_img.shape[:2]
    h2, w2 = ref_img.shape[:2]
    # Resize ref to same height for display
    scale = h1 / h2
    ref_disp = cv2.resize(ref_img, (max(1, int(w2 * scale)), h1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(src_img, cmap="gray")
    ax1.set_title("OHRC Source", fontsize=14)
    ax1.axis("off")
    ax2.imshow(ref_disp, cmap="gray")
    ax2.set_title("LRO NAC Reference", fontsize=14)
    ax2.axis("off")

    colors = plt.cm.plasma(np.linspace(0, 1, max(1, len(inlier_matches))))
    for i, m in enumerate(inlier_matches[:80]):   # cap at 80 for readability
        pt_src = kp_src[m.queryIdx].pt
        pt_ref = kp_ref[m.trainIdx].pt
        pt_ref_scaled = (pt_ref[0] * scale, pt_ref[1] * scale)
        c = colors[i]
        ax1.plot(pt_src[0], pt_src[1], "o", color=c, markersize=4)
        ax2.plot(pt_ref_scaled[0], pt_ref_scaled[1], "o", color=c, markersize=4)
        con = ConnectionPatch(xyA=pt_ref_scaled, xyB=pt_src,
                              coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1,
                              color=c, linewidth=0.8, alpha=0.7)
        ax2.add_artist(con)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_match_csv(kp_src, kp_ref, inlier_matches, H, out_path):
    """Save inlier match table with residuals to CSV."""
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Src_X", "Src_Y", "Ref_X", "Ref_Y",
                         "Proj_X", "Proj_Y", "Residual_X", "Residual_Y", "Residual_px"])
        for m in inlier_matches:
            sx, sy = kp_src[m.queryIdx].pt
            rx, ry = kp_ref[m.trainIdx].pt
            pt = np.array([[[sx, sy]]], dtype=np.float32)
            proj = cv2.perspectiveTransform(pt, H)[0][0]
            res_x = proj[0] - rx
            res_y = proj[1] - ry
            res   = float(np.sqrt(res_x ** 2 + res_y ** 2))
            writer.writerow([f"{sx:.2f}", f"{sy:.2f}", f"{rx:.2f}", f"{ry:.2f}",
                             f"{proj[0]:.2f}", f"{proj[1]:.2f}",
                             f"{res_x:.3f}", f"{res_y:.3f}", f"{res:.3f}"])


def generate_products(src_img, ref_img, kp_src, kp_ref, result, tc_id, descriptor, out_dir):
    """Master product generator — warp, overlay, match vis, CSV."""
    os.makedirs(out_dir, exist_ok=True)

    H             = result.get("H")
    inlier_matches = result.get("inlier_matches", [])
    phase         = result.get("phase", "?")

    if H is not None:
        warped = warp_source(src_img, H, ref_img.shape)
        ov_path = os.path.join(out_dir, "overlay.png")
        save_overlay(src_img, ref_img, warped, ov_path, tc_id, descriptor, phase)
        cv2.imwrite(os.path.join(out_dir, "warped_src.tif"), warped)
    else:
        print(f"    [Products] No homography — skipping overlay/warp.")

    if inlier_matches and H is not None:
        mv_path  = os.path.join(out_dir, "match_lines.png")
        csv_path = os.path.join(out_dir, "match_table.csv")
        save_match_visualization(src_img, ref_img, kp_src, kp_ref, inlier_matches, mv_path)
        save_match_csv(kp_src, kp_ref, inlier_matches, H, csv_path)
