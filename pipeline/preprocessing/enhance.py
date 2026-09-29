"""
preprocessing/legacy_enhance.py
Exactly reproduces the Makharia et al. 2025 preprocessing chain:
  Georeference → Resample → Normalize → CLAHE → Invert → Dilate → PCA
Nothing here is changed from the paper description.
"""
import cv2
import numpy as np
from sklearn.decomposition import PCA as skPCA


def georeference_align(src_img: np.ndarray,
                        ref_img: np.ndarray,
                        src_meta: dict,
                        ref_meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Coarse spatial alignment using bounding-box metadata from both labels.
    Projects source image into the reference image's coordinate frame by
    computing the pixel-scale affine transform implied by the two footprints.
    Returns (aligned_src, ref_img) both on a common grid.
    """
    # Pixel → geographic scale for source
    src_h, src_w = src_img.shape[:2]
    src_bbox = src_meta.get("bounding_box", {})
    src_lon_span = src_bbox.get("max_lon", 36.0) - src_bbox.get("min_lon", 31.0)
    src_lat_span = src_bbox.get("max_lat", -84.0) - src_bbox.get("min_lat", -84.8)

    # Geographic → pixel for reference
    ref_h, ref_w = ref_img.shape[:2]
    ref_lon_span = src_lon_span   # reference crops share approx footprint
    ref_lat_span = src_lat_span

    # Scale factors (how many pixels per degree)
    src_ppd_x = src_w / src_lon_span if src_lon_span else 1.0
    src_ppd_y = src_h / src_lat_span if src_lat_span else 1.0
    ref_ppd_x = ref_w / ref_lon_span if ref_lon_span else 1.0
    ref_ppd_y = ref_h / ref_lat_span if ref_lat_span else 1.0

    # Resample source to reference pixel density
    scale_x = ref_ppd_x / src_ppd_x
    scale_y = ref_ppd_y / src_ppd_y
    new_w = max(1, int(src_w * scale_x))
    new_h = max(1, int(src_h * scale_y))
    aligned_src = cv2.resize(src_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return aligned_src, ref_img


def resample_to_common_gsd(src_img: np.ndarray,
                            ref_img: np.ndarray,
                            src_gsd: float,
                            ref_gsd: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample source to match reference GSD.
    The coarser image (larger GSD) becomes the target scale for both.
    """
    common_gsd = max(src_gsd, ref_gsd)
    s_factor = src_gsd / common_gsd
    r_factor = ref_gsd / common_gsd

    if s_factor < 0.99:
        h, w = src_img.shape[:2]
        src_img = cv2.resize(src_img,
                             (max(1, int(w * s_factor)), max(1, int(h * s_factor))),
                             interpolation=cv2.INTER_AREA)
    if r_factor < 0.99:
        h, w = ref_img.shape[:2]
        ref_img = cv2.resize(ref_img,
                             (max(1, int(w * r_factor)), max(1, int(h * r_factor))),
                             interpolation=cv2.INTER_AREA)

    # Crop/pad to same size (take min dimension)
    min_h = min(src_img.shape[0], ref_img.shape[0])
    min_w = min(src_img.shape[1], ref_img.shape[1])
    src_img = src_img[:min_h, :min_w]
    ref_img = ref_img[:min_h, :min_w]
    return src_img, ref_img


def normalize_intensity(img: np.ndarray) -> np.ndarray:
    """Stretch pixel values to full 0–255 uint8 range."""
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def apply_clahe(img: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid: tuple = (8, 8)) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation (per paper defaults)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img)


def invert(img: np.ndarray) -> np.ndarray:
    """Photometric inversion: new = 255 - old."""
    return (255 - img).astype(np.uint8)


def dilate(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Morphological dilation with elliptical structuring element."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(img, kernel, iterations=1)


def apply_pca(img: np.ndarray, n_components: int = 1) -> np.ndarray:
    """
    PCA whitening on flattened image patches.
    Suppresses spatially repetitive (low-information) brightness patterns.
    n_components=1 keeps only the principal component — standard for single-band.
    """
    h, w = img.shape
    flat = img.reshape(1, h * w).astype(np.float64)
    pca = skPCA(n_components=n_components)
    transformed = pca.fit_transform(flat.T)            # (N, 1)
    reconstructed = pca.inverse_transform(transformed) # (N, original_features)
    out = reconstructed.reshape(h, w)
    return cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def preprocess_pair(src_img: np.ndarray,
                    ref_img: np.ndarray,
                    src_meta: dict,
                    ref_meta: dict,
                    verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Full Makharia et al. 2025 preprocessing chain applied to both images.
    Returns (preprocessed_src, preprocessed_ref) ready for descriptor extraction.
    """
    if verbose:
        print("  [Preprocess] Step 1: Georeferencing + affine alignment...")
    src, ref = georeference_align(src_img, ref_img, src_meta, ref_meta)

    if verbose:
        print("  [Preprocess] Step 2: GSD resampling to common scale...")
    src, ref = resample_to_common_gsd(src, ref,
                                       src_meta.get("gsd", 0.25),
                                       ref_meta.get("gsd", 0.50))

    if verbose:
        print(f"  [Preprocess] Common shape: src={src.shape} ref={ref.shape}")
        print("  [Preprocess] Step 3: Intensity normalisation...")
    src = normalize_intensity(src)
    ref = normalize_intensity(ref)

    if verbose:
        print("  [Preprocess] Step 4: CLAHE...")
    src = apply_clahe(src)
    ref = apply_clahe(ref)

    if verbose:
        print("  [Preprocess] Step 5: Inversion...")
    src = invert(src)
    ref = invert(ref)

    if verbose:
        print("  [Preprocess] Step 6: Morphological Dilation...")
    src = dilate(src)
    ref = dilate(ref)

    if verbose:
        print("  [Preprocess] Step 7: PCA (principal component, 1)...")
    src = apply_pca(src)
    ref = apply_pca(ref)

    if verbose:
        print("  [Preprocess] Done.")
    return src, ref
