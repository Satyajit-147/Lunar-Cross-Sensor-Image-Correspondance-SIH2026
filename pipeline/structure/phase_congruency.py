"""
structure/phase_congruency.py
Stage 2: Converts raw brightness images into Phase Congruency maps.

Phase Congruency measures structural discontinuities (edges, ridges, valleys)
independent of contrast sign or magnitude. A crater rim produces a strong PC
response whether the sunlit side is on the left or the right — this is what
defeats the relief-inversion problem.

Uses the monogenic signal formulation via the `phasepack` library.
"""
import cv2
import numpy as np
import phasepack.phasecongmono as phasepack_pc


def compute_phase_congruency(img: np.ndarray,
                              nscale: int = 4,
                              min_wavelength: int = 3,
                              mult: float = 2.1) -> np.ndarray:
    """
    Compute the Phase Congruency map for a grayscale image.

    Parameters
    ----------
    img : (H, W) uint8 grayscale image.
    nscale : number of wavelet scales.
    min_wavelength : wavelength of smallest scale filter.
    mult : scaling factor between successive filters.

    Returns
    -------
    (H, W) uint8 Phase Congruency map normalised to [0, 255].
    """
    result = phasepack_pc(img, nscale=nscale, minWaveLength=min_wavelength, mult=mult)
    PC = result[0]
    return cv2.normalize(PC, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
"""
