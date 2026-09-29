"""
illumination/compute_normals.py
Stage 1a: Computes per-pixel surface normals from a DEM using finite-difference gradients.

Given a DEM grid Z(x, y), the surface normal at each pixel is:
    N_raw = (-dZ/dx, -dZ/dy, 1)
    N     = N_raw / ||N_raw||

A flat patch → normal (0,0,1) pointing straight up.
A steep crater wall → normal tilts sideways, catching light differently.
"""
import numpy as np


def compute_normals(dem: np.ndarray,
                    pixel_size_x: float,
                    pixel_size_y: float) -> np.ndarray:
    """
    Compute unit surface normals from a DEM.

    Parameters
    ----------
    dem : (H, W) float array of elevation values.
    pixel_size_x, pixel_size_y : ground resolution of pixels in metres.

    Returns
    -------
    (H, W, 3) float array of unit normal vectors (nx, ny, nz).
    """
    dy, dx = np.gradient(dem, pixel_size_y, pixel_size_x)

    # Surface normal components: N = (-dZ/dx, -dZ/dy, 1)
    nx = -dx
    ny = -dy
    nz = np.ones_like(dem)

    # Normalise to unit length
    magnitude = np.sqrt(nx**2 + ny**2 + nz**2)
    magnitude[magnitude == 0] = 1e-6  # avoid division by zero

    nx /= magnitude
    ny /= magnitude
    nz /= magnitude

    return np.stack((nx, ny, nz), axis=-1)
""", "Description": "Extracted from src/rendering/illumination.py — pure normal computation logic"
