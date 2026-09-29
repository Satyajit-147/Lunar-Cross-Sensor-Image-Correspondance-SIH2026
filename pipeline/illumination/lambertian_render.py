"""
illumination/lambertian_render.py
Stage 1b: Renders a synthetic reflectance image using the Lambertian cosine law.

Given per-pixel surface normals N(x,y) and a sun direction vector L:
    I(x, y) = albedo * max(0,  N(x,y) · L )

The sun direction is derived from the OHRC's own metadata:
    L_x = cos(elevation) * sin(azimuth)   (East)
    L_y = cos(elevation) * cos(azimuth)   (North)
    L_z = sin(elevation)                  (Up)

This produces facet-orientation shading but does NOT model cast shadows.
"""
import numpy as np


def sun_direction_vector(sun_azimuth_deg: float,
                         sun_elevation_deg: float) -> np.ndarray:
    """
    Convert sun azimuth/elevation (degrees) to a 3D unit vector in ENU frame.

    Parameters
    ----------
    sun_azimuth_deg : compass bearing of the sun (0=N, 90=E, 180=S, 270=W).
    sun_elevation_deg : angle above the horizon.

    Returns
    -------
    (3,) unit vector pointing toward the sun.
    """
    az = np.radians(sun_azimuth_deg)
    el = np.radians(sun_elevation_deg)
    return np.array([
        np.sin(az) * np.cos(el),   # East
        np.cos(az) * np.cos(el),   # North
        np.sin(el),                # Up
    ])


def lambertian_render(normals: np.ndarray,
                      sun_azimuth_deg: float,
                      sun_elevation_deg: float,
                      albedo: float = 1.0) -> np.ndarray:
    """
    Render a synthetic grayscale image using the Lambertian cosine law.

    Parameters
    ----------
    normals : (H, W, 3) array of unit surface normals.
    sun_azimuth_deg, sun_elevation_deg : sun position from OHRC metadata.
    albedo : uniform surface reflectance (default 1.0).

    Returns
    -------
    (H, W) float array in [0, 1] — synthetic reflectance image.
    """
    L = sun_direction_vector(sun_azimuth_deg, sun_elevation_deg)
    reflectance = np.dot(normals, L)
    reflectance = np.clip(reflectance, 0, 1) * albedo
    return reflectance
"""
