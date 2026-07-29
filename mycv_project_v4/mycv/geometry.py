"""
mycv.geometry
=============
Geometric transformations using affine and projective mapping.

Functions
---------
bilinear_interpolate : Vectorised sub-pixel sampler (shared by all warps)
rotate_image         : Centre-rotation via backward affine mapping
warp_perspective     : Projective warp via inverse Homography + bilinear interp
"""

import numpy as np


def bilinear_interpolate(
    image: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    """
    Vectorised bilinear interpolation over a single-channel float image.

    For a real-valued source coordinate (xs[i,j], ys[i,j]):

        I(xs, ys) = (1-dx)(1-dy)*I[y0,x0] + dx*(1-dy)*I[y0,x1]
                  + (1-dx)*dy*I[y1,x0]     + dx*dy*I[y1,x1]

    where x0 = floor(xs), x1 = x0+1, dx = xs - x0 (and likewise for y).
    Out-of-bounds coordinates are clamped to the edge (replication padding).

    Parameters
    ----------
    image : np.ndarray  shape (H, W), float64
    xs    : np.ndarray  shape (H_out, W_out) — fractional column indices
    ys    : np.ndarray  shape (H_out, W_out) — fractional row    indices

    Returns
    -------
    np.ndarray  shape (H_out, W_out), float64
    """
    H, W = image.shape

    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1

    dx = (xs - x0).astype(np.float64)
    dy = (ys - y0).astype(np.float64)

    x0 = np.clip(x0, 0, W - 1);  x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1);  y1 = np.clip(y1, 0, H - 1)

    return (
        (1 - dx) * (1 - dy) * image[y0, x0]
        +      dx * (1 - dy) * image[y0, x1]
        + (1 - dx) *      dy * image[y1, x0]
        +      dx  *      dy * image[y1, x1]
    )


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate `image` counter-clockwise by `angle_deg` degrees about its centre.

    Transformation:  M = T_back . R(theta) . T_origin   (3x3 homogeneous)
    Backward mapping: src = M^{-1} . dst, then bilinear sample.

    Parameters
    ----------
    image     : np.ndarray  shape (H, W), uint8 or float
    angle_deg : float — counter-clockwise rotation angle in degrees

    Returns
    -------
    np.ndarray  shape (H, W), dtype uint8
    """
    image_f = image.astype(np.float64)
    H, W    = image_f.shape
    theta   = np.deg2rad(angle_deg)
    cx, cy  = (W - 1) / 2.0, (H - 1) / 2.0

    T_to = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    R    = np.array([[ np.cos(theta), -np.sin(theta), 0],
                     [ np.sin(theta),  np.cos(theta), 0],
                     [0, 0, 1]], dtype=np.float64)
    T_bk = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)

    M_inv = np.linalg.inv(T_bk @ R @ T_to)

    xv, yv = np.meshgrid(np.arange(W), np.arange(H))
    grid   = np.stack([xv.ravel(), yv.ravel(),
                       np.ones(H * W)]).astype(np.float64)
    src    = M_inv @ grid

    xs = src[0].reshape(H, W)
    ys = src[1].reshape(H, W)

    return np.clip(bilinear_interpolate(image_f, xs, ys), 0, 255).astype(np.uint8)


def warp_perspective(
    image: np.ndarray,
    H: np.ndarray,
    output_shape: tuple,
) -> np.ndarray:
    """
    Apply a 3x3 Homography matrix H to warp `image` into a new view-plane.

    Projective backward mapping
    ---------------------------
    For every destination pixel p' = [x', y', 1]^T:

        p_tilde = H^{-1} . p'  =  [x_tilde, y_tilde, w_tilde]^T

    Division by w_tilde converts from the projective plane P^2 back to R^2:

        xs = x_tilde / w_tilde,    ys = y_tilde / w_tilde

    This division is the non-linear step that makes perspective possible.
    w_tilde encodes the foreshortening factor of depth — without it, parallel
    lines cannot converge and you get only an affine (linear) warp.

    Parameters
    ----------
    image        : np.ndarray  shape (H_in, W_in), uint8 or float
    H            : np.ndarray  shape (3, 3) — forward Homography (src -> dst)
                   Inverted internally for backward mapping.
    output_shape : (H_out, W_out) — destination canvas size

    Returns
    -------
    np.ndarray  shape (H_out, W_out), dtype uint8
    """
    if H.shape != (3, 3):
        raise ValueError(f"H must be a 3x3 matrix, got {H.shape}.")

    H_out, W_out = output_shape
    image_f      = image.astype(np.float64)
    H_inv        = np.linalg.inv(H.astype(np.float64))

    # Dense grid of destination pixels in homogeneous coords: (3, H_out*W_out)
    xv, yv = np.meshgrid(np.arange(W_out), np.arange(H_out))
    ones   = np.ones(H_out * W_out, dtype=np.float64)
    dst_grid = np.stack([xv.ravel(), yv.ravel(), ones])

    # Apply inverse Homography
    src_hom = H_inv @ dst_grid               # (3, H_out*W_out)
    x_tilde = src_hom[0]
    y_tilde = src_hom[1]
    w_tilde = src_hom[2]

    # Projective division: P^2 -> R^2
    # Guard against division by zero at points at infinity
    w_safe = np.where(np.abs(w_tilde) < 1e-10, 1e-10, w_tilde)
    xs = (x_tilde / w_safe).reshape(H_out, W_out)
    ys = (y_tilde / w_safe).reshape(H_out, W_out)

    warped = bilinear_interpolate(image_f, xs, ys)
    return np.clip(warped, 0, 255).astype(np.uint8)
