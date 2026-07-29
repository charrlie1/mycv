"""
mycv.features
=============
Classical feature extraction from gradient and edge data.

Functions
---------
harris_corner_response : Structure-tensor corner scoring map R(x,y)
hough_line_transform   : Polar-space accumulator array for line detection
"""

import numpy as np
from .filters import convolve2d


# ---------------------------------------------------------------------------
#  Internal helper
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Return a normalised 2-D Gaussian kernel of the given size and sigma."""
    k = size // 2
    y, x = np.mgrid[-k : k + 1, -k : k + 1]
    g = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    return (g / g.sum()).astype(np.float64)


# ---------------------------------------------------------------------------
#  Harris Corner Detector
# ---------------------------------------------------------------------------

def harris_corner_response(
    Gx: np.ndarray,
    Gy: np.ndarray,
    k: float = 0.05,
    window_size: int = 5,
    sigma: float = 1.4,
) -> np.ndarray:
    """
    Compute the Harris corner response map R from Sobel gradients.

    Structure Tensor (second-moment matrix)
    ----------------------------------------
    For each pixel, the 2x2 structure tensor M aggregates gradient energy
    over a local Gaussian window W:

        M = sum_W [ Ixx  Ixy ]    where  Ixx = Gx^2,  Ixy = Gx*Gy,  Iyy = Gy^2
                  [ Ixy  Iyy ]

    Eigenvalues lambda_1, lambda_2 of M characterise local structure:
        lambda_1 ~ lambda_2 >> 0  ->  corner
        lambda_1 >> lambda_2 ~ 0  ->  edge
        lambda_1 ~ lambda_2 ~ 0   ->  flat region

    Corner Response Function
    ------------------------
    Harris avoids eigendecomposition using algebraic invariants:

        R = det(M) - k * tr(M)^2
          = (lambda_1 * lambda_2) - k * (lambda_1 + lambda_2)^2
          = (Ixx*Iyy - Ixy^2) - k * (Ixx + Iyy)^2

    R >> 0 -> corner;   R << 0 -> edge;   |R| ~ 0 -> flat.

    Parameters
    ----------
    Gx          : np.ndarray  shape (H, W) — horizontal Sobel gradient
    Gy          : np.ndarray  shape (H, W) — vertical   Sobel gradient
    k           : float  Harris sensitivity constant, typically in [0.04, 0.06]
    window_size : int    Gaussian window side length
    sigma       : float  Gaussian sigma

    Returns
    -------
    np.ndarray  shape (H, W), float64 — raw Harris response map R.
        Threshold at R > 0 and apply non-maximum suppression to find corners.
    """
    Gx = Gx.astype(np.float64)
    Gy = Gy.astype(np.float64)

    # Gradient outer products
    Ixx = Gx * Gx
    Ixy = Gx * Gy
    Iyy = Gy * Gy

    # Gaussian-weighted summation over the local window
    # (convolution with Gaussian == windowed sum Sigma_W)
    gauss = _gaussian_kernel(size=window_size, sigma=sigma)
    Sxx = convolve2d(Ixx, gauss, padding="same")
    Sxy = convolve2d(Ixy, gauss, padding="same")
    Syy = convolve2d(Iyy, gauss, padding="same")

    # Corner response: R = det(M) - k * tr(M)^2
    det   = Sxx * Syy - Sxy ** 2      # lambda_1 * lambda_2
    trace = Sxx + Syy                  # lambda_1 + lambda_2
    return det - k * trace ** 2


# ---------------------------------------------------------------------------
#  Hough Line Transform
# ---------------------------------------------------------------------------

def hough_line_transform(
    edges: np.ndarray,
    n_thetas: int = 180,
    threshold: int = 0,
) -> dict:
    """
    Detect straight lines in a binary edge map using the Hough Transform.

    Polar Line Parameterisation
    ---------------------------
    Every line in Cartesian space can be written as:

        rho = x*cos(theta) + y*sin(theta),
        theta in [-90, 90),  rho in [-d, d]

    where d = sqrt(H^2 + W^2) is the image diagonal.

    For each edge pixel (x, y) and each angle theta_i we compute rho and cast
    a "vote" in the 2-D accumulator array A[rho_bin, theta_bin].
    Lines in the image appear as peaks in A.

    Vectorisation Strategy
    ----------------------
    Let N = number of edge pixels, T = n_thetas.

    Build an (N, T) matrix of rho values:
        rho_matrix = xs[:, None] * cos_t[None, :] + ys[:, None] * sin_t[None, :]

    Bin all (N*T) votes into the accumulator with a single np.add.at call —
    no Python-level loop over pixels.

    Parameters
    ----------
    edges     : np.ndarray  shape (H, W), uint8 — binary edge map {0, 255}
    n_thetas  : int         number of angle bins in [-pi/2, pi/2)
    threshold : int         minimum votes to include in the returned line list

    Returns
    -------
    dict with keys
        'accumulator' np.ndarray (n_rhos, n_thetas) int32 — raw vote matrix
        'thetas'      np.ndarray (n_thetas,) float64      — angle values (rad)
        'rhos'        np.ndarray (n_rhos,)   float64      — rho bin centres
        'lines'       list of (rho, theta_deg) tuples where votes > threshold
    """
    H, W   = edges.shape
    d      = int(np.ceil(np.hypot(H, W)))
    n_rhos = 2 * d + 1                          # bins: -d ... +d

    thetas   = np.linspace(-np.pi / 2, np.pi / 2, n_thetas, endpoint=False)
    rho_bins = np.linspace(-d, d, n_rhos)

    cos_t = np.cos(thetas)   # (T,)
    sin_t = np.sin(thetas)   # (T,)

    # Edge pixel coordinates
    ys, xs = np.where(edges > 0)             # each shape (N,)
    N = len(xs)

    if N == 0:
        return {
            "accumulator": np.zeros((n_rhos, n_thetas), dtype=np.int32),
            "thetas": thetas,
            "rhos": rho_bins,
            "lines": [],
        }

    # Vectorised rho computation: (N, T) via broadcasting
    rho_matrix = (
        xs[:, np.newaxis] * cos_t[np.newaxis, :]
        + ys[:, np.newaxis] * sin_t[np.newaxis, :]
    )

    # Discretise rho values: shift from [-d, +d] -> [0, 2d] then round to int
    rho_idx = np.round(rho_matrix + d).astype(np.int32)
    rho_idx = np.clip(rho_idx, 0, n_rhos - 1)

    # Scatter-accumulate votes: flatten (N, T) -> (N*T,) indices
    accumulator = np.zeros((n_rhos, n_thetas), dtype=np.int32)
    theta_idx   = np.broadcast_to(
        np.arange(n_thetas, dtype=np.int32)[np.newaxis, :],
        (N, n_thetas),
    )

    flat_idx = rho_idx.ravel() * n_thetas + theta_idx.ravel()
    np.add.at(accumulator.ravel(), flat_idx, 1)

    # Extract peaks above threshold
    r_idx, t_idx = np.where(accumulator > threshold)
    votes = accumulator[r_idx, t_idx]
    order = np.argsort(-votes)               # sort descending by vote count

    lines = [
        (float(rho_bins[r_idx[i]]), float(np.degrees(thetas[t_idx[i]])))
        for i in order
    ]

    return {
        "accumulator": accumulator,
        "thetas": thetas,
        "rhos": rho_bins,
        "lines": lines,
    }
