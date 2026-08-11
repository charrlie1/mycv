"""
mycv.features
=============
Classical feature extraction from gradient and edge data.

Functions
---------
harris_corner_response : Structure-tensor corner scoring map R(x,y)
hough_line_transform   : Polar-space accumulator array for line detection
extract_object_metrics : Morphological bounding-box and shape metrics
draw_bounding_box      : In-place rectangle drawing on RGB arrays
classify_object        : Deterministic color+shape object classification
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


# ---------------------------------------------------------------------------
#  Morphological Feature Extraction and Object Classification
# ---------------------------------------------------------------------------

def extract_object_metrics(binary_mask: np.ndarray) -> dict | None:
    """
    Extract morphological metrics from a binary mask.

    Uses np.where to find all non-zero pixels and determines the absolute
    minimum and maximum coordinates to form the bounding box.

    Parameters
    ----------
    binary_mask : np.ndarray
        2D binary mask (uint8 or bool) where non-zero pixels represent objects.

    Returns
    -------
    dict or None
        Dictionary with keys: 'bbox', 'width', 'height', 'area', 'aspect_ratio'
        Returns None if the mask is completely empty (no non-zero pixels).

        - bbox: tuple (y1, x1, y2, x2) — top-left and bottom-right corners
        - width: int — bounding box width (x2 - x1)
        - height: int — bounding box height (y2 - y1)
        - area: int — bounding box area (width * height)
        - aspect_ratio: float — width / height (0.0 if height is 0)
    """
    # Find all non-zero pixel coordinates
    ys, xs = np.where(binary_mask > 0)

    # Handle empty mask edge case
    if len(ys) == 0 or len(xs) == 0:
        return None

    # Compute bounding box from min/max coordinates
    y1 = int(np.min(ys))
    y2 = int(np.max(ys))
    x1 = int(np.min(xs))
    x2 = int(np.max(xs))

    bbox = (y1, x1, y2, x2)

    # Calculate dimensions
    width = x2 - x1
    height = y2 - y1

    # Calculate area
    area = width * height

    # Calculate aspect ratio with safe division-by-zero handling
    if height == 0:
        aspect_ratio = 0.0
    else:
        aspect_ratio = width / height

    return {
        "bbox": bbox,
        "width": width,
        "height": height,
        "area": area,
        "aspect_ratio": aspect_ratio,
    }


def draw_bounding_box(
    image: np.ndarray,
    bbox: tuple,
    color: tuple = (255, 0, 0),
    thickness: int = 2
) -> None:
    """
    Draw a rectangle on an RGB image in-place using pure NumPy slicing.

    No loops are used. The function overwrites exact pixel values at the
    borders of the bounding box based on the given thickness.

    Parameters
    ----------
    image : np.ndarray
        3D RGB NumPy array of shape (H, W, 3), modified in-place.
    bbox : tuple
        Bounding box as (y1, x1, y2, x2) — top-left and bottom-right corners.
    color : tuple, optional
        RGB color values (default: (255, 0, 0) — red).
    thickness : int, optional
        Line thickness in pixels (default: 2).
    """
    y1, x1, y2, x2 = bbox
    H, W = image.shape[:2]

    # Convert color to array for broadcasting
    color_arr = np.array(color, dtype=image.dtype)

    # Clamp bounding box to image bounds
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(H - 1, y2)
    x2 = min(W - 1, x2)

    # Draw top horizontal border
    image[y1:y1 + thickness, x1:x2 + 1] = color_arr

    # Draw bottom horizontal border
    image[y2 - thickness + 1:y2 + 1, x1:x2 + 1] = color_arr

    # Draw left vertical border
    image[y1:y2 + 1, x1:x1 + thickness] = color_arr

    # Draw right vertical border
    image[y1:y2 + 1, x2 - thickness + 1:x2 + 1] = color_arr


def classify_object(image: np.ndarray, metrics_dict: dict) -> str:
    """
    Classify an object based on color and shape.

    Crops the original RGB image using the bounding box coordinates,
    calculates the mean RGB value, and finds the closest target color
    using vectorized Euclidean distance. Shape is classified based on
    aspect ratio.

    Parameters
    ----------
    image : np.ndarray
        Original RGB image as a 3D NumPy array of shape (H, W, 3).
    metrics_dict : dict
        Dictionary from extract_object_metrics containing 'bbox' and 'aspect_ratio'.

    Returns
    -------
    str
        Combined classification string, e.g., 'Blue Rectangle', 'Red Square'.
        Returns 'Unknown' if metrics_dict is None or invalid.
    """
    if metrics_dict is None or "bbox" not in metrics_dict:
        return "Unknown"

    bbox = metrics_dict["bbox"]
    y1, x1, y2, x2 = bbox

    # Crop the ROI from the original image
    cropped = image[y1:y2 + 1, x1:x2 + 1]

    # Handle empty crop
    if cropped.size == 0:
        return "Unknown"

    # Calculate mean RGB value across spatial axes (axis 0 and 1)
    mean_rgb = np.mean(cropped, axis=(0, 1))

    # Define target colors dictionary
    target_colors = {
        "Red": np.array([255, 0, 0], dtype=np.float64),
        "Green": np.array([0, 255, 0], dtype=np.float64),
        "Blue": np.array([0, 0, 255], dtype=np.float64),
        "Yellow": np.array([255, 255, 0], dtype=np.float64),
        "Black": np.array([0, 0, 0], dtype=np.float64),
        "White": np.array([255, 255, 255], dtype=np.float64),
    }

    # Stack target colors into a matrix for vectorized distance calculation
    color_names = list(target_colors.keys())
    color_matrix = np.stack([target_colors[name] for name in color_names])

    # Vectorized Euclidean distance: ||mean_rgb - target||_2
    # distances[i] = sqrt(sum((mean_rgb - color_matrix[i])^2))
    diff = color_matrix - mean_rgb[np.newaxis, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=1))

    # Find the index of the minimum distance (closest color)
    closest_idx = np.argmin(distances)
    color_name = color_names[closest_idx]

    # Shape classification based on aspect ratio
    aspect_ratio = metrics_dict.get("aspect_ratio", 1.0)

    if 0.8 <= aspect_ratio <= 1.2:
        shape_name = "Square"
    else:
        shape_name = "Rectangle"

    return f"{color_name} {shape_name}"