"""
mycv.features
=============
Classical feature extraction from gradient and edge data.

Functions
---------
harris_corner_response : Structure-tensor corner scoring map R(x,y)
detect_harris_corners   : Thresholded, non-max-suppressed Harris corner points
hough_line_transform    : Polar-space accumulator array for line detection
count_hough_lines       : Peak-clustered Hough line count (one peak per physical line)
extract_object_metrics  : Full region-property engine (bbox, moments, shape descriptors)
convex_hull             : Monotone-chain convex hull of a 2-D point set
draw_bounding_box       : In-place rectangle drawing on RGB arrays
classify_object         : Deterministic color+shape object classification

Bounding-box convention
------------------------
All bounding boxes returned by this module use the EXCLUSIVE convention
(y1, x1, y2, x2), consistent with `mycv.detection.find_template_matches`
and `mycv.morphology.component_properties`:

    width  = x2 - x1
    height = y2 - y1
    crop   = image[y1:y2, x1:x2]      (no +1 anywhere)
"""

import numpy as np
from .filters import convolve2d
from .morphology import grayscale_dilate


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
        Threshold at R > 0 and apply non-maximum suppression to find corners
        (see `detect_harris_corners`, which does this for you).
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


def detect_harris_corners(
    Gx: np.ndarray,
    Gy: np.ndarray,
    k: float = 0.05,
    window_size: int = 5,
    sigma: float = 1.4,
    threshold: float = 0.01,
    nms_radius: int = 3,
    mask: np.ndarray = None,
) -> np.ndarray:
    """
    Full Harris corner detection pipeline: response -> threshold -> NMS -> count.

    A raw `harris_corner_response` map is not a set of corners — it is a
    continuous scalar field with broad plateaus around each true corner.
    This function completes the pipeline:

        1. Compute R = harris_corner_response(Gx, Gy, ...)
        2. Threshold: keep R > threshold * R.max()   (relative threshold,
           so the same `threshold` works across images of different contrast)
        3. Local-maximum suppression: keep only pixels where R equals the
           max of its (2*nms_radius+1)^2 neighbourhood — implemented with
           `morphology.grayscale_dilate`, a vectorised sliding-window MAX
           filter (no Python pixel loop)
        4. Optionally restrict candidates to a region mask (e.g. to count
           corners belonging to a single detected object)

    Parameters
    ----------
    Gx, Gy      : np.ndarray  shape (H, W) — Sobel gradients
    k           : float  Harris sensitivity constant
    window_size : int    Gaussian window for the structure tensor
    sigma       : float  Gaussian sigma for the structure tensor
    threshold   : float  fraction of R.max() a pixel must exceed to be
                  considered a corner candidate, in [0, 1]
    nms_radius  : int    non-maximum-suppression neighbourhood radius
    mask        : np.ndarray, optional  shape (H, W) — restrict corners to
                  mask > 0 (e.g. a single object's foreground pixels)

    Returns
    -------
    np.ndarray  shape (N, 2), int — corner coordinates as (y, x) rows,
        sorted by descending response strength. `len(result)` is the
        corner count.
    """
    R = harris_corner_response(Gx, Gy, k=k, window_size=window_size, sigma=sigma)

    if mask is not None:
        R = np.where(mask > 0, R, -np.inf)

    finite = R[np.isfinite(R)]
    if finite.size == 0 or finite.max() <= 0:
        return np.empty((0, 2), dtype=np.int64)

    thresh_val = threshold * finite.max()
    candidate = R > thresh_val

    window = 2 * nms_radius + 1
    local_max_map = grayscale_dilate(R, size=window)
    is_local_max = (R == local_max_map) & candidate & np.isfinite(R)

    ys, xs = np.where(is_local_max)
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    order = np.argsort(-R[ys, xs])
    return np.stack([ys[order], xs[order]], axis=1).astype(np.int64)


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
                      NOTE: one physical line typically produces several
                      adjacent accumulator peaks. Use `count_hough_lines`
                      for a declustered, one-peak-per-line count.
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


def _nms_2d(accumulator: np.ndarray, win_r: int, win_t: int) -> np.ndarray:
    """Vectorised 2-D local-maximum mask with independent window radii per axis."""
    acc = accumulator.astype(np.float64)
    padded = np.pad(acc, ((win_r, win_r), (win_t, win_t)),
                     mode="constant", constant_values=-np.inf)
    H, W = acc.shape
    sh, sw = 2 * win_r + 1, 2 * win_t + 1
    s0, s1 = padded.strides
    patches = np.lib.stride_tricks.as_strided(
        padded, shape=(H, W, sh, sw), strides=(s0, s1, s0, s1),
    )
    local_max = patches.max(axis=(-2, -1))
    return acc == local_max


def count_hough_lines(
    edges: np.ndarray,
    n_thetas: int = 180,
    vote_threshold: int = 50,
    nms_rho: int = 10,
    nms_theta: int = 10,
) -> dict:
    """
    Detect distinct physical lines by clustering nearby Hough accumulator peaks.

    One physical line generically produces a small cluster of adjacent
    (rho, theta) accumulator bins above threshold, not a single isolated
    peak. This function runs 2-D non-maximum suppression over the
    accumulator (with independent window radii along the rho and theta
    axes, since the two axes have very different physical units) so that
    only one representative peak survives per cluster, giving a count
    that matches the number of physical lines rather than the number of
    accumulator bins above threshold.

    Parameters
    ----------
    edges          : np.ndarray  shape (H, W), uint8 — binary edge map
    n_thetas       : int    angle bins passed through to `hough_line_transform`
    vote_threshold : int    minimum votes for a bin to be a peak candidate
    nms_rho        : int    NMS window radius along the rho axis (bins)
    nms_theta      : int    NMS window radius along the theta axis (bins)

    Returns
    -------
    dict with keys
        'lines' : list of (rho, theta_deg, votes) tuples, one per detected
                  physical line, sorted by descending vote count
        'count' : int — number of detected lines, len(lines)
    """
    result = hough_line_transform(edges, n_thetas=n_thetas, threshold=vote_threshold)
    acc, thetas, rhos = result["accumulator"], result["thetas"], result["rhos"]

    is_peak = _nms_2d(acc, nms_rho, nms_theta) & (acc > vote_threshold)
    r_idx, t_idx = np.where(is_peak)
    votes = acc[r_idx, t_idx]
    order = np.argsort(-votes)

    lines = [
        (float(rhos[r_idx[i]]), float(np.degrees(thetas[t_idx[i]])), int(votes[i]))
        for i in order
    ]
    return {"lines": lines, "count": len(lines)}


# ---------------------------------------------------------------------------
#  Convex hull (monotone chain) — needed for solidity
# ---------------------------------------------------------------------------

def convex_hull(points: np.ndarray) -> np.ndarray:
    """
    Compute the convex hull of a 2-D point set using Andrew's monotone chain.

    Algorithm (O(N log N))
    -----------------------
    1. Sort points lexicographically (by x, then y).
    2. Build the lower hull: sweep left to right, popping the last point
       whenever the last three points make a non-left (clockwise or
       collinear) turn, via the 2-D cross product:

           cross(O, A, B) = (A_x-O_x)(B_y-O_y) - (A_y-O_y)(B_x-O_x)

       cross <= 0 means A does not turn left at B, so A cannot be on the
       hull and is discarded.
    3. Build the upper hull the same way, sweeping right to left.
    4. Concatenate (dropping the duplicated endpoints) for the full hull,
       in counter-clockwise order.

    Parameters
    ----------
    points : np.ndarray  shape (N, 2) — (x, y) integer or float coordinates

    Returns
    -------
    np.ndarray  shape (M, 2) — hull vertices in counter-clockwise order.
        M < 3 means the input is degenerate (all points on a line, or
        fewer than 3 distinct points) and has zero enclosed area.
    """
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(pts) < 3:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=np.float64)


def _polygon_area(polygon: np.ndarray) -> float:
    """Shoelace formula: A = 1/2 |sum_i (x_i y_{i+1} - x_{i+1} y_i)|."""
    if polygon is None or len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _perimeter_from_mask(binary_mask: np.ndarray) -> float:
    """
    Estimate the boundary perimeter via Moore-neighbour contour tracing.

    A naive perimeter estimator that simply counts 4-neighbour
    foreground/background transitions ("pixel-edge counting") treats the
    boundary as a Manhattan staircase and systematically OVER-estimates
    the true perimeter of any curved or diagonal boundary by roughly a
    factor of 4/pi (~1.27x) — a circle's edge-transition count comes out
    around 1.27x its true circumference, which would make `circularity`
    (4*pi*A / P^2) badly undershoot 1.0 even for a perfect circle.

    This function instead walks the actual 8-connected boundary contour
    (Moore-neighbour tracing with a backtrack pointer, Jacob's stopping
    criterion) starting from the topmost-then-leftmost foreground pixel,
    and sums true Euclidean step lengths: 1.0 for an orthogonal step,
    sqrt(2) for a diagonal step. This is the standard discrete-perimeter
    estimator used by region-property tools (e.g. it matches the
    chain-code-length convention used by `skimage.measure.perimeter`)
    and gives circularity ~1.0 for a discretised circle.

    Parameters
    ----------
    binary_mask : np.ndarray  shape (H, W) — single (ideally 8-connected)
                  foreground blob; non-zero pixels are foreground

    Returns
    -------
    float — estimated boundary perimeter (chain-code arc length)
    """
    binary = binary_mask > 0
    H, W = binary.shape
    ys, xs = np.where(binary)
    if ys.size == 0:
        return 0.0
    if ys.size == 1:
        return 4.0   # a lone pixel's own unit-square boundary

    # Clockwise neighbour offsets starting from North, and their step lengths.
    dirs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    step_len = [1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2)]

    order = np.lexsort((xs, ys))          # sort by y, then x
    start = (int(ys[order[0]]), int(xs[order[0]]))

    def is_fg(p):
        return 0 <= p[0] < H and 0 <= p[1] < W and binary[p]

    current = start
    backtrack_dir = 6                      # West — background by construction
                                            # (start is the topmost-leftmost fg pixel)
    perimeter = 0.0
    max_iter = 4 * ys.size + 8             # safety cap against pathological input

    for _ in range(max_iter):
        found = False
        for i in range(1, 9):
            d = (backtrack_dir + i) % 8
            ny, nx = current[0] + dirs[d][0], current[1] + dirs[d][1]
            if is_fg((ny, nx)):
                perimeter += step_len[d]
                backtrack_dir = (d + 4) % 8    # opposite direction from new current
                current = (ny, nx)
                found = True
                break
        if not found or current == start:
            break

    return float(perimeter)


def _boundary_points(binary_mask: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Restrict a foreground point set to boundary pixels only (pixels with at
    least one background 4-neighbour). The convex hull only ever needs
    extremal points, and boundary pixels are a superset of the hull
    vertices, so pre-filtering keeps `convex_hull`'s O(N log N) sweep fast
    on large filled objects without changing the result.
    """
    binary = binary_mask > 0
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    fg = padded[1:-1, 1:-1]
    is_boundary = fg & (
        (~padded[1:-1, 2:]) | (~padded[1:-1, :-2]) |
        (~padded[2:, 1:-1]) | (~padded[:-2, 1:-1])
    )
    keep = is_boundary[ys, xs]
    if not keep.any():
        return np.stack([xs, ys], axis=1)
    return np.stack([xs[keep], ys[keep]], axis=1)


# ---------------------------------------------------------------------------
#  Morphological Feature Extraction and Object Classification
# ---------------------------------------------------------------------------

def extract_object_metrics(
    binary_mask: np.ndarray,
    Gx: np.ndarray = None,
    Gy: np.ndarray = None,
    edges: np.ndarray = None,
) -> dict | None:
    """
    Extract a full set of region properties from a binary mask.

    Bounding box (EXCLUSIVE convention)
    ------------------------------------
    bbox = (y1, x1, y2, x2) where y2, x2 are EXCLUSIVE, matching
    `mycv.detection` and `mycv.morphology.component_properties`:

        bbox_width  = x2 - x1     (correct inclusive pixel count)
        bbox_height = y2 - y1
        crop        = image[y1:y2, x1:x2]     (no +1 needed anywhere)

    Area — two distinct quantities
    -------------------------------
    `bbox_area` = bbox_width * bbox_height is the area of the rectangle,
    which generally OVER-counts an irregular object (it includes
    background pixels inside the box). `pixel_area` is the true number
    of foreground pixels (`np.count_nonzero`), and is what should be used
    for filtering, moment normalisation, and shape descriptors.

    Shape descriptors
    ------------------
    - extent       = pixel_area / bbox_area                (in [0, 1])
    - moments      = raw/central second moments (m00, m10, m01,
                     mu20, mu11, mu02), giving the centroid and the
                     2x2 covariance structure of the pixel distribution
    - orientation  = 0.5 * arctan2(2*mu11, mu20 - mu02)     (radians)
    - eccentricity = sqrt(1 - lambda2/lambda1) from the eigenvalues of
                     the covariance matrix [[mu20, mu11], [mu11, mu02]]
    - perimeter    = discrete foreground/background boundary transitions
    - circularity  = 4*pi*pixel_area / perimeter^2          (1.0 = circle)
    - solidity     = pixel_area / convex_hull_area           (1.0 = convex)

    Parameters
    ----------
    binary_mask : np.ndarray
        2D binary mask (uint8 or bool) where non-zero pixels represent objects.
    Gx, Gy : np.ndarray, optional  shape (H, W) — full-image Sobel gradients.
        If both are given, `corner_count` is added using
        `detect_harris_corners` restricted to this object's mask.
    edges : np.ndarray, optional  shape (H, W) — full-image binary edge map.
        If given, `line_count` is added using `count_hough_lines`
        restricted to this object's bounding box.

    Returns
    -------
    dict or None
        Returns None if the mask is completely empty (no non-zero pixels).
        Keys: bbox, bbox_width, bbox_height, bbox_area, pixel_area,
        extent, aspect_ratio, centroid, moments, orientation,
        orientation_deg, eccentricity, perimeter, circularity, solidity,
        convex_hull, and (if requested) corner_count / line_count.
        `width`, `height`, `area` are kept as deprecated aliases for
        `bbox_width`, `bbox_height`, `bbox_area` for backward compatibility.
    """
    ys, xs = np.where(binary_mask > 0)
    if len(ys) == 0:
        return None

    # ---- bounding box (exclusive) --------------------------------------
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    bbox = (y1, x1, y2, x2)

    bbox_width  = x2 - x1
    bbox_height = y2 - y1
    bbox_area   = bbox_width * bbox_height
    pixel_area  = int(xs.size)
    extent      = pixel_area / bbox_area if bbox_area > 0 else 0.0
    aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0.0

    # ---- moments, centroid, orientation, eccentricity -------------------
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    m00 = float(pixel_area)
    m10 = float(x.sum())
    m01 = float(y.sum())
    cx, cy = m10 / m00, m01 / m00

    xc, yc = x - cx, y - cy
    mu20 = float((xc * xc).sum())
    mu02 = float((yc * yc).sum())
    mu11 = float((xc * yc).sum())

    if abs(mu20 - mu02) < 1e-12 and abs(mu11) < 1e-12:
        orientation = 0.0
    else:
        orientation = 0.5 * float(np.arctan2(2.0 * mu11, mu20 - mu02))

    common = np.sqrt(max((mu20 - mu02) ** 2 + 4.0 * mu11 ** 2, 0.0))
    lambda1 = (mu20 + mu02 + common) / 2.0
    lambda2 = (mu20 + mu02 - common) / 2.0
    if lambda1 <= 1e-12:
        eccentricity = 0.0
    else:
        eccentricity = float(np.sqrt(max(1.0 - max(lambda2, 0.0) / lambda1, 0.0)))

    # ---- perimeter and circularity --------------------------------------
    perimeter = _perimeter_from_mask(binary_mask)
    if perimeter > 0:
        circularity = float(np.clip(4.0 * np.pi * pixel_area / (perimeter ** 2), 0.0, 1.0))
    else:
        circularity = 0.0

    # ---- solidity via convex hull ----------------------------------------
    boundary_pts = _boundary_points(binary_mask, xs, ys)
    hull = convex_hull(boundary_pts)
    hull_area = _polygon_area(hull)
    solidity = float(pixel_area / hull_area) if hull_area > 0 else None

    metrics = {
        "bbox": bbox,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": bbox_area,
        "pixel_area": pixel_area,
        "extent": extent,
        "aspect_ratio": aspect_ratio,
        "centroid": (cx, cy),
        "moments": {"m00": m00, "m10": m10, "m01": m01,
                    "mu20": mu20, "mu11": mu11, "mu02": mu02},
        "orientation": orientation,
        "orientation_deg": float(np.degrees(orientation)),
        "eccentricity": eccentricity,
        "perimeter": perimeter,
        "circularity": circularity,
        "solidity": solidity,
        "convex_hull": hull,
        # --- deprecated aliases, kept for backward compatibility ---
        "width": bbox_width,
        "height": bbox_height,
        "area": bbox_area,
    }

    if Gx is not None and Gy is not None:
        corners = detect_harris_corners(Gx, Gy, mask=binary_mask)
        metrics["corner_count"] = int(len(corners))
        metrics["corners"] = corners

    if edges is not None:
        roi_edges = np.zeros_like(edges)
        roi_edges[y1:y2, x1:x2] = edges[y1:y2, x1:x2]
        line_result = count_hough_lines(roi_edges)
        metrics["line_count"] = line_result["count"]

    return metrics


def draw_bounding_box(
    image: np.ndarray,
    bbox: tuple,
    color: tuple = (255, 0, 0),
    thickness: int = 2
) -> None:
    """
    Draw a rectangle on an RGB image in-place using pure NumPy slicing.

    No loops are used. bbox uses the EXCLUSIVE convention (y1, x1, y2, x2)
    — see the module docstring — so image[y1:y2, x1:x2] is exactly the
    box interior with no off-by-one adjustment.

    Parameters
    ----------
    image : np.ndarray
        3D RGB NumPy array of shape (H, W, 3), modified in-place.
    bbox : tuple
        Bounding box as (y1, x1, y2, x2) — exclusive bottom-right corner.
    color : tuple, optional
        RGB color values (default: (255, 0, 0) — red).
    thickness : int, optional
        Line thickness in pixels (default: 2).
    """
    y1, x1, y2, x2 = bbox
    H, W = image.shape[:2]

    color_arr = np.array(color, dtype=image.dtype)

    # Clamp bounding box to image bounds
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(H, y2)
    x2 = min(W, x2)

    if y2 <= y1 or x2 <= x1:
        return   # degenerate / fully-clipped box — nothing to draw

    t = max(1, thickness)

    # Top / bottom horizontal borders
    image[y1:min(y1 + t, y2), x1:x2] = color_arr
    image[max(y2 - t, y1):y2, x1:x2] = color_arr

    # Left / right vertical borders
    image[y1:y2, x1:min(x1 + t, x2)] = color_arr
    image[y1:y2, max(x2 - t, x1):x2] = color_arr


def classify_object(
    image: np.ndarray,
    metrics_dict: dict,
    mask: np.ndarray = None,
) -> str:
    """
    Classify an object based on color and shape.

    Colour is computed from the object's actual foreground pixels when a
    mask is available, rather than the whole bounding box — a bounding
    box generally contains background pixels too (e.g. 200 object pixels
    inside a 1000-pixel box), which would otherwise bias the mean colour
    toward the background. Priority order for the colour source:

        1. `mask` argument, if given  ->  image[y1:y2, x1:x2][mask_crop > 0]
        2. `metrics_dict["mean_rgb"]`, if already precomputed
           (e.g. by `morphology.component_properties`)
        3. fallback: mean over the whole bounding box crop (legacy
           behaviour, used only when no mask information is available)

    Shape is classified from `aspect_ratio` (bbox_width / bbox_height).

    Parameters
    ----------
    image : np.ndarray
        Original RGB image as a 3D NumPy array of shape (H, W, 3).
    metrics_dict : dict
        Dictionary from `extract_object_metrics` (or
        `morphology.component_properties`) containing at least 'bbox'.
    mask : np.ndarray, optional  shape (H, W) — full-image binary mask.
        When given, colour is computed only from mask pixels inside the
        bounding box.

    Returns
    -------
    str
        Combined classification string, e.g., 'Blue Rectangle', 'Red Square'.
        Returns 'Unknown' if metrics_dict is None or invalid.
    """
    if metrics_dict is None or "bbox" not in metrics_dict:
        return "Unknown"

    y1, x1, y2, x2 = metrics_dict["bbox"]
    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        return "Unknown"

    if mask is not None:
        mask_crop = mask[y1:y2, x1:x2] > 0
        if mask_crop.any():
            mean_rgb = cropped[mask_crop].mean(axis=0)
        else:
            mean_rgb = np.mean(cropped, axis=(0, 1))
    elif "mean_rgb" in metrics_dict:
        mean_rgb = np.asarray(metrics_dict["mean_rgb"], dtype=np.float64)
    else:
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
    diff = color_matrix - mean_rgb[np.newaxis, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=1))

    closest_idx = np.argmin(distances)
    color_name = color_names[closest_idx]

    # Shape classification based on aspect ratio
    aspect_ratio = metrics_dict.get("aspect_ratio", 1.0)

    if 0.8 <= aspect_ratio <= 1.2:
        shape_name = "Square"
    else:
        shape_name = "Rectangle"

    return f"{color_name} {shape_name}"
