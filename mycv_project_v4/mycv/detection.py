"""
mycv.detection
==============
Object detection primitives: template matching, multi-scale image pyramids,
and bounding-box post-processing via Non-Maximum Suppression.

Functions
---------
match_template_ncc   : Normalised Cross-Correlation template matching
gaussian_pyramid     : Anti-aliased image pyramid via blur-then-decimate
non_max_suppression  : IoU-based redundant bounding-box suppression

Mathematical background
-----------------------
NCC is the Pearson correlation coefficient evaluated between a sliding
image patch and a fixed template. Values in [-1, 1], where +1 is a
perfect match regardless of global brightness or contrast offset.

The Gaussian pyramid satisfies the Nyquist-Shannon sampling theorem by
low-pass filtering the image before halving its spatial resolution,
preventing aliasing from high-frequency components that cannot be
represented at the coarser scale.

NMS implements the Intersection-over-Union (IoU) criterion — the Jaccard
index of two rectangle sets — to suppress spatially redundant detections
and retain only the highest-scoring bounding box per object instance.
"""

import numpy as np


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _make_gaussian_kernel5() -> np.ndarray:
    """
    Return the standard 5x5 separable Gaussian kernel (sigma ~ 1.0).

    Coefficients are the binomial row [1, 4, 6, 4, 1] / 16, outer-producted
    with itself and normalised to unit sum.
    """
    b = np.array([1, 4, 6, 4, 1], dtype=np.float64)
    k = np.outer(b, b)
    return k / k.sum()


def _extract_patches(image: np.ndarray, kH: int, kW: int) -> np.ndarray:
    """
    Zero-copy extraction of all overlapping (kH, kW) patches via as_strided.

    Parameters
    ----------
    image : np.ndarray  shape (H, W), float64 — source image (not padded)
    kH    : int         patch height
    kW    : int         patch width

    Returns
    -------
    np.ndarray  shape (H-kH+1, W-kW+1, kH, kW), float64
        patches[i, j] is the kH x kW patch whose top-left corner is (i, j).
    """
    H, W   = image.shape
    H_out  = H - kH + 1
    W_out  = W - kW + 1
    s0, s1 = image.strides
    return np.lib.stride_tricks.as_strided(
        image,
        shape=(H_out, W_out, kH, kW),
        strides=(s0, s1, s0, s1),
    )


# ---------------------------------------------------------------------------
#  1. Normalised Cross-Correlation (NCC) Template Matching
# ---------------------------------------------------------------------------

def match_template_ncc(
    image: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    """
    Compute the Normalised Cross-Correlation (NCC) response map between a
    grayscale image and a template.

    Mathematical definition — Pearson correlation over image patches
    ----------------------------------------------------------------
    Let P_{xy} be the kH x kW image patch centred at output position (x,y),
    and T the template (same size). Define:

        mu_P  = mean(P_{xy})          (local patch mean)
        mu_T  = mean(T)               (template mean,  scalar)
        s_P   = std(P_{xy})           (local patch std)
        s_T   = std(T)                (template std,   scalar)

    The NCC at position (x,y) is the Pearson correlation coefficient:

        NCC(x,y) = cov(P_{xy}, T) / (s_P * s_T)

    where the cross-covariance is:

        cov(P_{xy}, T) = mean((P_{xy} - mu_P) * (T - mu_T))

    This is equivalent to the dot product of the zero-mean, unit-norm
    patch vector and the zero-mean, unit-norm template vector. Values
    lie in [-1, 1]; NCC = +1 indicates a perfect linear match regardless
    of additive or multiplicative intensity offsets.

    Vectorisation strategy
    ----------------------
    The patch tensor P of shape (H_out, W_out, kH, kW) is built in a
    single as_strided call (zero data copy). All statistics are computed
    by reducing over the last two axes (-2, -1), yielding (H_out, W_out)
    maps for each quantity with no Python-level pixel loop.

    Division-by-zero safety
    -----------------------
    Flat patches (s_P = 0) or a flat template (s_T = 0) produce a zero
    denominator. These locations are clamped to NCC = 0 via np.where,
    since a flat patch has no discriminative information.

    Parameters
    ----------
    image    : np.ndarray  shape (H, W), dtype uint8 or float
    template : np.ndarray  shape (kH, kW), dtype uint8 or float
               Must satisfy kH <= H and kW <= W.

    Returns
    -------
    np.ndarray  shape (H-kH+1, W-kW+1), float64
        NCC response map. Values in [-1, 1]. Peaks indicate best matches.
    """
    if image.ndim != 2 or template.ndim != 2:
        raise ValueError("Both image and template must be 2-D grayscale arrays.")

    kH, kW = template.shape
    H, W   = image.shape

    if kH > H or kW > W:
        raise ValueError(
            f"Template ({kH}x{kW}) cannot be larger than image ({H}x{W})."
        )

    img = image.astype(np.float64)
    tpl = template.astype(np.float64)

    # ── Template statistics (scalars) ────────────────────────────────────
    mu_T  = tpl.mean()
    tpl_0 = tpl - mu_T                      # zero-mean template
    s_T   = tpl_0.std()                     # template standard deviation

    # ── Zero-copy patch tensor: shape (H_out, W_out, kH, kW) ─────────────
    patches = _extract_patches(img, kH, kW)  # no data copied

    # ── Local patch statistics — reduce over (kH, kW) axes ───────────────
    mu_P  = patches.mean(axis=(-2, -1), keepdims=True)   # (H_out, W_out, 1, 1)
    P_0   = patches - mu_P                               # zero-mean patches
    s_P   = P_0.std(axis=(-2, -1))                       # (H_out, W_out)

    # ── Cross-covariance: mean of elementwise product ─────────────────────
    # tpl_0 broadcasts (kH, kW) against P_0 (H_out, W_out, kH, kW)
    cross_cov = (P_0 * tpl_0).mean(axis=(-2, -1))        # (H_out, W_out)

    # ── NCC = cross-covariance / (s_P * s_T) ─────────────────────────────
    denom = s_P * s_T

    # Guard: if denominator is (near) zero, the patch is flat -> NCC = 0
    ncc = np.where(denom < 1e-10, 0.0, cross_cov / denom)

    return np.clip(ncc, -1.0, 1.0)


def find_template_matches(
    ncc_map: np.ndarray,
    template_shape: tuple,
    threshold: float = 0.8,
    nms_iou: float = 0.3,
) -> tuple:
    """
    Extract bounding boxes from an NCC response map above a threshold,
    then apply Non-Maximum Suppression to remove duplicates.

    Parameters
    ----------
    ncc_map        : np.ndarray  shape (H_out, W_out) — NCC response map
    template_shape : (kH, kW)    — template dimensions, used to build boxes
    threshold      : float        — minimum NCC score to consider a match
    nms_iou        : float        — IoU threshold passed to non_max_suppression

    Returns
    -------
    boxes  : np.ndarray  shape (M, 4)  [y1, x1, y2, x2] of surviving detections
    scores : np.ndarray  shape (M,)    corresponding NCC scores
    """
    kH, kW = template_shape

    ys, xs = np.where(ncc_map >= threshold)
    if len(ys) == 0:
        return np.empty((0, 4), dtype=np.float64), np.empty((0,), dtype=np.float64)

    scores = ncc_map[ys, xs]
    boxes  = np.stack([ys, xs, ys + kH, xs + kW], axis=1).astype(np.float64)

    return non_max_suppression(boxes, scores, iou_threshold=nms_iou)


# ---------------------------------------------------------------------------
#  2. Gaussian Image Pyramid
# ---------------------------------------------------------------------------

def gaussian_pyramid(
    image: np.ndarray,
    levels: int = 4,
) -> list:
    """
    Build a Gaussian image pyramid by iteratively blurring and decimating.

    Nyquist-Shannon necessity of low-pass filtering before decimation
    -----------------------------------------------------------------
    Downsampling by factor 2 (taking every 2nd pixel) halves the Nyquist
    frequency from f_N to f_N/2.  Any signal component with spatial
    frequency f > f_N/2 cannot be represented at the coarser resolution
    and — critically — it does NOT disappear.  Instead, it *aliases*:
    it folds back into the representable band and masquerades as a
    lower-frequency signal, corrupting the downsampled image irreversibly.

    The Gaussian kernel acts as an ideal low-pass filter (in the discrete
    sense), attenuating all energy above f_N/2 before the decimation step.
    This ensures the sampled signal satisfies the theorem's condition:
    all signal energy lies strictly below the new Nyquist limit.

    Pyramid construction
    --------------------
    Level 0 : original image  I_0 = I
    Level k : I_k = (I_{k-1} * G)[::2, ::2]

    where G is the 5x5 Gaussian kernel and * denotes 2D convolution
    (delegated to mycv.filters.convolve2d with 'same' padding).

    Parameters
    ----------
    image  : np.ndarray  shape (H, W), uint8 or float
    levels : int         total number of pyramid levels (including level 0)

    Returns
    -------
    list of np.ndarray, length == levels
        pyramid[0] is the original image (float64).
        pyramid[k] has shape approximately (H/2^k, W/2^k).
    """
    if image.ndim != 2:
        raise ValueError("gaussian_pyramid expects a 2-D grayscale image.")
    if levels < 1:
        raise ValueError("levels must be >= 1.")

    # Import here to avoid circular dependency; filters has no dependency on detection
    from .filters import convolve2d

    kernel  = _make_gaussian_kernel5()
    pyramid = [image.astype(np.float64)]

    for _ in range(levels - 1):
        current = pyramid[-1]
        if current.shape[0] < 2 or current.shape[1] < 2:
            break  # image too small to downsample further
        blurred    = convolve2d(current, kernel, padding="same")
        downsampled = blurred[::2, ::2]
        pyramid.append(downsampled)

    return pyramid


# ---------------------------------------------------------------------------
#  3. Non-Maximum Suppression via Intersection over Union
# ---------------------------------------------------------------------------

def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> tuple:
    """
    Suppress redundant bounding boxes using the Greedy IoU-NMS algorithm.

    Intersection over Union — the geometry
    ---------------------------------------
    Two axis-aligned bounding boxes A = [y1_A, x1_A, y2_A, x2_A] and
    B = [y1_B, x1_B, y2_B, x2_B] have:

        Intersection height : h = max(0, min(y2_A, y2_B) - max(y1_A, y1_B))
        Intersection width  : w = max(0, min(x2_A, x2_B) - max(x1_A, x1_B))
        Intersection area   : I = h * w

    The intersection is always a rectangle (possibly degenerate with zero
    area when the boxes do not overlap). The max/min operations enforce
    that negative extents — which would arise when the boxes are disjoint
    — are clamped to zero.

    The Union area is:
        U = area(A) + area(B) - I          (inclusion-exclusion principle)

    The Jaccard index (IoU) is:
        IoU(A, B) = I / U  in [0, 1]

    IoU = 1 only if A and B are identical; IoU = 0 when they are disjoint.

    Greedy NMS algorithm
    --------------------
    1. Sort all boxes by score, descending.
    2. Take the highest-scoring box; add it to the output set.
    3. Compute IoU between that box and every remaining candidate.
    4. Suppress (discard) any candidate with IoU > iou_threshold.
    5. Repeat from step 2 with the unsuppressed candidates.

    The IoU computation in step 3 is fully vectorised: a single
    np.maximum / np.minimum broadcast over the entire remaining-candidate
    array, with no per-box Python loop.

    Parameters
    ----------
    boxes         : np.ndarray  shape (N, 4), float — [y1, x1, y2, x2]
    scores        : np.ndarray  shape (N,),   float — detection confidence
    iou_threshold : float       suppress boxes with IoU > this value

    Returns
    -------
    kept_boxes  : np.ndarray  shape (M, 4)
    kept_scores : np.ndarray  shape (M,)
        M <= N surviving detections, sorted by score descending.
    """
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must be shape (N, 4), got {boxes.shape}.")
    if len(scores) != len(boxes):
        raise ValueError("boxes and scores must have the same length.")

    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float64), np.empty((0,), dtype=np.float64)

    boxes  = boxes.astype(np.float64)
    scores = scores.astype(np.float64)

    # Pre-compute individual box areas
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Sort by score descending; work with index array
    order = np.argsort(-scores)

    kept = []
    while order.size > 0:
        # The highest-scoring remaining candidate
        best = order[0]
        kept.append(best)

        if order.size == 1:
            break

        rest = order[1:]   # all other remaining candidates

        # ── Vectorised IoU against 'best' ─────────────────────────────────
        # Intersection rectangle
        inter_y1 = np.maximum(boxes[best, 0], boxes[rest, 0])
        inter_x1 = np.maximum(boxes[best, 1], boxes[rest, 1])
        inter_y2 = np.minimum(boxes[best, 2], boxes[rest, 2])
        inter_x2 = np.minimum(boxes[best, 3], boxes[rest, 3])

        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter   = inter_h * inter_w   # (len(rest),)

        union   = areas[best] + areas[rest] - inter
        iou     = np.where(union > 0, inter / union, 0.0)

        # Retain only candidates with IoU <= threshold (not suppressed)
        order = rest[iou <= iou_threshold]

    kept_idx = np.array(kept, dtype=np.int64)
    return boxes[kept_idx], scores[kept_idx]
