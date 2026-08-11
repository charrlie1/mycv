"""
mycv.morphology
===============
Binary morphological operations grounded in Set Theory.

A binary image I is a SET of foreground pixel coordinates:
    F = { (x,y) | I(x,y) = 1 }

A structuring element B is a second set defining a neighbourhood shape.

Functions
---------
dilate                    : Minkowski sum  F (+) B  — expands foreground regions
erode                     : Minkowski diff F (-) B  — shrinks foreground regions
opening                   : Erosion  then Dilation  — removes thin protrusions / noise
closing                   : Dilation then Erosion   — fills small holes / gaps
grayscale_dilate          : Sliding-window MAX filter (grayscale morphology) —
                             used for local-maximum suppression (e.g. Harris corners)
label_connected_components: Two-pass union-find binary connected-component labelling
component_properties      : Per-component region properties (area, bbox, centroid, ...)

All binary operations use np.lib.stride_tricks.as_strided to build a zero-copy
patch tensor (H_out, W_out, kH, kW) and apply logical ANY / ALL reductions.
"""

import numpy as np


def _extract_patches(binary: np.ndarray, kH: int, kW: int) -> np.ndarray:
    """
    Zero-copy extraction of all overlapping (kH, kW) patches from a 2-D
    zero-padded binary array.

    Returns
    -------
    np.ndarray  shape (H, W, kH, kW), bool
        patches[i, j] is the kH x kW neighbourhood centred on pixel (i, j).
    """
    pH, pW = kH // 2, kW // 2
    padded = np.pad(binary, ((pH, pH), (pW, pW)),
                    mode="constant", constant_values=0)
    H, W   = binary.shape
    s0, s1 = padded.strides
    return np.lib.stride_tricks.as_strided(
        padded,
        shape=(H, W, kH, kW),
        strides=(s0, s1, s0, s1),
    )


def dilate(image: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    """
    Binary dilation: F (+) B (Minkowski sum).

    A pixel belongs to the output iff the structuring element B, centred
    on that pixel, overlaps at least one foreground pixel (ANY logic):

        out(x,y) = 1  <=>  exists (i,j) in B : image(x+i, y+j) = 1

    Parameters
    ----------
    image  : np.ndarray  shape (H, W), uint8, values in {0, 255}
    kernel : np.ndarray  shape (kH, kW), binary structuring element.
             Defaults to a 3x3 block of ones.

    Returns
    -------
    np.ndarray  shape (H, W), uint8, values in {0, 255}
    """
    if kernel is None:
        kernel = np.ones((3, 3), dtype=np.bool_)

    binary  = (image > 0)
    kH, kW  = kernel.shape
    patches = _extract_patches(binary, kH, kW)           # (H, W, kH, kW)

    se_mask = kernel.astype(np.bool_)
    masked  = patches & se_mask[np.newaxis, np.newaxis, :, :]

    # A pixel is foreground if ANY neighbour (within B) is foreground
    dilated = masked.any(axis=(-2, -1))

    return (dilated * 255).astype(np.uint8)


def erode(image: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    """
    Binary erosion: F (-) B (Minkowski difference).

    A pixel belongs to the output iff the entire structuring element B fits
    inside the foreground set (ALL logic):

        out(x,y) = 1  <=>  forall (i,j) in B : image(x+i, y+j) = 1

    Parameters
    ----------
    image  : np.ndarray  shape (H, W), uint8, values in {0, 255}
    kernel : np.ndarray  shape (kH, kW), binary structuring element.
             Defaults to a 3x3 block of ones.

    Returns
    -------
    np.ndarray  shape (H, W), uint8, values in {0, 255}
    """
    if kernel is None:
        kernel = np.ones((3, 3), dtype=np.bool_)

    binary  = (image > 0)
    kH, kW  = kernel.shape
    patches = _extract_patches(binary, kH, kW)           # (H, W, kH, kW)

    se_mask = kernel.astype(np.bool_)
    active  = se_mask[np.newaxis, np.newaxis, :, :]

    # Inactive SE cells are treated as always satisfied (True)
    filled_patch = np.where(active, patches, True)

    # A pixel survives erosion only if ALL active SE cells are foreground
    eroded = filled_patch.all(axis=(-2, -1))

    return (eroded * 255).astype(np.uint8)


def opening(image: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    """
    Morphological opening: (F (-) B) (+) B.

    Erosion followed by dilation. Removes thin protrusions, isolated noise
    pixels, and breaks narrow bridges while preserving large regions.

    Parameters
    ----------
    image  : np.ndarray  shape (H, W), uint8, values in {0, 255}
    kernel : structuring element (see dilate / erode)

    Returns
    -------
    np.ndarray  shape (H, W), uint8
    """
    return dilate(erode(image, kernel), kernel)


def closing(image: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    """
    Morphological closing: (F (+) B) (-) B.

    Dilation followed by erosion. Fills small holes inside foreground regions
    and closes narrow gaps between nearby shapes.

    Parameters
    ----------
    image  : np.ndarray  shape (H, W), uint8, values in {0, 255}
    kernel : structuring element (see dilate / erode)

    Returns
    -------
    np.ndarray  shape (H, W), uint8
    """
    return erode(dilate(image, kernel), kernel)


# ---------------------------------------------------------------------------
#  Grayscale morphology — sliding-window MAX filter
# ---------------------------------------------------------------------------

def grayscale_dilate(image: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Grayscale dilation: a sliding-window MAX filter.

        out(x,y) = max_{(i,j) in window} image(x+i, y+j)

    Unlike binary `dilate` (which ORs a boolean neighbourhood), this
    operates on real-valued response maps and is the standard building
    block for local-maximum (non-maximum suppression) tests:

        is_local_max(x,y)  <=>  image(x,y) == grayscale_dilate(image)(x,y)

    Out-of-window pixels are padded with -inf so they never win the max
    and never spuriously suppress a true border maximum.

    Parameters
    ----------
    image : np.ndarray  shape (H, W), float
    size  : int  odd window side length (default 3)

    Returns
    -------
    np.ndarray  shape (H, W), float64 — windowed maximum map
    """
    if size % 2 == 0:
        raise ValueError("size must be odd.")

    img = image.astype(np.float64)
    k = size // 2
    padded = np.pad(img, k, mode="constant", constant_values=-np.inf)

    H, W = img.shape
    s0, s1 = padded.strides
    patches = np.lib.stride_tricks.as_strided(
        padded,
        shape=(H, W, size, size),
        strides=(s0, s1, s0, s1),
    )
    return patches.max(axis=(-2, -1))


# ---------------------------------------------------------------------------
#  Connected-Component Labelling (two-pass union-find)
# ---------------------------------------------------------------------------

def label_connected_components(binary_mask: np.ndarray, connectivity: int = 8) -> tuple:
    """
    Label connected components of a binary mask using two-pass union-find.

    Why connected components?
    --------------------------
    A colour mask may contain several disjoint blobs (e.g. two red objects
    in the same frame). Averaging all foreground pixels with
    `calculate_centroid` collapses them into a single centroid lying
    *between* the objects, which is wrong for tracking. Connected-component
    labelling first partitions the foreground set F into disjoint subsets
    F_1, F_2, ..., F_k so that each F_i can be processed independently
    (area, bbox, centroid, colour, ...).

    Algorithm
    ---------
    First pass (raster scan): for each foreground pixel, look at its
    already-visited causal neighbours (up + left, plus the two diagonals
    for 8-connectivity). If none are foreground, assign a fresh
    provisional label. If one or more are foreground, assign the minimum
    of their labels and record a union-find equivalence between all of
    them (they belong to the same physical blob even though they were
    given different provisional labels, e.g. a "U" shape).

    Second pass: resolve every provisional label to its union-find root
    and relabel roots to consecutive integers 1..num_labels.

    Note on performance
    --------------------
    This is a genuine two-pass union-find implementation with an explicit
    raster-scan Python loop (connected-component labelling is inherently
    sequential — it cannot be vectorised into pure NumPy ufuncs without
    external libraries). For very large images this will be slower than
    `scipy.ndimage.label`; if `scipy` is an acceptable dependency for your
    deployment, prefer it there. This implementation exists to keep `mycv`
    fully dependency-free.

    Parameters
    ----------
    binary_mask  : np.ndarray  shape (H, W) — non-zero pixels are foreground
    connectivity : int  4 or 8 (default 8)

    Returns
    -------
    labels     : np.ndarray  shape (H, W), int64 — 0 is background,
                 1..num_labels index each connected component
    num_labels : int — number of connected components found
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8.")

    binary = (binary_mask > 0)
    H, W = binary.shape
    labels = np.zeros((H, W), dtype=np.int64)

    # Union-find over provisional labels. parent[0] is an unused sentinel.
    parent = [0]

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:          # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    if connectivity == 8:
        causal_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]
    else:
        causal_offsets = [(-1, 0), (0, -1)]

    next_label = 1
    for y in range(H):
        row = binary[y]
        for x in range(W):
            if not row[x]:
                continue
            neighbour_labels = []
            for dy, dx in causal_offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and labels[ny, nx] > 0:
                    neighbour_labels.append(int(labels[ny, nx]))

            if not neighbour_labels:
                labels[y, x] = next_label
                parent.append(next_label)
                next_label += 1
            else:
                m = min(neighbour_labels)
                labels[y, x] = m
                for lbl in neighbour_labels:
                    union(m, lbl)

    # Second pass: resolve every label to its root, relabel to 1..num_labels
    root_to_final = {}
    out_label = 0
    flat = labels.ravel()
    for i in range(flat.size):
        v = flat[i]
        if v == 0:
            continue
        r = find(int(v))
        final = root_to_final.get(r)
        if final is None:
            out_label += 1
            root_to_final[r] = out_label
            final = out_label
        flat[i] = final

    return labels.reshape(H, W), out_label


def component_properties(labels: np.ndarray, num_labels: int, image: np.ndarray = None) -> list:
    """
    Compute per-component region properties from a label map.

    Bounding boxes use the EXCLUSIVE convention (y1, x1, y2, x2) consistent
    with the rest of `mycv` (see `features.extract_object_metrics` and
    `detection.find_template_matches`): `image[y1:y2, x1:x2]` crops exactly
    the component's bounding box, no off-by-one arithmetic required.

    Parameters
    ----------
    labels     : np.ndarray  shape (H, W), int — output of
                 `label_connected_components`
    num_labels : int  — number of components (as returned alongside `labels`)
    image      : np.ndarray, optional  shape (H, W, 3) — if given, each
                 component's `mean_rgb` is computed from its member pixels
                 only (not the whole bounding box)

    Returns
    -------
    list of dict, one per component (label 1..num_labels), each with keys:
        'label', 'bbox', 'pixel_area', 'bbox_area', 'extent',
        'aspect_ratio', 'centroid', and 'mean_rgb' (if `image` given)
    """
    H, W = labels.shape
    yy, xx = np.mgrid[0:H, 0:W]

    props = []
    for lbl in range(1, num_labels + 1):
        mask = labels == lbl
        ys = yy[mask]
        xs = xx[mask]
        if ys.size == 0:
            continue

        y1, y2 = int(ys.min()), int(ys.max()) + 1     # exclusive
        x1, x2 = int(xs.min()), int(xs.max()) + 1     # exclusive
        bbox_h, bbox_w = y2 - y1, x2 - x1
        bbox_area = bbox_h * bbox_w
        pixel_area = int(ys.size)
        extent = pixel_area / bbox_area if bbox_area > 0 else 0.0
        aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 0.0

        entry = {
            "label": lbl,
            "bbox": (y1, x1, y2, x2),
            "pixel_area": pixel_area,
            "bbox_area": bbox_area,
            "extent": extent,
            "aspect_ratio": aspect_ratio,
            "centroid": (float(xs.mean()), float(ys.mean())),
        }
        if image is not None:
            entry["mean_rgb"] = image[mask].mean(axis=0)

        props.append(entry)

    return props
