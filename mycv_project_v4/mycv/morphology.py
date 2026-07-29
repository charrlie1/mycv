"""
mycv.morphology
===============
Binary morphological operations grounded in Set Theory.

A binary image I is a SET of foreground pixel coordinates:
    F = { (x,y) | I(x,y) = 1 }

A structuring element B is a second set defining a neighbourhood shape.

Functions
---------
dilate  : Minkowski sum  F (+) B  — expands foreground regions
erode   : Minkowski diff F (-) B  — shrinks foreground regions
opening : Erosion  then Dilation  — removes thin protrusions / noise
closing : Dilation then Erosion   — fills small holes / gaps

All operations use np.lib.stride_tricks.as_strided to build a zero-copy
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
