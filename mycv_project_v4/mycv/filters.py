"""
mycv.filters
============
Spatial filtering via 2-D discrete convolution.

Functions
---------
convolve2d           : Pure-NumPy 2-D convolution engine (stride tricks + einsum)
sobel_edge_detection : Sobel gradient-magnitude and direction maps

Kernels
-------
SOBEL_X : 3x3 horizontal Sobel kernel  (dI/dx)
SOBEL_Y : 3x3 vertical Sobel kernel    (dI/dy)
"""

import numpy as np


# Sobel kernels
# Kx: outer product of Gaussian smoother [1,2,1]^T and finite-difference [-1,0,+1]
SOBEL_X = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1],
], dtype=np.float64)

SOBEL_Y = SOBEL_X.T   # Ky = Kx^T — detects horizontal edges


def convolve2d(
    image: np.ndarray,
    kernel: np.ndarray,
    padding: str = "same",
) -> np.ndarray:
    """
    Perform a true 2-D discrete convolution on a single-channel image.

    Discrete convolution:
        (I * K)[x, y] = sum_i sum_j I[x-i, y-j] * K[i, j]

    The kernel is flipped on both axes (np.flip) to implement true convolution
    rather than cross-correlation. Patch extraction uses as_strided (zero copies)
    and contraction uses np.einsum in one vectorised pass.

    Parameters
    ----------
    image   : np.ndarray  shape (H, W)
    kernel  : np.ndarray  shape (kH, kW)
    padding : 'same'  -> output shape == input shape (zero-pad)
              'valid' -> no padding; output shrinks by (kH-1, kW-1)

    Returns
    -------
    np.ndarray  float64
    """
    if image.ndim != 2:
        raise ValueError("image must be a 2-D array. Convert to grayscale first.")

    image = image.astype(np.float64)
    kH, kW = kernel.shape

    if padding == "same":
        pad_h, pad_w = kH // 2, kW // 2
        image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                       mode="constant", constant_values=0)
    elif padding != "valid":
        raise ValueError("padding must be 'same' or 'valid'.")

    H_p, W_p = image.shape
    H_out = H_p - kH + 1
    W_out = W_p - kW + 1

    # Zero-copy (H_out, W_out, kH, kW) patch view via stride tricks
    s0, s1 = image.strides
    patches = np.lib.stride_tricks.as_strided(
        image,
        shape=(H_out, W_out, kH, kW),
        strides=(s0, s1, s0, s1),
    )

    # True convolution requires flipping the kernel on both spatial axes
    kernel_flipped = np.flip(kernel)

    # Single einsum: contract over (kH, kW) patch axes -> (H_out, W_out)
    return np.einsum("hwij,ij->hw", patches, kernel_flipped)


def sobel_edge_detection(image: np.ndarray) -> dict:
    """
    Compute Sobel gradient maps for a grayscale image.

        |nabla I| = sqrt(Gx^2 + Gy^2)    — gradient magnitude
        theta     = arctan2(Gy, Gx)       — gradient direction

    Parameters
    ----------
    image : np.ndarray  shape (H, W), uint8 or float64

    Returns
    -------
    dict with keys
        'Gx'        np.ndarray (H, W) float64  — horizontal gradient
        'Gy'        np.ndarray (H, W) float64  — vertical gradient
        'magnitude' np.ndarray (H, W) uint8    — normalised gradient magnitude
        'direction' np.ndarray (H, W) float64  — angle in radians, in (-pi, pi)
    """
    Gx = convolve2d(image, SOBEL_X, padding="same")
    Gy = convolve2d(image, SOBEL_Y, padding="same")

    magnitude = np.sqrt(Gx**2 + Gy**2)
    direction = np.arctan2(Gy, Gx)

    max_val = magnitude.max()
    mag_norm = (
        (magnitude / max_val * 255).astype(np.uint8)
        if max_val > 0
        else magnitude.astype(np.uint8)
    )

    return {"Gx": Gx, "Gy": Gy, "magnitude": mag_norm, "direction": direction}
