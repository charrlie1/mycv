"""
mycv.core
=========
Point operations on image tensors.

Functions
---------
rgb_to_grayscale  : ITU-R BT.601 luminosity conversion
threshold         : Binary pointwise thresholding
"""

import numpy as np


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale using the ITU-R BT.601 luminosity formula.

    Y = 0.2989·R + 0.5870·G + 0.1140·B

    Parameters
    ----------
    image : np.ndarray  shape (H, W, 3), dtype uint8, values in [0, 255]

    Returns
    -------
    np.ndarray  shape (H, W), dtype uint8
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image of shape (H, W, 3), got {image.shape}.")

    weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)
    grayscale = np.dot(image.astype(np.float64), weights)
    return np.clip(grayscale, 0, 255).astype(np.uint8)


def threshold(image: np.ndarray, tau: int = 127) -> np.ndarray:
    """
    Apply binary thresholding to a grayscale image.

    T(x, y) = 255  if I(x,y) >= tau,  else 0

    Parameters
    ----------
    image : np.ndarray  shape (H, W), dtype uint8
    tau   : int  threshold value in [0, 255]

    Returns
    -------
    np.ndarray  shape (H, W), dtype uint8, values in {0, 255}
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image, got shape {image.shape}.")
    if not (0 <= tau <= 255):
        raise ValueError(f"Threshold tau must be in [0, 255], got {tau}.")

    return (image >= tau).astype(np.uint8) * 255
