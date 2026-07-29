"""
mycv.color
==========
Photometric and colour-space transformations.

Functions
---------
histogram_equalize : Contrast enhancement via CDF remapping
rgb_to_hsv         : Perceptual colour-space conversion (RGB cube -> HSV cylinder)
"""

import numpy as np


def histogram_equalize(image: np.ndarray) -> np.ndarray:
    """
    Enhance contrast of a grayscale image using histogram equalisation.

    Algorithm
    ---------
    1. Compute the histogram h[k] — counts per intensity level k in [0, 255].
    2. Normalise to the PDF:  p(k) = h[k] / N   (N = total pixels)
    3. Compute the CDF:       cdf(k) = sum_{j=0}^{k} p(j)
    4. Build lookup table:    T(k) = round(255 * cdf(k))
    5. Apply via fancy index: output = T[image]

    The mapping T stretches the CDF toward the identity line, spreading
    intensities to fill [0, 255] as uniformly as the discrete histogram allows.

    Parameters
    ----------
    image : np.ndarray  shape (H, W), dtype uint8

    Returns
    -------
    np.ndarray  shape (H, W), dtype uint8
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image, got shape {image.shape}.")

    # Step 1-2: histogram -> PDF
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 255))
    pdf = hist / hist.sum()

    # Step 3: prefix-sum -> CDF
    cdf = np.cumsum(pdf)

    # Step 4: build the 256-entry lookup table
    lut = np.round(cdf * 255).astype(np.uint8)

    # Step 5: apply lookup table — pure fancy indexing, zero loops
    return lut[image]


def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to the HSV colour space.

    Geometry
    --------
    RGB occupies a unit cube [0,1]^3. HSV maps it to a cylinder:
        V (Value)      = max(R,G,B)           — height of the cylinder
        S (Saturation) = Delta / V             — radius (0=grey axis, 1=surface)
        H (Hue)        = piecewise angle in [0, 360)  — angular position

    Parameters
    ----------
    image : np.ndarray  shape (H, W, 3), dtype uint8, values in [0, 255]

    Returns
    -------
    np.ndarray  shape (H, W, 3), dtype float32
        Channel 0: H in [0, 360)
        Channel 1: S in [0, 1]
        Channel 2: V in [0, 1]
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB image, got {image.shape}.")

    img = image.astype(np.float32) / 255.0
    R, G, B = img[..., 0], img[..., 1], img[..., 2]

    # V and Delta
    V    = np.max(img, axis=-1)
    Cmin = np.min(img, axis=-1)
    delta = V - Cmin

    # S: avoid division by zero on pure-black pixels (V == 0)
    S = np.where(V == 0, 0.0, delta / V)

    # H: piecewise across the three cube faces
    # Replace delta==0 with 1.0 to avoid NaN; result is masked out afterward
    d_safe = np.where(delta == 0, 1.0, delta)

    H_R = (60.0 * ((G - B) / d_safe)) % 360.0
    H_G =  60.0 * ((B - R) / d_safe + 2.0)
    H_B =  60.0 * ((R - G) / d_safe + 4.0)

    max_is_R = (V == R)
    max_is_G = (V == G) & ~max_is_R
    max_is_B = ~max_is_R & ~max_is_G

    H = (
        np.where(max_is_R, H_R, 0.0) +
        np.where(max_is_G, H_G, 0.0) +
        np.where(max_is_B, H_B, 0.0)
    )
    # Achromatic pixels (delta == 0) have undefined hue; set to 0 by convention
    H = np.where(delta == 0, 0.0, H)

    return np.stack([H, S, V], axis=-1).astype(np.float32)
