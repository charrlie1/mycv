"""
mycv — Custom Computer Vision Library  v4.0.0
=============================================
A pure-NumPy image processing and computer vision library built from
mathematical first principles. No OpenCV. No SciPy. No skimage.

Modules
-------
core      : Point ops        — rgb_to_grayscale, threshold
color     : Photometrics     — histogram_equalize, rgb_to_hsv
filters   : Spatial filters  — convolve2d, sobel_edge_detection
morphology: Binary set ops   — dilate, erode, opening, closing
geometry  : Transformations  — rotate_image, warp_perspective, bilinear_interpolate
features  : Feature extract  — harris_corner_response, hough_line_transform
detection : Detection        — match_template_ncc, gaussian_pyramid, non_max_suppression
tracking  : Motion & tracking — compute_motion_mask, color_mask,
                                calculate_centroid, TemporalSmoother
"""

from .core       import rgb_to_grayscale, threshold
from .color      import histogram_equalize, rgb_to_hsv
from .filters    import convolve2d, sobel_edge_detection, SOBEL_X, SOBEL_Y
from .morphology import dilate, erode, opening, closing
from .geometry   import bilinear_interpolate, rotate_image, warp_perspective
from .features   import harris_corner_response, hough_line_transform
from .detection  import (
    match_template_ncc, find_template_matches,
    gaussian_pyramid, non_max_suppression,
)
from .tracking   import (
    compute_motion_mask,
    color_mask, calculate_centroid,
    TemporalSmoother,
)

__version__ = "4.0.0"
__author__  = "Tolu"

__all__ = [
    "rgb_to_grayscale", "threshold",
    "histogram_equalize", "rgb_to_hsv",
    "convolve2d", "sobel_edge_detection", "SOBEL_X", "SOBEL_Y",
    "dilate", "erode", "opening", "closing",
    "bilinear_interpolate", "rotate_image", "warp_perspective",
    "harris_corner_response", "hough_line_transform",
    "match_template_ncc", "find_template_matches",
    "gaussian_pyramid", "non_max_suppression",
    "compute_motion_mask",
    "color_mask", "calculate_centroid",
    "TemporalSmoother",
]
