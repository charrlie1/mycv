"""
mycv — Custom Computer Vision Library  v4.1.0
=============================================
A pure-NumPy image processing and computer vision library built from
mathematical first principles. No OpenCV. No SciPy. No skimage.

Modules
-------
core        : Point ops         — rgb_to_grayscale, threshold
color       : Photometrics      — histogram_equalize, rgb_to_hsv
filters     : Spatial filters   — convolve2d, sobel_edge_detection
morphology  : Binary set ops    — dilate, erode, opening, closing,
                                   grayscale_dilate, connected components
geometry    : Transformations   — rotate_image, warp_perspective, bilinear_interpolate
features    : Feature extract   — harris_corner_response, detect_harris_corners,
                                   hough_line_transform, count_hough_lines,
                                   extract_object_metrics, convex_hull,
                                   draw_bounding_box, classify_object
detection   : Detection         — match_template_ncc, match_template_ncc_multiscale,
                                   gaussian_pyramid, non_max_suppression
tracking    : Motion & tracking — compute_motion_mask, color_mask, color_mask_hue_wrap,
                                   calculate_centroid, TemporalSmoother,
                                   KalmanCentroidTracker, MultiObjectKalmanTracker
calibration : Projective geom.  — solve_homography_dlt, ransac_homography
streaming   : Live video (opt.) — StreamReader (requires the optional `av` package)

Changelog (v4.0.0 -> v4.1.0)
------------------------------
Corrections:
  - features.extract_object_metrics: bbox width/height are now correctly
    inclusive (+1); the exported bbox convention is now EXCLUSIVE
    (y1, x1, y2, x2) and standardised across features/detection/morphology,
    so `image[y1:y2, x1:x2]` always crops the box directly.
  - features.extract_object_metrics: now returns true `pixel_area`
    (foreground pixel count) distinct from `bbox_area` (rectangle area).
  - features.classify_object: now accepts an optional `mask` and computes
    mean colour from object pixels only, not the whole bounding box.
  - features.draw_bounding_box: rewritten for the new exclusive convention.
  - tracking.KalmanCentroidTracker.predict(): now COMMITS the predicted
    state, so repeated calls during occlusion correctly propagate the
    object forward instead of re-predicting from stale state.
  - tracking.kalman_filter_update: uses np.linalg.solve instead of an
    explicit matrix inverse, and the numerically robust Joseph-form
    covariance update by default.
  - tracking.KalmanCentroidTracker: supports real per-frame `dt`.

New capabilities:
  - True pixel area, extent, second moments, orientation, eccentricity,
    perimeter, circularity, and solidity (via a new monotone-chain
    `convex_hull`) in `extract_object_metrics`.
  - `detect_harris_corners` (thresholded, NMS'd corner points) and
    `count_hough_lines` (peak-clustered line count).
  - `morphology.label_connected_components` + `component_properties`
    for robust multi-object colour tracking.
  - `tracking.MultiObjectKalmanTracker` for tracking several objects
    with persistent IDs.
  - Kalman measurement gating (`mahalanobis_gate`) and adaptive
    measurement noise (`confidence` argument to `update`).
  - `detection.match_template_ncc_multiscale` for scale-invariant
    template matching via the Gaussian pyramid.
  - `detection.match_template_ncc` now guards against unbounded memory
    use on large frames/templates.
  - `tracking.color_mask_hue_wrap` automatically handles hue wrap-around
    (e.g. red spanning 0/360 degrees).
  - New `mycv.calibration` module: normalised DLT homography estimation
    and RANSAC for outlier-robust homography fitting.
  - New optional `mycv.streaming` module for RTSP/HTTP/UDP video
    ingestion with reconnection and frame-dropping (requires `av`).
"""

from .core       import rgb_to_grayscale, threshold
from .color      import histogram_equalize, rgb_to_hsv
from .filters    import convolve2d, sobel_edge_detection, SOBEL_X, SOBEL_Y
from .morphology import (
    dilate, erode, opening, closing,
    grayscale_dilate,
    label_connected_components, component_properties,
)
from .geometry   import bilinear_interpolate, rotate_image, warp_perspective
from .features   import (
    harris_corner_response, detect_harris_corners,
    hough_line_transform, count_hough_lines,
    extract_object_metrics, convex_hull,
    draw_bounding_box, classify_object,
)
from .detection  import (
    match_template_ncc, find_template_matches, match_template_ncc_multiscale,
    gaussian_pyramid, non_max_suppression,
)
from .tracking   import (
    compute_motion_mask,
    color_mask, color_mask_hue_wrap, calculate_centroid,
    TemporalSmoother,
    kalman_filter_predict, kalman_filter_update, mahalanobis_gate,
    KalmanCentroidTracker, MultiObjectKalmanTracker,
)
from .calibration import (
    normalize_points, solve_homography_dlt, reprojection_error, ransac_homography,
)
"""# .streaming is intentionally NOT imported here: it depends on the
# optional `av` package and importing mycv should never require it.
# Access it explicitly: `from mycv.streaming import StreamReader`."""

__version__ = "4.1.0"
__author__  = "Toluwanimicharles"

version = __version__
author = __author__

__all__ = [
    "rgb_to_grayscale", "threshold",
    "histogram_equalize", "rgb_to_hsv",
    "convolve2d", "sobel_edge_detection", "SOBEL_X", "SOBEL_Y",
    "dilate", "erode", "opening", "closing",
    "grayscale_dilate", "label_connected_components", "component_properties",
    "bilinear_interpolate", "rotate_image", "warp_perspective",
    "harris_corner_response", "detect_harris_corners",
    "hough_line_transform", "count_hough_lines",
    "extract_object_metrics", "convex_hull",
    "draw_bounding_box", "classify_object",
    "match_template_ncc", "find_template_matches", "match_template_ncc_multiscale",
    "gaussian_pyramid", "non_max_suppression",
    "compute_motion_mask",
    "color_mask", "color_mask_hue_wrap", "calculate_centroid",
    "TemporalSmoother",
    "kalman_filter_predict", "kalman_filter_update", "mahalanobis_gate",
    "KalmanCentroidTracker", "MultiObjectKalmanTracker",
    "normalize_points", "solve_homography_dlt", "reprojection_error", "ransac_homography",
]
