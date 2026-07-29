# mycv v4.0.0 — Custom Computer Vision Library

Pure-NumPy image processing and computer vision. No OpenCV. No SciPy. No skimage.

## Structure
```
mycv_project_v4/
├── mycv/
│   ├── __init__.py       <- Public API v4
│   ├── core.py           <- rgb_to_grayscale, threshold
│   ├── color.py          <- histogram_equalize, rgb_to_hsv
│   ├── filters.py        <- convolve2d, sobel_edge_detection
│   ├── morphology.py     <- dilate, erode, opening, closing
│   ├── geometry.py       <- rotate_image, warp_perspective, bilinear_interpolate
│   ├── features.py       <- harris_corner_response, hough_line_transform
│   ├── detection.py      <- match_template_ncc, gaussian_pyramid, non_max_suppression
│   └── tracking.py       <- compute_motion_mask, color_mask,
│                            calculate_centroid, TemporalSmoother
├── main.py               <- 21-step end-to-end pipeline
├── mycv_math_v4.pdf      <- Full mathematical monograph
└── README.md
```

## Quickstart
```bash
pip install numpy pillow
# Place any .jpg as test_image.jpg, then:
python main.py
```

## v4 API (tracking module)
```python
import mycv, numpy as np

# Motion detection
mask = mycv.compute_motion_mask(frame_t, frame_t_minus_1, threshold=30)

# Colour tracking
hsv      = mycv.rgb_to_hsv(rgb)
cmask    = mycv.color_mask(hsv, lower=[0,0.3,0.3], upper=[60,1,1])
cx, cy   = mycv.calculate_centroid(cmask)   # (-1,-1) if no object

# Temporal smoothing
smoother = mycv.TemporalSmoother(n_frames=10, height=H, width=W, channels=3)
smooth   = smoother.update(new_frame)       # uint8, rolling N-frame average
```

## Author
Built by **Tolu** — Obafemi Awolowo University, EEE Department.
