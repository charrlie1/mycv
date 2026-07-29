# mycv — A Pure-NumPy Computer Vision Library

![Status](https://img.shields.io/badge/status-beta-orange)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![NumPy](https://img.shields.io/badge/numpy-1.20+-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> **mycv** is a from-scratch, dependency-free computer vision library.
> Every algorithm; from colour-space geometry to template matching, is
> implemented directly in vectorised NumPy. There are no calls to OpenCV,
> scikit-image, or any compiled vision backend inside the mathematical core.

The purpose of `mycv` is not to outperform OpenCV. Its purpose is to make
every operation **transparent**: each function is a readable, line-by-line
translation of a closed-form mathematical definition, so that a student or
researcher can read the code and understand *why* it works, not merely that
it works.

---

## Philosophy

Most computer-vision code today is a thin wrapper around opaque C++ binaries.
You call `cv2.cvtColor(...)` and a black box returns an answer. `mycv` takes
the opposite stance:

- **No black boxes.** Every transform is plain NumPy — broadcasting,
  `np.einsum`, and zero-copy `np.lib.stride_tricks.as_strided` patch tensors.
- **Math first.** Each module corresponds to a section of the companion
  monograph [`mycv_math_v4.pdf`](mycv_math_v4.pdf), which derives every
  formula from first principles (luminance weighting, CDF remapping, the
  RGB→HSV cylinder map, discrete convolution, morphological set theory,
  Pearson-correlation NCC, Nyquist–Shannon anti-aliasing, and IoU geometry).
- **Type-safe arithmetic.** Inputs are upcast before any arithmetic and
  clipped back to `uint8` only at function boundaries. The notorious `uint8`
  underflow in frame differencing is resolved by casting to `int16` *before*
  subtraction.
- **Hardware-agnostic core.** The library consumes and produces NumPy arrays
  only. Camera capture and windowed display are handled by a separate I/O
  layer (`live_demo.py`), keeping the mathematics fully decoupled from the
  hardware.

---

## Repository Layout

```
MYCV_PROJECT_V4/
├── mycv/                 # The pure-NumPy mathematical engine
│   ├── core.py           #   Grayscale (BT.601), binary thresholding
│   ├── color.py          #   Histogram equalisation, RGB → HSV
│   ├── filters.py        #   2-D convolution, Sobel edge detection
│   ├── morphology.py     #   Dilation, erosion, opening, closing (set theory)
│   ├── geometry.py       #   Centre-rotation, bilinear interpolation, homography
│   ├── features.py       #   Harris corner response, Hough line transform
│   ├── detection.py      #   NCC template matching, Gaussian pyramid, NMS
│   ├── tracking.py       #   Motion mask, colour mask, centroid, ring-buffer smoother
│   └── __init__.py
├── live_demo.py          # Real-time I/O + visualisation (pygame / PyAV)
├── main.py               # Batch demonstration over a synthetic sequence
├── mycv_math_v4.pdf      # Mathematical monograph (the blueprint)
├── outputs/              # Generated result images
└── README.md
```

---

## Installation

The mathematical core requires **only NumPy**.

```bash
git clone https://github.com/charrlie1/mycv.git
cd mycv
python -m pip install numpy
```

Run any script from the project root and the package imports directly:

```bash
python main.py
```

### Optional: real-time demo dependencies

The live camera / video demo needs a display and I/O backend, but these are
**never** used for image processing — only for capturing frames and drawing
the window:

```bash
python -m pip install pygame av
```

### Optional: editable install

If you add a `pyproject.toml` later, you may install the package in editable
mode so that `import mycv` works from any directory:

```bash
python -m pip install -e .
```

---

## Quick Start

### Colour-based tracking

```python
import numpy as np
from mycv.color import rgb_to_hsv
from mycv.tracking import color_mask, calculate_centroid

frame = ...  # np.ndarray of shape (H, W, 3), dtype uint8

hsv  = rgb_to_hsv(frame)
mask = color_mask(
    hsv,
    lower_bound=np.array([70.0, 0.25, 0.25], dtype=np.float32),  # green
    upper_bound=np.array([170.0, 1.0, 1.0],  dtype=np.float32),
)
cx, cy = calculate_centroid(mask)   # (-1, -1) if the mask is empty
```

### Motion detection between two frames

```python
from mycv.core import rgb_to_grayscale
from mycv.tracking import compute_motion_mask

gray_now  = rgb_to_grayscale(frame_now)
gray_prev = rgb_to_grayscale(frame_prev)

motion = compute_motion_mask(gray_now, gray_prev, threshold=30)
```

### Template matching with non-maximum suppression

```python
from mycv.detection import match_template_ncc, find_template_matches

ncc = match_template_ncc(gray_image, template)
boxes, scores = find_template_matches(
    ncc_map=ncc,
    template_shape=template.shape,
    threshold=0.8,
    nms_iou=0.3,
)
```

---

## Module Reference

| Module | Domain | Key functions |
|---|---|---|
| `core.py` | Pointwise linear algebra | `rgb_to_grayscale`, `threshold` |
| `color.py` | Probability & colour geometry | `histogram_equalize`, `rgb_to_hsv` |
| `filters.py` | Discrete convolution | `convolve2d`, Sobel gradients |
| `morphology.py` | Set theory / lattices | `dilate`, `erode`, `opening`, `closing` |
| `geometry.py` | Affine & projective geometry | Centre-rotation, bilinear warp, homography |
| `features.py` | Accumulator geometry | Harris corner response, Hough lines |
| `detection.py` | Correlation & sampling theory | `match_template_ncc`, `gaussian_pyramid`, `non_max_suppression`, `find_template_matches` |
| `tracking.py` | Unsigned arithmetic & ring buffers | `compute_motion_mask`, `color_mask`, `calculate_centroid`, `TemporalSmoother` |

---

## Real-Time Demo

`live_demo.py` is the hardware harness. It acquires frames (camera, video
file, or a synthetic source), hands each one to `mycv` as a NumPy array, and
renders the result. **All tracking and detection math is performed by `mycv`;
pygame only captures pixels and draws the window.**

```bash
# Verify the pipeline with no hardware required
python live_demo.py --source synthetic --mode color

# Live webcam
python live_demo.py --source camera --mode color

# A recorded video file
python live_demo.py --source path/to/video.mp4 --mode template

# Headless run (e.g. over SSH, no display)
python live_demo.py --source synthetic --headless
```

### Supported `--source` values

| Value | Meaning |
|---|---|
| `synthetic` | Generated moving square (testing) |
| `camera` | First webcam via `pygame.camera` |
| `0`, `1`, … | Specific webcam index |
| `video.mp4` | Decoded via PyAV / FFmpeg |
| `dshow:video=NAME` | Windows DirectShow device |
| `v4l2:/dev/video0` | Linux V4L2 device |
| `avfoundation:0:1` | macOS AVFoundation device |

### Live keyboard controls

| Key | Action |
|---|---|
| `c` / `m` / `t` | Switch to colour / motion / template mode |
| `s` | Sample the centre colour into the HSV range |
| `g` `r` `b` `y` | Colour presets (red handles hue wrap-around) |
| `=` / `-` | Raise / lower the NCC detection threshold |
| `[` / `]` | Lower / raise the motion threshold |
| `←` / `→` | Shrink / grow the captured template |
| `o` / `p` | Toggle mask overlay / morphological cleanup |
| `h` | Print help · `q` / `Esc` quit |

> **Performance note.** Normalised cross-correlation builds a patch tensor of
> shape `(H−kH+1, W−kW+1, kH, kW)`, so memory grows quickly with resolution.
> The demo therefore downscales frames to `--max-dim` (default `180`) before
> detection. Lower it for speed, raise it cautiously for small targets, or use
> `--detect-every N` to run NCC on every *N*-th frame.

---

## Mathematical Documentation

The complete derivations live in [`mycv_math_v4.pdf`](mycv_math_v4.pdf):

| § | Topic | Representative result |
|---|---|---|
| 2 | Luminance & thresholding | $Y = 0.2989R + 0.5870G + 0.1140B$ |
| 3 | Histogram equalisation & HSV | $T(k) = \mathrm{round}(255 \cdot \mathrm{CDF}(k))$ |
| 4 | Discrete convolution & Sobel | $(I * K)[x,y] = \sum_{i,j} I[x-i,y-j]\,K[i,j]$ |
| 5 | Morphology as set theory | $F \oplus B$ (ANY), $F \ominus B$ (ALL) |
| 6 | Projective warping | $x_s = \tilde{x}/\tilde{w}$ |
| 7 | Harris & Hough | $R = \det M - k(\mathrm{tr}\,M)^2$ |
| 8 | NCC, pyramid, NMS | $\mathrm{NCC} = \mathrm{cov}(P,T)/(\sigma_P\sigma_T)$ |
| 9 | Tracking & ring buffers | $D = \|I_t^{\mathrm{int16}} - I_{t-1}^{\mathrm{int16}}\|$ |

---

## What `mycv` Is *Not*

- It is **not** a performance-optimised production detector. Where clarity
  and correctness conflict with raw speed, clarity wins. For throughput on
  4K video, a compiled backend (or the planned CuPy path) is appropriate.
- It is **not** a learned model. Template matching is classical correlation,
  not a neural network; it suits rigid, known-appearance targets.

---

## Roadmap

- [ ] Connected-component labelling for multi-object identity tracking
- [ ] Kalman filtering for predictive centroid smoothing
- [ ] Optional CuPy backend (`pip install mycv[gpu]`)
- [ ] Unit-test suite (`pytest`) with coverage reporting
- [ ] Sphinx documentation site generated from docstrings

---

## License

Distributed under the MIT License. See `LICENSE`.

---

## Author

**Abodunrin Charles Toluwanimi**
Department of Electrical and Electronic Engineering
Obafemi Awolowo University, Ile-Ife

Built as a study in implementing computer vision from its mathematical
foundations, with no reliance on pre-compiled vision libraries.
