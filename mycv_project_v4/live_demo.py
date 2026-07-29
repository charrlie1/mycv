#!/usr/bin/env python3
"""
live_demo.py

Real-time demonstration driver for the mycv pure-NumPy computer-vision library.

This script provides a live video I/O layer and interactive visualisation for:

    - colour-based tracking
    - frame-difference motion tracking
    - normalised cross-correlation template detection

It intentionally avoids OpenCV for all mathematical image processing.

Supported sources
-----------------

    --source synthetic
        Generated moving square. Useful for testing.

    --source camera
        First available webcam through pygame.camera.

    --source 0, --source 1, ...
        Specific webcam index through pygame.camera.

    --source video.mp4
        Video file through PyAV.

    --source "dshow:video=Integrated Camera"
        Windows DirectShow device through PyAV / FFmpeg.

    --source "v4l2:/dev/video0"
        Linux V4L2 device through PyAV / FFmpeg.

    --source "avfoundation:0:1"
        macOS AVFoundation device through PyAV / FFmpeg.

Controls
--------

    q / Esc     quit
    c           colour-tracking mode
    m           motion-tracking mode
    t           capture centre template and enter template-detection mode
    d           enter template-detection mode
    s           sample colour from centre pixel region
    g           green colour preset
    r           red colour preset, with hue wrap-around
    b           blue colour preset
    y           yellow colour preset
    o           toggle mask overlay
    p           toggle morphological cleanup
    = / +       increase detection threshold
    -           decrease detection threshold
    [           decrease motion threshold
    ]           increase motion threshold
    left        decrease template size
    right       increase template size
    h           print help
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


from mycv.core import rgb_to_grayscale
from mycv.color import rgb_to_hsv
from mycv.morphology import opening, closing
from mycv.tracking import (
    compute_motion_mask,
    color_mask,
    calculate_centroid,
    TemporalSmoother,
)
from mycv.detection import (
    match_template_ncc,
    find_template_matches,
)


try:
    import pygame
    import pygame.camera

    PYGAME_AVAILABLE = True
    PYGAME_ERROR = None
except Exception as exc:
    pygame = None
    PYGAME_AVAILABLE = False
    PYGAME_ERROR = exc


try:
    import av

    PYAV_AVAILABLE = True
    PYAV_ERROR = None
except Exception as exc:
    av = None
    PYAV_AVAILABLE = False
    PYAV_ERROR = exc


HELP_TEXT = """
mycv live demo
==============

Modes:
    c       colour tracking
    m       motion tracking
    t       capture centre template and switch to template detection
    d       switch to template detection

Colour controls:
    s       sample colour from centre region
    g       green preset
    r       red preset, handles hue wrap-around
    b       blue preset
    y       yellow preset

Detection / motion controls:
    = / +   increase detection threshold
    -       decrease detection threshold
    [       decrease motion threshold
    ]       increase motion threshold
    left    decrease template size
    right   increase template size

Display controls:
    o       toggle mask overlay
    p       toggle morphological cleanup
    h       print help
    q/Esc   quit

Example source strings:
    --source synthetic
    --source camera
    --source 0
    --source video.mp4
    --source "dshow:video=Integrated Camera"
    --source "v4l2:/dev/video0"
    --source "avfoundation:0:1"
"""


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------
def resize_nearest(image: np.ndarray, max_dim: int) -> np.ndarray:
    """
    Downscale an image using nearest-neighbour sampling.

    This is intentionally simple and dependency-free. It is used to keep
    real-time template matching tractable, because full-resolution NCC is
    memory-intensive and computationally expensive.
    """

    if max_dim is None or max_dim < 8:
        return image

    H, W = image.shape[:2]

    if max(H, W) <= max_dim:
        return image

    scale = float(max_dim) / float(max(H, W))

    new_H = max(1, int(round(H * scale)))
    new_W = max(1, int(round(W * scale)))

    rows = np.linspace(0, H - 1, new_H).astype(np.int64)
    cols = np.linspace(0, W - 1, new_W).astype(np.int64)

    return image[rows[:, None], cols[None, :]]


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """
    Normalise an incoming frame to uint8 RGB.

    Handles:
        - grayscale images,
        - floating-point images in [0, 1],
        - numeric arrays needing clipping.
    """

    img = np.asarray(image)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected RGB or grayscale image, got shape {img.shape}.")

    if img.dtype == np.uint8:
        return img

    if np.issubdtype(img.dtype, np.floating):
        finite = img[np.isfinite(img)]
        if finite.size > 0 and float(finite.max()) <= 1.0:
            img = img * 255.0

    return np.clip(img, 0, 255).astype(np.uint8)


def hsv_mask_wrapped(
    hsv_image: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> np.ndarray:
    """
    HSV range mask with support for hue wrap-around.

    mycv.tracking.color_mask performs a single axis-aligned HSV box query.
    Red-like colours often require two intervals because hue is periodic:

        [0, 20] union [340, 360]

    This helper splits the query automatically when necessary.
    """

    lower = np.asarray(lower_bound, dtype=np.float32).copy()
    upper = np.asarray(upper_bound, dtype=np.float32).copy()

    # Clamp saturation and value to their valid physical range.
    for channel in (1, 2):
        lower[channel] = float(np.clip(lower[channel], 0.0, 1.0))
        upper[channel] = float(np.clip(upper[channel], 0.0, 1.0))

    # Bring hue bounds into a more manageable range.
    while upper[0] < 0.0:
        lower[0] += 360.0
        upper[0] += 360.0

    while lower[0] > 360.0:
        lower[0] -= 360.0
        upper[0] -= 360.0

    # Explicit inverted interval, e.g. [330, 30].
    if lower[0] > upper[0]:
        mask_a = color_mask(
            hsv_image,
            np.array([lower[0], lower[1], lower[2]], dtype=np.float32),
            np.array([360.0, upper[1], upper[2]], dtype=np.float32),
        )
        mask_b = color_mask(
            hsv_image,
            np.array([0.0, lower[1], lower[2]], dtype=np.float32),
            np.array([upper[0], upper[1], upper[2]], dtype=np.float32),
        )
        return np.maximum(mask_a, mask_b)

    # Lower hue below zero, e.g. [-15, 15].
    if lower[0] < 0.0:
        mask_a = color_mask(
            hsv_image,
            np.array([0.0, lower[1], lower[2]], dtype=np.float32),
            np.array([upper[0], upper[1], upper[2]], dtype=np.float32),
        )
        mask_b = color_mask(
            hsv_image,
            np.array([lower[0] + 360.0, lower[1], lower[2]], dtype=np.float32),
            np.array([360.0, upper[1], upper[2]], dtype=np.float32),
        )
        return np.maximum(mask_a, mask_b)

    # Upper hue above 360, e.g. [350, 370].
    if upper[0] > 360.0:
        mask_a = color_mask(
            hsv_image,
            np.array([lower[0], lower[1], lower[2]], dtype=np.float32),
            np.array([360.0, upper[1], upper[2]], dtype=np.float32),
        )
        mask_b = color_mask(
            hsv_image,
            np.array([0.0, lower[1], lower[2]], dtype=np.float32),
            np.array([upper[0] - 360.0, upper[1], upper[2]], dtype=np.float32),
        )
        return np.maximum(mask_a, mask_b)

    return color_mask(hsv_image, lower, upper)


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------
class SyntheticSource:
    """
    Synthetic moving-square source.

    This is useful for verifying that the full pipeline works without
    requiring camera permissions, a display camera backend, or a video file.
    """

    def __init__(self, width: int = 320, height: int = 240, fps: int = 30) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.t = 0
        self.rng = np.random.default_rng(0)

    def read(self) -> np.ndarray:
        H = self.height
        W = self.width

        img = np.zeros((H, W, 3), dtype=np.uint8)

        # Dark background.
        img[..., 0] = 18
        img[..., 1] = 20
        img[..., 2] = 26

        square = max(12, min(H, W) // 5)
        y = H // 2 - square // 2

        max_x = max(1, W - square)
        x = int((self.t * 3) % max_x)

        # Green square.
        img[y:y + square, x:x + square] = (55, 220, 85)

        # Mild noise.
        noise = self.rng.integers(-6, 7, size=img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        self.t += 1

        return img

    def close(self) -> None:
        pass


class PygameCameraSource:
    """
    Webcam source using pygame.camera.

    This is the simplest pure-Python-friendly webcam route for many desktop
    systems, but it is platform-dependent. If it fails, use a PyAV device
    source such as dshow, v4l2, or avfoundation.
    """

    def __init__(self, device: int | str = 0, width: int = 320, height: int = 240, fps: int = 30) -> None:
        if not PYGAME_AVAILABLE:
            raise RuntimeError(
                f"pygame is not available. Install pygame-ce or pygame. Details: {PYGAME_ERROR}"
            )

        pygame.camera.init()

        cameras = pygame.camera.list_cameras()
        if not cameras:
            raise RuntimeError(
                "pygame.camera.list_cameras() found no cameras. "
                "Check camera privacy settings, permissions, and device index."
            )

        if isinstance(device, int):
            if 0 <= device < len(cameras):
                device_path = cameras[device]
            else:
                device_path = cameras[0]
        else:
            device_path = device

        self.device = device_path
        self.fps = float(fps)

        try:
            self.cam = pygame.camera.Camera(device_path, (int(width), int(height)), "RGB")
        except Exception:
            # Fall back to the camera's default size and colourspace.
            self.cam = pygame.camera.Camera(device_path)

        self.cam.start()

        self.width = int(width)
        self.height = int(height)

    def read(self) -> np.ndarray | None:
        try:
            surface = self.cam.get_image()
            w, h = surface.get_size()
            data = pygame.image.tostring(surface, "RGB")
            frame = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3).copy()

            self.width = w
            self.height = h

            return frame

        except Exception:
            return None

    def close(self) -> None:
        try:
            self.cam.stop()
        except Exception:
            pass

        try:
            pygame.camera.quit()
        except Exception:
            pass


class PyAVVideoSource:
    """
    Video-file source using PyAV.

    This decodes compressed video through FFmpeg and returns RGB NumPy arrays.
    """

    def __init__(self, path: str | Path) -> None:
        if not PYAV_AVAILABLE:
            raise RuntimeError(
                f"PyAV is not available. Install it with: python -m pip install av. Details: {PYAV_ERROR}"
            )

        self.path = str(path)
        self._open()

    def _open(self) -> None:
        self.container = av.open(self.path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"

        self._decoder = self.container.decode(video=0)

        self.width = int(self.stream.codec_context.width)
        self.height = int(self.stream.codec_context.height)

        rate = self.stream.average_rate
        self.fps = float(rate) if rate is not None else 30.0

    def read(self) -> np.ndarray | None:
        while True:
            try:
                frame = next(self._decoder)

            except StopIteration:
                # Loop the video for demonstration purposes.
                self.container.close()
                self._open()
                continue

            except Exception:
                return None

            if frame is None:
                continue

            rgb = frame.to_ndarray(format="rgb24")
            self.height, self.width = rgb.shape[:2]

            return rgb

    def close(self) -> None:
        try:
            self.container.close()
        except Exception:
            pass


class PyAVDeviceSource:
    """
    Live device source using PyAV / FFmpeg input devices.

    Examples:
        Windows:
            dshow:video=Integrated Camera

        Linux:
            v4l2:/dev/video0

        macOS:
            avfoundation:0:1
    """

    def __init__(
        self,
        fmt: str,
        device: str,
        width: int = 320,
        height: int = 240,
        fps: int = 30,
    ) -> None:
        if not PYAV_AVAILABLE:
            raise RuntimeError(
                f"PyAV is not available. Install it with: python -m pip install av. Details: {PYAV_ERROR}"
            )

        self.fmt = fmt
        self.device = device
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)

        options = {
            "framerate": str(int(fps)),
            "video_size": f"{int(width)}x{int(height)}",
        }

        if fmt == "dshow" and not device.startswith(("video=", "audio=")):
            device = f"video={device}"

        try:
            self.container = av.open(device, format=fmt, options=options)
        except Exception:
            # Some devices reject the requested options; retry with defaults.
            self.container = av.open(device, format=fmt)

        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"

        self._decoder = self.container.decode(video=0)

    def read(self) -> np.ndarray | None:
        try:
            frame = next(self._decoder)
        except Exception:
            return None

        if frame is None:
            return None

        rgb = frame.to_ndarray(format="rgb24")
        self.height, self.width = rgb.shape[:2]

        return rgb

    def close(self) -> None:
        try:
            self.container.close()
        except Exception:
            pass


def make_source(source: str, width: int, height: int, fps: int):
    """
    Construct the requested frame source.
    """

    source = str(source)

    if source.lower() == "synthetic":
        return SyntheticSource(width=width, height=height, fps=fps)

    if source.lower() == "camera":
        return PygameCameraSource(device=0, width=width, height=height, fps=fps)

    if source.isdigit():
        return PygameCameraSource(device=int(source), width=width, height=height, fps=fps)

    path = Path(source).expanduser()
    if path.exists():
        return PyAVVideoSource(path)

    if ":" in source:
        fmt, _, device = source.partition(":")
        fmt = fmt.lower()

        if fmt in {"dshow", "v4l2", "avfoundation", "gdigrab"}:
            return PyAVDeviceSource(
                fmt=fmt,
                device=device,
                width=width,
                height=height,
                fps=fps,
            )

    raise ValueError(
        f"Unknown source '{source}'. Use synthetic, camera, an integer camera index, "
        "a video-file path, or an FFmpeg device string such as "
        "'dshow:video=Integrated Camera'."
    )


# ---------------------------------------------------------------------------
# Vision pipeline
# ---------------------------------------------------------------------------
class VisionPipeline:
    """
    Unified tracking and detection pipeline.

    Modes:
        colour      HSV mask, morphology, centroid
        motion      frame differencing, morphology, centroid
        template    NCC template matching, NMS, bounding boxes
    """

    def __init__(
        self,
        mode: str = "color",
        max_dim: int = 180,
        smooth_frames: int = 0,
        template_size: int = 28,
        detection_threshold: float = 0.55,
        motion_threshold: int = 25,
        detect_every: int = 1,
        use_morphology: bool = True,
    ) -> None:
        self.mode = mode
        self.max_dim = int(max_dim)
        self.smooth_frames = int(smooth_frames)
        self.template_size = int(template_size)
        self.detection_threshold = float(detection_threshold)
        self.motion_threshold = int(motion_threshold)
        self.detect_every = max(1, int(detect_every))
        self.use_morphology = bool(use_morphology)

        # Default green HSV bounds.
        self.lower_bound = np.array([70.0, 0.25, 0.25], dtype=np.float32)
        self.upper_bound = np.array([170.0, 1.0, 1.0], dtype=np.float32)

        self.smoother: TemporalSmoother | None = None
        self.prev_gray: np.ndarray | None = None
        self.template: np.ndarray | None = None

        self.frame_count = 0

        self.display_rgb: np.ndarray | None = None
        self.gray: np.ndarray | None = None
        self.mask: np.ndarray | None = None

        self.centroid: tuple[float, float] = (-1.0, -1.0)
        self.boxes = np.empty((0, 4), dtype=np.float64)
        self.scores = np.empty((0,), dtype=np.float64)

        self.last_error = ""

    def _ensure_smoother(self, rgb: np.ndarray) -> None:
        if self.smooth_frames <= 1:
            self.smoother = None
            return

        H, W, C = rgb.shape

        if (
            self.smoother is None
            or self.smoother.H != H
            or self.smoother.W != W
            or self.smoother.C != C
        ):
            self.smoother = TemporalSmoother(
                n_frames=self.smooth_frames,
                height=H,
                width=W,
                channels=C,
            )

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        if not self.use_morphology:
            return mask

        return closing(opening(mask))

    def set_color_preset(self, name: str) -> None:
        name = name.lower()

        if name == "green":
            self.lower_bound = np.array([70.0, 0.25, 0.25], dtype=np.float32)
            self.upper_bound = np.array([170.0, 1.0, 1.0], dtype=np.float32)

        elif name == "red":
            # Negative hue is handled by hsv_mask_wrapped.
            self.lower_bound = np.array([-20.0, 0.30, 0.20], dtype=np.float32)
            self.upper_bound = np.array([20.0, 1.0, 1.0], dtype=np.float32)

        elif name == "blue":
            self.lower_bound = np.array([190.0, 0.30, 0.20], dtype=np.float32)
            self.upper_bound = np.array([260.0, 1.0, 1.0], dtype=np.float32)

        elif name == "yellow":
            self.lower_bound = np.array([35.0, 0.30, 0.30], dtype=np.float32)
            self.upper_bound = np.array([85.0, 1.0, 1.0], dtype=np.float32)

        else:
            raise ValueError(f"Unknown colour preset: {name}")

    def sample_color_from_center(
        self,
        rgb: np.ndarray | None = None,
        tol_h: float = 18.0,
        tol_s: float = 0.25,
        tol_v: float = 0.25,
    ) -> bool:
        if rgb is None:
            rgb = self.display_rgb

        if rgb is None:
            return False

        H, W = rgb.shape[:2]
        y = H // 2
        x = W // 2

        y0 = max(0, y - 2)
        y1 = min(H, y + 3)
        x0 = max(0, x - 2)
        x1 = min(W, x + 3)

        patch = rgb[y0:y1, x0:x1]

        if patch.size == 0:
            return False

        hsv = rgb_to_hsv(patch)
        mean_hsv = hsv.reshape(-1, 3).mean(axis=0)

        h, s, v = map(float, mean_hsv)

        self.lower_bound = np.array(
            [
                h - tol_h,
                max(0.0, s - tol_s),
                max(0.0, v - tol_v),
            ],
            dtype=np.float32,
        )

        self.upper_bound = np.array(
            [
                h + tol_h,
                min(1.0, s + tol_s),
                min(1.0, v + tol_v),
            ],
            dtype=np.float32,
        )

        return True

    def capture_template(self) -> bool:
        if self.gray is None:
            return False

        H, W = self.gray.shape

        ts = int(min(self.template_size, H - 4, W - 4))

        if ts < 8:
            self.last_error = "Template size is too small for this frame."
            return False

        y = (H - ts) // 2
        x = (W - ts) // 2

        self.template = self.gray[y:y + ts, x:x + ts].astype(np.float64).copy()
        self.template_size = ts
        self.last_error = ""

        return True

    def process(self, rgb: np.ndarray) -> "VisionPipeline":
        if rgb is None:
            return self

        rgb = to_uint8_rgb(rgb)
        rgb = resize_nearest(rgb, self.max_dim)

        self._ensure_smoother(rgb)

        if self.smoother is not None:
            rgb = self.smoother.update(rgb)

        gray = rgb_to_grayscale(rgb)

        self.frame_count += 1

        self.display_rgb = rgb
        self.gray = gray

        H, W = gray.shape

        self.mask = None
        self.centroid = (-1.0, -1.0)

        if self.mode != "template":
            self.boxes = np.empty((0, 4), dtype=np.float64)
            self.scores = np.empty((0,), dtype=np.float64)

        # -------------------------------------------------------------------
        # Colour tracking
        # -------------------------------------------------------------------
        if self.mode == "color":
            hsv = rgb_to_hsv(rgb)

            mask = hsv_mask_wrapped(
                hsv,
                self.lower_bound,
                self.upper_bound,
            )

            mask = self._clean_mask(mask)

            self.mask = mask
            self.centroid = calculate_centroid(mask)

        # -------------------------------------------------------------------
        # Motion tracking
        # -------------------------------------------------------------------
        elif self.mode == "motion":
            if self.prev_gray is None or self.prev_gray.shape != gray.shape:
                self.prev_gray = gray
                self.mask = np.zeros_like(gray)

            else:
                mask = compute_motion_mask(
                    frame_t=gray,
                    frame_t_minus_1=self.prev_gray,
                    threshold=self.motion_threshold,
                )

                mask = self._clean_mask(mask)

                self.mask = mask
                self.centroid = calculate_centroid(mask)

        # -------------------------------------------------------------------
        # Template detection
        # -------------------------------------------------------------------
        elif self.mode == "template":
            if (
                self.template is None
                or self.template.shape[0] > H
                or self.template.shape[1] > W
            ):
                self.capture_template()

            if self.template is not None:
                run_detection = (
                    self.frame_count % self.detect_every == 0
                    or len(self.boxes) == 0
                )

                if run_detection:
                    try:
                        ncc_map = match_template_ncc(gray, self.template)

                        self.boxes, self.scores = find_template_matches(
                            ncc_map=ncc_map,
                            template_shape=self.template.shape,
                            threshold=self.detection_threshold,
                            nms_iou=0.3,
                        )

                        self.last_error = ""

                    except Exception as exc:
                        self.last_error = str(exc)
                        self.boxes = np.empty((0, 4), dtype=np.float64)
                        self.scores = np.empty((0,), dtype=np.float64)

                if len(self.boxes) > 0:
                    y1, x1, y2, x2 = self.boxes[0]
                    self.centroid = (
                        float((x1 + x2) * 0.5),
                        float((y1 + y2) * 0.5),
                    )

        else:
            raise ValueError(f"Unknown pipeline mode: {self.mode}")

        self.prev_gray = gray

        return self


# ---------------------------------------------------------------------------
# Pygame drawing helpers
# ---------------------------------------------------------------------------
def draw_boxes_pygame(
    screen,
    boxes: np.ndarray,
    sx: float,
    sy: float,
    color: tuple[int, int, int] = (255, 255, 0),
    width: int = 2,
) -> None:
    for box in boxes:
        y1, x1, y2, x2 = map(float, box)

        rect = pygame.Rect(
            int(x1 * sx),
            int(y1 * sy),
            int((x2 - x1) * sx),
            int((y2 - y1) * sy),
        )

        pygame.draw.rect(screen, color, rect, width)


def draw_cross_pygame(
    screen,
    centroid: tuple[float, float],
    sx: float,
    sy: float,
    color: tuple[int, int, int] = (255, 255, 255),
    radius: int = 8,
    width: int = 2,
) -> None:
    cx, cy = centroid

    if cx < 0 or cy < 0:
        return

    x = int(cx * sx)
    y = int(cy * sy)

    pygame.draw.line(screen, color, (x - radius, y), (x + radius, y), width)
    pygame.draw.line(screen, color, (x, y - radius), (x, y + radius), width)


def draw_hud_pygame(screen, font, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    y = 4

    for text, color in lines:
        surface = font.render(text, True, color)
        screen.blit(surface, (4, y))
        y += surface.get_height() + 2


# ---------------------------------------------------------------------------
# Headless runner
# ---------------------------------------------------------------------------
def run_headless(args: argparse.Namespace) -> int:
    try:
        source = make_source(
            source=args.source,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )

    except Exception as exc:
        print(f"Source error: {exc}")
        return 1

    pipeline = VisionPipeline(
        mode=args.mode,
        max_dim=args.max_dim,
        smooth_frames=args.smooth_frames,
        template_size=args.template_size,
        detection_threshold=args.detection_threshold,
        motion_threshold=args.motion_threshold,
        detect_every=args.detect_every,
        use_morphology=True,
    )

    try:
        for i in range(args.headless_frames):
            rgb = source.read()

            if rgb is None:
                print("Source returned no frame.")
                break

            pipeline.process(rgb)

            if i % max(1, args.headless_print_every) == 0:
                cx, cy = pipeline.centroid

                print(
                    f"frame={i:04d} "
                    f"mode={pipeline.mode:<8} "
                    f"proc_size={pipeline.gray.shape[1]}x{pipeline.gray.shape[0]} "
                    f"centroid=({cx:7.2f},{cy:7.2f}) "
                    f"boxes={len(pipeline.boxes)}"
                )

    except KeyboardInterrupt:
        print("Interrupted.")

    finally:
        source.close()

    return 0


# ---------------------------------------------------------------------------
# GUI runner
# ---------------------------------------------------------------------------
def run_gui(args: argparse.Namespace) -> int:
    if not PYGAME_AVAILABLE:
        print("pygame is required for the graphical live demo.")
        print("Install it with:")
        print("    python -m pip install pygame-ce")
        print("or:")
        print("    python -m pip install pygame")
        print()
        print(f"Import error: {PYGAME_ERROR}")
        return 1

    try:
        source = make_source(
            source=args.source,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )

    except Exception as exc:
        print(f"Source error: {exc}")
        return 1

    pipeline = VisionPipeline(
        mode=args.mode,
        max_dim=args.max_dim,
        smooth_frames=args.smooth_frames,
        template_size=args.template_size,
        detection_threshold=args.detection_threshold,
        motion_threshold=args.motion_threshold,
        detect_every=args.detect_every,
        use_morphology=True,
    )

    try:
        first_frame = source.read()

        if first_frame is None:
            print("Could not read the first frame from the source.")
            return 1

        pipeline.process(first_frame)

        pygame.init()
        pygame.display.init()
        pygame.font.init()

        H, W = pipeline.gray.shape
        scale = max(1, int(args.display_scale))

        screen = pygame.display.set_mode((int(W * scale), int(H * scale)))
        pygame.display.set_caption("mycv live demo")

        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 20)

        overlay = True
        running = True

        fps_smooth = 0.0
        last_time = time.perf_counter()

        try:
            target_fps = float(getattr(source, "fps", args.fps))
        except Exception:
            target_fps = float(args.fps)

        if not np.isfinite(target_fps) or target_fps <= 0:
            target_fps = float(args.fps)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False

                    elif event.key == pygame.K_c:
                        pipeline.mode = "color"

                    elif event.key == pygame.K_m:
                        pipeline.mode = "motion"

                    elif event.key == pygame.K_d:
                        pipeline.mode = "template"

                    elif event.key == pygame.K_t:
                        pipeline.mode = "template"
                        pipeline.capture_template()

                    elif event.key == pygame.K_s:
                        pipeline.sample_color_from_center()

                    elif event.key == pygame.K_g:
                        pipeline.set_color_preset("green")

                    elif event.key == pygame.K_r:
                        pipeline.set_color_preset("red")

                    elif event.key == pygame.K_b:
                        pipeline.set_color_preset("blue")

                    elif event.key == pygame.K_y:
                        pipeline.set_color_preset("yellow")

                    elif event.key == pygame.K_o:
                        overlay = not overlay

                    elif event.key == pygame.K_p:
                        pipeline.use_morphology = not pipeline.use_morphology

                    elif event.key in (pygame.K_EQUALS,):
                        pipeline.detection_threshold = min(
                            0.95,
                            pipeline.detection_threshold + 0.05,
                        )

                    elif event.key == pygame.K_MINUS:
                        pipeline.detection_threshold = max(
                            0.05,
                            pipeline.detection_threshold - 0.05,
                        )

                    elif event.key == pygame.K_LEFTBRACKET:
                        pipeline.motion_threshold = max(
                            5,
                            pipeline.motion_threshold - 5,
                        )

                    elif event.key == pygame.K_RIGHTBRACKET:
                        pipeline.motion_threshold = min(
                            120,
                            pipeline.motion_threshold + 5,
                        )

                    elif event.key == pygame.K_LEFT:
                        pipeline.template_size = max(
                            8,
                            pipeline.template_size - 2,
                        )

                    elif event.key == pygame.K_RIGHT:
                        pipeline.template_size = min(
                            96,
                            pipeline.template_size + 2,
                        )

                    elif event.key == pygame.K_h:
                        print(HELP_TEXT)

            rgb = source.read()

            if rgb is None:
                print("Source returned no frame. Exiting.")
                break

            pipeline.process(rgb)

            if pipeline.gray is None:
                continue

            if pipeline.gray.shape != (H, W):
                H, W = pipeline.gray.shape
                screen = pygame.display.set_mode((int(W * scale), int(H * scale)))

            vis = pipeline.display_rgb

            # Optional red mask overlay.
            if overlay and pipeline.mask is not None and pipeline.mask.shape == vis.shape[:2]:
                vis = vis.copy()
                idx = pipeline.mask > 0

                if np.any(idx):
                    red = np.array([255, 60, 60], dtype=np.float32)
                    vis[idx] = np.clip(
                        0.45 * vis[idx].astype(np.float32) + 0.55 * red,
                        0,
                        255,
                    ).astype(np.uint8)

            frame_surface = pygame.image.frombuffer(
                np.ascontiguousarray(vis).tobytes(),
                (W, H),
                "RGB",
            )

            display_surface = pygame.transform.scale(frame_surface, screen.get_size())
            screen.blit(display_surface, (0, 0))

            sx = screen.get_width() / float(W)
            sy = screen.get_height() / float(H)

            if pipeline.mode == "template" and len(pipeline.boxes) > 0:
                draw_boxes_pygame(screen, pipeline.boxes, sx, sy)

            draw_cross_pygame(
                screen,
                pipeline.centroid,
                sx,
                sy,
                color=(255, 255, 255),
            )

            # Centre sampling marker.
            centre_x = int((W // 2) * sx)
            centre_y = int((H // 2) * sy)
            pygame.draw.circle(screen, (0, 255, 255), (centre_x, centre_y), 3, 1)

            template_shape = (
                "None"
                if pipeline.template is None
                else f"{pipeline.template.shape[0]}x{pipeline.template.shape[1]}"
            )

            hud_lines = [
                (
                    f"mode:{pipeline.mode}  proc:{W}x{H}  fps:{fps_smooth:5.1f}  source:{args.source}",
                    (255, 255, 255),
                ),
                (
                    f"centroid:({pipeline.centroid[0]:6.1f},{pipeline.centroid[1]:6.1f})  "
                    f"morph:{int(pipeline.use_morphology)}  overlay:{int(overlay)}",
                    (0, 255, 140),
                ),
            ]

            if pipeline.mode == "color":
                hud_lines.append(
                    (
                        f"H range:[{pipeline.lower_bound[0]:6.1f},{pipeline.upper_bound[0]:6.1f}]  "
                        f"S:[{pipeline.lower_bound[1]:4.2f},{pipeline.upper_bound[1]:4.2f}]  "
                        f"V:[{pipeline.lower_bound[2]:4.2f},{pipeline.upper_bound[2]:4.2f}]",
                        (200, 220, 255),
                    )
                )

            elif pipeline.mode == "motion":
                hud_lines.append(
                    (
                        f"motion threshold:{pipeline.motion_threshold}  use [ and ] to adjust",
                        (255, 220, 180),
                    )
                )

            elif pipeline.mode == "template":
                hud_lines.append(
                    (
                        f"thr:{pipeline.detection_threshold:4.2f}  matches:{len(pipeline.boxes)}  "
                        f"template:{template_shape}  detect_every:{pipeline.detect_every}",
                        (255, 255, 140),
                    )
                )

            if pipeline.last_error:
                hud_lines.append(
                    (
                        f"error:{pipeline.last_error[:70]}",
                        (255, 90, 90),
                    )
                )

            hud_lines.append(
                (
                    "q:quit c:color m:motion t:template s:sample g/r/b/y presets o:overlay p:morph",
                    (190, 190, 190),
                )
            )

            hud_lines.append(
                (
                    "=/-:det thr  [ ]:motion thr  left/right:template size  h:help",
                    (190, 190, 190),
                )
            )

            draw_hud_pygame(screen, font, hud_lines)

            pygame.display.flip()

            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            if dt > 0:
                instantaneous_fps = 1.0 / dt

                if fps_smooth == 0.0:
                    fps_smooth = instantaneous_fps
                else:
                    fps_smooth = 0.9 * fps_smooth + 0.1 * instantaneous_fps

            clock.tick(target_fps)

    except KeyboardInterrupt:
        print("Interrupted.")

    except Exception as exc:
        print(f"Runtime error: {exc}")
        return 1

    finally:
        try:
            source.close()
        except Exception:
            pass

        try:
            pygame.quit()
        except Exception:
            pass

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="mycv live tracking and detection demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--source",
        type=str,
        default="synthetic",
        help=(
            "Frame source. Use synthetic, camera, an integer camera index, "
            "a video-file path, or an FFmpeg device string such as "
            "'dshow:video=Integrated Camera'."
        ),
    )

    parser.add_argument("--width", type=int, default=320, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=240, help="Requested capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested capture frame rate.")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["color", "motion", "template"],
        default="color",
        help="Initial vision mode.",
    )

    parser.add_argument(
        "--max-dim",
        type=int,
        default=180,
        help=(
            "Maximum processed image dimension. Frames are downscaled to this "
            "size before tracking/detection to keep NCC tractable."
        ),
    )

    parser.add_argument(
        "--display-scale",
        type=int,
        default=3,
        help="Integer display magnification.",
    )

    parser.add_argument(
        "--smooth-frames",
        type=int,
        default=0,
        help="Number of frames in the temporal ring-buffer smoother. Use 0 to disable.",
    )

    parser.add_argument(
        "--template-size",
        type=int,
        default=28,
        help="Initial square template size in processed-frame pixels.",
    )

    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.55,
        help="Initial NCC detection threshold.",
    )

    parser.add_argument(
        "--motion-threshold",
        type=int,
        default=25,
        help="Initial frame-difference motion threshold.",
    )

    parser.add_argument(
        "--detect-every",
        type=int,
        default=1,
        help="Run template detection every N frames. Increase for performance.",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without a GUI window. Useful for SSH or debugging.",
    )

    parser.add_argument(
        "--headless-frames",
        type=int,
        default=120,
        help="Number of frames to process in headless mode.",
    )

    parser.add_argument(
        "--headless-print-every",
        type=int,
        default=10,
        help="Print status every N frames in headless mode.",
    )

    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List pygame.camera devices and exit.",
    )

    args = parser.parse_args()

    if args.list_cameras:
        if not PYGAME_AVAILABLE:
            print(f"pygame is not available: {PYGAME_ERROR}")
            return 1

        try:
            pygame.camera.init()
            cameras = pygame.camera.list_cameras()
            print("pygame.camera devices:")
            for i, camera in enumerate(cameras):
                print(f"    {i}: {camera}")
            pygame.camera.quit()
        except Exception as exc:
            print(f"Could not list cameras: {exc}")
            return 1

        return 0

    if args.headless:
        return run_headless(args)

    return run_gui(args)


if __name__ == "__main__":
    raise SystemExit(main())