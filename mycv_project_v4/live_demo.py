#!/usr/bin/env python3
"""
live_demo.py

Updated real-time demonstration driver for mycv v4.1.

Integrates:

    - colour tracking
    - motion tracking
    - template detection
    - multi-scale template detection
    - connected-component labelling
    - component properties
    - single-object Kalman tracking
    - multi-object Kalman tracking
    - bounding-box extraction and drawing
    - object classification
    - optional Harris corner and Hough line feature counting
    - optional mycv.streaming.StreamReader for files, RTSP, HTTP, UDP, etc.

No OpenCV is used for image mathematics.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


try:
    from mycv.core import rgb_to_grayscale
    from mycv.color import rgb_to_hsv
    from mycv.filters import sobel_edge_detection
    from mycv.morphology import (
        opening,
        closing,
        label_connected_components,
        component_properties,
    )
    from mycv.features import (
        extract_object_metrics,
        draw_bounding_box,
        classify_object,
    )
    from mycv.tracking import (
        compute_motion_mask,
        color_mask_hue_wrap,
        calculate_centroid,
        TemporalSmoother,
        KalmanCentroidTracker,
        MultiObjectKalmanTracker,
    )
    from mycv.detection import (
        match_template_ncc,
        match_template_ncc_multiscale,
        find_template_matches,
    )

except Exception as exc:
    print("Failed to import mycv modules.")
    print()
    print("Common causes:")
    print("  1. mycv/__init__.py contains raw uncommented text.")
    print("  2. mycv/__init__.py uses 'all' instead of '__all__'.")
    print("  3. Exported names in __all__ contain trailing spaces.")
    print("  4. A module file is missing or has a syntax error.")
    print()
    print(f"Import error: {exc}")
    raise SystemExit(1)


try:
    from mycv.streaming import StreamReader
    STREAMING_AVAILABLE = True
    STREAMING_ERROR = None
except Exception as exc:
    StreamReader = None
    STREAMING_AVAILABLE = False
    STREAMING_ERROR = exc


try:
    import pygame
    import pygame.camera

    PYGAME_AVAILABLE = True
    PYGAME_ERROR = None
except Exception as exc:
    pygame = None
    PYGAME_AVAILABLE = False
    PYGAME_ERROR = exc


HELP_TEXT = """
mycv live demo v4.1
===================

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

Tracking controls:
    k       toggle Kalman filtering
    u       toggle multi-object tracking
    i       toggle object bounding box + classification
    o       toggle mask overlay
    p       toggle morphological cleanup
    f       toggle corner/line feature counting

Detection controls:
    n       toggle multi-scale template matching
    = / +   increase detection threshold
    -       decrease detection threshold
    [       decrease motion threshold
    ]       increase motion threshold
    left    decrease template size
    right   increase template size

Other:
    h       print help
    q/Esc   quit

Example sources:
    --source synthetic
    --source camera
    --source 0
    --source video.mp4
    --source "rtsp://192.168.1.100:554/stream"
    --source "dshow:video=Integrated Camera"
    --source "v4l2:/dev/video0"
    --source "avfoundation:0:1"
"""


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------
def resize_nearest(image, max_dim):
    """
    Downscale with nearest-neighbour sampling.

    This keeps connected-component labelling, NCC, and feature extraction
    tractable in real time.
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


def to_uint8_rgb(image):
    """
    Convert an incoming frame to uint8 RGB.
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


def confidence_from_area(area, H, W):
    """
    Convert object pixel area into a Kalman confidence value in (0, 1].

    A very small mask usually produces a noisy centroid, so its measurement
    should be trusted less. A mask occupying about 0.5% of the frame or
    more receives full confidence.
    """

    frame_area = float(max(1, H * W))
    fraction = float(area) / frame_area

    confidence = fraction / 0.005
    confidence = float(np.clip(confidence, 0.10, 1.0))

    return confidence


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------
class SyntheticSource:
    """
    Synthetic moving square for testing.
    """

    def __init__(self, width=320, height=240, fps=30):
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.t = 0
        self.rng = np.random.default_rng(0)

    def read(self):
        H = self.height
        W = self.width

        img = np.zeros((H, W, 3), dtype=np.uint8)

        img[..., 0] = 18
        img[..., 1] = 20
        img[..., 2] = 26

        square = max(12, min(H, W) // 5)
        y = H // 2 - square // 2

        max_x = max(1, W - square)
        x = int((self.t * 3) % max_x)

        img[y:y + square, x:x + square] = (55, 220, 85)

        noise = self.rng.integers(-6, 7, size=img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        self.t += 1

        return img, time.perf_counter()

    def close(self):
        pass


class PygameCameraSource:
    """
    Webcam source through pygame.camera.
    """

    def __init__(self, device=0, width=320, height=240, fps=30):
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
            self.cam = pygame.camera.Camera(device_path)

        self.cam.start()

        self.width = int(width)
        self.height = int(height)

    def read(self):
        try:
            surface = self.cam.get_image()
            w, h = surface.get_size()
            data = pygame.image.tostring(surface, "RGB")
            frame = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3).copy()

            self.width = w
            self.height = h

            return frame, time.perf_counter()

        except Exception:
            return None, None

    def close(self):
        try:
            self.cam.stop()
        except Exception:
            pass

        try:
            pygame.camera.quit()
        except Exception:
            pass


class StreamSource:
    """
    Video-file / network-stream / FFmpeg-device source through mycv.streaming.

    This uses mycv.streaming.StreamReader, which requires the optional `av`
    package. It supports local files, RTSP, HTTP, HTTPS, UDP, RTP, and
    FFmpeg device strings such as:

        dshow:video=Integrated Camera
        v4l2:/dev/video0
        avfoundation:0:1
    """

    def __init__(self, url, rtsp_transport="tcp", timeout_us=5_000_000, reconnect_delay=2.0):
        if not STREAMING_AVAILABLE:
            raise RuntimeError(
                "mycv.streaming.StreamReader is unavailable because the optional "
                f"'av' package is missing or failed to import. Details: {STREAMING_ERROR}"
            )

        self.reader = StreamReader(
            url=url,
            rtsp_transport=rtsp_transport,
            timeout_us=timeout_us,
            reconnect_delay=reconnect_delay,
        ).start()

        self.fps = 30.0

    def read(self):
        frame, timestamp = self.reader.read(timeout=1.0)
        return frame, timestamp

    def close(self):
        try:
            self.reader.stop()
        except Exception:
            pass


def make_source(source, width, height, fps):
    """
    Construct the requested frame source.

    Source strings:

        synthetic
        camera
        0, 1, 2, ...
        video.mp4
        rtsp://...
        http://...
        dshow:video=Camera Name
        v4l2:/dev/video0
        avfoundation:0:1
    """

    source = str(source)

    if source.lower() == "synthetic":
        return SyntheticSource(width=width, height=height, fps=fps)

    if source.lower() == "camera":
        return PygameCameraSource(device=0, width=width, height=height, fps=fps)

    if source.isdigit():
        return PygameCameraSource(device=int(source), width=width, height=height, fps=fps)

    return StreamSource(source)


# ---------------------------------------------------------------------------
# Vision pipeline
# ---------------------------------------------------------------------------
class VisionPipeline:
    """
    Unified real-time pipeline for mycv v4.1.
    """

    def __init__(
        self,
        mode="color",
        max_dim=180,
        smooth_frames=0,
        template_size=28,
        detection_threshold=0.55,
        motion_threshold=25,
        detect_every=1,
        use_morphology=True,
        enable_kalman=True,
        enable_object_info=True,
        multi_object=False,
        use_multiscale=False,
        multiscale_levels=3,
        feature_counts=False,
        process_noise=1e-2,
        measurement_noise=1e-1,
        gate_threshold=0.0,
        min_area=20,
        max_match_distance=50.0,
        max_missed=5,
    ):
        self.mode = mode
        self.max_dim = int(max_dim)
        self.smooth_frames = int(smooth_frames)
        self.template_size = int(template_size)
        self.detection_threshold = float(detection_threshold)
        self.motion_threshold = int(motion_threshold)
        self.detect_every = max(1, int(detect_every))

        self.use_morphology = bool(use_morphology)
        self.use_kalman = bool(enable_kalman)
        self.show_object_info = bool(enable_object_info)
        self.multi_object = bool(multi_object)
        self.use_multiscale = bool(use_multiscale)
        self.multiscale_levels = int(multiscale_levels)
        self.feature_counts = bool(feature_counts)

        self.min_area = int(min_area)
        self.max_match_distance = float(max_match_distance)
        self.max_missed = int(max_missed)

        gate = None
        if gate_threshold is not None and float(gate_threshold) > 0:
            gate = float(gate_threshold)

        self.single_tracker = KalmanCentroidTracker(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            dt=1.0,
            gate_threshold=gate,
        )

        self.multi_tracker = MultiObjectKalmanTracker(
            max_match_distance=self.max_match_distance,
            max_missed=self.max_missed,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            dt=1.0,
        )

        # Default green HSV bounds.
        self.lower_bound = np.array([70.0, 0.25, 0.25], dtype=np.float32)
        self.upper_bound = np.array([170.0, 1.0, 1.0], dtype=np.float32)

        self.smoother = None
        self.prev_gray = None
        self.template = None

        self.frame_count = 0

        self.display_rgb = None
        self.gray = None
        self.mask = None

        self.components = []
        self.selected_mask = None

        self.centroid = (-1.0, -1.0)
        self.track_positions = {}
        self.raw_detections = []

        self.boxes = np.empty((0, 4), dtype=np.float64)
        self.scores = np.empty((0,), dtype=np.float64)

        self.object_metrics = None
        self.object_label = ""
        self.object_bbox = None

        self.lost_frames = 0
        self._last_mode = None

        self.last_error = ""

    def set_mode(self, mode):
        """
        Change pipeline mode and reset temporal tracking state.
        """

        if mode == self.mode:
            return

        self.mode = mode

        self.single_tracker.reset()
        self.multi_tracker.reset()

        self.components = []
        self.selected_mask = None
        self.track_positions = {}
        self.raw_detections = []

        self.object_metrics = None
        self.object_label = ""
        self.object_bbox = None

        self.lost_frames = 0

    def _ensure_smoother(self, rgb):
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

    def _clean_mask(self, mask):
        if not self.use_morphology:
            return mask

        return closing(opening(mask))

    def set_color_preset(self, name):
        """
        Apply a simple HSV colour preset.
        """

        name = name.lower()

        if name == "green":
            self.lower_bound = np.array([70.0, 0.25, 0.25], dtype=np.float32)
            self.upper_bound = np.array([170.0, 1.0, 1.0], dtype=np.float32)

        elif name == "red":
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

        self.single_tracker.reset()
        self.multi_tracker.reset()
        self.lost_frames = 0

    def sample_color_from_center(self, rgb=None, tol_h=18.0, tol_s=0.25, tol_v=0.25):
        """
        Sample a small centre patch and set HSV bounds from its mean HSV.
        """

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

        self.single_tracker.reset()
        self.multi_tracker.reset()
        self.lost_frames = 0

        return True

    def capture_template(self):
        """
        Capture a centre template from the current processed grayscale frame.
        """

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

        self.single_tracker.reset()
        self.multi_tracker.reset()
        self.lost_frames = 0

        return True

    def _component_detections(self, mask, rgb):
        """
        Label connected components and return component properties.
        """

        labels, num_labels = label_connected_components(mask, connectivity=8)

        if num_labels == 0:
            return [], None

        props = component_properties(labels, num_labels, image=rgb)

        props = [p for p in props if p["pixel_area"] >= self.min_area]

        if not props:
            return [], None

        props.sort(key=lambda p: p["pixel_area"], reverse=True)

        selected_label = props[0]["label"]
        selected_mask = (labels == selected_label)

        return props, selected_mask

    def _update_object_info(self, rgb, gray):
        """
        Extract bounding-box metrics and classify the current selected object.
        """

        self.object_metrics = None
        self.object_label = ""
        self.object_bbox = None

        if not self.show_object_info:
            return

        # -------------------------------------------------------------------
        # Mask-based modes: colour or motion.
        # -------------------------------------------------------------------
        if self.mode in {"color", "motion"} and self.selected_mask is not None:
            Gx = None
            Gy = None
            edges = None

            if self.feature_counts:
                sob = sobel_edge_detection(gray)
                Gx = sob["Gx"]
                Gy = sob["Gy"]

                mag = sob["magnitude"]
                edge_thresh = max(15, int(mag.max() * 0.25))
                edges = (mag >= edge_thresh).astype(np.uint8) * 255

            metrics = extract_object_metrics(
                self.selected_mask,
                Gx=Gx,
                Gy=Gy,
                edges=edges,
            )

            if metrics is not None:
                self.object_metrics = metrics
                self.object_bbox = metrics["bbox"]

                try:
                    self.object_label = classify_object(
                        rgb,
                        metrics,
                        mask=self.selected_mask,
                    )
                except Exception:
                    self.object_label = "Unknown"

            return

        # -------------------------------------------------------------------
        # Template mode: use best detection as an approximate object box.
        # -------------------------------------------------------------------
        if self.mode == "template" and len(self.boxes) > 0:
            y1, x1, y2, x2 = map(int, self.boxes[0])

            y2 = max(y2, y1 + 1)
            x2 = max(x2, x1 + 1)

            width = x2 - x1
            height = y2 - y1

            aspect_ratio = width / height if height > 0 else 0.0

            metrics = {
                "bbox": (y1, x1, y2, x2),
                "bbox_width": width,
                "bbox_height": height,
                "bbox_area": width * height,
                "pixel_area": width * height,
                "aspect_ratio": aspect_ratio,
                "centroid": (
                    float(x1 + x2) * 0.5,
                    float(y1 + y2) * 0.5,
                ),
            }

            self.object_metrics = metrics
            self.object_bbox = metrics["bbox"]

            try:
                self.object_label = classify_object(rgb, metrics, mask=None)
            except Exception:
                self.object_label = "Template"

    def process(self, rgb, timestamp=None):
        """
        Process one RGB frame.
        """

        if rgb is None:
            return self

        rgb = to_uint8_rgb(rgb)
        rgb = resize_nearest(rgb, self.max_dim)

        self._ensure_smoother(rgb)

        if self.smoother is not None:
            rgb = self.smoother.update(rgb)

        gray = rgb_to_grayscale(rgb)

        if self.mode != self._last_mode:
            self.single_tracker.reset()
            self.multi_tracker.reset()
            self.lost_frames = 0
            self._last_mode = self.mode

        self.frame_count += 1

        self.display_rgb = rgb
        self.gray = gray

        H, W = gray.shape

        self.mask = None
        self.components = []
        self.selected_mask = None

        self.track_positions = {}
        self.raw_detections = []

        raw_centroid = (-1.0, -1.0)
        raw_area = 0

        if self.mode != "template":
            self.boxes = np.empty((0, 4), dtype=np.float64)
            self.scores = np.empty((0,), dtype=np.float64)

        # -------------------------------------------------------------------
        # Colour tracking.
        # -------------------------------------------------------------------
        if self.mode == "color":
            hsv = rgb_to_hsv(rgb)

            mask = color_mask_hue_wrap(
                hsv,
                self.lower_bound,
                self.upper_bound,
            )

            mask = self._clean_mask(mask)

            self.mask = mask

            props, selected_mask = self._component_detections(mask, rgb)

            self.components = props
            self.selected_mask = selected_mask

            if props:
                raw_centroid = props[0]["centroid"]
                raw_area = props[0]["pixel_area"]

                detections = [p["centroid"] for p in props]
                self.raw_detections = detections

                if self.multi_object and self.use_kalman:
                    self.track_positions = self.multi_tracker.update(detections)
                elif self.multi_object:
                    self.track_positions = {
                        i + 1: det for i, det in enumerate(detections)
                    }

        # -------------------------------------------------------------------
        # Motion tracking.
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

                props, selected_mask = self._component_detections(mask, rgb)

                self.components = props
                self.selected_mask = selected_mask

                if props:
                    raw_centroid = props[0]["centroid"]
                    raw_area = props[0]["pixel_area"]

                    detections = [p["centroid"] for p in props]
                    self.raw_detections = detections

                    if self.multi_object and self.use_kalman:
                        self.track_positions = self.multi_tracker.update(detections)
                    elif self.multi_object:
                        self.track_positions = {
                            i + 1: det for i, det in enumerate(detections)
                        }

        # -------------------------------------------------------------------
        # Template detection.
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
                        if self.use_multiscale:
                            self.boxes, self.scores = match_template_ncc_multiscale(
                                gray,
                                self.template,
                                levels=self.multiscale_levels,
                                threshold=self.detection_threshold,
                                nms_iou=0.3,
                            )
                        else:
                            ncc_map = match_template_ncc(gray, self.template)

                            self.boxes, self.scores = find_template_matches(
                                ncc_map=ncc_map,
                                template_shape=self.template.shape,
                                threshold=self.detection_threshold,
                                nms_iou=0.3,
                            )

                        self.last_error = ""

                    except MemoryError as exc:
                        self.last_error = str(exc)
                        self.boxes = np.empty((0, 4), dtype=np.float64)
                        self.scores = np.empty((0,), dtype=np.float64)

                    except Exception as exc:
                        self.last_error = str(exc)
                        self.boxes = np.empty((0, 4), dtype=np.float64)
                        self.scores = np.empty((0,), dtype=np.float64)

                if len(self.boxes) > 0:
                    centres = []

                    for box in self.boxes:
                        y1, x1, y2, x2 = box
                        centres.append(
                            (
                                float((x1 + x2) * 0.5),
                                float((y1 + y2) * 0.5),
                            )
                        )

                    self.raw_detections = centres

                    raw_centroid = centres[0]
                    raw_area = int(
                        max(1.0, float(self.boxes[0][3] - self.boxes[0][1]))
                        * max(1.0, float(self.boxes[0][2] - self.boxes[0][0]))
                    )

                    if self.multi_object and self.use_kalman:
                        self.track_positions = self.multi_tracker.update(centres)
                    elif self.multi_object:
                        self.track_positions = {
                            i + 1: det for i, det in enumerate(centres)
                        }

                else:
                    if self.multi_object and self.use_kalman:
                        self.track_positions = self.multi_tracker.update([])

        else:
            raise ValueError(f"Unknown pipeline mode: {self.mode}")

        # -------------------------------------------------------------------
        # Single-object Kalman smoothing / prediction.
        # -------------------------------------------------------------------
        if not self.multi_object:
            if raw_centroid[0] >= 0 and raw_centroid[1] >= 0:
                self.lost_frames = 0

                if self.use_kalman:
                    confidence = confidence_from_area(raw_area, H, W)

                    smoothed = self.single_tracker.update(
                        raw_centroid,
                        dt=1.0,
                        confidence=confidence,
                    )

                    self.centroid = (float(smoothed[0]), float(smoothed[1]))
                else:
                    self.centroid = (float(raw_centroid[0]), float(raw_centroid[1]))

            else:
                self.lost_frames += 1

                if self.use_kalman and self.lost_frames <= 60:
                    predicted = self.single_tracker.predict(dt=1.0)

                    if predicted[0] >= 0 and predicted[1] >= 0:
                        self.centroid = (float(predicted[0]), float(predicted[1]))
                    else:
                        self.centroid = (-1.0, -1.0)
                else:
                    if self.lost_frames > 60:
                        self.single_tracker.reset()

                    self.centroid = (-1.0, -1.0)

        else:
            # In multi-object mode, keep the largest raw detection as a
            # convenience centroid for object classification/HUD.
            self.centroid = (
                float(raw_centroid[0]),
                float(raw_centroid[1]),
            ) if raw_centroid[0] >= 0 else (-1.0, -1.0)

        self.prev_gray = gray

        # -------------------------------------------------------------------
        # Bounding-box extraction and classification.
        # -------------------------------------------------------------------
        self._update_object_info(rgb, gray)

        return self


# ---------------------------------------------------------------------------
# Pygame drawing helpers
# ---------------------------------------------------------------------------
def draw_boxes_pygame(screen, boxes, sx, sy, color=(255, 255, 0), width=2):
    for box in boxes:
        y1, x1, y2, x2 = map(float, box)

        rect = pygame.Rect(
            int(x1 * sx),
            int(y1 * sy),
            int((x2 - x1) * sx),
            int((y2 - y1) * sy),
        )

        pygame.draw.rect(screen, color, rect, width)


def draw_cross_pygame(screen, centroid, sx, sy, color=(255, 255, 255), radius=8, width=2):
    cx, cy = centroid

    if cx < 0 or cy < 0:
        return

    x = int(cx * sx)
    y = int(cy * sy)

    pygame.draw.line(screen, color, (x - radius, y), (x + radius, y), width)
    pygame.draw.line(screen, color, (x, y - radius), (x, y + radius), width)


def draw_tracks_pygame(screen, track_positions, sx, sy, font, color=(0, 255, 160)):
    for tid, (x, y) in track_positions.items():
        if x < 0 or y < 0:
            continue

        px = int(x * sx)
        py = int(y * sy)

        pygame.draw.circle(screen, color, (px, py), 5, 1)

        label = font.render(str(tid), True, color)
        screen.blit(label, (px + 6, py - 8))


def draw_hud_pygame(screen, font, lines):
    y = 4

    for text, color in lines:
        surface = font.render(text, True, color)
        screen.blit(surface, (4, y))
        y += surface.get_height() + 2


# ---------------------------------------------------------------------------
# Headless runner
# ---------------------------------------------------------------------------
def run_headless(args):
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
        use_morphology=not args.no_morphology,
        enable_kalman=not args.no_kalman,
        enable_object_info=not args.no_object_info,
        multi_object=args.multi,
        use_multiscale=args.multiscale,
        multiscale_levels=args.multiscale_levels,
        feature_counts=args.feature_counts,
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
        gate_threshold=args.gate,
        min_area=args.min_area,
        max_match_distance=args.max_match_distance,
        max_missed=args.max_missed,
    )

    try:
        for i in range(args.headless_frames):
            frame, timestamp = source.read()

            if frame is None:
                print("Source returned no frame.")
                break

            pipeline.process(frame, timestamp)

            if i % max(1, args.headless_print_every) == 0:
                cx, cy = pipeline.centroid
                label = pipeline.object_label if pipeline.object_label else "-"

                print(
                    f"frame={i:04d} "
                    f"mode={pipeline.mode:<8} "
                    f"proc_size={pipeline.gray.shape[1]}x{pipeline.gray.shape[0]} "
                    f"centroid=({cx:7.2f},{cy:7.2f}) "
                    f"components={len(pipeline.components):02d} "
                    f"tracks={len(pipeline.track_positions):02d} "
                    f"kalman={int(pipeline.use_kalman)} "
                    f"label={label}"
                )

    except KeyboardInterrupt:
        print("Interrupted.")

    finally:
        source.close()

    return 0


# ---------------------------------------------------------------------------
# GUI runner
# ---------------------------------------------------------------------------
def run_gui(args):
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
        use_morphology=not args.no_morphology,
        enable_kalman=not args.no_kalman,
        enable_object_info=not args.no_object_info,
        multi_object=args.multi,
        use_multiscale=args.multiscale,
        multiscale_levels=args.multiscale_levels,
        feature_counts=args.feature_counts,
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
        gate_threshold=args.gate,
        min_area=args.min_area,
        max_match_distance=args.max_match_distance,
        max_missed=args.max_missed,
    )

    try:
        first_frame, first_timestamp = source.read()

        if first_frame is None:
            print("Could not read the first frame from the source.")
            return 1

        pipeline.process(first_frame, first_timestamp)

        pygame.init()
        pygame.display.init()
        pygame.font.init()

        H, W = pipeline.gray.shape
        scale = max(1, int(args.display_scale))

        screen = pygame.display.set_mode((int(W * scale), int(H * scale)))
        pygame.display.set_caption("mycv live demo v4.1")

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
                        pipeline.set_mode("color")

                    elif event.key == pygame.K_m:
                        pipeline.set_mode("motion")

                    elif event.key == pygame.K_d:
                        pipeline.set_mode("template")

                    elif event.key == pygame.K_t:
                        pipeline.set_mode("template")
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

                    elif event.key == pygame.K_k:
                        pipeline.use_kalman = not pipeline.use_kalman
                        pipeline.single_tracker.reset()
                        pipeline.multi_tracker.reset()
                        pipeline.lost_frames = 0

                    elif event.key == pygame.K_u:
                        pipeline.multi_object = not pipeline.multi_object
                        pipeline.single_tracker.reset()
                        pipeline.multi_tracker.reset()
                        pipeline.track_positions = {}

                    elif event.key == pygame.K_i:
                        pipeline.show_object_info = not pipeline.show_object_info

                    elif event.key == pygame.K_o:
                        overlay = not overlay

                    elif event.key == pygame.K_p:
                        pipeline.use_morphology = not pipeline.use_morphology

                    elif event.key == pygame.K_f:
                        pipeline.feature_counts = not pipeline.feature_counts

                    elif event.key == pygame.K_n:
                        pipeline.use_multiscale = not pipeline.use_multiscale

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

            frame, timestamp = source.read()

            if frame is None:
                print("Source returned no frame. Exiting.")
                break

            pipeline.process(frame, timestamp)

            if pipeline.gray is None:
                continue

            if pipeline.gray.shape != (H, W):
                H, W = pipeline.gray.shape
                screen = pygame.display.set_mode((int(W * scale), int(H * scale)))

            # Always copy because draw_bounding_box writes in-place.
            vis = pipeline.display_rgb.copy()

            # Optional red mask overlay.
            if overlay and pipeline.mask is not None and pipeline.mask.shape == vis.shape[:2]:
                idx = pipeline.mask > 0

                if np.any(idx):
                    red = np.array([255, 60, 60], dtype=np.float32)
                    vis[idx] = np.clip(
                        0.45 * vis[idx].astype(np.float32) + 0.55 * red,
                        0,
                        255,
                    ).astype(np.uint8)

            # Draw selected object bounding box using mycv.features.draw_bounding_box.
            if pipeline.show_object_info and pipeline.object_bbox is not None:
                try:
                    draw_bounding_box(
                        vis,
                        pipeline.object_bbox,
                        color=(0, 255, 255),
                        thickness=2,
                    )
                except Exception:
                    pass

            frame_surface = pygame.image.frombuffer(
                np.ascontiguousarray(vis).tobytes(),
                (W, H),
                "RGB",
            )

            display_surface = pygame.transform.scale(frame_surface, screen.get_size())
            screen.blit(display_surface, (0, 0))

            sx = screen.get_width() / float(W)
            sy = screen.get_height() / float(H)

            # Draw component boxes for mask-based multi-object tracking.
            if pipeline.mode in {"color", "motion"} and pipeline.show_object_info:
                for prop in pipeline.components:
                    y1, x1, y2, x2 = prop["bbox"]

                    rect = pygame.Rect(
                        int(x1 * sx),
                        int(y1 * sy),
                        int((x2 - x1) * sx),
                        int((y2 - y1) * sy),
                    )

                    pygame.draw.rect(screen, (0, 220, 120), rect, 1)

            # Template mode: draw all detection boxes lightly.
            if pipeline.mode == "template" and len(pipeline.boxes) > 0:
                draw_boxes_pygame(
                    screen,
                    pipeline.boxes,
                    sx,
                    sy,
                    color=(255, 255, 0),
                    width=1,
                )

            # Draw tracks or single centroid.
            if pipeline.multi_object and pipeline.track_positions:
                draw_tracks_pygame(screen, pipeline.track_positions, sx, sy, font)

            elif not pipeline.multi_object:
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

            # Draw classification label near bounding box.
            if pipeline.show_object_info and pipeline.object_label and pipeline.object_bbox is not None:
                label_surf = font.render(pipeline.object_label, True, (0, 255, 255))

                label_x = int(pipeline.object_bbox[1] * sx)
                label_y = int(pipeline.object_bbox[0] * sy) - 18

                if label_y < 0:
                    label_y = int(pipeline.object_bbox[0] * sy) + 4

                screen.blit(label_surf, (label_x, label_y))

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
                    f"kalman:{int(pipeline.use_kalman)}  multi:{int(pipeline.multi_object)}  "
                    f"obj_info:{int(pipeline.show_object_info)}",
                    (0, 255, 140),
                ),
            ]

            if pipeline.object_label:
                hud_lines.append(
                    (
                        f"object: {pipeline.object_label}",
                        (0, 255, 255),
                    )
                )

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
                        f"multi-scale:{int(pipeline.use_multiscale)}  template:{template_shape}",
                        (255, 255, 140),
                    )
                )

            if pipeline.mode in {"color", "motion"}:
                hud_lines.append(
                    (
                        f"components:{len(pipeline.components)}  tracks:{len(pipeline.track_positions)}  "
                        f"min_area:{pipeline.min_area}",
                        (180, 255, 180),
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
                    "q:quit c:color m:motion t:template s:sample g/r/b/y presets",
                    (190, 190, 190),
                )
            )

            hud_lines.append(
                (
                    "k:kalman u:multi i:object o:overlay p:morph f:features n:multiscale h:help",
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
def main():
    parser = argparse.ArgumentParser(
        description="mycv v4.1 live tracking, detection, classification, and Kalman demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--source",
        type=str,
        default="synthetic",
        help=(
            "Frame source. Use synthetic, camera, an integer camera index, "
            "a video-file path, a network stream URL, or an FFmpeg device string."
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
        help="Maximum processed image dimension.",
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
        help="Run template detection every N frames.",
    )

    parser.add_argument(
        "--no-morphology",
        action="store_true",
        help="Disable morphological cleanup at startup.",
    )

    parser.add_argument(
        "--no-kalman",
        action="store_true",
        help="Disable Kalman filtering at startup.",
    )

    parser.add_argument(
        "--no-object-info",
        action="store_true",
        help="Disable bounding-box and classification overlay at startup.",
    )

    parser.add_argument(
        "--multi",
        action="store_true",
        help="Start in multi-object tracking mode.",
    )

    parser.add_argument(
        "--multiscale",
        action="store_true",
        help="Start with multi-scale template matching enabled.",
    )

    parser.add_argument(
        "--multiscale-levels",
        type=int,
        default=3,
        help="Number of pyramid levels for multi-scale template matching.",
    )

    parser.add_argument(
        "--feature-counts",
        action="store_true",
        help="Enable Harris corner count and Hough line count in object metrics.",
    )

    parser.add_argument(
        "--process-noise",
        type=float,
        default=1e-2,
        help="Kalman process noise variance.",
    )

    parser.add_argument(
        "--measurement-noise",
        type=float,
        default=1e-1,
        help="Kalman measurement noise variance.",
    )

    parser.add_argument(
        "--gate",
        type=float,
        default=0.0,
        help="Mahalanobis gating threshold for single-object Kalman. 0 disables gating. A common 2-D value is 9.21.",
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=20,
        help="Minimum connected-component pixel area to keep.",
    )

    parser.add_argument(
        "--max-match-distance",
        type=float,
        default=50.0,
        help="Maximum association distance for multi-object Kalman tracking.",
    )

    parser.add_argument(
        "--max-missed",
        type=int,
        default=5,
        help="Maximum missed frames before a multi-object track is deleted.",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without a GUI window.",
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