"""
mycv.tracking
=============
Pure-NumPy motion detection, colour-based object tracking,
and temporal video noise reduction.

Functions
---------
compute_motion_mask  : Frame-difference binary motion mask
color_mask           : HSV range filter (replaces cv2.inRange)
calculate_centroid   : 2-D centre-of-mass of a binary mask

Classes
-------
TemporalSmoother     : Circular-buffer rolling-average denoiser

Mathematical background
-----------------------
Motion detection computes the absolute per-pixel intensity difference
between two consecutive frames and thresholds it, detecting locations
where the scene has changed.  Correct uint8 subtraction requires a safe
path to avoid unsigned integer underflow.

Colour masking treats HSV as a 3-D bounded box query: a pixel is
"inside" the target colour if each of its three channel values lies
within the specified lower and upper bounds simultaneously.

Centroid estimation is the discrete 2-D expected value of pixel
coordinates weighted by the binary mask — equivalent to the arithmetic
mean of all foreground pixel positions.

Temporal smoothing averages the last N frames along the time axis using
a pre-allocated circular (ring) buffer, which avoids O(N) memory shifts
on every update and achieves constant-time writes via modulo indexing.
"""

import numpy as np


# ============================================================
#  1.  Motion Detection
# ============================================================

def compute_motion_mask(
    frame_t: np.ndarray,
    frame_t_minus_1: np.ndarray,
    threshold: int = 30,
) -> np.ndarray:
    """
    Compute a binary motion mask from two consecutive grayscale frames.

    The uint8 underflow problem
    ---------------------------
    NumPy uint8 arrays are stored as unsigned 8-bit integers in [0, 255].
    Subtracting two uint8 values wraps around modulo 256:

        frame_t[y, x]         = 10   (dark pixel, current frame)
        frame_t_minus_1[y, x] = 200  (bright pixel, previous frame)
        difference (uint8)    = 10 - 200  =>  66  (wraps: 256 - 190 = 66)

    The correct absolute difference is 190, not 66.  Direct uint8
    subtraction silently produces a wrong positive value wherever
    frame_t < frame_t_minus_1, masking real motion as a small number
    and corrupting the threshold decision.

    Safe solution
    -------------
    Cast both frames to int16 (signed, range -32768..32767) before
    subtracting so that no underflow can occur, then apply np.abs to
    produce the true absolute difference, and finally cast back to uint8.
    The entire pipeline is a sequence of vectorised NumPy operations
    with no Python-level loop:

        diff = |int16(frame_t) - int16(frame_t_minus_1)|

    Thresholding is then a single boolean comparison multiplied by 255:

        mask(x,y) = 255  if  diff(x,y) >= threshold,  else 0

    Parameters
    ----------
    frame_t        : np.ndarray  shape (H, W), dtype uint8 — current frame
    frame_t_minus_1: np.ndarray  shape (H, W), dtype uint8 — previous frame
    threshold      : int  intensity-difference threshold in [0, 255]

    Returns
    -------
    np.ndarray  shape (H, W), dtype uint8, values in {0, 255}
    """
    if frame_t.shape != frame_t_minus_1.shape:
        raise ValueError(
            f"Frame shapes must match: {frame_t.shape} vs {frame_t_minus_1.shape}."
        )
    if frame_t.ndim != 2:
        raise ValueError("Frames must be 2-D grayscale arrays.")

    # Cast to signed int16 BEFORE subtraction to prevent uint8 underflow
    diff = np.abs(
        frame_t.astype(np.int16) - frame_t_minus_1.astype(np.int16)
    )                                        # shape (H, W), dtype int16

    # Single vectorised threshold: boolean * 255 -> {0, 255}
    return (diff >= threshold).astype(np.uint8) * 255


# ============================================================
#  2.  Colour-Based Object Tracking
# ============================================================

def color_mask(
    hsv_image: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> np.ndarray:
    """
    Create a binary mask for pixels within an HSV colour range.

    This replaces cv2.inRange with pure NumPy logical operations.

    A pixel at (y, x) is foreground iff its HSV triple
    (H, S, V) satisfies ALL three simultaneous inequalities:

        lower_bound[0] <= H(y,x) <= upper_bound[0]   (Hue)
        lower_bound[1] <= S(y,x) <= upper_bound[1]   (Saturation)
        lower_bound[2] <= V(y,x) <= upper_bound[2]   (Value)

    This is equivalent to testing membership in the axis-aligned 3-D
    box [lower_bound, upper_bound] in HSV space.  Each inequality is
    evaluated as a vectorised NumPy comparison producing a (H, W) bool
    array; the three pairs are combined with bitwise AND (&) to enforce
    the simultaneous requirement without any Python-level loop.

    Hue wrap-around note
    --------------------
    Hue is periodic (red spans both 0° and 360°).  If lower_bound[0]
    > upper_bound[0] (e.g., lower=330, upper=30 for red), call this
    function twice with two non-wrapping intervals and OR the results.

    Parameters
    ----------
    hsv_image   : np.ndarray  shape (H, W, 3), float32
                  Channel 0: H in [0, 360),  1: S in [0,1],  2: V in [0,1]
                  (output of mycv.rgb_to_hsv)
    lower_bound : array-like  shape (3,)  — [H_min, S_min, V_min]
    upper_bound : array-like  shape (3,)  — [H_max, S_max, V_max]

    Returns
    -------
    np.ndarray  shape (H, W), dtype uint8, values in {0, 255}
    """
    if hsv_image.ndim != 3 or hsv_image.shape[2] != 3:
        raise ValueError(f"Expected HSV image (H, W, 3), got {hsv_image.shape}.")

    lb = np.asarray(lower_bound, dtype=np.float32)
    ub = np.asarray(upper_bound, dtype=np.float32)

    H = hsv_image[..., 0]   # (H_img, W_img)
    S = hsv_image[..., 1]
    V = hsv_image[..., 2]

    # Three simultaneous range tests, combined with bitwise AND
    mask = (
        (H >= lb[0]) & (H <= ub[0]) &
        (S >= lb[1]) & (S <= ub[1]) &
        (V >= lb[2]) & (V <= ub[2])
    )                                        # (H_img, W_img) bool

    return mask.astype(np.uint8) * 255


def calculate_centroid(
    binary_mask: np.ndarray,
) -> tuple:
    """
    Calculate the 2-D centroid (centre of mass) of a binary mask.

    Mathematical definition
    -----------------------
    Given a binary mask M : Omega -> {0, 1} with foreground pixel set

        F = { (y, x) in Omega : M(y, x) = 1 }

    the centroid is the expected value of the coordinate distribution:

        x_c = (1/|F|)  *  sum_{(y,x) in F}  x
        y_c = (1/|F|)  *  sum_{(y,x) in F}  y

    This is the discrete 2-D analogue of the first raw moment normalised
    by the zeroth moment (total mass), identical to the arithmetic mean
    of all foreground pixel (x, y) positions.

    Implementation
    --------------
    np.nonzero(mask) returns the row (y) and column (x) indices of all
    non-zero pixels in two 1-D arrays of length |F|.  Computing the
    mean of each array with np.mean produces the centroid in O(|F|)
    time with no Python-level loop over pixels.

    Division-by-zero safety
    -----------------------
    When the mask is entirely black (|F| = 0) no centroid is defined.
    The function returns (-1, -1) to signal this to the caller rather
    than raising a ZeroDivisionError.

    Parameters
    ----------
    binary_mask : np.ndarray  shape (H, W), dtype uint8, values in {0, 255}

    Returns
    -------
    (cx, cy) : tuple of float  — centroid coordinates (column, row),
               or (-1, -1) if the mask is completely black.
    """
    if binary_mask.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got shape {binary_mask.shape}.")

    # Indices of all foreground pixels
    ys, xs = np.nonzero(binary_mask)         # each shape (|F|,)

    if xs.size == 0:
        return (-1, -1)                      # mask is entirely black

    cx = float(np.mean(xs))                  # mean column index = x_c
    cy = float(np.mean(ys))                  # mean row index    = y_c
    return (cx, cy)


# ============================================================
#  3.  Temporal Smoothing — Circular Buffer Frame Averager
# ============================================================

class TemporalSmoother:
    """
    Reduce temporal video noise by averaging the last N frames.

    Why a circular (ring) buffer?
    -----------------------------
    A naive implementation appends new frames to a list and discards the
    oldest:
        frames.append(new_frame)
        frames.pop(0)
        average = np.mean(frames, axis=0)

    list.pop(0) requires O(N) pointer shifts to move every element one
    position forward.  Appending reallocates when the list grows.  For a
    1080p RGB video at 30 fps with N = 10 frames this is ~186 MB of data
    to shuffle on every single frame — entirely unnecessary.

    The circular buffer pre-allocates a fixed 4-D tensor of shape
    (N, H, W, C) once at initialisation.  A single integer write_idx
    tracks where the next frame goes:

        write_idx = frame_count % N

    Writing a new frame is a single O(H*W*C) in-place assignment:

        buffer[write_idx] = new_frame

    The modulo operation maps frame_count onto [0, N-1] cyclically, so
    the oldest frame is always at position (write_idx + 1) % N and the
    newest at write_idx — without ever moving any data.  This gives
    O(1) amortised index arithmetic and one O(H*W*C) write per update,
    compared to O(N * H * W * C) for the list-pop approach.

    Averaging
    ---------
    The smoothed frame is the arithmetic mean across the temporal axis:

        I_smooth = (1/N) * sum_{k=0}^{N-1} buffer[k]

    Computed as np.mean(buffer, axis=0) in float32 to avoid uint8
    overflow (255 * N exceeds uint8 range for N > 1), then cast back
    to uint8 after clipping.

    Parameters
    ----------
    n_frames : int  number of frames to average (buffer capacity)
    height   : int  frame height in pixels
    width    : int  frame width in pixels
    channels : int  number of channels — 1 for grayscale, 3 for RGB
    """

    def __init__(
        self,
        n_frames: int,
        height: int,
        width: int,
        channels: int = 3,
    ) -> None:
        if n_frames < 1:
            raise ValueError("n_frames must be >= 1.")

        self.N        = n_frames
        self.H        = height
        self.W        = width
        self.C        = channels
        self._count   = 0          # total frames seen (never resets)

        # Pre-allocate the 4-D ring buffer in float32 to avoid overflow
        # during accumulation.  Initialised to zero (black frames).
        shape = (n_frames, height, width) if channels == 1 else \
                (n_frames, height, width, channels)
        self._buffer = np.zeros(shape, dtype=np.float32)

    @property
    def is_filled(self) -> bool:
        """True once at least N frames have been written."""
        return self._count >= self.N

    def update(self, new_frame: np.ndarray) -> np.ndarray:
        """
        Insert a new frame into the ring buffer and return the
        temporally-smoothed result.

        The insertion index is:
            write_idx = total_frames_seen % N

        This cycles through [0, N-1] indefinitely, overwriting the
        oldest slot on each pass.

        Parameters
        ----------
        new_frame : np.ndarray  shape (H, W) or (H, W, C), dtype uint8

        Returns
        -------
        np.ndarray  shape matching new_frame, dtype uint8
            Per-pixel mean of the last min(N, frames_seen) frames.
        """
        expected_shape = (self.H, self.W) if self.C == 1 else \
                         (self.H, self.W, self.C)
        if new_frame.shape != expected_shape:
            raise ValueError(
                f"Expected frame shape {expected_shape}, "
                f"got {new_frame.shape}."
            )

        # Circular write: O(1) index arithmetic + one array assignment
        write_idx = self._count % self.N
        self._buffer[write_idx] = new_frame.astype(np.float32)
        self._count += 1

        # Average over the temporal axis (axis=0) in float32
        # — no uint8 overflow possible (max value = 255 * N)
        n_valid = min(self._count, self.N)   # avoid averaging empty slots
        if n_valid == self.N:
            # All slots filled: average the entire buffer
            smoothed = self._buffer.mean(axis=0)
        else:
            # Buffer not yet full: average only the written slots
            smoothed = self._buffer[:n_valid].mean(axis=0)

        return np.clip(smoothed, 0, 255).astype(np.uint8)

    def reset(self) -> None:
        """Clear the buffer and reset the frame counter."""
        self._buffer[:] = 0.0
        self._count = 0
