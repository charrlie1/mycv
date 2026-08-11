"""
mycv.tracking
=============
Pure-NumPy motion detection, colour-based object tracking,
and temporal video noise reduction.

Functions
---------
compute_motion_mask   : Frame-difference binary motion mask
color_mask            : HSV range filter (replaces cv2.inRange)
color_mask_hue_wrap    : color_mask that transparently handles hue wrap-around
calculate_centroid    : 2-D centre-of-mass of a binary mask
kalman_filter_predict : Predict next state using Kalman filter
kalman_filter_update  : Update Kalman state with measurement
mahalanobis_gate      : Statistical distance test for measurement gating

Classes
-------
TemporalSmoother      : Circular-buffer rolling-average denoiser
KalmanCentroidTracker : Kalman filter for smooth single-centroid prediction
MultiObjectKalmanTracker : Track multiple simultaneous objects with IDs

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
mean of all foreground pixel positions. For scenes with multiple
disjoint objects of the same colour, prefer
`morphology.label_connected_components` + `morphology.component_properties`
to get one centroid per object instead of a single averaged blob.

Temporal smoothing averages the last N frames along the time axis using
a pre-allocated circular (ring) buffer, which avoids O(N) memory shifts
on every update and achieves constant-time writes via modulo indexing.

Kalman filtering provides optimal recursive state estimation for linear
Gaussian systems. For centroid tracking, we model position and velocity
as a 4-D state vector [x, y, vx, vy] with constant-velocity dynamics.
The filter alternates between prediction (projecting state forward in
time) and update (correcting prediction with noisy measurements).
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
    Hue is periodic (red spans both 0 deg and 360 deg).  If
    lower_bound[0] > upper_bound[0] (e.g., lower=330, upper=30 for red),
    this function alone treats it as an empty (never-true) range,
    because a single AND-combined interval cannot represent a wrapped
    range. Use `color_mask_hue_wrap` instead, which detects the wrapped
    case automatically and ORs together the two non-wrapping sub-ranges.

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


def color_mask_hue_wrap(
    hsv_image: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> np.ndarray:
    """
    Hue-wrap-aware wrapper around `color_mask`.

    Hue wrap-around must be handled explicitly: colours like red straddle
    the 0/360 degree seam (e.g. lower=330, upper=30). Calling `color_mask`
    directly with lower > upper degenerately matches nothing, since a
    single AND'd interval [330, 30] is empty. This function detects that
    case and automatically ORs together the two non-wrapping sub-ranges
    [lower, 360) and [0, upper], so callers don't have to remember to
    special-case red (or any other hue that straddles the seam) at every
    call site.

    Parameters
    ----------
    hsv_image, lower_bound, upper_bound : see `color_mask`

    Returns
    -------
    np.ndarray  shape (H, W), dtype uint8, values in {0, 255}
    """
    lb = np.asarray(lower_bound, dtype=np.float32)
    ub = np.asarray(upper_bound, dtype=np.float32)

    if lb[0] <= ub[0]:
        return color_mask(hsv_image, lb, ub)

    mask_hi = color_mask(hsv_image, [lb[0], lb[1], lb[2]], [360.0, ub[1], ub[2]])
    mask_lo = color_mask(hsv_image, [0.0, lb[1], lb[2]], [ub[0], ub[1], ub[2]])
    return np.maximum(mask_hi, mask_lo)


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

    Multi-object caveat
    --------------------
    If `binary_mask` contains more than one disjoint blob (e.g. two
    same-colour objects visible at once), this single centroid will fall
    somewhere BETWEEN them, which is generally not a meaningful position
    for either object. Run `morphology.label_connected_components` first
    and compute a centroid per component (via
    `morphology.component_properties`) for robust multi-object tracking.

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


# ============================================================
#  4.  Kalman Filter for Centroid Tracking
# ============================================================

def kalman_filter_predict(
    state: np.ndarray,
    P: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
) -> tuple:
    """
    Predict the next state using the Kalman filter prediction step.

    Mathematical formulation
    ------------------------
    The Kalman filter maintains a Gaussian belief over the state vector
    characterized by mean (state estimate) and covariance (uncertainty):

        x_k ~ N(state, P)

    The prediction step projects this belief forward in time using the
    state transition model:

        state_pred = F @ state           (projected state estimate)
        P_pred     = F @ P @ F.T + Q     (projected covariance)

    where:
        - F is the state transition matrix encoding system dynamics
        - Q is the process noise covariance (model uncertainty)

    For constant-velocity motion tracking with state [x, y, vx, vy]:

        F = [[1, 0, dt, 0],      x_new   = x + dt*vx
             [0, 1, 0, dt],      y_new   = y + dt*vy
             [0, 0, 1, 0],       vx_new  = vx
             [0, 0, 0, 1]]       vy_new  = vy

    Parameters
    ----------
    state : np.ndarray  shape (4,) — current state [x, y, vx, vy]
    P     : np.ndarray  shape (4, 4) — current state covariance
    F     : np.ndarray  shape (4, 4) — state transition matrix
    Q     : np.ndarray  shape (4, 4) — process noise covariance

    Returns
    -------
    (state_pred, P_pred) : tuple
        state_pred : np.ndarray  shape (4,) — predicted state
        P_pred     : np.ndarray  shape (4, 4) — predicted covariance
    """
    # Project state forward: x_pred = F @ x
    state_pred = F @ state

    # Project covariance forward: P_pred = F @ P @ F.T + Q
    P_pred = F @ P @ F.T + Q

    return state_pred, P_pred


def kalman_filter_update(
    state_pred: np.ndarray,
    P_pred: np.ndarray,
    measurement: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    joseph_form: bool = True,
) -> tuple:
    """
    Update the predicted state with a new measurement (Kalman update step).

    Mathematical formulation
    ------------------------
    Given a noisy measurement z related to the state by the observation
    model z = H @ x + noise, the update step corrects the prediction:

    Innovation (measurement residual):
        y = z - H @ state_pred

    Innovation covariance:
        S = H @ P_pred @ H.T + R

    Optimal Kalman gain (minimizes posterior error covariance):
        K = P_pred @ H.T @ inv(S)

    Updated state estimate:
        state = state_pred + K @ y

    Updated covariance:
        P = (I - K @ H) @ P_pred                          [standard form]
        P = (I-KH) P_pred (I-KH).T + K R K.T               [Joseph form]

    The Kalman gain K automatically balances trust between prediction
    and measurement based on their relative uncertainties (P_pred vs R).

    Numerical stability
    --------------------
    K is obtained by solving the linear system S.T @ K.T = (P_pred @ H.T).T
    via `np.linalg.solve` rather than explicitly forming `inv(S)` — this
    is the standard numerically-preferred pattern (avoids computing a
    full matrix inverse just to immediately multiply by it). For a small
    2x2 innovation covariance the difference versus explicit inversion is
    minor, but it is best practice and costs nothing here.

    The Joseph-form covariance update is used by default. It is
    algebraically equal to the standard form only when K is the exact
    optimal gain; under floating-point rounding (or a suboptimal/gated
    gain) it remains guaranteed symmetric positive semi-definite, while
    the standard form can drift slightly asymmetric or lose positive-
    definiteness over many iterations. Pass `joseph_form=False` to use
    the cheaper standard form.

    Parameters
    ----------
    state_pred : np.ndarray  shape (4,) — predicted state from predict()
    P_pred     : np.ndarray  shape (4, 4) — predicted covariance
    measurement: np.ndarray  shape (2,) — observed centroid [x, y]
    H          : np.ndarray  shape (2, 4) — observation matrix
    R          : np.ndarray  shape (2, 2) — measurement noise covariance
    joseph_form: bool  use the numerically-robust Joseph-form covariance
                 update (default True)

    Returns
    -------
    (state, P) : tuple
        state : np.ndarray  shape (4,) — updated state estimate
        P     : np.ndarray  shape (4, 4) — updated covariance
    """
    # Innovation (residual between measurement and prediction)
    innovation = measurement - H @ state_pred

    # Innovation covariance: S = H @ P @ H.T + R
    S = H @ P_pred @ H.T + R

    # Kalman gain via linear solve rather than explicit inverse:
    #   K @ S = P_pred @ H.T   =>   S.T @ K.T = (P_pred @ H.T).T
    # S is symmetric (H P H.T + R, with R symmetric), so S.T == S.
    PHt = P_pred @ H.T
    K = np.linalg.solve(S, PHt.T).T

    # Updated state estimate: x = x_pred + K @ y
    state = state_pred + K @ innovation

    I = np.eye(state.shape[0])
    IKH = I - K @ H
    if joseph_form:
        P = IKH @ P_pred @ IKH.T + K @ R @ K.T
    else:
        P = IKH @ P_pred

    return state, P


def mahalanobis_gate(
    state_pred: np.ndarray,
    P_pred: np.ndarray,
    measurement: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
) -> float:
    """
    Squared Mahalanobis distance between a measurement and a Kalman
    prediction, for measurement gating.

        d^2 = innovation.T @ inv(S) @ innovation
        S   = H @ P_pred @ H.T + R

    A large d^2 means the measurement is statistically implausible given
    the filter's current belief (e.g. a false detection, or the tracker
    latching onto a different object). Gating rejects measurements whose
    d^2 exceeds a chi-squared critical value — for a 2-D measurement,
    d^2 > 9.21 corresponds to roughly the 99% confidence threshold.

    Parameters
    ----------
    state_pred, P_pred, measurement, H, R : see `kalman_filter_update`

    Returns
    -------
    float — squared Mahalanobis distance d^2
    """
    innovation = measurement - H @ state_pred
    S = H @ P_pred @ H.T + R
    return float(innovation @ np.linalg.solve(S, innovation))


class KalmanCentroidTracker:
    """
    Track a moving object's centroid using a Kalman filter.

    Why use a Kalman filter for centroid smoothing?
    -----------------------------------------------
    Simple temporal averaging (like TemporalSmoother) treats all frames
    equally and introduces latency proportional to the window size.
    In contrast, the Kalman filter:

    1. Provides optimal recursive estimation for linear-Gaussian systems
    2. Explicitly models object dynamics (e.g., constant velocity motion)
    3. Adapts to measurement quality via the Kalman gain
    4. Can predict position during temporary occlusions (no measurement)
    5. Has minimal latency — each measurement is processed immediately

    State-space model
    -----------------
    We use a 4-D state vector representing 2D position and velocity:

        state = [x, y, vx, vy]^T

    The constant-velocity motion model assumes:
        x(t+dt)  = x(t) + dt * vx(t)
        y(t+dt)  = y(t) + dt * vy(t)
        vx(t+dt) = vx(t)
        vy(t+dt) = vy(t)

    This yields the state transition matrix F (rebuilt per-call if `dt`
    varies — see `update`/`predict`):
        F = [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]

    The observation matrix H extracts only position from the state:
        H = [[1, 0, 0, 0],      [x]     [x]
             [0, 1, 0, 0]]  =>  [y]  =  [y]

    Prediction/update lifecycle
    -----------------------------
    Each frame calls EITHER `update(centroid)` (measurement available)
    OR `predict()` (no measurement — e.g. occlusion), never both for the
    same frame. Both methods COMMIT the prediction step to
    `self.state`/`self.P` internally via a single shared `_step_predict`
    helper, so repeated `predict()` calls during an occlusion correctly
    propagate the object forward frame by frame using its last known
    velocity, rather than re-predicting one step from the same stale
    state every time.

    Tuning parameters
    -----------------
    process_noise (Q): Controls how much we expect the target to deviate
                       from constant-velocity motion. Higher values make
                       the filter more responsive to sudden accelerations.

    measurement_noise (R): Represents expected centroid detection noise.
                          Higher values make the filter trust predictions
                          more than measurements (smoother but laggy).

    Parameters
    ----------
    process_noise    : float  variance of process noise (default: 1e-3)
    measurement_noise: float  variance of measurement noise (default: 1e-1)
    dt               : float  default time step between frames (default: 1.0)
    gate_threshold   : float or None  chi-squared squared-Mahalanobis gate
                       (see `mahalanobis_gate`). None disables gating
                       (default). A common 2-D choice is ~9.21 (99%).
    """

    def __init__(
        self,
        process_noise: float = 1e-3,
        measurement_noise: float = 1e-1,
        dt: float = 1.0,
        gate_threshold: float = None,
    ) -> None:
        if process_noise <= 0:
            raise ValueError("process_noise must be positive.")
        if measurement_noise <= 0:
            raise ValueError("measurement_noise must be positive.")

        self.dt = dt
        self.gate_threshold = gate_threshold

        # State vector: [x, y, vx, vy]
        self.state = np.zeros(4, dtype=np.float64)

        # State covariance (initial uncertainty)
        self.P = np.eye(4, dtype=np.float64) * 1.0

        # Process noise covariance
        self.process_noise_var = process_noise
        self.Q = np.eye(4, dtype=np.float64) * process_noise

        # Measurement noise covariance (base_R is the un-scaled reference
        # used by the adaptive-noise `confidence` argument in `update`)
        self.base_R = np.eye(2, dtype=np.float64) * measurement_noise
        self.R = self.base_R.copy()

        # Observation matrix (we only observe position, not velocity)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        self._initialized = False
        self._build_F(dt)
        self.last_gated = False   # True if the last update() rejected its measurement

    def _build_F(self, dt: float) -> None:
        """(Re)build the constant-velocity state transition matrix for a given dt."""
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

    def _step_predict(self, dt: float = None) -> None:
        """
        Run and COMMIT one prediction step. Shared by `update` and
        `predict` so the state always advances exactly once per frame,
        regardless of which one is called.
        """
        if dt is not None and dt != self.dt:
            self.dt = dt
            self._build_F(dt)
        self.state, self.P = kalman_filter_predict(self.state, self.P, self.F, self.Q)

    def initialize(self, initial_centroid: tuple) -> None:
        """
        Initialize the filter with an initial centroid measurement.

        Parameters
        ----------
        initial_centroid : tuple  (x, y) — first detected centroid
        """
        self.state[0] = initial_centroid[0]  # x
        self.state[1] = initial_centroid[1]  # y
        self.state[2] = 0.0                   # vx (unknown initially)
        self.state[3] = 0.0                   # vy (unknown initially)
        self.P = np.eye(4, dtype=np.float64) * 1.0
        self._initialized = True
        self.last_gated = False

    def update(self, centroid: tuple, dt: float = None, confidence: float = None) -> tuple:
        """
        Update the tracker with a new centroid measurement.

        Performs both prediction and update steps of the Kalman filter.
        If `gate_threshold` was set and this measurement's squared
        Mahalanobis distance from the prediction exceeds it, the
        measurement is REJECTED (treated as a false/implausible
        detection): the filter keeps the predicted state instead of
        being corrupted by it, and `self.last_gated` is set to True.

        Parameters
        ----------
        centroid   : tuple  (x, y) — measured centroid position
        dt         : float, optional — override this frame's time step
                     (for variable frame rate; see the class docstring)
        confidence : float in (0, 1], optional — adaptive measurement
                     noise. A smaller/noisier detection (e.g. a small
                     pixel_area mask) should pass a smaller confidence,
                     which scales R up (trust the measurement less):

                         R_effective = base_R / confidence

                     Omit to use the fixed `measurement_noise` from
                     construction.

        Returns
        -------
        (x, y) : tuple  smoothed/predicted centroid position
        """
        measurement = np.array(centroid, dtype=np.float64)

        if not self._initialized:
            self.initialize(centroid)
            return centroid

        self._step_predict(dt)

        R_eff = self.base_R / confidence if confidence is not None else self.base_R

        if self.gate_threshold is not None:
            d2 = mahalanobis_gate(self.state, self.P, measurement, self.H, R_eff)
            if d2 > self.gate_threshold:
                # Reject: keep the (already-committed) predicted state.
                self.last_gated = True
                return (float(self.state[0]), float(self.state[1]))

        self.last_gated = False
        self.R = R_eff
        self.state, self.P = kalman_filter_update(
            self.state, self.P, measurement, self.H, R_eff
        )

        return (float(self.state[0]), float(self.state[1]))

    def predict(self, dt: float = None) -> tuple:
        """
        Predict the next centroid position without a new measurement.

        Useful for handling temporary occlusions or dropped frames. Only
        performs the prediction step (no measurement update) — but,
        unlike a stateless prediction, this COMMITS the result to
        `self.state`/`self.P`. Calling `predict()` repeatedly across
        several frames therefore correctly propagates the object forward
        frame-by-frame at its last known velocity, instead of predicting
        the same one-step-ahead position from stale state every time.

        Parameters
        ----------
        dt : float, optional — override this frame's time step

        Returns
        -------
        (x, y) : tuple  predicted centroid position, or (-1, -1) if the
                 tracker has not yet been initialized.
        """
        if not self._initialized:
            return (-1.0, -1.0)

        self._step_predict(dt)
        return (float(self.state[0]), float(self.state[1]))

    def reset(self) -> None:
        """Reset the tracker to its initial uninitialized state."""
        self.state = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 1.0
        self.R = self.base_R.copy()
        self._initialized = False
        self.last_gated = False


class MultiObjectKalmanTracker:
    """
    Track multiple simultaneous objects, each with its own Kalman filter
    and a persistent track ID.

    This is the natural next step once `morphology.label_connected_components`
    can produce multiple per-frame detections: a single `KalmanCentroidTracker`
    can only follow one object, so tracking several requires (a) one filter
    per object and (b) an association step that decides which detection
    belongs to which existing track each frame.

    Association strategy
    ---------------------
    This implementation uses greedy nearest-centroid assignment: at each
    frame, every (track, detection) pair is scored by Euclidean distance
    between the track's predicted position and the detection; pairs are
    matched greedily from smallest to largest distance, skipping any pair
    that reuses an already-matched track or detection, and rejecting a
    match if the distance exceeds `max_match_distance`. This is O(T*D log(T*D))
    and is simple and fast; a Hungarian-algorithm (optimal bipartite
    assignment) implementation would improve match quality when many
    tracks are close together, at higher implementation cost — a natural
    future upgrade using the same track bookkeeping here.

    Track lifecycle
    -----------------
    - A detection that matches no existing track spawns a new track.
    - A track that matches no detection this frame gets a `predict()`-only
      step (see `KalmanCentroidTracker.predict`) and its `missed` counter
      incremented.
    - A track is deleted once `missed` exceeds `max_missed` consecutive
      frames without a matching detection.

    Parameters
    ----------
    max_match_distance : float  maximum centroid distance (pixels) for a
                          detection to be associated with an existing track
    max_missed         : int    consecutive missed frames before a track
                          is deleted
    process_noise, measurement_noise, dt : passed through to each new
                          `KalmanCentroidTracker`
    """

    def __init__(
        self,
        max_match_distance: float = 50.0,
        max_missed: int = 5,
        process_noise: float = 1e-3,
        measurement_noise: float = 1e-1,
        dt: float = 1.0,
    ) -> None:
        self.max_match_distance = max_match_distance
        self.max_missed = max_missed
        self._tracker_kwargs = dict(
            process_noise=process_noise, measurement_noise=measurement_noise, dt=dt,
        )
        self.tracks = {}     # track_id -> KalmanCentroidTracker
        self._missed = {}    # track_id -> consecutive missed-frame count
        self._next_id = 1

    def update(self, detections: list, dt: float = None) -> dict:
        """
        Associate this frame's detections with existing tracks, update
        matched tracks, predict-only unmatched tracks, spawn new tracks
        for unmatched detections, and prune stale tracks.

        Parameters
        ----------
        detections : list of (x, y) tuples — this frame's measured centroids
                     (e.g. one per connected component from
                     `morphology.component_properties`)
        dt         : float, optional — this frame's time step, forwarded
                     to every track's predict/update call

        Returns
        -------
        dict  track_id -> (x, y) current position, for every live track
        """
        track_ids = list(self.tracks.keys())

        # Build the pairwise distance matrix between predicted track
        # positions and this frame's detections.
        pairs = []
        if track_ids and detections:
            predicted = {tid: self.tracks[tid].predict(dt) for tid in track_ids}
            for ti, tid in enumerate(track_ids):
                px, py = predicted[tid]
                for di, (dx, dy) in enumerate(detections):
                    dist = float(np.hypot(px - dx, py - dy))
                    if dist <= self.max_match_distance:
                        pairs.append((dist, ti, di))
            pairs.sort(key=lambda p: p[0])

        matched_tracks = set()
        matched_dets = set()
        assignment = {}   # track_id -> detection_index
        for dist, ti, di in pairs:
            tid = track_ids[ti]
            if tid in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(tid)
            matched_dets.add(di)
            assignment[tid] = di

        # Update matched tracks with their assigned detection.
        # NOTE: predict() was already called above to build the distance
        # matrix, which commits the internal state — so we call the raw
        # Kalman update step directly here rather than update() (which
        # would otherwise predict a second time for this frame).
        for tid, di in assignment.items():
            tracker = self.tracks[tid]
            measurement = np.array(detections[di], dtype=np.float64)
            tracker.state, tracker.P = kalman_filter_update(
                tracker.state, tracker.P, measurement, tracker.H, tracker.base_R,
            )
            self._missed[tid] = 0

        # Unmatched existing tracks: already predict()-ed above; just
        # bump their missed counter (or handle the no-detections-at-all
        # case where predict() hasn't run yet).
        for tid in track_ids:
            if tid not in assignment:
                if not detections:
                    self.tracks[tid].predict(dt)
                self._missed[tid] = self._missed.get(tid, 0) + 1

        # Unmatched detections spawn new tracks.
        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            new_tracker = KalmanCentroidTracker(**self._tracker_kwargs)
            new_tracker.initialize(det)
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = new_tracker
            self._missed[tid] = 0

        # Prune stale tracks.
        for tid in list(self.tracks.keys()):
            if self._missed.get(tid, 0) > self.max_missed:
                del self.tracks[tid]
                del self._missed[tid]

        return {tid: (float(t.state[0]), float(t.state[1])) for tid, t in self.tracks.items()}

    def reset(self) -> None:
        """Remove all tracks and reset ID assignment."""
        self.tracks.clear()
        self._missed.clear()
        self._next_id = 1
