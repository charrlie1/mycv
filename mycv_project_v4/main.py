
import numpy as np
from PIL import Image
import mycv



def save(array: np.ndarray, path: str) -> None:
    Image.fromarray(array).save(path)
    print(f"        -> {path}")


def to_u8(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)



def main() -> None:
    print(f"\nmycv v{mycv.__version__} by {mycv.__author__}")
    print("=" * 60)


    print("\n[  I/O  ] Loading test_image.jpg ...")
    rgb = np.array(Image.open("test_image.jpg").convert("RGB"), dtype=np.uint8)
    H, W = rgb.shape[:2]
    print(f"          Shape: {rgb.shape}")

    # ── v2/v3 pipeline (outputs 01-18) ───────────────────────────────────
    gray     = mycv.rgb_to_grayscale(rgb)
    save(gray,                                     "output_01_grayscale.jpg")
    save(mycv.histogram_equalize(gray),            "output_02_equalized.jpg")
    hsv      = mycv.rgb_to_hsv(rgb)
    save((hsv[...,2]*255).astype(np.uint8),        "output_03_hsv_value.jpg")
    save((hsv[...,1]*255).astype(np.uint8),        "output_04_hsv_saturation.jpg")
    edges_d  = mycv.sobel_edge_detection(gray)
    edge_map = edges_d["magnitude"]
    Gx, Gy   = edges_d["Gx"], edges_d["Gy"]
    save(edge_map,                                 "output_05_edges.jpg")
    binary   = mycv.threshold(edge_map, tau=80)
    save(binary,                                   "output_06_threshold.jpg")
    se       = np.ones((3,3), dtype=np.bool_)
    save(mycv.dilate(binary, se),                  "output_07_dilated.jpg")
    save(mycv.erode(binary, se),                   "output_08_eroded.jpg")
    save(mycv.opening(binary, se),                 "output_09_opened.jpg")
    save(mycv.closing(binary, se),                 "output_10_closed.jpg")
    save(mycv.rotate_image(gray, 30.0),            "output_11_rotated.jpg")
    H_mat    = np.array([[1.,0.3,0.],[0.,1.,0.],[0.,0.001,1.]])
    save(mycv.warp_perspective(gray,H_mat,gray.shape), "output_12_homography.jpg")
    save(to_u8(mycv.harris_corner_response(Gx.astype(np.float64),
                                           Gy.astype(np.float64))),
                                                   "output_13_harris.jpg")
    hough    = mycv.hough_line_transform(binary, n_thetas=180)
    save(to_u8(hough["accumulator"].astype(np.float64)),
                                                   "output_14_hough_accum.jpg")
    template = gray[:32, :32]
    ncc_map  = mycv.match_template_ncc(gray, template)
    save(to_u8(ncc_map),                           "output_15_ncc_map.jpg")
    pyramid  = mycv.gaussian_pyramid(gray, levels=4)
    save(to_u8(pyramid[1]),                        "output_16_pyramid_l1.jpg")
    save(to_u8(pyramid[2]),                        "output_17_pyramid_l2.jpg")
    save(to_u8(pyramid[3]),                        "output_18_pyramid_l3.jpg")


    print("\n[TRACK  ] Motion detection (synthetic frame pair) ...")
    
    rng        = np.random.default_rng(seed=7)
    noise      = rng.integers(-40, 40, size=gray.shape, dtype=np.int16)
    frame_prev = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    motion     = mycv.compute_motion_mask(gray, frame_prev, threshold=30)
    save(motion, "output_19_motion_mask.jpg")
    moved_px   = int(np.count_nonzero(motion))
    print(f"          Moving pixels detected: {moved_px:,}  "
          f"({100*moved_px/(H*W):.1f}% of frame)")

    # ── v4 Step 2: Colour mask + centroid ────────────────────────────────
    print("\n[TRACK  ] HSV colour mask (warm hues: red/orange/yellow) ...")
    # Target: warm hues H in [0, 60) deg, saturated (S >= 0.3), bright (V >= 0.3)
    lower = np.array([0.0,  0.30, 0.30], dtype=np.float32)
    upper = np.array([60.0, 1.00, 1.00], dtype=np.float32)
    cmask = mycv.color_mask(hsv, lower, upper)
    save(cmask, "output_20_color_mask.jpg")
    cx, cy = mycv.calculate_centroid(cmask)
    if cx == -1:
        print("          Centroid: no matching pixels found.")
    else:
        print(f"          Centroid: x={cx:.1f} px,  y={cy:.1f} px")

    # ── v4 Step 3: Temporal smoothing ─────────────────────────────────────
    print("\n[TRACK  ] Temporal smoothing (N=5 ring buffer) ...")
    smoother = mycv.TemporalSmoother(n_frames=5, height=H, width=W, channels=3)
    for i in range(5):
        # Feed slightly noisy versions of the same frame to fill the buffer
        frame_noise = rng.integers(-15, 15, size=rgb.shape, dtype=np.int16)
        noisy_frame = np.clip(rgb.astype(np.int16) + frame_noise, 0, 255).astype(np.uint8)
        smoothed = smoother.update(noisy_frame)
    save(smoothed, "output_21_temporal_smooth.jpg")
    print(f"          Buffer filled: {smoother.is_filled}")
    print(f"          Output shape:  {smoothed.shape}  dtype: {smoothed.dtype}")

    print("\n" + "=" * 60)
    print("Pipeline complete. 21 output images written to disk.\n")


if __name__ == "__main__":
    main()
