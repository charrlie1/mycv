"""
mycv.calibration
=================
Projective geometry: homography estimation via Direct Linear Transformation
(DLT), with point normalisation for numerical stability and RANSAC for
robustness to outlier correspondences.

Functions
---------
normalize_points     : Isotropic point normalisation (Hartley normalisation)
solve_homography_dlt : Normalised DLT homography estimation via SVD
reprojection_error   : Per-point reprojection error under a homography
ransac_homography    : Outlier-robust homography estimation via RANSAC

Mathematical background
------------------------
A homography H is a 3x3 matrix mapping one projective plane to another:

    p' ~ H p        (~ denotes equality up to scale, since points are
                       homogeneous: [x, y, 1])

Direct Linear Transformation (DLT) rearranges each correspondence
p_i <-> p_i' into two linear equations in the 9 unknown entries of H,
stacks all correspondences into a matrix A, and solves A h = 0 via SVD
(h is the singular vector for the smallest singular value — the
least-squares null-space solution).

Point normalisation (Hartley, 1997) is essential for numerical
conditioning: raw pixel coordinates can span hundreds to thousands of
units, which makes the entries of A wildly different in magnitude and
destabilises the SVD. Translating each point set to have its centroid
at the origin and scaling so the mean distance from the origin is
sqrt(2) puts all coordinates on a comparable scale before solving, and
the resulting H is un-normalised afterward.

Real point correspondences (e.g. from feature matching) generally
contain outliers. RANSAC repeatedly fits a homography from a minimal
random sample of 4 correspondences, counts how many of ALL
correspondences agree with it within a reprojection-error threshold
(inliers), and keeps the model with the most inliers — then refits
using all of that model's inliers for a final, outlier-free estimate.
"""

import numpy as np


def normalize_points(points: np.ndarray) -> tuple:
    """
    Isotropic point normalisation for numerically stable DLT.

    Translates the point set so its centroid is at the origin, then
    scales isotropically so the mean distance from the origin is
    sqrt(2) (i.e. an "average" point sits at roughly (1, 1)).

    Parameters
    ----------
    points : np.ndarray  shape (N, 2) — (x, y) coordinates

    Returns
    -------
    (normalized_points, T) : tuple
        normalized_points : np.ndarray  shape (N, 2)
        T                 : np.ndarray  shape (3, 3) — the similarity
            transform applied, in homogeneous coordinates, so that
            un-normalising a homography H_n computed from normalized
            points is: H = inv(T_dst) @ H_n @ T_src
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be shape (N, 2), got {pts.shape}.")

    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.sqrt((shifted ** 2).sum(axis=1)).mean()
    scale = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0

    T = np.array([
        [scale, 0.0,   -scale * centroid[0]],
        [0.0,   scale, -scale * centroid[1]],
        [0.0,   0.0,   1.0],
    ], dtype=np.float64)

    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    hom = np.hstack([pts, ones])                 # (N, 3)
    normalized = (T @ hom.T).T[:, :2]

    return normalized, T


def solve_homography_dlt(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """
    Estimate a 3x3 homography H such that dst ~ H @ src, via normalised DLT.

    Derivation
    ----------
    For p ~ [x, y, 1] and p' ~ [x', y', 1] with p' = H p (up to scale),
    the cross product p' x (H p) = 0 (since they're parallel vectors)
    expands into two independent linear equations per correspondence
    (the third is a linear combination of the first two):

        [-x, -y, -1,  0,  0,  0,  x'x, x'y, x'] h = 0
        [ 0,  0,  0, -x, -y, -1,  y'x, y'y, y'] h = 0

    where h = vec(H) is the 9 unknowns. Stacking all N correspondences
    gives a (2N, 9) matrix A; h is the right singular vector of A
    corresponding to its smallest singular value (the least-squares
    solution to A h = 0 subject to ||h|| = 1).

    Parameters
    ----------
    src_points, dst_points : np.ndarray  shape (N, 2), N >= 4 — matched points

    Returns
    -------
    np.ndarray  shape (3, 3) — homography H, normalised so H[2, 2] == 1
    """
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    if src.shape != dst.shape:
        raise ValueError("src_points and dst_points must have the same shape.")
    n = src.shape[0]
    if n < 4:
        raise ValueError(f"Need at least 4 point correspondences, got {n}.")

    src_n, T_src = normalize_points(src)
    dst_n, T_dst = normalize_points(dst)

    A = np.zeros((2 * n, 9), dtype=np.float64)
    for i in range(n):
        x, y = src_n[i]
        xp, yp = dst_n[i]
        A[2 * i]     = [-x, -y, -1, 0, 0, 0, xp * x, xp * y, xp]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, yp * x, yp * y, yp]

    _, _, Vt = np.linalg.svd(A)
    H_n = Vt[-1].reshape(3, 3)

    H = np.linalg.inv(T_dst) @ H_n @ T_src
    return H / H[2, 2]


def reprojection_error(H: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """
    Per-point Euclidean reprojection error under homography H.

        error_i = || (H p_i) / w_i  -  p'_i ||_2

    where (x_tilde, y_tilde, w_tilde) = H @ [x_i, y_i, 1] and the
    projective division by w_tilde converts back to Cartesian coordinates.

    Parameters
    ----------
    H          : np.ndarray  shape (3, 3)
    src_points : np.ndarray  shape (N, 2)
    dst_points : np.ndarray  shape (N, 2)

    Returns
    -------
    np.ndarray  shape (N,) — per-point reprojection error
    """
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)

    ones = np.ones((src.shape[0], 1), dtype=np.float64)
    hom = np.hstack([src, ones])                  # (N, 3)
    proj = (H @ hom.T).T                           # (N, 3)

    w = np.where(np.abs(proj[:, 2]) < 1e-12, 1e-12, proj[:, 2])
    proj_xy = proj[:, :2] / w[:, np.newaxis]

    return np.sqrt(((proj_xy - dst) ** 2).sum(axis=1))


def ransac_homography(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    n_iterations: int = 1000,
    reprojection_threshold: float = 3.0,
    min_inliers: int = 4,
    seed: int = None,
) -> tuple:
    """
    Robustly estimate a homography from correspondences that may contain
    outliers, using RANSAC (RANdom SAmple Consensus).

    Algorithm
    ---------
    Repeat `n_iterations` times:
        1. Randomly sample 4 correspondences (the minimal set for DLT).
        2. Fit a candidate H from just those 4 via `solve_homography_dlt`.
        3. Compute `reprojection_error` for ALL N correspondences under
           this candidate H; count inliers (error < threshold).
        4. Keep the candidate with the most inliers seen so far.
    Finally, refit H using DLT on ALL inliers of the best candidate
    (least-squares refinement using every agreeing point, not just the
    minimal sample that found it).

    Parameters
    ----------
    src_points, dst_points : np.ndarray  shape (N, 2), N >= 4
    n_iterations            : int    number of random samples to try
    reprojection_threshold  : float  max reprojection error (pixels) to
                              count a correspondence as an inlier
    min_inliers             : int    minimum inlier count to accept a
                              result; raises RuntimeError otherwise
    seed                    : int, optional — RNG seed for reproducibility

    Returns
    -------
    (H, inlier_mask) : tuple
        H            : np.ndarray  shape (3, 3) — refined homography
        inlier_mask  : np.ndarray  shape (N,), bool — inliers of the
                       final refined H under `reprojection_threshold`
    """
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    n = src.shape[0]
    if n < 4:
        raise ValueError(f"Need at least 4 point correspondences, got {n}.")

    rng = np.random.default_rng(seed)

    best_count = -1
    best_inliers = None

    for _ in range(n_iterations):
        idx = rng.choice(n, 4, replace=False)
        try:
            H_candidate = solve_homography_dlt(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue

        errors = reprojection_error(H_candidate, src, dst)
        inliers = errors < reprojection_threshold
        count = int(inliers.sum())

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < min_inliers:
        raise RuntimeError(
            f"RANSAC failed to find a homography with at least "
            f"{min_inliers} inliers (best: {max(best_count, 0)})."
        )

    # Final refit using all inliers of the best model
    H_refined = solve_homography_dlt(src[best_inliers], dst[best_inliers])
    final_errors = reprojection_error(H_refined, src, dst)
    final_inliers = final_errors < reprojection_threshold

    return H_refined, final_inliers
