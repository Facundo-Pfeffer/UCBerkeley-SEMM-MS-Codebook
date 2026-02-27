from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _polygon_area(poly: np.ndarray) -> float:
    """Signed area of a simple polygon given as (x, y) points."""
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    )


def _estimate_num_cycles(centered: np.ndarray, n_angle_bins: int = 72) -> int:
    """
    Estimate how many times the hysteresis loop was traced (overlaid cycles)
    by counting radius layers in each angle bin. In each bin, radii are sorted
    and clusters are separated by gaps above the 95th percentile of gaps.
    Median cluster count over bins = num_cycles. Same for all strain levels.
    """
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    radii = np.sqrt(np.sum(centered**2, axis=1))
    angles_norm = np.where(angles < 0, angles + 2 * np.pi, angles)
    bin_edges = np.linspace(0, 2 * np.pi, n_angle_bins + 1)
    cycle_counts = []
    for i in range(n_angle_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (angles_norm >= lo) & (angles_norm < hi) if i < n_angle_bins - 1 else (angles_norm >= lo) & (angles_norm <= hi)
        if not np.any(mask) or np.sum(mask) < 2:
            continue
        r_bin = np.sort(radii[mask])
        if len(r_bin) < 2:
            cycle_counts.append(1)
            continue
        gaps = np.diff(r_bin)
        gap_threshold = float(np.percentile(gaps, 95.0))
        n_clusters = 1 + int(np.sum(gaps > max(gap_threshold, 1e-12)))
        n_clusters = min(n_clusters, 8)
        cycle_counts.append(n_clusters)
    if not cycle_counts:
        return 1
    return max(1, int(round(np.median(cycle_counts))))


def compute_loop_area(
    points: np.ndarray,
    *,
    n_bins: int = 400,
    min_points_per_bin: int = 2,
) -> float:
    """
    Energy dissipated per cycle (area of one hysteresis loop) in the (u, F) plane.

    Standard, order-independent approximation used in practice:

      1. Project the loop onto the displacement axis u and discretize
         [u_min, u_max] into bins.
      2. In each bin, find the upper envelope F_max(u) and lower envelope
         F_min(u) of the cloud.
      3. Approximate the enclosed area as
             W_D ≈ ∑_bins (F_max - F_min) * Δu.

    This uses only the envelopes of the loop and is robust to:
      - Point ordering (no reliance on trace sequence),
      - Multiple overlaid cycles (repeating the same loop many times does
        not change F_max/F_min in each bin).
    The same algorithm is applied to all strain levels.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("compute_loop_area expects an (N, 2) array of points.")

    u = pts[:, 0]
    F = pts[:, 1]

    u_min = float(np.min(u))
    u_max = float(np.max(u))
    if not np.isfinite(u_min) or not np.isfinite(u_max) or u_max <= u_min:
        return 0.0

    n_bins = int(n_bins)
    if n_bins < 10:
        n_bins = 10

    bin_edges = np.linspace(u_min, u_max, n_bins + 1)
    bin_indices = np.digitize(u, bin_edges) - 1  # -> [0, n_bins-1]
    widths = bin_edges[1:] - bin_edges[:-1]

    area = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.count_nonzero(mask) < min_points_per_bin:
            continue
        f_slice = F[mask]
        f_top = float(np.max(f_slice))
        f_bottom = float(np.min(f_slice))
        area += (f_top - f_bottom) * float(widths[i])

    return abs(area)


def compute_envelope_point_mask(
    points: np.ndarray,
    *,
    n_bins: int = 400,
    min_points_per_bin: int = 2,
    atol: float = 1e-9,
) -> np.ndarray:
    """
    Return a boolean mask marking which points lie on the loop envelopes
    used for the area calculation.

    For each displacement bin, this selects all points whose force is at
    the local maximum or minimum (within a small tolerance). These are
    exactly the points that define F_max and F_min in ``compute_loop_area``.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("compute_envelope_point_mask expects an (N, 2) array of points.")

    u = pts[:, 0]
    F = pts[:, 1]

    u_min = float(np.min(u))
    u_max = float(np.max(u))
    if not np.isfinite(u_min) or not np.isfinite(u_max) or u_max <= u_min:
        return np.zeros(len(pts), dtype=bool)

    n_bins = int(n_bins)
    if n_bins < 10:
        n_bins = 10

    bin_edges = np.linspace(u_min, u_max, n_bins + 1)
    bin_indices = np.digitize(u, bin_edges) - 1  # -> [0, n_bins-1]

    used = np.zeros(len(pts), dtype=bool)
    for i in range(n_bins):
        idx = np.where(bin_indices == i)[0]
        if idx.size < min_points_per_bin:
            continue
        f_slice = F[idx]
        f_top = float(np.max(f_slice))
        f_bottom = float(np.min(f_slice))
        top_mask = f_slice >= f_top - atol
        bottom_mask = f_slice <= f_bottom + atol
        used[idx[top_mask | bottom_mask]] = True

    return used


@dataclass
class IsolatorMetrics:
    U0: float
    P0: float
    K0: float
    WD: float
    K2: float
    K1: float
    zeta_eff: float
    u0_point: Tuple[float, float]
    p0_point: Tuple[float, float]


def compute_isolator_metrics(points: np.ndarray, omega_over_omegan: float = 1.0) -> IsolatorMetrics:
    """
    Compute U0, P0, K0, W_D, K2, K1 and effective damping ratio zeta_eff
    from a digitized hysteresis cloud.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("compute_isolator_metrics expects an (N, 2) array.")

    u = pts[:, 0]
    F = pts[:, 1]

    # Maximum displacement and associated point.
    # We want the marker to sit on the *bottom* branch of the loop at the
    # extreme displacement (as in the original plots), not randomly on top
    # or bottom depending on point ordering.
    abs_u = np.abs(u)
    max_abs_u = float(np.max(abs_u))
    # All points whose |u| is within 0.1% of the maximum.
    near_max = np.where(abs_u >= max_abs_u * 0.999)[0]
    if near_max.size == 0:
        idx_u0 = int(np.argmax(abs_u))
    else:
        # Among near-maximum |u| points, pick the one with minimum force
        # (bottom of the loop).
        idx_u0 = int(near_max[np.argmin(F[near_max])])
    U0 = max_abs_u
    u0_point = (float(u[idx_u0]), float(F[idx_u0]))

    # Maximum force and associated point
    idx_p0 = int(np.argmax(np.abs(F)))
    P0 = float(abs(F[idx_p0]))
    p0_point = (float(u[idx_p0]), float(F[idx_p0]))

    K0 = P0 / U0 if U0 > 0 else float("nan")

    # Energy per cycle (kip·in) as area of loop
    WD = compute_loop_area(pts)

    # Loss stiffness
    K2 = WD / (np.pi * U0**2) if U0 > 0 else float("nan")

    # Storage stiffness from K0^2 = K1^2 + K2^2
    if np.isfinite(K0) and np.isfinite(K2) and K0 >= K2:
        K1 = float(np.sqrt(K0**2 - K2**2))
    else:
        K1 = float("nan")

    # Equivalent damping ratio; assume steady-state excitation with given ratio
    if K1 > 0 and omega_over_omegan > 0:
        zeta_eff = WD / (2.0 * np.pi * omega_over_omegan * K1 * U0**2)
    else:
        zeta_eff = float("nan")

    return IsolatorMetrics(
        U0=U0,
        P0=P0,
        K0=K0,
        WD=WD,
        K2=K2,
        K1=K1,
        zeta_eff=zeta_eff,
        u0_point=u0_point,
        p0_point=p0_point,
    )

