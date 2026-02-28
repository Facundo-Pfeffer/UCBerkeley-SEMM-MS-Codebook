from __future__ import annotations

"""
Iterative equivalent-SDOF response for the CE223 base-isolator system.

The goal is to:
- Map isolator hysteresis metrics (K1, zeta_eff vs shear strain) to an
  equivalent linear SDOF representation, and
- Iterate between time-domain response (Newmark) and updated shear strain
  until the assumed strain is consistent with the deformation obtained
  under the ground motion.

This module is intentionally generic: it does not hard-code a particular
ground motion file. Callers provide:
- ground acceleration history ug_ddot(t),
- time step dt,
- effective isolated mass m (consistent units),
and receive:
- iteration records (strain, K1, zeta, U_max per pass),
- final time histories u(t), u_dot(t), u_ddot(t).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from axis_limits_config import get_axis_limits
from isolator_curve_digitizer import IsolatorCurveDigitizer
from isolator_metrics import IsolatorMetrics, compute_isolator_metrics
from newmark_sdof import newmark_sdof


BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = BASE_DIR / "input_diagrams"

# Effective rubber thickness of the isolator stack [inches].
# Shear strain γ (%) and peak displacement U_max [in] are related by
#     γ = 100 * U_max / H_IN.
RUBBER_THICKNESS_IN = 3.25

# Number of HDR bearings in the isolation system (problem statement: 4 bearings).
# Hysteresis curves give per-bearing K1; system stiffness for the SDOF is NUM_BEARINGS * K1.
NUM_BEARINGS = 4

# Match strain labels and numeric values used in the dashboard.
STRAIN_LABELS: Dict[str, str] = {
    "strain10.png": "ε ≈ 9.8%",
    "strain74.png": "ε ≈ 74%",
    "strain124.png": "ε ≈ 124%",
    "strain180.png": "ε ≈ 180%",
}

STRAIN_PERCENT: Dict[str, float] = {
    "strain10.png": 9.8,
    "strain74.png": 74.0,
    "strain124.png": 124.0,
    "strain180.png": 180.0,
}

# Target peak displacements U0 [in] used to calibrate the digitized curves.
KNOWN_MAX_DISP: Dict[str, float] = {
    "strain10.png": 0.32,
    "strain74.png": 2.40,
    "strain124.png": 4.00,
    "strain180.png": 5.85,
}


@dataclass
class IsolatorLibraryEntry:
    strain_percent: float
    U0_in: float
    K1: float
    zeta_eff: float


def _load_isolator_library() -> List[IsolatorLibraryEntry]:
    """
    Compute (γ, U0, K1, zeta_eff) from the digitized isolator curves.

    This mirrors the logic used in the isolator dashboard so that the
    equivalent-SDOF properties are consistent with the reported metrics.
    """
    ordered_names = sorted(
        [
            name
            for name in STRAIN_PERCENT
            if (IMG_DIR / name).exists() and get_axis_limits(name, IMG_DIR) is not None
        ],
        key=lambda n: STRAIN_PERCENT.get(n, 0.0),
    )

    entries: List[IsolatorLibraryEntry] = []

    for name in ordered_names:
        limits = get_axis_limits(name, IMG_DIR)
        if limits is None:
            continue

        img = IMG_DIR / name
        if not img.exists():
            continue

        digitizer = IsolatorCurveDigitizer(limits, min_bin_density=3)
        pts = digitizer.digitize(
            img,
            max_points=16000,
            shuffle=True,
            known_max_displacement=KNOWN_MAX_DISP.get(name),
        )

        # Apply the same small noise filter used in the dashboard for strain74.
        if name == "strain74.png":
            noise1 = (pts[:, 0] < -2.3) & (pts[:, 1] > -2.46)
            noise2 = (pts[:, 0] < -1.9) & (pts[:, 0] > -2.0) & (pts[:, 1] > -1.5)
            pts = pts[~(noise1 | noise2)]

        metrics: IsolatorMetrics = compute_isolator_metrics(pts)
        entries.append(
            IsolatorLibraryEntry(
                strain_percent=float(STRAIN_PERCENT[name]),
                U0_in=float(metrics.U0),
                K1=float(metrics.K1),
                zeta_eff=float(metrics.zeta_eff),
            )
        )

    if not entries:
        raise RuntimeError(
            "No isolator images/axis limits found while building the equivalent-SDOF library. "
            "Ensure input_diagrams/strain*.png and axis_limits_config are available."
        )

    return entries


def build_equivalent_property_maps() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build interpolation-ready arrays for γ [%] → (K1, zeta_eff).

    Returns
    -------
    strains : np.ndarray
        Shear strains in percent (sorted).
    K1_vals : np.ndarray
        Storage stiffness K1 corresponding to each strain (same units as
        the isolator metrics, e.g. kip/in).
    zeta_vals : np.ndarray
        Effective damping ratios ζ_eff corresponding to each strain.
    """
    entries = _load_isolator_library()
    strains = np.array([e.strain_percent for e in entries], dtype=float)
    K1_vals = np.array([e.K1 for e in entries], dtype=float)
    zeta_vals = np.array([e.zeta_eff for e in entries], dtype=float)
    return strains, K1_vals, zeta_vals


@dataclass
class IterationRecord:
    iteration: int
    gamma_percent: float
    U_max_in: float
    K1: float
    zeta_eff: float


def equivalent_properties_from_displacement(
    U_max_in: float,
    m: float,
    strains: np.ndarray,
    K1_vals: np.ndarray,
    zeta_vals: np.ndarray,
    H_in: float,
) -> Tuple[float, float, float, float]:
    """
    Given a trial peak displacement U_max [in], compute:
    - shear strain γ [%],
    - equivalent K1(γ),
    - equivalent damping ratio ζ_eff(γ),
    - viscous coefficient c for the SDOF (assuming classical viscous damping).

    Parameters
    ----------
    U_max_in : float
        Trial peak relative displacement in inches.
    m : float
        Effective isolated mass (consistent units with k and c).
    strains, K1_vals, zeta_vals : np.ndarray
        Lookup arrays from build_equivalent_property_maps().
    H_in : float
        Effective rubber thickness in inches.
    """
    if H_in <= 0.0:
        raise ValueError("H_in must be positive.")

    gamma = 100.0 * U_max_in / H_in  # [%]
    # Clamp γ to the range of available data for interpolation.
    g_clamped = float(np.clip(gamma, float(strains.min()), float(strains.max())))

    K1 = float(np.interp(g_clamped, strains, K1_vals))
    zeta = float(np.interp(g_clamped, strains, zeta_vals))

    # Equivalent viscous coefficient for the SDOF, using K1 as the elastic stiffness.
    # c = 2 ζ ω_n m,  with ω_n = sqrt(k/m).
    if m <= 0.0:
        raise ValueError("Mass m must be positive.")
    omega_n = float(np.sqrt(K1 / m))
    c = 2.0 * zeta * omega_n * m

    return gamma, K1, zeta, c


def iterate_isolator_response(
    ug_ddot: np.ndarray,
    dt: float,
    m: float,
    U_guess_in: float,
    max_iter: int = 10,
    tol_rel: float = 0.002,
) -> Tuple[List[IterationRecord], np.ndarray]:
    """
    Perform an iterative equivalent-SDOF analysis for the base-isolated system.

    Starting from an initial guess for peak displacement U_guess_in, this
    routine:
    1. Builds an equivalent SDOF (K1, ζ_eff, c) from the isolator library.
    2. Solves m u¨ + c u˙ + K1 u = -m u¨_g(t) with Newmark's method.
    3. Extracts a new peak displacement U_max from the response.
    4. Repeats until U_max converges within tol_rel or max_iter is reached.

    Returns
    -------
    records : list[IterationRecord]
        Per-iteration summary (γ, U_max, K1, ζ_eff).
    response : np.ndarray
        Final Newmark response array of shape (n, 3) with columns (u, v, a).
    """
    ug_ddot = np.asarray(ug_ddot, dtype=float).ravel()
    n = ug_ddot.size
    if n == 0:
        raise ValueError("ug_ddot must contain at least one time point.")

    strains, K1_vals, zeta_vals = build_equivalent_property_maps()
    H_in = RUBBER_THICKNESS_IN

    records: List[IterationRecord] = []
    U_trial = float(U_guess_in)
    response = None

    for it in range(1, max_iter + 1):
        # Properties used for this iteration come from the current trial
        # peak displacement U_trial.
        gamma, K1, zeta, c = equivalent_properties_from_displacement(
            U_trial, m, strains, K1_vals, zeta_vals, H_in
        )

        # System stiffness = 4 × per-bearing K1 (four HDR bearings in parallel).
        # ζ_eff is unchanged for the system; c_system = 2 ζ √(k_system m).
        k_system = NUM_BEARINGS * K1
        omega_n = float(np.sqrt(k_system / m))
        c_system = 2.0 * zeta * omega_n * m

        p = -m * ug_ddot  # Effective force in relative coordinates.

        resp = newmark_sdof(m, k_system, c_system, p, dt, u0=0.0, v0=0.0, method="constant")
        u = resp[:, 0]
        U_max = float(np.max(np.abs(u)))

        # Record the trial pair (γ, U_trial) and associated properties.
        records.append(
            IterationRecord(
                iteration=it,
                gamma_percent=100.0 * U_trial / H_in if H_in > 0.0 else gamma,
                U_max_in=U_trial,
                K1=K1,
                zeta_eff=zeta,
            )
        )

        # Check convergence based on change in peak displacement.
        if it > 1:
            rel_change = abs(U_max - U_trial) / max(U_trial, 1e-8)
            if rel_change <= tol_rel:
                response = resp
                break

        U_trial = U_max
        response = resp

    if response is None:
        # Should not happen, but keep type checker happy.
        response = np.zeros((n, 3), dtype=float)

    return records, response

