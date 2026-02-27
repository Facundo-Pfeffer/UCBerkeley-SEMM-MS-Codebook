from __future__ import annotations

"""
Frequency-domain SDOF response for CE223 hysteresis models (A, B, C).

This module implements the FFT-based base-excitation analysis described in the
assignment, using the model-dependent dynamic stiffness

    H(ω) = -m / (R̂(ω) - m ω²),

where R̂(ω) is the complex dynamic stiffness for each damping model:

  - Model A (Kelvin–Voigt):          R̂(ω) = k + i c ω
  - Model B (hysteretic idealization):R̂(ω) = k (1 + i δ sgn(ω))
  - Model C (fractional Kelvin–Voigt):
        R̂(ω) = k_C + c_α (i ω)^α
    with the two-sided definition
        (i ω)^α = |ω|^α [cos(πα/2) + i sgn(ω) sin(πα/2)]
    to enforce conjugate symmetry for real-valued time histories.

The ground motion is read from a PEER-format .AT2 file (Kobe KBU090) and
converted to m/s² for consistency with the SDOF parameters in sdof_hysteresis.
"""

from pathlib import Path
from typing import Dict, Tuple

import math
import numpy as np


def load_peer_at2_to_mps2(path: Path) -> Tuple[np.ndarray, float]:
    """
    Load a PEER .AT2 acceleration record and return (ug_ddot, dt).

    The file is expected to contain header lines starting with '%' including
    one of the form

        %NPTS= XXXX, DT=  .0100 SEC,

    followed by acceleration values in units of g. The output acceleration is
    returned in m/s² using 1 g ≈ 9.80665 m/s².
    """
    if not path.exists():
        raise FileNotFoundError(f"Ground motion file not found: {path}")

    # Parse DT from header
    dt: float | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.lstrip().startswith("%NPTS"):
                text = line.upper()
                if "DT=" in text:
                    dt_part = text.split("DT=")[1]
                    if "SEC" in dt_part:
                        dt_str = dt_part.split("SEC")[0].strip().replace(",", "")
                    else:
                        dt_str = dt_part.strip().split()[0]
                    dt = float(dt_str)
                break

    if dt is None:
        raise ValueError(f"Could not parse DT from header of {path}")

    # Load accelerations in g, skipping header lines starting with '%'
    acc_data = np.loadtxt(path, comments="%")
    acc_g = np.asarray(acc_data, dtype=float).ravel()
    if acc_g.size < 2:
        raise ValueError(f"Ground motion file {path} has too few samples.")

    g_to_mps2 = 9.80665
    ug_ddot = acc_g * g_to_mps2
    return ug_ddot, dt


def _dynamic_stiffness_models(
    omega: np.ndarray, params: Dict[str, float]
) -> Dict[str, np.ndarray]:
    """
    Build dynamic stiffness R̂(ω) for Models A, B, C over a vector of ω.

    Parameters
    ----------
    omega : np.ndarray
        Angular frequency array [rad/s] from FFT (two-sided).
    params : dict
        sdof_parameters() dict with keys m, k, c, delta, alpha, c_alpha, omega_n.
    """
    k = float(params["k"])
    c = float(params["c"])
    delta = float(params["delta"])
    alpha = float(params["alpha"])
    c_alpha = float(params["c_alpha"])
    omega_n = float(params["omega_n"])

    omega = np.asarray(omega, dtype=float)
    sgn = np.sign(omega)

    # Model A: Kelvin–Voigt
    R_A = k + 1j * c * omega

    # Model B: hysteretic idealization with two-sided implementation
    R_B = k * (1.0 + 1j * delta * sgn)

    # Model C: fractional Kelvin–Voigt with storage-matched stiffness k_C
    cos_term = math.cos(math.pi * alpha / 2.0)
    sin_term = math.sin(math.pi * alpha / 2.0)
    k_C = k - c_alpha * (omega_n**alpha) * cos_term

    abs_omega_alpha = np.abs(omega) ** alpha
    frac = c_alpha * abs_omega_alpha * (cos_term + 1j * sgn * sin_term)
    R_C = k_C + frac

    return {
        "Model A (viscous)": R_A,
        "Model B (hysteretic)": R_B,
        "Model C (fractional)": R_C,
    }


def sdof_frequency_response_for_models(
    ug_ddot: np.ndarray,
    dt: float,
    params: Dict[str, float],
    zero_pad_factor: int = 2,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute SDOF response (u, u_dot, u_ddot_abs) for Models A, B, C via FFT.

    Parameters
    ----------
    ug_ddot : np.ndarray
        Ground acceleration ü_g(t) [m/s²].
    dt : float
        Time step [s].
    params : dict
        sdof_parameters() output with m, k, c, delta, alpha, c_alpha, omega_n.
    zero_pad_factor : int, optional
        Zero-padding factor for FFT length (default 2).

    Returns
    -------
    t : np.ndarray
        Time vector [s] for the original record (no padding).
    responses : dict
        Mapping model label → array of shape (n, 3) with columns
        (u, u_dot, u_ddot_abs).
    """
    ug = np.asarray(ug_ddot, dtype=float).ravel()
    n = ug.size
    if n == 0:
        raise ValueError("ug_ddot must contain at least one time point.")

    m = float(params["m"])
    if m <= 0.0:
        raise ValueError("Mass m must be positive.")
    if dt <= 0.0:
        raise ValueError("Time step dt must be positive.")

    if zero_pad_factor < 1:
        zero_pad_factor = 1
    n_fft = int(zero_pad_factor * n)

    ug_padded = np.zeros(n_fft, dtype=float)
    ug_padded[:n] = ug

    Ug = np.fft.fft(ug_padded)
    omega = 2.0 * math.pi * np.fft.fftfreq(n_fft, d=dt)

    R_models = _dynamic_stiffness_models(omega, params)
    responses: Dict[str, np.ndarray] = {}

    for label, R_hat in R_models.items():
        denom = R_hat - m * omega**2
        denom_safe = np.where(np.abs(denom) < 1e-12, 1e-12 + 0j, denom)
        H = -m / denom_safe

        U = H * Ug
        V = 1j * omega * U
        Uddot_rel = -(omega**2) * U
        Uddot_abs = Uddot_rel + Ug

        u_t = np.fft.ifft(U).real[:n]
        v_t = np.fft.ifft(V).real[:n]
        a_abs_t = np.fft.ifft(Uddot_abs).real[:n]

        responses[label] = np.column_stack((u_t, v_t, a_abs_t))

    t = np.arange(0.0, n * dt, dt)
    return t, responses

