from __future__ import annotations

"""
Frequency-domain SDOF response utilities (local copy for base isolator project).

See also `CE223_EarthquakeProtectiveSystems/fft_sdof_response.py` for the
course-level version. This module is kept self-contained so that scripts in
`base_isolators_desgin` can import a robust FFT-based solver without
modifying Python's import path.
"""

import math
from typing import Literal

import numpy as np


def sdof_response_fft_ground_motion(
    ug_ddot: np.ndarray,
    dt: float,
    m: float,
    k: float,
    c: float,
    *,
    zero_pad_factor: int = 2,
    return_components: Literal["relative", "absolute"] = "absolute",
) -> np.ndarray:
    """
    Compute SDOF response to a ground-acceleration record using FFT.

    Equation in relative coordinates:
        m u¨(t) + c u˙(t) + k u(t) = -m ü_g(t)

    Frequency-domain solution:
        U(ω) = H(ω) Ü_g(ω),
        H(ω) = -m / (k + i c ω - m ω²).

    Velocity and absolute acceleration follow from:
        V(ω)      = i ω U(ω),
        Ü_rel(ω)  = -ω² U(ω),
        Ü_abs(ω)  = Ü_rel(ω) + Ü_g(ω).

    Parameters
    ----------
    ug_ddot : np.ndarray
        Ground acceleration ü_g(t).
    dt : float
        Time step [s].
    m : float
        Mass.
    k : float
        Stiffness.
    c : float
        Viscous damping coefficient.
    zero_pad_factor : int, optional
        Factor by which to zero-pad the record before FFT (default 2).
        Zero-padding reduces wrap-around artefacts of the circular
        convolution implicit in the FFT.
    return_components : {"relative", "absolute"}, optional
        If "relative", returns columns (u, u_dot, u_ddot_rel).
        If "absolute", returns columns (u, u_dot, u_ddot_abs).

    Returns
    -------
    np.ndarray
        Array of shape (n, 3) with the requested time histories.
    """
    ug = np.asarray(ug_ddot, dtype=float).ravel()
    n = ug.size
    if n == 0:
        raise ValueError("Ground motion array ug_ddot must be non-empty.")
    if dt <= 0.0:
        raise ValueError("Time step dt must be positive.")
    if m <= 0.0:
        raise ValueError("Mass m must be positive.")

    if zero_pad_factor < 1:
        zero_pad_factor = 1
    n_fft = int(zero_pad_factor * n)
    ug_padded = np.zeros(n_fft, dtype=float)
    ug_padded[:n] = ug

    Ug = np.fft.fft(ug_padded)
    omega = 2.0 * math.pi * np.fft.fftfreq(n_fft, d=dt)

    R_hat = k + 1j * c * omega
    denom = R_hat - m * omega**2
    denom_safe = np.where(np.abs(denom) < 1e-12, 1e-12 + 0j, denom)
    H = -m / denom_safe

    U = H * Ug
    V = 1j * omega * U
    Uddot_rel = -(omega**2) * U
    Uddot_abs = Uddot_rel + Ug

    u_t = np.fft.ifft(U).real[:n]
    v_t = np.fft.ifft(V).real[:n]

    if return_components == "relative":
        a_t = np.fft.ifft(Uddot_rel).real[:n]
    else:
        a_t = np.fft.ifft(Uddot_abs).real[:n]

    return np.column_stack((u_t, v_t, a_t))

