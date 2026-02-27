from __future__ import annotations

"""
Frequency-domain SDOF response utilities for CE223.

The core routine `sdof_response_fft_ground_motion` computes the linear
single-degree-of-freedom (SDOF) response to a base acceleration record
using FFT-based convolution.

Equation of motion in relative coordinates:

    m u¨(t) + c u˙(t) + k u(t) = -m ü_g(t)

Taking Fourier transforms:

    (k + i c ω - m ω²) U(ω) = -m Ü_g(ω)
    ⇒ U(ω) = H(ω) Ü_g(ω),
       H(ω) = -m / (k + i c ω - m ω²).

Velocity and absolute acceleration follow from spectral derivatives:

    V(ω) = i ω U(ω),
    Ü_rel(ω) = -ω² U(ω),
    Ü_abs(ω) = Ü_rel(ω) + Ü_g(ω).

An FFT with zero-padding is used to approximate the convolution while
reducing wrap-around effects; the time histories are truncated back to
the original record length.
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

    Parameters
    ----------
    ug_ddot : np.ndarray
        Ground acceleration time history ü_g(t) [consistent accel units].
    dt : float
        Time step [s].
    m : float
        Mass.
    k : float
        Stiffness.
    c : float
        Viscous damping coefficient.
    zero_pad_factor : int, optional
        Factor by which to extend the record length with zeros before
        applying the FFT (default 2). A value ≥ 2 helps mitigate
        wrap-around effects in the convolution.
    return_components : {"relative", "absolute"}, optional
        If "relative", returns columns (u, u_dot, u_ddot_rel).
        If "absolute", returns columns (u, u_dot, u_ddot_abs).

    Returns
    -------
    np.ndarray
        Array of shape (n, 3) with time histories corresponding to the
        chosen `return_components`.
    """
    ug = np.asarray(ug_ddot, dtype=float).ravel()
    n = ug.size
    if n == 0:
        raise ValueError("Ground motion array ug_ddot must be non-empty.")
    if dt <= 0.0:
        raise ValueError("Time step dt must be positive.")
    if m <= 0.0:
        raise ValueError("Mass m must be positive.")

    # Zero pad to reduce circular-convolution artefacts
    if zero_pad_factor < 1:
        zero_pad_factor = 1
    n_fft = int(zero_pad_factor * n)
    ug_padded = np.zeros(n_fft, dtype=float)
    ug_padded[:n] = ug

    Ug = np.fft.fft(ug_padded)
    omega = 2.0 * math.pi * np.fft.fftfreq(n_fft, d=dt)  # rad/s

    R_hat = k + 1j * c * omega
    denom = R_hat - m * omega**2

    # Avoid division by zero at resonant frequencies
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

