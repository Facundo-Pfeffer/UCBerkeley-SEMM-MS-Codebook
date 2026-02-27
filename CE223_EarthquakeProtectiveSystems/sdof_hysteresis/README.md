# SDOF Hysteresis and Frequency-Domain Earthquake Response

This module builds interactive dashboards to compare three SDOF damping models:

1. **Model A – Kelvin–Voigt (viscous)**  
   \(f(t) = k\,u(t) + c\,\dot{u}(t)\)

2. **Model B – Hysteretic (structural, harmonic idealization)**  
   \(f(t) = k\,u_0\sin(\omega t) + k\delta\,u_0\cos(\omega t)\)

3. **Model C – Fractional Kelvin–Voigt**  
   \(f(t) = \tilde{k}\,u(t) + c_\alpha D^\alpha u(t)\)

The goals are to:

- Visualize **force–displacement hysteresis loops** under prescribed harmonic motion.
- Compare **storage** and **loss** stiffnesses \(K_1(\omega)\), \(K_2(\omega)\).
- Compute **energy dissipated per cycle** (EDC) both analytically and numerically.
- Use a **frequency-domain SDOF solver** to compute earthquake response for the three models.

## 1. Harmonic hysteresis loops and EDC

For a prescribed displacement \(u(t) = u_0 \sin(\omega t)\), all three models admit the
harmonic decomposition:

\[
f(t) = K_1(\omega)\,u_0 \sin(\omega t) + K_2(\omega)\,u_0 \cos(\omega t),
\]

with:

- \(K_1(\omega)\): **storage stiffness**, in phase with \(u(t)\).
- \(K_2(\omega)\): **loss stiffness**, in quadrature with \(u(t)\).

The **energy dissipated per cycle** is:

\[
\mathrm{EDC}(\omega) = \oint f\,du
= \int_0^T f(t)\,\dot{u}(t)\,dt
= \pi\,K_2(\omega)\,u_0^2,
\]

so \(K_2\) directly controls the loop area.

### Key files

- `sdof_hysteresis_plotly.py`
  - Provides a small CE-style color palette.
  - `create_sdof_hysteresis_figure(...)`: builds a multi-trace hysteresis figure with
    consistent styling and legends.

- `build_sdof_hysteresis_dashboard.py`
  - Defines model parameters \((m, k, c, \delta, c_\alpha, \alpha)\).
  - Generates a dense time grid with at least 1000 points per cycle.
  - Evaluates \(u(t)\), \(f(t)\) for each model and frequency.
  - Builds:
    - A hysteresis dashboard (`sdof_hysteresis_dashboard.html`) with:
      - Loops for Models A/B/C at selected \(\omega\) (including resonance).
      - An EDC vs \(\omega\) plot, showing:
        - analytical EDC curves from closed-form formulas,
        - numerical EDC computed as a polygonal area in \((u, f)\)-space
          (MATLAB-`polyarea` equivalent),
      - text boxes explaining the models, fractional derivative, and physical meaning.
    - A simpler loops-only view (`sdof_hysteresis_loops.html`).

### Numerical EDC computation (polyarea)

For each model and frequency:

1. Define a uniform time grid over one period \(T = 2\pi/\omega\) with at least 1000 points.
2. Compute:
   - \(u_j = u(t_j)\),
   - \(f_j = f(t_j)\).
3. Enforce closure by appending \((u_0, f_0)\) to the end of the arrays.
4. Apply a polygon-area routine (NumPy implementation of `polyarea(u, f)`), interpret
   the result as \(\mathrm{EDC}\), and use \(|\cdot|\) to handle loop orientation.
5. Compare the numerical EDC against the analytical formula above as a consistency check.

At resonance, all three models are tuned so that \(\mathrm{EDC}\) coincides, and the
dashboard highlights this agreement.

## 2. Frequency-domain earthquake response

The second part of the dashboard analyzes the **earthquake response** of each SDOF
model using the frequency-domain approach.

### Governing equation and transfer function

In relative coordinates \(u(t)\), with ground acceleration \(\ddot{u}_g(t)\):

\[
m\ddot{u}(t) + f_d(t) + k\,u(t) = -m\ddot{u}_g(t).
\]

Introducing \(\hat{R}(\omega)\) such that \(F(\omega) = \hat{R}(\omega) U(\omega)\),
and Fourier transforming:

\[
\bigl(\hat{R}(\omega) - m\omega^2\bigr) U(\omega) = -m\,\ddot{U}_g(\omega),
\]

so the relative-displacement transfer function is:

\[
U(\omega) = H(\omega)\,\ddot{U}_g(\omega),
\qquad
H(\omega) = -\frac{m}{\hat{R}(\omega) - m\omega^2}.
\]

Model-specific \(\hat{R}(\omega)\) are implemented in `sdof_frequency_response.py`:

- Model A: \(\hat{R}^{(A)}(\omega) = k + i c \omega\).
- Model B: \(\hat{R}^{(B)}(\omega) = k\bigl(1 + i\delta\,\operatorname{sgn}(\omega)\bigr)\).
- Model C: \(\hat{R}^{(C)}(\omega) = \tilde{k} + c_\alpha (i\omega)^\alpha\),
  with \(\tilde{k}\) chosen so that storage stiffness matches \(k\) at \(\omega_n\).

Once \(U(\omega)\) is known, spectral differentiation gives:

\[
V(\omega) = i\omega U(\omega), \qquad \ddot{U}(\omega) = -\omega^2 U(\omega),
\]

and the absolute acceleration spectrum:

\[
\ddot{U}_{\mathrm{abs}}(\omega) = -\omega^2 U(\omega) + \ddot{U}_g(\omega).
\]

Time histories follow by inverse FFT.

### Key file: `sdof_frequency_response.py`

- `load_peer_at2_to_mps2(path)`  
  Reads a PEER AT2 file, parses `DT`, converts gravity units to \( \mathrm{m/s}^2 \).

- `_dynamic_stiffness_models(omega, params)`  
  Returns \(\hat{R}^{(A)}(\omega)\), \(\hat{R}^{(B)}(\omega)\), \(\hat{R}^{(C)}(\omega)\).

- `sdof_frequency_response_for_models(ug_ddot, dt, params)`  
  Orchestrates FFT-based response computation for all three models:
  computes the DFT of \(\ddot{u}_g(t)\), builds \(H(\omega)\) and \(U(\omega)\) for each model,
  and recovers \(u(t)\), \(\dot{u}(t)\), and \(\ddot{u}_{\mathrm{abs}}(t)\) via inverse FFT.

## 3. Usage

From `CE223_EarthquakeProtectiveSystems/sdof_hysteresis/`:

```bash
python build_sdof_hysteresis_dashboard.py
```

The script will:

- Build harmonic hysteresis loops for Models A/B/C.
- Compute analytical and numerical EDC vs frequency.
- Load the Kobe ground motion from
  `../input_ground_motion/RSN1108_KOBE_KBU090.AT2`.
- Use `sdof_frequency_response_for_models` to compute earthquake responses.
- Write the canonical dashboards under
  `../highlighted_htmls/sdof_hysteresis_dashboard.html` and
  `../highlighted_htmls/sdof_hysteresis_loops.html`.

