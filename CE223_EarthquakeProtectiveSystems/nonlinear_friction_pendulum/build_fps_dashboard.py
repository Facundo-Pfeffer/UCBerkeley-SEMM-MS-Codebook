from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import numpy as np
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.subplots import make_subplots


G_SI = 9.80665  # m/s²
# Floor NSC spectrum grid / plot: log Tp cannot include 0; use a small T_min [s].
FLOOR_SPECTRUM_T_MIN = 0.01
FLOOR_SPECTRUM_T_MAX = 10.0
FLOOR_SPECTRUM_REF_F_HZ = 2.0
FLOOR_SPECTRUM_REF_T = 1.0 / FLOOR_SPECTRUM_REF_F_HZ
BASE_DIR = Path(__file__).resolve().parent
CE223_DIR = BASE_DIR.parent
HIGHLIGHTED_HTML_DIR = CE223_DIR / "highlighted_htmls"
HIGHLIGHTED_HTML_DIR.mkdir(parents=True, exist_ok=True)
KOBE_PATH = CE223_DIR / "input_ground_motion" / "RSN1108_KOBE_KBU090.AT2"
SYLMAR_PATH = BASE_DIR / "SYLMAR360.txt"
OUTPUT_HTML = HIGHLIGHTED_HTML_DIR / "CE223_FPS_Bilinear_Kobe_Sylmar.html"

MATLAB_COLORS = {
    "dark_blue": "rgb(0, 70, 140)",
    "black": "rgb(20, 20, 20)",
    "crimson": "rgb(180, 20, 60)",
    "dark_green": "rgb(0, 100, 0)",
    "gray": "rgb(110, 110, 110)",
}


@dataclass(frozen=True)
class FpsProblemData:
    total_mass: float = 1.47e6
    n_bearings: int = 15
    radius: float = 1.0
    friction_coefficient: float = 0.03
    yield_displacement_mm: float = 0.03

    @property
    def weight(self) -> float:
        return self.total_mass * G_SI

    @property
    def yield_displacement(self) -> float:
        return self.yield_displacement_mm * 1.0e-3


@dataclass(frozen=True)
class BilinearFpsParameters:
    mass: float
    pendulum_stiffness: float
    characteristic_strength: float
    yield_force: float
    elastic_stiffness: float
    hardening_modulus: float
    post_yield_stiffness: float


@dataclass
class GroundMotionRecord:
    name: str
    dt: float
    acceleration_mps2: np.ndarray

    @property
    def time_array(self) -> np.ndarray:
        return np.arange(self.acceleration_mps2.size, dtype=float) * self.dt

    @property
    def acceleration_g(self) -> np.ndarray:
        return self.acceleration_mps2 / G_SI


@dataclass
class TimeHistoryResult:
    time: np.ndarray
    ground_acceleration: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    relative_acceleration: np.ndarray
    absolute_acceleration: np.ndarray
    restoring_force: np.ndarray

    @property
    def peak_displacement(self) -> float:
        return float(np.max(np.abs(self.displacement)))

    @property
    def peak_velocity(self) -> float:
        return float(np.max(np.abs(self.velocity)))

    @property
    def peak_force(self) -> float:
        return float(np.max(np.abs(self.restoring_force)))

    @property
    def peak_abs_acc(self) -> float:
        return float(np.max(np.abs(self.absolute_acceleration)))


@dataclass
class FloorSpectrumResult:
    periods: np.ndarray
    frequencies: np.ndarray
    peak_abs_component_acceleration: np.ndarray


@dataclass
class BilinearState:
    plastic_displacement: float = 0.0
    back_force: float = 0.0


class GroundMotionLoader:
    @staticmethod
    def load_peer_file(path: Path) -> GroundMotionRecord:
        if not path.exists():
            raise FileNotFoundError(f"Ground motion file not found: {path}")
        dt = GroundMotionLoader._parse_dt(path)
        acc_g = np.loadtxt(path, comments="%")
        acc_g = np.asarray(acc_g, dtype=float).ravel()
        if acc_g.size < 2:
            raise ValueError(f"Ground motion file has too few points: {path}")
        return GroundMotionRecord(name=path.name, dt=dt, acceleration_mps2=acc_g * G_SI)

    @staticmethod
    def _parse_dt(path: Path) -> float:
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                if "DT=" not in line.upper():
                    continue
                upper = line.upper()
                dt_part = upper.split("DT=")[1]
                dt_str = dt_part.replace("SEC", "").replace(",", " ").strip().split()[0]
                return float(dt_str)
        raise ValueError(f"Could not parse DT from {path}")


class FpsParameterBuilder:
    @staticmethod
    def from_problem_data(data: FpsProblemData) -> BilinearFpsParameters:
        weight = data.weight
        kp = weight / data.radius
        q = data.friction_coefficient * weight
        fy = q
        k = fy / data.yield_displacement
        h = (k * kp) / (k - kp)
        return BilinearFpsParameters(
            mass=data.total_mass,
            pendulum_stiffness=kp,
            characteristic_strength=q,
            yield_force=fy,
            elastic_stiffness=k,
            hardening_modulus=h,
            post_yield_stiffness=kp,
        )


class BilinearConstitutiveModel:
    def __init__(self, parameters: BilinearFpsParameters) -> None:
        self.parameters = parameters

    def update(self, displacement: float, previous_state: BilinearState) -> tuple[float, BilinearState, float]:
        k = self.parameters.elastic_stiffness
        h = self.parameters.hardening_modulus
        fy = self.parameters.yield_force

        trial_force = k * (displacement - previous_state.plastic_displacement)
        shifted_trial_force = trial_force - previous_state.back_force
        trial_yield_cond = abs(shifted_trial_force) - fy

        if trial_yield_cond <= 0.0:
            return trial_force, BilinearState(previous_state.plastic_displacement, previous_state.back_force), k

        sign_trial = 1.0 if shifted_trial_force >= 0.0 else -1.0
        gamma_increment = trial_yield_cond / (k + h)

        force = trial_force - k * gamma_increment * sign_trial
        plastic_displacement = previous_state.plastic_displacement + gamma_increment * sign_trial
        back_force = previous_state.back_force + h * gamma_increment * sign_trial
        algorithmic_tangent = (k * h) / (k + h)

        return force, BilinearState(plastic_displacement, back_force), algorithmic_tangent


class NonlinearNewmarkSolver:
    def __init__(self, model: BilinearConstitutiveModel, beta: float = 1.0 / 4.0, gamma: float = 1.0 / 2.0) -> None:
        self.model = model
        self.beta = beta
        self.gamma = gamma

    def solve(self, record: GroundMotionRecord, tolerance: float = 1e-8, max_iterations: int = 60) -> TimeHistoryResult:
        m = self.model.parameters.mass
        dt = record.dt
        ug = record.acceleration_mps2
        n = ug.size

        u = np.zeros(n)
        v = np.zeros(n)
        a = np.zeros(n)
        f = np.zeros(n)
        abs_a = np.zeros(n)

        state = BilinearState()
        f[0], state, _ = self.model.update(0.0, state)
        a[0] = (-f[0] - m * ug[0]) / m
        abs_a[0] = a[0] + ug[0]

        beta = self.beta
        gamma = self.gamma
        a0 = 1.0 / (beta * dt * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a6 = dt * (1.0 - gamma)
        a7 = gamma * dt

        for i in range(1, n):
            u_guess = u[i - 1]
            converged_state = state
            converged_force = f[i - 1]
            converged_acc = a[i - 1]
            converged_vel = v[i - 1]

            for _ in range(max_iterations):
                trial_state_input = BilinearState(state.plastic_displacement, state.back_force)
                force_i, state_i, k_alg = self.model.update(u_guess, trial_state_input)
                acc_i = a0 * (u_guess - u[i - 1]) - a2 * v[i - 1] - a3 * a[i - 1]
                residual = m * acc_i + force_i + m * ug[i]
                tangent = m * a0 + k_alg
                delta_u = -residual / tangent
                u_guess += delta_u
                converged_state = state_i
                converged_force = force_i
                converged_acc = acc_i
                converged_vel = v[i - 1] + a6 * a[i - 1] + a7 * acc_i

                if abs(delta_u) < tolerance and abs(residual) < tolerance * max(1.0, self.model.parameters.characteristic_strength):
                    break

            u[i] = u_guess
            f[i] = converged_force
            a[i] = converged_acc
            v[i] = converged_vel
            state = converged_state
            abs_a[i] = a[i] + ug[i]

        return TimeHistoryResult(
            time=record.time_array,
            ground_acceleration=ug,
            displacement=u,
            velocity=v,
            relative_acceleration=a,
            absolute_acceleration=abs_a,
            restoring_force=f,
        )


class EquivalentLinearSolver:
    def __init__(self, parameters: BilinearFpsParameters) -> None:
        self.parameters = parameters

    def solve(self, record: GroundMotionRecord, iterations: int = 12) -> TimeHistoryResult:
        m = self.parameters.mass
        kp = self.parameters.pendulum_stiffness
        q = self.parameters.characteristic_strength
        umax = max(self.parameters.yield_force / self.parameters.elastic_stiffness, 1e-6)

        for _ in range(iterations):
            k_eff = kp + q / max(umax, 1e-6)
            zeta = (2.0 * q) / (math.pi * k_eff * max(umax, 1e-6))
            zeta = float(np.clip(zeta, 0.02, 0.35))
            c_eff = 2.0 * zeta * math.sqrt(m * k_eff)
            result = LinearNewmarkSolver.solve_sdof_base_excitation(record, m, c_eff, k_eff)
            umax = max(result.peak_displacement, 1e-6)

        return result


class LinearNewmarkSolver:
    @staticmethod
    def solve_sdof_base_excitation(
        record: GroundMotionRecord,
        mass: float,
        damping: float,
        stiffness: float,
        beta: float = 1.0 / 4.0,
        gamma: float = 1.0 / 2.0,
    ) -> TimeHistoryResult:
        ug = record.acceleration_mps2
        dt = record.dt
        n = ug.size

        u = np.zeros(n)
        v = np.zeros(n)
        a = np.zeros(n)
        f = np.zeros(n)
        abs_a = np.zeros(n)

        a[0] = (-damping * v[0] - stiffness * u[0] - mass * ug[0]) / mass
        abs_a[0] = a[0] + ug[0]
        f[0] = stiffness * u[0] + damping * v[0]

        a0 = 1.0 / (beta * dt * dt)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2.0 * beta) - 1.0)
        k_eff = stiffness + a0 * mass + a1 * damping

        for i in range(1, n):
            p_i = -mass * ug[i]
            p_eff = (
                p_i
                + mass * (a0 * u[i - 1] + a2 * v[i - 1] + a3 * a[i - 1])
                + damping * (a1 * u[i - 1] + a4 * v[i - 1] + a5 * a[i - 1])
            )
            u[i] = p_eff / k_eff
            a[i] = a0 * (u[i] - u[i - 1]) - a2 * v[i - 1] - a3 * a[i - 1]
            v[i] = v[i - 1] + dt * ((1.0 - gamma) * a[i - 1] + gamma * a[i])
            f[i] = stiffness * u[i] + damping * v[i]
            abs_a[i] = a[i] + ug[i]

        return TimeHistoryResult(
            time=record.time_array,
            ground_acceleration=ug,
            displacement=u,
            velocity=v,
            relative_acceleration=a,
            absolute_acceleration=abs_a,
            restoring_force=f,
        )


class FloorSpectrumCalculator:
    @staticmethod
    def compute(
        support_abs_acceleration: np.ndarray,
        dt: float,
        damping_ratio: float = 0.02,
        min_period: float = FLOOR_SPECTRUM_T_MIN,
        max_period: float = FLOOR_SPECTRUM_T_MAX,
        n_points: int = 220,
        min_points_per_cycle: int = 24,
    ) -> FloorSpectrumResult:
        if max_period <= min_period:
            raise ValueError("max_period must be greater than min_period.")
        periods = np.geomspace(min_period, max_period, n_points)
        frequencies = 1.0 / periods
        peaks = np.zeros(n_points, dtype=float)
        support_abs_acceleration = np.asarray(support_abs_acceleration, dtype=float).ravel().copy()
        n = support_abs_acceleration.size
        time_original = np.arange(n, dtype=float) * dt

        max_frequency = float(np.max(frequencies))
        target_dt = 1.0 / (max_frequency * float(max(min_points_per_cycle, 8)))
        substeps = max(1, int(math.ceil(dt / target_dt)))
        dt_internal = dt / float(substeps)
        time_internal = np.arange((n - 1) * substeps + 1, dtype=float) * dt_internal
        support_internal = np.interp(time_internal, time_original, support_abs_acceleration)
        pseudo_record = GroundMotionRecord("support", dt_internal, support_internal)

        for i, period in enumerate(periods):
            omega = 2.0 * math.pi / period
            k = omega * omega
            c = 2.0 * damping_ratio * omega
            result = LinearNewmarkSolver.solve_sdof_base_excitation(
                pseudo_record,
                mass=1.0,
                damping=c,
                stiffness=k,
            )
            component_abs_acc = result.absolute_acceleration
            peaks[i] = float(np.max(np.abs(component_abs_acc)))

        # Ascending Tp (short period left → long period right) for period-axis spectrum plots.
        sort_idx = np.argsort(periods, kind="mergesort")
        periods = periods[sort_idx]
        frequencies = frequencies[sort_idx]
        peaks = peaks[sort_idx]

        return FloorSpectrumResult(
            periods=periods,
            frequencies=frequencies,
            peak_abs_component_acceleration=peaks,
        )


class FigureFactory:
    @staticmethod
    def _max_marker_coordinates(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        idx = int(np.argmax(np.abs(y)))
        return float(x[idx]), float(y[idx])

    @staticmethod
    def time_history_5panel(title: str, result: TimeHistoryResult, total_weight: float) -> go.Figure:
        time = result.time
        ground_g = result.ground_acceleration / G_SI
        abs_acc_g = result.absolute_acceleration / G_SI
        rel_acc_g = result.relative_acceleration / G_SI
        force_mn = result.restoring_force / 1e6
        force_over_weight = result.restoring_force / total_weight
        disp_mm = result.displacement * 1e3
        vel_mps = result.velocity

        fig = make_subplots(
            rows=5,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.045,
            subplot_titles=(
                "Isolation Displacement u(t) [mm]",
                "Isolation Velocity u̇(t) [m/s]",
                "Normalized Restoring Force F(t)/W [-]",
                "Absolute Acceleration üt(t) [g]",
                "Ground Acceleration üg(t) [g]",
            ),
        )

        series = [
            (disp_mm, MATLAB_COLORS["dark_blue"], "u_max", "mm", "Displacement u", ".3f"),
            (vel_mps, MATLAB_COLORS["black"], "u_dot_max", "m/s", "Velocity u_dot", ".4f"),
            (force_over_weight, MATLAB_COLORS["crimson"], "F/W_max", "-", "Normalized restoring force F/W", ".5f"),
            (abs_acc_g, MATLAB_COLORS["dark_green"], "a_abs_max", "g", "Absolute acceleration a_abs", ".4f"),
            (ground_g, MATLAB_COLORS["gray"], "a_g_max", "g", "Ground acceleration a_g", ".4f"),
        ]

        for row_idx, (y_values, color, peak_label, unit_label, hover_name, hover_fmt) in enumerate(series, start=1):
            y_max = float(np.max(np.abs(y_values)))
            y_padding = 0.18 * max(y_max, 1e-9)
            y_min = float(np.min(y_values)) - y_padding
            y_top = float(np.max(y_values)) + y_padding

            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=y_values,
                    mode="lines",
                    line=dict(color=color, width=2.2),
                    showlegend=False,
                    customdata=np.column_stack((disp_mm, vel_mps, force_over_weight, force_mn, rel_acc_g, abs_acc_g, ground_g)),
                    hovertemplate=(
                        f"Time: %{{x:.3f}} s<br>"
                        f"{hover_name}: %{{y{hover_fmt}}}"
                        + (f" {unit_label}" if unit_label != "-" else "")
                        + "<br>"
                        "u: %{customdata[0]:.3f} mm<br>"
                        "u_dot: %{customdata[1]:.4f} m/s<br>"
                        "F/W: %{customdata[2]:.5f}<br>"
                        "F: %{customdata[3]:.4f} MN<br>"
                        "a_rel: %{customdata[4]:.4f} g<br>"
                        "a_abs: %{customdata[5]:.4f} g<br>"
                        "a_g: %{customdata[6]:.4f} g<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=1,
            )
            x_peak, y_peak = FigureFactory._max_marker_coordinates(time, y_values)
            fig.add_trace(
                go.Scatter(
                    x=[x_peak],
                    y=[y_peak],
                    mode="markers",
                    marker=dict(size=10, color=MATLAB_COLORS["black"]),
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )
            fig.add_annotation(
                x=x_peak,
                y=y_peak,
                xref="x",
                yref=f"y{'' if row_idx == 1 else row_idx}",
                text=f"{peak_label} = {abs(y_peak):.3f}" + (f" {unit_label}" if unit_label != "-" else ""),
                showarrow=True,
                arrowhead=2,
                ax=24,
                ay=-24,
                font=dict(size=14, color=MATLAB_COLORS["black"]),
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor=MATLAB_COLORS["black"],
                borderwidth=1,
            )
            fig.update_yaxes(range=[y_min, y_top], row=row_idx, col=1, title_font=dict(size=15), tickfont=dict(size=13))

        fig.update_layout(
            template="plotly_white",
            height=1200,
            title=dict(text=title, x=0.5, xanchor="center", font=dict(size=22)),
            margin=dict(t=110),
            font=dict(size=14),
            hovermode="x unified",
        )
        time_end = float(time[-1]) if time.size > 0 else 0.0
        fig.update_xaxes(range=[0.0, time_end], title_font=dict(size=16), tickfont=dict(size=13))
        fig.update_xaxes(title_text="Time [s]", row=5, col=1, title_font=dict(size=16), tickfont=dict(size=13))
        fig.update_annotations(font=dict(size=16))
        return fig

    @staticmethod
    def hysteresis(title: str, result: TimeHistoryResult, total_weight: float) -> go.Figure:
        disp_mm = result.displacement * 1e3
        force_mn = result.restoring_force / 1e6
        force_over_weight = result.restoring_force / total_weight
        time = result.time
        velocity = result.velocity
        rel_acc_g = result.relative_acceleration / G_SI
        abs_acc_g = result.absolute_acceleration / G_SI
        ground_g = result.ground_acceleration / G_SI
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=disp_mm,
                y=force_over_weight,
                mode="lines",
                line=dict(color=MATLAB_COLORS["dark_blue"], width=2.4),
                name="F-u",
                customdata=np.column_stack((time, velocity, rel_acc_g, abs_acc_g, ground_g, force_mn)),
                hovertemplate=(
                    "Displacement u: %{x:.3f} mm<br>"
                    "Normalized restoring force F/W: %{y:.5f}<br>"
                    "Restoring force F: %{customdata[5]:.4f} MN<br>"
                    "Time: %{customdata[0]:.3f} s<br>"
                    "Velocity u_dot: %{customdata[1]:.4f} m/s<br>"
                    "Relative acceleration a_rel: %{customdata[2]:.4f} g<br>"
                    "Absolute acceleration a_abs: %{customdata[3]:.4f} g<br>"
                    "Ground acceleration a_g: %{customdata[4]:.4f} g<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            template="plotly_white",
            height=430,
            title=dict(text=title, x=0.5, xanchor="center", font=dict(size=22)),
            xaxis=dict(title="Displacement [mm]", title_font=dict(size=16), tickfont=dict(size=13)),
            yaxis=dict(title="Normalized Force F/W [-]", title_font=dict(size=16), tickfont=dict(size=13)),
            font=dict(size=14),
        )
        return fig

    @staticmethod
    def floor_spectra_comparison(
        title: str,
        exact_spectrum: FloorSpectrumResult,
        equivalent_spectrum: FloorSpectrumResult,
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=exact_spectrum.periods,
                y=exact_spectrum.peak_abs_component_acceleration / G_SI,
                mode="lines",
                line=dict(color=MATLAB_COLORS["dark_blue"], width=2.6),
                name="Nonlinear isolation üₜ",
                customdata=np.column_stack((exact_spectrum.frequencies,)),
                hovertemplate=(
                    "Period Tp: %{x:.4f} s<br>"
                    "Frequency fp: %{customdata[0]:.3f} Hz<br>"
                    "Peak NSC abs. accel.: %{y:.4f} g<br>"
                    "Floor motion: nonlinear isolation<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=equivalent_spectrum.periods,
                y=equivalent_spectrum.peak_abs_component_acceleration / G_SI,
                mode="lines",
                line=dict(color=MATLAB_COLORS["crimson"], width=2.4, dash="dash"),
                name="Equiv.-linear isolation üₜ",
                customdata=np.column_stack((equivalent_spectrum.frequencies,)),
                hovertemplate=(
                    "Period Tp: %{x:.4f} s<br>"
                    "Frequency fp: %{customdata[0]:.3f} Hz<br>"
                    "Peak NSC abs. accel.: %{y:.4f} g<br>"
                    "Floor motion: equivalent-linear isolation<extra></extra>"
                ),
            )
        )
        fig.add_vline(
            x=FLOOR_SPECTRUM_REF_T,
            line_dash="dot",
            line_color=MATLAB_COLORS["black"],
            annotation_text=f"{FLOOR_SPECTRUM_REF_F_HZ:g} Hz (Tp = {FLOOR_SPECTRUM_REF_T:g} s)",
            annotation_position="top",
            annotation_font_size=12,
        )
        t_lo = FLOOR_SPECTRUM_T_MIN
        t_hi = FLOOR_SPECTRUM_T_MAX
        fig.update_layout(
            template="plotly_white",
            height=430,
            title=dict(text=title, x=0.5, xanchor="center", font=dict(size=22)),
            xaxis=dict(
                type="log",
                title="NSC period Tp [s] (log scale)",
                range=[math.log10(t_lo), math.log10(t_hi)],
                title_font=dict(size=16),
                tickfont=dict(size=13),
                exponentformat="power",
            ),
            yaxis=dict(title="Peak NSC Absolute Acceleration [g]", title_font=dict(size=16), tickfont=dict(size=13)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            font=dict(size=14),
        )
        return fig


class HtmlReportBuilder:
    def __init__(self, params: BilinearFpsParameters, data: FpsProblemData) -> None:
        self.params = params
        self.data = data

    @staticmethod
    def fig_to_div(fig: go.Figure, include_js: bool = False) -> str:
        return to_html(fig, include_plotlyjs=include_js, full_html=False, config=dict(displayModeBar=True, responsive=True))

    def build(
        self,
        nonlinear_results: dict[str, TimeHistoryResult],
        equivalent_results: dict[str, TimeHistoryResult],
        floor_spectra: dict[str, tuple[FloorSpectrumResult, FloorSpectrumResult]],
    ) -> str:
        kobe_nl = nonlinear_results["Kobe"]
        sylmar_nl = nonlinear_results["Sylmar"]
        kobe_el = equivalent_results["Kobe"]
        sylmar_el = equivalent_results["Sylmar"]

        fig_kobe_nl = FigureFactory.time_history_5panel("Kobe — Nonlinear Bilinear FPS Response", kobe_nl, self.data.weight)
        fig_kobe_nl_h = FigureFactory.hysteresis("Kobe — Nonlinear Hysteresis F-u", kobe_nl, self.data.weight)
        fig_sylmar_nl = FigureFactory.time_history_5panel("Sylmar — Nonlinear Bilinear FPS Response", sylmar_nl, self.data.weight)
        fig_sylmar_nl_h = FigureFactory.hysteresis("Sylmar — Nonlinear Hysteresis F-u", sylmar_nl, self.data.weight)

        fig_kobe_el = FigureFactory.time_history_5panel("Kobe — Equivalent Linear Response", kobe_el, self.data.weight)
        fig_kobe_el_h = FigureFactory.hysteresis("Kobe — Equivalent Linear F-u", kobe_el, self.data.weight)
        fig_sylmar_el = FigureFactory.time_history_5panel("Sylmar — Equivalent Linear Response", sylmar_el, self.data.weight)
        fig_sylmar_el_h = FigureFactory.hysteresis("Sylmar — Equivalent Linear F-u", sylmar_el, self.data.weight)

        fig_floor_kobe = FigureFactory.floor_spectra_comparison(
            "Kobe Floor Spectrum Comparison (ζp = 2%)",
            floor_spectra["Kobe"][0],
            floor_spectra["Kobe"][1],
        )
        fig_floor_sylmar = FigureFactory.floor_spectra_comparison(
            "Sylmar Floor Spectrum Comparison (ζp = 2%)",
            floor_spectra["Sylmar"][0],
            floor_spectra["Sylmar"][1],
        )

        sections = [
            self.fig_to_div(fig_kobe_nl, include_js=False),
            self.fig_to_div(fig_kobe_nl_h, include_js=False),
            self.fig_to_div(fig_sylmar_nl, include_js=False),
            self.fig_to_div(fig_sylmar_nl_h, include_js=False),
            self.fig_to_div(fig_kobe_el, include_js=False),
            self.fig_to_div(fig_kobe_el_h, include_js=False),
            self.fig_to_div(fig_sylmar_el, include_js=False),
            self.fig_to_div(fig_sylmar_el_h, include_js=False),
            self.fig_to_div(fig_floor_kobe, include_js=False),
            self.fig_to_div(fig_floor_sylmar, include_js=False),
        ]

        summary_table = self._build_peak_table(nonlinear_results, equivalent_results)

        return dedent(
            f"""
            <!DOCTYPE HTML>
            <html lang="en">
            <head>
              <meta charset="utf-8" />
              <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
              <title>CE223 – Friction Pendulum (FPS) — Kobe &amp; Sylmar</title>
              <link rel="stylesheet" href="../../assets/css/main.css" />
              <noscript><link rel="stylesheet" href="../../assets/css/noscript.css" /></noscript>
              <style>
                /* Match site header width (Phantom shell); same pattern as isolator_dashboard.html */
                .ce223-dashboard .container {{
                    max-width: 68em;
                    margin-left: auto;
                    margin-right: auto;
                }}
                #main.ce223-dashboard {{
                    padding-top: 0.75rem;
                }}
                .inner-report {{
                    font-family: Arial, Helvetica, sans-serif;
                    font-size: 1rem;
                    line-height: 1.55;
                    color: #2c3e50;
                    max-width: 100%;
                    margin: 0;
                    padding: 0 1rem;
                }}
                .inner-report header.major {{
                    text-align: left;
                    margin-bottom: 1.25rem;
                }}
                .inner-report header.major h2 {{
                    font-size: 1.85rem;
                    font-weight: 700;
                    color: #003262;
                    margin: 0 0 0.5rem 0;
                    letter-spacing: 0.01em;
                    text-transform: none;
                }}
                .summary-lead {{
                    font-size: 1.05rem;
                    color: #6b7280;
                    max-width: 68em;
                    margin: 0;
                    line-height: 1.6;
                }}
                .inner-report .box {{
                    background: #fff;
                    border: 1px solid #e5e7eb;
                    border-radius: 8px;
                    padding: 1.5rem;
                    margin-bottom: 1.5rem;
                }}
                .inner-report .box h3 {{
                    font-size: 1.2rem;
                    font-weight: 700;
                    color: #003262;
                    margin: 0 0 0.75rem 0;
                    border-left: 4px solid #FDB515;
                    padding-left: 0.75rem;
                }}
                .inner-report .box p {{
                    margin: 0 0 0.75rem 0;
                }}
                .inner-report .box p:last-child {{
                    margin-bottom: 0;
                }}
                .report-section {{
                    background: #fff;
                    border: 1px solid #e5e7eb;
                    border-left: 4px solid #003262;
                    border-radius: 8px;
                    padding: 1.5rem;
                    margin-bottom: 1.5rem;
                }}
                .report-section h2 {{
                    font-size: 1.35rem;
                    font-weight: 700;
                    color: #003262;
                    margin: 0 0 0.5rem 0;
                }}
                .report-section > p {{
                    margin: 0 0 0.75rem 0;
                    color: #374151;
                }}
                .eq {{
                    background: #f9fafb;
                    border-left: 4px solid #003262;
                    padding: 0.9rem 1rem;
                    margin: 0.75rem 0;
                    overflow-x: auto;
                    -webkit-overflow-scrolling: touch;
                }}
                .eq mjx-container {{
                    min-width: max-content;
                }}
                .summary-table-wrap {{
                    overflow-x: auto;
                    -webkit-overflow-scrolling: touch;
                    margin: 1rem 0;
                }}
                .summary-table {{
                    width: 100%;
                    min-width: 32rem;
                    border-collapse: collapse;
                    font-size: 0.95rem;
                    border: 1px solid #e5e7eb;
                    border-radius: 6px;
                    overflow: hidden;
                }}
                .summary-table thead th {{
                    background: #003262;
                    color: #fff;
                    font-weight: 600;
                    text-align: center;
                    padding: 0.55rem 0.5rem;
                }}
                .summary-table thead th:first-child,
                .summary-table thead th:nth-child(2) {{
                    text-align: left;
                }}
                .summary-table tbody td {{
                    padding: 0.5rem 0.5rem;
                    border-top: 1px solid #e5e7eb;
                    color: #2c3e50;
                }}
                .summary-table tbody td:first-child {{
                    font-weight: 600;
                    color: #003262;
                }}
                .summary-table tbody td:nth-child(2) {{
                    text-align: left;
                }}
                .summary-table tbody td:nth-child(n+3) {{
                    text-align: right;
                }}
                .summary-table tbody tr:nth-child(even) {{
                    background: #f9fafb;
                }}
                .plot-embed {{
                    border: 1px solid #e5e7eb;
                    border-radius: 6px;
                    padding: 0.5rem;
                    background: #fff;
                    margin: 1rem 0;
                    overflow-x: auto;
                }}
                .plot-embed .plotly {{
                    max-width: 100%;
                }}
                .inner-report .js-plotly-plot .modebar {{
                    top: 56px !important;
                }}
                @media (max-width: 736px) {{
                    #main.ce223-dashboard {{
                        padding-top: 0.5rem;
                    }}
                    .inner-report {{
                        padding: 0 0.75rem;
                    }}
                    .inner-report header.major h2 {{
                        font-size: 1.45rem;
                    }}
                    .summary-lead {{
                        font-size: 1rem;
                    }}
                    .inner-report .box, .report-section {{
                        padding: 1rem;
                    }}
                    .inner-report .box h3 {{
                        font-size: 1.1rem;
                    }}
                    .report-section h2 {{
                        font-size: 1.2rem;
                    }}
                    .summary-table {{
                        font-size: 0.85rem;
                    }}
                    .summary-table thead th, .summary-table tbody td {{
                        padding: 0.4rem 0.35rem;
                    }}
                    .inner-report .js-plotly-plot .modebar {{
                        top: 74px !important;
                    }}
                }}
              </style>
              <script src="https://cdn.plot.ly/plotly-3.3.1.min.js"></script>
              <script>
                window.MathJax = {{
                  tex: {{
                    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                    processEscapes: true,
                    processEnvironments: true
                  }},
                  options: {{
                    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
                  }}
                }};
              </script>
              <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
              <script async src="../../assets/js/navigation.js"></script>
            </head>
            <body class="is-preload">
              <div id="page-wrapper">
                <header id="header"></header>

                <section id="main" class="wrapper style1 ce223-dashboard">
                  <div class="container inner-report">
                    <header class="major">
                      <h2>CE223 – Friction Pendulum (FPS) — Kobe &amp; Sylmar</h2>
                      <p class="summary-lead">
                        This report compares a <strong>rigid‑superstructure friction pendulum (FPS)</strong> idealization under two horizontal records:
                        <strong>Kobe KBU090</strong> and <strong>Sylmar 360°</strong>. Part (a) solves a <strong>nonlinear bilinear</strong> regularization with Newmark and return mapping;
                        part (b) uses an <strong>equivalent viscously damped linear</strong> model; part (c) reports <strong>floor spectra</strong> for linear nonstructural oscillators ($\\zeta_p=2\\%$)
                        driven by absolute deck acceleration $u_t(t)=\\ddot u(t)+\\ddot u_g(t)$.
                      </p>
                    </header>

                    <section class="box">
                      <h3>Model definition</h3>
                      <p>The rigid-superstructure FPS system is modeled as one SDOF in relative coordinates, and the governing equation is:</p>
                      <div class="eq">$$M\\ddot u(t) + F_r(u,\\dot u) = -M\\ddot u_g(t), \\quad F_r \\approx K_p u + Q\\,\\mathrm{{sgn}}(\\dot u).$$</div>
                      <p>To avoid force discontinuity at velocity reversals, a bilinear regularization with kinematic hardening and very small yield displacement is used. The matching relations are:</p>
                      <div class="eq">
                        $$F_y = \\mu Mg, \\quad k = \\frac{{F_y}}{{u_y}}, \\quad K_p=\\frac{{Mg}}{{R}}, \\quad H=\\frac{{kK_p}}{{k-K_p}}.$$
                      </div>
                      <p>Inputs: $M={self.data.total_mass:.3e}$ kg, $R={self.data.radius:.2f}$ m, $\\mu={self.data.friction_coefficient:.3f}$, $u_y={self.data.yield_displacement_mm:.4f}$ mm, $n_b={self.data.n_bearings}$.</p>
                      <p>Computed: $K_p={self.params.pendulum_stiffness/1e6:.3f}$ MN/m, $Q={self.params.characteristic_strength/1e6:.3f}$ MN, $k={self.params.elastic_stiffness/1e9:.3f}$ GN/m, $H={self.params.hardening_modulus/1e6:.3f}$ MN/m.</p>
                    </section>

                    <section class="box">
                      <h3>Numerical procedure</h3>
                      <p><strong>Part (a) Nonlinear Bilinear:</strong> At each time step $t_{{n+1}}$, the algorithm performs Newton iterations on displacement $u_{{n+1}}$. For each iterate it computes trial force $F^{{tr}}=k(u-u_n^p)$, trial yield check $f^{{tr}}=|F^{{tr}}-q_n|-F_y$, then applies either elastic update ($f^{{tr}}\\le 0$) or plastic correction ($f^{{tr}}>0$) with return mapping. The algorithmic tangent is $k$ (elastic) or $kH/(k+H)$ (plastic). Convergence is checked on both residual and displacement correction.</p>
                      <div class="eq">$$\\Delta\\gamma = \\frac{{f^{{tr}}}}{{k+H}}, \\quad F_{{n+1}} = F^{{tr}} - k\\Delta\\gamma\\,\\mathrm{{sgn}}\\!\\left(F^{{tr}}-q_n\\right), \\quad q_{{n+1}} = q_n + H\\Delta\\gamma\\,\\mathrm{{sgn}}\\!\\left(F^{{tr}}-q_n\\right).$$</div>
                      <p><strong>Part (b) Equivalent Linear:</strong> The effective stiffness and damping are iteratively updated from response amplitude. At each outer iteration, linear Newmark is solved with updated $(k_{{eff}}, c_{{eff}})$; then $u_{{max}}$ is re-estimated and properties are updated until stable.</p>
                      <div class="eq">$$k_{{eff}} = K_p + \\frac{{Q}}{{u_{{max}}}}, \\qquad \\zeta_{{eff}} = \\frac{{2Q}}{{\\pi k_{{eff}}u_{{max}}}}, \\qquad c_{{eff}} = 2\\zeta_{{eff}}\\sqrt{{M k_{{eff}}}}.$$</div>
                      <p><strong>Part (c) Floor Spectra:</strong> The input to NSC oscillators is the <em>absolute</em> floor acceleration from parts (a) and (b). For each $T_p$ with fixed $\\zeta_p=2\\%$, an SDOF is solved and the peak absolute component acceleration is extracted.</p>
                      <div class="eq">$$\\ddot z + 2\\zeta_p\\omega_p\\dot z + \\omega_p^2 z = -u_t(t), \\qquad a_{{p,abs}}(t)=\\ddot z(t)+u_t(t), \\qquad S_{{a,floor}}(T_p)=\\max|a_{{p,abs}}(t)|.$$</div>
                    </section>

                    <section class="box">
                      <h3>Peak response comparison</h3>
                      {summary_table}
                    </section>

                    <section class="report-section">
                      <h2>Part (a) — Nonlinear bilinear response</h2>
                      <p>For each ground motion, the time histories of isolation displacement, isolation velocity, restoring force, absolute structural acceleration, and ground acceleration are presented together with the corresponding force–displacement relation.</p>
                      <div class="plot-embed">{sections[0]}</div>
                      <div class="plot-embed">{sections[1]}</div>
                      <div class="plot-embed">{sections[2]}</div>
                      <div class="plot-embed">{sections[3]}</div>
                    </section>
                    <section class="report-section">
                      <h2>Part (b) — Equivalent linear response</h2>
                      <p>These plots use the same layout as part (a) for direct comparison. Differences reflect the limitations of a single linearized pair $(k_{{eff}},c_{{eff}})$ under transient loading and unloading.</p>
                      <div class="plot-embed">{sections[4]}</div>
                      <div class="plot-embed">{sections[5]}</div>
                      <div class="plot-embed">{sections[6]}</div>
                      <div class="plot-embed">{sections[7]}</div>
                    </section>
                    <section class="report-section">
                      <h2>Part (c) — Floor spectra ($\\zeta_p = 2\\%$)</h2>
                      <p>Each NSC is a linear 2% oscillator; the only difference between curves is the floor motion $u_t(t)=\\ddot u(t)+\\ddot u_g(t)$ from part (a) versus part (b). Oscillator periods $T_p$ are spaced uniformly in log-space from 0.01 s to 10.0 s (a logarithmic period axis cannot include $T_p=0$). The dotted reference marks $f_p=2$ Hz ($T_p=0.5$ s), a band often used for stiff acceleration-sensitive components.</p>
                      <p><strong>Note for $f_p$ above 2 Hz:</strong> The equivalent-linear isolation model does <em>not</em> guarantee a larger floor spectrum than the nonlinear one. Bilinear hysteresis and reversals can inject high-frequency content into the <em>true</em> $u_t(t)$, while the fitted viscously damped oscillator tends to produce a smoother deck motion with less energy above a few Hz. The NSC spectrum then often lies <em>above</em> the equivalent-linear curve in that band even though every oscillator is linear.</p>
                      <div class="plot-embed">{sections[8]}</div>
                      <div class="plot-embed">{sections[9]}</div>
                    </section>
                  </div>
                </section>
              </div>
            </body>
            </html>
            """
        ).strip()

    @staticmethod
    def _build_peak_table(
        nonlinear_results: dict[str, TimeHistoryResult],
        equivalent_results: dict[str, TimeHistoryResult],
    ) -> str:
        rows = []
        for motion in ("Kobe", "Sylmar"):
            nl = nonlinear_results[motion]
            el = equivalent_results[motion]
            rows.append(
                f"<tr><td>{motion}</td><td>Nonlinear</td><td>{nl.peak_displacement*1e3:.3f}</td><td>{nl.peak_velocity:.3f}</td><td>{nl.peak_force/1e6:.3f}</td><td>{nl.peak_abs_acc/G_SI:.3f}</td></tr>"
            )
            rows.append(
                f"<tr><td>{motion}</td><td>Equivalent Linear</td><td>{el.peak_displacement*1e3:.3f}</td><td>{el.peak_velocity:.3f}</td><td>{el.peak_force/1e6:.3f}</td><td>{el.peak_abs_acc/G_SI:.3f}</td></tr>"
            )
        return (
            '<div class="summary-table-wrap"><table class="summary-table"><thead><tr>'
            "<th scope='col'>Motion</th><th scope='col'>Model</th><th scope='col'>Peak |u| [mm]</th>"
            "<th scope='col'>Peak |u̇| [m/s]</th><th scope='col'>Peak |F| [MN]</th><th scope='col'>Peak |üt| [g]</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )


def main() -> None:
    problem_data = FpsProblemData()
    parameters = FpsParameterBuilder.from_problem_data(problem_data)
    constitutive = BilinearConstitutiveModel(parameters)

    kobe_record = GroundMotionLoader.load_peer_file(KOBE_PATH)
    sylmar_record = GroundMotionLoader.load_peer_file(SYLMAR_PATH)
    records = {"Kobe": kobe_record, "Sylmar": sylmar_record}

    nonlinear_solver = NonlinearNewmarkSolver(constitutive)
    equivalent_solver = EquivalentLinearSolver(parameters)

    nonlinear_results: dict[str, TimeHistoryResult] = {}
    equivalent_results: dict[str, TimeHistoryResult] = {}
    floor_spectra: dict[str, tuple[FloorSpectrumResult, FloorSpectrumResult]] = {}

    for name, record in records.items():
        nonlinear_result = nonlinear_solver.solve(record)
        equivalent_result = equivalent_solver.solve(record)
        nonlinear_results[name] = nonlinear_result
        equivalent_results[name] = equivalent_result

        exact_spec = FloorSpectrumCalculator.compute(nonlinear_result.absolute_acceleration, record.dt, damping_ratio=0.02)
        eq_spec = FloorSpectrumCalculator.compute(equivalent_result.absolute_acceleration, record.dt, damping_ratio=0.02)
        floor_spectra[name] = (exact_spec, eq_spec)

    report = HtmlReportBuilder(parameters, problem_data).build(nonlinear_results, equivalent_results, floor_spectra)
    OUTPUT_HTML.write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
