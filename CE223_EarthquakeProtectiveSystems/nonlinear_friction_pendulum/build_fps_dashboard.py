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
        min_period: float = 0.05,
        max_period: float = 2.0,
        n_points: int = 180,
        min_points_per_cycle: int = 24,
    ) -> FloorSpectrumResult:
        periods = np.linspace(min_period, max_period, n_points)
        frequencies = 1.0 / periods
        peaks = np.zeros_like(periods)
        support_abs_acceleration = np.asarray(support_abs_acceleration, dtype=float).ravel()
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
            peaks[i] = np.max(np.abs(component_abs_acc))

        return FloorSpectrumResult(periods=periods, frequencies=frequencies, peak_abs_component_acceleration=peaks)


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
        exact_periods = 1.0 / np.maximum(exact_spectrum.frequencies, 1e-12)
        equivalent_periods = 1.0 / np.maximum(equivalent_spectrum.frequencies, 1e-12)
        fig.add_trace(
            go.Scatter(
                x=exact_spectrum.frequencies,
                y=exact_spectrum.peak_abs_component_acceleration / G_SI,
                mode="lines",
                line=dict(color=MATLAB_COLORS["dark_blue"], width=2.6),
                name="Exact (nonlinear input)",
                customdata=np.column_stack((exact_periods,)),
                hovertemplate=(
                    "Frequency fp: %{x:.3f} Hz<br>"
                    "Period Tp: %{customdata[0]:.4f} s<br>"
                    "Peak NSC abs. accel.: %{y:.4f} g<br>"
                    "Model: Nonlinear floor input<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=equivalent_spectrum.frequencies,
                y=equivalent_spectrum.peak_abs_component_acceleration / G_SI,
                mode="lines",
                line=dict(color=MATLAB_COLORS["crimson"], width=2.4, dash="dash"),
                name="Equivalent linear input",
                customdata=np.column_stack((equivalent_periods,)),
                hovertemplate=(
                    "Frequency fp: %{x:.3f} Hz<br>"
                    "Period Tp: %{customdata[0]:.4f} s<br>"
                    "Peak NSC abs. accel.: %{y:.4f} g<br>"
                    "Model: Equivalent-linear floor input<extra></extra>"
                ),
            )
        )
        fig.add_vline(x=2.0, line_dash="dot", line_color=MATLAB_COLORS["black"])
        fig.update_layout(
            template="plotly_white",
            height=430,
            title=dict(text=title, x=0.5, xanchor="center", font=dict(size=22)),
            xaxis=dict(title="NSC Frequency fp [Hz]", range=[0.5, 20.0], title_font=dict(size=16), tickfont=dict(size=13)),
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
            self.fig_to_div(fig_kobe_nl, include_js=True),
            self.fig_to_div(fig_kobe_nl_h),
            self.fig_to_div(fig_sylmar_nl),
            self.fig_to_div(fig_sylmar_nl_h),
            self.fig_to_div(fig_kobe_el),
            self.fig_to_div(fig_kobe_el_h),
            self.fig_to_div(fig_sylmar_el),
            self.fig_to_div(fig_sylmar_el_h),
            self.fig_to_div(fig_floor_kobe),
            self.fig_to_div(fig_floor_sylmar),
        ]

        summary_table = self._build_peak_table(nonlinear_results, equivalent_results)

        return dedent(
            f"""
            <!doctype html>
            <html lang="en">
            <head>
              <meta charset="utf-8" />
              <meta name="viewport" content="width=device-width, initial-scale=1" />
              <title>CE223 FPS Dashboard — Kobe & Sylmar</title>
              <style>
                :root {{
                  --ucb-blue: #003262;
                  --border: rgba(0,50,98,0.18);
                  --bg: #f8fafc;
                  --text: #2C3E50;
                }}
                * {{ box-sizing: border-box; }}
                body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; color: var(--text); background: var(--bg); line-height: 1.6; font-size: 1.03rem; }}
                .wrap {{ max-width: 1080px; margin: 0 auto; padding: 2rem 1.3rem 3rem; }}
                .site-header {{ background:#ffffff; border-bottom:1px solid #e5e7eb; }}
                .site-header-inner {{ max-width:1080px; margin:0 auto; padding:0.8rem 1.3rem; display:flex; align-items:center; justify-content:space-between; gap:1rem; }}
                .site-header-brand {{ color:var(--ucb-blue); font-weight:700; text-decoration:none; font-size:1.05rem; }}
                .site-header-links {{ display:flex; flex-wrap:wrap; gap:0.55rem; }}
                .site-header-links a {{ text-decoration:none; color:var(--ucb-blue); border:1px solid var(--border); border-radius:999px; padding:0.35rem 0.7rem; font-size:0.9rem; background:#fff; }}
                header {{ border-bottom: 3px solid var(--ucb-blue); padding-bottom: 1rem; margin-bottom: 1rem; }}
                header h1 {{ color: var(--ucb-blue); margin: 0 0 0.35rem; font-size: 2rem; }}
                .card {{ background: #fff; border: 1px solid var(--border); border-radius: 12px; padding: 1rem 1.1rem; margin: 1rem 0; }}
                .card h2 {{ font-size: 1.45rem; margin: 0 0 0.55rem; }}
                .eq {{ background:#f9fafb; border-left: 4px solid var(--ucb-blue); padding:0.7rem 0.9rem; margin:0.7rem 0; }}
                table {{ width:100%; border-collapse: collapse; margin-top:0.7rem; }}
                th, td {{ border:1px solid #e5e7eb; padding:0.45rem 0.55rem; text-align:left; }}
                th {{ background: var(--ucb-blue); color:#fff; }}
                .plot {{ border: 1px solid var(--border); border-radius: 10px; padding: 0.45rem; margin-top: 0.8rem; background:#fff; }}
              </style>
              <script>
                window.MathJax = {{
                  tex: {{
                    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
                  }}
                }};
              </script>
              <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
            </head>
            <body>
              <div class="site-header">
                <div class="site-header-inner">
                  <a class="site-header-brand" href="../../webpage/index.html">Facundo L. Pfeffer</a>
                  <div class="site-header-links">
                    <a href="../../webpage/index.html">Portfolio home</a>
                    <a href="../../webpage/cee223-earthquake-protective-systems.html">CE223 page</a>
                    <a href="../../webpage/cee225-dynamics.html">CEE225 dynamics</a>
                  </div>
                </div>
              </div>
              <div class="wrap">
                <header>
                  <h1>Friction Pendulum System Dashboard — Kobe & Sylmar</h1>
                  <p>Part (a): nonlinear bilinear regularization solved with Newmark + return mapping. Part (b): equivalent viscously damped linear model. Part (c): floor spectra for NSCs (ζp = 2%).</p>
                </header>

                <div class="card">
                  <h2>Model Definition</h2>
                  <p>The rigid-superstructure FPS system is modeled as one SDOF in relative coordinates, and the governing equation is:</p>
                  <div class="eq">$$M\\ddot u(t) + F_r(u,\\dot u) = -M\\ddot u_g(t), \\quad F_r \\approx K_p u + Q\\,\\mathrm{{sgn}}(\\dot u).$$</div>
                  <p>To avoid force discontinuity at velocity reversals, a bilinear regularization with kinematic hardening and very small yield displacement is used. The matching relations are:</p>
                  <div class="eq">
                    $$F_y = \\mu Mg, \\quad k = \\frac{{F_y}}{{u_y}}, \\quad K_p=\\frac{{Mg}}{{R}}, \\quad H=\\frac{{kK_p}}{{k-K_p}}.$$
                  </div>
                  <p>Inputs: $M={self.data.total_mass:.3e}$ kg, $R={self.data.radius:.2f}$ m, $\\mu={self.data.friction_coefficient:.3f}$, $u_y={self.data.yield_displacement_mm:.4f}$ mm, $n_b={self.data.n_bearings}$.</p>
                  <p>Computed: $K_p={self.params.pendulum_stiffness/1e6:.3f}$ MN/m, $Q={self.params.characteristic_strength/1e6:.3f}$ MN, $k={self.params.elastic_stiffness/1e9:.3f}$ GN/m, $H={self.params.hardening_modulus/1e6:.3f}$ MN/m.</p>
                </div>

                <div class="card">
                  <h2>Numerical Procedure Used</h2>
                  <p><strong>Part (a) Nonlinear Bilinear:</strong> At each time step $t_{{n+1}}$, the algorithm performs Newton iterations on displacement $u_{{n+1}}$. For each iterate it computes trial force $F^{{tr}}=k(u-u_n^p)$, trial yield check $f^{{tr}}=|F^{{tr}}-q_n|-F_y$, then applies either elastic update ($f^{{tr}}\\le 0$) or plastic correction ($f^{{tr}}>0$) with return mapping. The algorithmic tangent is $k$ (elastic) or $kH/(k+H)$ (plastic). Convergence is checked on both residual and displacement correction.</p>
                  <div class="eq">$$\\Delta\\gamma = \\frac{{f^{{tr}}}}{{k+H}}, \\quad F_{{n+1}} = F^{{tr}} - k\\Delta\\gamma\\,\\mathrm{{sgn}}(F^{{tr}}-q_n), \\quad q_{{n+1}} = q_n + H\\Delta\\gamma\\,\\mathrm{{sgn}}(F^{{tr}}-q_n).$$</div>
                  <p><strong>Part (b) Equivalent Linear:</strong> The effective stiffness and damping are iteratively updated from response amplitude. At each outer iteration, linear Newmark is solved with updated $(k_{{eff}}, c_{{eff}})$; then $u_{{max}}$ is re-estimated and properties are updated until stable.</p>
                  <div class="eq">$$k_{{eff}} = K_p + \\frac{{Q}}{{u_{{max}}}}, \\qquad \\zeta_{{eff}} = \\frac{{2Q}}{{\\pi k_{{eff}}u_{{max}}}}, \\qquad c_{{eff}} = 2\\zeta_{{eff}}\\sqrt{{M k_{{eff}}}}.$$</div>
                  <p><strong>Part (c) Floor Spectra:</strong> The input to NSC oscillators is the <em>absolute</em> floor acceleration from parts (a) and (b). For each $T_p$ with fixed $\\zeta_p=2\\%$, an SDOF is solved and the peak absolute component acceleration is extracted.</p>
                  <div class="eq">$$\\ddot z + 2\\zeta_p\\omega_p\\dot z + \\omega_p^2 z = -u_t(t), \\qquad a_{{p,abs}}(t)=\\ddot z(t)+u_t(t), \\qquad S_{{a,floor}}(T_p)=\\max|a_{{p,abs}}(t)|.$$</div>
                </div>

                <div class="card">
                  <h2>Peak Response Comparison</h2>
                  {summary_table}
                </div>

                <div class="card"><h2>Part (a) — Nonlinear Bilinear Response</h2><p>For each ground motion, the time histories of isolation displacement, isolation velocity, restoring force, absolute structural acceleration, and ground acceleration are presented together with the corresponding force-displacement relation.</p><div class="plot">{sections[0]}</div><div class="plot">{sections[1]}</div><div class="plot">{sections[2]}</div><div class="plot">{sections[3]}</div></div>
                <div class="card"><h2>Part (b) — Equivalent Linear Response</h2><p>These plots are generated in the same format as part (a), so direct visual comparison is immediate. Differences mainly reflect the inability of a single linearized pair $(k_{{eff}},c_{{eff}})$ to reproduce all nonlinear hysteretic effects during transient loading and unloading.</p><div class="plot">{sections[4]}</div><div class="plot">{sections[5]}</div><div class="plot">{sections[6]}</div><div class="plot">{sections[7]}</div></div>
                <div class="card"><h2>Part (c) — Floor Spectra (ζp = 2%)</h2><p>Absolute floor acceleration $u_t(t)=\\ddot u(t)+\\ddot u_g(t)$ is used as support input to nonstructural-component oscillators. The vertical line at 2 Hz marks the high-frequency range typically associated with stiff components.</p><p>When the equivalent-linear spectrum diverges from the nonlinear spectrum in this frequency range, the equivalent-linear approximation may misestimate acceleration-sensitive nonstructural demands.</p><div class="plot">{sections[8]}</div><div class="plot">{sections[9]}</div></div>
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
            "<table><thead><tr><th>Motion</th><th>Model</th><th>Peak |u| [mm]</th>"
            "<th>Peak |u̇| [m/s]</th><th>Peak |F| [MN]</th><th>Peak |üt| [g]</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
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
