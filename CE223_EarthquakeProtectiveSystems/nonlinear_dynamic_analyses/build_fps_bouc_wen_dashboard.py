"""Self-contained dashboard for CE223 Problem 1: FPS with a Bouc--Wen model.

The script explicitly works through the four requested items:
- part (a): expected ideal FPS hysteresis for the prescribed cyclic motion,
- part (b): Bouc--Wen calibration for the same cyclic test,
- part (c): nonlinear time-history analysis for Kobe and Sylmar, and
- part (d): comparison against an internal bilinear plasticity model.

Assumption stated explicitly:
- because the numerical results from Homework Assignment 4 are not embedded in this
  file, part (d) is carried out against a self-contained bilinear plasticity model
  built from the same FPS properties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import numpy as np
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.subplots import make_subplots


# -----------------------------------------------------------------------------
# Constants, paths, and plotting style
# -----------------------------------------------------------------------------

G_SI = 9.80665  # m/s²
BASE_DIR = Path(__file__).resolve().parent
CE223_DIR = BASE_DIR.parent
HIGHLIGHTED_HTML_DIR = CE223_DIR / "highlighted_htmls"
HIGHLIGHTED_HTML_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_HTML = HIGHLIGHTED_HTML_DIR / "CE223_FPS_BoucWen_Kobe_Sylmar.html"


def first_existing_path(*candidates: Path) -> Path:
    """Return the first existing path from a list of candidates."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    formatted = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not locate the required ground-motion file. Checked:\n" + formatted
    )


KOBE_CANDIDATES = (
    BASE_DIR / "RSN1108_KOBE_KBU090.AT2",
    BASE_DIR / "input_ground_motion" / "RSN1108_KOBE_KBU090.AT2",
    BASE_DIR.parent / "input_ground_motion" / "RSN1108_KOBE_KBU090.AT2",
)
SYLMAR_CANDIDATES = (
    BASE_DIR / "SYLMAR360.txt",
    BASE_DIR / "input_ground_motion" / "SYLMAR360.txt",
    BASE_DIR.parent / "input_ground_motion" / "SYLMAR360.txt",
)

MATLAB_COLORS = {
    "dark_blue": "rgb(0, 70, 140)",
    "black": "rgb(20, 20, 20)",
    "crimson": "rgb(180, 20, 60)",
    "dark_green": "rgb(0, 100, 0)",
    "gray": "rgb(110, 110, 110)",
    "gold": "rgb(210, 150, 0)",
    "purple": "rgb(90, 50, 130)",
}


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FpsProblemData:
    """Input values stated in the FPS problem statement."""
    total_mass: float = 1.47e6
    n_bearings: int = 15
    radius: float = 1.0
    friction_coefficient: float = 0.03
    yield_displacement_mm: float = 0.03

    @property
    def weight(self) -> float:
        return self.total_mass * G_SI

    @property
    def bearing_weight(self) -> float:
        return self.weight / float(self.n_bearings)

    @property
    def yield_displacement(self) -> float:
        return self.yield_displacement_mm * 1.0e-3


@dataclass(frozen=True)
class BoucWenFpsParameters:
    """Bouc--Wen parameters and derived FPS stiffness/strength values."""
    total_mass: float
    n_bearings: int
    radius: float
    friction_coefficient: float
    yield_displacement: float
    alpha: float
    beta_bw: float
    gamma_bw: float
    exponent_n: float

    @property
    def total_weight(self) -> float:
        return self.total_mass * G_SI

    @property
    def bearing_weight(self) -> float:
        return self.total_weight / float(self.n_bearings)

    @property
    def total_k2(self) -> float:
        return self.total_weight / self.radius

    @property
    def bearing_k2(self) -> float:
        return self.bearing_weight / self.radius

    @property
    def total_k1(self) -> float:
        return self.total_k2 / self.alpha

    @property
    def bearing_k1(self) -> float:
        return self.bearing_k2 / self.alpha

    @property
    def total_characteristic_strength(self) -> float:
        return self.friction_coefficient * self.total_weight

    @property
    def bearing_characteristic_strength(self) -> float:
        return self.friction_coefficient * self.bearing_weight

    @property
    def total_hysteretic_strength(self) -> float:
        return (1.0 - self.alpha) * self.total_k1 * self.yield_displacement

    @property
    def bearing_hysteretic_strength(self) -> float:
        return (1.0 - self.alpha) * self.bearing_k1 * self.yield_displacement


@dataclass(frozen=True)
class BilinearFpsParameters:
    """Parameters used by the comparison bilinear plasticity model."""
    mass: float
    pendulum_stiffness: float
    characteristic_strength: float
    yield_force: float
    elastic_stiffness: float
    hardening_modulus: float
    post_yield_stiffness: float


@dataclass
class GroundMotionRecord:
    """Acceleration record stored in SI units with a constant time step."""
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
    """Response histories returned by the dynamic solvers."""
    time: np.ndarray
    ground_acceleration: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    relative_acceleration: np.ndarray
    absolute_acceleration: np.ndarray
    restoring_force: np.ndarray
    hysteretic_parameter: np.ndarray | None = None

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
class CyclicTestResult:
    """Cyclic displacement-control response for one bearing."""
    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    hysteretic_parameter: np.ndarray
    bouc_wen_force: np.ndarray
    ideal_force: np.ndarray
    bearing_weight: float
    cycle_frequency_hz: float
    n_cycles: int

    @property
    def bouc_wen_force_over_weight(self) -> np.ndarray:
        return self.bouc_wen_force / self.bearing_weight

    @property
    def ideal_force_over_weight(self) -> np.ndarray:
        return self.ideal_force / self.bearing_weight

    @property
    def last_cycle_mask(self) -> np.ndarray:
        period = 1.0 / self.cycle_frequency_hz
        return self.time >= (self.n_cycles - 1) * period


@dataclass(frozen=True)
class CalibrationCaseResult:
    """One Bouc--Wen calibration trial and its cyclic response."""
    label: str
    parameters: BoucWenFpsParameters
    cyclic_result: CyclicTestResult


@dataclass
class BilinearState:
    """Internal variables for the one-dimensional bilinear return mapping."""
    plastic_displacement: float = 0.0
    back_force: float = 0.0


# -----------------------------------------------------------------------------
# Ground-motion input
# -----------------------------------------------------------------------------


class GroundMotionLoader:
    """Reads AT2-style or two-column ground-motion acceleration files."""
    @staticmethod
    def load_acceleration_file(path: Path, name: str | None = None) -> GroundMotionRecord:
        """Load one acceleration record and return it in m/s²."""
        if not path.exists():
            raise FileNotFoundError(f"Ground motion file not found: {path}")

        # AT2 files usually store DT in the header; plain text exports may not.
        dt = GroundMotionLoader._parse_dt(path)
        numeric_rows = GroundMotionLoader._read_numeric_rows(path)
        if numeric_rows.size == 0:
            raise ValueError(f"No numeric acceleration data found in: {path}")

        # Two-column files are interpreted as [time, acceleration in g].
        if numeric_rows.ndim == 2 and numeric_rows.shape[1] >= 2 and GroundMotionLoader._looks_like_time_column(numeric_rows[:, 0]):
            time = numeric_rows[:, 0]
            acc_g = numeric_rows[:, 1]
            dt_from_time = float(np.median(np.diff(time)))
            dt = dt_from_time if dt is None else dt
        else:
            acc_g = numeric_rows.ravel()

        if dt is None:
            raise ValueError(f"Could not parse a time step from {path}. Add a DT=... header or use a two-column time/acceleration file.")
        if acc_g.size < 2:
            raise ValueError(f"Ground motion file has too few data points: {path}")

        return GroundMotionRecord(
            name=name or path.stem,
            dt=float(dt),
            acceleration_mps2=np.asarray(acc_g, dtype=float).ravel() * G_SI,
        )

    @staticmethod
    def _parse_dt(path: Path) -> float | None:
        """Extract the time step from a header containing DT."""
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                upper = line.upper()
                if "DT=" not in upper and "DT =" not in upper:
                    continue
                rhs = upper.split("DT", maxsplit=1)[1].replace("=", " ")
                rhs = rhs.replace("SEC", " ").replace(",", " ")
                for token in rhs.split():
                    try:
                        return float(token)
                    except ValueError:
                        continue
        return None

    @staticmethod
    def _read_numeric_rows(path: Path) -> np.ndarray:
        """Read numeric rows while skipping headers and comments."""
        rows: list[list[float]] = []
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                clean = line.strip().replace(",", " ")
                if not clean or clean.startswith(("#", "%", "!")):
                    continue

                row: list[float] = []
                for token in clean.split():
                    token = token.replace("D", "E").replace("d", "E")
                    try:
                        row.append(float(token))
                    except ValueError:
                        row = []
                        break

                if row:
                    rows.append(row)

        if not rows:
            return np.array([], dtype=float)
        row_lengths = {len(row) for row in rows}
        if len(row_lengths) == 1:
            return np.asarray(rows, dtype=float)
        return np.asarray([value for row in rows for value in row], dtype=float)

    @staticmethod
    def _looks_like_time_column(values: np.ndarray) -> bool:
        """Detect a uniformly spaced first column."""
        if values.size < 3:
            return False
        diffs = np.diff(values)
        if not np.all(diffs > 0.0):
            return False
        return bool(np.std(diffs) <= 1e-4 * max(abs(float(np.mean(diffs))), 1e-12))


# -----------------------------------------------------------------------------
# Mechanical parameter construction
# -----------------------------------------------------------------------------


class FpsParameterBuilder:
    """Builds calibrated parameter objects from the problem statement."""
    @staticmethod
    def bouc_wen_from_problem_data(
        data: FpsProblemData,
        beta_bw: float = 0.5,
        gamma_bw: float = 0.5,
        exponent_n: float = 5.0,
        alpha: float | None = None,
    ) -> BoucWenFpsParameters:
        """Match FPS stiffness/strength and assign Bouc--Wen shape parameters."""
        if alpha is None:
            # This alpha enforces both alpha*K1 = K2 and (1−α)*K1*uy = mu*W.
            ratio = data.yield_displacement / (data.friction_coefficient * data.radius)
            alpha = ratio / (1.0 + ratio)

        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1.")
        if data.yield_displacement <= 0.0:
            raise ValueError("yield displacement must be positive.")
        if beta_bw < 0.0 or gamma_bw < 0.0:
            raise ValueError("Bouc-Wen beta and gamma should be nonnegative for this calibration.")
        if exponent_n < 1.0:
            raise ValueError("The Bouc-Wen exponent should be at least 1.0 for this implementation.")

        return BoucWenFpsParameters(
            total_mass=data.total_mass,
            n_bearings=data.n_bearings,
            radius=data.radius,
            friction_coefficient=data.friction_coefficient,
            yield_displacement=data.yield_displacement,
            alpha=float(alpha),
            beta_bw=float(beta_bw),
            gamma_bw=float(gamma_bw),
            exponent_n=float(exponent_n),
        )

    @staticmethod
    def bilinear_from_problem_data(data: FpsProblemData) -> BilinearFpsParameters:
        """Create the ideal bilinear FPS parameters for comparison."""
        weight = data.weight
        kp = weight / data.radius  # Geometric FPS stiffness, W/R.
        q = data.friction_coefficient * weight  # Coulomb strength, mu*W.
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


# -----------------------------------------------------------------------------
# Force models and nonlinear solvers
# -----------------------------------------------------------------------------


class BoucWenForceModel:
    """Evaluates total and per-bearing restoring forces."""
    def __init__(self, parameters: BoucWenFpsParameters) -> None:
        self.parameters = parameters

    def total_force(self, displacement: np.ndarray | float, z: np.ndarray | float) -> np.ndarray | float:
        """Return the total isolator force for the full bearing set."""
        p = self.parameters
        return p.alpha * p.total_k1 * displacement + (1.0 - p.alpha) * p.total_k1 * p.yield_displacement * z

    def bearing_force(self, displacement: np.ndarray | float, z: np.ndarray | float) -> np.ndarray | float:
        """Return the force in one representative bearing."""
        p = self.parameters
        return p.alpha * p.bearing_k1 * displacement + (1.0 - p.alpha) * p.bearing_k1 * p.yield_displacement * z

    def ideal_bearing_force(self, displacement: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Return the ideal FPS force used as the cyclic target."""
        p = self.parameters
        return p.bearing_k2 * displacement + p.bearing_characteristic_strength * sign_with_memory(velocity)


class BoucWenCyclicSolver:
    """Integrates the Bouc--Wen internal variable for prescribed cyclic motion."""
    def __init__(self, model: BoucWenForceModel, points_per_cycle: int = 12000) -> None:
        self.model = model
        self.points_per_cycle = points_per_cycle

    def solve(self, amplitude: float = 0.4, frequency_hz: float = 1.0, n_cycles: int = 3) -> CyclicTestResult:
        """Run a displacement-controlled sinusoidal cyclic test."""
        if amplitude <= 0.0:
            raise ValueError("amplitude must be positive.")
        if frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive.")
        if n_cycles < 1:
            raise ValueError("n_cycles must be at least 1.")

        p = self.model.parameters
        period = 1.0 / frequency_hz
        dt = period / float(self.points_per_cycle)
        time = np.arange(0.0, n_cycles * period + 0.5 * dt, dt)
        omega = 2.0 * math.pi * frequency_hz
        displacement = amplitude * np.sin(omega * time)
        velocity = amplitude * omega * np.cos(omega * time)
        z = np.zeros(time.size, dtype=float)

        for i in range(1, time.size):
            # The FPS yield displacement is extremely small. A backward-Euler update
            # avoids the RK overshoot that previously had to be hidden by clipping.
            z[i] = solve_bouc_wen_z_backward_euler(
                z_old=float(z[i - 1]),
                velocity_new=float(velocity[i]),
                dt=float(time[i] - time[i - 1]),
                beta_bw=p.beta_bw,
                gamma_bw=p.gamma_bw,
                exponent_n=p.exponent_n,
                yield_displacement=p.yield_displacement,
            )

        bw_force = self.model.bearing_force(displacement, z)
        ideal_force = self.model.ideal_bearing_force(displacement, velocity)

        return CyclicTestResult(
            time=time,
            displacement=displacement,
            velocity=velocity,
            hysteretic_parameter=z,
            bouc_wen_force=np.asarray(bw_force, dtype=float),
            ideal_force=np.asarray(ideal_force, dtype=float),
            bearing_weight=p.bearing_weight,
            cycle_frequency_hz=frequency_hz,
            n_cycles=n_cycles,
        )


class BoucWenDynamicSolver:
    """State-vector Newmark solver for the Bouc--Wen FPS dynamic response.

    The Newton unknown is the same state vector used in the theory,
    y = [u, u_dot, z]^T. The mechanical part is enforced with the
    average-acceleration Newmark equations, and the Bouc--Wen internal
    variable is advanced with a backward-Euler equation.
    """

    def __init__(
        self,
        model: BoucWenForceModel,
        max_internal_dt: float = 1.0e-4,
        beta_newmark: float = 1.0 / 4.0,
        gamma_newmark: float = 1.0 / 2.0,
        tolerance: float = 1.0e-8,
        max_iterations: int = 40,
    ) -> None:
        self.model = model
        self.max_internal_dt = max_internal_dt
        self.beta_newmark = beta_newmark
        self.gamma_newmark = gamma_newmark
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def solve(
        self,
        record: GroundMotionRecord,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> TimeHistoryResult:
        """Integrate the state vector over one ground-motion record."""
        p = self.model.parameters
        time = record.time_array
        ground_acc = record.acceleration_mps2
        n_steps = ground_acc.size

        tolerance = self.tolerance if tolerance is None else tolerance
        max_iterations = self.max_iterations if max_iterations is None else max_iterations

        y = np.zeros((n_steps, 3), dtype=float)  # columns: u, u_dot, z
        a = np.zeros(n_steps, dtype=float)

        a[0] = self._acceleration(y[0], float(ground_acc[0]))

        for i in range(1, n_steps):
            dt_record = float(time[i] - time[i - 1])
            n_sub = max(1, int(math.ceil(dt_record / self.max_internal_dt)))
            h = dt_record / float(n_sub)
            ag0 = float(ground_acc[i - 1])
            ag1 = float(ground_acc[i])

            y_sub = y[i - 1].copy()
            a_sub = float(a[i - 1])

            for j in range(n_sub):
                r1 = (j + 1.0) / float(n_sub)
                ag_new = ag0 + (ag1 - ag0) * r1
                debug_context = {
                    "record_name": record.name,
                    "record_step": i,
                    "record_steps": n_steps - 1,
                    "substep": j + 1,
                    "substeps": n_sub,
                    "time_old": float(time[i - 1] + j * h),
                    "time_new": float(time[i - 1] + (j + 1) * h),
                    "dt_record": dt_record,
                    "dt_substep": h,
                    "ag_old": ag0,
                    "ag_new": ag_new,
                    "u_old": float(y_sub[0]),
                    "v_old": float(y_sub[1]),
                    "a_old": a_sub,
                    "z_old": float(y_sub[2]),
                }

                try:
                    y_sub = self._solve_state_step(
                        y_previous=y_sub,
                        a_previous=a_sub,
                        ground_acceleration_next=ag_new,
                        dt=h,
                        tolerance=tolerance,
                        max_iterations=max_iterations,
                        debug_context=debug_context,
                    )
                except RuntimeError as exc:
                    raise RuntimeError(str(exc)) from exc

                if not np.all(np.isfinite(y_sub)):
                    raise RuntimeError(self._format_state_failure(debug_context, y_sub))

                a_sub = self._acceleration(y_sub, ag_new)

            y[i] = y_sub
            a[i] = a_sub

        restoring_force = np.asarray(self.model.total_force(y[:, 0], y[:, 2]), dtype=float)
        absolute_acc = a + ground_acc

        return TimeHistoryResult(
            time=time,
            ground_acceleration=ground_acc,
            displacement=y[:, 0],
            velocity=y[:, 1],
            relative_acceleration=a,
            absolute_acceleration=absolute_acc,
            restoring_force=restoring_force,
            hysteretic_parameter=y[:, 2],
        )

    def _acceleration(self, state: np.ndarray, ground_acceleration: float) -> float:
        """Relative acceleration from equilibrium at state [u, u_dot, z]."""
        p = self.model.parameters
        restoring_force = float(self.model.total_force(float(state[0]), float(state[2])))
        return -ground_acceleration - restoring_force / p.total_mass

    def _zdot(self, velocity: float, z_value: float) -> float:
        """Bouc--Wen z-rate evaluated from the state-vector components."""
        p = self.model.parameters
        return bouc_wen_z_rate(
            velocity=velocity,
            z=z_value,
            beta_bw=p.beta_bw,
            gamma_bw=p.gamma_bw,
            exponent_n=p.exponent_n,
            yield_displacement=p.yield_displacement,
        )

    def _state_residual(
        self,
        y_next: np.ndarray,
        y_previous: np.ndarray,
        a_previous: float,
        ground_acceleration_next: float,
        dt: float,
    ) -> np.ndarray:
        """Residual for Newmark kinematics and backward-Euler z evolution."""
        beta = self.beta_newmark
        gamma = self.gamma_newmark

        u_next, v_next, z_next = map(float, y_next)
        u_previous, v_previous, z_previous = map(float, y_previous)
        a_next = self._acceleration(y_next, ground_acceleration_next)

        r_u = (
            u_next
            - u_previous
            - dt * v_previous
            - dt * dt * ((0.5 - beta) * a_previous + beta * a_next)
        )
        r_v = (
            v_next
            - v_previous
            - dt * ((1.0 - gamma) * a_previous + gamma * a_next)
        )
        r_z = z_next - z_previous - dt * self._zdot(v_next, z_next)

        return np.array([r_u, r_v, r_z], dtype=float)

    def _finite_difference_jacobian(
        self,
        y_next: np.ndarray,
        y_previous: np.ndarray,
        a_previous: float,
        ground_acceleration_next: float,
        dt: float,
        residual_at_y: np.ndarray,
    ) -> np.ndarray:
        """Finite-difference Jacobian for the three-component state residual."""
        jacobian = np.zeros((3, 3), dtype=float)
        perturbations = np.array([
            max(1.0e-8 * max(abs(float(y_next[0])), 1.0), 1.0e-10),
            max(1.0e-8 * max(abs(float(y_next[1])), 1.0), 1.0e-10),
            max(1.0e-8 * max(abs(float(y_next[2])), 1.0), 1.0e-10),
        ], dtype=float)

        for column, perturbation in enumerate(perturbations):
            perturbed = y_next.copy()
            perturbed[column] += perturbation
            residual_perturbed = self._state_residual(
                y_next=perturbed,
                y_previous=y_previous,
                a_previous=a_previous,
                ground_acceleration_next=ground_acceleration_next,
                dt=dt,
            )
            jacobian[:, column] = (residual_perturbed - residual_at_y) / perturbation

        return jacobian

    def _residual_norm(self, residual: np.ndarray, y_next: np.ndarray, y_previous: np.ndarray) -> float:
        displacement_scale = max(abs(float(y_next[0])), abs(float(y_previous[0])), 1.0e-3)
        velocity_scale = max(abs(float(y_next[1])), abs(float(y_previous[1])), 1.0e-3)
        return max(
            abs(float(residual[0])) / displacement_scale,
            abs(float(residual[1])) / velocity_scale,
            abs(float(residual[2])),
        )

    def _correction_norm(self, correction: np.ndarray, y_next: np.ndarray) -> float:
        displacement_scale = max(abs(float(y_next[0])), 1.0e-3)
        velocity_scale = max(abs(float(y_next[1])), 1.0e-3)
        return max(
            abs(float(correction[0])) / displacement_scale,
            abs(float(correction[1])) / velocity_scale,
            abs(float(correction[2])),
        )

    def _converged(self, residual: np.ndarray, correction: np.ndarray | None, y_next: np.ndarray, y_previous: np.ndarray, tolerance: float) -> bool:
        displacement_scale = max(abs(float(y_next[0])), abs(float(y_previous[0])), 1.0e-3)
        velocity_scale = max(abs(float(y_next[1])), abs(float(y_previous[1])), 1.0e-3)

        residual_ok = (
            abs(float(residual[0])) <= max(1.0e-10, tolerance * displacement_scale)
            and abs(float(residual[1])) <= max(1.0e-9, tolerance * velocity_scale)
            and abs(float(residual[2])) <= max(1.0e-8, tolerance)
        )
        if correction is None:
            return residual_ok

        correction_ok = self._correction_norm(correction, y_next) <= 1.0e-9
        practical_residual_ok = (
            abs(float(residual[0])) <= max(1.0e-9, 100.0 * tolerance * displacement_scale)
            and abs(float(residual[1])) <= max(1.0e-8, 100.0 * tolerance * velocity_scale)
            and abs(float(residual[2])) <= max(1.0e-7, 100.0 * tolerance)
        )
        return residual_ok or (correction_ok and practical_residual_ok)

    def _solve_state_step(
        self,
        y_previous: np.ndarray,
        a_previous: float,
        ground_acceleration_next: float,
        dt: float,
        tolerance: float,
        max_iterations: int,
        debug_context: dict[str, float | int | str],
    ) -> np.ndarray:
        """Newton iteration for y_{n+1} = [u, u_dot, z]^T."""
        y_trial = y_previous.copy()
        y_trial[0] = y_previous[0] + dt * y_previous[1] + 0.5 * dt * dt * a_previous
        y_trial[1] = y_previous[1] + dt * a_previous

        iteration_history: list[dict[str, float | int | str]] = []
        previous_norm = math.inf

        for iteration in range(max_iterations):
            residual = self._state_residual(
                y_next=y_trial,
                y_previous=y_previous,
                a_previous=a_previous,
                ground_acceleration_next=ground_acceleration_next,
                dt=dt,
            )
            residual_norm = self._residual_norm(residual, y_trial, y_previous)
            if self._converged(residual, None, y_trial, y_previous, tolerance):
                return y_trial

            jacobian = self._finite_difference_jacobian(
                y_next=y_trial,
                y_previous=y_previous,
                a_previous=a_previous,
                ground_acceleration_next=ground_acceleration_next,
                dt=dt,
                residual_at_y=residual,
            )
            try:
                jacobian_condition = float(np.linalg.cond(jacobian))
            except np.linalg.LinAlgError:
                jacobian_condition = math.inf

            try:
                correction = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError:
                correction = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]

            correction_norm = self._correction_norm(correction, y_trial)
            iteration_entry: dict[str, float | int | str] = {
                "iteration": iteration,
                "residual_norm": residual_norm,
                "r_u": float(residual[0]),
                "r_v": float(residual[1]),
                "r_z": float(residual[2]),
                "u_new": float(y_trial[0]),
                "v_new": float(y_trial[1]),
                "z_new": float(y_trial[2]),
                "du_correction": float(correction[0]),
                "dv_correction": float(correction[1]),
                "dz_correction": float(correction[2]),
                "correction_norm": correction_norm,
                "jacobian_condition": jacobian_condition,
                "line_search_trials": "not started",
            }
            iteration_history.append(iteration_entry)

            trial_summaries: list[str] = []
            step_scale = 1.0
            accepted = False
            best_trial: np.ndarray | None = None
            best_trial_norm = math.inf
            best_trial_residual: np.ndarray | None = None

            for line_search_iteration in range(12):
                trial = y_trial + step_scale * correction
                if not np.all(np.isfinite(trial)):
                    trial_summaries.append(f"ls={line_search_iteration}, scale={step_scale:.3e}, nonfinite trial")
                    step_scale *= 0.5
                    continue

                try:
                    trial_residual = self._state_residual(
                        y_next=trial,
                        y_previous=y_previous,
                        a_previous=a_previous,
                        ground_acceleration_next=ground_acceleration_next,
                        dt=dt,
                    )
                except (OverflowError, FloatingPointError, ValueError) as exc:
                    trial_summaries.append(f"ls={line_search_iteration}, scale={step_scale:.3e}, evaluation failed: {exc}")
                    step_scale *= 0.5
                    continue

                trial_norm = self._residual_norm(trial_residual, trial, y_previous)
                trial_summaries.append(f"ls={line_search_iteration}, scale={step_scale:.3e}, norm={trial_norm:.6e}")

                if trial_norm < best_trial_norm:
                    best_trial = trial
                    best_trial_norm = trial_norm
                    best_trial_residual = trial_residual

                if (
                    trial_norm <= residual_norm * (1.0 + 1.0e-8)
                    or trial_norm <= previous_norm * (1.0 + 1.0e-8)
                    or self._converged(trial_residual, step_scale * correction, trial, y_previous, tolerance)
                ):
                    y_trial = trial
                    previous_norm = min(previous_norm, trial_norm)
                    accepted = True
                    break

                step_scale *= 0.5

            iteration_entry["line_search_trials"] = "; ".join(trial_summaries[-8:])

            if not accepted:
                # Finite-difference Newton can occasionally stall at the residual floor.
                # Accept the best trial if it satisfies the practical convergence test.
                if best_trial is not None and best_trial_residual is not None and self._converged(
                    best_trial_residual,
                    best_trial - y_trial,
                    best_trial,
                    y_previous,
                    tolerance,
                ):
                    return best_trial

                raise RuntimeError(
                    self._format_newton_failure(
                        reason="State-vector Newton line search failed to find an acceptable step.",
                        debug_context=debug_context,
                        iteration_history=iteration_history,
                        y_trial=y_trial,
                    )
                )

            if self._converged(
                self._state_residual(
                    y_next=y_trial,
                    y_previous=y_previous,
                    a_previous=a_previous,
                    ground_acceleration_next=ground_acceleration_next,
                    dt=dt,
                ),
                correction,
                y_trial,
                y_previous,
                tolerance,
            ):
                return y_trial

        raise RuntimeError(
            self._format_newton_failure(
                reason="Maximum state-vector Newton iterations reached without convergence.",
                debug_context=debug_context,
                iteration_history=iteration_history,
                y_trial=y_trial,
            )
        )

    def _format_newton_failure(
        self,
        reason: str,
        debug_context: dict[str, float | int | str],
        iteration_history: list[dict[str, float | int | str]],
        y_trial: np.ndarray,
    ) -> str:
        p = self.model.parameters
        context_lines = [
            "Bouc-Wen state-vector Newmark diagnostic",
            f"Reason: {reason}",
            "",
            "Location:",
            f"  record              = {debug_context.get('record_name')}",
            f"  record step         = {debug_context.get('record_step')} / {debug_context.get('record_steps')}",
            f"  substep             = {debug_context.get('substep')} / {debug_context.get('substeps')}",
            f"  time_old            = {float(debug_context.get('time_old', math.nan)):.9e} s",
            f"  time_new            = {float(debug_context.get('time_new', math.nan)):.9e} s",
            f"  dt_record           = {float(debug_context.get('dt_record', math.nan)):.9e} s",
            f"  dt_substep          = {float(debug_context.get('dt_substep', math.nan)):.9e} s",
            f"  ag_old              = {float(debug_context.get('ag_old', math.nan)):.9e} m/s^2",
            f"  ag_new              = {float(debug_context.get('ag_new', math.nan)):.9e} m/s^2",
            "",
            "State entering the failed substep:",
            f"  u_old               = {float(debug_context.get('u_old', math.nan)):.9e} m",
            f"  v_old               = {float(debug_context.get('v_old', math.nan)):.9e} m/s",
            f"  a_old               = {float(debug_context.get('a_old', math.nan)):.9e} m/s^2",
            f"  z_old               = {float(debug_context.get('z_old', math.nan)):.9e}",
            "",
            "Current state-vector unknown:",
            f"  u_guess             = {float(y_trial[0]):.9e} m",
            f"  v_guess             = {float(y_trial[1]):.9e} m/s",
            f"  z_guess             = {float(y_trial[2]):.9e}",
            "",
            "Model parameters:",
            f"  alpha               = {p.alpha:.9e}",
            f"  beta_bw             = {p.beta_bw:.9e}",
            f"  gamma_bw            = {p.gamma_bw:.9e}",
            f"  exponent_n          = {p.exponent_n:.9e}",
            f"  yield_displacement  = {p.yield_displacement:.9e} m",
            f"  total_k1            = {p.total_k1:.9e} N/m",
            f"  total_k2            = {p.total_k2:.9e} N/m",
            f"  total_Q             = {p.total_characteristic_strength:.9e} N",
            "",
            "Last Newton iterations:",
        ]
        for item in iteration_history[-8:]:
            context_lines.append(
                "  "
                f"it={int(item.get('iteration', -1)):02d}, "
                f"norm={float(item.get('residual_norm', math.nan)):.6e}, "
                f"R_u={float(item.get('r_u', math.nan)):.6e}, "
                f"R_v={float(item.get('r_v', math.nan)):.6e}, "
                f"R_z={float(item.get('r_z', math.nan)):.6e}, "
                f"u={float(item.get('u_new', math.nan)):.6e}, "
                f"v={float(item.get('v_new', math.nan)):.6e}, "
                f"z={float(item.get('z_new', math.nan)):.6e}, "
                f"du={float(item.get('du_correction', math.nan)):.6e}, "
                f"dv={float(item.get('dv_correction', math.nan)):.6e}, "
                f"dz={float(item.get('dz_correction', math.nan)):.6e}, "
                f"corr_norm={float(item.get('correction_norm', math.nan)):.3e}, "
                f"condJ={float(item.get('jacobian_condition', math.nan)):.3e}"
            )
            line_search_trials = item.get("line_search_trials", "")
            if line_search_trials:
                context_lines.append(f"    line search: {line_search_trials}")
        context_lines.extend([
            "",
            "Suggested checks:",
            "  - The solver now uses y = [u, u_dot, z]^T and backward Euler for z.",
            "  - If this diagnostic appears, reduce max_internal_dt or relax tolerance slightly.",
            "  - This diagnostic is numerical convergence information, not a physical clipping rule.",
        ])
        return "\n".join(context_lines)

    @staticmethod
    def _format_state_failure(debug_context: dict[str, float | int | str], y_new: np.ndarray) -> str:
        return "\n".join([
            "Bouc-Wen state-vector Newmark state became nonfinite after a substep.",
            f"record      = {debug_context.get('record_name')}",
            f"step        = {debug_context.get('record_step')} / {debug_context.get('record_steps')}",
            f"substep     = {debug_context.get('substep')} / {debug_context.get('substeps')}",
            f"time_new    = {float(debug_context.get('time_new', math.nan)):.9e} s",
            f"u_new       = {float(y_new[0]):.9e}",
            f"v_new       = {float(y_new[1]):.9e}",
            f"z_new       = {float(y_new[2]):.9e}",
        ])


class BilinearConstitutiveModel:
    """One-dimensional bilinear model with kinematic hardening."""
    def __init__(self, parameters: BilinearFpsParameters) -> None:
        self.parameters = parameters

    def update(self, displacement: float, previous_state: BilinearState) -> tuple[float, BilinearState, float]:
        """Return force, updated state, and tangent at a trial displacement."""
        k = self.parameters.elastic_stiffness
        h = self.parameters.hardening_modulus
        fy = self.parameters.yield_force

        # Elastic predictor measured relative to the current plastic displacement.
        trial_force = k * (displacement - previous_state.plastic_displacement)
        shifted_trial_force = trial_force - previous_state.back_force
        trial_yield_condition = abs(shifted_trial_force) - fy

        if trial_yield_condition <= 0.0:
            return trial_force, BilinearState(previous_state.plastic_displacement, previous_state.back_force), k

        # Plastic corrector for one-dimensional kinematic hardening.
        sign_trial = 1.0 if shifted_trial_force >= 0.0 else -1.0
        plastic_multiplier = trial_yield_condition / (k + h)
        force = trial_force - k * plastic_multiplier * sign_trial
        plastic_displacement = previous_state.plastic_displacement + plastic_multiplier * sign_trial
        back_force = previous_state.back_force + h * plastic_multiplier * sign_trial
        algorithmic_tangent = (k * h) / (k + h)

        return force, BilinearState(plastic_displacement, back_force), algorithmic_tangent


class NonlinearBilinearNewmarkSolver:
    """Average-acceleration Newmark solver with Newton iterations."""
    def __init__(self, model: BilinearConstitutiveModel, beta: float = 1.0 / 4.0, gamma: float = 1.0 / 2.0) -> None:
        self.model = model
        self.beta = beta
        self.gamma = gamma

    def solve(
        self,
        record: GroundMotionRecord,
        tolerance: float = 1e-8,
        max_iterations: int = 80,
    ) -> TimeHistoryResult:
        """Solve the bilinear response with Newton iterations at each time step."""
        m = self.model.parameters.mass
        dt = record.dt
        ug = record.acceleration_mps2
        n = ug.size

        u = np.zeros(n, dtype=float)
        v = np.zeros(n, dtype=float)
        a = np.zeros(n, dtype=float)
        f = np.zeros(n, dtype=float)
        abs_a = np.zeros(n, dtype=float)

        state = BilinearState()
        f[0], state, _ = self.model.update(0.0, state)
        a[0] = (-f[0] - m * ug[0]) / m
        abs_a[0] = a[0] + ug[0]

        beta = self.beta
        gamma = self.gamma
        # Newmark constants for the average-acceleration update.
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

            # Newton iteration on displacement at the new time step.
            for _ in range(max_iterations):
                trial_state = BilinearState(state.plastic_displacement, state.back_force)
                force_i, state_i, k_alg = self.model.update(u_guess, trial_state)
                acc_i = a0 * (u_guess - u[i - 1]) - a2 * v[i - 1] - a3 * a[i - 1]
                residual = m * acc_i + force_i + m * ug[i]
                tangent = m * a0 + k_alg  # Dynamic tangent: inertia plus material tangent.
                delta_u = -residual / tangent
                u_guess += delta_u
                converged_state = state_i
                converged_force = force_i
                converged_acc = acc_i
                converged_vel = v[i - 1] + a6 * a[i - 1] + a7 * acc_i

                force_scale = max(1.0, self.model.parameters.characteristic_strength)
                if abs(delta_u) < tolerance and abs(residual) < tolerance * force_scale:
                    break

            u[i] = u_guess
            v[i] = converged_vel
            a[i] = converged_acc
            f[i] = converged_force
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


# -----------------------------------------------------------------------------
# Plotly figures and HTML report
# -----------------------------------------------------------------------------


class FigureFactory:
    """Factory methods for the figures shown in the dashboard."""

    @staticmethod
    def _max_marker_coordinates(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        idx = int(np.argmax(np.abs(y)))
        return float(x[idx]), float(y[idx])

    @staticmethod
    def _cyclic_components(result: CyclicTestResult, parameters: BoucWenFpsParameters) -> dict[str, np.ndarray]:
        """Return per-bearing cyclic quantities used by the cyclic hovers."""
        time = np.asarray(result.time, dtype=float)
        displacement = np.asarray(result.displacement, dtype=float)
        velocity = np.asarray(result.velocity, dtype=float)
        z = np.asarray(result.hysteretic_parameter, dtype=float)
        period = 1.0 / result.cycle_frequency_hz
        cycle_number = np.minimum(np.floor(time / period).astype(int) + 1, result.n_cycles)

        linear_force = parameters.alpha * parameters.bearing_k1 * displacement
        hysteretic_force = (
            (1.0 - parameters.alpha)
            * parameters.bearing_k1
            * parameters.yield_displacement
            * z
        )
        ideal_linear_force = parameters.bearing_k2 * displacement
        ideal_friction_force = parameters.bearing_characteristic_strength * sign_with_memory(velocity)

        return {
            "time": time,
            "cycle": cycle_number.astype(float),
            "displacement_m": displacement,
            "displacement_mm": displacement * 1.0e3,
            "velocity": velocity,
            "z": z,
            "linear_kN": linear_force / 1.0e3,
            "hysteretic_kN": hysteretic_force / 1.0e3,
            "bouc_wen_kN": np.asarray(result.bouc_wen_force, dtype=float) / 1.0e3,
            "linear_over_weight": linear_force / result.bearing_weight,
            "hysteretic_over_weight": hysteretic_force / result.bearing_weight,
            "bouc_wen_over_weight": np.asarray(result.bouc_wen_force, dtype=float) / result.bearing_weight,
            "ideal_linear_kN": ideal_linear_force / 1.0e3,
            "ideal_friction_kN": ideal_friction_force / 1.0e3,
            "ideal_kN": np.asarray(result.ideal_force, dtype=float) / 1.0e3,
            "ideal_linear_over_weight": ideal_linear_force / result.bearing_weight,
            "ideal_friction_over_weight": ideal_friction_force / result.bearing_weight,
            "ideal_over_weight": np.asarray(result.ideal_force, dtype=float) / result.bearing_weight,
        }

    @staticmethod
    def _cyclic_customdata(components: dict[str, np.ndarray]) -> np.ndarray:
        """Shared cyclic customdata table for detailed hover labels."""
        return np.column_stack((
            components["time"],
            components["cycle"],
            components["displacement_m"],
            components["velocity"],
            components["z"],
            components["linear_kN"],
            components["hysteretic_kN"],
            components["bouc_wen_kN"],
            components["linear_over_weight"],
            components["hysteretic_over_weight"],
            components["bouc_wen_over_weight"],
            components["ideal_linear_kN"],
            components["ideal_friction_kN"],
            components["ideal_kN"],
            components["ideal_over_weight"],
            components["displacement_mm"],
            components["ideal_linear_over_weight"],
            components["ideal_friction_over_weight"],
        ))

    @staticmethod
    def _bouc_wen_cyclic_hover(parameters: BoucWenFpsParameters) -> str:
        """Hover text for normalized Bouc--Wen cyclic plots.

        The plotted ordinate is normalized force, so the force-decomposition
        terms are also reported normalized by the one-bearing weight W_b.
        """
        return (
            "Cycle: %{customdata[1]:.0f}<br>"
            "t: %{customdata[0]:.4f} s<br>"
            "u: %{customdata[2]:.6f} m = %{x:.3f} mm<br>"
            "u̇: %{customdata[3]:.6f} m/s<br>"
            "z: %{customdata[4]:.6f}<br>"
            "αK1u/Wb: %{customdata[8]:.6f}<br>"
            "(1−α)K1uyz/Wb: %{customdata[9]:.6f}<br>"
            "F(t)/Wb: %{customdata[10]:.6f}<extra></extra>"
        )

    @staticmethod
    def _ideal_cyclic_hover(parameters: BoucWenFpsParameters) -> str:
        """Hover text for normalized ideal-FPS cyclic plots."""
        return (
            "Cycle: %{customdata[1]:.0f}<br>"
            "t: %{customdata[0]:.4f} s<br>"
            "u: %{customdata[2]:.6f} m = %{x:.3f} mm<br>"
            "u̇: %{customdata[3]:.6f} m/s<br>"
            "K2u/Wb: %{customdata[16]:.6f}<br>"
            "Q sign(u̇)/Wb: %{customdata[17]:.6f}<br>"
            "Fideal(t)/Wb: %{customdata[14]:.6f}<extra></extra>"
        )

    @staticmethod
    def ideal_cyclic_hysteresis(result: CyclicTestResult, parameters: BoucWenFpsParameters) -> go.Figure:
        """Plot the complete ideal FPS cyclic path from the initial state."""
        components = FigureFactory._cyclic_components(result, parameters)
        customdata = FigureFactory._cyclic_customdata(components)
        period = 1.0 / result.cycle_frequency_hz
        initial_mask = result.time <= 0.25 * period
        amplitude_mm = float(np.max(np.abs(components["displacement_mm"])))
        force_scale = float(np.max(np.abs(components["ideal_over_weight"])))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=components["displacement_mm"],
                y=components["ideal_over_weight"],
                mode="lines",
                line=dict(color=MATLAB_COLORS["dark_blue"], width=3.0),
                name="Ideal FPS complete path from t = 0",
                customdata=customdata,
                hovertemplate="Model: ideal FPS<br>" + FigureFactory._ideal_cyclic_hover(parameters),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=components["displacement_mm"][initial_mask],
                y=components["ideal_over_weight"][initial_mask],
                mode="lines",
                line=dict(color=MATLAB_COLORS["black"], width=2.6, dash="dot"),
                name="Initial loading branch",
                customdata=customdata[initial_mask],
                hovertemplate="Initial loading branch<br>" + FigureFactory._ideal_cyclic_hover(parameters),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[0.0],
                mode="markers+text",
                marker=dict(size=11, color=MATLAB_COLORS["black"], symbol="circle"),
                text=["start: z = 0"],
                textposition="top right",
                name="Initial state",
                hovertemplate="Initial state<br>t: 0.0000 s<br>u: 0.000000 m = 0.000 mm<br>u_dot: "
                              f"{result.velocity[0]:.6f} m/s<br>z: 0.000000<br>F: 0.000000<extra></extra>",
            )
        )
        fig.update_layout(
            template="plotly_white",
            height=560,
            title=dict(text="Part (a) — Ideal FPS cyclic path, including initial loading", x=0.5, xanchor="center", font=dict(size=22)),
            xaxis=dict(title="Displacement u [mm]", range=[-1.08 * amplitude_mm, 1.08 * amplitude_mm], title_font=dict(size=16), tickfont=dict(size=13), zeroline=True),
            yaxis=dict(title="Normalized force F/Wb [-]", range=[-1.18 * force_scale, 1.18 * force_scale], title_font=dict(size=16), tickfont=dict(size=13), zeroline=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            font=dict(size=14),
        )
        return fig

    @staticmethod
    def cyclic_hysteresis(result: CyclicTestResult, parameters: BoucWenFpsParameters) -> go.Figure:
        """Plot complete ideal and Bouc--Wen cyclic paths from the initial state."""
        components = FigureFactory._cyclic_components(result, parameters)
        customdata = FigureFactory._cyclic_customdata(components)
        period = 1.0 / result.cycle_frequency_hz
        initial_mask = result.time <= 0.25 * period
        amplitude_mm = float(np.max(np.abs(components["displacement_mm"])))
        force_scale = max(float(np.max(np.abs(components["ideal_over_weight"]))), float(np.max(np.abs(components["bouc_wen_over_weight"]))))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=components["displacement_mm"],
                y=components["ideal_over_weight"],
                mode="lines",
                line=dict(color=MATLAB_COLORS["dark_blue"], width=3.0, dash="dash"),
                name="Ideal FPS complete path from t = 0",
                customdata=customdata,
                hovertemplate="Model: ideal FPS<br>" + FigureFactory._ideal_cyclic_hover(parameters),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=components["displacement_mm"],
                y=components["bouc_wen_over_weight"],
                mode="lines",
                line=dict(color=MATLAB_COLORS["crimson"], width=2.8),
                name="Bouc-Wen complete path from t = 0",
                customdata=customdata,
                hovertemplate="Model: Bouc-Wen<br>" + FigureFactory._bouc_wen_cyclic_hover(parameters),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=components["displacement_mm"][initial_mask],
                y=components["bouc_wen_over_weight"][initial_mask],
                mode="lines",
                line=dict(color=MATLAB_COLORS["black"], width=2.6, dash="dot"),
                name="Initial loading branch",
                customdata=customdata[initial_mask],
                hovertemplate="Initial loading branch<br>" + FigureFactory._bouc_wen_cyclic_hover(parameters),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[0.0],
                mode="markers+text",
                marker=dict(size=11, color=MATLAB_COLORS["black"], symbol="circle"),
                text=["start: z = 0"],
                textposition="top right",
                name="Initial state",
                hovertemplate="Initial state<br>t: 0.0000 s<br>u: 0.000000 m = 0.000 mm<br>u_dot: "
                              f"{result.velocity[0]:.6f} m/s<br>z: 0.000000<br>F: 0.000000<extra></extra>",
            )
        )
        fig.update_layout(
            template="plotly_white",
            height=640,
            title=dict(text="Part (b) — Bouc-Wen approximation, complete path from the unloaded state", x=0.5, xanchor="center", font=dict(size=22)),
            xaxis=dict(title="Displacement u [mm]", range=[-1.08 * amplitude_mm, 1.08 * amplitude_mm], title_font=dict(size=16), tickfont=dict(size=13), zeroline=True),
            yaxis=dict(title="Normalized force F/Wb [-]", range=[-1.18 * force_scale, 1.18 * force_scale], title_font=dict(size=16), tickfont=dict(size=13), zeroline=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            font=dict(size=14),
        )
        return fig

    @staticmethod
    def cyclic_initial_loading_zoom(result: CyclicTestResult, parameters: BoucWenFpsParameters) -> go.Figure:
        """Zoom into the first milliseconds of the cyclic test so the loading branch is visible."""
        components = FigureFactory._cyclic_components(result, parameters)
        customdata = FigureFactory._cyclic_customdata(components)

        # The FPS yield displacement is 0.03 mm, so z saturates within a very small
        # fraction of the first cycle. A zoomed view is needed; otherwise the branch
        # is compressed at the origin on a +/-400 mm displacement plot.
        first_positive_saturation = np.where(components["z"] >= 0.98)[0]
        if first_positive_saturation.size > 0:
            i_end = int(min(max(first_positive_saturation[0] * 4, 40), result.time.size - 1))
        else:
            period = 1.0 / result.cycle_frequency_hz
            i_end = int(np.searchsorted(result.time, 0.01 * period))
        zoom_mask = np.arange(result.time.size) <= i_end

        amplitude_mm = max(float(np.max(components["displacement_mm"][zoom_mask])), 1.0e-6)
        force_scale = max(float(np.max(np.abs(components["bouc_wen_over_weight"][zoom_mask]))), 1.0e-6)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=components["displacement_mm"][zoom_mask],
                y=components["bouc_wen_over_weight"][zoom_mask],
                mode="lines+markers",
                line=dict(color=MATLAB_COLORS["crimson"], width=3.0),
                marker=dict(size=5),
                name="Bouc-Wen initial loading branch",
                customdata=customdata[zoom_mask],
                hovertemplate="Initial loading branch<br>" + FigureFactory._bouc_wen_cyclic_hover(parameters),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[0.0],
                mode="markers+text",
                marker=dict(size=11, color=MATLAB_COLORS["black"], symbol="circle"),
                text=["start: z = 0"],
                textposition="top right",
                name="Initial state",
                hovertemplate="Initial state<br>t: 0.000000 s<br>u: 0.000000 m = 0.000 mm<br>u_dot: "
                              f"{result.velocity[0]:.6f} m/s<br>z: 0.000000<br>F/Wb: 0.000000<extra></extra>",
            )
        )
        fig.update_layout(
            template="plotly_white",
            height=500,
            title=dict(text="Part (b) — Zoom of the initial Bouc-Wen loading branch", x=0.5, xanchor="center", font=dict(size=22)),
            xaxis=dict(title="Displacement u [mm]", range=[-0.04 * amplitude_mm, 1.12 * amplitude_mm], title_font=dict(size=16), tickfont=dict(size=13), zeroline=True),
            yaxis=dict(title="Normalized force F/Wb [-]", range=[-0.04 * force_scale, 1.18 * force_scale], title_font=dict(size=16), tickfont=dict(size=13), zeroline=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            font=dict(size=14),
        )
        return fig

    @staticmethod
    def cyclic_calibration_iterations(calibration_cases: list[CalibrationCaseResult]) -> go.Figure:
        """Plot every calibration trial over all cycles, not only the last loop."""
        if not calibration_cases:
            raise ValueError("At least one calibration case is required.")

        fig = go.Figure()
        colors = [MATLAB_COLORS["gray"], MATLAB_COLORS["gold"], MATLAB_COLORS["crimson"], MATLAB_COLORS["purple"]]
        dashes = ["dot", "dashdot", "solid", "longdash"]

        force_scale = 0.0
        amplitude_mm = 0.0
        for i, case in enumerate(calibration_cases):
            result = case.cyclic_result
            parameters = case.parameters
            components = FigureFactory._cyclic_components(result, parameters)
            customdata = FigureFactory._cyclic_customdata(components)
            force_scale = max(force_scale, float(np.max(np.abs(components["bouc_wen_over_weight"]))))
            amplitude_mm = max(amplitude_mm, float(np.max(np.abs(components["displacement_mm"]))))
            fig.add_trace(
                go.Scatter(
                    x=components["displacement_mm"],
                    y=components["bouc_wen_over_weight"],
                    mode="lines",
                    line=dict(color=colors[i % len(colors)], width=2.2 if i < len(calibration_cases) - 1 else 2.8, dash=dashes[i % len(dashes)]),
                    name=case.label,
                    customdata=customdata,
                    hovertemplate=(f"Case: {case.label}<br>" + FigureFactory._bouc_wen_cyclic_hover(parameters)),
                )
            )

        reference = calibration_cases[-1]
        components = FigureFactory._cyclic_components(reference.cyclic_result, reference.parameters)
        customdata = FigureFactory._cyclic_customdata(components)
        force_scale = max(force_scale, float(np.max(np.abs(components["ideal_over_weight"]))))
        amplitude_mm = max(amplitude_mm, float(np.max(np.abs(components["displacement_mm"]))))
        fig.add_trace(
            go.Scatter(
                x=components["displacement_mm"],
                y=components["ideal_over_weight"],
                mode="lines",
                line=dict(color=MATLAB_COLORS["dark_blue"], width=3.0, dash="dash"),
                name="Ideal FPS complete path from t = 0",
                customdata=customdata,
                hovertemplate="Case: ideal FPS<br>" + FigureFactory._ideal_cyclic_hover(reference.parameters),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[0.0],
                mode="markers+text",
                marker=dict(size=10, color=MATLAB_COLORS["black"]),
                text=["start"],
                textposition="top right",
                name="Initial state",
                hovertemplate="Initial state<br>t: 0.0000 s<br>u: 0.000 mm<br>F/Wb: 0.000000<br>z: 0.000000<extra></extra>",
            )
        )
        fig.update_layout(
            template="plotly_white",
            height=620,
            title=dict(
                text="Part (b) — Calibration iterations, complete paths from t = 0",
                x=0.5,
                xanchor="center",
                y=0.995,
                yanchor="top",
                font=dict(size=22),
            ),
            xaxis=dict(title="Displacement u [mm]", range=[-1.08 * amplitude_mm, 1.08 * amplitude_mm], title_font=dict(size=16), tickfont=dict(size=13), zeroline=True),
            yaxis=dict(title="Normalized force F/Wb [-]", range=[-1.18 * force_scale, 1.18 * force_scale], title_font=dict(size=16), tickfont=dict(size=13), zeroline=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(t=130),
            font=dict(size=14),
        )
        return fig

    @staticmethod
    def z_history(result: CyclicTestResult, parameters: BoucWenFpsParameters) -> go.Figure:
        """Plot z(t) with a full-history panel and an initial-transition zoom."""
        components = FigureFactory._cyclic_components(result, parameters)
        customdata = FigureFactory._cyclic_customdata(components)

        first_positive_saturation = np.where(components["z"] >= 0.98)[0]
        if first_positive_saturation.size > 0:
            zoom_end_index = int(min(max(first_positive_saturation[0] * 5, 60), result.time.size - 1))
        else:
            period = 1.0 / result.cycle_frequency_hz
            zoom_end_index = int(np.searchsorted(result.time, 0.01 * period))
        zoom_mask = np.arange(result.time.size) <= zoom_end_index

        hover = (
            "Cycle: %{customdata[1]:.0f}<br>"
            "t: %{x:.6f} s<br>"
            "u: %{customdata[2]:.8f} m = %{customdata[15]:.5f} mm<br>"
            "u̇: %{customdata[3]:.6f} m/s<br>"
            "z: %{y:.8f}<br>"
            "αK1u/Wb: %{customdata[8]:.8f}<br>"
            "(1−α)K1uyz/Wb: %{customdata[9]:.8f}<br>"
            "F(t)/Wb: %{customdata[10]:.8f}<extra></extra>"
        )

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.16,
            subplot_titles=(
                "Complete z(t) history from the initial condition",
                "Zoom near t = 0 showing z(0)=0 and the rapid positive transition",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=result.time,
                y=result.hysteretic_parameter,
                mode="lines",
                line=dict(color=MATLAB_COLORS["dark_blue"], width=2.2),
                name="z(t), complete history",
                customdata=customdata,
                hovertemplate=hover,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=result.time[zoom_mask],
                y=result.hysteretic_parameter[zoom_mask],
                mode="lines+markers",
                line=dict(color=MATLAB_COLORS["crimson"], width=2.4),
                marker=dict(size=5),
                name="z(t), initial zoom",
                customdata=customdata[zoom_mask],
                hovertemplate=hover,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[0.0],
                mode="markers+text",
                marker=dict(size=10, color=MATLAB_COLORS["black"]),
                text=["z(0)=0"],
                textposition="top right",
                name="Initial state",
                hovertemplate="Initial state<br>t: 0.000000 s<br>z: 0.000000<br>u: 0.000000 mm<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[0.0],
                mode="markers+text",
                marker=dict(size=10, color=MATLAB_COLORS["black"]),
                text=["z(0)=0"],
                textposition="top right",
                name="Initial state, zoom",
                showlegend=False,
                hovertemplate="Initial state<br>t: 0.000000 s<br>z: 0.000000<br>u: 0.000000 mm<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.update_layout(
            template="plotly_white",
            height=720,
            title=dict(text="Part (b) — Bouc-Wen hysteretic parameter from the initial state", x=0.5, xanchor="center", font=dict(size=20)),
            font=dict(size=14),
            hovermode="closest",
        )
        fig.update_xaxes(title_text="Time [s]", title_font=dict(size=16), tickfont=dict(size=13), row=1, col=1)
        fig.update_xaxes(title_text="Time [s]", title_font=dict(size=16), tickfont=dict(size=13), row=2, col=1)
        fig.update_yaxes(title_text="z(t) [-]", range=[-1.05, 1.05], title_font=dict(size=16), tickfont=dict(size=13), zeroline=True, row=1, col=1)
        fig.update_yaxes(title_text="z(t) [-]", range=[-0.05, 1.05], title_font=dict(size=16), tickfont=dict(size=13), zeroline=True, row=2, col=1)
        fig.update_annotations(font=dict(size=15))
        return fig

    @staticmethod
    def time_history_5panel(
        title: str,
        result: TimeHistoryResult,
        total_weight: float,
        parameters: BoucWenFpsParameters | None = None,
    ) -> go.Figure:
        """Create the five-panel response history figure with complete hover data."""
        time = result.time
        ground_g = result.ground_acceleration / G_SI
        abs_acc_g = result.absolute_acceleration / G_SI
        rel_acc_g = result.relative_acceleration / G_SI
        force_mn = result.restoring_force / 1.0e6
        force_over_weight = result.restoring_force / total_weight
        disp_mm = result.displacement * 1.0e3
        vel_mps = result.velocity

        has_components = parameters is not None and result.hysteretic_parameter is not None
        if has_components:
            z_values = np.asarray(result.hysteretic_parameter, dtype=float)
            linear_mn = parameters.alpha * parameters.total_k1 * result.displacement / 1.0e6
            hysteretic_mn = (
                (1.0 - parameters.alpha)
                * parameters.total_k1
                * parameters.yield_displacement
                * z_values
            ) / 1.0e6
            linear_over_weight = linear_mn * 1.0e6 / total_weight
            hysteretic_over_weight = hysteretic_mn * 1.0e6 / total_weight
            customdata = np.column_stack((disp_mm, vel_mps, force_over_weight, force_mn, rel_acc_g, abs_acc_g, ground_g, z_values, linear_mn, hysteretic_mn, linear_over_weight, hysteretic_over_weight))
            component_hover = (
                "z: %{customdata[7]:.6f}<br>"
                "αK1u/W: %{customdata[10]:.6f}<br>"
                "(1−α)K1uyz/W: %{customdata[11]:.6f}<br>"
                "F(t)/W: %{customdata[2]:.6f}<br>"
            )
        else:
            customdata = np.column_stack((disp_mm, vel_mps, force_over_weight, force_mn, rel_acc_g, abs_acc_g, ground_g))
            component_hover = ""

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
            (vel_mps, MATLAB_COLORS["black"], "u_dot_max", "m/s", "Velocity u̇", ".4f"),
            (force_over_weight, MATLAB_COLORS["crimson"], "F/W_max", "-", "Normalized restoring force F/W", ".5f"),
            (abs_acc_g, MATLAB_COLORS["dark_green"], "a_abs_max", "g", "Absolute acceleration a_abs", ".4f"),
            (ground_g, MATLAB_COLORS["gray"], "a_g_max", "g", "Ground acceleration a_g", ".4f"),
        ]

        for row_idx, (y_values, color, peak_label, unit_label, hover_name, hover_fmt) in enumerate(series, start=1):
            y_max = float(np.max(np.abs(y_values)))
            y_padding = 0.18 * max(y_max, 1.0e-9)
            y_min = float(np.min(y_values)) - y_padding
            y_top = float(np.max(y_values)) + y_padding

            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=y_values,
                    mode="lines",
                    line=dict(color=color, width=2.2),
                    showlegend=False,
                    customdata=customdata,
                    hovertemplate=(
                        f"Time: %{{x:.3f}} s<br>"
                        f"{hover_name}: %{{y{hover_fmt}}}"
                        + (f" {unit_label}" if unit_label != "-" else "")
                        + "<br>"
                        "u: %{customdata[0]:.3f} mm<br>"
                        "u̇: %{customdata[1]:.4f} m/s<br>"
                        "F(t)/W: %{customdata[2]:.6f}<br>"
                        "a_rel: %{customdata[4]:.4f} g<br>"
                        "a_abs: %{customdata[5]:.4f} g<br>"
                        "a_g: %{customdata[6]:.4f} g<br>"
                        + component_hover
                        + "<extra></extra>"
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
            axis_suffix = "" if row_idx == 1 else str(row_idx)
            fig.add_annotation(
                x=x_peak,
                y=y_peak,
                xref=f"x{axis_suffix}",
                yref=f"y{axis_suffix}",
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
    def hysteresis(
        title: str,
        result: TimeHistoryResult,
        total_weight: float,
        parameters: BoucWenFpsParameters | None = None,
    ) -> go.Figure:
        """Create a force-displacement plot with complete response hover data."""
        disp_mm = result.displacement * 1.0e3
        force_mn = result.restoring_force / 1.0e6
        force_over_weight = result.restoring_force / total_weight
        time = result.time
        velocity = result.velocity
        rel_acc_g = result.relative_acceleration / G_SI
        abs_acc_g = result.absolute_acceleration / G_SI
        ground_g = result.ground_acceleration / G_SI

        has_components = parameters is not None and result.hysteretic_parameter is not None
        if has_components:
            z_values = np.asarray(result.hysteretic_parameter, dtype=float)
            linear_mn = parameters.alpha * parameters.total_k1 * result.displacement / 1.0e6
            hysteretic_mn = (
                (1.0 - parameters.alpha)
                * parameters.total_k1
                * parameters.yield_displacement
                * z_values
            ) / 1.0e6
            linear_over_weight = linear_mn * 1.0e6 / total_weight
            hysteretic_over_weight = hysteretic_mn * 1.0e6 / total_weight
            customdata = np.column_stack((time, velocity, rel_acc_g, abs_acc_g, ground_g, force_mn, z_values, linear_mn, hysteretic_mn, linear_over_weight, hysteretic_over_weight))
            hovertemplate = (
                "Displacement u: %{x:.3f} mm<br>"
                "Time: %{customdata[0]:.3f} s<br>"
                "u̇: %{customdata[1]:.4f} m/s<br>"
                "Relative acceleration a_rel: %{customdata[2]:.4f} g<br>"
                "Absolute acceleration a_abs: %{customdata[3]:.4f} g<br>"
                "Ground acceleration a_g: %{customdata[4]:.4f} g<br>"
                "z: %{customdata[6]:.6f}<br>"
                "αK1u/W: %{customdata[9]:.6f}<br>"
                "(1−α)K1uyz/W: %{customdata[10]:.6f}<br>"
                "F(t)/W: %{y:.6f}<extra></extra>"
            )
        else:
            customdata = np.column_stack((time, velocity, rel_acc_g, abs_acc_g, ground_g, force_mn))
            hovertemplate = (
                "Displacement u: %{x:.3f} mm<br>"
                "Normalized restoring force F/W: %{y:.6f}<br>"
                "Time: %{customdata[0]:.3f} s<br>"
                "u̇: %{customdata[1]:.4f} m/s<br>"
                "Relative acceleration a_rel: %{customdata[2]:.4f} g<br>"
                "Absolute acceleration a_abs: %{customdata[3]:.4f} g<br>"
                "Ground acceleration a_g: %{customdata[4]:.4f} g<extra></extra>"
            )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=disp_mm,
                y=force_over_weight,
                mode="lines",
                line=dict(color=MATLAB_COLORS["dark_blue"], width=2.4),
                name="F-u",
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        )
        fig.update_layout(
            template="plotly_white",
            height=500,
            title=dict(text=title, x=0.5, xanchor="center", font=dict(size=22)),
            xaxis=dict(title="Displacement u [mm]", title_font=dict(size=16), tickfont=dict(size=13), zeroline=True),
            yaxis=dict(title="Normalized force F/W [-]", title_font=dict(size=16), tickfont=dict(size=13), zeroline=True),
            font=dict(size=14),
        )
        return fig

    @staticmethod
    def hysteresis_comparison(
        title: str,
        bouc_wen_result: TimeHistoryResult,
        bilinear_result: TimeHistoryResult,
        total_weight: float,
        parameters: BoucWenFpsParameters,
    ) -> go.Figure:
        """Plot Bouc--Wen and bilinear force-displacement loops with detailed hover data."""
        fig = go.Figure()
        z_values = np.asarray(bouc_wen_result.hysteretic_parameter, dtype=float)
        bw_linear_mn = parameters.alpha * parameters.total_k1 * bouc_wen_result.displacement / 1.0e6
        bw_hysteretic_mn = (
            (1.0 - parameters.alpha)
            * parameters.total_k1
            * parameters.yield_displacement
            * z_values
        ) / 1.0e6
        bw_linear_over_weight = bw_linear_mn * 1.0e6 / total_weight
        bw_hysteretic_over_weight = bw_hysteretic_mn * 1.0e6 / total_weight
        bw_customdata = np.column_stack((
            bouc_wen_result.time,
            bouc_wen_result.velocity,
            bouc_wen_result.relative_acceleration / G_SI,
            bouc_wen_result.absolute_acceleration / G_SI,
            bouc_wen_result.ground_acceleration / G_SI,
            bouc_wen_result.restoring_force / 1.0e6,
            z_values,
            bw_linear_mn,
            bw_hysteretic_mn,
            bw_linear_over_weight,
            bw_hysteretic_over_weight,
        ))
        fig.add_trace(
            go.Scatter(
                x=bouc_wen_result.displacement * 1.0e3,
                y=bouc_wen_result.restoring_force / total_weight,
                mode="lines",
                line=dict(color=MATLAB_COLORS["crimson"], width=2.4),
                name="Bouc-Wen",
                customdata=bw_customdata,
                hovertemplate=(
                    "Model: Bouc-Wen<br>"
                    "u: %{x:.3f} mm<br>"
                    "t: %{customdata[0]:.3f} s<br>"
                    "u̇: %{customdata[1]:.4f} m/s<br>"
                    "a_rel: %{customdata[2]:.4f} g<br>"
                    "a_abs: %{customdata[3]:.4f} g<br>"
                    "a_g: %{customdata[4]:.4f} g<br>"
                    "z: %{customdata[6]:.6f}<br>"
                    "αK1u/W: %{customdata[9]:.6f}<br>"
                    "(1−α)K1uyz/W: %{customdata[10]:.6f}<br>"
                    "F(t)/W: %{y:.6f}<extra></extra>"
                ),
            )
        )
        bilinear_customdata = np.column_stack((
            bilinear_result.time,
            bilinear_result.velocity,
            bilinear_result.relative_acceleration / G_SI,
            bilinear_result.absolute_acceleration / G_SI,
            bilinear_result.ground_acceleration / G_SI,
            bilinear_result.restoring_force / 1.0e6,
        ))
        fig.add_trace(
            go.Scatter(
                x=bilinear_result.displacement * 1.0e3,
                y=bilinear_result.restoring_force / total_weight,
                mode="lines",
                line=dict(color=MATLAB_COLORS["black"], width=2.4, dash="dash"),
                name="Plasticity",
                customdata=bilinear_customdata,
                hovertemplate=(
                    "Model: plasticity<br>"
                    "u: %{x:.3f} mm<br>"
                    "F(t)/W: %{y:.6f}<br>"
                    "t: %{customdata[0]:.3f} s<br>"
                    "u̇: %{customdata[1]:.4f} m/s<br>"
                    "a_rel: %{customdata[2]:.4f} g<br>"
                    "a_abs: %{customdata[3]:.4f} g<br>"
                    "a_g: %{customdata[4]:.4f} g<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            template="plotly_white",
            height=540,
            title=dict(text=title, x=0.5, xanchor="center", font=dict(size=22)),
            xaxis=dict(title="Displacement u [mm]", title_font=dict(size=16), tickfont=dict(size=13), zeroline=True),
            yaxis=dict(title="Normalized force F/W [-]", title_font=dict(size=16), tickfont=dict(size=13), zeroline=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            font=dict(size=14),
        )
        return fig


class HtmlReportBuilder:
    """Combines figures, tables, and explanatory text into one HTML page."""
    def __init__(self, bw_params: BoucWenFpsParameters, bilinear_params: BilinearFpsParameters, data: FpsProblemData) -> None:
        self.bw_params = bw_params
        self.bilinear_params = bilinear_params
        self.data = data

    @staticmethod
    def fig_to_div(fig: go.Figure, include_js: bool = False) -> str:
        """Convert a Plotly figure to an embeddable HTML fragment."""
        return to_html(fig, include_plotlyjs=include_js, full_html=False, config=dict(displayModeBar=True, responsive=True))

    def build(
        self,
        cyclic_result: CyclicTestResult,
        calibration_cases: list[CalibrationCaseResult],
        bw_results: dict[str, TimeHistoryResult],
        bilinear_results: dict[str, TimeHistoryResult],
    ) -> str:
        """Assemble all figures, tables, and static HTML into the dashboard."""
        fig_ideal = FigureFactory.ideal_cyclic_hysteresis(cyclic_result, self.bw_params)
        fig_cyclic = FigureFactory.cyclic_hysteresis(cyclic_result, self.bw_params)
        fig_initial_zoom = FigureFactory.cyclic_initial_loading_zoom(cyclic_result, self.bw_params)
        fig_calibration_iterations = FigureFactory.cyclic_calibration_iterations(calibration_cases)
        fig_z = FigureFactory.z_history(cyclic_result, self.bw_params)

        fig_kobe_bw = FigureFactory.time_history_5panel("Part (c) — Kobe motion: Bouc-Wen FPS response", bw_results["Kobe"], self.data.weight, self.bw_params)
        fig_kobe_bw_h = FigureFactory.hysteresis("Part (c) — Kobe motion: normalized hysteresis F/W-u", bw_results["Kobe"], self.data.weight, self.bw_params)
        fig_sylmar_bw = FigureFactory.time_history_5panel("Part (c) — Sylmar motion: Bouc-Wen FPS response", bw_results["Sylmar"], self.data.weight, self.bw_params)
        fig_sylmar_bw_h = FigureFactory.hysteresis("Part (c) — Sylmar motion: normalized hysteresis F/W-u", bw_results["Sylmar"], self.data.weight, self.bw_params)

        fig_kobe_compare = FigureFactory.hysteresis_comparison(
            "Part (d) — Kobe motion: Bouc-Wen versus plasticity",
            bw_results["Kobe"],
            bilinear_results["Kobe"],
            self.data.weight,
            self.bw_params,
        )
        fig_sylmar_compare = FigureFactory.hysteresis_comparison(
            "Part (d) — Sylmar motion: Bouc-Wen versus plasticity",
            bw_results["Sylmar"],
            bilinear_results["Sylmar"],
            self.data.weight,
            self.bw_params,
        )

        sections = [
            self.fig_to_div(fig_ideal, include_js=False),
            self.fig_to_div(fig_cyclic, include_js=False),
            self.fig_to_div(fig_initial_zoom, include_js=False),
            self.fig_to_div(fig_calibration_iterations, include_js=False),
            self.fig_to_div(fig_z, include_js=False),
            self.fig_to_div(fig_kobe_bw, include_js=False),
            self.fig_to_div(fig_kobe_bw_h, include_js=False),
            self.fig_to_div(fig_sylmar_bw, include_js=False),
            self.fig_to_div(fig_sylmar_bw_h, include_js=False),
            self.fig_to_div(fig_kobe_compare, include_js=False),
            self.fig_to_div(fig_sylmar_compare, include_js=False),
        ]

        peak_table = self._build_peak_table(bw_results, bilinear_results)
        parameter_table = self._build_parameter_table()
        calibration_table = self._build_calibration_table(calibration_cases)
        difference_table = self._build_difference_table(bw_results, bilinear_results)

        return dedent(
            rf"""
            <!DOCTYPE HTML>
            <html lang="en">
            <head>
              <meta charset="utf-8" />
              <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
              <title>CE223 – FPS Bouc-Wen Dashboard</title>
              <link rel="stylesheet" href="../../assets/css/main.css" />
              <noscript><link rel="stylesheet" href="../../assets/css/noscript.css" /></noscript>
              <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
              <script>
                window.MathJax = {{
                  tex: {{
                    inlineMath: [['$', '$'], ['\\(', '\\)']],
                    displayMath: [['$$', '$$'], ['\\[', '\\]']],
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
              <style>
                .ce223-dashboard .container {{
                    max-width: 72em;
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
                .inner-report .js-plotly-plot .modebar {{
                    top: 56px !important;
                }}
                :root {{
                    --berkeley-blue: #003262;
                    --california-gold: #FDB515;
                    --text: #1f2937;
                    --muted: #4b5563;
                    --line: #d1d5db;
                    --panel: #ffffff;
                    --bg: #f7f9fc;
                }}
                * {{ box-sizing: border-box; }}
                body {{
                    margin: 0;
                    font-family: Arial, Helvetica, sans-serif;
                    background: var(--bg);
                    color: var(--text);
                    line-height: 1.6;
                }}
                .page {{
                    max-width: 100%;
                    margin: 0;
                    padding: 0 0 2.5rem 0;
                }}
                .hero {{
                    background: linear-gradient(135deg, #ffffff 0%, #f4f7fb 100%);
                    border: 1px solid var(--line);
                    border-radius: 14px;
                    padding: 26px 28px;
                    margin-bottom: 22px;
                }}
                .hero h1 {{
                    margin: 0 0 10px 0;
                    font-size: 2rem;
                    color: var(--berkeley-blue);
                }}
                .hero p {{
                    margin: 0.35rem 0;
                    color: var(--muted);
                }}
                .card {{
                    background: var(--panel);
                    border: 1px solid var(--line);
                    border-radius: 12px;
                    padding: 20px 22px;
                    margin-bottom: 20px;
                }}
                .card h2, .card h3 {{
                    margin-top: 0;
                    color: var(--berkeley-blue);
                }}
                .card h2 {{
                    border-left: 5px solid var(--california-gold);
                    padding-left: 12px;
                    font-size: 1.35rem;
                }}
                .equation {{
                    overflow-x: auto;
                    padding: 4px 0;
                }}
                .summary-table-wrap {{ overflow-x: auto; }}
                .summary-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 0.85rem;
                    font-size: 0.97rem;
                }}
                .summary-table th, .summary-table td {{
                    border: 1px solid var(--line);
                    padding: 10px 12px;
                    text-align: left;
                }}
                .summary-table th {{
                    background: #eef3f8;
                    font-weight: 700;
                }}
                .plot-embed {{
                    margin-top: 14px;
                }}
                ul {{
                    margin-top: 0.35rem;
                    padding-left: 1.2rem;
                }}
                code {{
                    background: #eff3f8;
                    padding: 1px 5px;
                    border-radius: 4px;
                }}
              </style>
            </head>
            <body class="is-preload">
              <div id="page-wrapper">
                <header id="header"></header>

                <section id="main" class="wrapper style1 ce223-dashboard">
                  <div class="container inner-report">
                    <header class="major">
                      <h2>CE223 – Friction Pendulum System with Bouc-Wen Model</h2>
                      <p class="summary-lead">
                        This dashboard calibrates a Bouc-Wen model for a friction pendulum system and then scales the calibrated model to a rigid mass supported by 15 identical bearings. The dynamic response is computed under the Kobe and Sylmar records and compared with an internal bilinear plasticity model.
                      </p>
                    </header>

                    <div class="page">
                <section class="hero">
                  <h1>CE223 Problem 1 — Friction Pendulum System with Bouc-Wen Model</h1>
                  <p>This report is self-contained and explicitly follows parts (a) through (d) of the assignment.</p>
                  <p><strong>Assumption made explicitly for part (d):</strong> the original Homework Assignment 4 output files are not embedded in this script, so the comparison is carried out against an internal bilinear plasticity model created from the same FPS properties.</p>
                </section>

                <section class="card">
                  <h2>Problem data and governing equations</h2>
                  <p>The rigid superstructure has total mass \(M = 1.47\times 10^6\,\mathrm{{kg}}\) and is supported by 15 identical FPS bearings. The given bearing data are \(R=1.0\,\mathrm{{m}}\), \(\mu=0.03\), and \(u_y = 0.03\,\mathrm{{mm}}\).</p>
                  <p>The prescribed cyclic motion for parts (a) and (b) is:</p>
                  <div class="equation">\[u(t)=0.4\sin(2\pi t)\ \mathrm{{m}}.\]</div>
                  <p>The Bouc--Wen restoring force for a single equivalent FPS component is written as:</p>
                  <div class="equation">\[F(t)=\alpha K_1 u(t) + (1-\alpha) K_1 u_y z(t).\]</div>
                  <p>The hysteretic variable satisfies:</p>
                  <div class="equation">\[u_y\dot z + \gamma z|\dot u||z|^{{n-1}} + \beta \dot u |z|^n - \dot u = 0.\]</div>
                  <p>For the earthquake-response analysis, the rigid-mass equation of motion is:</p>
                  <div class="equation">\[M\ddot u(t) + F_\mathrm{{tot}}\bigl(u(t),z(t)\bigr) = -M\ddot u_g(t),\qquad F_\mathrm{{tot}} = N_b F_b.\]</div>
                  <p>The mechanical matching used in the calibration is:</p>
                  <div class="equation">\[K_2 = \frac{{W}}{{R}}, \qquad Q = \mu W, \qquad \alpha K_1 = K_2, \qquad (1-\alpha)K_1u_y = Q.\]</div>
                  {parameter_table}
                </section>

                <section class="card">
                  <h2>Part (a) — Expected hysteretic loop for the prescribed cyclic test</h2>
                  <p>The ideal FPS response is represented as a bilinear force law with secondary slope \(K_2=W_b/R\) and characteristic friction strength \(Q_b=\mu W_b\) for one bearing. Because the prescribed displacement is \(u(t)=0.4\sin(2\pi t)\), the cyclic loop is confined to the interval \([-400,400]\,\mathrm{{mm}}\).</p>
                  <div class="plot-embed">{sections[0]}</div>
                </section>

                <section class="card">
                  <h2>Part (b) — Bouc-Wen approximation of the cyclic loop</h2>
                  <p>The Bouc--Wen model is calibrated by fixing \(\alpha\) from the mechanical matching above and then adjusting \(\beta\), \(\gamma\), and \(n\) to reproduce the rounded transition of the ideal loop. In the calibration trials below, \(\beta+\gamma=1\) is kept fixed while \(n\) is increased to sharpen the transition.</p>
                  {calibration_table}
                  <div class="plot-embed">{sections[1]}</div>
                  <div class="plot-embed">{sections[2]}</div>
                  <div class="plot-embed">{sections[3]}</div>
                  <div class="plot-embed">{sections[4]}</div>
                </section>

                <section class="card">
                  <h2>Part (c) — Nonlinear time-history analysis with the Bouc-Wen model</h2>
                  <p>For each motion, the required five stacked histories are shown first: displacement, velocity, normalized force \((F/W)\), absolute acceleration in units of \(g\), and ground acceleration in units of \(g\). The corresponding \((F/W)\)-versus-displacement loop is shown afterward.</p>
                  <h3>Kobe motion</h3>
                  <div class="plot-embed">{sections[5]}</div>
                  <div class="plot-embed">{sections[6]}</div>
                  <h3>Sylmar motion</h3>
                  <div class="plot-embed">{sections[7]}</div>
                  <div class="plot-embed">{sections[8]}</div>
                </section>

                <section class="card">
                  <h2>Part (d) — Comparison with the plasticity model</h2>
                  <p>The comparison below uses the self-contained bilinear plasticity model defined in this script. It has the same total mass, the same characteristic strength \(Q\), and the same post-yield slope \(K_2=W/R\). The first table reports the absolute peak responses. The second table reports the percent difference of the Bouc--Wen response relative to the plasticity model.</p>
                  {peak_table}
                  {difference_table}
                  <div class="plot-embed">{sections[9]}</div>
                  <div class="plot-embed">{sections[10]}</div>
                </section>
                    </div>
                  </div>
                </section>
              </div>
            </body>
            </html>
            """
        ).strip()

    def _build_parameter_table(self) -> str:
        """Create the table of mechanical parameters used in the report."""
        p = self.bw_params
        b = self.bilinear_params
        rows = [
            ("Total mass", f"{self.data.total_mass:.3e}", "kg"),
            ("Total weight", f"{self.data.weight / 1e6:.4f}", "MN"),
            ("Number of bearings", f"{self.data.n_bearings:d}", "-"),
            ("Dish radius", f"{self.data.radius:.3f}", "m"),
            ("Friction coefficient", f"{self.data.friction_coefficient:.4f}", "-"),
            ("Yield displacement", f"{self.data.yield_displacement_mm:.5f}", "mm"),
            ("Bouc-Wen alpha", f"{p.alpha:.6f}", "-"),
            ("Bouc-Wen beta", f"{p.beta_bw:.4f}", "-"),
            ("Bouc-Wen gamma", f"{p.gamma_bw:.4f}", "-"),
            ("Bouc-Wen n", f"{p.exponent_n:.2f}", "-"),
            ("Bearing K1", f"{p.bearing_k1 / 1e6:.6f}", "MN/m"),
            ("Bearing K2", f"{p.bearing_k2 / 1e6:.6f}", "MN/m"),
            ("Bearing αK1 = K2", f"{p.alpha * p.bearing_k1 / 1e6:.6f}", "MN/m"),
            ("Bearing (1−α)K1 uy = Q", f"{p.bearing_hysteretic_strength / 1e3:.6f}", "kN"),
            ("Total K1", f"{p.total_k1 / 1e9:.4f}", "GN/m"),
            ("Total K2", f"{p.total_k2 / 1e6:.4f}", "MN/m"),
            ("Total αK1 = K2", f"{p.alpha * p.total_k1 / 1e6:.4f}", "MN/m"),
            ("Total (1−α)K1 uy = Q", f"{p.total_hysteretic_strength / 1e6:.4f}", "MN"),
            ("Total Q", f"{p.total_characteristic_strength / 1e6:.4f}", "MN"),
            ("Plasticity elastic stiffness", f"{b.elastic_stiffness / 1e9:.4f}", "GN/m"),
            ("Plasticity hardening modulus", f"{b.hardening_modulus / 1e6:.4f}", "MN/m"),
        ]
        body = "".join(f"<tr><td>{name}</td><td>{value}</td><td>{unit}</td></tr>" for name, value, unit in rows)
        return (
            '<div class="summary-table-wrap"><table class="summary-table"><thead><tr>'
            "<th scope='col'>Quantity</th><th scope='col'>Value</th><th scope='col'>Unit</th>"
            f"</tr></thead><tbody>{body}</tbody></table></div>"
        )

    @staticmethod
    def _build_calibration_table(calibration_cases: list[CalibrationCaseResult]) -> str:
        """Summarize each Bouc--Wen calibration trial."""
        rows = []
        for case in calibration_cases:
            result = case.cyclic_result
            mask = result.last_cycle_mask
            error = result.bouc_wen_force_over_weight[mask] - result.ideal_force_over_weight[mask]
            max_error = float(np.max(np.abs(error)))
            rms_error = float(math.sqrt(np.mean(error * error)))
            area_bw = abs(integrate_trapezoid(result.bouc_wen_force_over_weight[mask], result.displacement[mask]))
            area_ideal = abs(integrate_trapezoid(result.ideal_force_over_weight[mask], result.displacement[mask]))
            area_error = 100.0 * (area_bw - area_ideal) / max(area_ideal, 1e-14)
            rows.append(
                "<tr>"
                f"<td>{case.label}</td>"
                f"<td>{case.parameters.beta_bw:.3f}</td>"
                f"<td>{case.parameters.gamma_bw:.3f}</td>"
                f"<td>{case.parameters.exponent_n:.1f}</td>"
                f"<td>{max_error:.5f}</td>"
                f"<td>{rms_error:.5f}</td>"
                f"<td>{area_error:+.2f}</td>"
                "</tr>"
            )
        return (
            '<div class="summary-table-wrap"><table class="summary-table"><thead><tr>'
            "<th scope='col'>Case</th><th scope='col'>β</th><th scope='col'>γ</th><th scope='col'>n</th>"
            "<th scope='col'>Max |Δ(F/Wb)|</th><th scope='col'>RMS |Δ(F/Wb)|</th>"
            "<th scope='col'>Loop area error [%]</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )

    @staticmethod
    def _build_difference_table(
        bw_results: dict[str, TimeHistoryResult],
        bilinear_results: dict[str, TimeHistoryResult],
    ) -> str:
        """Create a percent-difference table using the plasticity model as reference."""
        rows = []
        for motion in ("Kobe", "Sylmar"):
            bw = bw_results[motion]
            pl = bilinear_results[motion]

            def pct(a: float, b: float) -> float:
                return 100.0 * (a - b) / max(abs(b), 1.0e-14)

            rows.append(
                "<tr>"
                f"<td>{motion}</td>"
                f"<td>{pct(bw.peak_displacement, pl.peak_displacement):+.2f}</td>"
                f"<td>{pct(bw.peak_velocity, pl.peak_velocity):+.2f}</td>"
                f"<td>{pct(bw.peak_force, pl.peak_force):+.2f}</td>"
                f"<td>{pct(bw.peak_abs_acc, pl.peak_abs_acc):+.2f}</td>"
                "</tr>"
            )
        return (
            '<div class="summary-table-wrap"><table class="summary-table"><thead><tr>'
            "<th scope='col'>Motion</th><th scope='col'>Δ Peak |u| [%]</th>"
            "<th scope='col'>Δ Peak |u̇| [%]</th><th scope='col'>Δ Peak |F| [%]</th>"
            "<th scope='col'>Δ Peak |ü_t| [%]</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )


    @staticmethod
    def _build_peak_table(
        bw_results: dict[str, TimeHistoryResult],
        bilinear_results: dict[str, TimeHistoryResult],
    ) -> str:
        """Create the response-peak comparison table."""
        rows = []
        for motion in ("Kobe", "Sylmar"):
            for model_name, result in (("Bouc-Wen", bw_results[motion]), ("Plasticity", bilinear_results[motion])):
                rows.append(
                    "<tr>"
                    f"<td>{motion}</td>"
                    f"<td>{model_name}</td>"
                    f"<td>{result.peak_displacement * 1e3:.3f}</td>"
                    f"<td>{result.peak_velocity:.4f}</td>"
                    f"<td>{result.peak_force / 1e6:.4f}</td>"
                    f"<td>{result.peak_abs_acc / G_SI:.4f}</td>"
                    "</tr>"
                )
        return (
            '<div class="summary-table-wrap"><table class="summary-table"><thead><tr>'
            "<th scope='col'>Motion</th><th scope='col'>Model</th><th scope='col'>Peak |u| [mm]</th>"
            "<th scope='col'>Peak |u̇| [m/s]</th><th scope='col'>Peak |F| [MN]</th>"
            "<th scope='col'>Peak |ü_t| [g]</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )


# -----------------------------------------------------------------------------
# Small helpers and script entry point
# -----------------------------------------------------------------------------


def bouc_wen_z_rate(
    velocity: float,
    z: float,
    beta_bw: float,
    gamma_bw: float,
    exponent_n: float,
    yield_displacement: float,
) -> float:
    """Return dz/dt for the scalar Bouc--Wen evolution equation."""
    abs_z = abs(z)
    return (
        velocity
        - gamma_bw * abs(velocity) * z * abs_z ** (exponent_n - 1.0)
        - beta_bw * velocity * abs_z ** exponent_n
    ) / yield_displacement


def bouc_wen_z_rate_derivative_wrt_z(
    velocity: float,
    z: float,
    beta_bw: float,
    gamma_bw: float,
    exponent_n: float,
    yield_displacement: float,
) -> float:
    """Return partial derivative of dz/dt with respect to z."""
    abs_z = abs(z)
    if abs_z == 0.0:
        abs_power = 1.0 if exponent_n == 1.0 else 0.0
        sign_z = 0.0
    else:
        abs_power = abs_z ** (exponent_n - 1.0)
        sign_z = 1.0 if z > 0.0 else -1.0

    d_z_abs_to_n_minus_1 = exponent_n * abs_power
    d_abs_to_n = exponent_n * abs_power * sign_z
    return -(
        gamma_bw * abs(velocity) * d_z_abs_to_n_minus_1
        + beta_bw * velocity * d_abs_to_n
    ) / yield_displacement


def bouc_wen_z_rate_derivative_wrt_velocity(
    velocity: float,
    z: float,
    beta_bw: float,
    gamma_bw: float,
    exponent_n: float,
    yield_displacement: float,
) -> float:
    """Return partial derivative of dz/dt with respect to velocity."""
    abs_z = abs(z)
    sign_v = 0.0 if velocity == 0.0 else (1.0 if velocity > 0.0 else -1.0)
    return (
        1.0
        - gamma_bw * sign_v * z * abs_z ** (exponent_n - 1.0)
        - beta_bw * abs_z ** exponent_n
    ) / yield_displacement


def solve_bouc_wen_z_backward_euler(
    z_old: float,
    velocity_new: float,
    dt: float,
    beta_bw: float,
    gamma_bw: float,
    exponent_n: float,
    yield_displacement: float,
    tolerance: float = 1.0e-12,
    max_iterations: int = 50,
) -> float:
    """Advance z by one backward-Euler step without artificial state clipping."""
    if dt <= 0.0 or velocity_new == 0.0:
        return z_old

    z_new = z_old
    previous_norm = math.inf
    for _ in range(max_iterations):
        rate = bouc_wen_z_rate(
            velocity=velocity_new,
            z=z_new,
            beta_bw=beta_bw,
            gamma_bw=gamma_bw,
            exponent_n=exponent_n,
            yield_displacement=yield_displacement,
        )
        residual = z_new - z_old - dt * rate
        residual_norm = abs(residual)
        if residual_norm < tolerance:
            return float(z_new)

        tangent = 1.0 - dt * bouc_wen_z_rate_derivative_wrt_z(
            velocity=velocity_new,
            z=z_new,
            beta_bw=beta_bw,
            gamma_bw=gamma_bw,
            exponent_n=exponent_n,
            yield_displacement=yield_displacement,
        )
        if tangent == 0.0:
            raise RuntimeError("Zero tangent in the backward-Euler Bouc-Wen z update.")

        correction = -residual / tangent
        step_scale = 1.0
        accepted = False
        for _line_search in range(20):
            trial = z_new + step_scale * correction
            if math.isfinite(trial):
                trial_rate = bouc_wen_z_rate(
                    velocity=velocity_new,
                    z=trial,
                    beta_bw=beta_bw,
                    gamma_bw=gamma_bw,
                    exponent_n=exponent_n,
                    yield_displacement=yield_displacement,
                )
                trial_residual = trial - z_old - dt * trial_rate
                trial_norm = abs(trial_residual)
                if trial_norm < residual_norm or trial_norm < previous_norm:
                    z_new = trial
                    previous_norm = trial_norm
                    accepted = True
                    break
            step_scale *= 0.5

        if not accepted:
            raise RuntimeError("Backward-Euler Bouc-Wen z update failed to find a decreasing step.")

    raise RuntimeError("Backward-Euler Bouc-Wen z update did not converge.")


def integrate_trapezoid(y_values: np.ndarray, x_values: np.ndarray) -> float:
    """Return ∫ y dx without triggering NumPy's deprecated trapz warning."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_values, x_values))
    return float(np.trapz(y_values, x_values))


def sign_with_memory(values: np.ndarray) -> np.ndarray:
    """Return signs while carrying the previous nonzero sign through zeros."""
    signs = np.sign(np.asarray(values, dtype=float))
    previous = 1.0
    for i, value in enumerate(signs):
        if value == 0.0:
            signs[i] = previous
        else:
            previous = value
    return signs


def main() -> None:
    """Run the calibration, solve the records, and write the dashboard."""
    problem_data = FpsProblemData()

    # Parameter iterations used to show how n controls the loop transition sharpness.
    calibration_specs = [
        ("Trial 1: β = 0.50, γ = 0.50, n = 1", 0.5, 0.5, 1.0),
        ("Trial 2: β = 0.50, γ = 0.50, n = 3", 0.5, 0.5, 3.0),
        ("Final choice: β = 0.50, γ = 0.50, n = 5", 0.5, 0.5, 5.0),
    ]
    calibration_cases: list[CalibrationCaseResult] = []
    for label, beta_bw, gamma_bw, exponent_n in calibration_specs:
        params = FpsParameterBuilder.bouc_wen_from_problem_data(
            problem_data,
            beta_bw=beta_bw,
            gamma_bw=gamma_bw,
            exponent_n=exponent_n,
        )
        model = BoucWenForceModel(params)
        cyclic = BoucWenCyclicSolver(model).solve(amplitude=0.4, frequency_hz=1.0, n_cycles=3)
        calibration_cases.append(CalibrationCaseResult(label=label, parameters=params, cyclic_result=cyclic))

    bw_params = calibration_cases[-1].parameters
    bw_model = BoucWenForceModel(bw_params)
    cyclic_result = calibration_cases[-1].cyclic_result
    bilinear_params = FpsParameterBuilder.bilinear_from_problem_data(problem_data)

    # Ground motions are loaded after calibration so solver parameters are already fixed.
    kobe_path = first_existing_path(*KOBE_CANDIDATES)
    sylmar_path = first_existing_path(*SYLMAR_CANDIDATES)
    records = {
        "Kobe": GroundMotionLoader.load_acceleration_file(kobe_path, name="Kobe"),
        "Sylmar": GroundMotionLoader.load_acceleration_file(sylmar_path, name="Sylmar"),
    }

    bw_solver = BoucWenDynamicSolver(bw_model, max_internal_dt=1.0e-4)
    bilinear_solver = NonlinearBilinearNewmarkSolver(BilinearConstitutiveModel(bilinear_params))

    bw_results: dict[str, TimeHistoryResult] = {}
    bilinear_results: dict[str, TimeHistoryResult] = {}

    for motion_name, record in records.items():
        # Both models use the same input record and total structural mass.
        bw_results[motion_name] = bw_solver.solve(record)
        bilinear_results[motion_name] = bilinear_solver.solve(record)

    report = HtmlReportBuilder(bw_params, bilinear_params, problem_data).build(
        cyclic_result=cyclic_result,
        calibration_cases=calibration_cases,
        bw_results=bw_results,
        bilinear_results=bilinear_results,
    )
    OUTPUT_HTML.write_text(report, encoding="utf-8")
    print(f"Wrote self-contained dashboard to {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
