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
OUTPUT_HTML = HIGHLIGHTED_HTML_DIR / "CE223_LRB_BoucWen_Kobe_Sylmar.html"

MATLAB_COLORS = {
    "dark_blue": "rgb(0, 70, 140)",
    "black": "rgb(20, 20, 20)",
    "crimson": "rgb(180, 20, 60)",
    "dark_green": "rgb(0, 100, 0)",
    "gray": "rgb(110, 110, 110)",
    "light_gray": "rgb(175, 175, 175)",
    "gold": "rgb(210, 150, 0)",
    "purple": "rgb(90, 50, 130)",
}


@dataclass(frozen=True)
class LrbProblemData:
    """Problem constants and calibration readings for the LRB example.

    The graph gives the secondary branch slope as S = 0.92 kN/mm and
    marks the steep corner tangent as 10S. The peak load is read from the
    scanned loop and is used only to estimate the transition displacement.
    The selected value gives a derived branch intercept close to 100 kN.
    """
    total_mass: float = 50.96e6
    n_bearings: int = 200
    cyclic_amplitude: float = 0.235
    cyclic_frequency_hz: float = 1.0
    secondary_slope_kN_per_mm: float = 0.92
    initial_slope_multiplier: float = 10.0
    peak_load_at_amplitude_kN: float = 316.0
    reference_beta_bw: float = 0.50
    reference_gamma_bw: float = 0.50
    reference_exponent_n: float = 1.0

    @property
    def total_weight(self) -> float:
        return self.total_mass * G_SI

    @property
    def bearing_weight(self) -> float:
        return self.total_weight / float(self.n_bearings)

    @property
    def amplitude_mm(self) -> float:
        return self.cyclic_amplitude * 1.0e3

    @property
    def bearing_secondary_slope(self) -> float:
        """Secondary branch slope K2 = S in N/m."""
        return self.secondary_slope_kN_per_mm * 1.0e6

    @property
    def bearing_initial_tangent(self) -> float:
        """Initial/corner tangent K1 = 10S in N/m."""
        return self.initial_slope_multiplier * self.bearing_secondary_slope

    @property
    def alpha(self) -> float:
        return self.bearing_secondary_slope / self.bearing_initial_tangent

    @property
    def bearing_yield_displacement(self) -> float:
        """Estimate uy from the peak load and the two graph slopes.

        For a bilinear backbone, the large-displacement branch is
        F ≈ K2 u + (K1 - K2) uy. Therefore uy follows from the measured
        point (U, F_peak) on the upper branch.
        """
        peak_force = self.peak_load_at_amplitude_kN * 1.0e3
        branch_force = self.bearing_secondary_slope * self.cyclic_amplitude
        tangent_gap = self.bearing_initial_tangent - self.bearing_secondary_slope
        return (peak_force - branch_force) / tangent_gap

    @property
    def bearing_characteristic_strength(self) -> float:
        """Derived intercept force (K1 - K2) uy, reported but not fitted directly."""
        return (self.bearing_initial_tangent - self.bearing_secondary_slope) * self.bearing_yield_displacement

    @property
    def yield_displacement(self) -> float:
        return self.bearing_yield_displacement

    @property
    def bearing_test_slope(self) -> float:
        """Backward-compatible alias for K2 = S."""
        return self.bearing_secondary_slope

    @property
    def bearing_corner_tangent(self) -> float:
        """Backward-compatible alias for K1 = 10S."""
        return self.bearing_initial_tangent

    @property
    def corner_slope_multiplier(self) -> float:
        return self.initial_slope_multiplier

    @property
    def total_corner_tangent(self) -> float:
        return float(self.n_bearings) * self.bearing_initial_tangent

    @property
    def total_test_slope(self) -> float:
        return float(self.n_bearings) * self.bearing_secondary_slope

    @property
    def total_characteristic_strength(self) -> float:
        return float(self.n_bearings) * self.bearing_characteristic_strength


@dataclass(frozen=True)
class BoucWenLrbParameters:
    """Derived Bouc-Wen parameters used by the cyclic and dynamic solvers."""
    total_mass: float
    n_bearings: int
    bearing_corner_tangent: float
    bearing_test_slope: float
    bearing_characteristic_strength: float
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
    def total_corner_tangent(self) -> float:
        return float(self.n_bearings) * self.bearing_corner_tangent

    @property
    def total_test_slope(self) -> float:
        return float(self.n_bearings) * self.bearing_test_slope

    @property
    def total_characteristic_strength(self) -> float:
        return float(self.n_bearings) * self.bearing_characteristic_strength

    @property
    def bearing_hysteretic_strength(self) -> float:
        return (1.0 - self.alpha) * self.bearing_corner_tangent * self.yield_displacement

    @property
    def total_hysteretic_strength(self) -> float:
        return float(self.n_bearings) * self.bearing_hysteretic_strength


@dataclass(frozen=True)
class EquivalentLinearProperties:
    """Equivalent linear oscillator properties computed from one cyclic loop."""
    mass: float
    stiffness: float
    damping_ratio: float
    damping: float
    circular_frequency: float
    period: float
    amplitude: float
    dissipated_energy: float
    stored_energy: float


@dataclass
class GroundMotionRecord:
    """Ground acceleration record stored in SI units."""
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
    """Container for response histories produced by a time-history solver."""
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
    """Container for a prescribed cyclic displacement test of one bearing."""
    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    hysteretic_parameter: np.ndarray
    bouc_wen_force: np.ndarray
    target_force: np.ndarray
    bearing_weight: float
    cycle_frequency_hz: float
    n_cycles: int

    @property
    def bouc_wen_force_over_weight(self) -> np.ndarray:
        return self.bouc_wen_force / self.bearing_weight

    @property
    def target_force_over_weight(self) -> np.ndarray:
        return self.target_force / self.bearing_weight

    @property
    def last_cycle_mask(self) -> np.ndarray:
        period = 1.0 / self.cycle_frequency_hz
        return self.time >= (self.n_cycles - 1) * period


@dataclass(frozen=True)
class CalibrationCaseResult:
    """One calibration trial and its resulting cyclic loop."""
    label: str
    parameters: BoucWenLrbParameters
    cyclic_result: CyclicTestResult


class GroundMotionLoader:
    """Read PEER-style and simple numeric ground-motion files."""
    @staticmethod
    def load_acceleration_file(path: Path, name: str | None = None) -> GroundMotionRecord:
        """Load an acceleration file and return the record in m/s²."""
        if not path.exists():
            raise FileNotFoundError(f"Ground motion file not found: {path}")

        dt = GroundMotionLoader._parse_dt(path)
        numeric_rows = GroundMotionLoader._read_numeric_rows(path)
        if numeric_rows.size == 0:
            raise ValueError(f"No numeric acceleration data found in: {path}")

        if numeric_rows.ndim == 2 and numeric_rows.shape[1] >= 2 and GroundMotionLoader._looks_like_time_column(numeric_rows[:, 0]):
            time = np.asarray(numeric_rows[:, 0], dtype=float)
            acc_g = np.asarray(numeric_rows[:, 1], dtype=float)
            dt_from_time = float(np.median(np.diff(time)))
            dt = dt_from_time if dt is None else dt
        else:
            acc_g = np.asarray(numeric_rows, dtype=float).ravel()

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
        if values.size < 3:
            return False
        diffs = np.diff(values)
        if not np.all(diffs > 0.0):
            return False
        return bool(np.std(diffs) <= 1.0e-4 * max(abs(float(np.mean(diffs))), 1.0e-12))


class LrbParameterBuilder:
    """Build a consistent Bouc-Wen parameter set from the problem data."""
    @staticmethod
    def bouc_wen_from_problem_data(
        data: LrbProblemData,
        beta_bw: float = 0.50,
        gamma_bw: float = 0.50,
        exponent_n: float = 2.0,
    ) -> BoucWenLrbParameters:
        """Create the parameter object and validate the basic Bouc-Wen inputs."""
        if not 0.0 < data.alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1.")
        if data.yield_displacement <= 0.0:
            raise ValueError("yield displacement must be positive.")
        if beta_bw < 0.0 or gamma_bw < 0.0:
            raise ValueError("Bouc-Wen beta and gamma should be nonnegative for this calibration.")
        if exponent_n < 1.0:
            raise ValueError("The Bouc-Wen exponent should be at least 1.0 for this implementation.")

        return BoucWenLrbParameters(
            total_mass=data.total_mass,
            n_bearings=data.n_bearings,
            bearing_corner_tangent=data.bearing_corner_tangent,
            bearing_test_slope=data.bearing_test_slope,
            bearing_characteristic_strength=data.bearing_characteristic_strength,
            yield_displacement=data.yield_displacement,
            alpha=data.alpha,
            beta_bw=float(beta_bw),
            gamma_bw=float(gamma_bw),
            exponent_n=float(exponent_n),
        )


class BoucWenForceModel:
    """Evaluate bearing and system restoring forces from displacement and z."""
    def __init__(self, parameters: BoucWenLrbParameters) -> None:
        self.parameters = parameters

    def total_force(self, displacement: np.ndarray | float, z: np.ndarray | float) -> np.ndarray | float:
        p = self.parameters
        return p.alpha * p.total_corner_tangent * displacement + (1.0 - p.alpha) * p.total_corner_tangent * p.yield_displacement * z

    def bearing_force(self, displacement: np.ndarray | float, z: np.ndarray | float) -> np.ndarray | float:
        p = self.parameters
        return p.alpha * p.bearing_corner_tangent * displacement + (1.0 - p.alpha) * p.bearing_corner_tangent * p.yield_displacement * z

    def target_bearing_force(self, displacement: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        p = self.parameters
        return p.bearing_test_slope * displacement + p.bearing_characteristic_strength * sign_with_memory(velocity)


class BoucWenCyclicSolver:
    """Integrate the Bouc-Wen internal variable for prescribed sinusoidal motion."""
    def __init__(self, model: BoucWenForceModel, points_per_cycle: int = 16000) -> None:
        self.model = model
        self.points_per_cycle = points_per_cycle

    def solve(self, amplitude: float = 0.235, frequency_hz: float = 1.0, n_cycles: int = 3) -> CyclicTestResult:
        """Run a prescribed cyclic displacement test for one bearing."""
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

        z = self._integrate_z(
            time=time,
            amplitude=amplitude,
            omega=omega,
            beta_bw=p.beta_bw,
            gamma_bw=p.gamma_bw,
            exponent_n=p.exponent_n,
            yield_displacement=p.yield_displacement,
        )

        # The guide curve is smooth because the scanned test loop has rounded reversals.
        reference_z = self._integrate_z(
            time=time,
            amplitude=amplitude,
            omega=omega,
            beta_bw=0.50,
            gamma_bw=0.50,
            exponent_n=1.0,
            yield_displacement=p.yield_displacement,
        )

        bw_force = self.model.bearing_force(displacement, z)
        # This is an eyeballed smooth recorded-loop guide. It intentionally avoids the
        # discontinuous sign law because the experimental loop has rounded reversals.
        target_force = (
            p.alpha * p.bearing_corner_tangent * displacement
            + (1.0 - p.alpha) * p.bearing_corner_tangent * p.yield_displacement * reference_z
        )

        return CyclicTestResult(
            time=time,
            displacement=displacement,
            velocity=velocity,
            hysteretic_parameter=z,
            bouc_wen_force=np.asarray(bw_force, dtype=float),
            target_force=np.asarray(target_force, dtype=float),
            bearing_weight=p.bearing_weight,
            cycle_frequency_hz=frequency_hz,
            n_cycles=n_cycles,
        )

    @staticmethod
    def _integrate_z(
        time: np.ndarray,
        amplitude: float,
        omega: float,
        beta_bw: float,
        gamma_bw: float,
        exponent_n: float,
        yield_displacement: float,
    ) -> np.ndarray:
        """Integrate the internal variable z with a fourth-order Runge-Kutta rule."""
        z = np.zeros(time.size, dtype=float)

        def velocity_at(t_value: float) -> float:
            return amplitude * omega * math.cos(omega * t_value)

        def dz_dt(t_value: float, z_value: float) -> float:
            v_value = velocity_at(t_value)
            abs_z = abs(z_value)
            return (
                v_value
                - gamma_bw * abs(v_value) * z_value * abs_z ** (exponent_n - 1.0)
                - beta_bw * v_value * abs_z ** exponent_n
            ) / yield_displacement

        for i in range(1, time.size):
            dt = float(time[i] - time[i - 1])
            t0 = float(time[i - 1])
            z0 = float(z[i - 1])
            k1 = dz_dt(t0, z0)
            k2 = dz_dt(t0 + 0.5 * dt, z0 + 0.5 * dt * k1)
            k3 = dz_dt(t0 + 0.5 * dt, z0 + 0.5 * dt * k2)
            k4 = dz_dt(t0 + dt, z0 + dt * k3)
            z_next = z0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if not np.isfinite(z_next) or abs(z_next) > 10.0:
                raise FloatingPointError(
                    "Bouc-Wen hysteretic variable became unstable. "
                    "Try a smaller time step or different beta/gamma/n parameters."
                )
            z[i] = z_next

        return z


class BoucWenDynamicSolver:
    """State-vector Newmark solver for the Bouc-Wen isolated mass.

    The unknown advanced at each step is y = [u, v, z]^T, where u is the
    relative displacement, v is the relative velocity, and z is the Bouc-Wen
    hysteretic parameter. The first two residual equations are the Newmark
    kinematic relations, while the third residual equation is a backward-Euler
    discretization of the z evolution law.
    """

    def __init__(
        self,
        model: BoucWenForceModel,
        beta_newmark: float = 1.0 / 4.0,
        gamma_newmark: float = 1.0 / 2.0,
        tolerance: float = 1.0e-9,
        max_iterations: int = 35,
        max_internal_dt: float | None = None,
    ) -> None:
        self.model = model
        self.beta_newmark = beta_newmark
        self.gamma_newmark = gamma_newmark
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        if max_internal_dt is not None and max_internal_dt <= 0.0:
            raise ValueError("max_internal_dt must be positive when provided.")
        self.max_internal_dt = max_internal_dt

    def solve(self, record: GroundMotionRecord) -> TimeHistoryResult:
        """Solve the nonlinear response using a coupled state-vector Newton solve."""
        p = self.model.parameters
        time = record.time_array
        ground_acc = record.acceleration_mps2
        n_steps = ground_acc.size

        u = np.zeros(n_steps, dtype=float)
        v = np.zeros(n_steps, dtype=float)
        z = np.zeros(n_steps, dtype=float)
        a = np.zeros(n_steps, dtype=float)
        restoring_force = np.zeros(n_steps, dtype=float)

        y = np.zeros(3, dtype=float)  # y = [u, v, z]^T
        restoring_force[0] = float(self.model.total_force(y[0], y[2]))
        a[0] = self._acceleration(y, float(ground_acc[0]))

        for i in range(1, n_steps):
            dt_total = float(time[i] - time[i - 1])
            if dt_total <= 0.0:
                raise ValueError("Ground-motion time array must be strictly increasing.")

            # Substepping keeps the coupled z evolution stable while storing output
            # only at the original ground-motion sample times.
            if self.max_internal_dt is None:
                n_substeps = 1
            else:
                n_substeps = max(1, int(math.ceil(dt_total / self.max_internal_dt)))
            dt_sub = dt_total / float(n_substeps)

            ag_start = float(ground_acc[i - 1])
            ag_end = float(ground_acc[i])
            a_current = float(a[i - 1])

            for substep in range(1, n_substeps + 1):
                eta = float(substep) / float(n_substeps)
                ag_sub = ag_start + eta * (ag_end - ag_start)
                y = self._solve_step(
                    y_previous=y.copy(),
                    a_previous=a_current,
                    ground_acceleration_next=ag_sub,
                    dt=dt_sub,
                )
                a_current = self._acceleration(y, ag_sub)

            u[i], v[i], z[i] = y
            restoring_force[i] = float(self.model.total_force(y[0], y[2]))
            a[i] = a_current

        absolute_acc = a + ground_acc

        return TimeHistoryResult(
            time=time,
            ground_acceleration=ground_acc,
            displacement=u,
            velocity=v,
            relative_acceleration=a,
            absolute_acceleration=absolute_acc,
            restoring_force=restoring_force,
            hysteretic_parameter=z,
        )

    def _solve_step(
        self,
        y_previous: np.ndarray,
        a_previous: float,
        ground_acceleration_next: float,
        dt: float,
    ) -> np.ndarray:
        """Newton iteration for y_{n+1} = [u_{n+1}, v_{n+1}, z_{n+1}]^T."""
        beta = self.beta_newmark
        gamma = self.gamma_newmark

        # Newmark predictor gives a stable first guess before enforcing equilibrium.
        y_trial = y_previous.copy()
        y_trial[0] = y_previous[0] + dt * y_previous[1] + 0.5 * dt * dt * a_previous
        y_trial[1] = y_previous[1] + dt * a_previous

        for _ in range(self.max_iterations):
            residual = self._state_residual(
                y_next=y_trial,
                y_previous=y_previous,
                a_previous=a_previous,
                ground_acceleration_next=ground_acceleration_next,
                dt=dt,
            )
            jacobian = self._finite_difference_jacobian(
                y_next=y_trial,
                y_previous=y_previous,
                a_previous=a_previous,
                ground_acceleration_next=ground_acceleration_next,
                dt=dt,
                residual_at_y=residual,
            )

            try:
                correction = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError:
                correction = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]

            y_trial = y_trial + correction
            self._check_state(y_trial)

            updated_residual = self._state_residual(
                y_next=y_trial,
                y_previous=y_previous,
                a_previous=a_previous,
                ground_acceleration_next=ground_acceleration_next,
                dt=dt,
            )
            if self._converged(updated_residual, correction, y_trial):
                return y_trial

        raise RuntimeError("Bouc-Wen state-vector Newmark iteration did not converge.")

    def _state_residual(
        self,
        y_next: np.ndarray,
        y_previous: np.ndarray,
        a_previous: float,
        ground_acceleration_next: float,
        dt: float,
    ) -> np.ndarray:
        """Residual for Newmark kinematics and the z evolution equation."""
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

        # Backward-Euler update for z uses the same end-of-step state vector.
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
        """Finite-difference Jacobian for the three coupled residual equations."""
        jacobian = np.zeros((3, 3), dtype=float)
        base_scale = np.array([1.0e-2, 1.0e-2, 1.0], dtype=float)
        for j in range(3):
            perturb = math.sqrt(np.finfo(float).eps) * max(abs(float(y_next[j])), float(base_scale[j]))
            y_perturbed = y_next.copy()
            y_perturbed[j] += perturb
            residual_perturbed = self._state_residual(
                y_next=y_perturbed,
                y_previous=y_previous,
                a_previous=a_previous,
                ground_acceleration_next=ground_acceleration_next,
                dt=dt,
            )
            jacobian[:, j] = (residual_perturbed - residual_at_y) / perturb
        return jacobian

    def _acceleration(self, state: np.ndarray, ground_acceleration: float) -> float:
        """Relative acceleration from equilibrium at a given state vector."""
        p = self.model.parameters
        restoring_force = float(self.model.total_force(float(state[0]), float(state[2])))
        return -ground_acceleration - restoring_force / p.total_mass

    def _zdot(self, velocity: float, z_value: float) -> float:
        """Bouc-Wen evolution law evaluated at the end-of-step state."""
        p = self.model.parameters
        abs_z = abs(z_value)
        return (
            velocity
            - p.gamma_bw * abs(velocity) * z_value * abs_z ** (p.exponent_n - 1.0)
            - p.beta_bw * velocity * abs_z ** p.exponent_n
        ) / p.yield_displacement

    @staticmethod
    def _check_state(state: np.ndarray) -> None:
        """Catch numerical divergence without artificially clipping the response."""
        if not np.all(np.isfinite(state)) or abs(float(state[2])) > 50.0:
            raise FloatingPointError("Bouc-Wen state vector became numerically unstable.")

    def _converged(self, residual: np.ndarray, correction: np.ndarray, state: np.ndarray) -> bool:
        """Scaled convergence check for mixed units in [u, v, z]."""
        state_scale = np.maximum(np.abs(state), np.array([1.0e-3, 1.0e-3, 1.0], dtype=float))
        residual_scale = np.array([1.0e-6, 1.0e-6, 1.0e-6], dtype=float)
        correction_error = float(np.max(np.abs(correction) / state_scale))
        residual_error = float(np.max(np.abs(residual) / residual_scale))
        return correction_error < self.tolerance and residual_error < 1.0


class EquivalentLinearBuilder:
    """Compute secant stiffness and equivalent damping from cyclic energy."""
    @staticmethod
    def from_cyclic_result(result: CyclicTestResult, parameters: BoucWenLrbParameters) -> EquivalentLinearProperties:
        """Match secant stiffness and energy dissipation at the test amplitude."""
        mask = result.last_cycle_mask
        displacement = result.displacement[mask]
        bearing_force = result.bouc_wen_force[mask]
        total_force = float(parameters.n_bearings) * bearing_force
        amplitude = float(np.max(np.abs(displacement)))
        force_max = float(np.max(np.abs(total_force)))
        stiffness = force_max / max(amplitude, 1.0e-12)

        bearing_energy = abs(integrate_trapezoid(bearing_force, displacement))
        dissipated_energy = float(parameters.n_bearings) * bearing_energy
        stored_energy = 0.5 * stiffness * amplitude * amplitude
        # No cap is imposed here; the table should show the damping implied by the loop.
        damping_ratio = float(dissipated_energy / max(4.0 * math.pi * stored_energy, 1.0e-12))
        if not math.isfinite(damping_ratio) or damping_ratio < 0.0:
            raise FloatingPointError("Equivalent damping ratio is not finite or is negative.")
        circular_frequency = math.sqrt(stiffness / parameters.total_mass)
        damping = 2.0 * damping_ratio * parameters.total_mass * circular_frequency
        period = 2.0 * math.pi / circular_frequency

        return EquivalentLinearProperties(
            mass=parameters.total_mass,
            stiffness=stiffness,
            damping_ratio=damping_ratio,
            damping=damping,
            circular_frequency=circular_frequency,
            period=period,
            amplitude=amplitude,
            dissipated_energy=dissipated_energy,
            stored_energy=stored_energy,
        )


class LinearNewmarkSolver:
    """Average-acceleration Newmark solver for a linear base-excited SDOF."""
    @staticmethod
    def solve_sdof_base_excitation(
        record: GroundMotionRecord,
        mass: float,
        damping: float,
        stiffness: float,
        beta: float = 1.0 / 4.0,
        gamma: float = 1.0 / 2.0,
    ) -> TimeHistoryResult:
        """Solve a linear SDOF oscillator subjected to ground acceleration."""
        ug = record.acceleration_mps2
        dt = record.dt
        n = ug.size

        u = np.zeros(n, dtype=float)
        v = np.zeros(n, dtype=float)
        a = np.zeros(n, dtype=float)
        f = np.zeros(n, dtype=float)
        abs_a = np.zeros(n, dtype=float)

        a[0] = (-damping * v[0] - stiffness * u[0] - mass * ug[0]) / mass
        abs_a[0] = a[0] + ug[0]
        f[0] = stiffness * u[0] + damping * v[0]

        # Standard average-acceleration Newmark constants.

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


class FigureFactory:
    """Create Plotly figures used in the HTML report."""
    @staticmethod
    def _max_marker_coordinates(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        idx = int(np.argmax(np.abs(y)))
        return float(x[idx]), float(y[idx])


    @staticmethod
    def cyclic_hysteresis(result: CyclicTestResult, parameters: BoucWenLrbParameters) -> go.Figure:
        """Plot the cyclic calibration path from the initial unloaded state."""
        period = 1.0 / result.cycle_frequency_hz
        initial_mask = result.time <= 0.25 * period
        last_mask = result.last_cycle_mask

        displacement_mm = result.displacement * 1.0e3
        target_kN = result.target_force / 1.0e3
        bw_kN = result.bouc_wen_force / 1.0e3
        velocity = result.velocity
        z = result.hysteretic_parameter
        time = result.time

        alpha_k1_kN_per_mm = parameters.alpha * parameters.bearing_corner_tangent / 1.0e6
        linear_kN = (parameters.alpha * parameters.bearing_corner_tangent * result.displacement) / 1.0e3
        hysteretic_kN = (
            (1.0 - parameters.alpha)
            * parameters.bearing_corner_tangent
            * parameters.yield_displacement
            * z
        ) / 1.0e3
        target_hysteretic_kN = target_kN - linear_kN

        displacement_initial_mm = displacement_mm[initial_mask]
        bw_initial_kN = bw_kN[initial_mask]
        time_initial = time[initial_mask]
        velocity_initial = velocity[initial_mask]
        z_initial = z[initial_mask]
        linear_initial_kN = linear_kN[initial_mask]
        hysteretic_initial_kN = hysteretic_kN[initial_mask]

        zero_x, zero_y = find_zero_force_crossings(displacement_mm[last_mask], bw_kN[last_mask])
        force_scale = max(float(np.max(np.abs(target_kN))), float(np.max(np.abs(bw_kN))))
        amplitude_mm = float(np.max(np.abs(result.displacement)) * 1.0e3)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=displacement_mm,
                y=target_kN,
                mode="lines",
                line=dict(color=MATLAB_COLORS["dark_blue"], width=2.8, dash="dash"),
                name="Eyeballed recorded-loop guide from t = 0",
                customdata=np.column_stack((time, velocity, z, linear_kN, target_hysteretic_kN, target_kN)),
                hovertemplate=(
                    "Model: eyeballed recorded-loop guide<br>"
                    "t: %{customdata[0]:.4f} s<br>"
                    "u: %{x:.3f} mm<br>"
                    "u_dot: %{customdata[1]:.5f} m/s<br>"
                    "z_guide: %{customdata[2]:.5f}<br>"
                    "alpha K1 = K2: " + f"{alpha_k1_kN_per_mm:.3f}" + " kN/mm<br>"
                    "alpha K1 u = K2 u: %{customdata[3]:.3f} kN<br>"
                    "(1-alpha) K1 uy z: %{customdata[4]:.3f} kN<br>"
                    "F: %{customdata[5]:.3f} kN<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=displacement_mm,
                y=bw_kN,
                mode="lines",
                line=dict(color=MATLAB_COLORS["crimson"], width=3.0),
                name="Selected Bouc-Wen path from t = 0",
                customdata=np.column_stack((time, velocity, z, linear_kN, hysteretic_kN, bw_kN)),
                hovertemplate=(
                    "Model: selected Bouc-Wen path<br>"
                    "t: %{customdata[0]:.4f} s<br>"
                    "u: %{x:.3f} mm<br>"
                    "u_dot: %{customdata[1]:.5f} m/s<br>"
                    "z: %{customdata[2]:.5f}<br>"
                    "alpha K1 = K2: " + f"{alpha_k1_kN_per_mm:.3f}" + " kN/mm<br>"
                    "alpha K1 u = K2 u: %{customdata[3]:.3f} kN<br>"
                    "(1-alpha) K1 uy z: %{customdata[4]:.3f} kN<br>"
                    "F: %{customdata[5]:.3f} kN<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=displacement_initial_mm,
                y=bw_initial_kN,
                mode="lines",
                line=dict(color=MATLAB_COLORS["black"], width=2.4, dash="dot"),
                name="First quarter-cycle from rest",
                customdata=np.column_stack((time_initial, velocity_initial, z_initial, linear_initial_kN, hysteretic_initial_kN, bw_initial_kN)),
                hovertemplate=(
                    "First quarter-cycle from rest<br>"
                    "t: %{customdata[0]:.4f} s<br>"
                    "u: %{x:.3f} mm<br>"
                    "u_dot: %{customdata[1]:.5f} m/s<br>"
                    "z: %{customdata[2]:.5f}<br>"
                    "alpha K1 = K2: " + f"{alpha_k1_kN_per_mm:.3f}" + " kN/mm<br>"
                    "alpha K1 u = K2 u: %{customdata[3]:.3f} kN<br>"
                    "(1-alpha) K1 uy z: %{customdata[4]:.3f} kN<br>"
                    "F: %{customdata[5]:.3f} kN<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[0.0],
                mode="markers+text",
                marker=dict(size=10, color=MATLAB_COLORS["black"], symbol="circle"),
                text=["initial state"],
                textposition="top right",
                name="Initial state (0,0)",
                hovertemplate="Initial state<br>t: 0.0000 s<br>u: 0.000 mm<br>F: 0.000 kN<br>z: 0.00000<extra></extra>",
            )
        )
        if zero_x.size > 0:
            fig.add_trace(
                go.Scatter(
                    x=zero_x,
                    y=zero_y,
                    mode="markers",
                    marker=dict(size=9, color=MATLAB_COLORS["dark_green"], symbol="diamond"),
                    name="Last-cycle F = 0 crossings",
                    hovertemplate="Last-cycle zero-force crossing<br>u: %{x:.3f} mm<br>F: %{y:.3f} kN<extra></extra>",
                )
            )

        fig.update_layout(
            template="plotly_white",
            height=720,
            title=dict(
                text="Cyclic Calibration — Lead Rubber Bearing",
                x=0.5,
                y=0.985,
                xanchor="center",
                yanchor="top",
                font=dict(size=22),
            ),
            margin=dict(t=170, r=35, b=75, l=70),
            xaxis=dict(
                title="Displacement u [mm]",
                range=[-1.12 * amplitude_mm, 1.12 * amplitude_mm],
                title_font=dict(size=16),
                tickfont=dict(size=13),
                zeroline=True,
                zerolinewidth=1.6,
                zerolinecolor=MATLAB_COLORS["gray"],
            ),
            yaxis=dict(
                title="Load F [kN]",
                range=[-1.18 * force_scale, 1.18 * force_scale],
                title_font=dict(size=16),
                tickfont=dict(size=13),
                zeroline=True,
                zerolinewidth=1.6,
                zerolinecolor=MATLAB_COLORS["gray"],
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            font=dict(size=14),
        )
        return fig




    @staticmethod
    def cyclic_z_history(result: CyclicTestResult, parameters: BoucWenLrbParameters) -> go.Figure:
        """Plot the cyclic evolution of the hysteretic parameter z(t)."""
        time = result.time
        z = result.hysteretic_parameter
        disp_mm = result.displacement * 1.0e3
        vel_mps = result.velocity
        linear_kN = (parameters.alpha * parameters.bearing_corner_tangent * result.displacement) / 1.0e3
        hysteretic_kN = (
            (1.0 - parameters.alpha)
            * parameters.bearing_corner_tangent
            * parameters.yield_displacement
            * z
        ) / 1.0e3
        total_force_kN = result.bouc_wen_force / 1.0e3

        z_max = float(np.max(np.abs(z))) if z.size > 0 else 0.0
        z_pad = 0.18 * max(z_max, 1.0e-9)
        x_peak, y_peak = FigureFactory._max_marker_coordinates(time, z)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=time,
                y=z,
                mode="lines",
                line=dict(color=MATLAB_COLORS["dark_blue"], width=2.6),
                name="z(t)",
                customdata=np.column_stack((disp_mm, vel_mps, linear_kN, hysteretic_kN, total_force_kN)),
                hovertemplate=(
                    "Time: %{x:.4f} s<br>"
                    "z: %{y:.5f}<br>"
                    "u: %{customdata[0]:.3f} mm<br>"
                    "u_dot: %{customdata[1]:.5f} m/s<br>"
                    "alpha K1 u = K2 u: %{customdata[2]:.3f} kN<br>"
                    "(1-alpha) K1 uy z: %{customdata[3]:.3f} kN<br>"
                    "F: %{customdata[4]:.3f} kN<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[x_peak],
                y=[y_peak],
                mode="markers",
                marker=dict(size=10, color=MATLAB_COLORS["black"]),
                showlegend=False,
                hovertemplate="Peak |z|<br>t: %{x:.4f} s<br>z: %{y:.5f}<extra></extra>",
            )
        )
        fig.add_annotation(
            x=x_peak,
            y=y_peak,
            text=f"z_max = {abs(y_peak):.5f}",
            showarrow=True,
            arrowhead=2,
            ax=24,
            ay=-24,
            font=dict(size=14, color=MATLAB_COLORS["black"]),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=MATLAB_COLORS["black"],
            borderwidth=1,
        )
        fig.update_layout(
            template="plotly_white",
            height=470,
            title=dict(text="Cyclic Calibration — Hysteretic Parameter z(t)", x=0.5, xanchor="center", font=dict(size=22)),
            xaxis=dict(title="Time [s]", title_font=dict(size=16), tickfont=dict(size=13), zeroline=True),
            yaxis=dict(title="Hysteretic Parameter z [-]", range=[float(np.min(z)) - z_pad, float(np.max(z)) + z_pad], title_font=dict(size=16), tickfont=dict(size=13), zeroline=True),
            font=dict(size=14),
            hovermode="x unified",
        )
        return fig




    @staticmethod
    def z_time_history(
        title: str,
        result: TimeHistoryResult,
        parameters: BoucWenLrbParameters,
    ) -> go.Figure:
        """Plot the dynamic evolution of the Bouc-Wen hysteretic parameter z(t)."""
        if result.hysteretic_parameter is None:
            raise ValueError("z_time_history requires a nonlinear result with hysteretic_parameter.")

        time = result.time
        z = np.asarray(result.hysteretic_parameter, dtype=float)
        disp_mm = result.displacement * 1.0e3
        vel_mps = result.velocity
        force_over_weight = result.restoring_force / parameters.total_weight
        force_mn = result.restoring_force / 1.0e6
        rel_acc_g = result.relative_acceleration / G_SI
        abs_acc_g = result.absolute_acceleration / G_SI
        ground_g = result.ground_acceleration / G_SI
        linear_mn = parameters.alpha * parameters.total_corner_tangent * result.displacement / 1.0e6
        hysteretic_mn = (
            (1.0 - parameters.alpha)
            * parameters.total_corner_tangent
            * parameters.yield_displacement
            * z
        ) / 1.0e6

        z_max = float(np.max(np.abs(z))) if z.size > 0 else 0.0
        z_pad = 0.18 * max(z_max, 1.0e-9)
        x_peak, y_peak = FigureFactory._max_marker_coordinates(time, z)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=time,
                y=z,
                mode="lines",
                line=dict(color=MATLAB_COLORS["dark_blue"], width=2.6),
                name="z(t)",
                customdata=np.column_stack((disp_mm, vel_mps, force_over_weight, force_mn, rel_acc_g, abs_acc_g, ground_g, linear_mn, hysteretic_mn)),
                hovertemplate=(
                    "Time: %{x:.3f} s<br>"
                    "z: %{y:.5f}<br>"
                    "u: %{customdata[0]:.3f} mm<br>"
                    "u_dot: %{customdata[1]:.4f} m/s<br>"
                    "F/W: %{customdata[2]:.5f}<br>"
                    "F: %{customdata[3]:.4f} MN<br>"
                    "a_rel: %{customdata[4]:.4f} g<br>"
                    "a_abs: %{customdata[5]:.4f} g<br>"
                    "a_g: %{customdata[6]:.4f} g<br>"
                    "alpha K1 u = K2 u: %{customdata[7]:.4f} MN<br>"
                    "(1-alpha)K1 uy z: %{customdata[8]:.4f} MN<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[x_peak],
                y=[y_peak],
                mode="markers",
                marker=dict(size=10, color=MATLAB_COLORS["black"]),
                showlegend=False,
                hovertemplate="Peak |z|<br>t: %{x:.3f} s<br>z: %{y:.5f}<extra></extra>",
            )
        )
        fig.add_annotation(
            x=x_peak,
            y=y_peak,
            text=f"z_max = {abs(y_peak):.5f}",
            showarrow=True,
            arrowhead=2,
            ax=24,
            ay=-24,
            font=dict(size=14, color=MATLAB_COLORS["black"]),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=MATLAB_COLORS["black"],
            borderwidth=1,
        )
        fig.update_layout(
            template="plotly_white",
            height=470,
            title=dict(text=title, x=0.5, xanchor="center", font=dict(size=22)),
            xaxis=dict(title="Time [s]", title_font=dict(size=16), tickfont=dict(size=13), zeroline=True),
            yaxis=dict(title="Hysteretic Parameter z [-]", range=[float(np.min(z)) - z_pad, float(np.max(z)) + z_pad], title_font=dict(size=16), tickfont=dict(size=13), zeroline=True),
            font=dict(size=14),
            hovermode="x unified",
        )
        return fig




    @staticmethod
    def interactive_cyclic_calibration_html(parameters: BoucWenLrbParameters, data: LrbProblemData) -> str:
        """Return a Plotly-native calibration panel for beta, gamma, and n.

        The plotted hysteresis traces show the complete cyclic path from the
        initial condition, not only the stabilized final cycle.
        """
        beta_value = parameters.beta_bw
        gamma_value = parameters.gamma_bw
        exponent_value = parameters.exponent_n
        derived_q_value = parameters.bearing_characteristic_strength / 1.0e3
        uy_value = parameters.yield_displacement * 1.0e3
        peak_force_value = data.peak_load_at_amplitude_kN
        amplitude_mm = data.cyclic_amplitude * 1.0e3
        secondary_slope = parameters.bearing_test_slope / 1.0e6
        initial_tangent = parameters.bearing_corner_tangent / 1.0e6
        alpha_value = parameters.alpha

        return dedent(
            f"""
            <div class="interactive-calibration-panel">
              <div class="interactive-calibration-header">
                <h3>Interactive Bouc-Wen calibration</h3>
                <p>
                  The Plotly sliders below update only the cyclic calibration plot in the browser. The time-history
                  figures continue to use the parameters printed in the tables. Once a better combination is found,
                  copy the selected values into <code>LrbProblemData</code> and regenerate the dashboard.
                </p>
              </div>

              <p class="slider-note">
                The slider ranges are intentionally broad enough for calibration without making the controls unreadable: <code>β</code> and <code>γ</code> vary from 0.00 to 3.00,
                and <code>n</code> varies from 1.00 to 6.00. The value <code>u_y = {uy_value:.2f} mm</code> per bearing
                is estimated from the plotted peak load and the slopes <code>K1=10S</code> and <code>K2=S</code>.
                The reported <code>Q = {derived_q_value:.1f} kN</code> is therefore a derived intercept, not a fitted input; the current reading intentionally places it near 100 kN.
                The plotted hysteresis path starts at the unloaded state <code>(u,F)=(0,0)</code>. The hover information reports the decomposition
                <code>F = alpha K1 u + (1-alpha)K1 uy z</code>, with <code>alpha K1 = K2</code>.
              </p>

              <div id="lrb-cyclic-calibration-interactive" class="calibration-plot"></div>
              <script>
                (function () {{
                  const plotId = "lrb-cyclic-calibration-interactive";
                  const amplitudeMm = {amplitude_mm:.12g};
                  const slopeK2 = {secondary_slope:.12g};
                  const kBoucWen = {initial_tangent:.12g};
                  const alpha = {alpha_value:.12g};
                  const uyMm = {uy_value:.12g};
                  const peakLoadKn = {peak_force_value:.12g};
                  const derivedQKn = (1.0 - alpha) * kBoucWen * uyMm;
                  const omega = 2.0 * Math.PI;
                  const cycles = 3;
                  const pointsPerCycle = 3200;
                  const dt = 1.0 / pointsPerCycle;
                  const nSteps = cycles * pointsPerCycle + 1;
                  const lastStart = (cycles - 1) * pointsPerCycle;

                  const state = {{
                    beta: {beta_value:.12g},
                    gamma: {gamma_value:.12g},
                    exponent: {exponent_value:.12g}
                  }};

                  const time = new Array(nSteps);
                  const displacement = new Array(nSteps);
                  const velocity = new Array(nSteps);
                  for (let i = 0; i < nSteps; i += 1) {{
                    const t = i * dt;
                    time[i] = t;
                    displacement[i] = amplitudeMm * Math.sin(omega * t);
                    velocity[i] = amplitudeMm * omega * Math.cos(omega * t);
                  }}

                  function makeRange(start, stop, step) {{
                    const values = [];
                    const nValues = Math.round((stop - start) / step);
                    for (let i = 0; i <= nValues; i += 1) {{
                      values.push(Number((start + i * step).toFixed(6)));
                    }}
                    return values;
                  }}

                  function nearestIndex(values, selected) {{
                    let idx = 0;
                    let best = Math.abs(values[0] - selected);
                    for (let i = 1; i < values.length; i += 1) {{
                      const error = Math.abs(values[i] - selected);
                      if (error < best) {{
                        idx = i;
                        best = error;
                      }}
                    }}
                    return idx;
                  }}

                  function buildSliderSteps(parameter, values) {{
                    return values.map(function (value) {{
                      return {{
                        method: "skip",
                        label: "",
                        args: [{{parameter: parameter, value: value}}]
                      }};
                    }});
                  }}

                  function computeLoop(beta, gamma, exponent) {{
                    const z = new Array(nSteps).fill(0.0);

                    function dzdt(tValue, zValue) {{
                      const vValue = amplitudeMm * omega * Math.cos(omega * tValue);
                      const absZ = Math.abs(zValue);
                      const term1 = vValue;
                      const term2 = gamma * Math.abs(vValue) * zValue * Math.pow(absZ, exponent - 1.0);
                      const term3 = beta * vValue * Math.pow(absZ, exponent);
                      return (term1 - term2 - term3) / uyMm;
                    }}

                    for (let i = 1; i < nSteps; i += 1) {{
                      const t0 = time[i - 1];
                      const z0 = z[i - 1];
                      const k1 = dzdt(t0, z0);
                      const k2 = dzdt(t0 + 0.5 * dt, z0 + 0.5 * dt * k1);
                      const k3 = dzdt(t0 + 0.5 * dt, z0 + 0.5 * dt * k2);
                      const k4 = dzdt(t0 + dt, z0 + dt * k3);
                      const zNext = z0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
                      z[i] = (Number.isFinite(zNext) && Math.abs(zNext) <= 10.0) ? zNext : NaN;
                    }}

                    function linearAt(i) {{ return alpha * kBoucWen * displacement[i]; }}
                    function hystereticAt(i) {{ return (1.0 - alpha) * kBoucWen * uyMm * z[i]; }}
                    function forceAt(i) {{ return linearAt(i) + hystereticAt(i); }}

                    const x = [];
                    const y = [];
                    const custom = [];
                    for (let i = 0; i < nSteps; i += 1) {{
                      const linear = linearAt(i);
                      const hysteretic = hystereticAt(i);
                      const force = linear + hysteretic;
                      x.push(displacement[i]);
                      y.push(force);
                      custom.push([time[i], velocity[i] / 1000.0, z[i], linear, hysteretic, force]);
                    }}

                    const initialX = [];
                    const initialY = [];
                    const initialCustom = [];
                    const initialEnd = Math.round(0.25 * pointsPerCycle);
                    for (let i = 0; i <= initialEnd; i += 1) {{
                      const linear = linearAt(i);
                      const hysteretic = hystereticAt(i);
                      const force = linear + hysteretic;
                      initialX.push(displacement[i]);
                      initialY.push(force);
                      initialCustom.push([time[i], velocity[i] / 1000.0, z[i], linear, hysteretic, force]);
                    }}

                    const zeroX = [];
                    const zeroY = [];
                    for (let i = lastStart; i < nSteps - 1; i += 1) {{
                      const f0 = forceAt(i);
                      const f1 = forceAt(i + 1);
                      if (!Number.isFinite(f0) || !Number.isFinite(f1)) {{
                        continue;
                      }}
                      if (f0 === 0.0) {{
                        zeroX.push(displacement[i]);
                        zeroY.push(0.0);
                      }} else if (f0 * f1 < 0.0) {{
                        const ratio = -f0 / (f1 - f0);
                        zeroX.push(displacement[i] + ratio * (displacement[i + 1] - displacement[i]));
                        zeroY.push(0.0);
                      }}
                    }}

                    return {{x: x, y: y, custom: custom, initialX: initialX, initialY: initialY, initialCustom: initialCustom, zeroX: zeroX, zeroY: zeroY}};
                  }}

                  function titleText() {{
                    return "Interactive cyclic calibration — β=" + state.beta.toFixed(2)
                      + ", γ=" + state.gamma.toFixed(2)
                      + ", n=" + state.exponent.toFixed(2)
                      + ", uy=" + uyMm.toFixed(2) + " mm, derived Q=" + derivedQKn.toFixed(1) + " kN";
                  }}

                  function betaPrefix() {{ return "β = " + state.beta.toFixed(2) + "  "; }}
                  function gammaPrefix() {{ return "γ = " + state.gamma.toFixed(2) + "  "; }}
                  function exponentPrefix() {{ return "n = " + state.exponent.toFixed(2) + "  "; }}

                  const betaValues = makeRange(0.00, 3.00, 0.025);
                  const gammaValues = makeRange(0.00, 3.00, 0.025);
                  const exponentValues = makeRange(1.00, 6.00, 0.05);

                  const target = computeLoop(0.50, 0.50, 1.00);
                  const model = computeLoop(state.beta, state.gamma, state.exponent);

                  const componentHover =
                    "t: %{{customdata[0]:.4f}} s<br>" +
                    "u: %{{x:.2f}} mm<br>" +
                    "u_dot: %{{customdata[1]:.5f}} m/s<br>" +
                    "z: %{{customdata[2]:.5f}}<br>" +
                    "alpha K1 = K2: " + slopeK2.toFixed(3) + " kN/mm<br>" +
                    "alpha K1 u = K2 u: %{{customdata[3]:.2f}} kN<br>" +
                    "(1-alpha)K1 uy z: %{{customdata[4]:.2f}} kN<br>" +
                    "F: %{{customdata[5]:.2f}} kN<extra></extra>";

                  const traces = [
                    {{
                      x: target.x,
                      y: target.y,
                      customdata: target.custom,
                      mode: "lines",
                      type: "scatter",
                      name: "Reference rounded guide from t = 0",
                      line: {{color: "rgb(0, 70, 140)", width: 3.0, dash: "dash"}},
                      hovertemplate: "Guide<br>" + componentHover
                    }},
                    {{
                      x: model.x,
                      y: model.y,
                      customdata: model.custom,
                      mode: "lines",
                      type: "scatter",
                      name: "Bouc-Wen path from t = 0",
                      line: {{color: "rgb(180, 20, 60)", width: 3.0}},
                      hovertemplate: "Bouc-Wen<br>" + componentHover
                    }},
                    {{
                      x: model.initialX,
                      y: model.initialY,
                      customdata: model.initialCustom,
                      mode: "lines",
                      type: "scatter",
                      name: "First quarter-cycle from rest",
                      line: {{color: "rgb(20, 20, 20)", width: 2.4, dash: "dot"}},
                      hovertemplate: "Initial loading<br>" + componentHover
                    }},
                    {{
                      x: [0.0],
                      y: [0.0],
                      mode: "markers+text",
                      type: "scatter",
                      name: "Initial state (0,0)",
                      text: ["initial state"],
                      textposition: "top right",
                      marker: {{size: 10, color: "rgb(20, 20, 20)"}},
                      hovertemplate: "Initial state<br>t: 0.0000 s<br>u: 0.00 mm<br>F: 0.00 kN<br>z: 0.00000<extra></extra>"
                    }},
                    {{
                      x: model.zeroX,
                      y: model.zeroY,
                      mode: "markers",
                      type: "scatter",
                      name: "Last-cycle F = 0 crossings",
                      marker: {{size: 9, color: "rgb(0, 100, 0)", symbol: "diamond"}},
                      hovertemplate: "Last-cycle zero-force crossing<br>u: %{{x:.2f}} mm<br>F: 0.00 kN<extra></extra>"
                    }}
                  ];

                  const sliderFont = {{size: 13, color: "rgb(0, 50, 98)"}};
                  const sliderLabelFont = {{size: 14, color: "rgb(0, 50, 98)"}};
                  const layout = {{
                    template: "plotly_white",
                    height: 880,
                    title: {{
                      text: titleText(),
                      x: 0.5,
                      y: 1.125,
                      xanchor: "center",
                      yanchor: "top",
                      pad: {{b: 10}},
                      font: {{size: 20}}
                    }},
                    xaxis: {{title: "Displacement u [mm]", range: [-265, 265], zeroline: true}},
                    yaxis: {{title: "Load F [kN]", range: [-360, 360], zeroline: true}},
                    legend: {{orientation: "h", yanchor: "bottom", y: 1.04, xanchor: "center", x: 0.5}},
                    margin: {{l: 70, r: 35, t: 220, b: 290}},
                    font: {{size: 14}},
                    annotations: [
                      {{
                        text: "β",
                        x: 0.105,
                        y: -0.070,
                        xref: "paper",
                        yref: "paper",
                        showarrow: false,
                        xanchor: "right",
                        yanchor: "middle",
                        font: sliderLabelFont
                      }},
                      {{
                        text: "γ",
                        x: 0.105,
                        y: -0.235,
                        xref: "paper",
                        yref: "paper",
                        showarrow: false,
                        xanchor: "right",
                        yanchor: "middle",
                        font: sliderLabelFont
                      }},
                      {{
                        text: "n",
                        x: 0.105,
                        y: -0.400,
                        xref: "paper",
                        yref: "paper",
                        showarrow: false,
                        xanchor: "right",
                        yanchor: "middle",
                        font: sliderLabelFont
                      }}
                    ],
                    sliders: [
                      {{
                        active: nearestIndex(betaValues, state.beta),
                        x: 0.14,
                        y: -0.08,
                        len: 0.78,
                        xanchor: "left",
                        yanchor: "top",
                        pad: {{t: 18, b: 6}},
                        currentvalue: {{prefix: betaPrefix(), visible: true, xanchor: "right", offset: 12, font: sliderFont}},
                        steps: buildSliderSteps("beta", betaValues)
                      }},
                      {{
                        active: nearestIndex(gammaValues, state.gamma),
                        x: 0.14,
                        y: -0.245,
                        len: 0.78,
                        xanchor: "left",
                        yanchor: "top",
                        pad: {{t: 18, b: 6}},
                        currentvalue: {{prefix: gammaPrefix(), visible: true, xanchor: "right", offset: 12, font: sliderFont}},
                        steps: buildSliderSteps("gamma", gammaValues)
                      }},
                      {{
                        active: nearestIndex(exponentValues, state.exponent),
                        x: 0.14,
                        y: -0.41,
                        len: 0.78,
                        xanchor: "left",
                        yanchor: "top",
                        pad: {{t: 18, b: 6}},
                        currentvalue: {{prefix: exponentPrefix(), visible: true, xanchor: "right", offset: 12, font: sliderFont}},
                        steps: buildSliderSteps("exponent", exponentValues)
                      }}
                    ]
                  }};

                  function updateCurve() {{
                    const updated = computeLoop(state.beta, state.gamma, state.exponent);
                    Plotly.restyle(plotId, {{x: [updated.x], y: [updated.y], customdata: [updated.custom]}}, [1]);
                    Plotly.restyle(plotId, {{x: [updated.initialX], y: [updated.initialY], customdata: [updated.initialCustom]}}, [2]);
                    Plotly.restyle(plotId, {{x: [updated.zeroX], y: [updated.zeroY]}}, [4]);
                    Plotly.relayout(plotId, {{
                      "title.text": titleText(),
                      "sliders[0].currentvalue.prefix": betaPrefix(),
                      "sliders[1].currentvalue.prefix": gammaPrefix(),
                      "sliders[2].currentvalue.prefix": exponentPrefix()
                    }});
                  }}

                  Plotly.newPlot(plotId, traces, layout, {{displayModeBar: true, responsive: true}}).then(function (plotDiv) {{
                    plotDiv.on("plotly_sliderchange", function (eventData) {{
                      if (!eventData || !eventData.step || !eventData.step.args) {{
                        return;
                      }}
                      const payload = eventData.step.args[0];
                      if (!payload || payload.parameter === undefined) {{
                        return;
                      }}
                      state[payload.parameter] = Number(payload.value);
                      updateCurve();
                    }});
                  }});
                }})();
              </script>
            </div>
            """
        ).strip()




    @staticmethod
    def time_history_5panel(
        title: str,
        result: TimeHistoryResult,
        total_weight: float,
        parameters: BoucWenLrbParameters | None = None,
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

        z_values: np.ndarray | None = None
        linear_mn: np.ndarray | None = None
        hysteretic_mn: np.ndarray | None = None
        if parameters is not None and result.hysteretic_parameter is not None:
            z_values = np.asarray(result.hysteretic_parameter, dtype=float)
            linear_mn = parameters.alpha * parameters.total_corner_tangent * result.displacement / 1.0e6
            hysteretic_mn = (
                (1.0 - parameters.alpha)
                * parameters.total_corner_tangent
                * parameters.yield_displacement
                * z_values
            ) / 1.0e6

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

        has_components = z_values is not None and linear_mn is not None and hysteretic_mn is not None
        if has_components:
            customdata = np.column_stack((disp_mm, vel_mps, force_over_weight, force_mn, rel_acc_g, abs_acc_g, ground_g, z_values, linear_mn, hysteretic_mn))
            component_hover = (
                "z: %{customdata[7]:.5f}<br>"
                "alpha K1 u = K2 u: %{customdata[8]:.4f} MN<br>"
                "(1-alpha)K1 uy z: %{customdata[9]:.4f} MN<br>"
            )
        else:
            customdata = np.column_stack((disp_mm, vel_mps, force_over_weight, force_mn, rel_acc_g, abs_acc_g, ground_g))
            component_hover = ""

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
                        "u_dot: %{customdata[1]:.4f} m/s<br>"
                        "F/W: %{customdata[2]:.5f}<br>"
                        "F: %{customdata[3]:.4f} MN<br>"
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
        parameters: BoucWenLrbParameters | None = None,
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
            linear_mn = parameters.alpha * parameters.total_corner_tangent * result.displacement / 1.0e6
            hysteretic_mn = (
                (1.0 - parameters.alpha)
                * parameters.total_corner_tangent
                * parameters.yield_displacement
                * z_values
            ) / 1.0e6
            customdata = np.column_stack((time, velocity, rel_acc_g, abs_acc_g, ground_g, force_mn, z_values, linear_mn, hysteretic_mn))
            hovertemplate = (
                "Displacement u: %{x:.3f} mm<br>"
                "Normalized restoring force F/W: %{y:.5f}<br>"
                "Restoring force F: %{customdata[5]:.4f} MN<br>"
                "Time: %{customdata[0]:.3f} s<br>"
                "Velocity u_dot: %{customdata[1]:.4f} m/s<br>"
                "Relative acceleration a_rel: %{customdata[2]:.4f} g<br>"
                "Absolute acceleration a_abs: %{customdata[3]:.4f} g<br>"
                "Ground acceleration a_g: %{customdata[4]:.4f} g<br>"
                "z: %{customdata[6]:.5f}<br>"
                "alpha K1 u = K2 u: %{customdata[7]:.4f} MN<br>"
                "(1-alpha)K1 uy z: %{customdata[8]:.4f} MN<extra></extra>"
            )
        else:
            customdata = np.column_stack((time, velocity, rel_acc_g, abs_acc_g, ground_g, force_mn))
            hovertemplate = (
                "Displacement u: %{x:.3f} mm<br>"
                "Normalized restoring force F/W: %{y:.5f}<br>"
                "Restoring force F: %{customdata[5]:.4f} MN<br>"
                "Time: %{customdata[0]:.3f} s<br>"
                "Velocity u_dot: %{customdata[1]:.4f} m/s<br>"
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
            height=470,
            title=dict(text=title, x=0.5, xanchor="center", font=dict(size=22)),
            xaxis=dict(title="Displacement u [mm]", title_font=dict(size=16), tickfont=dict(size=13), zeroline=True),
            yaxis=dict(title="Normalized Force F/W [-]", title_font=dict(size=16), tickfont=dict(size=13), zeroline=True),
            font=dict(size=14),
        )
        return fig




    @staticmethod
    def hysteresis_comparison(
        title: str,
        nonlinear_result: TimeHistoryResult,
        equivalent_result: TimeHistoryResult,
        total_weight: float,
        parameters: BoucWenLrbParameters,
    ) -> go.Figure:
        """Compare nonlinear and equivalent-linear force-displacement loops."""
        fig = go.Figure()

        z_values = np.asarray(nonlinear_result.hysteretic_parameter, dtype=float)
        nonlinear_linear_mn = parameters.alpha * parameters.total_corner_tangent * nonlinear_result.displacement / 1.0e6
        nonlinear_hysteretic_mn = (
            (1.0 - parameters.alpha)
            * parameters.total_corner_tangent
            * parameters.yield_displacement
            * z_values
        ) / 1.0e6
        nonlinear_custom = np.column_stack((
            nonlinear_result.time,
            nonlinear_result.velocity,
            nonlinear_result.relative_acceleration / G_SI,
            nonlinear_result.absolute_acceleration / G_SI,
            nonlinear_result.ground_acceleration / G_SI,
            nonlinear_result.restoring_force / 1.0e6,
            z_values,
            nonlinear_linear_mn,
            nonlinear_hysteretic_mn,
        ))
        fig.add_trace(
            go.Scatter(
                x=nonlinear_result.displacement * 1.0e3,
                y=nonlinear_result.restoring_force / total_weight,
                mode="lines",
                line=dict(color=MATLAB_COLORS["crimson"], width=2.4),
                name="Nonlinear Bouc-Wen",
                customdata=nonlinear_custom,
                hovertemplate=(
                    "Model: Nonlinear Bouc-Wen<br>"
                    "u: %{x:.3f} mm<br>"
                    "F/W: %{y:.5f}<br>"
                    "F: %{customdata[5]:.4f} MN<br>"
                    "t: %{customdata[0]:.3f} s<br>"
                    "u_dot: %{customdata[1]:.4f} m/s<br>"
                    "a_rel: %{customdata[2]:.4f} g<br>"
                    "a_abs: %{customdata[3]:.4f} g<br>"
                    "a_g: %{customdata[4]:.4f} g<br>"
                    "z: %{customdata[6]:.5f}<br>"
                    "alpha K1 u = K2 u: %{customdata[7]:.4f} MN<br>"
                    "(1-alpha)K1 uy z: %{customdata[8]:.4f} MN<extra></extra>"
                ),
            )
        )

        equivalent_custom = np.column_stack((
            equivalent_result.time,
            equivalent_result.velocity,
            equivalent_result.relative_acceleration / G_SI,
            equivalent_result.absolute_acceleration / G_SI,
            equivalent_result.ground_acceleration / G_SI,
            equivalent_result.restoring_force / 1.0e6,
        ))
        fig.add_trace(
            go.Scatter(
                x=equivalent_result.displacement * 1.0e3,
                y=equivalent_result.restoring_force / total_weight,
                mode="lines",
                line=dict(color=MATLAB_COLORS["black"], width=2.4, dash="dash"),
                name="Equivalent linear",
                customdata=equivalent_custom,
                hovertemplate=(
                    "Model: Equivalent linear<br>"
                    "u: %{x:.3f} mm<br>"
                    "F/W: %{y:.5f}<br>"
                    "F: %{customdata[5]:.4f} MN<br>"
                    "t: %{customdata[0]:.3f} s<br>"
                    "u_dot: %{customdata[1]:.4f} m/s<br>"
                    "a_rel: %{customdata[2]:.4f} g<br>"
                    "a_abs: %{customdata[3]:.4f} g<br>"
                    "a_g: %{customdata[4]:.4f} g<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            template="plotly_white",
            height=500,
            title=dict(text=title, x=0.5, xanchor="center", font=dict(size=22)),
            xaxis=dict(title="Displacement u [mm]", title_font=dict(size=16), tickfont=dict(size=13), zeroline=True),
            yaxis=dict(title="Normalized Force F/W [-]", title_font=dict(size=16), tickfont=dict(size=13), zeroline=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            font=dict(size=14),
        )
        return fig




class HtmlReportBuilder:
    """Assemble the complete self-contained HTML dashboard."""
    def __init__(self, bw_params: BoucWenLrbParameters, eq_props: EquivalentLinearProperties, data: LrbProblemData) -> None:
        self.bw_params = bw_params
        self.eq_props = eq_props
        self.data = data

    @staticmethod
    def fig_to_div(fig: go.Figure, include_js: bool = False) -> str:
        return to_html(fig, include_plotlyjs=include_js, full_html=False, config=dict(displayModeBar=True, responsive=True))

    def build(
        self,
        cyclic_result: CyclicTestResult,
        calibration_cases: list[CalibrationCaseResult],
        bw_results: dict[str, TimeHistoryResult],
        equivalent_results: dict[str, TimeHistoryResult],
    ) -> str:
        """Build all figures, tables, and explanatory sections for the report."""
        fig_cyclic = FigureFactory.cyclic_hysteresis(cyclic_result, self.bw_params)
        fig_cyclic_z = FigureFactory.cyclic_z_history(cyclic_result, self.bw_params)
        fig_kobe_bw = FigureFactory.time_history_5panel("Kobe — Nonlinear Bouc-Wen LRB Response", bw_results["Kobe"], self.data.total_weight, self.bw_params)
        fig_kobe_bw_h = FigureFactory.hysteresis("Kobe — Nonlinear Bouc-Wen Hysteresis F/W-u", bw_results["Kobe"], self.data.total_weight, self.bw_params)
        fig_kobe_bw_z = FigureFactory.z_time_history("Kobe — Nonlinear Bouc-Wen Hysteretic Parameter z(t)", bw_results["Kobe"], self.bw_params)
        fig_sylmar_bw = FigureFactory.time_history_5panel("Sylmar — Nonlinear Bouc-Wen LRB Response", bw_results["Sylmar"], self.data.total_weight, self.bw_params)
        fig_sylmar_bw_h = FigureFactory.hysteresis("Sylmar — Nonlinear Bouc-Wen Hysteresis F/W-u", bw_results["Sylmar"], self.data.total_weight, self.bw_params)
        fig_sylmar_bw_z = FigureFactory.z_time_history("Sylmar — Nonlinear Bouc-Wen Hysteretic Parameter z(t)", bw_results["Sylmar"], self.bw_params)

        fig_kobe_eq = FigureFactory.time_history_5panel("Kobe — Equivalent Linear Response", equivalent_results["Kobe"], self.data.total_weight)
        fig_kobe_eq_h = FigureFactory.hysteresis("Kobe — Equivalent Linear F/W-u", equivalent_results["Kobe"], self.data.total_weight)
        fig_sylmar_eq = FigureFactory.time_history_5panel("Sylmar — Equivalent Linear Response", equivalent_results["Sylmar"], self.data.total_weight)
        fig_sylmar_eq_h = FigureFactory.hysteresis("Sylmar — Equivalent Linear F/W-u", equivalent_results["Sylmar"], self.data.total_weight)

        fig_kobe_compare = FigureFactory.hysteresis_comparison(
            "Kobe — Nonlinear versus Equivalent Linear Hysteresis",
            bw_results["Kobe"],
            equivalent_results["Kobe"],
            self.data.total_weight,
            self.bw_params,
        )
        fig_sylmar_compare = FigureFactory.hysteresis_comparison(
            "Sylmar — Nonlinear versus Equivalent Linear Hysteresis",
            bw_results["Sylmar"],
            equivalent_results["Sylmar"],
            self.data.total_weight,
            self.bw_params,
        )

        sections = [
            self.fig_to_div(fig_cyclic, include_js=False),
            self.fig_to_div(fig_cyclic_z, include_js=False),
            self.fig_to_div(fig_kobe_bw, include_js=False),
            self.fig_to_div(fig_kobe_bw_h, include_js=False),
            self.fig_to_div(fig_kobe_bw_z, include_js=False),
            self.fig_to_div(fig_sylmar_bw, include_js=False),
            self.fig_to_div(fig_sylmar_bw_h, include_js=False),
            self.fig_to_div(fig_sylmar_bw_z, include_js=False),
            self.fig_to_div(fig_kobe_eq, include_js=False),
            self.fig_to_div(fig_kobe_eq_h, include_js=False),
            self.fig_to_div(fig_sylmar_eq, include_js=False),
            self.fig_to_div(fig_sylmar_eq_h, include_js=False),
            self.fig_to_div(fig_kobe_compare, include_js=False),
            self.fig_to_div(fig_sylmar_compare, include_js=False),
        ]

        parameter_table = self._build_parameter_table()
        calibration_table = self._build_calibration_table(calibration_cases)
        equivalent_table = self._build_equivalent_linear_table()
        peak_table = self._build_peak_table(bw_results, equivalent_results, self.data.total_weight)
        interactive_calibration = FigureFactory.interactive_cyclic_calibration_html(self.bw_params, self.data)

        return dedent(
            f"""
            <!DOCTYPE HTML>
            <html lang="en">
            <head>
              <meta charset="utf-8" />
              <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
              <title>CE223 – LRB Bouc-Wen Dashboard</title>
              <link rel="stylesheet" href="../../assets/css/main.css" />
              <noscript><link rel="stylesheet" href="../../assets/css/noscript.css" /></noscript>
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
                .eq {{
                    background: #f9fafb;
                    border-left: 4px solid #003262;
                    padding: 0.9rem 1rem;
                    margin: 0.75rem 0;
                    overflow-x: auto;
                    -webkit-overflow-scrolling: touch;
                }}
                .summary-table-wrap {{
                    overflow-x: auto;
                    -webkit-overflow-scrolling: touch;
                    margin: 1rem 0;
                }}
                .summary-table {{
                    width: 100%;
                    min-width: 36rem;
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
                .interactive-calibration-panel {{
                    border: 1px solid #e5e7eb;
                    border-radius: 8px;
                    background: #fff;
                    padding: 1rem;
                    margin: 1rem 0;
                }}
                .interactive-calibration-header h3 {{
                    font-size: 1.15rem;
                    font-weight: 700;
                    color: #003262;
                    margin: 0 0 0.35rem 0;
                    border-left: 4px solid #FDB515;
                    padding-left: 0.75rem;
                }}
                .interactive-calibration-header p,
                .slider-note {{
                    color: #374151;
                    margin: 0.4rem 0 0.8rem 0;
                }}
                .calibration-control-grid {{
                    display: grid;
                    grid-template-columns: repeat(3, minmax(12rem, 1fr));
                    gap: 1rem;
                    margin: 1rem 0 0.85rem 0;
                }}
                .calibration-control {{
                    display: grid;
                    grid-template-columns: auto 1fr auto;
                    gap: 0.75rem;
                    align-items: center;
                    background: #f9fafb;
                    border: 1px solid #e5e7eb;
                    border-radius: 8px;
                    padding: 0.75rem;
                }}
                .control-title {{
                    font-weight: 700;
                    color: #003262;
                    min-width: 1.6rem;
                }}
                .control-value {{
                    font-variant-numeric: tabular-nums;
                    font-weight: 700;
                    color: #111827;
                    min-width: 3.2rem;
                    text-align: right;
                }}
                .calibration-control input[type="range"] {{
                    width: 100%;
                    accent-color: #003262;
                }}
                .calibration-plot {{
                    width: 100%;
                    height: 800px;
                }}
                .inner-report .js-plotly-plot .modebar {{
                    top: 56px !important;
                }}
                @media (max-width: 736px) {{
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
                    .summary-table {{
                        font-size: 0.85rem;
                    }}
                    .summary-table thead th, .summary-table tbody td {{
                        padding: 0.4rem 0.35rem;
                    }}
                    .calibration-control-grid {{
                        grid-template-columns: 1fr;
                    }}
                    .calibration-control {{
                        grid-template-columns: auto 1fr auto;
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
                      <h2>CE223 – Lead Rubber Bearing System with Bouc-Wen Model</h2>
                      <p class="summary-lead">
                        This dashboard calibrates a Bouc-Wen model for one lead rubber bearing and then scales the calibrated bearing model to a rigid mass supported by 200 identical bearings. The dynamic response is computed under the Kobe and Sylmar records and compared with an equivalent viscously damped linear oscillator.
                      </p>
                    </header>

                    <section class="box">
                      <h3>Model definition</h3>
                      <p>The recorded loop is represented by the Bouc-Wen restoring force:</p>
                      <div class="eq">$$F(t)=\\alpha K_{{\\mathrm{{BW}}}}u(t)+(1-\\alpha)K_{{\\mathrm{{BW}}}}u_y z(t),\\qquad \\alpha K_{{\\mathrm{{BW}}}}=K_2=S.$$</div>
                      <p>The hysteretic parameter is governed by:</p>
                      <div class="eq">$$\\dot z(t)=\\frac{{1}}{{u_y}}\\left[\\dot u(t)-\\gamma |\\dot u(t)|z(t)|z(t)|^{{n-1}}-\\beta \\dot u(t)|z(t)|^n\\right].$$</div>
                      <p>The test plot is interpreted with $K_2=S=0.92$ kN/mm for the secondary branch of the recorded loop. The steep tangent near the rounded corners is taken as $K_1=10S=9.20$ kN/mm. In the Bouc--Wen force law used here, $K_{{\\mathrm{{BW}}}}=K_1$, so that $\\alpha K_{{\\mathrm{{BW}}}}=K_2$ and $\\alpha=K_2/K_1=S/(10S)=0.10$. The transition displacement is estimated from the plotted peak point by $u_y=(F_{{\\max}}-K_2U)/(K_1-K_2)$. The intercept force $Q=(K_1-K_2)u_y$ is then reported as a derived quantity, not prescribed directly.</p>
                      {parameter_table}
                    </section>

                    <section class="report-section">
                      <h2>Part (a) — Cyclic calibration</h2>
                      <p>The prescribed displacement is $u(t)=0.235\\sin(2\\pi t)$ m. The main calibration figure now separates two different ideas that were previously hidden. The crimson and dashed blue curves show the stabilized loop from the last cycle, while the dotted black curve shows the initial loading branch from the undeformed state. Therefore, the point $(u,F)=(0,0)$ appears explicitly as the initial state.</p>
                      <p>The stabilized loop is not expected to pass through the origin. Once $z$ has saturated, the force at $u=0$ is approximately $F=\\pm(1-\\alpha)K_1u_y$, so the zero-force crossings occur at nonzero displacements. The green markers identify those $F=0$ crossings on the stabilized loop.</p>
                      <p>The calibration is performed in two stages. First, the force scale and slopes are fixed from the test loop: $K_2=S$, $K_1=10S$, $K_{{\\mathrm{{BW}}}}=K_1$, $\\alpha=K_2/K_1$, and $(1-\\alpha)K_{{\\mathrm{{BW}}}}u_y=Q$. The transition displacement $u_y$ is estimated from the upper-branch point near the maximum displacement, giving a derived $Q$ close to 100 kN per bearing. Second, $\\beta$, $\\gamma$, and $n$ are adjusted to reproduce the overall loop shape. Keeping $\\beta+\\gamma=1$ keeps the saturation level near $|z|=1$; the selected exercise value $\\beta=\\gamma=0.50$ is used with $n=1$.</p>
                      {calibration_table}
                      {interactive_calibration}
                      <div class="plot-embed">{sections[0]}</div>
                    </section>

                    <section class="report-section">
                      <h2>Part (b) — Nonlinear Bouc-Wen time-history results</h2>
                      <p>For each ground motion, the five requested histories are shown first, followed by the nonlinear $F/W$ versus displacement loop.</p>
                      <div class="plot-embed">{sections[1]}</div>
                      <div class="plot-embed">{sections[2]}</div>
                      <div class="plot-embed">{sections[3]}</div>
                      <div class="plot-embed">{sections[4]}</div>
                    </section>

                    <section class="report-section">
                      <h2>Part (c) — Equivalent viscously damped linear oscillator</h2>
                      <p>The equivalent linear system is calibrated from the final cyclic Bouc-Wen loop at $U=0.235$ m. The effective stiffness is taken as $k_{{eff}}=F_{{max}}/U$, and the equivalent viscous damping ratio is obtained by matching the loop energy: $\\xi_{{eq}}=E_D/(4\\pi E_{{S,0}})$.</p>
                      {equivalent_table}
                      {peak_table}
                      <div class="plot-embed">{sections[5]}</div>
                      <div class="plot-embed">{sections[6]}</div>
                      <div class="plot-embed">{sections[7]}</div>
                      <div class="plot-embed">{sections[8]}</div>
                      <div class="plot-embed">{sections[9]}</div>
                      <div class="plot-embed">{sections[10]}</div>
                    </section>
                  </div>
                </section>
              </div>
            </body>
            </html>
            """
        ).strip()

    def _build_parameter_table(self) -> str:
        p = self.bw_params
        rows = [
            ("Total mass", f"{self.data.total_mass:.3e}", "kg"),
            ("Total weight", f"{self.data.total_weight / 1.0e6:.4f}", "MN"),
            ("Number of bearings", f"{self.data.n_bearings:d}", "-"),
            ("Cyclic displacement amplitude", f"{self.data.cyclic_amplitude * 1.0e3:.3f}", "mm"),
            ("Secondary slope K2 = S per bearing", f"{p.bearing_test_slope / 1.0e6:.4f}", "kN/mm"),
            ("Initial tangent K1 = 10S per bearing", f"{p.bearing_corner_tangent / 1.0e6:.4f}", "kN/mm"),
            ("Alpha = K2/K1 = S/(10S)", f"{p.alpha:.4f}", "-"),
            ("Derived intercept Q = (K1 - K2) uy per bearing", f"{p.bearing_characteristic_strength / 1.0e3:.3f}", "kN"),
            ("Yield displacement uy", f"{p.yield_displacement * 1.0e3:.4f}", "mm"),
            ("Bouc-Wen beta", f"{p.beta_bw:.4f}", "-"),
            ("Bouc-Wen gamma", f"{p.gamma_bw:.4f}", "-"),
            ("Bouc-Wen n", f"{p.exponent_n:.2f}", "-"),
            ("Total initial tangent K1 = 10S", f"{p.total_corner_tangent / 1.0e9:.4f}", "GN/m"),
            ("Total secondary slope K2 = S", f"{p.total_test_slope / 1.0e6:.4f}", "MN/m"),
            ("Total derived Q", f"{p.total_characteristic_strength / 1.0e6:.4f}", "MN"),
        ]
        body = "".join(f"<tr><td>{name}</td><td>{value}</td><td>{unit}</td></tr>" for name, value, unit in rows)
        return (
            '<div class="summary-table-wrap"><table class="summary-table"><thead><tr>'
            "<th scope='col'>Quantity</th><th scope='col'>Value</th><th scope='col'>Unit</th>"
            f"</tr></thead><tbody>{body}</tbody></table></div>"
        )

    @staticmethod
    def _build_calibration_table(calibration_cases: list[CalibrationCaseResult]) -> str:
        rows = []
        for case in calibration_cases:
            result = case.cyclic_result
            mask = result.last_cycle_mask
            error = result.bouc_wen_force[mask] - result.target_force[mask]
            max_error = float(np.max(np.abs(error))) / 1.0e3
            rms_error = float(math.sqrt(np.mean(error * error))) / 1.0e3
            area_bw = abs(integrate_trapezoid(result.bouc_wen_force[mask], result.displacement[mask]))
            area_target = abs(integrate_trapezoid(result.target_force[mask], result.displacement[mask]))
            area_error = 100.0 * (area_bw - area_target) / max(area_target, 1.0e-14)
            rows.append(
                "<tr>"
                f"<td>{case.label}</td>"
                f"<td>{case.parameters.beta_bw:.3f}</td>"
                f"<td>{case.parameters.gamma_bw:.3f}</td>"
                f"<td>{case.parameters.exponent_n:.1f}</td>"
                f"<td>{max_error:.3f}</td>"
                f"<td>{rms_error:.3f}</td>"
                f"<td>{area_error:+.2f}</td>"
                "</tr>"
            )
        return (
            '<div class="summary-table-wrap"><table class="summary-table"><thead><tr>'
            "<th scope='col'>Case</th><th scope='col'>β</th><th scope='col'>γ</th><th scope='col'>n</th>"
            "<th scope='col'>Max |ΔF| to guide [kN]</th><th scope='col'>RMS |ΔF| to guide [kN]</th>"
            "<th scope='col'>Guide area error [%]</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )

    def _build_equivalent_linear_table(self) -> str:
        e = self.eq_props
        rows = [
            ("Effective amplitude", f"{e.amplitude * 1.0e3:.3f}", "mm"),
            ("Effective stiffness", f"{e.stiffness / 1.0e6:.4f}", "MN/m"),
            ("Equivalent period", f"{e.period:.4f}", "s"),
            ("Equivalent circular frequency", f"{e.circular_frequency:.4f}", "rad/s"),
            ("Equivalent damping ratio", f"{100.0 * e.damping_ratio:.2f}", "%"),
            ("Equivalent viscous coefficient", f"{e.damping / 1.0e6:.4f}", "MN s/m"),
            ("Dissipated energy per cycle", f"{e.dissipated_energy / 1.0e6:.4f}", "MJ"),
            ("Stored strain energy", f"{e.stored_energy / 1.0e6:.4f}", "MJ"),
        ]
        body = "".join(f"<tr><td>{name}</td><td>{value}</td><td>{unit}</td></tr>" for name, value, unit in rows)
        return (
            '<div class="summary-table-wrap"><table class="summary-table"><thead><tr>'
            "<th scope='col'>Quantity</th><th scope='col'>Value</th><th scope='col'>Unit</th>"
            f"</tr></thead><tbody>{body}</tbody></table></div>"
        )

    @staticmethod
    def _build_peak_table(
        bw_results: dict[str, TimeHistoryResult],
        equivalent_results: dict[str, TimeHistoryResult],
        total_weight: float,
    ) -> str:
        rows = []
        for motion in ("Kobe", "Sylmar"):
            for model_name, result in (("Nonlinear Bouc-Wen", bw_results[motion]), ("Equivalent Linear", equivalent_results[motion])):
                rows.append(
                    "<tr>"
                    f"<td>{motion}</td>"
                    f"<td>{model_name}</td>"
                    f"<td>{result.peak_displacement * 1.0e3:.3f}</td>"
                    f"<td>{result.peak_velocity:.4f}</td>"
                    f"<td>{result.peak_force / 1.0e6:.4f}</td>"
                    f"<td>{result.peak_force / total_weight:.5f}</td>"
                    f"<td>{result.peak_abs_acc / G_SI:.4f}</td>"
                    "</tr>"
                )
        return (
            '<div class="summary-table-wrap"><table class="summary-table"><thead><tr>'
            "<th scope='col'>Motion</th><th scope='col'>Model</th><th scope='col'>Peak |u| [mm]</th>"
            "<th scope='col'>Peak |u̇| [m/s]</th><th scope='col'>Peak |F| [MN]</th>"
            "<th scope='col'>Peak |F/W| [-]</th><th scope='col'>Peak |ü_t| [g]</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )


def find_zero_force_crossings(displacement: np.ndarray, force: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return linearly interpolated coordinates where a force-displacement curve crosses F=0."""
    displacement = np.asarray(displacement, dtype=float).ravel()
    force = np.asarray(force, dtype=float).ravel()
    if displacement.size != force.size:
        raise ValueError("displacement and force must have the same length.")

    zero_displacements: list[float] = []
    zero_forces: list[float] = []
    for i in range(force.size - 1):
        f0 = float(force[i])
        f1 = float(force[i + 1])
        if not math.isfinite(f0) or not math.isfinite(f1):
            continue
        if f0 == 0.0:
            zero_displacements.append(float(displacement[i]))
            zero_forces.append(0.0)
        elif f0 * f1 < 0.0:
            ratio = -f0 / (f1 - f0)
            x_zero = float(displacement[i] + ratio * (displacement[i + 1] - displacement[i]))
            zero_displacements.append(x_zero)
            zero_forces.append(0.0)

    return np.asarray(zero_displacements, dtype=float), np.asarray(zero_forces, dtype=float)


def sign_with_memory(values: np.ndarray) -> np.ndarray:
    """Return signs while keeping the previous sign at exact zero crossings."""
    signs = np.sign(np.asarray(values, dtype=float))
    previous = 1.0
    for i, value in enumerate(signs):
        if value == 0.0:
            signs[i] = previous
        else:
            previous = value
    return signs


def integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate y with respect to x using the NumPy trapezoidal rule."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def main() -> None:
    """Generate the LRB Bouc-Wen dashboard HTML file."""
    problem_data = LrbProblemData()

    calibration_specs = [
        (
            (
                "Selected: "
                f"β={problem_data.reference_beta_bw:.2f}, "
                f"γ={problem_data.reference_gamma_bw:.2f}, "
                f"n={problem_data.reference_exponent_n:.2f}"
            ),
            problem_data.reference_beta_bw,
            problem_data.reference_gamma_bw,
            problem_data.reference_exponent_n,
        ),
    ]
    calibration_cases: list[CalibrationCaseResult] = []
    for label, beta_bw, gamma_bw, exponent_n in calibration_specs:
        params = LrbParameterBuilder.bouc_wen_from_problem_data(
            problem_data,
            beta_bw=beta_bw,
            gamma_bw=gamma_bw,
            exponent_n=exponent_n,
        )
        model = BoucWenForceModel(params)
        cyclic = BoucWenCyclicSolver(model).solve(
            amplitude=problem_data.cyclic_amplitude,
            frequency_hz=problem_data.cyclic_frequency_hz,
            n_cycles=3,
        )
        calibration_cases.append(CalibrationCaseResult(label=label, parameters=params, cyclic_result=cyclic))

    bw_params = calibration_cases[-1].parameters
    bw_model = BoucWenForceModel(bw_params)
    cyclic_result = calibration_cases[-1].cyclic_result
    equivalent_properties = EquivalentLinearBuilder.from_cyclic_result(cyclic_result, bw_params)

    records = {
        "Kobe": GroundMotionLoader.load_acceleration_file(KOBE_PATH, name="Kobe"),
        "Sylmar": GroundMotionLoader.load_acceleration_file(SYLMAR_PATH, name="Sylmar"),
    }

    bw_solver = BoucWenDynamicSolver(bw_model, max_internal_dt=2.0e-3)
    bw_results: dict[str, TimeHistoryResult] = {}
    equivalent_results: dict[str, TimeHistoryResult] = {}

    for motion_name, record in records.items():
        bw_results[motion_name] = bw_solver.solve(record)
        equivalent_results[motion_name] = LinearNewmarkSolver.solve_sdof_base_excitation(
            record=record,
            mass=equivalent_properties.mass,
            damping=equivalent_properties.damping,
            stiffness=equivalent_properties.stiffness,
        )

    report = HtmlReportBuilder(bw_params, equivalent_properties, problem_data).build(
        cyclic_result=cyclic_result,
        calibration_cases=calibration_cases,
        bw_results=bw_results,
        equivalent_results=equivalent_results,
    )
    OUTPUT_HTML.write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
