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


def make_output_directory() -> Path:
    """Return a writable dashboard directory."""
    candidates = (
        CE223_DIR / "highlighted_htmls",
        BASE_DIR / "highlighted_htmls",
        BASE_DIR / "fvd_dashboard_story_profile_output",
    )
    for directory in candidates:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            test_file = directory / ".write_test"
            test_file.write_text("", encoding="utf-8")
            test_file.unlink(missing_ok=True)
            return directory
        except OSError:
            continue
    raise PermissionError("No writable highlighted_htmls directory could be created.")


HIGHLIGHTED_HTML_DIR = make_output_directory()

SYLMAR_CANDIDATES = (
    BASE_DIR / "SYLMAR360.txt",
    CE223_DIR / "input_ground_motion" / "SYLMAR360.txt",
    Path.cwd() / "SYLMAR360.txt",
)
OUTPUT_HTML = HIGHLIGHTED_HTML_DIR / "CE223_FVD_Retrofit_Sylmar.html"

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
class FvdProblemData:
    """Input data for the three-story FVD retrofit calculation.

    Geometry, gravity loading, frame properties, Rayleigh damping targets, and
    FVD design controls are stored in one place. The FVD damping target follows
    the convention used in the damper-design procedure: it is the supplemental
    damping ratio contributed by the FVD system alone, not the total damping
    ratio after adding Rayleigh damping.
    """
    young_modulus: float = 200.0e9
    column_second_moment_area: float = 9.115e-4
    story_height: float = 3.7

    n_moment_frames: int = 2
    columns_per_frame: int = 5

    n_grid_bays: int = 4
    n_lettered_bays: int = 4
    grid_bay_width: float = 10.0
    lettered_bay_width: float = 8.0

    lower_floor_pressure: float = 7.0e3
    roof_pressure: float = 5.6e3

    rayleigh_damping_ratio: float = 0.05
    rayleigh_modes: tuple[int, int] = (1, 3)

    supplemental_fvd_first_mode_damping_ratio: float = 0.35
    dampers_per_story: int = 4
    damper_horizontal_projection_factor: float = 1.0

    # Initial trial for the fictitious-braced iteration. A value of 0.50 means
    # the first trial fictitious story stiffness is 50% of the original story
    # stiffness at each story. The final result is insensitive to this value as
    # long as the iteration converges.
    initial_fictitious_story_stiffness_scale: float = 0.50

    braced_period_relative_tolerance: float = 1.0e-10
    max_fictitious_braced_iterations: int = 50

    @property
    def n_stories(self) -> int:
        return 3

    @property
    def floor_area(self) -> float:
        return (
                self.n_grid_bays * self.grid_bay_width
                * self.n_lettered_bays * self.lettered_bay_width
        )

    @property
    def n_lateral_columns_per_story(self) -> int:
        return self.n_moment_frames * self.columns_per_frame


@dataclass(frozen=True)
class ModalProperties:
    omega: np.ndarray
    periods: np.ndarray
    modes: np.ndarray
    generalized_mass: np.ndarray
    generalized_stiffness: np.ndarray
    participation_factors: np.ndarray
    effective_modal_mass: np.ndarray


@dataclass(frozen=True)
class StructuralMatrices:
    mass: np.ndarray
    stiffness: np.ndarray
    rayleigh_damping: np.ndarray
    story_stiffness: np.ndarray
    story_height: np.ndarray
    rayleigh_mass_coefficient: float
    rayleigh_stiffness_coefficient: float
    original_modal_damping: np.ndarray


@dataclass(frozen=True)
class FictitiousIterationRow:
    """One iteration of the auxiliary braced-system period matching loop."""
    iteration: int
    trial_story_stiffness: np.ndarray
    trial_period: float
    period_error: float


@dataclass(frozen=True)
class FvdDesignResult:
    """Computed FVD design quantities and damping matrices."""
    target_braced_fundamental_period: float
    supplemental_fvd_first_mode_damping_ratio: float
    converged_fictitious_story_stiffness: np.ndarray
    effective_horizontal_story_damping: np.ndarray
    damper_coefficient_per_device: np.ndarray
    fvd_damping_matrix: np.ndarray
    total_damping_matrix: np.ndarray
    retrofitted_modal_damping: np.ndarray
    fvd_modal_damping: np.ndarray
    iteration_history: tuple[FictitiousIterationRow, ...]


@dataclass
class GroundMotionRecord:
    """Ground acceleration record stored in SI units."""
    name: str
    dt: float
    acceleration_mps2: np.ndarray
    source_path: Path

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.acceleration_mps2.size, dtype=float) * self.dt

    @property
    def acceleration_g(self) -> np.ndarray:
        return self.acceleration_mps2 / G_SI


@dataclass
class LinearMDOFResult:
    """Response histories for a linear base-excited MDOF system."""
    label: str
    time: np.ndarray
    ground_acceleration: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    relative_acceleration: np.ndarray
    absolute_acceleration: np.ndarray
    interstory_drift: np.ndarray
    interstory_drift_ratio: np.ndarray

    @property
    def peak_floor_displacement(self) -> np.ndarray:
        return np.max(np.abs(self.displacement), axis=0)

    @property
    def peak_interstory_drift_ratio(self) -> np.ndarray:
        return np.max(np.abs(self.interstory_drift_ratio), axis=0)

    @property
    def peak_absolute_acceleration_g(self) -> np.ndarray:
        return np.max(np.abs(self.absolute_acceleration), axis=0) / G_SI

    @property
    def peak_roof_displacement(self) -> float:
        return float(self.peak_floor_displacement[-1])


class GroundMotionLoader:
    """Read PEER-style and simple numeric ground-motion files."""

    @staticmethod
    def find_sylmar_path() -> Path:
        for path in SYLMAR_CANDIDATES:
            if path.exists():
                return path
        candidates = "\n".join(f"  - {path}" for path in SYLMAR_CANDIDATES)
        raise FileNotFoundError(
            "SYLMAR360.txt was not found. Place the file in one of these locations:\n"
            f"{candidates}"
        )

    @staticmethod
    def load_acceleration_file(path: Path, name: str | None = None) -> GroundMotionRecord:
        if not path.exists():
            raise FileNotFoundError(f"Ground motion file not found: {path}")

        dt = GroundMotionLoader._parse_dt(path)
        numeric_rows = GroundMotionLoader._read_numeric_rows(path)
        if numeric_rows.size == 0:
            raise ValueError(f"No numeric acceleration data found in: {path}")

        if (
                numeric_rows.ndim == 2
                and numeric_rows.shape[1] >= 2
                and GroundMotionLoader._looks_like_time_column(numeric_rows[:, 0])
        ):
            time = np.asarray(numeric_rows[:, 0], dtype=float)
            acc_g = np.asarray(numeric_rows[:, 1], dtype=float)
            dt_from_time = float(np.median(np.diff(time)))
            dt = dt_from_time if dt is None else dt
        else:
            acc_g = np.asarray(numeric_rows, dtype=float).ravel()

        if dt is None:
            raise ValueError(
                f"Could not parse a time step from {path}. "
                "Add a DT=... header or use a two-column time/acceleration file."
            )
        if acc_g.size < 2:
            raise ValueError(f"Ground motion file has too few data points: {path}")

        return GroundMotionRecord(
            name=name or path.stem,
            dt=float(dt),
            acceleration_mps2=np.asarray(acc_g, dtype=float).ravel() * G_SI,
            source_path=path,
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


class ShearBuildingBuilder:
    """Build mass, stiffness, and damping matrices for the FVD retrofit example."""

    @staticmethod
    def story_drift_matrix(n_stories: int) -> np.ndarray:
        """Return B, where story_drifts = B @ floor_displacements.

        For a three-story model, B @ [u1, u2, u3] gives
        [u1, u2 - u1, u3 - u2]. The first component uses u0 = 0 because the
        floor coordinates are relative to the moving ground.
        """
        b = np.eye(n_stories, dtype=float)
        for i in range(1, n_stories):
            b[i, i - 1] = -1.0
        return b

    @staticmethod
    def build(data: FvdProblemData) -> tuple[StructuralMatrices, ModalProperties]:
        masses = np.array([
            data.lower_floor_pressure * data.floor_area / G_SI,
            data.lower_floor_pressure * data.floor_area / G_SI,
            data.roof_pressure * data.floor_area / G_SI,
        ], dtype=float)

        mass = np.diag(masses)

        column_stiffness = (
                12.0
                * data.young_modulus
                * data.column_second_moment_area
                / data.story_height ** 3
        )
        story_stiffness = np.full(
            data.n_stories,
            data.n_lateral_columns_per_story * column_stiffness,
            dtype=float,
        )

        b = ShearBuildingBuilder.story_drift_matrix(data.n_stories)
        stiffness = b.T @ np.diag(story_stiffness) @ b

        modal = ModalAnalysis.solve(mass, stiffness)
        i_mode, j_mode = data.rayleigh_modes
        omega_i = float(modal.omega[i_mode - 1])
        omega_j = float(modal.omega[j_mode - 1])

        xi = data.rayleigh_damping_ratio
        a_m = 2.0 * xi * omega_i * omega_j / (omega_i + omega_j)
        a_k = 2.0 * xi / (omega_i + omega_j)
        rayleigh = a_m * mass + a_k * stiffness
        original_modal_damping = ModalAnalysis.modal_damping_ratios(mass, rayleigh, modal)

        matrices = StructuralMatrices(
            mass=mass,
            stiffness=stiffness,
            rayleigh_damping=rayleigh,
            story_stiffness=story_stiffness,
            story_height=np.full(data.n_stories, data.story_height, dtype=float),
            rayleigh_mass_coefficient=float(a_m),
            rayleigh_stiffness_coefficient=float(a_k),
            original_modal_damping=original_modal_damping,
        )
        return matrices, modal


class ModalAnalysis:
    """Undamped modal analysis and modal damping utilities."""

    @staticmethod
    def solve(mass: np.ndarray, stiffness: np.ndarray) -> ModalProperties:
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(mass, stiffness))
        order = np.argsort(np.real(eigenvalues))
        eigenvalues = np.real(eigenvalues[order])
        modes = np.real(eigenvectors[:, order])

        if np.any(eigenvalues <= 0.0):
            raise ValueError("The generalized eigenproblem produced nonpositive eigenvalues.")

        omega = np.sqrt(eigenvalues)
        periods = 2.0 * math.pi / omega

        for j in range(modes.shape[1]):
            roof_value = modes[-1, j]
            if abs(roof_value) > 1.0e-14:
                modes[:, j] = modes[:, j] / roof_value

        generalized_mass = np.array([
            modes[:, j].T @ mass @ modes[:, j]
            for j in range(modes.shape[1])
        ])
        generalized_stiffness = np.array([
            modes[:, j].T @ stiffness @ modes[:, j]
            for j in range(modes.shape[1])
        ])

        r = np.ones(mass.shape[0], dtype=float)
        participation_factors = np.array([
            (modes[:, j].T @ mass @ r) / generalized_mass[j]
            for j in range(modes.shape[1])
        ])
        effective_modal_mass = participation_factors ** 2 * generalized_mass

        return ModalProperties(
            omega=omega,
            periods=periods,
            modes=modes,
            generalized_mass=generalized_mass,
            generalized_stiffness=generalized_stiffness,
            participation_factors=participation_factors,
            effective_modal_mass=effective_modal_mass,
        )

    @staticmethod
    def modal_damping_ratios(
            mass: np.ndarray,
            damping: np.ndarray,
            modal: ModalProperties,
    ) -> np.ndarray:
        ratios = []
        for j in range(modal.modes.shape[1]):
            phi = modal.modes[:, j]
            c_n = float(phi.T @ damping @ phi)
            m_n = float(phi.T @ mass @ phi)
            ratios.append(c_n / (2.0 * float(modal.omega[j]) * m_n))
        return np.asarray(ratios, dtype=float)


class FvdDesigner:
    """Size linear FVDs through the auxiliary fictitious-braced system.

    The input damping target is the supplemental first-mode damping ratio supplied
    by the FVD system alone. The method temporarily represents that desired
    viscous effect with fictitious story springs placed at the same stories as
    the dampers. The fictitious springs are scaled until the first period of the
    auxiliary braced system reaches the target braced period. The converged
    fictitious stiffnesses are then converted into horizontal viscous coefficients
    and finally into one physical damper coefficient per device.

    The fictitious springs are not part of the final structural model; only the
    resulting FVD damping matrix is added to the Rayleigh damping matrix.
    """

    @staticmethod
    def design(
            data: FvdProblemData,
            matrices: StructuralMatrices,
            modal: ModalProperties,
    ) -> FvdDesignResult:
        """Return the FVD coefficients and damping matrices.

        Calculation flow:

        1. Read the original fundamental period from the unretrofitted model.
        2. Convert the supplemental FVD damping target into the target fundamental
           period of the auxiliary braced system.
        3. Iterate on fictitious story stiffnesses until the auxiliary braced
           system reaches that target period.
        4. Convert the converged fictitious stiffnesses into effective horizontal
           story damping coefficients.
        5. Divide each story damping coefficient among the physical dampers acting
           in parallel.
        6. Assemble C_FVD and C_total = C_Rayleigh + C_FVD.
        """
        supplemental_first_mode_damping_ratio = (
            data.supplemental_fvd_first_mode_damping_ratio
        )
        if supplemental_first_mode_damping_ratio <= 0.0:
            raise ValueError("The supplemental first-mode FVD damping ratio must be positive.")

        original_fundamental_period = float(modal.periods[0])
        target_braced_fundamental_period = (
                original_fundamental_period
                / math.sqrt(1.0 + 2.0 * supplemental_first_mode_damping_ratio)
        )

        # First trial: keep the same story distribution as the original lateral
        # stiffness and scale it by the configured trial factor.
        trial_fictitious_story_stiffness = (
                data.initial_fictitious_story_stiffness_scale
                * matrices.story_stiffness.copy()
        )
        iteration_history: list[FictitiousIterationRow] = []

        for iteration in range(data.max_fictitious_braced_iterations):
            trial_braced_fundamental_period = (
                FvdDesigner._compute_fictitious_braced_fundamental_period(
                    mass=matrices.mass,
                    original_stiffness=matrices.stiffness,
                    fictitious_story_stiffness=trial_fictitious_story_stiffness,
                )
            )

            relative_period_error = (
                                            trial_braced_fundamental_period - target_braced_fundamental_period
                                    ) / target_braced_fundamental_period

            iteration_history.append(
                FictitiousIterationRow(
                    iteration=iteration,
                    trial_story_stiffness=trial_fictitious_story_stiffness.copy(),
                    trial_period=float(trial_braced_fundamental_period),
                    period_error=float(relative_period_error),
                )
            )

            if abs(relative_period_error) <= data.braced_period_relative_tolerance:
                break

            # Scale all fictitious springs by one factor. This preserves the chosen
            # damper layout while correcting the auxiliary first-period error.
            update_denominator = 1.0 - (
                    (
                            target_braced_fundamental_period ** 2
                            - trial_braced_fundamental_period ** 2
                    )
                    / (
                            target_braced_fundamental_period ** 2
                            - original_fundamental_period ** 2
                    )
            )
            if abs(update_denominator) < 1.0e-14:
                raise ZeroDivisionError(
                    "The fictitious-spring update denominator is too small."
                )

            trial_fictitious_story_stiffness = (
                    trial_fictitious_story_stiffness / update_denominator
            )
        else:
            raise RuntimeError("The fictitious-braced-system iteration did not converge.")

        converged_fictitious_story_stiffness = trial_fictitious_story_stiffness

        # Harmonic first-mode sizing relation: k_hat = omega_1 c_h.
        effective_horizontal_story_damping = (
                original_fundamental_period
                * converged_fictitious_story_stiffness
                / (2.0 * math.pi)
        )

        damper_coefficient_per_device = (
                effective_horizontal_story_damping
                / (
                        float(data.dampers_per_story)
                        * data.damper_horizontal_projection_factor ** 2
                )
        )

        story_drift_matrix = ShearBuildingBuilder.story_drift_matrix(
            data.n_stories
        )
        fvd_damping = (
                story_drift_matrix.T
                @ np.diag(effective_horizontal_story_damping)
                @ story_drift_matrix
        )
        total_damping = matrices.rayleigh_damping + fvd_damping

        fvd_modal_damping = ModalAnalysis.modal_damping_ratios(
            matrices.mass,
            fvd_damping,
            modal,
        )
        retrofitted_modal_damping = ModalAnalysis.modal_damping_ratios(
            matrices.mass,
            total_damping,
            modal,
        )

        return FvdDesignResult(
            target_braced_fundamental_period=target_braced_fundamental_period,
            supplemental_fvd_first_mode_damping_ratio=supplemental_first_mode_damping_ratio,
            converged_fictitious_story_stiffness=converged_fictitious_story_stiffness,
            effective_horizontal_story_damping=effective_horizontal_story_damping,
            damper_coefficient_per_device=damper_coefficient_per_device,
            fvd_damping_matrix=fvd_damping,
            total_damping_matrix=total_damping,
            retrofitted_modal_damping=retrofitted_modal_damping,
            fvd_modal_damping=fvd_modal_damping,
            iteration_history=tuple(iteration_history),
        )

    @staticmethod
    def _compute_fictitious_braced_fundamental_period(
            mass: np.ndarray,
            original_stiffness: np.ndarray,
            fictitious_story_stiffness: np.ndarray,
    ) -> float:
        """Compute T_1 of the temporary system K_original + K_hat.

        The vector fictitious_story_stiffness stores one trial horizontal spring
        stiffness per story. These story springs are assembled as:

            K_hat = B.T @ diag(fictitious_story_stiffness) @ B

        The function returns only the first period of this temporary elastic
        system.
        """
        story_drift_matrix = ShearBuildingBuilder.story_drift_matrix(
            original_stiffness.shape[0]
        )
        fictitious_bracing_stiffness = (
                story_drift_matrix.T
                @ np.diag(fictitious_story_stiffness)
                @ story_drift_matrix
        )
        braced_modal = ModalAnalysis.solve(
            mass,
            original_stiffness + fictitious_bracing_stiffness,
        )
        return float(braced_modal.periods[0])


class LinearNewmarkMDOFSolver:
    """Average-acceleration Newmark solver for a linear base-excited MDOF system."""

    @staticmethod
    def solve(
            label: str,
            record: GroundMotionRecord,
            mass: np.ndarray,
            damping: np.ndarray,
            stiffness: np.ndarray,
            story_height: np.ndarray,
            beta: float = 1.0 / 4.0,
            gamma: float = 1.0 / 2.0,
    ) -> LinearMDOFResult:
        ag = record.acceleration_mps2
        dt = record.dt
        n_steps = ag.size
        n_dof = mass.shape[0]

        r = np.ones(n_dof, dtype=float)
        u = np.zeros((n_steps, n_dof), dtype=float)
        v = np.zeros((n_steps, n_dof), dtype=float)
        a = np.zeros((n_steps, n_dof), dtype=float)

        p0 = -mass @ r * ag[0]
        a[0, :] = np.linalg.solve(mass, p0 - damping @ v[0, :] - stiffness @ u[0, :])

        a0 = 1.0 / (beta * dt * dt)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2.0 * beta) - 1.0)

        k_eff = stiffness + a0 * mass + a1 * damping

        for i in range(1, n_steps):
            p_i = -mass @ r * ag[i]
            p_eff = (
                    p_i
                    + mass @ (a0 * u[i - 1, :] + a2 * v[i - 1, :] + a3 * a[i - 1, :])
                    + damping @ (a1 * u[i - 1, :] + a4 * v[i - 1, :] + a5 * a[i - 1, :])
            )

            u[i, :] = np.linalg.solve(k_eff, p_eff)
            a[i, :] = a0 * (u[i, :] - u[i - 1, :]) - a2 * v[i - 1, :] - a3 * a[i - 1, :]
            v[i, :] = v[i - 1, :] + dt * ((1.0 - gamma) * a[i - 1, :] + gamma * a[i, :])

        absolute_acceleration = a + ag[:, None] * r[None, :]
        b = ShearBuildingBuilder.story_drift_matrix(n_dof)
        interstory_drift = u @ b.T
        interstory_drift_ratio = interstory_drift / story_height[None, :]

        return LinearMDOFResult(
            label=label,
            time=record.time,
            ground_acceleration=ag,
            displacement=u,
            velocity=v,
            relative_acceleration=a,
            absolute_acceleration=absolute_acceleration,
            interstory_drift=interstory_drift,
            interstory_drift_ratio=interstory_drift_ratio,
        )


class FigureFactory:
    """Create Plotly figures used in the HTML report."""

    @staticmethod
    def ground_motion(record: GroundMotionRecord) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=record.time,
            y=record.acceleration_g,
            mode="lines",
            line=dict(color=MATLAB_COLORS["black"], width=1.7),
            name=record.name,
            hovertemplate="t: %{x:.3f} s<br>a_g: %{y:.5f} g<extra></extra>",
        ))
        fig.update_layout(
            template="plotly_white",
            height=420,
            title=dict(text=f"Ground Motion — {record.name}", x=0.5, font=dict(size=22)),
            xaxis=dict(title="Time [s]"),
            yaxis=dict(title="Ground acceleration [g]", zeroline=True),
            font=dict(size=14),
            hovermode="x unified",
        )
        return fig

    @staticmethod
    def _peak_index_and_coordinates(
            time: np.ndarray,
            values: np.ndarray,
    ) -> tuple[int, float, float]:
        """Return the index, time, and signed value of the maximum absolute response."""
        peak_index = int(np.argmax(np.abs(values)))
        return peak_index, float(time[peak_index]), float(values[peak_index])

    @staticmethod
    def _set_padded_y_range(
            fig: go.Figure,
            row: int,
            col: int,
            values: np.ndarray,
    ) -> None:
        """Pad a subplot y-axis so peak markers and callouts are not clipped."""
        values = np.asarray(values, dtype=float)
        y_min = float(np.min(values))
        y_max = float(np.max(values))
        span = y_max - y_min
        if span <= 1.0e-12:
            span = max(abs(y_min), abs(y_max), 1.0)
        pad = 0.18 * span
        fig.update_yaxes(range=[y_min - pad, y_max + pad], row=row, col=col)

    @staticmethod
    def _add_peak_marker_and_annotation(
            fig: go.Figure,
            row: int,
            col: int,
            time: np.ndarray,
            values: np.ndarray,
            series_label: str,
            response_label: str,
            unit_label: str,
            value_format: str,
            line_color: str,
            ax: int,
            ay: int,
    ) -> None:
        """Add the same peak marker/callout format used in the reference dashboard."""
        _, peak_time, peak_value = FigureFactory._peak_index_and_coordinates(time, values)
        peak_abs = abs(peak_value)
        annotation_text = (
            f"{series_label}<br>"
            f"{response_label} = {peak_abs:{value_format}} {unit_label}"
        )

        fig.add_trace(
            go.Scatter(
                x=[peak_time],
                y=[peak_value],
                mode="markers",
                marker=dict(
                    size=10,
                    color=MATLAB_COLORS["black"],
                    line=dict(color=line_color, width=2),
                ),
                showlegend=False,
                hovertemplate=(
                    f"{series_label} peak<br>"
                    "t: %{x:.3f} s<br>"
                    f"signed value: %{{y:{value_format}}} {unit_label}<br>"
                    f"{response_label}: {peak_abs:{value_format}} {unit_label}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
        fig.add_annotation(
            x=peak_time,
            y=peak_value,
            text=annotation_text,
            showarrow=True,
            arrowhead=2,
            ax=ax,
            ay=ay,
            font=dict(size=13, color=MATLAB_COLORS["black"]),
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor=line_color,
            borderwidth=1,
            row=row,
            col=col,
        )

    @staticmethod
    def floor_displacements(original: LinearMDOFResult, retrofitted: LinearMDOFResult) -> go.Figure:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.075)
        labels = ("Floor 1 (u1)", "Floor 2 (u2)", "Roof (u3)")

        for j, label in enumerate(labels):
            row = j + 1
            original_u = 1.0e3 * original.displacement[:, j]
            retrofitted_u = 1.0e3 * retrofitted.displacement[:, j]

            fig.add_trace(go.Scatter(
                x=original.time,
                y=original_u,
                mode="lines",
                line=dict(color=MATLAB_COLORS["dark_blue"], width=1.9),
                name=f"Original {label}",
                legendgroup="original",
                showlegend=j == 0,
                hovertemplate=f"{label}<br>t: %{{x:.3f}} s<br>u: %{{y:.3f}} mm<extra>Original</extra>",
            ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=retrofitted.time,
                y=retrofitted_u,
                mode="lines",
                line=dict(color=MATLAB_COLORS["crimson"], width=1.9),
                name=f"Retrofitted {label}",
                legendgroup="retrofitted",
                showlegend=j == 0,
                hovertemplate=f"{label}<br>t: %{{x:.3f}} s<br>u: %{{y:.3f}} mm<extra>Retrofitted</extra>",
            ), row=row, col=1)

            FigureFactory._add_peak_marker_and_annotation(
                fig=fig,
                row=row,
                col=1,
                time=original.time,
                values=original_u,
                series_label="Original",
                response_label="Peak |u|",
                unit_label="mm",
                value_format=".3f",
                line_color=MATLAB_COLORS["dark_blue"],
                ax=36,
                ay=-30,
            )
            FigureFactory._add_peak_marker_and_annotation(
                fig=fig,
                row=row,
                col=1,
                time=retrofitted.time,
                values=retrofitted_u,
                series_label="Retrofitted",
                response_label="Peak |u|",
                unit_label="mm",
                value_format=".3f",
                line_color=MATLAB_COLORS["crimson"],
                ax=-36,
                ay=30,
            )

            FigureFactory._set_padded_y_range(fig, row, 1, np.r_[original_u, retrofitted_u])
            fig.update_yaxes(title_text=f"{label}<br>u [mm]", row=row, col=1, zeroline=True)

        fig.update_layout(
            template="plotly_white",
            height=860,
            title=dict(text="Relative Floor Displacement Histories", x=0.5, font=dict(size=22)),
            xaxis3=dict(title="Time [s]"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            font=dict(size=14),
            hovermode="x unified",
            margin=dict(t=115, r=45, b=70, l=75),
        )
        return fig

    @staticmethod
    def interstory_drift_ratios(original: LinearMDOFResult, retrofitted: LinearMDOFResult) -> go.Figure:
        """Plot IDR histories with peak callouts attached to the peak markers.

        The annotations remain arrow callouts to the peak points. When the original
        and retrofitted peak times are close, the labels are placed on opposite
        sides of the markers and separated according to which peak is higher in
        the subplot.
        """
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.075)
        labels = ("Story 1", "Story 2", "Story 3")

        for j, label in enumerate(labels):
            row = j + 1
            original_idr = 100.0 * original.interstory_drift_ratio[:, j]
            retrofitted_idr = 100.0 * retrofitted.interstory_drift_ratio[:, j]

            fig.add_trace(go.Scatter(
                x=original.time,
                y=original_idr,
                mode="lines",
                line=dict(color=MATLAB_COLORS["dark_blue"], width=1.9),
                name=f"Original {label}",
                legendgroup="original",
                showlegend=j == 0,
                hovertemplate=f"{label}<br>t: %{{x:.3f}} s<br>IDR: %{{y:.4f}} %<extra>Original</extra>",
            ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=retrofitted.time,
                y=retrofitted_idr,
                mode="lines",
                line=dict(color=MATLAB_COLORS["crimson"], width=1.9),
                name=f"Retrofitted {label}",
                legendgroup="retrofitted",
                showlegend=j == 0,
                hovertemplate=f"{label}<br>t: %{{x:.3f}} s<br>IDR: %{{y:.4f}} %<extra>Retrofitted</extra>",
            ), row=row, col=1)

            _, original_peak_time, original_peak_value = FigureFactory._peak_index_and_coordinates(
                original.time,
                original_idr,
            )
            _, retrofitted_peak_time, retrofitted_peak_value = FigureFactory._peak_index_and_coordinates(
                retrofitted.time,
                retrofitted_idr,
            )

            record_duration = float(original.time[-1] - original.time[0])
            peaks_close_in_time = abs(original_peak_time - retrofitted_peak_time) <= 0.025 * record_duration

            if peaks_close_in_time:
                original_is_upper_peak = original_peak_value >= retrofitted_peak_value

                if original_is_upper_peak:
                    original_ax, original_ay = 95, -72
                    retrofitted_ax, retrofitted_ay = -95, 72
                else:
                    original_ax, original_ay = 95, 72
                    retrofitted_ax, retrofitted_ay = -95, -72
            else:
                original_ax = 60
                original_ay = -44 if original_peak_value >= 0.0 else 44
                retrofitted_ax = -60
                retrofitted_ay = -44 if retrofitted_peak_value >= 0.0 else 44

            FigureFactory._add_peak_marker_and_annotation(
                fig=fig,
                row=row,
                col=1,
                time=original.time,
                values=original_idr,
                series_label="Original",
                response_label="Peak IDR",
                unit_label="%",
                value_format=".4f",
                line_color=MATLAB_COLORS["dark_blue"],
                ax=original_ax,
                ay=original_ay,
            )
            FigureFactory._add_peak_marker_and_annotation(
                fig=fig,
                row=row,
                col=1,
                time=retrofitted.time,
                values=retrofitted_idr,
                series_label="Retrofitted",
                response_label="Peak IDR",
                unit_label="%",
                value_format=".4f",
                line_color=MATLAB_COLORS["crimson"],
                ax=retrofitted_ax,
                ay=retrofitted_ay,
            )

            combined_idr = np.r_[original_idr, retrofitted_idr]
            y_min = float(np.min(combined_idr))
            y_max = float(np.max(combined_idr))
            y_span = max(y_max - y_min, abs(y_min), abs(y_max), 1.0e-6)
            fig.update_yaxes(
                title_text=f"{label}<br>IDR [%]",
                range=[y_min - 0.34 * y_span, y_max + 0.34 * y_span],
                row=row,
                col=1,
                zeroline=True,
            )

        fig.update_layout(
            template="plotly_white",
            height=900,
            title=dict(text="Interstory Drift Ratio Histories", x=0.5, font=dict(size=22)),
            xaxis3=dict(title="Time [s]"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            font=dict(size=14),
            hovermode="x unified",
            margin=dict(t=115, r=60, b=70, l=75),
        )
        return fig

    @staticmethod
    def absolute_accelerations(original: LinearMDOFResult, retrofitted: LinearMDOFResult) -> go.Figure:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.075)
        labels = ("Floor 1", "Floor 2", "Roof")

        for j, label in enumerate(labels):
            row = j + 1
            original_a = original.absolute_acceleration[:, j] / G_SI
            retrofitted_a = retrofitted.absolute_acceleration[:, j] / G_SI

            fig.add_trace(go.Scatter(
                x=original.time,
                y=original_a,
                mode="lines",
                line=dict(color=MATLAB_COLORS["dark_blue"], width=1.9),
                name=f"Original {label}",
                legendgroup="original",
                showlegend=j == 0,
                hovertemplate=f"{label}<br>t: %{{x:.3f}} s<br>a_abs: %{{y:.4f}} g<extra>Original</extra>",
            ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=retrofitted.time,
                y=retrofitted_a,
                mode="lines",
                line=dict(color=MATLAB_COLORS["crimson"], width=1.9),
                name=f"Retrofitted {label}",
                legendgroup="retrofitted",
                showlegend=j == 0,
                hovertemplate=f"{label}<br>t: %{{x:.3f}} s<br>a_abs: %{{y:.4f}} g<extra>Retrofitted</extra>",
            ), row=row, col=1)

            FigureFactory._add_peak_marker_and_annotation(
                fig=fig,
                row=row,
                col=1,
                time=original.time,
                values=original_a,
                series_label="Original",
                response_label="Peak |a_abs|",
                unit_label="g",
                value_format=".4f",
                line_color=MATLAB_COLORS["dark_blue"],
                ax=36,
                ay=-30,
            )
            FigureFactory._add_peak_marker_and_annotation(
                fig=fig,
                row=row,
                col=1,
                time=retrofitted.time,
                values=retrofitted_a,
                series_label="Retrofitted",
                response_label="Peak |a_abs|",
                unit_label="g",
                value_format=".4f",
                line_color=MATLAB_COLORS["crimson"],
                ax=-36,
                ay=30,
            )

            FigureFactory._set_padded_y_range(fig, row, 1, np.r_[original_a, retrofitted_a])
            fig.update_yaxes(title_text=f"{label}<br>a_abs [g]", row=row, col=1, zeroline=True)

        fig.update_layout(
            template="plotly_white",
            height=860,
            title=dict(text="Absolute Floor Acceleration Histories", x=0.5, font=dict(size=22)),
            xaxis3=dict(title="Time [s]"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            font=dict(size=14),
            hovermode="x unified",
            margin=dict(t=115, r=45, b=70, l=75),
        )
        return fig

    @staticmethod
    def peak_comparison(original: LinearMDOFResult, retrofitted: LinearMDOFResult) -> go.Figure:
        """Compare peak IDR, floor acceleration, and floor displacement values."""
        idr_categories = ["Story 1", "Story 2", "Story 3"]
        idr_original = 100.0 * original.peak_interstory_drift_ratio
        idr_retrofitted = 100.0 * retrofitted.peak_interstory_drift_ratio

        acc_categories = ["Floor 1", "Floor 2", "Roof"]
        acc_original = original.peak_absolute_acceleration_g
        acc_retrofitted = retrofitted.peak_absolute_acceleration_g

        disp_categories = ["Floor 1 (u1)", "Floor 2 (u2)", "Roof (u3)"]
        disp_original = 1.0e3 * np.max(np.abs(original.displacement), axis=0)
        disp_retrofitted = 1.0e3 * np.max(np.abs(retrofitted.displacement), axis=0)

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.18,
            subplot_titles=(
                "Peak interstory drift ratio",
                "Peak absolute floor acceleration",
                "Peak relative floor displacement",
            ),
        )

        def add_grouped_bars(
                row: int,
                categories: list[str],
                original_values: np.ndarray,
                retrofitted_values: np.ndarray,
                unit_label: str,
                value_format: str,
                showlegend: bool,
        ) -> None:
            fig.add_trace(go.Bar(
                x=categories,
                y=original_values,
                marker_color=MATLAB_COLORS["dark_blue"],
                name="Original",
                text=[f"{value:{value_format}}" for value in original_values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=f"%{{x}}<br>Original: %{{y:{value_format}}} {unit_label}<extra></extra>",
                showlegend=showlegend,
            ), row=row, col=1)
            fig.add_trace(go.Bar(
                x=categories,
                y=retrofitted_values,
                marker_color=MATLAB_COLORS["crimson"],
                name="Retrofitted",
                text=[f"{value:{value_format}}" for value in retrofitted_values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=f"%{{x}}<br>Retrofitted: %{{y:{value_format}}} {unit_label}<extra></extra>",
                showlegend=showlegend,
            ), row=row, col=1)

            max_value = float(max(np.max(original_values), np.max(retrofitted_values), 1.0e-12))
            fig.update_yaxes(range=[0.0, 1.25 * max_value], row=row, col=1)

        add_grouped_bars(1, idr_categories, idr_original, idr_retrofitted, "%", ".4f", True)
        add_grouped_bars(2, acc_categories, acc_original, acc_retrofitted, "g", ".4f", False)
        add_grouped_bars(3, disp_categories, disp_original, disp_retrofitted, "mm", ".3f", False)

        fig.update_yaxes(title_text="IDR [%]", row=1, col=1)
        fig.update_yaxes(title_text="|a_abs| [g]", row=2, col=1)
        fig.update_yaxes(title_text="|u| [mm]", row=3, col=1)

        fig.update_layout(
            template="plotly_white",
            height=980,
            title=dict(text="Peak Response Comparison", x=0.5, font=dict(size=22)),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
            font=dict(size=14),
            barmode="group",
            margin=dict(t=135, r=45, b=80, l=75),
        )
        return fig

    @staticmethod
    def interstory_drift_profile(original: LinearMDOFResult, retrofitted: LinearMDOFResult) -> go.Figure:
        """Plot peak interstory drift ratio against story number."""
        story_numbers = np.array([1.0, 2.0, 3.0], dtype=float)
        original_idr = 100.0 * original.peak_interstory_drift_ratio
        retrofitted_idr = 100.0 * retrofitted.peak_interstory_drift_ratio

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=original_idr,
            y=story_numbers,
            mode="lines+markers+text",
            name="Original",
            line=dict(color=MATLAB_COLORS["dark_blue"], width=2.2),
            marker=dict(color=MATLAB_COLORS["dark_blue"], size=10),
            text=[f"{value:.4f} %" for value in original_idr],
            textposition="middle right",
            hovertemplate="Story %{y:.0f}<br>Peak IDR: %{x:.4f} %<extra>Original</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=retrofitted_idr,
            y=story_numbers,
            mode="lines+markers+text",
            name="Retrofitted",
            line=dict(color=MATLAB_COLORS["crimson"], width=2.2),
            marker=dict(color=MATLAB_COLORS["crimson"], size=10),
            text=[f"{value:.4f} %" for value in retrofitted_idr],
            textposition="middle left",
            hovertemplate="Story %{y:.0f}<br>Peak IDR: %{x:.4f} %<extra>Retrofitted</extra>",
        ))

        max_value = float(max(np.max(original_idr), np.max(retrofitted_idr), 1.0e-8))
        fig.update_layout(
            template="plotly_white",
            height=540,
            title=dict(text="Peak Interstory Drift Ratio Profile", x=0.5, font=dict(size=22)),
            xaxis=dict(title="Peak interstory drift ratio [%]", range=[0.0, 1.18 * max_value]),
            yaxis=dict(
                title="Story number",
                tickmode="array",
                tickvals=[1, 2, 3],
                ticktext=["Story 1", "Story 2", "Story 3"],
                range=[0.75, 3.25],
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            font=dict(size=14),
            margin=dict(t=110, r=50, b=70, l=95),
        )
        return fig

    @staticmethod
    def fictitious_iteration(design: FvdDesignResult) -> go.Figure:
        iterations = [row.iteration for row in design.iteration_history]
        periods = [row.trial_period for row in design.iteration_history]
        stiffness = [row.trial_story_stiffness[0] / 1.0e6 for row in design.iteration_history]
        errors = [100.0 * row.period_error for row in design.iteration_history]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=iterations,
            y=periods,
            mode="lines+markers",
            line=dict(color=MATLAB_COLORS["dark_blue"], width=2.5),
            name="Trial first period",
            hovertemplate="Iteration: %{x}<br>Trial T1: %{y:.6f} s<extra></extra>",
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=iterations,
            y=stiffness,
            mode="lines+markers",
            line=dict(color=MATLAB_COLORS["crimson"], width=2.5),
            name="Fictitious story stiffness",
            hovertemplate="Iteration: %{x}<br>k_hat: %{y:.3f} MN/m<extra></extra>",
        ), secondary_y=True)
        fig.add_hline(y=design.target_braced_fundamental_period,
                      line=dict(color=MATLAB_COLORS["black"], width=1.6, dash="dash"))
        fig.update_layout(
            template="plotly_white",
            height=480,
            title=dict(text="Fictitious-Spring Iteration", x=0.5, font=dict(size=22)),
            xaxis=dict(title="Iteration"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            font=dict(size=14),
        )
        fig.update_yaxes(title_text="Period [s]", secondary_y=False)
        fig.update_yaxes(title_text="Story fictitious stiffness [MN/m]", secondary_y=True)
        return fig


class HtmlReportBuilder:
    """Build a self-contained CE223-style HTML dashboard."""

    def __init__(
            self,
            data: FvdProblemData,
            matrices: StructuralMatrices,
            modal: ModalProperties,
            design: FvdDesignResult,
    ) -> None:
        self.data = data
        self.matrices = matrices
        self.modal = modal
        self.design = design

    @staticmethod
    def fig_to_div(fig: go.Figure, include_js: bool = False) -> str:
        return to_html(
            fig,
            include_plotlyjs=include_js,
            full_html=False,
            config=dict(displayModeBar=True, responsive=True),
        )

    def build(
            self,
            record: GroundMotionRecord | None,
            original_result: LinearMDOFResult | None,
            retrofitted_result: LinearMDOFResult | None,
            missing_motion_message: str | None = None,
    ) -> str:
        plotly_loaded = self.fig_to_div(FigureFactory.fictitious_iteration(self.design), include_js=True)

        response_plots = ""
        if record is not None and original_result is not None and retrofitted_result is not None:
            response_plots = "\n".join([
                '<h3>Ground-motion and response-history plots</h3>',
                f'<div class="plot-card">{self.fig_to_div(FigureFactory.ground_motion(record), include_js=False)}</div>',
                f'<div class="plot-card">{self.fig_to_div(FigureFactory.floor_displacements(original_result, retrofitted_result), include_js=False)}</div>',
                f'<div class="plot-card">{self.fig_to_div(FigureFactory.interstory_drift_ratios(original_result, retrofitted_result), include_js=False)}</div>',
                f'<div class="plot-card">{self.fig_to_div(FigureFactory.absolute_accelerations(original_result, retrofitted_result), include_js=False)}</div>',
                f'<div class="plot-card">{self.fig_to_div(FigureFactory.interstory_drift_profile(original_result, retrofitted_result), include_js=False)}</div>',
                f'<div class="plot-card">{self.fig_to_div(FigureFactory.peak_comparison(original_result, retrofitted_result), include_js=False)}</div>',
            ])

        response_summary = ""
        if original_result is not None and retrofitted_result is not None:
            response_summary = self._response_summary_table(original_result, retrofitted_result)
        elif missing_motion_message:
            response_summary = (
                '<div class="warning-box">'
                '<h3>Ground-motion response plots</h3>'
                f'<p>{escape_html(missing_motion_message)}</p>'
                '<p>The matrices, modal properties, and FVD design are still fully computed. '
                'The time-history plots are generated automatically when the Sylmar record is present.</p>'
                '</div>'
            )

        conclusions = ""
        if original_result is not None and retrofitted_result is not None:
            conclusions = self._conclusions_section(original_result, retrofitted_result)

        template = r"""
<!DOCTYPE HTML>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <title>CE223 – FVD Retrofit Dashboard</title>
  <link rel="stylesheet" href="../../assets/css/main.css" />
  <noscript><link rel="stylesheet" href="../../assets/css/noscript.css" /></noscript>
  <style>
    :root {
      --berkeley-blue: #003262;
      --cal-gold: #FDB515;
      --ink: #243447;
      --muted: #667085;
      --paper: #ffffff;
      --soft: #f6f7fb;
      --line: #e5e7eb;
      --equation: #f9fafb;
      --shadow: 0 10px 24px rgba(30, 44, 70, 0.06);
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      background: var(--soft);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
    }
    #main.ce223-dashboard { padding: 1.5rem 0 3.5rem 0; }
    .container {
      max-width: 72em;
      margin-left: auto;
      margin-right: auto;
      padding-left: 1rem;
      padding-right: 1rem;
    }
    .inner-report { font-size: 1rem; line-height: 1.58; }
    header.major {
      background: var(--paper);
      border: 1px solid var(--line);
      border-left: 5px solid var(--berkeley-blue);
      border-radius: 10px;
      box-shadow: var(--shadow);
      padding: 1.6rem 1.75rem;
      margin-bottom: 1.5rem;
    }
    header.major h2 {
      color: var(--berkeley-blue);
      font-size: 1.9rem;
      font-weight: 700;
      margin: 0 0 0.55rem 0;
    }
    .summary-lead { color: var(--muted); margin: 0; max-width: 68rem; }
    .box, .report-section {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 10px;
      box-shadow: var(--shadow);
      padding: 1.45rem 1.55rem;
      margin-bottom: 1.5rem;
    }
    .report-section { border-left: 5px solid var(--berkeley-blue); }
    .box h3, .report-section h2, .report-section h3 {
      color: var(--berkeley-blue);
      font-weight: 700;
      margin: 0 0 0.85rem 0;
    }
    .box h3, .report-section h3 {
      font-size: 1.12rem;
      border-left: 4px solid var(--cal-gold);
      padding-left: 0.75rem;
    }
    .report-section h2 { font-size: 1.35rem; margin-bottom: 0.55rem; }
    p { margin: 0 0 0.8rem 0; }
    .note-list { margin: 0.6rem 0 1rem 0; padding-left: 1.2rem; }
    .note-list li { margin: 0.25rem 0; }
    .eq {
      background: var(--equation);
      border-left: 4px solid var(--berkeley-blue);
      border-radius: 6px;
      padding: 0.85rem 1rem;
      margin: 0.85rem 0 1rem 0;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
    }
    .definition-box {
      border: 1px solid #dbe3ee;
      background: #fbfdff;
      border-radius: 8px;
      padding: 0.95rem 1rem;
      margin: 0.9rem 0;
    }
    .definition-box strong { color: var(--berkeley-blue); }
    .matrix-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 1rem;
      margin: 0.9rem 0 1.1rem 0;
    }
    .matrix-card {
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0.95rem 1rem;
      overflow-x: auto;
    }
    .matrix-card h4 {
      color: var(--berkeley-blue);
      margin: 0 0 0.55rem 0;
      font-size: 1rem;
      font-weight: 700;
    }
    .summary-table-wrap {
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
      margin: 1rem 0;
    }
    .summary-table {
      width: 100%;
      min-width: 34rem;
      border-collapse: collapse;
      font-size: 0.94rem;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
    }
    .summary-table thead th {
      background: var(--berkeley-blue);
      color: #fff;
      font-weight: 600;
      text-align: center;
      padding: 0.55rem 0.5rem;
    }
    .summary-table tbody td, .summary-table tbody th {
      padding: 0.5rem 0.5rem;
      border-top: 1px solid var(--line);
      color: var(--ink);
    }
    .summary-table tbody td:first-child, .summary-table tbody th:first-child {
      color: var(--berkeley-blue);
      font-weight: 700;
      text-align: left;
    }
    .summary-table tbody td:nth-child(n+2) {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .summary-table tbody tr:nth-child(even) { background: #f9fafb; }
    .plot-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0.5rem;
      background: #fff;
      margin: 1rem 0;
      overflow-x: auto;
    }
    .warning-box {
      background: #fff8e5;
      border: 1px solid #f0d485;
      border-left: 4px solid var(--cal-gold);
      border-radius: 8px;
      padding: 1rem 1.1rem;
      margin: 1rem 0;
    }
    code { background: #eef2f7; padding: 0.12rem 0.3rem; border-radius: 4px; font-size: 0.92em; }
    #header.site-header-fallback {
      min-height: 3.25rem;
      background: var(--berkeley-blue);
      border-bottom: 0.22rem solid var(--cal-gold);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 1.25rem;
      color: #fff;
    }
    #header.site-header-fallback h1 {
      margin: 0;
      font-size: 1rem;
      font-weight: 700;
      letter-spacing: 0.02em;
    }
    #header.site-header-fallback a {
      color: #fff;
      text-decoration: none;
    }
    #header.site-header-fallback nav ul {
      display: flex;
      gap: 1rem;
      list-style: none;
      margin: 0;
      padding: 0;
    }
    .js-plotly-plot .modebar { top: 56px !important; }
    @media (max-width: 736px) {
      #main.ce223-dashboard { padding-top: 0.9rem; }
      .container { padding-left: 0.75rem; padding-right: 0.75rem; }
      header.major, .box, .report-section { padding: 1rem; }
      header.major h2 { font-size: 1.45rem; }
      .summary-table { font-size: 0.84rem; }
    }
  </style>
  <script>
    window.MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
        processEscapes: true,
        processEnvironments: true
      },
      options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre'] }
    };
  </script>
  <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <script async src="../../assets/js/navigation.js"></script>
</head>
<body class="is-preload">
  <div id="page-wrapper">
    <header id="header"></header>
    <script>
      document.addEventListener("DOMContentLoaded", function () {
        window.setTimeout(function () {
          var header = document.getElementById("header");
          if (header && header.children.length === 0) {
            header.classList.add("site-header-fallback");
            header.innerHTML =
              '<h1><a href="../../index.html">SEMM</a></h1>' +
              '<nav id="nav"><ul>' +
              '<li><a href="../../index.html">Home</a></li>' +
              '<li><a href="../index.html">CE223</a></li>' +
              '<li><a href="./">Dashboards</a></li>' +
              '</ul></nav>';
          }
        }, 350);
      });
    </script>

    <section id="main" class="wrapper style1 ce223-dashboard">
      <div class="container inner-report">
        <header class="major">
          <h2>CE223 – Fluid Viscous Damper Retrofit</h2>
          <p class="summary-lead">
            This dashboard presents the calculation path for the three-story steel moment frame:
            the original shear-building matrices, the Rayleigh damping calibration, the fictitious-spring
            sizing of the linear fluid viscous dampers, and the final retrofitted damping matrix used for
            response-history analysis.
          </p>
        </header>

        <section class="box">
          <h3>Model idealization and sign convention</h3>
          <p>
            The generalized coordinates are the relative horizontal floor displacements with respect to
            the moving ground. The absolute displacement of floor \(i\) is \(u_i(t)+u_g(t)\).
            With this convention, ground acceleration enters as the effective inertia load shown below.
          </p>
          <div class="eq">
            $$\mathbf{u}(t)=\begin{bmatrix}u_1(t)&u_2(t)&u_3(t)\end{bmatrix}^T,
            \qquad
            \mathbf{r}=\begin{bmatrix}1&1&1\end{bmatrix}^T.$$
            $$\mathbf{M}\ddot{\mathbf{u}}(t)
            +\mathbf{C}\dot{\mathbf{u}}(t)
            +\mathbf{K}\mathbf{u}(t)
            =
            -\mathbf{M}\mathbf{r}\ddot u_g(t).$$
          </div>
          <p>
            A convenient way of obtaining the story drift vector from the floor displacement vector
            is to use a story-drift matrix. This matrix subtracts the displacement of the floor below
            from the displacement of the floor above. For the first story, the floor below is the base,
            and the base displacement is zero in the relative-coordinate system.
          </p>
          <div class="eq">
            $$\boldsymbol{\Delta}(t)
            =
            \begin{bmatrix}
            \Delta_1(t)\\
            \Delta_2(t)\\
            \Delta_3(t)
            \end{bmatrix}
            =
            \begin{bmatrix}
            u_1(t)\\
            u_2(t)-u_1(t)\\
            u_3(t)-u_2(t)
            \end{bmatrix}.$$
          </div>
          <p>
            This subtraction can be written compactly as:
          </p>
          <div class="eq">
            $$\boldsymbol{\Delta}(t)=\mathbf{B}\mathbf{u}(t),
            \qquad
            \mathbf{B}=
            \begin{bmatrix}
            1&0&0\\
            -1&1&0\\
            0&-1&1
            \end{bmatrix}.$$
          </div>
          <p>
            This same matrix is useful later because story springs and story dampers act on story
            drift or story drift velocity, while the equations of motion are written in terms of
            floor degrees of freedom.
          </p>
          <ul class="note-list">
            <li>The full floor plan contributes to the seismic mass of each floor.</li>
            <li>Lateral stiffness is assigned only to the moment frames on gridlines A and E.</li>
            <li>Rigid beam-diaphragm action permits the fixed-fixed column sidesway stiffness estimate.</li>
            <li>The K-braced FVD mechanism is treated as a horizontal damper mechanism, so the projection factor is \(\gamma=1\).</li>
          </ul>
          @@PROBLEM_SUMMARY@@
        </section>

        <section class="report-section">
          <h2>Part (a) — Original unretrofitted structure</h2>
          <p>
            The mass matrix is obtained from floor seismic weight divided by \(g\). The stiffness matrix is
            assembled from story stiffnesses through \(\mathbf{K}=\mathbf{B}^T\mathbf{K}_s\mathbf{B}\).
            Rayleigh damping is calibrated to 5 percent damping in modes 1 and 3.
          </p>
          <div class="eq">
            $$A_f=(40.0~\mathrm{m})(32.0~\mathrm{m})=1280~\mathrm{m}^2,
            \qquad
            m_i=\frac{q_iA_f}{g}.$$
            $$k_s=n_c\frac{12EI}{h^3},
            \qquad
            \mathbf{C}_{\mathrm{Rayleigh}}=a_M\mathbf{M}+a_K\mathbf{K}.$$
          </div>
          @@MODAL_SUMMARY@@
          <div class="matrix-grid">
            @@M_MATRIX@@
            @@K_MATRIX@@
            @@CR_MATRIX@@
          </div>
        </section>

        <section class="report-section">
          <h2>Part (b) — FVD preliminary design by fictitious springs</h2>
          <p>
            The design objective is to choose the physical damper coefficient \(C_d\) so that the
            supplemental damping contributed by the FVD system is 35 percent in the first mode.
            In the convention used for this design procedure, \(\xi_1=35\%\) is the first-mode
            contribution of the FVD system alone. The inherent Rayleigh damping of the original
            structure is retained separately and is added afterward through
            \(\mathbf{C}_{\mathrm{Rayleigh}}+\mathbf{C}_{\mathrm{FVD}}\).
          </p>
          <div class="eq">
            $$\xi_{1,\mathrm{FVD}}
            =
            \xi_{1,\mathrm{target}}
            =
            0.35.$$
          </div>

          <p>
            The fictitious-spring procedure is a sizing device that converts a desired first-mode
            viscous effect into damper coefficients at the selected damper locations. A viscous
            damper does not add elastic stiffness to the structure, so a period calculation cannot
            be performed directly from the dampers. The method therefore introduces an auxiliary
            elastic structure with artificial horizontal springs placed at the same locations as
            the proposed dampers. These artificial springs are adjusted until the auxiliary elastic
            structure reaches a target fundamental period. After convergence, the artificial spring
            constants are converted back into viscous coefficients.
          </p>

          <div class="definition-box">
            <p><strong>Notation used in this section.</strong></p>
            @@NOTATION_TABLE@@
          </div>

          <p>
            The connection between viscous damping and fictitious stiffness comes from first-mode
            harmonic motion. For harmonic displacement at frequency \(\omega_1\), the velocity
            amplitude is \(\omega_1\) times the displacement amplitude. Therefore, an effective
            horizontal viscous coefficient \(c_h\) can be represented, only for preliminary sizing,
            by the fictitious stiffness:
          </p>
          <div class="eq">
            $$\widehat{k}_0=\omega_1 c_h=\frac{2\pi}{T_1}c_h.$$
          </div>

          <p>
            The target period of the auxiliary system is obtained by first considering a reference
            supplemental damping matrix proportional to the unbraced stiffness:
          </p>
          <div class="eq">
            $$\mathbf{C}_L=\alpha_0\mathbf{K}.$$
          </div>
          <p>Projecting this reference damping matrix onto the first mode gives:</p>
          <div class="eq">
            $$\boldsymbol{\phi}_1^T\mathbf{C}_L\boldsymbol{\phi}_1
            =
            \alpha_0
            \boldsymbol{\phi}_1^T\mathbf{K}\boldsymbol{\phi}_1
            =
            \alpha_0 K_1
            =
            2\xi_{1,\mathrm{FVD}}\omega_1M_1.$$
          </div>
          <p>
            Since \(K_1=\omega_1^2M_1\), the proportionality constant is
            \(\alpha_0=2\xi_{1,\mathrm{FVD}}/\omega_1\). Multiplying the reference damping
            matrix by \(\omega_1\) gives an equivalent fictitious stiffness matrix:
          </p>
          <div class="eq">
            $$\widehat{\mathbf{K}}_0
            =
            \omega_1\mathbf{C}_L
            =
            2\xi_{1,\mathrm{FVD}}\mathbf{K}.$$
          </div>
          <p>
            Therefore, in the first mode, the auxiliary generalized stiffness increases from
            \(K_1\) to \(K_1(1+2\xi_{1,\mathrm{FVD}})\). The target fundamental period of the
            auxiliary fictitious-spring structure is:
          </p>
          <div class="eq">
            $$\widehat{T}_1
            =
            \frac{T_1}{\sqrt{1+2\xi_{1,\mathrm{FVD}}}}.$$
          </div>

          <p>
            This period is not the actual period of the final retrofitted structure. It is the
            period that the auxiliary elastic structure must reach so that the fictitious spring
            distribution represents the desired first-mode viscous damping effect.
          </p>

          <div class="definition-box">
            <p><strong>Iteration used to place the fictitious springs at the damper locations.</strong></p>
            <p>
              At iteration \(r\), the trial fictitious story stiffnesses are assembled into
              \(\widehat{\mathbf{K}}_0^{(r)}\). The first period of the auxiliary trial structure is
              computed from:
            </p>
            <div class="eq">
              $$\left[
              \mathbf{K}
              +
              \widehat{\mathbf{K}}_0^{(r)}
              -
              \left(\widehat{\omega}_{1,\mathrm{tr}}^{(r)}\right)^2\mathbf{M}
              \right]
              \widehat{\boldsymbol{\phi}}_{1,\mathrm{tr}}^{(r)}
              =
              \mathbf{0},
              \qquad
              \widehat{T}_{1,\mathrm{tr}}^{(r)}
              =
              \frac{2\pi}{\widehat{\omega}_{1,\mathrm{tr}}^{(r)}}.$$
            </div>
            <p>The fictitious stiffnesses are then scaled while preserving the selected damper distribution:</p>
            <div class="eq">
              $$\widehat{k}_{0,j}^{(r+1)}
              =
              \frac{\widehat{k}_{0,j}^{(r)}}
              {1-
              \left[
              \frac{
              \widehat{T}_1^2-
              \left(\widehat{T}_{1,\mathrm{tr}}^{(r)}\right)^2
              }{
              \widehat{T}_1^2-T_1^2
              }
              \right]},
              \qquad
              j=1,\ldots,n_s.$$
            </div>
            <p>
              The numerator inside the correction measures how far the current trial period is from the target
              auxiliary period. The denominator normalizes that error by the full period shift between the
              original structure and the target auxiliary structure. Once
              \(\widehat{T}_{1,\mathrm{tr}}^{(r)}=\widehat{T}_1\), the correction term becomes zero and the
              fictitious stiffnesses stop changing.
            </p>
          </div>

          <p>After convergence, each fictitious stiffness is converted back to an effective horizontal viscous coefficient:</p>
          <div class="eq">
            $$c_{h,j}=\frac{T_1}{2\pi}\widehat{k}_{0,j}.$$
          </div>
          <p>
            If \(n_d\) identical dampers act in parallel at that story and \(\gamma\) is the horizontal projection
            factor of each damper, the physical coefficient of one FVD is:
          </p>
          <div class="eq">
            $$C_{d,j}=\frac{c_{h,j}}{n_d\gamma^2}
            =
            \frac{T_1}{2\pi n_d\gamma^2}\widehat{k}_{0,j}.$$
          </div>
          <p>
            For the rigid K-braced layout used here, the damper action is horizontal, so
            \(\gamma=1\), and four dampers act in parallel at each story.
          </p>

          @@DESIGN_SUMMARY@@
          <div class="matrix-grid">
            @@KHAT_MATRIX@@
            @@CH_MATRIX@@
            @@CD_MATRIX@@
          </div>
          <div class="plot-card">@@FICTITIOUS_PLOT@@</div>
        </section>

        <section class="report-section">
          <h2>Part (c) — Retrofitted structure and response comparison</h2>
          <p>
            The physical lateral stiffness matrix of the building is unchanged by the FVD design.
            The dampers contribute only through the supplemental damping matrix. Since the 35 percent
            design target is the FVD contribution alone, the first-mode damping ratio of the total
            retrofitted model becomes approximately \(5\%+35\%=40\%\) when the Rayleigh and FVD
            matrices are added. With story-based assembly, the FVD matrix is:
          </p>
          <div class="eq">
            $$\mathbf{C}_{\mathrm{FVD}}
            =
            \mathbf{B}^T
            \mathbf{C}_{s,\mathrm{FVD}}
            \mathbf{B},
            \qquad
            \mathbf{C}_{\mathrm{total}}
            =
            \mathbf{C}_{\mathrm{Rayleigh}}
            +
            \mathbf{C}_{\mathrm{FVD}}.$$
          </div>
          <p>
            The response-history quantities used for comparison are interstory drift ratio and absolute
            floor acceleration:
          </p>
          <div class="eq">
            $$\mathrm{IDR}_i(t)=\frac{u_i(t)-u_{i-1}(t)}{h_i},
            \qquad
            \ddot u_{i,\mathrm{abs}}(t)=\ddot u_i(t)+\ddot u_g(t).$$
          </div>
          <div class="matrix-grid">
            @@CFVD_MATRIX@@
            @@CTOTAL_MATRIX@@
          </div>
          @@RESPONSE_SUMMARY@@
          @@CONCLUSIONS@@
          @@RESPONSE_PLOTS@@
        </section>

        <section class="box">
          <h3>References used by this dashboard</h3>
          <p>
            The fictitious-spring sizing procedure and passive supplemental damping terminology follow
            Christopoulos and Filiatrault. The modal analysis, Rayleigh damping calibration, and
            response-history formulation follow standard structural dynamics notation as presented by Chopra.
          </p>
          <ul class="note-list">
            <li>Christopoulos, C. and Filiatrault, A. (2006). <em>Principles of Passive Supplemental Damping and Seismic Isolation</em>. IUSS Press, Pavia.</li>
            <li>Chopra, A. K. (2017). <em>Dynamics of Structures: Theory and Applications to Earthquake Engineering</em>, 5th ed. Pearson.</li>
            <li>Constantinou, M. C. and Symans, M. D. (1992). <em>Experimental and Analytical Investigation of Seismic Response of Structures with Supplemental Fluid Viscous Dampers</em>. NCEER-92-0032.</li>
          </ul>
        </section>
      </div>
    </section>
  </div>
</body>
</html>
"""
        replacements = {
            "@@PROBLEM_SUMMARY@@": self._problem_summary_table(),
            "@@MODAL_SUMMARY@@": self._modal_summary_table(),
            "@@M_MATRIX@@": self._matrix_display("Mass matrix", r"\mathbf{M}", self.matrices.mass, r"\mathrm{kg}"),
            "@@K_MATRIX@@": self._matrix_display("Stiffness matrix", r"\mathbf{K}", self.matrices.stiffness,
                                                 r"\mathrm{N/m}"),
            "@@CR_MATRIX@@": self._matrix_display("Rayleigh damping matrix", r"\mathbf{C}_{\mathrm{Rayleigh}}",
                                                  self.matrices.rayleigh_damping, r"\mathrm{N\,s/m}"),
            "@@NOTATION_TABLE@@": self._notation_table(),
            "@@DESIGN_SUMMARY@@": self._design_summary_table(),
            "@@KHAT_MATRIX@@": self._matrix_display("Converged fictitious story stiffnesses", r"\widehat{\mathbf{k}}_0",
                                                    self.design.converged_fictitious_story_stiffness.reshape(1, -1),
                                                    r"\mathrm{N/m}"),
            "@@CH_MATRIX@@": self._matrix_display("Effective horizontal story damping", r"\mathbf{c}_h",
                                                  self.design.effective_horizontal_story_damping.reshape(1, -1),
                                                  r"\mathrm{N\,s/m}"),
            "@@CD_MATRIX@@": self._matrix_display("Per-device FVD coefficients", r"\mathbf{C}_d",
                                                  self.design.damper_coefficient_per_device.reshape(1, -1),
                                                  r"\mathrm{N\,s/m}"),
            "@@FICTITIOUS_PLOT@@": plotly_loaded,
            "@@CFVD_MATRIX@@": self._matrix_display("FVD damping matrix", r"\mathbf{C}_{\mathrm{FVD}}",
                                                    self.design.fvd_damping_matrix, r"\mathrm{N\,s/m}"),
            "@@CTOTAL_MATRIX@@": self._matrix_display("Total damping matrix", r"\mathbf{C}_{\mathrm{total}}",
                                                      self.design.total_damping_matrix, r"\mathrm{N\,s/m}"),
            "@@RESPONSE_SUMMARY@@": response_summary,
            "@@CONCLUSIONS@@": conclusions,
            "@@RESPONSE_PLOTS@@": response_plots,
        }
        for key, value in replacements.items():
            template = template.replace(key, value)
        return dedent(template).strip()

    def _problem_summary_table(self) -> str:
        d = self.data
        rows = [
            ("Floor area", f"{d.floor_area:.3f}", "m²"),
            ("Lateral columns per story", f"{d.n_lateral_columns_per_story:d}", "-"),
            ("Story height", f"{d.story_height:.3f}", "m"),
            ("Story stiffness", f"{self.matrices.story_stiffness[0]:.6e}", "N/m"),
            ("Original first-mode Rayleigh damping", f"{100.0 * d.rayleigh_damping_ratio:.3f}", "%"),
            ("Supplemental FVD first-mode target", f"{100.0 * d.supplemental_fvd_first_mode_damping_ratio:.3f}", "%"),
            ("Retrofitted first-mode damping", f"{100.0 * self.design.retrofitted_modal_damping[0]:.3f}", "%"),
        ]
        return make_key_value_table(rows)

    def _notation_table(self) -> str:
        rows = [
            (r"\(T_1,\omega_1,\boldsymbol{\phi}_1\)",
             "First period, circular frequency, and mode shape of the original unbraced structure."),
            (r"\(\widehat{(\cdot)}\)",
             "A hat denotes a quantity belonging to the auxiliary fictitious-spring structure."),
            (r"\(\widehat{T}_1\)", "Target first period assigned to the auxiliary fictitious-spring structure."),
            (r"\(\widehat{T}_{1,\mathrm{tr}}^{(r)}\)", "Trial first period at iteration r."),
            (r"\(\widehat{k}_{0,j}^{(r)}\)",
             "Trial fictitious horizontal stiffness assigned to location or story j at iteration r."),
            (r"\(\widehat{\mathbf{K}}_0^{(r)}\)",
             "Global stiffness matrix assembled from the trial fictitious springs."),
            (r"\(c_{h,j}\)", "Effective horizontal viscous coefficient required at location or story j."),
            (r"\(C_{d,j}\)", "Physical coefficient of one FVD at location or story j."),
        ]
        out = [
            '<div class="summary-table-wrap"><table class="summary-table"><thead><tr><th scope="col">Symbol</th><th scope="col">Meaning</th></tr></thead><tbody>']
        for symbol, meaning in rows:
            out.append(f"<tr><td>{symbol}</td><td>{escape_html(meaning)}</td></tr>")
        out.append("</tbody></table></div>")
        return "".join(out)

    def _modal_summary_table(self) -> str:
        rows = [
            '<div class="summary-table-wrap"><table class="summary-table"><thead><tr>'
            '<th scope="col">Mode</th><th scope="col">ω [rad/s]</th><th scope="col">T [s]</th>'
            '<th scope="col">Γ [-]</th><th scope="col">M_eff / M_total [%]</th>'
            '<th scope="col">ξ original [%]</th><th scope="col">ξ FVD [%]</th><th scope="col">ξ total [%]</th>'
            '</tr></thead><tbody>'
        ]
        total_mass = float(np.sum(np.diag(self.matrices.mass)))
        for i in range(self.modal.omega.size):
            rows.append(
                "<tr>"
                f"<td>{i + 1}</td>"
                f"<td>{self.modal.omega[i]:.6f}</td>"
                f"<td>{self.modal.periods[i]:.6f}</td>"
                f"<td>{self.modal.participation_factors[i]:.6f}</td>"
                f"<td>{100.0 * self.modal.effective_modal_mass[i] / total_mass:.4f}</td>"
                f"<td>{100.0 * self.matrices.original_modal_damping[i]:.4f}</td>"
                f"<td>{100.0 * self.design.fvd_modal_damping[i]:.4f}</td>"
                f"<td>{100.0 * self.design.retrofitted_modal_damping[i]:.4f}</td>"
                "</tr>"
            )
        rows.append("</tbody></table></div>")
        return "".join(rows)

    def _design_summary_table(self) -> str:
        rows = [
            ("Original first period", f"{self.modal.periods[0]:.6f}", "s"),
            ("Target auxiliary braced first period", f"{self.design.target_braced_fundamental_period:.6f}", "s"),
            ("Converged fictitious story stiffness", f"{self.design.converged_fictitious_story_stiffness[0]:.6e}",
             "N/m"),
            ("Effective horizontal story damping", f"{self.design.effective_horizontal_story_damping[0]:.6e}", "N s/m"),
            ("Coefficient per physical FVD", f"{self.design.damper_coefficient_per_device[0]:.6e}", "N s/m"),
            ("FVDs per story", f"{self.data.dampers_per_story:d}", "-"),
            ("Horizontal projection factor γ", f"{self.data.damper_horizontal_projection_factor:.3f}", "-"),
            ("Initial fictitious stiffness scale", f"{self.data.initial_fictitious_story_stiffness_scale:.3f}", "-"),
            ("Fictitious-spring iterations", f"{len(self.design.iteration_history):d}", "-"),
        ]
        return make_key_value_table(rows)

    def _conclusions_section(
            self,
            original: LinearMDOFResult,
            retrofitted: LinearMDOFResult,
    ) -> str:
        """Return a concise conclusions block based on the computed response measures."""
        original_idr = 100.0 * original.peak_interstory_drift_ratio
        retrofitted_idr = 100.0 * retrofitted.peak_interstory_drift_ratio
        original_acc = original.peak_absolute_acceleration_g
        retrofitted_acc = retrofitted.peak_absolute_acceleration_g
        original_disp = 1.0e3 * np.max(np.abs(original.displacement), axis=0)
        retrofitted_disp = 1.0e3 * np.max(np.abs(retrofitted.displacement), axis=0)

        def reduction(original_value: float, retrofitted_value: float) -> float:
            return 100.0 * (original_value - retrofitted_value) / max(original_value, 1.0e-12)

        peak_story = int(np.argmax(original_idr)) + 1
        peak_floor = int(np.argmax(original_acc)) + 1
        peak_floor_name = "Roof" if peak_floor == 3 else f"Floor {peak_floor}"
        peak_disp_dof = int(np.argmax(original_disp)) + 1
        peak_disp_name = "Roof (u3)" if peak_disp_dof == 3 else f"Floor {peak_disp_dof} (u{peak_disp_dof})"

        max_idr_original = float(np.max(original_idr))
        max_idr_retrofitted = float(np.max(retrofitted_idr))
        max_acc_original = float(np.max(original_acc))
        max_acc_retrofitted = float(np.max(retrofitted_acc))
        max_disp_original = float(np.max(original_disp))
        max_disp_retrofitted = float(np.max(retrofitted_disp))

        return (
            '<h3>Conclusions</h3>'
            '<div class="note">'
            '<ul class="note-list">'
            f'<li>The largest peak interstory drift ratio occurs at Story {peak_story}. '
            f'Its value decreases from {max_idr_original:.4f}% in the original structure '
            f'to {max_idr_retrofitted:.4f}% after retrofit, which corresponds to a reduction of '
            f'{reduction(max_idr_original, max_idr_retrofitted):.2f}%.</li>'
            f'<li>The largest peak relative floor displacement occurs at {peak_disp_name}. '
            f'Its value decreases from {max_disp_original:.3f} mm to {max_disp_retrofitted:.3f} mm, '
            f'which corresponds to a reduction of {reduction(max_disp_original, max_disp_retrofitted):.2f}%.</li>'
            f'<li>The largest peak absolute acceleration occurs at the {peak_floor_name}. '
            f'Its value decreases from {max_acc_original:.4f} g to {max_acc_retrofitted:.4f} g, '
            f'which corresponds to a reduction of {reduction(max_acc_original, max_acc_retrofitted):.2f}%.</li>'
            '<li>Based on the peak-response measures reported here, the fluid viscous damper retrofit '
            'is effective in reducing deformation demand and also reduces floor acceleration demand '
            'for the analyzed Sylmar excitation.</li>'
            '</ul>'
            '</div>'
        )

    def _response_summary_table(
            self,
            original: LinearMDOFResult,
            retrofitted: LinearMDOFResult,
    ) -> str:
        """Return a peak-response table in the same style as the reference dashboard."""

        def peak_abs_and_time(values: np.ndarray, time: np.ndarray) -> tuple[float, float]:
            peak_index = int(np.argmax(np.abs(values)))
            return float(abs(values[peak_index])), float(time[peak_index])

        rows = []

        for story_index in range(3):
            original_peak, original_time = peak_abs_and_time(
                100.0 * original.interstory_drift_ratio[:, story_index],
                original.time,
            )
            retrofit_peak, retrofit_time = peak_abs_and_time(
                100.0 * retrofitted.interstory_drift_ratio[:, story_index],
                retrofitted.time,
            )
            reduction = 100.0 * (original_peak - retrofit_peak) / max(original_peak, 1.0e-12)
            rows.append(
                "<tr>"
                f"<td>Story {story_index + 1}</td>"
                "<td>Peak |IDR|</td>"
                f"<td>{original_peak:.4f}</td>"
                f"<td>{original_time:.3f}</td>"
                f"<td>{retrofit_peak:.4f}</td>"
                f"<td>{retrofit_time:.3f}</td>"
                f"<td>{reduction:+.2f}</td>"
                "<td>%</td>"
                "</tr>"
            )

        floor_labels = ("Floor 1", "Floor 2", "Roof")
        for floor_index, floor_label in enumerate(floor_labels):
            original_peak, original_time = peak_abs_and_time(
                original.absolute_acceleration[:, floor_index] / G_SI,
                original.time,
            )
            retrofit_peak, retrofit_time = peak_abs_and_time(
                retrofitted.absolute_acceleration[:, floor_index] / G_SI,
                retrofitted.time,
            )
            reduction = 100.0 * (original_peak - retrofit_peak) / max(original_peak, 1.0e-12)
            rows.append(
                "<tr>"
                f"<td>{floor_label}</td>"
                "<td>Peak |absolute acceleration|</td>"
                f"<td>{original_peak:.4f}</td>"
                f"<td>{original_time:.3f}</td>"
                f"<td>{retrofit_peak:.4f}</td>"
                f"<td>{retrofit_time:.3f}</td>"
                f"<td>{reduction:+.2f}</td>"
                "<td>g</td>"
                "</tr>"
            )

        displacement_labels = ("Floor 1 (u1)", "Floor 2 (u2)", "Roof (u3)")
        for floor_index, floor_label in enumerate(displacement_labels):
            original_peak, original_time = peak_abs_and_time(
                1.0e3 * original.displacement[:, floor_index],
                original.time,
            )
            retrofit_peak, retrofit_time = peak_abs_and_time(
                1.0e3 * retrofitted.displacement[:, floor_index],
                retrofitted.time,
            )
            reduction = 100.0 * (original_peak - retrofit_peak) / max(original_peak, 1.0e-12)
            rows.append(
                "<tr>"
                f"<td>{floor_label}</td>"
                "<td>Peak |relative displacement|</td>"
                f"<td>{original_peak:.3f}</td>"
                f"<td>{original_time:.3f}</td>"
                f"<td>{retrofit_peak:.3f}</td>"
                f"<td>{retrofit_time:.3f}</td>"
                f"<td>{reduction:+.2f}</td>"
                "<td>mm</td>"
                "</tr>"
            )

        return (
                '<h3>Peak response summary</h3>'
                '<div class="summary-table-wrap"><table class="summary-table"><thead><tr>'
                "<th scope='col'>Location</th><th scope='col'>Quantity</th>"
                "<th scope='col'>Original peak</th><th scope='col'>t_peak [s]</th>"
                "<th scope='col'>Retrofitted peak</th><th scope='col'>t_peak [s]</th>"
                "<th scope='col'>Reduction [%]</th><th scope='col'>Unit</th>"
                "</tr></thead><tbody>"
                + "".join(rows)
                + "</tbody></table></div>"
        )

    @staticmethod
    def _matrix_display(title: str, symbol: str, matrix: np.ndarray, units: str) -> str:
        latex = matrix_to_latex(symbol=symbol, matrix=matrix, units=units)
        return (
            '<div class="matrix-card">'
            f'<h4>{escape_html(title)}</h4>'
            f'<div class="eq">$${latex}$$</div>'
            '</div>'
        )


def make_key_value_table(rows: list[tuple[str, str, str]]) -> str:
    out = [
        '<div class="summary-table-wrap"><table class="summary-table"><thead><tr>'
        '<th scope="col">Quantity</th><th scope="col">Value</th><th scope="col">Unit</th>'
        '</tr></thead><tbody>'
    ]
    for key, value, unit in rows:
        out.append(
            "<tr>"
            f"<td>{escape_html(key)}</td>"
            f"<td>{escape_html(value)}</td>"
            f"<td>{escape_html(unit)}</td>"
            "</tr>"
        )
    out.append("</tbody></table></div>")
    return "".join(out)


def format_latex_number(value: float) -> str:
    """Format a number for compact MathJax matrix display."""
    value = float(value)
    if abs(value) < 5.0e-13:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0 ** exponent)
    if -2 <= exponent <= 3:
        return f"{value:.5g}"
    return f"{mantissa:.4f}\\times 10^{{{exponent}}}"


def matrix_to_latex(symbol: str, matrix: np.ndarray, units: str) -> str:
    """Return a LaTeX bmatrix string with units."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix_to_latex expects a two-dimensional array.")
    rows = []
    for i in range(matrix.shape[0]):
        row = " & ".join(format_latex_number(matrix[i, j]) for j in range(matrix.shape[1]))
        rows.append(row)
    body = "\\\\\n".join(rows)
    return f"{symbol}=\\begin{{bmatrix}}\n{body}\n\\end{{bmatrix}}\\ {units}"


def escape_html(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def print_numeric_summary(
        data: FvdProblemData,
        matrices: StructuralMatrices,
        modal: ModalProperties,
        design: FvdDesignResult,
) -> None:
    print("\nCE223 FVD retrofit summary")
    print("-" * 72)
    print(f"Floor area: {data.floor_area:.3f} m^2")
    print(f"Story stiffness ks: {matrices.story_stiffness[0]:.6e} N/m")
    print(f"Masses: {np.diag(matrices.mass)} kg")
    print(f"Natural circular frequencies: {modal.omega} rad/s")
    print(f"Natural periods: {modal.periods} s")
    print(
        f"Rayleigh coefficients: a_M = {matrices.rayleigh_mass_coefficient:.6e} s^-1, a_K = {matrices.rayleigh_stiffness_coefficient:.6e} s")
    print(f"Supplemental FVD first-mode damping ratio: {design.supplemental_fvd_first_mode_damping_ratio:.6f}")
    print(f"Target auxiliary braced fundamental period: {design.target_braced_fundamental_period:.6f} s")
    print(f"Converged fictitious story stiffness: {design.converged_fictitious_story_stiffness} N/m")
    print(f"Effective horizontal story damping: {design.effective_horizontal_story_damping} N s/m")
    print(f"Damper coefficient per device: {design.damper_coefficient_per_device} N s/m")
    print(f"Original modal damping ratios: {matrices.original_modal_damping}")
    print(f"FVD modal damping ratios: {design.fvd_modal_damping}")
    print(f"Retrofitted modal damping ratios: {design.retrofitted_modal_damping}")


def main() -> None:
    data = FvdProblemData()
    matrices, modal = ShearBuildingBuilder.build(data)
    design = FvdDesigner.design(data, matrices, modal)

    print_numeric_summary(data, matrices, modal, design)

    record: GroundMotionRecord | None = None
    original_result: LinearMDOFResult | None = None
    retrofitted_result: LinearMDOFResult | None = None
    missing_motion_message: str | None = None

    try:
        sylmar_path = GroundMotionLoader.find_sylmar_path()
        record = GroundMotionLoader.load_acceleration_file(sylmar_path, name="Sylmar")
        original_result = LinearNewmarkMDOFSolver.solve(
            label="Original",
            record=record,
            mass=matrices.mass,
            damping=matrices.rayleigh_damping,
            stiffness=matrices.stiffness,
            story_height=matrices.story_height,
        )
        retrofitted_result = LinearNewmarkMDOFSolver.solve(
            label="Retrofitted",
            record=record,
            mass=matrices.mass,
            damping=design.total_damping_matrix,
            stiffness=matrices.stiffness,
            story_height=matrices.story_height,
        )
        print("\nPeak response quantities")
        print("-" * 72)
        print(f"Original peak IDR [%]: {100.0 * original_result.peak_interstory_drift_ratio}")
        print(f"Retrofitted peak IDR [%]: {100.0 * retrofitted_result.peak_interstory_drift_ratio}")
        print(f"Original peak abs. acceleration [g]: {original_result.peak_absolute_acceleration_g}")
        print(f"Retrofitted peak abs. acceleration [g]: {retrofitted_result.peak_absolute_acceleration_g}")

    except FileNotFoundError as exc:
        missing_motion_message = str(exc)
        print(f"\n{missing_motion_message}")

    report = HtmlReportBuilder(data, matrices, modal, design).build(
        record=record,
        original_result=original_result,
        retrofitted_result=retrofitted_result,
        missing_motion_message=missing_motion_message,
    )
    OUTPUT_HTML.write_text(report, encoding="utf-8")
    print(f"\nWrote {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
