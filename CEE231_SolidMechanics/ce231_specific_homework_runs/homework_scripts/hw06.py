
import numpy as np
import plotly.graph_objects as go
from typing import Iterable, Tuple


def _compute_compliance_matrix(stiffness_matrix: np.ndarray) -> np.ndarray:
	"""Return compliance matrix S given a 6x6 stiffness matrix C (Voigt).

	The algorithm strictly follows the original implementation (matrix inverse).
	"""
	return np.linalg.inv(stiffness_matrix)


def _generate_spherical_grid(num: int = 300) -> Tuple[np.ndarray, np.ndarray]:
	"""Generate spherical angles (phis, thetas) for uniform sampling.

	- phis in [0, 2π) with endpoint excluded
	- thetas in [0, π]
	"""
	phis = np.linspace(0, 2 * np.pi, num, endpoint=False)
	thetas = np.linspace(0, np.pi, num)
	return phis, thetas


def _direction_vector(theta: float, phi: float) -> np.ndarray:
	"""Unit direction vector for given spherical angles (physics convention)."""
	return np.array([
		np.sin(theta) * np.cos(phi),
		np.sin(theta) * np.sin(phi),
		np.cos(theta),
	])


def _stress_voigt_from_direction(direction: np.ndarray) -> np.ndarray:
	"""Return unit-direction stress in Voigt notation for modulus probing."""
	return np.array([
		direction[0] ** 2,
		direction[1] ** 2,
		direction[2] ** 2,
		direction[1] * direction[2],
		direction[2] * direction[0],
		direction[0] * direction[1],
	])


def _strain_tensor_from_voigt(strain_voigt: np.ndarray) -> np.ndarray:
	"""Convert 6-vector (Voigt) strain to 3x3 symmetric tensor.

	Uses engineering shear to tensor shear conversion (γ = 2ε).
	"""
	return np.array([
		[strain_voigt[0], strain_voigt[5] / 2, strain_voigt[4] / 2],
		[strain_voigt[5] / 2, strain_voigt[1], strain_voigt[3] / 2],
		[strain_voigt[4] / 2, strain_voigt[3] / 2, strain_voigt[2]],
	])


def _compute_directional_youngs_modulus(S: np.ndarray, theta: float, phi: float) -> Tuple[float, float, float, float, np.ndarray]:
	"""Compute directional Young's modulus and mapped 3D point.

	Returns (x, y, z, E) where (x, y, z) is the point at radius E in direction (θ, φ).
	"""
	d = _direction_vector(theta, phi)
	stress_v = _stress_voigt_from_direction(d)
	strain_v = S @ stress_v
	strain_t = _strain_tensor_from_voigt(strain_v)
	strain_dd = d @ strain_t @ d
	E = 1.0 / strain_dd
	return (
		float(E * np.sin(theta) * np.cos(phi)),
		float(E * np.sin(theta) * np.sin(phi)),
		float(E * np.cos(theta)),
		float(E),
		d,
	)


def plot_directional_youngs_modulus(C: np.ndarray, resolution: int = 300, colorscale: str = "icefire", material_name: str | None = None) -> None:
	"""Plot the directional Young's modulus as a 3D point cloud.

	This function preserves the original algorithm and numerical behavior while
	organizing the computation into smaller, well‑named helpers.
	"""
	S = _compute_compliance_matrix(C)
	phis, thetas = _generate_spherical_grid(resolution)

	# Accumulate point cloud
	x, y, z, Ed = [], [], [], []
	# metadata containers for richer hover information
	theta_list, phi_list = [], []
	dx_list, dy_list, dz_list = [], [], []
	for phi in phis:
		for theta in thetas:
			x_i, y_i, z_i, E_i, d = _compute_directional_youngs_modulus(S, theta, phi)
			x.append(x_i); y.append(y_i); z.append(z_i); Ed.append(E_i)
			theta_list.append(theta); phi_list.append(phi)
			dx_list.append(float(d[0])); dy_list.append(float(d[1])); dz_list.append(float(d[2]))

	# Convert to arrays for plotting (shape consistent with original)
	x, y, z, Ed = map(np.array, (x, y, z, Ed))
	x = x.flatten(); y = y.flatten(); z = z.flatten()

	customdata = np.stack([
		Ed,
		np.degrees(theta_list),
		np.degrees(phi_list),
		dx_list,
		dy_list,
		dz_list,
	], axis=1)
	fig = go.Figure(data=[
		go.Scatter3d(
			x=x, y=y, z=z,
			mode='markers',
			marker=dict(size=2.5, color=Ed, colorscale=colorscale, colorbar=dict(title='E (GPa)'), opacity=1.0),
			customdata=customdata,
			hovertemplate=(
				"<b>E</b>: %{customdata[0]:.3f} GPa"+
				"<br><b>θ</b>: %{customdata[1]:.1f}°  <b>φ</b>: %{customdata[2]:.1f}°"+
				"<br><b>d</b>: (%{customdata[3]:.3f}, %{customdata[4]:.3f}, %{customdata[5]:.3f})<extra></extra>"
			),
		)
	])

	fig.update_layout(
		title=f"Directional Young’s Modulus (Point Cloud){' — ' + material_name if material_name else ''}",
		scene=dict(xaxis_title="e₁", yaxis_title="e₂", zaxis_title="e₃", aspectmode="data"),
		template="plotly_white",
	)
	fig.show()


# Backwards‑compatible API used in the original script
def run_assignment_1(C: np.ndarray) -> None:
	plot_directional_youngs_modulus(C)

def _cubic_stiffness(c11: float, c12: float, c44: float) -> np.ndarray:
	"""Return 6x6 cubic crystal stiffness matrix (Voigt notation)."""
	return np.array([
		[c11, c12, c12, 0, 0, 0],
		[c12, c11, c12, 0, 0, 0],
		[c12, c12, c11, 0, 0, 0],
		[0, 0, 0, c44, 0, 0],
		[0, 0, 0, 0, c44, 0],
		[0, 0, 0, 0, 0, c44],
	], dtype=float)


def main() -> None:
	# Example 1: Fe

    c11, c12, c44 = 231.4, 134.7, 116.4
    C = _cubic_stiffness(c11, c12, c44)
    plot_directional_youngs_modulus(C, material_name="Fe")

	# Example 2: Nb
    c11, c12, c44 = 240.2, 125.6, 28.2
    C = _cubic_stiffness(c11, c12, c44)
    plot_directional_youngs_modulus(C, material_name="Nb")

	# Example 3 (general symmetric matrix already in Voigt form): Niti Alloy
    C = np.array([
		[246, 131, 96, 0, -4, 0],
		[131, 236, 130, 0, -2, 0],
		[96, 130, 196, 0, -2, 0],
		[0, 0, 0, 84, 0, -1],
		[-4, -2, -2, 0, 5, 0],
		[0, 0, 0, -1, 0, 92],
	], dtype=float)
    plot_directional_youngs_modulus(C, material_name="NiTi Alloy")


if __name__ == "__main__":
	main()