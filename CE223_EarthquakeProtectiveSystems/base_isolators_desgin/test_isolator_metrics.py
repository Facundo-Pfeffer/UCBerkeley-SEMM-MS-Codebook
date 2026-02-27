import numpy as np

from isolator_metrics import compute_loop_area


def _rectangle_cloud(width: float = 1.0, height: float = 2.0, n: int = 200) -> np.ndarray:
    """
    Construct a simple rectangular hysteresis loop as a point cloud.

    Bottom branch: F = 0 over u in [0, width]
    Top branch:    F = height over u in [0, width]

    The true enclosed area is width * height.
    """
    u_bottom = np.linspace(0.0, width, n)
    f_bottom = np.zeros_like(u_bottom)
    u_top = np.linspace(0.0, width, n)
    f_top = np.full_like(u_top, height)

    u = np.concatenate([u_bottom, u_top])
    f = np.concatenate([f_bottom, f_top])
    pts = np.column_stack([u, f])

    rng = np.random.default_rng(0)
    rng.shuffle(pts, axis=0)
    return pts


def test_rectangle_single_cycle_area_matches_analytic():
    """Area between top and bottom branches matches width * height."""
    width = 1.0
    height = 2.0
    pts = _rectangle_cloud(width=width, height=height, n=400)

    wd = compute_loop_area(pts)
    assert np.isclose(wd, width * height, rtol=1e-2, atol=1e-3)


def test_rectangle_multiple_cycles_same_energy():
    """Duplicating the loop (multiple overlaid cycles) does not change W_D."""
    width = 1.0
    height = 2.0
    base_pts = _rectangle_cloud(width=width, height=height, n=400)

    # Simulate several overlaid cycles by concatenating and shuffling.
    pts = np.vstack([base_pts, base_pts, base_pts])
    rng = np.random.default_rng(1)
    rng.shuffle(pts, axis=0)

    wd = compute_loop_area(pts)
    assert np.isclose(wd, width * height, rtol=1e-2, atol=1e-3)

