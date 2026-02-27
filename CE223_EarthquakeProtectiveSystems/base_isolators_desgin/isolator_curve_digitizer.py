from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class AxisLimits:
    """
    Physical axis limits for a 2D plot.

    x corresponds to horizontal (displacement), y to vertical (force).
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return self.x_min, self.x_max, self.y_min, self.y_max


class IsolatorCurveDigitizer:
    """
    Digitizer for scanned isolator force–displacement plots.

    The workflow is:
    - Load a PNG image.
    - Detect the main plotting rectangle automatically using dark-pixel density.
    - Map dark pixels inside that rectangle to physical (x, y) coordinates
      using user-specified axis limits.
    - Optionally downsample or shuffle the resulting cloud of points.
    """

    def __init__(
        self,
        axis_limits: AxisLimits,
        *,
        dark_threshold: int | None = None,
        row_frac_threshold: float = 0.15,
        col_frac_threshold: float = 0.10,
        border_trim_pixels: int = 4,
        min_bin_density: int | None = None,
        min_component_area: int = 200,
    ) -> None:
        """
        Parameters
        ----------
        axis_limits:
            Physical axis limits of the plot.
        dark_threshold:
            Optional grayscale threshold (0–255). Pixels <= threshold are
            considered "ink". If None, a data-driven threshold is computed.
        row_frac_threshold, col_frac_threshold:
            Fractions of the maximum dark-pixel count used to decide which
            rows/columns belong to the plotting region.
        border_trim_pixels:
            Number of pixels to trim from each side of the detected plot
            rectangle to avoid axis/frame lines.
        min_bin_density:
            Optional minimum number of points per (x, y) histogram bin for an
            additional data-space filter. If None (default), that filter is
            disabled.
        min_component_area:
            Minimum number of pixels for a connected dark region in image
            space; smaller regions (digits, specks) are removed before mapping
            to (u, F).
        """
        self.axis_limits = axis_limits
        self.dark_threshold = dark_threshold
        self.row_frac_threshold = row_frac_threshold
        self.col_frac_threshold = col_frac_threshold
        self.border_trim_pixels = border_trim_pixels
        self.min_bin_density = min_bin_density
        self.min_component_area = min_component_area

    # Public API -----------------------------------------------------

    def digitize(
        self,
        image_path: str | Path,
        *,
        max_points: int | None = None,
        shuffle: bool = True,
        known_max_displacement: float | None = None,
    ) -> np.ndarray:
        """
        Digitize a force–displacement plot image into a cloud of (x, y) points.

        The curve's pixel bounding box is mapped to the physical axis limits, so the
        leftmost/rightmost (and top/bottom) curve pixels get exactly axis_min/axis_max.
        This avoids underestimating U0/P0 when the curve does not reach the plot edges.

        Parameters
        ----------
        image_path:
            Path to the PNG image.
        max_points:
            Optional maximum number of points to return. If not None and the
            detected pixel cloud is larger, points are randomly subsampled.
        shuffle:
            If True, shuffle point order before optional subsampling.

        Returns
        -------
        points:
            Array of shape (N, 2) with columns [x, y] in physical units.
        """
        image_path = Path(image_path)
        gray = _load_grayscale(image_path)

        dark_mask = self._compute_dark_mask(gray)
        y0, y1, x0, x1 = self._detect_plot_rectangle(dark_mask)

        # Apply border trim to avoid axes/frame lines.
        y0 = max(y0 + self.border_trim_pixels, 0)
        y1 = min(y1 - self.border_trim_pixels, dark_mask.shape[0] - 1)
        x0 = max(x0 + self.border_trim_pixels, 0)
        x1 = min(x1 - self.border_trim_pixels, dark_mask.shape[1] - 1)
        if y1 <= y0 or x1 <= x0:
            raise RuntimeError("Plot rectangle collapsed after border trimming.")

        cropped_mask = dark_mask[y0 : y1 + 1, x0 : x1 + 1]

        # Suppress axis lines (rows/cols with very high dark density)
        cropped_mask = _suppress_axes(cropped_mask)
        # Remove small connected components (digits, specks) in image space
        cropped_mask = _remove_small_components(cropped_mask, self.min_component_area)

        ys, xs = np.nonzero(cropped_mask)
        if xs.size == 0:
            raise RuntimeError(f"No dark pixels detected inside plot for {image_path}")

        # Map pixel coordinates to physical coordinates.
        width = x1 - x0
        height = y1 - y0
        x_min, x_max, y_min, y_max = self.axis_limits.as_tuple()

        # Use the curve's bounding box in pixels so the curve spans the full axis range.
        # Otherwise the curve often doesn't reach the plot edges (inner margin), and we
        # underestimate U0 / P0.
        xs_lo, xs_hi = int(xs.min()), int(xs.max())
        ys_lo, ys_hi = int(ys.min()), int(ys.max())
        span_x = xs_hi - xs_lo
        span_y = ys_hi - ys_lo
        if span_x < 1:
            span_x = 1
        if span_y < 1:
            span_y = 1

        x_data = x_min + (xs.astype(float) - xs_lo) / span_x * (x_max - x_min)
        # Image y increases downward; data y increases upward.
        y_data = y_max - (ys.astype(float) - ys_lo) / span_y * (y_max - y_min)

        pts = np.column_stack([x_data, y_data])

        # Optional global rescaling in x if a trusted maximum displacement is known.
        if known_max_displacement is not None:
            current_max = float(np.max(np.abs(pts[:, 0])))
            if current_max > 0.0:
                scale = float(known_max_displacement) / current_max
                pts[:, 0] *= scale

        if shuffle:
            rng = np.random.default_rng(seed=0)
            rng.shuffle(pts, axis=0)

        if max_points is not None and pts.shape[0] > max_points:
            pts = pts[:max_points]

        return pts

    def digitize_to_csv(
        self,
        image_path: str | Path,
        csv_path: str | Path | None = None,
        *,
        max_points: int | None = None,
        shuffle: bool = True,
        header: str = "disp_in,force_kip",
    ) -> np.ndarray:
        """
        Digitize an image and save the point cloud to CSV.

        Returns the same NumPy array as :meth:`digitize` so callers can
        both persist to disk and work numerically in memory.
        """
        pts = self.digitize(
            image_path,
            max_points=max_points,
            shuffle=shuffle,
        )

        image_path = Path(image_path)
        if csv_path is None:
            csv_path = image_path.with_suffix(".csv")
        csv_path = Path(csv_path)

        _save_points_csv(pts, csv_path, header=header)
        return pts

    # Internal helpers -----------------------------------------------

    def _compute_dark_mask(self, gray: np.ndarray) -> np.ndarray:
        """Return boolean mask of where the image is considered 'ink'."""
        if self.dark_threshold is not None:
            thr = int(self.dark_threshold)
        else:
            # Data-driven threshold:
            # use a low percentile of intensities in the central band so that
            # only the darkest ink strokes (axes + curves) are selected.
            h, w = gray.shape
            h0, h1 = int(0.1 * h), int(0.9 * h)
            w0, w1 = int(0.1 * w), int(0.9 * w)
            central = gray[h0:h1, w0:w1]
            thr = int(np.percentile(central, 5))
        return gray <= thr

    def _detect_plot_rectangle(self, dark_mask: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Detect the axes rectangle by looking for contiguous bands of dark pixels.

        Returns
        -------
        (y0, y1, x0, x1) in pixel indices.
        """
        row_counts = dark_mask.sum(axis=1)
        col_counts = dark_mask.sum(axis=0)

        def _band_from_counts(counts: np.ndarray, frac_threshold: float) -> Tuple[int, int]:
            if counts.max() == 0:
                raise RuntimeError("No dark pixels in image; cannot locate plot.")
            mask = counts >= frac_threshold * counts.max()
            if not mask.any():
                # Fall back to using all rows/cols if heuristic fails.
                return 0, len(counts) - 1
            idx = np.where(mask)[0]
            return int(idx[0]), int(idx[-1])

        y0, y1 = _band_from_counts(row_counts, self.row_frac_threshold)
        x0, x1 = _band_from_counts(col_counts, self.col_frac_threshold)
        return y0, y1, x0, x1


def _load_grayscale(path: Path) -> np.ndarray:
    """Load an image as a NumPy grayscale array in [0, 255]."""
    img = Image.open(path).convert("L")
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected grayscale image at {path}, got shape {arr.shape}")
    return arr


def _suppress_axes(mask: np.ndarray, row_frac: float = 0.4, col_frac: float = 0.4) -> np.ndarray:
    """
    Remove rows/columns that look like plot axes (very dense dark pixels).

    Parameters
    ----------
    mask:
        Boolean mask of dark pixels inside the cropped plotting window.
    row_frac, col_frac:
        Fraction of the maximum row/column count used to flag an axis-like
        band. Rows/columns above this threshold (and a 1-pixel halo) are
        zeroed out.
    """
    h, w = mask.shape
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)

    if row_counts.max() > 0:
        row_threshold = row_frac * row_counts.max()
        axis_rows = np.where(row_counts >= row_threshold)[0]
        for r in axis_rows:
            r0 = max(0, r - 1)
            r1 = min(h, r + 2)
            mask[r0:r1, :] = False

    if col_counts.max() > 0:
        col_threshold = col_frac * col_counts.max()
        axis_cols = np.where(col_counts >= col_threshold)[0]
        for c in axis_cols:
            c0 = max(0, c - 1)
            c1 = min(w, c + 2)
            mask[:, c0:c1] = False

    return mask


def _suppress_isolated(mask: np.ndarray, min_neighbors: int = 2) -> np.ndarray:
    """
    Remove dark pixels that do not have enough dark neighbours.

    This helps filter out small specks and noise (e.g., scan artifacts)
    while preserving continuous strokes such as the hysteresis curve.
    """
    if not mask.any():
        return mask

    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    # Count dark pixels in the 3x3 neighbourhood (including the pixel itself)
    neigh = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return mask & (neigh >= min_neighbors)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    Remove connected components in the image mask that are too small.

    This targets annotation digits and specks, which occupy relatively few
    pixels compared to the main hysteresis curves.
    """
    if min_area <= 0 or not mask.any():
        return mask

    h, w = mask.shape
    labels = -np.ones_like(mask, dtype=int)

    for i in range(h):
        for j in range(w):
            if not mask[i, j] or labels[i, j] != -1:
                continue
            # Flood-fill this component
            stack = [(i, j)]
            labels[i, j] = 0  # temporary
            pixels = [(i, j)]
            while stack:
                ci, cj = stack.pop()
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < h and 0 <= nj < w and mask[ni, nj] and labels[ni, nj] == -1:
                        labels[ni, nj] = 0
                        stack.append((ni, nj))
                        pixels.append((ni, nj))
            if len(pixels) < min_area:
                for pi, pj in pixels:
                    mask[pi, pj] = False

    return mask


def _save_points_csv(points: np.ndarray, path: Path, *, header: str = "x,y") -> None:
    """
    Robust CSV saving helper.

    - Creates parent directories if necessary.
    - Writes a header row.
    """
    path = Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        points,
        delimiter=",",
        header=header,
        comments="",
    )


def digitize_many(
    image_paths: Sequence[str | Path],
    axis_limits: AxisLimits,
    *,
    max_points: int | None = None,
) -> dict[Path, np.ndarray]:
    """
    Convenience helper to digitize multiple images with shared axis limits.
    """
    dig = IsolatorCurveDigitizer(axis_limits)
    out: dict[Path, np.ndarray] = {}
    for p in image_paths:
        pts = dig.digitize(p, max_points=max_points)
        out[Path(p)] = pts
    return out


