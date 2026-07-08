from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate
from scipy.stats import gaussian_kde


@dataclass(frozen=True)
class EmpiricalSpectralDensity:
    method: str
    grid: np.ndarray
    density: np.ndarray
    cdf: np.ndarray
    pooled_eigenvalues: np.ndarray
    num_graphs: int
    per_graph_dimension: float
    local_count_half_width: float | None = None
    bandwidth: float | None = None

    def evaluate(self, x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        clipped = np.clip(x, self.grid[0], self.grid[-1])
        return np.interp(clipped, self.grid, self.density, left=0.0, right=0.0)

    def quantiles_to_energies(self, quantiles: np.ndarray | list[float]) -> np.ndarray:
        quantiles = np.asarray(quantiles, dtype=float)
        if np.any((quantiles < 0.0) | (quantiles > 1.0)):
            raise ValueError("quantiles must lie in [0, 1]")
        return np.interp(quantiles, self.cdf, self.grid)


def pool_eigenvalues(eigenvalue_samples: list[np.ndarray]) -> np.ndarray:
    if not eigenvalue_samples:
        raise ValueError("eigenvalue_samples must be non-empty")
    arrays = [np.asarray(sample, dtype=float) for sample in eigenvalue_samples if len(sample) > 0]
    if not arrays:
        raise ValueError("eigenvalue_samples contain no eigenvalues")
    return np.sort(np.concatenate(arrays))


def _density_grid_from_samples(
    pooled_eigenvalues: np.ndarray,
    num_points: int,
    padding_fraction: float = 0.05,
) -> np.ndarray:
    lo = float(np.min(pooled_eigenvalues))
    hi = float(np.max(pooled_eigenvalues))
    spread = hi - lo
    padding = padding_fraction * spread if spread > 0 else 1.0
    return np.linspace(lo - padding, hi + padding, num_points)


def _normalize_density_on_grid(grid: np.ndarray, density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    density = np.maximum(np.asarray(density, dtype=float), 0.0)
    cdf = integrate.cumulative_trapezoid(density, grid, initial=0.0)
    total_mass = float(cdf[-1])
    if total_mass <= 0:
        raise ValueError("estimated density has non-positive total mass")
    density = density / total_mass
    cdf = cdf / total_mass
    return density, cdf


def estimate_ensemble_histogram_density(
    eigenvalue_samples: list[np.ndarray],
    num_graphs: int,
    per_graph_dimension: float,
    num_points: int = 2001,
    num_bins: int = 120,
) -> EmpiricalSpectralDensity:
    pooled = pool_eigenvalues(eigenvalue_samples)
    hist, edges = np.histogram(pooled, bins=num_bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    grid = _density_grid_from_samples(pooled, num_points=num_points)
    density = np.interp(grid, centers, hist, left=0.0, right=0.0)
    density, cdf = _normalize_density_on_grid(grid, density)
    return EmpiricalSpectralDensity(
        method="histogram",
        grid=grid,
        density=density,
        cdf=cdf,
        pooled_eigenvalues=pooled,
        num_graphs=num_graphs,
        per_graph_dimension=per_graph_dimension,
    )


def estimate_ensemble_kde_density(
    eigenvalue_samples: list[np.ndarray],
    num_graphs: int,
    per_graph_dimension: float,
    num_points: int = 2001,
    bandwidth_scale: float = 1.0,
) -> EmpiricalSpectralDensity:
    pooled = pool_eigenvalues(eigenvalue_samples)
    kde = gaussian_kde(pooled)
    kde.set_bandwidth(kde.factor * bandwidth_scale)
    grid = _density_grid_from_samples(pooled, num_points=num_points)
    density = kde(grid)
    density, cdf = _normalize_density_on_grid(grid, density)
    return EmpiricalSpectralDensity(
        method="kde",
        grid=grid,
        density=density,
        cdf=cdf,
        pooled_eigenvalues=pooled,
        num_graphs=num_graphs,
        per_graph_dimension=per_graph_dimension,
        bandwidth=float(kde.factor),
    )


def estimate_local_count_density(
    eigenvalue_samples: list[np.ndarray],
    num_graphs: int,
    per_graph_dimension: float,
    local_count_half_width: float,
    num_points: int = 2001,
) -> EmpiricalSpectralDensity:
    if local_count_half_width <= 0:
        raise ValueError("local_count_half_width must be positive")
    pooled = pool_eigenvalues(eigenvalue_samples)
    grid = _density_grid_from_samples(pooled, num_points=num_points)
    sorted_values = pooled
    left = np.searchsorted(sorted_values, grid - local_count_half_width, side="left")
    right = np.searchsorted(sorted_values, grid + local_count_half_width, side="right")
    counts = right - left
    density = counts / (num_graphs * per_graph_dimension * 2.0 * local_count_half_width)
    density, cdf = _normalize_density_on_grid(grid, density)
    return EmpiricalSpectralDensity(
        method="local_count",
        grid=grid,
        density=density,
        cdf=cdf,
        pooled_eigenvalues=pooled,
        num_graphs=num_graphs,
        per_graph_dimension=per_graph_dimension,
        local_count_half_width=local_count_half_width,
    )


def estimate_empirical_spectral_density(
    eigenvalue_samples: list[np.ndarray],
    num_graphs: int,
    per_graph_dimension: float,
    method: str = "kde",
    num_points: int = 2001,
    num_bins: int = 120,
    bandwidth_scale: float = 1.0,
    local_count_half_width: float = 0.20,
) -> EmpiricalSpectralDensity:
    if method == "kde":
        return estimate_ensemble_kde_density(
            eigenvalue_samples=eigenvalue_samples,
            num_graphs=num_graphs,
            per_graph_dimension=per_graph_dimension,
            num_points=num_points,
            bandwidth_scale=bandwidth_scale,
        )
    if method == "histogram":
        return estimate_ensemble_histogram_density(
            eigenvalue_samples=eigenvalue_samples,
            num_graphs=num_graphs,
            per_graph_dimension=per_graph_dimension,
            num_points=num_points,
            num_bins=num_bins,
        )
    if method == "local_count":
        return estimate_local_count_density(
            eigenvalue_samples=eigenvalue_samples,
            num_graphs=num_graphs,
            per_graph_dimension=per_graph_dimension,
            local_count_half_width=local_count_half_width,
            num_points=num_points,
        )
    raise ValueError(f"unsupported empirical density method {method!r}")
