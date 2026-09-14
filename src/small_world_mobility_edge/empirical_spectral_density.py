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


@dataclass(frozen=True)
class CrossFittedEmpiricalSpectralDensity:
    """One graph-level cross-fitting fold and its training-only density."""

    fold: int
    training_graph_ids: tuple[int, ...]
    held_out_graph_ids: tuple[int, ...]
    density_model: EmpiricalSpectralDensity


def estimate_cross_fitted_variable_local_count_densities(
    eigenvalue_samples: list[np.ndarray],
    graph_ids: list[int] | np.ndarray,
    *,
    num_folds: int = 5,
    bulk_local_count_half_width: float = 0.20,
    tail_local_count_half_width: float = 0.08,
    tail_quantile_cutoff: float = 0.95,
    num_points: int = 2001,
) -> list[CrossFittedEmpiricalSpectralDensity]:
    """Fit deterministic graph-level folds without train/test leakage.

    Graph ``g`` belongs to fold ``g % num_folds``. Every density model is
    estimated only from graphs outside its held-out fold.
    """
    if num_folds < 2:
        raise ValueError("num_folds must be at least two")
    ids = np.asarray(graph_ids, dtype=int)
    if len(eigenvalue_samples) != len(ids):
        raise ValueError("eigenvalue_samples and graph_ids must have equal length")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("graph_ids must be unique")
    if len(ids) < num_folds:
        raise ValueError("at least one graph per fold is required")
    models: list[CrossFittedEmpiricalSpectralDensity] = []
    for fold in range(num_folds):
        held_mask = ids % num_folds == fold
        train_mask = ~held_mask
        held_ids = tuple(int(value) for value in ids[held_mask])
        train_ids = tuple(int(value) for value in ids[train_mask])
        if not held_ids or not train_ids:
            raise ValueError(f"fold {fold} has an empty train or held-out set")
        training_samples = [
            np.asarray(sample, dtype=float)
            for sample, keep in zip(eigenvalue_samples, train_mask)
            if keep
        ]
        mean_dimension = float(np.mean([len(sample) for sample in training_samples]))
        density = estimate_variable_local_count_density(
            eigenvalue_samples=training_samples,
            num_graphs=len(training_samples),
            per_graph_dimension=mean_dimension,
            bulk_local_count_half_width=bulk_local_count_half_width,
            tail_local_count_half_width=tail_local_count_half_width,
            tail_quantile_cutoff=tail_quantile_cutoff,
            num_points=num_points,
        )
        if not np.isclose(density.cdf[0], 0.0, atol=1e-12):
            raise AssertionError("cross-fitted CDF must start at zero")
        if not np.isclose(density.cdf[-1], 1.0, atol=1e-12):
            raise AssertionError("cross-fitted CDF must end at one")
        if np.any(np.diff(density.cdf) < -1e-12):
            raise AssertionError("cross-fitted CDF must be nondecreasing")
        models.append(
            CrossFittedEmpiricalSpectralDensity(
                fold=fold,
                training_graph_ids=train_ids,
                held_out_graph_ids=held_ids,
                density_model=density,
            )
        )
    return models


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


def estimate_variable_local_count_density(
    eigenvalue_samples: list[np.ndarray],
    num_graphs: int,
    per_graph_dimension: float,
    bulk_local_count_half_width: float,
    tail_local_count_half_width: float,
    tail_quantile_cutoff: float,
    num_points: int = 2001,
) -> EmpiricalSpectralDensity:
    if bulk_local_count_half_width <= 0:
        raise ValueError("bulk_local_count_half_width must be positive")
    if tail_local_count_half_width <= 0:
        raise ValueError("tail_local_count_half_width must be positive")
    if not 0.5 <= tail_quantile_cutoff < 1.0:
        raise ValueError("tail_quantile_cutoff must lie in [0.5, 1)")
    pooled = pool_eigenvalues(eigenvalue_samples)
    grid = _density_grid_from_samples(pooled, num_points=num_points)
    sorted_values = pooled
    empirical_quantiles = np.searchsorted(sorted_values, grid, side="right") / len(sorted_values)
    tail_mask = (empirical_quantiles <= 1.0 - tail_quantile_cutoff) | (empirical_quantiles >= tail_quantile_cutoff)
    half_widths = np.where(tail_mask, tail_local_count_half_width, bulk_local_count_half_width)
    left = np.searchsorted(sorted_values, grid - half_widths, side="left")
    right = np.searchsorted(sorted_values, grid + half_widths, side="right")
    counts = right - left
    density = counts / (num_graphs * per_graph_dimension * 2.0 * half_widths)
    density, cdf = _normalize_density_on_grid(grid, density)
    return EmpiricalSpectralDensity(
        method="variable_local_count",
        grid=grid,
        density=density,
        cdf=cdf,
        pooled_eigenvalues=pooled,
        num_graphs=num_graphs,
        per_graph_dimension=per_graph_dimension,
        local_count_half_width=bulk_local_count_half_width,
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
    tail_local_count_half_width: float | None = None,
    tail_quantile_cutoff: float = 0.95,
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
    if method == "variable_local_count":
        return estimate_variable_local_count_density(
            eigenvalue_samples=eigenvalue_samples,
            num_graphs=num_graphs,
            per_graph_dimension=per_graph_dimension,
            bulk_local_count_half_width=local_count_half_width,
            tail_local_count_half_width=tail_local_count_half_width
            if tail_local_count_half_width is not None
            else local_count_half_width,
            tail_quantile_cutoff=tail_quantile_cutoff,
            num_points=num_points,
        )
    raise ValueError(f"unsupported empirical density method {method!r}")
