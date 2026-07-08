from __future__ import annotations

from dataclasses import dataclass
import warnings

import networkx as nx
import numpy as np
from scipy import integrate


@dataclass(frozen=True)
class UnfoldingResult:
    eigenvalues: np.ndarray
    unfolded: np.ndarray
    mean_counting_function: np.ndarray


@dataclass(frozen=True)
class UnfoldingComparison:
    exact_spacings: np.ndarray
    local_spacings: np.ndarray
    spacing_midpoints: np.ndarray
    absolute_difference: np.ndarray
    relative_difference: np.ndarray


@dataclass(frozen=True)
class LocalWindowSpacingResult:
    spacings: np.ndarray
    selected_eigenvalues: np.ndarray
    raw_gaps: np.ndarray
    analyzed_dimension: int
    rho_at_E: float
    window_min: float
    window_max: float
    energy: float
    num_eigenvalues: int
    num_gaps: int
    density_method: str
    reweight_within_window: bool
    midpoint_locations: np.ndarray
    midpoint_densities: np.ndarray
    mean_spacing: float


def generate_random_regular_graph(
    n: int,
    degree: int,
    seed: int | np.random.Generator | None = None,
) -> nx.Graph:
    """Return a random simple d-regular graph on n vertices.

    Mathematically, every vertex has degree exactly ``degree``.
    This is the baseline ensemble in the Jakobson-Miller-Rivin-Rudnick paper.
    """
    return nx.random_regular_graph(degree, n, seed=seed)


def adjacency_matrix(graph: nx.Graph) -> np.ndarray:
    """Return the adjacency matrix A of a graph as a dense symmetric array.

    For the regular-graph baseline, the adjacency spectrum is the main object:
    its global density should approach the Kesten-McKay law, and its local
    statistics are what we later unfold and compare against GOE.
    """
    return nx.to_numpy_array(graph, dtype=float)


def adjacency_eigendecomposition(graph: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    """Compute all eigenvalues and eigenvectors of the adjacency matrix.

    If A v = lambda v, then:
    - ``lambda`` probes the spectral density and spacing statistics;
    - ``v`` lets us measure localization via the inverse participation ratio.
    """
    adj = adjacency_matrix(graph)
    eigenvalues, eigenvectors = np.linalg.eigh(adj)
    return eigenvalues, eigenvectors


def inverse_participation_ratio(eigenvectors: np.ndarray) -> np.ndarray:
    """Return IPR(v) = sum_i |v_i|^4 for each eigenvector column.

    For a fully delocalized normalized vector, IPR is of order 1/n.
    For a localized vector, IPR stays much larger.
    """
    return np.sum(np.abs(eigenvectors) ** 4, axis=0)


def participation_ratio(eigenvectors: np.ndarray) -> np.ndarray:
    """Return PR(v) = 1 / IPR(v) for each eigenvector column."""
    ipr = inverse_participation_ratio(eigenvectors)
    return 1.0 / np.maximum(ipr, 1e-15)


def level_spacings(eigenvalues: np.ndarray) -> np.ndarray:
    """Return consecutive spacings s_i = lambda_{i+1} - lambda_i."""
    return np.diff(np.sort(np.asarray(eigenvalues, dtype=float)))


def adjacent_gap_ratios(eigenvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return adjacent gap ratios and the eigenvalue locations they live near.

    If delta_i = lambda_{i+1} - lambda_i, then

        r_i = min(delta_i, delta_{i+1}) / max(delta_i, delta_{i+1}).

    We associate ``r_i`` with the middle eigenvalue ``lambda_{i+1}``.
    This is useful because the ratio is unfolding-free and can be binned
    across the spectrum.
    """
    eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))
    spacings = np.diff(eigenvalues)
    if len(spacings) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    left = spacings[:-1]
    right = spacings[1:]
    denom = np.maximum(left, right)
    numer = np.minimum(left, right)
    valid = denom > 0
    ratios = numer[valid] / denom[valid]
    centers = eigenvalues[1:-1][valid]
    return ratios, centers


def kesten_mckay_density(x: np.ndarray | float, degree: int) -> np.ndarray:
    """Evaluate the Kesten-McKay density for a d-regular adjacency spectrum.

    For d >= 2, the limiting density is

        rho_d(x) = d * sqrt(4(d-1) - x^2) / (2*pi*(d^2 - x^2))

    on |x| <= 2*sqrt(d-1), and 0 outside.
    """
    x = np.asarray(x, dtype=float)
    support_radius = 2.0 * np.sqrt(degree - 1.0)
    density = np.zeros_like(x, dtype=float)
    inside = np.abs(x) < support_radius
    if np.any(inside):
        numerator = degree * np.sqrt(np.maximum(4.0 * (degree - 1.0) - x[inside] ** 2, 0.0))
        denominator = 2.0 * np.pi * (degree**2 - x[inside] ** 2)
        density[inside] = numerator / denominator
    return density


def kesten_mckay_grid(
    degree: int,
    num_points: int = 20_001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a grid, density, and CDF for the Kesten-McKay law.

    The CDF is built numerically using cumulative trapezoidal integration.
    It gives the mean counting function used for unfolding.
    """
    support_radius = 2.0 * np.sqrt(degree - 1.0)
    grid = np.linspace(-support_radius, support_radius, num_points)
    density = kesten_mckay_density(grid, degree)
    cdf = integrate.cumulative_trapezoid(density, grid, initial=0.0)
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    return grid, density, cdf


def unfold_with_kesten_mckay(
    eigenvalues: np.ndarray,
    degree: int,
    num_points: int = 20_001,
) -> UnfoldingResult:
    """Unfold the adjacency spectrum using the Kesten-McKay mean density.

    If F_d is the Kesten-McKay CDF, then the mean counting function is

        N_bar(lambda) = n * F_d(lambda),

    so the unfolded eigenvalues are xi_i = N_bar(lambda_i).
    Consecutive unfolded spacings should have mean close to 1 and are the
    quantities compared against the GOE Wigner surmise in Phase 2.
    """
    eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))
    grid, _, cdf = kesten_mckay_grid(degree=degree, num_points=num_points)
    clipped = np.clip(eigenvalues, grid[0], grid[-1])
    interpolated_cdf = np.interp(clipped, grid, cdf)
    mean_counting_function = len(eigenvalues) * interpolated_cdf
    return UnfoldingResult(
        eigenvalues=eigenvalues,
        unfolded=mean_counting_function,
        mean_counting_function=mean_counting_function,
    )


def unfolded_spacings(unfolded_eigenvalues: np.ndarray) -> np.ndarray:
    """Return spacings in unfolded coordinates."""
    return np.diff(np.sort(np.asarray(unfolded_eigenvalues, dtype=float)))


def local_density_unfolded_spacings(
    eigenvalues: np.ndarray,
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate unfolded spacings by local-density normalization.

    If delta_i = lambda_{i+1} - lambda_i, then Cook's description is

        s_i^(local) ~= n * rho(lambda_i^*) * delta_i,

    where rho is the local mean density and lambda_i^* is a representative
    energy in the gap. We use the midpoint of the gap.
    """
    eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))
    gaps = np.diff(eigenvalues)
    midpoints = 0.5 * (eigenvalues[:-1] + eigenvalues[1:])
    local_density = kesten_mckay_density(midpoints, degree)
    spacings = len(eigenvalues) * local_density * gaps
    return spacings, midpoints


def remove_trivial_regular_eigenvalue(
    eigenvalues: np.ndarray,
    degree: int,
    atol: float = 1e-10,
) -> np.ndarray:
    """Remove one trivial adjacency eigenvalue equal to ``degree`` if present."""
    eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))
    if len(eigenvalues) == 0:
        return eigenvalues

    index = int(np.argmin(np.abs(eigenvalues - degree)))
    if np.isclose(eigenvalues[index], degree, atol=atol, rtol=0.0):
        return np.delete(eigenvalues, index)
    return eigenvalues


def adjacent_gap_ratios_from_spacings(spacings: np.ndarray) -> np.ndarray:
    """Return adjacent gap ratios computed from an existing spacing sequence."""
    spacings = np.asarray(spacings, dtype=float)
    if len(spacings) < 2:
        return np.array([], dtype=float)

    left = spacings[:-1]
    right = spacings[1:]
    denom = np.maximum(left, right)
    numer = np.minimum(left, right)
    valid = denom > 0
    return numer[valid] / denom[valid]


def local_window_spacings(
    eigenvalues: np.ndarray,
    E: float,
    window_half_width: float,
    density_at_E: float | None = None,
    degree: int | None = None,
    density_method: str = "kesten_mckay",
    analyzed_dimension: int | None = None,
    reweight_within_window: bool = False,
    normalize_mean: bool = False,
    min_eigenvalues: int = 6,
) -> LocalWindowSpacingResult:
    """Return locally rescaled spacings from a spectral window around ``E``.

    The default Cook-style normalization uses the fixed density at the chosen
    energy:

        s_i(E) = n * rho(E) * (lambda_{i+1} - lambda_i).

    When ``reweight_within_window=True``, the density is instead evaluated at
    each gap midpoint:

        s_i = n * rho(lambda_i^*) * (lambda_{i+1} - lambda_i),
        lambda_i^* = 0.5 * (lambda_i + lambda_{i+1}).
    """
    if window_half_width <= 0:
        raise ValueError("window_half_width must be positive")

    eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))
    if degree is not None:
        eigenvalues = remove_trivial_regular_eigenvalue(eigenvalues, degree=degree)

    window_min = float(E - window_half_width)
    window_max = float(E + window_half_width)
    selected = eigenvalues[(eigenvalues >= window_min) & (eigenvalues <= window_max)]

    if len(selected) < min_eigenvalues:
        raise ValueError(
            f"Window [{window_min:.6f}, {window_max:.6f}] around E={E:.6f} "
            f"contains only {len(selected)} eigenvalues; need at least {min_eigenvalues}."
        )

    gaps = np.diff(selected)
    if len(gaps) != len(selected) - 1:
        raise AssertionError("number of gaps must be one less than number of selected eigenvalues")

    midpoint_locations = 0.5 * (selected[:-1] + selected[1:])
    n_effective = len(eigenvalues) if analyzed_dimension is None else int(analyzed_dimension)
    if n_effective <= 0:
        raise ValueError("analyzed_dimension must be positive")

    if density_method == "kesten_mckay":
        if degree is None:
            raise ValueError("degree is required for density_method='kesten_mckay'")
        if density_at_E is None:
            rho_at_E = float(kesten_mckay_density(E, degree))
        else:
            rho_at_E = float(density_at_E)
    elif density_method == "empirical":
        if density_at_E is None:
            raise ValueError("density_at_E is required for density_method='empirical'")
        if reweight_within_window:
            raise ValueError("reweight_within_window is not supported for density_method='empirical'")
        rho_at_E = float(density_at_E)
    else:
        raise ValueError(f"unsupported density_method={density_method!r}")

    if rho_at_E <= 0:
        raise ValueError(
            f"rho(E)={rho_at_E:.6e} at E={E:.6f}; choose an energy in the Kesten-McKay bulk."
        )
    if rho_at_E < 1e-3:
        warnings.warn(
            f"rho(E)={rho_at_E:.6e} at E={E:.6f} is very small; local spacing estimates may be unstable.",
            RuntimeWarning,
            stacklevel=2,
        )

    if reweight_within_window:
        midpoint_densities = kesten_mckay_density(midpoint_locations, degree)
        spacings = n_effective * midpoint_densities * gaps
    else:
        midpoint_densities = np.full_like(midpoint_locations, rho_at_E, dtype=float)
        spacings = n_effective * rho_at_E * gaps

    if normalize_mean:
        mean_spacing = float(np.mean(spacings))
        if mean_spacing <= 0 or not np.isfinite(mean_spacing):
            raise ValueError("cannot normalize local spacings with non-finite or non-positive mean")
        spacings = spacings / mean_spacing

    if not np.all(np.isfinite(spacings)):
        raise ValueError("local_window_spacings produced non-finite values")

    return LocalWindowSpacingResult(
        spacings=spacings,
        selected_eigenvalues=selected,
        raw_gaps=gaps,
        analyzed_dimension=n_effective,
        rho_at_E=rho_at_E,
        window_min=window_min,
        window_max=window_max,
        energy=float(E),
        num_eigenvalues=len(selected),
        num_gaps=len(gaps),
        density_method=density_method,
        reweight_within_window=reweight_within_window,
        midpoint_locations=midpoint_locations,
        midpoint_densities=np.asarray(midpoint_densities, dtype=float),
        mean_spacing=float(np.mean(spacings)),
    )


def compare_unfolding_methods(
    eigenvalues: np.ndarray,
    degree: int,
) -> UnfoldingComparison:
    """Compare exact counting-function unfolding to Cook's local rule.

    The exact McKay-law unfolding gives

        s_i^(exact) = N_bar(lambda_{i+1}) - N_bar(lambda_i)
                     = n * integral_{lambda_i}^{lambda_{i+1}} rho(t) dt.

    The local-density approximation gives

        s_i^(local) ~= n * rho(midpoint_i) * (lambda_{i+1} - lambda_i).

    For bulk gaps in a large random regular graph, these should be very close.
    """
    eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))
    exact = unfolded_spacings(unfold_with_kesten_mckay(eigenvalues, degree=degree).unfolded)
    local, midpoints = local_density_unfolded_spacings(eigenvalues, degree=degree)
    absolute_difference = np.abs(exact - local)
    relative_difference = absolute_difference / np.maximum(np.abs(exact), 1e-12)
    return UnfoldingComparison(
        exact_spacings=exact,
        local_spacings=local,
        spacing_midpoints=midpoints,
        absolute_difference=absolute_difference,
        relative_difference=relative_difference,
    )


def wigner_surmise_goe(s: np.ndarray | float) -> np.ndarray:
    """Evaluate the GOE Wigner surmise density.

        P_W(s) = (pi / 2) * s * exp(-pi s^2 / 4),   s >= 0.
    """
    s = np.asarray(s, dtype=float)
    density = np.zeros_like(s, dtype=float)
    nonnegative = s >= 0
    density[nonnegative] = 0.5 * np.pi * s[nonnegative] * np.exp(-0.25 * np.pi * s[nonnegative] ** 2)
    return density


def bin_average_by_eigenvalue(
    eigenvalue_locations: np.ndarray,
    values: np.ndarray,
    num_bins: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average a spectral observable over eigenvalue bins.

    This is useful for:
    - IPR versus eigenvalue,
    - adjacent gap ratio versus eigenvalue,
    - later finite-size windowed diagnostics.
    """
    eigenvalue_locations = np.asarray(eigenvalue_locations, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(eigenvalue_locations) != len(values):
        raise ValueError("locations and values must have the same length")
    if len(values) == 0:
        return np.array([]), np.array([]), np.array([])

    edges = np.linspace(eigenvalue_locations.min(), eigenvalue_locations.max(), num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(num_bins, np.nan, dtype=float)
    counts = np.zeros(num_bins, dtype=int)

    which = np.digitize(eigenvalue_locations, edges) - 1
    which = np.clip(which, 0, num_bins - 1)
    for idx in range(num_bins):
        mask = which == idx
        counts[idx] = int(np.sum(mask))
        if counts[idx] > 0:
            means[idx] = float(np.mean(values[mask]))

    return centers, means, counts
