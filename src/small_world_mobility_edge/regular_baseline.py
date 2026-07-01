from __future__ import annotations

from dataclasses import dataclass

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
