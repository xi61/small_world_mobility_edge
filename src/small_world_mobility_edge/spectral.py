from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


GOE_GAP_RATIO = 0.5307
POISSON_GAP_RATIO = 0.3863


@dataclass(frozen=True)
class SpectrumWindow:
    center: float
    width: float
    count: int
    mean_gap_ratio: float
    mean_ipr: float
    mean_pr: float
    label: str


def symmetric_eigpairs(adjacency: sparse.spmatrix, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    n = adjacency.shape[0]
    if k is None or k >= n - 1 or n <= 256:
        dense = adjacency.toarray()
        evals, evecs = np.linalg.eigh(dense)
        return evals, evecs
    evals, evecs = eigsh(adjacency.astype(float), k=k, which="BE")
    order = np.argsort(evals)
    return evals[order], evecs[:, order]


def inverse_participation_ratio(eigenvectors: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(eigenvectors) ** 4, axis=0)


def participation_ratio(eigenvectors: np.ndarray) -> np.ndarray:
    ipr = inverse_participation_ratio(eigenvectors)
    return 1.0 / np.maximum(ipr, 1e-15)


def adjacent_gap_ratios(eigenvalues: np.ndarray) -> np.ndarray:
    spacings = np.diff(np.sort(eigenvalues))
    if len(spacings) < 2:
        return np.array([], dtype=float)
    left = spacings[:-1]
    right = spacings[1:]
    denom = np.maximum(left, right)
    numer = np.minimum(left, right)
    valid = denom > 0
    return numer[valid] / denom[valid]


def classify_gap_statistics(mean_gap_ratio: float) -> str:
    if np.isnan(mean_gap_ratio):
        return "undetermined"
    if abs(mean_gap_ratio - GOE_GAP_RATIO) <= abs(mean_gap_ratio - POISSON_GAP_RATIO):
        return "goe_like"
    return "poisson_like"


def scan_spectrum_windows(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    windows: int = 8,
) -> list[SpectrumWindow]:
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    ipr = inverse_participation_ratio(eigenvectors)
    pr = 1.0 / np.maximum(ipr, 1e-15)

    bins = np.array_split(np.arange(len(eigenvalues)), windows)
    results: list[SpectrumWindow] = []
    for idx in bins:
        if len(idx) < 3:
            continue
        evals = eigenvalues[idx]
        ratios = adjacent_gap_ratios(evals)
        mean_ratio = float(np.mean(ratios)) if len(ratios) else float("nan")
        results.append(
            SpectrumWindow(
                center=float(np.mean(evals)),
                width=float(evals[-1] - evals[0]),
                count=len(idx),
                mean_gap_ratio=mean_ratio,
                mean_ipr=float(np.mean(ipr[idx])),
                mean_pr=float(np.mean(pr[idx])),
                label=classify_gap_statistics(mean_ratio),
            )
        )
    return results


def mobility_edge_candidates(windows: list[SpectrumWindow]) -> list[float]:
    candidates: list[float] = []
    for left, right in zip(windows, windows[1:]):
        if left.label != right.label:
            candidates.append(0.5 * (left.center + right.center))
    return candidates
