"""Controlled one-factor comparison tables and figures for PA families."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .mobility_edge_postprocess import GOE_MEAN_GAP_RATIO, POISSON_MEAN_GAP_RATIO, write_rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parameter_dir(
    root: Path,
    family: str,
    n: int,
    m: int,
    parameter_name: str,
    value: float,
    regime: str | None = None,
) -> Path:
    slug = str(value).replace(".", "p")
    family_root = root / family
    if regime:
        family_root = family_root / regime
    return family_root / f"n_{n}_m_{m}" / f"{parameter_name}_{slug}"


def finite(values) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def mean(values) -> float:
    values = finite(values)
    return float(np.mean(values)) if len(values) else math.nan


def load_parameter(
    root: Path,
    family: str,
    n: int,
    m: int,
    parameter_name: str,
    value: float,
    regime: str | None = None,
) -> dict[str, object]:
    directory = parameter_dir(root, family, n, m, parameter_name, value, regime)
    required = [
        directory / "run_manifest.json",
        directory / "structural" / "structural_summary.json",
        directory / "structural" / "degree_ccdf.csv",
        directory / "global_spectrum" / "empirical_density_cdf.npz",
        directory / "coarse" / "q0_1_dq_0p002" / "joined_localization_level_statistics.csv",
        directory / "coarse" / "q0_1_dq_0p002" / "candidate_regions.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing parameter outputs:\n" + "\n".join(missing))
    with np.load(required[3], allow_pickle=False) as data:
        density = {key: np.asarray(data[key]) for key in ("grid", "density", "cdf")}
    fine_path = directory / "fine" / "fine_window_summary.csv"
    structural_by_graph_path = directory / "structural" / "structural_by_graph.csv"
    atom_path = directory / "global_spectrum" / "atom_multiplicity_by_graph.csv"
    return {
        "value": value,
        "directory": directory,
        "manifest": json.loads(required[0].read_text()),
        "structural_summary": json.loads(required[1].read_text()),
        "degree_ccdf": read_csv(required[2]),
        "density": density,
        "coarse": read_csv(required[4]),
        "candidates": json.loads(required[5].read_text()),
        "fine": read_csv(fine_path) if fine_path.is_file() else [],
        "structural_by_graph": read_csv(structural_by_graph_path) if structural_by_graph_path.is_file() else [],
        "atoms": read_csv(atom_path) if atom_path.is_file() else [],
    }


def summary_row(item: dict[str, object], *, family: str, parameter_name: str) -> dict[str, object]:
    value = float(item["value"])
    coarse = item["coarse"]
    structural = item["structural_summary"]
    q = np.asarray([float(row["q_center"]) for row in coarse])
    nipr = np.asarray([float(row.get("eigenvector_mean_n_times_ipr", "nan")) for row in coarse])
    energy = np.asarray([float(row["E_mean"]) for row in coarse])
    raw_r = np.asarray([float(row["graph_equal_mean_raw_r_positive_conditioned"]) for row in coarse])
    valid_ipr = np.isfinite(nipr)
    maximum_index = int(np.nanargmax(nipr)) if np.any(valid_ipr) else 0
    bulk = (q >= 0.10) & (q <= 0.90) & np.isfinite(raw_r)
    edge = ((q <= 0.10) | (q >= 0.90)) & np.isfinite(raw_r)
    atom_rows = item["atoms"] or item["structural_by_graph"]
    zero_fraction = mean(
        float(row.get("zero_multiplicity", 0.0)) / max(int(float(row.get("n", 5000))) - 1, 1)
        for row in atom_rows
    )
    fine = item["fine"]
    categories = sorted({candidate["category"] for candidate in item["candidates"]})
    row: dict[str, object] = {
        parameter_name: value,
        "mean_maximum_degree": structural.get("mean_maximum_degree", math.nan),
        "mean_degree_gini": structural.get("mean_degree_gini", math.nan),
        "mean_zero_multiplicity_fraction": zero_fraction,
        "positive_edge_degree_spectral_slope": structural.get("positive_slope_mean", math.nan),
        "negative_edge_degree_spectral_slope": structural.get("negative_slope_mean", math.nan),
        "maximum_coarse_n_times_ipr": float(nipr[maximum_index]) if np.any(valid_ipr) else math.nan,
        "maximum_ipr_q": float(q[maximum_index]) if np.any(valid_ipr) else math.nan,
        "maximum_ipr_E": float(energy[maximum_index]) if np.any(valid_ipr) else math.nan,
        "bulk_raw_r_positive_conditioned": float(np.nanmean(raw_r[bulk])) if np.any(bulk) else math.nan,
        "edge_raw_r_positive_conditioned": float(np.nanmean(raw_r[edge])) if np.any(edge) else math.nan,
        "closest_coarse_raw_r_distance_to_poisson": float(np.nanmin(np.abs(raw_r - POISSON_MEAN_GAP_RATIO))),
        "candidate_categories": categories,
        "candidate_intervals": [
            [candidate.get("q_low"), candidate.get("q_high")]
            for candidate in item["candidates"]
            if candidate["category"] != "NO_SUSTAINED_SIGNAL"
        ],
        "fine_window_count": len(fine),
        "fine_min_KS_to_GOE": min(
            [float(value) for value in (row.get("crossfit_unfolded_KS_to_GOE", "nan") for row in fine) if np.isfinite(float(value))],
            default=math.nan,
        ),
        "fine_min_KS_to_Poisson": min(
            [float(value) for value in (row.get("crossfit_unfolded_KS_to_Poisson", "nan") for row in fine) if np.isfinite(float(value))],
            default=math.nan,
        ),
        "localization_spacing_classification": (
            "IPR_AND_R_SHIFT"
            if "IPR_AND_R_SHIFT" in categories
            else "IPR_ONLY"
            if "IPR_ONLY" in categories
            else "ATOM_CONTAMINATED"
            if "ATOM_CONTAMINATED" in categories
            else "NO_SUSTAINED_SIGNAL"
        ),
    }
    if family == "affine_linear":
        row["realized_degree_ccdf_alpha_hill_q0p95"] = structural.get(
            "q0.95_hill_degree_ccdf_alpha_mean", math.nan
        )
        row["realized_degree_ccdf_alpha_log_ccdf_q0p95"] = structural.get(
            "q0.95_ccdf_regression_exponent_mean", math.nan
        )
    else:
        row["mean_maximum_degree_over_n"] = structural.get("mean_maximum_degree_over_n", math.nan)
        row["mean_maximum_degree_over_total_degree"] = structural.get(
            "mean_maximum_degree_over_total_degree", math.nan
        )
        row["mean_top_1_percent_degree_share"] = structural.get(
            "mean_top_1_percent_degree_share", math.nan
        )
        row["power_law_exponent_interpretation"] = (
            "NOT_APPLICABLE_SUBLINEAR_KERNEL_STRETCHED_EXPONENTIAL_DIAGNOSTIC"
            if value < 1.0
            else "LINEAR_ATTACHMENT_ENDPOINT"
            if math.isclose(value, 1.0, abs_tol=1e-12)
            else "NOT_APPLICABLE_SUPERLINEAR_KERNEL"
        )
    return row


def plot_family(items: list[dict[str, object]], family: str, parameter_name: str, output_dir: Path) -> None:
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(items)))

    figure, axis = plt.subplots(figsize=(8.5, 5.8))
    for color, item in zip(colors, items):
        rows = item["degree_ccdf"]
        if family == "affine_linear":
            realized = float(
                item["structural_summary"].get(
                    "q0.95_hill_degree_ccdf_alpha_mean", math.nan
                )
            )
            label = (
                f"requested {parameter_name}={item['value']:g}; "
                f"realized Hill={realized:.3f}"
            )
        else:
            label = f"{parameter_name}={item['value']:g} (kernel exponent)"
        axis.loglog(
            [float(row["degree"]) for row in rows],
            [float(row["mean_ccdf"]) for row in rows],
            color=color,
            label=label,
        )
    axis.set_xlabel("degree k")
    axis.set_ylabel(r"mean empirical $P(D\geq k)$")
    axis.grid(alpha=0.2, which="both")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "figure_A_degree_ccdf_comparison.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.5, 5.8))
    for color, item in zip(colors, items):
        density = item["density"]
        axis.plot(density["grid"], density["density"], color=color, label=f"{parameter_name}={item['value']:g}")
    axis.set_xlabel("energy E")
    axis.set_ylabel("normalized continuous density")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "figure_B_global_esd_comparison.png", dpi=180)
    plt.close(figure)

    for filename, metric, ylabel in (
        ("figure_C_n_times_ipr_vs_quantile.png", "eigenvector_mean_n_times_ipr", r"mean $N\,\mathrm{IPR}$"),
        ("figure_D_raw_r_vs_quantile.png", "graph_equal_mean_raw_r_positive_conditioned", "graph-equal mean raw r"),
    ):
        figure, axis = plt.subplots(figsize=(9.2, 5.8))
        for color, item in zip(colors, items):
            rows = item["coarse"]
            axis.plot(
                [float(row["q_center"]) for row in rows],
                [float(row.get(metric, "nan")) for row in rows],
                color=color,
                label=f"{parameter_name}={item['value']:g}",
            )
        if "raw_r" in filename:
            axis.axhline(GOE_MEAN_GAP_RATIO, color="black", ls="--", label="GOE")
            axis.axhline(POISSON_MEAN_GAP_RATIO, color="firebrick", ls=":", label="Poisson")
        else:
            axis.set_yscale("log")
        axis.set_xlabel("continuous non-Perron spectral quantile q")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, ncol=2)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)

    for x_name, filename in (
        ("q_center", "figure_E_localization_level_statistics_vs_quantile.png"),
        ("E_mean", "figure_F_localization_level_statistics_vs_energy.png"),
    ):
        figure, axes = plt.subplots(2, 1, figsize=(9.5, 8.0), sharex=True)
        for color, item in zip(colors, items):
            rows = item["coarse"]
            x = [float(row[x_name]) for row in rows]
            axes[0].plot(x, [float(row.get("eigenvector_mean_n_times_ipr", "nan")) for row in rows], color=color, label=f"{parameter_name}={item['value']:g}")
            axes[1].plot(x, [float(row["graph_equal_mean_raw_r_positive_conditioned"]) for row in rows], color=color)
        axes[0].set_yscale("log")
        axes[0].set_ylabel(r"mean $N\,\mathrm{IPR}$")
        axes[1].set_ylabel("graph-equal mean raw r")
        axes[1].axhline(GOE_MEAN_GAP_RATIO, color="black", ls="--")
        axes[1].axhline(POISSON_MEAN_GAP_RATIO, color="firebrick", ls=":")
        axes[1].set_xlabel("quantile q" if x_name == "q_center" else "energy E")
        axes[0].legend(frameon=False, ncol=2)
        for axis in axes:
            axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)


def build_family_summary(
    *,
    root: Path,
    family: str,
    n: int,
    m: int,
    parameter_name: str,
    values: list[float],
    regime: str | None = None,
) -> Path:
    items = [load_parameter(root, family, n, m, parameter_name, value, regime) for value in values]
    family_root = root / family / regime if regime else root / family
    output_dir = family_root / f"n_{n}_m_{m}" / "controlled_family_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [summary_row(item, family=family, parameter_name=parameter_name) for item in items]
    write_rows(output_dir / "family_summary.csv", rows)
    plot_family(items, family, parameter_name, output_dir)
    manifest = {
        "family": family,
        "attachment_regime": regime,
        "n": n,
        "m": m,
        "controlled_parameter": parameter_name,
        "values": values,
        "one_factor_at_a_time": True,
        "source_directories": [str(item["directory"]) for item in items],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output_dir


def build_nonlinear_hub_edge_cross_regime_summary(
    *,
    root: Path,
    n: int,
    m: int,
    sublinear_values: list[float],
    superlinear_values: list[float],
) -> Path:
    """Compare the empirical nonlinear hub--edge relation across zeta<1 and zeta>1."""
    rows: list[dict[str, object]] = []
    source_paths: list[str] = []
    for regime, values in (
        ("sublinear", sublinear_values),
        ("superlinear", superlinear_values),
    ):
        for value in values:
            directory = parameter_dir(
                root,
                "nonlinear",
                n,
                m,
                "attachment_zeta",
                value,
                "sublinear" if regime == "sublinear" else None,
            )
            source = (
                directory
                / "global_spectrum"
                / "nonlinear_hub_edge"
                / "hub_edge_summary.csv"
            )
            if not source.is_file():
                raise FileNotFoundError(source)
            source_paths.append(str(source))
            for row in read_csv(source):
                rows.append(
                    {
                        **row,
                        "attachment_regime": regime,
                        "attachment_zeta": value,
                    }
                )
    output_dir = (
        root
        / "nonlinear"
        / f"n_{n}_m_{m}"
        / "nonlinear_hub_edge_cross_regime_summary"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "hub_edge_cross_regime_summary.csv", rows)

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    for side, marker in (
        ("perron", "^"),
        ("positive_nonperron", "o"),
        ("negative_absolute", "s"),
    ):
        selected = [
            row
            for row in rows
            if row["record_type"] == "fixed_rank"
            and row["spectral_side"] == side
            and int(row["rank_or_top_count"]) == 1
        ]
        selected.sort(key=lambda row: float(row["attachment_zeta"]))
        axes[0].plot(
            [float(row["attachment_zeta"]) for row in selected],
            [float(row["median_lambda_over_sqrt_degree"]) for row in selected],
            marker=marker,
            label=side,
        )
    for side, marker in (("positive_nonperron", "o"), ("negative_absolute", "s")):
        selected = [
            row
            for row in rows
            if row["record_type"] == "loglog_fit"
            and row["spectral_side"] == side
            and int(row["rank_or_top_count"]) == 50
        ]
        selected.sort(key=lambda row: float(row["attachment_zeta"]))
        axes[1].plot(
            [float(row["attachment_zeta"]) for row in selected],
            [float(row["mean_slope"]) for row in selected],
            marker=marker,
            label=side,
        )
    axes[0].axhline(1.0, color="black", ls="--", lw=1, label=r"$|\lambda|/\sqrt{d}=1$")
    axes[0].set_ylabel(r"median rank-1 $|\lambda|/\sqrt{d_{(1)}}$")
    axes[1].axhline(0.5, color="black", ls="--", lw=1, label=r"$b=1/2$")
    axes[1].set_ylabel(r"mean top-50 slope $b$")
    for axis in axes:
        axis.axvline(0.5, color="#6c757d", ls=":", lw=1)
        axis.axvline(1.0, color="#6c757d", ls="-.", lw=1)
        axis.set_xlabel(r"nonlinear attachment exponent $\zeta$")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle("Nonlinear PA hub--spectral-edge relation across regimes")
    figure.tight_layout()
    figure.savefig(output_dir / "hub_edge_cross_regime_comparison.png", dpi=180)
    plt.close(figure)
    manifest = {
        "analysis": "nonlinear_hub_edge_cross_regime_summary_v1",
        "n": n,
        "m": m,
        "sublinear_attachment_zeta": sublinear_values,
        "superlinear_attachment_zeta": superlinear_values,
        "linear_sqrt_degree_relation_assumed": False,
        "sources": source_paths,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", choices=("affine_linear", "nonlinear"), required=True)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--values", type=float, nargs="+", required=True)
    parser.add_argument("--regime", choices=("sublinear", "linear", "superlinear"))
    args = parser.parse_args()
    parameter_name = "degree_ccdf_alpha" if args.family == "affine_linear" else "attachment_zeta"
    print(
        build_family_summary(
            root=args.root,
            family=args.family,
            n=args.n,
            m=args.m,
            parameter_name=parameter_name,
            values=args.values,
            regime=args.regime,
        )
    )


if __name__ == "__main__":
    main()
