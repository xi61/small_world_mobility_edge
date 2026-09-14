"""One-factor-at-a-time summaries for PA m/density sensitivity runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .mobility_edge_family_summary import load_parameter, summary_row
from .mobility_edge_postprocess import (
    GOE_MEAN_GAP_RATIO,
    POISSON_MEAN_GAP_RATIO,
    write_rows,
)


def _slug(value: float) -> str:
    return str(float(value)).replace(".", "p")


def build_m_sensitivity_summary(
    *,
    root: Path,
    family: str,
    n: int,
    parameter_name: str,
    parameter_value: float,
    m_values: list[int],
) -> Path:
    items = [
        load_parameter(root, family, n, m, parameter_name, parameter_value)
        for m in m_values
    ]
    output_dir = (
        root
        / family
        / f"n_{n}_m_sensitivity"
        / f"{parameter_name}_{_slug(parameter_value)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for m, item in zip(m_values, items):
        row = summary_row(item, family=family, parameter_name=parameter_name)
        row["m"] = m
        row["mean_degree_approximately"] = 2 * m
        row["density_sensitivity_not_degree_matched"] = True
        rows.append(row)
    write_rows(output_dir / "m_sensitivity_summary.csv", rows)

    colors = plt.cm.viridis([index / max(len(items) - 1, 1) for index in range(len(items))])
    figure, axes = plt.subplots(2, 1, figsize=(9.5, 8.0), sharex=True)
    for color, m, item in zip(colors, m_values, items):
        coarse = item["coarse"]
        q = [float(row["q_center"]) for row in coarse]
        axes[0].plot(
            q,
            [float(row.get("eigenvector_mean_n_times_ipr", "nan")) for row in coarse],
            color=color,
            label=f"m={m} (mean degree about {2*m})",
        )
        axes[1].plot(
            q,
            [float(row["graph_equal_mean_raw_r_positive_conditioned"]) for row in coarse],
            color=color,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"mean $N\,\mathrm{IPR}$")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].axhline(GOE_MEAN_GAP_RATIO, color="black", ls="--", label="GOE")
    axes[1].axhline(POISSON_MEAN_GAP_RATIO, color="firebrick", ls=":", label="Poisson")
    axes[1].set_ylabel("graph-equal mean raw r")
    axes[1].set_xlabel("continuous non-Perron spectral quantile q")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"PA m/density sensitivity: N={n}, {parameter_name}={parameter_value:g}"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "m_sensitivity_localization_level_statistics.png", dpi=180)
    plt.close(figure)

    manifest = {
        "family": family,
        "n": n,
        "fixed_parameter": {"name": parameter_name, "value": parameter_value},
        "controlled_parameter": "m",
        "m_values": m_values,
        "one_factor_at_a_time": True,
        "interpretation": "m changes the graph density (mean degree approximately 2m); this is not degree matched",
        "source_directories": [str(item["directory"]) for item in items],
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
    parser.add_argument("--parameter-value", type=float, required=True)
    parser.add_argument("--m-values", type=int, nargs="+", default=(2, 5, 10, 20))
    args = parser.parse_args()
    parameter_name = (
        "degree_ccdf_alpha" if args.family == "affine_linear" else "attachment_zeta"
    )
    print(
        build_m_sensitivity_summary(
            root=args.root,
            family=args.family,
            n=args.n,
            parameter_name=parameter_name,
            parameter_value=args.parameter_value,
            m_values=list(args.m_values),
        )
    )


if __name__ == "__main__":
    main()
