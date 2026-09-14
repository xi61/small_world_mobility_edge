"""Reproducibly select one informative parameter for an m-sensitivity run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ELIGIBLE_CATEGORIES = {"IPR_ONLY", "IPR_AND_R_SHIFT"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def select_parameter(
    summary_path: Path,
    *,
    parameter_name: str,
    maximum_zero_atom_fraction: float = 0.05,
) -> dict[str, object]:
    rows = read_rows(summary_path)
    candidates: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for row in rows:
        value = float(row[parameter_name])
        category = row["localization_spacing_classification"]
        zero_fraction = float(row.get("mean_zero_multiplicity_fraction", "nan"))
        maximum_ipr = float(row["maximum_coarse_n_times_ipr"])
        fine_count = int(float(row.get("fine_window_count", 0)))
        reasons = []
        if category not in ELIGIBLE_CATEGORIES:
            reasons.append("no sustained non-atom localization category")
        if not math.isfinite(zero_fraction) or zero_fraction > maximum_zero_atom_fraction:
            reasons.append("zero-atom fraction is unavailable or too large")
        if fine_count <= 0:
            reasons.append("no fine spacing windows")
        eligible = not reasons
        score = maximum_ipr * (1.25 if category == "IPR_AND_R_SHIFT" else 1.0)
        record = {
            "parameter_value": value,
            "category": category,
            "mean_zero_multiplicity_fraction": zero_fraction,
            "maximum_coarse_n_times_ipr": maximum_ipr,
            "fine_window_count": fine_count,
            "eligible": eligible,
            "exclusion_reasons": reasons,
            "score": score,
        }
        audit.append(record)
        if eligible:
            candidates.append(record)
    if not candidates:
        raise ValueError("no parameter satisfies the predeclared m-sensitivity rule")
    selected = max(candidates, key=lambda row: (float(row["score"]), -float(row["parameter_value"])))
    return {
        "parameter_name": parameter_name,
        "selected_value": selected["parameter_value"],
        "selection_rule": {
            "eligible_categories": sorted(ELIGIBLE_CATEGORIES),
            "maximum_zero_atom_fraction": maximum_zero_atom_fraction,
            "requires_fine_spacing_windows": True,
            "score": "maximum_coarse_n_times_ipr times 1.25 for IPR_AND_R_SHIFT, else times 1",
            "tie_break": "smaller parameter value",
        },
        "selected_record": selected,
        "all_parameter_audit": audit,
        "interpretation": "selection identifies a sensitivity target, not a mobility edge",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--parameter-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_parameter(args.summary, parameter_name=args.parameter_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
