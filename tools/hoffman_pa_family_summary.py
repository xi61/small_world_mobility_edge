from __future__ import annotations

import json
import os
from pathlib import Path

from small_world_mobility_edge.preferential_attachment.mobility_edge_family_summary import (
    build_family_summary,
    build_nonlinear_hub_edge_cross_regime_summary,
)
from small_world_mobility_edge.preferential_attachment.mobility_edge_selection import (
    select_parameter,
)


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"required environment variable {name} is missing")
    return value


def main() -> None:
    root = Path(required("PA_OUTPUT_ROOT"))
    family = required("PA_MODEL_FAMILY")
    n = int(os.environ.get("PA_N", "5000"))
    m = int(os.environ.get("PA_M", "10"))
    values = [
        float(value)
        for value in required("PA_PARAMETER_VALUES").replace(":", ",").split(",")
    ]
    parameter_name = (
        "degree_ccdf_alpha" if family == "affine_linear" else "attachment_zeta"
    )
    output_dir = build_family_summary(
        root=root,
        family=family,
        n=n,
        m=m,
        parameter_name=parameter_name,
        values=values,
        regime=os.environ.get("PA_REGIME", "").strip() or None,
    )
    print(output_dir, flush=True)
    regime = os.environ.get("PA_REGIME", "").strip() or None
    if family == "nonlinear" and regime == "sublinear":
        superlinear_values = [
            float(value)
            for value in os.environ.get(
                "PA_CROSS_REGIME_SUPERLINEAR_VALUES", "1.05,1.10,1.20,1.30,1.50"
            ).replace(":", ",").split(",")
        ]
        cross_regime = build_nonlinear_hub_edge_cross_regime_summary(
            root=root,
            n=n,
            m=m,
            sublinear_values=values,
            superlinear_values=superlinear_values,
        )
        print(cross_regime, flush=True)
    if os.environ.get("PA_SELECT_SENSITIVITY", "1") == "1":
        selection = select_parameter(
            output_dir / "family_summary.csv", parameter_name=parameter_name
        )
        selection_path = output_dir / "m_sensitivity_selection.json"
        selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
        print(selection_path, flush=True)


if __name__ == "__main__":
    main()
