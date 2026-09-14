from __future__ import annotations

import json

from small_world_mobility_edge.preferential_attachment.mobility_edge_production_aggregate import (
    load_production_cache,
    write_nonlinear_hub_spectral_edge_diagnostic,
)

from hoffman_pa_mobility_edge_chunk import config_from_environment


def main() -> None:
    config = config_from_environment()
    if config.model_family != "nonlinear":
        raise SystemExit("nonlinear hub-edge postprocessing requires PA_MODEL_FAMILY=nonlinear")
    graphs, structural, _, _ = load_production_cache(config, chunk_size=5)
    result = write_nonlinear_hub_spectral_edge_diagnostic(
        config, graphs, structural, config.parameter_dir
    )
    destination = config.parameter_dir / "global_spectrum" / "nonlinear_hub_edge"
    manifest = {
        **config.metadata(),
        "analysis": "nonlinear_hub_spectral_edge_diagnostic_v1",
        "reuse_classification": "REPOSTPROCESS_ONLY",
        "source_cache": str(config.partial_dir),
        "result": result,
    }
    (destination / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(destination, flush=True)


if __name__ == "__main__":
    main()
