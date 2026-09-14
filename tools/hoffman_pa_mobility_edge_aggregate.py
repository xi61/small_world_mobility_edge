from __future__ import annotations

from small_world_mobility_edge.preferential_attachment.mobility_edge_production_aggregate import (
    aggregate_production_parameter,
)

from hoffman_pa_mobility_edge_chunk import config_from_environment


def main() -> None:
    config = config_from_environment()
    destination = aggregate_production_parameter(
        config, chunk_size=int(__import__("os").environ.get("PA_CHUNK_SIZE", "5"))
    )
    print(f"aggregated into {destination}", flush=True)


if __name__ == "__main__":
    main()
