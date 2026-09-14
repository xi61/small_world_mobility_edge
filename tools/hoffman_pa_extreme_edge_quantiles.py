from __future__ import annotations

import os
from pathlib import Path

from small_world_mobility_edge.preferential_attachment.extreme_edge_quantiles import (
    run_comparison,
)


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"required environment variable {name} is missing")
    return value


def main() -> None:
    destination = run_comparison(
        Path(required("PA_OUTPUT_ROOT")),
        Path(required("PA_EXTREME_EDGE_OUTPUT_ROOT")),
        chunk_size=int(os.environ.get("PA_CHUNK_SIZE", "5")),
    )
    print(destination, flush=True)


if __name__ == "__main__":
    main()
