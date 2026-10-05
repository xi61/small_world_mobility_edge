#!/usr/bin/env python3
"""Run the deterministic end-to-end PA diagonal-disorder pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from small_world_mobility_edge.random_schrodinger.pa_diagonal_uniform_noise import (
    load_configuration,
    run_controlled_pilot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/pa_diagonal_uniform_noise/production.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_controlled_pilot(
        config=load_configuration(args.config),
        output=args.output_dir,
    )
    print(result)


if __name__ == "__main__":
    main()
