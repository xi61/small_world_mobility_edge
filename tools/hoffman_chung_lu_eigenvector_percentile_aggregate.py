from __future__ import annotations

from small_world_mobility_edge import chung_lu_eigenvector_percentile_scan as scan


def main() -> None:
    for specification in scan.TAU_SPECIFICATIONS:
        output_dir = scan.aggregate_parameter(specification.displayed_tau)
        print(f"aggregated displayed_tau={specification.displayed_tau} into {output_dir}", flush=True)


if __name__ == "__main__":
    main()
