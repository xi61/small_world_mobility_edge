from __future__ import annotations

from small_world_mobility_edge.preferential_attachment import full_spectrum_scan as scan


def main() -> None:
    for item in scan.PREFERENCE_SPECIFICATIONS:
        destination = scan.aggregate_parameter(item.attachment_tail_exponent)
        print(
            "aggregated alpha=%g delta=%g into %s"
            % (item.attachment_tail_exponent, item.delta, destination),
            flush=True,
        )


if __name__ == "__main__":
    main()
