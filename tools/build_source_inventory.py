#!/usr/bin/env python3
"""Write a deterministic hash/provenance inventory for publishable files."""

from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/source_inventory.csv"


def git_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    excluded = {"docs/source_inventory.csv"}
    return [
        ROOT / item.decode()
        for item in result.stdout.split(b"\0")
        if item and item.decode() not in excluded
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def role(relative: str) -> str:
    suffix = Path(relative).suffix.lower()
    if relative.startswith("src/"):
        return "scientific source"
    if relative.startswith("tests/"):
        return "automated test"
    if relative.startswith("tools/") or relative.startswith("scripts/"):
        return "execution or reporting utility"
    if relative.startswith("experiments/"):
        return "experiment configuration"
    if relative.startswith("docs/reports/") and suffix in {".png", ".pdf", ".csv"}:
        return "recorded report artifact"
    if relative.startswith("docs/") or relative in {"README.md", "LICENSE", "CITATION.cff"}:
        return "documentation or metadata"
    if relative.startswith("outputs/"):
        return "compact recorded result"
    return "project support file"


def provenance(relative: str) -> str:
    diagonal = {
        "src/small_world_mobility_edge/random_schrodinger/pa_diagonal_uniform_noise.py",
        "src/small_world_mobility_edge/random_schrodinger/pa_diagonal_uniform_noise_reporting.py",
        "tests/test_pa_diagonal_uniform_noise.py",
        "tools/hoffman_pa_diagonal_uniform_noise.py",
        "experiments/pa_diagonal_uniform_noise/production.json",
    }
    if relative in diagonal:
        return "immutable executed source snapshot pa_diagonal_uniform_noise_20260924_v1"
    if relative.startswith("docs/reports/"):
        return "curated recorded artifact from completed experiment"
    if relative.startswith(("tools/build_source_inventory.py", "tools/audit_report_figures.py", "tools/package_combined_report_figures.py", "tools/run_pa_diagonal_uniform_noise_smoke.py", "tools/run_anderson_rrg_smoke.py", "tools/validate_release.py")):
        return "publication assembly"
    imported_prefixes = (
        "src/small_world_mobility_edge/random_schrodinger/",
        "src/small_world_mobility_edge/preferential_attachment/linear_pa_two_part",
        "experiments/",
        "docs/history/",
        "scripts/",
        "tests/test_anderson_repair_extension.py",
        "tests/test_linear_pa_two_part.py",
        "tests/test_pa_coupling_handoff.py",
        "tests/test_pa_diagonal_uniform_noise.py",
        "tests/test_random_schrodinger_",
        "tools/hoffman_anderson_",
        "tools/hoffman_linear_pa_two_part",
        "tools/hoffman_pa_coupling_handoff",
        "tools/hoffman_pa_diagonal_uniform_noise.py",
        "tools/hoffman_random_schrodinger",
        "tools/hoffman_retry_linear_pa_two_part_case.sh",
        "tools/hoffman_submit_linear_pa_two_part",
        "tools/sync_random_schrodinger_results.sh",
    )
    if relative.startswith(imported_prefixes):
        return "authoritative research workspace"
    return "existing public checkout at base commit 67f3f56"


def main() -> None:
    rows = []
    for path in sorted(git_files()):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "role": role(relative),
                "provenance": provenance(relative),
            }
        )
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
