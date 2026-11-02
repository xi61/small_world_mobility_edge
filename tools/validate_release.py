#!/usr/bin/env python3
"""Run repository-level publication safety and completeness checks."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    "", ".bib", ".cff", ".cfg", ".csv", ".html", ".json", ".md", ".py", ".sh", ".tex", ".toml", ".txt", ".yaml", ".yml"
}
PRIVATE_PATTERNS = {
    "macOS user path": re.compile("/" + r"Users/[^/<$\s]+/"),
    "named cluster account": re.compile(r"/u/(?:scratch|home)/(?:[a-zA-Z]/)?[a-zA-Z][a-zA-Z0-9_-]+/"),
    "private key material": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "common secret assignment": re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|password)\s*[:=]\s*['\"][^<'\"]{12,}"),
}


def repository_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    files = [path for path in repository_files() if path.is_file()]
    oversized = [path for path in files if path.stat().st_size >= 100 * 1024 * 1024]
    if oversized:
        fail("files at or above GitHub's 100 MiB limit: " + ", ".join(str(path.relative_to(ROOT)) for path in oversized))

    findings: list[str] = []
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text()
        except UnicodeDecodeError:
            continue
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(content):
                findings.append(f"{path.relative_to(ROOT)}: {label}")
    if findings:
        fail("private-path or secret scan failed:\n  " + "\n  ".join(findings))

    for path in ROOT.glob("experiments/**/*.json"):
        with path.open() as handle:
            json.load(handle)

    shell_files = [str(path.relative_to(ROOT)) for path in files if path.suffix == ".sh"]
    for shell_file in shell_files:
        subprocess.run(["bash", "-n", shell_file], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "tools/audit_report_figures.py"], cwd=ROOT, check=True)

    required = (
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "docs/reproducibility.md",
        "docs/provenance.md",
        "docs/source_inventory.csv",
        "docs/figure_reproduction.csv",
    )
    missing = [item for item in required if not (ROOT / item).is_file()]
    if missing:
        fail("required publication files missing: " + ", ".join(missing))
    print(f"release validation passed for {len(files)} publishable files")


if __name__ == "__main__":
    main()
