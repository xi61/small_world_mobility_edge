# Reproduction tools

This directory contains the actual entry points used to run, aggregate,
transfer, audit, validate, and report the Chung--Lu, fixed-degree, and
preferential-attachment production experiments.

- `hoffman_*_chunk.py` and `hoffman_*_submit*.sh` split deterministic graph
  ranges across SGE array tasks.
- `hoffman_*_aggregate.py` and `hoffman_aggregate_*.sh` validate chunks before
  writing compact summaries and figures.
- `sync_*.sh` copy result products from Hoffman without mixing them into source
  code.
- `audit_*.py`, `validate_*.py`, `summarize_*.py`, and `build_*.py` implement
  integrity checks and report generation.
- `run_*_smoke.py` exercises bounded local versions of the PA disorder and
  random-regular Anderson models without weakening production freeze gates.
- `audit_report_figures.py`, `build_source_inventory.py`, and
  `validate_release.py` define the publication audit boundary.

The wrappers derive account-neutral Hoffman paths from `USER`; override them with
`HOFFMAN_SCRATCH`, `HOFFMAN_HOME`, or `REPO_ROOT` when needed. Set
`PYTHON_MODULE` to a Hoffman module providing Python 3.11 or newer. See
[`docs/hoffman_reproduction.md`](../docs/hoffman_reproduction.md). Create the
`logs/` directory before `qsub`, because SGE opens the output file before the
job body runs.

The package itself requires Python 3.11 or newer. Historical manifests record
the environment used for the archived computation; use a current compatible
module when validating or rerunning the public code.
