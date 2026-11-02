# Reproducibility guide

This project distinguishes three levels of reproduction because the original
production eigensystems are much larger than a practical Git repository.

## 1. Validate the published checkout

Create a Python 3.11+ environment, install the package, run all tests, then run
the repository validator:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
python3 tools/audit_report_figures.py
python3 tools/validate_release.py
```

`validate_release.py` checks JSON syntax, shell syntax, figure presence and
checksums, required publication metadata, the GitHub 100 MiB per-file limit,
and common private-path/secret patterns.

## 2. Run bounded local examples

```bash
make smoke
make smoke-pa
make smoke-disorder
make smoke-anderson
make smoke-multifractality
```

The first command covers regular, Erdős–Rényi, Chung–Lu, and exact-degree
rewired paths. The second covers the general PA pipeline. The third constructs
one deterministic `N=80` affine-PA graph, draws the configured uniform
potential, diagonalizes `H(a)`, verifies the eigensystem, and writes the full
pilot diagnostic bundle. The fourth generates a connected random 3-regular
graph, applies uniform diagonal disorder, verifies complete central-window
coverage against a full dense spectrum, and records localization/gap tables.

All examples write under ignored `outputs/`. Matplotlib uses a noninteractive
backend in production/report modules.
The multifractality example creates real affine-PA graphs at N=64 and 128,
checks their eigenpair residuals, writes small caches, and exercises both
case analysis and cross-size aggregation. It is not a production estimate.

### October results from committed tables

The [multifractality guide](experiments/pa_multifractality.md) gives an
aggregate command requiring no external arrays. The
[anatomy scripts](../tools/pa_spectral_anatomy/README.md) list the exact
external-input schemas for cache-level regeneration of graph pictures.
The [result index](results/README.md) maps the new figures to their generators.

## 3. Reproduce production experiments

Production was designed for an SGE/Grid Engine cluster. Set account-local
paths rather than editing source:

```bash
export PROJECT_ROOT="$PWD"
export HOFFMAN_SCRATCH="/path/to/scratch"
export HOFFMAN_HOME="/path/to/home"
export PYTHON_MODULE="python/<3.11-or-newer-module>"
export PYTHONPATH="$PROJECT_ROOT/src"
mkdir -p logs
```

The exact scheduler recipes are in `docs/hoffman_reproduction.md`. The primary
newer workflows are:

| Study | Configuration | Scientific/report code | Execution family |
|---|---|---|---|
| Three-model PA edge/bulk | `experiments/single_graph_edge_bulk/production_models.json` | `preferential_attachment/single_graph_edge_bulk.py` | `hoffman_pa_single_graph_edge_bulk.*` |
| Clean affine-PA two-part study | `experiments/linear_pa_two_part_20260924/config.json` | `linear_pa_two_part.py`, `linear_pa_two_part_reporting.py` | `hoffman_linear_pa_two_part_*` |
| PA diagonal uniform noise | `experiments/pa_diagonal_uniform_noise/production.json` | `pa_diagonal_uniform_noise*.py` | `hoffman_pa_diagonal_uniform_noise.py` plus freeze/array wrappers |
| Anderson repair/extension | `experiments/random_schrodinger/anderson_repair_extension_20260924_v1.json` | `random_schrodinger/anderson_*.py` | `hoffman_anderson_*` |
| Geometry/tunneling corrections | `experiments/random_schrodinger/correction_plan_v1.json` | `geometry.py`, `tunneling*.py` | `hoffman_random_schrodinger_correction_*` |

Freeze manifests bind a production run to a source/configuration identity.
The production CLIs intentionally reject a live tree that does not match the
freeze. Do not bypass that check; create a new run root and freeze instead.

### External input layout

Large inputs are not committed. Put recovered clean affine-PA cases at
`external_data/linear_pa_two_part_20260924_v3/cases` or update
`clean_source_root` in a copied configuration. `external_data/` is ignored.
Each case must retain its graph cache, resolved case record, eigensystem arrays,
tables, and COMPLETE manifests because the diagonal-disorder workflow validates
all of those identities before reuse.

### Figure regeneration

Run each scientific aggregate first. Then package the combined report figures:

```bash
python3 tools/package_combined_report_figures.py \
  --clean-report <clean-report-root> \
  --disorder-run <diagonal-disorder-run-root> \
  --output-dir <combined-report-root>/figures
python3 tools/audit_report_figures.py --write
```

The 84 PA report figures are produced by
`preferential_attachment/single_graph_edge_bulk.py` from the three cases in
`production_models.json`, with the held-out CDF caches named there. The exact
per-figure mapping is machine readable in `docs/figure_reproduction.csv`.

## Seed policy

- Smoke tests accept explicit CLI seeds or use documented deterministic
  defaults.
- Production graph seeds are fixed in JSON or derived from a recorded base
  seed and case coordinates.
- PA diagonal-noise potential seeds use NumPy `SeedSequence` over the base
  seed, `N`, alpha-grid index, and a fixed namespace integer. One potential is
  paired across amplitudes on a graph.
- Anderson potential seeds and graph seeds are tabulated in the production
  configuration; amplitudes are paired within streams.
- Rewiring seeds are distinct from source graph seeds.

Reproducibility means replaying the stated algorithm, versions, seeds, and
inputs. It does not guarantee byte-identical floating-point output across
different BLAS/LAPACK implementations.
