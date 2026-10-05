# Spectral Localization in Sparse Random Graphs

Reproducible numerical experiments for eigenvalue statistics, eigenvector
localization, local-mode interactions, and diagonal disorder on sparse random
graphs. The repository collects the complete local implementation used across
the project: random regular, Erdős–Rényi, Chung–Lu, exact-degree rewired
controls, preferential attachment, affine-PA local-mode analysis, PA diagonal
disorder, and the random 3-regular Anderson benchmark.

The project name is historical. No distinct Watts–Strogatz model is silently
substituted for “small world”; the implemented ensembles are named explicitly
below.

## Quick start

Python 3.11 or newer is required.

```bash
git clone https://github.com/xi61/small_world_mobility_edge.git
cd small_world_mobility_edge
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

Run the cross-model dense smoke test:

```bash
python3 -m small_world_mobility_edge.model_smoke \
  --model all --n 160 --degree 6 --tau 2.5 --num-graphs 3 \
  --output-dir outputs/smoke
```

Run the PA and paired diagonal-disorder pilots:

```bash
python3 -m small_world_mobility_edge.preferential_attachment.experiments \
  --smoke-test --output-dir outputs/smoke/pa
python3 tools/run_pa_diagonal_uniform_noise_smoke.py \
  --output-dir outputs/smoke/pa_diagonal_uniform_noise
python3 tools/run_anderson_rrg_smoke.py \
  --output-dir outputs/smoke/anderson_rrg
```

These are integration checks, not substitutes for the recorded production
runs. `make test`, `make smoke`, `make smoke-pa`, `make smoke-disorder`,
`make smoke-anderson`, and `make validate` provide the same entry points.

## Implemented models and analyses

| Model or comparison | Operator / graph | Primary code | Reproducible entry point |
|---|---|---|---|
| Random simple regular | adjacency of a simple `k`-regular graph | `regular_baseline.py` | `model_smoke --model regular` |
| Erdős–Rényi | `G(n,p)`, full graph or largest component | `erdos_renyi_baseline.py`, `erdos_renyi_extreme_edge.py` | `model_smoke --model erdos-renyi` |
| Chung–Lu | expected-degree graph with constant or Pareto weights | `chung_lu_baseline.py` and diagnostic modules | `model_smoke --model chung-lu` |
| Fixed realized degree | finite simple double-edge-swap chain | `fixed_degree_configuration_baseline.py` | `model_smoke --model fixed-degree-rewire` |
| Preferential attachment | BA, affine linear, nonlinear, and Holme–Kim | `preferential_attachment/` | `preferential_attachment.experiments --smoke-test` |
| Clean affine-PA local modes | raw adjacency, full spectrum, graph-ball reference modes | `linear_pa_two_part.py` and reporting module | `tools/hoffman_linear_pa_two_part.py` |
| PA diagonal disorder | `H(a)=A+a diag(xi)`, `xi ~ Uniform[-1,1)` | `pa_diagonal_uniform_noise.py` and reporting module | `tools/run_pa_diagonal_uniform_noise_smoke.py` or production wrapper |
| Random 3-regular Anderson | regular adjacency plus uniform diagonal disorder | `anderson_rrg_benchmark.py`, `anderson_reporting.py` | `tools/run_anderson_rrg_smoke.py` or production wrappers |
| Geometry / tunneling studies | induced graph balls, local bases, Schur diagnostics | `random_schrodinger/geometry.py`, `tunneling*.py`, `handoff_*.py` | matching `tools/hoffman_*` workflows |

The [experiment catalog](docs/experiment_catalog.md) documents component
selection, normalization, spectral windows, atom handling, exponent
conventions, and supported diagnostic combinations. The
[reproducibility guide](docs/reproducibility.md) separates quick checks,
production reruns, and reproduction from archived caches.

## Reports and figures

Two self-contained report packages are tracked:

- `docs/reports/pa_three_model_edge_bulk_conclusive_report/`: TeX, compiled
  PDF, tables, and all 84 referenced figures for the three-model PA
  edge/bulk study.
- `docs/reports/PA_and_Anderson_report_20260924/`: TeX, bibliography,
  compiled PDF, and the clean affine-PA plus diagonal-disorder figures.

Every packaged image is mapped to its generator, input tier, configuration,
command template, and checksum in
[`docs/figure_reproduction.csv`](docs/figure_reproduction.csv). Verify and
refresh that map with:

```bash
python3 tools/audit_report_figures.py --write
```

The recorded production images are committed so the reports render without
multi-gigabyte eigensystem caches. With those external caches available,
`tools/package_combined_report_figures.py` rebuilds the combined figure bundle.

## Reproducibility contract

- Graph, potential, and rewiring seeds are explicit in configurations,
  manifests, or graph-level tables.
- Spacings are formed within individual graph spectra; eigenvalues from
  different graphs are never sorted together to manufacture gaps.
- Raw gaps, mean-rescaled gaps, analytical unfolding, and empirical-CDF
  unfolding are distinct quantities in code and output names.
- Perron removal, component choice, zero-atom masks, multiplicity tolerances,
  and rank conventions are recorded per experiment.
- Historical run outputs are read-only inputs. New runs write to ignored
  `outputs/` or user-supplied run roots.
- The file-level [source inventory](docs/source_inventory.csv) records hashes,
  roles, and whether an item came from the public base, authoritative research
  workspace, immutable executed snapshot, or publication assembly.

See [provenance](docs/provenance.md), [data availability](docs/data_availability.md),
and the [release validation record](docs/release_validation.md).

## Repository layout

- `src/small_world_mobility_edge/`: scientific generators and analyses.
- `tests/`: deterministic unit and controlled-fixture tests.
- `experiments/`: frozen model grids, seeds, thresholds, and run policies.
- `tools/`: local and Hoffman/SGE execution, validation, aggregation, transfer,
  reporting, and release-audit entry points.
- `scripts/`: report-construction utilities.
- `outputs/`: selected compact historical results; new output is ignored.
- `docs/reports/`: self-contained publication report packages.
- `docs/history/`: dated research/audit records retained as provenance, not as
  current instructions.

## Scientific limitations

The checked-in results are finite-size computational evidence. Several PA and
disorder grids contain only one graph or one potential stream per condition;
those comparisons do not establish a universal mobility edge, critical
disorder, or asymptotic phase transition. The exact-degree control preserves
the labeled degree sequence but a finite swap chain is not claimed to be a
uniform configuration-model sample. The local-mode workflow is explicitly
a-posteriori where exact eigenvectors select regions or modes.

## License and citation

No open-source reuse license is granted; see [`LICENSE`](LICENSE). Cite the
repository and exact commit used; machine-readable metadata is in
[`CITATION.cff`](CITATION.cff).
