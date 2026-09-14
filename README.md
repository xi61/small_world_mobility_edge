# Small-World Mobility Edge

Numerical experiments for spectral density, level statistics, and eigenvector
localization in sparse random graphs. This is the single aggregate repository
for random simple `k`-regular, Erdős--Rényi, Chung--Lu, fixed-realized-degree,
and preferential-attachment experiments. It contains the generators,
diagnostic and Hoffman reproduction pipelines, compact finalized results, and
the preferential-attachment report package.

## Install and verify

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

Run the four non-PA model/control paths at a modest size:

```bash
python3 -m small_world_mobility_edge.model_smoke \
  --model all --n 160 --degree 6 --tau 2.5 --num-graphs 3 \
  --output-dir outputs/smoke
```

The smoke runner writes graph invariants, normalized empirical density,
within-graph spacing and adjacent-gap-ratio summaries, and eigenvalue-aligned
IPR plots. It is a quick integration check; the specialized modules and
Hoffman entry points are the production paths.

## Repository map

- `src/small_world_mobility_edge/`: model generators and scientific analysis.
- `tools/`: chunk, aggregate, transfer, cache-audit, validation, and report
  entry points used for the Hoffman runs.
- `outputs/smoke/regular/`: the available random-regular reference result.
- `outputs/erdos_renyi/`: ER density, local-spacing, and edge outputs.
- `outputs/chung_lu/`: Chung--Lu density, spacing, localization, resonance,
  projector, and degree/spectral-tail outputs.
- `outputs/configuration_model/`: exact-realized-degree rewiring controls paired
  with Chung--Lu samples.
- `outputs/preferential_attachment/`: BA, affine-linear, nonlinear, rewired,
  mobility-edge, extreme-edge, and single-graph edge/bulk outputs.
- `docs/reports/pa_three_model_edge_bulk_conclusive_report/`: self-contained
  TeX source, recorded figures/tables, and compiled PA report PDF.

The [results inventory](docs/results_inventory.md) states which outputs are
tracked and which high-volume artifacts are regenerated. The
[Hoffman reproduction guide](docs/hoffman_reproduction.md) maps scheduler
wrappers to each model family. The [experiment catalog](docs/experiment_catalog.md)
records filtering, normalization, seed, exponent, and model--diagnostic
conventions.

## Capability map

| Model/comparison | Generator | Diagnostics entry point | Small example | Status |
|---|---|---|---|---|
| Random simple `k`-regular | [`regular_baseline.py`](src/small_world_mobility_edge/regular_baseline.py) | [`regular_baseline_experiments.py`](src/small_world_mobility_edge/regular_baseline_experiments.py), [`local_spacing_by_energy.py`](src/small_world_mobility_edge/local_spacing_by_energy.py) | `python3 -m small_world_mobility_edge.model_smoke --model regular` | Generator, Kesten--McKay unfolding, density, spacing, ratios, and IPR implemented |
| Erdős--Rényi `G(n,p)` | [`erdos_renyi_baseline.py`](src/small_world_mobility_edge/erdos_renyi_baseline.py) | [`erdos_renyi_spacing_by_energy.py`](src/small_world_mobility_edge/erdos_renyi_spacing_by_energy.py), [`erdos_renyi_extreme_edge.py`](src/small_world_mobility_edge/erdos_renyi_extreme_edge.py) | `python3 -m small_world_mobility_edge.model_smoke --model erdos-renyi --component-mode largest_connected_component` | Full graph/LCC, ensemble density, local spacing, and sparse edge eigenpairs implemented |
| Power-law Chung--Lu | [`chung_lu_baseline.py`](src/small_world_mobility_edge/chung_lu_baseline.py) | [`chung_lu_spacing_by_energy.py`](src/small_world_mobility_edge/chung_lu_spacing_by_energy.py), [`chung_lu_eigenvector_percentile_scan.py`](src/small_world_mobility_edge/chung_lu_eigenvector_percentile_scan.py) | `python3 -m small_world_mobility_edge.model_smoke --model chung-lu --tau 2.5` | Weight modes, mean solve, cutoffs, density/spacing, localization, resonance, and projector diagnostics implemented |
| Fixed realized-degree control | [`fixed_degree_configuration_baseline.py`](src/small_world_mobility_edge/fixed_degree_configuration_baseline.py) | [`fixed_degree_configuration_joint_tail.py`](src/small_world_mobility_edge/fixed_degree_configuration_joint_tail.py) | `python3 -m small_world_mobility_edge.model_smoke --model fixed-degree-rewire` | Exact labeled degrees preserved by finite double-edge swaps; no independent stub-pairing sampler exists locally |
| Degree/spectral-tail matching | Chung--Lu and PA generators above | [`chung_lu_degree_spectral_tail_match.py`](src/small_world_mobility_edge/chung_lu_degree_spectral_tail_match.py), [`preferential_attachment/experiments.py`](src/small_world_mobility_edge/preferential_attachment/experiments.py) | `python3 -m small_world_mobility_edge.chung_lu_degree_spectral_tail_match --smoke-test --n 400 --tau 2.5 --degree 8 --num-graphs 3 --k-per-side 40 --target-tail-count 20 --output-dir outputs/smoke/tail` | Degree CCDF, fitted exponents, cutoff sensitivity, spectral CCDF, rank match, and optional rewired control implemented |
| Preferential attachment | [`preferential_attachment/baseline.py`](src/small_world_mobility_edge/preferential_attachment/baseline.py) | [`preferential_attachment/experiments.py`](src/small_world_mobility_edge/preferential_attachment/experiments.py) and related modules | `python3 -m small_world_mobility_edge.preferential_attachment.experiments --smoke-test --output-dir outputs/smoke/pa` | BA, affine linear, nonlinear, Holme--Kim, rewired controls, and compatible PA--Chung--Lu comparisons retained |

## Important conventions

- Spacings are always formed inside each graph; only the resulting gaps or
  summaries are pooled across graphs.
- Eigenvalue-density histograms, local-count density estimates, and spacing
  histograms are different objects and are named separately in outputs.
- Regular graphs drop one trivial adjacency eigenvalue `d`. ER, Chung--Lu,
  and PA workflows record or remove the Perron eigenpair according to each
  experiment, and zero-atom handling is explicit in manifests.
- Chung--Lu's canonical `tau` is a density exponent, so its weight CCDF has
  exponent `tau - 1`. Affine PA uses the degree-CCDF exponent `alpha`.
  Nonlinear attachment uses the kernel exponent `zeta`; these parameters are
  not interchangeable.
- The fixed-degree control is a simple graph produced by
  `networkx.double_edge_swap`. It has no self-loops or parallel edges and
  preserves every labeled degree, but a finite swap chain is not claimed to be
  a uniform configuration-model sample or to preserve connectivity.

## Output policy

Final plots, summary tables, manifests, research notes, and the report package
are tracked. New runs still write beneath `outputs/`, which remains ignored by
default so an accidental rerun cannot add gigabytes to Git.

The tracked publication tier deliberately excludes scheduler logs, cache
directories, `partials/`, `replay_partials/`, superseded runs, per-window
`.npz` checkpoints, and high-volume per-graph `.csv.gz` tables. Those are
restart or analysis intermediates, not omitted model code; the source and
`tools/` pipelines regenerate them. See the results inventory for exact scope.
