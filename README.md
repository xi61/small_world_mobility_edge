# Small-World Mobility Edge

Numerical experiments for spectral density, level statistics, and eigenvector
localization in sparse random graphs.  The public code covers random regular,
Erdős--Rényi, Chung--Lu, degree-preserving controls, and preferential-
attachment families.  Generated production caches are intentionally not a
complete part of the repository.

## Install and verify

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

Run all four non-PA families at a modest size:

```bash
python3 -m small_world_mobility_edge.model_smoke \
  --model all --n 160 --degree 6 --tau 2.5 --num-graphs 3 \
  --output-dir outputs/smoke
```

The smoke runner writes graph invariants, normalized density estimates,
within-graph spacing and gap-ratio summaries, and eigenvalue-aligned IPR plots.
It is a quick integration check; the specialized experiment modules below are
the scientific production paths.

## Capability map

| Model/comparison | Generator | Diagnostics entry point | Small example | Status |
|---|---|---|---|---|
| Random simple `k`-regular | [`regular_baseline.py`](src/small_world_mobility_edge/regular_baseline.py) | [`regular_baseline_experiments.py`](src/small_world_mobility_edge/regular_baseline_experiments.py), [`local_spacing_by_energy.py`](src/small_world_mobility_edge/local_spacing_by_energy.py) | `python3 -m small_world_mobility_edge.model_smoke --model regular` | Generator, Kesten--McKay unfolding, density, spacing, ratios, and IPR implemented |
| Erdős--Rényi `G(n,p)` | [`erdos_renyi_baseline.py`](src/small_world_mobility_edge/erdos_renyi_baseline.py) | [`erdos_renyi_spacing_by_energy.py`](src/small_world_mobility_edge/erdos_renyi_spacing_by_energy.py), [`erdos_renyi_extreme_edge.py`](src/small_world_mobility_edge/erdos_renyi_extreme_edge.py) | `python3 -m small_world_mobility_edge.model_smoke --model erdos-renyi --component-mode largest_connected_component` | Full graph/LCC, ensemble density, local spacing, and sparse edge eigenpairs implemented |
| Power-law Chung--Lu | [`chung_lu_baseline.py`](src/small_world_mobility_edge/chung_lu_baseline.py) | [`chung_lu_spacing_by_energy.py`](src/small_world_mobility_edge/chung_lu_spacing_by_energy.py), [`chung_lu_eigenvector_percentile_scan.py`](src/small_world_mobility_edge/chung_lu_eigenvector_percentile_scan.py) | `python3 -m small_world_mobility_edge.model_smoke --model chung-lu --tau 2.5` | Weight modes, mean solve, cutoffs, density/spacing, localization, resonance, and projector diagnostics implemented |
| Fixed realized-degree control | [`fixed_degree_configuration_baseline.py`](src/small_world_mobility_edge/fixed_degree_configuration_baseline.py) | [`fixed_degree_configuration_joint_tail.py`](src/small_world_mobility_edge/fixed_degree_configuration_joint_tail.py) | `python3 -m small_world_mobility_edge.model_smoke --model fixed-degree-rewire` | Exact labeled degrees preserved by finite double-edge swaps; no independent stub-pairing sampler exists locally |
| Degree/spectral-tail matching | Chung--Lu and PA generators above | [`chung_lu_degree_spectral_tail_match.py`](src/small_world_mobility_edge/chung_lu_degree_spectral_tail_match.py), [`preferential_attachment/experiments.py`](src/small_world_mobility_edge/preferential_attachment/experiments.py) | `python3 -m small_world_mobility_edge.chung_lu_degree_spectral_tail_match --smoke-test --n 400 --tau 2.5 --degree 8 --num-graphs 3 --k-per-side 40 --target-tail-count 20 --output-dir outputs/smoke/tail` | Degree CCDF, fitted exponents, cutoff sensitivity, spectral CCDF, rank match, and optional rewired control implemented |
| Preferential attachment | [`preferential_attachment/baseline.py`](src/small_world_mobility_edge/preferential_attachment/baseline.py) | [`preferential_attachment/experiments.py`](src/small_world_mobility_edge/preferential_attachment/experiments.py) and related modules | `python3 -m small_world_mobility_edge.preferential_attachment.experiments --smoke-test --output-dir outputs/smoke/pa` | BA, affine linear, nonlinear, Holme--Kim, rewired controls, and compatible PA--Chung--Lu comparisons retained |

See the [experiment catalog](docs/experiment_catalog.md) for filtering,
normalization, seed, exponent, and model--diagnostic support details.  The
[results inventory](docs/results_inventory.md) distinguishes runnable code
from archived production results that are not published.

## Important conventions

- Spacings are always formed inside each graph; only the resulting gaps or
  summaries are pooled across graphs.
- Eigenvalue-density histograms, local-count density estimates, and spacing
  histograms are different objects and are named separately in outputs.
- Regular graphs drop one trivial adjacency eigenvalue `d`.  ER, Chung--Lu,
  and PA workflows record or remove the Perron eigenpair according to each
  experiment, and zero-atom handling is explicit in manifests.
- Chung--Lu's canonical `tau` is a density exponent, so its weight CCDF has
  exponent `tau - 1`.  Affine PA uses the degree-CCDF exponent `alpha`.
  Nonlinear attachment uses the kernel exponent `zeta`; these parameters are
  not interchangeable.
- The so-called fixed-degree configuration control is a simple graph produced
  by `networkx.double_edge_swap`.  It has no self-loops or parallel edges and
  preserves every labeled degree, but a finite swap chain is not claimed to be
  a uniform configuration-model sample or to preserve connectivity.

## Output policy

New runs write beneath `outputs/`, which is ignored by Git.  A small set of
historical figures and tables predates that policy and remains tracked.  Do
not commit bulk spectra, caches, scheduler logs, private paths, or cluster
working copies.
