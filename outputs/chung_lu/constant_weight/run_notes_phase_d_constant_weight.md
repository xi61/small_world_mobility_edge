# Phase D Run Log

Date: 2026-07-07

## Purpose

Phase D adds a homogeneous constant-weight Chung-Lu control and compares it directly against the existing Erdős-Rényi Phase C pipeline.

The goal is:

- keep the same local-spacing experiment,
- change only the graph generator,
- verify that when all expected degrees are equal, constant-weight Chung-Lu behaves like homogeneous Erdős-Rényi up to sampling variation.

This phase intentionally preserves:

- the random-regular control code,
- the Phase C ER code,
- the Phase C empirical-density and local-window workflow.

## Parameter Snapshot

```python
N = 2000
ACTIVE_AVERAGE_DEGREES = [3, 6]
ALL_AVERAGE_DEGREES = [3, 6, 10, 20]
NUM_DENSITY_GRAPHS = 12
NUM_SPACING_GRAPHS = 12
DENSITY_SEED = 31415
SPACING_SEED = 27182
COMPONENT_MODE = "largest_connected_component"
DENSITY_METHOD = "local_count"
LOCAL_COUNT_HALF_WIDTH = 0.20
DENSITY_NUM_POINTS = 2001
DENSITY_NUM_BINS = 120
DENSITY_BANDWIDTH_SCALE = 1.0
EXPLICIT_E_VALUES = [-2.5, -1.5, 0.0, 1.5, 2.5]
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
WINDOW_HALF_WIDTH = 0.20
NUM_BINS = 30
MIN_EIGENVALUES = 10
SMALL_SPACING_THRESHOLD = 0.20
KS_SAMPLE_SIZE = 4096
```

## Command

```bash
python3 src/small_world_mobility_edge/chung_lu_spacing_by_energy.py
```

## Phase D Methodology

### Models compared

1. Erdős-Rényi:

```text
G(n, p),    p = c / (n - 1)
```

2. Constant-weight Chung-Lu:

```text
nx.expected_degree_graph(weights, selfloops=False)
```

with equal weights

```text
w_i = c * n / (n - 1)
```

for every vertex.

This choice makes the loopless Chung-Lu edge probability match the ER target scale:

```text
p = c / (n - 1)
```

### Shared pipeline

For both models, with matched:

- `n`
- average degree `c`
- density-estimation seeds
- spacing-analysis seeds
- component mode
- density-estimation method
- spectral locations `E`
- local window width

the script does the following:

1. Generate a density-estimation ensemble.
2. Generate a spacing-analysis ensemble.
3. Restrict to the largest connected component when requested.
4. Compute sorted adjacency eigenvalues.
5. Remove the Perron/top eigenvalue from bulk-spacing analysis.
6. Estimate the ensemble spectral density:

```text
rho_hat(E)
```

using the same empirical method for both models.

7. Choose spectral locations by:
   - explicit `E` values,
   - ER ensemble quantiles converted to energies, then reused for both models.

8. For each graph and each window:

```text
[E - window_half_width, E + window_half_width]
```

form only within-graph consecutive gaps

```text
delta_i = lambda_{i+1} - lambda_i
```

and rescale using the same local-spacing rule as Phase C:

```text
s_i(E) = n_analyzed * rho_hat(E) * delta_i
```

where `n_analyzed` is the size of the analyzed matrix, here usually the largest connected component size.

9. Pool the already-rescaled within-graph spacings across realizations.
10. Compare local spacing histograms with:

```text
P_GOE(s) = (pi / 2) * s * exp(-pi s^2 / 4)
P_Poisson(s) = exp(-s)
```

11. Compare numerical summaries:
   - mean adjacent-gap ratio,
   - small-gap probability `P(s < 0.2)`,
   - agreement within combined sampling error.

## Generated Output Directories

- [avgdeg_3](../../../outputs/chung_lu/constant_weight/avgdeg_3)
- [avgdeg_6](../../../outputs/chung_lu/constant_weight/avgdeg_6)

Per degree, the saved files are:

- `degree_distribution_comparison.png`
- `global_density_comparison.png`
- `local_spacing_histograms_quantiles.png`
- `local_spacing_histograms_explicit_E.png`
- `local_spacing_summary_quantiles.png`
- `local_spacing_summary_explicit_E.png`
- `degree_summary.csv`
- `degree_comparison.csv`
- `erdos_renyi_quantile_summary.csv`
- `chung_lu_quantile_summary.csv`
- `model_comparison_quantiles.csv`

## Degree Summary

### avgdeg = 3

| model | average_degree | number_of_graphs | target_mean_degree | mean_realized_degree | sd_realized_degree | mean_num_components | mean_largest_component_size | mean_isolated_vertices | mean_analyzed_dimension |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| erdos_renyi | 3 | 12 | 3 | 3.000666666666666 | 0.05164975285223508 | 112.83333333333333 | 1876.1666666666667 | 101.25 | 1876.1666666666667 |
| chung_lu | 3 | 12 | 3 | 2.9901666666666666 | 0.048326227272819856 | 113.08333333333333 | 1877.9166666666667 | 103.5 | 1877.9166666666667 |

### avgdeg = 6

| model | average_degree | number_of_graphs | target_mean_degree | mean_realized_degree | sd_realized_degree | mean_num_components | mean_largest_component_size | mean_isolated_vertices | mean_analyzed_dimension |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| erdos_renyi | 6 | 12 | 6 | 6.021166666666666 | 0.06730910155034867 | 6.416666666666667 | 1994.5833333333333 | 5.416666666666667 | 1994.5833333333333 |
| chung_lu | 6 | 12 | 6 | 5.981916666666667 | 0.06526652815164678 | 5.333333333333333 | 1995.5833333333333 | 4.25 | 1995.5833333333333 |

## Quantile Comparison Output

### avgdeg = 3

| E | q | ER mean r | Chung-Lu mean r | diff r | combined se(r) | within 2se? | ER P(s<0.2) | Chung-Lu P(s<0.2) | diff P(s<0.2) | combined se(P) | within 2se? |
|---:|---:|---:|---:|---:|---:|:---|---:|---:|---:|---:|:---|
| -2.4270363729897224 | 0.1 | 0.5462433493761888 | 0.5208528939456561 | -0.025390455430532644 | 0.013966370485055257 | True | 0.036908681834873794 | 0.03471855267355433 | -0.0021901291613194637 | 0.009835565138209795 | True |
| -1.3604272007676221 | 0.25 | 0.5290754009986965 | 0.5400032826072171 | 0.01092788160852054 | 0.008086457113791946 | True | 0.037707315619740917 | 0.022467045388850092 | -0.015240270230890824 | 0.005026291884720679 | False |
| -0.0012260326487519913 | 0.5 | 0.5063922536163675 | 0.47319661236177657 | -0.03319564125459096 | 0.013529329636804319 | False | 0.24565660497182365 | 0.2481714169489102 | 0.0025148119770865562 | 0.0190348960808388 | True |
| 1.3551409427559122 | 0.75 | 0.5312979266392198 | 0.5353017102554598 | 0.0040037836162399865 | 0.010952163736029214 | True | 0.03728074084901516 | 0.03223131708579064 | -0.005049423763224523 | 0.00654840538129525 | True |
| 2.4207493481566096 | 0.9 | 0.5202154049896063 | 0.5299631568286179 | 0.009747751839011665 | 0.016059609226611978 | True | 0.035933242878990614 | 0.03882296337007229 | 0.0028897204910816737 | 0.00891054241948019 | True |

### avgdeg = 6

| E | q | ER mean r | Chung-Lu mean r | diff r | combined se(r) | within 2se? | ER P(s<0.2) | Chung-Lu P(s<0.2) | diff P(s<0.2) | combined se(P) | within 2se? |
|---:|---:|---:|---:|---:|---:|:---|---:|---:|---:|---:|:---|
| -3.3550313338795066 | 0.1 | 0.5440719524757238 | 0.551306829204174 | 0.007234876728450135 | 0.013959622571513485 | True | 0.023860251055158873 | 0.025162887059287417 | 0.001302636004128544 | 0.006638371556517044 | True |
| -1.8776164035357177 | 0.25 | 0.5006297167772932 | 0.5373348540973085 | 0.036705137320015346 | 0.013393550018638426 | False | 0.04408526819451591 | 0.03131824291410146 | -0.01276702528041445 | 0.006716084302200748 | True |
| -0.0029701856594091737 | 0.5 | 0.5199734474154157 | 0.5295822404361522 | 0.009608793020736472 | 0.008910279161697949 | True | 0.04148381504787873 | 0.029333041305779323 | -0.012150773742099406 | 0.008045336679181241 | True |
| 1.8700111180827887 | 0.75 | 0.5188543561658493 | 0.544105870276565 | 0.02525151411071569 | 0.014238959871285416 | True | 0.03072387999799894 | 0.022668468560547984 | -0.008055411437450956 | 0.005732179754711103 | True |
| 3.348018830257959 | 0.9 | 0.5374189970245026 | 0.5322539199331051 | -0.005165077091397463 | 0.01189129850091689 | True | 0.04229244233720533 | 0.035055491771909676 | -0.0072369505652956564 | 0.009333605435308001 | True |

## Interpretation Checkpoint

- Phase D supports the intended homogeneous-control statement: constant-weight Chung-Lu is close to Erdős-Rényi when expected degrees are all equal.
- The realized degree distributions match closely and both stay near the target average degree.
- The global spectral densities are visually similar at matched parameters.
- The local spacing summaries are mostly within combined sampling variation.

For `avgdeg = 3`:

- the center anomaly at `q = 0.50`, `E ≈ 0` persists in both homogeneous models
- ER:
  - `mean r ≈ 0.506`
  - `P(s < 0.2) ≈ 0.246`
- constant-weight Chung-Lu:
  - `mean r ≈ 0.473`
  - `P(s < 0.2) ≈ 0.248`

This is consistent with the anomaly being a sparse homogeneous effect, not something unique to the ER implementation.

For `avgdeg = 6`:

- both models remain broadly GOE-like across the quantile windows
- the agreement is strong enough that Phase D functions as the intended homogeneous cross-check

## Files Added / Used In Phase D

- [chung_lu_baseline.py](../../../src/small_world_mobility_edge/chung_lu_baseline.py)
- [chung_lu_spacing_by_energy.py](../../../src/small_world_mobility_edge/chung_lu_spacing_by_energy.py)
- [test_chung_lu_phase_d.py](../../../tests/test_chung_lu_phase_d.py)

## Validation

Command used:

```bash
python3 -m unittest discover -s tests
```

Result:

- `Ran 11 tests ... OK`

## Stop Point

Phase D only is complete here.
Heavy-tailed Chung-Lu has **not** been started.
