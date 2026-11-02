# PA diagonal disorder: fixed-graph amplitude sweep

[Disorder models](../models/disorder_and_anderson.md) · [Production config](../../experiments/pa_diagonal_uniform_noise/production.json) · [Replicated extension](anderson_benchmark.md)

## Operator and experiment

$$H(a)=A+a\operatorname{diag}(\xi),\qquad
\xi_i\overset{\mathrm{iid}}\sim U[-1,1),\qquad
a\in\{0,0.5,1,2,4,8,16,32\}.$$

A is the **unscaled** adjacency of a saved clean affine-PA graph. The run
uses N=500,10000, m=10, α=1.05,1.1,1.25,1.5,2,2.5,3, one graph per
condition. One potential vector is paired across all positive amplitudes on
each graph. There are 14 clean and 98 positive-amplitude operators: 112 full
eigensystems. The adjacency, degrees, and vertex labels stay fixed as a changes.

This raw-amplitude grid is distinct from the replicated extension's labels
a/√20. There is no hidden division of A or a by √20 in this experiment.

For a normalized eigenvector, the exact decomposition

$$E=\psi^TA\psi+a\sum_i\xi_i|\psi_i|^2$$

separates hopping and potential contributions. The diagnostics additionally
measure IPR, N·IPR, participation, cumulative coordinate mass, support sizes,
edge ranks, energy-binned localization, and neighboring-gap statistics.

## Results and interpretation

Full-spectrum median N·IPR rises strongly between a=0 and a=32. For example,
at α=1.05 it changes from ≈8.655 to 73.16 at N=500 and ≈13.27 to 250.9
at N=10000. At α=2 the changes are ≈3.355 to 53.83 and ≈3.422 to 103.4.
The [selected summary](../results/latest_summary_tables/noise_selected_summary.csv)
and [112-operator completion table](../results/latest_summary_tables/noise_completion.csv)
retain the case coordinates.

Clean full-spectrum summaries include solver-basis values inside repeated
eigenspaces. The primary individual-state comparison masks the clean atom;
positive-disorder multiplicities are checked separately. A decrease in mean
edge N·IPR can coexist with a decrease in 80%-mass support: IPR emphasizes
the largest coordinates, while support size includes much of the tail.
Rank-window comparisons select current eigenvectors at each amplitude; they
do not track the same state continuously through crossings.

The colored workbook's **Energy profiles** are 60-bin energy summaries with
common edges across amplitudes within α, not a list of exact eigenvalues.
Its **Mass matrices** describe individual top-ten algebraic eigenvectors and
cumulative mass, including rank 1. At a=0 rank 1 is Perron, whereas the
non-Perron clean-edge analyses start at rank 2. Color records magnitude, not
an independently assigned localization phase.

## Code, figures, and reproduction

The implementation recovered from the immutable executed source snapshot is
[`pa_diagonal_uniform_noise.py`](../../src/small_world_mobility_edge/random_schrodinger/pa_diagonal_uniform_noise.py),
with [`pa_diagonal_uniform_noise_reporting.py`](../../src/small_world_mobility_edge/random_schrodinger/pa_diagonal_uniform_noise_reporting.py).
The case/global figures are packaged in the
[PA/Anderson report](../reports/PA_and_Anderson_report_20260924/), with input and
generator mappings in [figure_reproduction.csv](../figure_reproduction.csv).

```bash
python3 tools/run_pa_diagonal_uniform_noise_smoke.py --output-dir outputs/smoke/pa_diagonal_uniform_noise
python3 tools/hoffman_pa_diagonal_uniform_noise.py --help
```

The pilot generates one small graph and a complete diagnostic bundle. A
production rerun needs the original clean cache layout, a copied configuration
with its source root set, and a new frozen run identity. Follow the
[reproducibility guide](../reproducibility.md); do not reuse a completed run's
freeze after changing its source or configuration.
