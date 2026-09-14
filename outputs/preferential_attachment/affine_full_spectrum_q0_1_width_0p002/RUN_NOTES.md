# Affine preferential-attachment full-spectrum run

## Run identity

- Date: 2026-08-13
- Hoffman project: `<HOFFMAN_SCRATCH>/small_world_mobility_edge_pa_full_20260813`
- Array job: `14319818` (300 tasks, five graphs per task)
- Corrected aggregation job: `14323911`
- Completion: 1,500/1,500 graphs and all three aggregates completed without task errors.

## Aligned design

Each setting uses `N=5000`, `m=10`, 500 graphs, exact mean degree 19.978,
the corresponding Chung--Lu graph-seed list, Perron removal, exact-zero
removal for the continuous scans, and 500 full-spectrum quantile bins of
width 0.002.  Each dense eigendecomposition is shared by the eigenvalue and
eigenvector analyses.

The affine attachment law is

`P(new vertex -> i | G_t) proportional to degree_i(t) + delta`,

with degree upper-tail exponent `alpha = 2 + delta/m`:

| alpha | delta | interpretation |
|---:|---:|---|
| 1.8 | -2.0 | weakest preference in this comparison |
| 1.5 | -5.0 | intermediate preference |
| 1.05 | -9.5 | strongest preference |

`alpha` is a preferential-attachment degree-CCDF exponent.  It is not the
same mathematical exponent as the Chung--Lu expected-weight density label;
the numerical labels are aligned for comparison, but identical degree laws
cannot be imposed in standard affine PA for the legacy Chung--Lu tails below
CCDF exponent one.

## Initial results

| alpha | mean maximum degree | mean zero multiplicity | any disconnected graph / isolated K2 | any exact E=+/-1 multiplicity | maximum mean N*IPR (q, E) | gap-ratio bins closer to GOE |
|---:|---:|---:|---:|---:|---:|---:|
| 1.8 | 458.964 | 0 | no | no | 342.20 (0.001, -19.26) | 500/500 |
| 1.5 | 688.574 | 0 | no | no | 351.25 (0.001, -23.02) | 500/500 |
| 1.05 | 2175.914 | 1854.914 | no | no | 708.98 (0.993, 20.48) | 496/498 valid bins |

For `alpha=1.05`, the two bins closer to Poisson are immediately adjacent to
the removed zero atom (`q=0.497` and `q=0.503`) and contain only 164 and 107
contributing graphs.  The two central displayed bins have no positive gaps.
They should not be interpreted as a bulk Poisson regime.  Away from that
central rank-deficient region, the positive spacings remain much closer to
GOE even though the eigenvectors have several strong concentration bands.

The preferential-attachment construction begins from a complete graph and
adds every new vertex with ten edges, so every realization is connected and
has no isolated K2 component.  The Chung--Lu small-fragment mechanism does
not recur.  The strong-preference setting instead develops an extensive zero
atom inside the connected graph.  In a representative graph, 3,245 vertices
remain at degree ten; they form an independent set and their neighbor union
contains only 1,641 vertices.  This alone guarantees at least 1,604 zero
modes by rank deficiency, compared with 1,844 observed zero modes in that
graph.  This is an internal neighborhood-span mechanism, not a fragment
mechanism.

## Output files

Every parameter directory contains:

- `eigenvalue_spacing_histograms.png`: local unfolded-spacing histograms with
  GOE and Poisson curves;
- `eigenvalue_spacing_summary.png` and CSV: graph-equal mean adjacent-gap
  ratio, absolute distance to GOE/Poisson expectations, KS distances for the
  positive-spacing distributions, and zero/small-spacing fractions;
- `eigenvalue_spacing_heatmap.png`: full-spectrum positive-spacing density;
- `eigenvector_percentile_scan_summary.png` and CSV: IPR, participation
  ratio, top-r mass, and coordinates needed for target mass;
- `eigenvector_coordinate_mass_heatmap.png`: full-spectrum cumulative top-r
  coordinate mass;
- `fragment_multiplicity_summary.csv`: components, isolated K2 counts, zero
  multiplicity, and exact E=+/-1 multiplicity;
- `run_manifest.json`: conventions, parameters, seeds, and output inventory.
