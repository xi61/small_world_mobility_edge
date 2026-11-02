# Replicated disorder and the random-regular Anderson benchmark

[Disorder models](../models/disorder_and_anderson.md) · [Configuration and seeds](../../experiments/random_schrodinger/anderson_repair_extension_20260924_v1.json)

## Designs that must remain separate

| Setting | Fresh replicated PA | Random 3-regular |
|---|---|---|
| Models | Affine α=1.05, nonlinear ζ=1.5, affine α=2 | Connected simple degree-three graphs |
| N | 1250,2500,5000 | 1250,2500,5000 |
| Graphs per model/size | 3 | 3 |
| Potential streams per graph | 2 | 2 |
| Amplitudes | a/√20=0,0.5,2,8 | Raw a=0,4,8,10,16; a=9 added |
| Physical central window | \|E\|≤0.1√20 | \|E\|≤0.1 |
| Unique operators | 27(1+2·3)=189 | 9(1+2·5)=99 |

Both use H=A+a diag(ξ), ξ uniform on [−1,1), with hopping one and no
post-disorder rescaling. The clean operator is counted once per graph.
Potential fields are paired across positive amplitudes. The a=9 regular
extension added 18 operators to the original 81 under the configured
conditional refinement rule.

The earlier pilot repair is also implemented: 48 graph/potential/amplitude
inputs become 45 distinct operators after deduplicating the three clean
graphs. The stable clean zero-space dimensions of its three N=5000 PA models
are 1844,240,0. Repeated blocks use projector/subspace diagnostics rather
than arbitrary solver-basis IPR comparisons.

## Statistics and numerical checks

For consecutive levels, δᵢ=Eᵢ₊₁−Eᵢ and

$$r_i=\frac{\min(\delta_i,\delta_{i+1})}{\max(\delta_i,\delta_{i+1})}.$$

The ratio requires no unfolding. The configured references are 2 log 2−1
≈0.38629 for Poisson, ≈0.53 for GOE, and ≈0.5359 for the GOE surmise.
Ratios use consecutive original-level triplets with their middle level in
the window and retained guards. Triplets touching an excluded clean atom are
discarded, **not reconnected across the deleted block**.

The regular benchmark solves a guarded central interval and checks interval
completeness against dense reference spectra. Residual, normalization,
sampled orthogonality, index/energy alignment, and cache-identity checks are
required before aggregation. PA additionally retains edge bands and total-
width-0.02 quantile windows centered at 25%,50%,75%.

For each operator/window, vector-level medians are computed first. Potential
replicates are averaged within a graph, then graphs receive equal weight.
The graph is the uncertainty unit; two potentials on one adjacency are not
two independent graphs. Unfolding references leave out the entire target
graph, including all of its potentials.

## Recorded finite-size behavior

In the affine α=1.05 q75 **spectral window**, the graph-averaged median
participation fractions at N=1250,2500,5000 are approximately
0.1076,0.1118,0.1087 when clean and 0.00830,0.00705,0.00559 at a/√20=8.
Participation fraction is 1/(N·IPR). “q75” is not the 75th percentile of
participation. The clean sequence is approximately size-stable here; the
strong-disorder sequence decreases.

The regular benchmark's central mean raw gap ratios are:

| a | N=1250 | N=2500 | N=5000 |
|---:|---:|---:|---:|
| 0 | 0.5388 | 0.5258 | 0.5444 |
| 4 | 0.4916 | 0.5319 | 0.5352 |
| 8 | 0.4579 | 0.4278 | 0.4172 |
| 9 | 0.3672 | 0.4036 | 0.4004 |
| 10 | 0.4103 | 0.4004 | 0.3991 |
| 16 | 0.3752 | 0.3891 | 0.4210 |

This provides a disorder-driven shift toward the Poisson reference, with
visible finite-sample variation rather than monotone values in every row.
In the literature convention with uniform disorder [−W/2,W/2], W=2a.
Any regular-graph critical-disorder comparison must use that conversion and
cannot be transferred into a critical PA amplitude.

Source tables: [PA graph points](../results/latest_summary_tables/anderson_pa_graph_points.csv),
[PA uncertainty](../results/latest_summary_tables/anderson_pa_graph_uncertainty.csv),
[regular graph points](../results/latest_summary_tables/anderson_rrg_graph_points.csv),
[regular uncertainty](../results/latest_summary_tables/anderson_rrg_graph_uncertainty.csv).

## Code and reproduction

The modules in `src/small_world_mobility_edge/random_schrodinger/` are
`disorder_replication.py`, `anderson_rrg_benchmark.py`, `anderson_utils.py`,
`anderson_reporting.py`, and `zero_subspace_correction.py`. The SGE execution
family is `tools/hoffman_anderson_*`, including the separate a=9 array.

```bash
python3 tools/run_anderson_rrg_smoke.py --output-dir outputs/smoke/anderson_rrg
```

This small benchmark verifies complete central-window coverage against the
full spectrum. The [Hoffman guide](../hoffman_reproduction.md) and frozen
configuration specify production seed tables, windows, resource requests,
and the conditional a=9 rule. Production results are finite-size evidence;
these grids do not determine a critical exponent or a PA mobility edge.
