# Preferential-attachment mobility-edge implementation and execution report

**Project:** `small_world_mobility_edge`
**Production date:** August 23, 2026
**Production platform:** Hoffman2 SGE/Grid Engine
**Main setting:** $N=5000$, $G=500$, full adjacency spectrum, Perron eigenpair removed

## Executive result

The complete affine-linear and nonlinear preferential-attachment experiments are now implemented and executed. The strongest reproducible result is **spectral-edge eigenvector concentration without a broad GOE-to-Poisson transition in the neighboring eigenvalue spacings**. The central bulk remains close to GOE in every non-atom, $m\geq5$ production family. Exact-degree rewiring changes the detailed edge spectrum and reduces some localization, but does not remove the edge-concentration pattern; this makes the realized degree sequence the leading mechanism, with attachment history and graph organization contributing secondary corrections.

No result is labeled a mobility edge from IPR alone. The report keeps four effects separate:

1. extended, GOE-like bulk behavior;
2. concentrated edge eigenvectors with still-GOE-like local spacings;
3. exact atoms and sparse-graph rank deficiencies;
4. a genuine joint IPR and GOE-to-Poisson crossover, which was **not established** in these runs.

## A. Implementation

### Graph models and canonical parameters

The affine-linear model attaches each new vertex to $m$ distinct old vertices with weights

\[
  w_i=d_i+\delta,
  \qquad \delta=m(\alpha-2),
\]

where `degree_ccdf_alpha` is defined through

\[
  \Pr(D\geq k)\sim k^{-\alpha}.
\]

The nonlinear model uses

\[
  w_i=d_i^\zeta,
\]

with canonical parameter `attachment_zeta`. It does not assume that the resulting degree distribution is a power law. Both generators start from $K_{m+1}$, choose $m$ distinct targets without replacement, produce a simple undirected graph, and are deterministic under the saved seed.

### Gap-first statistics

For each graph, the continuous non-Perron eigenvalues are sorted before any pooling. Every within-graph gap is formed once,

\[
  \delta_i=\lambda_{i+1}-\lambda_i,
  \qquad q_i^{\mathrm{gap}}=(i+1/2)/M,
\]

and each raw adjacent-gap ratio is assigned to its middle eigenvalue,

\[
  r_i=\frac{\min(\delta_i,\delta_{i+1})}
  {\max(\delta_i,\delta_{i+1})},
  \qquad q_i^r=(i+1)/M.
\]

The primary level statistic is the graph-equal mean raw $r$, with GOE and Poisson references $0.53590$ and $0.38629$. No gap is formed across graphs. Exact atoms are segmented before the continuous scan, so removing an atom cannot create an artificial gap bridging its two sides. Inclusive and positive-gap-conditioned ratios, zero gaps, zero/nonzero pairs, omitted zero/zero pairs, and contributing graph counts are all retained.

### Cross-fitted unfolding and histogram gate

Unfolded spacings are secondary. Graphs are placed deterministically into five folds by `graph_index % 5`; each held-out fold is unfolded using an empirical density/CDF estimated only from the other four folds. Each displayed histogram must contain at least 100 positive within-graph gaps. A narrower window is marked insufficient or symmetrically widened, with both nominal and actual bounds recorded.

### Future-proof cache and resume behavior

New production chunks retain per-eigenpair eigenvalue, rank, quantile, IPR, $N\mathrm{IPR}$, participation statistics, top-10/50/100/200 mass, and fixed-mass radii, without retaining dense eigenvectors. They also retain complete degree arrays and orderings, concentration measures, component statistics, atom multiplicities, model metadata, seeds, and SGE provenance. Resume logic reopens and validates both files in a chunk, including graph indices, seeds, shapes, offsets, parameter metadata, and required fields; filename existence alone is not accepted.

### Code map

- `src/small_world_mobility_edge/preferential_attachment/baseline.py`: canonical affine and nonlinear generators; nonlinear distinct-target sampling; ζ=1 consistency.
- `src/small_world_mobility_edge/preferential_attachment/full_spectrum_scan.py`: production configuration, per-eigenpair caching, graph structure, exact-degree rewiring, content validation, and structural replay.
- `src/small_world_mobility_edge/preferential_attachment/mobility_edge_postprocess.py`: atom segmentation, gap-first raw ratios, fivefold cross-fitting, adaptive windows, joined summaries, and candidate screening.
- `src/small_world_mobility_edge/preferential_attachment/mobility_edge_cached_pipeline.py`: legacy-cache reprocessing, global spectrum, coarse/fine outputs, and plots.
- `src/small_world_mobility_edge/preferential_attachment/mobility_edge_production_aggregate.py`: production-cache aggregation, degree/tail validation, heatmaps, and cached fine localization.
- `src/small_world_mobility_edge/preferential_attachment/mobility_edge_family_summary.py`: aligned family tables and Figures A–F.
- `src/small_world_mobility_edge/preferential_attachment/mobility_edge_selection.py`: reproducible parameter selection for the density sensitivity.
- `src/small_world_mobility_edge/preferential_attachment/mobility_edge_m_sensitivity_summary.py`: $m\in\{2,5,10,20\}$ comparisons.
- `src/small_world_mobility_edge/preferential_attachment/mobility_edge_rewiring_comparison.py`: paired original-minus-rewired spectrum and quantile diagnostics.
- `src/small_world_mobility_edge/empirical_spectral_density.py`: graph-level fivefold empirical density/CDF estimation.
- `tools/audit_pa_legacy_cache.py` and `tools/hoffman_pa_*`: cache audit, deterministic SGE arrays, aggregation, summaries, plot backfill, and copy-back.

Focused tests cover deterministic graph generation, distinct targets, ζ=1 consistency, gap assignment, no cross-graph gaps, raw-r scale invariance, zero-gap handling, histogram gating/widening, fold separation, CDF normalization, exact-degree rewiring, canonical metadata, and content-aware resume. The implemented suite passed 90 tests before production, and a Hoffman smoke run passed before the full arrays were submitted.

## B. Hoffman execution

All graph generation, $5000\times5000$ eigendecomposition, aggregation, fine scans, density estimation, and rewiring production ran on Hoffman2. Array tasks normally covered five graphs each.

| Phase | Hoffman job IDs | Result |
|---|---|---|
| Legacy cache audit/durable protection | read-only audit and copy | 300 expected partial files for each of α=1.8, 1.5, 1.05 verified; indices 0–499, seeds, sizes, and SHA-256 checksums saved |
| Legacy spectral reprocessing | 14510955; plot backfill 14511099 | complete; no eigendecomposition |
| Legacy structural seed replay | 14510937–14510939; reaggregation 14511081–14511083 | complete; graph generation only |
| New affine α=2.0 | array 14510944; retries 14511088, 14511089; aggregate 14511090 | complete; tasks 49–54 had stale/incomplete metadata and task 76 exited 137, then passed targeted retries |
| New affine α=1.25 | array 14510946; aggregate 14510962 | complete |
| New affine α=1.10 | array 14510947; aggregate 14510963 | complete |
| Affine family summary/selection | 14511039 | complete; selected α*=1.25 |
| Affine $m$-sensitivity | arrays 14512459–14512461; aggregates 14512462–14512464; summary 14512465 | complete for $m=2,5,20$, reusing $m=10$ |
| Nonlinear ζ sweep | arrays 14512487–14512491; aggregates 14512492–14512496; family summary 14512497 | complete; selected ζ*=1.30 |
| Nonlinear $m$-sensitivity | arrays 14512776–14512778; aggregates 14512779, 14512780, 14512782; summary 14512783 | complete for $m=2,5,20$, reusing $m=10$ |
| Selected affine exact-degree rewiring | array 14513116; aggregate 14513118; comparison 14513120 | complete; all 100 array tasks, aggregate, and comparison exited 0 |
| Selected nonlinear exact-degree rewiring | array 14513117; retries 14513168 and 14513169; aggregate 14513171; comparison 14513173 | complete; original tasks 31 and 34–36 exited 137 at only 1.5–1.6 GB, then passed unchanged-seed targeted retries |
| Durable production-cache copy | 14513349 | complete, exit 0; source and destination each contain 23,487 files and 6,984,667,701 bytes |

The original failed outputs and logs were preserved. The corrected analysis is identified as `pa_mobility_edge_gap_first_v2_atom_segmented`; superseded atom-bridging results remain under `superseded/pa_mobility_edge_gap_first_v1_atom_bridge_bug`.

## C. Reuse audit and durable data

The α=1.8, 1.5, and 1.05 eigenspectra and localization summaries were reprocessed from the existing cache. No eigendecomposition was repeated for them. Degree arrays missing from the old spectral cache were recovered by deterministic seed replay without diagonalization.

New eigendecompositions were run only for affine α=2.0, 1.25, 1.10; the nonlinear ζ sweep; new $m$-sensitivity points; and two selected rewired controls. The already valid selected $m=10$ result was reused in each $m$-summary.

The protected legacy cache is at:

`<HOFFMAN_HOME>/small_world_mobility_edge_cache/pa_full_20260813`

with provenance at:

`<HOFFMAN_HOME>/small_world_mobility_edge_cache/pa_full_20260813_provenance`

The complete production tree, including the large per-eigenpair partial caches, is preserved durably on Hoffman at:

`<HOFFMAN_HOME>/small_world_mobility_edge_cache/pa_mobility_20260823`

The scratch production tree remains at:

`<HOFFMAN_SCRATCH>/small_world_mobility_edge_pa_mobility_20260823/outputs/preferential_attachment/mobility_edge`

Local copy-back excludes only `partials/`; it retains manifests, provenance, graph-level compressed tables, structural summaries, coarse/fine tables, cross-fit models, and figures. Post-copy validation found an exact remote/local summary-tree match of 20,287 files and 1,236,862,214 bytes; all 117 copied JSON files parse successfully.

## D. Affine-linear PA results

All rows below use $N=5000,m=10,G=500$. Edge $r$ is the positive-gap-conditioned raw statistic. The “classification” is a screening result, not a declaration of a mobility edge.

| α | realized Hill α | mean $d_{\max}$ | Gini | zero fraction | edge slopes $b_+,b_-$ | max $N\mathrm{IPR}$ ($q$) | bulk/edge raw r | classification |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2.00 | 1.911 | 374.9 | 0.3466 | 0 | 0.461, 0.482 | 279.5 (0.001) | 0.5308 / 0.5307 | IPR only |
| 1.80 | 1.753 | 459.0 | 0.3595 | 0 | 0.462, 0.482 | 342.2 (0.001) | 0.5307 / 0.5302 | IPR only |
| 1.50 | 1.469 | 688.6 | 0.3883 | 0 | 0.466, 0.488 | 351.2 (0.001) | 0.5307 / 0.5297 | IPR only |
| 1.25 | 1.204 | 1131.5 | 0.4284 | 0 | 0.474, 0.497 | 372.0 (0.003) | 0.5306 / 0.5295 | IPR + screening-level r shift |
| 1.10 | 1.016 | 1756.7 | 0.4667 | 0.1491 | 0.481, 0.504 | 586.5 (0.995) | 0.5309 / 0.5277 | atom-contaminated for selection |
| 1.05 | 0.954 | 2175.9 | 0.4830 | 0.3711 | 0.488, 0.511 | 709.0 (0.993) | 0.5305 / 0.5252 | atom-contaminated for selection |

The realized degree tails track the requested ordering, with finite-$N$ bias at the heaviest tails. The observed continuous spectral range broadens from approximately $[-21.74,20.28]$ at α=2.0 to $[-107.59,36.77]$ at α=1.05, and becomes increasingly asymmetric only at the extreme edges; the median positive/negative magnitudes remain balanced. There are no $E=\pm1$ atoms in the $m=10$ affine family; α=1.10 and 1.05 instead develop mean zero multiplicities 745.3 and 1854.9. Thus this connected PA family does not reproduce the Chung–Lu isolated-$K_2$, $E=\pm1$ mechanism.

The fitted edge relation

\[
  \log|\lambda_{(j)}|=a+b\log d_{(j)}
\]

has $b\approx0.46\text{–}0.51$, close to the degree-driven $1/2$ relation. Eigenvector concentration increases strongly at both spectral edges as the degree tail becomes heavier. In contrast, non-atom bulk and edge raw-r means remain near 0.53, much closer to GOE than Poisson. Fine cross-fitted histograms are likewise closer to GOE: for selected α=1.25, the smallest saved KS distances across fine windows are 0.0079 to GOE and 0.1551 to Poisson. The detected negative-edge, central, and positive-edge regions are therefore candidate concentration regions, not evidence of a broad mobility edge.

The reproducible selection rule chose α*=1.25 because it has strong sustained non-atom edge concentration, no zero atom, and enough continuous spectral resolution.

### Affine density sensitivity

At α=1.25, changing $m=2,5,10,20$ changes the mean degree approximately to $4,10,20,40$; this is a density sensitivity, not a degree-matched test. At $m=2$, the zero fraction is 0.476 and mean $E=\pm1$ multiplicities are about 105 each. At $m=5$, the zero fraction is 0.141, while $m=10,20$ have no zero atom. Maximum coarse $N\mathrm{IPR}$ decreases from 951.1 to 570.98, 371.95, and 179.79 as $m$ increases. The $m=2$ fine histograms can be closer to Poisson, but the extensive atoms/rank deficiency prevent interpreting that as a clean continuous-spectrum crossover. For $m\geq5$, and most cleanly $m\geq10$, the bulk remains GOE-like.

## E. Nonlinear PA results

All rows below use $N=5000,m=10,G=500$. No degree power-law exponent is inferred from ζ; the structural variables measure hub condensation directly.

| ζ | mean $d_{\max}$ | $d_{\max}/N$ | top-1% degree share | Gini | zero fraction | max $N\mathrm{IPR}$ ($q$) | bulk/edge raw r | classification |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1.05 | 448.1 | 0.0896 | 0.1069 | 0.3539 | 0 | 342.6 (0.001) | 0.5307 / 0.5303 | IPR only |
| 1.10 | 550.0 | 0.1100 | 0.1213 | 0.3619 | 0 | 416.8 (0.001) | 0.5306 / 0.5308 | IPR only |
| 1.20 | 895.5 | 0.1791 | 0.1624 | 0.3816 | 0 | 433.4 (0.001) | 0.5304 / 0.5294 | IPR + screening-level r shift |
| 1.30 | 1575.5 | 0.3151 | 0.2262 | 0.4078 | 0 | 495.3 (0.003) | 0.5311 / 0.5293 | IPR + screening-level r shift |
| 1.50 | 3353.7 | 0.6707 | 0.3768 | 0.4625 | 0.0590 | 686.8 (0.003) | 0.5307 / 0.5277 | atom-contaminated for selection |

Increasing ζ produces clear condensation: the largest vertex grows from about 9% to 67% of all vertices in degree, while the upper 1% takes 10.7% to 37.7% of degree mass. The continuous spectral range broadens from approximately $[-24.09,22.20]$ at ζ=1.05 to $[-128.71,36.77]$ at ζ=1.50, and the edge eigenvectors become more concentrated. Nevertheless, the non-atom bulk raw ratio stays around 0.530–0.531.

The finest ζ=1.50 scan finds a few extremely negative-edge cells whose histograms are closer to Poisson, but their raw $r\approx0.448\text{–}0.473$ remains between the two references and the effect does not persist through the interior. Central histogram anomalies occur near the removed zero atom and are not supported by the primary raw-r statistic. The evidence therefore supports a narrow extreme-edge effect at most, not a broad mobility edge.

The selection rule chose ζ*=1.30 because it shows strong nontrivial condensation and edge concentration without the 5.9% zero atom that excludes ζ=1.50.

### Nonlinear density sensitivity

At ζ=1.30, $m=2,5,10,20$ gives mean zero fractions 0.451, 0.0527, 0, and 0. Maximum $N\mathrm{IPR}$ falls from 984.6 to 705.6, 495.3, and 287.6 as density increases. At $m=2$, the maximum degree is 2396 and takes 12.0% of total degree; sparse atom effects also pull the bulk raw ratio down to 0.5155. At $m\geq5$ the bulk returns to approximately 0.531, and at $m\geq10$ the spectrum is atom-free. This again separates sparse structural degeneracy from the clean edge-localization question.

## F. Exact-degree rewiring

Both controls use 10 requested double-edge swaps per edge and a maximum-tries factor of 20. Graphs are paired by graph index and graph seed, and exact labeled degree-array equality is required.

### Nonlinear ζ=1.30, $m=10$

All 500 pairs pass exact labeled-degree equality and remain connected. Mean edge overlap is 0.07761. The rank-matched ESD changes modestly: mean two-sample KS distance is 0.02348, while the mean lower and upper non-Perron edges shift by −0.399 and −0.817 under rewiring. Neither version has $0$ or $\pm1$ atoms.

Across central bulk bins, rewired minus original averages are:

- $\Delta(N\mathrm{IPR})=-0.331$;
- $\Delta r=-0.000361$;
- $\Delta C_{50}=-0.00950$.

The original negative-edge candidate is $q\in[0,0.046]$; after rewiring it is $q\in[0,0.042]$. The positive-edge IPR-only region changes from $[0.954,1]$ to $[0.960,1]$. Thus rewiring reduces some coordinate concentration but preserves the qualitative edge pattern and GOE-like bulk spacing.

### Affine α=1.25, $m=10$

All 500 pairs pass exact labeled-degree equality and remain connected. Mean edge overlap is 0.07201. The mean rank-matched ESD KS distance is 0.03829, and the mean lower and upper non-Perron edges shift by −0.391 and −0.680. Neither version has $0$ or $\pm1$ atoms.

Across central bulk bins, rewired minus original averages are:

- $\Delta(N\mathrm{IPR})=-0.672$;
- $\Delta r=+0.000244$;
- $\Delta C_{50}=-0.01770$.

The original negative-edge interval $[0,0.058]$ contracts to $[0,0.046]$, and the positive-edge interval $[0.942,1]$ contracts to $[0.954,1]$. The weaker original central IPR-only interval $[0.490,0.512]$ disappears after rewiring. Thus organization beyond degree sequence affects the magnitude and the small central feature, while the principal edge concentration and GOE-like bulk spacing survive.

## G. Main result paths

The local result root is:

`outputs/preferential_attachment/mobility_edge/`

Main aligned family figures are saved under each `controlled_family_summary/` directory:

- `figure_A_degree_ccdf_comparison.png`
- `figure_B_global_esd_comparison.png`
- `figure_C_n_times_ipr_vs_quantile.png`
- `figure_D_raw_r_vs_quantile.png`
- `figure_E_localization_level_statistics_vs_quantile.png`
- `figure_F_localization_level_statistics_vs_energy.png`
- `family_summary.csv`

Every parameter directory also contains:

- `structural/degree_ccdf.png` and exact structural CSV/JSON summaries;
- `global_spectrum/global_continuous_esd.png`, atom tables, and degree/spectral-edge matching tables;
- `coarse/q0_1_dq_0p002/eigenvector_localization_metrics.png`;
- `coarse/q0_1_dq_0p002/eigenvector_coordinate_mass_heatmap.png`;
- `coarse/q0_1_dq_0p002/localization_and_raw_r_vs_quantile.png`;
- `coarse/q0_1_dq_0p002/localization_and_raw_r_vs_energy.png`;
- `coarse/q0_1_dq_0p002/joined_localization_level_statistics.csv`;
- `fine/fine_window_summary.csv`, `fine/joined_fine_localization_spacing.csv`, and fine histogram panels;
- `unfolding/crossfit_k5/` fold models and manifest;
- `rewiring_comparison/original_vs_exact_degree_rewired.png` for the selected controls.

Density-sensitivity summaries are under:

- `affine_linear/n_5000_m_sensitivity/degree_ccdf_alpha_1p25/`;
- `nonlinear/n_5000_m_sensitivity/attachment_zeta_1p3/`.

## H. Interpretation

1. **Extended bulk:** In atom-free $m=10$ families, bulk raw $r\approx0.530\text{–}0.531$, close to GOE and far from Poisson. Cross-fitted spacing histograms generally agree.
2. **Localized/concentrated edges with GOE-like spacings:** $N\mathrm{IPR}$, top-coordinate mass, and hub concentration rise strongly at both edges as affine tails become heavier or nonlinear attachment becomes more superlinear. The neighboring eigenvalue spacings usually remain GOE-like. This is the dominant observation.
3. **Atoms and motifs:** Connected PA graphs at $m=10$ eliminate the Chung–Lu isolated-$K_2$, $E=\pm1$ mechanism. Heavy affine tails, strong superlinearity, or low $m$ instead create zero atoms/rank deficiency; $m=2$ also creates $\pm1$ multiplicity. Atom-contaminated low ratios are not Poisson evidence.
4. **Rewiring mechanism:** Exact-degree rewiring retains the qualitative edge concentration and almost unchanged bulk $r$, while shifting the detailed spectrum and localization magnitude. Degree heterogeneity is therefore the leading mechanism, but graph organization beyond degrees is not irrelevant.
5. **Mobility-edge conclusion:** The experiments do not establish a broad spectral interval with a simultaneous sustained IPR rise and GOE-to-Poisson crossover. The defensible wording is “degree-driven edge localization/prelocalization with GOE-like local level statistics over most of the continuous spectrum,” plus narrow extreme-edge and atom-related exceptions requiring separate finite-size study.
