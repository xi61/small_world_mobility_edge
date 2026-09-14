# Preferential-attachment sublinear and extreme-edge execution report

Authoritative methodology: `pa_mobility_edge_gap_first_v2_atom_segmented`.

## Code and execution scope

The existing nonlinear generator was generalized to finite zeta >= 0; no duplicate generator was introduced. Existing affine/superlinear spectra were postprocessed, while only graph indices beyond 499 were added where the two narrowest ensemble-CDF edge cells required more support.

Changed implementation points:

- `preferential_attachment/baseline.py`: nonlinear kernel validation and the explicit zeta=0 uniform-weight boundary.
- `preferential_attachment/full_spectrum_scan.py`: sublinear regime validation, metadata, and separate canonical output hierarchy.
- `preferential_attachment/mobility_edge_production_aggregate.py`: stretched-exponential and nonlinear hub-edge diagnostics.
- `preferential_attachment/mobility_edge_family_summary.py`: sublinear family and cross-regime hub-edge comparisons.
- `preferential_attachment/extreme_edge_quantiles.py`: v2 gap-first cross-fitted ensemble-CDF edge analysis, support audit, adaptive companions, and comparison figures.
- `preferential_attachment/extreme_edge_augmentation.py`: deterministic, eigenvalues-only graph augmentation with content validation.
- Hoffman wrappers and focused tests under `tools/` and `tests/`.

## Reuse audit

- Sublinear zeta=0,0.25,0.40,0.50,0.60,0.75,0.90: `NEW_EIGENDECOMPOSITION_REQUIRED` (500 new graphs per point).
- Affine alpha=2.00,1.50,1.25,1.05 and superlinear zeta=1.05,1.20,1.50 base ensembles: `REPOSTPROCESS_ONLY`; their original 500 spectra were not recomputed.
- Narrow resolvable edge cells: `AUGMENT_GRAPHS_ONLY`; deterministic graph indices began at 500 and stopped at the recorded per-model target/G_max.
- Superlinear hub-edge comparison: eigenspectra were reused; degree arrays were reconstructed by deterministic graph replay where absent.

## Gap, quantile, and inference conventions

Every gap is formed inside one graph and one atom-free segment after positive-edge Perron removal. Raw adjacent-gap ratios are computed before unfolding. Requested edge cells use the historical Chung--Lu-aligned ensemble empirical-CDF energy coordinate, implemented as a held-out five-fold training-only CDF evaluated at each within-graph gap midpoint. The graph-local rank coordinate is retained alongside it. Consequently, the two width-5e-5 cells are resolvable in the ensemble-CDF coordinate even though they are finer than the single-graph rank lattice. A supported displayed histogram requires at least 100 positive within-graph gaps and 50 contributing graphs; this is pooled support across independent graphs, not 100 gaps from every graph.

The principal spacing histogram is the unit-mean five-fold cross-fitted unfolded spacing distribution. Raw-gap histograms are scale diagnostics only. Exact atoms, zero gaps, positive-conditioned ratios, and inclusive ratios remain separate products.

## Hoffman SGE jobs

| job_id | stage | model_parameter | graph_range | tasks | failed_records | nonzero_exits | final_status | retry_note |
|---|---|---|---|---|---|---|---|---|
| 14534086 | sublinear smoke | array smoke cases | small N smoke | 2 | 0 | 0 | SUCCESS |  |
| 14534098 | sublinear smoke aggregate | smoke family | small N smoke | 1 | 0 | 0 | SUCCESS |  |
| 14534099 | extreme-edge base postprocess | 4 affine + 3 superlinear | existing G=500 caches | 1 | 0 | 1 | FAILED_RETRIED | manifest schema validation correctly rejected source |
| 14534114 | sublinear production array | zeta=0 | 0-499 | 100 | 0 | 0 | SUCCESS |  |
| 14534115 | sublinear production array | zeta=0.25 | 0-499 | 100 | 0 | 0 | SUCCESS |  |
| 14534116 | sublinear production array | zeta=0.40 | 0-499 | 100 | 0 | 0 | SUCCESS |  |
| 14534117 | sublinear production array | zeta=0.50 | 0-499 | 100 | 0 | 0 | SUCCESS |  |
| 14534118 | sublinear production array | zeta=0.60 | 0-499 | 100 | 0 | 0 | SUCCESS |  |
| 14534119 | sublinear production array | zeta=0.75 | 0-499 | 100 | 0 | 0 | SUCCESS |  |
| 14534120 | sublinear production array | zeta=0.90 | 0-499 | 100 | 0 | 0 | SUCCESS |  |
| 14534126 | sublinear aggregate | zeta=0 | 0-499 | 1 | 0 | 0 | SUCCESS |  |
| 14534127 | sublinear aggregate | zeta=0.25 | 0-499 | 1 | 0 | 0 | SUCCESS |  |
| 14534129 | sublinear aggregate | zeta=0.40 | 0-499 | 1 | 0 | 0 | SUCCESS |  |
| 14534130 | sublinear aggregate | zeta=0.50 | 0-499 | 1 | 0 | 0 | SUCCESS |  |
| 14534131 | sublinear aggregate | zeta=0.60 | 0-499 | 1 | 0 | 0 | SUCCESS |  |
| 14534132 | sublinear aggregate | zeta=0.75 | 0-499 | 1 | 0 | 0 | SUCCESS |  |
| 14534133 | sublinear aggregate | zeta=0.90 | 0-499 | 1 | 0 | 0 | SUCCESS |  |
| 14534139 | sublinear family summary | all sublinear | aggregated G=500 products | 1 | 0 | 1 | FAILED_RETRIED | old Hoffman Matplotlib mathtext rejected sqrt label |
| 14535125 | sublinear family summary retry | all sublinear | aggregated G=500 products | 1 | 0 | 0 | SUCCESS | aggregation-only retry; no spectra recomputed |
| 14534140 | superlinear hub-edge initial | zeta=1.05 | 0-499 replay + cached spectra | 1 | 0 | 0 | SUPERSEDED | top-rank comparator defect produced NaN top-10 relation |
| 14534141 | superlinear hub-edge initial | zeta=1.10 | 0-499 replay + cached spectra | 1 | 0 | 0 | SUPERSEDED | top-rank comparator defect produced NaN top-10 relation |
| 14534142 | superlinear hub-edge initial | zeta=1.20 | 0-499 replay + cached spectra | 1 | 0 | 0 | SUPERSEDED | top-rank comparator defect produced NaN top-10 relation |
| 14534143 | superlinear hub-edge initial | zeta=1.30 | 0-499 replay + cached spectra | 1 | 0 | 0 | SUPERSEDED | top-rank comparator defect produced NaN top-10 relation |
| 14534144 | superlinear hub-edge initial | zeta=1.50 | 0-499 replay + cached spectra | 1 | 0 | 0 | SUPERSEDED | top-rank comparator defect produced NaN top-10 relation |
| 14534248 | superlinear hub-edge corrected | zeta=1.05 | 0-499 replay + cached spectra | 1 | 0 | 0 | SUCCESS | no eigenspectrum recomputation |
| 14534249 | superlinear hub-edge corrected | zeta=1.10 | 0-499 replay + cached spectra | 1 | 0 | 0 | SUCCESS | no eigenspectrum recomputation |
| 14534250 | superlinear hub-edge corrected | zeta=1.20 | 0-499 replay + cached spectra | 1 | 0 | 0 | SUCCESS | no eigenspectrum recomputation |
| 14534252 | superlinear hub-edge corrected | zeta=1.30 | 0-499 replay + cached spectra | 1 | 0 | 0 | SUCCESS | no eigenspectrum recomputation |
| 14534253 | superlinear hub-edge corrected | zeta=1.50 | 0-499 replay + cached spectra | 1 | 0 | 0 | SUCCESS | no eigenspectrum recomputation |
| 14534176 | extreme-edge base postprocess retry | 4 affine + 3 superlinear | existing G=500 caches | 1 | 0 | 1 | FAILED_RETRIED | old Hoffman Matplotlib set_ticks API mismatch after numerical products |
| 14534268 | extreme-edge base postprocess corrected | 4 affine + 3 superlinear | existing G=500 caches | 1 | 0 | 0 | SUCCESS | no eigenspectrum recomputation |
| 14534322 | extreme-edge augmentation | affine alpha=2.00 | 500-1524 | 205 | 0 | 0 | SUCCESS | target G=1525 |
| 14534323 | extreme-edge augmentation | affine alpha=1.50 | 500-1839 | 268 | 0 | 0 | SUCCESS | target G=1840 |
| 14534324 | extreme-edge augmentation | affine alpha=1.25 | 500-1739 | 248 | 0 | 0 | SUCCESS | target G=1740 |
| 14534325 | extreme-edge augmentation | affine alpha=1.05 | 500-1999 | 300 | 0 | 0 | SUCCESS | G_max=2000 |
| 14534326 | extreme-edge augmentation | superlinear zeta=1.05 | 500-1304 | 161 | 0 | 0 | SUCCESS | target G=1305 |
| 14534327 | extreme-edge augmentation | superlinear zeta=1.20 | 500-1999 | 300 | 0 | 0 | SUCCESS | G_max=2000 |
| 14534328 | extreme-edge augmentation | superlinear zeta=1.50 | 500-1999 | 300 | 0 | 0 | SUCCESS | G_max=2000 |
| 14534329 | extreme-edge final aggregate | 4 affine + 3 superlinear | base + augmentation caches | 1 | 0 | 0 | SUCCESS | wallclock 167.607 s; maxvmem 3.541G |

## Sublinear family summary

| zeta | mean dmax | Gini | top-1% share | zero mult. | max NIPR | q(max) | bulk r | negative r | positive r | R+ rank1 | R- rank1 | b+ top50 | b- top50 | candidates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 84.54 | 0.2611 | 0.03308 | 0 | 36.17 | 0.001 | 0.5311 | 0.5301 | 0.5292 | 1.981 | 1.33 | 0.8881 | 0.6613 | IPR_ONLY [0,0.02]; IPR_AND_R_SHIFT [0.98,1] |
| 0.25 | 102.8 | 0.2774 | 0.03852 | 0 | 44.4 | 0.001 | 0.5303 | 0.5313 | 0.53 | 1.722 | 1.267 | 0.7252 | 0.6041 | IPR_ONLY [0,0.022]; IPR_AND_R_SHIFT [0.978,1] |
| 0.4 | 119.3 | 0.2882 | 0.04315 | 0 | 53.46 | 0.001 | 0.5311 | 0.5305 | 0.5301 | 1.551 | 1.222 | 0.6426 | 0.573 | IPR_ONLY [0,0.024]; IPR_AND_R_SHIFT [0.976,1] |
| 0.5 | 133.6 | 0.296 | 0.04713 | 0 | 61.85 | 0.001 | 0.5308 | 0.53 | 0.5281 | 1.429 | 1.191 | 0.5827 | 0.547 | IPR_ONLY [0,0.026]; IPR_AND_R_SHIFT [0.974,1] |
| 0.6 | 153.5 | 0.3044 | 0.05208 | 0 | 74.24 | 0.001 | 0.5307 | 0.5313 | 0.5292 | 1.301 | 1.155 | 0.5328 | 0.5247 | IPR_ONLY [0,0.028]; IPR_AND_R_SHIFT [0.974,1] |
| 0.75 | 197.2 | 0.3181 | 0.06239 | 0 | 108.7 | 0.001 | 0.5303 | 0.5303 | 0.5294 | 1.111 | 1.1 | 0.4673 | 0.4938 | IPR_ONLY [0,0.03]; IPR_ONLY [0.97,1] |
| 0.9 | 277.2 | 0.3341 | 0.07861 | 0 | 184.8 | 0.001 | 0.5307 | 0.5309 | 0.5301 | 0.9985 | 1.043 | 0.4303 | 0.4711 | IPR_ONLY [0,0.034]; IPR_ONLY [0.966,1] |

### zeta=0

- Degree structure: mean d_max=84.54, d_max/N=0.01691, d_max/sum(d)=0.0008463, mean degree=19.98, minimum degree=10, Gini=0.2611, top-1/top-5/top-10/top-1% shares=0.0008463/0.003991/0.007673/0.03308, degree-m fraction=0.09081, highest-degree birth rank=8.272, mean component count=1. All graphs connected=True.
- Stretched-exponential diagnostic: mean slope=-0.09518, mean R2=0.9987; this is a validation diagnostic, not a forced fit.
- Spectrum: continuous range [-12.65, 18.45], mean Perron=31.59, mean negative edge=-12.19, mean positive non-Perron edge=18.18; mean zero multiplicity=0, mean -1 multiplicity=0, mean +1 multiplicity=0.
- Localization/spacing: max coarse NIPR=36.17 at q=0.001; bulk raw r=0.5311; negative-edge raw r=0.5301; positive-edge raw r=0.5292; positive-edge inclusive raw r=0.5292; mean positive-edge KS(GOE)/KS(Poisson)=0.01289/0.2129; maximum coarse zero-gap fraction=0.
- Hub-edge test: rank-1 R+=1.981, R-=1.33; top-50 b+=0.8881 (R2=0.7253), b-=0.6613 (R2=0.9766).
- Candidate screen: IPR_ONLY [0,0.02]; IPR_AND_R_SHIFT [0.98,1].

### zeta=0.25

- Degree structure: mean d_max=102.8, d_max/N=0.02057, d_max/sum(d)=0.00103, mean degree=19.98, minimum degree=10, Gini=0.2774, top-1/top-5/top-10/top-1% shares=0.00103/0.004817/0.009212/0.03852, degree-m fraction=0.1041, highest-degree birth rank=7.006, mean component count=1. All graphs connected=True.
- Stretched-exponential diagnostic: mean slope=-0.2643, mean R2=0.9987; this is a validation diagnostic, not a forced fit.
- Spectrum: continuous range [-13.27, 17.79], mean Perron=33.18, mean negative edge=-12.82, mean positive non-Perron edge=17.45; mean zero multiplicity=0, mean -1 multiplicity=0, mean +1 multiplicity=0.
- Localization/spacing: max coarse NIPR=44.4 at q=0.001; bulk raw r=0.5303; negative-edge raw r=0.5313; positive-edge raw r=0.53; positive-edge inclusive raw r=0.53; mean positive-edge KS(GOE)/KS(Poisson)=0.01301/0.2133; maximum coarse zero-gap fraction=0.
- Hub-edge test: rank-1 R+=1.722, R-=1.267; top-50 b+=0.7252 (R2=0.7987), b-=0.6041 (R2=0.9782).
- Candidate screen: IPR_ONLY [0,0.022]; IPR_AND_R_SHIFT [0.978,1].

### zeta=0.4

- Degree structure: mean d_max=119.3, d_max/N=0.02387, d_max/sum(d)=0.001195, mean degree=19.98, minimum degree=10, Gini=0.2882, top-1/top-5/top-10/top-1% shares=0.001195/0.005552/0.01057/0.04315, degree-m fraction=0.1133, highest-degree birth rank=6.54, mean component count=1. All graphs connected=True.
- Stretched-exponential diagnostic: mean slope=-0.5141, mean R2=0.9985; this is a validation diagnostic, not a forced fit.
- Spectrum: continuous range [-13.83, 17.29], mean Perron=34.4, mean negative edge=-13.32, mean positive non-Perron edge=16.9; mean zero multiplicity=0, mean -1 multiplicity=0, mean +1 multiplicity=0.
- Localization/spacing: max coarse NIPR=53.46 at q=0.001; bulk raw r=0.5311; negative-edge raw r=0.5305; positive-edge raw r=0.5301; positive-edge inclusive raw r=0.5301; mean positive-edge KS(GOE)/KS(Poisson)=0.01147/0.2133; maximum coarse zero-gap fraction=0.
- Hub-edge test: rank-1 R+=1.551, R-=1.222; top-50 b+=0.6426 (R2=0.862), b-=0.573 (R2=0.9799).
- Candidate screen: IPR_ONLY [0,0.024]; IPR_AND_R_SHIFT [0.976,1].

### zeta=0.5

- Degree structure: mean d_max=133.6, d_max/N=0.02673, d_max/sum(d)=0.001338, mean degree=19.98, minimum degree=10, Gini=0.296, top-1/top-5/top-10/top-1% shares=0.001338/0.006223/0.01181/0.04713, degree-m fraction=0.1197, highest-degree birth rank=6.958, mean component count=1. All graphs connected=True.
- Stretched-exponential diagnostic: mean slope=-0.8311, mean R2=0.9986; this is a validation diagnostic, not a forced fit.
- Spectrum: continuous range [-14.35, 16.91], mean Perron=35.39, mean negative edge=-13.75, mean positive non-Perron edge=16.49; mean zero multiplicity=0, mean -1 multiplicity=0, mean +1 multiplicity=0.
- Localization/spacing: max coarse NIPR=61.85 at q=0.001; bulk raw r=0.5308; negative-edge raw r=0.53; positive-edge raw r=0.5281; positive-edge inclusive raw r=0.5281; mean positive-edge KS(GOE)/KS(Poisson)=0.01339/0.2115; maximum coarse zero-gap fraction=0.
- Hub-edge test: rank-1 R+=1.429, R-=1.191; top-50 b+=0.5827 (R2=0.9027), b-=0.547 (R2=0.9816).
- Candidate screen: IPR_ONLY [0,0.026]; IPR_AND_R_SHIFT [0.974,1].

### zeta=0.6

- Degree structure: mean d_max=153.5, d_max/N=0.03069, d_max/sum(d)=0.001536, mean degree=19.98, minimum degree=10, Gini=0.3044, top-1/top-5/top-10/top-1% shares=0.001536/0.00709/0.0134/0.05208, degree-m fraction=0.1271, highest-degree birth rank=6.48, mean component count=1. All graphs connected=True.
- Stretched-exponential diagnostic: mean slope=-1.398, mean R2=0.9985; this is a validation diagnostic, not a forced fit.
- Spectrum: continuous range [-14.96, 16.72], mean Perron=36.55, mean negative edge=-14.28, mean positive non-Perron edge=16.06; mean zero multiplicity=0, mean -1 multiplicity=0, mean +1 multiplicity=0.
- Localization/spacing: max coarse NIPR=74.24 at q=0.001; bulk raw r=0.5307; negative-edge raw r=0.5313; positive-edge raw r=0.5292; positive-edge inclusive raw r=0.5292; mean positive-edge KS(GOE)/KS(Poisson)=0.01286/0.2131; maximum coarse zero-gap fraction=0.
- Hub-edge test: rank-1 R+=1.301, R-=1.155; top-50 b+=0.5328 (R2=0.9451), b-=0.5247 (R2=0.9825).
- Candidate screen: IPR_ONLY [0,0.028]; IPR_AND_R_SHIFT [0.974,1].

### zeta=0.75

- Degree structure: mean d_max=197.2, d_max/N=0.03944, d_max/sum(d)=0.001974, mean degree=19.98, minimum degree=10, Gini=0.3181, top-1/top-5/top-10/top-1% shares=0.001974/0.00904/0.01695/0.06239, degree-m fraction=0.1395, highest-degree birth rank=6.336, mean component count=1. All graphs connected=True.
- Stretched-exponential diagnostic: mean slope=-3.509, mean R2=0.9983; this is a validation diagnostic, not a forced fit.
- Spectrum: continuous range [-16.3, 16.24], mean Perron=38.8, mean negative edge=-15.43, mean positive non-Perron edge=15.6; mean zero multiplicity=0, mean -1 multiplicity=0, mean +1 multiplicity=0.
- Localization/spacing: max coarse NIPR=108.7 at q=0.001; bulk raw r=0.5303; negative-edge raw r=0.5303; positive-edge raw r=0.5294; positive-edge inclusive raw r=0.5294; mean positive-edge KS(GOE)/KS(Poisson)=0.01327/0.2122; maximum coarse zero-gap fraction=0.
- Hub-edge test: rank-1 R+=1.111, R-=1.1; top-50 b+=0.4673 (R2=0.9813), b-=0.4938 (R2=0.9855).
- Candidate screen: IPR_ONLY [0,0.03]; IPR_ONLY [0.97,1].

### zeta=0.9

- Degree structure: mean d_max=277.2, d_max/N=0.05545, d_max/sum(d)=0.002775, mean degree=19.98, minimum degree=10, Gini=0.3341, top-1/top-5/top-10/top-1% shares=0.002775/0.01252/0.02319/0.07861, degree-m fraction=0.1542, highest-degree birth rank=5.996, mean component count=1. All graphs connected=True.
- Stretched-exponential diagnostic: mean slope=-13.85, mean R2=0.9983; this is a validation diagnostic, not a forced fit.
- Spectrum: continuous range [-18.69, 17.65], mean Perron=42.1, mean negative edge=-17.37, mean positive non-Perron edge=16.61; mean zero multiplicity=0, mean -1 multiplicity=0, mean +1 multiplicity=0.
- Localization/spacing: max coarse NIPR=184.8 at q=0.001; bulk raw r=0.5307; negative-edge raw r=0.5309; positive-edge raw r=0.5301; positive-edge inclusive raw r=0.5301; mean positive-edge KS(GOE)/KS(Poisson)=0.01335/0.2121; maximum coarse zero-gap fraction=0.
- Hub-edge test: rank-1 R+=0.9985, R-=1.043; top-50 b+=0.4303 (R2=0.9882), b-=0.4711 (R2=0.9894).
- Candidate screen: IPR_ONLY [0,0.034]; IPR_ONLY [0.966,1].

## Exact extreme positive-edge intervals

| model | q interval | G | graphs | + gaps | zero | ratios | raw r +/- SE | |r-GOE| | |r-Pois| | KS GOE | KS Pois | energy range | status | atom | histogram panel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| affine alpha=2 | [0.995,0.996) | 1525 | 1525 | 7625 | 0 | 7627 | 0.5338 +/- 0.00342 | 0.002086 | 0.1475 | 0.01448 | 0.2114 | [13.27,14] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_2p0/crossfit_unfolded_histogram_panels.png |
| affine alpha=2 | [0.996,0.997) | 1525 | 1525 | 7621 | 0 | 7607 | 0.537 +/- 0.00345 | 0.001103 | 0.1507 | 0.01244 | 0.2133 | [14,14.87] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_2p0/crossfit_unfolded_histogram_panels.png |
| affine alpha=2 | [0.997,0.998) | 1525 | 1525 | 7602 | 0 | 7651 | 0.532 +/- 0.00351 | 0.003873 | 0.1457 | 0.01437 | 0.2061 | [14.87,15.86] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_2p0/crossfit_unfolded_histogram_panels.png |
| affine alpha=2 | [0.998,0.999) | 1525 | 1525 | 7624 | 0 | 7596 | 0.5344 +/- 0.00349 | 0.001493 | 0.1481 | 0.01788 | 0.2096 | [15.86,16.97] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_2p0/crossfit_unfolded_histogram_panels.png |
| affine alpha=2 | [0.999,0.9995) | 1525 | 1525 | 3787 | 0 | 3798 | 0.5274 +/- 0.005 | 0.008539 | 0.1411 | 0.0293 | 0.1951 | [16.97,17.67] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_2p0/crossfit_unfolded_histogram_panels.png |
| affine alpha=2 | [0.9995,0.9999) | 1525 | 1479 | 2755 | 0 | 2218 | 0.5409 +/- 0.00616 | 0.004989 | 0.1546 | 0.02792 | 0.2021 | [17.67,18.61] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_2p0/crossfit_unfolded_histogram_panels.png |
| affine alpha=2 | [0.9999,0.99995) | 1525 | 219 | 222 | 0 | 66 | 0.5257 +/- 0.0306 | 0.01024 | 0.1394 | 0.09364 | 0.2174 | [18.6,18.88] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_2p0/crossfit_unfolded_histogram_panels.png |
| affine alpha=2 | [0.99995,1) | 1525 | 118 | 120 | 0 | 27 | 0.5949 +/- 0.0555 | 0.05902 | 0.2086 | 0.1694 | 0.1121 | [18.87,19.6] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_2p0/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.5 | [0.995,0.996) | 1840 | 1840 | 9212 | 0 | 9187 | 0.5294 +/- 0.00321 | 0.006532 | 0.1431 | 0.03291 | 0.2013 | [15.37,16.73] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p5/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.5 | [0.996,0.997) | 1840 | 1840 | 9221 | 0 | 9197 | 0.5279 +/- 0.00325 | 0.007959 | 0.1416 | 0.03993 | 0.1968 | [16.72,18.41] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p5/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.5 | [0.997,0.998) | 1840 | 1840 | 9154 | 0 | 9197 | 0.5334 +/- 0.00324 | 0.002469 | 0.1471 | 0.03515 | 0.1958 | [18.4,20.19] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p5/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.5 | [0.998,0.999) | 1840 | 1840 | 9188 | 0 | 9183 | 0.5289 +/- 0.00318 | 0.007027 | 0.1426 | 0.03121 | 0.1953 | [20.18,21.88] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p5/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.5 | [0.999,0.9995) | 1840 | 1836 | 4534 | 0 | 4576 | 0.5279 +/- 0.00462 | 0.007983 | 0.1416 | 0.03981 | 0.1987 | [21.87,22.83] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p5/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.5 | [0.9995,0.9999) | 1840 | 1752 | 3361 | 0 | 2652 | 0.5493 +/- 0.00569 | 0.01343 | 0.163 | 0.05081 | 0.1942 | [22.82,24.06] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p5/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.5 | [0.9999,0.99995) | 1840 | 232 | 232 | 0 | 105 | 0.5766 +/- 0.0228 | 0.04067 | 0.1903 | 0.09692 | 0.2136 | [24.07,24.39] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p5/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.5 | [0.99995,1) | 1840 | 131 | 136 | 0 | 23 | 0.5976 +/- 0.0555 | 0.06168 | 0.2113 | 0.08104 | 0.2048 | [24.38,25.62] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p5/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.25 | [0.995,0.996) | 1740 | 1740 | 8660 | 0 | 8674 | 0.5161 +/- 0.0034 | 0.01977 | 0.1298 | 0.07415 | 0.1773 | [17.29,19.48] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p25/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.25 | [0.996,0.997) | 1740 | 1740 | 8777 | 0 | 8672 | 0.5178 +/- 0.00343 | 0.01811 | 0.1315 | 0.08362 | 0.1661 | [19.46,22.43] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p25/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.25 | [0.997,0.998) | 1740 | 1740 | 8657 | 0 | 8695 | 0.5206 +/- 0.00336 | 0.01531 | 0.1343 | 0.06265 | 0.1757 | [22.39,25.23] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p25/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.25 | [0.998,0.999) | 1740 | 1740 | 8670 | 0 | 8715 | 0.5332 +/- 0.00318 | 0.002717 | 0.1469 | 0.03135 | 0.1981 | [25.23,27.25] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p25/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.25 | [0.999,0.9995) | 1740 | 1739 | 4312 | 0 | 4317 | 0.536 +/- 0.00457 | 6.726e-05 | 0.1497 | 0.02894 | 0.1963 | [27.24,28.25] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p25/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.25 | [0.9995,0.9999) | 1740 | 1664 | 3170 | 0 | 2528 | 0.5382 +/- 0.00562 | 0.002253 | 0.1519 | 0.04052 | 0.1913 | [28.23,29.41] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p25/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.25 | [0.9999,0.99995) | 1740 | 240 | 242 | 0 | 84 | 0.5081 +/- 0.03 | 0.02784 | 0.1218 | 0.05118 | 0.2059 | [29.39,29.69] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p25/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.25 | [0.99995,1) | 1740 | 111 | 113 | 0 | 22 | 0.4226 +/- 0.0474 | 0.1133 | 0.0363 | 0.1186 | 0.1637 | [29.68,30.44] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p25/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.05 | [0.995,0.996) | 2000 | 2000 | 6391 | 0 | 6302 | 0.4865 +/- 0.00396 | 0.04937 | 0.1002 | 0.145 | 0.1192 | [27.42,31.58] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p05/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.05 | [0.996,0.997) | 2000 | 1999 | 6235 | 0 | 6276 | 0.4749 +/- 0.00376 | 0.06096 | 0.08865 | 0.05199 | 0.1793 | [31.54,33.41] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p05/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.05 | [0.997,0.998) | 2000 | 2000 | 6199 | 0 | 6302 | 0.5319 +/- 0.00377 | 0.003959 | 0.1456 | 0.01184 | 0.2139 | [33.4,34.27] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p05/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.05 | [0.998,0.999) | 2000 | 2000 | 6274 | 0 | 6286 | 0.5289 +/- 0.00382 | 0.006975 | 0.1426 | 0.01108 | 0.2137 | [34.26,35.06] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p05/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.05 | [0.999,0.9995) | 2000 | 1970 | 3115 | 0 | 3024 | 0.5442 +/- 0.00527 | 0.008289 | 0.1579 | 0.01881 | 0.2117 | [35.05,35.51] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p05/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.05 | [0.9995,0.9999) | 2000 | 1631 | 2066 | 0 | 1242 | 0.5461 +/- 0.00745 | 0.01024 | 0.1598 | 0.01413 | 0.2146 | [35.5,36.03] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p05/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.05 | [0.9999,0.99995) | 2000 | 102 | 102 | 0 | 21 | 0.3936 +/- 0.0626 | 0.1423 | 0.00735 | 0.1568 | 0.1397 | [36.02,36.16] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p05/crossfit_unfolded_histogram_panels.png |
| affine alpha=1.05 | [0.99995,1) | 2000 | 40 | 41 | 0 | 6 | 0.5473 +/- 0.124 | 0.01136 | 0.161 | 0.1656 | 0.1523 | [36.15,36.42] | INSUFFICIENT_EDGE_SUPPORT_AT_N5000 | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | affine_linear/affine_linear_degree_ccdf_alpha_1p05/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.05 | [0.995,0.996) | 1305 | 1305 | 6494 | 0 | 6543 | 0.5369 +/- 0.00375 | 0.0009785 | 0.1506 | 0.01575 | 0.2157 | [13.7,14.56] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p05/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.05 | [0.996,0.997) | 1305 | 1305 | 6543 | 0 | 6511 | 0.5379 +/- 0.00367 | 0.001962 | 0.1516 | 0.02078 | 0.2105 | [14.56,15.61] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p05/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.05 | [0.997,0.998) | 1305 | 1305 | 6507 | 0 | 6522 | 0.5314 +/- 0.00375 | 0.004529 | 0.1451 | 0.01679 | 0.2094 | [15.61,16.78] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p05/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.05 | [0.998,0.999) | 1305 | 1305 | 6530 | 0 | 6537 | 0.5297 +/- 0.00377 | 0.006153 | 0.1435 | 0.0226 | 0.2027 | [16.78,18.09] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p05/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.05 | [0.999,0.9995) | 1305 | 1305 | 3222 | 0 | 3248 | 0.5237 +/- 0.00527 | 0.01217 | 0.1374 | 0.03312 | 0.192 | [18.08,18.9] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p05/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.05 | [0.9995,0.9999) | 1305 | 1256 | 2369 | 0 | 1883 | 0.5362 +/- 0.00667 | 0.0002863 | 0.1499 | 0.04665 | 0.1925 | [18.89,20.03] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p05/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.05 | [0.9999,0.99995) | 1305 | 177 | 177 | 0 | 66 | 0.5169 +/- 0.0318 | 0.01904 | 0.1306 | 0.06397 | 0.1905 | [20.02,20.34] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p05/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.05 | [0.99995,1) | 1305 | 112 | 116 | 0 | 22 | 0.4454 +/- 0.0564 | 0.09053 | 0.05907 | 0.1459 | 0.1071 | [20.35,21.58] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p05/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.2 | [0.995,0.996) | 2000 | 2000 | 9973 | 0 | 9998 | 0.5344 +/- 0.00309 | 0.001532 | 0.1481 | 0.03409 | 0.1984 | [15.14,16.7] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p2/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.2 | [0.996,0.997) | 2000 | 2000 | 10042 | 0 | 10000 | 0.5296 +/- 0.00308 | 0.006322 | 0.1433 | 0.03984 | 0.1901 | [16.67,18.73] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p2/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.2 | [0.997,0.998) | 2000 | 2000 | 10007 | 0 | 9968 | 0.5301 +/- 0.00311 | 0.00584 | 0.1438 | 0.04894 | 0.1881 | [18.72,21] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p2/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.2 | [0.998,0.999) | 2000 | 2000 | 9947 | 0 | 10013 | 0.5257 +/- 0.00314 | 0.0102 | 0.1394 | 0.04493 | 0.1827 | [21,23.38] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p2/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.2 | [0.999,0.9995) | 2000 | 1994 | 4992 | 0 | 4957 | 0.5414 +/- 0.00434 | 0.005516 | 0.1551 | 0.05198 | 0.1843 | [23.37,24.86] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p2/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.2 | [0.9995,0.9999) | 2000 | 1889 | 3632 | 0 | 2909 | 0.5444 +/- 0.00538 | 0.008505 | 0.1581 | 0.0586 | 0.1793 | [24.84,26.87] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p2/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.2 | [0.9999,0.99995) | 2000 | 271 | 273 | 0 | 109 | 0.557 +/- 0.0219 | 0.02112 | 0.1707 | 0.09033 | 0.1893 | [26.85,27.41] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p2/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.2 | [0.99995,1) | 2000 | 135 | 137 | 0 | 21 | 0.5212 +/- 0.0629 | 0.01472 | 0.1349 | 0.1226 | 0.1994 | [27.4,29.18] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p2/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.5 | [0.995,0.996) | 2000 | 2000 | 9369 | 0 | 9367 | 0.5152 +/- 0.00315 | 0.02065 | 0.129 | 0.07414 | 0.1679 | [14.21,18.23] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p5/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.5 | [0.996,0.997) | 2000 | 2000 | 9449 | 0 | 9414 | 0.5009 +/- 0.00318 | 0.03505 | 0.1146 | 0.1057 | 0.1466 | [18.16,25.45] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p5/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.5 | [0.997,0.998) | 2000 | 2000 | 9542 | 0 | 9405 | 0.4816 +/- 0.00312 | 0.05434 | 0.09527 | 0.1369 | 0.1363 | [25.36,32.09] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p5/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.5 | [0.998,0.999) | 2000 | 2000 | 9276 | 0 | 9416 | 0.5167 +/- 0.0031 | 0.01917 | 0.1304 | 0.023 | 0.2008 | [32.08,34.34] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p5/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.5 | [0.999,0.9995) | 2000 | 1996 | 4651 | 0 | 4657 | 0.5284 +/- 0.00444 | 0.007467 | 0.1421 | 0.02146 | 0.2046 | [34.32,35.12] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p5/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.5 | [0.9995,0.9999) | 2000 | 1885 | 3438 | 0 | 2685 | 0.5282 +/- 0.00561 | 0.00768 | 0.1419 | 0.03111 | 0.2039 | [35.1,35.91] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p5/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.5 | [0.9999,0.99995) | 2000 | 228 | 228 | 0 | 64 | 0.545 +/- 0.0323 | 0.009119 | 0.1587 | 0.04318 | 0.2284 | [35.91,36.1] | OK | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p5/crossfit_unfolded_histogram_panels.png |
| superlinear zeta=1.5 | [0.99995,1) | 2000 | 98 | 99 | 0 | 18 | 0.4651 +/- 0.0684 | 0.07083 | 0.07878 | 0.1361 | 0.1748 | [36.09,36.67] | INSUFFICIENT_EDGE_SUPPORT_AT_N5000 | NONE_IN_REQUESTED_POSITIVE_EDGE_WINDOW | nonlinear/nonlinear_attachment_zeta_1p5/crossfit_unfolded_histogram_panels.png |

## Resolution audit

- [0.9999,0.99995): ensemble/CDF coordinate is mathematically resolvable; final support statuses: affine alpha=2=OK (222 gaps), affine alpha=1.5=OK (232 gaps), affine alpha=1.25=OK (242 gaps), affine alpha=1.05=OK (102 gaps), superlinear zeta=1.05=OK (177 gaps), superlinear zeta=1.2=OK (273 gaps), superlinear zeta=1.5=OK (228 gaps).
- [0.99995,1.0): ensemble/CDF coordinate is mathematically resolvable; final support statuses: affine alpha=2=OK (120 gaps), affine alpha=1.5=OK (136 gaps), affine alpha=1.25=OK (113 gaps), affine alpha=1.05=INSUFFICIENT_EDGE_SUPPORT_AT_N5000 (41 gaps), superlinear zeta=1.05=OK (116 gaps), superlinear zeta=1.2=OK (137 gaps), superlinear zeta=1.5=INSUFFICIENT_EDGE_SUPPORT_AT_N5000 (99 gaps).

## Figures

- `extreme_positive_edge_quantiles/affine_linear/affine_linear_degree_ccdf_alpha_1p05/crossfit_unfolded_histogram_panels.png`
- `extreme_positive_edge_quantiles/affine_linear/affine_linear_degree_ccdf_alpha_1p05/edge_evolution.png`
- `extreme_positive_edge_quantiles/affine_linear/affine_linear_degree_ccdf_alpha_1p05/interval_08_adaptive_companion_histograms.png`
- `extreme_positive_edge_quantiles/affine_linear/affine_linear_degree_ccdf_alpha_1p05/raw_gap_histogram_panels.png`
- `extreme_positive_edge_quantiles/affine_linear/affine_linear_degree_ccdf_alpha_1p25/crossfit_unfolded_histogram_panels.png`
- `extreme_positive_edge_quantiles/affine_linear/affine_linear_degree_ccdf_alpha_1p25/edge_evolution.png`
- `extreme_positive_edge_quantiles/affine_linear/affine_linear_degree_ccdf_alpha_1p25/raw_gap_histogram_panels.png`
- `extreme_positive_edge_quantiles/affine_linear/affine_linear_degree_ccdf_alpha_1p5/crossfit_unfolded_histogram_panels.png`
- `extreme_positive_edge_quantiles/affine_linear/affine_linear_degree_ccdf_alpha_1p5/edge_evolution.png`
- `extreme_positive_edge_quantiles/affine_linear/affine_linear_degree_ccdf_alpha_1p5/raw_gap_histogram_panels.png`
- `extreme_positive_edge_quantiles/affine_linear/affine_linear_degree_ccdf_alpha_2p0/crossfit_unfolded_histogram_panels.png`
- `extreme_positive_edge_quantiles/affine_linear/affine_linear_degree_ccdf_alpha_2p0/edge_evolution.png`
- `extreme_positive_edge_quantiles/affine_linear/affine_linear_degree_ccdf_alpha_2p0/raw_gap_histogram_panels.png`
- `extreme_positive_edge_quantiles/model_comparison/graph_equal_raw_r_heatmap.png`
- `extreme_positive_edge_quantiles/model_comparison/signed_ks_preference_heatmap.png`
- `extreme_positive_edge_quantiles/nonlinear/nonlinear_attachment_zeta_1p05/crossfit_unfolded_histogram_panels.png`
- `extreme_positive_edge_quantiles/nonlinear/nonlinear_attachment_zeta_1p05/edge_evolution.png`
- `extreme_positive_edge_quantiles/nonlinear/nonlinear_attachment_zeta_1p05/raw_gap_histogram_panels.png`
- `extreme_positive_edge_quantiles/nonlinear/nonlinear_attachment_zeta_1p2/crossfit_unfolded_histogram_panels.png`
- `extreme_positive_edge_quantiles/nonlinear/nonlinear_attachment_zeta_1p2/edge_evolution.png`
- `extreme_positive_edge_quantiles/nonlinear/nonlinear_attachment_zeta_1p2/raw_gap_histogram_panels.png`
- `extreme_positive_edge_quantiles/nonlinear/nonlinear_attachment_zeta_1p5/crossfit_unfolded_histogram_panels.png`
- `extreme_positive_edge_quantiles/nonlinear/nonlinear_attachment_zeta_1p5/edge_evolution.png`
- `extreme_positive_edge_quantiles/nonlinear/nonlinear_attachment_zeta_1p5/interval_08_adaptive_companion_histograms.png`
- `extreme_positive_edge_quantiles/nonlinear/nonlinear_attachment_zeta_1p5/raw_gap_histogram_panels.png`
- `nonlinear/n_5000_m_10/nonlinear_hub_edge_cross_regime_summary/hub_edge_cross_regime_comparison.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p0/global_spectrum/global_continuous_esd.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p0/global_spectrum/nonlinear_hub_edge/hub_spectral_edge_scaling.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p0/structural/degree_ccdf.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p0/structural/sublinear_stretched_exponential_ccdf.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p25/global_spectrum/global_continuous_esd.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p25/global_spectrum/nonlinear_hub_edge/hub_spectral_edge_scaling.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p25/structural/degree_ccdf.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p25/structural/sublinear_stretched_exponential_ccdf.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p4/global_spectrum/global_continuous_esd.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p4/global_spectrum/nonlinear_hub_edge/hub_spectral_edge_scaling.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p4/structural/degree_ccdf.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p4/structural/sublinear_stretched_exponential_ccdf.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p5/global_spectrum/global_continuous_esd.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p5/global_spectrum/nonlinear_hub_edge/hub_spectral_edge_scaling.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p5/structural/degree_ccdf.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p5/structural/sublinear_stretched_exponential_ccdf.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p6/global_spectrum/global_continuous_esd.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p6/global_spectrum/nonlinear_hub_edge/hub_spectral_edge_scaling.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p6/structural/degree_ccdf.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p6/structural/sublinear_stretched_exponential_ccdf.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p75/global_spectrum/global_continuous_esd.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p75/global_spectrum/nonlinear_hub_edge/hub_spectral_edge_scaling.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p75/structural/degree_ccdf.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p75/structural/sublinear_stretched_exponential_ccdf.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p9/global_spectrum/global_continuous_esd.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p9/global_spectrum/nonlinear_hub_edge/hub_spectral_edge_scaling.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p9/structural/degree_ccdf.png`
- `nonlinear/sublinear/n_5000_m_10/attachment_zeta_0p9/structural/sublinear_stretched_exponential_ccdf.png`
- `nonlinear/sublinear/n_5000_m_10/controlled_family_summary/figure_A_degree_ccdf_comparison.png`
- `nonlinear/sublinear/n_5000_m_10/controlled_family_summary/figure_B_global_esd_comparison.png`
- `nonlinear/sublinear/n_5000_m_10/controlled_family_summary/figure_C_n_times_ipr_vs_quantile.png`
- `nonlinear/sublinear/n_5000_m_10/controlled_family_summary/figure_D_raw_r_vs_quantile.png`
- `nonlinear/sublinear/n_5000_m_10/controlled_family_summary/figure_E_localization_level_statistics_vs_quantile.png`
- `nonlinear/sublinear/n_5000_m_10/controlled_family_summary/figure_F_localization_level_statistics_vs_energy.png`

## Portable numerical sources

- `nonlinear/sublinear/n_5000_m_10/controlled_family_summary/family_summary.csv`
- `nonlinear/n_5000_m_10/nonlinear_hub_edge_cross_regime_summary/hub_edge_cross_regime_summary.csv`
- `extreme_positive_edge_quantiles/model_comparison/extreme_edge_model_by_interval_summary.csv`
- `extreme_positive_edge_quantiles/model_comparison/augmentation_requirement_audit.csv`
- `pa_sublinear_extreme_edge_sge_job_ledger.csv`
- `pa_sublinear_extreme_edge_sync_checksum_audit.csv`

## Validation and interpretation

The production validators enforce consecutive graph indices, deterministic seeds, model/parameter/N/m/version agreement, readable chunks, no graph-boundary gap, no atom bridge, Perron removal, pre-unfolding raw r, and fold exclusion. Unsupported exact cells are labeled rather than plotted inferentially; adaptive companion intervals never replace their nominal requested rows.

Across the sublinear family, hub strength and edge NIPR increase smoothly toward zeta=1, while positive-conditioned raw r remains close to the GOE reference and cross-fitted unfolded KS is much smaller to GOE than to Poisson. The zeta=1/2 structural landmark is not accompanied by a demonstrated spectral transition. The hub-edge regressions retain strong rank association but do not support one universal lambda/sqrt(d) prefactor or exponent across nonlinear regimes, especially under superlinear condensation. High IPR with GOE-like positive spacings is therefore reported as edge localization/prelocalization, not a clean Anderson-style mobility edge.
