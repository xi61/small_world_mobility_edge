# Abnormal-band microscope report

Report refinement: `pa_abnormal_band_microscope_scientific_interpretation_v2`
Numerical pipeline: `pa_abnormal_band_microscope_v1` / `pa_abnormal_band_microscope_aggregate_v1`
Authoritative method: `pa_mobility_edge_gap_first_v2_atom_segmented`

All results use $N=5000$, $m=10$, graph-equal statistics, within-graph gaps, atom-segmented spectra, five-fold training-only ensemble-CDF unfolding, and the same eigenvalue triplet for raw and unfolded ratios. The authoritative PA results were not modified.

## Executive conclusion

The two primary models contain reproducible finite-rank low-$r$ regions, but the anomaly is partly amplified by the steep edge-density gradient.

| Model | Main microscope result | Interpretation |
|---|---|---|
| affine α=1.05 | On the targeted $5\times10^{-5}$ grid, the contiguous band is $q\approx[0.99500,0.99710)$; the shifted-grid interval is $[0.995025,0.997075)$. The lowest cell, $[0.99650,0.99655)$, has raw/unfolded $r=0.4535/0.5019$. | H2 + H3 + H4: density-gradient contribution, narrow finite-rank band, and a residual unfolded non-GOE signal. The broad scan is already depressed at $q=0.994$, so the true lower onset is not bracketed. |
| superlinear ζ=1.50 | The fine-grid band is $q\approx[0.99620,0.99845)$; the shifted interval is $[0.996175,0.998475)$. The minimum cell, $[0.99790,0.99795)$, has raw/unfolded $r=0.4777/0.5038$. | H2 + H3 + H4: a stable, localized finite-rank band with a material density correction but a significant remaining unfolded depression. |
| affine α=1.25 | One $10^{-4}$ cell reaches raw/unfolded $r=0.5250/0.5199$, but its shifted counterpart moves and the confidence intervals include GOE. | Weak intermediate finite-rank crossover; H1/H3, not a robust H4 band. |
| affine α=2.00 | No cell is below the $r_{\rm GOE}-0.01=0.5259$ band criterion. The finest-grid minimum is raw/unfolded $r=0.5351/0.5379$. | Clean GOE-like control. The narrow contributing rank range is expected at the edge and is not, by itself, an H3 anomaly. |

All standard and half-shifted cells pass both support gates. No new graphs were required.

## Answers to the required scientific questions

### 1. Does affine α=1.05 really contain a low-r band near q=0.995–0.997?

Yes. At $dq=5\times10^{-5}$, 41 of 50 standard cells have raw $r<0.5259$, and the contiguous minimum-containing interval is $[0.99500,0.99710)$. Its standard/shifted overlap is 0.00205 in $q$, nearly the full band. At the lowest cell, 313 ratios from 308 graphs give

- raw $r=0.4535$, bootstrap 95% CI ([0.4230,0.4847]);
- unfolded $r=0.5019$, bootstrap 95% CI ([0.4728,0.5290]).

Both upper confidence limits are below the GOE reference $r_{\rm GOE}=0.5359$.

### 2. Where does the α=1.05 band begin and end?

The upper endpoint is resolved at approximately $q=0.99710$ on the standard fine grid and $q=0.997075$ after a half-bin shift.

The lower endpoint is not resolved by this experiment. The $dq=0.0002$ band touches the requested lower boundary at $q=0.9940$, and the narrowest sliding window has its minimum at $q_c=0.99430$. Therefore the defensible statement is:

- within the prespecified fine region, a band spans approximately $q=0.99500$ to $0.99710$;
- in the full microscope region, the depression begins at or below $q=0.994$.

A future extension below 0.994 would be needed only to locate the onset, not to establish the requested 0.995–0.997 anomaly.

### 3. Does the signal survive all three resolutions?

Yes, although the coarse coordinate exposes more than one local trough.

| Model | $dq=0.0002$ standard band | $dq=0.0001$ standard band | $dq=0.00005$ standard band |
|---|---:|---:|---:|
| α=1.05 | [0.9940, 0.9958) | [0.9945, 0.9958) | [0.9950, 0.9971) |
| ζ=1.50 | [0.9966, 0.9982) | [0.9962, 0.9982) | [0.9962, 0.99845) |

Here a band means the contiguous minimum-containing cells with raw $r<r_{\rm GOE}-0.01$. The α=1.05 endpoint movement shows real substructure in the ensemble-CDF coordinate rather than a perfectly flat interval; the fine targeted grid nevertheless resolves a coherent band.

### 4. Does it survive half-bin shifts?

Mostly yes.

- α=1.05: standard/shifted overlap is 0.00160 at $dq=0.0002$, only 0.00015 at $dq=0.0001$, and 0.00205 at $dq=0.00005$. Thus the $10^{-4}$ curve is boundary-sensitive, but the primary fine-grid band is not.
- ζ=1.50: overlaps are 0.00150, 0.00200, and 0.00225, respectively; its location is especially stable.
- α=1.25: the isolated standard and shifted low cells do not overlap.
- α=2.00: neither grid contains a cell below the band threshold.

### 5. Does it survive sliding-window analysis?

Yes for both primary models and for all requested widths.

- α=1.05 minima for widths 0.0002, 0.0005, and 0.001 are raw/unfolded $r=0.4644/0.4649$, (0.4688/0.5212), and (0.4637/0.4616).
- ζ=1.50 minima are (0.4986/0.5079), (0.4791/0.4963), and (0.4816/0.4873).

The valley therefore does not fragment into isolated fine cells. For α=1.05, the moving-window minimum at $q_c=0.99430$ also confirms that the depression reaches the lower scan boundary.

### 6. Is unfolded r also depressed?

Yes, but less strongly at the sharpest raw minima.

- α=1.05: $0.4535\rightarrow0.5019$ in the minimum fine cell.
- ζ=1.50: $0.4777\rightarrow0.5038$.

This is material evidence for H2, but the unfolded bootstrap upper limits (0.5290 and 0.5270) remain below GOE. Fixed-rank CDF-unfolded results also remain low: 0.4796 for α=1.05 ranks 11–20 and 0.5011 for ζ=1.50 ranks 11–20. Density variation is therefore a contributor, not a complete explanation.

### 7. Does P$r$ shift systematically?

Yes. At the α=1.05 minimum, raw (P$r$) is closer to the Poisson ratio reference than GOE (KS 0.114 versus 0.176), while same-triplet unfolding switches the preference back toward GOE (0.093 versus 0.183). At the ζ=1.50 minimum, both raw and unfolded (P$r$) remain closer to GOE (0.127/0.081 versus Poisson 0.201/0.227), but their means are depressed.

The single-spacing histograms remain more GOE-like than Poisson-like in both models. For example, unfolded spacing KS distances at the minima are 0.088 versus 0.174 for α=1.05 and 0.101 versus 0.169 for ζ=1.50. This supports a consecutive-spacing effect rather than wholesale Poissonization of (P(s)).

### 8. Are consecutive spacings unusually correlated?

Only modestly, and not in a way that uniquely explains the anomaly.

- α=1.05 minimum: raw correlation (-0.100), CI ([-0.213,0.009]); unfolded (-0.119), CI ([-0.224,-0.019]).
- ζ=1.50 minimum: raw (-0.091), CI ([-0.166,-0.008]); unfolded (-0.090), CI ([-0.175,-0.001]).
- α=2.00 control minimum: approximately (-0.259), stronger than either abnormal model.

The fractions with spacing ratio outside [0.5,2] are 0.604/0.518 (raw/unfolded) for α=1.05 and 0.567/0.530 for ζ=1.50. There is some excess imbalance, especially before unfolding, but simple negative alternation is not the sole mechanism.

### 9. Which fixed edge ranks create the effect?

The effect is a narrow finite-rank feature.

- α=1.05: ranks 11–20 are strongest (raw/CDF-unfolded $r=0.4715/0.4796$); ranks 21–40 remain low at (0.4910/0.4914), and recovery is nearly complete by ranks 81–120 ((0.5306/0.5306)). The fine minimum receives ranks p10/median/p90 = 10/11/12.8.
- ζ=1.50: ranks 11–20 give (0.4940/0.5011); ranks 21–40 give (0.5210/0.5241), with recovery at ranks 41 and beyond. The fine minimum receives ranks 9/10/12.
- α=1.25: ranks 11–20 and 21–40 are mildly low, (0.5183/0.5200) and (0.5217/0.5223).
- α=2.00: fixed-rank values remain approximately 0.524–0.534 without an intermediate-rank valley comparable to the primary models.

### 10. Does the same phenomenon occur for ζ=1.50?

Yes. It is more stable in $q$ under boundary shifts than the α=1.05 band, although its raw depression is less extreme. Its fine minimum uses 466 ratios from 457 graphs; raw and unfolded confidence intervals are ([0.4551,0.4990]) and ([0.4823,0.5270]).

### 11. Is α=1.25 an intermediate version?

At fixed rank, yes: ranks 11–40 are mildly depressed and localization is much stronger than in the control. In fine $q$ bins, however, the evidence is weak. The minimum raw/unfolded values (0.5250/0.5199) have confidence intervals containing GOE, and the standard/shifted low cells do not overlap. It is best described as a weak, boundary-sensitive finite-rank crossover rather than a resolved band.

### 12. Is α=2.00 clean?

Yes. No standard or shifted $q$ cell is below the $r_{\rm GOE}-0.01$ criterion. At its lowest $10^{-4}$ cell, ratio-distribution KS distances strongly favor GOE over Poisson (raw 0.055 versus 0.240; unfolded 0.054 versus 0.241), and spacing KS distances do the same (approximately 0.016–0.018 versus 0.209).

### 13. Final classification

- α=1.05: H2 + H3 + H4, with some $10^{-4}$-grid boundary sensitivity. It is a density-modulated finite-rank non-GOE band, not a pure bin artifact.
- ζ=1.50: H2 + H3 + H4. It is the cleanest stable finite-rank band in the ensemble-CDF coordinate.
- α=1.25: H1 + weak H3; unresolved as a true $q$-band.
- α=2.00: null/GOE control (H5 in the sense that apparent single-cell minima are ordinary fluctuations, not a band).

## Localization overlay

Where cached eigenvectors exist, localization supports the finite-rank interpretation. At the fine minima:

| Model | NIPR | PR/N | C10 | C50 | C200 |
|---|---:|---:|---:|---:|---:|
| α=2.00 | 96.9 | 0.01134 | 0.334 | 0.490 | 0.636 |
| α=1.25 | 386.5 | 0.00300 | 0.447 | 0.519 | 0.650 |
| ζ=1.50 | 437.3 | 0.00267 | 0.482 | 0.515 | 0.591 |

The α=1.05 authoritative legacy eigenspectrum cache contains eigenvalues but not per-eigenpair localization arrays, so NIPR, PR/N, and concentration values are unavailable for that model. Recomputing 2,000 eigenvector sets was not justified because all spacing/ratio cells were already supported; the rank mapping and fixed-rank experiment still locate its effect sharply.

## Negative-edge mirror

The negative edge has a raw analogue, but it is more density-sensitive.

- α=1.05 negative $dq=0.0002$ minimum: raw/unfolded $r=0.4871/0.5391$.
- ζ=1.50 negative minimum: (0.4850/0.5042).
- At fixed negative ranks 1–5, α=1.05 changes $0.4018\rightarrow0.5350$ and ζ=1.50 changes $0.3980\rightarrow0.5346$ under CDF unfolding.

Thus the spectacular raw negative-edge rank-1–5 dip is mostly an edge-density artifact. The positive intermediate-rank band is more robust after unfolding.

## Statistical stability and numerical checks

At the primary fine minima, all four independent graph blocks reproduce a low raw value:

- α=1.05 raw block values: 0.4515, 0.4075, 0.5131, 0.4457;
- ζ=1.50 raw block values: 0.4900, 0.5163, 0.4532, 0.4511.

Individual fine-cell block counts are 66–90 ratios for α=1.05 and 104–129 for ζ=1.50, so the α=1.05 block values are descriptive rather than separately inferential; the full graph-bootstrap intervals are the primary uncertainty statements.

No tested cell contains a zero gap, and no gap in the broad 0.995–0.998 solver audit is below (10^{-8}) relative scale. The anomaly is not a near-degeneracy or eigensolver-tolerance artifact.

## Support and augmentation

Every requested standard and shifted cell satisfies both requirements:

- histogram: at least 100 positive gaps from at least 50 graphs;
- mean $r$: at least 100 valid ratios from at least 50 ratio-contributing graphs.

Validation covered 25/40/50 cells for α=1.05, 25/40/70 for ζ=1.50, and 25/40 for both comparators, with shifted grids containing one fewer cell. The validation result is `PASS` with zero issues. The yield rule was

\[
G_{\rm target}=\left\lceil\frac{100}{\widehat y_r}\,1.25\right\rceil,
\]

capped at 10,000 graphs. Because every cell already passed, the decision was `NO_AUGMENTATION_REQUIRED` and no additional eigendecompositions were launched.

## Provenance

- Cache-only reanalysis array: Hoffman job `14541581`, tasks 1–4, all exit 0.
- Fine microscope array: Hoffman job `14541697`, tasks 1–4, all exit 0.
- Initial aggregate attempt: Hoffman job `14541731`, exit 1 from an old-Matplotlib API incompatibility; no numerical outputs were damaged.
- Corrected aggregate and validation: Hoffman job `14549767`, exit 0.
- Remote diagnostic root: `<HOFFMAN_SCRATCH>/small_world_mobility_edge_pa_mobility_20260823/outputs/preferential_attachment/mobility_edge/diagnostics/abnormal_band_microscope/`.
- Local diagnostic root: `outputs/preferential_attachment/mobility_edge/diagnostics/abnormal_band_microscope/`.
- Validation: [`model_comparison/validation_report.json`](model_comparison/validation_report.json).
- Cache checksums, graph-index ranges, seeds/job provenance, and validation notes: `cached_n5000_reanalysis/<model>/source_cache_checksums.csv` and per-model manifests.

## Main figures and data

Primary α=1.05 diagnostics:

- [`dq_0p00005/standard_vs_half_shifted_r.png`](alpha_1p05/dq_0p00005/standard_vs_half_shifted_r.png)
- [`dq_0p00005/unfolded_spacing_histogram_panel.png`](alpha_1p05/dq_0p00005/unfolded_spacing_histogram_panel.png)
- [`ratio_histograms/candidate_low_band_p_r.png`](alpha_1p05/ratio_histograms/candidate_low_band_p_r.png)
- [`spacing_pair_correlations/candidate_low_band_pairs.png`](alpha_1p05/spacing_pair_correlations/candidate_low_band_pairs.png)
- [`sliding_windows/sliding_window_r.png`](alpha_1p05/sliding_windows/sliding_window_r.png)
- [`dq_0p00005/q_to_edge_rank.png`](alpha_1p05/dq_0p00005/q_to_edge_rank.png)
- [`fixed_rank/fixed_rank_r_and_localization.png`](alpha_1p05/fixed_rank/fixed_rank_r_and_localization.png)
- [`dq_0p00005/independent_graph_block_replication.png`](alpha_1p05/dq_0p00005/independent_graph_block_replication.png)

Primary ζ=1.50 diagnostics:

- [`dq_0p00005/standard_vs_half_shifted_r.png`](zeta_1p50/dq_0p00005/standard_vs_half_shifted_r.png)
- [`dq_0p00005/unfolded_spacing_histogram_panel.png`](zeta_1p50/dq_0p00005/unfolded_spacing_histogram_panel.png)
- [`ratio_histograms/candidate_low_band_p_r.png`](zeta_1p50/ratio_histograms/candidate_low_band_p_r.png)
- [`spacing_pair_correlations/candidate_low_band_pairs.png`](zeta_1p50/spacing_pair_correlations/candidate_low_band_pairs.png)
- [`sliding_windows/sliding_window_r.png`](zeta_1p50/sliding_windows/sliding_window_r.png)
- [`dq_0p00005/q_to_edge_rank.png`](zeta_1p50/dq_0p00005/q_to_edge_rank.png)
- [`fixed_rank/fixed_rank_r_and_localization.png`](zeta_1p50/fixed_rank/fixed_rank_r_and_localization.png)
- [`dq_0p00005/localization_r_ks_overlay.png`](zeta_1p50/dq_0p00005/localization_r_ks_overlay.png)
- [`dq_0p00005/independent_graph_block_replication.png`](zeta_1p50/dq_0p00005/independent_graph_block_replication.png)

Cross-model figures:

- [`dq_0p0001_raw_r_heatmap.png`](model_comparison/dq_0p0001_raw_r_heatmap.png)
- [`dq_0p0001_unfolded_r_heatmap.png`](model_comparison/dq_0p0001_unfolded_r_heatmap.png)
- [`dq_0p0001_signed_ks_heatmap.png`](model_comparison/dq_0p0001_signed_ks_heatmap.png)
- [`model_digest.csv`](model_comparison/model_digest.csv)
- [`support_and_augmentation_recommendations.csv`](model_comparison/support_and_augmentation_recommendations.csv)
