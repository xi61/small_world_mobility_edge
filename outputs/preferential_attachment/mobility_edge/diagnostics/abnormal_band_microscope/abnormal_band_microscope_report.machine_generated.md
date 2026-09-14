# Abnormal-band microscope report

Analysis version: pa_abnormal_band_microscope_aggregate_v1. Authoritative outputs were not modified.

## Executive result

Affine alpha=1.05 has a supported fine-grid low-r band spanning approximately q=0.99500-0.99710 on the standard dq_0p00005 grid. The lowest cell is [0.99650,0.99655) with raw r=0.4535 and same-triplet unfolded r=0.5019 from 313 ratios.

Superlinear zeta=1.50 has a corresponding band around q=0.99620-0.99845; its lowest fine cell has raw/unfolded r=0.4777/0.5038.

Graph augmentation decision: NO_AUGMENTATION_REQUIRED.

## Fine-grid model digest

| Model | Grid | Candidate band | Lowest cell raw / unfolded r | Ratios / graphs | Rank p10 / median / p90 | Classification |
|---|---|---|---:|---:|---:|---|
| affine alpha=2.00 | dq_0p0001 | 0.99740-0.99750 | 0.5351 / 0.5379 | 758 / 705 | 12.0 / 13.0 / 14.0 | H3 finite-rank hub-mode band |
| affine alpha=1.25 | dq_0p0001 | 0.99590-0.99600 | 0.5250 / 0.5199 | 870 / 794 | 19.0 / 21.0 / 23.0 | H1 bin-boundary sensitive; H3 finite-rank hub-mode band |
| affine alpha=1.05 | dq_0p00005 | 0.99500-0.99710 | 0.4535 / 0.5019 | 313 / 308 | 10.0 / 11.0 / 12.8 | H2 density-gradient contribution; H3 finite-rank hub-mode band; H4 robust intermediate non-GOE band |
| superlinear zeta=1.50 | dq_0p00005 | 0.99620-0.99845 | 0.4777 / 0.5038 | 466 / 457 | 9.0 / 10.0 / 12.0 | H2 density-gradient contribution; H3 finite-rank hub-mode band; H4 robust intermediate non-GOE band |

## Required scientific questions

1. Affine alpha=1.05 low-r band: yes if the standard/shifted overlap and independent blocks above remain nonempty; the exact supported interval is reported in the digest.
2. Estimated start/end: q=0.99500-0.99710 on the finest requested standard grid.
3. Resolution stability: see each model's dq_0p0002, dq_0p0001, and, for the two primary models, dq_0p00005 standard summaries and histogram panels.
4. Half-bin shift: alpha=1.05 overlap width=0.00205; zeta=1.50 overlap width=0.00225.
5. Sliding windows: alpha=1.05 minimum raw r=0.4644 at q=0.99430; zeta=1.50 minimum=0.4986 at q=0.99765.
6. Same-triplet unfolding: alpha=1.05 lowest-cell unfolded r=0.5019; density correction is material. Zeta=1.50 unfolded r=0.5038.
7. Full P(r): candidate-band panels compare observed raw and unfolded distributions with GOE and Poisson references under each primary model directory.
8. Consecutive spacings: correlations, confidence intervals, scatter, and log spacing-ratio plots are saved; these distinguish broad imbalance from simple negative alternation.
9. Fixed edge ranks: alpha=1.05 minimum fixed window is 11-20; zeta=1.50 minimum is 11-20.
10. Zeta=1.50: classification is H2 density-gradient contribution; H3 finite-rank hub-mode band; H4 robust intermediate non-GOE band.
11. Alpha=1.25: lowest fine raw/unfolded r=0.5250/0.5199; classification is H1 bin-boundary sensitive; H3 finite-rank hub-mode band.
12. Alpha=2.00 control: lowest fine raw/unfolded r=0.5351/0.5379; classification is H3 finite-rank hub-mode band.
13. Primary classifications: alpha=1.05: H2 density-gradient contribution; H3 finite-rank hub-mode band; H4 robust intermediate non-GOE band; zeta=1.50: H2 density-gradient contribution; H3 finite-rank hub-mode band; H4 robust intermediate non-GOE band.

## Statistical stability

- affine alpha=2.00: at its minimum fine cell the independent 500-graph block raw-r range is 0.5326-0.6357 across 4 blocks; bootstrap CI is [0.5175,0.5533].
- affine alpha=1.25: at its minimum fine cell the independent 500-graph block raw-r range is 0.4970-0.5379 across 4 blocks; bootstrap CI is [0.5081,0.5422].
- affine alpha=1.05: at its minimum fine cell the independent 500-graph block raw-r range is 0.4075-0.5131 across 4 blocks; bootstrap CI is [0.4230,0.4847].
- superlinear zeta=1.50: at its minimum fine cell the independent 500-graph block raw-r range is 0.4511-0.5163 across 4 blocks; bootstrap CI is [0.4551,0.4990].

## Support and augmentation

Separate histogram (100 positive gaps, 50 graphs) and mean-r (100 ratios, 50 graphs) gates were applied to every cell. Unsupported cells are masked in comparison heatmaps. The complete yield-based decision table is model_comparison/support_and_augmentation_recommendations.csv.

## Provenance and paths

- Cache-only Hoffman array job IDs: 14541581.
- Fine microscope Hoffman array job IDs: 14541697.
- Comparison/report Hoffman job ID: 14549767.
- Remote result root: <HOFFMAN_SCRATCH>/small_world_mobility_edge_pa_mobility_20260823/outputs/preferential_attachment/mobility_edge/diagnostics/abnormal_band_microscope.
- Local synchronized result root: outputs/preferential_attachment/mobility_edge/diagnostics/abnormal_band_microscope/.
- Upstream base/augmentation cache checksums and job metadata are recorded under cached_n5000_reanalysis/MODEL/source_cache_checksums.csv.

## Main output map

- Per-model nested summaries, shifted grids, histograms, rank maps, block replication, negative-edge mirror, fixed-rank results, and sliding windows live under alpha_2p0, alpha_1p25, alpha_1p05, and zeta_1p50.
- Cross-model raw-r, unfolded-r, and signed KS heatmaps live under model_comparison.
- Cells failing ratio support are masked; no outermost q>0.9999 computation was launched.
