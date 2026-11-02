# Compact recorded results

These files contain existing experimental measurements, not new production
simulations performed for the publication update.

| Folder | Content | Reproduction |
|---|---|---|
| [pa_multifractality](pa_multifractality/) | 14 case manifests and five tables per case, four aggregate tables, seven figures, controlled fixtures | [Full method and commands](../experiments/pa_multifractality.md) |
| [pa_spectral_anatomy](pa_spectral_anatomy/) | Bin sensitivity, exact gaps, mass sets, shells, sign-pair overlaps, leakage and star tests; selected figures | [Audit and plotting scripts](../../tools/pa_spectral_anatomy/README.md) |
| [latest_summary_tables](latest_summary_tables/) | Exact-level, near-zero, diagonal-noise, and replicated-Anderson summaries | [Geometry](../experiments/single_graph_geometry.md), [disorder](../models/disorder_and_anderson.md) |

The multifractality aggregate can be regenerated from committed tables alone,
without large eigensystems. Per-vector recomputation and graph pictures need
the external caches described on the experiment pages. Original import
checksums are in [october_import_provenance.csv](../october_import_provenance.csv);
current publication checksums are in [source_inventory.csv](../source_inventory.csv).
Case manifests retain source-cache identities but use portable source paths.
The noise completion table replaces its machine-local path prefix with
`<HOFFMAN_SCRATCH>`; all numerical fields are unchanged.

The two older self-contained report packages and their 96 figures have a
separate [figure reproduction map](../figure_reproduction.csv). For the
October figures:

| Published figure(s) | Generator / source |
|---|---|
| `pa_multifractality/report/figures/multifractality_alpha_*.png` | `multifractality._plot_aggregate`, from the committed case tables |
| `pa_spectral_anatomy/figures/` audit-named figures | `tools/audit_pa_noncentral_targets.py`, original saved graph/eigensystem and binned report tables |
| `hubs.png`, `graph_a.png`, `spectrum10000.png` | `make_pa_followup_figures.py`; curated copies from the October project report |
| `spectrum5000.png` | `make_n5000_spectrum_visual.py` |
| `zeros5000.png` | `make_n5000_zero_retained_histogram.py` |
| `nearzero_connectors.png` | `analyze_zero_crossing_modes.py` |
| `nearzero_support.png`, `individual_edge.png` | Report-stage renderings of the committed summary/level tables; reproducible equivalents from `tools/plot_published_anatomy.py` |

```bash
python3 tools/plot_published_anatomy.py --output-dir outputs/anatomy/from_tables
```
