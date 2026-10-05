# Figure reproduction map

`figure_reproduction.csv` maps each packaged figure to its generating module, entry point, inputs, configuration, command template, checksum, and packaging step.

| Report package | Packaged figures |
|---|---:|
| `PA_and_Anderson_report_20260924` | 12 |
| `pa_three_model_edge_bulk_conclusive_report` | 84 |

The PA conclusive report contains 84 TeX-referenced figures. The combined report contains 11 TeX-referenced figures plus one retained alternate disorder comparison. Commands containing angle-bracket fields require the corresponding production run roots or immutable caches; the repository intentionally does not commit those multi-gigabyte intermediates.

Run `python tools/audit_report_figures.py --write` to verify all recorded assets and refresh the map.
