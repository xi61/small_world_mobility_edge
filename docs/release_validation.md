# Release validation record

Release candidate: `codex/reproducible-release-20261005`

Validation date: 2026-10-05

The release is checked with:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q src tools scripts tests
python3 tools/run_pa_diagonal_uniform_noise_smoke.py --output-dir outputs/smoke/pa_diagonal_uniform_noise
python3 tools/audit_report_figures.py --write
python3 tools/build_source_inventory.py
python3 tools/validate_release.py
```

## Recorded outcome

- Unit and controlled-fixture suite: **171 tests passed**.
- Cross-model smoke: regular, Erdős–Rényi, Chung–Lu, and exact-degree rewired
  paths completed; every empirical density integrated to `1.000000`.
- General PA smoke: three `N=400`, `m=4` graph tasks completed and aggregated.
- PA diagonal-disorder smoke: COMPLETE for the deterministic `N=80` operator.
- Random 3-regular Anderson smoke: COMPLETE; maximum relative residual
  `5.77e-15`, with the central interval checked against a full spectrum.
- Figure audit: **96 packaged figures verified** across two report packages.
- Source/provenance inventory: **2,703 hashed entries** (the generated
  inventory intentionally excludes itself).
- Publication validator: passed for **2,704 publishable files**.
- Package build: `small_world_mobility_edge-0.3.0-py3-none-any.whl` built
  successfully; build SHA-256
  `6a6979dc111f07fae729a2614fdc7255f9ee3e3be00824ebba1ede2f875f38ff`.
- TeX validation: both self-contained reports compiled successfully with
  Tectonic. The combined report emitted only non-fatal underfull-box warnings.

The validator checks the GitHub per-file size limit, private path/secret
patterns, JSON parsing, shell syntax, required metadata, and report-figure
presence/checksums. Byte-compilation also passed; historical LaTeX strings in
the report revision utility were marked raw to eliminate invalid-escape
warnings.
