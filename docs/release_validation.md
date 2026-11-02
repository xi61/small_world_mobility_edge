# Release validation record

## November 2, 2026 update

- **186 unit/fixture tests passed**, including the original model suite,
  multifractality limits and cache aggregation guards, graph-audit kernels,
  and unchanged-checksum checks for imported numerical tables and figures.
- The multifractality end-to-end smoke passed on actual N=64 and N=128 PA
  graphs, including eigensystem residuals and both analysis stages.
- Rebuilding all 14 recorded multifractality cases' aggregate tables produced
  8456 moment-exponent rows, 3477 histogram-scaling rows, and seven figure
  sets. The two slope tables agree with the archived results to maximum
  absolute difference 1.34×10⁻¹⁵; both exported source-summary tables agree
  numerically exactly. Original recorded files were not overwritten.
- The four-model dense smoke passed; every density integrated to 1.000000.
- PA diagonal-disorder and random-regular Anderson smoke tests completed.
- The ported N=5000 exact-spectrum plot, both zero-retained histogram
  scripts, and N=10000 hub/shell/spectrum plotting script completed against
  original saved inputs, writing to separate temporary outputs.
- New table-only edge/near-zero plots rendered successfully and were visually
  checked. All local links in the 14 new/updated navigation pages resolved.
- All Python source/tools/tests byte-compiled; shell scripts are now syntax-
  checked **one file at a time**. The earlier multi-file `bash -n` invocation
  checked only its first script.
- All 96 figures in the two older report packages still passed their checksum
  audit. October artifact checks are part of the new unit tests and inventory.
- No Hoffman submission, new production ensemble, or optional missing-vector
  eigensolve was performed. Original source and result trees were read-only.

Commands run from the publication checkout (temporary output paths abbreviated):

```bash
python3 -m unittest discover -s tests -q
python3 -m compileall -q src tools tests
python3 tools/run_pa_multifractality_smoke.py --output-dir outputs/smoke/pa_multifractality
python3 tools/hoffman_pa_multifractality.py --config experiments/pa_multifractality_20261021/config.json aggregate --output-root docs/results/pa_multifractality/cases --report-dir outputs/rebuilt_multifractality
python3 tools/run_pa_diagonal_uniform_noise_smoke.py --output-dir outputs/smoke/disorder
python3 tools/run_anderson_rrg_smoke.py --output-dir outputs/smoke/anderson
python3 tools/plot_published_anatomy.py --output-dir outputs/anatomy
python3 tools/build_source_inventory.py
python3 tools/validate_release.py
git diff --check
```

## Earlier local release validation

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
