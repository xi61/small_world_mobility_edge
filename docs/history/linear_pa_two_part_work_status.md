# Parallel work status

Updated 2026-09-24 for the linear-PA two-part execution.

## Active source and ownership

- Project root: `<SOURCE_WORKSPACE>`.
- This root is not a Git repository; the authoritative executable revisions are immutable Hoffman source/configuration freezes.
- This task owns only the clean affine linear-PA localization, geometry, local-reference, and coupling analysis requested by `Linear_PA_two_part_Codex_implementation_prompt.md`.
- No diagonal-disorder, nonlinear-PA, or random-3-regular Anderson source or jobs were changed or launched by this execution.

Actively implemented files:

- `src/small_world_mobility_edge/preferential_attachment/linear_pa_two_part.py`
- `src/small_world_mobility_edge/preferential_attachment/linear_pa_two_part_reporting.py`
- `tools/hoffman_linear_pa_two_part.py`
- `tools/hoffman_linear_pa_two_part_*.sh`
- `tools/hoffman_submit_linear_pa_two_part_pipeline.sh`
- `tools/hoffman_retry_linear_pa_two_part_case.sh`
- `tests/test_linear_pa_two_part.py`
- `experiments/linear_pa_two_part_20260924/config.json`
- `experiments/linear_pa_two_part_20260924/pilot_config.json`

Shared pre-existing modules used read-only are `preferential_attachment/baseline.py`, `random_schrodinger/common.py`, and `random_schrodinger/tunneling.py`.

## Completed scientific execution

- Valid production root: `<HOFFMAN_SCRATCH>/linear_pa_two_part_20260924_v3`.
- Scientific freeze: `<production root>/freeze/executed_source_v3`.
- Scientific source/configuration ID: `ff36916b3b1552ce2c81de603e3c84e3f94ebf1c29ef2afc1405d8877eedb012`.
- Job ledger: `<production root>/JOB_LEDGER.csv`.
- Logs: `<production root>/logs`.
- Resolved production configuration: `N={500,10000}`, configurable default `alpha={1.05,1.10,1.25,1.50,2.00,2.50,3.00}`, one independent graph per `(N,alpha)`, `m=10`, clean unscaled simple adjacency.
- Freeze job `14888541`, validation job `14888542`, and end-to-end pilot `14888543` completed successfully. Validation passed 16 unit tests plus 14 controlled validation fixtures in the scientific freeze. The pilot produced 12 targets, 164 dictionary rows, and 42 local eigensystem caches.
- Part 1 arrays `14888544` (`N=500`) and `14888545` (`N=10000`) completed all seven alpha cases. Every requested full spectrum and full eigenvector matrix is retained with residual and cache identities.
- Part 2 array `14888546` (`N=500`) completed all seven cases with 95 fixed targets per case. Array `14888547` (`N=10000`) completed all seven cases with 129 fixed targets per case.
- Every production Part 2 case has status `COMPLETE`. Scientifically inadequate reference descriptions and zero eligible Schur rows are retained as outcomes, not converted to job failures.
- Scheduler-only adjustment for `14888547`: pending tasks were changed from four to two slots and task concurrency was raised to seven after live memory telemetry showed roughly 2 GB peak use. The already-running first task retained four slots. Scientific source, configuration, graph caches, and outputs were unchanged.
- Aggregate job `14888548` reached the completed science tables but failed in the plotting stage because Hoffman Matplotlib rejects the newer two-argument `set_xticks` API. The failed report directory and log are preserved; no scientific case output was affected.

## Report-only corrections

- The report-only revision changes reporting, tests, and launchers only. It does not regenerate graphs, eigendecompositions, or Part 2 target analyses.
- It uses `tools/hoffman_linear_pa_two_part_report_v2.sh` and `tools/hoffman_submit_linear_pa_two_part_report_v2.sh`.
- V4 report root: `<HOFFMAN_SCRATCH>/linear_pa_two_part_20260924_v3_reporting_v4`; freeze `14889915`, validation `14889916`, report `14889917`. All completed and the validation passed 18 tests plus 14 controlled fixtures. V4 is preserved but its narrative counted only the strict reference setting while the aggregate row counts included every sensitivity setting, so it did not surface the one qualified non-reference description.
- Corrected V5 report root: `<HOFFMAN_SCRATCH>/linear_pa_two_part_20260924_v3_reporting_v5`.
- V5 reporting freeze label: `executed_reporting_source_v5`.
- V5 retains the Hoffman Matplotlib compatibility fix and explicitly distinguishes dictionary/pass counts across all predeclared sensitivities, mass-informed sensitivities, and the strict `(x,y,r)=(0.80,0.20,2)` reference setting. It does not change any scientific table or rerun a simulation/eigendecomposition.
- The V5 report-only job ledger is authoritative for its submitted job IDs and final statuses.

## Preserved preliminary attempts

- V1 root: `<HOFFMAN_SCRATCH>/linear_pa_two_part_20260924_v1`, source ID `60586f1713558d4f8e4808bfee11e1eaf43d90e6ed6643cc78aab642235e5e80`. Its pilot exposed the acceptance-configuration propagation defect. Production was not accepted.
- V2 root: `<HOFFMAN_SCRATCH>/linear_pa_two_part_20260924_v2`, source ID `62ebf1f313e5ad2dbc738f5d19701c397af9d09f2baf6e40d25a45b4d91f5bf2`. Its pilot exposed over-restrictive mechanical locality caps and the alpha-slug scheduler gate mismatch. Production was not accepted.
- Historical and preliminary outputs remain preserved in their separate roots. No earlier result was overwritten.

## Disorder and Anderson boundary

- No additional PA diagonal-disorder or random-3-regular Anderson task was started by this execution.
- No existing disorder/benchmark job was cancelled, deleted, overwritten, or restarted.
- Any second task auditing those pipelines should remain read-only unless the user explicitly expands its authority.
