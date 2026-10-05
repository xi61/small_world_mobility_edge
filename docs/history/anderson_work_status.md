# Anderson/random Schrodinger work status

**Updated:** 2026-09-24 (America/Los_Angeles)

## Ownership and scope

This chat owns only diagonal-disorder experiments with `H(a)=A+a*diag(xi)` on the three PA families and the separate random 3-regular Anderson benchmark. Clean `a=0` calculations are controls. The completed clean-PA coupling/tunneling run, its shared `handoff_*` modules, `PARALLEL_WORK_STATUS.md`, and its output root are read-only dependencies.

## Completion state

- Workflow complete on Hoffman at 2026-09-24T10:17:59Z; no owned job remains queued or running.
- Historical six-sweep PA pilot reused cache-only: 48 inputs resolved to 45 unique checkpoints plus three deduplicated `a=0` references. No pilot eigensolve was repeated.
- Three missing clean-baseline vector supplements completed.
- Fresh PA replication completed: 27 graphs and 189 operator instances across three models, three sizes, three graph replicates, two positive-disorder potential streams, and the frozen amplitudes.
- Random 3-regular benchmark completed: 9 graphs and 81 base operator instances. The frozen conditional rule triggered, so all 18 authorized `a=9` streams were added, for 99 total RRG operator instances.
- All 270 PA/base-RRG checkpoints passed numerical validation; worst relative residual was `1.3473816867098723e-13`. The 18 conditional `a=9` checkpoints had worst relative residual `1.9464936278519965e-14`.
- Final scientific aggregate completed under job `14888175`. Visual QA found a two-line orphan PDF page; reporting-only freeze `executed_anderson_source_v3_reporting` and job `14888208` repaired pagination without changing scientific arrays or conclusions. The delivered bundle contains seven figures plus Markdown and the repaired PDF.
- The clean-PA coupling/tunneling result (zero accepted predictive two-mode cases) remains unchanged and outside this execution.

## Intended isolated paths

- Remote run root: `<HOFFMAN_SCRATCH>/random_schrodinger_repair_and_extension_20260924_v1`
- Local review root: `outputs/random_schrodinger_repair_and_extension_20260924_v1`
- Configuration: `experiments/random_schrodinger/anderson_repair_extension_20260924_v1.json`

## Shared read-only dependencies

`common.py`, `geometry.py`, `handoff_metrics.py`, `handoff_coupling.py`, `handoff_reporting.py`, `tunneling.py`, `tunneling_correction.py`, `preferential_attachment/baseline.py`, `regular_baseline.py`, and all historical/clean-PA outputs.

## Job ledger

The complete persistent accounting ledger, including repaired failed attempts and retained production smoke units, is `outputs/random_schrodinger_repair_and_extension_20260924_v1/JOB_LEDGER.csv` locally and `<HOFFMAN_SCRATCH>/random_schrodinger_repair_and_extension_20260924_v1/JOB_LEDGER.csv` on Hoffman.

Executed scientific source freezes: `executed_anderson_source_v1` and repaired `executed_anderson_source_v2`; the v2 scientific source/configuration identity is `41e699639d68bf2d7dc883aa798e9929fb3a5256bfc733a7f008727955a1ea20`. The reporting-only v3 identity is `f900cb012c269738ed7b571772072248d04ec72821fa40ca47908524d4167a1a`.
