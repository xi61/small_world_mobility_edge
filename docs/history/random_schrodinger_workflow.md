# Random-Schrödinger Hoffman workflow

All commands below run from the authoritative Hoffman project:

    <HOFFMAN_SCRATCH>/small_world_mobility_edge_pa_mobility_20260823/project

All new outputs are isolated below:

    <HOFFMAN_SCRATCH>/random_schrodinger_20260921

The shell must remove the unsupported inherited locale variable at submission
time, using `env -u LC_ALL qsub ...`.  Every numerical script loads
`python/3.9.6`, bounds BLAS/OpenMP threads to the requested Grid Engine
slots, and records `JOB_ID` and `SGE_TASK_ID` in its manifest.

## Staged launch commands

Tests:

    env -u LC_ALL qsub tools/hoffman_random_schrodinger_tests.sh

Stage 1 one-case smoke and production:

    env -u LC_ALL qsub -t 3 -v RS_BASELINE_SAMPLES=2,RS_OUTPUT_ROOT=<HOFFMAN_SCRATCH>/random_schrodinger_20260921/smoke/stage1 tools/hoffman_random_schrodinger_stage1.sh
    env -u LC_ALL qsub tools/hoffman_random_schrodinger_stage1.sh

Stage 2 controlled validation and ER pilot:

    env -u LC_ALL qsub tools/hoffman_random_schrodinger_validation.sh
    env -u LC_ALL qsub -v RS_ER_N=1000,RS_ER_SEED=21901001,RS_ER_LABEL=pilot tools/hoffman_random_schrodinger_er.sh

Published-size ER run, released only after the pilot's cost, residual,
orthogonality, component, and full-coverage checks pass:

    env -u LC_ALL qsub -v RS_ER_N=10000,RS_ER_SEED=21901001,RS_ER_LABEL=published_size tools/hoffman_random_schrodinger_er.sh

First ER/PA/Chung--Lu applications:

    env -u LC_ALL qsub tools/hoffman_random_schrodinger_applications.sh

Stage 3 bounded pilot:

    env -u LC_ALL qsub tools/hoffman_random_schrodinger_disorder.sh

Aggregation after the relevant job IDs finish:

    env -u LC_ALL qsub tools/hoffman_random_schrodinger_aggregate.sh

Use Grid Engine `-hold_jid` when submitting dependent stages together.
The applications array must not begin its ER task until the ER pilot is
complete.  The aggregate job should be held on all production arrays.

## Resume and preservation behavior

- Historical output directories are never written.
- Stage 1 smoke and production use different roots.
- Each disorder amplitude has its own manifest and cache identity.  Rerunning
  a sweep skips a completed matching amplitude, refuses a mismatched cache,
  and resumes missing or failed amplitudes.
- ER, graph, selected-eigenvector, potential, and per-amplitude caches include
  graph/operator identities, scaling, vertex mapping, seeds, selection, and
  implementation version.
- Failed array tasks can be resubmitted with `qsub -t task_id`; successful
  disorder amplitudes are not recomputed.
- Aggregation reads manifests and labels missing artifacts `NOT_FOUND`;
  it does not infer that queued work is complete.

## Synchronization

From the local project, run:

    tools/sync_random_schrodinger_results.sh

The script uses filtered `rsync` to transfer manifests, tables, figures, and
logs while leaving large graph/eigenvector caches on Hoffman.  It writes to
`outputs/random_schrodinger_20260921` and never modifies historical results.
