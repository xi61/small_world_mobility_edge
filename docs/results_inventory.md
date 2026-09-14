# Results inventory

The repository separates runnable scientific code from historical production
artifacts.  A missing cache does not mean the generator or analysis is absent.

## Tracked snapshots

The existing Git history contains selected, compact snapshots under
`outputs/`:

- random-regular local-density and spacing figures;
- Erdős--Rényi density/local-spacing figures and an extreme-edge summary;
- Chung--Lu power-law density/local-spacing figures and numerical summaries.

These files are illustrative historical outputs.  Their run manifests and
tables, when present, are the authority for their settings; current module
defaults may describe later experiments.

## Runnable without historical data

The following workflows can generate fresh seeded results from source:

- random-regular generation, density, unfolding, spacing, gap ratios, and IPR;
- ER full-graph/LCC density and spacing plus sparse extreme-edge diagnostics;
- Chung--Lu weight generation, density/spacing, localization,
  resonance/projector, and degree/spectral-tail workflows;
- exact-realized-degree double-edge-swap comparisons;
- PA generation and the dense/sparse diagnostic workflows described in the
  PA package README;
- the shared quick non-PA integration runner.

## Historical data not published

Full production spectra, scheduler logs, chunk caches, copied cluster working
directories, large report build trees, and later PA/Chung--Lu production
results are not committed.  Cache-dependent postprocessors require the
corresponding external result directory and should fail clearly when it is not
supplied.  The public repository therefore supports rerunning a simulation but
does not promise byte-for-byte reproduction of every archived production
figure.

New outputs are written under `outputs/` and are ignored by Git.  Publish a new
small result only after recording its configuration, seeds, code commit,
filtering rules, sample counts, and normalization.
