# Small-World Mobility Edge

This project is a coding and literature workspace for the question:

Does a mobility edge appear in the spectrum of small-world and sparse random graph models?

The initial focus is on adjacency-matrix spectra, localization diagnostics, and level statistics for:

- Watts-Strogatz style small-world graphs
- Chung-Lu graphs with prescribed expected degrees
- Configuration models with fixed degree sequences
- Preferential-attachment graphs
- Random regular graphs as a control ensemble

## Research goal

The working hypothesis is that some sparse or heavy-tailed graph ensembles may show a crossover, or a genuine asymptotic transition, between:

- delocalized eigenvectors with GOE-like local level statistics
- localized eigenvectors with Poisson-like local level statistics

The code is organized so you can scan energy windows across the spectrum and track:

- inverse participation ratio (IPR)
- participation ratio (PR)
- adjacent-gap ratio
- rough GOE-vs-Poisson classification by window

## Project layout

- `docs/literature_map.md`: papers already collected for this problem
- `docs/research_notes.md`: problem setup and coding agenda
- `src/small_world_mobility_edge/models.py`: graph ensemble generators
- `src/small_world_mobility_edge/spectral.py`: eigenvalue and localization diagnostics
- `experiments/ws_mobility_scan.py`: first experiment script for a Watts-Strogatz scan
- `tests/`: basic structural tests

## Quick start

Run the first experiment:

```bash
cd /Users/xihe/Developer/small_world_mobility_edge
PYTHONPATH=src python3 experiments/ws_mobility_scan.py
```

Run tests:

```bash
cd /Users/xihe/Developer/small_world_mobility_edge
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Current environment assumptions

This scaffold only assumes `numpy` and `scipy`, which are already available here. It does not require `networkx`.

## Immediate next coding targets

1. Add finite-size scaling experiments for windowed gap statistics.
2. Compare Watts-Strogatz against random regular controls at matched mean degree.
3. Add heavy-tail degree samplers for Chung-Lu and configuration models.
4. Add batch experiment output to CSV for later plotting.
