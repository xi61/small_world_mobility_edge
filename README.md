# Small-World Mobility Edge

This repository now keeps only the random-regular baseline code used to study
spectral statistics relevant to the mobility-edge project.

Current focus:

- random `k`-regular graph generation
- adjacency eigenvalues and eigenvectors
- Kesten-McKay spectral density
- inverse participation ratio (IPR)
- adjacent gap ratios
- McKay-law unfolding and GOE spacing comparison

## Kept files

- `src/small_world_mobility_edge/models.py`
- `src/small_world_mobility_edge/regular_baseline.py`
- `src/small_world_mobility_edge/regular_baseline_experiments.py`
- `src/small_world_mobility_edge/__init__.py`
- `pyproject.toml`

## Quick start

Run the baseline experiment:

```bash
cd /Users/xihe/Developer/small_world_mobility_edge
python3 src/small_world_mobility_edge/regular_baseline_experiments.py
```
