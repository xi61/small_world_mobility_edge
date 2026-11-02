# Random regular and Erdős–Rényi

[All models](../../README.md#browse-by-model) · [Full catalog](../experiment_catalog.md)

The starting question was whether spacing statistics change from the spectral
bulk to the edge. Random regular graphs provide a controlled-degree baseline;
ER introduces random degrees without a prescribed heavy tail.

| Experiment | Operator and selection | Code |
|---|---|---|
| Regular baseline | Simple k-regular adjacency; remove the trivial k eigenvalue | `regular_baseline.py`, `regular_baseline_experiments.py` |
| ER density and aggregate CDF | G(N,p), p=d/(N−1); full graph or LCC as recorded | `erdos_renyi_baseline.py`, `erdos_renyi_spacing_by_energy.py` |
| ER extreme edge | A/√d, sparse or dense edge selection, localization diagnostics | `erdos_renyi_extreme_edge.py` |

These names are relative to `src/small_world_mobility_edge/`. The aggregate CDF
estimates a reference distribution from multiple graphs. For one graph's
ordered retained levels Eᵢ, unfolding uses ξᵢ=M F(Eᵢ), followed by differences
ξᵢ₊₁−ξᵢ. M and F must use the same retained-spectrum convention. Spacings are
never formed between levels belonging to different graphs.

For k-regular adjacency the continuous Kesten–McKay reference is

$$\rho_k(E)=\frac{k\sqrt{4(k-1)-E^2}}{2\pi(k^2-E^2)},\qquad |E|\leq2\sqrt{k-1}.$$

Raw gaps, per-window mean-rescaled gaps, analytic unfolding, and empirical-CDF
unfolding answer different normalization questions. The catalog records which
path is used. The later [single-graph study](../experiments/single_graph_geometry.md)
retains one realization's energy, eigenvector, and graph geometry together;
an ensemble reference CDF can still be used to unfold that graph's gaps.

```bash
python3 -m small_world_mobility_edge.model_smoke --model regular --n 160 --degree 6 --num-graphs 3 --output-dir outputs/smoke
python3 -m small_world_mobility_edge.model_smoke --model erdos-renyi --n 160 --degree 6 --num-graphs 3 --output-dir outputs/smoke
```
