# Preferential-Attachment Spectral Pipeline Handoff

## Superlinear implementation and fine edge-resolution follow-up

Date: 2026-08-24

Project workspace: `<SOURCE_WORKSPACE>`
Canonical migrated project: `<PUBLICATION_CHECKOUT>`

## 1. Purpose of this handoff

This document gives the next ChatGPT session enough context to design the next preferential-attachment (PA) experiment without reconstructing the project from scratch. It records:

- which PA families and controls have actually been run;
- the mathematical meaning of the spectral and localization statistics;
- exact code locations and short code excerpts implementing the pipeline;
- the results already obtained and the limits on their interpretation;
- the complete exact-degree rewiring control;
- why simply choosing ever-narrower quantile bins does not always improve edge inference; and
- a concrete specification for a superlinear/fine-edge follow-up.

The authoritative consolidated execution report is:

`<PUBLICATION_CHECKOUT>/docs/preferential_attachment_mobility_edge_execution_report.md`

The authoritative production version is `pa_mobility_edge_gap_first_v2_atom_segmented`. Earlier v1 mobility-edge outputs that allowed a spacing ratio to bridge a repeated-eigenvalue atom must not be used for conclusions.

## 2. Executive scientific result

The current experiments support **strong preferential-attachment edge localization with an otherwise GOE-like positive-gap bulk**, but they do not yet establish a sharp Anderson-style mobility edge.

The robust pattern is:

1. Eigenvectors near both nontrivial spectral edges have much larger normalized IPR than bulk eigenvectors.
2. In clean, atom-free regimes, the adjacent-gap ratio remains close to the GOE value through most of the spectrum, including much of the edge region.
3. Stronger condensation or lower density creates repeated-eigenvalue atoms and zero gaps. Those atoms can make inclusive level statistics look more Poisson-like even when the positive-conditioned gaps remain GOE-like.
4. Exact-degree rewiring preserves most of the edge-localization pattern, showing that the degree sequence and hub concentration are the main drivers. Rewiring nevertheless contracts some edge-candidate intervals and removes a central IPR-only anomaly, so PA-specific organization contributes secondary structure.
5. Fine quantile screens identify candidate edge bands. A mobility-edge claim requires a sustained, atom-free change in both localization and level statistics, with adequate independent graph support and finite-size checks. That standard has not yet been met.

## 3. Experiment inventory: what was actually run

### 3.1 Dense Barabási–Albert smoke test

- Model: standard BA, attachment probability proportional to degree.
- Parameters: `N=400`, `m=4`, `num_graphs=3`.
- Spectrum: full dense eigendecomposition.
- Controls: exact-degree rewiring.
- Purpose: numerical correctness, eigenpair checks, normalization checks, and end-to-end file generation.
- Scientific status: validation only; not large enough for tail or ensemble inference.

### 3.2 Standard BA sparse tail pilot

- Model: standard BA.
- Parameters: `N=5000`, `m=10`, `num_graphs=5`.
- Spectrum: 180 sparse eigenpairs from each edge; Perron mode removed from the positive edge.
- Analyses: degree and spectral tails, rank relation, edge spacings, localization, exact-degree rewiring, and an unpaired matched Chung–Lu comparison.
- Scientific status: useful pilot evidence, but `G=5` remains a small number of independent graph realizations.

### 3.3 Legacy affine full-spectrum family

- Model: affine-linear PA.
- Parameters: `N=5000`, `m=10`, `G=500` for each `alpha in {1.80, 1.50, 1.05}`.
- Spectrum: full dense spectrum.
- Scientific status: exploratory predecessor. Use the corrected family below for authoritative conclusions.

### 3.4 Corrected affine-linear production family

- Model: affine-linear PA.
- Parameters: `alpha in {2.00, 1.80, 1.50, 1.25, 1.10, 1.05}`, `N=5000`, `m=10`, `G=500` per alpha.
- Method: gap-first atom segmentation, graph-equal aggregation, and five-fold cross-fitting.
- Coarse and fine quantile scans: includes `dq=0.002`, `0.001`, `0.0005`, and `0.0002` products.
- Selected clean family member: `alpha=1.25`.

### 3.5 Nonlinear superlinear production family

- Model: nonlinear PA with attachment weight `w_i = d_i^zeta`.
- Parameters: `zeta in {1.05, 1.10, 1.20, 1.30, 1.50}`, `N=5000`, `m=10`, `G=500` per zeta.
- Method: the same production spectral and postprocessing pipeline as the corrected affine family.
- Selected clean family member: `zeta=1.30`.
- `zeta=1.50` is strongly condensed and contains a non-negligible zero-eigenvalue atom, so it is not the cleanest member for spacing-universality inference.

Important: the superlinear generator is already implemented. The next phase should extend and audit it, not implement it from scratch.

### 3.6 Exact-degree rewiring controls

- Affine: `alpha=1.25`, 500 paired original/rewired graphs.
- Nonlinear: `zeta=1.30`, 500 paired original/rewired graphs.
- Pairing: identical graph index and original graph seed, followed by a separate deterministic rewiring seed.
- Invariant: the exact labeled degree vector was checked after every production rewiring.

### 3.7 Density sensitivity

- Affine selected point: `alpha=1.25`, `m in {2,5,10,20}`.
- Nonlinear selected point: `zeta=1.30`, `m in {2,5,10,20}`.
- Purpose: distinguish localization/atom effects caused by sparse density from effects associated with PA nonlinearity.

### 3.8 Implemented but not part of the saved evidence

Holme–Kim graph generation exists in the codebase, but no executed Holme–Kim output was found in the authoritative result tree. It must not be described as an already completed experiment.

## 4. Mathematical model definitions

### 4.1 Standard BA attachment

At time `t`, a new vertex attaches to an existing vertex `i` with probability

\[
P(i\mid G_t)=\frac{d_i(t)}{\sum_j d_j(t)}.
\]

The generator call is in:

`<PUBLICATION_CHECKOUT>/src/small_world_mobility_edge/preferential_attachment/baseline.py`, lines 226–232.

```python
graph = nx.barabasi_albert_graph(
    n=config.n,
    m=config.m,
    seed=graph_seed,
)
```

### 4.2 Affine-linear PA

Affine attachment uses

\[
w_i=d_i+\delta,
\qquad
P(i\mid G_t)=\frac{d_i+\delta}{\sum_j(d_j+\delta)}.
\]

The project parameterizes it through the desired degree-tail exponent

\[
\alpha=2+\frac{\delta}{m},
\qquad
\delta=m(\alpha-2).
\]

The positivity condition for the minimum eligible degree `m` is

\[
m+\delta=m(\alpha-1)>0,
\]

which requires `alpha > 1`.

Source: `baseline.py`, lines 97–108.

```python
def affine_offset_from_alpha(*, alpha: float, m: int) -> float:
    if alpha <= 1.0:
        raise ValueError("alpha must exceed 1")
    return float(m) * (float(alpha) - 2.0)
```

The weighted sampling step is in `baseline.py`, lines 140–151:

```python
weights = degrees[:new_node].astype(np.float64) + delta
probabilities = weights / weights.sum()
targets = rng.choice(
    new_node,
    size=m,
    replace=False,
    p=probabilities,
)
```

Lower `alpha` means more negative `delta`, stronger relative preference for already-large degrees, and more hub condensation.

### 4.3 Nonlinear superlinear PA

The nonlinear model uses

\[
w_i=d_i^{\zeta},
\qquad
P(i\mid G_t)=\frac{d_i^\zeta}{\sum_j d_j^\zeta}.
\]

- `zeta=1`: ordinary linear PA.
- `zeta>1`: superlinear PA; large-degree vertices receive more than a proportional advantage.
- Increasing `zeta` can cause condensation, where a small number of vertices capture a macroscopic fraction of edges.

Source: `baseline.py`, lines 155–191; the stable weight computation is at lines 176–185.

```python
log_weights = zeta * np.log(degrees[:new_node].astype(np.float64))
log_weights -= float(np.max(log_weights))
weights = np.exp(log_weights)
probabilities = weights / weights.sum()
targets = rng.choice(
    new_node,
    size=m,
    replace=False,
    p=probabilities,
)
```

The logarithmic implementation avoids numerical overflow when `d_i^zeta` becomes large. Unlike the affine family, a strongly superlinear model should not automatically be assigned a stationary degree power-law exponent.

## 5. Spectral and localization mathematics

### 5.1 Adjacency eigenproblem

For an undirected graph with symmetric adjacency matrix `A`, solve

\[
A v_k=\lambda_k v_k,
\qquad
\lVert v_k\rVert_2=1.
\]

The dense production decomposition is in `full_spectrum_scan.py`, lines 173–181:

```python
adjacency = nx.to_numpy_array(graph, dtype=np.float64)
eigenvalues, eigenvectors = np.linalg.eigh(adjacency)
```

The largest positive adjacency eigenvalue is the Perron mode. It is structurally special and is removed before ordinary positive-edge statistics. The production removal/check is in `full_spectrum_scan.py`, lines 1220–1229.

### 5.2 IPR, normalized IPR, and effective support

For a normalized eigenvector `v`,

\[
\operatorname{IPR}(v)=\sum_{i=1}^N |v_i|^4.
\]

The reported normalized value is

\[
N\operatorname{IPR}=N\sum_i |v_i|^4.
\]

Why its scale diagnoses localization:

- Perfectly flat vector: `|v_i|=1/sqrt(N)`, hence `IPR=1/N` and `NIPR=1`.
- Real random/Haar-delocalized vector: `E[IPR]=3/(N+2)`, hence `E[NIPR]` approaches 3.
- Equal mass on only `k` vertices: `|v_i|=1/sqrt(k)` on the support, hence `IPR=1/k` and `NIPR=N/k`.

Therefore high `NIPR` means localized, while a value near the real-random baseline of 3 means delocalized. A rough effective support estimate is

\[
k_{\mathrm{eff}}\approx\frac{N}{N\operatorname{IPR}}.
\]

For the BA pilot value `NIPR=134.46` at `N=5000`, this gives about `5000/134.46 = 37` effectively occupied vertices. This is a heuristic for a representative vector, not an exact support count for an ensemble mean.

Production code computes vertex masses and IPR in `full_spectrum_scan.py`, lines 1140–1152:

```python
masses = np.square(eigenvectors)
ipr = np.sum(np.square(masses), axis=0)
participation_ratio = 1.0 / ipr
normalized_ipr = float(n) * ipr
```

The complementary quantity `PR/N` satisfies

\[
\frac{\operatorname{PR}}{N}=rac{1}{N\operatorname{IPR}}.
\]

### 5.3 Local mass concentration

For sorted masses `p_(1) >= p_(2) >= ...`, where `p_i=|v_i|^2`, define

\[
C_K=\sum_{j=1}^{K}p_{(j)}.
\]

`C10`, `C50`, and `C200` measure how much eigenvector mass is captured by the largest 10, 50, or 200 coordinates. The project also records the number of coordinates needed to capture 90% of the mass. These metrics verify that a high IPR is a real concentration effect rather than a plotting artifact.

### 5.4 Raw adjacent-gap ratio

Within one graph and one atom-free spectral segment, sort eigenvalues and form positive gaps

\[
\Delta_i=\lambda_{i+1}-\lambda_i.
\]

The raw adjacent-gap ratio is

\[
r_i=\frac{\min(\Delta_i,\Delta_{i+1})}
{\max(\Delta_i,\Delta_{i+1})}.
\]

It is not the mean gap. It is a dimensionless ratio of two neighboring gaps and does not require spectral unfolding. Reference means are

\[
\langle r\rangle_{\mathrm{GOE}}\approx0.53590,
\qquad
\langle r\rangle_{\mathrm{Poisson}}\approx0.38629.
\]

The constants are defined in `mobility_edge_postprocess.py`, lines 27–34. The gap-first construction, atom segmentation, and ratio construction are in lines 109–190. In particular, the atom bridge is explicitly omitted at lines 140–146, and ratios are made only inside valid segments at lines 157–177.

Interpretation:

- `r` near 0.536: GOE-like level repulsion and correlated gaps.
- `r` near 0.386: Poisson-like uncorrelated gaps.
- A pile of exactly zero gaps is neither clean GOE nor clean Poisson evidence. It signals an eigenvalue atom or degeneracy and must be reported separately.

The production summary first averages within each graph/bin and then gives every graph equal weight. It does not join the last eigenvalue from one graph to the first eigenvalue of another.

### 5.5 Cross-fitted unfolded spacings

Unfolded spacings require an estimated local density. To avoid fitting and testing on the same observations, the production code uses five folds: fit the unfolding map on four folds and evaluate on the held-out fold. The no-leakage logic is in `mobility_edge_postprocess.py`, lines 309–359.

Raw `r` is the primary local universality diagnostic because it is unfolding-free. Cross-fitted spacing CDFs/histograms and their GOE/Poisson KS distances are secondary confirmation.

## 6. What “too small” meant in the smoke and pilot tests

“Too small” referred to statistical sample support, not to eigenvalue magnitudes.

For the dense smoke test:

- only `G=3` independent graph realizations were generated;
- each rank window held five gaps per graph;
- pooling therefore produced only 15 gaps per window and only 12 adjacent-gap ratios.

Those counts can validate code, but they cannot support stable tail exponents, uncertainty estimates, or fine spectral-edge conclusions.

The rough number of states/gaps in a quantile bin of width `dq` is

\[
n_{\mathrm{pooled}}\approx G M\,dq,
\]

where `M` is the usable number of eigenvalues per graph. Crucially, the number of independent graph realizations is still `G`, not `G M dq`.

For `N=400`, `G=3`:

- `dq=0.002` gives only about `3*400*0.002=2.4` pooled states before boundary and atom losses;
- `dq=0.001` gives only about 1.2;
- using `[0.90,1.00]` as one broad tail gives roughly 120 states, but they still come from only three independent graphs and do not resolve fine structure inside the tail.

Therefore scanning from `q=0.90` to `q=1` with increasingly narrow bins does **not** fix the smoke-test limitation. Narrower bins reduce observations. The remedy is a combination of larger `G`, adequate `N`, adaptive minimum-count bins, and fixed-rank edge windows.

At `N=5000`, `G=500`, the approximate pooled counts are much better:

- `dq=0.001`: about 2,500 states;
- `dq=0.0005`: about 1,250 states;
- `dq=0.0002`: about 500 states.

But a single graph has quantile resolution only about

\[
1/M\approx1/5000=0.0002.
\]

Below that width, many graphs contribute zero or one eigenvalue, so an adjacent-gap ratio cannot be formed within the bin. Finer-than-`0.0002` edge work at `N=5000` should therefore use fixed spectral ranks or larger `N`, not merely smaller quantile widths.

## 7. Why the BA pilot used five graphs while later families used 500

The phrase “five graphs” means five independently generated graph realizations (`num_graphs=5`). Statistics were aggregated across those five realizations, using graph-aware grouping where applicable.

This was not intended to match the final affine/nonlinear production scale. The standard BA test was an early sparse-solver pilot designed to verify tail extraction and compare original, rewired, and Chung–Lu spectra at manageable cost. The later production question required stable quantile-resolved full-spectrum estimates, so it used `G=500` per parameter point.

Consequences:

- `G=5` can reveal a strong recurring pattern and help debug the pipeline.
- It is weak for graph-to-graph uncertainty, rare-tail estimation, and fine spectral-bin inference.
- `G=500` supports graph-equal means, cross-fitting, contributor counts, paired rewiring differences, and much narrower bins.
- The five-graph Chung–Lu comparison is unpaired and remains a pilot, not evidence at the same strength as the 500-pair exact-degree rewiring controls.

## 8. How to read Figure E

Authoritative affine figure:

`<PUBLICATION_CHECKOUT>/outputs/preferential_attachment/mobility_edge/affine_linear/n_5000_m_10/controlled_family_summary/figure_E_localization_level_statistics_vs_quantile.png`

Authoritative nonlinear figure:

`<PUBLICATION_CHECKOUT>/outputs/preferential_attachment/mobility_edge/nonlinear/n_5000_m_10/controlled_family_summary/figure_E_localization_level_statistics_vs_quantile.png`

### Top panel: normalized IPR versus spectral quantile

- `q≈0`: negative spectral edge.
- `q≈1`: positive non-Perron spectral edge.
- Large `NIPR` at either edge means eigenvectors are concentrated on far fewer than `N` vertices.
- A high-edged “bowl” therefore means edge localization with a more delocalized middle.
- Interior narrow peaks can represent motif bands, hub-related structure, or atom-adjacent bands and should not automatically be called mobility edges.
- The logarithmic vertical scale can make changes in peak width or height visually unintuitive.

### Bottom panel: graph-mean raw `r`

The bottom panel is **not mean gap size**. It is the graph-equal mean of the raw adjacent-gap ratios defined above. The dashed reference at about 0.5359 is GOE; the dotted reference at about 0.3863 is Poisson. “Positive-conditioned” means ratios involving zero gaps were excluded and zero-gap fractions were reported separately.

Thus a curve near 0.53 says the positive spacings retain GOE-like repulsion even where localization strengthens. A drop caused only by an eigenvalue atom is not clean evidence of Poisson localized statistics.

## 9. Results by experiment family

### 9.1 Standard BA sparse tail pilot

#### Abstract

The BA pilot found the expected hub/eigenvalue rank relation and strong edge localization. Its spectral-gap ratios stayed GOE-like. The result is numerically and qualitatively useful, but the five-graph scale is insufficient for a final finite-size or tail-law claim.

#### Detail

At `N=5000`, `m=10`, `G=5`:

| Object | Edge | Rank slope | R² | Median `|lambda|/sqrt(d)` |
|---|---:|---:|---:|---:|
| BA original | positive | 0.4401 | 0.9958 | 1.056 |
| BA original | negative | 0.4657 | 0.9972 | 1.065 |
| degree-rewired | positive | 0.4527 | — | — |
| degree-rewired | negative | 0.4724 | — | — |

The slope near one-half supports the hub approximation

\[
|\lambda_j|\propto \sqrt{d_j}.
\]

At the `q0=0.976` tail threshold, the estimated CCDF exponents were:

- degree: 2.0357;
- `|lambda|`: 4.6172;
- `lambda²`: 2.3086.

From `|lambda|≈sqrt(d)`, a degree CCDF exponent `beta` predicts an absolute-eigenvalue exponent of roughly `2 beta` and a `lambda²` exponent of roughly `beta`. The pilot predictions were 4.071 and 2.036. The deviations are plausible finite-tail and finite-ensemble effects and should not be overinterpreted.

At the closest non-Perron positive edge:

\[
N\operatorname{IPR}=134.455,
\qquad
r=0.52581.
\]

The first number means very strong edge localization: the rough effective support is only about 37 of 5,000 vertices. The second remains much closer to GOE than Poisson.

The matched Chung–Lu pilot at the same threshold gave degree, `|lambda|`, and `lambda²` exponents 2.12295, 4.88207, and 2.44103. Because it used only five unpaired graphs, it should be described as a consistency check, not a definitive control.

#### Summary

BA already displays localized edge eigenvectors without a corresponding clean Poisson gap regime. The hub-rank mechanism is visible. More graph realizations and controlled finite-size scaling are needed for tail exponents and a mobility-edge statement.

### 9.2 Corrected affine-linear family

#### Abstract

Stronger affine preference creates larger hubs, larger edge NIPR, and—at the most condensed points—zero-eigenvalue atoms. Across the clean family, the bulk and positive-conditioned edge gap ratios remain GOE-like.

#### Detail

| `alpha` | mean `dmax` | zero-gap fraction | max mean `NIPR` | bulk `r` | edge `r` |
|---:|---:|---:|---:|---:|---:|
| 2.00 | 374.9 | 0 | 279.5 | 0.5308 | 0.5307 |
| 1.80 | 459.0 | 0 | 342.2 | 0.5307 | 0.5302 |
| 1.50 | 688.6 | 0 | 351.2 | 0.5307 | 0.5297 |
| 1.25 | 1131.5 | 0 | 372.0 | 0.5306 | 0.5295 |
| 1.10 | 1756.7 | 0.1491 | 586.5 | 0.5309 | 0.5277 |
| 1.05 | 2175.9 | 0.3711 | 709.0 | 0.5305 | 0.5252 |

The selected clean point is `alpha=1.25`: it has strong hub and edge-localization effects without a zero atom. Its fine positive-spacing comparison reached a minimum GOE KS distance of about 0.0079, versus about 0.1551 to Poisson.

The production screen marked the following intervals as candidates:

- negative edge: `[0,0.058]`;
- central IPR-only band: `[0.490,0.512]`;
- positive edge: `[0.942,1]`.

These are screening intervals, not automatically mobility edges. Candidate construction is in `mobility_edge_postprocess.py`, lines 709–830. A label requires joint, sustained localization and spacing evidence; IPR alone is insufficient.

#### Summary

The affine family shows a controlled transition from clean edge localization to highly condensed, atom-contaminated spectra. `alpha=1.25` is the best current clean point for edge-resolution and rewiring work.

### 9.3 Nonlinear superlinear family

#### Abstract

Increasing `zeta` produces progressively larger hubs and stronger peak localization. `zeta=1.30` is the clean selected superlinear point; `zeta=1.50` is more condensed but begins to contain a zero-eigenvalue atom. The numerical trend—not visual “pointiness”—is the evidence for stronger localization.

#### Detail

| `zeta` | mean `dmax` | `dmax/N` | top-1% degree share | zero-gap fraction | max mean `NIPR` | bulk `r` | edge `r` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.05 | 448.1 | 0.0896 | 0.1069 | 0 | 342.6 | 0.5307 | 0.5303 |
| 1.10 | 550.0 | 0.1100 | 0.1213 | 0 | 416.8 | 0.5306 | 0.5308 |
| 1.20 | 895.5 | 0.1791 | 0.1624 | 0 | 433.4 | 0.5304 | 0.5294 |
| 1.30 | 1575.5 | 0.3151 | 0.2262 | 0 | 495.3 | 0.5311 | 0.5293 |
| 1.50 | 3353.7 | 0.6707 | 0.3768 | 0.0590 | 686.8 | 0.5307 | 0.5277 |

The curve looking smoother does not mean localization is weaker. Smoothing is affected by aggregation, bin counts, line interpolation, and the logarithmic vertical scale. The relevant comparisons are numeric:

- within the nonlinear family, `dmax`, `dmax/N`, top-1% degree share, and peak `NIPR` all rise substantially as `zeta` rises;
- the clean selected nonlinear point `zeta=1.30` has mean `dmax=1575.5` and peak `NIPR=495.3`;
- the clean selected affine point `alpha=1.25` has mean `dmax=1131.5` and peak `NIPR=372.0`.

Therefore `zeta=1.30` is more localized than `alpha=1.25` by those selected-point metrics. It is **not** correct to claim that every superlinear case exceeds every affine case: the extreme affine point `alpha=1.05` has peak `NIPR≈709`, slightly above `zeta=1.50` at about 687, and the two parameterizations are not degree-matched.

The width and height of the high-`NIPR` edge region, together with concentration metrics and atom status, matter more than how pointed the plotted curve looks.

#### Summary

Superlinearity strengthens hub condensation and edge localization across its own parameter grid. `zeta=1.30` is the appropriate atom-free point for the next fine-edge study; `zeta=1.50` should be retained as a condensation/atom stress test rather than treated as the clean primary case.

### 9.4 Exact-degree rewiring controls

#### Abstract

Degree-preserving rewiring keeps the main edge-localization pattern but changes finer spectral structure. This indicates that the degree sequence is the primary cause, while PA-specific adjacency organization modifies candidate widths, central anomalies, and the exact spectral density.

#### Detail: complete rewiring pipeline

For each original graph:

1. Generate the original graph using the saved family, parameter, graph index, and deterministic graph seed.
2. Copy the graph.
3. Apply NetworkX double-edge swaps with a separate deterministic rewiring seed.
4. Request approximately `10*E` successful swaps, with a maximum-attempt multiplier of 20.
5. Recompute the spectrum and every spectral statistic using the same Perron rule, atom tolerances, bins, folds, and localization definitions.
6. Verify exact equality of the labeled degree array and check connectedness.
7. Compare original and rewired observations at the same graph index, seed, and spectral bin.

A double-edge swap chooses two edges such as `(a,b)` and `(c,d)` and replaces them with a valid crossing such as `(a,d)` and `(c,b)`, rejecting self-loops and duplicate edges. Every participating vertex loses one incident edge and gains one, so its degree is unchanged.

The production checks are in `full_spectrum_scan.py`, lines 1202–1217. The paired-difference code is in `mobility_edge_rewiring_comparison.py`, lines 46–124; lines 96–101 implement:

```python
paired["delta_nipr"] = (
    paired["mean_nipr_rewired"] - paired["mean_nipr_original"]
)
paired["delta_raw_r"] = (
    paired["mean_raw_r_rewired"] - paired["mean_raw_r_original"]
)
```

That is always

\[
\Delta=\text{rewired}-\text{original}
\]

for the same pair and bin.

Preserved invariants:

- node count and node labels;
- edge count;
- exact labeled degree vector, including hub identities;
- mean and maximum degree, degree CCDF, degree Gini, and top-degree shares;
- a simple undirected graph with no self-loops or parallel edges;
- original graph index/seed pairing;
- spectral analysis, quantile bins, Perron removal, atom rules, folds, and thresholds.

Connectedness is checked but is not a mathematical invariant of arbitrary double-edge swaps. All saved production pairs passed the connectedness and exact-degree checks.

Randomized or weakened features:

- which vertices are adjacent;
- PA age/growth-history correlations encoded in the edge arrangement;
- assortativity and degree-degree mixing;
- clustering, triangles, short cycles, and other motifs;
- community/local organization;
- hub-neighbor overlap and mesoscopic structure.

Thus rewiring asks: if exact degrees and exact hub identities are held fixed, which spectral effects survive removal of the original PA wiring pattern? A surviving effect is degree-led; a disappearing effect depends on organization beyond the degree sequence. Finite double-edge swapping approximates a conditional random-graph control and is not a proof of perfectly uniform sampling over all simple graphs with that degree sequence.

Affine `alpha=1.25`, 500 paired graphs:

- exact labeled-degree checks: 500/500 passed;
- connectedness checks: 500/500 passed;
- mean original/rewired edge overlap: 0.0720;
- full-spectrum two-sample ESD KS distance: 0.03829;
- raw-energy edge shifts: `Delta E_minus=-0.391`, `Delta E_plus=-0.680`;
- central-bulk `Delta(NIPR)=-0.672`;
- central-bulk `Delta r=+0.000244`;
- central-bulk `Delta C50=-0.01770`;
- negative candidate contracted from `[0,0.058]` to `[0,0.046]`;
- positive candidate contracted from `[0.942,1]` to `[0.954,1]`;
- the central IPR-only band disappeared after rewiring.

The value 0.03829 is the maximum separation between the **original and rewired empirical spectral-distribution CDFs**. It is not a distance to Poisson. It says rewiring changes the global spectral distribution by a modest but measurable amount, about 3.8% at its largest CDF separation.

GOE versus Poisson is diagnosed separately by raw `r` and cross-fitted positive spacings. Here the bulk `Delta r=0.000244` is tiny relative to the GOE–Poisson mean separation

\[
0.53590-0.38629=0.14961.
\]

The gap regime therefore remains GOE-like. The energy shifts mean the rewired extreme eigenvalues moved to lower raw energies; they do not imply a Poisson transition. `Delta(NIPR)=-0.672` and `Delta C50=-0.01770` indicate somewhat weaker bulk concentration after rewiring, while contraction of the candidate bands shows a real organizational contribution at the edges.

Nonlinear `zeta=1.30`, 500 paired graphs:

- full-spectrum ESD KS distance: 0.02348;
- raw-energy shifts: `Delta E_minus=-0.399`, `Delta E_plus=-0.817`;
- central-bulk `Delta(NIPR)=-0.331`;
- central-bulk `Delta r=-0.000361`;
- central-bulk `Delta C50=-0.00950`.

#### Summary

“Not much difference” is accurate for the bulk universality statistic `r`, but it is too broad as a description of the entire control. Rewiring produces modest global spectral-density changes and meaningful candidate-band/localization changes. The main edge localization survives, pointing to degree/hub concentration as the dominant mechanism.

### 9.5 Density sensitivity

#### Abstract

Low `m` generates rank deficiency, repeated eigenvalues, and much stronger localization in both selected families. Increasing `m` suppresses atoms and peak NIPR while leaving the clean positive-conditioned bulk near GOE.

#### Detail

Affine `alpha=1.25`:

| `m` | zero-gap fraction | max mean `NIPR` | bulk `r` |
|---:|---:|---:|---:|
| 2 | 0.4761 | 951.1 | 0.5198 |
| 5 | 0.1409 | 571.0 | 0.5306 |
| 10 | 0 | 372.0 | 0.5306 |
| 20 | 0 | 179.8 | 0.5306 |

Nonlinear `zeta=1.30`:

| `m` | zero-gap fraction | max mean `NIPR` | bulk `r` |
|---:|---:|---:|---:|
| 2 | 0.4509 | 984.6 | 0.5155 |
| 5 | 0.0527 | 705.6 | 0.5311 |
| 10 | 0 | 495.3 | 0.5311 |
| 20 | 0 | 287.6 | 0.5308 |

At affine `m=2`, the `+1` and `-1` eigenvalues each had approximate multiplicity 105, illustrating why zero/inclusive-gap effects can contaminate a naive Poisson comparison.

The m-sensitivity comparison is not degree-sequence matched: changing `m` changes mean degree to approximately `2m`. That limitation is stated in `mobility_edge_m_sensitivity_summary.py`, lines 43–50.

#### Summary

Density is a major control variable. Atom-free `m=10` should remain the primary comparison, while `m=2` and `m=5` serve as sparse/degeneracy stress tests. An apparent reduction of inclusive `r` at low `m` must not be called Poisson localization without separating repeated eigenvalues.

## 10. Code and output map for the next session

### Core source files

- Graph generators: `<PUBLICATION_CHECKOUT>/src/small_world_mobility_edge/preferential_attachment/baseline.py`
- Full-spectrum production scan: `<PUBLICATION_CHECKOUT>/src/small_world_mobility_edge/preferential_attachment/full_spectrum_scan.py`
- Atom-segmented postprocessing: `<PUBLICATION_CHECKOUT>/src/small_world_mobility_edge/preferential_attachment/mobility_edge_postprocess.py`
- Paired rewiring comparison: `<PUBLICATION_CHECKOUT>/src/small_world_mobility_edge/preferential_attachment/mobility_edge_rewiring_comparison.py`
- Density sensitivity summary: `<PUBLICATION_CHECKOUT>/src/small_world_mobility_edge/preferential_attachment/mobility_edge_m_sensitivity_summary.py`

### Authoritative output roots

- Standard BA pilot: `<PUBLICATION_CHECKOUT>/outputs/preferential_attachment/barabasi_albert/n_5000_m_10`
- Affine production family: `<PUBLICATION_CHECKOUT>/outputs/preferential_attachment/mobility_edge/affine_linear/n_5000_m_10`
- Nonlinear production family: `<PUBLICATION_CHECKOUT>/outputs/preferential_attachment/mobility_edge/nonlinear/n_5000_m_10`
- Affine selected rewiring: `<PUBLICATION_CHECKOUT>/outputs/preferential_attachment/mobility_edge/affine_linear/n_5000_m_10/alpha_1p25`
- Nonlinear selected rewiring: `<PUBLICATION_CHECKOUT>/outputs/preferential_attachment/mobility_edge/nonlinear/n_5000_m_10/zeta_1p30`

Before any implementation, inspect manifests, `family_summary.csv`, fine-bin summaries, candidate interval tables, contributor counts, and plot-generation metadata under those roots. Do not silently mix files from superseded v1 runs.

## 11. Required next phase: superlinear and finer edge examination

### 11.1 First principle

Do not create a second nonlinear generator unless the current one fails an audit. `generate_nonlinear_preferential_attachment_graph` and the production configuration already support `d^zeta` attachment. The next task is to inspect, extend, and validate the existing production path.

### 11.2 Audit existing fine outputs before launching new computation

For at least `zeta=1.30`, and secondarily `zeta=1.50`, inspect all already-saved edge products for:

- `dq=0.001`;
- `dq=0.0005`;
- `dq=0.0002`;
- negative edge `q in [0,0.05]`;
- positive non-Perron edge `q in [0.95,1]`.

For every bin, report total state/gap/ratio counts and the number of contributing graphs. Do not report a stable-looking line if most graphs cannot contribute a ratio.

### 11.3 Add fixed-rank edge windows

At `N=5000`, quantiles finer than about `0.0002` exceed the per-graph spectral resolution. Add graph-local fixed-rank windows on both edges:

- gaps/ratios 1–5;
- 6–10;
- 11–20;
- 21–40;
- optionally 41–80 if counts justify it.

On the positive side, rank after removing the Perron eigenpair. Never construct gaps across graphs or across an atom boundary.

Report the raw energy range and quantile range covered by each rank window so the fixed-rank and quantile views can be compared.

### 11.4 Recommended adaptive-bin supplement

Alongside fixed `dq=0.0002`, construct outward-to-inward adaptive edge windows that require, at minimum:

- 100 positive gaps or 100 raw-r observations after atom removal; and
- a stated minimum number of contributing graphs.

The exact graph-contributor threshold should be selected after auditing the existing counts. It must be reported, not hidden.

### 11.5 Recommended superlinear grid

Reuse saved points where possible and consider the extended grid

\[
\zeta\in\{1.00,1.05,1.10,1.20,1.30,1.35,1.40,1.50\}.
\]

- `zeta=1.00` provides the direct linear baseline under the same custom generator.
- `zeta=1.30` is the primary clean superlinear point.
- `zeta=1.35` and `1.40` fill the gap before the atom-contaminated `1.50` point.
- `zeta=1.50` is a condensation/atom stress test.

For new directly comparable runs, keep `N=5000`, `m=10`, `G=500`, deterministic seed construction, fold assignment, atom tolerances, and postprocessing version fixed. If resources permit, add an `N`-scaling subset rather than making bins narrower than the spectral resolution.

### 11.6 Required metrics

For each family point and edge window, retain:

- `NIPR` and `PR/N`;
- `C10`, `C50`, `C200`, and 90%-mass support fraction;
- positive-conditioned raw `r`;
- inclusive raw `r` plus zero-gap/atom fraction;
- five-fold cross-fitted spacing KS distance to GOE and Poisson;
- `dmax`, `dmax/N`, degree Gini, and top-1% degree share;
- atom locations and multiplicities;
- raw eigenvalue bounds;
- number of states, gaps, ratios, and contributing graphs.

### 11.7 Validation gates

The implementation must verify:

1. no gaps or ratios cross graph boundaries;
2. no ratios bridge a repeated-eigenvalue atom;
3. Perron removal is applied consistently before positive-edge ranks;
4. no observation appears in both the training and held-out unfolding fold;
5. graph-equal estimates are distinguished from pooled-observation estimates;
6. fine-bin contributor counts and uncertainties are saved;
7. resume logic is content-aware and validates configuration/version metadata;
8. new runs cannot overwrite authoritative existing outputs accidentally;
9. exact model, seed, `N`, `m`, `G`, zeta, bin definitions, atom tolerance, and git/code version appear in the manifest.

### 11.8 Scientific conclusion gate

Call a region a mobility-edge candidate only if it shows a sustained change over neighboring quantile or rank windows in both:

- localization observables (`NIPR`, `PR/N`, mass concentration); and
- atom-free level statistics (`r` and cross-fitted spacing comparisons).

The change must also have adequate graph support and remain qualitatively stable under reasonable changes of bin width/rank window and, ideally, system size. If only IPR rises while positive-conditioned `r` stays GOE-like, describe the result as **edge localization or prelocalization**, not a demonstrated GOE-to-Poisson mobility edge.

## 12. Ready-to-use prompt for the next ChatGPT session

> Read this handoff completely, then inspect the canonical source and output paths it identifies. Do not implement a new nonlinear PA generator until you verify the existing `d^zeta` generator and production configuration. Audit the saved `zeta=1.30` and `zeta=1.50` fine-edge outputs at `dq=0.001`, `0.0005`, and `0.0002`, including per-bin graph-contributor counts, atom handling, Perron removal, and cross-fit metadata. Then propose a concrete implementation plan for fixed-rank edge windows (1–5, 6–10, 11–20, 21–40 gaps/ratios on each edge), adaptive minimum-count windows, and an extended superlinear grid. Quote the relevant current code lines and identify every file to change. State estimated compute, storage, and validation cost. Do not launch remote/HPC jobs or modify the authoritative code/results until I approve the plan. Preserve the v2 atom-segmented gap-first rules, graph-equal aggregation, deterministic pairing, and all metadata checks. The scientific target is a joint, atom-free localization and level-statistics transition; if raw `r` stays GOE-like while NIPR rises, report edge localization/prelocalization rather than claiming a mobility edge.

## 13. Bottom-line interpretation

The completed work already includes a production nonlinear superlinear PA family and fine quantile products. The next useful step is not blindly narrowing the quantile bins. It is to audit the existing resolution, add rank-based edge coordinates, enforce contributor-count thresholds, and—if more resolution is scientifically necessary—increase `N` or use an `N`-scaling design.

The current evidence is internally consistent:

- higher PA concentration produces larger hubs and stronger edge localization;
- clean positive gaps remain predominantly GOE-like;
- atoms at strong condensation or low density can mimic a loss of level repulsion if not segmented;
- exact-degree rewiring leaves the dominant edge effect in place, making the degree sequence the leading mechanism while preserving a secondary role for PA-specific wiring.

That is a strong result, but it is a result about edge localization and its mechanism—not yet a demonstrated sharp mobility edge.
