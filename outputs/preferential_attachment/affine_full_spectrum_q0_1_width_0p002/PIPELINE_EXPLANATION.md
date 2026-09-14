# Preferential attachment: literature, mathematics, implementation, and results

## 1. Literature basis

The production implementation was resumed from the affine preferential-attachment
backup made on 2026-08-12. The following primary sources were subsequently used to
verify the mathematical convention and interpret the spectral results.

1. H. A. Simon, [*On a Class of Skew Distribution Functions*](https://doi.org/10.1093/biomet/42.3-4.425), Biometrika 42 (1955), 425--440. This is historical reinforcement/cumulative-advantage background, not the graph algorithm used here.
2. D. de Solla Price, [*A General Theory of Bibliometric and Other Cumulative Advantage Processes*](https://doi.org/10.1002/asi.4630270505), JASIS 27 (1976), 292--306. Price formalizes the idea that the arrival rate of another success is proportional to previous successes and motivates a baseline term for the first success.
3. A.-L. Barabasi and R. Albert, [*Emergence of Scaling in Random Networks*](https://arxiv.org/abs/cond-mat/9910332), Science 286 (1999), 509--512. This gives the two defining network mechanisms: growth and preferential attachment to well-connected vertices.
4. S. N. Dorogovtsev, J. F. F. Mendes, and A. N. Samukhin, [*Structure of Growing Networks with Preferential Linking*](https://arxiv.org/abs/cond-mat/0004434), PRL 85 (2000), 4633--4636. This introduces initial attractiveness, the source of the affine shift used here.
5. P. L. Krapivsky, S. Redner, and F. Leyvraz, [*Connectivity of Growing Random Networks*](https://arxiv.org/abs/cond-mat/0005139), PRL 85 (2000), 4629--4632. This places the linear attachment kernel at the boundary between sublinear stretched-exponential behavior and superlinear concentration on a dominant vertex.
6. B. Bollobas, O. Riordan, J. Spencer, and G. Tusnady, [*The Degree Sequence of a Scale-Free Random Graph Process*](https://doi.org/10.1002/rsa.1009), Random Structures and Algorithms 18 (2001), 279--290. This gives a rigorous degree-sequence result for the unshifted Barabasi--Albert process.
7. K.-I. Goh, B. Kahng, and D. Kim, [*Spectra and Eigenvectors of Scale-Free Networks*](https://arxiv.org/abs/cond-mat/0103337), PRE 64 (2001), 051903. This is interpretation literature for adjacency spectral tails and localized edge eigenvectors; it is not used to generate graphs.
8. Y. Y. Atas, E. Bogomolny, O. Giraud, and G. Roux, [*The Distribution of the Ratio of Consecutive Level Spacings in Random Matrix Ensembles*](https://arxiv.org/abs/1212.5611), PRL 110 (2013), 084101. This supports the adjacent-gap-ratio diagnostic.

## 2. Mathematical realization of "popular vertices attract more links"

At growth time `t`, let `d_i(t)` be the current degree of existing vertex `i`.
For one edge of the new vertex, the affine rule is

\[
  \Pr(t\to i\mid G_t)
  =\frac{d_i(t)+\delta}
  {\sum_{j<t}\left(d_j(t)+\delta\right)}.
\]

The numerator is the popularity score. A vertex with more existing neighbors
has more ways to be encountered and is assigned proportionally more probability.
The denominator is the normalization that makes these scores probabilities.

The offset `delta` is initial attractiveness:

- `delta=0` is the standard Barabasi--Albert rule;
- positive `delta` adds a common baseline and weakens the relative degree advantage;
- negative `delta` makes the relative advantage of a high-degree vertex stronger;
- because every new vertex is born with degree `m`, positivity requires
  `m+delta>0`.

The implementation samples `m` distinct targets without replacement, reusing
the normalized `degree+delta` weights on the remaining targets. It therefore
constructs a simple graph and never creates a duplicate edge from one new vertex.

### Tail-exponent derivation

At large `t`, the graph has approximately `mt` edges and `t` vertices, so

\[
 \sum_j(d_j+\delta)\simeq(2m+\delta)t.
\]

Each new vertex supplies `m` edges. The continuum mean-field equation is

\[
 \frac{d d_i}{dt}
 =\frac{m(d_i+\delta)}{(2m+\delta)t}.
\]

With `d_i(t_i)=m` at the birth time `t_i`, integration gives

\[
 d_i(t)+\delta
 =(m+\delta)\left(\frac{t}{t_i}\right)^\beta,
 \qquad
 \beta=\frac{m}{2m+\delta}.
\]

Birth times are approximately uniform, hence

\[
 \Pr(D\geq k)
 \asymp
 \left(\frac{m+\delta}{k+\delta}\right)^{1/\beta}
 \asymp k^{-\alpha},
 \qquad
 \alpha=\frac1\beta=2+\frac{\delta}{m}.
\]

Thus the degree-density exponent is `gamma=alpha+1=3+delta/m`. The code uses
the upper-tail/CDF convention `alpha`, so

\[
 \delta=m(\alpha-2).
\]

For `m=10`, the three settings are `(alpha,delta)=(1.8,-2)`, `(1.5,-5)`,
and `(1.05,-9.5)`. They move from weaker to much stronger preferential
advantage. The condition `delta>-m` is equivalent to `alpha>1`.

## 3. Difference from Chung--Lu and rewiring

Chung--Lu starts from a fixed expected-weight sequence and samples an edge
between `i,j` from a probability determined by `w_i w_j/sum(w)`. Conditional
on the weights, edge probabilities do not change because another edge was
just sampled.

Preferential attachment is history-dependent. Adding an edge changes a degree,
which changes the probability law used at later growth steps. It also creates
age--degree correlations: old vertices have had more opportunities to receive
edges.

This is not rewiring. Degree-preserving rewiring begins with a finished graph,
keeps every degree fixed, and randomizes edge endpoints. No rewired graph was
included in this full-spectrum production run. An optional rewiring control
exists only in the postponed sparse-tail backup.

The numerical labels `1.8,1.5,1.05` are not identical tail laws across the two
models. The legacy Chung--Lu labels are expected-weight density exponents; the
PA labels here are degree upper-tail exponents. Exact matching is impossible
within standard affine PA when the requested Chung--Lu CCDF exponent is below
one. What was aligned was `N`, graph count, mean degree, quantile width, and
the numerical seed lists.

## 4. Graph and compute design

- `N=5000` vertices.
- `m=10` new edges per added vertex.
- Initial graph: complete graph on `m+1=11` vertices.
- Exact edge count:
  `mN-m(m+1)/2=49,945`.
- Exact realized mean degree: `2E/N=19.978`, versus Chung--Lu target 20.
- 500 graphs for each of three attachment settings; 1,500 graphs total.
- Corresponding Chung--Lu graph-seed bases: 27282, 27307, and 27352.
- Full dense symmetric eigendecomposition, shared by the eigenvalue and
  eigenvector analyses.
- Full continuous non-Perron spectral range `q in [0,1]`, width 0.002, giving
  500 bins.
- Hoffman array job 14319818: 300 tasks, five graphs per task.

The complete seed graph and ten outgoing edges per new vertex guarantee that
every realization is connected and has minimum degree at least ten. Therefore
this model cannot generate isolated vertices or isolated K2 components.

## 5. Per-graph spectral preprocessing

For each graph, the code diagonalizes the full adjacency matrix. It then:

1. saves all non-Perron eigenvalues for the eigenvalue analysis;
2. removes the largest Perron eigenpair;
3. removes exact-zero eigenpairs from the continuous scans, using tolerance
   `1e-8`;
4. separately records the zero multiplicity and exact `E=+1` and `E=-1`
   multiplicities;
5. applies both the eigenvalue and eigenvector pipelines to the same graph.

Removing an atom from the continuous scan does not erase it from the audit.
This separation is essential: otherwise a large atom can make a spacing
histogram look Poisson-like merely by inserting many zero gaps.

## 6. Eigenvalue pipeline

### Global density and unfolding

The exact-zero-removed non-Perron eigenvalues from all 500 graphs estimate an
ensemble density. At energy `E`, the local-count estimator is conceptually

\[
 \widehat\rho_h(E)
 =\frac{\#\{\lambda:|\lambda-E|\leq h\}}
 {G\,\overline M\,(2h)}.
\]

The code uses `h=0.20` in the bulk and `h=0.08` in the two 5% tails, then
renormalizes the density and numerically integrates

\[
 \widehat F(E)=\int_{-\infty}^{E}\widehat\rho_h(x)\,dx.
\]

Within one graph with continuous dimension `M`, selected eigenvalues are
unfolded by

\[
 \xi_i=M\widehat F(\lambda_i),\qquad
 s_i=\xi_{i+1}-\xi_i.
\]

Gaps are always formed within one graph. Eigenvalues from distinct graphs are
never placed next to each other to manufacture a gap.

### Spacing diagnostics

For adjacent gaps `delta_i=lambda_{i+1}-lambda_i`, the statistic is

\[
 r_i=\frac{\min(\delta_i,\delta_{i+1})}
 {\max(\delta_i,\delta_{i+1})}.
\]

The benchmarks used by the code are

\[
 \mathbb E[r]_{\rm GOE}=0.53590,
 \qquad
 \mathbb E[r]_{\rm Poisson}=2\log 2-1\simeq0.38629.
\]

The output reports absolute distance from both expectations. Positive unfolded
spacings are also normalized to mean one and compared by KS distance with

\[
 F_{\rm GOE}(s)=1-e^{-\pi s^2/4},
 \qquad
 F_{\rm Poisson}(s)=1-e^{-s}.
\]

Histograms are pooled for shape and sample size, whereas mean gap ratios are
first computed per graph and then averaged with equal graph weights.

## 7. Eigenvector pipeline

For a normalized eigenvector `v`, define coordinate masses `p_i=|v_i|^2` and
sort them as `p_(1)>=p_(2)>=...`. The scan records

\[
 \operatorname{IPR}(v)=\sum_i p_i^2,
 \quad
 \operatorname{PR}(v)=\operatorname{IPR}(v)^{-1},
 \quad
 C_r(v)=\sum_{j=1}^r p_{(j)},
 \quad
 r_x=\min\{r:C_r\geq x\}.
\]

For a Haar/GOE eigenvector,

\[
 \mathbb E[\operatorname{IPR}]=\frac{3}{N+2},
 \qquad N\operatorname{IPR}\approx3,
 \qquad \operatorname{PR}/N\approx1/3.
\]

Large `N*IPR`, large top-r mass, and small `r_x/N` indicate localization.
Metrics are averaged over eigenvectors inside each graph/bin and then over
graphs with equal graph weight. The heatmap displays `C_r` over all 500
continuous spectral quantiles and coordinate ranks 1--200.

The eigenvector percentile is a within-graph continuous spectral rank. The
eigenvalue spacing percentile is an ensemble-CDF energy window. They approach
the same energy ordering at large sample size but are not literally the same
selector; energy `E` must be used when comparing a narrow feature.

## 8. Results

### Fragment and exact-atom audit

| alpha | mean maximum degree | mean zero multiplicity | disconnected or isolated K2 | exact E=+/-1 multiplicity |
|---:|---:|---:|---:|---:|
| 1.8 | 458.964 | 0 | none | none |
| 1.5 | 688.574 | 0 | none | none |
| 1.05 | 2175.914 | 1854.914 | none | none |

The Chung--Lu isolated-K2 `E=+/-1` mechanism does not recur. Connectivity and
the absence of K2 components are built into this PA construction, not merely
observed by chance.

The strong-preference run instead has a true extensive zero atom inside a
connected graph. For one representative `alpha=1.05` graph, 3,245 vertices
remain at the minimum degree ten. These vertices form an independent set and
their combined neighbor set has only 1,641 vertices. With the adjacency matrix
ordered as this independent set `S` and the remaining vertices,

\[
 A=\begin{pmatrix}0&B\\B^T&C\end{pmatrix}.
\]

Vectors supported on `S` and lying in `ker(B^T)` are zero eigenvectors, so

\[
 \operatorname{nullity}(A)\geq |S|-\operatorname{rank}(B)
 \geq3245-1641=1604.
\]

The observed nullity in that graph is 1,844. This explains most of the zero
atom without disconnected pieces or identical neighborhoods.

### Eigenvectors

For `alpha=1.8` and `1.5`, the continuous bulk is close to the Haar scale,
while both spectral edges are strongly localized. The maximum mean `N*IPR`
values are 342.20 and 351.25 at the negative edge (`q=0.001`).

For `alpha=1.05`, several interior bands and much stronger edge concentration
appear. The largest mean value is `N*IPR=708.98` at `q=0.993`, `E=20.48`,
with top-50 mass 0.538. Near energy `E=+1`, the closest bin has `q=0.665`,
`N*IPR=44.62`, top-50 mass 0.447, and top-200 mass 0.694. Near `E=-1`, the
corresponding values are `q=0.335`, `N*IPR=46.41`, top-50 mass 0.442, and
top-200 mass 0.689. These are concentration bands near `+/-1`, but there is
no exact `+/-1` eigenvalue multiplicity.

### Eigenvalues

- `alpha=1.8`: all 500 quantile bins have mean gap ratio closer to GOE than
  Poisson.
- `alpha=1.5`: all 500 bins are closer to GOE.
- `alpha=1.05`: 496 of 498 valid bins are closer to GOE.

For `alpha=1.05`, `q=0.499` and `0.501` contain no positive spacings because
they fall in the central interval created after the extensive zero atom is
removed. The adjacent bins `q=0.497` and `0.503` have mean ratios 0.446 and
0.444 and are closer to Poisson, but only 164 and 107 graphs contribute.
These transition bins are not evidence of a bulk Poisson regime.

At `E=+1`, even the strong-preference run remains GOE-like:
mean `r=0.5262`, distance 0.0097 to GOE versus 0.1399 to Poisson, and KS
distance 0.0277 to GOE versus 0.1973 to Poisson. Thus localized eigenvectors
near `E=+1` coexist with repulsive local statistics among the nondegenerate
eigenvalues.

## 9. Comparison with Chung--Lu

The narrow Chung--Lu `E=1` peak was dominated by an exact atom generated by
isolated K2 components. For example, the Chung--Lu density-exponent 1.5 run at
`q=0.683` had mean `N*IPR=1204.65`, top-50 mass 0.777, atom fraction 0.503,
zero-gap fraction 0.447, inclusive mean gap ratio 0.362, and conditional
positive-gap ratio 0.539.

The comparable PA `alpha=1.5` bin near energy `E=1` occurs at `q=0.617`, not
`q=0.683`, and has mean `N*IPR=3.52`, top-50 mass 0.100, no exact atom, and
mean gap ratio 0.525. Therefore the Chung--Lu K2 resonance does not transfer.

The result that does transfer is the broader eigenvector/eigenvalue mismatch:
edge eigenvectors can concentrate strongly while nearby nondegenerate
eigenvalue spacings remain GOE-like. This is consistent with prior spectral
studies of scale-free networks that found localized adjacency eigenvectors at
high-degree vertices, but it is not by itself a finite-size proof of a sharp
mobility edge.

## 10. Code map

- Affine exponent map and kernel: `src/small_world_mobility_edge/preferential_attachment/baseline.py`, lines 32--43 and 54--89.
- Connectivity, edge count, and component diagnostics: the same file, lines 91--117.
- Parameters, seed bases, quantile grid, and GOE/Poisson constants: `full_spectrum_scan.py`, lines 34--130.
- Dense diagonalization and zero/Perron removal: lines 163--182.
- IPR, PR, top-r mass, and target-mass radius: lines 185--245.
- Exact-zero and `E=+/-1` audit: lines 304--342.
- Chunk partials that share the eigendecomposition: lines 345--395.
- Graph-equal eigenvector aggregation: lines 424--470.
- Density, CDF, unfolding, gap ratios, KS distances, and histograms: lines 530--694.
- Component/multiplicity aggregation: lines 697--729.
- Final manifests and figures: lines 896--952.
- Hoffman task mapping: `tools/hoffman_preferential_attachment_full_spectrum_chunk.py`, lines 8--35.
- Regression/smoke tests: `tests/test_preferential_attachment_full_spectrum.py`, lines 14--64.
