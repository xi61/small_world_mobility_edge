# Technical handoff: preferential attachment and Chung–Lu spectral-localization experiments

**Project:** `small_world_mobility_edge`
**Prepared for:** independent review and design of the next preferential-attachment experiments
**Date:** August 22, 2026

## 1. Purpose and short conclusion

This document explains what has actually been implemented, what each output measures, what the present results do and do not establish, and which controls are still needed. It is intended to be self-contained enough that another ChatGPT session can inspect the model and propose the next experiments without relying on the earlier conversation.

There are currently **two preferential-attachment experiment pipelines**, not one:

1. A five-graph **standard Barabási–Albert pilot** at (N=5000,m=10). It computes only sparse spectral edges and was designed primarily for degree–spectral-tail matching and a degree-preserving rewiring comparison. It is useful mechanism evidence, but it is not a full-spectrum or high-power localization study.
2. A 500-graph-per-parameter **affine preferential-attachment production experiment** at (N=5000,m=10). It computes the full adjacency eigendecomposition, eigenvalue-spacing histograms and GOE/Poisson diagnostics, eigenvector IPR and top-r mass across the full non-Perron spectrum, and exact multiplicities at (0 and ±1).

The principal current findings are:

- Preferential attachment does **not** reproduce the Chung–Lu exact (E=\pm1) resonance caused by isolated two-vertex components. The implemented growth rule makes every graph connected with minimum degree at least (m=10), so those components are structurally impossible.
- The strongest affine preference, degree-tail exponent (alpha=1.05), creates a large **zero-eigenvalue atom** in a connected graph. This is a rank-deficiency/independent-set phenomenon, not fragmentation.
- Eigenvectors become strongly concentrated near spectral edges, but the positive local eigenvalue spacings remain overwhelmingly closer to GOE than to Poisson at (N=5000). Thus the present data do not establish a conventional GOE-to-Poisson mobility edge.
- The standard Barabási–Albert tail pilot supports the relation ( |\lambda_{(j)}|\approx c\sqrt{d_{(j)}} ), and a degree-preserving rewiring gives a similar relation. This suggests that much of the spectral-edge tail is controlled by the degree sequence, although the five-graph sample is too small for a definitive conclusion.
- The labels (1.8,1.5,1.05) are **not exponent-matched between the current affine preferential-attachment and legacy Chung–Lu runs**. This is the most important design issue to fix before treating them as controlled model comparisons.

## 2. The graph models

### 2.1 Standard Barabási–Albert preferential attachment

The standard Barabási–Albert model grows a graph sequentially. The implementation begins with the complete graph (K_{m+1}). Every new vertex creates (m) edges to distinct existing vertices. In the standard model, the probability of choosing existing vertex (i) is proportional to its current degree:

\[
  \Pr(t\to i\mid G_t)
  =\frac{d_i(t)}{\sum_{j<t}d_j(t)}.
\]

This is the mathematical realization of “popular vertices are more likely to receive new connections.” Adding an edge changes a degree, so it also changes later attachment probabilities. The graph is therefore a history-dependent growth process.

With this rule, the asymptotic degree probability mass function behaves as

\[
  p_D(k)\asymp k^{-3},
\]

and its upper tail behaves as

\[
  \Pr(D\ge k)=1-F_D(k)\asymp k^{-2}.
\]

The code identifies this as the Barabási–Albert model and records the theoretical **degree upper-tail exponent** (2). See `src/small_world_mobility_edge/preferential_attachment/baseline.py`, lines 12–23 and 136–173.

### 2.2 Affine linear preferential attachment

The production full-spectrum experiment uses the more general attachment kernel

\[
  \Pr(t\to i\mid G_t)
  =\frac{d_i(t)+\delta}
  {\sum_{j<t}[d_j(t)+\delta]}.
\]

The offset (delta) is an initial-attractiveness parameter:

- (delta=0) gives standard Barabási–Albert attachment;
- (delta>0) adds the same baseline attractiveness to every vertex and weakens the relative advantage of high degree;
- (-m<delta<0) strengthens the relative advantage of already-high-degree vertices;
- (delta>-m) is required because each newly born vertex initially has degree (m), and its attractiveness (m+delta) must remain positive.

The implementation samples (m) **distinct** targets without replacement with weights (d_i+delta), so it produces a simple undirected graph. The kernel and sampling loop are in `src/small_world_mobility_edge/preferential_attachment/baseline.py`, lines 85–122.

#### Derivation of the power law

At large time (t), the normalization satisfies

\[
  \sum_j[d_j(t)+\delta]\simeq(2m+\delta)t.
\]

Since each new vertex contributes (m) new edges, the continuum approximation for a vertex born at time (t_i) is

\[
  \frac{d d_i}{dt}
  =\frac{m[d_i(t)+\delta]}{(2m+\delta)t}.
\]

Using (d_i(t_i)=m), integration gives

\[
  d_i(t)+\delta
  =(m+\delta)\left(\frac{t}{t_i}\right)^\beta,
  \qquad
  \beta=\frac{m}{2m+\delta}.
\]

Birth times are approximately uniform, so the degree upper tail is

\[
  \Pr(D\ge k)\asymp k^{-\alpha},
  \qquad
  \alpha=\frac1\beta=2+\frac{\delta}{m}.
\]

Equivalently, the degree-density exponent is

\[
  \gamma=\alpha+1=3+\frac{\delta}{m}.
\]

The code accepts the degree **CDF/upper-tail exponent** (alpha) and sets

\[
  \delta=m(\alpha-2).
\]

For (m=10), the production settings are:

| PA upper-tail exponent (alpha) | Offset (delta) | Minimum newborn attractiveness (m+delta) | Interpretation |
|---:|---:|---:|---|
| 1.8 | -2.0 | 8.0 | moderately stronger preference than BA |
| 1.5 | -5.0 | 5.0 | stronger preference |
| 1.05 | -9.5 | 0.5 | extremely strong preference, near the admissible boundary |

The exponent conversion and positivity check are implemented at `src/small_world_mobility_edge/preferential_attachment/baseline.py`, lines 85–96.

### 2.3 Optional Holme–Kim model

The generator layer also supports the Holme–Kim model through NetworkX. It adds a triad-formation probability (p\in[0,1]), so after preferential attachment it can close triangles and increase clustering. This is not “rewiring,” and it is not used in the checked 500-graph production outputs. The code path is at `src/small_world_mobility_edge/preferential_attachment/baseline.py`, lines 174–182.

### 2.4 Preferential attachment is not rewiring

Preferential attachment constructs a graph through growth. A high-degree vertex receives a larger attachment probability at later times, and old vertices have had more opportunities to receive edges.

Degree-preserving rewiring starts with a completed graph and applies double-edge swaps. For example, edges ((a,b)) and ((c,d)) may be replaced by ((a,d)) and ((c,b)) when this creates no loops or duplicate edges. Every labeled vertex keeps exactly the same degree. Rewiring removes much of the growth history, age–degree organization, and local wiring pattern while preserving the degree sequence.

Consequently, the comparison

\[
  \text{PA graph}\quad\text{versus}\quad
  \text{degree-preserving rewiring of that same graph}
\]

is the cleanest implemented test of whether a spectral feature is explained mainly by degrees or also by the special organization created by growth.

## 3. Parameter meanings and graph invariants

In the production affine experiment:

- (N=5000) is the **number of vertices in each graph**, not the number of graphs.
- (m=10) is the number of edges created by every arriving vertex.
- (G=500) is the number of independent graph realizations for each (alpha).
- The seed graph is (K_{11}).
- The exact number of edges is

\[
  E=\frac{m(m+1)}2+m[N-(m+1)]
   =mN-\frac{m(m+1)}2
   =49{,}945.
\]

- The exact realized mean degree is

\[
  \bar d=\frac{2E}{N}=19.978,
\]

which is very close to the Chung–Lu target (20).
- Every graph is connected and has minimum degree (10). Isolated vertices and isolated (K_2) components cannot occur.
- There are 500 width-(0.002) quantile bins over the full continuous non-Perron spectrum, (q\in[0,1]).

The production constants and preference specifications are in `src/small_world_mobility_edge/preferential_attachment/full_spectrum_scan.py`, lines 38–97. The graph creation, full diagonalization, component audit, and exact multiplicity counts are at lines 308–359. Final aggregation and manifest creation are at lines 920–956.

## 4. Current preferential-attachment experiments

### 4.1 Sparse Barabási–Albert tail and rewiring pilot

This pilot uses (N=5000,m=10,G=5), standard BA ((delta=0,alpha=2)), and five paired degree-preserving rewirings. It computes 180 algebraically largest and 180 algebraically smallest eigenvalues using a sparse solver rather than diagonalizing the whole adjacency matrix.

It runs four related tests.

#### A. Degree–spectral-tail matching

The phrase “match the long tails of the spectral density to the tails of the degree sequence” is implemented as the following hypothesis. If high-degree vertices create edge eigenvalues satisfying

\[
  |\lambda|\asymp\sqrt D,
\]

and the degree upper tail is

\[
  \Pr(D>x)\sim x^{-\alpha},
\]

then

\[
  \Pr(|\lambda|>x)
  \approx\Pr(D>x^2)
  \sim x^{-2\alpha},
\]

and, equivalently,

\[
  \Pr(\lambda^2>y)\sim y^{-\alpha}.
\]

The code therefore estimates upper-tail exponents for (D), (|\lambda|), and (lambda^2) using Hill estimates and log-CDF regressions across lower tail thresholds (0.95,0.96,0.97,0.976,0.98). It reports the deviations

\[
  \widehat\beta_{|\lambda|}-2\widehat\alpha_D,
  \qquad
  \widehat\gamma_{\lambda^2}-\widehat\alpha_D.
\]

These computations are in `src/small_world_mobility_edge/preferential_attachment/experiments.py`, lines 282–368.

#### B. Rank-matched scaling

The largest degree and largest non-Perron edge eigenvalue are also compared by rank:

\[
  \log |\lambda_{(j)}|=a+b\log d_{(j)}.
\]

The predicted slope is (b=1/2). The output additionally reports (R^2), Spearman rank correlation, and the ratio (|\lambda_{(j)}|/\sqrt{d_{(j)}}). See `preferential_attachment/experiments.py`, lines 371–397.

#### C. Sparse edge spacings and localization

The retained edge eigenvalues are divided into fixed, disjoint rank windows. Within each graph and window, gaps are normalized by their own mean. Adjacent gap ratios and eigenvector localization measures are then aggregated. These are edge-rank diagnostics only; they do not constitute a global density or a full-spectrum histogram.

#### D. Exact-degree rewiring control

Every BA graph is rewired by double-edge swaps, and the code verifies exact equality of the labeled degree sequence before comparing tails, edge spacings, and edge-eigenvector localization. The observed edge overlap is only about (1.85\%), indicating that the wiring has been substantially changed even though degrees are identical. The finite swap chain is an approximate randomized simple graph conditional on the degree sequence; uniform sampling from all such graphs is not claimed.

#### Pilot results

At the primary lower-tail threshold (q=0.976), the five BA graphs give:

| Quantity | BA | Degree-preserving rewiring |
|---|---:|---:|
| estimated degree-tail exponent | 2.036 | 2.036 by construction |
| estimated (|\lambda|)-tail exponent | 4.617 | 4.560 |
| estimated (lambda^2)-tail exponent | 2.309 | 2.280 |
| (|\lambda|) exponent minus twice degree exponent | 0.546 | 0.489 |
| (lambda^2) exponent minus degree exponent | 0.273 | 0.244 |

The rank-matched fits are stronger than the direct tail-exponent estimates:

| Variant and side | slope (b) | (R^2) | median (|\lambda|/\sqrt d) |
|---|---:|---:|---:|
| BA, positive non-Perron | 0.440 | 0.996 | 1.056 |
| BA, negative absolute | 0.466 | 0.997 | 1.065 |
| rewired, positive non-Perron | 0.453 | 0.995 | 1.058 |
| rewired, negative absolute | 0.472 | 0.996 | 1.067 |

The similarity after rewiring supports a degree-sequence explanation for much of the edge-tail relation. The deviations from the ideal exponent and slope predictions should be treated as finite-size/threshold evidence, not as an exact asymptotic validation, because only five graphs and 180 eigenvalues per side were retained.

The same pilot includes an **unpaired** Chung–Lu ensemble with degree upper-tail exponent (2), (N=5000), target mean degree (20), five graphs, and the same sparse solver settings. It compares degree and sparse spectral tails only; it does not load a matching Chung–Lu full-spectrum IPR or spacing result. This scope is recorded in `outputs/preferential_attachment/barabasi_albert/n_5000_m_10/run_manifest.json`.

### 4.2 Affine full-spectrum production experiment

For each (alpha\in\{1.8,1.5,1.05\}), 500 graphs are generated and the full (5000\times5000) adjacency matrix is diagonalized. One eigendecomposition is shared by all eigenvalue and eigenvector diagnostics for that graph.

For each graph, the pipeline:

1. saves all non-Perron eigenvalues for density, atom, and spacing analysis;
2. removes the largest Perron eigenpair;
3. removes exact-zero eigenpairs from the *continuous-spectrum* scans using tolerance (10^{-8});
4. retains zero multiplicity and exact (E=\pm1) multiplicities in a separate audit;
5. assigns the remaining eigenpairs to spectral bins;
6. calculates IPR, participation ratio, top-r mass, mass radii, unfolded spacings, gap ratios, and distributional distances;
7. writes graph-level partial files, then forms graph-equal ensemble summaries.

The full pipeline is implemented in `src/small_world_mobility_edge/preferential_attachment/full_spectrum_scan.py`; the main graph analysis is at lines 308–359 and the joint final aggregation is at lines 920–956.

## 5. Eigenvalue-spacing and histogram test

### 5.1 What the histogram represents

The main histogram is **not simply a histogram of raw adjacency eigenvalues**. It is a histogram of locally unfolded adjacent eigenvalue spacings. The global empirical spectral density is also saved, but it serves primarily to unfold the spectrum.

After removing the Perron eigenvalue and separating exact zeros, the eigenvalues from all (G=500) graphs estimate the ensemble density. At grid energy (E), the local-count estimate is

\[
  \widehat\rho_h(E)
  =\frac{\#\{\lambda:|\lambda-E|\le h\}}
  {G\,\overline M\,(2h)},
\]

where (overline M) is the mean retained continuous dimension. The implementation uses half-width (h=0.20) in the bulk and (h=0.08) in the outer (5\%) tails. It normalizes the density and computes

\[
  \widehat F(E)=\int_{-\infty}^{E}\widehat\rho_h(x)\,dx.
\]

Within graph (g), an eigenvalue is unfolded as

\[
  \xi_i^{(g)}=M_g\widehat F(\lambda_i^{(g)}),
  \qquad
  s_i^{(g)}=\xi_{i+1}^{(g)}-\xi_i^{(g)}.
\]

Gaps are never formed across different graphs. The pooled histogram is created only after within-graph gaps have been constructed. The density/CDF estimator is implemented in `src/small_world_mobility_edge/empirical_spectral_density.py`, especially lines 55–63 and 115–179. The Chung–Lu energy-bin selection, unfolding, and aggregation are in `src/small_world_mobility_edge/chung_lu_spacing_by_energy.py`, principally lines 231–618; the affine PA full-spectrum code implements the same design locally.

### 5.2 GOE and Poisson comparisons

The unfolded spacing CDFs used as benchmarks are

\[
  F_{\mathrm{GOE}}(s)=1-e^{-\pi s^2/4},
  \qquad
  F_{\mathrm{Poisson}}(s)=1-e^{-s}.
\]

The GOE/Wigner form vanishes at (s=0), representing level repulsion. The Poisson density is maximal at (s=0), representing independent levels without repulsion.

The unfolding-free adjacent-gap ratio is

\[
  r_i=\frac{\min(\delta_i,\delta_{i+1})}
  {\max(\delta_i,\delta_{i+1})},
  \qquad
  \delta_i=\lambda_{i+1}-\lambda_i.
\]

The reference expectations are

\[
  \mathbb E[r]_{\mathrm{GOE}}\simeq0.53590,
  \qquad
  \mathbb E[r]_{\mathrm{Poisson}}=2\log2-1\simeq0.38629.
\]

Each summary bin reports the graph-equal mean gap ratio, its absolute distance from both expectations, and KS distances between positive unfolded spacings and both benchmark CDFs. Graph-equal aggregation prevents graphs with more selected gaps from dominating the reported mean.

### 5.3 Exact atoms must be treated separately

A repeated eigenvalue creates zero gaps. Those zeros can produce a decreasing histogram and a low mean gap ratio even if the surrounding **positive** spacings retain GOE-like repulsion. Therefore the pipeline records:

- the fraction of eigenvalues in an exact atom;
- the fraction of degenerate raw gaps;
- an inclusive ratio that retains zero/nonzero neighboring pairs;
- a conditional ratio and KS test using positive gaps only.

This distinction was essential for interpreting the Chung–Lu (E=1) resonance.

## 6. Eigenvector localization and IPR test

For a normalized eigenvector (v), define coordinate masses

\[
  p_i=|v_i|^2,
  \qquad \sum_i p_i=1.
\]

The inverse participation ratio is

\[
  \operatorname{IPR}(v)=\sum_i p_i^2=\sum_i|v_i|^4.
\]

Its reciprocal is the participation ratio,

\[
  \operatorname{PR}(v)=\operatorname{IPR}(v)^{-1}.
\]

Interpretation:

- a flat vector has (operatorname{IPR}=1/N) and (operatorname{PR}=N);
- a vector on (k) equally weighted coordinates has (operatorname{IPR}=1/k) and (operatorname{PR}=k);
- a real Haar/GOE vector has
  \(mathbb E[\operatorname{IPR}]=3/(N+2)\), so (N\operatorname{IPR}\approx3) and (operatorname{PR}/N\approx1/3).

The scan also sorts (p_{(1)}\ge p_{(2)}\ge\cdots) and records

\[
  C_r=\sum_{j=1}^r p_{(j)},
  \qquad
  r_x=\min\{r:C_r\ge x\}.
\]

Large (N\operatorname{IPR}), large top-r mass (C_r), and small (r_x/N) all indicate coordinate concentration. The heatmap shows mean cumulative top-r mass as a function of spectral quantile and coordinate rank (r\le200).

At one fixed (N), these metrics establish concentration but not asymptotic localization. A mobility-edge claim requires size scaling such as

\[
  \operatorname{IPR}_N(E)\asymp N^{-D_2(E)}.
\]

Here (D_2=1) indicates extended eigenvectors, (D_2=0) localized eigenvectors, and (0<D_2<1) an intermediate/nonergodic scaling. The eigenvalue-spacing and eigenvector crossovers should be tested in the same energy windows across several (N).

## 7. Chung–Lu comparator

### 7.1 Graph generation

The Chung–Lu model begins with expected degrees or weights (w_1,\ldots,w_N) and independently includes edge (i\!-!j) with probability approximately

\[
  p_{ij}=\frac{w_iw_j}{\sum_k w_k}.
\]

The code uses NetworkX’s expected-degree graph without self-loops. It imposes a structural cutoff and solves a clipped scaling equation so the final expected weights have the target mean degree while satisfying the admissibility condition

\[
  w_{\max}^2\le\sum_i w_i.
\]

Expected-weight construction and cutoff logic are in `src/small_world_mobility_edge/chung_lu_baseline.py`, lines 107–129 and 149–340; graph sampling is at lines 345–349.

Unlike preferential attachment, conditional on the final weights the edge draws are independent and there is no graph-growth history. Low expected degrees are allowed, so disconnected vertices, small components, and pendant structures can occur.

### 7.2 Exponent convention in the legacy Chung–Lu runs

The legacy localization runs use an expected-weight **density exponent** (tau_d):

\[
  f_W(w)\propto w^{-\tau_d},
  \qquad
  \Pr(W>w)\propto w^{-(\tau_d-1)}.
\]

The old folder labeled `tau_0p8` used (0.8) as an upper-tail exponent, so it corresponds to density exponent (tau_d=1.8). The other displayed labels `1.05` and `1.5` were passed as density exponents. This mixed legacy mapping is explicitly preserved in `src/small_world_mobility_edge/chung_lu_eigenvector_percentile_scan.py`, lines 70–111.

This means the current numerical PA and Chung–Lu labels are not matched:

| displayed number | affine PA meaning | legacy Chung–Lu meaning | same tail law? |
|---:|---|---|---|
| 1.8 / legacy 0.8 | degree CCDF exponent (1.8) | weight-density exponent (1.8), hence weight CCDF exponent (0.8) | no |
| 1.5 | degree CCDF exponent (1.5) | weight-density exponent (1.5), hence weight CCDF exponent (0.5) | no |
| 1.05 | degree CCDF exponent (1.05) | weight-density exponent (1.05), hence weight CCDF exponent (0.05) | no |

Moreover, an unbounded power-law density requires (tau_d > 1), while affine PA requires degree CCDF exponent (alpha>1). Numerically copying a label does not align these constraints or distributions.

### 7.3 Chung–Lu IPR experiment

For (N=5000), target mean degree (20), and (G=500), the eigenvector scan removes the Perron and exact-zero eigenpairs and divides (q\in[0.5,1]) into 250 bins of width (0.002). Metrics are averaged inside each graph/bin and then across graphs with graph-equal weights and graph-level standard errors. The calculation and aggregation are in `src/small_world_mobility_edge/chung_lu_eigenvector_percentile_scan.py`, lines 160–364 and 580–735.

The upper half is sufficient for positive-energy and upper-edge behavior, but the affine PA production scan covers (q\in[0,1]) to display both spectral edges.

The strongest narrow Chung–Lu peaks are:

| weight-density exponent | peak quantile | mean energy | mean (N\operatorname{IPR}) | mean top-50 mass |
|---:|---:|---:|---:|---:|
| 1.8 | 0.685 | 0.998 | 127 | 0.365 |
| 1.5 | 0.683 | 1.000 | 1205 | 0.777 |
| 1.05 | 0.633 | 1.005 | 1751 | 0.929 |

Because each curve falls after the narrow peak, this is a resonance band rather than a one-way mobility edge.

### 7.4 Chung–Lu (E=\pm1) diagnosis

An isolated two-vertex component has adjacency matrix

\[
 A_{K_2}=\begin{pmatrix}0&1\\1&0\end{pmatrix},
 \qquad\operatorname{spec}(A_{K_2})=\{-1,+1\}.
\]

Its eigenvectors ((1,\pm1)/\sqrt2) have IPR (1/2), or (N\operatorname{IPR}=2500) at (N=5000). Repeated copies create exact (±1) multiplicity and very large apparent localization.

For the Chung–Lu density-exponent (1.5) run:

- 3,544 isolated (K_2) components occur across 500 graphs, or 7.088 per graph;
- they account for (94.3\%) of the full-graph (E=+1) multiplicity;
- at (q=0.683), the full-graph mean (N\operatorname{IPR}) is 1204.65 and top-50 mass is 0.777;
- passing to the same graph’s largest connected component lowers them to 79.34 and 0.551;
- the residual exact (+1) modes inside the largest connected component disappear in its 2-core, implicating degree-one structures rather than isolated components alone.

Because eigenvectors inside a degenerate eigenspace can be arbitrarily rotated, individual IPR values there are basis-dependent. The exact-(E=\pm1) audit therefore uses the invariant diagonal of the spectral projector (VV^T). This is implemented in `src/small_world_mobility_edge/chung_lu_e1_projector_core_scan.py`, lines 53–83 and 156–220; the full graph, largest connected component, and 2-core construction is at lines 239–319.

In the (q=0.683) bin, (50.3\%) of selected eigenvalues equal (+1) and (44.6\%) of raw gaps are zero. The inclusive mean gap ratio is (0.362), but this is not Poisson evidence: a zero gap adjacent to a positive gap contributes (r=0). Conditional on two positive gaps, the mean ratio is (0.539), close to GOE (0.5359) and far from Poisson (0.3863). The atom/positive-gap separation is implemented in `src/small_world_mobility_edge/chung_lu_eigenvalue_resonance_scan.py`, lines 69–99 and 240–318.

### 7.5 Chung–Lu upper edge

Away from exact resonances, the Chung–Lu density-exponent (1.5) run shows a mismatch between the two diagnostics. From (q=0.799) to (q=0.999), mean (N\operatorname{IPR}) grows from about (10.2) to (33.9), and top-200 mass grows from (0.491) to (0.805). Yet near (q=0.94), the mean gap ratio is about (0.531), with KS distance (0.011) to GOE and (0.209) to Poisson; near (q=0.9965), the mean gap ratio remains about (0.535).

Thus the present finite-(N) result is **edge eigenvector concentration with GOE-like nondegenerate level statistics**, not a textbook paired localization transition.

## 8. Affine preferential-attachment results

### 8.1 Components and exact atoms

| PA exponent (alpha) | mean maximum degree | mean zero multiplicity | disconnected graphs or isolated (K_2) | exact (E=\pm1) multiplicity |
|---:|---:|---:|---:|---:|
| 1.8 | 458.964 | 0 | 0 | 0 |
| 1.5 | 688.574 | 0 | 0 | 0 |
| 1.05 | 2175.914 | 1854.914 | 0 | 0 |

The absence of the Chung–Lu (K_2) mechanism is built into the graph generator: these PA graphs are connected and have minimum degree ten.

The (alpha=1.05) zero atom has a different cause. In a representative graph, 3,245 vertices remain at degree ten and form an independent set (S); together they have only 1,641 distinct neighbors. Ordering the adjacency matrix as (S) and its complement gives

\[
 A=\begin{pmatrix}0&B\\B^T&C\end{pmatrix}.
\]

Vectors supported on (S) in (ker(B^T)) are zero modes, hence

\[
 \operatorname{nullity}(A)
 \ge |S|-\operatorname{rank}(B)
 \ge3245-1641=1604.
\]

The observed nullity for that graph is 1844. Therefore most of the extensive zero atom is explained by an independent set with a smaller neighbor union, not by disconnected fragments.

### 8.2 Eigenvectors

- For (alpha=1.8) and (1.5), the continuous bulk is near the Haar scale, while both spectral edges are strongly concentrated. The maximum mean (N\operatorname{IPR}) values are 342.20 and 351.25 at the negative edge (q=0.001).
- For (alpha=1.05), the largest mean (N\operatorname{IPR}) is 708.98 at (q=0.993), (E=20.48), with top-50 mass 0.538.
- Near (E=+1) for (alpha=1.05), the closest bin is (q=0.665), with (N\operatorname{IPR}=44.62), top-50 mass 0.447, and top-200 mass 0.694. Near (E=-1), (q=0.335) has (N\operatorname{IPR}=46.41), top-50 mass 0.442, and top-200 mass 0.689. These are concentration bands, but not exact atoms.

### 8.3 Eigenvalues

- All 500 bins for (alpha=1.8) and all 500 for (alpha=1.5) have graph-equal mean gap ratios closer to GOE than Poisson.
- For (alpha=1.05), 496 of 498 valid bins are closer to GOE.
- The two exceptions lie beside the interval removed with the extensive zero atom: (q=0.497) and (0.503) have ratios (0.446) and (0.444), but only 164 and 107 graphs contribute. They should not be interpreted as a bulk Poisson phase.
- Near (E=+1) for (alpha=1.05), the mean ratio is (0.5262), its distance is (0.0097) to GOE versus (0.1399) to Poisson, and its KS distance is (0.0277) to GOE versus (0.1973) to Poisson.

The affine results therefore repeat the broader Chung–Lu pattern: strong eigenvector concentration can coexist with GOE-like correlations among the remaining nondegenerate eigenvalues.

## 9. What is comparable and what is confounded

### Aligned quantities

- (N=5000);
- approximately the same mean degree, (20) versus (19.978);
- (G=500) for the full-spectrum runs;
- width-(0.002) quantile resolution;
- full adjacency eigendecomposition;
- Perron removal and separate exact-zero audit;
- IPR, top-r, mass-radius, gap-ratio, KS, and histogram concepts.

### Not aligned

1. **Tail exponent:** the same numerical label denotes different exponents.
2. **Minimum degree and components:** PA has minimum degree ten and is connected; Chung–Lu permits degree zero/one and fragmentation.
3. **History:** PA has growth and age correlations; Chung–Lu has independent edges conditional on weights.
4. **Degree sequence:** the realized PA and Chung–Lu degree sequences are not the same.
5. **Spectral selection:** the Chung–Lu eigenvector scan covers (q\in[0.5,1]); affine PA covers (q\in[0,1]).
6. **Seeds:** reusing the same integer seed lists across different generators does not create paired random graphs or common random numbers. The comparison is still between independent ensembles.
7. **Rewiring:** full-spectrum affine PA was not rerun after degree-preserving rewiring. Rewiring exists only in the five-graph sparse BA pilot.
8. **System size:** the principal studies use only (N=5000), so they cannot determine IPR scaling exponents or prove a mobility edge.

The disappearance of isolated (K_2) multiplicity after switching models therefore cannot be attributed solely to “preference.” It also follows directly from the PA construction’s connected seed and minimum-degree constraint.

## 10. Recommended next experimental design

The next design should first decide which causal question is intended.

### Question A: Does growth history matter after fixing degrees?

Use each affine PA graph and a degree-preserving rewiring of that exact graph. Run the **same full-spectrum** eigenvalue and eigenvector pipeline on both. This controls the complete labeled degree sequence and changes wiring/history. Aggregate paired graph-level differences with confidence intervals.

This is the highest-priority extension because the code already has a validated rewiring implementation in the sparse BA pipeline.

### Question B: How do PA and independent-edge graphs differ at matched degree tails?

Match the **same degree CCDF exponent (alpha)** rather than the same numerical legacy label. For Chung–Lu, choose weights whose upper-tail exponent is (alpha), which corresponds to weight-density exponent (alpha+1) when that relation is valid. Then verify the *realized* degree CCDF rather than assuming expected weights and degrees are identical.

Compare at least:

- affine PA;
- degree-preserving rewiring of affine PA;
- Chung–Lu with matched realized degree tail and mean degree;
- if possible, a simple configuration-model graph with the same realized degree sequence.

The last two separate independent-edge/degree-sequence effects from growth correlations.

### Question C: Is there an asymptotic mobility edge?

Repeat aligned energy-window measurements for several sizes, for example

\[
 N\in\{1000,2000,5000,10000\},
\]

with graph counts chosen to keep graph-level uncertainty comparable. Estimate (D_2(E)) from the slope of (log\operatorname{IPR}) versus (log N), and scale (r_{50},r_{80},r_{90}) with (N). In the same energy windows, test whether gap ratios and spacing CDFs move from GOE toward Poisson.

Quantile alignment alone is insufficient for narrow motifs because an atom or zero removal shifts ranks. Save and compare both energy (E) and within-graph quantile (q).

### Question D: What causes the (alpha=1.05) zero atom?

For every graph, record:

- number of minimum-degree vertices;
- size of their induced subgraph;
- size of their neighbor union;
- lower bound (|S|-|N(S)|);
- numerical rank/nullity of the corresponding bipartite block;
- overlap of the zero-eigenspace projector with (S).

Then vary (N), (m), and (alpha) to locate whether the nullity fraction converges to a positive value and where it turns on.

### Question E: Do the spectral and degree tails match in affine PA?

Extend the five-graph BA tail pipeline to (alpha=1.8,1.5,1.05), with more graphs and several (N). For every original/rewired pair, report:

- empirical degree, (|\lambda|), and (lambda^2) upper-tail CDFs;
- Hill and log-CDF exponent stability across thresholds;
- rank slope (b), (R^2), and (|\lambda|/\sqrt d);
- finite-size drift and uncertainty;
- sensitivity to the number of retained sparse eigenpairs.

This directly addresses whether the long spectral-density tails match the degree-sequence tails, while the full-spectrum experiment addresses local spacings and eigenvectors.

### Statistical safeguards

1. Keep histogram pooling for display, but use graph-equal or paired graph-level estimates for inference.
2. Always report the number of contributing graphs and gaps in each bin.
3. Separate exact atoms and zero gaps from positive-spacing statistics.
4. Predefine energy/quantile windows or correct for searching across hundreds of bins.
5. Use bootstrap or graph-level standard errors; do not treat eigenvalues within one graph as independent graph realizations.
6. Do not call a single-(N) IPR spike a mobility edge.
7. Validate realized degree-tail matching and structural properties before interpreting model differences.

## 11. Outputs to inspect

### Affine PA full-spectrum outputs

Base directory:

`outputs/preferential_attachment/affine_full_spectrum_q0_1_width_0p002/`

Parameter directories:

- `n_5000_m_10_preference_tau_1p8_avgdeg_19p978/`
- `n_5000_m_10_preference_tau_1p5_avgdeg_19p978/`
- `n_5000_m_10_preference_tau_1p05_avgdeg_19p978/`

Most useful files in each directory:

- `run_manifest.json`: exact model, exponent convention, seeds, and numerical settings;
- `fragment_multiplicity_summary.csv`: components, degrees, zeros, and exact (\pm1) audit;
- `global_empirical_spectral_density.png`: global continuous density;
- `eigenvalue_spacing_histograms.png`: selected unfolded-spacing histograms;
- `eigenvalue_spacing_summary.csv` and `.png`: all-bin gap-ratio and GOE/Poisson comparisons;
- `eigenvalue_spacing_heatmap.png`: spacing-density evolution over the spectrum;
- `eigenvector_percentile_scan_summary.csv` and `.png`: IPR, participation fraction, top-r, and mass-radius curves;
- `eigenvector_coordinate_mass_heatmap.png`: spectral quantile versus cumulative coordinate mass;
- `empirical_density_cdf.npz`: density and CDF used for unfolding;
- `partials/`: graph-level/chunk intermediates used to build the summaries, not a separate scientific model.

### Sparse BA pilot outputs

Base directory:

`outputs/preferential_attachment/barabasi_albert/n_5000_m_10/`

Most useful files:

- `run_manifest.json`;
- `degree_spectral_tail_summary.csv`;
- `rank_match_summary.csv`;
- `spacing_localization_summary.csv`;
- `spacing_histograms.png`;
- the figures and tables under `outputs/preferential_attachment/comparisons/ba_vs_rewired/`;
- the unpaired tail comparison under `outputs/preferential_attachment/comparisons/ba_vs_chung_lu_tau_2p0/`.

The folder `n_400_m_4` is a smaller validation/pilot run: (N=400) vertices and (m=4) new edges per arriving vertex. It is not 400 graphs.

### Chung–Lu outputs

- Eigenvector scans: `outputs/chung_lu/power_law/eigenvectors/percentile_scan_q0p5_1_width_0p002/`.
- Baseline eigenvalue-spacing histograms and summaries: `outputs/chung_lu/power_law/tau_*_avgdeg_*/local_spacing_grid_quantiles.png` and `spacing_summary.csv`; each folder's `run_manifest.json` identifies the exact exponent convention and graph count.
- Exact (E=1) eigenvalue/eigenvector alignment: `outputs/chung_lu/power_law/eigenvalues/resonance_zoom_q0p60_0p72_width_0p002/`.
- Full graph/largest-connected-component comparison: `outputs/chung_lu/power_law/eigenvectors/e1_resonance_lcc_percentile_scan_q0p5_1_width_0p002/`.
- Basis-invariant (E=\pm1) projector/component audit: `outputs/chung_lu/power_law/eigenvectors/e1_projector_component_scan/`.
- Degree–spectral-tail pilot: `outputs/chung_lu/degree_spectral_tail_match/`.

## 12. Literature basis

The implemented model and interpretation follow these sources:

1. H. A. Simon, [*On a Class of Skew Distribution Functions*](https://doi.org/10.1093/biomet/42.3-4.425), *Biometrika* 42 (1955), 425–440 — cumulative advantage background.
2. D. de Solla Price, [*A General Theory of Bibliometric and Other Cumulative Advantage Processes*](https://doi.org/10.1002/asi.4630270505), *JASIS* 27 (1976), 292–306 — preferential accumulation with an initial-attractiveness idea.
3. A.-L. Barabási and R. Albert, [*Emergence of Scaling in Random Networks*](https://arxiv.org/abs/cond-mat/9910332), *Science* 286 (1999), 509–512 — growth plus degree-proportional attachment.
4. S. N. Dorogovtsev, J. F. F. Mendes, and A. N. Samukhin, [*Structure of Growing Networks with Preferential Linking*](https://arxiv.org/abs/cond-mat/0004434), *Physical Review Letters* 85 (2000), 4633–4636 — affine/initial-attractiveness attachment.
5. P. L. Krapivsky, S. Redner, and F. Leyvraz, [*Connectivity of Growing Random Networks*](https://arxiv.org/abs/cond-mat/0005139), *Physical Review Letters* 85 (2000), 4629–4632 — linear versus nonlinear attachment kernels.
6. B. Bollobás, O. Riordan, J. Spencer, and G. Tusnády, [*The Degree Sequence of a Scale-Free Random Graph Process*](https://doi.org/10.1002/rsa.1009), *Random Structures & Algorithms* 18 (2001), 279–290 — rigorous BA degree-sequence results.
7. K.-I. Goh, B. Kahng, and D. Kim, [*Spectra and Eigenvectors of Scale-Free Networks*](https://arxiv.org/abs/cond-mat/0103337), *Physical Review E* 64 (2001), 051903 — spectral tails and eigenvector concentration in scale-free networks.
8. Y. Y. Atas, E. Bogomolny, O. Giraud, and G. Roux, [*The Distribution of the Ratio of Consecutive Level Spacings in Random Matrix Ensembles*](https://arxiv.org/abs/1212.5611), *Physical Review Letters* 110 (2013), 084101 — adjacent-gap-ratio diagnostics.

## 13. Questions for the next reviewer

When using this report to design the next experiment, please answer these questions explicitly:

1. Is the target estimand the effect of growth history **conditional on the same degree sequence**, or the total difference between PA and Chung–Lu ensembles? These require different controls.
2. Which common exponent convention should be used, and how should realized rather than nominal tail matching be verified?
3. What (N)-grid and graph counts give enough power to estimate (D_2(E)) and a simultaneous GOE-to-Poisson crossover?
4. Should the next production run prioritize full-spectrum affine-PA rewiring, matched-degree Chung–Lu/configuration controls, or finite-size scaling?
5. How should inference account for searching over 500 spectral bins and for correlations between adjacent bins?
6. What diagnostics best distinguish a true localized phase from exact motif atoms, rank-deficiency zero modes, and finite-size edge concentration?

The recommended starting point is a paired full-spectrum comparison of each affine PA graph with its exact-degree rewiring, followed by an exponent-matched independent-edge/configuration control and then a multi-(N) scaling experiment.
