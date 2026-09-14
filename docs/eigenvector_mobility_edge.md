# Eigenvector localization and the mobility-edge scan

## What counts as an eigenvector mobility edge?

For an adjacency eigenpair

\[
A v_\alpha=\lambda_\alpha v_\alpha,
\qquad
\sum_{i=1}^{N}|v_\alpha(i)|^2=1,
\]

write \(p_i=|v_\alpha(i)|^2\). A mobility edge is a spectral location at which the scaling of the eigenvectors changes from extended to localized. At finite \(N\), one should look for a simultaneous crossover in several diagnostics rather than call any single jump a mobility edge:

1. \(\operatorname{IPR}=\sum_i p_i^2\) changes from order \(N^{-1}\) toward order one. Equivalently, \(N\operatorname{IPR}\), which is approximately 3 for a real Haar/GOE vector, begins to grow strongly with \(N\).
2. The participation fraction \(\operatorname{PR}/N=1/(N\operatorname{IPR})\) decreases toward zero.
3. The cumulative top-\(r\) mass \(C_r=\sum_{j=1}^{r}p_{(j)}\) rises, where \(p_{(1)}\geq p_{(2)}\geq\cdots\).
4. The mass radius \(r_x=\min\{r:C_r\geq x\}\) changes from order \(N\) toward order one or another sublinear scale.
5. In the conventional localized regime, nearby eigenvalue statistics also change from GOE-like level repulsion toward Poisson statistics. The eigenvalue and eigenvector crossovers need not coincide sharply at finite size.

A sharp bend across spectral quantile at one value of \(N\) is evidence of a finite-size crossover, not proof of an asymptotic mobility edge. The decisive test is size scaling. If

\[
\operatorname{IPR}_N(E)\asymp N^{-D_2(E)},
\]

then \(D_2=1\) is extended, \(0<D_2<1\) is nonergodic or multifractal extended, and \(D_2=0\) is localized.

The inequalities also connect the diagnostics.  If the top \(r\) coordinates
carry mass \(C_r=m\), Cauchy--Schwarz gives

\[
\operatorname{IPR}=\sum_i p_i^2
\geq \sum_{i=1}^{r}p_{(i)}^2
\geq \frac{m^2}{r}.
\]

Thus \(C_{200}=0.9\) alone implies
​\(\operatorname{IPR}\geq0.9^2/200=0.00405\),
​\(N\operatorname{IPR}\geq20.25\), and
​\(\operatorname{PR}\leq247\) at \(N=5000\).  This bound is conservative:
the measured tau 1.05 edge vectors have a mean effective PR of only about
32.7 coordinates.

## The existing extreme-tail eigenvector experiment

The saved result folders are:

- `outputs/chung_lu/power_law/eigenvectors/n_5000_tau_0p8_avgdeg_20p0`
- `outputs/chung_lu/power_law/eigenvectors/n_5000_tau_1p05_avgdeg_20p0`
- `outputs/chung_lu/power_law/eigenvectors/n_5000_tau_1p5_avgdeg_20p0`

Each uses \(N=5000\), nominal mean degree 20, 500 graph realizations, the full graph, a deterministic quantile weight sequence, and eight very narrow bins between quantiles 0.998 and 0.9999. These bins are useful for resolving variation *within the extreme tail*. They do not include a bulk reference and therefore cannot locate the onset of localization.

The endpoint summaries are:

| Displayed tau | Quantile | Mean energy | Mean \(N\)IPR | Mean PR/\(N\) | Top-10 mass | Top-200 mass |
|---:|---:|---:|---:|---:|---:|---:|
| 0.8 | 0.99810 | 17.062 | 29.28 | 0.03456 | 0.1656 | 0.7311 |
| 0.8 | 0.99975 | 18.303 | 32.37 | 0.03130 | 0.1763 | 0.7546 |
| 1.05 | 0.99810 | 9.130 | 111.00 | 0.00958 | 0.3605 | 0.8691 |
| 1.05 | 0.99975 | 11.002 | 160.43 | 0.00654 | 0.4361 | 0.9009 |
| 1.5 | 0.99810 | 15.569 | 30.50 | 0.03316 | 0.1623 | 0.7884 |
| 1.5 | 0.99975 | 16.909 | 34.71 | 0.02920 | 0.1796 | 0.8099 |

The first screenshot is therefore correctly read as a smooth progression, not a visible discrete jump between its ultra-fine bins. All three series are already strongly concentrated at the first displayed bin. In particular, for displayed tau 1.5 the top 200 coordinates carry about 78.8% to 81.0%, while those 200 coordinates are only 4% of 5000.

The tau 1.05 output is much more concentrated than the other two. It reaches about 90.1% mass in the top 200 coordinates and its mean participation ratio at the final bin is only

\[
5000\times0.00654\approx32.7
\]

effective coordinates.

The old and new q labels require one caution.  The legacy diagnostic ranked
the near-zero atom together with the non-Perron spectrum, whereas the new scan
first removes eigenpairs satisfying ​\(|\lambda|\leq10^{-8}\).  If the old
zero fraction is \(z\), an old upper-tail label approximately maps to
​\(q_{\rm continuous}=(q_{\rm old}-z)/(1-z)\).  This is why old labels such
as 0.998--0.9999 land in approximately the last few new continuous-spectrum
bins rather than defining a separate discrepancy.

## Why the broad-bin plot appears to jump

The broad experiment spans the bulk and the edge, so it can reveal a bulk-to-tail crossover that the 0.998--0.9999 zoom cannot show. This is not a contradiction:

- broad bins answer “where does concentration begin to change?”;
- the existing fine bins answer “how does an already concentrated extreme tail continue to change?”

The saved broad plot is also not a direct one-for-one comparison with all three \(N=5000\) tail runs: its manifest uses a smaller local experiment and broad, unequal bins. The new scan below keeps \(N=5000\), 500 graphs, and equal bin widths throughout.

## How large is top-200 mass at N=5000?

A perfectly flat vector has \(C_{200}=200/5000=0.04\), but this is too weak a null model because sorting the coordinates of a random delocalized vector creates large order statistics. For a real Haar/GOE vector, let \(X=Z^2\) with \(Z\sim N(0,1)\). If the largest fraction \(p=r/N\) of coordinates is retained and \(a=\Phi^{-1}(1-p/2)\), the asymptotic retained mass is

\[
C_r^{\mathrm{GOE}}
\approx
\mathbb E[X\mathbf 1\{X>a^2\}]
=2\{a\phi(a)+1-\Phi(a)\}.
\]

At \(N=5000\), this gives:

| Metric | GOE/Haar benchmark |
|---|---:|
| Top-10 mass | 0.0228 |
| Top-50 mass | 0.0845 |
| Top-100 mass | 0.1440 |
| Top-200 mass | 0.2389 |
| Coordinates for 50% mass | about 620 |
| Coordinates for 80% mass | about 1,580 |
| Coordinates for 90% mass | about 2,223 |
| Coordinates for 95% mass | about 2,765 |

Consequently, top-200 masses of 0.73--0.90 are far above both the flat 0.04 baseline and the correct sorted-Haar value of about 0.239. At this system size it is accurate to call these eigenvectors strongly concentrated or localized-looking. It is not yet accurate to claim asymptotic localization without repeating the experiment over several \(N\).

## Should localization be normalized by the number of nonzero coordinates?

The literal number of nonzero coordinates is not a reliable localization measure.

1. A generic eigenvector on a connected component has almost no coordinates that are exactly zero, even when most of its mass is carried by a small set.
2. “Nonzero” becomes dependent on an arbitrary floating-point threshold.
3. For a disconnected graph, exact zeros may only reveal that the eigenvector belongs to one block/component. That is geometric support, not the same as amplitude localization within that component.
4. Degenerate eigenspaces can be rotated, changing individual coordinate supports and IPRs while leaving the eigenspace unchanged.

It is still useful to report the numerical nonzero fraction as a structural diagnostic, but the decision should be based primarily on IPR/PR, entropy support, top-\(r\) mass, and the mass radii \(r_{50},r_{80},r_{90},r_{95}\). The new scan reports all of these, so top 200 can be compared with both the total \(N=5000\) and the actual number of coordinates required to reach a prescribed mass.

## New equal-percentile scan

The new experiment uses 250 bins

\[
[0.500,0.502),[0.502,0.504),\ldots,[0.998,1.000].
\]

The width \(\Delta q=0.002\) is fine enough to localize a crossover while retaining several eigenvectors per graph in typical continuous spectra. The much smaller width 0.0002 used in the old extreme-tail scan would often contain only one or two eigenvectors per graph and is unnecessarily noisy for a 0.5-wide scan.

For consistency with the legacy diagnostics, the discrete rank selector uses
`floor(q_low M)` and `ceil(q_high M)` for a spectrum of size \(M\).  Thus the
percentile intervals have exactly equal width, but adjacent intervals can
share one boundary eigenvector when an endpoint falls between integer ranks.
This induces correlation between neighboring plotted points, not the broad
trend.  A future strictly disjoint-bin version should assign eigenvectors by
their rank midpoints.

For every graph, the pipeline:

1. rebuilds the legacy expected-weight sequence;
2. samples the Chung--Lu graph using the same graph seed as its eigenvalue counterpart;
3. computes the full adjacency eigendecomposition;
4. removes the Perron eigenpair;
5. removes exact-zero eigenvalues and the matching eigenvector columns;
6. defines quantiles within this continuous non-Perron spectrum;
7. computes graph-level localization metrics in each bin;
8. aggregates with graph-equal means and graph-level standard errors.

The legacy exponent conventions are deliberately preserved:

| Displayed tau | Convention in the source eigenvalue run | Density exponent passed to generator | Cutoff coefficient | Graph-seed base |
|---:|---|---:|---:|---:|
| 0.8 | CCDF exponent | 1.8 | 0.999 | 27282 |
| 1.05 | density exponent | 1.05 | 1.0 | 27307 |
| 1.5 | density exponent | 1.5 | 1.0 | 27352 |

This mixed legacy mapping is recorded in every new manifest. It is necessary to reproduce the weight sequences behind the saved eigenvalue folders rather than imposing a new convention after the fact.

## Results of the 250-bin scan

The following table compares a representative interior bin, an operational
edge-crossover bin, and the last bin.  The reported quantile is the bin
midpoint.  `NIPR` means \(N\sum_i|v_i|^4\), and \(r_{90}\) is the smallest
number of largest coordinates carrying 90% of the mass.

| Displayed tau | q midpoint | Mean energy | Mean NIPR | Mean PR/N | Entropy support/N | Top-10 mass | Top-200 mass | Mean r90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8 | 0.799 | 1.860 | 5.65 | 0.1790 | 0.3223 | 0.0489 | 0.3622 | 1,645 |
| 0.8 | 0.927 | 5.047 | 8.54 | 0.1177 | 0.2458 | -- | 0.4591 | 1,408 |
| 0.8 | 0.999 | 17.716 | 30.93 | 0.03276 | 0.08651 | 0.1712 | 0.7446 | 651 |
| 1.05 | 0.799 | 3.263 | 14.30 | 0.07121 | 0.1754 | 0.1103 | 0.5272 | 1,083 |
| 1.05 | 0.923 | 4.307 | 23.12 | 0.04507 | 0.1307 | -- | 0.6007 | 881 |
| 1.05 | 0.999 | 11.002 | 160.43 | 0.006539 | 0.02497 | 0.4361 | 0.9009 | 199 |
| 1.5 | 0.799 | 1.947 | 10.20 | 0.1007 | 0.2042 | 0.0782 | 0.4911 | 1,123 |
| 1.5 | 0.901 | 5.612 | 11.61 | 0.08658 | 0.1771 | -- | 0.5462 | 1,023 |
| 1.5 | 0.999 | 16.544 | 33.85 | 0.02995 | 0.06570 | 0.1764 | 0.8050 | 363 |

Every row, including the interior rows, is more concentrated than the
Haar/GOE vector baseline \(N\operatorname{IPR}\simeq3\).  The data therefore
show a *smooth strengthening of localization toward the edge*, rather than a
clean transition from fully ergodic Haar vectors to localized vectors.

A descriptive two-segment regression was fitted separately to log-NIPR,
log-entropy support, and top-200 mass for \(q\geq0.8\).  For every candidate
bin midpoint \(q_c\), the fit is
\(y(q)=a+bq+c(q-q_c)_+\); the reported bend minimizes residual sum of
squares over \(q_c\).  It places the strongest change in slope in these
ranges:

| Displayed tau | Operational q range | Approximate energy range |
|---:|---:|---:|
| 0.8 | 0.927--0.939 | 5.05--5.92 |
| 1.05 | 0.915--0.931 | 4.20--4.43 |
| 1.5 | 0.897--0.905 | 5.32--5.91 |

These are operational crossover locations, not confidence intervals and not
proof of asymptotic mobility edges.  A finite-size scaling study is required
to decide whether \(r_{90}\) becomes \(O(1)\), sublinear, or proportional to
\(N\).

## The narrow motif resonances are not the mobility edge

The fine scan also exposes sharp, non-monotone spikes close to
​\(E=1\):

| Displayed tau | q near spike | Mean energy | Peak mean NIPR |
|---:|---:|---:|---:|
| 0.8 | 0.685 | 0.998 | 127 |
| 1.05 | 0.633 | 1.005 | 1,751 |
| 1.5 | 0.683 | 1.000 | 1,205 |

There are smaller features near \(E=\sqrt 2\).  Their narrowness in energy,
non-monotonicity in quantile, and proximity to exact small-tree eigenvalues
identify them as motif resonances.  An audit of representative graphs found
14 isolated two-vertex components for displayed tau 1.05 and six for tau
1.5; each isolated edge supplies an exact \(+1\) mode.  The remaining
multiplicity is consistent with repeated pendant-neighborhood structures
inside the giant component.  Because an eigensolver can choose an arbitrary
basis within a degenerate eigenspace, the IPR of an individual exactly
degenerate eigenvector is basis-dependent.  Those spikes must therefore be
reported separately from the broad high-energy crossover.

## Comparison with the eigenvalue-spacing evidence

The paired spacing/IPR experiment available for displayed tau 1.5 remains
GOE-like well past the eigenvector crossover.  Around \(q=0.94\), it reports
mean gap ratio about 0.531 and a much smaller KS distance to GOE (0.011) than
to Poisson (0.209).  Even at \(q=0.9965\), its mean gap ratio is about 0.535
and its GOE distance remains much smaller than its Poisson distance, while
the NIPR has risen to about 31.

Thus the eigenvector and eigenvalue results are related but are **not a
textbook-matched mobility-edge signal** in this finite system.  The
eigenvectors become strongly concentrated and prelocalized toward the upper
edge, whereas the unfolded local spacings remain substantially GOE-like.
The plausible interpretations are finite-size prelocalization, a
nonergodic-extended regime, or hub/motif-driven concentration.  Size scaling
of both diagnostics in the same bins is the discriminating next experiment.

## Reproducibility artifacts

The implementation is in
`src/small_world_mobility_edge/chung_lu_eigenvector_percentile_scan.py`.
The Hoffman chunk, streaming aggregation, and submission entry points are in
`tools/hoffman_chung_lu_eigenvector_percentile_chunk.py`,
`tools/hoffman_chung_lu_eigenvector_percentile_aggregate.py`, and
`tools/hoffman_submit_chung_lu_eigenvector_percentile.sh`.

Each result directory contains the 500-graph table, 250-row summary, coordinate
decay table, two figures, and a manifest recording the exact seed and weight
conventions:

- `outputs/chung_lu/power_law/eigenvectors/percentile_scan_q0p5_1_width_0p002/n_5000_tau_0p8_avgdeg_20p0`
- `outputs/chung_lu/power_law/eigenvectors/percentile_scan_q0p5_1_width_0p002/n_5000_tau_1p05_avgdeg_20p0`
- `outputs/chung_lu/power_law/eigenvectors/percentile_scan_q0p5_1_width_0p002/n_5000_tau_1p5_avgdeg_20p0`
