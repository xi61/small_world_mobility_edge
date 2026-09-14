# Eigenvalue/eigenvector alignment at the E=1 resonance

## Question

The eigenvector percentile scan shows an extremely narrow localization band
near continuous non-Perron spectral quantile 0.68 and energy 1.  This follow-up
tests whether the eigenvalue distribution changes in the same window.  It is
an additive experiment: its outputs live under
`outputs/chung_lu/power_law/eigenvalues/resonance_zoom_q0p60_0p72_width_0p002`
and do not replace any existing eigenvalue result.

## Matched design

The scan uses the same 500 graphs, graph-seed streams, deterministic expected
weights, mixed legacy tau conventions, structural cutoffs, full-graph mode,
Perron removal, and exact-zero removal as the eigenvector scan.  It uses 60
equal percentile intervals of width 0.002 from q=0.60 through q=0.72.  This
window contains all three observed E=1 eigenvector peaks: approximately q=0.685
for displayed tau 0.8, q=0.633 for tau 1.05, and q=0.683 for tau 1.5.

## Why ordinary unfolded spacings are insufficient here

An exact spectral atom produces repeated eigenvalues and hence zero gaps.  If
one first deletes zero gaps and normalizes only the remaining gaps, the primary
signature of the atom disappears.  The scan therefore reports three layers:

1. the fraction of selected eigenvalues satisfying |lambda-1| <= 1e-8;
2. the fraction of raw adjacent gaps satisfying |Delta lambda| <= 1e-10 and
   an inclusive adjacent-gap ratio in which a zero/nonzero neighboring pair
   contributes zero;
3. conventional gap ratios and GOE/Poisson KS distances conditional on both
   gaps, or the unfolded spacing, being positive.

The third layer describes the continuous spectrum surrounding the atom.  It
must not be used to argue that the atom itself is GOE or Poisson.

## Alignment criterion

For each displayed tau, the experiment compares:

- the bin maximizing the E=1 atom fraction;
- the bin maximizing the degenerate-gap fraction;
- the bin maximizing eigenvector N times IPR in the saved matched scan.

Coincidence within one bin establishes peak alignment at resolution 0.002.
It establishes a localized resonance band, not by itself an asymptotic
mobility edge.  A mobility-edge claim instead requires a sustained regime
change and size scaling.

## Results

All three 500-graph aggregates completed.  The eigenvalue-atom maximum and
the eigenvector mean-NIPR maximum fall in exactly the same 0.002-wide bin for
all three displayed tau values:

| Displayed tau | Eigenvalue-atom peak q | Eigenvector-NIPR peak q | Mean energy | Atom fraction in peak bin | Degenerate-gap peak q | Peak NIPR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.8 | 0.685 | 0.685 | 0.99753 | 0.0409 | 0.685 | 127.4 |
| 1.05 | 0.633 | 0.633 | 1.00485 | 0.7842 | 0.635 | 1,750.8 |
| 1.5 | 0.683 | 0.683 | 1.00019 | 0.5031 | 0.683 | 1,204.6 |

The one-bin displacement of the degenerate-gap maximum for displayed tau
1.05 is expected because an eigenvalue of multiplicity \(m\) contributes
\(m\) atom eigenvalues but \(m-1\) internal zero gaps, and graph-to-graph rank
variation moves boundary gaps between adjacent bins.

The alignment is not merely equality of the maximum bin.  The
full-width-at-half-maximum intervals are:

| Displayed tau | E=1 atom | Degenerate gaps | Eigenvector NIPR |
|---:|---:|---:|---:|
| 0.8 | 0.685--0.687 | 0.685--0.687 | 0.685--0.687 |
| 1.05 | 0.623--0.643 | 0.623--0.643 | 0.621--0.643 |
| 1.5 | 0.681--0.685 | 0.681--0.685 | 0.681--0.685 |

Across all 60 bins, Pearson correlations between the atom fraction and NIPR
are 0.986, 0.989, and 0.998 for displayed tau 0.8, 1.05, and 1.5.  The
corresponding degenerate-gap/NIPR correlations are 0.976, 0.988, and 0.998.
This establishes that the eigenvalue and eigenvector resonance profiles are
the same feature at the current resolution.

The atom strength varies substantially with tau.  Across the full continuous
non-Perron spectrum, a graph contains on average 0.732, 16.018, and 7.516
exact \(+1\) eigenvalues for displayed tau 0.8, 1.05, and 1.5.  In the
respective peak bins, the pooled exact-\(+1\) counts are 180/4,399,
1,083/1,381, and 1,303/2,590 selected eigenvalues.

## Spacing interpretation

At the aligned peak:

| Displayed tau | Inclusive gap ratio | Positive/positive gap ratio | KS to GOE | KS to Poisson | Positive gaps | Regular ratios |
|---:|---:|---:|---:|---:|---:|---:|
| 0.8 | 0.5040 | 0.5117 | 0.0205 | 0.2035 | 3,856 | 3,336 |
| 1.05 | 0.3965 | 0.4933 | 0.1569 | 0.2880 | 217 | 86 |
| 1.5 | 0.3623 | 0.5390 | 0.0483 | 0.2421 | 1,157 | 775 |

The inclusive statistic falls because every zero/nonzero neighboring-gap pair
has ratio zero.  In particular, displayed tau 1.5 falls below the Poisson
benchmark even though, after conditioning on two positive gaps, its ratio is
0.539, essentially the GOE value 0.5359.  Its positive-spacing distribution is
also much closer to GOE than Poisson.  Displayed tau 0.8 gives the same
qualitative conclusion.  Displayed tau 1.05 leaves only 86 regular ratios at
the peak and is flagged as underpowered for that statistic; even there, the
conditional positive spacings are closer to GOE than Poisson.

Therefore the aligned eigenvalue threshold is an **exact spectral
atom/degeneracy threshold**, not a conventional transition from GOE level
repulsion to Poisson spacings.  Together with the falling localization on the
far side of the band, this supports the localized-motif-resonance
interpretation.  It does not turn the band into a one-way mobility edge.

## Reproducibility artifacts

The three additive output folders are:

- `outputs/chung_lu/power_law/eigenvalues/resonance_zoom_q0p60_0p72_width_0p002/n_5000_tau_0p8_avgdeg_20p0`
- `outputs/chung_lu/power_law/eigenvalues/resonance_zoom_q0p60_0p72_width_0p002/n_5000_tau_1p05_avgdeg_20p0`
- `outputs/chung_lu/power_law/eigenvalues/resonance_zoom_q0p60_0p72_width_0p002/n_5000_tau_1p5_avgdeg_20p0`

Each contains the 500-graph table, 60-bin eigenvalue summary, paired
eigenvalue/eigenvector table, alignment figure, peak-alignment JSON, and
manifest.  The implementation is
`src/small_world_mobility_edge/chung_lu_eigenvalue_resonance_scan.py`.
