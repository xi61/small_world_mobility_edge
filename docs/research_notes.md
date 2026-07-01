# Research Notes

## Main question

Is there a mobility edge in the spectrum of small-world random graph models?

## Candidate graph models

1. Watts-Strogatz small-world graphs.
2. Chung-Lu graphs with heavy-tailed expected degrees.
3. Configuration models with heavy-tailed degree sequences.
4. Preferential-attachment graphs.
5. Random `d`-regular graphs as a control.

## Operational definition for numerics

For a finite graph, call an energy window more localized when:

- the mean IPR is larger
- the mean participation ratio is smaller
- the mean adjacent-gap ratio is closer to the Poisson benchmark `~0.386`

Call an energy window more delocalized when:

- the mean IPR is smaller
- the mean participation ratio is larger
- the mean adjacent-gap ratio is closer to the GOE benchmark `~0.531`

## Practical cautions

- A finite-size crossover is not yet an asymptotic mobility edge.
- Heavy-tailed degree fluctuations can create localized states near hubs.
- Small-world rewiring may create satellite bands or outlier states rather than a clean single transition.
- Preferential-attachment graphs are not exchangeable; degree-age correlations may matter.

## Suggested first experiments

1. Watts-Strogatz:
   sweep rewiring probability `p` at fixed `n, k`.
2. Random regular:
   use as baseline for GOE-like spacing and delocalized eigenvectors.
3. Chung-Lu:
   compare light-tail versus heavy-tail expected degrees.
4. Configuration model:
   check whether the spectral edges localize earlier than the bulk.
5. Preferential attachment:
   inspect whether hub-driven top eigenvectors dominate apparent localization.

## What would count as evidence

The strongest numerical evidence would be:

- stable energy-window separation between GOE-like and Poisson-like spacing
- matching localization crossover in IPR
- the crossover sharpening as `n` increases

That would justify a more serious asymptotic conjecture.
