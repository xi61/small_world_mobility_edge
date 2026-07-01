# Literature Map

This file records the papers already identified for the mobility-edge project, starting from the two papers you listed.

## Papers you explicitly mentioned

1. D. Jakobson, S. D. Miller, I. Rivin, Z. Rudnick, *Eigenvalue spacings for regular graphs*.
   Link: https://arxiv.org/abs/hep-th/0310002
   Use here: numerical evidence that generic `k`-regular graphs approach GOE level-spacing behavior.

2. A. Aggarwal, C. Bordenave, P. Lopatto, *Mobility Edge for Lévy Matrices*.
   Link: https://arxiv.org/abs/2210.09458
   Use here: a rigorous mobility-edge result in a heavy-tailed matrix model; good template for what a proof or numerical phase diagram should look like.

## Closely related papers for sparse-graph localization and Poisson statistics

3. J. Alt, R. Ducatez, A. Knowles, *Poisson statistics and localization at the spectral edge of sparse Erdős-Rényi graphs*.
   Link: https://arxiv.org/abs/2106.12519
   Use here: direct evidence that sparse graphs can have localized edge states and Poisson eigenvalue statistics.

4. Y. He, A. Knowles, M. Marcozzi, *Local law and complete eigenvector delocalization for supercritical Erdős-Rényi graphs*.
   Link: https://arxiv.org/abs/1808.09437
   Use here: bulk and supercritical sparse-graph delocalization benchmark.

5. P. Bourgade, J. Huang, H.-T. Yau, *Eigenvector Statistics of Sparse Random Matrices*.
   Link: https://arxiv.org/abs/1609.09022
   Use here: asymptotically Gaussian bulk eigenvectors for sparse matrices under degree growth assumptions.

6. I. Dumitriu, S. Pal, *Sparse regular random graphs: Spectral density and eigenvectors*.
   Link: https://arxiv.org/abs/0910.5306
   Use here: random-regular control ensemble; delocalization guidance.

## Small-world specific papers

7. P. Nakkirt, *The Eigenvalue Distribution of the Watt-Strogatz Random Graph*.
   Link: https://arxiv.org/abs/2009.00332
   Use here: moments and limiting eigenvalue-distribution information for Watts-Strogatz graphs.

8. A. Mishra, J. N. Bandyopadhyay, S. Jalan, *Multifractal analysis of eigenvectors of smallworld networks*.
   Link: https://arxiv.org/abs/2010.09024
   Use here: numerical localization and multifractality evidence in small-world adjacency spectra.

9. R. Kuehn, J. M. van Mourik, *Spectra of Modular and Small-World Matrices*.
   Link: https://arxiv.org/abs/1012.0529
   Use here: cavity-method style analysis, localized satellite bands, and localized states in band gaps.

10. R. Monasson, *Diffusion, localization and dispersion relations on "small-world" lattices*.
    Link: https://arxiv.org/abs/cond-mat/9903347
    Use here: early spectral and localization analysis for small-world Laplacians.

## Why this set is enough to start coding

Together, these papers give the minimum viable research frame:

- random regular graphs as a GOE-like control
- sparse random graphs with proven localized edge states and Poisson statistics
- heavy-tailed random matrices with a rigorous mobility edge
- small-world graph papers showing localization-sensitive spectral structure

That is enough to design experiments which ask whether a small-world graph family shows:

- delocalized bulk with GOE-type spacing
- localized tails or satellite bands with Poisson-type spacing
- a stable crossover energy that sharpens with system size

## Initial numerical question

For each graph family and each system size `n`, scan energy windows across the adjacency spectrum and record:

- window center
- mean adjacent-gap ratio
- mean IPR
- fraction of large-IPR eigenvectors

The practical mobility-edge candidate is the energy where the windowed observables cross from GOE-like and delocalized to Poisson-like and localized.
