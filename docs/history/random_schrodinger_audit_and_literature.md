# Random-Schrödinger extension: audit and source-to-diagnostic note

Date: 2026-09-21.  This note separates existing project facts, published
results, and the numerical extensions introduced by this run.

## Repository and Hoffman audit

The user-selected workspace resolves to
`<SOURCE_WORKSPACE>`.  It is a
non-Git project copy.  The separately referenced Git checkout at
`<PUBLICATION_CHECKOUT>` is clean at commit `67f3f56`
and two commits ahead of `origin/main`; the active copy has the same scientific
implementation with older hard-coded temporary-directory defaults plus
additional reports/results.  New manifests therefore record a deterministic
source-tree SHA-256 when no Git commit is available.

No `AGENTS.md` exists in the active tree.  The repository name and README use
“small world” descriptively.  Searches of `src/`, `tools/`, `experiments/`, and
`docs/` found PA, Chung–Lu, Erdős–Rényi, random-regular, Holme–Kim, and
degree-preserving rewiring code, but no established Watts–Strogatz or other
distinct SW generator.  This extension therefore does not invent one.

### Existing reusable PA result

`src/small_world_mobility_edge/preferential_attachment/single_graph_edge_bulk.py`
created the exact prior induced-subgraph reports.  Hoffman job `14612758`
completed three `N=5000, m=10` graph tasks under
`<HOFFMAN_SCRATCH>/small_world_mobility_edge_single_graph_20260901/outputs`.
The synchronized copies are below
`outputs/preferential_attachment/single_graph_edge_bulk/`.

Each case contains:

- `tables/graph_realization.npz`: exact edges, degree array, graph seed, `N`,
  and `m`; labels and eigenvector coordinates are both `0,...,N-1`;
- `tables/selected_eigenvectors.npz`: the edge-50 and bulk-50 eigenvectors,
  their full-spectrum indices, and eigenvalues;
- `spectrum/single_graph_spectral_arrays.npz`: algebraically sorted full
  eigenvalues, residual-audited IPR, `N*IPR`, raw gaps, ratios, and rank
  selections;
- `run_manifest.json`: generator and selection provenance.

The old Stage-1 code already computed top-10 mass, induced-subgraph statistics,
full-graph mean/median distances, and a uniform random-vertex baseline.  It did
not store the 10-by-10 distance matrix or 45 pair records, degree-matched
controls, component/core annotations, center/shell geometry, non-Perron top-10
as a separate selection, or adjacent-eigenvector cross-geometry.

The three graphs are connected simple graphs with exactly 49,945 edges and
mean degree 19.978.  Their generator definitions are in
`src/small_world_mobility_edge/preferential_attachment/baseline.py`:

- affine PA attaches `m` distinct targets without replacement with weight
  `degree + delta`, where `delta=m(alpha-2)` and `alpha` is the asymptotic
  degree-CCDF exponent; the seed graph is `K_(m+1)`;
- nonlinear PA attaches with weight `degree**zeta`; `zeta` is a kernel
  exponent, not a degree-tail exponent;
- all PA outputs here are loop-free, simple, connected, and have minimum
  degree 10.

The historical operator is unscaled adjacency `B=A` with no regularizer.  The
old gap pipeline excludes the Perron eigenvalue for its non-Perron spectral
statistics and removes the audited zero atom without bridging it.  This
extension preserves those labels while adding explicitly named diagnostics.

### Existing Chung–Lu and control conventions

`src/small_world_mobility_edge/chung_lu_baseline.py` uses NetworkX's
independent-edge expected-degree graph with `selfloops=False`.  It can use
quantile or sampled Pareto weights, solves a clipped-mean equation, and checks
`max(w)^2 <= sum(w)`.  Its `tau` is a weight-density exponent: the weight CCDF
exponent is `tau-1`.  A PA degree-CCDF exponent `alpha` is therefore nominally
matched by Chung–Lu `tau=alpha+1`, subject to a separately reported finite-size
cutoff and realized-tail audit.  The model can be disconnected and can contain
isolates, leaves, and small components.  Existing degree-preserving rewiring is
a finite double-edge-swap chain on a simple graph and is not relabeled as
configuration sampling.

### Hoffman execution facts

The validated environment remains SGE/Grid Engine with account/project
`campus_prj`, module `python/3.9.6`, NumPy 1.22.2, SciPy 1.7.1, NetworkX 2.6.2,
and Matplotlib 3.4.3.  Existing jobs request `h_rt`, `h_data`, and the `shared`
parallel environment.  There were no active jobs for this account at the audit
time.  The protected historical cache remains
`<HOFFMAN_HOME>/small_world_mobility_edge_cache/pa_full_20260813`; new work
writes below `<HOFFMAN_SCRATCH>/random_schrodinger_20260921`.

## Literature-to-method mapping

### Alt, Ducatez, Knowles (2021)

Source: [Delocalization Transition for Critical Erdős–Rényi
Graphs](https://link.springer.com/article/10.1007/s00220-021-04167-y).

Published facts used here:

- Section 1.1, especially (1.1)--(1.3), treats eigenvector mass as `|v_x|^2`,
  describes localization around graph-distance-separated resonant vertices,
  and states the heuristic comparison between exponentially small tunneling
  and local level spacing.  It does not prescribe a finite-sample tunneling
  estimator for arbitrary graph ensembles.
- Section 1.5 and Definition A.17/Lemma A.18 construct a star tuning fork: two
  degree-`D` stars joined to a common base.  The exact modes at `+/-sqrt(D)` are
  supported on the two stars.  These are validation fixtures, not a claim that
  PA edge states have the same structure.
- Appendix A.1 gives the radial `(p,q)`-tree profile and distinguishes
  component decay from shell population.
- Appendix A.1/Figure 10 reports one qualitative numerical illustration:
  `G(N,d/N)`, `N=10,000`, `d=0.6 log N`, restricted to the giant component,
  using `A/sqrt(d)`, and plotting every eigenvalue against the associated
  eigenvector infinity norm.  The Perron point is retained in that plot.

Figure 10 does not report an RNG seed, repetition count, software, eigensolver,
or simulation algorithm.  Our reproduction records its own seed and solver and
targets the reported qualitative scatter, not pixel-identical data.  A partial
spectrum is never labeled as the Figure-10 full-spectrum replication.

### Alt, Ducatez, Knowles (2024)

Source: [Localized phase for the Erdős–Rényi
graph](https://arxiv.org/html/2305.16294v2).

Published facts used here:

- Section 1.1, (1.2)--(1.3), defines a tunneling-error norm for approximate
  localized states and compares it with the separation of their approximate
  energies.  The discussion explicitly warns that local truncated profiles
  can be too inaccurate.
- The paper sets `H=A/sqrt(d)` and normalized degree
  `alpha_x=D_x/d`; (1.4) defines `Lambda(alpha)=alpha/sqrt(alpha-1)`.
- Definition 2.1 uses
  `lambda(x)=lambda_2(H^(V\\{x}))`: all competing vertices in `V` are removed,
  the second eigenpair (after the Perron mode) is computed, and its eigenvector
  is embedded in the original vertex space.  Section 2.1 defines
  `M^(X)=M|_(X^c)`, so removed coordinates have zero rows and columns.
- Proposition 2.3 controls both approximate-energy error and
  `||(H-lambda(x))u(x)||`; Proposition 2.4 lower-bounds separations between the
  approximate energies.  The approximate states are nonorthogonal, and that
  nonorthogonality must be measured rather than assumed away.

The paper is a theorem/proof, not a published Monte Carlo protocol.  It gives
no simulation seed, repetition count, or general-purpose estimator for PA or
Chung–Lu.  This project implements the ER construction with its exact center
and removal conventions; any use of analogous centers on PA/CL is labeled
exploratory.  Separately, our local-ball projection and Schur-complement
experiments are new numerical diagnostics, not paper replications.

### Castellano and Pastor-Satorras (2017)

Source: [Relating Topological Determinants of Complex Networks to Their
Spectral Properties](https://arxiv.org/abs/1703.10438).

The paper compares the principal adjacency eigenpair with the isolated
hub-neighborhood and maximum-core subgraphs and discusses qualitative
differences between linear PA and static heterogeneous networks.  We use this
only to motivate recording full degrees, core numbers, hub neighborhoods, and
shell masses.  We do not promote its principal-eigenvector conclusions to a
theorem about all non-Perron edge eigenvectors.

### Pastor-Satorras and Castellano (2015)

Source: [Distinct types of eigenvector localization in
networks](https://arxiv.org/abs/1505.06024).

This paper distinguishes hub localization from maximum-core localization of
the principal eigenvector.  It motivates the same hub/core annotations and
the need not to call every high-mass leaf a separate center.  Again, its scope
is the principal eigenvector, not the complete upper-edge spectrum.

## New diagnostic definitions

Stage 1 uses unweighted BFS distances in the complete analyzed graph.  Paths
may leave the selected set; unreachable pairs remain infinity and receive an
explicit flag.  Candidate centers are local mass maxima among the deterministic
top-50 mass vertices.  Candidates are sorted by radius-`r` ball mass, vertex
mass, and degree (descending, then vertex ID ascending), greedily separated by
a graph-distance threshold, and retained until their ball union captures 80%
of total mass or four centers are reached.  The primary rule uses radius two
and separation three; a radius/separation grid is saved.  Failure to reach the
mass target is labeled unresolved.

Our effective-coupling experiment uses independently diagonalized induced
balls with a Dirichlet boundary.  For nonorthogonal columns `U`, it stores
`S=U*U`, `K=U*HU`, the Gram condition number, and a symmetrically
orthogonalized representation.  For an orthonormal two-state representation it
reports direct coupling, uncoupled detuning, predicted two-state splitting,
and the observed full-operator adjacent gap as distinct quantities.  The
mediated diagnostic uses the resolvent identity
`H_eff(z)=zI-[P*(zI-H)^(-1)P]^(-1)`, algebraically equivalent to the stated
Schur complement for orthonormal `P`, and performs sparse linear solves rather
than a dense inverse.

Stage 3 retains `B=A` and implements exactly
`H(a)=B+a*diag(xi)`, `xi~Uniform[-1,1)`, using NumPy's documented half-open RNG
interval.  With scalar normalization `c=1`, operator-unit and raw-adjacency
half-widths are both `a`.  The pilot freezes `s_ref=sqrt(20)` from the
model-prescribed mean degree `2m`, rather than letting the scale drift with
each realization, and reports each realization's
`s_B=sqrt(trace(B^2)/N_eff)=sqrt(realized mean degree)` (19.978 for the three
cached graphs).
