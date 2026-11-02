# Publication audit — November 2, 2026

Status: prepared and verified locally; GitHub publication is pending write access.

## What was missing from GitHub

The public `main` resolved to `00a2c25`. The local publication branch was
three commits ahead (`54838f2`, `67f3f56`, `1b0a53e`) with no divergent
public commits. A complete-looking local README therefore did not describe
what a GitHub visitor could access.

| Scientific content | Before this update | Included in the update |
|---|---|---|
| Broad ER / Chung–Lu / fixed-degree / PA workflows | Much of the suite existed only in the unpushed release | Generators, tests, settings, aggregate pipelines, and report bundles |
| Single-graph geometry | Main PA workflow prepared locally; October follow-up scripts outside release | Exact-level/bin audit, graph shells, induced subgraphs, signed-connector diagnostics, figures and tables |
| Hub interactions | Local-mode/coupling code already prepared but unpublished | Implementation plus formulas, pair identity, overlap distinction, and recorded acceptance outcome |
| Multifractality | Missing from the prepared release as well as public main | Core module, tests, frozen grid, portable SGE workflow, real-graph smoke, all 14 compact case outputs |
| PA diagonal noise | Executed snapshot recovered in unpushed release | Raw-amplitude design, 112-operator results, reporting code, reproduction path |
| Replicated Anderson / regular benchmark | Implemented locally but unpublished | Nested graph/potential design, distinct amplitude/window conventions, 189/99-operator results and code |

The review used local source/configurations, immutable executed code where
available, completed experiment tables, the latest project report, and the
project discussions about individual levels, eigenvector signs, repeated
subspaces, and model comparisons. Private chat transcripts are not published.

## Mathematical corrections carried into the current pages

- Histogram crowding is separated from eigenvalue multiplicity.
- A single graph's geometry is separated from the ensemble reference CDF.
- Top-mass induced edges are separated from distances in the full graph.
- Signed amplitudes, not squared masses, cancel at connectors.
- Same-hub positive/negative mass overlap is separated from overlap of two
  neighboring positive modes on different hubs.
- The local-mode study's one accepted multimode description is not relabeled
  as a successful two-mode tunneling explanation.
- The multifractality papers' bulk singularity and their numerical protocol
  are separated from the PA finite-size observations. Degenerate-window
  flags, changing spectral-window widths, and single-graph replication are
  stated next to the curves they affect.
- Raw a in the final noise sweep is separated from a/√20 in the replicated
  PA extension and W=2a in the regular Anderson convention.

## Publication changes versus scientific reruns

Imported mathematical calculations and recorded numerical tables retain the
executed definitions. Public-path changes remove machine-local roots and
expose explicit CLI inputs. New checks reject mixed multifractality case
identities and invalid selected vectors. The aggregate report now describes
window drift accurately and reports configured rather than hard-coded smoke
settings. All original research outputs remain untouched.

No Hoffman job or new production ensemble was launched for this update.
Bounded smoke tests and regeneration of aggregate curves from committed
tables validate the public execution paths. See the updated
[validation record](release_validation.md) for commands and outcomes.

## Publication access check

The terminal's HTTPS Git transport has no configured login. The connected
GitHub integration can read this repository but a Git-data write returned
HTTP 403, “Resource not accessible by integration.” No remote branch or file
was changed. Publication requires authenticated Git or a connection with
repository-content write permission; credentials should not be pasted into
the project or chat.
