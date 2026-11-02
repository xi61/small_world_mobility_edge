# Provenance and preservation

The publication tree was assembled without modifying the original research
workspace or the existing public checkout.

## Source tiers

1. **Public baseline and prepared local release.** At the November 2 audit,
   public `main` was `00a2c25`. Commits `54838f2`, `67f3f56`, and `1b0a53e`
   existed locally but had not been pushed. They already contained the broad
   model suite, local-mode/disorder code, and two report packages. Earlier
   documentation incorrectly called `67f3f56` a public baseline.
2. **Authoritative research workspace.** Newer clean affine-PA, geometry,
   tunneling, Anderson, tests, configurations, and scheduler tools were copied
   from the working research tree without changing the originals.
3. **Immutable executed snapshot.** The final PA diagonal-uniform-noise module,
   reporting module, CLI, test, and production configuration were recovered
   from the source snapshot embedded in its completed run. This matters: the
   executed snapshot is the authoritative version for that report, not an
   earlier live-workspace draft.
4. **Recorded report artifacts.** Compiled PDFs and final figures were copied
   as evidence products. Their large intermediate eigensystems remain outside
   Git.
5. **Publication assembly.** Release-only smoke, packaging, inventory, CI, and
   validation utilities were added to make the boundary auditable.
6. **October experiments.** Cached-PA multifractality, spectral-bin audits,
   exact-level/hub/near-zero plotting scripts, and compact recorded results
   were imported for this update. Original-file checksums are in
   [october_import_provenance.csv](october_import_provenance.csv). Publication
   changes parameterize paths, add validation, and correct interpretation;
   recorded numerical tables are not recomputed or altered.

`docs/source_inventory.csv` provides a SHA-256, byte count, role, and source
tier for every publishable file except the inventory itself. It is generated
by `tools/build_source_inventory.py`.

## Preservation rules

- The source workspace and historical output trees are treated as read-only.
- Existing compact outputs keep their canonical relative paths.
- Dated files under `docs/history/` are evidence snapshots. Current commands
  live in the README and reproducibility guides.
- Large caches, scheduler logs, partials, and local transfer roots are not
  rewritten into Git. Their required schemas and identities remain in code,
  configurations, and manifests.
- Absolute user paths and named cluster accounts are replaced by documented
  placeholders or environment variables.

The assembly branch is `codex/reproducible-release-20261005`. The November
update is ready locally; publication to the existing repository's `main`
is pending authenticated write access. See the [publication audit](publication_update_20261102.md).
