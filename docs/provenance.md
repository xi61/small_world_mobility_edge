# Provenance and preservation

The publication tree was assembled without modifying the original research
workspace or the existing public checkout.

## Source tiers

1. **Existing public checkout.** The release starts from commit `67f3f56`,
   which already contained the regular, Erdős–Rényi, Chung–Lu, fixed-degree,
   broad PA, selected compact results, and the 84-figure PA report package.
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

The release branch is `codex/reproducible-release-20261005`. Publishing it to
the configured remote is a separate external action.
