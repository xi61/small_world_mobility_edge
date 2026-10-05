# Data and artifact availability

The repository includes code, tests, configurations, compact result tables,
manifests, final report figures, TeX sources, and compiled PDFs. This is enough
to audit the published claims, run controlled examples, and reproduce every
figure when the production caches are supplied.

The following high-volume artifacts are intentionally not Git objects:

- full dense eigenvector matrices and residual arrays;
- per-graph compressed localization tables;
- restartable partials and replay partials;
- scheduler logs and software caches;
- copied cluster run roots and transfer archives.

Those files are deterministic or restartable products, but some individual
arrays are hundreds of megabytes and complete run trees are multi-gigabyte.
Committing them would exceed normal GitHub limits. They should be distributed
as versioned release assets or through an external archival repository if
public raw-data distribution is desired.

To attach external data, preserve the original case directory structure and
manifests. The code verifies graph hashes, source/configuration identities,
potential hashes, operator definitions, dimensions, and COMPLETE status before
aggregation. Do not rename or edit a manifest to force acceptance.

The exact boundary of checked-in historical results is documented in
`docs/results_inventory.md`; report-figure checksums and generators are in
`docs/figure_reproduction.csv`.
