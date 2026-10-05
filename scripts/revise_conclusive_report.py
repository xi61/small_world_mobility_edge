#!/usr/bin/env python3
"""Revise the conclusive single-graph report using only the original figure suite."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "docs/reports/pa_three_model_edge_bulk_conclusive_report"
TEX = PKG / "pa_three_model_edge_bulk_conclusive_report.tex"

MODELS = [
    ("affine_alpha_1p05", r"affine $\alpha=1.05$"),
    ("nonlinear_zeta_1p5", r"nonlinear $\zeta=1.5$"),
    ("affine_alpha_2p0", r"affine $\alpha=2.0$"),
]


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    i = text.index(start)
    j = text.index(end, i)
    return text[:i] + replacement + text[j:]


def diagnostics_section() -> str:
    per_model = {
        "affine_alpha_1p05": {
            "density": r"The left panel makes the large zero atom visible, while the right panel removes the zero modes and the Perron root to reveal the continuous levels used for gap statistics. The narrow central spike explains why raw gaps near zero are small; it is a density effect, not by itself evidence that the corresponding eigenvectors are extended or localized.",
            "spacing": r"The raw histograms vary strongly across energy because the spectral density varies. After local unfolding, most populated interior bins are qualitatively closer to the GOE curve than to the Poisson curve, while the extreme bins contain few gaps and are visibly noisy. The global ratio mean is about 0.531, close to GOE, but that full-spectrum average does not resolve the rank-specific edge fluctuations shown by the microscope.",
            "energy": r"The energy--$N\,\mathrm{IPR}$ cloud has a pronounced localization bowl: the continuous bulk near $E=0$ is least localized, whereas the spectral tails rise by one to two orders of magnitude. The positive-edge rings sit high in the cloud; the isolated far-right Perron point must not be mixed with the non-Perron window summaries.",
            "edge": r"Read the four panels together. Individual $r_i$ values oscillate across almost the entire unit interval, so no single point or short run establishes Poisson or GOE statistics. By contrast, $N\,\mathrm{IPR}$ is already in the hundreds near the edge, rises above $10^3$ around ranks 18--31, and remains far above its bulk scale. $M_{10}$ stays near 0.45--0.50 through most of the top 50. The scatter panel shows that very large $N\,\mathrm{IPR}$ occurs at both small and large $r_i$; spacing ratio and localization strength are therefore not monotonically linked.",
            "bulk": r"The centered atom-free bulk still has irregular pointwise $r_i$, but its localization measures are much smaller: $N\,\mathrm{IPR}$ is mostly in the tens and $M_{10}$ is generally about 0.1--0.3 near the selected center. The larger values at negative offsets show that the chosen 50-level display spans a localization gradient; the nested numerical tables are the controlled comparison.",
            "compare": r"The boxplots separate edge from bulk most clearly in $N\,\mathrm{IPR}$, $M_{10}$, top-support degree, induced density, and full-graph distance. Edge supports range from very dense hub cores to less dense deeper states, whereas bulk supports cluster near zero density. The edge distance z-scores are strongly negative, meaning substantially shorter paths than uniformly random ten-vertex sets.",
        },
        "nonlinear_zeta_1p5": {
            "density": r"The full spectrum contains a smaller but still visible zero atom. Removing that atom and the Perron root produces 4,759 continuous non-Perron levels. As in the affine stress case, the sharp central density peak makes raw gaps near zero small without implying localization on its own.",
            "spacing": r"Unfolding largely removes the energy-dependent gap scale. The well-populated interior panels are closer to the GOE shape than to the exponential Poisson reference; the outermost panels have only tens of gaps and should be read as descriptive. The full continuous-spectrum ratio mean is about 0.536, while the non-Perron top-10 strict mean is 0.387. That contrast is a narrow-edge effect, not a whole-spectrum Poisson regime.",
            "energy": r"The continuous bulk near $E=0$ lies around $N\,\mathrm{IPR}\approx 3$--15, whereas the positive edge rises to the hundreds and occasionally near $10^3$. The positive-edge rings are cleanly separated vertically from the bulk rings, making eigenvector localization the strongest visual edge signal.",
            "edge": r"This is the four-panel figure highlighted in the request. The upper-left points are individual $r_i=\min(s_{i-1},s_i)/\max(s_{i-1},s_i)$ values; their large scatter is expected. Rank 2 is near zero because one adjacent gap reaches toward the isolated Perron root, which is why the strict non-Perron table begins its usable ratios at rank 3. The top-10 strict mean is near Poisson, but the top-25 and top-50 means move upward. Meanwhile $N\,\mathrm{IPR}$ stays in the hundreds through roughly rank 30 and then drops toward about 100; $M_{10}$ stays near 0.48 through roughly rank 30 before declining. The scatter panel again has no monotone $r_i$--localization relation.",
            "bulk": r"Around the selected atom-free center, $N\,\mathrm{IPR}$ remains tightly confined near 8--16 and $M_{10}$ near 0.06--0.10, despite large fluctuations in individual $r_i$. This is the cleanest example of why the bulk can be extended even when a short sequence of ratios happens to contain both Poisson-like and GOE-like values.",
            "compare": r"The edge and bulk distributions are almost disjoint in $N\,\mathrm{IPR}$ and $M_{10}$. Edge top-mass vertices also have much higher degree, form substantially denser induced subgraphs, and have shorter full-graph paths. The bulk induced density is exactly zero for these displayed supports, while the edge density is often large; this is a structural localization signature independent of the ambiguous ratio average.",
        },
        "affine_alpha_2p0": {
            "density": r"This model has no zero atom in the stored graph: 4,999 continuous non-Perron levels remain. Its broad, smooth density makes it the clean control for separating edge localization from zero-mode or rank-deficiency effects.",
            "spacing": r"The unfolded interior histograms track the GOE reference reasonably well, while extreme-edge panels have far fewer gaps and visible discreteness. The full-spectrum ratio mean is about 0.525, again close to GOE. Thus localization at this edge does not require an overall Poisson gap-ratio law in this single graph.",
            "energy": r"The $N\,\mathrm{IPR}$ bowl is smooth and symmetric apart from the isolated Perron point: values are near 4 in the bulk and rise into the tens or hundreds toward the tails. This provides the clearest visual mobility-edge-like crossover of the three graphs, although one realization cannot locate a sharp transition.",
            "edge": r"The upper-right and lower-left panels decline fairly steadily with rank: $N\,\mathrm{IPR}$ falls from hundreds to a few tens, and $M_{10}$ falls from roughly 0.4 to 0.15--0.2. Individual ratios still jump between small and large values, and the scatter plot shows no single-valued relation between $r_i$ and $N\,\mathrm{IPR}$. The localization crossover is much clearer than any crossover in $r_i$.",
            "bulk": r"The bulk localization panels are nearly flat: $N\,\mathrm{IPR}$ stays around 3.5--4.3 and $M_{10}$ around 0.03. Pointwise gap ratios remain noisy, but their fluctuations occur on an eigenvector background that is uniformly extended relative to the edge.",
            "compare": r"Every eigenvector/support metric separates edge and bulk: edge $N\,\mathrm{IPR}$ and $M_{10}$ are much larger, its selected vertices have degrees in the hundreds rather than about ten, induced density is substantial rather than zero, and pairwise distance is roughly 1--2 rather than about 3. The strongly negative distance z-scores confirm compactness relative to uniform random supports.",
        },
    }

    chunks = [
        r"\section{Interpretation of the original diagnostic figures}",
        r"\label{sec:original-figures}",
        "The interpretations below refer to the original recorded plots, not to newly drawn window-summary curves. No new diagnostic figure is introduced in this revision. The figures display edge ranks 1--50 including the Perron state, whereas the numerical edge tables use non-Perron ranks 2 onward. This distinction is kept explicit.",
        r"\subsection{How to read the four-panel eigenpair microscopes}",
        r"The upper-left panel reports an \emph{individual} adjacent-gap ratio for each eligible eigenvalue, not a moving mean. The dashed GOE and dotted Poisson lines are ensemble reference means, so a single point crossing either line is not a classification. The upper-right panel reports $N\,\mathrm{IPR}$ on a logarithmic scale; higher means more localization. The lower-left panel reports $M_{10}$, the sum of the ten largest squared eigenvector coordinates; it is not a distance. The lower-right panel asks whether spacing ratio and localization co-vary. Across all three graphs they do not do so monotonically.",
        r"\subsection{How to read the spacing histograms}",
        r"Raw gaps depend on the local spectral density, so raw histograms from different energy bands cannot be compared as though they share one physical scale. The unfolded plots rescale the local mean spacing to one and are the appropriate visual comparison with GOE and Poisson reference curves. Extreme-tail panels contain the fewest gaps and therefore have the largest finite-sample noise. These are one-graph diagnostics, not ensemble tests.",
        r"\subsection{How to read the induced-support drawings}",
        r"Each drawing keeps only the ten vertices carrying the largest eigenvector masses. Node area encodes mass, color encodes full-graph degree, and black lines are only edges among those ten vertices. Missing black lines do not mean that two vertices are far apart in the full graph; full-graph shortest-path distances can pass through omitted vertices and are summarized separately. Edge supports form visible hub-rich cores, while the centered bulk supports are mostly isolated in the induced drawings.",
        r"\setcounter{section}{9}",
        r"\section{Recorded spacing and edge-bulk diagnostics}",
    ]

    for key, label in MODELS:
        d = per_model[key]
        chunks.extend([
            rf"\subsection{{{label}}}",
            rf"\paragraph{{Eigenvalue density.}} {d['density']}",
            r"\begin{figure}[H]\centering", rf"\includegraphics[width=.94\linewidth]{{figures/{key}/spectrum/single_graph_eigenvalue_density.png}}", rf"\caption{{Original single-graph eigenvalue-density diagnostic for {label}.}}\end{{figure}}",
            rf"\paragraph{{Raw and unfolded local spacing.}} {d['spacing']}",
            r"\begin{figure}[H]\centering", rf"\includegraphics[width=.94\linewidth]{{figures/{key}/histograms/single_graph_local_raw_spacing_histograms.png}}", rf"\caption{{Original local raw-spacing histograms for {label}.}}\end{{figure}}",
            r"\begin{figure}[H]\centering", rf"\includegraphics[width=.94\linewidth]{{figures/{key}/histograms/single_graph_local_unfolded_spacing_histograms.png}}", rf"\caption{{Original local unfolded-spacing histograms for {label}.}}\end{{figure}}",
            r"\begin{figure}[H]\centering", rf"\includegraphics[width=.94\linewidth]{{figures/{key}/histograms/single_graph_global_spacing_and_ratio.png}}", rf"\caption{{Original full continuous-spectrum spacing and ratio diagnostic for {label}.}}\end{{figure}}",
            rf"\paragraph{{Localization versus energy.}} {d['energy']}",
            r"\begin{figure}[H]\centering", rf"\includegraphics[width=.94\linewidth]{{figures/{key}/energy_ipr/single_graph_localization_vs_energy.png}}", rf"\caption{{Original localization-versus-energy diagnostic for {label}.}}\end{{figure}}",
            rf"\paragraph{{Edge eigenpair microscope.}} {d['edge']}",
            r"\begin{figure}[H]\centering", rf"\includegraphics[width=.94\linewidth]{{figures/{key}/edge/edge_rank_diagnostics.png}}", rf"\caption{{Original four-panel edge-rank eigenpair microscope for {label}.}}\end{{figure}}",
            rf"\paragraph{{Bulk eigenpair microscope.}} {d['bulk']}",
            r"\begin{figure}[H]\centering", rf"\includegraphics[width=.94\linewidth]{{figures/{key}/bulk/bulk_offset_diagnostics.png}}", rf"\caption{{Original four-panel centered bulk-offset eigenpair microscope for {label}.}}\end{{figure}}",
            rf"\paragraph{{Direct edge-versus-bulk comparison.}} {d['compare']}",
            r"\begin{figure}[H]\centering", rf"\includegraphics[width=.94\linewidth]{{figures/{key}/edge_vs_bulk/edge_vs_bulk_comparison.png}}", rf"\caption{{Original direct edge-versus-bulk comparison for {label}.}}\end{{figure}}",
        ])
    return "\n".join(chunks) + "\n"


def support_figure(paths: list[tuple[str, str]], caption: str) -> str:
    blocks = [r"\begin{figure}[H]\centering"]
    for i, (path, label) in enumerate(paths):
        blocks.extend([r"\begin{minipage}[t]{.48\linewidth}\centering", rf"\includegraphics[width=\linewidth]{{{path}}}", rf"\caption*{{\footnotesize {label}}}\end{{minipage}}"])
        if i % 2 == 0:
            blocks.append(r"\hfill")
        elif i != len(paths) - 1:
            blocks.append(r"\par\medskip")
    blocks.append(rf"\caption{{{caption}}}\end{{figure}}")
    return "\n".join(blocks)


def induced_section() -> str:
    chunks = [
        r"\section{Original induced top-mass support graphs}",
        "These are the original test drawings: edge ranks 1--10 include the Perron state at rank 1, and the bulk panels are the centered atom-free bulk-10 window. They are retained exactly as recorded; no replacement or newly generated induced graph is included. Coordinate tables remain omitted.",
        "Across the three models, the edge drawings visually agree with the numerical density, component, LCC, degree, and distance summaries: the selected vertices usually form one dense hub-rich component. The bulk drawings are almost edgeless, often ten singleton components. This does not contradict full-graph distances near two or three because paths through vertices outside the displayed ten are not drawn.",
    ]
    bulk_order = ["m001", "m002", "m003", "m004", "p000", "p001", "p002", "p003", "p004", "p005"]
    for key, label in MODELS:
        chunks.append(rf"\subsection{{{label}: original edge ranks 1--10}}")
        edge_paths = [(f"figures/{key}/induced_subgraphs/edge/edge_rank_{rank:03d}.png", f"edge rank {rank:03d}") for rank in range(1, 11)]
        for i in range(0, 10, 4):
            chunks.append(support_figure(edge_paths[i:i + 4], f"Original edge induced supports, ranks {i + 1}--{min(i + 4, 10)}."))
        chunks.append(rf"\subsection{{{label}: original centered bulk window}}")
        bulk_paths = [(f"figures/{key}/induced_subgraphs/bulk/bulk_offset_{off}.png", f"bulk offset {off}") for off in bulk_order]
        for i in range(0, 10, 4):
            chunks.append(support_figure(bulk_paths[i:i + 4], "Original centered bulk induced supports."))
    return "\n".join(chunks) + "\n"


def style_tables(text: str) -> str:
    text = text.replace(r"\toprule", r"\hline").replace(r"\midrule", r"\hline").replace(r"\bottomrule", r"\hline")
    specs = {
        "llrrrrrrrr": "|l|l|r|r|r|r|r|r|r|r|",
        "llrr|rr|rr|rr|rr": "|l|l|r|r|r|r|r|r|r|r|r|r|",
        "ll|rr|rr|rr|rr|rr|rr": "|l|l|r|r|r|r|r|r|r|r|r|r|r|r|",
        "llrrrrrr": "|l|l|r|r|r|r|r|r|",
        "lllrrrr": "|l|l|l|r|r|r|r|",
        "p{8.15in}r": "|p{8.15in}|r|",
    }
    for old, new in specs.items():
        text = text.replace(rf"\begin{{tabular}}{{{old}}}", rf"\rowcolors{{2}}{{white}}{{lightblue}}\begin{{tabular}}{{{new}}}")
        text = text.replace(rf"\begin{{longtable}}{{{old}}}", rf"\rowcolors{{2}}{{white}}{{lightblue}}\begin{{longtable}}{{{new}}}")
    lines = text.splitlines()
    in_table = False
    for idx, line in enumerate(lines):
        if "\\begin{tabular}" in line or "\\begin{longtable}" in line:
            in_table = True
        if in_table and line.lstrip().startswith(("Model &", "Region &", "TeX-relative filename &")):
            end = ""
            body = line
            for suffix in (r"\\\hline\endfirsthead", r"\\\hline\endhead", r"\\\hline", r"\\"):
                if body.endswith(suffix):
                    end = suffix
                    body = body[: -len(suffix)]
                    break
            cells = [c.strip() for c in body.split("&")]
            lines[idx] = " & ".join(rf"\TH{{{cell}}}" for cell in cells) + end
        if in_table and line.rstrip().endswith(r"\\"):
            lines[idx] = line.rstrip() + r"\hline"
        if "\\end{tabular}" in line or "\\end{longtable}" in line:
            in_table = False
    return "\n".join(lines) + "\n"


def manifest_section(tex: str) -> tuple[str, list[str]]:
    paths = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)
    if len(paths) != 84 or len(set(paths)) != 84:
        raise RuntimeError(f"Expected 84 unique original figures, found {len(paths)} refs / {len(set(paths))} unique")
    rows = []
    for rel in sorted(paths):
        fp = PKG / rel
        if not fp.exists():
            raise FileNotFoundError(fp)
        rows.append(rf"\path{{{rel}}} & {fp.stat().st_size} \\")
    section = "\n".join([
        r"\begin{landscape}\section{Explicit figure-file manifest}",
        r"Every path below is quoted literally by an \texttt{\textbackslash includegraphics} command and included in the package.",
        r"\fontsize{5}{5.8}\selectfont\setlength{\tabcolsep}{2pt}",
        r"\rowcolors{2}{white}{lightblue}\begin{longtable}{|p{8.15in}|r|}\hline",
        r"\TH{TeX-relative filename} & \TH{Bytes} \\\hline\endfirsthead",
        r"\hline\TH{TeX-relative filename} & \TH{Bytes} \\\hline\endhead",
        *rows,
        r"\end{longtable}\end{landscape}",
    ]) + "\n"
    return section, paths


def write_manifests(paths: list[str]) -> None:
    with (PKG / "figure_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tex_relative_path", "original_source", "role", "packaged_path", "exists", "bytes", "sha256"])
        for rel in paths:
            fp = PKG / rel
            source = rel.replace("figures/", "outputs/preferential_attachment/single_graph_edge_bulk/", 1)
            role = "original induced support graph" if "induced_subgraphs" in rel else "original core diagnostic"
            writer.writerow([rel, source, role, rel, fp.exists(), fp.stat().st_size, hashlib.sha256(fp.read_bytes()).hexdigest()])
    (PKG / "FIGURES.md").write_text("# Figures quoted by the TeX source\n\n" + "\n".join(f"- {p}" for p in paths) + "\n", encoding="utf-8")
    (PKG / "README.md").write_text(
        "Three-model edge/bulk report package\n\nCompile here with:\ntectonic pa_three_model_edge_bulk_conclusive_report.tex\n\nAll 84 includegraphics paths are original recorded test figures, relative to this directory, and listed in figure_manifest.csv and FIGURES.md.\n",
        encoding="utf-8",
    )


def main() -> None:
    text = TEX.read_text(encoding="utf-8")
    text = text.replace(r"\definecolor{navy}{HTML}{17324D}\definecolor{blue}{HTML}{2A6F97}", r"\definecolor{navy}{HTML}{17324D}\definecolor{blue}{HTML}{2A6F97}" + "\n" + r"\definecolor{lightblue}{HTML}{EAF3F8}\definecolor{tableline}{HTML}{94A3B8}")
    text = text.replace(r"\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\renewcommand{\arraystretch}{1.12}", r"\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\renewcommand{\arraystretch}{1.18}" + "\n" + r"\setlength{\arrayrulewidth}{0.45pt}\arrayrulecolor{tableline}" + "\n" + r"\newcommand{\TH}[1]{\cellcolor{navy}\color{white}\bfseries #1}")
    text = replace_between(text, r"\section{Window-size synthesis figures}", r"\section{Interpretation}", "")
    text = replace_between(
        text, r"\section{Interpretation}", r"\begin{landscape}\section{Random 1,000-set reference distributions}",
        r"\setcounter{section}{8}" + "\n" + r"\section{Numerical interpretation and random-support controls}" + "\n" +
        r"The spacing-ratio conclusions are weaker than the eigenvector conclusions. Edge and bulk strict mean $r$ overlap in several windows, so these three fixed graphs do not support a universal edge-Poisson claim. In contrast, every window size shows substantially larger edge IPR, $N\,\mathrm{IPR}$, and $M_{10}$, with hub-heavy, dense, compact top-mass supports.\n\n" +
        "Induced density is the number of displayed edges divided by $\\binom{10}{2}=45$. Components and LCC are properties of the ten-vertex induced graph; LCC is the number of vertices in its largest connected component. Distances are shortest paths in the full graph. For each model, 1,000 uniformly sampled ten-vertex sets define $z=(T_{\\rm obs}-\\bar T_{\\rm rand})/s_{\\rm rand}$ and the inclusive percentile. Negative distance $z$ means compact; positive density or degree $z$ means unusually connected or hub-heavy. These controls are not degree-matched, and z-scores are not Gaussian p-values.\n"
    )
    text = replace_between(text, r"\section{Recorded spacing and edge-bulk diagnostics}", r"\section{Induced top-mass support graphs}", diagnostics_section())
    text = replace_between(text, r"\section{Induced top-mass support graphs}", r"\begin{landscape}\section{Explicit figure-file manifest}", induced_section())
    text = style_tables(text)
    marker = r"\begin{landscape}\section{Explicit figure-file manifest}"
    end = r"\section{Machine-readable outputs and limitations}"
    i, j = text.index(marker), text.index(end, text.index(marker))
    provisional = text[:i] + text[j:]
    manifest, paths = manifest_section(provisional)
    text = text[:i] + manifest + text[j:]
    TEX.write_text(text, encoding="utf-8")
    write_manifests(paths)


if __name__ == "__main__":
    main()
