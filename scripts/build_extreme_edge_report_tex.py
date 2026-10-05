#!/usr/bin/env python3
"""Convert the updated extreme-edge Markdown report to portable LaTeX."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from build_extreme_edge_report_docx import parse_blocks


PLAIN = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_\discretionary{}{}{}", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "<": r"\textless{}", ">": r"\textgreater{}", "|": r"\textbar{}",
}

UNICODE = {
    "α": r"\ensuremath{\alpha}", "ζ": r"\ensuremath{\zeta}",
    "Δ": r"\ensuremath{\Delta}", "λ": r"\ensuremath{\lambda}",
    "δ": r"\ensuremath{\delta}", "π": r"\ensuremath{\pi}",
    "≥": r"\ensuremath{\geq}", "≤": r"\ensuremath{\leq}",
    "≈": r"\ensuremath{\approx}", "→": r"\ensuremath{\rightarrow}",
    "−": r"\ensuremath{-}", "±": r"\ensuremath{\pm}",
    "×": r"\ensuremath{\times}", "…": r"\ldots{}",
    "–": "--", "—": "---", "“": "``", "”": "''",
    "‘": "`", "’": "'", " ": "~",
}

INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*]+\*)")


def escape_text(value: str) -> str:
    return "".join(UNICODE.get(ch, PLAIN.get(ch, ch)) for ch in value)


def escape_code(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40,}", value):
        return r"\seqsplit{" + value + "}"
    escaped = escape_text(value)
    for marker in ("/", "=", ",", ":"):
        escaped = escaped.replace(marker, marker + r"\discretionary{}{}{}")
    return escaped


def inline_to_tex(value: str) -> str:
    out = []
    position = 0
    for match in INLINE.finditer(value):
        if match.start() > position:
            out.append(escape_text(value[position:match.start()]))
        token = match.group(0)
        if token.startswith("**"):
            inner = token[2:-2]
            if re.fullmatch(r"[A-Z0-9_]{12,}", inner):
                parts = inner.split("_")
                out.append(r"\textbf{\texttt{" + r"\_\hspace{0pt}".join(parts) + "}}")
            else:
                out.append(r"\textbf{" + inline_to_tex(inner) + "}")
        elif token.startswith("`"):
            inner = token[1:-1]
            if "/" in inner and re.fullmatch(r"[A-Za-z0-9_./:-]+", inner):
                out.append(r"\nolinkurl{" + inner + "}")
            else:
                out.append(r"\texttt{" + escape_code(inner) + "}")
        else:
            out.append(r"\emph{" + inline_to_tex(token[1:-1]) + "}")
        position = match.end()
    if position < len(value):
        out.append(escape_text(value[position:]))
    return "".join(out)


def column_fractions(count: int):
    return {
        2: [0.42, 0.50],
        3: [0.27, 0.28, 0.37],
        4: [0.24, 0.21, 0.21, 0.26],
        5: [0.20, 0.17, 0.17, 0.17, 0.22],
        6: [0.15, 0.17, 0.15, 0.09, 0.14, 0.22],
        9: [0.14] + [0.09] * 8,
    }[count]


def table_to_tex(rows) -> str:
    count = len(rows[0])
    fractions = column_fractions(count)
    specs = []
    for index, fraction in enumerate(fractions):
        ragged = index == 0 or (count <= 6 and index == count - 1)
        align = r">{\RaggedRight\arraybackslash}" if ragged else r">{\centering\arraybackslash}"
        specs.append(align + f"p{{{fraction:.2f}\\linewidth}}")
    spec = "".join(specs)
    size = r"\scriptsize" if count >= 6 else r"\footnotesize"

    def header_line():
        cells = [r"{\color{white}\bfseries " + inline_to_tex(cell) + "}" for cell in rows[0]]
        return r"\rowcolor{Navy}" + " & ".join(cells) + r" \\"

    out = [r"\begingroup", size, r"\setlength{\tabcolsep}{3pt}",
           r"\renewcommand{\arraystretch}{1.16}", f"\\begin{{longtable}}{{{spec}}}",
           r"\toprule", header_line(), r"\midrule", r"\endfirsthead",
           r"\toprule", header_line(), r"\midrule", r"\endhead",
           r"\midrule", rf"\multicolumn{{{count}}}{{r}}{{\scriptsize Continued on next page}} \\",
           r"\endfoot", r"\bottomrule", r"\endlastfoot"]
    for row_index, row in enumerate(rows[1:], 1):
        prefix = r"\rowcolor{PaleGray}" if row_index % 2 == 0 else ""
        out.append(prefix + " & ".join(inline_to_tex(cell) for cell in row) + r" \\")
    out.extend([r"\end{longtable}", r"\endgroup"])
    return "\n".join(out)


PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[letterpaper,top=0.82in,bottom=0.82in,left=0.9in,right=0.9in,headheight=14pt]{geometry}
\usepackage{amsmath,amssymb}
\usepackage[table]{xcolor}
\usepackage{booktabs,longtable,array,ragged2e}
\usepackage{seqsplit}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{pdflscape}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}

\definecolor{Navy}{HTML}{0B2545}
\definecolor{Blue}{HTML}{1D5D8C}
\definecolor{Accent}{HTML}{2A7F9E}
\definecolor{BodyText}{HTML}{24313D}
\definecolor{Muted}{HTML}{607080}
\definecolor{PaleBlue}{HTML}{E8EEF5}
\definecolor{PaleGray}{HTML}{F4F6F8}

\AtBeginDocument{\color{BodyText}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\LTleft}{0pt}
\setlength{\LTright}{0pt}
\setcounter{secnumdepth}{0}
\setlist[itemize]{leftmargin=1.25em,itemsep=2pt,topsep=3pt}
\setlist[enumerate]{leftmargin=1.55em,itemsep=4pt,topsep=3pt}
\titleformat{\section}{\color{Navy}\Large\bfseries}{}{0pt}{}[\vspace{2pt}\color{PaleBlue}\titlerule]
\titleformat{\subsection}{\color{Blue}\large\bfseries}{}{0pt}{}
\titleformat{\subsubsection}{\color{BodyText}\normalsize\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{14pt}{6pt}
\titlespacing*{\subsection}{0pt}{11pt}{4pt}
\titlespacing*{\subsubsection}{0pt}{9pt}{3pt}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\scriptsize\color{Navy}\bfseries PREFERENTIAL-ATTACHMENT SPECTRAL AUDIT}
\fancyhead[R]{\scriptsize\color{Muted} UPDATED 30 AUGUST 2026}
\fancyfoot[R]{\scriptsize\color{Muted} PAGE \thepage}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
\raggedbottom
\emergencystretch=3em

\begin{document}
"""


def build(source: Path, target: Path) -> None:
    blocks = parse_blocks(source.read_text(encoding="utf-8"))
    out = [PREAMBLE]
    list_kind = None
    landscape = False
    first_heading = True

    def close_list():
        nonlocal list_kind
        if list_kind:
            out.append(rf"\end{{{list_kind}}}")
            list_kind = None

    for block in blocks:
        kind = block[0]
        if kind in {"bullet", "number"}:
            desired = "itemize" if kind == "bullet" else "enumerate"
            if list_kind != desired:
                close_list()
                out.append(rf"\begin{{{desired}}}")
                list_kind = desired
            out.append(r"\item " + inline_to_tex(block[1]))
            continue

        close_list()
        if kind == "heading":
            level, text = block[1], block[2]
            if text == "Why gap and ratio counts can differ" and not landscape:
                out.extend([r"\clearpage", r"\begin{landscape}"])
                landscape = True
            if level == 1 and first_heading:
                out.extend([
                    r"{\color{Accent}\small\bfseries TECHNICAL FORENSIC REPORT\par}",
                    r"\vspace{5pt}",
                    r"{\color{Navy}\fontsize{24}{28}\selectfont\bfseries " + inline_to_tex(text) + r"\par}",
                    r"\vspace{5pt}{\color{Accent}\rule{\linewidth}{1.2pt}}\vspace{7pt}",
                ])
                first_heading = False
            else:
                if text.startswith("Appendix A:"):
                    out.append(r"\clearpage")
                command = {2: "section", 3: "subsection", 4: "subsubsection"}[level]
                out.append(rf"\{command}{{{inline_to_tex(text)}}}")
            continue
        if kind == "paragraph":
            text = inline_to_tex(block[1])
            if block[1].startswith("**Status:**"):
                out.extend([
                    r"\begin{center}",
                    r"\setlength{\fboxsep}{8pt}\colorbox{PaleBlue}{\parbox{0.94\linewidth}{" + text + "}}",
                    r"\end{center}",
                ])
            else:
                out.append(text + r"\par")
            continue
        if kind == "equation":
            out.extend([r"\begin{equation*}", block[1].strip(), r"\end{equation*}"])
            continue
        if kind == "code":
            out.extend([r"\begin{quote}\ttfamily\small", escape_code(block[1]), r"\end{quote}"])
            continue
        if kind == "table":
            out.append(table_to_tex(block[1]))
            if len(block[1][0]) >= 8 and landscape:
                out.extend([r"\end{landscape}", r"\clearpage"])
                landscape = False
            continue
        raise RuntimeError(f"Unhandled block {block}")

    close_list()
    if landscape:
        out.append(r"\end{landscape}")
    out.extend([r"\end{document}", ""])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: build_extreme_edge_report_tex.py SOURCE.md TARGET.tex")
    build(Path(sys.argv[1]), Path(sys.argv[2]))
