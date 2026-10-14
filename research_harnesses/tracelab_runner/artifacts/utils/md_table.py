"""Render GitHub-Flavored-Markdown tables.

Experiment ``analyze.py`` scripts emit a ``.md`` table next to their LaTeX ``.tex`` so the web
detail page (``web/app/src/pages/exp/[...slug].astro``) can render the full table with the standard
``.readme table`` styling — same numbers as the paper, no LaTeX parsing on the web side.

Keep the output plain GFM: a header row, an alignment row, then body rows. Group headings (the
LaTeX ``\\multicolumn`` "Per session" rows) become bold lines BETWEEN tables, since GFM has no
column spanning. Use :func:`section_tables` for the grouped-block layout the cost/timing tables use.
"""

from __future__ import annotations

from typing import Iterable, Sequence

_SEP = {"l": ":--", "r": "--:", "c": ":-:"}


def gfm_table(headers: Sequence[str], rows: Iterable[Sequence[str]], align: Sequence[str] | None = None) -> str:
    """One GFM table. ``align`` is per-column 'l'/'r'/'c' (default: first left, rest right)."""
    headers = list(headers)
    ncol = len(headers)
    align = list(align) if align else (["l"] + ["r"] * (ncol - 1))
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(_SEP[a] for a in align) + " |",
    ]
    for row in rows:
        cells = [str(c) for c in row]
        cells += [""] * (ncol - len(cells))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def section_tables(
    headers: Sequence[str],
    sections: Iterable[tuple[str, Iterable[Sequence[str]]]],
    align: Sequence[str] | None = None,
) -> str:
    """Several tables sharing one header, each under a bold ``**label**`` group heading.

    ``sections`` is an iterable of ``(label, rows)``. Mirrors the grouped granularity blocks
    (Per session / Per request / Per step) of the cost and timing tables.
    """
    blocks: list[str] = []
    for label, rows in sections:
        blocks.append(f"**{label}**\n\n{gfm_table(headers, rows, align)}")
    return "\n".join(blocks)
