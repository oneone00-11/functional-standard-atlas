"""Assemble Additional file 2 from the Notes PDF and the seventeen figures.

The delivered file was assembled by hand, and it drifted twice: it carried a
value the source had corrected, and it predated a prose change nobody noticed.
Its page 16-32 font repair was also applied by hand, which made every later
rebuild a choice between losing the repair and shipping a stale file.

This does the assembly, so neither is a choice any more. The Notes PDF has to be
converted from .docx outside this repository -- and the converter matters: WPS
writing through Quartz leaves a text layer whose words run together, which a
similarity check reads. Word's export of the same file is clean.
`atlas.submission_package.text_layer` is the check; this module refuses to
assemble a Notes PDF that fails it.

Usage (PYTHONPATH=src):
    python -m atlas.build_additional_file_2 --notes <Supplemental_Note.pdf> \
        --figures <dir with Supplemental_Fig_S1..S17.pdf> --out <Additional_file_2.pdf>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

N_FIGURES = 17
# The Notes PDF is 16 pages and each of the seventeen figures is one, so a
# complete file is 33. This was 32, one short, and the check therefore exited 1
# on the file that actually ships -- a gate red against a correct artefact,
# which teaches everyone to ignore it. Note length is what moves this number,
# so it is stated rather than derived: if the Notes grow a page, that is a fact
# worth being told about rather than absorbing silently.
EXPECTED_PAGES = 33


def figure_paths(figures: Path) -> list[Path]:
    """S1 to S17 in numeric order, which is not the order a glob returns."""
    out = []
    for n in range(1, N_FIGURES + 1):
        p = figures / f"Supplemental_Fig_S{n}.pdf"
        if not p.exists():
            sys.exit(f"[af2] missing figure: {p}")
        out.append(p)
    return out


def build(notes: Path, figures: Path, out: Path) -> dict:
    import pypdf

    from atlas.submission_package import text_layer

    tl = text_layer(notes)
    if len(tl["glued"]) > 5:
        sys.exit(f"[af2] refusing to assemble: {notes.name} has {len(tl['glued'])} glued "
                 f"tokens in its text layer, so words have run together. Re-export the "
                 f".docx with Word (Save as PDF), not WPS. First offenders: "
                 f"{', '.join(tl['glued'][:3])}")

    writer = pypdf.PdfWriter()
    for page in pypdf.PdfReader(str(notes)).pages:
        writer.add_page(page)
    n_notes = len(writer.pages)

    per_figure = []
    for p in figure_paths(figures):
        r = pypdf.PdfReader(str(p))
        for page in r.pages:
            writer.add_page(page)
        per_figure.append((p.name, len(r.pages)))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        writer.write(fh)
    total = len(writer.pages)
    return {"notes_pages": n_notes, "figures": per_figure, "total_pages": total,
            "notes_glued": len(tl["glued"]), "notes_words": tl["words"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", required=True)
    ap.add_argument("--figures", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-pages", type=int, default=EXPECTED_PAGES)
    a = ap.parse_args()

    r = build(Path(a.notes).expanduser(), Path(a.figures).expanduser(),
              Path(a.out).expanduser())
    print(f"[af2] notes {r['notes_pages']} pages ({r['notes_words']:,} words, "
          f"{r['notes_glued']} glued) + {len(r['figures'])} figures")
    odd = [(n, k) for n, k in r["figures"] if k != 1]
    for n, k in odd:
        print(f"  {n}: {k} pages")
    print(f"[af2] wrote {a.out}: {r['total_pages']} pages")
    if r["total_pages"] != a.expect_pages:
        print(f"  PAGE COUNT: expected {a.expect_pages}, got {r['total_pages']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
