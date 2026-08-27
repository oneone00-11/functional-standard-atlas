"""Pin the main-figure order against the manuscript.

`build_submission()` copies `MAIN_FIGS[i]` to `Zhang_Fig{i+1}`, so the list is
not a collection but an ordering: changing a position renumbers a delivered
figure, and nothing downstream notices. The manuscript's Code availability
section states that every figure regenerates from the archive, which is only
true while this order matches the paper.

That has now drifted twice. Both times the list carried a comment saying it was
pinned to the manuscript, and both times the comment was read as description
rather than constraint -- the second drift left `Zhang_Fig3` holding the
head-to-head panel while the paper's Figure 3 was the ceiling distribution. A
comment cannot fail a build, so the pin lives here instead.

SOURCE OF TRUTH: the submitted manuscript (Figures 1-6, first-citation order).
When the manuscript renumbers, change EXPECTED_MAIN_FIGS in this file and
`MAIN_FIGS` in `atlas.supplement` together; the caption fragments below are
what make the intended order checkable by eye.
"""

from __future__ import annotations

from atlas.supplement import DEMOTED_FIGS, MAIN_FIGS

# Figure number -> (figure stem, opening words of that figure's caption).
# Keep the captions in step with the manuscript; they are the human-readable
# half of this pin and the reason a reviewer can audit it without the docx.
EXPECTED_BY_NUMBER = {
    1: ("fig_atlas_overview", "The functional-standard atlas"),
    2: ("fig_territory_corrected", "The territory map as measured, and as a fraction"),
    3: ("fig_mavedb_ceilings", "Attenuation ceilings across human MaveDB deposits"),
    4: ("fig_splice_head_to_head", "Ordering claims tested pairwise"),
    5: ("fig_rna_readout", "Predictor performance is readout-specific"),
    6: ("fig_classification", "Classification performance and evidence strength"),
}

EXPECTED_MAIN_FIGS = [EXPECTED_BY_NUMBER[n][0]
                      for n in sorted(EXPECTED_BY_NUMBER)]


def test_main_figs_matches_the_manuscript():
    """MAIN_FIGS must equal the manuscript's Figure 1-6 order, exactly."""
    assert MAIN_FIGS == EXPECTED_MAIN_FIGS, (
        "MAIN_FIGS no longer matches the manuscript's figure order.\n"
        f"  expected: {EXPECTED_MAIN_FIGS}\n"
        f"  actual:   {MAIN_FIGS}\n"
        "If the manuscript renumbered, update BOTH this file and "
        "atlas.supplement.MAIN_FIGS. If it did not, the reordering is a bug: "
        "build_submission() would ship mislabelled Zhang_Fig*.")


def test_submission_filenames_are_what_the_paper_cites():
    """Spell out the Zhang_FigN mapping the copy loop produces."""
    produced = {i: stem for i, stem in enumerate(MAIN_FIGS, 1)}
    expected = {n: stem for n, (stem, _) in EXPECTED_BY_NUMBER.items()}
    assert produced == expected, (
        "Zhang_FigN would not hold the figure the manuscript calls Figure N: "
        f"{ {n: (produced.get(n), expected[n]) for n in expected if produced.get(n) != expected[n]} }")


def test_main_and_demoted_figures_are_disjoint():
    """A demoted figure must not also be shipped as a main figure."""
    overlap = set(MAIN_FIGS) & set(DEMOTED_FIGS)
    assert not overlap, f"figures both main and demoted: {sorted(overlap)}"
