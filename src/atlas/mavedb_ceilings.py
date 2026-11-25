"""Compute the attenuation ceiling of every MaveDB deposit that permits one.

The survey (``atlas.mavedb_survey``) establishes which deposits publish what a
reliability estimate needs. This module computes the estimate itself, so the
paper can describe the ceiling *distribution* across the resource rather than
generalising from seven assays.

Estimators are the ones the primary analysis uses (``atlas.reliability``):

  replicates  mean pairwise Spearman correlation among the reconciling replicate
              columns, raised to the reliability of their mean by Spearman-Brown
  errors      rel = 1 - mean(SE^2) / var(score)

Error columns come in four shapes. They are tried in this order and the rule
used is recorded, because they are not equally trustworthy:

  1  se / sem / standard_error / SE_*      used directly
  2  a CI pair (lower, upper)              SE = (upper - lower) / (2 * 1.96)
  3  sd / sigma / std_dev                  used as if it were an SE
  4  variance                              SE = sqrt(variance)

Rule 3 is flagged: a deposited SD may be the spread across replicates rather
than the standard error of their mean, and the two differ by sqrt(n). Treating
it as an SE understates reliability, so the resulting ceiling is a lower bound.

**Reliability is deliberately not clipped.** ``rel = 1 - mean(SE^2)/var`` can
only leave [0, 1] from below, and it does so when the error column is not on the
scale of the score — which is a property of the deposit worth counting, not an
error to hide. Out-of-range values are classified, not squashed:

  ok            0 < rel < 0.99
  degenerate    rel <= 0      error column not on the score's scale
  suspicious    rel >= 0.99   SE vanishing against var; almost always the same
  insufficient  fewer than 50 usable rows, or var(score) == 0

The current generation (results/mavedb_ceilings_v2.tsv) is not a new survey of the
tables. v1's eleven computed columns (`one`) are carried for the 2,409 deposits they
stand for, and two steps are added:

* The annotation (G16), now in this module: ``title`` and ``human`` from the survey,
  ``target_genes`` from the metadata, ``platform`` (v1: `platform_series` on the 2,452
  candidates), and ``stratum`` (designed_stability for the platform, then human, then
  other_natural). Run with v1's flags and v1's platform algorithm it reproduces v1's
  five columns row by row, or the build stops; it is then run with the survey v2's
  human flag and platform (atlas.mavedb_records).
* The error column (G14): `se_from` takes the first column that looks like an SE. For
  the 43 candidates by error column that carry two or more such columns, the score table
  is fetched again (cached in data/raw/mavedb_ceilings_v2/) and the column chosen by the
  relations between the columns (A3.4's tolerance: rtol 0.5%, on 99% of the rows):
  ``z_column``          a column equals score / SE_x row by row (the deposit's own z):
                        SE_x is used (BAP1's SGE deposits: SE_bind_continuous)
  ``test_statistic``    score itself equals a column over SE_x: the score is a test
                        statistic and the deposit carries no error on its scale; status
                        ``no_score_scale_error``, no ceiling (RAD51C's SGE deposits)
  ``first_of_several``  neither: the first, as before, and said so
  in the column ``error_column_choice``. ``revised`` says what a row changed against v1:
  error_column, human, platform.

Usage (with PYTHONPATH=src):

    python -m atlas.mavedb_ceilings                       # the current generation
    python -m atlas.mavedb_ceilings --survey-all --survey results/mavedb_survey_v1.tsv \
        --out results/mavedb_ceilings_v1.tsv [--limit 30]   # v1's run over every candidate
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from atlas.mavedb_survey import API, PAUSE, TIMEOUT
from atlas.reliability import spearman_brown

SCORE = "score"
MIN_N = 50
CI_Z = 1.959963984540054          # two-sided 95%

RE_SE = re.compile(r"(^|_)(se|sem|std[_.]?err|stderr|standard[_.]?error)($|_)", re.I)
RE_SD = re.compile(r"(^|_)(sd|sigma\d*|std[_.]?dev)($|_)", re.I)
RE_VAR = re.compile(r"(^|_)variance($|_)", re.I)
RE_LO = re.compile(r"(low(er)?|lo)($|_)", re.I)
RE_HI = re.compile(r"(high(er)?|upper|hi)($|_)", re.I)
RE_CI = re.compile(r"ci", re.I)


# Rocklin/Tsuboyama-lineage protease-resistance stability data on de novo designed
# mini-proteins. Identified from target names (HHH_rd*, EEHEE_rd*, EHEE_rd*, TrROS,
# villin/pin1-style scaffolds) and titles containing "trypsin"/"chymotrypsin"/
# "digestion". These are not human variant effect measurements, and their published
# interval is a curve-fitting CI on dG rather than a reproducibility interval, so
# they are reported separately and never pooled into the main distribution.
RE_DESIGNED = re.compile(
    r"HHH_rd|EEHEE_rd|EHEE_rd|HEEH_rd|TrROS|villin|pin1|rd\d_\d{4}", re.I)
RE_PROTEASE = re.compile(r"trypsin|chymotrypsin|digestion", re.I)


RE_ASSAY_SUFFIX = re.compile(
    r"\s*(chymotrypsin|trypsin)\s*digestion\s*$|\s*combined\s+scores?\s*$", re.I)


def platform_of(title: str, targets: str = "") -> str:
    """`designed_stability` for the protease-resistance mini-protein platform."""
    blob = f"{title} {targets}"
    if RE_DESIGNED.search(blob) or RE_PROTEASE.search(blob):
        return "designed_stability"
    return "other"


def platform_series(titles: pd.Series, targets: pd.Series) -> pd.Series:
    """Platform per deposit, propagated across each protein's deposit group.

    The platform deposits three score sets per protein: trypsin, chymotrypsin
    and a combined score. Only the first two name a protease, so classifying on
    title alone leaves the combined deposit — the one carrying the fitting CI
    whose ceiling is in question — filed as an ordinary deposit. Grouping on the
    target stem and marking the whole group keeps the three together.
    """
    stem = titles.fillna("").map(lambda t: RE_ASSAY_SUFFIX.sub("", str(t)).strip())
    direct = pd.Series([platform_of(str(t), str(g))
                        for t, g in zip(titles.fillna(""), targets.fillna(""))],
                       index=titles.index)
    flagged = set(stem[direct == "designed_stability"])
    return stem.map(lambda k: "designed_stability" if k in flagged else "other")


def _ci_pairs(cols: list[str]) -> list[tuple[str, str]]:
    """CI bounds that share a stem, e.g. score_95CI_low / score_95CI_high."""
    lo = [c for c in cols if RE_CI.search(c) and RE_LO.search(c)]
    hi = [c for c in cols if RE_CI.search(c) and RE_HI.search(c)]
    out = []
    for a in lo:
        stem = RE_LO.sub("", a)
        for b in hi:
            if RE_HI.sub("", b) == stem:
                out.append((a, b))
                break
    return out


def se_from(df: pd.DataFrame) -> tuple[pd.Series | None, str, str]:
    """Return (per-variant SE, column(s) used, rule name), by priority."""
    cols = [c for c in df.columns if c != SCORE]
    for c in cols:
        if RE_SE.search(c):
            return pd.to_numeric(df[c], errors="coerce"), c, "se"
    for a, b in _ci_pairs(cols):
        lo = pd.to_numeric(df[a], errors="coerce")
        hi = pd.to_numeric(df[b], errors="coerce")
        return (hi - lo).abs() / (2 * CI_Z), f"{a}+{b}", "ci"
    for c in cols:
        if RE_SD.search(c):
            return pd.to_numeric(df[c], errors="coerce"), c, "sd_as_se"
    for c in cols:
        if RE_VAR.search(c):
            return np.sqrt(pd.to_numeric(df[c], errors="coerce").clip(lower=0)), c, "variance"
    return None, "", ""


def reliability_from_replicates(df: pd.DataFrame, reps: list[str]) -> tuple[float, int]:
    """Spearman-Brown reliability of the mean of `reps`."""
    sub = df[reps].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < MIN_N:
        return float("nan"), len(sub)
    rs = [sub[a].corr(sub[b], method="spearman") for a, b in combinations(reps, 2)]
    rs = [r for r in rs if np.isfinite(r)]
    if not rs:
        return float("nan"), len(sub)
    return spearman_brown(float(np.mean(rs)), k=len(reps)), len(sub)


def classify_rel(rel: float, n: int, var: float) -> str:
    if n < MIN_N or not np.isfinite(var) or var <= 0 or not np.isfinite(rel):
        return "insufficient"
    if rel <= 0:
        return "degenerate"
    if rel >= 0.99:
        return "suspicious"
    return "ok"


def _fetch_table(urn: str) -> pd.DataFrame:
    r = requests.get(f"{API}/score-sets/{urn}/scores", timeout=TIMEOUT)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), low_memory=False)


def one(urn: str, permits: str, reconciling: str) -> dict:
    try:
        df = _fetch_table(urn)
    except Exception as exc:                       # noqa: BLE001 — recorded
        row = _blank(urn, permits)
        row["note"] = f"fetch/parse failed: {type(exc).__name__}"
        return row
    return from_table(df, urn, permits, reconciling)


def _blank(urn: str, permits: str) -> dict:
    return {"urn": urn, "permits": permits, "method": "", "error_column": "",
            "n_variants": None, "var_score": None, "mean_se2": None,
            "reliability_raw": None, "ceiling": None, "status": "insufficient",
            "note": ""}


def from_table(df: pd.DataFrame, urn: str, permits: str, reconciling: str,
               se_column: str | None = None) -> dict:
    """The eleven columns from one score table; se_column, when given, is the error
    column to use in place of `se_from`'s first choice."""
    row = _blank(urn, permits)
    if SCORE not in df.columns:
        row["note"] = "no score column"
        return row
    score = pd.to_numeric(df[SCORE], errors="coerce")
    var = float(score.var())
    row["var_score"] = var

    reps = [c for c in (reconciling or "").split(";") if c and c in df.columns]
    if permits == "replicates" and len(reps) >= 2:
        rel, n = reliability_from_replicates(df, reps)
        row.update(method="replicates", error_column=";".join(reps),
                   n_variants=n, reliability_raw=rel)
    else:
        if se_column is not None:
            se, col, rule = pd.to_numeric(df[se_column], errors="coerce"), se_column, "se"
        else:
            se, col, rule = se_from(df)
        if se is None:
            row["note"] = "no usable error column"
            return row
        sub = pd.concat([score.rename(SCORE), se.rename("se")], axis=1).dropna()
        n = len(sub)
        row["n_variants"] = n
        if n >= MIN_N and var > 0:
            mse = float((sub["se"] ** 2).mean())
            row["mean_se2"] = mse
            row["reliability_raw"] = 1 - mse / float(sub[SCORE].var())
        row.update(method=rule, error_column=col)
        if rule == "sd_as_se":
            row["note"] = "SD used as SE; ceiling is a lower bound"

    rel = row["reliability_raw"]
    row["status"] = classify_rel(rel if rel is not None else float("nan"),
                                 row["n_variants"] or 0, var)
    if row["status"] == "ok":
        row["ceiling"] = float(np.sqrt(rel))
    return row


# ==========================================================================
# the current generation: the annotation and the error column
# ==========================================================================
COMPUTED = ["urn", "permits", "method", "error_column", "n_variants", "var_score",
            "mean_se2", "reliability_raw", "ceiling", "status", "note"]
ANNOTATED = ["title", "human", "target_genes", "platform", "stratum"]
COLUMNS_V2 = COMPUTED + ANNOTATED + ["error_column_choice", "revised"]
RELATION_RTOL = 0.005          # A3.4's: the deposits round their values
RELATION_SHARE = 0.99


def stratum_of(platform: str, human: bool) -> str:
    if platform == "designed_stability":
        return "designed_stability"
    return "human" if bool(human) else "other_natural"


def annotate(computed: pd.DataFrame, survey: pd.DataFrame, metadata: pd.DataFrame,
             platform: pd.Series | None = None) -> pd.DataFrame:
    """The five annotation columns. platform None: v1's algorithm (`platform_series` on
    these rows); else urn -> platform."""
    t = computed.copy()
    sv, md = survey.set_index("urn"), metadata.set_index("urn")
    t["title"] = t["urn"].map(sv["title"])
    t["human"] = t["urn"].map(sv["human"]).astype(bool)
    t["target_genes"] = t["urn"].map(md["target_genes"])
    if platform is None:
        t["platform"] = platform_series(t["title"], t["target_genes"]).to_numpy()
    else:
        t["platform"] = t["urn"].map(platform)
    t["stratum"] = [stratum_of(p, h) for p, h in zip(t["platform"], t["human"])]
    return t[COMPUTED + ANNOTATED]


def se_like(columns: list[str]) -> list[str]:
    return [c for c in columns if c != SCORE and RE_SE.search(c)]


def _holds(lhs: pd.Series, rhs: pd.Series) -> tuple[bool, int, float]:
    both = lhs.notna() & rhs.notna() & np.isfinite(lhs) & np.isfinite(rhs)
    n = int(both.sum())
    if n < MIN_N:
        return False, n, float("nan")
    share = float(np.isclose(lhs[both], rhs[both], rtol=RELATION_RTOL, atol=1e-9).mean())
    return share >= RELATION_SHARE, n, share


def choose_error_column(df: pd.DataFrame, se_cols: list[str]) -> dict:
    """z_column, test_statistic or first_of_several (see the module's docstring)."""
    x = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    score = x[SCORE]
    others = [c for c in x.columns if c != SCORE and x[c].notna().sum() >= MIN_N]
    for se in se_cols:
        z = (score / x[se]).replace([np.inf, -np.inf], np.nan)
        for c in others:
            if c == se:
                continue
            ok, n, share = _holds(x[c], z)
            if ok:
                return {"choice": "z_column", "se": se, "via": c, "rows": n, "share": share}
    for se in se_cols:
        for c in others:
            if c == se:
                continue
            ok, n, share = _holds(score, (x[c] / x[se]).replace([np.inf, -np.inf], np.nan))
            if ok:
                return {"choice": "test_statistic", "se": se, "via": c, "rows": n,
                        "share": share}
    return {"choice": "first_of_several", "se": se_cols[0], "via": "", "rows": None,
            "share": None}


def from_table_v2(df: pd.DataFrame, urn: str, permits: str, reconciling: str,
                  se_cols: list[str]) -> tuple[dict, dict]:
    """The eleven columns for one of the 43, with the choice behind them."""
    present = [c for c in se_cols if c in df.columns]
    if SCORE not in df.columns or not present:
        return from_table(df, urn, permits, reconciling), {"choice": "unavailable"}
    ch = choose_error_column(df, present)
    if ch["choice"] == "test_statistic":
        row = _blank(urn, permits)
        row.update(method="se", status="no_score_scale_error",
                   var_score=float(pd.to_numeric(df[SCORE], errors="coerce").var()),
                   note=(f"score is {ch['via']} / {ch['se']}: a test statistic; the deposit "
                         "carries no error on the score's scale"))
        return row, ch
    return from_table(df, urn, permits, reconciling, se_column=ch["se"]), ch


TABLES = Path(__file__).resolve().parents[2] / "data" / "raw" / "mavedb_ceilings_v2"


def fetch_tables(urns: list[str], cache: Path = TABLES) -> dict:
    """The score tables of these deposits, cached as <urn>.csv with the time fetched."""
    import json
    from datetime import datetime, timezone
    cache.mkdir(parents=True, exist_ok=True)
    log_path = cache / "fetch_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    for urn in urns:
        dest = cache / (urn.replace(":", "_") + ".csv")
        if dest.exists() and urn in log and "error" not in log[urn]:
            continue
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for k in range(4):
            try:
                r = requests.get(f"{API}/score-sets/{urn}/scores", timeout=TIMEOUT)
                r.raise_for_status()
                dest.write_bytes(r.content)
                log[urn] = {"fetched_utc": now, "bytes": len(r.content)}
                break
            except Exception as exc:               # noqa: BLE001
                log[urn] = {"fetched_utc": now, "error": f"{type(exc).__name__}: {exc}"[:200]}
                time.sleep(5 * 2 ** k)
        log_path.write_text(json.dumps(log, indent=1) + "\n")
        time.sleep(PAUSE)
    return log


def _empty(x) -> bool:
    return x is None or (isinstance(x, str) and x == "") or (not isinstance(x, str) and pd.isna(x))


def _same(a, b) -> bool:
    """Equal as the TSV holds them: an empty string and NaN are both an empty cell."""
    if _empty(a) or _empty(b):
        return _empty(a) and _empty(b)
    if isinstance(a, (int, float, np.integer, np.floating)) and \
            isinstance(b, (int, float, np.integer, np.floating)):
        return float(a) == float(b)
    return str(a) == str(b)


def build_v2(v1: pd.DataFrame, survey_v1: pd.DataFrame, metadata_v1: pd.DataFrame,
             survey_v2: pd.DataFrame, metadata_v2: pd.DataFrame,
             cache: Path = TABLES) -> tuple[pd.DataFrame, dict]:
    # 1. the annotation step, as v1 ran it, must give v1's five columns
    again = annotate(v1[COMPUTED], survey_v1, metadata_v1)
    mismatch = {c: int(sum(not _same(a, b) for a, b in zip(again[c], v1[c])))
                for c in ANNOTATED}
    record = {"annotation_reproduces_v1": mismatch}
    if any(mismatch.values()):
        return pd.DataFrame(), record
    # 2. the 43: candidates by error column with two or more SE-like columns
    cols = survey_v1.set_index("urn")["all_columns"].fillna("").astype(str)
    multi = v1[(v1["permits"] == "errors")
               & v1["urn"].map(lambda u: len(se_like(cols.get(u, "").split(";"))) >= 2)]
    log = fetch_tables(multi["urn"].tolist(), cache)
    computed = v1[COMPUTED].copy()
    choice = pd.Series("", index=v1.index, dtype=object)
    chosen = {}
    for i, r in multi.iterrows():
        f = cache / (r["urn"].replace(":", "_") + ".csv")
        if not f.exists():
            chosen[r["urn"]] = {"choice": "fetch_failed", "error": log.get(r["urn"])}
            continue
        df = pd.read_csv(f, low_memory=False)
        row, ch = from_table_v2(df, r["urn"], r["permits"], "",
                                se_like(cols.get(r["urn"], "").split(";")))
        for c in COMPUTED:
            computed.at[i, c] = row[c]
        choice.at[i] = ch["choice"]
        chosen[r["urn"]] = ch
    # the recomputed rows as the TSV will hold them (v1's file keeps 16 significant
    # digits), so that "revised" compares like with like
    computed = pd.read_csv(io.StringIO(computed.to_csv(sep="\t", index=False)), sep="\t")
    # 3. the annotation with the current rules
    platform = survey_v2.set_index("urn")["platform"]
    out = annotate(computed, survey_v2, metadata_v2, platform=platform)
    out["error_column_choice"] = choice.to_numpy()
    rev = []
    for i in out.index:
        what = []
        if any(not _same(out.at[i, c], v1.at[i, c]) for c in COMPUTED):
            what.append("error_column")
        if bool(out.at[i, "human"]) != bool(v1.at[i, "human"]):
            what.append("human")
        if out.at[i, "platform"] != v1.at[i, "platform"]:
            what.append("platform")
        rev.append(";".join(what))
    out["revised"] = rev
    record |= {"multi_se_deposits": int(len(multi)),
               "experiment_sets": int(multi["urn"].str.extract(r"urn:mavedb:(\d+)")[0].nunique()),
               "choices": {k: int(v) for k, v in choice[choice != ""].value_counts().items()},
               "per_deposit": chosen, "fetched": log}
    return out[COLUMNS_V2], record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey-all", action="store_true",
                    help="v1's run: fetch and compute every candidate of --survey")
    ap.add_argument("--survey", default="results/mavedb_survey_v1.tsv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    if not args.survey_all:
        import json
        from atlas import paths
        res = paths.RESULTS
        v1 = pd.read_csv(res / "mavedb_ceilings_v1.tsv", sep="\t")
        t, rec = build_v2(v1, pd.read_csv(res / "mavedb_survey_v1.tsv", sep="\t"),
                          pd.read_csv(res / "mavedb_metadata_v1.tsv", sep="\t"),
                          pd.read_csv(paths.path("mavedb_survey"), sep="\t"),
                          pd.read_csv(paths.path("mavedb_metadata"), sep="\t"))
        if t.empty:
            print("the annotation step does not reproduce v1:", rec, file=sys.stderr)
            return 1
        out = Path(args.out) if args.out else paths.path("mavedb_ceilings")
        out.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(out, sep="\t", index=False)
        (TABLES / "build_record.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
        print(f"wrote {out.name} ({len(t)} rows); annotation reproduces v1: "
              f"{rec['annotation_reproduces_v1']}; the {rec['multi_se_deposits']} deposits: "
              f"{rec['choices']}; revised: {t['revised'].replace('', 'none').value_counts().to_dict()}")
        return 0

    s = pd.read_csv(args.survey, sep="\t").fillna("")
    cand = s[s["permits"].isin(["replicates", "errors"])]
    if args.limit:
        cand = cand.head(args.limit)
    print(f"candidates: {len(cand):,}", file=sys.stderr)

    rows = []
    for i, (_, r) in enumerate(cand.iterrows(), 1):
        rows.append(one(r["urn"], r["permits"], r.get("reconciling_replicates", "")))
        if i % 50 == 0:
            print(f"  {i}/{len(cand)}", file=sys.stderr)
        time.sleep(PAUSE)

    d = pd.DataFrame(rows)
    out = Path(args.out or "results/mavedb_ceilings_v1.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, sep="\t", index=False)

    n = len(d)
    print(f"\nwrote {out}  ({n:,} deposits)")
    for st in ("ok", "degenerate", "suspicious", "insufficient"):
        k = int((d["status"] == st).sum())
        print(f"  {st:<14} {k:>6,}  ({k / n:5.1%})")
    ok = d[d["status"] == "ok"]["ceiling"]
    if len(ok):
        q = ok.quantile([0, .25, .5, .75, 1]).round(3).tolist()
        print(f"\n  ceiling five-number: min {q[0]}  Q1 {q[1]}  median {q[2]}  "
              f"Q3 {q[3]}  max {q[4]}")
        below = int((ok < 0.45).sum())
        print(f"  below the 0.45 correction bound: {below:,} / {len(ok):,} "
              f"({below / len(ok):.1%})")
    print("\n  rules used:", d[d.status == "ok"]["method"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
