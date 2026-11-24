"""ClinVar record status for every variant of the frozen matrix, annotated by the atlas.

v5's "ClinVar recorded" stratum read data/external/clinvar_recorded_ids.parquet, the
companion benchmark's old analysis set (21,394 variant ids; all 3,893 of BRCA1 among
them), which no longer matches any definition in the companion release. Here the atlas
annotates its own 64,178 variants with the release's ClinVar VCF (GRCh38 weekly,
fileDate 2026-06-15) and the release's rules, so that the two papers mean the same thing
by an arm:

  * a variant is recorded when the VCF has a record at its exact chrom, pos, ref and alt
    (no review-status filter; no normalisation: the VCF is left-aligned, and a matrix
    deletion anchored on the base before it may not be, so non-SNVs are matched as they
    stand and the number matched is reported);
  * CLNSIG is collapsed to P/LP, B/LB, VUS, Conflicting or Other;
  * arm: classified (P/LP or B/LB), recorded_unclassified, unrecorded; arm_strict also
    files a record with an empty CLNSIG as recorded_no_assertion.

The four functions below are the release's, ported as written and not imported
(phase1/src/evid_build_set.py: annotate_clinvar, clinvar_arm, clinvar_arm_strict;
scripts/config.py: classify_clnsig). One line differs: annotate_clinvar's
`classify = _load_clnsig_classifier()`, which loaded scripts/config.py by path, is
`classify = classify_clnsig`. tests/test_clinvar_status.py compares the sources.

Subcommands (from the repository root, PYTHONPATH=src):

  fetch     copy the VCF from the companion working tree (it is not in git:
            phase1/data/evidence/clinvar/clinvar_20260615.vcf.gz) to
            data/refs/clinvar/, and check it: sha256 equal to the value the release's
            analysis_set_v1.manifest.json records (read with git show), md5 equal to
            NCBI's (the .md5 file beside it); NCBI's public copies looked up by their
            headers only (HTTP HEAD for the VCF, the .md5 file read; nothing downloaded),
            the first whose size and md5 are the copy's recorded; and the three release
            files the rules and the check come from, with git show, into
            data/raw/companion_v3.1.0/
  annotate  results/clinvar_status_v1.parquet (variant_id, variation_id, clnsig_raw,
            clnsig_class, clnrevstat, arm, arm_strict), its provenance record, and
            results/clinvar_status_counts_v1.tsv: the number of variants by gene x
            stratum x variant class x arm x arm_strict, the number of non-SNVs matched,
            and the comparison with the old list (counts only). Stops unless the
            release analysis set's 8,853 SNVs get its clinvar_arm and
            clinvar_arm_strict row by row.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from atlas.evaluate import classify_region
from atlas.external_scores import COMPANION_DIR, DOI, TAG, TAG_COMMIT, _git, companion_repo
from atlas.manifest import register_file, sha256_file

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
FROZEN = REPO / "data" / "frozen" / "frozen-matrix-v1.parquet"
OLD_LIST = REPO / "data" / "external" / "clinvar_recorded_ids.parquet"
VCF_NAME = "clinvar_20260615.vcf.gz"
VCF_RELEASE_PATH = f"phase1/data/evidence/clinvar/{VCF_NAME}"
VCF = REPO / "data" / "refs" / "clinvar" / VCF_NAME
VCF_SHA256 = "10d86b892aae1f035e1950844e13fb039dad50be1087a0d1445c60d29191a342"
VCF_MD5 = "e19ba3d834e402f65cdc42293dcceace"
FILE_DATE = "2026-06-15"
NCBI_URLS = (                  # the archive first: the weekly folder is not kept for ever
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2026/" + VCF_NAME,
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly/" + VCF_NAME,
)

MANIFEST_REL = "phase1/data/evidence/analysis_set_v1.manifest.json"
ANALYSIS_SET = "phase1/data/evidence/analysis_set_v1.parquet"
RULE_FILES = ("phase1/src/evid_build_set.py", "scripts/config.py")
FILES = RULE_FILES + (MANIFEST_REL,)

OUT_TABLE = RESULTS / "clinvar_status_v1.parquet"
OUT_PROV = RESULTS / "clinvar_status_v1.provenance.json"
OUT_SUMMARY = RESULTS / "clinvar_status_counts_v1.tsv"   # a name the artefact manifest takes
COLUMNS = ["variant_id", "variation_id", "clnsig_raw", "clnsig_class", "clnrevstat",
           "arm", "arm_strict"]
ARMS = ["classified", "recorded_unclassified", "unrecorded"]
ARMS_STRICT = ["classified", "recorded_unclassified", "recorded_no_assertion", "unrecorded"]
STRATA = ("coding_or_utr", "splice_1_2", "splice_3_10", "splice_11_50", "splice_deep")
EXPECTED_RELEASE_SET = 8853


# ==========================================================================
# the release's rules, ported as written (see the module docstring)
# ==========================================================================
def classify_clnsig(clnsig: str) -> str:
    """Collapse a raw CLNSIG value into one of:
    P/LP, B/LB, VUS, Conflicting, Other.
    """
    if not clnsig:
        return "Other"
    s = clnsig.lower()
    # Conflicting first — it can co-occur with other tokens.
    if "conflicting" in s:
        return "Conflicting"
    if "pathogenic" in s and "benign" not in s:
        # Pathogenic, Likely_pathogenic, Pathogenic/Likely_pathogenic
        return "P/LP"
    if "benign" in s and "pathogenic" not in s:
        return "B/LB"
    if "uncertain_significance" in s or s == "uncertain_significance":
        return "VUS"
    return "Other"


def annotate_clinvar(keys: set[tuple[str, int, str, str]], vcf: Path) -> pd.DataFrame:
    """One pass over the ClinVar VCF, keeping the records our variants match.

    A variant is `recorded` if the release carries a record at its exact
    chrom/pos/ref/alt. The clinical-significance field is collapsed by the
    project's own classify_clnsig, so `classified` means P/LP or B/LB and
    `recorded_unclassified` means VUS, Conflicting or Other.
    """
    classify = classify_clnsig
    wanted_chroms = {k[0] for k in keys}
    rows, file_date = [], None
    with gzip.open(vcf, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("##"):
                if line.startswith("##fileDate"):
                    file_date = line.strip().split("=", 1)[1]
                continue
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t", 8)
            if p[0] not in wanted_chroms:
                continue
            key = (p[0], int(p[1]), p[3], p[4])
            if key not in keys:
                continue
            info = dict(kv.split("=", 1) for kv in p[7].split(";") if "=" in kv)
            rows.append({
                "chrom": p[0], "pos": int(p[1]), "ref": p[3], "alt": p[4],
                "variation_id": p[2],
                "clnsig_raw": info.get("CLNSIG", ""),
                "clnsig_class": classify(info.get("CLNSIG", "")),
                "clnrevstat": info.get("CLNREVSTAT", ""),
            })
    out = pd.DataFrame(rows)
    out.attrs["file_date"] = file_date
    return out


def clinvar_arm(cls: str | float) -> str:
    if not isinstance(cls, str) or cls == "":
        return "unrecorded"
    return "classified" if cls in ("P/LP", "B/LB") else "recorded_unclassified"


def clinvar_arm_strict(cls: str | float, clnsig_raw: str | float) -> str:
    """The same three arms, except that a record carrying NO clinical assertion at
    all is separated out.

    classify_clnsig maps an empty CLNSIG to "Other", which clinvar_arm then files
    under recorded_unclassified -- so an allele-only ClinVar entry, which nobody has
    ever assessed, sits in the same arm as a variant ClinVar looked at and could not
    call. On this set 588 rows are such records and 576 of them are BRCA1, which is
    85% of BRCA1's middle arm; separating them moves that arm's likelihood ratio by
    a full evidence tier. The three arms the brief specifies stay as they are and
    this runs beside them.
    """
    if not isinstance(cls, str) or cls == "":
        return "unrecorded"
    if cls in ("P/LP", "B/LB"):
        return "classified"
    if not isinstance(clnsig_raw, str) or clnsig_raw.strip() == "":
        return "recorded_no_assertion"
    return "recorded_unclassified"


# ==========================================================================
# the VCF and the release's files
# ==========================================================================
def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def release_vcf_sha256() -> str:
    """The VCF's sha256 as the release's analysis-set manifest records it."""
    return json.loads(_git("show", f"{TAG}:{MANIFEST_REL}"))["sources"]["clinvar_vcf"]["sha256"]


def fetch(dest: Path = VCF, out: Path = COMPANION_DIR) -> dict:
    """Copy the VCF (read only, from the companion working tree) and the release files;
    stop unless both hashes are right."""
    commit = _git("rev-parse", f"{TAG}^{{commit}}").decode().strip()
    if commit != TAG_COMMIT:
        raise SystemExit(f"{TAG} points to {commit}, not {TAG_COMMIT}")
    for rel in FILES:
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_git("show", f"{TAG}:{rel}"))
        register_file(out / "MANIFEST.json", path, release_path=rel, tag=TAG,
                      commit=TAG_COMMIT, doi=DOI, read_with=f"git show {TAG}:{rel}")
    src = companion_repo() / VCF_RELEASE_PATH
    want_sha = release_vcf_sha256()
    if want_sha != VCF_SHA256:
        raise SystemExit(f"the release manifest records {want_sha}, not {VCF_SHA256}")
    ncbi_md5 = (src.parent / (src.name + ".md5")).read_text().split()[0]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or sha256_file(dest) != VCF_SHA256:
        shutil.copyfile(src, dest)
    got = {"sha256": sha256_file(dest), "md5": md5_file(dest)}
    if got["sha256"] != VCF_SHA256 or got["md5"] != VCF_MD5 or ncbi_md5 != VCF_MD5:
        raise SystemExit(f"the copied VCF is not the release's: {got}, NCBI's md5 {ncbi_md5}")
    rec = {"source": f"<companion working tree>/{VCF_RELEASE_PATH}",
           "copied_to": str(dest.relative_to(REPO)) if dest.is_relative_to(REPO) else str(dest),
           "bytes": dest.stat().st_size, **got,
           "sha256_in_release_manifest": want_sha, "md5_in_ncbi_md5_file": ncbi_md5,
           "release_files": list(FILES), "tag": TAG, "commit": commit, "doi": DOI,
           "public_url": public_url(dest.stat().st_size, got["md5"])}
    (dest.parent / (dest.name + ".copy.json")).write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def public_url(size: int, md5: str, urls=NCBI_URLS, timeout: float = 30.0) -> dict:
    """NCBI's public copies of the VCF, by their headers only (the VCF is not
    downloaded): each URL's status, size and Last-Modified, and the md5 in the .md5 file
    beside it. A URL counts when its size and md5 are the copy's."""
    import urllib.request
    head = {"User-Agent": "atlas-check/1.0"}
    found, checked = None, []
    for url in urls:
        rec = {"url": url}
        try:
            req = urllib.request.Request(url, method="HEAD", headers=head)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                rec |= {"status": r.status, "content_length": int(r.headers["Content-Length"]),
                        "last_modified": r.headers.get("Last-Modified")}
            with urllib.request.urlopen(urllib.request.Request(url + ".md5", headers=head),
                                        timeout=timeout) as r:
                rec["md5"] = r.read(1000).decode().split()[0]
        except Exception as e:                                  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
        rec["same_file"] = rec.get("content_length") == size and rec.get("md5") == md5
        checked.append(rec)
        found = found or (url if rec["same_file"] else None)
    from datetime import datetime, timezone
    return {"url": found, "note": None if found else "没有找到 (not found)",
            "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "checked": checked, "how": "HTTP HEAD for the VCF; the .md5 file read"}


# ==========================================================================
# the annotation
# ==========================================================================
def variant_class(ref: str, alt: str) -> str:
    return "snv" if len(ref) == 1 and len(alt) == 1 else "non_snv"


def annotate(matrix: pd.DataFrame, vcf: Path = VCF) -> tuple[pd.DataFrame, str | None]:
    """One row per variant of the matrix, in its order: the matched record's fields
    (empty when there is none) and the two arms."""
    m = matrix[["variant_id", "chrom", "pos", "ref", "alt"]].copy()
    m["chrom"], m["pos"] = m["chrom"].astype(str), m["pos"].astype(int)
    keys = set(zip(m["chrom"], m["pos"], m["ref"], m["alt"]))
    cv = annotate_clinvar(keys, vcf)
    file_date = cv.attrs.get("file_date")
    if cv.empty:
        cv = pd.DataFrame(columns=["chrom", "pos", "ref", "alt", "variation_id",
                                   "clnsig_raw", "clnsig_class", "clnrevstat"])
    dup = cv.duplicated(["chrom", "pos", "ref", "alt"], keep=False)
    if dup.any():
        raise SystemExit(f"{int(dup.sum())} VCF records share a chrom/pos/ref/alt with "
                         "another: one variant would get two rows")
    t = m.merge(cv, on=["chrom", "pos", "ref", "alt"], how="left")
    t["arm"] = t["clnsig_class"].map(clinvar_arm)
    t["arm_strict"] = [clinvar_arm_strict(c, r) for c, r in zip(t["clnsig_class"],
                                                                 t["clnsig_raw"])]
    if len(t) != len(m) or not (t["variant_id"].to_numpy() == m["variant_id"].to_numpy()).all():
        raise SystemExit("the annotation changed the matrix's rows")
    return t, file_date


def release_check(t: pd.DataFrame, analysis_set: pd.DataFrame) -> dict:
    """The release analysis set's rows: clinvar_arm and clinvar_arm_strict, row by row."""
    a = analysis_set.set_index("variant_id")
    ours = t.set_index("variant_id").reindex(a.index)
    out = {"n": int(len(a)), "missing_from_matrix": int(ours["arm"].isna().sum())}
    for col, rel in (("arm", "clinvar_arm"), ("arm_strict", "clinvar_arm_strict")):
        same = ours[col].to_numpy() == a[rel].astype(str).to_numpy()
        out[col] = {"against": rel, "identical": int(same.sum()),
                    "differing": a.index[~same].tolist()[:20]}
    for col in ("variation_id", "clnsig_raw", "clnsig_class", "clnrevstat"):
        x = ours[col].fillna("").astype(str).to_numpy()
        y = a[col].fillna("").astype(str).to_numpy()
        out[f"{col}_identical"] = int((x == y).sum())
    out["holds"] = (out["missing_from_matrix"] == 0
                    and all(out[c]["identical"] == out["n"] for c in ("arm", "arm_strict")))
    return out


def summary(t: pd.DataFrame, matrix: pd.DataFrame, old_ids: set[str]) -> pd.DataFrame:
    """Counts: by gene x stratum x variant class x arm x arm_strict ('arms'), the
    non-SNVs matched ('non_snv_matched'), and against the old list ('old_list')."""
    d = t[["variant_id", "arm", "arm_strict"]].merge(
        matrix[["variant_id", "gene", "hgvs_c", "ref", "alt"]], on="variant_id")
    d["stratum"] = d["hgvs_c"].map(classify_region)
    d["variant_class"] = [variant_class(r, a) for r, a in zip(d["ref"], d["alt"])]
    d["recorded"] = d["arm"] != "unrecorded"
    d["in_old_list"] = d["variant_id"].isin(old_ids)
    rows = []
    g = (d.groupby(["gene", "stratum", "variant_class", "arm", "arm_strict"], sort=True)
         .size().reset_index(name="n"))
    rows += [{"part": "arms", **r} for r in g.to_dict("records")]
    for (gene, vc), p in list(d.groupby(["gene", "variant_class"], sort=True)) + [
            (("all", "all"), d)]:
        rows.append({"part": "matched", "gene": gene, "stratum": "all", "variant_class": vc,
                     "arm": "recorded", "arm_strict": "", "n": int(p["recorded"].sum()),
                     "of": int(len(p))})
    for gene, p in list(d.groupby("gene", sort=True)) + [("all", d)]:
        for what, mask in (("old_list_now_unrecorded", p["in_old_list"] & ~p["recorded"]),
                           ("recorded_not_in_old_list", ~p["in_old_list"] & p["recorded"]),
                           ("old_list_and_recorded", p["in_old_list"] & p["recorded"]),
                           ("old_list", p["in_old_list"])):
            rows.append({"part": "old_list", "gene": gene, "stratum": "all",
                         "variant_class": "all", "arm": what, "arm_strict": "",
                         "n": int(mask.sum()), "of": int(len(p))})
    return pd.DataFrame(rows, columns=["part", "gene", "stratum", "variant_class", "arm",
                                       "arm_strict", "n", "of"])


def run(vcf: Path = VCF) -> dict:
    if not vcf.exists():
        raise SystemExit(f"{vcf} missing; run `python -m atlas.clinvar_status fetch`")
    copy = json.loads((vcf.parent / (vcf.name + ".copy.json")).read_text())
    if sha256_file(vcf) != VCF_SHA256:
        raise SystemExit("the VCF is not the release's")
    matrix = pd.read_parquet(FROZEN)
    t, file_date = annotate(matrix, vcf)
    if file_date != FILE_DATE:
        raise SystemExit(f"the VCF's fileDate is {file_date}, not {FILE_DATE}")
    s = pd.read_parquet(COMPANION_DIR / ANALYSIS_SET)
    if len(s) != EXPECTED_RELEASE_SET:
        raise SystemExit(f"the release analysis set has {len(s)} rows")
    check = release_check(t, s)
    old_ids = set(pd.read_parquet(OLD_LIST)["variant_id"])
    summ = summary(t, matrix, old_ids)
    RESULTS.mkdir(exist_ok=True)
    out = t[COLUMNS].copy()
    out.to_parquet(OUT_TABLE, index=False)
    summ.to_csv(OUT_SUMMARY, sep="\t", index=False)
    prov = {
        "written_by": "atlas.clinvar_status annotate",
        "output": str(OUT_TABLE.relative_to(REPO)), "output_sha256": sha256_file(OUT_TABLE),
        "counts": str(OUT_SUMMARY.relative_to(REPO)), "counts_sha256": sha256_file(OUT_SUMMARY),
        "rows": int(len(out)),
        "vcf": {"path": copy["copied_to"], "sha256": copy["sha256"], "md5": copy["md5"],
                "bytes": copy["bytes"], "file_date": file_date,
                "release": "ClinVar GRCh38 weekly VCF, fileDate 2026-06-15",
                "sha256_in_release_manifest": copy["sha256_in_release_manifest"],
                "md5_in_ncbi_md5_file": copy["md5_in_ncbi_md5_file"],
                "copied_from": copy["source"], "public_url": copy["public_url"]},
        "rules": {"source": {rel: sha256_file(COMPANION_DIR / rel) for rel in RULE_FILES},
                  "functions": {"annotate_clinvar": "phase1/src/evid_build_set.py",
                                "clinvar_arm": "phase1/src/evid_build_set.py",
                                "clinvar_arm_strict": "phase1/src/evid_build_set.py",
                                "classify_clnsig": "scripts/config.py"},
                  "ported": "as written; annotate_clinvar's classifier is the ported "
                            "classify_clnsig rather than scripts/config.py loaded by path",
                  "match": "chrom, pos, ref and alt all equal; no review-status filter; "
                           "no normalisation of non-SNVs"},
        "n_by_arm": {"arm": out["arm"].value_counts().to_dict(),
                   "arm_strict": out["arm_strict"].value_counts().to_dict(),
                   "recorded": int((out["arm"] != "unrecorded").sum())},
        "release_check": check,
        "old_list": {"file": str(OLD_LIST.relative_to(REPO)), "sha256": sha256_file(OLD_LIST),
                     "ids": len(old_ids)},
        "release": {"tag": TAG, "commit": TAG_COMMIT, "doi": DOI},
    }
    OUT_PROV.write_text(json.dumps(prov, indent=1, ensure_ascii=False, default=str) + "\n")
    print(f"wrote {OUT_TABLE.relative_to(REPO)} ({len(out)} rows), "
          f"{OUT_SUMMARY.relative_to(REPO)}; release check "
          f"{'holds' if check['holds'] else 'DOES NOT hold'}")
    if not check["holds"]:
        raise SystemExit("the release analysis set's arms differ: stopping")
    return prov


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "annotate"])
    args = ap.parse_args(argv)
    if args.command == "fetch":
        print(json.dumps(fetch(), indent=1))
    else:
        run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
