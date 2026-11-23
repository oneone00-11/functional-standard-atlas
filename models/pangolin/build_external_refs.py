"""Reference sequence and gene database for the external genes: chromosomes 1, 7, 16, 17.

The core genes were scored on data/refs/grch38_subset.fa and grch38_subset.gtf.db
(Ensembl release 112, chromosomes 2, 3, 13, 16 and 17; see NOTES.md). The external genes
lie on 1 (SFPQ), 7 (XRCC2, SBDS), 16 (CTCF) and 17 (RAD51D). This builds new files by the
same recipe and leaves grch38_subset.* untouched:

  grch38_external.fa (+ .fai)  the release-112 chromosome FASTAs 1, 7, 16 and 17,
                               decompressed and concatenated in that order
  grch38_external.gtf          the rows of the release-112 GTF on those chromosomes,
                               in the GTF's order
  grch38_external.gtf.db       gffutils.create_db(gtf, db, force=True, keep_order=True,
                               merge_strategy="create_unique", disable_infer_genes=True,
                               disable_infer_transcripts=True), the recipe of NOTES.md

Chromosomes 1 and 7 are downloaded from Ensembl when absent; 16 and 17, and the GTF, are
the files already in data/refs/. Checks, written with the hashes to
data/refs/grch38_external.build.json; any failure stops the build:
  - the GTF archive is the one the companion release's gene databases came from;
  - decompressed, chromosomes 1 and 7 equal byte for byte the FASTAs on which the
    companion release scored its 3-50 bp variants (sha256 from its run log);
  - chromosomes 16 and 17 equal byte for byte their records in grch38_subset.fa;
  - the GTF rows of chromosomes 1 and 7 equal the release's chr1.gtf and chr7.gtf.

Run with the Pangolin environment (gffutils, pyfaidx):
    <pangolin venv>/bin/python models/pangolin/build_external_refs.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
REPO = MODEL_DIR.parents[1]
REFS = REPO / "data" / "refs"
CHROMS = ("1", "7", "16", "17")
FASTA_GZ = "Homo_sapiens.GRCh38.dna.chromosome.{c}.fa.gz"
ENSEMBL = "https://ftp.ensembl.org/pub/release-112/fasta/homo_sapiens/dna/" + FASTA_GZ
GTF_GZ = REFS / "Homo_sapiens.GRCh38.112.gtf.gz"
SUBSET = REFS / "grch38_subset.fa"
OUT_FA = REFS / "grch38_external.fa"
OUT_GTF = REFS / "grch38_external.gtf"
OUT_DB = REFS / "grch38_external.gtf.db"
RECORD = REFS / "grch38_external.build.json"

# What the companion release scored its 3-50 bp variants on, read from its run logs
# (copied by `python -m atlas.external_scores companion`), never transcribed.
RUN_LOGS = REPO / "data/raw/companion_v3.1.0/scripts/stage_E18/run_log"
SPLICEAI_LOG = RUN_LOGS / "run_spliceai_20261023T082635Z.json"
REFS_LOG = RUN_LOGS / "build_pangolin_refs_20261023T044027Z.json"
CHUNK = 1 << 22


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(c: str) -> dict:
    dest = REFS / FASTA_GZ.format(c=c)
    url = ENSEMBL.format(c=c)
    fetched = False
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
        fetched = True
    return {"file": dest.name, "url": url, "downloaded_now": fetched,
            "bytes": dest.stat().st_size, "sha256": sha256(dest)}


def build_fasta() -> dict:
    """Decompress and concatenate; hash each chromosome's decompressed record."""
    per = {}
    tmp = OUT_FA.with_suffix(".fa.part")
    with open(tmp, "wb") as out:
        for c in CHROMS:
            h, n, header = hashlib.sha256(), 0, None
            with gzip.open(REFS / FASTA_GZ.format(c=c), "rb") as f:
                for block in iter(lambda: f.read(CHUNK), b""):
                    if header is None:
                        header = block.split(b"\n", 1)[0].decode()
                    h.update(block)
                    n += len(block)
                    out.write(block)
            per[c] = {"sha256": h.hexdigest(), "bytes": n, "header": header}
    tmp.rename(OUT_FA)
    import pyfaidx
    pyfaidx.Faidx(str(OUT_FA))                      # writes grch38_external.fa.fai
    return per


def subset_record(c: str, header: str, length: int) -> str:
    """sha256 of chromosome c's record (header and sequence) in grch38_subset.fa."""
    fai = {line.split("\t")[0]: int(line.split("\t")[2])
           for line in (SUBSET.with_suffix(".fa.fai")).read_text().splitlines()}
    start = fai[c] - len(header.encode()) - 1      # the header line ('>' included) and '\n'
    h = hashlib.sha256()
    with open(SUBSET, "rb") as f:
        f.seek(start)
        if f.read(1) != b">":
            raise SystemExit(f"chromosome {c}: no record header where the index says")
        f.seek(start)
        left = length
        while left:
            block = f.read(min(CHUNK, left))
            if not block:
                break
            h.update(block)
            left -= len(block)
    return h.hexdigest()


def build_gtf() -> dict:
    """The GTF's rows on CHROMS, in its order; hash each chromosome's rows."""
    hashes = {c: hashlib.sha256() for c in CHROMS}
    rows = {c: 0 for c in CHROMS}
    tmp = OUT_GTF.with_suffix(".gtf.part")
    with gzip.open(GTF_GZ, "rt") as f, open(tmp, "w") as out:
        for line in f:
            c = line.split("\t", 1)[0]
            if c in hashes:
                out.write(line)
                hashes[c].update(line.encode())
                rows[c] += 1
    tmp.rename(OUT_GTF)
    return {c: {"rows": rows[c], "sha256": hashes[c].hexdigest()} for c in CHROMS}


def build_db() -> None:
    import gffutils
    gffutils.create_db(str(OUT_GTF), str(OUT_DB), force=True, keep_order=True,
                       merge_strategy="create_unique", disable_infer_genes=True,
                       disable_infer_transcripts=True)


def release_hashes() -> tuple[dict, dict, str]:
    """{chrom: (sha256, bytes)} of the release's FASTAs, {chrom: sha256} of its GTF
    subsets, and the sha256 of the GTF archive, from the release's run logs."""
    def by_name(log: Path, key: str) -> dict:
        return {Path(x["path"]).name: x for x in json.loads(log.read_text())[key]}
    fa = by_name(SPLICEAI_LOG, "inputs")
    out = by_name(REFS_LOG, "outputs")
    gz = by_name(REFS_LOG, "inputs")[GTF_GZ.name]["sha256"]
    return ({c: (fa[f"chr{c}.fa"]["sha256"], fa[f"chr{c}.fa"]["bytes"]) for c in ("1", "7")},
            {c: out[f"chr{c}.gtf"]["sha256"] for c in ("1", "7")}, gz)


def main() -> int:
    started = _utc()
    RELEASE_FASTA, RELEASE_GTF, RELEASE_GTF_GZ = release_hashes()
    failures, rec = [], {"written_by": "models/pangolin/build_external_refs.py",
                         "started_utc": started, "chromosomes": list(CHROMS),
                         "release_hashes_from": [str(SPLICEAI_LOG.relative_to(REPO)),
                                                 str(REFS_LOG.relative_to(REPO))]}
    gtf_gz = sha256(GTF_GZ)
    rec["gtf_gz"] = {"file": GTF_GZ.name, "sha256": gtf_gz}
    if gtf_gz != RELEASE_GTF_GZ:
        failures.append("the GTF archive is not the companion release's")
    rec["fasta_gz"] = {c: download(c) for c in CHROMS}
    per = build_fasta()
    rec["fasta_records"] = per
    for c, (sha, n) in RELEASE_FASTA.items():
        same = per[c]["sha256"] == sha and per[c]["bytes"] == n
        rec.setdefault("checks", {})[f"chr{c}_equals_release_fasta"] = same
        if not same:
            failures.append(f"chromosome {c} differs from the release's FASTA")
    for c in ("16", "17"):
        same = subset_record(c, per[c]["header"], per[c]["bytes"]) == per[c]["sha256"]
        rec["checks"][f"chr{c}_equals_grch38_subset"] = same
        if not same:
            failures.append(f"chromosome {c} differs from grch38_subset.fa")
    gtf = build_gtf()
    rec["gtf_rows"] = gtf
    for c, sha in RELEASE_GTF.items():
        same = gtf[c]["sha256"] == sha
        rec["checks"][f"chr{c}_gtf_equals_release"] = same
        if not same:
            failures.append(f"the GTF rows of chromosome {c} differ from the release's")
    if failures:
        rec["failures"] = failures
        RECORD.write_text(json.dumps(rec, indent=1) + "\n")
        print("stopping:", failures, file=sys.stderr)
        return 1
    build_db()
    import gffutils
    import pyfaidx
    rec["outputs"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                      for p in (OUT_FA, OUT_FA.with_suffix(".fa.fai"), OUT_GTF, OUT_DB)}
    rec["versions"] = {"python": platform.python_version(), "gffutils": gffutils.__version__,
                       "pyfaidx": pyfaidx.__version__}
    rec["ended_utc"] = _utc()
    RECORD.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec["checks"]), "->", OUT_FA.name, OUT_GTF.name, OUT_DB.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
