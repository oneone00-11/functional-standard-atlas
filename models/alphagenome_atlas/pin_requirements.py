"""requirements.txt for the AlphaGenome Atlas environment, pinned with hashes after the run
(CONVENTIONS.md rule 3), as the companion release pins its scoring environments
(scripts/stage_E17/pin_requirements.py).

Run with the environment's own interpreter, after score.py has run:
    ~/work/envs/alphagenome-atlas/bin/python models/alphagenome_atlas/pin_requirements.py

For every package of `pip freeze`, the file pip picks on this machine (a wheel where there
is one) is downloaded and hashed, and the downloaded wheel's RECORD is compared with the
installed distribution's: the same files with the same hashes, so the hash pins what is
installed and not merely a file of the same name. Writes requirements.txt
(`name==version --hash=sha256:...`) and requirements.check.json beside it.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import platform
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "requirements.txt"
CHECK = HERE / "requirements.check.json"


def run(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def tilde(path: str) -> str:
    """A path with the home directory written as ~, so the record names no local account."""
    home = str(Path.home()) + "/"
    return "~/" + path[len(home):] if path.startswith(home) else path


def record_of_wheel(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(".dist-info/RECORD"))
        rows = csv.reader(io.StringIO(z.read(name).decode()))
        return {r[0]: r[1] for r in rows if r and r[1]}


def record_installed(dist_name: str) -> dict[str, str]:
    d = metadata.distribution(dist_name)
    return {str(f): f"{f.hash.mode}={f.hash.value}" for f in d.files or [] if f.hash}


def main() -> None:
    freeze = sorted(x for x in run(sys.executable, "-m", "pip", "freeze").split() if x)
    files, problems = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for spec in freeze:
            dest = Path(tmp) / spec
            dest.mkdir(parents=True)
            r = subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--quiet",
                                "--disable-pip-version-check", "--prefer-binary", "--dest",
                                str(dest), spec], capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(f"{spec}: pip download failed\n{r.stderr[-800:]}")
            got = list(dest.iterdir())
            if len(got) != 1:
                raise SystemExit(f"{spec}: downloaded {[g.name for g in got]}")
            f = got[0]
            rec = {"spec": spec, "file": f.name, "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                   "kind": "wheel" if f.suffix == ".whl" else "sdist"}
            if rec["kind"] == "wheel":
                theirs = record_of_wheel(f)
                ours = record_installed(spec.split("==")[0])
                common = set(theirs) & set(ours)
                same = sum(theirs[k] == ours[k] for k in common)
                rec |= {"record_files": len(theirs), "installed_files_hashed": len(ours),
                        "compared": len(common), "identical": same}
                if same != len(common) or not common:
                    problems.append(spec)
            files.append(rec)
    header = ["# The AlphaGenome Atlas environment of models/alphagenome_atlas/score.py, read with",
              "# pip freeze after the run (CONVENTIONS.md rule 3), each with the sha256 of the file",
              "# pip picks here. Written by pin_requirements.py; do not edit by hand. Install with",
              "# pip install --require-hashes --no-deps -r requirements.txt (Python "
              f"{platform.python_version()})."]
    OUT.write_text("\n".join(header + [f"{r['spec']} --hash=sha256:{r['sha256']}" for r in files])
                   + "\n")
    CHECK.write_text(json.dumps({
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "interpreter": tilde(sys.executable), "python": platform.python_version(),
        "platform": platform.platform(), "packages": len(files),
        "wheels_whose_record_matches_the_installed_files": sum(
            1 for r in files if r["kind"] == "wheel" and r["spec"] not in problems),
        "problems": problems, "files": files}, indent=1) + "\n")
    print(f"{len(files)} packages -> {OUT.name}; RECORD mismatches: {problems or 'none'}")
    if problems:
        raise SystemExit("an installed package is not the wheel downloaded: stopping")


if __name__ == "__main__":
    main()
