"""SHA-256 manifest helpers.

Every file under data/ must be registered in a MANIFEST.json next to it.
These helpers are the only sanctioned way to create or verify that record.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

CHUNK = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 of a file, streamed in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(manifest_path: str | Path) -> dict:
    p = Path(manifest_path)
    if not p.exists():
        return {"files": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def register_file(manifest_path: str | Path, file_path: str | Path, **meta) -> dict:
    """Hash file_path and add/update its entry in manifest_path.

    Extra keyword metadata (e.g. urn=..., title=...) is stored alongside the hash.
    Returns the updated manifest dict.
    """
    manifest_path = Path(manifest_path)
    file_path = Path(file_path)
    manifest = load_manifest(manifest_path)
    entry = {
        "sha256": sha256_file(file_path),
        "bytes": file_path.stat().st_size,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        **meta,
    }
    manifest.setdefault("files", {})[file_path.name] = entry
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def verify_manifest(manifest_path: str | Path) -> list[str]:
    """Re-hash every registered file; return a list of mismatch descriptions
    (empty list = all good)."""
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    problems: list[str] = []
    for name, entry in sorted(manifest.get("files", {}).items()):
        fp = manifest_path.parent / name
        if not fp.exists():
            problems.append(f"MISSING: {name}")
            continue
        actual = sha256_file(fp)
        if actual != entry["sha256"]:
            problems.append(f"HASH MISMATCH: {name}")
    return problems
