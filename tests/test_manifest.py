"""Guardrail tests: the hash manifest round-trips and detects tampering."""

import json

from atlas.manifest import register_file, sha256_file, verify_manifest


def test_register_and_verify_roundtrip(tmp_path):
    data = tmp_path / "scores.csv"
    data.write_bytes(b"variant,score\na,1.0\n")

    manifest_path = tmp_path / "MANIFEST.json"
    register_file(manifest_path, data, urn="urn:mavedb:test", title="toy")

    # Manifest records the correct hash.
    manifest = json.loads(manifest_path.read_text())
    assert manifest["files"]["scores.csv"]["sha256"] == sha256_file(data)
    assert manifest["files"]["scores.csv"]["urn"] == "urn:mavedb:test"

    # Clean state verifies with no problems.
    assert verify_manifest(manifest_path) == []


def test_verify_detects_tampering(tmp_path):
    data = tmp_path / "scores.csv"
    data.write_bytes(b"variant,score\na,1.0\n")
    manifest_path = tmp_path / "MANIFEST.json"
    register_file(manifest_path, data)

    # Tamper with the data file -> manifest must complain.
    data.write_bytes(b"variant,score\na,999.0\n")
    problems = verify_manifest(manifest_path)
    assert problems == ["HASH MISMATCH: scores.csv"]


def test_verify_detects_missing_file(tmp_path):
    data = tmp_path / "scores.csv"
    data.write_bytes(b"x")
    manifest_path = tmp_path / "MANIFEST.json"
    register_file(manifest_path, data)

    data.unlink()
    assert verify_manifest(manifest_path) == ["MISSING: scores.csv"]


def test_register_files_in_subdirectories_do_not_collide(tmp_path):
    """Regression: same-named files in different subdirs must stay separate."""
    for sub in ("set_a", "set_b"):
        d = tmp_path / sub
        d.mkdir()
        (d / "scores.csv").write_bytes(f"data-{sub}".encode())

    manifest_path = tmp_path / "MANIFEST.json"
    register_file(manifest_path, tmp_path / "set_a" / "scores.csv")
    register_file(manifest_path, tmp_path / "set_b" / "scores.csv")

    manifest = json.loads(manifest_path.read_text())
    assert set(manifest["files"]) == {"set_a/scores.csv", "set_b/scores.csv"}
    assert verify_manifest(manifest_path) == []
