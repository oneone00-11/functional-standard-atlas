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
