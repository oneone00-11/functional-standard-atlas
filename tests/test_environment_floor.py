"""Keep the declared Python floor in step with what the pins actually need.

A clean extract of the v2.3.0 archive could not be installed at all. `make setup`
ran a bare `python3`, which is 3.9 on macOS, and pip then reported "no matching
distribution found for pandas==3.0.5" -- true for 3.9, and pointing at entirely
the wrong thing. The real constraint is numpy 2.5.1 and scipy 1.18.0, both of
which declare Requires-Python >= 3.12.

The archive exists so a reviewer can rebuild the paper. An archive that cannot
install fails that at the first step, so the floor is now checked by `make
setup` and asserted here.

If a pin is raised or lowered, MIN_PY in the Makefile, the README note and
REQUIRED below all move together.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REQUIRED = (3, 12)


def test_running_interpreter_meets_the_floor():
    assert sys.version_info[:2] >= REQUIRED, (
        f"running Python {sys.version_info.major}.{sys.version_info.minor}; "
        f"the pinned dependencies need >= {REQUIRED[0]}.{REQUIRED[1]}")


def test_makefile_declares_and_guards_the_floor():
    mk = (REPO / "Makefile").read_text()
    m = re.search(r"^MIN_PY\s*:?=\s*(\d+)\.(\d+)", mk, re.M)
    assert m, "Makefile no longer declares MIN_PY; `make setup` would fail obscurely again"
    assert (int(m.group(1)), int(m.group(2))) == REQUIRED, (
        f"Makefile MIN_PY is {m.group(0)}, this test expects "
        f"{REQUIRED[0]}.{REQUIRED[1]}")
    assert "check-python" in mk and "setup: check-python" in mk, (
        "`setup` no longer depends on check-python, so an old interpreter would "
        "reach pip and produce a misleading resolver error")


def test_pinned_deps_do_not_exceed_the_declared_floor():
    """The floor must cover every pin, not just the two we know about."""
    md = pytest.importorskip("importlib.metadata")
    req = (REPO / "requirements.txt").read_text()
    names = [ln.split("==")[0].strip() for ln in req.splitlines()
             if "==" in ln and not ln.strip().startswith("#")]
    worst, worst_pkg = REQUIRED, None
    for name in names:
        try:
            spec = md.metadata(name)["Requires-Python"]
        except Exception:
            continue                      # not installed here; nothing to check
        if not spec:
            continue
        m = re.search(r">=\s*(\d+)\.(\d+)", spec)
        if m:
            v = (int(m.group(1)), int(m.group(2)))
            if v > worst:
                worst, worst_pkg = v, name
    assert worst <= REQUIRED, (
        f"{worst_pkg} requires Python >= {worst[0]}.{worst[1]}, above the "
        f"declared floor {REQUIRED[0]}.{REQUIRED[1]}. Raise MIN_PY in the "
        "Makefile, the README note and REQUIRED here together.")
