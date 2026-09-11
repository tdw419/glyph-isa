"""GH-24 S1 golden test: tests/test_gh24_ascii_bridge.py.

Verifies, per the roadmap GH-24 gate:
  1. Fixed-stride invariant — exactly w chars per row, h rows.
  2. Pure-function determinism — identical memory => byte-identical canvas.
  3. Dual-artifact rule — canvas is raw text; metadata is JSON sidecar.
  4. Golden receipt — projects a REAL GH-18 admit receipt (executed live
     via the committed test harness helpers) and asserts exact canvas
     bytes + ABI-matched box legend (BOX2 [736..768), argv 750, result
     754, table 1568..1583).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "tools"), str(_REPO / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.geos_ascii_bridge import (      # noqa: E402
    BOXES, MARKERS, N, project, project_metadata, project_surface,
)
from tools.geos_hilbert import hilbert_xy2d_true  # noqa: E402


# ── 1. fixed-stride invariant ───────────────────────────────────────────

def test_s1_fixed_stride_invariant():
    mem = [0] * (N * N)
    mem[750] = 6
    mem[754] = 18
    for (w, h) in [(80, 25), (40, 10), (128, 128), (17, 9)]:
        canvas = project(mem, w=w, h=h)
        rows = canvas.split("\n")
        assert len(rows) == h
        assert all(len(r) == w for r in rows)


def test_s1_hilbert_addresses_are_canonical():
    # word -> (x,y) and its inverse must round-trip through the SAME
    # implementation the bridge imports (geos_hilbert, not a parallel).
    import tools.geos_hilbert as gh
    for word in (700, 703, 710, 718, 736, 750, 754, 1568, 1570, 1583, 1600):
        x, y = gh.hilbert_d2xy_true(N, word)
        assert hilbert_xy2d_true(N, x, y) == word


# ── 2. pure-function determinism ────────────────────────────────────────

def test_s1_pure_function_determinism():
    mem = [0] * (N * N)
    mem[703] = 0xFEED0006
    mem[750] = 6
    mem[754] = 18
    mem[1570] = 0x1E0004
    c1 = project(mem)
    c2 = project(list(mem))          # fresh copy, same values
    assert c1 == c2
    # input memory must not be mutated by projection
    assert mem[750] == 6 and mem[754] == 18


# ── 3. dual-artifact rule ───────────────────────────────────────────────

def test_s1_dual_artifact_canvas_raw_metadata_json():
    mem = [0] * (N * N)
    mem[754] = 18
    canvas, meta = project_surface(mem, tick=7)
    assert isinstance(canvas, str)
    assert "\x1b" not in canvas and "{" not in canvas   # raw text, no JSON/ANSI
    import json
    m = json.loads(json.dumps(meta))                    # JSON-serializable
    assert m["tick"] == 7
    by_name = {b["name"]: b for b in m["boxes"]}
    assert tuple(by_name["BOX2"]["rect"]) == (736, 767)
    assert tuple(by_name["TABLE"]["rect"]) == (1568, 1583)
    assert set(by_name["TABLE"]) >= {"name", "rect", "memory_words", "bus_lane", "writable"}


def test_s1_unlit_words_render_empty():
    mem = [0] * (N * N)
    canvas = project(mem)
    assert set(canvas) == {".", "\n"}


# ── 4. golden receipt: REAL GH-18 admit run ─────────────────────────────

def _real_gh18_receipt():
    import test_gh18_syscall_abi as T
    with tempfile.TemporaryDirectory() as d:
        runner, _ = T._run(Path(d), mode="admit", name="admit.npy")
        seeds = dict.fromkeys(
            range(T.GH18_TABLE_WORD, T.GH18_TABLE_WORD + T.GH18_NSLOTS), 0)
        T._seed_admit_tile(runner, seeds)
        receipt = runner.drive(seeds=seeds, max_instructions=60000)
    assert receipt["halted"] is True and receipt["faulted"] is False
    return receipt["memory"]


def test_s1_golden_gh18_receipt_exact_canvas_bytes():
    mem = _real_gh18_receipt()
    # landed ABI first: the projected state IS the GH-18 receipt
    assert mem[750] == 6            # ARGV_IN
    assert mem[754] == 18           # MAILBOX_RESULT: triple(6)
    assert mem[703] == 0xFEED0006   # exit word
    assert mem[950] == 0xCAFE0018   # kernel status
    assert mem[952] == 0x00020018   # ABI version
    assert mem[1570] == 0x1E0004    # table slot 2, live tile PC

    canvas = project(mem)
    rows = canvas.split("\n")
    assert len(rows) == 25 and all(len(r) == 80 for r in rows)

    # marker cells at their canonical Hilbert coordinates
    import tools.geos_hilbert as gh
    def _cell(word):
        x, y = gh.hilbert_d2xy_true(N, word)
        return rows[y][x]
    assert _cell(750) == MARKERS[750]   # '>'
    assert _cell(754) == MARKERS[754]   # '@'
    assert _cell(703) == MARKERS[703]   # 'X'
    assert _cell(1570) == "T"           # lit table slot
    # slot 2 (1570) lit, but neighbours (1568,1583) unlit — they seeded 0
    x68, y68 = gh.hilbert_d2xy_true(N, 1568)
    assert rows[y68][x68] == "."

    # determinism of the golden artifact itself
    assert project(list(mem)) == canvas

    # golden-bytes assertion against the committed snapshot
    golden_path = Path(__file__).parent / "fixtures" / "gh24_s1_golden_canvas.txt"
    assert golden_path.exists(), "golden canvas fixture missing"
    assert canvas == golden_path.read_text(), "golden canvas drifted"
