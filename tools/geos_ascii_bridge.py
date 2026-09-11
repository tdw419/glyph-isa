"""GH-24 S1: Spatial Observation Bridge — ASCII world projection.

Pure-function projector from GlyphCPUv2 memory (16,384 words) to a
fixed-layout ASCII canvas. Word addresses map to 2D via
tools/geos_hilbert.py:hilbert_d2xy_true (N=128) — one import shares the
coordinate space with tools/mkv_infinite_map.py instead of a parallel
build, per the GH-24 roadmap line.

MONITORING INVARIANT: reads only committed bus state passed in by the
caller — never mid-task speculative registers. Non-invasive and
deterministic by construction; read-only, zero GPU, VCC-trivial.

DUAL-ARTIFACT RULE: the canvas is NEVER JSON-wrapped (fixed-stride
string preserves 2D adjacency for attention); JSON only as the sidecar
metadata — named boxes `{name, rect, memory_words, bus_lane, writable}`
+ tick.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_REPO = Path(__file__).resolve().parent
# Works both from scratch staging and from tools/ once landed.
for _p in (_REPO, *_REPO.parents):
    if (_p / "tools" / "geos_hilbert.py").exists():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from tools.geos_hilbert import hilbert_d2xy_true  # noqa: E402

# 128 x 128 = 16,384 words = GlyphRunner ram_words.
N = 128

# ── Landed ABI box legend (GH-18; mirrors tests/test_gh18_syscall_abi.py) ──
# rect is the inclusive word-address range [lo, hi].
BOXES: List[Dict] = [
    {"name": "KERNEL",   "rect": (0, 699),     "bus_lane": "kernel", "writable": False},
    {"name": "BOX0",     "rect": (700, 717),   "bus_lane": "task_a", "writable": True},
    {"name": "BOX1",     "rect": (718, 735),   "bus_lane": "task_b", "writable": True},
    {"name": "BOX2",     "rect": (736, 767),   "bus_lane": "tile_abi", "writable": True},
    {"name": "STATUS",   "rect": (950, 967),   "bus_lane": "kernel", "writable": False},
    {"name": "TABLE",    "rect": (1568, 1583), "bus_lane": "tile_abi", "writable": True},
    {"name": "TILE_RECT", "rect": (1600, 1695), "bus_lane": "tile_abi", "writable": True},
    {"name": "DATA",     "rect": (768, 949),   "bus_lane": "kernel", "writable": True},
]

# Character per region for lit (non-zero) words. Markers (below) override.
_REGION_CHAR = {
    "KERNEL": "K", "BOX0": "A", "BOX1": "B", "BOX2": "S",
    "STATUS": "V", "TABLE": "T", "TILE_RECT": "x", "DATA": "D",
}

# ABI-word markers: a lit word at this address renders as this char.
# argv 750 / result 754 (BOX2 tile-ABI), exit word 703 (BOX0).
MARKERS: Dict[int, str] = {750: ">", 754: "@", 703: "X"}

EMPTY = "."


def _region_of(word: int) -> str:
    for box in BOXES:
        lo, hi = box["rect"]
        if lo <= word <= hi:
            return _REGION_CHAR[box["name"]]
    return "D"


def project(memory: Sequence[int], origin: Tuple[int, int] = (0, 0),
            w: int = 80, h: int = 25) -> str:
    """Project a committed memory image to a fixed-stride ASCII canvas.

    Exactly h rows of exactly w characters, newline-joined, no trailing
    newline. A cell renders EMPTY if the word is zero (unlit); otherwise
    its Hilbert (x, y) cell gets the region char, overridden by MARKERS
    for named ABI words. Words whose Hilbert cell falls outside the
    origin..origin+(w,h) viewport are clipped.
    """
    ox, oy = origin
    rows: List[List[str]] = [[EMPTY] * w for _ in range(h)]
    for word in range(min(len(memory), N * N)):
        val = int(memory[word]) & 0xFFFFFFFF
        if val == 0:
            continue
        x, y = hilbert_d2xy_true(N, word)
        cx, cy = x - ox, y - oy
        if 0 <= cx < w and 0 <= cy < h:
            rows[cy][cx] = MARKERS.get(word) or _region_of(word)
    return "\n".join("".join(row) for row in rows)


def project_metadata(memory: Sequence[int], origin: Tuple[int, int] = (0, 0),
                     w: int = 80, h: int = 25, tick: int = 0) -> Dict:
    """JSON sidecar for project(): named boxes + tick, per the roadmap
    dual-artifact rule. Canvas itself is never wrapped in JSON."""
    ox, oy = origin

    def _lit(lo: int, hi: int) -> int:
        top = min(hi, len(memory) - 1)
        return sum(1 for wd in range(lo, top + 1)
                   if (int(memory[wd]) & 0xFFFFFFFF) != 0)

    boxes = []
    for box in BOXES:
        lo, hi = box["rect"]
        entry = dict(box)
        entry["memory_words"] = _lit(lo, hi)
        boxes.append(entry)
    return {
        "tick": tick,
        "canvas_rect": {"origin": list(origin), "w": w, "h": h},
        "n": N,
        "boxes": boxes,
    }


def project_surface(memory: Sequence[int], origin: Tuple[int, int] = (0, 0),
                    w: int = 80, h: int = 25, tick: int = 0):
    """Convenience: (canvas_str, metadata_dict)."""
    return (project(memory, origin=origin, w=w, h=h),
            project_metadata(memory, origin=origin, w=w, h=h, tick=tick))
