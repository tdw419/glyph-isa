"""Single source of truth for Hilbert curve math used across tools/ and systems/.

Two incompatible variants exist in the codebase's history and are preserved
here under distinct names rather than unified, because unifying them would
silently corrupt already-encoded containers:

- hilbert_d2xy_true / hilbert_xy2d_true: the correct Hacker's Delight Hilbert
  curve (unit-step, used by vcc_engine_v2.py, spadsl.py, spadsl_semantic.py).
- hilbert_d2xy_legacy_qemu: a buggy variant that does NOT trace a valid
  Hilbert curve (it takes Chebyshev steps), but is baked into existing
  QEMU-captured MKV containers (qemu_capture_simple.py, qemu_to_mkv.py,
  vac3_demo.py). Do not "fix" this without a migration plan for existing data.
"""

from typing import Tuple


def hilbert_d2xy_true(n: int, d: int) -> Tuple[int, int]:
    x, y = 0, 0
    s = 1
    t = d
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def hilbert_xy2d_true(n: int, x: int, y: int) -> int:
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 & (x // s)
        ry = 1 & (y // s)
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x %= s
        y %= s
        s //= 2
    return d


def hilbert_d2xy_legacy_qemu(n: int, d: int) -> Tuple[int, int]:
    x, y = 0, 0
    s = 1
    while s < n:
        rx = (d >> 1) & 1
        ry = d & 1
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d >>= 2
        s <<= 1
    return x, y
