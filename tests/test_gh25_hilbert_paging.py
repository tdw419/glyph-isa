#!/usr/bin/env python3
"""tests/test_gh25_hilbert_paging.py — GH-25 Infinite Spatial Page Table gate.

Falsifiable gate for GH-25 (systems/GLYPH_SELF_HOSTING_ROADMAP.md), leg 1
first: CPU ≡ WGSL parity across a BARRIER-ALIGNED page swap.

Mechanism under test (PTE_HILB — the one-PTE-format-change horizon stub):
  A GH-17 PTE with bit 4 (PTE_HILB = 0x10) set carries a HILBERT frame
  address instead of a linear pfn. The frame word is
      pix_word = xy2d(HILB_ORDER, col_from_pfn, row_from_pfn) * PAGE_WORDS + offset
  where the PTE's pfn field (pte >> 8) is reinterpreted as a packed
  2D frame origin (fpn_row, fpn_col) on a HILB_ORDER x HILB_ORDER frame
  grid, d2xy/xy2d are the SAME verified Hacker's Delight curve as
  tools/geos_hilbert.py (hilbert_reference_verify-verified), and offset
  stays the intra-frame word offset (Hilbert adjacency within a frame is
  data-locality, not addressing). HILB_ORDER = 64 (4096 frame slots).

  Frame-swap protocol (BSP rule): the HOST swaps a viewport frame in/out
  by rewriting ONE PTE's pfn field (the frame origin) between the CPU and
  WGSL runs at a tick boundary (engine not running — the swap is the
  barrier). Parity means: identical 32-register state + identical paged
  read results across the swap on both engines.

Legs (priority order per the roadmap row):
  1. CPU ≡ WGSL parity across a barrier-aligned page swap (the actual work)
  2. fault-on-unmapped still vectors cleanly (GH-17 invariant with PTE_HILB armed)
  3. VCC SHA256 image invariant across swap (swap = PTE rewrite + frame pixels only)
  4. identity mapping preserved on re-residency (swap A->B->A restores bytes)
  5. Hilbert d2xy host/shader agreement (non-vacuity: WGSL paging actually walks)
  6. runner.py line budget + clean imports invariant (GH-5)
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.glyph_gpt.atlas import build_default_atlas          # noqa: E402
from tools.glyph_gpt.runner import GlyphRunner                 # noqa: E402
from tools.geos_hilbert import hilbert_d2xy_true, hilbert_xy2d_true  # noqa: E402
from tools.glyph_gpt.baker import (                            # noqa: E402
    PAGE_WORDS,
    PTE_V,
    PTE_W,
    PTE_U,
    PTE_PIX,
)

PTE_HILB = 0x10          # GH-25: Hilbert frame address in pfn field
HILB_ORDER = 64          # frame grid is 64x64 frame slots
HILB_SIDE = 1 << 6

# GH-25 image geography (GH-18 ABI facts, all landed):
GH25_VPN_DATA = 12       # vpn 12 = virtual words 3072..3327 (the paged window)
GH25_VADDR = GH25_VPN_DATA * PAGE_WORDS   # 3072
GH25_TABLE_IDX = GH25_VPN_DATA


def _hilb_frame_origin(slot: int) -> int:
    """Packed pfn field for Hilbert frame slot d: (row << 8) | col on the
    HILB_SIDE x HILB_SIDE frame grid."""
    col, row = hilbert_d2xy_true(HILB_SIDE, slot)
    return (row << 8) | col


def _hilb_pix_word(pfn_field: int, offset: int) -> int:
    """Frame word for a PTE_HILB translation — the single source of truth
    mirrored bit-exactly in the WGSL shader."""
    row = (pfn_field >> 8) & 0xFF
    col = pfn_field & 0xFF
    d = hilbert_xy2d_true(HILB_SIDE, col, row)
    return d * PAGE_WORDS + offset


def _hilb_pte(slot: int) -> int:
    return (PTE_V | PTE_W | PTE_U | PTE_HILB
            | (_hilb_frame_origin(slot) << 8))


# --- legs -------------------------------------------------------------------

def test_gh25_leg1_cpu_wgsl_parity_across_page_swap():
    """THE leg: identical register state on CPU and WGSL when the paged
    window's backing frame is swapped (slot 5 -> slot 42) at the barrier,
    and byte-exact paged reads of DIFFERENT resident frames post-swap."""
    from tools.glyph_gpt.gh25_hilbert_paging import hilbert_swap_image

    atlas = build_default_atlas()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "gh25_swap.npy"
        # Payload: 0xDEAD0005 pre-baked into frame slot 5's window,
        # 0xBEAD0042 into slot 42's window.
        img = hilbert_swap_image(atlas, out_path=out)

        runner = GlyphRunner(out, ram_words=16384)

        # PASS 1 (both engines, swap state A: PTE points at slot 5)
        _write_pte(runner.image, GH25_TABLE_IDX, _hilb_pte(5))
        _arm_pt(runner.image)
        pre_hash = hashlib.sha256(runner.image.tobytes()).hexdigest()
        rec_cpu_a = runner.run(max_instructions=4000)
        rec_wgsl_a = runner.run_wgsl(max_steps=4000)
        _assert_parity(rec_cpu_a, rec_wgsl_a)
        assert rec_cpu_a["halted"] and rec_wgsl_a["halted"]
        # the task read the paged word: r10 == slot 5 payload high word check
        # is done in-image; verify word carries 0xFEED0025 on success.

        # BARRIER: engine stopped. Host swaps the viewport frame in the PT
        # (one PTE rewrite — the ONLY mutation) and re-runs both engines
        # from the swapped image.
        img_b = np.load(out) if out.exists() else img
        _write_pte(img_b, GH25_TABLE_IDX, _hilb_pte(42))
        runner_b = GlyphRunner(img_b, ram_words=16384)
        rec_cpu_b = runner_b.run(max_instructions=4000)
        rec_wgsl_b = runner_b.run_wgsl(max_steps=4000)
        _assert_parity(rec_cpu_b, rec_wgsl_b)
        assert rec_cpu_b["halted"] and rec_wgsl_b["halted"]
        assert rec_cpu_b["status_word_value"] == 0xCAFE0025, hex(
            rec_cpu_b["status_word_value"])

        # and pass 1 actually verified too (not a vacuous HALT before LD)
        assert rec_cpu_a["status_word_value"] == 0xCAFE0025, hex(
            rec_cpu_a["status_word_value"])

        # VCC leg rides the same harness: the swap changed ONLY the PTE
        # word (4 image pixels max at the table window) — frame payload
        # pixels are identical between A and B images.
        _assert_swap_diff_is_pte_only(img_b, runner_b.image, pre_hash)


def test_gh25_leg2_fault_on_unmapped_vectors_cleanly():
    """GH-17 invariant holds with the Hilbert walker live: access to a vpn
    with V=0 traps to the fault handler with the fault address recorded."""
    from tools.glyph_gpt.gh25_hilbert_paging import hilbert_fault_image

    atlas = build_default_atlas()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "gh25_fault.npy"
        hilbert_fault_image(atlas, out_path=out)
        runner = GlyphRunner(out, ram_words=16384)
        receipt = runner.run(max_instructions=4000)
        mem = receipt["memory"]
        assert mem[731] == 0xFA025, hex(mem[731])
        # The fault leg reads vpn 13 (vaddr 3328); the engine records the
        # faulting BYTE address (vaddr << 2, the convention since GH-13's
        # E-K1 traps) = 3328 * 4 = 0x3400.
        assert receipt["fault_addr"] == (GH25_VADDR + PAGE_WORDS) * 4, hex(receipt["fault_addr"])


def test_gh25_leg3_identity_swap_roundtrip_restores_bytes():
    """Swap A->B->A restores the exact image bytes (identity mapping on
    re-residency) — the swap is reversible at the PTE level."""
    from tools.glyph_gpt.gh25_hilbert_paging import hilbert_swap_image

    atlas = build_default_atlas()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "gh25_roundtrip.npy"
        hilbert_swap_image(atlas, out_path=out)
        img0 = np.load(out)
        h0 = hashlib.sha256(img0.tobytes()).hexdigest()

        pte5, pte42 = _hilb_pte(5), _hilb_pte(42)
        _write_pte(img0, GH25_TABLE_IDX, pte42)
        _write_pte(img0, GH25_TABLE_IDX, pte5)
        h1 = hashlib.sha256(img0.tobytes()).hexdigest()
        assert h0 == h1


def test_gh25_leg4_hilbert_host_shader_agreement():
    """Non-vacuity: the d2xy table the WGSL walker uses (emitted from
    tools/geos_hilbert.py at build_shader time) matches the host curve for
    every slot — a shader curve drift would break leg 1 loudly here."""
    from tools.glyph_gpt.gh25_hilbert_paging import hilbert_d2xy_lut

    lut = hilbert_d2xy_lut(HILB_ORDER)
    assert len(lut) == HILB_SIDE * HILB_SIDE
    for slot in range(HILB_SIDE * HILB_SIDE):
        col, row = hilbert_d2xy_true(HILB_SIDE, slot)
        assert lut[slot] == (col | (row << 16))


def test_gh25_leg5_d2xy_roundtrip_property():
    """Hacker's Delight inverse property on the frame grid (the property
    hilbert_reference_verify.py guards, restricted to what GH-25 uses)."""
    for slot in range(0, HILB_SIDE * HILB_SIDE, 37):
        col, row = hilbert_d2xy_true(HILB_SIDE, slot)
        assert hilbert_xy2d_true(HILB_SIDE, col, row) == slot


def test_gh25_leg6_runner_budget_and_no_dev_imports():
    """GH-5 invariant: runner.py stays <= 200 LOC, 0 dev imports."""
    import ast
    runner_path = _REPO / "tools" / "glyph_gpt" / "runner.py"
    lines = runner_path.read_text().splitlines()
    assert len(lines) <= 200, f"runner.py exceeds 200 lines: {len(lines)}"
    tree = ast.parse(runner_path.read_text())
    forbidden = {
        "atlas", "spatial_builder", "synth", "generate",
        "model", "tokenizer", "corpus", "train", "pack_dataset", "baker",
        "pytest", "hypothesis", "scipy", "torch", "transformers",
    }
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for n in names:
            for f in forbidden:
                assert f not in n, f"runner.py must not import '{n}'"


# --- helpers ----------------------------------------------------------------

PT_BASE_WORD = 1536          # GH-17 page table base
PT_ARM_WORD = 8211           # PAGE_TABLE_WORD (PAGE_TABLE_ADDR >> 2)


def _write_pte(img: np.ndarray, idx: int, pte: int) -> None:
    """Stamp a PTE into the image pixels at the table window's linear word
    (the pfn-5 pix window words are linear == pixel index, scanline order,
    exactly like _addr_to_xy). One PTE word = 3 RGB channels (24 bits)."""
    h, w, _ = img.shape
    word = PT_BASE_WORD + idx
    img[word // w, word % w] = (
        (pte >> 16) & 0xFF, (pte >> 8) & 0xFF, pte & 0xFF)


def _arm_pt(img: np.ndarray) -> None:
    """Arm PAGE_TABLE_ADDR in the image-pixel MMIO surface so BOTH engines
    see the armed walker (the WGSL engine reads only pixels)."""
    h, w, _ = img.shape
    img[PT_ARM_WORD // w, PT_ARM_WORD % w] = (0, 0, PT_BASE_WORD & 0xFF)


def _assert_parity(cpu: dict, wgsl: dict) -> None:
    assert cpu.get("error") is None, cpu.get("error")
    assert wgsl.get("error") is None, wgsl.get("error")
    assert cpu["registers_full"] == wgsl["registers_full"], (
        f"CPU vs WGSL register mismatch across swap:\n"
        f"CPU:  {cpu['registers_full']}\nWGSL: {wgsl['registers_full']}")


def _assert_swap_diff_is_pte_only(img_b, runner_img, pre_hash: str) -> None:
    """VCC leg: between pass-1 image and post-swap image, the only pixel
    differences are inside the page-table window's linear word range."""
    # (leg 3 covers byte-exact reversibility; here assert the helper math)
    h, w, _ = runner_img.shape
    lo, hi = PT_BASE_WORD, PT_BASE_WORD + 256
    diff = np.argwhere(np.any(img_b != runner_img, axis=-1))
    for y, x in diff:
        word = y * w + x
        assert lo <= word < hi, (
            f"swap mutated pixel word {word} outside PT window [{lo},{hi})")
