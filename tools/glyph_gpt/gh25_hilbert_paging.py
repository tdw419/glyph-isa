"""tools/glyph_gpt/gh25_hilbert_paging.py — GH-25 Infinite Spatial Page Table.

Horizon stub per systems/GLYPH_SELF_HOSTING_ROADMAP.md GH-25: ONE PTE
format change (PTE_HILB) + the barrier-aligned swap harness. NOT a new
subsystem.

PTE_HILB (bit 4, 0x10):
    When set, the PTE's pfn field (pte >> 8) no longer carries a linear
    frame number. It carries a PACKED 2D FRAME ORIGIN (fpn_row << 8 |
    fpn_col) on a HILB_SIDE x HILB_SIDE grid of PAGE_WORDS-sized frames.
    The physical frame word is

        pix_word = xy2d(HILB_SIDE, fpn_col, fpn_row) * PAGE_WORDS + offset

    where xy2d/d2xy are the verified Hacker's Delight curve from
    tools/geos_hilbert.py (same implementation verified by
    hilbert_reference_verify.py and shared with tools/mkv_infinite_map.py's
    storage map — one curve, two zoom levels, TWO-TABLE RULE respected).

    The intra-frame word offset stays LINEAR: Hilbert order governs frame
    placement (viewport adjacency = frame-slot adjacency in 2D), not bytes
    within a frame.

    GlyphCPUv2 mirrors this bit-exactly (tools/glyph_isa_v2.py, GH-25
    block in the LD/ST page-walk paths); the WGSL shader receives the same
    rule plus a d2xy LUT emitted by hilbert_d2xy_lut() — the LUT is
    generated from tools.geos_hilbert at shader build time, so the two
    engines cannot drift.

BARRIER-ALIGNED SWAP PROTOCOL (the parity harness contract):
    Frame swaps happen ONLY while both engines are stopped (the swap IS
    the barrier — a discrete state transition, per the BSP rule). The
    host rewrites one PTE word in the image pixels and re-runs. The gate
    is tests/test_gh25_hilbert_paging.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent.parent
for _p in (str(_REPO), str(_REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.geos_hilbert import hilbert_d2xy_true  # noqa: E402
from tools.glyph_isa_v2 import (                  # noqa: E402
    KFAULT_PC_ADDR,
    MODE_LATCH_ADDR,
    PAGE_TABLE_WORD,
    PAGE_TABLE_BASE_WORD,
    PAGE_WORDS,
    PTE_V,
    PTE_W,
    PTE_U,
)
PTE_HILB = 0x10
HILB_SIDE = 64            # frame grid is 64x64 frame slots

# --- image geography (GH-17/18 landed ABI facts; nothing legacy moves) ------
GH25_VPN_DATA = 12              # paged window = virtual words 3072..3327
GH25_VADDR = GH25_VPN_DATA * PAGE_WORDS
GH25_STATUS_WORD = 950
GH25_KERNEL_OK = 0xCAFE0025     # 0xCAFE << 16 | 0x25 (gate tail id)
GH25_FAULT_WORD = 731
GH25_FLAG_WORD = 717            # nonzero-read flag (non-vacuity marker; 732 is GH-16's
                                # TICKS_COUNT receipt word — do not collide)
GH25_FAULT_SEEN = 0xFA025       # verdict: faulted in GH-25 image (fits 24-bit pixel)
GH25_PAYLOAD_SLOT5 = 0x0AD0005  # baked into slot-5 frame word 0 (24-bit pixel-safe)
GH25_PAYLOAD_SLOT42 = 0x0AD0042  # baked into slot-42 frame word 0


def pack_const(v: int) -> List[str]:
    """Glyph lines building v (32-bit) in r14: HI<<16 | LO, LDI-safe."""
    hi, lo = (v >> 16) & 0xFFFF, v & 0xFFFF
    lines = [f"LDI r14 {lo}"]
    if hi:
        lines += [f"LDI r13 {hi}", "LDI r4 16", "SHL r13 r4", "OR r14 r13"]
    return lines


def hilbert_frame_pix_word(pfn_field: int, offset: int) -> int:
    """THE translation rule — mirrored bit-exactly in the WGSL shader."""
    row = (pfn_field >> 8) & 0xFF
    col = pfn_field & 0xFF
    # xy2d inline (full-grid reflection, bitwise): must equal
    # tools.geos_hilbert.hilbert_xy2d_true(HILB_SIDE, col, row).
    d = 0
    s = HILB_SIDE >> 1
    x, y = col, row
    while s > 0:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = HILB_SIDE - 1 - x
                y = HILB_SIDE - 1 - y
            x, y = y, x
        s >>= 1
    return d * PAGE_WORDS + offset


def hilbert_frame_pte(slot: int) -> int:
    """PTE for Hilbert frame slot d (d2xy gives the packed frame origin)."""
    col, row = hilbert_d2xy_true(HILB_SIDE, slot)
    pfn_field = (row << 8) | col
    return PTE_V | PTE_W | PTE_U | PTE_HILB | (pfn_field << 8)


def hilbert_d2xy_lut(side: int = HILB_SIDE) -> np.ndarray:
    """u32 LUT[slot] = (col & 0xFFFF) | (row << 16) for every frame slot —
    emitted from tools.geos_hilbert at shader build time so the WGSL walker
    provably uses the SAME curve as the host (gate leg 4)."""
    lut = np.zeros(side * side, dtype=np.uint32)
    for slot in range(side * side):
        col, row = hilbert_d2xy_true(side, slot)
        lut[slot] = (col & 0xFFFF) | (row << 16)
    return lut


# --- glyph assembly ---------------------------------------------------------
# Minimal self-contained kernel (GH-17 paged-kernel pattern, NOT the GH-18
# admit kernel — GH-25 needs a paged task, not a syscall task):
#   1. SUPER: zero receipt words, identity-map vpn 0..3 + the PT window
#      (vpn 6), write the vpn-12 PTE (V|W|U|HILB, slot-5 frame origin),
#      arm PAGE_TABLE_ADDR, latch USER, KJMP task A.
#   2. Task A (USER): LD r10 <- [3072] (the paged window), set the
#      nonzero-read flag, JMP :__kdone.
#   3. :__kdone: status = 0xCAFE0025, HALT.
# The fault mode maps vpn 12 (Hilbert) and faults on UNMAPPED vpn 13.
#
# NOTE (GH-23 receipt, dbg_gh23_brk2): once the walker is armed, later STs
# to the PT region itself walk the live table — every PT write here happens
# BEFORE the arming store (the GH-20 pre-arming loop discipline).


def hilbert_paging_program_text(mode: str = "swap",
                                task_a_pc: int = 0,
                                kfault_pc: int = 0) -> str:
    a: List[str] = []
    add = a.append
    add(":__entry")
    add("JMP :__kmain")
    add(":__kmain")
    # ---- zero receipt words ----
    add(f"LDI r15 {GH25_STATUS_WORD}")
    add("LDI r14 0")
    add("ST r15 r14")
    add(f"LDI r15 {GH25_FAULT_WORD}")
    add("LDI r14 0")
    add("ST r15 r14")
    add(f"LDI r15 {GH25_FLAG_WORD}")
    add("LDI r14 0")
    add("ST r15 r14")
    # ---- identity map vpn 0..3 ----
    for p in range(4):
        val = PTE_V | PTE_W | PTE_U | (p << 8)
        add(f"LDI r15 {PAGE_TABLE_BASE_WORD + p}")
        add(f"LDI r14 {val}")
        add("ST r15 r14")
    # ---- page-table window itself (vpn 6, identity) ----
    add(f"LDI r15 {PAGE_TABLE_BASE_WORD + 6}")
    add(f"LDI r14 {PTE_V | PTE_W | (6 << 8)}")
    add("ST r15 r14")
    # ---- vpn 12 PTE: NOT written by the kernel — the BAKED IMAGE carries
    #      the pass-A PTE (slot 5, stamped in _two_pass_bake). The parity
    #      harness swaps frames by rewriting THIS PTE word between passes
    #      (the barrier); a kernel-side write would clobber the swap.
    # ---- arm the walker LAST (pre-arming discipline, see note) ----
    add(f"LDI r15 {PAGE_TABLE_WORD}")
    add(f"LDI r14 {PAGE_TABLE_BASE_WORD}")
    add("ST r15 r14")
    # ---- vectors ----
    if mode == "fault":
        add(f"LDI r15 {KFAULT_PC_ADDR >> 2}")
        add(f"LDI r14 {kfault_pc}")
        add("ST r15 r14")
    # ---- enter task A in USER mode ----
    add(f"LDI r15 {MODE_LATCH_ADDR >> 2}")
    add("LDI r14 1")
    add("ST r15 r14")
    add(f"LDI r30 {task_a_pc}")
    add("KJMP r30")
    add("JMP :__kdone")

    # ---- task A (USER) ----
    add(":__task_a")
    if mode == "swap":
        # r10 <- [3072] through the Hilbert walker
        add(f"LDI r15 {GH25_VADDR}")
        add("LD r10 r15")
        # non-vacuity: flag word 717 = 1 iff the paged read returned the
        # frame payload (a dead walker reads 0 / zero padding and leaves
        # the flag at 0). LD reads the 24-bit pixel word.
        add(f"LDI r15 {GH25_FLAG_WORD}")
        add(f"LDI r14 {GH25_PAYLOAD_SLOT5 & 0xFFFFFF}")
        add("CMP r10 r14")
        add("JZ :__flag_set")
        add("JMP :__kdone")        # mismatch: flag stays 0
        add(":__flag_set")
        add("LDI r14 1")
        add("ST r15 r14")
        add("JMP :__kdone")
    else:  # fault: read vpn 13 -> unmapped -> trap to :__kfault
        add(f"LDI r15 {GH25_VADDR + PAGE_WORDS}")
        add("LD r10 r15")
        add("JMP :__kdone")

    # ---- fault handler ----
    add(":__kfault")
    add(f"LDI r15 {GH25_FAULT_WORD}")
    add(f"LDI r14 {GH25_FAULT_SEEN}")
    add("ST r15 r14")
    add("JMP :__kdone")

    # ---- done: status = 0xCAFE0025 ----
    add(":__kdone")
    add("LDI r3 51966")
    add("LDI r4 16")
    add("SHL r3 r4")
    add("LDI r9 37")
    add("ADD r3 r9")
    add(f"LDI r15 {GH25_STATUS_WORD}")
    add("ST r15 r3")
    add("HALT")
    return "\n".join(a) + "\n"


def _frame_payload(word0: int) -> np.ndarray:
    """One PAGE_WORDS frame: word 0 = word0, rest zero."""
    frame = np.zeros((PAGE_WORDS, 3), dtype=np.uint8)
    frame[0] = ((word0 >> 16) & 0xFF, (word0 >> 8) & 0xFF, word0 & 0xFF)
    return frame


def _stamp_frame(img: np.ndarray, slot: int, word0: int) -> None:
    """Write a frame payload at the Hilbert slot's frame origin (image
    pixels, scanline order — the same linear layout _addr_to_xy uses)."""
    col, row = hilbert_d2xy_true(HILB_SIDE, slot)
    base = hilbert_frame_pix_word((row << 8) | col, 0)
    h, w, _ = img.shape
    payload = _frame_payload(word0)
    for i in range(PAGE_WORDS):
        word = base + i
        if word < h * w:
            img[word // w, word % w] = payload[i]


def _write_table_word(img: np.ndarray, idx: int, pte: int) -> None:
    """Stamp a PTE word into the image's page-table window (one 24-bit
    pixel per word, scanline order)."""
    h, w, _ = img.shape
    word = PAGE_TABLE_BASE_WORD + idx
    img[word // w, word % w] = (
        (pte >> 16) & 0xFF, (pte >> 8) & 0xFF, pte & 0xFF)


def _two_pass_bake(mode: str, out_path: Optional[Union[str, Path]],
                   cols_instrs: int, min_rows: int) -> np.ndarray:
    """Pass 1 fixes the packed pixel PCs (:__task_a/:__kfault), pass 2
    bakes the final image (GH-17 paged_kernel_image discipline)."""
    from rv64i_to_glyph import assemble_glyph_to_pixels
    from tools.glyph_gpt.baker import bake_image

    _, coords1 = assemble_glyph_to_pixels(
        hilbert_paging_program_text(mode), cols_instrs=cols_instrs,
        min_rows=min_rows)

    def packed(label: str) -> int:
        col, row = coords1[label]
        return (col & 0xFFFF) | ((row & 0xFFFF) << 16)

    txt = hilbert_paging_program_text(
        mode, task_a_pc=packed(":__task_a"), kfault_pc=packed(":__kfault"))
    img = bake_image(txt, atlas=None, cols_instrs=cols_instrs,
                     min_rows=min_rows, out_path=None)
    # stamp the Hilbert frames AFTER the bake (they live in the zero
    # padding BELOW the text rows — min_rows guards the height: slot 42's
    # frame word base = xy2d(64,(5,2))*256 + 255 = 11007, so the image must
    # be at least 344 rows tall at 32 px/row; 360 gives headroom).
    if mode == "swap":
        _stamp_frame(img, 5, GH25_PAYLOAD_SLOT5)
        _stamp_frame(img, 42, GH25_PAYLOAD_SLOT42)
        # pass-A PTE: stamp the vpn-12 Hilbert PTE (slot 5) into the image's
        # page-table window — the kernel does NOT write this word, the image
        # is the pass-A state and the harness rewrites it for pass B.
        _write_table_word(img, GH25_VPN_DATA, hilbert_frame_pte(5))
    if out_path is not None:
        p = Path(out_path)
        if p.suffix == ".npy":
            np.save(p, img)
        elif p.suffix == ".npz":
            np.savez(p, image=img)
        else:
            from PIL import Image
            Image.fromarray(img.astype(np.uint8)).save(p)
    return img


def hilbert_swap_image(atlas, out_path: Optional[Union[str, Path]] = None,
                       cols_instrs: int = 8, min_rows: int = 360) -> np.ndarray:
    """Bake the GH-25 swap-parity image: kernel + slot-5 frame (0x0AD0005)
    + slot-42 frame (0x0AD0042) at their Hilbert frame origins."""
    return _two_pass_bake("swap", out_path, cols_instrs, min_rows)


def hilbert_fault_image(atlas, out_path: Optional[Union[str, Path]] = None,
                        cols_instrs: int = 8, min_rows: int = 360) -> np.ndarray:
    """Bake the GH-25 fault-leg image: task A reads unmapped vpn 13."""
    return _two_pass_bake("fault", out_path, cols_instrs, min_rows)
