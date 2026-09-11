#!/usr/bin/env python3
"""GH-20: Pixel-FS v2 — bake the admit-mode Syscall ABI v2 kernel with a
statically seeded FSTAB (image IS the disk).

The FS v2 data plane lives in the GH-8b pixel-alias window [1024,1280)
(engine: glyph_isa_v2._fs_pix_write/_fs_pix_read — 2 px/word, lo24 +
hi8-BLUE). The bake seeds the FSTAB via the ENGINE-EXACT layout so the
image booted on any host already carries:

  slot 0 [1024,1032):  name='A', start=1040, len=2, in_use=1,
                       refcount=2, parent=0, next=0, rsvd=0
  slot 1 [1032,1040):  all zeros (free — fs_append must NOT clobber it)
  data   [1040,1048):  file A's extent words (two payload words)

Persistence contract (receipt: GH-8b write-through mirror + boot RAM
zeroing): the KERNEL boots with memory[] zeroed — a runtime-only seed
would die with the first drive()'s fresh CPU. Pixels are the persisted
truth; they are stamped at BAKE time, so every boot (first run, replay,
md5 fixpoint) reads the same FSTAB without host seeding.
"""
from __future__ import annotations

from typing import Any, Optional, Union
from pathlib import Path

import numpy as np

# ── FS v2 ABI constants (the finalized gate contract) ────────────────────
FSV2_FSTAB_WORD = 1024       # 2 slots x 8 words [1024,1040)
FSV2_NSLOTS = 2
FSV2_SLOT_WORDS = 8          # name,start,len,in_use,refcount,parent,next,rsvd
FSV2_DATA_WORD = 1040        # data extents [1040,1280)
FSV2_DATA_END = 1280
FSV2_N_APPEND = 10           # SYS 10 = fs_append
FSV2_N_RENAME = 11           # SYS 11 = fs_rename
FSV2_N_UNLINK = 12           # SYS 12 = fs_unlink

# static seed: file 'A' lives in slot 0 with a 2-word extent, shared
# (refcount 2 — the fs_unlink gate leg refuses on it)
FSV2_SEED_A_NAME = ord('A')
FSV2_SEED_A_START = FSV2_DATA_WORD
FSV2_SEED_A_LEN = 2
FSV2_SEED_A_IN_USE = 1
FSV2_SEED_A_REFCOUNT = 2     # the shared-file fixture (unlink must refuse)
FSV2_SEED_A_DATA = (0x11111111, 0x22222222)


def _gh20_seed_fstab() -> "dict[int, int]":
    """The static FSTAB the bake stamps into the fs pixel window."""
    seed: "dict[int, int]" = {}
    # slot 0
    seed[FSV2_FSTAB_WORD + 0] = FSV2_SEED_A_NAME
    seed[FSV2_FSTAB_WORD + 1] = FSV2_SEED_A_START
    seed[FSV2_FSTAB_WORD + 2] = FSV2_SEED_A_LEN
    seed[FSV2_FSTAB_WORD + 3] = FSV2_SEED_A_IN_USE
    seed[FSV2_FSTAB_WORD + 4] = FSV2_SEED_A_REFCOUNT
    seed[FSV2_FSTAB_WORD + 5] = 0       # parent
    seed[FSV2_FSTAB_WORD + 6] = 0       # next chain
    seed[FSV2_FSTAB_WORD + 7] = 0       # reserved
    # slot 1 stays zero (free)
    # file A's extent
    for i, w in enumerate(FSV2_SEED_A_DATA):
        seed[FSV2_DATA_WORD + i] = w
    return seed


def _gh20_bake_stamp(img: np.ndarray, words: "dict[int, int]") -> None:
    """Stamp fs-window words into the image with the ENGINE-EXACT alias
    layout (glyph_isa_v2.GlyphCPUv2._fs_pix_write): word w -> pixel
    (2w) = bits 23..0, pixel (2w+1) = bits 31..24 in BLUE. Runs AFTER
    syscall_abi_kernel_image() so the boot prologue cannot re-zero the
    seeded window (the admit-mode prologue does not touch [1024,1280),
    but stamping post-bake makes the invariant structural, not hopeful).
    """
    h, w, _ = img.shape
    for word, value in words.items():
        value &= 0xFFFFFFFF
        lo, hi = value & 0xFFFFFF, (value >> 24) & 0xFF
        lin = (word * 2) % (w * h)
        img[lin // w, lin % w] = ((lo >> 16) & 0xFF, (lo >> 8) & 0xFF, lo & 0xFF)
        lin2 = (word * 2 + 1) % (w * h)
        img[lin2 // w, lin2 % w] = (0, 0, hi)


def pixel_fs_v2_kernel_image(
    atlas: Any,
    mode: str = "admit",
    status_word: int = 950,
    timer_quantum: int = 35,
    cols_instrs: int = 8,
    min_rows: int = 16,
    out_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """GH-20: bake ONE image — the admit-mode Syscall ABI v2 kernel with
    the Pixel-FS v2 FSTAB statically seeded into the fs pixel window.

    The FS ops themselves (SYS 10/11/12) enter through the SAME GH-18
    door as every other tile: autoatlas.admit_syscall (proof =
    admission). The bake only guarantees the ABI's data plane: the
    syscall table, the tile rect, BOX2, the PTE_PIX page maps, and the
    seeded FSTAB — all in-image before the first instruction runs.

    LAYOUT RECEIPT (2026-09-09, probes debug_gh20_probe4b/8/9/10): the
    fs window [1024,1280) aliases to image pixels [2048,2560) (2 px per
    word) — which on the default 52-row bake WRAPS modulo 1664 pixels
    straight into live program text at cell 96+ (the prologue's PTE
    immediates). First draft stamped the window naively and the boot
    died at :__task_6's KJMP with status 0, no fault. FIX: bake the
    image TALL (min_rows=81 => 81+ rows), so pixels [2048,2560) are
    rows 64..80 — deep in the assembler's all-zero padding, clear of
    the program text (cells 0..270), the tile rect (cells 247..270),
    and the reserved pfn-5 table window (pixels [1312,1328), row 41).
    """
    # deferred import: baker re-exports this module (GH-20 test import
    # path) — a module-level import would be circular.
    from glyph_gpt.baker import syscall_abi_kernel_image
    # mode="fs_v2": the admit-mode prologue (pixel-mapped table page +
    # BOX2 arming) PLUS the vpn-4 fs-alias PTE (words [1024,1280) ->
    # pixels, identity by construction) and the FS drive task leg (task
    # A probe-issues SYS 10/11/12 — without it no FS number ever reaches
    # the table; receipt: probe17, 2026-09-09).
    # min_rows=81: see the layout receipt above — the fs window aliases
    # to pixels [2048,2560), which must land in the zero padding (the
    # default 52-row image wraps them into live program text).
    img = syscall_abi_kernel_image(
        atlas, mode="fs_v2", status_word=status_word,
        timer_quantum=timer_quantum, cols_instrs=cols_instrs,
        min_rows=max(min_rows, 81), out_path=None)
    # The window words stamp 1:1 (2 px/word) into the padding.
    _gh20_bake_stamp(img, _gh20_seed_fstab())
    if out_path is not None:
        p = Path(out_path)
        if p.suffix == ".npy":
            np.save(p, img)
        elif p.suffix == ".npz":
            np.savez(p, image=img)
        else:
            from PIL import Image
            Image.fromarray(img, "RGB").save(p, format="PNG")
    return img
