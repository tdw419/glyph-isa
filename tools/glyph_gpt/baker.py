#!/usr/bin/env python3
"""baker.py — GH-1 standalone image container baker.

Assembles caller text + RoutineAtlas tiles + data words into ONE spatial pixel
container (.glyph.png or .npy). The emitted image IS the complete self-hosting
runtime state: code region at origin, atlas payload linked, data words
initialized at bake time.

Usage:
  python3 tools/glyph_gpt/baker.py program.glyph -o program.glyph.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent), str(_HERE.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rv64i_to_glyph import assemble_glyph_to_pixels  # noqa: E402
import glyph_ir as gi  # noqa: E402
from glyph_isa_v2 import (  # noqa: E402
    KFAULT_PC_ADDR, KSYS_PC_ADDR, MODE_LATCH_ADDR,
    FAULT_ADDR_ADDR,
    BOX0_LO_ADDR, BOX0_HI_ADDR, BOX1_LO_ADDR, BOX1_HI_ADDR,
    BOX2_LO_ADDR, BOX2_HI_ADDR,
    SYS_A0_ADDR, SYS_N_ADDR, SYS_A1_ADDR,
    KTICK_PC_ADDR, TIMER_COUNT_ADDR, TIMER_RELOAD_ADDR, TICK_PC_ADDR,
    PAGE_TABLE_ADDR, PAGE_TABLE_WORD, PAGE_WORDS, PAGE_TABLE_BASE_WORD,
    PTE_V, PTE_W, PTE_U, PTE_PIX,
)

# GH-15 Step 4: RAM words the resident kernels own. A bake-time data seed
# landing in any of these ranges is rejected BEFORE emission (same table
# the GlyphIR StaticVerifier enforces for module data sections).
_BAKER_RESERVED_RANGES = [
    (750, 760, "GH-9 argv/receipt block"),
    (800, 896, "GH-9 mailbox patch window"),
    (950, 968, "kernel status/verdict words"),
    (1024, 1280, "GH-8b pixel-FS window"),
    (1536, 1792, "GH-17 page table window"),
]


def _check_data_words(data_words) -> None:
    """Loud, pre-emission rejection of data seeds in kernel-reserved RAM."""
    items = (data_words.items() if isinstance(data_words, dict)
             else data_words)
    for addr, _val in items:
        w = int(addr)
        for lo, hi, who in _BAKER_RESERVED_RANGES:
            if lo <= w < hi:
                raise gi.StaticVerificationError(
                    f"data word {w} lands in reserved {who} "
                    f"[{lo},{hi}) — rejected before emission")


def bake_image(
    program_text: str,
    atlas: Optional[Any] = None,
    data_words: Optional[Union[Dict[int, int], List[Tuple[int, int]]]] = None,
    cols_instrs: int = 8,
    min_rows: int = 16,
    out_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Bake a self-hosting spatial image container from program text and atlas.

    Args:
        program_text: Glyph assembly source.
        atlas: Optional RoutineAtlas supplying :atlas_* subroutine tiles.
        data_words: Optional dict or list of (word_address, 32bit_value) to seed.
        cols_instrs: Instructions per row (width = cols_instrs * 4 pixels).
        min_rows: Minimum rows in output image.
        out_path: Optional path to save (.png, .npy, .npz).

    Returns:
        np.ndarray of shape (H, W, 3) uint8 RGB pixels.
    """
    # 1. Resolve atlas tiles if caller contains unlinked :atlas_ references
    if atlas is not None and "CALL :atlas_" in program_text:
        linked = atlas.link(program_text, cols_instrs=cols_instrs)
    else:
        linked = program_text

    # 2. Seed data words at bake time via initial store instructions
    #    (GH-15 Step 4: seeds in kernel-reserved RAM are rejected BEFORE
    #    any pixels exist — the same contract the IR verifier enforces.)
    if data_words:
        _check_data_words(data_words)
        items = data_words.items() if isinstance(data_words, dict) else data_words
        init_lines: List[str] = []
        for addr, val in items:
            init_lines.append(f"LDI r14 {int(val) & 0xFFFFFFFF}")
            init_lines.append(f"LDI r15 {int(addr) & 0xFFFFFFFF}")
            init_lines.append("ST r15 r14")
        init_str = "\n".join(init_lines) + "\n"
        if ":__entry" in linked:
            linked = linked.replace(":__entry", f":__entry\n{init_str}", 1)
        else:
            linked = init_str + linked

    # 3. Assemble complete linked text into 2D spatial pixel buffer
    pixels, _coords = assemble_glyph_to_pixels(
        linked, cols_instrs=cols_instrs, min_rows=min_rows
    )

    # 4. Optional serialization to disk
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".npy":
            np.save(p, pixels)
        elif p.suffix == ".npz":
            np.savez(p, image=pixels)
        else:
            from PIL import Image
            img = Image.fromarray(pixels, "RGB")
            img.save(p, format="PNG")

    return pixels


def _raise_lines_to_ir(lines, cols_instrs: int = 8):
    """GH-15 Step 4 (Step 5 dedup): raise glyph program text (post-link)
    into a GlyphIRModule for static verification. Parsing delegates to
    glyph_ir.raise_lines_to_ir (the ONE raiser); this lane keeps only
    its module-shape policy: __baker_entry, program-sized window, data
    section placeholder, bounds clear of the reserved kernel ranges."""
    import glyph_ir as _gi

    # Count instructions the way the legacy baker did (label-with-op
    # counts once) so the data-section placeholder size is unchanged.
    import re as _re
    n_instr = 0
    for ln in lines:
        s = ln.split(";", 1)[0].strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith(":"):
            m = _re.match(r":([A-Za-z0-9_]+)\s*(.*)", s)
            if m and m.group(2).strip():
                n_instr += 1
            continue
        n_instr += 1

    policy = _gi.RaisePolicy(
        entry_label="__baker_entry",
        fuse_entry=True,
        cont_prefix="__baker_blk",
        strict_operands=False,
        window="program", cols_instrs=cols_instrs,
        preserves=(),
        data_bounds=(0, 749),
        extra_metadata={"origin": "baker"},
    )
    return _gi.raise_lines_to_ir(
        lines, name="baked_program", policy=policy,
        data_sections=[_gi.DataSection(symbol="text", base_word=0,
                                       words=[0] * max(n_instr, 1))],
    )


def ir_bake_bytes(
    program_text: str,
    atlas: Optional[Any] = None,
    data_words: Optional[Union[Dict[int, int], List[Tuple[int, int]]]] = None,
    cols_instrs: int = 8,
    min_rows: int = 16,
    out_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """GH-15 Step 4: IR-routed bake. The linked program text is raised to
    a GlyphIRModule and StaticVerifier-checked (window capacity,
    contract bounds, jump targets) BEFORE pixel emission; emission itself
    is the byte-exact legacy path — the IR stage only ever REJECTS."""
    import glyph_ir as _gi
    if atlas is not None and "CALL :atlas_" in program_text:
        linked = atlas.link(program_text, cols_instrs=cols_instrs)
    else:
        linked = program_text
    module = _raise_lines_to_ir(linked.splitlines(), cols_instrs=cols_instrs)
    _gi.StaticVerifier(module).verify()
    return bake_image(program_text, atlas=atlas, data_words=data_words,
                      cols_instrs=cols_instrs, min_rows=min_rows,
                      out_path=out_path)


def kernel_program_text(status_word: int = 950) -> str:
    """GH-2 resident kernel: all four builder contracts in glyph code.

    Seeds inputs, CALLs each atlas tile, verifies results IN-IMAGE (CMP/JZ
    for register results, compare loops for memory results), counts passes
    in r9, writes 0xCAFE0000 | n_passed to status_word, HALTs. On any
    in-image failure it writes progress-so-far through the same path and
    HALTs — the host judges success purely from receipt["memory"][status].
    """
    a = []
    add = a.append
    # ---- contract 1: double(21) -> a0 == 42
    add(":__kmain")
    add("LDI r9 0")            # pass counter
    add("LDI r10 21")
    add("CALL :atlas_double")
    add("LDI r2 42")
    add("CMP r10 r2")
    add("JZ :__k1_pass")
    add("JMP :__kfail")
    add(":__k1_pass")
    add("LDI r15 1")
    add("ADD r9 r15")
    # ---- contract 2: accumulate([7,9,5] @500) -> a0 == 21
    add("LDI r11 500")
    add("LDI r14 7")
    add("ST r11 r14")
    add("LDI r14 9")
    add("LDI r15 501")
    add("ST r15 r14")
    add("LDI r14 5")
    add("LDI r15 502")
    add("ST r15 r14")
    add("LDI r12 3")
    add("CALL :atlas_accumulate")
    add("LDI r2 21")
    add("CMP r10 r2")
    add("JZ :__k2_pass")
    add("JMP :__kfail")
    add(":__k2_pass")
    add("LDI r15 1")
    add("ADD r9 r15")
    # ---- contract 3: memcpy(500 -> 600, 3 words [111,222,333])
    add("LDI r11 500")
    add("LDI r14 111")
    add("ST r11 r14")
    add("LDI r14 222")
    add("LDI r15 501")
    add("ST r15 r14")
    add("LDI r14 333")
    add("LDI r15 502")
    add("ST r15 r14")
    add("LDI r12 600")
    add("LDI r13 3")
    add("CALL :atlas_memcpy")
    # in-image verify: compare mem[500+i] vs mem[600+i]
    add("LDI r11 500")
    add("LDI r12 600")
    add("LDI r13 3")
    add(":__k3_loop")
    add("CMP r13 r0")
    add("JZ :__k3_done")
    add("LD r14 r11")
    add("LD r15 r12")
    add("CMP r14 r15")
    add("JZ :__k3_next")
    add("JMP :__kfail")
    add(":__k3_next")
    add("LDI r4 1")
    add("ADD r11 r4")
    add("ADD r12 r4")
    add("SUB r13 r4")
    add("JMP :__k3_loop")
    add(":__k3_done")
    add("LDI r15 1")
    add("ADD r9 r15")
    # ---- contract 4: tile_clear(400, 4 words, 0xDEAD)
    add("LDI r11 400")
    add("LDI r14 57005")
    add("LDI r13 4")
    add("CALL :atlas_tile_clear")
    # in-image verify: mem[400+i] == 0xDEAD
    add("LDI r11 400")
    add("LDI r13 4")
    add(":__k4_loop")
    add("CMP r13 r0")
    add("JZ :__k4_done")
    add("LD r14 r11")
    add("LDI r15 57005")
    add("CMP r14 r15")
    add("JZ :__k4_next")
    add("JMP :__kfail")
    add(":__k4_next")
    add("LDI r4 1")
    add("ADD r11 r4")
    add("SUB r13 r4")
    add("JMP :__k4_loop")
    add(":__k4_done")
    add("LDI r15 1")
    add("ADD r9 r15")
    # ---- success: status = 0xCAFE0000 | r9, halt
    add("LDI r3 51966")        # 0xCAFE
    add("LDI r4 16")
    add("SHL r3 r4")           # 0xCAFE0000
    add("ADD r3 r9")
    add(f"LDI r15 {status_word}")
    add("ST r15 r3")
    add("HALT")
    # ---- shared failure exit: record progress-so-far, halt
    add(":__kfail")
    add("LDI r3 51966")
    add("LDI r4 16")
    add("SHL r3 r4")
    add("ADD r3 r9")
    add(f"LDI r15 {status_word}")
    add("ST r15 r3")
    add("HALT")
    return "\n".join(a) + "\n"


def bake_kernel_image(
    atlas: Any,
    status_word: int = 950,
    cols_instrs: int = 8,
    min_rows: int = 16,
    out_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """GH-2: bake ONE image containing the resident :__kmain kernel plus the
    atlas payload. The kernel runs every builder contract in-image; the host
    only reads receipt['halted'], receipt['faulted'], and memory[status_word].
    """
    return bake_image(
        kernel_program_text(status_word),
        atlas=atlas,
        cols_instrs=cols_instrs,
        min_rows=min_rows,
        out_path=out_path,
    )


def self_extending_kernel_program_text(
    patch_pixel_addr: int = 0,
    status_word: int = 950,
    mailbox_flag: int = 960,
    mailbox_n_px: int = 961,
    mailbox_data: int = 964,
    n_patch_instrs: int = 16,
) -> str:
    """GH-3: glyph assembly for self-extending kernel with patch window."""
    lines = [
        ":__entry",
        "LDI r31 4351",
        "JMP :__kmain",
        ":__kmain",
        # Check if MAILBOX_FLAG == 1
        f"LDI r15 {mailbox_flag}",
        "LD r1 r15",
        "LDI r2 1",
        "CMP r1 r2",
        "JZ :__do_patch",
        "JMP :__exec_ext",
        ":__do_patch",
        # Clear MAILBOX_FLAG: memory[mailbox_flag] = 0
        "LDI r2 0",
        "ST r15 r2",
        # Read MAILBOX_N_PX
        f"LDI r15 {mailbox_n_px}",
        "LD r13 r15",
        # Source memory address in mailbox
        f"LDI r10 {mailbox_data}",
        # Destination linear pixel address in image
        f"LDI r11 {patch_pixel_addr}",
        ":__patch_loop",
        "CMP r13 r0",
        "JZ :__patch_done",
        "LD r14 r10",
        "PARALLEL_ST r11 r14 1",
        "LDI r4 1",
        "ADD r10 r4",
        "ADD r11 r4",
        "SUB r13 r4",
        "JMP :__patch_loop",
        ":__patch_done",
        ":__exec_ext",
        # Set up test input: r10 = 7
        "LDI r10 7",
        "CALL :__patch_window",
        # Check contract: result in r10 == 21 (7 * 3)
        "LDI r2 21",
        "CMP r10 r2",
        "JZ :__ext_pass",
        "JMP :__ext_fail",
        ":__ext_pass",
        "LDI r3 51966",  # 0xCAFE
        "LDI r4 16",
        "SHL r3 r4",
        "LDI r4 3",
        "ADD r3 r4",     # 0xCAFE0003
        f"LDI r15 {status_word}",
        "ST r15 r3",
        "HALT",
        ":__ext_fail",
        "LDI r3 57005",  # 0xDEAD
        "LDI r4 16",
        "SHL r3 r4",
        "LDI r4 3",
        "ADD r3 r4",     # 0xDEAD0003
        f"LDI r15 {status_word}",
        "ST r15 r3",
        "HALT",
        ":__patch_window",
    ]
    for _ in range(n_patch_instrs):
        lines.append("HALT")
    return "\n".join(lines) + "\n"


def bake_self_extending_kernel_image(
    atlas: Any,
    status_word: int = 950,
    mailbox_flag: int = 960,
    mailbox_n_px: int = 961,
    mailbox_data: int = 964,
    cols_instrs: int = 8,
    min_rows: int = 16,
    n_patch_instrs: int = 16,
    out_path: Optional[Union[str, Path]] = None,
) -> tuple[np.ndarray, dict]:
    """GH-3: bake ONE self-extending kernel image containing a reserved patch
    window, an MMIO-mailbox loader loop (PARALLEL_ST), and dynamic dispatch.
    """
    txt1 = self_extending_kernel_program_text(
        patch_pixel_addr=0,
        status_word=status_word,
        mailbox_flag=mailbox_flag,
        mailbox_n_px=mailbox_n_px,
        mailbox_data=mailbox_data,
        n_patch_instrs=n_patch_instrs,
    )
    if atlas is not None and "CALL :atlas_" in txt1:
        linked1 = atlas.link(txt1, cols_instrs=cols_instrs)
    else:
        linked1 = txt1
    _, coords1 = assemble_glyph_to_pixels(linked1, cols_instrs=cols_instrs, min_rows=min_rows)
    col, row = coords1[":__patch_window"]
    patch_pixel_addr = row * (cols_instrs * 4) + (col * 4)
    patch_pixel_count = n_patch_instrs * 4

    txt2 = self_extending_kernel_program_text(
        patch_pixel_addr=patch_pixel_addr,
        status_word=status_word,
        mailbox_flag=mailbox_flag,
        mailbox_n_px=mailbox_n_px,
        mailbox_data=mailbox_data,
        n_patch_instrs=n_patch_instrs,
    )
    pixels = bake_image(
        txt2,
        atlas=atlas,
        cols_instrs=cols_instrs,
        min_rows=min_rows,
        out_path=out_path,
    )
    patch_info = {
        "patch_pixel_addr": patch_pixel_addr,
        "patch_pixel_count": patch_pixel_count,
        "patch_col": col,
        "patch_row": row,
        "n_patch_instrs": n_patch_instrs,
    }
    return pixels, patch_info


# --- GH-6 syscall/UART image ABI (fixed word indices) -----------------------
# Kept below the atlas-payload region the GH-2 kernel uses (contracts at
# 400..600). Byte addresses are 4x word addresses; the user box is expressed
# in BYTES (BOX0_LO/HI are byte ranges per glyph_isa_v2._addr_in_box), while
# the task's own ST/LD instructions use WORD addresses (byte_to_word_mem
# lowering) — so the box bytes are the word indices times 4.
GH6_UART_WORD = 700       # handler packs 'H','E','L','L' little-endian here
GH6_UART_WORD2 = 701      # then 'O'
GH6_UART_LEN_WORD = 702   # total bytes the handler appended (5)
GH6_EXIT_WORD = 703       # task's syscall exit status (kernel-visible)
GH6_FAULT_WORD = 704      # fault-leg verdict (0xFA171 = violation seen)
GH6_BOX0_LO_BYTE = 4 * 700   # box starts here (bytes)...
GH6_BOX0_HI_BYTE = 4 * 715   # ...and ends here (bytes)
# Inside the box (byte addresses): the UART ABI words 700..704 AND the
# task's scratch word at 4*705 (packed 'H'|'E'<<8|'L'<<16). The task's own
# STs use WORD indices (byte_to_word_mem lowering) — box bytes are the
# word indices times 4. The task's exit word IS GH6_EXIT_WORD (703).
GH6_TASK_BUF_WORD = 705
GH6_TASK_EXIT_WORD = GH6_EXIT_WORD


def _pack_hello4() -> int:
    """'H'|'E'<<8|'L'<<16 — the packed 3-byte HELLO prefix (24-bit LDI-safe)."""
    return ord('H') | (ord('E') << 8) | (ord('L') << 16)


# MMIO word indices (from tools/glyph_isa_v2.py constants, addr >> 2)
KSYS_PC_WORD = KSYS_PC_ADDR >> 2
KFAULT_PC_WORD = KFAULT_PC_ADDR >> 2
MODE_LATCH_WORD = MODE_LATCH_ADDR >> 2
BOX0_LO_WORD = BOX0_LO_ADDR >> 2
BOX2_LO_WORD = BOX2_LO_ADDR >> 2
SYS_A0_WORD = SYS_A0_ADDR >> 2

# Packed pixel PCs of the syscall handler / fault handler / done tail /
# user-task entry, bound by syscalluart_kernel_image() before the second
# (final) assemble pass — glyph code cannot name a pixel coordinate, so the
# baker injects them (same loader-seeds-the-PC pattern as the E-K2 harness).
_GH6_SYSCALL_PC = 0
_GH6_FAULT_PC = 0
_GH6_RET_PC = 0
_GH6_TASK_PC = 0
GH6_N = 6  # the syscall number the user task issues


def _uart_kernel_program_text(
    status_word: int = 950,
    fault_leg: bool = False,
) -> str:
    """GH-6 glyph assembly: resident kernel + USER box task + syscall path.

    Sequence (all in-image; the host only runs the image and reads memory):
      1. SUPER-mode prologue seeds the UART ABI words, programs BOX0
         (byte range [4*705, 4*715)), writes the packed pixel PC of the
         syscall handler into KSYS_PC and of the fault handler into
         KFAULT_PC, latches MODE_LATCH = MODE_USER, then KJMPs into the
         user task.
      2. KJMP is the engine's one-shot latch consumer (glyph_isa_v2:
         only KJMP checks MODE_LATCH == MODE_USER): mode drops to SUPER
         and immediately re-enters USER for the task; the latch clears.
      3. The task (USER, standalone) issues SYSCALL 6: the engine
         marshals r17/r10/r11 -> SYS_N/SYS_A0/SYS_A1, saves SYSCALL_PC,
         drops to SUPER and jumps KSYS_PC.
      4. The handler (SUPER) reads SYS_N/SYS_A0/SYS_A1, copies the packed
         HELLO word into the UART region, appends the byte count, writes
         the byte-count result back into SYS_A0, then SYSRETs.
      5. SYSRET restores the task's register file, delivers the result
         into r10, re-enters USER and resumes after the SYSCALL.
      6. The task writes its exit status (0xFEED0000 | syscall number) to
         its in-box exit word, then KJMPs to the loader-seeded done tail:
         the boundary jump drops back to SUPER (latch is 0, so it stays
         SUPER) and the kernel tail's status store is a legal kernel
         store. KJMP -- not RET -- is the privilege boundary here.
      7. The kernel tail (reached only when the task completed without
         faulting) re-seeds the status word and HALTs with
         0xCAFE0000 | 6.

    Mode-arithmetic note: glyph word ops mask at 32 bits and the CPU's
    mode is a host-side Python bool, so the kernel stores MODE_USER (1)
    into the latch; KJMP compares == MODE_USER, one-shots and clears it.
    The fault leg is the same image except the task's first act is a
    deliberate out-of-box store: E-K1 fires, the fault handler records
    GH6_FAULT_SEEN and falls into the done tail.
    """
    a: list[str] = []
    add = a.append
    add(":__entry")
    add("JMP :__kmain")
    add(":__kmain")
    # ---- seed the UART ABI words (kernel-visible region) ----
    add(f"LDI r15 {GH6_UART_WORD}")
    add("LDI r14 0")
    add("ST r15 r14")
    add(f"LDI r15 {GH6_UART_WORD2}")
    add("ST r15 r14")
    add(f"LDI r15 {GH6_UART_LEN_WORD}")
    add("ST r15 r14")
    if fault_leg:
        add(f"LDI r15 {GH6_FAULT_WORD}")
        add("ST r15 r14")
    # ---- program BOX0 (byte range) ----
    add(f"LDI r15 {BOX0_LO_WORD}")
    add(f"LDI r14 {GH6_BOX0_LO_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX0_LO_WORD + 1}")
    add(f"LDI r14 {GH6_BOX0_HI_BYTE}")
    add("ST r15 r14")
    # ---- arm the vectors (loader-seeded packed pixel PCs) ----
    add(f"LDI r15 {KSYS_PC_WORD}")
    add(f"LDI r14 {_GH6_SYSCALL_PC}")
    add("ST r15 r14")
    add(f"LDI r15 {KFAULT_PC_WORD}")
    add(f"LDI r14 {_GH6_FAULT_PC if fault_leg else _GH6_RET_PC}")
    add("ST r15 r14")
    # ---- latch USER and enter the task ----
    # KJMP is the only engine instruction that consumes MODE_LATCH, so
    # entry into the task must be KJMP (CALLR would keep SUPER mode).
    add(f"LDI r15 {MODE_LATCH_WORD}")
    add("LDI r14 1")           # MODE_USER: KJMP compares == 1, one-shots + clears
    add("ST r15 r14")
    add(f"LDI r30 {_GH6_TASK_PC}")
    add("KJMP r30")
    add("JMP :__kdone")
    # ---- syscall handler (SUPER): uart append + result ----
    add(":__ksys")
    # Read the engine-marshaled ABI words: LDI the MMIO word index, then
    # LD through it ('LD rd rX' dereferences the VALUE of rX as an address).
    add(f"LDI r15 {SYS_N_ADDR >> 2}")
    add("LD r5 r15")           # r5 = SYS_N (syscall number)
    add(f"LDI r15 {SYS_A0_ADDR >> 2}")
    add("LD r6 r15")           # r6 = SYS_A0 (task buffer word index)
    add(f"LDI r15 {SYS_A1_ADDR >> 2}")
    add("LD r7 r15")           # r7 = SYS_A1 (byte count)
    add("LD r8 r6")            # r8 = packed 'H'|'E'<<8|'L'<<16 from the buffer
    add("LDI r14 76")          # 'L'
    add("LDI r4 24")
    add("SHL r14 r4")          # 'L' << 24
    add("OR r8 r14")           # r8 = 'HELL' packed little-endian (4 bytes)
    add(f"LDI r15 {GH6_UART_WORD}")
    add("ST r15 r8")
    add(f"LDI r15 {GH6_UART_WORD2}")
    add("LDI r14 79")          # 'O'
    add("ST r15 r14")
    add(f"LDI r15 {GH6_UART_LEN_WORD}")
    add("LDI r14 5")           # 5 bytes appended ('HELL' + 'O')
    add("ST r15 r14")
    add(f"LDI r15 {SYS_A0_WORD}")
    add("ST r15 r14")          # result (bytes written) -> SYS_A0 for SYSRET
    add("SYSRET")

    add(":__kfault")
    if fault_leg:
        add(f"LDI r15 {GH6_FAULT_WORD}")
        add("LDI r14 1024369")  # 0xFA171 — violation seen
        add("ST r15 r14")
    add("LDI r9 6")
    # fall through to the same done/halt tail
    # ---- done: status = 0xCAFE0000 | r9, halt ----
    add(":__kdone")
    add("LDI r3 51966")        # 0xCAFE
    add("LDI r4 16")
    add("SHL r3 r4")
    add("ADD r3 r9")
    add(f"LDI r15 {status_word}")
    add("ST r15 r3")
    add("HALT")
    # ---- user task (standalone; ends with the KJMP privilege boundary) ----
    add(":__utask")
    if fault_leg:
        # The violation itself: word 900 (byte 3600) is outside BOX0, so
        # this USER store vectors to KFAULT_PC and the handler records it.
        add("LDI r15 900")
        add("LDI r14 1")
        add("ST r15 r14")
        add("KJMP r30")        # r30 unreachable here (E-K1 fires on the ST)
    add(f"LDI r17 {GH6_N}")
    add(f"LDI r10 {GH6_TASK_BUF_WORD}")
    add("LDI r11 4")
    add(f"LDI r14 {_pack_hello4()}")
    add(f"LDI r15 {GH6_TASK_BUF_WORD}")
    add("ST r15 r14")
    add("SYSCALL r12")
    add(f"LDI r15 {GH6_TASK_EXIT_WORD}")
    add("LDI r14 65261")       # 0xFEED
    add("LDI r4 16")
    add("SHL r14 r4")
    add("ADD r14 r17")
    add("ST r15 r14")
    # Exit through the privilege boundary: KJMP back to the kernel done
    # tail. The latch is 0 (entry one-shot cleared it), so the engine
    # drops to SUPER and STAYS SUPER for the kernel-tail status store.
    add(f"LDI r30 {_GH6_RET_PC}")
    add("KJMP r30")
    return "\n".join(a) + "\n"


def syscalluart_kernel_image(
    atlas: Any,
    status_word: int = 950,
    fault_leg: bool = False,
    cols_instrs: int = 8,
    min_rows: int = 16,
    out_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """GH-6: bake ONE image with the resident syscall kernel. Two passes:
    pass 1 fixes the packed pixel PCs (:__ksys/:__kfault/:__utask), pass 2
    bakes the final image. The atlas is accepted for GH-2 continuity but
    the syscall path needs no atlas tiles."""
    txt1 = _uart_kernel_program_text(status_word, fault_leg)
    _, coords1 = assemble_glyph_to_pixels(txt1, cols_instrs=cols_instrs, min_rows=min_rows)

    def packed(label: str) -> int:
        col, row = coords1[label]
        # engine packing: low 16 bits = column, high 16 = row
        return (col & 0xFFFF) | ((row & 0xFFFF) << 16)

    # Pass 2 re-materializes the program with the real packed PCs baked in
    # (assemble is deterministic, so the label coordinates are identical).
    return _bake_uart_with_pcs(
        status_word=status_word,
        fault_leg=fault_leg,
        syscall_pc=packed(":__ksys"),
        fault_pc=packed(":__kfault"),
        ret_pc=packed(":__kdone"),
        task_pc=packed(":__utask"),
        cols_instrs=cols_instrs,
        min_rows=min_rows,
        out_path=out_path,
    )


def _bake_uart_with_pcs(
    status_word: int,
    fault_leg: bool,
    syscall_pc: int,
    fault_pc: int,
    ret_pc: int,
    task_pc: int,
    cols_instrs: int,
    min_rows: int,
    out_path: Optional[Union[str, Path]],
) -> np.ndarray:
    """Second pass: bind the real packed pixel PCs into the program text."""
    global _GH6_SYSCALL_PC, _GH6_FAULT_PC, _GH6_RET_PC, _GH6_TASK_PC
    _GH6_SYSCALL_PC, _GH6_FAULT_PC = syscall_pc, fault_pc
    _GH6_RET_PC, _GH6_TASK_PC = ret_pc, task_pc
    try:
        return bake_image(
            _uart_kernel_program_text(status_word, fault_leg),
            atlas=None,
            cols_instrs=cols_instrs,
            min_rows=min_rows,
            out_path=out_path,
        )
    finally:
        _GH6_SYSCALL_PC = _GH6_FAULT_PC = _GH6_RET_PC = _GH6_TASK_PC = 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Bake Glyph code into spatial image container")
    parser.add_argument("input", help="Path to input .glyph source file")
    parser.add_argument("-o", "--output", default=None, help="Output image file (.glyph.png)")
    parser.add_argument("--cols", type=int, default=8, help="Columns of instructions (default: 8)")
    parser.add_argument("--min-rows", type=int, default=16, help="Minimum image rows (default: 16)")
    args = parser.parse_args()

    src_path = Path(args.input)
    text = src_path.read_text()
    out_path = args.output or src_path.with_suffix(".glyph.png")

    img = bake_image(text, cols_instrs=args.cols, min_rows=args.min_rows, out_path=out_path)
    print(f"Baked image {img.shape[1]}x{img.shape[0]} px saved to {out_path}")


# --- GH-7 multiprocessing image ABI (fixed word indices) ---------------------
# Same region scheme as GH-6 (words 700+; below the atlas payload region).
# BOX0 = task A arena [700..717), BOX1 = task B arena [718..735). The kernel
# owns the turn word at 705 (inside BOX0, written only in SUPER mode).
GH7_TURN_WORD = 705         # kernel round-robin marker (SUPER-only writes)
GH7_UART_A_WORD = 710       # handler A packs 'A','A','A','A' little-endian
GH7_UART_A_WORD2 = 711
GH7_UART_A_LEN = 712        # bytes the handler appended (4)
GH7_TASK_A_BUF = 713        # task A scratch (in BOX0)
GH7_EXIT_A = 703            # task A exit status word (in BOX0)
GH7_UART_B_WORD = 720       # handler B packs 'B','B','B','B' little-endian
GH7_UART_B_WORD2 = 721
GH7_UART_B_LEN = 722        # bytes the handler appended (4)
GH7_EXIT_B = 723            # task B exit status word (in BOX1)
GH7_TASK_B_BUF = 727        # task B scratch (in BOX1)
GH7_BADSYS_WORD = 730       # unknown-syscall marker ('E' = 69)
GH7_FAULT_WORD = 731        # fault-leg verdict (0xFA171 = violation seen)
GH7_BOX0_LO_BYTE = 4 * 700  # BOX0 = task A arena...
GH7_BOX0_HI_BYTE = 4 * 717  # ...[700..717)
GH7_BOX1_LO_BYTE = 4 * 718  # BOX1 = task B arena...
GH7_BOX1_HI_BYTE = 4 * 735  # ...[718..735)
GH7_N_A = 6                 # task A's syscall number
GH7_N_B = 7                 # task B's syscall number
GH7_FAULT_SEEN = 0xFA171    # fault-leg verdict value (same as GH-6)

# MMIO word indices the GH-7 kernel programs (BOX1 wasn't needed by GH-6)
BOX1_LO_WORD = BOX1_LO_ADDR >> 2
BOX1_HI_WORD = BOX1_HI_ADDR >> 2

# Packed pixel PCs of the GH-7 dispatch/handler/fault/done/tail labels,
# bound by multiproc_kernel_image() before the second (final) assemble pass.
_GH7_DISPATCH_PC = 0
_GH7_KSYS_PC = 0
_GH7_FAULT_PC = 0
_GH7_RET_PC = 0
_GH7_TASK_A_PC = 0
_GH7_TASK_B_PC = 0


def _pack_word4(ch: str) -> int:
    """'A' packed little-endian x4: 0x41414141."""
    v = ord(ch)
    return v | (v << 8) | (v << 16) | (v << 24)


def _mp_syscall_handler(n: int, ch: str, uart: int, uart2: int, uart_len: int) -> List[str]:
    """One SUPER-mode dispatcher slice: services syscall n (task A: 6, B: 7).

    Reached only when SYS_N == n (the :__ksys selector checked first); an
    unknown number records 'E' at GH7_BADSYS_WORD and falls through without
    touching the UART words. Services the task's payload: copies the packed
    4-byte word from the task's buffer word into the task's own UART region,
    appends the byte count, writes the result back to SYS_A0, SYSRETs.
    """
    a: List[str] = []
    add = a.append
    add(f":__ksys_{n}")
    add(f"LDI r15 {SYS_N_ADDR >> 2}")
    add("LD r5 r15")                    # r5 = SYS_N
    add(f"LDI r4 {n}")
    add("CMP r5 r4")
    add(f"JZ :__ksys_{n}_go")
    # unknown syscall for this dispatcher: record 'E' (69) and SYSRET dry
    add(f"LDI r15 {GH7_BADSYS_WORD}")
    add("LDI r14 69")                   # 'E'
    add("ST r15 r14")
    add("SYSRET")
    add(f":__ksys_{n}_go")
    add(f"LDI r15 {SYS_A0_ADDR >> 2}")
    add("LD r6 r15")                    # r6 = task buffer WORD index
    add(f"LDI r15 {SYS_A1_ADDR >> 2}")
    add("LD r7 r15")                    # r7 = byte count (4)
    add("LD r8 r6")                     # r8 = packed payload word
    add(f"LDI r15 {uart}")
    add("ST r15 r8")
    add(f"LDI r15 {uart2}")
    add("LDI r14 0")
    add("ST r15 r14")
    add(f"LDI r15 {uart_len}")
    add("ST r15 r7")                    # bytes appended
    add(f"LDI r15 {SYS_A0_WORD}")
    add("ST r15 r7")                    # result -> SYS_A0 for SYSRET
    add("SYSRET")
    return a


def _mp_user_task(
    n: int,
    buf_word: int,
    exit_word: int,
    fault_leg: bool = False,
) -> List[str]:
    """One USER-mode task slice: (optional violation) SYSCALL n with the
    packed payload, exit status write, KJMP back to the kernel dispatch
    loop (the privilege boundary; latch is 0, so it lands back in SUPER)."""
    a: List[str] = []
    add = a.append
    add(f":__task_{n}")
    if fault_leg:
        # The violation itself: word 900 (byte 3600) is outside every box,
        # so this USER store vectors to KFAULT_PC in USER mode.
        add("LDI r15 900")
        add("LDI r14 1")
        add("ST r15 r14")
    add(f"LDI r17 {n}")                 # syscall number in a7 = r17
    add(f"LDI r10 {buf_word}")          # a0 = r10 = buffer word index
    add("LDI r11 4")                    # a1 = r11 = byte count
    # payload char is the task's letter (A for syscall 6, B for 7 -- GH-7's
    # ABI numbering starts at 6, so the letter is NOT chr(64+n) == 'F').
    add(f"LDI r14 {_pack_word4('AB'[n - GH7_N_A])}")
    add(f"LDI r15 {buf_word}")
    add("ST r15 r14")                   # store the packed payload
    add("SYSCALL r12")
    add(f"LDI r15 {exit_word}")
    add("LDI r14 65261")                # 0xFEED
    add("LDI r4 16")
    add("SHL r14 r4")
    add("ADD r14 r17")
    add("ST r15 r14")                   # 0xFEED0000 | n
    # Return to the kernel dispatch loop through the privilege boundary.
    add(f"LDI r30 {_GH7_DISPATCH_PC if n == GH7_N_A else _GH7_RET_PC}")
    add("KJMP r30")
    return a


def _mp_kernel_program_text(
    status_word: int = 950,
    fault_leg: bool = False,
) -> str:
    """GH-7 glyph assembly: resident scheduler kernel + two USER box tasks.

    Sequence (all in-image; the host only runs the image and reads memory):
      1. SUPER-mode prologue seeds both UART ABI regions to 0, seeds the
         turn word, programs BOX0 and BOX1, arms KSYS_PC/KFAULT_PC with
         loader-seeded packed pixel PCs (two-pass bake), then enters task A:
         MODE_LATCH = 1, KJMP :__task_6 (KJMP is the only latch consumer).
      2. Task A runs USER in BOX0, SYSCALLs 6; the engine marshals
         r17/r10/r11 -> SYS_N/SYS_A0/SYS_A1, drops to SUPER, jumps KSYS_PC
         (== :__ksys selector). The handler services it, SYSRET re-enters USER.
      3. Task A writes its exit word and KJMPs back to :__dispatch (latch 0,
         so it stays SUPER) — the round-robin switch point.
      4. :__dispatch re-arms MODE_LATCH = 1 and KJMPs into task B; task B
         runs the same pattern with syscall 7 and KJMPs to :__kdone.
      5. :__kdone writes status = 0xCAFE0000 | 7 and HALTs.
    """
    a: List[str] = []
    add = a.append
    add(":__entry")
    add("JMP :__kmain")
    add(":__kmain")
    # ---- seed the UART regions and the turn word ----
    for w in (GH7_UART_A_WORD, GH7_UART_A_WORD2, GH7_UART_A_LEN,
              GH7_UART_B_WORD, GH7_UART_B_WORD2, GH7_UART_B_LEN,
              GH7_BADSYS_WORD, GH7_FAULT_WORD, GH7_TURN_WORD):
        add(f"LDI r15 {w}")
        add("LDI r14 0")
        add("ST r15 r14")
    # ---- program BOX0 (task A) and BOX1 (task B) ----
    add(f"LDI r15 {BOX0_LO_WORD}")
    add(f"LDI r14 {GH7_BOX0_LO_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX0_LO_WORD + 1}")
    add(f"LDI r14 {GH7_BOX0_HI_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX1_LO_WORD}")
    add(f"LDI r14 {GH7_BOX1_LO_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX1_LO_WORD + 1}")
    add(f"LDI r14 {GH7_BOX1_HI_BYTE}")
    add("ST r15 r14")
    # ---- arm the vectors (loader-seeded packed pixel PCs) ----
    add(f"LDI r15 {KSYS_PC_WORD}")
    add(f"LDI r14 {_GH7_KSYS_PC}")
    add("ST r15 r14")
    add(f"LDI r15 {KFAULT_PC_WORD}")
    add(f"LDI r14 {_GH7_FAULT_PC if fault_leg else _GH7_RET_PC}")
    add("ST r15 r14")
    # ---- enter task A: latch USER, KJMP ----
    add(f"LDI r15 {MODE_LATCH_WORD}")
    add("LDI r14 1")           # MODE_USER: KJMP one-shots the latch + clears
    add("ST r15 r14")
    add(f"LDI r30 {_GH7_TASK_A_PC}")
    add("KJMP r30")
    add("JMP :__kdone")
    # ---- syscall dispatcher (SUPER): reached via KSYS_PC ----
    # ONE resident dispatcher services BOTH tasks' syscall numbers: select
    # on SYS_N (6 -> :__ksys_6 slice, 7 -> :__ksys_7 slice), else record
    # 'E' and SYSRET dry. A single-number handler would strand the other
    # task's SYSCALL (its handler is never the KSYS_PC vector).
    add(":__ksys")
    add(f"LDI r15 {SYS_N_ADDR >> 2}")
    add("LD r5 r15")                    # r5 = SYS_N
    add(f"LDI r4 {GH7_N_A}")
    add("CMP r5 r4")
    add(f"JZ :__ksys_{GH7_N_A}")
    add(f"LDI r4 {GH7_N_B}")
    add("CMP r5 r4")
    add(f"JZ :__ksys_{GH7_N_B}")
    # unknown syscall: record 'E' (69) and SYSRET dry
    add(f"LDI r15 {GH7_BADSYS_WORD}")
    add("LDI r14 69")                   # 'E'
    add("ST r15 r14")
    add("SYSRET")
    for line in _mp_syscall_handler(GH7_N_A, "A", GH7_UART_A_WORD,
                                    GH7_UART_A_WORD2, GH7_UART_A_LEN):
        add(line)
    for line in _mp_syscall_handler(GH7_N_B, "B", GH7_UART_B_WORD,
                                    GH7_UART_B_WORD2, GH7_UART_B_LEN):
        add(line)
    # ---- round-robin dispatch loop: switch A -> B ----
    add(":__dispatch")
    add(f"LDI r15 {MODE_LATCH_WORD}")
    add("LDI r14 1")
    add("ST r15 r14")
    add(f"LDI r30 {_GH7_TASK_B_PC}")
    add("KJMP r30")
    add("JMP :__kdone")
    # ---- fault handler (SUPER): records the violation, halts via kernel ----
    add(":__kfault")
    if fault_leg:
        add(f"LDI r15 {GH7_FAULT_WORD}")
        add(f"LDI r14 {GH7_FAULT_SEEN}")
        add("ST r15 r14")
    # fall through to the done/halt tail
    # ---- done: status = 0xCAFE0000 | 7, halt ----
    add(":__kdone")
    add("LDI r9 7")           # last scheduled task id (fault path falls in here too)
    add("LDI r3 51966")        # 0xCAFE
    add("LDI r4 16")
    add("SHL r3 r4")
    add("ADD r3 r9")
    add(f"LDI r15 {status_word}")
    add("ST r15 r3")
    add("HALT")
    # ---- user task A (USER, BOX0): SYSCALL 6, exit, switch back ----
    for line in _mp_user_task(GH7_N_A, GH7_TASK_A_BUF, GH7_EXIT_A, fault_leg):
        add(line)
    # ---- user task B (USER, BOX1): SYSCALL 7, exit, final boundary ----
    for line in _mp_user_task(GH7_N_B, GH7_TASK_B_BUF, GH7_EXIT_B, False):
        add(line)
    return "\n".join(a) + "\n"


def multiproc_kernel_image(
    atlas: Any,
    status_word: int = 950,
    fault_leg: bool = False,
    cols_instrs: int = 8,
    min_rows: int = 16,
    out_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """GH-7: bake ONE image with the resident two-task scheduler kernel.
    Two passes: pass 1 fixes the packed pixel PCs (:__ksys/:__dispatch/
    :__kfault/:__task_6/:__task_7), pass 2 bakes the final image. The atlas
    is accepted for GH-2 continuity but the scheduler needs no atlas tiles."""
    txt1 = _mp_kernel_program_text(status_word, fault_leg)
    _, coords1 = assemble_glyph_to_pixels(txt1, cols_instrs=cols_instrs, min_rows=min_rows)

    def packed(label: str) -> int:
        col, row = coords1[label]
        # engine packing: low 16 bits = column, high 16 = row
        return (col & 0xFFFF) | ((row & 0xFFFF) << 16)

    global _GH7_DISPATCH_PC, _GH7_KSYS_PC, _GH7_FAULT_PC, _GH7_RET_PC, _GH7_TASK_A_PC, _GH7_TASK_B_PC
    _GH7_DISPATCH_PC = packed(":__dispatch")
    _GH7_KSYS_PC = packed(":__ksys")      # the selector, not one task's slice
    _GH7_FAULT_PC = packed(":__kfault")   # must be bound BEFORE pass 2 or the
    _GH7_RET_PC = packed(":__kdone")      # fault leg vectors to pixel (0,0)
    _GH7_TASK_A_PC = packed(":__task_6")
    _GH7_TASK_B_PC = packed(":__task_7")
    try:
        # Pass 2 re-materializes the program with the real packed PCs baked
        # in (assemble is deterministic, so the label coords are identical).
        return bake_image(
            _mp_kernel_program_text(status_word, fault_leg),
            atlas=None,
            cols_instrs=cols_instrs,
            min_rows=min_rows,
            out_path=out_path,
        )
    finally:
        _GH7_DISPATCH_PC = _GH7_FAULT_PC = _GH7_RET_PC = _GH7_TASK_A_PC = _GH7_TASK_B_PC = 0


# --- GH-8 in-image filesystem ABI (fixed word indices) -----------------------
# BOX0 = task arena [700..768) — wide enough to hold both tasks' exit words
# AND the task<->kernel staging windows. The FS itself lives ABOVE the arena
# in kernel-owned words (user stores there fault): FSTAB at 780 (2 slots x 4
# words: name, start, len, in_use), file data region at 800.
GH8_FSTAB_WORD = 1024       # GH-8b: in the pixel-aliased FS window [1024,1280)
GH8_FSTAB_NSLOTS = 2
GH8_SLOT_WORDS = 4          # name, start, len, in_use
GH8_DATA_WORD = 1044        # file data region (pixel-backed; outside boxes)
GH8_SCRATCH_WORD = 736      # task -> kernel staging (inside BOX0)
GH8_READOUT_WORD = 752      # kernel -> task read-out (inside BOX0)
GH8_EXIT_A = 703
GH8_EXIT_B = 723
GH8_BADSYS_WORD = 730       # unknown-syscall marker ('E')
GH8_FAULT_WORD = 731        # fault-leg verdict (0xFA171)
GH8_BOX0_LO_BYTE = 4 * 700
GH8_BOX0_HI_BYTE = 4 * 768
GH8_N_WRITE = 6             # create + write
GH8_N_READ = 7              # read
GH8_N_DEL = 8               # delete
GH8_NAME_DATA = (ord('D') | (ord('A') << 8) | (ord('T') << 16) | (ord('A') << 24))
GH8_PAYLOAD0 = 0x11223344
GH8_PAYLOAD1 = 0x55667788
GH8_FILE_BYTES = 8
GH8_FAULT_SEEN = 0xFA171

# Packed pixel PCs of the GH-8 selector/handler/fault/dispatch/done labels,
# bound by fs_kernel_image() before the second (final) assemble pass.
_GH8_KSYS_PC = 0
_GH8_FAULT_PC = 0
_GH8_RET_PC = 0
_GH8_DISPATCH_PC = 0
_GH8_TASK_A_PC = 0
_GH8_TASK_B_PC = 0


def _fs_store(word: int, value_expr: str, a: list, add) -> None:
    """Emit LDI r15 <word>; LDI/other r14 <value>; ST r15 r14 is caller's job.

    (kept trivial: the kernel text uses explicit LDI/ST pairs below; this
    helper exists only to document the ST convention: addr reg r15,
    value reg r14.)"""


def _fs_syscall_handler(n: int) -> List[str]:
    """One SUPER-mode FS dispatcher slice, reached only when SYS_N == n.

    Shared prologue facts (selector already checked SYS_N): r5 = SYS_N.
    FS layout constants are LDI-immediates; the FSTAB is scanned linearly.

    SYS 6 (create+write): a0 = name, a1 = byte count. Payload is already in
      the shared scratch window (task stored it before SWI). Kernel claims
      slot 0 (single-slot create for the gate; GH-14 generalizes), sets
      start = data region, len = a1, in_use = 1, copies 2 payload words
      scratch -> data region, returns byte count.
    SYS 7 (read): a0 = name, a1 = max bytes. Scans FSTAB for name + in_use;
      hit: copies len/4 words data -> read-out window, returns len.
      miss: writes 'E' into read-out word 0, returns 0.
    SYS 8 (delete): a0 = name. Scans FSTAB, clears in_use on hit.
    """
    a: List[str] = []
    add = a.append
    add(f":__fssys_{n}")
    add(f"LDI r15 {SYS_A0_ADDR >> 2}")
    add("LD r6 r15")                    # r6 = a0 = file name
    add(f"LDI r15 {SYS_A1_ADDR >> 2}")
    add("LD r7 r15")                    # r7 = a1 = byte count
    if n == GH8_N_WRITE:
        # GH-8b idempotence guard: if slot0.in_use == 1 the file already
        # lives in the image pixels (persisted from a previous boot) —
        # leave it untouched (create is a no-op) so offline re-runs READ
        # the stored bytes instead of replaying canonical payloads.
        add(f"LDI r15 {GH8_FSTAB_WORD + 3}")
        add("LD r9 r15")                # r9 = slot0.in_use
        add("LDI r8 1")
        add("CMP r9 r8")
        add(f"JZ :__fssys_{n}_done")    # exists: skip claim + payload copy
        # claim slot 0: name, start=DATA, len=a1, in_use=1
        add(f"LDI r15 {GH8_FSTAB_WORD}")
        add("ST r15 r6")                # slot0.name = a0
        add(f"LDI r15 {GH8_FSTAB_WORD + 1}")
        add(f"LDI r14 {GH8_DATA_WORD}")
        add("ST r15 r14")               # slot0.start = data region
        add(f"LDI r15 {GH8_FSTAB_WORD + 2}")
        add("ST r15 r7")                # slot0.len = a1
        add(f"LDI r15 {GH8_FSTAB_WORD + 3}")
        add("LDI r14 1")
        add("ST r15 r14")               # slot0.in_use = 1
        # copy the payload: scratch[0..1] -> data[0..1] (8-byte files)
        add(f"LDI r15 {GH8_SCRATCH_WORD}")
        add("LD r8 r15")
        add(f"LDI r15 {GH8_DATA_WORD}")
        add("ST r15 r8")
        add(f"LDI r15 {GH8_SCRATCH_WORD + 1}")
        add("LD r8 r15")
        add(f"LDI r15 {GH8_DATA_WORD + 1}")
        add("ST r15 r8")
        add(f":__fssys_{n}_done")
        # result = byte count -> SYS_A0, SYSRET
        add(f"LDI r15 {SYS_A0_WORD}")
        add("ST r15 r7")
        add("SYSRET")
    elif n == GH8_N_READ:
        # scan slot 0 then slot 1 for (name match AND in_use)
        add(f"LDI r15 {GH8_FSTAB_WORD}")
        add("LD r8 r15")                # r8 = slot0.name
        add("CMP r8 r6")
        add(f"JZ :__fsread_{n}_s0")
        add(f"LDI r15 {GH8_FSTAB_WORD + GH8_SLOT_WORDS}")
        add("LD r8 r15")                # r8 = slot1.name
        add("CMP r8 r6")
        add(f"JZ :__fsread_{n}_s1")
        # miss: ERR_NOFILE -> readout[0], result 0
        add(f"LDI r15 {GH8_READOUT_WORD}")
        add("LDI r14 69")               # 'E'
        add("ST r15 r14")
        add(f"LDI r15 {GH8_READOUT_WORD + 1}")
        add("LDI r14 0")
        add("ST r15 r14")
        add(f"LDI r15 {SYS_A0_WORD}")
        add("LDI r14 0")
        add("ST r15 r14")
        add("SYSRET")
        # hit slot 0: copy 2 words data -> readout
        add(f":__fsread_{n}_s0")
        add(f"LDI r15 {GH8_DATA_WORD}")
        add("LD r8 r15")
        add(f"LDI r15 {GH8_READOUT_WORD}")
        add("ST r15 r8")
        add(f"LDI r15 {GH8_DATA_WORD + 1}")
        add("LD r8 r15")
        add(f"LDI r15 {GH8_READOUT_WORD + 1}")
        add("ST r15 r8")
        add(f"LDI r15 {SYS_A0_WORD}")
        add(f"LDI r14 {GH8_FILE_BYTES}")
        add("ST r15 r14")
        add("SYSRET")
        # hit slot 1: same copy from slot1.start (== data region + 8)
        add(f":__fsread_{n}_s1")
        add(f"LDI r15 {GH8_DATA_WORD + 2}")
        add("LD r8 r15")
        add(f"LDI r15 {GH8_READOUT_WORD}")
        add("ST r15 r8")
        add(f"LDI r15 {GH8_DATA_WORD + 3}")
        add("LD r8 r15")
        add(f"LDI r15 {GH8_READOUT_WORD + 1}")
        add("ST r15 r8")
        add(f"LDI r15 {SYS_A0_WORD}")
        add(f"LDI r14 {GH8_FILE_BYTES}")
        add("ST r15 r14")
        add("SYSRET")
    else:  # GH8_N_DEL
        add(f"LDI r15 {GH8_FSTAB_WORD}")
        add("LD r8 r15")
        add("CMP r8 r6")
        add(f"JZ :__fsdel_{n}_s0")
        add(f"LDI r15 {GH8_FSTAB_WORD + GH8_SLOT_WORDS}")
        add("LD r8 r15")
        add("CMP r8 r6")
        add(f"JZ :__fsdel_{n}_s1")
        add(f"LDI r15 {SYS_A0_WORD}")
        add("LDI r14 0")
        add("ST r15 r14")
        add("SYSRET")
        add(f":__fsdel_{n}_s0")
        # unlink: clear name AND in_use (a re-read by this name must miss)
        add(f"LDI r15 {GH8_FSTAB_WORD}")
        add("LDI r14 0")
        add("ST r15 r14")
        add(f"LDI r15 {GH8_FSTAB_WORD + 3}")
        add("LDI r14 0")
        add("ST r15 r14")
        add(f"LDI r15 {SYS_A0_WORD}")
        add("LDI r14 1")
        add("ST r15 r14")
        add("SYSRET")
        add(f":__fsdel_{n}_s1")
        add(f"LDI r15 {GH8_FSTAB_WORD + GH8_SLOT_WORDS}")
        add("LDI r14 0")
        add("ST r15 r14")
        add(f"LDI r15 {GH8_FSTAB_WORD + GH8_SLOT_WORDS + 3}")
        add("LDI r14 0")
        add("ST r15 r14")
        add(f"LDI r15 {SYS_A0_WORD}")
        add("LDI r14 1")
        add("ST r15 r14")
        add("SYSRET")
    return a


def _fs_user_task(n: int, exit_word: int, fault_leg: bool = False) -> List[str]:
    """One USER-mode FS task slice.

    Task A (n=6): store the 2 payload words into the shared scratch window,
      SWI 6 (name 'DATA', 8 bytes), write exit word, KJMP to :__fsdispatch.
    Task B (n=7, or n=7+8 in del_leg): SWI 7 read 'DATA' (kernel copies into
      the read-out window), write exit word; del_leg adds SWI 8 delete + a
      second SWI 7 whose ERR marker overwrites the read-out window; KJMP to
      :__fsdone.
    """
    a: List[str] = []
    add = a.append
    add(f":__fstask_{n}")
    if fault_leg:
        # The violation itself: word 900 (byte 3600) is outside BOX0.
        add("LDI r15 900")
        add("LDI r14 1")
        add("ST r15 r14")
    if n == GH8_N_WRITE:
        add("LDI r17 6")                # a7 = syscall number
        add(f"LDI r10 {GH8_NAME_DATA}")  # a0 = file name 'DATA'
        add("LDI r11 8")                # a1 = byte count
        # stage the payload into the shared scratch window
        add(f"LDI r14 {GH8_PAYLOAD0}")
        add(f"LDI r15 {GH8_SCRATCH_WORD}")
        add("ST r15 r14")
        add(f"LDI r14 {GH8_PAYLOAD1}")
        add(f"LDI r15 {GH8_SCRATCH_WORD + 1}")
        add("ST r15 r14")
        add("SYSCALL r12")
        add(f"LDI r15 {exit_word}")
        add("LDI r14 65261")            # 0xFEED
        add("LDI r4 16")
        add("SHL r14 r4")
        add("ADD r14 r17")
        add("ST r15 r14")
        # switch back to the kernel dispatch loop (privilege boundary)
        add(f"LDI r30 {_GH8_DISPATCH_PC}")
        add("KJMP r30")
    else:
        add("LDI r17 7")                # a7 = read
        add(f"LDI r10 {GH8_NAME_DATA}")
        add("LDI r11 8")                # max bytes
        add("SYSCALL r12")
        add(f"LDI r15 {exit_word}")
        add("LDI r14 65261")
        add("LDI r4 16")
        add("SHL r14 r4")
        add("ADD r14 r17")
        add("ST r15 r14")
        if _GH8_DEL_LEG:
            # delete 'DATA', then re-read: the read-out window must show 'E'
            add("LDI r17 8")
            add(f"LDI r10 {GH8_NAME_DATA}")
            add("LDI r11 0")
            add("SYSCALL r12")
            add("LDI r17 7")
            add(f"LDI r10 {GH8_NAME_DATA}")
            add("LDI r11 8")
            add("SYSCALL r12")
        add(f"LDI r30 {_GH8_RET_PC}")
        add("KJMP r30")
    return a


# del-leg flag consumed by _fs_user_task during pass 2 (module-global, same
# pattern as the GH-6/GH-7 packed-PC globals; reset in fs_kernel_image's
# finally block).
_GH8_DEL_LEG = False


def _fs_kernel_program_text(
    status_word: int = 950,
    del_leg: bool = False,
    fault_leg: bool = False,
) -> str:
    """GH-8 glyph assembly: resident FS kernel + two USER box tasks.

    Sequence (all in-image; the host only runs the image and reads memory):
      1. SUPER prologue zeroes the FSTAB/data/staging words, programs BOX0,
         arms KSYS_PC/KFAULT_PC (loader-seeded packed pixel PCs, two-pass
         bake), latches MODE_LATCH = MODE_USER, KJMPs into task A.
      2. Task A (USER, BOX0) stages 2 payload words into the scratch window
         and SWIs 6 (create+write 'DATA', 8 bytes). The engine marshals
         r17/r10/r11 -> SYS_N/SYS_A0/SYS_A1, traps to :__fsksys; the
         selector routes to the :__fssys_6 slice: FSTAB slot 0 is claimed
         (name/start/len/in_use), payload copied scratch -> data region,
         byte count returned via SYS_A0; SYSRET re-enters USER.
      3. Task A writes its exit word and KJMPs to :__fsdispatch (latch 0 ->
         stays SUPER) -- the round-robin switch point.
      4. :__fsdispatch re-arms the latch and KJMPs into task B; task B SWIs
         7 (read 'DATA'); the kernel copies the file bytes into the read-out
         window; B writes its exit word (del_leg: then SWI 8 delete + SWI 7
         again -- the read-out window ends showing ERR_NOFILE). B KJMPs to
         :__fsdone; the kernel tail writes status 0xCAFE0000 | 8 and HALTs.
    """
    a: List[str] = []
    add = a.append
    add(":__entry")
    add("JMP :__kmain")
    add(":__kmain")
    # ---- zero the FS staging scratch + readout + exits (RAM semantics) ----
    # GH-8b: FSTAB and file data are NOT zeroed at boot — they live in the
    # image pixels and must survive reboot ("the screen is the hard drive").
    for w in (GH8_SCRATCH_WORD, GH8_SCRATCH_WORD + 1,
              GH8_READOUT_WORD, GH8_READOUT_WORD + 1,
              GH8_EXIT_A, GH8_EXIT_B, GH8_BADSYS_WORD, GH8_FAULT_WORD):
        add(f"LDI r15 {w}")
        add("LDI r14 0")
        add("ST r15 r14")
    # ---- program BOX0 (byte range, both tasks' arena) ----
    add(f"LDI r15 {BOX0_LO_WORD}")
    add(f"LDI r14 {GH8_BOX0_LO_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX0_LO_WORD + 1}")
    add(f"LDI r14 {GH8_BOX0_HI_BYTE}")
    add("ST r15 r14")
    # ---- arm the vectors (loader-seeded packed pixel PCs) ----
    add(f"LDI r15 {KSYS_PC_WORD}")
    add(f"LDI r14 {_GH8_KSYS_PC}")
    add("ST r15 r14")
    add(f"LDI r15 {KFAULT_PC_WORD}")
    add(f"LDI r14 {_GH8_FAULT_PC if fault_leg else _GH8_RET_PC}")
    add("ST r15 r14")
    # ---- enter task A: latch USER, KJMP ----
    add(f"LDI r15 {MODE_LATCH_WORD}")
    add("LDI r14 1")
    add("ST r15 r14")
    add(f"LDI r30 {_GH8_TASK_A_PC}")
    add("KJMP r30")
    add("JMP :__fsdone")
    # ---- FS syscall dispatcher (SUPER): reached via KSYS_PC ----
    add(":__fsksys")
    add(f"LDI r15 {SYS_N_ADDR >> 2}")
    add("LD r5 r15")                    # r5 = SYS_N
    add(f"LDI r4 {GH8_N_WRITE}")
    add("CMP r5 r4")
    add(f"JZ :__fssys_{GH8_N_WRITE}")
    add(f"LDI r4 {GH8_N_READ}")
    add("CMP r5 r4")
    add(f"JZ :__fssys_{GH8_N_READ}")
    add(f"LDI r4 {GH8_N_DEL}")
    add("CMP r5 r4")
    add(f"JZ :__fssys_{GH8_N_DEL}")
    # unknown syscall: record 'E' and SYSRET dry
    add(f"LDI r15 {GH8_BADSYS_WORD}")
    add("LDI r14 69")
    add("ST r15 r14")
    add("SYSRET")
    for n in (GH8_N_WRITE, GH8_N_READ, GH8_N_DEL):
        for line in _fs_syscall_handler(n):
            add(line)
    # ---- round-robin dispatch loop: switch A -> B ----
    add(":__fsdispatch")
    add(f"LDI r15 {MODE_LATCH_WORD}")
    add("LDI r14 1")
    add("ST r15 r14")
    add(f"LDI r30 {_GH8_TASK_B_PC}")
    add("KJMP r30")
    add("JMP :__fsdone")
    # ---- fault handler (SUPER): records the violation, halts via kernel ----
    add(":__fsfault")
    if fault_leg:
        add(f"LDI r15 {GH8_FAULT_WORD}")
        add(f"LDI r14 {GH8_FAULT_SEEN}")
        add("ST r15 r14")
    # fall through to the done/halt tail
    # ---- done: status = 0xCAFE0000 | 8, halt ----
    add(":__fsdone")
    add("LDI r9 8")
    add("LDI r3 51966")        # 0xCAFE
    add("LDI r4 16")
    add("SHL r3 r4")
    add("ADD r3 r9")
    add(f"LDI r15 {status_word}")
    add("ST r15 r3")
    add("HALT")
    # ---- user task A (USER, BOX0): create+write 'DATA' ----
    for line in _fs_user_task(GH8_N_WRITE, GH8_EXIT_A, fault_leg):
        add(line)
    # ---- user task B (USER, BOX0): read (del_leg: + delete + re-read) ----
    for line in _fs_user_task(GH8_N_READ, GH8_EXIT_B, False):
        add(line)
    return "\n".join(a) + "\n"


def fs_kernel_image(
    atlas: Any,
    status_word: int = 950,
    del_leg: bool = False,
    fault_leg: bool = False,
    cols_instrs: int = 8,
    min_rows: int = 16,
    out_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """GH-8: bake ONE image with the resident in-image filesystem kernel.
    Two passes: pass 1 fixes the packed pixel PCs (:__fsksys/:__fsdispatch/
    :__fsfault/:__fstask_6/:__fstask_7), pass 2 bakes the final image. The
    atlas is accepted for GH-2 continuity but the FS kernel needs no atlas
    tiles. Persistence model: the kernel's own ST stores land in the image
    array the runner executes (memory IS the disk); saving the mutated
    ndarray after run 1 and re-running it offline re-reads the same file."""
    global _GH8_DEL_LEG
    _GH8_DEL_LEG = del_leg
    try:
        txt1 = _fs_kernel_program_text(status_word, del_leg, fault_leg)
        _, coords1 = assemble_glyph_to_pixels(txt1, cols_instrs=cols_instrs, min_rows=min_rows)

        def packed(label: str) -> int:
            col, row = coords1[label]
            # engine packing: low 16 bits = column, high 16 = row
            return (col & 0xFFFF) | ((row & 0xFFFF) << 16)

        global _GH8_KSYS_PC, _GH8_FAULT_PC, _GH8_RET_PC, _GH8_DISPATCH_PC, _GH8_TASK_A_PC, _GH8_TASK_B_PC
        _GH8_KSYS_PC = packed(":__fsksys")
        _GH8_FAULT_PC = packed(":__fsfault")   # bound BEFORE pass 2 or the
        _GH8_RET_PC = packed(":__fsdone")      # fault leg vectors to (0,0)
        _GH8_DISPATCH_PC = packed(":__fsdispatch")
        _GH8_TASK_A_PC = packed(":__fstask_6")
        _GH8_TASK_B_PC = packed(":__fstask_7")
        # Pass 2 re-materializes the program with the real packed PCs baked
        # in (assemble is deterministic, so the label coords are identical).
        return bake_image(
            _fs_kernel_program_text(status_word, del_leg, fault_leg),
            atlas=None,
            cols_instrs=cols_instrs,
            min_rows=min_rows,
            out_path=out_path,
        )
    finally:
        _GH8_KSYS_PC = _GH8_FAULT_PC = _GH8_RET_PC = 0
        _GH8_DISPATCH_PC = _GH8_TASK_A_PC = _GH8_TASK_B_PC = 0
        _GH8_DEL_LEG = False


# --- GH-9 program loader ABI (fixed word indices) ----------------------------
# GH-3's patch window aimed at WHOLE PROGRAMS: the kernel patches the mailbox
# pixels into its own code region, copies the mailbox argv pair into the
# task's BOX0 arena, and KJMPs into the patched window USER-mode with the
# argv pointer in r10. BOX0 = [700..768) (GH-8 scheme); the argv block at
# 750 (argv0, argv1) and the result slot at 754 are in-arena so the
# injected program's LD/ST stay legal.
GH9_ARGV_WORD = 750        # argv block: argv0 @750, argv1 @751 (in BOX0)
GH9_ARGV_RESULT = 754      # program's result slot: argv base + 4 (in BOX0)
GH9_EXIT_WORD = 703        # injected program's exit word (in BOX0)
GH9_FAULT_WORD = 731       # fault-leg verdict (0xFA171)
GH9_BOX0_LO_BYTE = 4 * 700
GH9_BOX0_HI_BYTE = 4 * 768
GH9_MAILBOX_FLAG = 960     # 1 = patch+launch from mailbox (kernel clears)
GH9_MAILBOX_N_PX = 961     # pixel words to copy into the window
# Payload MUST stay below word 1024: GH-8b aliases words [1024,1280) to
# image pixels (2 px/word, reads wrap mod image size), so payload words
# there read back as kernel-code pixels. [800,896) holds the full 96-word
# window payload and is clear of BOX0 [700,768), status 950, flag 960,
# n_px 961, and the legacy GH-3 payload at 964.
GH9_MAILBOX_DATA = 800     # mailbox payload words start here
GH9_FAULT_SEEN = 0xFA171
GH9_N_INSTRS = 24          # patch-window capacity (instructions)
GH9_N_PX = GH9_N_INSTRS * 4  # pixel words in the window

# Packed pixel PCs of the GH-9 launch/fault/done labels + window, bound by
# loader_kernel_image() before the second (final) assemble pass.
_GH9_LAUNCH_PC = 0
_GH9_FAULT_PC = 0
_GH9_RET_PC = 0
_GH9_WINDOW_PC = 0
_GH9_FTASK_PC = 0


def _gh9_kernel_program_text(
    status_word: int = 950,
    fault_leg: bool = False,
) -> str:
    """GH-9 glyph assembly: resident loader kernel.

    Sequence (all in-image; the host runs the image and reads memory):
      1. SUPER prologue zeroes the launch ABI words, programs BOX0, arms
         KSYS_PC/KFAULT_PC (loader-seeded packed pixel PCs), latches
         MODE_LATCH = MODE_USER and KJMPs to the LAUNCH leg.
      2. LAUNCH leg (USER-mode context, but only stores in-box): reads the
         mailbox flag (word 960).
           - flag == 1 (fresh load): clear it, read mailbox n_px, copy
             n_px pixel words mailbox -> the reserved window via the
             PARALLEL_ST loop (GH-3 machinery), copy the mailbox argv pair
             (data[0], data[1]) into the argv block at 750/751, then KJMP
             into the window with r10 = argv pointer (750).
           - flag == 0 (offline re-run): the window already holds the
             resident program pixels; seed r10 = argv pointer and KJMP
             straight in (exec semantics: same text, fresh argv).
         The argv copy happens in the launch leg, NOT the host: the host
         only fills the mailbox; the kernel hands argv to the program.
      3. The injected program (host-assembled pixels) runs USER-mode
         inside the window: LD argv from r10, compute, ST the result to
         argv base + 4, write the exit word, HALT.
      4. KFAULT_PC handler: records 0xFA171 and falls into the done tail.
      5. Done tail: status = 0xCAFE0000 | 9, HALT. The engine halts the
         whole machine when the injected program's HALT executes, so the
         status word is written by whichever path ends the run — the
         fault/done tails cover the fault leg, and the online/offline
         success paths end at the program's own HALT (receipt reads the
         result + exit words; status 0xCAFE0009 is seeded by the launch
         leg BEFORE the KJMP so a clean run always shows it).

    fault_leg replaces the launch leg's KJMP with a KJMP to a fault task
    whose first store is word 900 (outside BOX0) — E-K1 fires, the handler
    records the verdict, and the clean-exit words stay zero.
    """
    a: List[str] = []
    add = a.append
    add(":__entry")
    add("JMP :__kmain")
    add(":__kmain")
    # ---- zero the KERNEL-owned receipt words ONLY. The mailbox flag/n_px
    # and the argv block are HOST-provided inputs (the host arms the
    # mailbox / seeds argv before step zero); zeroing them here destroyed
    # the online leg (flag read 0 -> the kernel took the offline branch
    # into an unpatched window) and blanked the offline leg's argv. ----
    for w in (GH9_ARGV_RESULT, GH9_EXIT_WORD, GH9_FAULT_WORD):
        add(f"LDI r15 {w}")
        add("LDI r14 0")
        add("ST r15 r14")
    # ---- program BOX0 (byte range) ----
    add(f"LDI r15 {BOX0_LO_WORD}")
    add(f"LDI r14 {GH9_BOX0_LO_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX0_LO_WORD + 1}")
    add(f"LDI r14 {GH9_BOX0_HI_BYTE}")
    add("ST r15 r14")
    # ---- arm the vectors (loader-seeded packed pixel PCs) ----
    add(f"LDI r15 {KSYS_PC_WORD}")
    add(f"LDI r14 {_GH9_LAUNCH_PC}")
    add("ST r15 r14")
    add(f"LDI r15 {KFAULT_PC_WORD}")
    add(f"LDI r14 {_GH9_FAULT_PC if fault_leg else _GH9_RET_PC}")
    add("ST r15 r14")
    # ---- seed the clean-run status NOW (a clean run ends at the program's
    # own HALT; the status word must already read 0xCAFE0009 by then) ----
    add("LDI r3 51966")        # 0xCAFE
    add("LDI r4 16")
    add("SHL r3 r4")
    add("LDI r4 9")
    add("ADD r3 r4")           # 0xCAFE0009
    add(f"LDI r15 {status_word}")
    add("ST r15 r3")
    # ---- enter the launch leg SUPER (no latch here). MODE_LATCH is a
    # ONE-SHOT consumed by the next KJMP (engine clears it on consume), so
    # it must be armed immediately before the KJMP that enters the WINDOW,
    # not before this leg: the launch leg clears the mailbox flag (word 960,
    # outside BOX0 -- an out-of-box store that would fault if we were in
    # USER mode) and runs the PARALLEL_ST patch loop. ----
    add(f"LDI r30 {_GH9_LAUNCH_PC}")
    add("KJMP r30")
    add("JMP :__g9done")
    # ---- launch leg (SUPER; mailbox-clear + patch loop are kernel work) ----
    add(":__g9launch")
    if fault_leg:
        # The violation itself: word 900 (byte 3600) is outside BOX0.
        # Latch USER first so the store trips E-K1 (stores in SUPER are
        # never box-checked), then take the fault-task entry directly.
        add(f"LDI r15 {MODE_LATCH_WORD}")
        add("LDI r14 1")
        add("ST r15 r14")
        add(f"LDI r30 {_GH9_FTASK_PC}")
        add("KJMP r30")       # latch consumed here: task runs USER
        add("JMP :__g9done")  # unreachable: E-K1 fires on the ST
    # mailbox flag == 1? (fresh load) else relaunch resident window
    add(f"LDI r15 {GH9_MAILBOX_FLAG}")
    add("LD r1 r15")
    add("LDI r2 1")
    add("CMP r1 r2")
    add("JZ :__g9doload")
    # offline relaunch: argv pointer in r10, straight into the window
    add(f"LDI r10 {GH9_ARGV_WORD}")
    add(f"LDI r15 {MODE_LATCH_WORD}")
    add("LDI r14 1")
    add("ST r15 r14")
    add(f"LDI r30 {_GH9_WINDOW_PC}")
    add("KJMP r30")
    add("JMP :__g9done")
    # fresh load: clear the flag, patch the window, launch. NB: the argv
    # block is NOT copied from the mailbox — the mailbox carries ONLY
    # program pixels; the HOST (process creator) seeds argv at 750/751
    # before step zero. Copying data[0..1] here clobbered the host's argv
    # with the program's first two pixel words.
    add(":__g9doload")
    add(f"LDI r15 {GH9_MAILBOX_FLAG}")
    add("LDI r14 0")
    add("ST r15 r14")
    # argv pointer into the launch register BEFORE the boundary KJMP
    add(f"LDI r10 {GH9_ARGV_WORD}")
    # patch the window: mailbox -> PARALLEL_ST loop (GH-3 machinery).
    # r13 = pixel-word count (bounded to the window capacity), r14 = source,
    # r15 = destination pixel address. NB: the host has already bound the
    # real window pixel address into _GH9_WINDOW_PC's low bits at bake time;
    # the loop copies linear pixel words (image IS addressable at addr%w*h).
    add(f"LDI r15 {GH9_MAILBOX_N_PX}")
    add("LD r13 r15")
    add(f"LDI r14 {GH9_MAILBOX_DATA}")
    # window destination pixel address = packed col/row of the window,
    # expanded to a linear pixel index by the baker (see loader_kernel_image).
    add(f"LDI r15 {_GH9_WINDOW_DST}")
    add(":__g9patchloop")
    add("LD r8 r14")           # (loop head) fetch is guarded below
    add("CMP r13 r0")
    add("JZ :__g9patchdone")
    add("PARALLEL_ST r15 r8 1")
    add("LDI r4 1")
    add("ADD r14 r4")
    add("ADD r15 r4")
    add("SUB r13 r4")
    add("JMP :__g9patchloop")
    add(":__g9patchdone")
    # arm the one-shot USER latch for the program entry (must be the NEXT
    # and ONLY KJMP after this store)
    add(f"LDI r15 {MODE_LATCH_WORD}")
    add("LDI r14 1")
    add("ST r15 r14")
    add(f"LDI r30 {_GH9_WINDOW_PC}")
    add("KJMP r30")
    add("JMP :__g9done")
    # ---- fault task (fault_leg only): entered USER via the latch; the very
    # first store goes outside BOX0 so E-K1 fires on it ----
    add(":__g9ftask")
    if fault_leg:
        add("LDI r15 900")
        add("LDI r14 1")
        add("ST r15 r14")     # E-K1: out-of-box USER store -> KFAULT_PC
    add("JMP :__g9done")
    # ---- fault handler (SUPER): record the verdict, fall to done ----
    add(":__g9fault")
    if fault_leg:
        add(f"LDI r15 {GH9_FAULT_WORD}")
        add(f"LDI r14 {GH9_FAULT_SEEN}")
        add("ST r15 r14")
    # fall through to the done tail
    # ---- done: status = 0xCAFE0000 | 9, halt ----
    add(":__g9done")
    add("LDI r3 51966")
    add("LDI r4 16")
    add("SHL r3 r4")
    add("LDI r4 9")
    add("ADD r3 r4")
    add(f"LDI r15 {status_word}")
    add("ST r15 r3")
    add("HALT")
    # ---- reserved patch window (patched at run time with the program) ----
    add(":__g9window")
    for _ in range(GH9_N_INSTRS):
        add("HALT")
    return "\n".join(a) + "\n"


# Window destination pixel address, bound by loader_kernel_image() at bake
# time (glyph code cannot name a pixel coordinate; the baker injects the
# linear pixel index of :__g9window, same two-pass pattern as the PCs).
_GH9_WINDOW_DST = 0


def loader_kernel_image(
    atlas: Any,
    status_word: int = 950,
    fault_leg: bool = False,
    cols_instrs: int = 8,
    min_rows: int = 16,
    out_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """GH-9: bake ONE image with the resident program-loader kernel.
    Two passes: pass 1 fixes the packed pixel PCs (:__g9launch/:__g9fault/
    :__g9done/:__g9window) and the window's linear destination pixel
    address; pass 2 bakes the final image. The atlas is accepted for GH-2
    continuity but the loader needs no atlas tiles. Loader model: the HOST
    assembles program pixels + argv into the mailbox/RAM before step zero
    (process creation), the kernel PARALLEL_STs the pixels into its own
    code region (patch) and KJMPs in USER-mode at the argv pointer —
    GH-3's patch window executed as exec()."""
    global _GH9_LAUNCH_PC, _GH9_FAULT_PC, _GH9_RET_PC, _GH9_WINDOW_PC, _GH9_FTASK_PC, _GH9_WINDOW_DST
    try:
        txt1 = _gh9_kernel_program_text(status_word, fault_leg)
        _, coords1 = assemble_glyph_to_pixels(txt1, cols_instrs=cols_instrs, min_rows=min_rows)

        def packed(label: str) -> int:
            col, row = coords1[label]
            # engine packing: low 16 bits = column, high 16 = row
            return (col & 0xFFFF) | ((row & 0xFFFF) << 16)

        _GH9_LAUNCH_PC = packed(":__g9launch")
        _GH9_FAULT_PC = packed(":__g9fault")   # bound BEFORE pass 2 or the
        _GH9_RET_PC = packed(":__g9done")      # fault leg vectors to (0,0)
        _GH9_WINDOW_PC = packed(":__g9window")
        _GH9_FTASK_PC = packed(":__g9ftask")
        wcol, wrow = coords1[":__g9window"]
        # linear pixel index of the window's first instruction pixel
        _GH9_WINDOW_DST = wrow * (cols_instrs * 4) + (wcol * 4)
        # Pass 2 re-materializes the program with the real PCs + window
        # destination baked in (assemble is deterministic, so the label
        # coords are identical).
        return bake_image(
            _gh9_kernel_program_text(status_word, fault_leg),
            atlas=None,
            cols_instrs=cols_instrs,
            min_rows=min_rows,
            out_path=out_path,
        )
    finally:
        _GH9_LAUNCH_PC = _GH9_FAULT_PC = _GH9_RET_PC = 0
        _GH9_WINDOW_PC = _GH9_FTASK_PC = _GH9_WINDOW_DST = 0


# --- GH-10 shell ABI (fixed word indices) -------------------------------------
# The shell polls a host-armed cmd buffer (words 903..905), dispatches
# echo/ps, and writes its output + prompt verdict to the uart buffer
# (words 910..912). Word placement: below the pixel-aliased FS window
# [1024,1280), clear of BOX0 [700,768), status 950, the GH-9 mailbox
# [800,896) + flag 960 / n_px 961, and the GH-3 legacy payload at 964.
GH10_CMD_FLAG = 903        # host -> shell: 1 = command ready (shell clears)
GH10_CMD_WORD0 = 904       # command word (host-stored: 'echo'/'ps' u32)
GH10_CMD_WORD1 = 905       # echo payload word (host-stored: 'hi' packed)
GH10_UART = 910            # shell output: one packed 4-byte word
GH10_UART_LEN = 911        # output byte count
GH10_ERR_WORD = 912        # unknown-command verdict ('E' = 69)
GH10_PROMPT = 0x3E202020   # idle prompt verdict: " >>" packed (32-bit word)
GH10_EXIT_WORD = 703       # shell session's exit word (in BOX0)
GH10_FAULT_WORD = 731      # fault-leg verdict (0xFA171)
GH10_FAULT_SEEN = 0xFA171
GH10_CMD_ECHO = (ord('e') | (ord('c') << 8) | (ord('h') << 16) | (ord('o') << 24))
GH10_CMD_PS = (ord('p') | (ord('s') << 8))
GH10_PS_WORD = (ord('A') | (ord('B') << 8) | (2 << 16))  # "AB" + task count
GH10_PS_LEN = 3
GH10_EXIT_OK = 0xFEED000A
GH10_BOX0_LO_BYTE = 4 * 700
GH10_BOX0_HI_BYTE = 4 * 768

# Packed pixel PCs of the GH-10 dispatch/fault/done labels, bound by
# shell_kernel_image() before the second (final) assemble pass.
_GH10_DISPATCH_PC = 0
_GH10_FAULT_PC = 0
_GH10_RET_PC = 0


def _gh10_kernel_program_text(
    status_word: int = 950,
    fault_leg: bool = False,
) -> str:
    """GH-10 glyph assembly: resident shell kernel.

    Sequence (all in-image; the host only arms the cmd buffer and runs):
      1. SUPER prologue zeroes the uart/error/exit receipt words, programs
         BOX0, arms KSYS_PC/KFAULT_PC (loader-seeded packed pixel PCs,
         two-pass bake), and KJMPs to the poll leg.
      2. Poll leg: reads CMD_FLAG (word 903).
           - flag == 1: clear it, load the command word (904) and payload
             word (905), and select:
               * 'echo': uart = payload word, uart_len = 2 ('hi' = 2 bytes
                 packed in the payload word's low 16 bits), exit = 0xFEED000A.
               * 'ps': uart = packed 'A'|'B'<<8|2<<16 (the two GH-7 task
                 letters + the task count), uart_len = 3, exit = 0xFEED000A.
               * anything else: ERR = 'E' (the uart stays untouched).
             then KJMP to the done tail.
           - flag == 0 (idle): uart = GH10_PROMPT (" >>" packed), fall
             through to the done tail.
      3. Done tail: status = 0xCAFE0000 | 10, HALT.

    fault_leg replaces the poll leg's first act with an out-of-box store
    (word 900): E-K1 fires, the handler records 0xFA171, the uart stays 0.

    32-bit decomposition notes (glyph has no byte lanes):
      - 'echo' = 0x6F636863. The LOW-24 compare uses LDI + AND 0xFFFFFF
        (SHR of a 24+-bit value would need the high half); the top byte
        ('o' << 24) is compared separately via SHR 24 + AND 0xFF. 24-bit
        AND masks stay LDI-safe.
      - 'ps' = 0x00007073: CMP against the LDI-safe full word.
      - uart_len for 'hi' is the constant 2; the payload word is copied
        verbatim (the host packed the bytes).
    """
    a: List[str] = []
    add = a.append
    add(":__entry")
    add("JMP :__kmain")
    add(":__kmain")
    # ---- zero the KERNEL-owned receipt words (host inputs at 903..905 and
    # the pixel FS are NOT touched) ----
    for w in (GH10_UART, GH10_UART_LEN, GH10_ERR_WORD, GH10_EXIT_WORD,
              GH10_FAULT_WORD):
        add(f"LDI r15 {w}")
        add("LDI r14 0")
        add("ST r15 r14")
    # ---- program BOX0 ----
    add(f"LDI r15 {BOX0_LO_WORD}")
    add(f"LDI r14 {GH10_BOX0_LO_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX0_LO_WORD + 1}")
    add(f"LDI r14 {GH10_BOX0_HI_BYTE}")
    add("ST r15 r14")
    # ---- arm the vectors ----
    add(f"LDI r15 {KSYS_PC_WORD}")
    add(f"LDI r14 {_GH10_DISPATCH_PC}")
    add("ST r15 r14")
    add(f"LDI r15 {KFAULT_PC_WORD}")
    add(f"LDI r14 {_GH10_FAULT_PC if fault_leg else _GH10_RET_PC}")
    add("ST r15 r14")
    add(f"LDI r30 {_GH10_DISPATCH_PC}")
    if fault_leg:
        # GH-6/7/9 fault-leg pattern: the ENTRY KJMP is the one-shot latch
        # consumer. Arm MODE_LATCH=USER before it so the KJMP lands the
        # poll leg in USER; the out-of-box ST there then trips E-K1.
        # (A latch store after the KJMP is inert — nothing consumes it.)
        add(f"LDI r15 {MODE_LATCH_WORD}")
        add("LDI r14 1")
        add("ST r15 r14")
    add("KJMP r30")
    add("JMP :__g10done")
    # ---- poll/dispatch leg (SUPER; cmd words live outside any box) ----
    add(":__g10poll")
    if fault_leg:
        # The violation itself: word 900 (byte 3600) is outside BOX0.
        # GH-6/7/9 pattern: the KJMP that enters this leg is the ONE-SHOT
        # latch consumer — it cleared MODE_LATCH and dropped us to USER.
        # A store in SUPER is never box-checked, so the violating ST must
        # run while still USER (latch already consumed — do NOT re-STORE
        # the latch; that would need another KJMP to take effect).
        add("LDI r15 900")
        add("LDI r14 1")
        add("ST r15 r14")     # E-K1: out-of-box USER store -> KFAULT_PC
        add("JMP :__g10done")
    add(f"LDI r15 {GH10_CMD_FLAG}")
    add("LD r1 r15")
    add("LDI r2 1")
    add("CMP r1 r2")
    add("JZ :__g10cmd")
    # ---- idle: prompt verdict, done ----
    add(f"LDI r4 {GH10_PROMPT & 0xFFFFFF}")       # low 24 bits
    add(f"LDI r5 {GH10_PROMPT >> 24}")            # top byte
    add("LDI r7 24")                              # shift count in a SPARE reg:
    add("SHL r5 r7")                              # (r4 as count clobbered the
    add("OR r4 r5")                               #  low-24 payload — GH-10 fix 2)
    add(f"LDI r15 {GH10_UART}")
    add("ST r15 r4")
    add("JMP :__g10done")
    # ---- command ready: consume the flag, load cmd + payload ----
    add(":__g10cmd")
    add(f"LDI r15 {GH10_CMD_FLAG}")
    add("LDI r14 0")
    add("ST r15 r14")
    add(f"LDI r15 {GH10_CMD_WORD0}")
    add("LD r5 r15")                              # r5 = command word
    add(f"LDI r15 {GH10_CMD_WORD1}")
    add("LD r6 r15")                              # r6 = payload word
    # select on LOW 24 bits of the command word. AND is TWO-operand in this
    # ISA (rd, rs2): 'AND r5 r7' -> r5 = r5 & r7; a third operand is ignored
    # by the assembler (rd=8/rs2=5 misparse made every verb compare miss).
    add("LDI r7 16777215")                        # 0xFFFFFF
    add("AND r5 r7")                              # r5 = cmd & 0xFFFFFF
    # 'echo' low24 = 'e' | 'c'<<8 | 'h'<<16 = 0x686365
    add("LDI r7 6841189")                         # 0x686365 = 'e'|'c'<<8|'h'<<16
    add("CMP r5 r7")
    add("JZ :__g10echo")
    # 'ps' low24 = 'p' | 's'<<8 = 0x7073
    add("LDI r7 29552")                           # 0x7370 = 'p'|'s'<<8
    add("CMP r5 r7")
    add("JZ :__g10ps")
    # top byte check for 'echo' ('o' << 24): full-24 select already matched
    # 'echo' — an unknown verb falls through here.
    add(f"LDI r15 {GH10_ERR_WORD}")
    add("LDI r14 69")                             # 'E'
    add("ST r15 r14")
    add(f"LDI r15 {GH10_EXIT_WORD}")
    add("LDI r14 10")
    add("ST r15 r14")
    add("JMP :__g10done")
    # ---- echo: uart = payload word, len = 2 ----
    add(":__g10echo")
    add(f"LDI r15 {GH10_UART}")
    add("ST r15 r6")
    add(f"LDI r15 {GH10_UART_LEN}")
    add("LDI r14 2")
    add("ST r15 r14")
    add(f"LDI r15 {GH10_EXIT_WORD}")
    add("LDI r14 65261")                          # 0xFEED low16 -> exit |=
    add("LDI r4 16")
    add("SHL r14 r4")
    add("LDI r4 10")
    add("ADD r14 r4")                             # 0xFEED000A
    add("ST r15 r14")
    add("JMP :__g10done")
    # ---- ps: uart = 'A'|'B'<<8|2<<16, len = 3 ----
    add(":__g10ps")
    add(f"LDI r15 {GH10_UART}")
    add(f"LDI r14 {GH10_PS_WORD}")
    add("ST r15 r14")
    add(f"LDI r15 {GH10_UART_LEN}")
    add(f"LDI r14 {GH10_PS_LEN}")
    add("ST r15 r14")
    add(f"LDI r15 {GH10_EXIT_WORD}")
    add("LDI r14 65261")
    add("LDI r4 16")
    add("SHL r14 r4")
    add("LDI r4 10")
    add("ADD r14 r4")
    add("ST r15 r14")
    add("JMP :__g10done")
    # ---- fault handler (SUPER): record the verdict, fall to done ----
    add(":__g10fault")
    if fault_leg:
        add(f"LDI r15 {GH10_FAULT_WORD}")
        add(f"LDI r14 {GH10_FAULT_SEEN}")
        add("ST r15 r14")
    # fall through to the done tail
    # ---- done: status = 0xCAFE0000 | 10, halt ----
    add(":__g10done")
    add("LDI r9 10")
    add("LDI r3 51966")        # 0xCAFE
    add("LDI r4 16")
    add("SHL r3 r4")
    add("ADD r3 r9")
    add(f"LDI r15 {status_word}")
    add("ST r15 r3")
    add("HALT")
    return "\n".join(a) + "\n"


def shell_kernel_image(
    atlas: Any,
    status_word: int = 950,
    fault_leg: bool = False,
    cols_instrs: int = 8,
    min_rows: int = 16,
    out_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """GH-10: bake ONE image with the resident shell kernel.
    Two passes: pass 1 fixes the packed pixel PCs (:__g10poll/:__g10fault/
    :__g10done), pass 2 bakes the final image. The atlas is accepted for
    GH-2 continuity but the shell needs no atlas tiles. Shell model: the
    HOST arms the cmd buffer (words 903..905) before step zero — process
    creation writes the command, the shell polls the flag in-image, services
    echo/ps/unknown, and reports through the uart buffer (words 910..912)."""
    global _GH10_DISPATCH_PC, _GH10_FAULT_PC, _GH10_RET_PC
    try:
        txt1 = _gh10_kernel_program_text(status_word, fault_leg)
        _, coords1 = assemble_glyph_to_pixels(txt1, cols_instrs=cols_instrs, min_rows=min_rows)

        def packed(label: str) -> int:
            col, row = coords1[label]
            # engine packing: low 16 bits = column, high 16 = row
            return (col & 0xFFFF) | ((row & 0xFFFF) << 16)

        _GH10_DISPATCH_PC = packed(":__g10poll")
        _GH10_FAULT_PC = packed(":__g10fault")   # bound BEFORE pass 2 or the
        _GH10_RET_PC = packed(":__g10done")      # fault leg vectors to (0,0)
        # Pass 2 re-materializes the program with the real packed PCs baked
        # in (assemble is deterministic, so the label coords are identical).
        return bake_image(
            _gh10_kernel_program_text(status_word, fault_leg),
            atlas=None,
            cols_instrs=cols_instrs,
            min_rows=min_rows,
            out_path=out_path,
        )
    finally:
        _GH10_DISPATCH_PC = _GH10_FAULT_PC = _GH10_RET_PC = 0


# --- GH-13 multi-agent workspaces ABI (fixed word indices) --------------------
# Generalizes GH-7's two boxes to THREE per-agent arenas using the engine's
# third permitted range (BOX2). BOX0 = agent A [700..717), BOX1 = agent B
# [718..735), BOX2 = agent C [736..753). Agent->agent and agent->kernel
# messages flow through a kernel-mediated mailbox word (754); no agent can
# write outside its own box (E-K1 trap = the isolation proof).
GH13_TURN_WORD = 705         # kernel round-robin marker (SUPER-only writes)
GH13_EXIT_A = 703            # agent A exit status word (in BOX0)
GH13_UART_A_WORD = 710       # agent A output (kernel-written)
GH13_UART_A_WORD2 = 711
GH13_UART_A_LEN = 712        # bytes the kernel appended (4)
GH13_TASK_A_BUF = 713        # agent A scratch (in BOX0)
GH13_EXIT_B = 723            # agent B exit status word (in BOX1)
GH13_READOUT_B = 720         # agent B read-out (kernel copies mailbox here)
GH13_TASK_B_BUF = 727        # agent B scratch (in BOX1)
GH13_VERDICT_B = 724         # B's in-box verify verdict: 'V' (86) or 'E'
GH13_EXIT_C = 739            # agent C exit status word (in BOX2)
GH13_UART_C_WORD0 = 746      # agent C read-out: A's payload (via KSYS)
GH13_UART_C_WORD1 = 747      # agent C read-out: B's verdict (via KSYS)
GH13_TASK_C_BUF = 741        # agent C scratch (in BOX2)
GH13_MAILBOX_WORD = 754      # agent->agent mailbox (kernel-mediated)
GH13_BADSYS_WORD = 758       # unknown-syscall marker ('E' = 69)
GH13_FAULT_WORD = 759        # fault-leg verdict (0xFA171)
GH13_BOX0_LO_BYTE = 4 * 700  # agent A arena [700..717)
GH13_BOX0_HI_BYTE = 4 * 717
GH13_BOX1_LO_BYTE = 4 * 718  # agent B arena [718..735)
GH13_BOX1_HI_BYTE = 4 * 735
GH13_BOX2_LO_BYTE = 4 * 736  # agent C arena [736..753)
GH13_BOX2_HI_BYTE = 4 * 753
GH13_BOX2_LO_WORD = BOX2_LO_ADDR >> 2
GH13_BOX2_HI_WORD = BOX2_HI_ADDR >> 2
GH13_N_A, GH13_N_B, GH13_N_C = 6, 7, 8
GH13_FAULT_SEEN = 0xFA171
GH13_PATTERN = 0xA5A5A5A5    # A writes; B verifies; C collects
GH13_VERIFIED = 86           # 'V'
GH13_STATUS_ID = 8           # final status = 0xCAFE0000 | 8

# Packed pixel PCs of the GH-13 selector/handler/fault/dispatch/agent labels,
# bound by multiagent_kernel_image() before the second (final) assemble pass.
_GH13_KSYS_PC = 0
_GH13_FAULT_PC = 0
_GH13_RET_PC = 0
_GH13_DISPATCH_B_PC = 0
_GH13_DISPATCH_C_PC = 0
_GH13_TASK_A_PC = 0
_GH13_TASK_B_PC = 0
_GH13_TASK_C_PC = 0


def _gh13_ksys_slice(n: int) -> List[str]:
    """One SUPER-mode dispatcher slice for agent syscall n (A: 6, B: 7, C: 8).

    Reached only when SYS_N == n (the :__ksys selector checked first); an
    unknown number records 'E' at GH13_BADSYS_WORD and SYSRETs dry.

    SYS 6 (agent A, writer): copies A's staged payload (r6 = mem[SYS_A0])
      into A's uart (710) AND the agent->agent mailbox (754), returns 4.
    SYS 7 (agent B, reader): copies the mailbox (754) into B's read-out
      (720) — B NEVER touches A's arena; the mailbox syscall is the ONLY
      channel — and returns 4. B verifies the pattern in its own box after
      SYSRET (user-side code, not here).
    SYS 8 (agent C, coordinator): packs A's payload (710) and B's verdict
      (724) into C's read-out (746/747), returns 8.
    """
    a: List[str] = []
    add = a.append
    add(f":__ksys_{n}")
    add(f"LDI r15 {SYS_A0_ADDR >> 2}")
    if n == GH13_N_A:
        add("LD r6 r15")                # r6 = a0 = A's buffer WORD index
        add("LD r7 r6")                 # r7 = mem[buf] = A's staged payload
        add(f"LDI r15 {GH13_UART_A_WORD}")
        add("ST r15 r7")                # A's own uart
        add(f"LDI r15 {GH13_MAILBOX_WORD}")
        add("ST r15 r7")                # the agent->agent mailbox
        add(f"LDI r15 {GH13_UART_A_LEN}")
        add("LDI r14 4")
        add("ST r15 r14")
        result = 4
    elif n == GH13_N_B:
        add(f"LDI r15 {GH13_MAILBOX_WORD}")
        add("LD r6 r15")                # r6 = mailbox contents
        add(f"LDI r15 {GH13_READOUT_B}")
        add("ST r15 r6")                # B's read-out (mailbox ONLY)
        result = 4
    else:
        add(f"LDI r15 {GH13_UART_A_WORD}")
        add("LD r6 r15")                # r6 = A's payload
        add(f"LDI r15 {GH13_UART_C_WORD0}")
        add("ST r15 r6")
        add(f"LDI r15 {GH13_VERDICT_B}")
        add("LD r7 r15")                # r7 = B's verdict
        add(f"LDI r15 {GH13_UART_C_WORD1}")
        add("ST r15 r7")
        result = 8
    add(f"LDI r15 {SYS_A0_WORD}")
    add(f"LDI r14 {result}")
    add("ST r15 r14")                   # result -> SYS_A0 for SYSRET
    add("SYSRET")
    return a


def _gh13_agent_task(
    n: int,
    buf_word: int,
    exit_word: int,
    fault_leg: bool = False,
) -> List[str]:
    """One USER-mode agent slice confined to its own box.

    A (n=6): (fault leg: out-of-box violation first) stages the pattern into
      its scratch, SYSCALL 6, exit write, KJMP to the B-dispatch leg.
    B (n=7): SYSCALL 7 (mailbox -> read-out), verifies the pattern IN ITS
      OWN BOX, writes 'V'/'E' at its verdict word, exit write, KJMP to the
      C-dispatch leg.
    C (n=8): SYSCALL 8 (collect A's payload + B's verdict), exit write,
      KJMP to the kernel done tail (latch 0 -> lands SUPER).
    """
    a: List[str] = []
    add = a.append
    add(f":__task_{n}")
    if fault_leg:
        # The violation itself: word 900 (byte 3600) is outside every box,
        # so this USER store vectors to KFAULT_PC in USER mode (the entering
        # KJMP is the one-shot latch consumer — it already dropped us USER).
        add("LDI r15 900")
        add("LDI r14 1")
        add("ST r15 r14")
    if n == GH13_N_A:
        add(f"LDI r17 {n}")             # a7 = syscall number
        add(f"LDI r10 {buf_word}")      # a0 = buffer word index
        add("LDI r11 4")                # a1 = byte count
        add(f"LDI r14 {GH13_PATTERN}")  # stage the payload in A's own box
        add(f"LDI r15 {buf_word}")
        add("ST r15 r14")
        add("SYSCALL r12")
    elif n == GH13_N_B:
        add(f"LDI r17 {n}")
        add(f"LDI r10 {buf_word}")      # a0 = B's read-out scratch
        add("LDI r11 4")                # a1 = max bytes
        add("SYSCALL r12")
        # verify the pattern IN B's own box (mailbox payload vs constant)
        add(f"LDI r15 {buf_word}")
        add("LD r2 r15")
        add(f"LDI r3 {GH13_PATTERN}")
        add("CMP r2 r3")
        add(f"JZ :__task_{n}_ok")
        add(f"LDI r15 {GH13_VERDICT_B}")
        add("LDI r14 69")               # 'E' on mismatch
        add("ST r15 r14")
        add(f"JMP :__task_{n}_exit")
        add(f":__task_{n}_ok")
        add(f"LDI r15 {GH13_VERDICT_B}")
        add(f"LDI r14 {GH13_VERIFIED}")  # 'V'
        add("ST r15 r14")
        add(f":__task_{n}_exit")
    else:
        add(f"LDI r17 {n}")
        add(f"LDI r10 {buf_word}")      # a0 = C's read-out scratch
        add("LDI r11 0")                # a1 = unused
        add("SYSCALL r12")
    # exit status write: 0xFEED0000 | n
    add(f"LDI r15 {exit_word}")
    add("LDI r14 65261")                # 0xFEED
    add("LDI r4 16")
    add("SHL r14 r4")
    add("ADD r14 r17")
    add("ST r15 r14")
    # Return through the privilege boundary. The destination dispatch leg
    # re-arms the latch for the NEXT agent (round-robin A -> B -> C).
    if n == GH13_N_A:
        add(f"LDI r30 {_GH13_DISPATCH_B_PC}")
    elif n == GH13_N_B:
        add(f"LDI r30 {_GH13_DISPATCH_C_PC}")
    else:
        add(f"LDI r30 {_GH13_RET_PC}")
    add("KJMP r30")
    return a


def _gh13_kernel_program_text(
    status_word: int = 950,
    fault_leg: bool = False,
) -> str:
    """GH-13 glyph assembly: resident multi-agent scheduler kernel.

    Sequence (all in-image; the host only runs the image and reads memory):
      1. SUPER-mode prologue zeroes the receipt words, programs BOX0/BOX1/
         BOX2 (three per-agent arenas), arms KSYS_PC/KFAULT_PC with
         loader-seeded packed pixel PCs (two-pass bake), then enters agent
         A: MODE_LATCH = 1, KJMP :__task_6 (KJMP is the only latch consumer).
      2. Agent A (writer, BOX0) stages the pattern, SYSCALL 6: the kernel
         copies it to A's uart AND the agent->agent mailbox. A exits and
         KJMPs to :__dispatch_b (latch 0 -> lands SUPER).
      3. :__dispatch_b re-arms the latch and KJMPs into agent B (reader/
         verifier, BOX1). B reads ONLY via the mailbox syscall (SYS 7),
         verifies the pattern in its own box ('V'), exits, KJMPs to
         :__dispatch_c.
      4. :__dispatch_c re-arms the latch and KJMPs into agent C
         (coordinator, BOX2). C SYSCALLs 8: the kernel packs A's payload +
         B's verdict into C's read-out. C exits, KJMPs to :__kdone.
      5. :__kdone writes status = 0xCAFE0000 | 8 and HALTs.
    """
    a: List[str] = []
    add = a.append
    add(":__entry")
    add("JMP :__kmain")
    add(":__kmain")
    # ---- zero the receipt words (all kernel-owned) ----
    for w in (GH13_TURN_WORD, GH13_EXIT_A, GH13_UART_A_WORD, GH13_UART_A_WORD2,
              GH13_UART_A_LEN, GH13_EXIT_B, GH13_READOUT_B, GH13_VERDICT_B,
              GH13_EXIT_C, GH13_UART_C_WORD0, GH13_UART_C_WORD1,
              GH13_MAILBOX_WORD, GH13_BADSYS_WORD, GH13_FAULT_WORD):
        add(f"LDI r15 {w}")
        add("LDI r14 0")
        add("ST r15 r14")
    # ---- program BOX0 (A), BOX1 (B), BOX2 (C) ----
    add(f"LDI r15 {BOX0_LO_WORD}")
    add(f"LDI r14 {GH13_BOX0_LO_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX0_LO_WORD + 1}")
    add(f"LDI r14 {GH13_BOX0_HI_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX1_LO_WORD}")
    add(f"LDI r14 {GH13_BOX1_LO_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX1_LO_WORD + 1}")
    add(f"LDI r14 {GH13_BOX1_HI_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {GH13_BOX2_LO_WORD}")
    add(f"LDI r14 {GH13_BOX2_LO_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {GH13_BOX2_LO_WORD + 1}")
    add(f"LDI r14 {GH13_BOX2_HI_BYTE}")
    add("ST r15 r14")
    # ---- arm the vectors (loader-seeded packed pixel PCs) ----
    add(f"LDI r15 {KSYS_PC_WORD}")
    add(f"LDI r14 {_GH13_KSYS_PC}")
    add("ST r15 r14")
    add(f"LDI r15 {KFAULT_PC_WORD}")
    add(f"LDI r14 {_GH13_FAULT_PC if fault_leg else _GH13_RET_PC}")
    add("ST r15 r14")
    # ---- enter agent A: latch USER, KJMP ----
    add(f"LDI r15 {MODE_LATCH_WORD}")
    add("LDI r14 1")           # MODE_USER: KJMP one-shots the latch + clears
    add("ST r15 r14")
    add(f"LDI r30 {_GH13_TASK_A_PC}")
    add("KJMP r30")
    add("JMP :__kdone")
    # ---- syscall dispatcher (SUPER): reached via KSYS_PC ----
    # ONE resident selector services ALL agents' syscall numbers (6/7/8);
    # a single-number handler would strand the other agents' SYSCALLs.
    add(":__ksys")
    add(f"LDI r15 {SYS_N_ADDR >> 2}")
    add("LD r5 r15")                    # r5 = SYS_N
    add(f"LDI r4 {GH13_N_A}")
    add("CMP r5 r4")
    add(f"JZ :__ksys_{GH13_N_A}")
    add(f"LDI r4 {GH13_N_B}")
    add("CMP r5 r4")
    add(f"JZ :__ksys_{GH13_N_B}")
    add(f"LDI r4 {GH13_N_C}")
    add("CMP r5 r4")
    add(f"JZ :__ksys_{GH13_N_C}")
    # unknown syscall: record 'E' (69) and SYSRET dry
    add(f"LDI r15 {GH13_BADSYS_WORD}")
    add("LDI r14 69")                   # 'E'
    add("ST r15 r14")
    add("SYSRET")
    for line in _gh13_ksys_slice(GH13_N_A):
        add(line)
    for line in _gh13_ksys_slice(GH13_N_B):
        add(line)
    for line in _gh13_ksys_slice(GH13_N_C):
        add(line)
    # ---- round-robin dispatch: switch A -> B, then B -> C ----
    add(":__dispatch_b")
    add(f"LDI r15 {MODE_LATCH_WORD}")
    add("LDI r14 1")
    add("ST r15 r14")
    add(f"LDI r30 {_GH13_TASK_B_PC}")
    add("KJMP r30")
    add("JMP :__kdone")
    add(":__dispatch_c")
    add(f"LDI r15 {MODE_LATCH_WORD}")
    add("LDI r14 1")
    add("ST r15 r14")
    add(f"LDI r30 {_GH13_TASK_C_PC}")
    add("KJMP r30")
    add("JMP :__kdone")
    # ---- fault handler (SUPER): records the violation, halts via kernel ----
    add(":__kfault")
    if fault_leg:
        add(f"LDI r15 {GH13_FAULT_WORD}")
        add(f"LDI r14 {GH13_FAULT_SEEN}")
        add("ST r15 r14")
    # fall through to the done/halt tail
    # ---- done: status = 0xCAFE0000 | 8, halt ----
    add(":__kdone")
    add(f"LDI r9 {GH13_STATUS_ID}")
    add("LDI r3 51966")        # 0xCAFE
    add("LDI r4 16")
    add("SHL r3 r4")
    add("ADD r3 r9")
    add(f"LDI r15 {status_word}")
    add("ST r15 r3")
    add("HALT")
    # ---- agent A (USER, BOX0): writer ----
    for line in _gh13_agent_task(GH13_N_A, GH13_TASK_A_BUF, GH13_EXIT_A,
                                 fault_leg):
        add(line)
    # ---- agent B (USER, BOX1): reader/verifier ----
    # B's a0/verify pointer is its READ-OUT word (720, inside BOX1) — the
    # kernel SYS 7 delivers the mailbox there; 727 is unused scratch.
    for line in _gh13_agent_task(GH13_N_B, GH13_READOUT_B, GH13_EXIT_B):
        add(line)
    # ---- agent C (USER, BOX2): coordinator, final boundary ----
    for line in _gh13_agent_task(GH13_N_C, GH13_TASK_C_BUF, GH13_EXIT_C):
        add(line)
    return "\n".join(a) + "\n"


# ────────────────────────────── GH-14 ────────────────────────────────────────
# Agent work protocol: N=2 host-driven agent SESSIONS share ONE baked image.
# Each session owns exactly one box (memory-ownership); all cross-session
# data flows through kernel-mediated channels: the mailbox (754) and the
# GH-8 pixel FS. Tickets + receipts live in the PIXEL-aliased FS window
# [1024,1280) — ordinary RAM words die with the runner, pixels persist.

GH14_TURN_WORD = 705        # kernel dispatch marker (SUPER-only)
GH14_EXIT_A, GH14_EXIT_B = 703, 723
GH14_UART_A = 710           # agent A receipt (its work result)
GH14_UART_B = 730           # agent B receipt (combined result)
GH14_READOUT_B = 720        # kernel delivers ticket payload / mailbox here
GH14_READOUT_B1 = 721
GH14_VERDICT = 724          # ticket-claim verdict: 'V' claimed / 'E' miss
GH14_MAILBOX = 754          # agent->agent mailbox (kernel-mediated)
GH14_SELECT_WORD = 760      # host seeds: 1 = agent A session, 2 = agent B
GH14_BADSYS_WORD = 758
GH14_FAULT_WORD = 759
GH14_BOX0_LO_BYTE = 4 * 700
GH14_BOX0_HI_BYTE = 4 * 717
GH14_BOX1_LO_BYTE = 4 * 718
GH14_BOX1_HI_BYTE = 4 * 735
GH14_N_A, GH14_N_B = 1, 2   # syscall numbers == agent ids (claim family)
GH14_N_SIGNAL = 3           # SYS 3: signal — a0 -> mailbox + FS receipt slot
GH14_N_RECV = 4             # SYS 4: recv — mailbox -> readout (720)
GH14_STATUS_ID = 2          # final status 0xCAFE0000 | 2

# ticket files (GH-8 pixel FS): slot0 = TICK_A, slot1 = TICK_B
GH14_SLOT0 = 1024           # slot0: name,start,len,in_use
GH14_SLOT1 = 1028           # slot1
GH14_SLOT_WORDS = 4
GH14_DATA_WORD = 1044       # slot0 data (2 words), slot1 data at 1046
GH14_RSLT_A = 1048          # pixel FS: agent A's result (SYS 3, slot 0)
GH14_RSLT_B = 1049          # pixel FS: agent B's combined result (slot 1)
GH14_TICK_A = (ord("T") | (ord("A") << 8) | (ord("C") << 16) | (ord("K") << 24))
GH14_TICK_B = (ord("T") | (ord("B") << 8) | (ord("C") << 16) | (ord("K") << 24))
GH14_A_PAYLOAD = 0x00000007
GH14_B_PAYLOAD = 0x00000005
GH14_FAULT_SEEN = 0xFA171

_GH14_KSYS_PC = _GH14_FAULT_PC = _GH14_RET_PC = 0
_GH14_TASK_A_PC = _GH14_TASK_B_PC = 0


def _gh14_seed_fs(a: List[str]) -> None:
    """SUPER prologue: seed the ticket FS once. GH-8 idempotence: if
    slot0.in_use is already 1 the tickets live in the persisted pixels
    (second boot / session B) — leave everything untouched so receipts
    survive across sessions."""
    a.append(f"LDI r15 {GH14_SLOT0 + 3}")
    a.append("LD r9 r15")                 # r9 = slot0.in_use
    a.append("LDI r8 1")
    a.append("CMP r9 r8")
    a.append("JZ :__fs_seeded")
    # claim slot0: name=TICK_A, start=DATA, len=2, in_use=1
    a.append(f"LDI r15 {GH14_SLOT0}")
    a.append(f"LDI r14 {GH14_TICK_A}")
    a.append("ST r15 r14")
    a.append(f"LDI r15 {GH14_SLOT0 + 1}")
    a.append(f"LDI r14 {GH14_DATA_WORD}")
    a.append("ST r15 r14")
    a.append(f"LDI r15 {GH14_SLOT0 + 2}")
    a.append("LDI r14 2")
    a.append("ST r15 r14")
    a.append(f"LDI r15 {GH14_SLOT0 + 3}")
    a.append("LDI r14 1")
    a.append("ST r15 r14")
    # slot0 data = A payload
    a.append(f"LDI r15 {GH14_DATA_WORD}")
    a.append(f"LDI r14 {GH14_A_PAYLOAD}")
    a.append("ST r15 r14")
    a.append(f"LDI r15 {GH14_DATA_WORD + 1}")
    a.append("LDI r14 0")
    a.append("ST r15 r14")
    # claim slot1: name=TICK_B, start=DATA+2, len=2, in_use=1; data = B payload
    a.append(f"LDI r15 {GH14_SLOT1}")
    a.append(f"LDI r14 {GH14_TICK_B}")
    a.append("ST r15 r14")
    a.append(f"LDI r15 {GH14_SLOT1 + 1}")
    a.append(f"LDI r14 {GH14_DATA_WORD + 2}")
    a.append("ST r15 r14")
    a.append(f"LDI r15 {GH14_SLOT1 + 2}")
    a.append("LDI r14 2")
    a.append("ST r15 r14")
    a.append(f"LDI r15 {GH14_SLOT1 + 3}")
    a.append("LDI r14 1")
    a.append("ST r15 r14")
    a.append(f"LDI r15 {GH14_DATA_WORD + 2}")
    a.append(f"LDI r14 {GH14_B_PAYLOAD}")
    a.append("ST r15 r14")
    a.append(f"LDI r15 {GH14_DATA_WORD + 3}")
    a.append("LDI r14 0")
    a.append("ST r15 r14")
    a.append(":__fs_seeded")


def _gh14_ksys_slice(n: int) -> List[str]:
    """One SUPER-mode dispatcher slice.

    SYS n (claim, n = agent id 1|2): a0 = expected ticket name. Scans the
      FSTAB for (name match AND in_use); on hit copies the 2 data words
      into the caller's readout (720/721), verdict 724 = 'V', result = 4.
      Miss: verdict 'E', result 0. SYS_A0 carries the result for SYSRET.
    SYS 3 (signal): a0 = value, a1 = slot index (0|1) -> mailbox 754 = a0
      AND pixel-FS receipt word (1048 + a1) = a0; result = 4.
    SYS 4 (recv): mailbox 754 -> readout 720 (kernel-mediated, the ONLY
      cross-agent channel); result = 4.
    """
    a: List[str] = []
    add = a.append
    add(f":__ksys_{n}")
    add(f"LDI r15 {SYS_A0_ADDR >> 2}")
    add("LD r6 r15")                      # r6 = a0
    add(f"LDI r15 {SYS_A1_ADDR >> 2}")
    add("LD r7 r15")                      # r7 = a1
    if n in (GH14_N_A, GH14_N_B):
        # scan slot0 then slot1 for name==a0
        add(f"LDI r15 {GH14_SLOT0}")
        add("LD r8 r15")
        add("CMP r8 r6")
        add(f"JZ :__claim_{n}_s0")
        add(f"LDI r15 {GH14_SLOT1}")
        add("LD r8 r15")
        add("CMP r8 r6")
        add(f"JZ :__claim_{n}_s1")
        # miss: verdict 'E', result 0
        add(f"LDI r15 {GH14_VERDICT}")
        add("LDI r14 69")
        add("ST r15 r14")
        add(f"LDI r15 {SYS_A0_WORD}")
        add("LDI r14 0")
        add("ST r15 r14")
        add("SYSRET")
        add(f":__claim_{n}_s0")
        # hit slot0: copy data words -> readout
        add(f"LDI r15 {GH14_DATA_WORD}")
        add("LD r8 r15")
        add(f"LDI r15 {GH14_READOUT_B}")
        add("ST r15 r8")
        add(f"LDI r15 {GH14_DATA_WORD + 1}")
        add("LD r8 r15")
        add(f"LDI r15 {GH14_READOUT_B1}")
        add("ST r15 r8")
        add(f"JMP :__claim_{n}_ok")
        add(f":__claim_{n}_s1")
        add(f"LDI r15 {GH14_DATA_WORD + 2}")
        add("LD r8 r15")
        add(f"LDI r15 {GH14_READOUT_B}")
        add("ST r15 r8")
        add(f"LDI r15 {GH14_DATA_WORD + 3}")
        add("LD r8 r15")
        add(f"LDI r15 {GH14_READOUT_B1}")
        add("ST r15 r8")
        add(f":__claim_{n}_ok")
        add(f"LDI r15 {GH14_VERDICT}")
        add(f"LDI r14 {ord('V')}")
        add("ST r15 r14")
        add(f"LDI r15 {SYS_A0_WORD}")
        add("LDI r14 4")
        add("ST r15 r14")
        add("SYSRET")
    elif n == GH14_N_SIGNAL:
        # mailbox = a0; receipt FS word (1048 + a1) = a0
        add(f"LDI r15 {GH14_MAILBOX}")
        add("ST r15 r6")
        add(f"LDI r15 {GH14_RSLT_A}")
        add("ADD r15 r7")
        add("ST r15 r6")
        add(f"LDI r15 {SYS_A0_WORD}")
        add("LDI r14 4")
        add("ST r15 r14")
        add("SYSRET")
    else:  # GH14_N_RECV: pixel-FS receipt (1048) -> readout. The mailbox
        # is a RAM word and dies with the session; A's receipt persists in
        # the pixel FS — the ONLY cross-session channel.
        add(f"LDI r15 {GH14_RSLT_A}")
        add("LD r8 r15")
        add(f"LDI r15 {GH14_READOUT_B}")
        add("ST r15 r8")
        add(f"LDI r15 {SYS_A0_WORD}")
        add("LDI r14 4")
        add("ST r15 r14")
        add("SYSRET")
    return a


def _gh14_agent_task(agent: int, exit_word: int, uart_word: int,
                     payload_reg_work: str, dispatch_pc: int,
                     fault_leg: bool = False) -> List[str]:
    """One USER-mode agent slice confined to its own box.

    claim (SYS <agent>) -> readout 720 = ticket payload -> work
    (payload_reg_work emits the multiply) -> uart = result ->
    signal/recv -> exit write -> KJMP to the next dispatch leg.
    """
    a: List[str] = []
    add = a.append
    add(f":__task_{agent}")
    if fault_leg:
        add("LDI r15 900")                # out-of-box violation (E-K1)
        add("LDI r14 1")
        add("ST r15 r14")
    # claim the ticket: r17 = agent id, a0 = ticket name
    add(f"LDI r17 {agent}")
    if agent == GH14_N_A:
        add(f"LDI r10 {GH14_TICK_A}")
    else:
        add(f"LDI r10 {GH14_TICK_B}")
    add("LDI r11 0")
    add("SYSCALL r12")
    # work: r2 = readout payload; payload_reg_work transforms r2 in place
    add(f"LDI r15 {GH14_READOUT_B}")
    add("LD r2 r15")
    for line in payload_reg_work:
        add(line)
    # receipt: uart = result
    add(f"LDI r15 {uart_word}")
    add("ST r15 r2")
    return a


def _gh14_agent_tail(kind: str, agent: int, exit_word: int,
                     next_pc: int) -> List[str]:
    """Post-work tail: signal/recv, exit write, KJMP out of the box."""
    a: List[str] = []
    add = a.append
    if kind == "signal":
        add(f"LDI r17 {GH14_N_SIGNAL}")
        add("LDI r10 0")                  # SYSRET returns in r10 — clear first
        add("ADD r10 r2")                 # a0 = the work result (r2)
        add(f"LDI r11 {0 if agent == GH14_N_A else 1}")
        add("SYSCALL r12")
    else:  # recv: mailbox -> 720, combine, signal slot 1
        add(f"LDI r17 {GH14_N_RECV}")
        add("LDI r10 0")
        add("LDI r11 0")
        add("SYSCALL r12")
        add(f"LDI r15 {GH14_READOUT_B}")
        add("LD r3 r15")                  # r3 = A's result
        add("ADD r2 r3")                  # combined = own + A's
        add(f"LDI r15 {GH14_UART_B}")
        add("ST r15 r2")
        add(f"LDI r17 {GH14_N_SIGNAL}")
        add("LDI r10 0")
        add("ADD r10 r2")
        add("LDI r11 1")
        add("SYSCALL r12")
    # exit status: 0xFEED0000 | agent (r17 was clobbered by the tail's
    # syscall number — restore the agent id explicitly)
    add(f"LDI r17 {agent}")
    add(f"LDI r15 {exit_word}")
    add("LDI r14 65261")                  # 0xFEED
    add("LDI r4 16")
    add("SHL r14 r4")
    add("ADD r14 r17")
    add("ST r15 r14")
    add(f"LDI r30 {next_pc if next_pc else _GH14_RET_PC}")
    add("KJMP r30")
    return a


def _gh14_kernel_program_text(
    status_word: int = 950,
    fault_leg: bool = False,
) -> str:
    """GH-14 glyph assembly: resident agent-protocol kernel.

    Sequence (all in-image; the host runs sessions through drive()):
      1. SUPER prologue: zero receipt words, program BOX0/BOX1, seed the
         ticket FS (idempotent), arm KSYS_PC/KFAULT_PC, read the host's
         agent selector (760) and KJMP into that agent's slice.
      2. Agent A (BOX0): claim TICK_A -> payload 7 -> result = 7*2 = 14 ->
         uart 710 -> SYS 3 signal (mailbox + FS 1048) -> exit 0xFEED0001
         -> KJMP :__kdone.
      3. Agent B (BOX1, next session on the same image): claim TICK_B ->
         payload 5 -> own = 5*3 = 15 -> SYS 4 recv (mailbox = 14) ->
         combined = 15 + 14 = 29 -> uart 730 -> SYS 3 signal slot 1
         (FS 1049) -> exit 0xFEED0002 -> KJMP :__kdone.
      4. :__kdone: status = 0xCAFE0000 | 2, HALT.
    """
    a: List[str] = []
    add = a.append
    add(":__entry")
    add("JMP :__kmain")
    add(":__kmain")
    for w in (GH14_TURN_WORD, GH14_EXIT_A, GH14_UART_A, GH14_EXIT_B,
              GH14_UART_B, GH14_READOUT_B, GH14_READOUT_B1, GH14_VERDICT,
              GH14_MAILBOX, GH14_BADSYS_WORD, GH14_FAULT_WORD):
        add(f"LDI r15 {w}")
        add("LDI r14 0")
        add("ST r15 r14")
    # program BOX0 (A) and BOX1 (B)
    add(f"LDI r15 {BOX0_LO_WORD}")
    add(f"LDI r14 {GH14_BOX0_LO_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX0_LO_WORD + 1}")
    add(f"LDI r14 {GH14_BOX0_HI_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX1_LO_WORD}")
    add(f"LDI r14 {GH14_BOX1_LO_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX1_LO_WORD + 1}")
    add(f"LDI r14 {GH14_BOX1_HI_BYTE}")
    add("ST r15 r14")
    # seed tickets (idempotent across sessions)
    _gh14_seed_fs(a)
    # arm the vectors
    add(f"LDI r15 {KSYS_PC_WORD}")
    add(f"LDI r14 {_GH14_KSYS_PC}")
    add("ST r15 r14")
    add(f"LDI r15 {KFAULT_PC_WORD}")
    if fault_leg:
        add(f"LDI r14 {_GH14_FAULT_PC}")
    else:
        add(f"LDI r14 {_GH14_RET_PC}")
    add("ST r15 r14")
    # host agent selection: 760 = 1 -> A, 2 -> B (default A when 0)
    add(f"LDI r15 {GH14_SELECT_WORD}")
    add("LD r5 r15")
    add("LDI r4 2")
    add("CMP r5 r4")
    add(f"JZ :__sel_b")
    # select A
    add(f"LDI r15 {MODE_LATCH_WORD}")
    add("LDI r14 1")
    add("ST r15 r14")
    add(f"LDI r30 {_GH14_TASK_A_PC}")
    add("KJMP r30")
    add("JMP :__kdone")
    add(":__sel_b")
    add(f"LDI r15 {MODE_LATCH_WORD}")
    add("LDI r14 1")
    add("ST r15 r14")
    add(f"LDI r30 {_GH14_TASK_B_PC}")
    add("KJMP r30")
    add("JMP :__kdone")
    # syscall dispatcher
    add(":__ksys")
    add(f"LDI r15 {SYS_N_ADDR >> 2}")
    add("LD r5 r15")
    add(f"LDI r4 {GH14_N_A}")
    add("CMP r5 r4")
    add(f"JZ :__ksys_{GH14_N_A}")
    add(f"LDI r4 {GH14_N_B}")
    add("CMP r5 r4")
    add(f"JZ :__ksys_{GH14_N_B}")
    add(f"LDI r4 {GH14_N_SIGNAL}")
    add("CMP r5 r4")
    add(f"JZ :__ksys_{GH14_N_SIGNAL}")
    add(f"LDI r4 {GH14_N_RECV}")
    add("CMP r5 r4")
    add(f"JZ :__ksys_{GH14_N_RECV}")
    add(f"LDI r15 {GH14_BADSYS_WORD}")
    add("LDI r14 69")
    add("ST r15 r14")
    add("SYSRET")
    for n in (GH14_N_A, GH14_N_B, GH14_N_SIGNAL, GH14_N_RECV):
        for line in _gh14_ksys_slice(n):
            add(line)
    # fault handler
    add(":__kfault")
    if fault_leg:
        add(f"LDI r15 {GH14_FAULT_WORD}")
        add(f"LDI r14 {GH14_FAULT_SEEN}")
        add("ST r15 r14")
    # done tail
    add(":__kdone")
    add(f"LDI r9 {GH14_STATUS_ID}")
    add("LDI r3 51966")                   # 0xCAFE
    add("LDI r4 16")
    add("SHL r3 r4")
    add("ADD r3 r9")
    add(f"LDI r15 {status_word}")
    add("ST r15 r3")
    add("HALT")
    # agent A (USER, BOX0): claim -> *2 -> uart -> signal slot 0
    for line in _gh14_agent_task(GH14_N_A, GH14_EXIT_A, GH14_UART_A,
                                 ["LDI r4 1", "SHL r2 r4"], 0, fault_leg):
        add(line)
    for line in _gh14_agent_tail("signal", GH14_N_A, GH14_EXIT_A, 0):
        add(line)
    # agent B (USER, BOX1): claim -> *3 -> recv -> combine -> signal slot 1
    for line in _gh14_agent_task(GH14_N_B, GH14_EXIT_B, GH14_UART_B,
                                 ["XOR r5 r5", "ADD r5 r2", "LDI r4 1",
                                  "SHL r2 r4", "ADD r2 r5"], 0):
        add(line)
    for line in _gh14_agent_tail("recv", GH14_N_B, GH14_EXIT_B, 0):
        add(line)
    return chr(10).join(a) + chr(10)


def agent_protocol_kernel_image(
    atlas: Any,
    status_word: int = 950,
    fault_leg: bool = False,
    cols_instrs: int = 8,
    min_rows: int = 80,   # GH-8b: pixel-FS window needs px rows 64..80
    out_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """GH-14: bake ONE image with the resident agent-protocol kernel.
    Two passes: pass 1 fixes the packed pixel PCs (:__ksys/:__kfault/
    :__kdone/:__sel_b/:__task_1/:__task_2), pass 2 bakes the final image."""
    txt1 = _gh14_kernel_program_text(status_word, fault_leg)
    _, coords1 = assemble_glyph_to_pixels(txt1, cols_instrs=cols_instrs, min_rows=min_rows)

    def packed(label: str) -> int:
        col, row = coords1[label]
        return (col & 0xFFFF) | ((row & 0xFFFF) << 16)

    global _GH14_KSYS_PC, _GH14_FAULT_PC, _GH14_RET_PC
    global _GH14_TASK_A_PC, _GH14_TASK_B_PC
    _GH14_KSYS_PC = packed(":__ksys")
    _GH14_FAULT_PC = packed(":__kfault")
    _GH14_RET_PC = packed(":__kdone")
    _GH14_TASK_A_PC = packed(":__task_1")
    _GH14_TASK_B_PC = packed(":__task_2")
    # the KJMP destinations that live OUTSIDE the agent slices: A returns
    # to :__kdone, B returns to :__kdone (host re-opens for each session).
    try:
        # Pass 2 re-materializes with the real packed PCs (globals set above).
        txt2 = _gh14_kernel_program_text(status_word, fault_leg)
        return bake_image(
            txt2,
            atlas=None,
            cols_instrs=cols_instrs,
            min_rows=min_rows,
            out_path=out_path,
        )
    finally:
        _GH14_KSYS_PC = _GH14_FAULT_PC = _GH14_RET_PC = 0
        _GH14_TASK_A_PC = _GH14_TASK_B_PC = 0


def multiagent_kernel_image(
    atlas: Any,
    status_word: int = 950,
    fault_leg: bool = False,
    cols_instrs: int = 8,
    min_rows: int = 16,
    out_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """GH-13: bake ONE image with the resident three-agent scheduler kernel.
    Two passes: pass 1 fixes the packed pixel PCs (:__ksys/:__kfault/
    :__dispatch_b/:__dispatch_c/:__task_6/:__task_7/:__task_8), pass 2 bakes
    the final image. The atlas is accepted for GH-2 continuity but the
    scheduler needs no atlas tiles."""
    txt1 = _gh13_kernel_program_text(status_word, fault_leg)
    _, coords1 = assemble_glyph_to_pixels(txt1, cols_instrs=cols_instrs, min_rows=min_rows)

    def packed(label: str) -> int:
        col, row = coords1[label]
        # engine packing: low 16 bits = column, high 16 = row
        return (col & 0xFFFF) | ((row & 0xFFFF) << 16)

    global _GH13_KSYS_PC, _GH13_FAULT_PC, _GH13_RET_PC
    global _GH13_DISPATCH_B_PC, _GH13_DISPATCH_C_PC
    global _GH13_TASK_A_PC, _GH13_TASK_B_PC, _GH13_TASK_C_PC
    _GH13_KSYS_PC = packed(":__ksys")        # the selector, not one agent's slice
    _GH13_FAULT_PC = packed(":__kfault")     # bound BEFORE pass 2 or the
    _GH13_RET_PC = packed(":__kdone")        # fault leg vectors to pixel (0,0)
    _GH13_DISPATCH_B_PC = packed(":__dispatch_b")
    _GH13_DISPATCH_C_PC = packed(":__dispatch_c")
    _GH13_TASK_A_PC = packed(":__task_6")
    _GH13_TASK_B_PC = packed(":__task_7")
    _GH13_TASK_C_PC = packed(":__task_8")
    try:
        # Pass 2 re-materializes the program with the real packed PCs baked
        # in (assemble is deterministic, so the label coords are identical).
        return bake_image(
            _gh13_kernel_program_text(status_word, fault_leg),
            atlas=None,
            cols_instrs=cols_instrs,
            min_rows=min_rows,
            out_path=out_path,
        )
    finally:
        _GH13_KSYS_PC = _GH13_FAULT_PC = _GH13_RET_PC = 0
        _GH13_DISPATCH_B_PC = _GH13_DISPATCH_C_PC = 0
        _GH13_TASK_A_PC = _GH13_TASK_B_PC = _GH13_TASK_C_PC = 0


# --- GH-16 preemptive scheduling ABI ----------------------------------------
GH16_UART_A_WORD = 710
GH16_UART_A_LEN  = 712
GH16_EXIT_A      = 703
GH16_UART_B_WORD = 720
GH16_UART_B_LEN  = 722
GH16_EXIT_B      = 723
GH16_FAULT_WORD  = 731
GH16_TICKS_COUNT = 732
GH16_TURN_WORD   = 705

GH16_VERIFY_WORD = 704
GH16_FLAG_WORD   = 706
GH16_SAVED_PC_WORD = 708
GH16_VERIFY_OK   = 0xFEEDCAFE

GH16_BOX0_LO_BYTE = 4 * 700
GH16_BOX0_HI_BYTE = 4 * 717
GH16_BOX1_LO_BYTE = 4 * 718
GH16_BOX1_HI_BYTE = 4 * 735

GH16_N_B = 7
GH16_EXIT_OK_B = 0xFEED0000 | GH16_N_B
GH16_FAULT_SEEN = 0xFA171
GH16_KERNEL_OK = 0xCAFE0016
GH16_PACK_B = 0x42424242

KTICK_PC_WORD     = KTICK_PC_ADDR >> 2
TIMER_COUNT_WORD  = TIMER_COUNT_ADDR >> 2
TIMER_RELOAD_WORD = TIMER_RELOAD_ADDR >> 2
TICK_PC_WORD      = TICK_PC_ADDR >> 2

_GH16_KTICK_PC = 0
_GH16_FAULT_PC = 0
_GH16_RET_PC = 0
_GH16_TASK_A_PC = 0
_GH16_TASK_B_PC = 0
_GH16_RESUME_A_PC = 0


def _preempt_kernel_program_text(
    status_word: int = 950,
    fault_leg: bool = False,
    timer_quantum: int = 20,
    context_save_leg: bool = False,
) -> str:
    a: List[str] = []
    add = a.append
    add(":__entry")
    add("JMP :__kmain")
    add(":__kmain")
    # ---- clear words ----
    for w in (GH16_UART_A_WORD, GH16_UART_A_LEN,
              GH16_UART_B_WORD, GH16_UART_B_LEN,
              GH16_FAULT_WORD, GH16_TICKS_COUNT,
              GH16_TURN_WORD, GH16_EXIT_A, GH16_EXIT_B,
              GH16_VERIFY_WORD, GH16_FLAG_WORD, GH16_SAVED_PC_WORD):
        add(f"LDI r15 {w}")
        add("LDI r14 0")
        add("ST r15 r14")
    # ---- program BOX0 and BOX1 ----
    add(f"LDI r15 {BOX0_LO_WORD}")
    add(f"LDI r14 {GH16_BOX0_LO_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX0_LO_WORD + 1}")
    add(f"LDI r14 {GH16_BOX0_HI_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX1_LO_WORD}")
    add(f"LDI r14 {GH16_BOX1_LO_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX1_LO_WORD + 1}")
    add(f"LDI r14 {GH16_BOX1_HI_BYTE}")
    add("ST r15 r14")
    # ---- arm fault vector ----
    add(f"LDI r15 {KFAULT_PC_WORD}")
    add(f"LDI r14 {_GH16_FAULT_PC if fault_leg else _GH16_RET_PC}")
    add("ST r15 r14")
    # ---- arm timer & tick handler ----
    add(f"LDI r15 {KTICK_PC_WORD}")
    add(f"LDI r14 {_GH16_KTICK_PC}")
    add("ST r15 r14")
    add(f"LDI r15 {TIMER_COUNT_WORD}")
    add(f"LDI r14 {timer_quantum}")
    add("ST r15 r14")
    add(f"LDI r15 {TIMER_RELOAD_WORD}")
    add(f"LDI r14 {timer_quantum}")
    add("ST r15 r14")
    # ---- enter task A in USER mode ----
    add(f"LDI r15 {MODE_LATCH_WORD}")
    add("LDI r14 1")
    add("ST r15 r14")
    add(f"LDI r30 {_GH16_TASK_A_PC}")
    add("KJMP r30")
    add("JMP :__kdone")

    # ---- tick handler (SUPER): reached when timer counts down to 0 ----
    add(":__ktick")
    # increment tick count
    add(f"LDI r15 {GH16_TICKS_COUNT}")
    add("LD r14 r15")
    add("LDI r13 1")
    add("ADD r14 r13")
    add("ST r15 r14")

    if context_save_leg:
        # Check if Task B is already done
        add(f"LDI r15 {GH16_FLAG_WORD}")
        add("LD r14 r15")
        add("XOR r13 r13")
        add("CMP r14 r13")
        add("JZ :__save_and_switch_b")
        # Flag != 0: Task B already finished, Task A was running and got ticked.
        # Just resume Task A at TICK_PC.
        add(f"LDI r15 {TICK_PC_WORD}")
        add("LD r30 r15")
        add(f"LDI r15 {MODE_LATCH_WORD}")
        add("LDI r14 1")
        add("ST r15 r14")
        add("KJMP r30")
        add("JMP :__kdone")

        add(":__save_and_switch_b")
        # Save Task A registers r1, r2 and TICK_PC
        add("LDI r15 700")
        add("ST r15 r1")
        add("LDI r15 701")
        add("ST r15 r2")
        add(f"LDI r15 {TICK_PC_WORD}")
        add("LD r14 r15")
        add(f"LDI r15 {GH16_SAVED_PC_WORD}")
        add("ST r15 r14")
        # set turn = 1
        add(f"LDI r15 {GH16_TURN_WORD}")
        add("LDI r14 1")
        add("ST r15 r14")
        # enter task B in USER mode
        add(f"LDI r15 {MODE_LATCH_WORD}")
        add("LDI r14 1")
        add("ST r15 r14")
        add(f"LDI r30 {_GH16_TASK_B_PC}")
        add("KJMP r30")
        add("JMP :__kdone")

        # switch back to task A
        add(":__resume_task_a")
        add("LDI r15 700")
        add("LD r1 r15")
        add("LDI r15 701")
        add("LD r2 r15")
        add(f"LDI r15 {GH16_SAVED_PC_WORD}")
        add("LD r30 r15")
        add(f"LDI r15 {GH16_TURN_WORD}")
        add("LDI r14 0")
        add("ST r15 r14")
        # arm latch and resume task A in USER mode!
        add(f"LDI r15 {MODE_LATCH_WORD}")
        add("LDI r14 1")
        add("ST r15 r14")
        add("KJMP r30")
        add("JMP :__kdone")
    else:
        # check turn word
        add(f"LDI r15 {GH16_TURN_WORD}")
        add("LD r14 r15")
        add("XOR r13 r13")
        add("CMP r14 r13")
        add("JZ :__switch_to_b")
        add("JMP :__kdone")

        add(":__switch_to_b")
        add(f"LDI r15 {GH16_TURN_WORD}")
        add("LDI r14 1")
        add("ST r15 r14")
        add(f"LDI r15 {MODE_LATCH_WORD}")
        add("LDI r14 1")
        add("ST r15 r14")
        add(f"LDI r30 {_GH16_TASK_B_PC}")
        add("KJMP r30")
        add("JMP :__kdone")

    # ---- fault handler (SUPER) ----
    add(":__kfault")
    add(f"LDI r15 {GH16_FAULT_WORD}")
    add(f"LDI r14 {GH16_FAULT_SEEN}")
    add("ST r15 r14")
    add("JMP :__kdone")

    # ---- done: status = 0xCAFE0016, HALT ----
    add(":__kdone")
    add("LDI r3 51966")
    add("LDI r4 16")
    add("SHL r3 r4")
    add("LDI r9 22")
    add("ADD r3 r9")
    add(f"LDI r15 {status_word}")
    add("ST r15 r3")
    add("HALT")

    # ---- task A in BOX0 ----
    add(":__task_a")
    if fault_leg:
        # Out-of-box write to timer MMIO word (word 8208)
        add(f"LDI r15 {TIMER_COUNT_WORD}")
        add("LDI r14 0")
        add("ST r15 r14")
        add("HALT")
    elif context_save_leg:
        # Context-save leg: initialize r1 and r2
        add("LDI r1 4386")       # 0x1122
        add("LDI r4 16")
        add("SHL r1 r4")
        add("LDI r5 13124")      # 0x3344
        add("OR r1 r5")          # r1 = 0x11223344

        add("LDI r2 21862")      # 0x5566
        add("SHL r2 r4")
        add("LDI r5 30600")      # 0x7788
        add("OR r2 r5")          # r2 = 0x55667788

        add(":__task_a_wait_b")
        add(f"LDI r15 {GH16_FLAG_WORD}")
        add("LD r14 r15")
        add("XOR r13 r13")
        add("CMP r14 r13")
        add("JZ :__task_a_wait_b")

        # Task B ran! Verify r1 and r2
        add("LDI r4 16")
        add("LDI r6 4386")
        add("SHL r6 r4")
        add("LDI r5 13124")
        add("OR r6 r5")
        add("CMP r1 r6")
        add("JZ :__r1_ok")
        add("JMP :__kdone")

        add(":__r1_ok")
        add("LDI r7 21862")
        add("SHL r7 r4")
        add("LDI r5 30600")
        add("OR r7 r5")
        add("CMP r2 r7")
        add("JZ :__r2_ok")
        add("JMP :__kdone")

        add(":__r2_ok")
        add("LDI r14 65261")     # 0xFEED
        add("SHL r14 r4")
        add("LDI r5 51966")      # 0xCAFE
        add("OR r14 r5")
        add(f"LDI r15 {GH16_VERIFY_WORD}")
        add("ST r15 r14")        # write 0xFEEDCAFE

        add(f"LDI r15 {GH16_EXIT_A}")
        add("LDI r14 65261")
        add("SHL r14 r4")
        add("ST r15 r14")
        add(f"LDI r30 {_GH16_RET_PC}")
        add("KJMP r30")
    else:
        add("XOR r1 r1")
        add("LDI r2 1")
        add(":__task_a_spin")
        add("ADD r1 r2")
        add("JMP :__task_a_spin")

    # ---- task B in BOX1 ----
    add(":__task_b")
    if context_save_leg:
        # Clobber r1 and r2 with garbage
        add("LDI r1 57005")      # 0xDEAD
        add("LDI r4 16")
        add("SHL r1 r4")
        add("LDI r5 48879")      # 0xBEEF
        add("OR r1 r5")          # r1 = 0xDEADBEEF

        add("LDI r2 47789")      # 0xBAAD
        add("SHL r2 r4")
        add("LDI r5 61453")      # 0xF00D
        add("OR r2 r5")          # r2 = 0xBAADF00D

        add(f"LDI r15 {GH16_UART_B_WORD}")
        add(f"LDI r14 {GH16_PACK_B}")
        add("ST r15 r14")
        add(f"LDI r15 {GH16_UART_B_LEN}")
        add("LDI r14 4")
        add("ST r15 r14")
        add(f"LDI r15 {GH16_EXIT_B}")
        add(f"LDI r14 {GH16_EXIT_OK_B}")
        add("ST r15 r14")

        # Set flag word = 1
        add(f"LDI r15 {GH16_FLAG_WORD}")
        add("LDI r14 1")
        add("ST r15 r14")

        # Yield back to resume Task A
        add(f"LDI r30 {_GH16_RESUME_A_PC}")
        add("KJMP r30")
    else:
        add(f"LDI r15 {GH16_UART_B_WORD}")
        add(f"LDI r14 {GH16_PACK_B}")
        add("ST r15 r14")
        add(f"LDI r15 {GH16_UART_B_LEN}")
        add("LDI r14 4")
        add("ST r15 r14")
        add(f"LDI r15 {GH16_EXIT_B}")
        add(f"LDI r14 {GH16_EXIT_OK_B}")
        add("ST r15 r14")
        add(f"LDI r30 {_GH16_RET_PC}")
        add("KJMP r30")

    return "\n".join(a) + "\n"


def preemptive_kernel_image(
    atlas: Any,
    status_word: int = 950,
    fault_leg: bool = False,
    timer_quantum: int = 20,
    context_save_leg: bool = False,
    cols_instrs: int = 8,
    min_rows: int = 16,
    out_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """GH-16: bake ONE image with the resident preemptive scheduler kernel.
    Two passes: pass 1 fixes the packed pixel PCs (:__ktick/:__kfault/:__kdone/
    :__task_a/:__task_b), pass 2 bakes the final image."""
    txt1 = _preempt_kernel_program_text(status_word, fault_leg, timer_quantum, context_save_leg)
    _, coords1 = assemble_glyph_to_pixels(txt1, cols_instrs=cols_instrs, min_rows=min_rows)

    def packed(label: str) -> int:
        col, row = coords1[label]
        return (col & 0xFFFF) | ((row & 0xFFFF) << 16)

    global _GH16_KTICK_PC, _GH16_FAULT_PC, _GH16_RET_PC, _GH16_TASK_A_PC, _GH16_TASK_B_PC, _GH16_RESUME_A_PC
    _GH16_KTICK_PC = packed(":__ktick")
    _GH16_FAULT_PC = packed(":__kfault")
    _GH16_RET_PC = packed(":__kdone")
    _GH16_TASK_A_PC = packed(":__task_a")
    _GH16_TASK_B_PC = packed(":__task_b")
    _GH16_RESUME_A_PC = packed(":__resume_task_a") if context_save_leg else 0
    try:
        return bake_image(
            _preempt_kernel_program_text(status_word, fault_leg, timer_quantum, context_save_leg),
            atlas=None,
            cols_instrs=cols_instrs,
            min_rows=min_rows,
            out_path=out_path,
        )
    finally:
        _GH16_KTICK_PC = _GH16_FAULT_PC = _GH16_RET_PC = _GH16_TASK_A_PC = _GH16_TASK_B_PC = _GH16_RESUME_A_PC = 0


# --- GH-17 spatial paging ABI ----------------------------------------------
GH17_KERNEL_OK     = 0xCAFE0017
GH17_FAULT_WORD    = 731
GH17_FAULT_VERDICT = 0xFA017
GH17_VERIFY_WORD   = 704
GH17_VERIFY_OK     = 0xFEED0017

_GH17_FAULT_PC = 0
_GH17_RET_PC = 0
_GH17_KTICK_PC = 0
_GH17_TASK_A_PC = 0
_GH17_TASK_B_PC = 0
_GH17_RESUME_A_PC = 0


def _paged_kernel_program_text(
    mode: str = "flat64k",
    status_word: int = 950,
    timer_quantum: int = 35,
) -> str:
    a: List[str] = []
    add = a.append
    add(":__entry")
    add("JMP :__kmain")
    add(":__kmain")

    # Clear status and receipt words
    add(f"LDI r15 {status_word}")
    add("LDI r14 0")
    add("ST r15 r14")
    add(f"LDI r15 {GH17_FAULT_WORD}")
    add("LDI r14 0")
    add("ST r15 r14")
    add(f"LDI r15 {GH17_VERIFY_WORD}")
    add("LDI r14 0")
    add("ST r15 r14")

    if mode == "flat64k":
        # Identity-map low pages 0..3 (words 0..1023)
        for p in range(4):
            val = PTE_V | PTE_W | PTE_U | (p << 8)
            add(f"LDI r15 {PAGE_TABLE_BASE_WORD + p}")
            add(f"LDI r14 {val}")
            add("ST r15 r14")
        # Page 6 (page table window itself, 1536..1791)
        add(f"LDI r15 {PAGE_TABLE_BASE_WORD + 6}")
        add(f"LDI r14 {PTE_V | PTE_W | (6 << 8)}")
        add("ST r15 r14")

        # Map high virtual pages across 64K space:
        # Page 16 (vaddr 4096) -> pfn 8
        add(f"LDI r15 {PAGE_TABLE_BASE_WORD + 16}")
        add(f"LDI r14 {PTE_V | PTE_W | PTE_U | (8 << 8)}")
        add("ST r15 r14")
        # Page 64 (vaddr 16384) -> pfn 9
        add(f"LDI r15 {PAGE_TABLE_BASE_WORD + 64}")
        add(f"LDI r14 {PTE_V | PTE_W | PTE_U | (9 << 8)}")
        add("ST r15 r14")
        # Page 128 (vaddr 32768) -> pfn 10
        add(f"LDI r15 {PAGE_TABLE_BASE_WORD + 128}")
        add(f"LDI r14 {PTE_V | PTE_W | PTE_U | (10 << 8)}")
        add("ST r15 r14")
        # Page 255 (vaddr 65280) -> pfn 11
        add(f"LDI r15 {PAGE_TABLE_BASE_WORD + 255}")
        add(f"LDI r14 {PTE_V | PTE_W | PTE_U | (11 << 8)}")
        add("ST r15 r14")

        # Arm page table base register in MMIO
        add(f"LDI r15 {PAGE_TABLE_WORD}")
        add(f"LDI r14 {PAGE_TABLE_BASE_WORD}")
        add("ST r15 r14")

        # Enter USER mode task
        add(f"LDI r15 {MODE_LATCH_WORD}")
        add("LDI r14 1")
        add("ST r15 r14")
        add(f"LDI r30 {_GH17_TASK_A_PC}")
        add("KJMP r30")
        add("JMP :__kdone")

        # Task body
        add(":__task_a")
        add("LDI r4 16")
        # Word 4096 = 0x1000CAFE (4096 << 16 | 51966)
        add("LDI r15 4096")
        add("LDI r14 4096")
        add("SHL r14 r4")
        add("LDI r5 51966")
        add("OR r14 r5")
        add("ST r15 r14")

        # Word 16384 = 0x4000BEEF (16384 << 16 | 48879)
        add("LDI r15 16384")
        add("LDI r14 16384")
        add("SHL r14 r4")
        add("LDI r5 48879")
        add("OR r14 r5")
        add("ST r15 r14")

        # Word 32768 = 0x8000FEED (32768 << 16 | 65261)
        add("LDI r15 32768")
        add("LDI r14 32768")
        add("SHL r14 r4")
        add("LDI r5 65261")
        add("OR r14 r5")
        add("ST r15 r14")

        # Word 65280 = 0xFF001234 (65280 << 16 | 4660)
        add("LDI r15 65280")
        add("LDI r14 65280")
        add("SHL r14 r4")
        add("LDI r5 4660")
        add("OR r14 r5")
        add("ST r15 r14")

        # Read back and verify 4096
        add("LDI r15 4096")
        add("LD r1 r15")
        add("LDI r14 4096")
        add("SHL r14 r4")
        add("LDI r5 51966")
        add("OR r14 r5")
        add("CMP r1 r14")
        add("JZ :__chk_16384")
        add("JMP :__kdone")

        add(":__chk_16384")
        add("LDI r15 16384")
        add("LD r1 r15")
        add("LDI r14 16384")
        add("SHL r14 r4")
        add("LDI r5 48879")
        add("OR r14 r5")
        add("CMP r1 r14")
        add("JZ :__chk_32768")
        add("JMP :__kdone")

        add(":__chk_32768")
        add("LDI r15 32768")
        add("LD r1 r15")
        add("LDI r14 32768")
        add("SHL r14 r4")
        add("LDI r5 65261")
        add("OR r14 r5")
        add("CMP r1 r14")
        add("JZ :__chk_65280")
        add("JMP :__kdone")

        add(":__chk_65280")
        add("LDI r15 65280")
        add("LD r1 r15")
        add("LDI r14 65280")
        add("SHL r14 r4")
        add("LDI r5 4660")
        add("OR r14 r5")
        add("CMP r1 r14")
        add("JZ :__flat_all_ok")
        add("JMP :__kdone")

        add(":__flat_all_ok")
        add(f"LDI r30 {_GH17_RET_PC}")
        add("KJMP r30")

    elif mode == "unmapped_fault":
        for p in range(4):
            val = PTE_V | PTE_W | PTE_U | (p << 8)
            add(f"LDI r15 {PAGE_TABLE_BASE_WORD + p}")
            add(f"LDI r14 {val}")
            add("ST r15 r14")
        # Arm KFAULT_PC
        add(f"LDI r15 {KFAULT_PC_WORD}")
        add(f"LDI r14 {_GH17_FAULT_PC}")
        add("ST r15 r14")
        # Arm PAGE_TABLE_ADDR
        add(f"LDI r15 {PAGE_TABLE_WORD}")
        add(f"LDI r14 {PAGE_TABLE_BASE_WORD}")
        add("ST r15 r14")
        # Enter USER mode
        add(f"LDI r15 {MODE_LATCH_WORD}")
        add("LDI r14 1")
        add("ST r15 r14")
        add(f"LDI r30 {_GH17_TASK_A_PC}")
        add("KJMP r30")
        add("JMP :__kdone")

        # Task: attempts store to unmapped word 25600 (page 100)
        add(":__task_a")
        add("LDI r15 25600")
        add("LDI r14 42")
        add("ST r15 r14")
        add("HALT")

        # Fault handler (SUPER mode): verifies fault_addr and writes verdict
        add(":__kfault")
        add(f"LDI r15 {FAULT_ADDR_ADDR >> 2}")
        add("LD r1 r15")
        # Expected byte address: 25600 * 4 = 102400 = (1 << 16) | 36864
        add("LDI r4 16")
        add("LDI r2 1")
        add("SHL r2 r4")
        add("LDI r5 36864")
        add("OR r2 r5")
        add("CMP r1 r2")
        add("JZ :__fault_matched")
        add("JMP :__kdone")

        add(":__fault_matched")
        # GH17_FAULT_VERDICT = 0xFA017 = (15 << 16) | 40983
        add(f"LDI r15 {GH17_FAULT_WORD}")
        add("LDI r14 15")
        add("SHL r14 r4")
        add("LDI r5 40983")
        add("OR r14 r5")
        add("ST r15 r14")
        add("JMP :__kdone")

    elif mode == "context_switch":
        # Low pages 0..3 identity mapped
        for p in range(4):
            val = PTE_V | PTE_W | PTE_U | (p << 8)
            add(f"LDI r15 {PAGE_TABLE_BASE_WORD + p}")
            add(f"LDI r14 {val}")
            add("ST r15 r14")
        # Page 8 (word 2048) -> pfn 8
        add(f"LDI r15 {PAGE_TABLE_BASE_WORD + 8}")
        add(f"LDI r14 {PTE_V | PTE_W | PTE_U | (8 << 8)}")
        add("ST r15 r14")
        # Page 16 (word 4096) -> pfn 9
        add(f"LDI r15 {PAGE_TABLE_BASE_WORD + 16}")
        add(f"LDI r14 {PTE_V | PTE_W | PTE_U | (9 << 8)}")
        add("ST r15 r14")

        # Arm PAGE_TABLE_ADDR
        add(f"LDI r15 {PAGE_TABLE_WORD}")
        add(f"LDI r14 {PAGE_TABLE_BASE_WORD}")
        add("ST r15 r14")

        # Clear flag and ticks words
        add(f"LDI r15 {GH16_FLAG_WORD}")
        add("LDI r14 0")
        add("ST r15 r14")
        add(f"LDI r15 {GH16_TICKS_COUNT}")
        add("LDI r14 0")
        add("ST r15 r14")

        # Arm timer downcounter and tick handler
        add(f"LDI r15 {KTICK_PC_WORD}")
        add(f"LDI r14 {_GH17_KTICK_PC}")
        add("ST r15 r14")
        add(f"LDI r15 {TIMER_COUNT_WORD}")
        add(f"LDI r14 {timer_quantum}")
        add("ST r15 r14")
        add(f"LDI r15 {TIMER_RELOAD_WORD}")
        add(f"LDI r14 {timer_quantum}")
        add("ST r15 r14")

        # Enter Task A in USER mode
        add(f"LDI r15 {MODE_LATCH_WORD}")
        add("LDI r14 1")
        add("ST r15 r14")
        add(f"LDI r30 {_GH17_TASK_A_PC}")
        add("KJMP r30")
        add("JMP :__kdone")

        # Tick handler (SUPER)
        add(":__ktick")
        add(f"LDI r15 {GH16_TICKS_COUNT}")
        add("LD r14 r15")
        add("LDI r13 1")
        add("ADD r14 r13")
        add("ST r15 r14")
        # Check flag (if Task B finished, resume Task A)
        add(f"LDI r15 {GH16_FLAG_WORD}")
        add("LD r14 r15")
        add("XOR r13 r13")
        add("CMP r14 r13")
        add("JZ :__save_and_switch_b")
        # Task B already finished; just resume Task A at TICK_PC
        add(f"LDI r15 {TICK_PC_WORD}")
        add("LD r30 r15")
        add(f"LDI r15 {MODE_LATCH_WORD}")
        add("LDI r14 1")
        add("ST r15 r14")
        add("KJMP r30")
        add("JMP :__kdone")

        add(":__save_and_switch_b")
        # Save Task A registers r1, r2 and TICK_PC
        add("LDI r15 700")
        add("ST r15 r1")
        add("LDI r15 701")
        add("ST r15 r2")
        add(f"LDI r15 {TICK_PC_WORD}")
        add("LD r14 r15")
        add(f"LDI r15 {GH16_SAVED_PC_WORD}")
        add("ST r15 r14")
        # Enter Task B in USER mode
        add(f"LDI r15 {MODE_LATCH_WORD}")
        add("LDI r14 1")
        add("ST r15 r14")
        add(f"LDI r30 {_GH17_TASK_B_PC}")
        add("KJMP r30")
        add("JMP :__kdone")

        # Resume Task A
        add(":__resume_task_a")
        add("LDI r15 700")
        add("LD r1 r15")
        add("LDI r15 701")
        add("LD r2 r15")
        add(f"LDI r15 {GH16_SAVED_PC_WORD}")
        add("LD r30 r15")
        add(f"LDI r15 {MODE_LATCH_WORD}")
        add("LDI r14 1")
        add("ST r15 r14")
        add("KJMP r30")
        add("JMP :__kdone")

        # Task A in USER mode: writes 0xAAAA1111 to word 2048 (> 1024)
        add(":__task_a")
        add("LDI r4 16")
        add("LDI r15 2048")
        add("LDI r14 43690")
        add("SHL r14 r4")
        add("LDI r5 4369")
        add("OR r14 r5")
        add("ST r15 r14")

        # Wait for Task B to complete
        add(":__task_a_wait_b")
        add(f"LDI r15 {GH16_FLAG_WORD}")
        add("LD r14 r15")
        add("XOR r13 r13")
        add("CMP r14 r13")
        add("JZ :__task_a_wait_b")

        # Task B finished! Verify word 2048 still holds 0xAAAA1111!
        add("LDI r15 2048")
        add("LD r1 r15")
        add("LDI r4 16")
        add("LDI r14 43690")
        add("SHL r14 r4")
        add("LDI r5 4369")
        add("OR r14 r5")
        add("CMP r1 r14")
        add("JZ :__task_a_verified")
        add("JMP :__kdone")

        add(":__task_a_verified")
        # GH17_VERIFY_OK = 0xFEED0017 = (65261 << 16) | 23
        add(f"LDI r15 {GH17_VERIFY_WORD}")
        add("LDI r14 65261")
        add("SHL r14 r4")
        add("LDI r5 23")
        add("OR r14 r5")
        add("ST r15 r14")
        add(f"LDI r30 {_GH17_RET_PC}")
        add("KJMP r30")

        # Task B in USER mode: writes 0xBBBB2222 to word 4096 (> 1024)
        add(":__task_b")
        add("LDI r4 16")
        add("LDI r15 4096")
        add("LDI r14 48059")
        add("SHL r14 r4")
        add("LDI r5 8738")
        add("OR r14 r5")
        add("ST r15 r14")
        # Clobber r1 and r2
        add("LDI r1 9999")
        add("LDI r2 8888")
        # Set flag word = 1
        add(f"LDI r15 {GH16_FLAG_WORD}")
        add("LDI r14 1")
        add("ST r15 r14")
        # Yield to resume Task A
        add(f"LDI r30 {_GH17_RESUME_A_PC}")
        add("KJMP r30")

    elif mode == "pixel_parity":
        # Low pages 0..3 identity mapped
        for p in range(4):
            val = PTE_V | PTE_W | PTE_U | (p << 8)
            add(f"LDI r15 {PAGE_TABLE_BASE_WORD + p}")
            add(f"LDI r14 {val}")
            add("ST r15 r14")
        # Map page 10 (word 2560) with PTE_PIX: pfn = 10
        # PTE = PTE_V | PTE_W | PTE_U | PTE_PIX | (10 << 8) = 1 | 2 | 4 | 8 | 2560 = 2575
        add(f"LDI r15 {PAGE_TABLE_BASE_WORD + 10}")
        add("LDI r14 2575")
        add("ST r15 r14")
        # Arm PAGE_TABLE_ADDR
        add(f"LDI r15 {PAGE_TABLE_WORD}")
        add(f"LDI r14 {PAGE_TABLE_BASE_WORD}")
        add("ST r15 r14")
        # Store 0x123456 (1193046) into word 2560
        add("LDI r15 2560")
        add("LDI r14 18")
        add("LDI r4 16")
        add("SHL r14 r4")
        add("LDI r5 13398")
        add("OR r14 r5")
        add("ST r15 r14")
        # Load back into r10
        add("LD r10 r15")
        add("HALT")

    # Common completion handler
    add(":__kdone")
    add("LDI r3 51966") # 0xCAFE
    add("LDI r4 16")
    add("SHL r3 r4")
    add("LDI r9 23")    # 0x0017
    add("ADD r3 r9")
    add(f"LDI r15 {status_word}")
    add("ST r15 r3")
    add("HALT")

    return "\n".join(a) + "\n"


def paged_kernel_image(
    atlas: Any,
    mode: str = "flat64k",
    status_word: int = 950,
    timer_quantum: int = 35,
    cols_instrs: int = 8,
    min_rows: int = 100,
    out_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """GH-17: bake ONE image with the resident spatial paging kernel.
    Two passes: pass 1 fixes the packed pixel PCs, pass 2 bakes the final image."""
    txt1 = _paged_kernel_program_text(mode, status_word, timer_quantum)
    _, coords1 = assemble_glyph_to_pixels(txt1, cols_instrs=cols_instrs, min_rows=min_rows)

    def packed(label: str) -> int:
        if label not in coords1:
            return 0
        col, row = coords1[label]
        return (col & 0xFFFF) | ((row & 0xFFFF) << 16)

    global _GH17_FAULT_PC, _GH17_RET_PC, _GH17_KTICK_PC, _GH17_TASK_A_PC, _GH17_TASK_B_PC, _GH17_RESUME_A_PC
    _GH17_FAULT_PC = packed(":__kfault")
    _GH17_RET_PC = packed(":__kdone")
    _GH17_KTICK_PC = packed(":__ktick")
    _GH17_TASK_A_PC = packed(":__task_a")
    _GH17_TASK_B_PC = packed(":__task_b")
    _GH17_RESUME_A_PC = packed(":__resume_task_a")
    try:
        return bake_image(
            _paged_kernel_program_text(mode, status_word, timer_quantum),
            atlas=None,
            cols_instrs=cols_instrs,
            min_rows=min_rows,
            out_path=out_path,
        )
    finally:
        _GH17_FAULT_PC = _GH17_RET_PC = _GH17_KTICK_PC = _GH17_TASK_A_PC = _GH17_TASK_B_PC = _GH17_RESUME_A_PC = 0


# --- GH-18 Syscall ABI v2: syscalls ARE atlas tiles --------------------------
# The kernel syscall table is an IN-IMAGE word table of packed pixel PCs
# into tile code in the image's tile rect. Dispatch is an INDEXED table
# lookup (LD TABLE_BASE+sys_n -> KJMP), replacing GH-7's linear CMP/JZ
# selector chain. The invariant baseline is the post-rewrite image.
#
# Word map (all inside the GH-17 page-table window [1536,1792) so the
# table + tile rect are identity-mapped by the kernel's own page-table
# setup: SUPER dispatch never consults the page walker on the hot path):
#   1568..1583  syscall table: 16 words, sys_n 6..21 -> entry = n - 6
#               (word 0/1 pre-seeded with the legacy fixed slices 6/7)
#   1600..1695  tile rect: 24 instructions x 4 px (the GH-9 window size —
#               an ingest-admitted tile's _pixel_words payload fits as-is)
# Every legacy architectural word stays untouched: boxes [700,768), FS
# window [1024,1280), status 950, ABI word 952 (NEW: 0x00020018),
# mailbox [800,896), page-table base word 1536.
GH18_TABLE_WORD = 1568      # syscall table base (16 entries, sys_n 6..21)
GH18_NSLOTS = 16            # power-of-two: mask = 15 -> unsigned bounds check
GH18_TABLE_MASK = GH18_NSLOTS - 1
# RESERVED pixel window for the table's IMAGE-PIXEL surface. The admit-mode
# kernel arms a PTE_PIX page map for vpn 6, so the dispatcher's table LD
# walks word w (in [1568,1584)) to pixel word 5*256 + (w & 0xFF) = pfn-5
# window [1312, 1328). PINNED CONSTANT, mode- and image-size-independent:
# growing program text must NEVER move or overwrite this window (ticket
# gh18-table-reservation, 2026-09-08 — same contract as the fs window's
# 2-px alias; guarded by test_gh18_syscall_table_window_reserved).
GH18_TABLE_PIX_WORD = 1312  # = 5*256 + (GH18_TABLE_WORD & 0xFF)
GH18_TILE_WORD = 1600       # tile rect base (24 instrs x 4 px = 96 words)
GH18_TILE_NPX = GH9_N_INSTRS * 4
GH18_ABI_WORD = 952         # kernel ABI version word
GH18_ABI_VERSION = 0x00020018
GH18_BADSYS_WORD = GH7_BADSYS_WORD      # unknown-syscall marker ('E' = 69)
GH18_EXIT_A = GH7_EXIT_A                # 703
GH18_UART_A = GH7_UART_A_WORD           # 710
GH18_UART_B = GH7_UART_B_WORD           # 720
GH18_EXIT_B = GH7_EXIT_B                # 723
GH18_FAULT_WORD = GH16_FAULT_WORD       # 731 (fault leg; unused by default)
GH18_TICKS_COUNT = GH16_TICKS_COUNT     # 732
GH18_TURN_WORD = GH16_TURN_WORD         # 705
GH18_VERIFY_WORD = GH16_VERIFY_WORD     # 704
GH18_VERIFY_OK = 0xFEED0018
GH18_KERNEL_OK = 0xCAFE0018
GH18_EXIT_OK = 0xFEED0000 | GH7_N_A      # 0xFEED0006
GH18_ARGV_WORD = GH9_ARGV_WORD          # 750: the tile's input word
GH18_RESULT_WORD = GH9_ARGV_RESULT      # 754: the tile's result word
GH18_BOX0_LO_BYTE = GH16_BOX0_LO_BYTE
GH18_BOX0_HI_BYTE = GH16_BOX0_HI_BYTE
GH18_BOX1_LO_BYTE = GH16_BOX1_LO_BYTE
GH18_BOX1_HI_BYTE = GH16_BOX1_HI_BYTE
# BOX2 = the tile-ABI window [736..768): the GH-18 tile's argv word (750)
# and result word (754) live here, and the admit-mode task stores the
# argv from USER mode. The kernel prologue arms BOX2 with this range
# (GH-16's boxes left BOX2 unset, so the argv store E-K1-faulted).
GH18_BOX2_LO_BYTE = 4 * 736
GH18_BOX2_HI_BYTE = 4 * 768
GH18_BOX2_LO_WORD = BOX2_LO_ADDR >> 2
GH18_N_A = GH7_N_A                      # legacy slice 6
GH18_N_B = GH7_N_B                      # legacy slice 7

# Packed pixel PCs + tile destination, bound by syscall_abi_kernel_image()
# before the second (final) assemble pass.
_GH18_DISPATCH_PC = 0
_GH18_KSYS_PC = 0
_GH18_FAULT_PC = 0
_GH18_RET_PC = 0
_GH18_TASK_A_PC = 0
_GH18_TASK_B_PC = 0
_GH18_TICK_PC = 0
_GH18_RESUME_A_PC = 0
_GH18_TILE_PC = 0            # packed pixel PC of the tile rect (tile entry)
GH18_TILE_PC = 0             # public mirror for the host-side admission gate


def _gh18_ksys_slice(n: int) -> List[str]:
    """GH-18 SUPER-mode fixed dispatcher slices for the legacy syscalls
    6 (A) and 7 (B). Unlike the GH-13 slices these read each task's OWN
    staged buffer word (SYS_A0-relative), NOT the GH-13 agent mailbox —
    in the GH-18 kernel word 754 is the tile result slot, so the mailbox
    ABI would alias the admission receipt into B's read-out.

    SYS 6 (task A): copies A's staged payload (mem[SYS_A0] = buffer word
      index 713) into A's uart (710), returns 4.
    SYS 7 (task B): copies B's staged payload (buffer word 727) into B's
      read-out (720), returns 4.
    """
    a: List[str] = []
    add = a.append
    add(f":__ksys_{n}")
    add(f"LDI r15 {SYS_A0_ADDR >> 2}")
    add("LD r6 r15")                    # r6 = a0 = the task's buffer WORD index
    add("LD r7 r6")                     # r7 = the staged payload
    if n == GH18_N_A:
        add(f"LDI r15 {GH18_UART_A}")
    else:
        add(f"LDI r15 {GH18_UART_B}")
    add("ST r15 r7")                    # the task's uart/read-out
    add(f"LDI r15 {SYS_A0_WORD}")
    add("LDI r14 4")
    add("ST r15 r14")                   # result -> SYS_A0 for SYSRET
    add("SYSRET")
    return a


def _gh18_pack_const(v: int) -> List[Tuple[str, str]]:
    """Glyph code lines building v (32-bit) in r14: HI<<16 | LO, LDI-safe."""
    hi, lo = (v >> 16) & 0xFFFF, v & 0xFFFF
    lines = [f"LDI r14 {lo}"]
    if hi:
        lines += [f"LDI r13 {hi}", "LDI r4 16", "SHL r13 r4", "OR r14 r13"]
    return lines


def _gh18_tile_payload_for_test(resume: Optional[int] = None) -> List[int]:
    """Pixel words of the reference admit tile (triple(r1)->r2 under the
    kernel argv ABI), assembled standalone then jump-relocated into the
    tile rect — the exact words syscall_abi_kernel_image bakes into the
    rect for the seeded modes (the test gate RAM-seeds ONLY the table
    word, so the tile code must already live in-image).

    `resume` = packed PC of :__ksys_done (the tile's KJMP-home target).
    Callers inside the bake MUST pass the pass-1 coords of the ACTUAL
    mode (paged_dispatch's ptloop shifts the dispatcher); the default
    resolves the baseline-mode coords, which match every mode whose
    prologue is mode-independent (admit/baseline).

    Receipt (2026-09-08, probes debug_gh18_fix5/6/8): the earlier draft
    (a) computed into r2 WITHOUT zeroing it — task A's boxed canary
    0x0ABC000C rides in r2 through SYSCALL (nothing marshals or zeroes
    it), so the tile stored (canary<<1)+argv instead of 3*argv; and
    (b) ended in HALT — correct only for the standalone oracle harness;
    in-image the tile must KJMP :__ksys_done or the run dies at the tile
    with the status word unwritten. Fixed: XOR r2 r2 before the shift-
    add, KJMP resume home last.
    """
    if resume is None:
        resume = _gh18_dispatch_resume()
    tile_text = (
        ":__entry\n"
        f"LDI r15 {GH18_ARGV_WORD}\n"
        "LD r1 r15\n"
        "XOR r2 r2\n"                    # dirty-register receipt: see above
        "LDI r13 1\n"
        "ADD r2 r1\n"                   # r2 = r1        (seed the accumulator)
        "SHL r2 r13\n"                  # r2 = r1 << 1   (SHL: rd = rd << rs2)
        "ADD r2 r1\n"                   # r2 = 2*r1 + r1 = 3*r1
        f"LDI r15 {GH18_RESULT_WORD}\n"
        "ST r15 r2\n"
        f"LDI r30 {resume}\n"           # :__ksys_done — KJMP home (SUPER)
        "KJMP r30\n"
    )
    from tools.glyph_gpt.autoatlas import ir_pixel_words
    words = ir_pixel_words(tile_text)
    _gh18_relocate_jumps(tile_text, words)
    return words


def _gh18_relocate_jumps(glyph_text: str, words: List[int],
                         mode: str = "baseline") -> None:
    """Relocate JMP/JZ immediates in-place from standalone tile coords to
    absolute tile-rect coords (same contract as autoatlas's GH-9 window
    relocation, target = the GH-18 tile rect).

    mode MUST match the image the tile will be stamped into: the tile
    rect's row shifts per mode (admit 0x1e0007 vs fs_v2 0x220005 —
    receipt 2026-09-09, probe23/probe25), so relocating with baseline
    coords stamps in-tile JZ/JMP targets ~80 cells short of the stamped
    body — the tile's `JZ :skip` landed in task-A's probe cells
    (212..217) and the dispatch looped forever (result word never
    written, exit_a 0, E_ATLAS_INJECT on every fs tile with a branch).
    """
    txt1 = _gh18_kernel_program_text(mode)
    _, coords = assemble_glyph_to_pixels(txt1, cols_instrs=8, min_rows=16)
    tcol, trow = coords[":__g18tile"]
    labels = assemble_glyph_to_pixels(glyph_text, cols_instrs=8)[1]
    cell = 0
    for raw in glyph_text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line or line.startswith(":"):
            continue
        op = line.split()[0]
        if op in ("JMP", "JZ", "CALL"):
            tcol2, trow2 = labels[line.split()[-1]]
            abs_cell = (trow + trow2) * 8 + (tcol + tcol2)
            imm = (abs_cell % 8) | ((abs_cell // 8) << 16)
            base = cell * 4
            words[base + 2] = imm & 0xFFFFFF
            words[base + 3] = (imm >> 24) & 0xFF
        cell += 1


def _gh18_user_task(
    n: int,
    buf_word: int,
    exit_word: int,
    mode: str = "baseline",
) -> List[str]:
    """GH-18 USER task slice. mode variants:
    - baseline/ingest: SYSCALL n (letter payload in a0 buffer, like GH-7).
    - unknown_syscall: SYSCALL 9 (in neither the fixed slices nor table).
    - neg_sysn: r17 = -1 (0xFFFFFFFF) — the marshaled SYS_N must hit the
      unsigned bounds check.
    - huge_sysn: r17 = 0x7FFFFFFF.
    - reentrancy: baseline plus registers preloaded with boxed canaries
      (r2..r4) that the preserve legs verify after SYSRET.
    - admit: baseline (SYS 6/7 letter payload intact) + boxed canaries
      + tile argv word + a SECOND syscall: SYS 8 (the admitted-tile
      number). This is the mode the admit legs run: a table-dispatched
      tile is only exercised when a task actually issues its number.
    - fs_v2: admit's shape with the FS ops instead of SYS 8: after the
      baseline SYS 6 + canaries, task A walks SYS 10/11/12 (GH-20
      Pixel-FS v2) with the SAME probe-then-issue pattern (probe the
      table slot; zero -> skip; non-zero -> stage argv and SYSCALL).
      After each issued FS call the task fetches the verdict into its
      OWN r2: 'LDI r15 754 / LD r2 r15'. This is load-bearing — SYSRET
      restores the PRE-TRAP register file and :__ksys_done zeroes
      SYS_A0 before returning, so a0 arrives as 0 and r10 is clobbered;
      without the fetch the task's r2 at HALT is its boot-time 0, not
      the op verdict (receipt: probe17 trace, 2026-09-09 — admit-mode
      task A issues SYS 6/8 only, no FS number ever reaches :__tbl).
    """
    a: List[str] = []
    add = a.append
    add(f":__task_{n}")
    if mode == "neg_sysn":
        # a7 = 0xFFFFFFFF: 'LDI r17 4294967295' wraps; build HI|LO.
        add("LDI r17 65535")
        add("LDI r13 16")
        add("LDI r4 16")
        add("SHL r17 r13")
        add("LDI r13 65535")
        add("OR r17 r13")
    elif mode == "huge_sysn":
        add("LDI r17 32767")
        add("LDI r13 16")
        add("SHL r17 r13")
    elif mode == "unknown_syscall" and n == GH18_N_A:
        # The leg's whole point: task A issues a number that is in NEITHER
        # the fixed slices (6/7) NOR the table -> unknown-syscall handler.
        # Task B stays baseline so both tasks still complete (leg contract).
        add("LDI r17 9")
        add(f"LDI r10 {buf_word}")
        add("LDI r11 4")
        add("SYSCALL r12")
        add(f"LDI r15 {exit_word}")
        add("LDI r14 65261")
        add("LDI r4 16")
        add("SHL r14 r4")
        add("ADD r14 r17")
        add("ST r15 r14")               # 0xFEED0000 | 9
        add(f"LDI r30 {_GH18_DISPATCH_PC if n == GH18_N_A else _GH18_RET_PC}")
        add("KJMP r30")
        return a
    else:
        add(f"LDI r17 {n}")
    add(f"LDI r10 {buf_word}")          # a0 = buffer word index
    add("LDI r11 4")                    # a1 = byte count
    if mode not in ("neg_sysn", "huge_sysn"):
        letter = _pack_word4("AB"[n - GH18_N_A])
        add(f"LDI r14 {letter}")
        add(f"LDI r15 {buf_word}")
        add("ST r15 r14")
    if mode in ("reentrancy", "admit"):
        # boxed canaries in r2..r4, stored in-box so the host can verify
        # they survived the syscall round trip (leg-4 contract proof).
        # NOTE the words: the previous draft stored the canaries into
        # 713/714/715 — 713 is ALSO task A's a0 payload word, so the
        # canary store CLOBBERED the 'A' payload before SYS 6 ran
        # (receipt: probe debug_gh18_tickfix7.py, UART_A == 0xc0cf10c).
        # The preserve legs read the boxes AFTER the tile leg, so the
        # canaries must live in words that no other leg writes. Also:
        # the canary constants were mis-decimalized (LDI 202174732 =
        # 0x0C0CF10C, not 0x0C0FE00C — LDI is exact, no truncation, but
        # the wrong number was in the source), so the old assertions
        # could never pass. Fixed constants, LDI-safe (<= 24 bits):
        #   0x0ABC000C / 0x0ABD000D / 0x0ABE000E in words 714/715/716.
        # (Receipt: the previous draft LDI'd 180224012/13/14 — those
        # decimalize to 0x0ABE000C/D/E, off by 2 in the high byte for the
        # first two; the preserve leg asserts the hex constants.)
        add("LDI r2 180092940")          # 0x0ABC000C
        add(f"LDI r15 {714}")
        add("ST r15 r2")
        add("LDI r2 180158477")          # 0x0ABD000D
        add(f"LDI r15 {715}")
        add("ST r15 r2")
        add("LDI r2 180224014")          # 0x0ABE000E
        add(f"LDI r15 {716}")
        add("ST r15 r2")
    if mode == "admit" and n == GH18_N_A:
        # Steering contract (oversight session 2026-09-08 14:10): task A
        # runs the baseline SYS 6 FIRST (call-chain intact, UART_A leg
        # stays green), THEN probes the table for a live SYS 8 slot and
        # only issues SYS 8 when the slot is non-zero (baseline runs keep
        # BADSYS clean and the two_syscalls leg green — zero-edit legs).
        #
        # Receipt (2026-09-08 run, UART_A=0/BADSYS=69 at cell trace): the
        # previous draft REPLACED the SYS 6 with the SYS 8 block and
        # staged the tile argv as 18 (the expected RESULT, not the input
        # 6 — the tile then tripled 18). Sequence now:
        #   1. SYS 6 (payload 'A' @713 -> uart 710, canaries pre-staged)
        #   2. probe mem[GH18_TABLE_WORD + 2] (the SYS 8 slot)
        #   3. zero -> skip straight to the exit tail (baseline behavior)
        #   4. non-zero -> stage argv[0] = 6 @750, sys_n=8, a0=750, SYS 8
        #   5. exit tail: 0xFEED0000|6, KJMP home
        add("SYSCALL r12")               # 1. the baseline SYS 6
        # 2. probe the SYS 8 table slot (r30 = slot word; XOR-flag CMP)
        add(f"LDI r15 {GH18_TABLE_WORD + 2}")
        add("LD r30 r15")
        add("XOR r4 r4")                 # flag-safe zero (r0 IS the flag)
        add("CMP r30 r4")
        # 3/4: JZ skips the tile leg when the slot is zero (unseeded);
        # fallthrough = slot live -> stage argv and issue SYS 8.
        add(f"JZ :__task_{n}_tail")
        add(f"LDI r14 {GH18_N_A}")       # argv[0] = 6 -> tile computes 18
        add(f"LDI r15 {GH18_ARGV_WORD}")
        add("ST r15 r14")
        add(f"LDI r17 {GH18_N_A + 2}")   # sys_n = 8 (first table slot)
        add(f"LDI r10 {GH18_ARGV_WORD}") # a0 = tile argv word
        add("LDI r11 4")
        add("SYSCALL r12")
        add(f":__task_{n}_tail")
        add(f"LDI r15 {exit_word}")
        add("LDI r14 65261")             # 0xFEED
        add("LDI r4 16")
        add("SHL r14 r4")
        add(f"LDI r13 {n}")
        add("ADD r14 r13")               # 0xFEED0000 | 6 (the TASK number)
        add("ST r15 r14")
        add(f"LDI r30 {_GH18_DISPATCH_PC if n == GH18_N_A else _GH18_RET_PC}")
        add("KJMP r30")
        return a
    if mode == "fs_v2" and n == GH18_N_A:
        # GH-20 Pixel-FS v2 drive leg (ticket gh20-spec-finalize). Task A:
        #   1. SYS 6 baseline (call-chain intact; UART_A leg stays green)
        #   2. canaries (same boxed constants as admit — leg parity)
        #   3. for sys_n in (10, 11, 12): probe the table slot; zero ->
        #      skip; non-zero -> stage argv, SYSCALL, fetch the verdict
        #      into r2 ('LDI r15 754 / LD r2 r15').
        # The r2 fetch is REQUIRED: SYSRET restores the pre-trap register
        # file and :__ksys_done zeroes SYS_A0 before returning, so a0
        # arrives as 0 — the result word 754 is the only verdict channel.
        # Receipts (2026-09-09): probe17 (admit-mode task A never issues
        # an FS number); probe16 trace (the :__ksys selector itself is
        # correct — JZ :__tbl fires for live table slots).
        add("SYSCALL r12")               # 1. the baseline SYS 6
        # 2. boxed canaries — IDENTICAL constants/stores to admit mode
        # (leg parity: the preserve legs assert these hex words).
        add("LDI r2 180092940")          # 0x0ABC000C
        add(f"LDI r15 {714}")
        add("ST r15 r2")
        add("LDI r2 180158477")          # 0x0ABD000D
        add(f"LDI r15 {715}")
        add("ST r15 r2")
        add("LDI r2 180224014")          # 0x0ABE000E
        add(f"LDI r15 {716}")
        add("ST r15 r2")
        # 3. the three FS ops, probe-then-issue, straight-line unrolled
        # (the ISA has no computed jumps; a loop would need self-modifying
        # code — unrolling is the proven shape from the admit leg).
        for fs_n in (10, 11, 12):
            add(f"LDI r15 {GH18_TABLE_WORD + (fs_n - 6)}")
            add("LD r30 r15")
            add("XOR r4 r4")             # flag-safe zero (r0 IS the flag)
            add("CMP r30 r4")
            add(f"JZ :__task_{n}_fs{fs_n}_skip")
            add("LDI r14 65")            # argv[0] = 'A' = NAME_A (the seed's
                                         # slot-0 name; the FS tiles branch
                                         # on it). Receipt (2026-09-09): the
                                         # first draft staged 713 — the
                                         # ADDRESS of the SYS 6 letter
                                         # payload, not its value — so the
                                         # tiles' branch-on-name saw 713 != 65
                                         # and every op returned errno 'E'
                                         # (69) with a perfect tile in the
                                         # rect (output/debug_gh20_probe53.py).
            add(f"LDI r15 {GH18_ARGV_WORD}")
            add("ST r15 r14")
            add("LDI r14 66")            # argv[1] = 'B' (fs_rename's new
                                         # name — leg 2 asserts slot 0's
                                         # name becomes NAME_B on-die; the
                                         # tile copies mem[751] into word
                                         # 1024). Harmless for append/unlink.
            add(f"LDI r15 {GH18_ARGV_WORD + 1}")
            add("ST r15 r14")
            add(f"LDI r17 {fs_n}")       # sys_n
            add(f"LDI r10 {GH18_ARGV_WORD}")   # a0 = tile argv word
            add("LDI r11 4")
            add("SYSCALL r12")
            add(f"LDI r15 {GH18_RESULT_WORD}") # fetch verdict into r2
            add("LD r2 r15")
            add(f":__task_{n}_fs{fs_n}_skip")
        add(f":__task_{n}_tail")
        add(f"LDI r15 {exit_word}")
        add("LDI r14 65261")             # 0xFEED
        add("LDI r4 16")
        add("SHL r14 r4")
        add(f"LDI r13 {n}")
        add("ADD r14 r13")               # 0xFEED0000 | 6
        add("ST r15 r14")
        add(f"LDI r30 {_GH18_DISPATCH_PC if n == GH18_N_A else _GH18_RET_PC}")
        add("KJMP r30")
        return a
    add("SYSCALL r12")
    add(f"LDI r15 {exit_word}")
    add("LDI r14 65261")                # 0xFEED
    add("LDI r4 16")
    add("SHL r14 r4")
    add("ADD r14 r17")
    add("ST r15 r14")                   # 0xFEED0000 | n
    # privilege boundary back to the kernel
    add(f"LDI r30 {_GH18_DISPATCH_PC if n == GH18_N_A else _GH18_RET_PC}")
    add("KJMP r30")
    return a


# Task modes whose image is driven with an admission-seeded table
# (admit legs + paged identity-map leg). fs_v2 = the GH-20 Pixel-FS v2
# drive image (admit prologue + vpn-4 fs-alias PTE + the FS task leg).
# posix = the GH-21 POSIX shim image (admit prologue + BOX2 arming; the
# C binary replaces task A's text at bake time — see posix_shim.py).
# libc = the GH-23 libc runtime image (identical prologue family; a
# REAL sys_brk tile and a 4-word stdout write window — see
# libc_runtime.py).
GH18_SEEDED_MODES = ("admit", "paged_dispatch", "fs_v2", "posix", "libc")


def _gh18_kernel_program_text(
    mode: str = "baseline",
    status_word: int = 950,
    timer_quantum: int = 35,
) -> str:
    """GH-18 glyph assembly: Syscall ABI v2 kernel.

    Sequence (all in-image; the host only runs the image and reads memory):
      1. SUPER prologue zeroes receipt words, seeds the ABI version word
         (952 = 0x00020018) IN-IMAGE, programs BOX0/BOX1, seeds the
         syscall table (16 zero entries = every table number unknown),
         arms KSYS_PC/KFAULT_PC (+ KTICK/timer for the re-entrancy leg),
         latches USER and KJMPs into task A.
      2. :__ksys dispatcher (SUPER, entered on every SYSCALL):
           read SYS_N; n in {6,7} -> the legacy fixed slices (call-chain
           compatibility, exercised by the regression invariant);
           else -> INDEXED TABLE DISPATCH:
             idx = (SYS_N - 6) & 15         ; power-of-two UNSIGNED mask
             LD  r6, TABLE + idx            ; packed tile PC (0 = unknown)
             JZ  :__ksys_unknown            ; 0 or OOB-negative all land here
             KJMP r6                        ; into the tile rect
         A negative sys_n wraps (& 15) inside the table — it reads a
         seeded-zero entry and vectors to the unknown handler, never an
         OOB read. 0x7FFFFFFF masks to idx 9 — an unseeded entry, ditto.
      3. The tile runs in the tile rect in SUPER (kernel work), reads its
         argv word, computes, stores the result, and KJMPs back to the
         :__ksys_done resume point, which writes SYS_A0 = 0 and SYSRETs
         (the engine restores the task's register file; the tile's
         clobbers never reach USER — the IRContract admission gate on the
         host side holds this for escalated tiles).
      4. :__ksys_unknown: 'E' -> BADSYS, SYS_A0 = 0, SYSRET (clean).
      5. Round-robin dispatch (GH-7 pattern) -> task B -> done tail
         (status 0xCAFE0018, HALT).

    Re-entrancy (mode == 'reentrancy'): the tick handler is armed and the
    engine defers the tick while mode == SUPER (glyph_isa_v2 GH-16 block
    only fires when was_user and mode is still USER after the step). A
    tick can therefore never land between SYSCALL marshal and SYSRET:
    SYS_N/A0/PC are only ever observed at rest (0) in the receipt.
    """
    a: List[str] = []
    add = a.append
    add(":__entry")
    add("JMP :__kmain")
    add(":__kmain")
    # ---- zero the receipt words ----
    for w in (GH18_BADSYS_WORD, GH18_EXIT_A, GH18_EXIT_B,
              GH18_UART_A, GH18_UART_A + 1, 712,
              GH18_UART_B, GH18_UART_B + 1, 722,
              GH18_RESULT_WORD, GH18_FAULT_WORD, GH18_TICKS_COUNT,
              GH18_TURN_WORD, GH18_VERIFY_WORD):
        add(f"LDI r15 {w}")
        add("LDI r14 0")
        add("ST r15 r14")
    # ---- ABI version word IN-IMAGE: mem[952] = 0x00020018 ----
    for line in _gh18_pack_const(GH18_ABI_VERSION):
        add(line)
    add(f"LDI r15 {GH18_ABI_WORD}")
    add("ST r15 r14")
    if mode in ("posix", "libc"):
        # GH-21: bump the ABI version word in-image (0x00020018 -> 0x00020019)
        # — the shim's ABI is a strict superset of GH-18's; the low-byte
        # bump distinguishes shim-capable images from GH-18/20 ones.
        # GH-23: bump again (0x00020019 -> 0x0002001A) for the libc-mode
        # image (a real brk tile + a 4-word write window).
        # (hi=0x0002, lo=LDI-safe: same shape as the _gh18_pack_const
        # block above.)
        lo = 26 if mode == "libc" else 25
        add(f"LDI r14 {lo}")
        add("LDI r13 2")
        add("LDI r4 16")
        add("SHL r13 r4")
        add("OR r14 r13")
        add(f"LDI r15 {GH18_ABI_WORD}")
        add("ST r15 r14")
    # ---- program BOX0/BOX1 ----
    add(f"LDI r15 {BOX0_LO_WORD}")
    add(f"LDI r14 {GH18_BOX0_LO_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX0_LO_WORD + 1}")
    add(f"LDI r14 {GH18_BOX0_HI_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX1_LO_WORD}")
    add(f"LDI r14 {GH18_BOX1_LO_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX1_LO_WORD + 1}")
    add(f"LDI r14 {GH18_BOX1_HI_BYTE}")
    add("ST r15 r14")
    # ---- arm BOX2 = the tile-ABI window [736..768) ----
    # GH-9 defined the argv word (750) and result word (754) INSIDE its
    # box (GH-9's BOX0 was [700..768)), but the GH-18 kernel inherited
    # GH-16's narrower boxes (A [700..717), B [718..735), BOX2 unset).
    # The admit-mode task's USER store of the tile argv into word 750
    # then trips the E-K1 out-of-box check -> fault_addr=3000 (0xBB8 =
    # word 750 * 4) before ANY syscall runs (receipt: probe
    # output/debug_gh18_stepwind.py, fault at task-A cell 171, step 188;
    # gate 2026-09-08 run: fault_addr=3000 on 3 admit legs). Arming
    # BOX2 over the tile-ABI window makes the USER argv/result stores
    # legal without touching BOX0/BOX1 (the tasks' own arenas).
    add(f"LDI r15 {BOX2_LO_WORD}")
    add(f"LDI r14 {GH18_BOX2_LO_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX2_LO_WORD + 1}")
    add(f"LDI r14 {GH18_BOX2_HI_BYTE}")
    add("ST r15 r14")
    # NOTE: this loop must NOT exist at runtime — it would overwrite the
    # host's admission seeds (the table is stamped on the live image AFTER
    # boot; RAM zero-init already gives the all-unknown baseline). The
    # LDI here only fixes the loop-carried register state for pass 1.
    add(f"LDI r13 {GH18_TABLE_WORD}")
    add("LDI r14 0")
    add("LDI r4 16")
    add(":__g18_tblloop")
    add("LDI r15 1")
    add("ADD r13 r15")
    add("SUB r4 r15")
    add("CMP r4 r0")
    add("JZ :__g18_tbldone")
    add("JMP :__g18_tblloop")
    add(":__g18_tbldone")
    # ---- arm the vectors ----
    add(f"LDI r15 {KSYS_PC_WORD}")
    add(f"LDI r14 {_GH18_KSYS_PC}")
    add("ST r15 r14")
    add(f"LDI r15 {KFAULT_PC_WORD}")
    add(f"LDI r14 {_GH18_FAULT_PC if mode == 'fault' else _GH18_RET_PC}")
    add("ST r15 r14")
    if mode == "reentrancy":
        # arm the GH-16 timer at the minimum quantum: if the tick could
        # fire mid-syscall, SYS_N/A0 would be clobbered and the receipt
        # would show torn state. The engine defers ticks while SUPER.
        add(f"LDI r15 {KTICK_PC_WORD}")
        add(f"LDI r14 {_GH18_TICK_PC}")
        add("ST r15 r14")
        add(f"LDI r15 {TIMER_COUNT_WORD}")
        add(f"LDI r14 {timer_quantum}")
        add("ST r15 r14")
        add(f"LDI r15 {TIMER_RELOAD_WORD}")
        add(f"LDI r14 {timer_quantum}")
        add("ST r15 r14")
    if mode in ("admit", "fs_v2", "posix", "libc"):
        # ── PIXEL-MAPPED TABLE PAGE (steering contract, oversight
        # 2026-09-08): the admitted table slot must persist IN THE IMAGE
        # so drive(seeds={}) still dispatches. With pt_base armed and vpn
        # 6 PTE_PIX-mapped, LD [1568..1583] walks to IMAGE PIXELS — the
        # runtime LD the contract demands.
        # Receipts (2026-09-08, probes debug_gh18_tblpath10..25):
        #   (a) Without paging, LD [1570] hits RAM[1570] (< len(memory))
        #       — pixels are unreachable, so RAM seeds die with each
        #       drive()'s fresh CPU and the e2e second drive saw 754=0.
        #   (b) Identity pix (pfn=6) WRAPS: pix_word = pfn*256 +
        #       (vaddr & 0xFF); 1570 & 0xFF = 0x22 = 34 (NOT 2), so
        #       pix_word = 6*256+34 = 1570 -> _addr_to_xy wraps modulo
        #       w*h -> pixel (0,2) = :__entry's JMP immediate — the
        #       entry jump got clobbered and the run fell into the tile
        #       HALTs (probe 8/9 trace: cell 0 -> 220 -> ... loop).
        #   (c) pfn=5: pix_word = 5*256+34 = 1314 (row 41, zero padding
        #       BELOW the program text) — slot i lives at pixel word
        #       1314+i, no wrap, no text collision. The tile rect CODE
        #       executes via KJMP (pixel fetch, no data walk), so vpn 6
        #       being pix-mapped never touches it as data.
        #   (d) PTE_U is REQUIRED on vpn 6: task A's probe LD [1570]
        #       runs in USER mode; V|W|PIX without U faulted at exactly
        #       1570 (probe 21/22/23: fault_addr=0x1888, task-A cells).
        for p in range(8 if mode != "libc" else 11):
            # GH-23: libc maps vpn 0..10 (words 0..2815) — vpn 10 covers
            # the C heap at word 2560 (GH23_HEAP_BASE, moved above the
            # transpiler's PTR_TABLE region 2048..2369; receipt
            # dbg_gh23_cron60..64: heap at 2048 overwrote the indirect-
            # call pointer table and qsort's jalr CALLR'd into kernel
            # text). All identity (pfn = vpn), V|W|U.
            if p == 6:
                val = (5 << 8) | PTE_V | PTE_W | PTE_U | PTE_PIX
            elif p == 4 and mode == "fs_v2":
                # GH-20 (fs_v2): vpn 4 (words [1024,1280), the GH-8b
                # pixel-alias FS window) PIX-mapped IDENTITY (pfn 4).
                # MUST be written in this pre-arming loop: once pt_base
                # is armed, every later ST to the PT region itself walks
                # the live table (vpn 6 -> pfn 5 PIX) and lands in image
                # pixels, never RAM (receipt 2026-09-09, probe22: the
                # previous draft wrote this PTE after the arming ST —
                # on-die vpn4 stayed the identity 0x407, the tile's
                # LD [1027] read zeroed RAM, every FS op skipped cleanly,
                # and admit_syscall diverged with result word 0/status 0;
                # probe20/21: dispatcher+tile cycle forever, r5 stuck 4).
                val = (4 << 8) | PTE_V | PTE_W | PTE_U | PTE_PIX
            else:
                val = (p << 8) | PTE_V | PTE_W | PTE_U
            add(f"LDI r15 {PAGE_TABLE_BASE_WORD + p}")
            add(f"LDI r14 {val}")
            add("ST r15 r14")
        add(f"LDI r15 {PAGE_TABLE_WORD}")
        add(f"LDI r14 {PAGE_TABLE_BASE_WORD}")
        add("ST r15 r14")
        if mode == "libc":
            # GH-23: seed the in-image program break. RAM word 723 is the
            # brk mirror the sys_brk tile reads/advances; the bake-time
            # boot runs SUPER, so this ST lands in RAM directly (an image-
            # pixel stamp CANNOT seed RAM — word 723 of the IMAGE is
            # kernel tick-handler text, receipt dbg_gh23_brk2: mem[723]
            # stayed 0 and the stamp corrupted row-22 text).
            # HEAP_BASE = 2560 (GH23_HEAP_BASE, libc_runtime.py) — above
            # the transpiler's PTR_TABLE region (receipt dbg_gh23_cron64).
            add("LDI r15 723")
            add("LDI r14 2560")
            add("ST r15 r14")
    if mode == "paged_dispatch":
        # Arm the GH-17 page table with an identity map of the low 8
        # pages (vpn 0..7 = words 0..2047). The syscall table (1568..1583)
        # and tile rect (1600..1695) live in vpn 6/7.
        # Receipts (2026-09-08, probes debug_gh18_paged10..18):
        #   Defect A: the loop mapped only 4 pages -> vpn 6/7 walked to
        #     INVALID PTEs once PAGE_TABLE_WORD was armed.
        #   Defect B (the silent killer): the PTE was built as
        #     'LDI r14 0 / SHL r14 r13 / OR 7' = (0 << vpn) | 7 = 7 ->
        #     pfn=0 for EVERY page, so every paged access collapsed into
        #     frame 0 (word 950 -> word 182). Status 0xcafe0018 was stored
        #     into word 182 and mem[950] stayed 0. Identity PTEs are
        #     (vpn << 8) | flags: r14 = vpn, SHL 8, OR 7.
        add("LDI r13 0")
        add(":__g18_ptloop")
        add(f"LDI r15 {PAGE_TABLE_BASE_WORD}")
        add("ADD r15 r13")
        add("ADD r14 r13")            # r14 = vpn (identity frame)
        add("LDI r4 8")
        add("SHL r14 r4")             # r14 = vpn << 8 (pfn field)
        add("LDI r5 7")               # V|W|U
        add("OR r14 r5")
        add("ST r15 r14")
        add("LDI r5 1")
        add("ADD r13 r5")
        add("LDI r4 8")
        add("CMP r13 r4")
        add("JZ :__g18_ptdone")
        add("JMP :__g18_ptloop")
        add(":__g18_ptdone")
        add(f"LDI r15 {PAGE_TABLE_WORD}")
        add(f"LDI r14 {PAGE_TABLE_BASE_WORD}")
        add("ST r15 r14")
    # ---- enter task A in USER mode ----
    add(f"LDI r15 {MODE_LATCH_WORD}")
    add("LDI r14 1")
    add("ST r15 r14")
    add(f"LDI r30 {_GH18_TASK_A_PC}")
    add("KJMP r30")
    add("JMP :__g18done")
    # ---- syscall dispatcher (SUPER): fixed slices + INDEXED TABLE ----
    add(":__ksys")
    add(f"LDI r15 {SYS_N_ADDR >> 2}")
    add("LD r5 r15")                    # r5 = SYS_N
    add(f"LDI r4 {GH18_N_A}")
    add("CMP r5 r4")
    add(f"JZ :__ksys_{GH18_N_A}")
    add(f"LDI r4 {GH18_N_B}")
    add("CMP r5 r4")
    add(f"JZ :__ksys_{GH18_N_B}")
    # -- INDEXED TABLE DISPATCH (the rewrite; replaces the linear chain) --
    add(":__tbl")
    add(f"LDI r4 {GH18_N_A}")
    add("SUB r5 r4")                    # r5 = n - 6 (wraps on negative)
    add(f"LDI r4 {GH18_TABLE_MASK}")
    add("AND r5 r4")                    # unsigned mask: -1 -> 15, huge -> low4
    add(f"LDI r15 {GH18_TABLE_WORD}")
    add("ADD r15 r5")                   # r15 = TABLE + idx
    add("LD r6 r15")                    # r6 = packed tile PC (0 = unknown)
    # Zero-test WITHOUT reading r0: r0 is the CMP FLAG register on this
    # engine (CMP writes equality there, JZ reads it back), so `CMP r6 r0`
    # compares against the stale flag of the selector compare above and
    # misroutes (found via the 2026-09-08 gate run: the unknown_syscall
    # leg silently dispatched instead of trapping). XOR r4 r4 gives a
    # true zero in r4 without touching the flag; CMP r6 r4 then sets the
    # flag to exactly (r6 == 0).
    add("XOR r4 r4")                    # r4 = 0 (flag-safe zero)
    add("CMP r6 r4")                    # flag = (r6 == 0)
    add("JZ :__ksys_unknown")
    add("KJMP r6")                      # into the tile rect (stays SUPER)
    add("JMP :__ksys_unknown")          # unreachable safety net
    # -- tile return/resume: SYSRET --
    # GH-23 defect fix (receipt dbg_gh23_cron52/cron53, 2026-09-09): this
    # resume point used to ZERO SYS_A0 before SYSRET — "so a stale result
    # never leaks into the next trap". But the engine's SYSRET contract is
    # explicitly 'deliver the dispatcher's result (it wrote SYS_A0) into
    # a0 = r10' (glyph_isa_v2.py, SYSRET branch). Zeroing here meant EVERY
    # syscall returned 0: invisible through GH-18/21 (their tasks read
    # verdicts from dedicated RAM words, never through a0) and fatal for
    # GH-23 where malloc's return value IS the heap pointer (arr = 0, the
    # array stores landed at words 0..3 and printf/qsort ran on garbage).
    # SYSCALL re-marshals a7/a0/a1 on every trap, so nothing needs clearing.
    add(":__ksys_done")
    add("SYSRET")
    # -- unknown syscall: 'E', clean SYSRET --
    add(":__ksys_unknown")
    add(f"LDI r15 {GH18_BADSYS_WORD}")
    add("LDI r14 69")                   # 'E'
    add("ST r15 r14")
    add(f"LDI r15 {SYS_A0_WORD}")
    add("LDI r14 0")
    add("ST r15 r14")
    add("SYSRET")
    for line in _gh18_ksys_slice(GH18_N_A):
        add(line)
    for line in _gh18_ksys_slice(GH18_N_B):
        add(line)
    # ---- round-robin dispatch loop ----
    add(":__g18dispatch")
    add(f"LDI r15 {MODE_LATCH_WORD}")
    add("LDI r14 1")
    add("ST r15 r14")
    add(f"LDI r30 {_GH18_TASK_B_PC}")
    add("KJMP r30")
    add("JMP :__g18done")
    # ---- tick handler (GH-16 pattern; re-entrancy leg only) ----
    add(":__g18tick")
    add(f"LDI r15 {GH18_TICKS_COUNT}")
    add("LD r14 r15")
    add("LDI r13 1")
    add("ADD r14 r13")
    add("ST r15 r14")
    add(f"LDI r15 {TICK_PC_WORD}")
    add("LD r30 r15")
    # Resume with JMPR, NOT KJMP: KJMP is the privilege boundary — it
    # drops to SUPER and then re-enters USER on MODE_LATCH==1, which
    # meant the handler's own tail (the MODE_LATCH store to the MMIO
    # word 0x8000, outside every task box) ran AS USER -> E-K1 fault at
    # 0x8000 (receipt: 2026-09-08 gate, fault_addr=0x8000 cell 173).
    # JMPR is the mode-preserving data jump: the handler entered SUPER
    # (the engine set mode on tick delivery) and STAYS SUPER for the
    # whole handler, resume included. Receipt: probe
    # output/debug_gh18_tickfix.py — fault eliminated, kernel reaches
    # 0xCAFE0018.
    add("JMPR r30")
    add("JMP :__g18done")
    # ---- fault handler ----
    add(":__g18fault")
    if mode == "fault":
        add(f"LDI r15 {GH18_FAULT_WORD}")
        add(f"LDI r14 {GH16_FAULT_SEEN}")
        add("ST r15 r14")
    add("JMP :__g18done")
    # ---- done: status = 0xCAFE0018, halt ----
    add(":__g18done")
    # At-rest marshal-word contract (reentrancy leg, 2026-09-08 gate
    # receipt): the engine's E-K2 SYSCALL marshal leaves SYS_N/SYS_A0
    # holding the LAST call's number/forever-after (task B's SYS 7 ->
    # mem[SYS_N]==7, mem[SYS_A0]==4 at HALT) — nothing ever clears them.
    # The done tail runs in SUPER (the task KJMPed home), so clearing the
    # MMIO words here is kernel-legal and leaves SYSRET result delivery
    # untouched (SYSRET reads SYS_A0 BEFORE this tail ever runs).
    add(f"LDI r15 {SYS_N_ADDR >> 2}")
    add("LDI r14 0")
    add("ST r15 r14")
    add(f"LDI r15 {SYS_A0_ADDR >> 2}")
    add("LDI r14 0")
    add("ST r15 r14")
    add("LDI r3 51966")
    add("LDI r4 16")
    add("SHL r3 r4")
    if mode == "fs_v2":
        # GH-20 drive image: status tail id 20 -> 0xCAFE0014 (the gate
        # contract pins KERNEL_OK = 0xCAFE0014 so receipts distinguish
        # the FS image from the GH-18 admit image's 0x18).
        add("LDI r9 20")                # 0x0014
    elif mode == "posix":
        # GH-21 shim image: status tail id 25 -> 0xCAFE0019 (distinguishes
        # the POSIX image from admit 0x18 / fs_v2 0x14).
        add("LDI r9 25")                # 0x0019
    elif mode == "libc":
        # GH-23 libc image: status tail id 26 -> 0xCAFE001A.
        add("LDI r9 26")                # 0x001A
    else:
        add("LDI r9 24")                # 0x0018
    add("ADD r3 r9")
    add(f"LDI r15 {status_word}")
    add("ST r15 r3")
    add("HALT")
    # ---- user tasks ----
    for line in _gh18_user_task(GH18_N_A, 713, GH18_EXIT_A, mode):
        add(line)
    for line in _gh18_user_task(GH18_N_B, 727, GH18_EXIT_B, "baseline"):
        add(line)
    # ---- the tile rect (ingest target; HALTs until patched) ----
    add(":__g18tile")
    for _ in range(GH9_N_INSTRS):
        add("HALT")
    return "\n".join(a) + "\n"


def syscall_abi_kernel_image(
    atlas: Any,
    mode: str = "baseline",
    status_word: int = 950,
    timer_quantum: int = 35,
    cols_instrs: int = 8,
    min_rows: int = 16,
    out_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """GH-18: bake ONE image with the Syscall ABI v2 kernel (indexed
    table dispatch). Two passes: pass 1 fixes the packed pixel PCs
    (:__ksys/:__g18dispatch/:__g18fault/:__g18done/:__task_6/:__task_7/
    :__g18tick/:__g18tile), pass 2 bakes the final image."""
    global _GH18_DISPATCH_PC, _GH18_KSYS_PC, _GH18_FAULT_PC, _GH18_RET_PC
    global _GH18_TASK_A_PC, _GH18_TASK_B_PC, _GH18_TICK_PC, _GH18_TILE_PC
    global GH18_TILE_PC
    try:
        txt1 = _gh18_kernel_program_text(mode, status_word, timer_quantum)
        _, coords1 = assemble_glyph_to_pixels(txt1, cols_instrs=cols_instrs, min_rows=min_rows)

        def packed(label: str) -> int:
            col, row = coords1[label]
            return (col & 0xFFFF) | ((row & 0xFFFF) << 16)

        _GH18_DISPATCH_PC = packed(":__g18dispatch")
        _GH18_KSYS_PC = packed(":__ksys")
        _GH18_FAULT_PC = packed(":__g18fault")
        _GH18_RET_PC = packed(":__g18done")
        _GH18_TASK_A_PC = packed(":__task_6")
        _GH18_TASK_B_PC = packed(":__task_7")
        _GH18_TICK_PC = packed(":__g18tick")
        _GH18_TILE_PC = packed(":__g18tile")
        GH18_TILE_PC = _GH18_TILE_PC
        img = bake_image(
            _gh18_kernel_program_text(mode, status_word, timer_quantum),
            atlas=None,
            cols_instrs=cols_instrs,
            min_rows=min_rows,
            out_path=out_path,
        )
        if mode in GH18_SEEDED_MODES:
            # The seeded legs RAM-seed ONLY the table word — the reference
            # tile must already live in-image at the tile rect (receipt:
            # 2026-09-08 run, the rect kept its 24 bake-time HALTs, table
            # dispatch jumped into them, run died at cell 215 with 754=0).
            # Patch the payload words straight into the rect pixels AFTER
            # the bake (word i = pixel i, 24-bit RGB, alpha byte unused).
            # fs_v2: the reference triple tile is the GH-18 gate fixture,
            # not an FS op — in the FS image the rect stays at its
            # bake-time HALTs and the ONLY table entries are the slots
            # admitted at runtime (SYS 10/11/12). Skip the patch.
            if mode != "fs_v2":
                payload = _gh18_tile_payload_for_test(
                    resume=_gh18_mode_resume(mode, coords1))
                h, w = img.shape[:2]
                cells_per_row = w // 4
                tcol, trow = coords1[":__g18tile"]
                for i, pw in enumerate(payload[:GH9_N_INSTRS * 4]):
                    cell = trow * cells_per_row + tcol + i // 4
                    x = (cell % cells_per_row) * 4 + (i % 4)
                    y = cell // cells_per_row
                    img[y, x] = ((pw >> 16) & 0xFF, (pw >> 8) & 0xFF, pw & 0xFF)
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
    finally:
        # GH18_TILE_PC deliberately survives the bake: it IS the admission
        # receipt — the host seeds the table with it after boot. Resetting
        # it here made every post-bake admission seed a 0 PC (KJMP-to-0 ->
        # :__entry -> prologue re-run). Reset only the pass-1 internals.
        _GH18_DISPATCH_PC = _GH18_KSYS_PC = _GH18_FAULT_PC = _GH18_RET_PC = 0
        _GH18_TASK_A_PC = _GH18_TASK_B_PC = _GH18_TICK_PC = 0
        _GH18_TILE_PC = GH18_TILE_PC


# --- GH-18 host-side admission: autoatlas.ingest() == the only door ----------
#
# NOTE: admit_syscall() itself lives in autoatlas.py (the tests import it
# from there — admission is an autoatlas operation: ingest() IS the door).
# baker.py supplies the tile-rect relocation helpers it needs.

def _gh18_mode_resume(mode: str, coords: Dict[str, Tuple[int, int]]) -> int:
    """Packed PC of :__ksys_done (the tile's KJMP-home target) for the
    ACTUAL mode's coords — paged_dispatch's prologue ptloop shifts the
    dispatcher block, so a mode-independent resume PC would be stale."""
    col, row = coords[":__ksys_done"]
    return (col & 0xFFFF) | ((row & 0xFFFF) << 16)


def _gh18_dispatch_resume(mode: str = "baseline") -> int:
    """Packed pixel PC of :__ksys_done — the tile's KJMP target. Resolved
    the same way the two-pass bake does (deterministic assemble).
    mode MUST match the image being admitted into: the admit-mode
    prologue (PTE_PIX arming) shifts the dispatcher block, so baseline
    coords mis-land the tile's KJMP home (receipt 2026-09-08,
    debug_gh18_decode3: baked admit tile exits LDI r30 0x110000, the
    _gh18_dispatch_resume() default gave 0x0d0005 -> on-die verify saw
    754 clobbered to 6 after a correct ST)."""
    txt1 = _gh18_kernel_program_text(mode)
    _, coords = assemble_glyph_to_pixels(txt1, cols_instrs=8, min_rows=16)
    col, row = coords[":__ksys_done"]
    return (col & 0xFFFF) | ((row & 0xFFFF) << 16)


def _gh18_tile_pc(mode: str = "admit") -> int:
    """Packed pixel PC of :__g18tile — recomputed deterministically, NOT
    read from the module global. Receipt (probe 22, 2026-09-08): baker.py
    loads as TWO module objects (glyph_gpt.baker via autoatlas's
    sys.path.insert and tools.glyph_gpt.baker via the test import); the
    bake writes GH18_TILE_PC only on the object the CALLER imported, so
    autoatlas's `B.GH18_TILE_PC` read the stale pre-bake 0 and every
    admission seeded KJMP-to-0 (prologue re-run, result 0). mode MUST
    match the baked image's mode: the admit-mode task text shifts the
    tile rect (probe 23: baseline 0x180006 vs admit 0x1b0004)."""
    txt1 = _gh18_kernel_program_text(mode)
    _, coords = assemble_glyph_to_pixels(txt1, cols_instrs=8, min_rows=16)
    col, row = coords[":__g18tile"]
    return (col & 0xFFFF) | ((row & 0xFFFF) << 16)


def __getattr__(name):
    # GH-18: admit_syscall lives in autoatlas (the admission door is
    # autoatlas.ingest). Lazy re-export keeps `baker.admit_syscall`
    # working for callers while autoatlas owns the implementation.
    if name == "admit_syscall":
        from glyph_gpt.autoatlas import admit_syscall
        return admit_syscall
    raise AttributeError(name)


# GH-20 re-exports: the Pixel-FS v2 bake + its ABI constants live in
# fs_v2.py (one definition; baker re-exports for the test import at
# tests/test_gh20_fs_v2.py — `from tools.glyph_gpt.baker import
# pixel_fs_v2_kernel_image, GH18_TABLE_WORD`).
from glyph_gpt.fs_v2 import (  # noqa: E402,F401
    pixel_fs_v2_kernel_image,
    FSV2_FSTAB_WORD,
    FSV2_NSLOTS,
    FSV2_SLOT_WORDS,
    FSV2_DATA_WORD,
    FSV2_DATA_END,
    FSV2_N_APPEND,
    FSV2_N_RENAME,
    FSV2_N_UNLINK,
    _gh20_seed_fstab,
    _gh20_bake_stamp,
)

# GH-21 re-exports: the POSIX Syscall Shim bake + its ABI constants live
# in posix_shim.py (one definition; baker re-exports for the test import
# at tests/test_gh21_posix_shim.py — `from tools.glyph_gpt.baker import
# posix_shim_kernel_image`).
from glyph_gpt.posix_shim import (  # noqa: E402,F401
    posix_shim_kernel_image,
    GH21_ABI_VERSION,
    GH21_KERNEL_OK,
    GH21_STDOUT_A,
    GH21_STDOUT_B,
    GH21_EXIT_CODE,
)

# GH-22 re-exports: the Device Driver ABI bake + its constants live in
# gh22_driver_abi.py (one definition; baker re-exports for the test import
# at tests/test_gh22_device_driver_abi.py — `from tools.glyph_gpt.baker
# import driver_abi_kernel_image`).
from glyph_gpt.gh22_driver_abi import (  # noqa: E402,F401
    driver_abi_kernel_image,
)

# GH-23 re-exports: the libc runtime bake + its ABI constants live in
# libc_runtime.py (one definition; baker re-exports for the test import
# at tests/test_gh23_libc_runtime.py — `from tools.glyph_gpt.baker
# import libc_runtime_kernel_image`).
from glyph_gpt.libc_runtime import (  # noqa: E402,F401
    libc_runtime_kernel_image,
    GH23_ABI_VERSION,
    GH23_KERNEL_OK,
    GH23_STDOUT_A,
    GH23_STDOUT_B,
    GH23_STDOUT_C,
    GH23_STDOUT_D,
    GH23_EXIT_CODE,
    GH23_BRK,
    GH23_HEAP_BASE,
)


if __name__ == "__main__":
    main()
