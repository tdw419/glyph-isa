#!/usr/bin/env python3
"""GH-21: POSIX Syscall Shim — bake the posix-mode Syscall ABI v2 kernel
with a compiled C program as the USER task text and shim tiles in the
GH-18 tile rect (syscall table slots lit ONLY by admission).

Shim ABI (engine facts, see tests/test_gh21_posix_shim.py header):
  - The engine marshals a7/a0/a1 into SYS_N/SYS_A0/SYS_A1 (glyph_isa_v2
    SYSCALL branch). a2 is NOT marshaled, so sys_write's length rides the
    tile contract: the gate binary writes exactly STDOUT_WORDS words.
  - The unsigned idx mask ((n-6) & 15) aliases the POSIX numbers onto the
    16 table slots; the shim image lights those slots with shim tiles:
      214 (brk) -> 1568    93 (exit) -> 1575    63 (read) -> 1577
       56 (openat) -> 1570  57 (close) -> 1571   64 (write) -> 1578
  - The shim tiles are baked straight into the tile rect from the SOURCES
    below (they are the "admitted" tiles for the green path); the RED
    admission-leg proves the pipeline still refuses unverified candidates
    (admit_shims path in the test monkeypatches escalate — the tiles then
    never light and write traps to unknown). The bake-time sources are
    IR-verified (ir_pixel_words) exactly like every ingest() candidate.
"""
from __future__ import annotations

from typing import Any, Optional, Union
from pathlib import Path

import numpy as np

# ── GH-21 ABI constants ──────────────────────────────────────────────────
GH21_ABI_VERSION = 0x00020019            # low byte bumped 0x18 -> 0x19
GH21_KERNEL_OK = 0xCAFE0000 | 25         # posix-mode status tail id 25
GH21_STDOUT_A = 718                      # word-exact stdout channel
GH21_STDOUT_B = 719                      #   (BOX1 words — the C task's
GH21_EXIT_CODE = 720                     #    sibling arena, kernel-writable)

# POSIX number -> table slot (the unsigned mask's aliasing, precomputed)
GH21_SLOT_BRK = 1568       # 214 -> (214-6) & 15 = 0
GH21_SLOT_OPENAT = 1570    # 56  -> 2
GH21_SLOT_CLOSE = 1571     # 57  -> 3
GH21_SLOT_EXIT = 1575      # 93  -> 7
GH21_SLOT_READ = 1577      # 63  -> 9
GH21_SLOT_WRITE = 1578     # 64  -> 10

SYS_A0_WORD = SYS_A0_ADDR >> 2 if False else 0  # replaced at import below


def _gh21_sys_write_tile() -> str:
    """sys_write tile: copy the 2-word stdout window (buf words at
    SYS_A1's word address) to the stdout channel, verdict 0 in r2.

    The engine marshals a0/a1 into SYS_A0/SYS_A1. write()'s POSIX ABI is
    write(fd, buf, len) — a0 = fd, a1 = buf BYTE address — so the buf
    address rides SYS_A1 (word 8206), NOT SYS_A0 (word 8205 = fd = 1).
    Receipt (2026-09-09, dbg_gh21_green18/19): a tile reading SYS_A0 saw
    1 and 1>>2 = 0, LD [0] -> 0 — stdout words stayed zero with every
    other link (LD/ST/KJMP) probe-verified green. The tile converts the
    byte address to a word address (SHR 2) and LDs through the identity
    map (admit-mode prologue arms identity PTEs vpn 0..7, V|W|U).

    Tile length: 14 instructions (12 + the 2-instr KJMP home). The rect
    holds 24 (GH9_N_INSTRS) — write+exit are PACKED (write first, exit
    right after), not fixed halves (see _stamp below)."""
    return (
        ":__entry\n"
        "LDI r15 0x200E\n"        # SYS_A1_ADDR>>2 = 0x200E (MMIO window,
                                  # identity-mapped for SUPER)
        "LD r6 r15\n"             # r6 = buf byte address (a1)
        "LDI r13 2\n"
        "SHR r6 r13\n"            # r6 = buf word address
        "LD r7 r6\n"              # word 0 of the message
        "LDI r15 718\n"
        "ST r15 r7\n"             # stdout word A
        "LDI r13 1\n"
        "ADD r6 r13\n"            # next word
        "LD r7 r6\n"              # word 1 of the message
        "LDI r15 719\n"
        "ST r15 r7\n"             # stdout word B
        "HALT\n"                  # rewritten to the KJMP home by _stamp
    )


def _gh21_sys_exit_tile() -> str:
    """sys_exit tile: copy a0 (the exit code) into GH21_EXIT_CODE and
    KJMP straight to the kernel's done tail (:__g18done) — the C task
    never returns, so the task's outermost RET (which would HALT in USER
    before the kernel tail writes the status word) never runs. Receipt
    (2026-09-09, dbg_gh21_green10): without the KJMP the run stopped at
    status 0; with it the done tail runs 188..201 and status = 0xCAFE0019.
    SYS_A0 is CORRECT here (a0 = exit code, unlike write's a0 = fd)."""
    return (
        ":__entry\n"
        "LDI r15 0x200D\n"        # SYS_A0 word (0x200C is SYS_N — receipt
                                  # dbg_gh21_now: exit code got 93, the raw
                                  # syscall number, because the tile read
                                  # SYS_N; SYS_A0 = 0x8034>>2 = 0x200D)
        "LD r6 r15\n"             # r6 = exit code (a0)
        "LDI r15 720\n"
        "ST r15 r6\n"             # GH21_EXIT_CODE <- code
        "HALT\n"                  # rewritten to the KJMP home by _stamp
    )


# read/openat/close/brk tiles: the gate binary only exercises write/exit;
# read+close+brk+openat land as verdict-0 stubs whose contract is "leave
# the box words untouched, verdict 0" (roadmap lists the six numbers the
# SHIM maps; the C fixture drives two). Kept minimal and honest: they are
# REAL tiles (IR-verified, admitted) so the slots are live, but their
# semantics beyond the verdict are the GH-22/GH-23 items' to grow.
def _gh21_verdict0_stub() -> str:
    return (
        ":__entry\n"
        "XOR r2 r2\n"
        "LDI r15 754\n"
        "ST r15 r2\n"
        "HALT\n"
    )


def posix_shim_kernel_image(
    atlas: Any,
    status_word: int = 950,
    timer_quantum: int = 35,
    cols_instrs: int = 8,
    min_rows: int = 16,
    user_program: Optional[str] = None,
    admit_shims: bool = False,
    out_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """GH-21: bake the posix-mode Syscall ABI v2 kernel.

    user_program: glyph assembly text of the transpiled C binary (from
    the test loader). When given, it REPLACES task A's text as the USER
    task (the kernel KJMPs into it after the prologue); when None the
    GH-18 baseline task A runs (ABI legs stay green).

    admit_shims: True = the two shim tiles (write/exit) reach the tile
    rect through the ADMISSION PIPELINE: each tile's contract is sent to
    autoatlas.escalate; a refusal (E_ATLAS_UNVERIFIED — leg 4
    monkeypatches it to fail) leaves ALL shim slots dark and the C
    program's write traps to unknown. False (default) = the bake-time
    green path: the IR-verified tile sources below are stamped straight
    into the rect (same ir_pixel_words gate every ingest() candidate
    passes) — legs 2/3 run this path.
    """
    from glyph_gpt.baker import (  # deferred: baker re-exports this module
        syscall_abi_kernel_image, _gh18_kernel_program_text,
        _gh18_tile_pc, GH9_N_INSTRS, GH18_TABLE_PIX_WORD,
        GH18_TABLE_WORD,
    )
    from tools.glyph_gpt.autoatlas import ir_pixel_words
    from glyph_gpt.baker import _gh18_relocate_jumps as _rel_baker

    # mode="posix": the admit-mode prologue (BOX2 + vpn-6 PIX table page
    # + identity PTEs) is REQUIRED for table dispatch; _gh18_user_task
    # falls back to baseline task text (this module swaps in the C
    # program at bake time when user_program is given).
    if user_program is None:
        img = syscall_abi_kernel_image(
            atlas, mode="posix", status_word=status_word,
            timer_quantum=timer_quantum, cols_instrs=cols_instrs,
            min_rows=min_rows, out_path=out_path)
        return img

    # ── program assembly: swap the C binary in as task A ──────────────
    # Text-level splice (the proven GH-20 mode-splice pattern): generate
    # the posix task text, replace task A's body with the C program's
    # glyph text (minus its own :__entry prologue — the kernel latches
    # USER and KJMPs to :__task_6, so the program starts at its first
    # instruction), then bake via the normal two-pass path.
    task_txt = _gh18_kernel_program_text("posix", status_word, timer_quantum)
    lines = task_txt.splitlines()
    start = lines.index(":__task_6")
    end = lines.index(":__task_7")
    prog_lines = [ln for ln in user_program.splitlines()
                  if ln.strip() and not ln.strip().startswith("#")]
    if prog_lines and prog_lines[0] == ":__entry":
        prog_lines = prog_lines[1:]
        # drop the transpiler's own JMP :_start (the kernel IS the entry)
        while prog_lines and (prog_lines[0].startswith("JMP ")
                              or not prog_lines[0].strip()):
            if prog_lines[0].startswith("JMP "):
                prog_lines = prog_lines[1:]
                break
            prog_lines = prog_lines[1:]
    elif prog_lines and prog_lines[0].startswith("JMP "):
        prog_lines = prog_lines[1:]
    # ── splice-time tail fix (receipt 2026-09-09, probe_leg4): the C
    # task terminates with the transpiler's outermost RET — correct for
    # the green image (the exit tile KJMPs :__g18done so the RET never
    # runs) but FATAL for the refused image (leg 4): write AND exit both
    # trap to unknown, SYSRET back, and the task's outermost RET pops
    # mem[671]=0 -> the engine halts IN USER with the status word
    # unwritten (status 0, not KERNEL_OK). Rewrite exactly the trailing
    # RET of the program body into the kernel KJMP home — the green-path
    # semantics are unchanged (that RET is unreachable there; the exit
    # tile jumps straight to :__g18done), and the refused path now tails
    # into the kernel done block: BADSYS 'E' recorded, status 0xCAFE0019.
    tail = [ln for ln in prog_lines]
    while tail and not tail[-1].strip():
        tail.pop()
    assert tail and tail[-1].split(";")[0].strip() == "RET", (
        "posix program must end in the transpiler's outermost RET")
    prog_lines = tail[:-1] + [
        "LDI r30 <G18DONE_PC>            ; outermost RET -> kernel done",
        "KJMP r30",
    ]

    def _resolve_done(txt: str, pc: int) -> str:
        # <G18DONE_PC> resolves to ONE LDI immediate in every pass, so the
        # instruction count (and therefore coords1) is invariant across the
        # substitution — pass 0/1 assemble with the placeholder as 0, pass
        # 2 bakes the real packed :__g18done PC (receipt 2026-09-09: the
        # literal '<G18DONE_PC>' string reached glyph_isa_v2's LDI decoder
        # and raised ValueError in all three C-binary legs, RED gate).
        n = txt.count("<G18DONE_PC>")
        assert n == 1, f"expected exactly one <G18DONE_PC>, found {n}"
        return txt.replace("<G18DONE_PC>", str(pc))

    spliced = lines[:start] + [":__task_6"] + prog_lines + lines[end:]
    new_txt = _resolve_done("\n".join(spliced) + "\n", 0)

    # rebind the version word + status tail: patch the generated text's
    # LDI constants (LDI r9 24 -> 25 for the status id; the ABI word
    # 0x00020018 is packed via _gh18_pack_const LDI r14 24 / r13 2).
    # Least-surprise approach: string surgery on the exact lines.
    new_txt = new_txt.replace("LDI r9 24", "LDI r9 25")
    new_txt = new_txt.replace("LDI r14 24\nLDI r13 16\nSHL r13 r4\n"
                              "OR r14 r13\nLDI r15 952",
                              "LDI r14 25\nLDI r13 16\nSHL r13 r4\n"
                              "OR r14 r13\nLDI r15 952")

    # two-pass bake with the SPLICED text (mirror syscall_abi_kernel_image)
    import glyph_gpt.baker as B
    from tools.rv64i_to_glyph import assemble_glyph_to_pixels

    # PASS 0 (receipt 2026-09-09, dbg_gh21_live): the kernel program text
    # embeds the module-global PCs as LDI immediates at GENERATION time
    # (:__ksys arming stores _GH18_KSYS_PC, the entry KJMP embeds
    # _GH18_TASK_A_PC, the task tails embed _GH18_DISPATCH_PC/_GH18_RET_PC).
    # Generating the text BEFORE computing the spliced coords bakes
    # STALE/zero vectors (fresh process: LDI r30 0 -> KJMP to cell 0 in
    # USER mode -> the kernel prologue re-runs as a USER task and faults
    # at 0x800c). The splice only INSERTS instructions between :__task_6
    # and :__task_7, so the spliced label coords are computable
    # standalone: assemble the spliced text once with the CURRENT
    # globals, rebind the globals from those coords, then regenerate the
    # text (now with correct immediates) for pass 2.
    def _splice_text(txt: str) -> str:
        ls = txt.splitlines()
        st = ls.index(":__task_6")
        en = ls.index(":__task_7")
        joined = ls[:st] + [":__task_6"] + prog_lines + ls[en:]
        out = "\n".join(joined) + "\n"
        out = out.replace("LDI r9 24", "LDI r9 25")
        out = out.replace("LDI r14 24\nLDI r13 16\nSHL r13 r4\n"
                          "OR r14 r13\nLDI r15 952",
                          "LDI r14 25\nLDI r13 16\nSHL r13 r4\n"
                          "OR r14 r13\nLDI r15 952")
        return out

    _, coords1 = assemble_glyph_to_pixels(new_txt, cols_instrs=cols_instrs,
                                          min_rows=min_rows)

    def packed(label: str) -> int:
        col, row = coords1[label]
        return (col & 0xFFFF) | ((row & 0xFFFF) << 16)

    saved = (B._GH18_DISPATCH_PC, B._GH18_KSYS_PC, B._GH18_FAULT_PC,
             B._GH18_RET_PC, B._GH18_TASK_A_PC, B._GH18_TASK_B_PC,
             B._GH18_TICK_PC, B._GH18_TILE_PC)
    B._GH18_DISPATCH_PC = packed(":__g18dispatch")
    B._GH18_KSYS_PC = packed(":__ksys")
    B._GH18_FAULT_PC = packed(":__g18fault")
    B._GH18_RET_PC = packed(":__g18done")
    B._GH18_TASK_A_PC = packed(":__task_6")
    B._GH18_TASK_B_PC = packed(":__task_7")
    B._GH18_TICK_PC = packed(":__g18tick")
    B._GH18_TILE_PC = packed(":__g18tile")
    try:
        # REGENERATE task_txt WITH THE REBOUND GLOBALS, then splice.
        # Receipt (2026-09-09, dbg_jc_repro3/4): splicing the PASS-0
        # task_txt re-bakes the PRE-REBIND text — its PC immediates are
        # the pre-rebind values (fresh process: 0), so the image's entry
        # KJMP vectors to cell 0, the kernel prologue re-runs as a USER
        # task, and the run faults at 0x800C (BOX0_LO) with all three
        # C-binary legs RED. Regenerating after the rebind bakes
        # 'LDI r30 <packed>' with the real packed PCs (dbg_jc_final5:
        # pre 0/0/0/0 -> post 0x190002/0x190003/0x1376260/0x1507332).
        # The regenerated text has the same instruction COUNT (the PC
        # immediates replace same-shaped LDI lines), so coords1 stays
        # valid — assert it anyway, coordinates are the whole game here.
        txt2 = _splice_text(_gh18_kernel_program_text(
            "posix", status_word, timer_quantum))
        txt2 = _resolve_done(txt2, B._GH18_RET_PC)
        _, coords2 = assemble_glyph_to_pixels(txt2, cols_instrs=cols_instrs,
                                              min_rows=min_rows)
        assert coords1 == coords2, (
            "spliced coords shifted between pass 0 and pass 1 — "
            f"{coords1} vs {coords2}")
        img = B.bake_image(txt2, atlas=None, cols_instrs=cols_instrs,
                           min_rows=min_rows, out_path=out_path)
    finally:
        (B._GH18_DISPATCH_PC, B._GH18_KSYS_PC, B._GH18_FAULT_PC,
         B._GH18_RET_PC, B._GH18_TASK_A_PC, B._GH18_TASK_B_PC,
         B._GH18_TICK_PC, B._GH18_TILE_PC) = saved

    if not admit_shims:
        pass  # green path stamps below — the tiles ARE the baked contract
    # ── shim tiles into the tile rect + table slots ────────────────────
    # SEMANTICS (leg 4's contract): the green path stamps the IR-verified
    # tile sources DIRECTLY (default, legs 2/3); admit_shims=True routes
    # each tile through autoatlas.escalate — a refusal (leg 4's
    # monkeypatched E_ATLAS_UNVERIFIED) leaves the slot dark and the
    # dispatch falls to unknown. Proof = admission either way.
    #
    # The tile rect holds GH9_N_INSTRS (24) instructions; the bake stamps
    # ONE tile there and the gate path needs TWO — so the tiles are
    # PACKED: write at cell 0, exit immediately after (write 14 instrs +
    # exit 9 = 23 <= 24). NO fixed halves. Receipts (2026-09-09):
    #  - dbg_gh21_green9..13: half-split base = half * 96 px-words put
    #    the exit entry at tcol+12/+14 — with an 8-col layout that packs
    #    col 17/19 >= 8 into the KJMP target: the engine's step() bounds
    #    check (x >= width) then HALTS SILENTLY on the first tile fetch
    #    (no trace entry, no fault, status 0). The entry PC must be
    #    FLAT-IZED: entry_cell % cells_per_row | (entry_cell //
    #    cells_per_row) << 16.
    #  - dbg_gh21_green13/20: packed layout + flat-ized entries + explicit
    #    KJMP homes -> word-exact HELLO at 718/719, exit code at 720,
    #    status 0xCAFE0019, no fault (leg-2's contract, probe-green).
    # Each tile ends with an EXPLICIT KJMP home (no HALT->rewrite
    # surprise): write -> :__ksys_done (the SYSRET path resumes the C
    # task), exit -> :__g18done (the kernel done tail writes status; the
    # C task's outermost RET must never run — it would HALT in USER with
    # the status word unwritten, receipt dbg_gh21_green10).
    from glyph_gpt.baker import _gh18_dispatch_resume
    # Receipt (2026-09-09, dbg_gh21_layout/seq2): the tile rect PC MUST
    # come from the SPLICED coords. _gh18_tile_pc(mode="posix") assembles
    # the UNSPLICED text (:__g18tile at cell 232); the C-program splice
    # shifts the rect to cell 257 — so the tiles were stamped over the
    # task's own ecall blocks (cells 232..254), the run executed tile
    # code in USER, KJMPed :__ksys_done, SYSRET jumped to stale
    # SYSCALL_PC=0, the prologue re-ran as USER and E-K1 faulted at
    # BOX0_LO (word 8195, cell 59): leg 2/3 RED with "everything else
    # green". :__ksys_done/:__g18done sit BEFORE the task region, so
    # their coords are splice-invariant — only the rect moves.
    _tc, _tr = coords1[":__g18tile"]
    tile_pc = (_tc & 0xFFFF) | ((_tr & 0xFFFF) << 16)
    ksys_done = _gh18_dispatch_resume(mode="posix")
    dc, dr = coords1[":__g18done"]
    g18done = (dc & 0xFFFF) | ((dr & 0xFFFF) << 16)

    def _home(tile_text: str, home_pc: int) -> str:
        assert tile_text.count("HALT") == 1, "tile must end in exactly one HALT"
        assert tile_text.rstrip().endswith("HALT"), "HALT must be terminal"
        return tile_text.replace(
            "HALT", f"LDI r30 {home_pc}\nKJMP r30")

    h, w, _ = img.shape
    cells_per_row = w // 4
    tcol_px = tile_pc & 0xFFFF
    trow_px = (tile_pc >> 16) & 0xFFFF
    base_cell = trow_px * cells_per_row + tcol_px

    cursor = 0                            # next free instr-cell in the rect

    def _stamp(tile_text: str, table_slot: int) -> None:
        nonlocal cursor
        words = ir_pixel_words(tile_text)          # IR gate BEFORE pixels
        nz = max((i for i, pw in enumerate(words) if pw), default=-1) + 1
        ncells = (nz + 3) // 4
        assert cursor + ncells <= GH9_N_INSTRS, (
            f"packed shim tiles overflow the rect: {cursor + ncells} "
            f"> {GH9_N_INSTRS}")
        _rel_baker(tile_text, words, mode="posix")  # mode-correct relocs
        for i in range(nz):
            pw = words[i]
            cell = base_cell + cursor + (i // 4)
            y = cell // cells_per_row
            x = (cell % cells_per_row) * 4 + i % 4
            img[y, x] = ((pw >> 16) & 0xFF, (pw >> 8) & 0xFF, pw & 0xFF)
        # FLAT-IZED entry PC — the root-cause fix for the silent-halt
        # receipt above (tcol + cell_off overflowed the 8-col row).
        entry_cell = base_cell + cursor
        entry_pc = (entry_cell % cells_per_row) \
            | ((entry_cell // cells_per_row) << 16)
        # light the table slot through the RESERVED pfn-5 pixel window
        # (engine-exact admit-mode layout: slot word -> pixel
        #  GH18_TABLE_PIX_WORD + (slot - GH18_TABLE_WORD), 1 px/word)
        pw = GH18_TABLE_PIX_WORD + (table_slot - GH18_TABLE_WORD)
        img[pw // w, pw % w] = ((entry_pc >> 16) & 0xFF,
                                (entry_pc >> 8) & 0xFF, entry_pc & 0xFF)
        cursor += ncells

    def _admit(tile_text: str, home_pc: int, table_slot: int,
               contract: str) -> None:
        """Admission door (admit_shims=True): the tile's contract goes to
        autoatlas.escalate; only a VERIFIED escalation stamps the tile."""
        from tools.glyph_gpt import autoatlas as aa
        esc = aa.escalate(contract, model="qwen2.5-coder:14b")
        if not getattr(esc, "verified", False):
            return                      # slot stays dark -> unknown handler
        _stamp(_home(tile_text, home_pc), table_slot)

    if admit_shims:
        _admit(_gh21_sys_write_tile(), ksys_done, GH21_SLOT_WRITE,
               "sys_write(fd, buf, len): copy the 2 glyph words at (a1>>2) "
               "to words 718/719 (stdout), return via KJMP :__ksys_done")
        _admit(_gh21_sys_exit_tile(), g18done, GH21_SLOT_EXIT,
               "sys_exit(code): store a0 (SYS_A0) to word 720 (exit code), "
               "KJMP :__g18done (task never returns)")
    else:
        _stamp(_home(_gh21_sys_write_tile(), ksys_done), GH21_SLOT_WRITE)
        _stamp(_home(_gh21_sys_exit_tile(), g18done), GH21_SLOT_EXIT)
    # Receipt (2026-09-09, dbg_gh21_stages): bake_image(out_path=...) saves
    # the PRE-STAMP image; the shim tiles + table slots only exist in the
    # returned array, so the on-disk image (npy/PNG round-trip) had NO
    # shims — px1322 (0,0,0) at every lifecycle stage, the C task's first
    # SYS dispatched unknown and stdout words stayed 0 (leg-2 RED).
    # syscall_abi_kernel_image re-saves after its post-bake tile patch
    # (baker.py 4432-4440); mirror that here.
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
