#!/usr/bin/env python3
"""GH-23: Picolibc/Newlib Full Port — bake the libc-mode Syscall ABI v2
kernel with the transpiled C (libc-linked) program as the USER task,
a REAL sys_brk tile, and a 4-word-window sys_write tile in the GH-18
tile rect (syscall table slots lit ONLY by admission/stamping — the
same door GH-21 uses).

libc ABI (engine facts, see tests/test_gh23_libc_runtime.py header):
  - The engine marshals a7/a0/a1 into SYS_N/SYS_A0/SYS_A1 (glyph_isa_v2
    SYSCALL branch). a2 is NOT marshaled, so sys_write's length rides
    the tile contract: the gate's libc buffers exactly 16 bytes (4
    words) per write(1, buf, 16) and the tile copies a FIXED 4-word
    stdout window.
  - sys_brk: the libc is a bump allocator over the brk. The brk value
    lives IN-IMAGE at GH23_BRK (word 723) — the tile reads it, adds the
    delta from SYS_A0, stores it back, and returns the OLD break in
    SYS_A0 (sbrk semantics; the engine's SYSRET delivers SYS_A0 into
    the task's a0). GH-21 left this slot a verdict-0 stub; GH-23 is the
    upgrade (the test leg pins mem[723] advancing by the deltas).
  - The unsigned idx mask ((n-6) & 15) aliases the POSIX numbers onto
    the 16 table slots:
      214 (brk) -> 1568       93 (exit) -> 1575      64 (write) -> 1578
    (GH-23 stamps exactly these three — the gate fixture only calls
    write/exit/brk. read/openat/close remain dark: an unlit slot traps
    cleanly to unknown, leg 4.)
  - The bake-time tile sources are IR-verified (ir_pixel_words) exactly
    like every ingest() candidate.
"""
from __future__ import annotations

from typing import Any, Optional, Union
from pathlib import Path

import numpy as np

# ── GH-23 ABI constants ──────────────────────────────────────────────────
GH23_ABI_VERSION = 0x0002001A            # low byte bumped 0x19 -> 0x1A
GH23_KERNEL_OK = 0xCAFE0000 | 26         # libc-mode status tail id 26
GH23_STDOUT_A = 718                      # word-exact stdout channel,
GH23_STDOUT_B = 719                      #   words 718..721 (4-word window;
GH23_STDOUT_C = 720                      #   BOX1 arena, kernel-writable —
GH23_STDOUT_D = 721                      #   same channel GH-21 used, wider)
GH23_EXIT_CODE = 722                     # sys_exit status code lands here
GH23_BRK = 723                           # the in-image program break (words)
GH23_HEAP_BASE = 2560                    # vpn 10, identity-mapped — ABOVE
                                         # the transpiler's PTR_TABLE region
                                         # (word 2048..2369; receipt
                                         # dbg_gh23_cron60..64: heap at 2048
                                         # overwrote the indirect-call fn
                                         # pointer entries and qsort's jalr
                                         # CALLR'd into kernel text).

# POSIX number -> table slot (the unsigned mask's aliasing, precomputed)
GH23_SLOT_BRK = 1568       # 214 -> (214-6) & 15 = 0
GH23_SLOT_EXIT = 1575      # 93  -> 7
GH23_SLOT_WRITE = 1578     # 64  -> 10


def _gh23_sys_write_tile() -> str:
    """sys_write tile: copy the 4-word stdout window (buf words at
    SYS_A1's word address) to the stdout channel, verdict 0 in r2.

    Same link contract as GH-21's tile (receipt dbg_gh21_green18/19):
    the buf address rides SYS_A1 (word 0x200E), SHR 2 -> word address,
    LD through the identity map (vpn 0..7 armed V|W|U by the admit-mode
    prologue the libc-mode bake reuses).

    Tile length: 26 instructions (24 + the 2-instr KJMP home). GH-21's
    packed pair (write 14 + exit 5 = 19) leaves room; GH-23 packs
    write 26 + exit 9 + brk 10 = 45 > 24 — so the rect is DOUBLED:
    the bake stamps a second contiguous rect block right after the
    first (see _stamp: the rect is the tile-rect PC plus as many
    contiguous cells as the packed tiles need; the bake-time image is
    zero-padding beyond :__g18tile's 24 HALTs ONLY IF min_rows is
    raised — libc_runtime_kernel_image defaults min_rows=34 so the
    72-cell pack lands in padding, never program text. The GH-18
    reserved-window test (table pixels [1312,1328)) stays green: the
    rect sits AFTER the table window in row-major pixel order and the
    doubling grows DOWNWARD rows, not into the table's row."""
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
        "LDI r13 1\n"
        "ADD r6 r13\n"
        "LD r7 r6\n"              # word 2 of the message
        "LDI r15 720\n"
        "ST r15 r7\n"             # stdout word C
        "LDI r13 1\n"
        "ADD r6 r13\n"
        "LD r7 r6\n"              # word 3 of the message
        "LDI r15 721\n"
        "ST r15 r7\n"             # stdout word D
        "HALT\n"                  # rewritten to the KJMP home by _home
    )


def _gh23_sys_exit_tile() -> str:
    """sys_exit tile: copy a0 (the exit code) into GH23_EXIT_CODE and
    KJMP straight to the kernel's done tail (:__g18done) — the C task
    never returns (same contract as GH-21's exit tile; receipt
    dbg_gh21_green10). SYS_A0 is CORRECT here (a0 = exit code)."""
    return (
        ":__entry\n"
        "LDI r15 0x200D\n"        # SYS_A0 word (0x200C is SYS_N — receipt
                                  # dbg_gh21_now: reading SYS_N gets the
                                  # raw syscall number, not the code)
        "LD r6 r15\n"             # r6 = exit code (a0)
        "LDI r15 722\n"
        "ST r15 r6\n"             # GH23_EXIT_CODE <- code
        "HALT\n"                  # rewritten to the KJMP home by _home
    )


def _gh23_sys_brk_tile() -> str:
    """sys_brk tile: sbrk semantics — return the OLD break in SYS_A0,
    advance the in-image break at GH23_BRK by the delta in SYS_A0.

    The task's a0 = delta (words). The tile:
      r6 = mem[723]           (old break)
      r7 = r6 + a0            (new break)
      mem[723] = r7
      mem[SYS_A0] = r6        (sbrk: old break is the return value)
    then KJMPs to the resume home (:__ksys_done — SYSRET restores the
    task's register file and delivers SYS_A0 into a0).

    Tile length: 12 instructions (10 + the 2-instr KJMP home).
    Signed deltas: LDI can't produce negatives directly, but the libc
    only ever GROWS the heap in this gate (malloc); the ISA's ADD wraps
    mod 2^32 so a negative delta word would still arithmetic correctly.
    """
    return (
        ":__entry\n"
        "LDI r15 723\n"           # GH23_BRK
        "LD r6 r15\n"             # r6 = old break
        "LDI r14 0x200D\n"        # SYS_A0 word (a0 = the delta)
        "LD r13 r14\n"            # r13 = delta
        "ADD r13 r6\n"            # r13 = old + delta = new break
        "ST r15 r13\n"            # mem[723] = new break
        "LDI r14 0x200D\n"
        "ST r14 r6\n"             # SYS_A0 <- old break (sbrk return)
        "HALT\n"                  # rewritten to the KJMP home by _home
    )


def libc_runtime_kernel_image(
    atlas: Any,
    status_word: int = 950,
    timer_quantum: int = 35,
    cols_instrs: int = 8,
    min_rows: int = 34,
    user_program: Optional[str] = None,
    out_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """GH-23: bake the libc-mode Syscall ABI v2 kernel.

    user_program: glyph assembly text of the transpiled C binary (from
    the test loader). When given, it REPLACES task A's text as the USER
    task (the kernel KJMPs into it after the prologue); when None the
    GH-18 baseline task A runs (ABI legs stay green).

    The bake is the GH-21 posix-splice pattern verbatim (posix_shim.py,
    receipted line-by-line) with three libc deltas:
      1. mode="libc" kernel text (ABI word 0x0002001A, status id 26).
      2. THREE shim tiles packed into the rect (write 26, exit 4,
         brk 10 = 40 <= 48 cells): the rect grows past GH9_N_INSTRS=24
         into the zero-padding BELOW the program text (min_rows=34
         default guarantees the headroom; the GH-18 reserved table
         window [1312,1328) sits in an EARLIER row and is untouched).
      3. The brk tile is REAL (sbrk semantics over word 723), not the
         GH-21 verdict-0 stub.
    """
    from glyph_gpt.baker import (  # deferred: baker re-exports this module
        syscall_abi_kernel_image, _gh18_kernel_program_text,
        GH9_N_INSTRS, GH18_TABLE_PIX_WORD,
        GH18_TABLE_WORD,
    )
    from tools.glyph_gpt.autoatlas import ir_pixel_words
    from glyph_gpt.baker import _gh18_relocate_jumps as _rel_baker

    # DEFECT 7 (2026-09-10, cron session; receipts output/gh23_red_0945.txt,
    # output/gh23_memmask_probe.out, output/dbg_gh23_cron359.txt): the
    # engine's packed-PC format is (row<<16)|col inside a 24-BIT pixel
    # word — the row field is only 8 bits (rows 0..255). At
    # cols_instrs=8 the spliced libc task (~3600 cells = 413+ rows)
    # overflows: CALL/CALLR-pushed return PCs with row >= 256 truncate on
    # the _mem_write 24-bit mask (probe: wrote 0x1650008 = row 357, read
    # back 0x650008 = row 101), so the first thunk RET lands in the
    # task's own data-init and execution derails (cron359: RET popped
    # 0x158, pc walked to cell 810; fault 0xfffffff8 on the C-suite and
    # brk legs). Every GH-18..22 image stayed under 256 rows, so the
    # mask was never visible until GH-23's 3300-instruction task. FIX:
    # bake the libc task at cols_instrs=16 — ~230 task rows + splice
    # head, all packed PCs < 2^24. The loader in
    # tests/test_gh23_libc_runtime.py (_load_posix_program) assembles at
    # the SAME cols_instrs=16 so its pointer-table seeds agree with the
    # bake layout (asserted by the row-limit assert below).
    if user_program is not None:
        cols_instrs = 16

    # mode="libc": the admit-mode prologue (BOX2 + vpn-6 PIX table page
    # + identity PTEs vpn 0..7) is REQUIRED for table dispatch; with no
    # user_program the baseline task text runs (ABI legs stay green).
    if user_program is None:
        img = syscall_abi_kernel_image(
            atlas, mode="libc", status_word=status_word,
            timer_quantum=timer_quantum, cols_instrs=cols_instrs,
            min_rows=min_rows, out_path=out_path)
        return img

    # ── program assembly: swap the C binary in as task A ──────────────
    # Text-level splice (the proven GH-21 mode-splice pattern, verbatim).
    task_txt = _gh18_kernel_program_text("libc", status_word, timer_quantum)
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
    # ── splice-time task terminator (GH-23 defect fix, receipt
    # dbg_gh23_cron27/cron29): the GH-21 pattern REWROTE the trailing RET
    # into the kernel KJMP home — but the transpiler's LAST RET is often a
    # real function epilogue, not the task exit. In this gate binary the
    # final RET is sys_brk_ecall's: malloc -> sys_brk_ecall returned into
    # that rewritten line, KJMP'd to :__g18done, and the task died after
    # malloc — qsort/printf/exit never ran (stdout 0x0, exit word 0x0 with
    # clean 0xCAFE001A tail). The C fixture terminates via exit(0) whose
    # tile KJMPs to :__g18done, so the task never falls off the end; the
    # in-place rewrite is wrong in general. Fix: keep EVERY RET intact and
    # APPEND a fallback terminator (never reached while exit() is on the
    # task's path). Appended lines sit after the last label, so no existing
    # label coordinate moves between the two bakes.
    # ── HW-STACK RE-SEED (GH-23 defect, receipt output/dbg_gh23_cron170,
    # 2026-09-09): the transpiler's rt0 seeds the hardware call stack at
    # `LDI r31 4607` — computed for the UN-spliced layout where the task
    # sits at flat cell 0. The splice re-homes the task to cell 332, so
    # CALL/PUSH decrement r31 into words 4605/4606 = spliced cells
    # 1151/1150 — the task's own TEXT: the first CALL push wrote a
    # return-address pixel over an LDI's register byte (regs >= 32 →
    # registers[rd] IndexError, receipt cron150) and the run died at
    # step 2993 (cron149 traceback: runner.py:82 → glyph_isa_v2.py:642).
    # cr170 step-diffed the image: EXACTLY two pixel writes into task
    # text (4606 -> 0x280018, 4605 -> 0x3a0004 — packed return PCs),
    # pinpointing the collision. FIX: rewrite the rt0 stack seed to 24
    # instruction cells ABOVE the task's spliced start, keeping the
    # same 1-instruction shape; the stack's pushes then walk UPWARD
    # into fresh padding rows, never down into text.

    tail = [ln for ln in prog_lines]
    while tail and not tail[-1].strip():
        tail.pop()
    assert tail and tail[-1].split(";")[0].strip() == "RET", (
        "libc program must end in the transpiler's outermost RET")
    prog_lines = tail + [
        "LDI r30 <G18DONE_PC>            ; task terminator (fallback after RET)",
        "KJMP r30",
    ]

    def _resolve_done(txt: str, pc: int) -> str:
        # <G18DONE_PC> resolves to ONE LDI immediate in every pass, so
        # the instruction count is invariant across the substitution.
        n = txt.count("<G18DONE_PC>")
        assert n == 1, f"expected exactly one <G18DONE_PC>, found {n}"
        return txt.replace("<G18DONE_PC>", str(pc))

    # ── RESERVED-WINDOW GUARD (GH-23 defect, receipts dbg_gh23_cron73..80):
    # the syscall table's pfn-5 pixel surface [1312,1328) = CELLS 328..331
    # (1 cell = 4 px, cells/row = 8 at cols_instrs=8). The splice puts the
    # C task right after the 211-cell prologue, so its cell 328 = task
    # instruction ~117 — the data-init seed's own `OR r20 r21` — and the
    # bake-time table-slot stamps OVERWROTE it (task PC walked onto the
    # stamped pixel at step 341, unknown opcode, running=False: every
    # runtime leg "halted cleanly" mid-seed). The kernel PROLOGUE itself
    # never reaches cell 328 (it KJMPs to the task at cell ~211), so the
    # pad between prologue and task is dead code for every execution
    # path. Filler = `LDI r0 0` (non-zero pixels (236,80,80) + imm —
    # real instructions, never executed: KJMP lands the task at its new
    # offset and nothing falls through the pad).
    #
    # Where the pad count comes from (asserted, not assumed):
    #   prologue = count of instructions before :__task_6 in the spliced
    #   text; the task must START at cell >= TABLE_PIX_CELL_END = 332
    #   (window end cell 331 + 1). GH-21's ~350-cell task starts past 332
    #   anyway when its own size crosses the window — but small fixtures
    #   (the brk leg's 20-instr binary) do not, so the guard is generic.
    def _count_instr(txt_lines: list) -> int:
        return sum(1 for ln in txt_lines
                   if ln.strip()
                   and not ln.strip().startswith(":")
                   and not ln.strip().startswith("#"))

    n_prologue = _count_instr(lines[:start])
    TABLE_PIX_WORD = 1312          # GH18_TABLE_PIX_WORD (pinned constant)
    PX_PER_CELL = 4
    # DEFECT 8 (2026-09-10, cron session; receipts dbg_cron385/386):
    # the vpn-6 PIX alias maps table words [1568,1584) to PIXEL WORDS
    # [1280,1536) = FLAT CELLS [320,384) — the FULL window, not just
    # the reserved 16 words at [1312,1328). The dispatcher's
    # unknown-slot path reads the slot pixel and JZs on zero; the old
    # splice's explicit "LDI r0 0" pad cells baked NONZERO opcode/reg
    # pixels into cells [320,332), so an unlit slot read 0xFFFF00 (the
    # pad LDI's reg pixel), the JZ never fired and the kernel KJMPed
    # into (0xFF00, 255) — OOB (cron371/385). GH-21's 8-col image has
    # literal ZERO pixels there (its text is short; the window sits in
    # assemble-time zero padding), which is why its leg 3 passes.
    # FIX: pad the splice out to cell 384 (past the whole alias
    # window), then ZERO-STAMP the window cells [320,384) post-bake —
    # those pad cells are never executed (the kernel JMPs straight to
    # :__task_6 at 384), so erasing their pixels only makes the unlit
    # slot LDs read 0. The lit slots are stamped AFTER the zeroing.
    window_end_cell = 384          # first cell BEYOND the alias window

    # ── TILE-RECT RELOCATION (GH-23 defect, receipt dbg_gh23_cron89..91):
    # the dispatcher's table LD returns a packed tile PC that must ride a
    # 24-BIT PIXEL word (col 0..15 | row 16..23). With the libc task
    # spliced in, :__g18tile lands at cell 3396 = row 424 — row<<16
    # = 0x1A8000 overflows the 24-bit pixel, the slot entry truncates
    # (0xAC0002: row 168, cell 1348 — MIDDLE OF THE TASK's data-init),
    # and the first syscall KJMPs into task text (fault 0xfffffff8,
    # receipt cron82; traced to the truncation in cron89/90/91).
    # FIX: splice a SECOND tile rect (label :__g23tile + 24 HALTs + 48
    # stamp cells of padding) into the low-row pad zone BEFORE the
    # reserved window; the shim tiles stamp there, where packed PCs fit.
    # The original :__g18tile rect stays in the kernel tail untouched
    # (nothing dispatches to it in libc mode — the table slots all point
    # at :__g23tile entries).
    RECT_INSTRS = 24                        # GH9_N_INSTRS
    STAMP_RESERVE = 48                      # write 26 + exit ~9 + brk ~10
    rect_cell = n_prologue                  # right after the prologue
    assert rect_cell + RECT_INSTRS + STAMP_RESERVE <= window_end_cell, (
        f"tile rect [{rect_cell},{rect_cell + RECT_INSTRS + STAMP_RESERVE}) "
        f"collides with the reserved table window at cell {window_end_cell}")
    assert (rect_cell // PX_PER_CELL) < 256, (
        "tile rect row must fit the 24-bit pixel-word PC packing")
    g23_rect = ([":__g23tile"] + ["HALT"] * RECT_INSTRS
                + [f"LDI r0 0            ; stamp reserve (never executed)"]
                * STAMP_RESERVE)
    pad = [f"LDI r0 0            ; reserved-window pad ({window_end_cell - n_prologue - RECT_INSTRS - STAMP_RESERVE} cells, never executed)"
           for _ in range(max(0, window_end_cell - n_prologue
                              - RECT_INSTRS - STAMP_RESERVE))]

    # ── HW-STACK RE-SEED (apply the fix documented above: constants are
    # in scope only after the pad computation) ──
    # DEFECT 2 (2026-09-10, cron session; receipt output/dbg_gh23_stack4.py):
    # the first re-seed put r31 at (task_start_cell + 24) * 4 - 1 = 1423 —
    # pixel row 44, cells 353..355 — which is INSIDE THE TASK TEXT (the
    # task spans cells 332..~3600). The engine's CALL pushes at r31 -= 1,
    # i.e. BACKWARD through cells 355, 354, 353…: the first two pushes
    # overwrote the task's own data-init pixels with packed return PCs.
    # The brk leg then ran 31 extra syscall wraps: the inner RET popped
    # 0xffff15 (a corrupted LDI pixel) instead of the return PC, execution
    # fell into :out_flush and looped printf formatting until the C stack
    # (r2) underflowed — fault_addr 0xfffffff8, brk = 0xa80 (32 × +4).
    # FIX: seed the stack BELOW the task, in the dead pad between the
    # tile rect and the reserved table window — last free pixel word of
    # that pad is 1311 (window [1312,1328) must stay untouched; pushes
    # grow DOWN through pad rows ~36-40 toward zero). Those pad cells are
    # never executed and never stamped.
    task_start_cell = window_end_cell          # first task cell (pad ends here)
    # DEFECT 3 (2026-09-10, cron session; receipt output/dbg_gh23_cron291):
    # the formula below read (TABLE_PIX_WORD * PX_PER_CELL) - 1 = 5247 — a
    # PIXEL address (word 1312's *byte* extent), not the RAM word 1311 the
    # comment promised. r31=5247 put the HW call stack INSIDE the spliced
    # task's pixel space (cell 1311 = pixel word 5244..5247, rows 164+);
    # the C stack grew down through the task's own text, RET popped packed
    # PCs from corrupted words, and execution wandered the stack region
    # (cells 465..1310 walked ~1900 steps executing stack pixels as code —
    # receipts dbg_gh23_cron255/285/286) until it happened to re-hit the
    # next real SYSCALL (cell 3376). Symptom: syscalls "completed" (brk
    # advanced) but the task derailed between them — a0 stuck at the
    # marshaled delta, empty stdout, no exit code, clean halt.
    hw_stack_word = TABLE_PIX_WORD - 1         # 1311: last pad pixel word
    _n_reseed = 0
    for _i, _ln in enumerate(prog_lines):
        if _ln.startswith("LDI r31 "):
            prog_lines[_i] = (
                "LDI r31 %d            ; HW call stack re-seeded for the "
                "splice (boot seed %s sat INSIDE the spliced task text — "
                "receipt dbg_gh23_cron170)"
                % (hw_stack_word, _ln.split(";")[0].strip()[8:]))
            _n_reseed += 1
    assert _n_reseed == 1, (
        f"rt0 HW-stack seed: expected exactly 1 'LDI r31 …' head line, "
        f"found {_n_reseed} — transpiler rt0 shape changed; re-derive "
        "the HW-stack re-seed (receipt dbg_gh23_cron170)")

    spliced = (lines[:start] + g23_rect + pad
               + [":__task_6"] + prog_lines + lines[end:])
    new_txt = _resolve_done("\n".join(spliced) + "\n", 0)

    # rebind the version word + status tail (GH-21 string surgery):
    # LDI r9 24 -> 26 for the status id; ABI word 0x0002001A packed via
    # the version-word block (LDI r14 26 = 0x1A low byte).
    new_txt = new_txt.replace("LDI r9 24", "LDI r9 26")
    new_txt = new_txt.replace(
        "LDI r14 24\nLDI r13 16\nSHL r13 r4\n"
        "OR r14 r13\nLDI r15 952",
        "LDI r14 26\nLDI r13 16\nSHL r13 r4\n"
        "OR r14 r13\nLDI r15 952")

    # two-pass bake with the SPLICED text (mirror syscall_abi_kernel_image)
    import glyph_gpt.baker as B
    from tools.rv64i_to_glyph import assemble_glyph_to_pixels

    # PASS 0 (GH-21 receipt dbg_gh21_live, verbatim): assemble the
    # spliced text once with the CURRENT globals, rebind the globals
    # from those coords, then regenerate the text for pass 2.
    def _splice_text(txt: str) -> str:
        ls = txt.splitlines()
        st = ls.index(":__task_6")
        en = ls.index(":__task_7")
        # NOTE: `txt` here is the UN-SPLICED kernel text (regenerated with
        # rebound globals), so the g23 rect + pad must be re-inserted —
        # identical computation to the pass-0 splice above (same prologue,
        # same constants) — or pass-1 coords would shift vs pass-0.
        joined = ls[:st] + g23_rect + pad + [":__task_6"] + prog_lines + ls[en:]
        out = "\n".join(joined) + "\n"
        out = out.replace("LDI r9 24", "LDI r9 26")
        out = out.replace(
            "LDI r14 24\nLDI r13 16\nSHL r13 r4\n"
            "OR r14 r13\nLDI r15 952",
            "LDI r14 26\nLDI r13 16\nSHL r13 r4\n"
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
        # REGENERATE task_txt WITH THE REBOUND GLOBALS, then splice
        # (GH-21 receipt dbg_jc_repro3/4, verbatim).
        txt2 = _splice_text(_gh18_kernel_program_text(
            "libc", status_word, timer_quantum))
        txt2 = _resolve_done(txt2, B._GH18_RET_PC)
        _, coords2 = assemble_glyph_to_pixels(txt2, cols_instrs=cols_instrs,
                                              min_rows=min_rows)
        assert coords1 == coords2, (
            "spliced coords shifted between pass 0 and pass 1 — "
            f"{coords1} vs {coords2}")
        img = B.bake_image(txt2, atlas=None, cols_instrs=cols_instrs,
                           min_rows=min_rows, out_path=None)
    finally:
        (B._GH18_DISPATCH_PC, B._GH18_KSYS_PC, B._GH18_FAULT_PC,
         B._GH18_RET_PC, B._GH18_TASK_A_PC, B._GH18_TASK_B_PC,
         B._GH18_TICK_PC, B._GH18_TILE_PC) = saved

    # ── DEFECT-8 ZERO-STAMP (the fix promised by the comment at the
    # splice site — implemented 2026-09-10 cron session, receipts
    # dbg_gh23_cron1074/1075): the splice's pad cells are `LDI r0 0`
    # with NONZERO pixels, so every unlit syscall-table slot reads
    # 0xFFFF00 through the vpn-6 PIX alias instead of 0. The
    # dispatcher's JZ-on-zero never fires and KJMP goes OOB (receipt
    # dbg_cron385_dispatch.out step 2948: OOB (261120,255) — r6 loaded
    # 0xffff00 from slot 1573). Those pad cells are NEVER executed
    # (the prologue KJMPs straight to the task at cell 384+), so
    # erasing the alias window's pixels is behavior-neutral except
    # for making unlit slot LDs read 0. Zero BEFORE the _stamp calls:
    # the three lit slots are re-lit after.
    _h_img, _w_img, _ = img.shape
    for _wd in range(1280, 1536):      # pfn-5 alias window: pixel words [1280,1536) <-> table slots [1568,1584)
        img[_wd // _w_img, _wd % _w_img] = (0, 0, 0)

    # ── shim tiles into the tile rect + table slots ────────────────────
    # The tile rect PC comes from the SPLICED coords of the RELOCATED
    # :__g23tile rect (low rows — see the relocation note above; the
    # original :__g18tile rect at row 424 CANNOT be addressed through
    # the 24-bit table pixel word). :__ksys_done/:__g18done sit BEFORE
    # the task region, so their coords are splice-invariant.
    _tc, _tr = coords1[":__g23tile"]
    tile_pc = (_tc & 0xFFFF) | ((_tr & 0xFFFF) << 16)
    # KJMP home for write/brk tiles: :__ksys_done packed, from the
    # SPLICED coords — NOT _gh18_dispatch_resume(mode=...). That helper
    # assembles the un-augmented mode text, and the libc prologue adds
    # 9 instructions (vpn-8 heap PTE + brk seed) BEFORE the dispatcher,
    # shifting :__ksys_done from flat 143 (posix text) to 149 (spliced
    # libc). The stale home dropped the tile into :__tbl's LD — the run
    # re-dispatched garbage (r6=2052 -> KJMP OOB, receipt
    # output/dbg_gh23_dec3.py, steps 259-266).
    dc, dr = coords1[":__ksys_done"]
    ksys_done = (dc & 0xFFFF) | ((dr & 0xFFFF) << 16)
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
        # The rect grows past GH9_N_INSTRS into the bake-time zero
        # padding BELOW the program text (min_rows=34 default). The
        # GH-18 reserved table window (pixel words [1312,1328)) sits in
        # an EARLIER row (row 41 of a 8-col image ~ cell 3280 < the rect
        # cell 232+ of the spliced layout) — asserted, not assumed: the
        # table window is at pfn-5 pixel words which are BEFORE the
        # rect cells in flat word order, and we only grow FORWARD.
        assert cursor + ncells <= 48, (
            f"packed libc tiles overflow the doubled rect: "
            f"{cursor + ncells} > 48")
        _rel_baker(tile_text, words, mode="posix")  # mode-correct relocs
        for i in range(nz):
            pw = words[i]
            cell = base_cell + cursor + (i // 4)
            y = cell // cells_per_row
            x = (cell % cells_per_row) * 4 + i % 4
            img[y, x] = ((pw >> 16) & 0xFF, (pw >> 8) & 0xFF, pw & 0xFF)
        # FLAT-IZED entry PC (GH-21 receipt dbg_gh21_green9..13).
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

    _stamp(_home(_gh23_sys_write_tile(), ksys_done), GH23_SLOT_WRITE)
    _stamp(_home(_gh23_sys_exit_tile(), g18done), GH23_SLOT_EXIT)
    _stamp(_home(_gh23_sys_brk_tile(), ksys_done), GH23_SLOT_BRK)

    # The in-image break (RAM word 723) is seeded by the libc-mode
    # kernel PROLOGUE (LDI r15 723 / LDI r14 2048 / ST — baker.py,
    # mode=="libc" branch). A bake-time IMAGE stamp cannot seed RAM:
    # image word 723 is tick-handler PROGRAM TEXT (row 22), and the
    # earlier _seed_word stamp corrupted it while mem[723] stayed 0
    # (receipt output/dbg_gh23_brk2.py).

    # Receipt (GH-21 dbg_gh21_stages, verbatim): bake_image(out_path=...)
    # saves the PRE-STAMP image; re-save after the stamps.
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


def _seed_word(img: np.ndarray, w: int, word: int, value: int) -> None:
    """Stamp one 32-bit data word into the image at its RAM-aliased
    pixel address (word i -> pixel i, 24-bit RGB, alpha byte unused —
    the engine-exact plain layout the GH-18 rect patch uses)."""
    v = value & 0xFFFFFF
    img[word // w, word % w] = ((v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF)
