#!/usr/bin/env python3
"""autoatlas.py — GH-12 capstone: the loop-closure piece.

The kernel requests a capability it lacks (routine-name resolution miss);
a local model drafts the glyph assembly; the oracle proves correctness
word-exactly; RoutineAtlas.register() double-gates it; the verified tile
is injected into the resident kernel through the GH-9 mailbox (the GH-11
drive() session plays the loader role) and re-dispatched. The admitted
program's pixels land in the live image, so the tile persists offline —
the same persistence model as GH-8b/GH-9.

FROZEN CONTRACT (2026-09-09, signed off in-session):
  detect:   launcher resolves routine by name -> miss (KeyError path)
  invoke:   escalate(name + contract_spec, expect_registers, input_registers)
            but ONLY if the name's family is on the escalatable whitelist
            — a corrupted mailbox payload name can NEVER trip the model.
  success:  verified text -> RoutineAtlas.register() -> mailbox inject ->
            kernel re-dispatch; result word-exact vs the oracle.
  failure:  E_ATLAS_UNVERIFIED after N attempts (full candidate history in
            the receipt) or E_ATLAS_FAMILY (whitelist, zero model calls).
            HARD-FAIL: no automatic frontier fallback, ever.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from glyph_ir import TERMINATOR_OPS as gi_TERMINATOR_OPS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # tools/

from glyph_gpt.escalate import escalate, EscalationResult          # noqa: E402
from glyph_gpt.oracle import run_oracle                            # noqa: E402
from glyph_gpt.baker import _gh9_kernel_program_text               # noqa: E402
# GH-26.3: module-level re-exports of the GH-18 layout constants so the
# shared admission tail + emit_admit share ONE definition with admit_syscall.
from glyph_gpt.baker import (                                      # noqa: E402
    GH18_TABLE_WORD, GH18_NSLOTS, GH18_RESULT_WORD, GH18_EXIT_A,
    GH18_EXIT_OK, GH18_TILE_WORD, GH9_N_INSTRS,
)
from rv64i_to_glyph import assemble_glyph_to_pixels                # noqa: E402

# ── kernel word map (baker.py constants, mirrored — do not renumber) ──
GH9_ARGV_WORD = 750        # argv0 @750, argv1 @751 (in BOX0)
GH9_ARGV_RESULT = 754      # injected program's result slot
GH9_EXIT_WORD = 703        # injected program's exit word
GH9_MAILBOX_FLAG = 960     # 1 = patch+launch (kernel clears)
GH9_MAILBOX_N_PX = 961     # pixel words to copy into the window
GH9_MAILBOX_DATA = 800     # 96-word payload window [800, 896)
GH9_N_INSTRS = 24          # patch-window capacity (instructions)
GH9_N_PX = GH9_N_INSTRS * 4
GH9_EXIT_OK = 0xFEED0009
KERNEL_STATUS_WORD = 950
KERNEL_OK = 0xCAFE0009

# GH-18 reserved syscall-table pixel window (module-level re-export of
# baker's pinned constant so tests + the admit write path share ONE
# definition; see baker.py layout contract + the reservation gate leg).
from glyph_gpt.baker import GH18_TABLE_PIX_WORD as GH18_TABLE_PIX_WORD  # noqa: E402,F401

# ── escalatable-family whitelist (frozen contract §1) ────────────────
# A miss escalates ONLY for these families. Arithmetic-utility class:
# pure register computation, no memory syscalls, no box writes, HALT.
ESCALATABLE_FAMILIES = {"utility", "arithmetic", "bitops", "ingested_c"}


@dataclass
class IngestResult:
    ok: bool = False
    code: str = "OK"                       # OK | E_ATLAS_UNVERIFIED | E_ATLAS_FAMILY | E_ATLAS_INJECT
    detail: str = ""
    name: str = ""
    family: str = ""
    escalations: int = 0                   # model invocations actually made
    glyph_text: str = ""
    oracle_registers: Optional[Dict[int, int]] = None
    kernel_result: Optional[int] = None
    kernel_exit: Optional[int] = None
    kernel_status: Optional[int] = None
    tile_r2: Optional[int] = None          # GH-20: the admitting task's r2
    table_word: int = 0                    # GH-18/20: 0 until ok, else the tile PC
    history: List[dict] = field(default_factory=list)


def _instr_count(glyph_text: str) -> int:
    """Instructions in the program (labels/comments/blank lines excluded)
    — the honest window-cap check, since the assembled image pads rows."""
    n = 0
    for line in glyph_text.splitlines():
        s = line.split(";", 1)[0].strip()
        if not s:
            continue
        if s.startswith(":"):
            # ':name' bare label = 0 instrs; ':name OP ...' = the op only.
            m = re.match(r":([A-Za-z0-9_]+)\s*(.*)", s)
            if m and m.group(2).strip():
                n += 1
            continue
        n += 1
    return n


def _window_origin() -> Tuple[int, int]:
    """(cell_col, cell_row) of the GH-9 patch window inside the baked
    kernel image — resolved the same way baker's two-pass bake does."""
    txt1 = _gh9_kernel_program_text(950, False)
    _, kcoords = assemble_glyph_to_pixels(txt1, cols_instrs=8, min_rows=16)
    return kcoords[":__g9window"]


_WINDOW_ORIGIN: Optional[Tuple[int, int]] = None


def _relocate_jumps(glyph_text: str, words: List[int],
                    cols_instrs: int = 8) -> None:
    """Patch JMP/JZ/CALL immediates in-place from standalone label coords
    to absolute window coords.

    The assembler resolves labels within the standalone tile image, but on
    dispatch the tile executes at the kernel's :__g9window offset. The
    engine fetches JMP/JZ/CALL targets as (cell_col | cell_row << 16) where
    cell_col is multiplied by INSTR_WIDTH at fetch time (glyph_isa_v2), so
    the absolute packed target is (wcol + tcol) | ((wrow + trow) << 16).
    """
    global _WINDOW_ORIGIN
    if _WINDOW_ORIGIN is None:
        _WINDOW_ORIGIN = _window_origin()
    wcol, wrow = _WINDOW_ORIGIN

    labels = assemble_glyph_to_pixels(glyph_text, cols_instrs=cols_instrs)[1]
    cell = 0
    for raw in glyph_text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line or line.startswith(":"):
            continue
        op = line.split()[0]
        if op in ("JMP", "JZ", "CALL"):
            tcol, trow = labels[line.split()[-1]]
            # Absolute instruction cell in the kernel image, then wrapped
            # per-cell-col: the engine multiplies the col field by
            # INSTR_WIDTH and silently stops if the resulting px lands past
            # the row width, so the col field must stay < cols_instrs.
            abs_cell = (wrow + trow) * cols_instrs + (wcol + tcol)
            imm = (abs_cell % cols_instrs) | ((abs_cell // cols_instrs) << 16)
            base = cell * 4
            words[base + 2] = imm & 0xFFFFFF
            words[base + 3] = (imm >> 24) & 0xFF
        cell += 1


def _pixel_words(glyph_text: str, cols_instrs: int = 8) -> List[int]:
    """Assemble to flat 24-bit pixel words (row-major), jump-relocated to
    the kernel window, sliced to the mailbox window — identical packing to
    the GH-9 loader gate (which flattens the full padded image then takes
    words[:96]). Trailing padding cells are 0x0 and never execute: the
    program HALTs first."""
    pixels, _ = assemble_glyph_to_pixels(glyph_text, cols_instrs=cols_instrs)
    h, w, _ = pixels.shape
    words = [((int(pixels[y, x][0]) << 16) | (int(pixels[y, x][1]) << 8)
              | int(pixels[y, x][2])) for y in range(h) for x in range(w)]
    words = words[:GH9_N_PX]
    _relocate_jumps(glyph_text, words, cols_instrs)
    return words


# ── GH-15 Step 3: the IR gate ─────────────────────────────────────────
# Every tile the ingest pipeline emits must raise into a GlyphIRModule
# and pass the StaticVerifier BEFORE pixel emission — the same dual-path
# contract the RV64I transpiler landed in 3ed16cd, applied to the
# autoatlas lane. Byte-exact: the IR stage only ever REJECTS.
# (GH-15 Step 5: parsing lives in glyph_ir.raise_lines_to_ir — the one
# raiser; this lane keeps only its module-shape policy.)

_GH15_TERMINATORS = gi_TERMINATOR_OPS  # shared single table (glyph_ir)


def _autoatlas_policy(cols_instrs: int = 8):
    import glyph_ir as gi
    return gi.RaisePolicy(
        entry_label="__entry", cont_prefix="__blk",
        strict_operands=True,
        window="gh9", cols_instrs=cols_instrs,
        preserves=("r31",),          # callstack reg survives tile entry
        data_bounds=(0, 749),        # tiles touch argv/box RAM, never the
                                     # mailbox/status/FS ranges above 750
        extra_metadata={"origin": "autoatlas"},
    )


def raise_tile_text_to_ir(glyph_text: str, name: str = "tile",
                          cols_instrs: int = 8) -> "object":
    """Raise glyph assembly text into a GlyphIRModule (shared raiser,
    autoatlas policy: GH-9 window budget, r31 preserved, strict
    operands). Raises StaticVerificationError on malformed text
    (unknown op shape, duplicate label) — before any pixels exist."""
    import glyph_ir as gi
    return gi.raise_lines_to_ir(
        glyph_text.splitlines(), name=name,
        policy=_autoatlas_policy(cols_instrs),
        window_max_words=GH9_N_INSTRS)


def _verify_tile_ir(glyph_text: str, cols_instrs: int = 8) -> None:
    """Dual-path gate: raise to IR + statically verify. Byte-exact
    legacy emission either way — this only ever REJECTS."""
    import glyph_ir as gi
    module = raise_tile_text_to_ir(glyph_text, name="ingest_tile",
                                   cols_instrs=cols_instrs)
    gi.StaticVerifier(module).verify()


def ir_pixel_words(glyph_text: str, cols_instrs: int = 8) -> List[int]:
    """The IR-routed payload: verify at the IR stage, then emit through
    the byte-exact legacy packer. Identical words to _pixel_words on any
    text that passes; a StaticVerificationError before any pixels."""
    _verify_tile_ir(glyph_text, cols_instrs)
    return _pixel_words(glyph_text, cols_instrs)


def ingest(
    atlas,
    name: str,
    family: str,
    contract: str,
    expect_registers: Dict[int, int],
    input_registers: Optional[Dict[int, int]] = None,
    runner=None,
    argv: Optional[Dict[int, int]] = None,
    max_attempts: int = 6,
    model: str = "qwen2.5-coder:14b",
    admission_vectors: Optional[List[Dict[str, Dict[int, int]]]] = None,
    seed_memory: Optional[Dict[int, int]] = None,
) -> IngestResult:
    """Atlas-miss -> whitelist -> escalate -> oracle -> register -> inject.

    atlas   : RoutineAtlas (register() refuses unverified tiles — gate 2)
    runner  : optional GlyphRunner; when given, the verified tile is ALSO
              injected through the GH-9 mailbox and re-dispatched, and the
              kernel's result word is checked word-exact against the oracle
              (gate 3: the kernel runs what the oracle proved).
    admission_vectors : extra oracle proofs required before registration,
              each {"seed_memory": {...}, "expect_registers": {...}}.
              Gate 1.5 (2026-09-08 receipt): a single golden vector can
              admit a DUAL function — an inverted-polarity popcount (counts
              CLEAR bits) passed 0xF0F0F0F0 because 16 set == 16 clear,
              then diverged on every other input. Each vector must pass
              word-exact; a diverging candidate re-escalates with feedback.
    """
    res = IngestResult(name=name, family=family)

    # ── gate 0: whitelist — corrupted names never reach the model ──
    if family not in ESCALATABLE_FAMILIES:
        res.code = "E_ATLAS_FAMILY"
        res.detail = (f"family '{family}' not escalatable "
                      f"(whitelist: {sorted(ESCALATABLE_FAMILIES)})")
        return res

    # ── gate 1+2: local draft -> oracle gate -> atlas registration ──
    # Kernel dispatch (runner given) demands the GH-9 program ABI: inputs
    # arrive in the argv block (mem[750]), result lands at mem[754], exit
    # word 0xFEED0009 at mem[703]. Composed into the contract so the model
    # targets the ABI directly; the oracle still verifies the r-register
    # semantics, gate 3 verifies the ABI stores word-exactly on-die.
    task = contract
    if runner is not None:
        out_reg = f"r{next(iter(expect_registers))}"
        argv_keys = sorted((argv or {0: 0}).keys())
        if len(argv_keys) == 1:
            load_insns = (
                f"arg0 is in memory[{GH9_ARGV_WORD}]. Load it FIRST: 'LDI r15 {GH9_ARGV_WORD}' then "
                "'LD r1 r15' (r1 is this task's input register).\n"
            )
        else:
            loads = [f"arg{w} is in memory[{GH9_ARGV_WORD + w}]. Load with 'LDI r15 {GH9_ARGV_WORD + w}' then 'LD r{w+1} r15'." for w in argv_keys]
            load_insns = "\n".join(loads) + "\n"
        task = (contract + "\nKERNEL ABI (required):\n"
                + load_insns +
                "ALL REGISTERS ARE DIRTY AT ENTRY (the kernel jumps into the "
                "window with leftover state): zero an accumulator/output "
                "with 'XOR rX rX' before accumulating into it. Never assume "
                "a register starts at 0 — this is the #1 cause of "
                "oracle-pass/kernel-fail divergence.\n"
                "After computing, store the result: 'LDI r15 754' then "
                f"'ST r15 {out_reg}'.\n"
                "Then write the exit word: 'LDI r14 4276944905', "
                "'LDI r15 703', 'ST r15 r14'.\n"
                "REGISTER DISCIPLINE: use r14 ONLY for the exit value and "
                "r15 ONLY as the address scratch — never 'LDI r15' twice "
                "in a row (the second LDI destroys the first value before "
                "it is stored).\n"
                "STRAIGHT-LINE FIRST: solve the contract in the fewest "
                "ops (constants via LDI + ALU); only loop if the contract "
                "demands iteration. Solve THE STATED CONTRACT, not the "
                "primer's example.\n"
                "HARD BUDGET: the whole program must be <= 24 instructions "
                "TOTAL — the 7 ABI lines above count, zeroing ops count, "
                "leaving <= 17 for the algorithm. HALT last.\n"
                "LOAD THE INPUT — 'LD r1 r15' reads mem[addr] INTO r1; r1 "
                "is NOT the memory address itself. Never solve the "
                "contract with r1 = 0: if r1 was never LOADED from "
                "mem[750], ADD r2 r1 accumulates nothing (the #1 silent "
                "failure — result stays 0).")
    # Oracle seed: under the kernel ABI the argv block IS the input
    # surface — argv {0: v} maps to mem[750+v]. input_registers (the
    # legacy direct-register path) maps to {750: first} when dispatching
    # through the kernel so both legs see the same vector.
    # runner=None seeds TOO (2026-09-08 gate receipt): the GH-18 admit
    # path calls ingest(runner=None, argv=None) with the tile ABI in the
    # contract — the drafter correctly reads mem[750], but with seed
    # None the oracle ran on EMPTY memory: LD got 0, 3*0 = 0, and every
    # syntactically-perfect candidate died as 'expected r2=0x12, got
    # 0x00000000'. The tile-ABI input word IS 750 in both paths, so the
    # argv->seed map is unconditional.
    seed = ({GH9_ARGV_WORD + w: v for w, v in (argv or {}).items()
             if argv} or None)
    if seed is None and input_registers:
        seed = {GH9_ARGV_WORD: next(iter(input_registers.values()))}
    # GH-20: explicit seed_memory passthrough — an FS tile's input plane
    # is the seeded FSTAB (word-indexed, NOT argv-relative), which the
    # argv->seed map above cannot express. Merged last so it can carry
    # BOTH planes when a caller passes argv too.
    if seed_memory:
        seed = {**(seed or {}), **seed_memory}
    esc = escalate(task, expect_registers=expect_registers,
                   seed_memory=seed,
                   input_registers=(None if runner is not None
                                    else input_registers),
                   max_attempts=max_attempts, model=model,
                   extra_vectors=admission_vectors)
    res.escalations = esc.attempts
    res.history = esc.history
    if not esc.verified or esc.glyph_text is None:
        res.code = "E_ATLAS_UNVERIFIED"
        res.detail = esc.error or "no candidate verified"
        return res
    res.glyph_text = esc.glyph_text
    if esc.oracle is not None and esc.oracle.registers:
        res.oracle_registers = {i: v for i, v
                                in enumerate(esc.oracle.registers)}

    # ── gates 2+3, escalated: IR gate → register → kernel re-dispatch ──
    # A candidate can pass the oracle yet diverge on-die (it depended on
    # register state the oracle seeds but the kernel leaves dirty). That
    # is NOT a dead end: re-escalate with the kernel's OBSERVED result
    # word as feedback so the next draft is kernel-robust, up to the same
    # max_attempts budget. Registered tiles that fail re-dispatch are
    # removed again — a broken tile never survives the loop.
    if runner is not None:
        expect_val = next(iter(expect_registers.values()))
        kernel_feedback = ""
        for _cycle in range(max_attempts):
            # IR gate: verify at the IR stage BEFORE pixels (GH-15 Step 3)
            try:
                _verify_tile_ir(esc.glyph_text)
            except Exception as e:  # StaticVerificationError (loud)
                res.code = "E_ATLAS_INJECT"
                res.detail = f"IR gate: {e}"
                return res
            atlas.register(name, esc.glyph_text, family=family)
            if _instr_count(esc.glyph_text) > GH9_N_INSTRS:
                res.code = "E_ATLAS_INJECT"
                res.detail = (f"tile needs {_instr_count(esc.glyph_text)} "
                              f"instrs, window holds {GH9_N_INSTRS}")
                return res
            words = _pixel_words(esc.glyph_text)
            seeds: Dict[int, int] = {GH9_MAILBOX_FLAG: 1,
                                     GH9_MAILBOX_N_PX: len(words)}
            for i, w in enumerate(words):
                seeds[GH9_MAILBOX_DATA + i] = w
            for w, v in (argv or {}).items():
                seeds[GH9_ARGV_WORD + w] = v
            receipt = runner.drive(seeds=seeds, max_instructions=60000)
            mem = receipt["memory"]      # ground-truth post-run RAM
            res.kernel_result = mem[GH9_ARGV_RESULT]
            res.kernel_exit = mem[GH9_EXIT_WORD]
            res.kernel_status = mem[KERNEL_STATUS_WORD]
            if (not receipt.get("faulted")
                    and res.kernel_result == (expect_val & 0xFFFFFFFF)
                    and res.kernel_exit == GH9_EXIT_OK):
                break                    # both gates green: admit for good
            # one of the on-die legs diverged — roll back and re-escalate
            atlas.tiles.pop(name, None)
            if receipt.get("faulted"):
                kernel_feedback = (
                    "Your program FAULTED the kernel when dispatched "
                    f"on-die: {res.kernel_status:#x}. Rewrite it within "
                    "the ABI below.")
            elif res.kernel_exit != GH9_EXIT_OK:
                kernel_feedback = (
                    "Your program did not write the exit word correctly "
                    f"when dispatched on-die (exit was "
                    f"{res.kernel_exit:#x}). Re-check the ABI stores.")
            else:
                kernel_feedback = (
                    "Your program passed the oracle but produced the "
                    f"WRONG result on the real kernel: it wrote "
                    f"{res.kernel_result:#x} where {expect_val:#x} was "
                    "expected. This almost always means it read a "
                    "register that is DIRTY on the kernel but zeroed in "
                    "the oracle. Zero EVERY register you accumulate or "
                    "branch on with 'XOR rX rX' before its first use.")
            res.history.append({"stage": "kernel", "glyph": esc.glyph_text,
                                "kernel_result": res.kernel_result,
                                "kernel_exit": res.kernel_exit,
                                "kernel_status": res.kernel_status,
                                "error": kernel_feedback})
            esc = escalate(task + f"\n\nIMPORTANT FEEDBACK from the last "
                                   f"on-die run:\n{kernel_feedback}",
                           expect_registers=expect_registers,
                           seed_memory=seed, input_registers=None,
                           max_attempts=max_attempts, model=model)
            res.escalations += esc.attempts
            if not esc.verified or esc.glyph_text is None:
                res.code = "E_ATLAS_UNVERIFIED"
                res.detail = (f"kernel-feedback re-escalation exhausted: "
                              f"{esc.error}")
                return res
            res.glyph_text = esc.glyph_text
            if esc.oracle is not None and esc.oracle.registers:
                res.oracle_registers = {i: v for i, v
                                        in enumerate(esc.oracle.registers)}
        else:
            res.code = "E_ATLAS_INJECT"
            res.detail = (f"still diverges on-die after {max_attempts} "
                          f"re-escalation cycles (last kernel result "
                          f"{res.kernel_result:#x}, expected "
                          f"{expect_val:#x})")
            return res
    res.ok = True
    return res


# ── GH-18: syscall-tile admission (the door ingest() already is) ────────

def _gh18_relocate_jumps(glyph_text: str, words: List[int],
                         mode: str = "baseline") -> None:
    """Relocate JMP/JZ immediates in-place from standalone tile coords to
    absolute tile-rect coords (delegates to baker's implementation).
    mode forwards to baker: the tile-rect row is mode-dependent
    (GH-20 receipt 2026-09-09, probe25 — baseline-coord relocation
    stamped fs_v2 in-tile jumps ~80 cells short)."""
    from glyph_gpt.baker import _gh18_relocate_jumps as _rel
    _rel(glyph_text, words, mode=mode)


def admit_syscall(
    runner,
    sys_n: int,
    contract: str = "triple(r1) -> r2: r2 = 3 * r1",
    argv: Optional[Dict[int, int]] = None,
    expected: int = 18,
    model: str = "qwen2.5-coder:14b",
    *,
    abi: str = "gh18",
):
    """Admit a new syscall tile through ingest() (proof = admission) and
    light its table word in the LIVE image.

    abi="gh18" (default): the GH-18 tile ABI — input @ word 750, result
    @ 754, tile rect destination, KJMP resume (receipts above).

    abi="gh20": the Pixel-FS v2 ABI (same dispatch, different tile):
    the tile reads/writes the FSTAB in the fs pixel window [1024,1280)
    directly (slot 0 = the seeded file 'A'); NO argv staging (the tile
    is a fixed-contract FS op — the drafter's oracle proof pins the
    exact FSTAB mutations word-exactly); the tile rect is stamped with
    the fs-alias layout for any word in [1024,1280) (the tile's LD/ST
    data traffic lands there by engine aliasing), table slot with the
    pfn-5 layout; verification matches the tile's r2 AND the result
    word 754 (both must equal `expected`, or — for refuse-path tiles —
    `expected` may be the clean-fail errno marker).

    Returns an IngestResult augmented with .table_word (0 when rejected:
    the table stays untouched — E_ATLAS_UNVERIFIED never lights a slot).
    """
    from glyph_gpt import baker as B
    from glyph_gpt.baker import (
        GH18_NSLOTS, GH18_TABLE_WORD, GH18_RESULT_WORD,
        GH18_TABLE_PIX_WORD,
        GH18_EXIT_A, GH18_EXIT_OK, GH9_N_INSTRS, _gh18_dispatch_resume,
        _gh18_tile_pc,
    )
    # NOTE the tile PC comes from _gh18_tile_pc() (deterministic
    # recompute), NOT from the GH18_TILE_PC global: baker.py loads as
    # two module objects (glyph_gpt.baker here vs tools.glyph_gpt.baker
    # in the tests), and the bake stamps the global only on the caller's
    # object — a global read here sees the stale pre-bake 0 and seeds
    # KJMP-to-0 (receipt: probe 22, debug_gh18_probe2*.py).
    kernel_status_word = KERNEL_STATUS_WORD  # 950, defined above in this module

    if sys_n < 6 or sys_n >= 6 + GH18_NSLOTS:
        raise ValueError(f"sys_n {sys_n} outside the table range "
                         f"[6, {6 + GH18_NSLOTS})")

    class _RectAtlas:
        """Atlas view whose tiles live in the GH-18 tile rect."""

        def __init__(self) -> None:
            self.tiles: dict = {}

    # Tile ABI in the CONTRACT text (2026-09-08 gate receipt): admit
    # passes runner=None into ingest(), so ingest's kernel-ABI preamble
    # never fires and the drafter saw only 'triple(r1) -> r2' — it has
    # no way to know the tile reads its input from mem[750] (nothing
    # preloads r1), so every candidate accumulated r1=0 -> result 0 ->
    # 6/6 E_ATLAS_UNVERIFIED ('expected r2=0x12, got 0'). The tile ABI
    # is: input @750, result @754, no exit word (admit_syscall rewrites
    # HALT into the KJMP home itself).
    # NOTE the concrete register, not '<out-reg>': with the literal
    # placeholder in the prompt the drafter derails (A/B receipt: same
    # prompt with 'ST r15 r2' yields a correct x3; with 'ST r15
    # <out-reg>' it drafts an x2 loop).
    out_reg = "r2"  # tile result reg (ABI-fixed r2, matches expect {2: N})
    # GH-20 ABI AUTO-ROUTE (receipt 2026-09-09, output/debug_gh20_probe52.py):
    # the committed RED gate (4e2f25b) calls admit_syscall WITHOUT abi= —
    # defaulting to gh18 primed qwen with the 3*argv multiply recipe and
    # every FS candidate computed 3*65=0xc3 (E_ATLAS_UNVERIFIED 6/6).
    # FS table slots [10..12] ARE the Pixel-FS v2 ops, so sys_n selects
    # the ABI; an explicit abi= kwarg still wins for direct callers.
    if abi == "gh18" and 10 <= sys_n <= 12:
        abi = "gh20"
    if abi == "gh20":
        # ── Pixel-FS v2 tile contract (GH-20 gate, ticket
        # gh20-spec-finalize): the tile reads/writes the FSTAB in the fs
        # pixel window [1024,1280) DIRECTLY (engine aliasing: 2 px/word,
        # lo24 + hi8-BLUE — glyph_isa_v2._fs_pix_write). No argv staging:
        # a fixed-contract FS op (the drafter's oracle proof pins the
        # exact FSTAB mutations word-exactly). The gate's admit calls
        # pass expected=0 (ok path) or the errno marker (refuse path) —
        # the same value lands in r2 AND the result word 754.
        # WARNING honored (ticket): NO MUL (this ISA: ADD/SUB/AND/OR/XOR/
        # SHL/SHR); decimalize hex constants carefully; CMP against r0
        # tests the FLAG register — XOR r4 r4 first.
        # GH-20 receipt (2026-09-09, probes 33-45): the original fs-tile
        # contract told qwen WHAT to do ("grow extent by staged words")
        # but not HOW the verdict channel works. qwen then drafted
        # argv-pass-through tiles (LD [750] / 3*x / ST 754) — the shape
        # the GH-18 MULTIPLY recipe primed — and 6/6 attempts died as
        # 'expected r2=0, got 0xc3'. Fix (probe 45, passes on attempt 2
        # with real qwen): name the CONCRETE recipe per op class —
        # branch-on-name against the slot-0 name field @1024, mutate the
        # FSTAB field, verdict into r2 AND 754 — plus the argv load line.
        # RECEIPT (2026-09-09, leg-2 divergence): the generic recipe says
        # "e.g. word 1026 = len", and qwen drafted EXACTLY that recipe
        # for fs_rename — mutating len and IGNORING the new name (r2=0 at
        # step 334, mem[1024] still 65 on-die). The rename recipe is
        # explicit: copy mem[751] into the slot-0 name word.
        # RECEIPT (2026-09-09, probe62): a substring match on the contract
        # text is WRONG for the structural legs, which pass contract=
        # "fs op tile" for all three ops — every op then got the append
        # recipe and the unlink oracle demanded r2=69 from a len+=1 tile
        # (0 != 69, E_ATLAS_UNVERIFIED 6/6). Route on SYS NUMBER: 10 =
        # append, 11 = rename, 12 = unlink (FSV2_N_* constants).
        if abi == "gh20" and sys_n == 11:
            op_recipe = (
                "\n:match"
                "\n  LDI r15 751        ; argv1 = the NEW name"
                "\n  LD r5 r15"
                "\n  LDI r15 1024       ; slot-0 name word — update IN PLACE"
                "\n  ST r15 r5"
                "\n  XOR r2 r2          ; verdict 0 = ok"
            )
        elif abi == "gh20" and sys_n == 12:
            op_recipe = (
                "\n:match"
                "\n  LDI r15 1028       ; slot-0 refcount word"
                "\n  LD r5 r15"
                "\n  XOR r6 r6"
                "\n  CMP r5 r6          ; flag = (refcount == 0)"
                "\n  JZ :ok             ; refcount 0 -> unlink would be safe"
                "\n  LDI r2 69          ; refuse: errno 'E', FSTAB untouched"
                "\n  JMP :done"
                "\n:ok"
                "\n  XOR r2 r2"
            )
        else:
            # fs_append (and any other FS op): grow slot 0's extent by one
            # word — len += 1 (the data plane test word lands at 1040+len).
            op_recipe = (
                "\n:match"
                "\n  LDI r15 1026       ; slot-0 len word"
                "\n  LD r5 r15"
                "\n  LDI r7 1"
                "\n  ADD r5 r7"
                "\n  ST r15 r5          ; len grows; slot 1 + data pixels beyond stay untouched"
                "\n  XOR r2 r2          ; verdict 0 = ok"
            )
        tile_contract = (
            contract + "\nFS-TILE ABI (required): argv0 is in memory[750]: "
            "'LDI r15 750' then 'LD r1 r15' (r1 starts at 0 — loading is "
            "the ONLY way it gets the name). The FSTAB lives in the fs "
            "pixel window, slot 0 base word 1024, layout [name,start,len,"
            "in_use,refcount,parent_dir,next_chain,reserved], 8 words per "
            "slot — access its fields DIRECTLY with LDI-immediate "
            "addresses (e.g. 'LDI r15 1024' / 'LD r3 r15' reads the name)."
            "\nRECIPE (branch-on-name, then mutate the field, then verdict):"
            "\n  LDI r3 65           ; slot-0 name is 'A' = 65"
            "\n  XOR r4 r4           ; r0 is the CMP FLAG register — never "
            "CMP against r0"
            "\n  CMP r1 r3           ; flag = (argv0 == name)"
            "\n  JZ :match           ; fallthrough = mismatch"
            "\n  LDI r2 69           ; errno 'E' on mismatch"
            "\n  JMP :done"
            + op_recipe +
            "\n:done"
            "\n  LDI r15 754"
            "\n  ST r15 r2           ; the verdict lands in r2 AND memory[754]"
            "\n  HALT last. NO loops — straight-line only. Emit the RECIPE "
            "VERBATIM (adjust only where this Task's contract explicitly "
            "says so)."
        )
    else:
        tile_contract = (
            contract + "\nTILE ABI (required): the input is in memory[750]. "
            "Load it FIRST: 'LDI r15 750' then 'LD r1 r15' — r1 starts at 0, "
            "reading memory is the ONLY way it gets the argument. Store the "
            f"result: 'LDI r15 754' then 'ST r15 {out_reg}'. No exit word. "
            "HALT last."
            # 2026-09-08 gate receipt (9-fail -> 2-fail residue): the ISA has
            # no MUL, and the drafter emitted 'LDI r3 3' + two ADDs — a dead
            # constant plus 2*x — which is syntactically VALID, so no
            # escalate KeyError hint ever fired and temperature-0 decoding
            # repeated the identical wrong program 6/6 (E_ATLAS_UNVERIFIED,
            # 'expected r2=0x12, got 0x0000000c'). The recipe goes in the
            # contract, same mechanism as the ABI lines above.
            + "\nMULTIPLY: this ISA has NO MUL opcode. 'r_out = 3 * r_in' with "
            "a CONSTANT is straight-line shift-add — emit EXACTLY:\n"
            "  XOR r2 r2\n  ADD r2 r1\n  LDI r13 1\n  SHL r2 r13\n  ADD r2 r1\n"
            "(two ADDs contribute two copies of r1; the SHL doubles the first "
            "copy -> 3*x total). A constant you LDI but never consume in an "
            "ALU/SHL op is dead code and computes the wrong value."
        )

    rect_atlas = _RectAtlas()
    if abi == "gh20":
        # FS tiles take no argv: input surface IS the FSTAB (seeded
        # in-image by pixel_fs_v2_kernel_image). argv=None AND the
        # ingest-side seed derivation (argv or {0:0}) suppressed — an
        # implicit mem[750]=0 seed would let a bogus candidate "verify"
        # against the wrong plane (receipt: the {2: 0} oracle proves the
        # FSTAB plane, not the argv plane).
        fs_argv = None
        fs_input_registers = None
    else:
        fs_argv = (argv if argv is not None else {0: 0})
        fs_input_registers = None
    # GH-20 (abi="gh20"): the tile's input plane is the FSTAB, not argv —
    # pass the seeded FSTAB as the oracle's seed_memory so the drafter's
    # oracle run sees the SAME bytes the on-die tile will (2026-09-09
    # receipt: with argv=None the ingest-side seed derivation collapses to
    # None and the oracle verified fs candidates against EMPTY RAM).
    fs_seed_memory = None
    if abi == "gh20":
        from glyph_gpt.fs_v2 import _gh20_seed_fstab
        fs_seed_memory = dict(_gh20_seed_fstab())
        if argv:
            for w, v in argv.items():
                fs_seed_memory[GH9_ARGV_WORD + w] = v & 0xFFFFFFFF
    # SINGLE ingest call (receipt 2026-09-09: an earlier draft called
    # ingest() twice — the first result was immediately discarded when
    # the second call rebound `res`, doubling the Ollama drafting cost
    # per admission and racing the atlas store for no effect).
    res = ingest(
        rect_atlas, f"syscall_{sys_n}", "utility", tile_contract,
        {2: expected},
        runner=None,                      # GH-9 window ABI does NOT apply:
        argv=fs_argv,                     # the GH-18 tile ABI is word-750
                                          # in / word-754 out; argv seeds the
                                          # ORACLE memory so the drafter's
                                          # LD r1 r15 reads the real argument
                                          # (was None: oracle ran on empty
                                          # RAM, every candidate computed
                                          # 3*0 — receipt above)
        input_registers=fs_input_registers,
        seed_memory=fs_seed_memory,
        model=model,
    )
    res.table_word = 0
    if not res.ok:
        return res
    # verified candidate (ingest set res.glyph_text) — the shared
    # stamp-and-verify tail does the rest (GH-26.3 refactor: the SAME
    # admission machinery now serves the agent emit_admit path; the
    # oracle remains the sole admission path, admit_syscall keeps its
    # drafter loop upstream of it).
    return _admit_verified_tile(runner, res.glyph_text, sys_n, expected,
                                abi, res=res)


def emit_admit(
    runner,
    tile_text: str,
    sys_n: int,
    expected: int,
    argv: Optional[Dict[int, int]] = None,
    source: str = "template",
    receipt_path: Optional[Path] = None,
    model: str = "qwen2.5-coder:14b",
    *,
    abi: str = "gh18",
):
    """GH-26.3 EMIT→ADMIT: an agent-drafted tile proposal routes ONLY
    through the GH-12/18 admission pipeline — IR StaticVerifier → oracle
    word-exact → table stamp. The agent has NO other registration route;
    a rejected candidate returns the oracle error and NOTHING partial
    lands (table word stays 0, image bytes untouched).

    Drafting discipline (S3 SAFE-ONLY verdict, 878330d): candidates come
    from the ABI-baked template/grammar library; GlyphGPT best-of-N is at
    most ONE drafting source (this function never calls it — the caller
    passes the drafted text; `source` records provenance in the receipt).

    Every attempt (pass or reject) appends one JSON receipt line to
    `receipt_path` (default output/agent_admissions.jsonl):
      {ts, intent, source, oracle_verdict, reason}
    Failed attempts are receipts too — they prove the oracle filters.

    sys_n bounds match admit_syscall; FS auto-route (slots 10..12) is
    NOT applied here — agent proposals declare their ABI explicitly.
    """
    import datetime as _dt
    path = Path(receipt_path) if receipt_path else (
        Path(__file__).resolve().parent.parent.parent
        / "output" / "agent_admissions.jsonl")

    def _receipt(verdict: str, reason: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "ts": _dt.datetime.now().astimezone().isoformat(),
                    "intent": "emit_admit",
                    "source": source,
                    "oracle_verdict": verdict,
                    "reason": reason,
                }) + "\n")
        except OSError:
            pass  # receipt failure never blocks the admission verdict

    res = IngestResult(name=f"syscall_{sys_n}", family="utility")
    if sys_n < 6 or sys_n >= 6 + GH18_NSLOTS:
        res.code = "E_EMIT_SYSN"
        res.detail = f"sys_n {sys_n} outside the table range [6, {6 + GH18_NSLOTS})"
        _receipt("rejected", res.detail)
        return res

    # ── gate A: IR StaticVerifier (BEFORE the oracle — a malformed
    # proposal dies here, image untouched) ──
    try:
        _verify_tile_ir(tile_text)
    except Exception as e:  # StaticVerificationError (loud)
        res.code = "E_EMIT_IR"
        res.detail = f"IR gate: {e}"
        _receipt("rejected", res.detail[:500])
        return res

    # ── gate B: oracle word-exact (the SOLE admission proof) —
    # the same argv->seed map admit_syscall's ingest call uses, so the
    # oracle sees the same mem[750] the on-die tile will ──
    seed = ({GH9_ARGV_WORD + w: v for w, v in (argv or {}).items()
             if argv} or None)
    ores = run_oracle(tile_text, expect_registers={2: expected},
                      seed_memory=seed)
    if not ores.passed:
        res.code = "E_EMIT_ORACLE"
        res.detail = ores.error or "oracle word-exact check failed"
        _receipt("rejected", res.detail[:500])
        return res
    res.oracle_registers = {i: v for i, v in enumerate(ores.registers or [])}

    # ── gate C: the shared GH-12/18 stamp-and-verify tail (identical to
    # admit_syscall's: IR payload words, tile-rect stamp, table stamp,
    # on-die re-dispatch, table rollback on divergence) ──
    return _admit_verified_tile(runner, tile_text, sys_n, expected,
                                abi, res=res,
                                on_verdict=_receipt)


def _admit_verified_tile(runner, tile_text: str, sys_n: int, expected: int,
                         abi: str = "gh18", res=None, on_verdict=None):
    """Shared post-oracle admission tail (GH-18/20): retarget the verified
    tile's HALT into the mode-correct KJMP home, stamp the tile rect +
    table word in the LIVE image (the ONLY two image mutations), then
    re-dispatch on-die and verify word-exactly. A diverging tile has its
    table slot rolled back — a broken tile never keeps its slot."""
    from glyph_gpt import baker as B
    from glyph_gpt.baker import (
        GH18_RESULT_WORD, GH18_EXIT_A, GH18_EXIT_OK, GH9_N_INSTRS,
        _gh18_dispatch_resume, _gh18_tile_pc,
    )
    if res is None:
        res = IngestResult(name=f"syscall_{sys_n}", family="utility")
    kernel_status_word = KERNEL_STATUS_WORD  # 950, defined above in this module
    tile_text = res.glyph_text = tile_text
    # re-target the tile: result already @754 via the contract; append
    # the tile-exit KJMP to the dispatcher's resume point.
    # mode="admit": the resume PC is mode-dependent (the admit-mode
    # prologue shifts :__ksys_done — receipt in baker._gh18_dispatch_resume,
    # debug_gh18_decode3: default/baseline resume 0x0d0005 vs the real
    # admit-mode home 0x110000 -> verify drive saw 754 clobbered to 6).
    # GH-20: img_mode ("fs_v2" vs "admit") — see the receipt above.
    # GH-20: the resume PC and tile PC are MODE-DEPENDENT — the fs_v2
    # task leg is longer than admit's, which shifts :__g18tile to
    # 0x220005 (vs admit's 0x1e0007; receipt 2026-09-09, probe23).
    # Stamping admit's coords into an fs_v2 image wrote tile words over
    # live kernel text and lit the table with a PC pointing at row 30
    # (the prologue's PTE immediates) — on-die dispatch diverged.
    img_mode = "fs_v2" if abi == "gh20" else "admit"
    resume = _gh18_dispatch_resume(mode=img_mode)
    tile_text = tile_text.replace("HALT", f"LDI r30 {resume}\nKJMP r30")
    words = ir_pixel_words(tile_text)
    if _instr_count(tile_text) > GH9_N_INSTRS:
        res.ok = False
        res.code = "E_ATLAS_INJECT"
        res.detail = (f"tile needs {_instr_count(tile_text)} instrs, "
                      f"tile rect holds {GH9_N_INSTRS}")
        return res
    _gh18_relocate_jumps(tile_text, words, mode=img_mode)

    # ── IMAGE-MUTATION RULE (steering contract, oversight 2026-09-08) ──
    # The admitted tile must persist IN THE IMAGE, not in per-drive RAM:
    # every drive() constructs a FRESH GlyphCPUv2 (runner.get_cpu()), so
    # seeds are volatile — gate receipt (gh18_gate_run2): admit verified
    # with seeds, then drive(seeds={}) saw 754=0 because the tile pixels
    # and table word died with the first drive's CPU. Layout is the
    # engine's GH-8b pixel aliasing (glyph_isa_v2._fs_pix_write): word i
    # (i in [1024,1280)) lives at pixel (i*2) = lo24 and pixel (i*2+1) =
    # hi8 in BLUE. The tile rect [1600,1696) is OUTSIDE the fs window, so
    # its words use the plain pixel layout: word i -> pixel i (24-bit RGB,
    # alpha byte unused — exactly how the bake stamps the rect).
    # These two writes (table word ∪ tile rect) are the ONLY image
    # mutations — patch isolation's positive whitelist.
    tile_word = B.GH18_TILE_WORD
    tile_pc = _gh18_tile_pc(mode=img_mode)
    table_word = GH18_TABLE_WORD + (sys_n - 6)
    img = runner.image
    h, w, _ = img.shape
    assert w % 4 == 0, f"image width {w} not instruction-aligned"
    cells_per_row = w // 4
    tcol_px = tile_pc & 0xFFFF          # instr-cell col of the tile rect
    trow_px = (tile_pc >> 16) & 0xFFFF  # instr-cell row

    def _pix_write_word(word: int, value: int) -> None:
        """Engine-exact word->pixels write (see _fs_pix_write / bake)."""
        value &= 0xFFFFFFFF
        lo, hi = value & 0xFFFFFF, (value >> 24) & 0xFF
        if 1024 <= word < 1280:
            # GH-8b fs-alias window: 2 pixels per word, hi pixel carries
            # only the BLUE byte (glyph_isa_v2._fs_pix_write is the engine
            # truth; the window is [1024,1280) ONLY — the table is NOT in
            # it, see the branch below). GH-20: the admit path uses this
            # branch BOTH for the tile's payload words (an FS tile's LD/
            # ST data traffic lands in the window by engine aliasing, so
            # the stamped payload must use the same layout) and for the
            # fs-writes the verification replays.
            lin, lin2 = word * 2, word * 2 + 1
            img[lin // w, lin % w] = ((lo >> 16) & 0xFF, (lo >> 8) & 0xFF, lo & 0xFF)
            img[lin2 // w, lin2 % w] = (0, 0, hi)
        elif GH18_TABLE_WORD <= word < GH18_TABLE_WORD + 16:
            # syscall table words [1568,1584): the admit-mode kernel arms
            # a PTE_PIX page map for vpn 6 in-image (see the admit-mode
            # prologue in baker), so the dispatcher's LD walks to IMAGE
            # PIXELS: pix_word = GH18_TABLE_PIX_WORD + (word & 0xFF),
            # pinned in baker as RESERVED for the table (see the layout
            # contract there). Receipts (probes
            # debug_gh18_tblpath10..33, 2026-09-08): (a) RAM writes die
            # with each drive()'s fresh CPU — the e2e leg's drive(seeds={})
            # saw 754=0; (b) identity pfn=6 WRAPS: pix_word = pfn*256 +
            # (word & 0xFF) with 1570 & 0xFF = 34 -> pixel word 1570 ->
            # _addr_to_xy modulo w*h -> pixel (0,2) = :__entry's JMP
            # immediate — clobbering the entry jump. pfn=5 lands in the
            # reserved zero window [1312,1328): pinned base + SLOT offset
            # (word - GH18_TABLE_WORD), NOT (word & 0xFF) — the window
            # starts at 1280+32, so adding the full low byte would skew
            # every non-zero slot (receipt: gate e2e E_ATLAS_INJECT,
            # 2026-09-08 reservation session — 1312+34=1346 vs 1314).
            pw = GH18_TABLE_PIX_WORD + ((word - GH18_TABLE_WORD) & 0xFF)
            img[pw // w, pw % w] = ((lo >> 16) & 0xFF, (lo >> 8) & 0xFF, lo & 0xFF)
        elif tile_word <= word < tile_word + GH9_N_INSTRS * 4:
            # Tile rect: 1 word = 1 pixel, EXACTLY the bake's layout
            # (baker.syscall_abi_kernel_image, rect patch after pass 2):
            # word i -> cell trow,tcol + i//4, pixel (cell*4 + i%4),
            # RGB = the word's 24 bits. Receipt (2026-09-08,
            # debug_gh18_rectdiff.py): the earlier draft here split the
            # word across 4 grayscale pixels and advanced the CELL per
            # word — the rect then held layout-garbage (baked words
            # 0xecff00.. valid opcodes vs stamped 0xec5050.. junk), the
            # dispatcher jumped in and computed 0 -> E_ATLAS_INJECT on
            # the e2e leg while the bake-provided tile passed.
            i = word - tile_word
            cell = trow_px * cells_per_row + tcol_px + i // 4
            y = cell // cells_per_row
            x = (cell % cells_per_row) * 4 + i % 4
            img[y, x] = (lo >> 16, (lo >> 8) & 0xFF, lo & 0xFF)
        else:
            raise ValueError(
                f"word {word} outside both the fs window [1024,1280) and "
                f"the tile rect [{tile_word},{tile_word + GH9_N_INSTRS * 4}) "
                "— refusing to stamp unknown words into the image")

    for i, pw in enumerate(words[:GH9_N_INSTRS * 4]):
        _pix_write_word(tile_word + i, pw)
    # FS tiles: the drafter's tile text RUNS with window addresses (its
    # LD/STs hit [1024,1280) by engine aliasing) but the assembled WORDS
    # themselves live in the tile rect at plain 1-word-per-pixel layout
    # (receipt: debug_gh18_rectdiff — the rect bake layout, not the fs
    # alias). Both ABIs stamp only the rect + table slot; an FS tile
    # mutates the FSTAB on-die.
    _pix_write_word(table_word, tile_pc)

    receipt = runner.drive(seeds={}, max_instructions=60000)
    mem = receipt["memory"]
    res.kernel_result = mem[GH18_RESULT_WORD]
    if abi == "gh20":
        # GH-20 verification: the result word 754 is the verdict channel.
        # The OLD draft ALSO required the admitting task's r2
        # (registers_full[2]) to equal expected — that can never hold:
        # SYSRET restores the PRE-TRAP register file and :__ksys_done
        # zeroes SYS_A0 before returning, so the task receives a0=0; its
        # r2 at HALT is whatever the task itself last computed (the
        # fs_v2 task leg fetches 754 into r2 so a LEG can assert it, but
        # it is task-side state, not the tile's return value — and
        # multi-op drives leave the LAST op's verdict there regardless
        # of which tile is being verified). Integrity set: result word
        # matches, no fault, clean kernel tail, and task A's exit word
        # is the full 0xFEED0000|6 (the drive leg ran to completion).
        res.tile_r2 = receipt["registers_full"][2]
        res.kernel_status = mem[kernel_status_word]
        if (not receipt.get("faulted")
                and mem[GH18_RESULT_WORD] == (expected & 0xFFFFFFFF)
                and mem[GH18_EXIT_A] == GH18_EXIT_OK
                and res.kernel_status == (0xCAFE0000 | 20)):
            res.ok = True
            res.table_word = tile_pc
            if on_verdict:
                on_verdict("admitted", f"tile live at slot {sys_n}, "
                           f"table_word={tile_pc:#x}")
        else:
            res.ok = False
            res.code = "E_ATLAS_INJECT"
            res.detail = (f"on-die dispatch diverged: result word "
                          f"{mem[GH18_RESULT_WORD]} != {expected}, "
                          f"exit_a {mem[GH18_EXIT_A]:#x}, "
                          f"status {res.kernel_status:#x}")
            # roll the table back — a broken tile never keeps its slot
            _pix_write_word(table_word, 0)
            if on_verdict:
                on_verdict("rejected", res.detail[:500])
        return res
    res.kernel_exit = mem[GH18_EXIT_A]
    res.kernel_status = mem[kernel_status_word]
    if (not receipt.get("faulted")
            and res.kernel_result == (expected & 0xFFFFFFFF)
            and res.kernel_exit == GH18_EXIT_OK):
        res.ok = True
        res.table_word = tile_pc
        if on_verdict:
            on_verdict("admitted", f"tile live at slot {sys_n}, "
                       f"table_word={tile_pc:#x}")
    else:
        res.ok = False
        res.code = "E_ATLAS_INJECT"
        res.detail = (f"on-die dispatch diverged: result {res.kernel_result}"
                      f" != {expected}")
        # roll the table back — a broken tile never keeps its slot
        _pix_write_word(table_word, 0)
        if on_verdict:
            on_verdict("rejected", res.detail[:500])
    return res
