#!/usr/bin/env python3
"""
oracle.py — execution-verification gate for generated/transpiled Glyph programs.

GH-12 substrate (AGENTS.md constitution: oracle-grounded verification).

Contract (proven live in tests/test_rv64i_to_glyph_stringc.py:347-372):
    glyph_text → assemble_glyph_to_pixels → GlyphCPUv2.step loop → HALT
    → register/memory expectations checked word-exactly.

This module is the hard gate in the escalation loop:
    local LLM drafts .glyph  →  run_oracle()  →  PASS: admit to atlas
                                              →  FAIL: discard, exact error
    Failed candidates NEVER touch the atlas; no frontier tokens spent.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent.parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from glyph_isa_v2 import GlyphCPUv2, OpcodeMapV2  # noqa: E402
from rv64i_to_glyph import assemble_glyph_to_pixels  # noqa: E402


@dataclass
class OracleResult:
    passed: bool
    steps: int = 0
    registers: Optional[List[int]] = None
    memory_hash: Optional[str] = None     # sha256 of pixel buffer at HALT (VCC contract)
    error: Optional[str] = None
    detail: Dict = field(default_factory=dict)


def run_oracle(glyph_text: str, cols_instrs: int = 64,
               max_instructions: int = 100_000,
               expect_registers: Optional[Dict[int, int]] = None,
               expect_memory_hash: Optional[str] = None,
               seed_memory: Optional[Dict[int, int]] = None,
               input_registers: Optional[Dict[int, int]] = None) -> OracleResult:
    """Assemble + execute glyph_text in GlyphCPUv2; check expectations.

    expect_registers: {reg_index: expected_value} — all must match at HALT.
    expect_memory_hash: sha256 of the final pixel buffer.
    seed_memory: {word_index: value} preloaded into RAM before step 0
                 (argv blocks, data sections — same pattern as the
                 stringc differential fixture's .data seeding).

    Returns OracleResult; passed=False with error set on assemble faults
    (SpatialMisalignmentFault), non-halt timeout, or expectation mismatch.
    Mismatch errors carry the exact reg/golden/actual triple so the
    drafting model can self-correct in one retry.
    """
    try:
        pixels, _labels = assemble_glyph_to_pixels(
            glyph_text, cols_instrs=cols_instrs, min_rows=16)
    except Exception as e:  # SpatialMisalignmentFault + decode errors
        return OracleResult(passed=False, error=f"assemble: {type(e).__name__}: {e}")

    # GH-20: fs_pix_enabled — FS-window LD/ST (words [1024,1280)) must
    # alias image pixels EXACTLY as the on-die engine does (the kernel's
    # vpn-4 PTE maps the window to pixels; without this flag the oracle's
    # plain-RAM reads disagreed with the die and every FS candidate
    # verified against the wrong plane, or failed against the right one —
    # receipt 2026-09-09, gate run 1: r2=0xc3 vs the expected 0).
    cpu = GlyphCPUv2(OpcodeMapV2(), cols_instrs=cols_instrs, fs_pix_enabled=True)
    cpu.memory = [0] * 16384
    if seed_memory:
        for w, v in seed_memory.items():
            v = int(v) & 0xFFFFFFFF
            cpu.memory[w] = v
            # GH-20: mirror fs-window seeds into the pixel plane — the
            # image IS the disk; RAM alone is invisible to a pix-mapped
            # window read.
            if 1024 <= w < 1280:
                lo, hi = v & 0xFFFFFF, (v >> 24) & 0xFF
                h, wd, _ = pixels.shape
                lin = (w * 2) % (wd * h)
                pixels[lin // wd, lin % wd] = ((lo >> 16) & 0xFF, (lo >> 8) & 0xFF, lo & 0xFF)
                lin2 = (w * 2 + 1) % (wd * h)
                pixels[lin2 // wd, lin2 % wd] = (0, 0, hi)
    # caller ABI: inputs arrive IN registers (e.g. popcount(r1) -> r2)
    if input_registers:
        for i, v in input_registers.items():
            cpu.registers[i] = int(v) & 0xFFFFFFFF
    cpu.pc = (0, 0)
    cpu.running = True
    steps = 0
    for steps in range(1, max_instructions + 1):
        if not cpu.step(pixels):
            break
    else:
        return OracleResult(
            passed=False, steps=steps, registers=list(cpu.registers),
            error=f"no-halt: {max_instructions} steps without HALT")

    regs = [int(r) & 0xFFFFFFFF for r in cpu.registers]
    mem_hash = hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()
    result = OracleResult(passed=True, steps=steps, registers=regs,
                          memory_hash=mem_hash,
                          detail={"pc": tuple(cpu.pc)})

    for idx, want in (expect_registers or {}).items():
        got = regs[idx]
        if got != (want & 0xFFFFFFFF):
            result.passed = False
            result.error = (f"contract: expected r{idx}=0x{want & 0xFFFFFFFF:08x}, "
                            f"got 0x{got:08x} at step {steps}")
            break
    if result.passed and expect_memory_hash and mem_hash != expect_memory_hash:
        result.passed = False
        result.error = f"contract: memory hash {mem_hash[:16]}… != {expect_memory_hash[:16]}…"
    return result


def check_generation(text: str, tokenizer, model, max_new_tokens: int = 512,
                     **oracle_kwargs) -> OracleResult:
    """generate.py helper: constrained-decode text from model then run_oracle.

    Ties GlyphGPT's generate loop to the hard gate: decode → strip fences →
    oracle. oracle_kwargs forwards expect_registers / seed_memory.
    """
    ids = tokenizer(text, return_tensors="np").input_ids
    out_ids = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    body = tokenizer.decode(out_ids[0][ids.shape[1]:], skip_special_tokens=True)
    # tolerate markdown fences from chat-style decoders
    if "```" in body:
        chunks = [c.split("```")[0] for c in body.split("```glyph") if "```" in c or c]
        body = "\n".join(chunks).replace("```", "")
    return run_oracle(body, **oracle_kwargs)
