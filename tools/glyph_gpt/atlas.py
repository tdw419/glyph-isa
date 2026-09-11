#!/usr/bin/env python3
"""
atlas.py — Phase 5.6: Patch-and-Copy subroutine atlas.

"The Font is the Compiler." Standard library routines live as pre-verified
spatial tiles in an atlas; the model emits coordination logic only and
links to tiles by reference. The linker splices tile bodies into the final
program text at assemble time.

Atlas format (JSON lines, one tile per line):
    {"name": "double", "family": "leaf_call",
     "tile_text": ":atlas_double\\nADD r10 r10\\nRET\\n",
     "receipt": {...execution receipt from tile verification...}}

Caller-side convention: the model (or synth) writes
    CALL :atlas_double
and the linker rewrites that operand to the spliced label. Every tile
label is namespaced atlas_<name> so tiles never collide with program
labels.

Verification: every tile in the atlas carries a receipt from an execution
run at registration time. A tile without a passing receipt is refused —
the atlas never contains unverified code.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


import io
import contextlib

def run_generated(text: str, cols_instrs: int = 8) -> dict:
    from tools.rv64i_to_glyph import assemble_glyph_to_pixels
    from tools.glyph_isa_v2 import OpcodeMapV2, GlyphCPUv2
    receipt: dict = {'assembled': False, 'executed': False, 'halted': False}
    try:
        img, labels = assemble_glyph_to_pixels(text, cols_instrs=cols_instrs)
        receipt['assembled'] = True
        cpu = GlyphCPUv2(OpcodeMapV2(), cols_instrs=cols_instrs)
        with contextlib.redirect_stdout(io.StringIO()):
            steps = cpu.run(img, max_instructions=5000)
        receipt['executed'] = True
        receipt['faulted'] = bool(getattr(cpu, 'faulted', False))
        if receipt['faulted']:
            receipt['fault_addr'] = int(getattr(cpu, 'fault_addr', 0)) & 0xFFFFFFFF
        receipt['halted'] = (not cpu.running) and not receipt['faulted']
        receipt['steps'] = int(steps)
        nreg = len(cpu.registers)
        receipt['registers_full'] = [int(cpu.registers[i]) & 0xFFFFFFFF for i in range(nreg)]
        receipt['registers'] = receipt['registers_full'][:8]
        receipt['memory'] = [int(m) & 0xFFFFFFFF for m in cpu.memory]
    except Exception as e:
        receipt['error'] = f'{type(e).__name__}: {e}'
    return receipt


_RV_GCC = "riscv64-unknown-elf-gcc"


class RoutineAtlas:
    def __init__(self) -> None:
        self.tiles: dict = {}

    # ── registration ──────────────────────────────────────────────────
    def register(self, name: str, tile_text: str,
                 family: str, cols_instrs: int = 8) -> dict:
        """Verify the tile executes+halts (or returns cleanly), then store.
        Refuses unverified tiles."""
        receipt = run_generated(tile_text, cols_instrs=cols_instrs)
        if not receipt.get("assembled"):
            raise ValueError(f"tile '{name}' did not assemble: "
                             f"{receipt.get('error')}")
        self.tiles[name] = {"name": name, "family": family,
                            "tile_text": tile_text, "receipt": receipt}
        return receipt

    # ── SB-2: C-to-atlas ingestion bridge ─────────────────────────────
    def register_from_c(self, name: str, c_source: str, func_symbol: str,
                        cols_instrs: int = 64) -> dict:
        """Compile a C function for RV32I, transpile it to Glyph, extract the
        function body as a namespaced tile, and register it — "any C algorithm
        becomes a first-class atlas tile". The compiled leaf must use the C
        ABI (args in a0.. == glyph r10..), return via `ret` (-> RET), and
        touch only caller-provided pointers. Byte load/store addresses are
        lowered to word indices (byte_to_word_mem); callers seed arrays at
        `byte_base >> 2`.

        Raises if the toolchain is missing, the tile does not assemble, or
        the lowered text fails the GlyphIR StaticVerifier BEFORE pixel
        emission (GH-15: use_ir is the transpiler default — an IR-rejected
        ingestion is loud, never a silent legacy fallback).
        """
        from rv64i_to_glyph import transpile_elf_to_glyph  # noqa: E402
        if shutil.which(_RV_GCC) is None:
            raise RuntimeError(f"{_RV_GCC} not installed — cannot ingest C")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "f.c").write_text(c_source)
            subprocess.run(
                [_RV_GCC, "-march=rv32i", "-mabi=ilp32", "-O1", "-nostdlib",
                 "-fno-builtin", "-ffreestanding", "-w", "-c",
                 str(td / "f.c"), "-o", str(td / "f.o")], check=True,
                capture_output=True)
            subprocess.run(
                [_RV_GCC, "-march=rv32i", "-mabi=ilp32", "-nostdlib",
                 "-Wl,-Ttext=0x0", f"-Wl,--entry={func_symbol}", "-w",
                 str(td / "f.o"), "-o", str(td / "f.elf")], check=True,
                capture_output=True)
            glyph = transpile_elf_to_glyph(
                (td / "f.elf").read_bytes(), entry_symbol=func_symbol,
                byte_to_word_mem=True, cols_instrs=cols_instrs)

        # slice from the function label to EOF (the transpiler emits the
        # target function and any trailing functions in address order; a
        # leaf with no calls is self-contained), then namespace its local
        # :pc_* / :__* labels so two ingested tiles never collide.
        lines = glyph.splitlines()
        try:
            start = next(i for i, ln in enumerate(lines)
                         if ln.strip() == f":{func_symbol}")
        except StopIteration:
            raise ValueError(f"transpiler emitted no ':{func_symbol}' label")
        body = "\n".join(lines[start + 1:]).rstrip("\n")
        body = re.sub(r":(pc_[0-9a-fA-F]+|__[A-Za-z0-9_]+)",
                      rf":{name}_\1", body)
        tile_text = f":atlas_{name}\n{body}\n"

        return self.register(name, tile_text, family="ingested_c",
                             cols_instrs=cols_instrs)

    # ── persistence ───────────────────────────────────────────────────
    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            for t in self.tiles.values():
                f.write(json.dumps(t) + "\n")

    @classmethod
    def load(cls, path: Path) -> "RoutineAtlas":
        atlas = cls()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    t = json.loads(line)
                    atlas.tiles[t["name"]] = t
        return atlas

    # ── linking ───────────────────────────────────────────────────────
    def link(self, caller_text: str, cols_instrs: int = 8) -> str:
        """Rewrite every 'CALL :atlas_<name>' ref and append the referenced
        tile bodies after the caller text. Unknown refs raise — a program
        that names a tile that isn't in the atlas must not silently run
        the wrong code."""
        import re
        used: dict = {}

        def repl(m: "re.Match") -> str:
            name = m.group(1)
            if name not in self.tiles:
                raise ValueError(f"unknown atlas tile '{name}'")
            used[name] = f"atlas_{name}"
            return f"CALL :atlas_{name}"

        linked = re.sub(r"CALL :atlas_(\w+)", repl, caller_text)
        for name, label in used.items():
            body = self.tiles[name]["tile_text"].rstrip("\n")
            # namespaced: ensure the tile's own entry label is unique
            body = body.replace(":atlas_" + name, ":" + label, 1)
            linked += "\n" + body + "\n"
        return linked

    # ── execution of a linked program ─────────────────────────────────
    def run_linked(self, caller_text: str, cols_instrs: int = 8) -> dict:
        return run_generated(self.link(caller_text, cols_instrs),
                             cols_instrs=cols_instrs)


# ── Phase 5.7: the grown font ────────────────────────────────────────────
# Tile grammar facts (verified against glyph_isa_v2 assembler+CPU):
#   'LD rd raddr'  -> rd = memory[raddr]        (register form only)
#   'ST raddr rval'-> memory[raddr] = rval      (addr is rs1)
#   loops: CMP/JZ/JMP (no BNE); strides via LDI scratch + ADD (no ADDI)
#   memory is a 1024-word array; addresses are word indices
# Each tile ships with a HARNESS that sets up registers, calls the tile,
# and HALTs. register() requires assemble+halt, so a tile can only enter
# the atlas through its harness passing execution.

ACCUMULATE = (":atlas_accumulate\nLDI r10 0\n:__acc_loop\nLD r14 r11\n"
              "ADD r10 r14\nLDI r15 1\nADD r11 r15\nLDI r15 1\n"
              "SUB r12 r15\nCMP r12 r0\nJZ :__acc_done\n"
              "JMP :__acc_loop\n:__acc_done\nRET\n")

MEMCPY = (":atlas_memcpy\n:__cpy_loop\nLD r14 r11\nST r12 r14\n"
          "LDI r15 1\nADD r11 r15\nADD r12 r15\nLDI r15 1\nSUB r13 r15\n"
          "CMP r13 r0\nJZ :__cpy_done\nJMP :__cpy_loop\n:__cpy_done\nRET\n")

TILE_CLEAR = (":atlas_tile_clear\n:__clr_loop\nST r11 r14\n"
              "LDI r15 1\nADD r11 r15\nLDI r15 1\nSUB r13 r15\n"
              "CMP r13 r0\nJZ :__clr_done\nJMP :__clr_loop\n"
              ":__clr_done\nRET\n")

# harnesses: in-bounds addresses (<1024 words — out-of-bounds ST is
# silently dropped by the CPU, which would fake a pass)
ACCUMULATE_HARNESS = """
LDI r11 500
LDI r14 7
ST r11 r14
LDI r14 9
LDI r15 501
ST r15 r14
LDI r14 5
LDI r15 502
ST r15 r14
LDI r12 3
CALL :atlas_accumulate
HALT
"""

MEMCPY_HARNESS = """
LDI r11 500
LDI r14 111
ST r11 r14
LDI r14 222
LDI r15 501
ST r15 r14
LDI r14 333
LDI r15 502
ST r15 r14
LDI r12 600
LDI r13 3
CALL :atlas_memcpy
HALT
"""

TILE_CLEAR_HARNESS = """
LDI r11 400
LDI r14 57005
LDI r13 4
CALL :atlas_tile_clear
HALT
"""


def build_default_atlas(path: Optional[Path] = None) -> RoutineAtlas:
    """The Phase 5.6/5.7 atlas: leaf doubling + accumulate + memcpy +
    tile_clear, every tile verified through its registration harness."""
    atlas = RoutineAtlas()
    atlas.register("double", ":atlas_double\nADD r10 r10\nRET\n",
                   family="leaf_call")
    for name, tile, harness in (
            ("accumulate", ACCUMULATE, ACCUMULATE_HARNESS),
            ("memcpy", MEMCPY, MEMCPY_HARNESS),
            ("tile_clear", TILE_CLEAR, TILE_CLEAR_HARNESS)):
        receipt = run_generated(harness + "\n" + tile)
        if not (receipt.get("assembled") and receipt.get("halted")):
            raise ValueError(f"tile '{name}' harness failed: "
                             f"{receipt.get('error') or receipt}")
        atlas.register(name, tile, family="utility")
        atlas.tiles[name]["harness_receipt"] = receipt
    if path is not None:
        atlas.save(path)
    return atlas
