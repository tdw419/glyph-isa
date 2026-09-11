#!/usr/bin/env python3
"""glyphc — Unified CLI compiler and runner for the Glyph ISA (GLS-1.0).

Commands:
  glyphc build   <source.glyph> [-o <out.png>] [--cols <N>] [--min-rows <M>]
  glyphc run     <image.png>    [--max-steps <N>] [--trace] [--json]
  glyphc disasm  <image.png>    [--cols <N>]
  glyphc verify  <file>
  glyphc project <image.png>    [--w <W>] [--h <H>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.glyph_isa_v2 import (  # noqa: E402
    OpcodeMapV2,
    _unpack_immediate,
    UNUSED_REGISTER,
    INSTR_WIDTH,
)
from tools.glyph_gpt.baker import bake_image  # noqa: E402
from tools.glyph_gpt.runner import GlyphRunner  # noqa: E402
from tools.geos_ascii_bridge import project  # noqa: E402


def cmd_build(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: input file '{in_path}' not found", file=sys.stderr)
        return 1

    out_path = Path(args.output) if args.output else in_path.with_suffix(".glyph.png")
    source_text = in_path.read_text(encoding="utf-8")

    try:
        img = bake_image(
            source_text,
            cols_instrs=args.cols,
            min_rows=args.min_rows,
            out_path=out_path,
        )
        print(f"✓ Baked '{in_path}' -> '{out_path}' ({img.shape[1]}x{img.shape[0]} px, {args.cols} cols)")
        return 0
    except Exception as e:
        print(f"Error during bake: {e}", file=sys.stderr)
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    img_path = Path(args.input)
    if not img_path.exists():
        print(f"Error: container file '{img_path}' not found", file=sys.stderr)
        return 1

    try:
        runner = GlyphRunner(img_path, cols_instrs=args.cols, ram_words=args.ram_words)
        receipt = runner.run(max_instructions=args.max_steps, trace=args.trace)

        if args.json:
            print(json.dumps(receipt, indent=2))
            return 0 if (receipt.get("halted") and not receipt.get("faulted")) else 1

        outputs = receipt.get("output", [])
        for out in outputs:
            print(f"OUTPUT: {out}")

        halted = receipt.get("halted", False)
        faulted = receipt.get("faulted", False)
        steps = receipt.get("steps", 0)

        status_str = "HALTED" if halted else ("FAULTED" if faulted else "STOPPED")
        print(f"[{status_str}] in {steps} steps | r0={receipt['registers_full'][0]} r10={receipt['registers_full'][10]}")

        if faulted:
            print(f"Fault Address: 0x{receipt.get('fault_addr', 0):08x}", file=sys.stderr)
            return 1
        return 0
    except Exception as e:
        print(f"Execution error: {e}", file=sys.stderr)
        return 1


def cmd_disasm(args: argparse.Namespace) -> int:
    img_path = Path(args.input)
    if not img_path.exists():
        print(f"Error: image file '{img_path}' not found", file=sys.stderr)
        return 1

    try:
        img = np.array(Image.open(img_path).convert("RGB"))
        h, w, _ = img.shape
        cols = args.cols or (w // INSTR_WIDTH)
        op_map = OpcodeMapV2()

        disassembled: List[str] = []
        for y in range(h):
            for c in range(cols):
                x = c * INSTR_WIDTH
                if x + 3 >= w:
                    break
                op_rgb = tuple(int(v) for v in img[y, x])
                opcode = op_map.rgb_to_opcode(op_rgb)
                if opcode is None:
                    continue

                rs1, rs2, rd = [int(v) for v in img[y, x + 1]]
                low_px = tuple(int(v) for v in img[y, x + 2])
                high_px = tuple(int(v) for v in img[y, x + 3])
                imm = _unpack_immediate(low_px, high_px)

                loc = f"[{c:02d},{y:02d}]"
                if opcode == "HALT":
                    disassembled.append(f"{loc} HALT")
                elif opcode == "LDI":
                    disassembled.append(f"{loc} LDI r{rd} {imm}")
                elif opcode in ("ADD", "SUB", "CMP", "AND", "OR", "XOR", "SHL", "SHR", "ROTR", "LD"):
                    disassembled.append(f"{loc} {opcode} r{rd} r{rs2}")
                elif opcode == "ST":
                    disassembled.append(f"{loc} ST r{rs1} r{rs2}")
                elif opcode in ("PRT", "PUSH", "POP"):
                    disassembled.append(f"{loc} {opcode} r{rd}")
                elif opcode == "SYSCALL":
                    if rd == UNUSED_REGISTER:
                        disassembled.append(f"{loc} SYSCALL {imm}")
                    else:
                        disassembled.append(f"{loc} SYSCALL r{rd}")
                elif opcode in ("JMP", "JZ", "CALL"):
                    tx = imm & 0xFFFF
                    ty = (imm >> 16) & 0xFFFF
                    disassembled.append(f"{loc} {opcode} {tx},{ty}")
                elif opcode in ("JMPR", "CALLR", "KJMP"):
                    disassembled.append(f"{loc} {opcode} r{rd}")
                elif opcode in ("RET", "SYSRET"):
                    disassembled.append(f"{loc} {opcode}")
                else:
                    disassembled.append(f"{loc} {opcode} r{rd} r{rs2} {imm}")

        for line in disassembled:
            print(line)
        return 0
    except Exception as e:
        print(f"Disassembly error: {e}", file=sys.stderr)
        return 1


def cmd_verify(args: argparse.Namespace) -> int:
    path = Path(args.input)
    if not path.exists():
        print(f"Error: file '{path}' not found", file=sys.stderr)
        return 1

    if path.suffix in (".png", ".npy", ".npz"):
        try:
            if path.suffix in (".npy", ".npz"):
                data = np.load(path)
                img = data if path.suffix == ".npy" else data[list(data.keys())[0]]
            else:
                img = np.array(Image.open(path).convert("RGB"))
            h, w, c = img.shape
            if c != 3:
                print(f"Verification failed: expected 3 RGB channels, got {c}", file=sys.stderr)
                return 1
            if w % INSTR_WIDTH != 0:
                print(f"Verification failed: width {w} not divisible by INSTR_WIDTH ({INSTR_WIDTH})", file=sys.stderr)
                return 1
            print(f"✓ Container '{path}' valid: {w}x{h} px ({w // INSTR_WIDTH} cols, {h} rows)")
            return 0
        except Exception as e:
            print(f"Verification failed: {e}", file=sys.stderr)
            return 1
    else:
        try:
            text = path.read_text(encoding="utf-8")
            img = bake_image(text, cols_instrs=args.cols, min_rows=args.min_rows)
            print(f"✓ Source '{path}' syntactically valid (bakes to {img.shape[1]}x{img.shape[0]} container)")
            return 0
        except Exception as e:
            print(f"Verification failed: {e}", file=sys.stderr)
            return 1


def cmd_project(args: argparse.Namespace) -> int:
    path = Path(args.input)
    if not path.exists():
        print(f"Error: image '{path}' not found", file=sys.stderr)
        return 1
    try:
        runner = GlyphRunner(path, cols_instrs=args.cols, ram_words=args.ram_words)
        runner.run(max_instructions=args.max_steps)
        cpu = runner.get_cpu()
        canvas = project(cpu.memory, w=args.w, h=args.h)
        print(canvas)
        return 0
    except Exception as e:
        print(f"Projection error: {e}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glyphc",
        description="Geometry OS Glyph ISA Compiler & Runtime Tool (GLS-1.0)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build
    p_build = subparsers.add_parser("build", help="Bake .glyph source into a .glyph.png spatial container")
    p_build.add_argument("input", help="Path to input .glyph source file")
    p_build.add_argument("-o", "--output", help="Path to output .glyph.png image file")
    p_build.add_argument("--cols", type=int, default=64, help="Instruction columns per row (default: 64)")
    p_build.add_argument("--min-rows", type=int, default=16, help="Minimum image rows (default: 16)")

    # run
    p_run = subparsers.add_parser("run", help="Execute a baked .glyph.png spatial container")
    p_run.add_argument("input", help="Path to .glyph.png or .npy container")
    p_run.add_argument("--cols", type=int, default=None, help="Instruction columns per row (auto-detected if omitted)")
    p_run.add_argument("--ram-words", type=int, default=None, help="RAM words to allocate (default: 1024)")
    p_run.add_argument("--max-steps", type=int, default=50000, help="Maximum execution cycles (default: 50000)")
    p_run.add_argument("--trace", action="store_true", help="Record step execution trace")
    p_run.add_argument("--json", action="store_true", help="Output execution receipt as JSON")

    # disasm
    p_disasm = subparsers.add_parser("disasm", help="Disassemble a .glyph.png spatial container back to assembly text")
    p_disasm.add_argument("input", help="Path to .glyph.png container")
    p_disasm.add_argument("--cols", type=int, default=None, help="Instruction columns per row (auto-detected if omitted)")

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify syntax and constraints of a .glyph source or container")
    p_verify.add_argument("input", help="Path to .glyph source or container")
    p_verify.add_argument("--cols", type=int, default=64, help="Instruction columns per row (default: 64)")
    p_verify.add_argument("--min-rows", type=int, default=16, help="Minimum image rows (default: 16)")

    # project
    p_project = subparsers.add_parser("project", help="Run container and project state to ASCII world view")
    p_project.add_argument("input", help="Path to .glyph.png container")
    p_project.add_argument("--cols", type=int, default=None, help="Instruction columns per row")
    p_project.add_argument("--ram-words", type=int, default=16384, help="RAM words (default: 16384)")
    p_project.add_argument("--max-steps", type=int, default=5000, help="Cycles to run before projection")
    p_project.add_argument("--w", type=int, default=80, help="ASCII canvas width")
    p_project.add_argument("--h", type=int, default=25, help="ASCII canvas height")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dispatch = {
        "build": cmd_build,
        "run": cmd_run,
        "disasm": cmd_disasm,
        "verify": cmd_verify,
        "project": cmd_project,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
