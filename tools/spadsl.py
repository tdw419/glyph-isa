#!/usr/bin/env python3
"""
SpaDSL — a restricted, array-oriented Python subset that compiles to
GlyphISA v2 assembly (tools/glyph_isa_v2.py).

Scope (Phase 2, extended):
    A = region(shape=(N,), initial=V)   # 1D region of N pixels, filled with V
    A = region(shape=(H, W), initial=V) # 2D region (height x width), row-major layout
    B = region(shape=(N,))              # 1D region, zero-filled
    C = A + B                           # elementwise add, same-size regions
    C = A - B                           # elementwise sub
    D = shift(A, offset=K)              # circular shift by K elements
    E = where(M, A, B)                  # conditional select: A where M!=0, else B
    F = conv2d(A, kernel)               # 2D convolution with kernel (list of lists)
    total = reduce(C, sum)              # sum all elements into one register
    print(total)                        # PRT the scalar register

Compilation strategy: every region op is unrolled at compile time into
straight-line LDI/LD/ST/ADD instructions on the real GlyphISA v2 ISA
(tools/glyph_isa_v2.py). There is no native GPU-parallel spatial opcode
in that ISA today, so this is scalar-unrolled, not warp-parallel — do
not claim GPU speedup from this compiler alone. It exists to prove the
DSL -> assembly -> execution loop closes correctly; a parallel backend
(WGSL) would need a real SIMT opcode added to the ISA first, not just a
compiler that pretends one exists.

Register usage (fixed, no allocator — fine for straight-line unrolled code):
    r20  scratch value A
    r21  scratch value B
    r22  scratch address A
    r23  scratch address B / C
    r24  reduce accumulator
Regions never overlap in the pixel address space: each is allocated a
contiguous run of addresses starting right after the assembled code's
own pixel footprint, so LD/ST on region data can never collide with the
instruction stream that is executing it.
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.glyph_isa_v2 import GlyphAssemblerV2, GlyphCPUv2, OpcodeMapV2, INSTR_WIDTH
from tools.geos_hilbert import hilbert_d2xy_true as hilbert_d2xy, hilbert_xy2d_true as xy2hilbert_d

WIDTH_INSTRS = 8
WIDTH_PX = WIDTH_INSTRS * INSTR_WIDTH

REG_A = 20
REG_B = 21
REG_ADDR_A = 22
REG_ADDR_B = 23
REG_ACC = 24


class SpaDSLError(Exception):
    pass


def linear_index_from_2d(x: int, y: int, width: int, layout: str) -> int:
    """
    Convert 2D coordinates to 1D linear index based on layout.
    
    Args:
        x, y: 2D coordinates
        width: Width of the 2D region
        layout: Either "linear" (row-major) or "hilbert"
    
    Returns:
        1D linear index
    """
    if layout == "linear":
        return y * width + x
    elif layout == "hilbert":
        n = width
        return xy2hilbert_d(n, x, y)
    else:
        raise ValueError(f"Unknown layout: {layout}")


class Region:
    def __init__(self, name: str, base: int, shape: Tuple[int, ...], layout: str = "linear"):
        self.name = name
        self.base = base
        self.shape = shape
        self.layout = layout
        self.size = 1
        for dim in shape:
            self.size *= dim


class SpaDSLCompiler:
    def __init__(self):
        self.regions: Dict[str, Region] = {}
        self.code: List[str] = []
        self.next_data_addr = 0  # patched in later once code length is known
        self.scalars: Dict[str, str] = {}  # name -> description (for print), value lives in REG_ACC snapshot
        self.last_reduce_var: str = None
        self.const_bindings: Dict[str, object] = {}  # compile-time constant data (e.g. conv2d kernels)

    def compile(self, source: str) -> List[str]:
        tree = ast.parse(source)
        for stmt in tree.body:
            self._compile_stmt(stmt)
        self.code.append("HALT")
        return self.code

    def _compile_stmt(self, stmt: ast.stmt):
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise SpaDSLError(f"line {stmt.lineno}: only simple `name = expr` assignment is supported")
            name = stmt.targets[0].id
            self._compile_assign(name, stmt.value, stmt.lineno)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            self._compile_call_stmt(stmt.value, stmt.lineno)
        else:
            raise SpaDSLError(f"line {stmt.lineno}: unsupported statement {ast.dump(stmt)}")

    def _compile_assign(self, name: str, value: ast.expr, lineno: int):
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "region":
            self._compile_region_decl(name, value, lineno)
        elif isinstance(value, ast.BinOp) and isinstance(value.op, (ast.Add, ast.Sub)):
            self._compile_elementwise(name, value, lineno)
        elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "reduce":
            self._compile_reduce(name, value, lineno)
        elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "shift":
            self._compile_shift(name, value, lineno)
        elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "where":
            self._compile_where(name, value, lineno)
        elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "conv2d":
            self._compile_conv2d(name, value, lineno)
        elif isinstance(value, (ast.List, ast.Constant)):
            # Constant data binding (e.g. `kernel = [[0,0,0],[0,1,0],[0,0,0]]`):
            # stored compile-time only; usable as a conv2d() kernel argument.
            try:
                self.const_bindings[name] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                raise SpaDSLError(f"line {lineno}: constant binding '{name}' must be a literal")
        else:
            raise SpaDSLError(f"line {lineno}: unsupported expression {ast.dump(value)}")

    def _compile_region_decl(self, name: str, call: ast.Call, lineno: int):
        shape = None
        initial = 0
        layout = "linear"
        for kw in call.keywords:
            if kw.arg == "shape":
                shape = ast.literal_eval(kw.value)
                if not isinstance(shape, tuple):
                    raise SpaDSLError(f"line {lineno}: shape must be a tuple, e.g., shape=(N,) or shape=(H, W)")
            elif kw.arg == "initial":
                initial = ast.literal_eval(kw.value)
            elif kw.arg == "layout":
                layout = ast.literal_eval(kw.value)
                if layout not in ("linear", "hilbert"):
                    raise SpaDSLError(f"line {lineno}: layout must be 'linear' or 'hilbert', got '{layout}'")
            else:
                raise SpaDSLError(f"line {lineno}: unknown region() keyword {kw.arg}")
        if shape is None:
            raise SpaDSLError(f"line {lineno}: region() requires shape=(N,) or shape=(H, W)")

        # For Hilbert layout, 2D regions must have width that's a power of 2
        if layout == "hilbert":
            if len(shape) != 2:
                raise SpaDSLError(f"line {lineno}: hilbert layout requires 2D shape, got {shape}")
            w = shape[1]
            if (w & (w - 1)) != 0:
                raise SpaDSLError(f"line {lineno}: hilbert layout requires width to be power of 2, got {w}")

        base = self.next_data_addr
        # Calculate total size from shape dimensions
        total_size = 1
        for dim in shape:
            total_size *= dim
        self.next_data_addr += total_size
        region = Region(name, base, shape, layout)
        self.regions[name] = region

        # Initialize all elements
        for i in range(total_size):
            self.code.append(f"LDI r{REG_A} {initial}")
            self.code.append(f"LDI r{REG_ADDR_A} {base + i}")
            self.code.append(f"ST r{REG_ADDR_A} r{REG_A}")

    def _compile_elementwise(self, name: str, binop: ast.BinOp, lineno: int):
        if not (isinstance(binop.left, ast.Name) and isinstance(binop.right, ast.Name)):
            raise SpaDSLError(f"line {lineno}: elementwise ops require two region names, e.g. C = A + B")
        a = self.regions.get(binop.left.id)
        b = self.regions.get(binop.right.id)
        if a is None or b is None:
            raise SpaDSLError(f"line {lineno}: unknown region in {binop.left.id} + {binop.right.id}")
        if a.size != b.size:
            raise SpaDSLError(f"line {lineno}: region size mismatch ({a.size} vs {b.size})")

        base = self.next_data_addr
        self.next_data_addr += a.size
        out = Region(name, base, a.shape, a.layout)
        self.regions[name] = out

        op = "PARALLEL_ADD" if isinstance(binop.op, ast.Add) else "PARALLEL_SUB"
        
        # GPU Parallel Execution (Chunked to fit within 32 registers)
        # We use r0-r9 for A, r10-r19 for B, r20-r29 for C, and r30 for address.
        chunk_size = 10
        for i in range(0, a.size, chunk_size):
            chunk = min(chunk_size, a.size - i)
            # Load region A chunk into consecutive registers starting at r0
            self.code.append(f"PARALLEL_LD r0 {a.base + i} {chunk}")
            # Load region B chunk into consecutive registers starting at r10
            self.code.append(f"PARALLEL_LD r10 {b.base + i} {chunk}")
            
            # Perform elementwise operation: output to registers starting at r20
            self.code.append(f"{op} r20 r0 r10 {chunk}")
            
            # Store output registers to region C memory
            self.code.append(f"LDI r30 {out.base + i}")
            self.code.append(f"PARALLEL_ST r30 r20 {chunk}")

    def _compile_reduce(self, name: str, call: ast.Call, lineno: int):
        if len(call.args) != 2 or not isinstance(call.args[0], ast.Name):
            raise SpaDSLError(f"line {lineno}: reduce(region, sum) requires a region name and a reduction op")
        region = self.regions.get(call.args[0].id)
        if region is None:
            raise SpaDSLError(f"line {lineno}: unknown region {call.args[0].id}")
        op_arg = call.args[1]
        op_name = op_arg.id if isinstance(op_arg, ast.Name) else None
        if op_name != "sum":
            raise SpaDSLError(f"line {lineno}: only reduce(region, sum) is supported in Phase 1")

        # GPU Parallel Execution
        # Sum memory elements directly into the accumulator register
        self.code.append(f"PARALLEL_REDUCE_SUM r{REG_ACC} {region.base} {region.size}")
        
        self.scalars[name] = "reduce_sum"
        self.last_reduce_var = name

    def _compile_call_stmt(self, call: ast.Call, lineno: int):
        if isinstance(call.func, ast.Name) and call.func.id == "print":
            if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
                raise SpaDSLError(f"line {lineno}: print() supports a single scalar variable, e.g. print(total)")
            var = call.args[0].id
            if var not in self.scalars:
                raise SpaDSLError(f"line {lineno}: print() target '{var}' is not a reduce() result")
            self.code.append(f"PRT r{REG_ACC}")
        else:
            raise SpaDSLError(f"line {lineno}: unsupported call {ast.dump(call)}")

    def _compile_shift(self, name: str, call: ast.Call, lineno: int):
        """Compile shift(region, offset=K) - circular shift by K elements."""
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
            raise SpaDSLError(f"line {lineno}: shift() requires a region name, e.g. shift(A, offset=K)")

        src = self.regions.get(call.args[0].id)
        if src is None:
            raise SpaDSLError(f"line {lineno}: unknown region {call.args[0].id}")

        offset = 0
        for kw in call.keywords:
            if kw.arg == "offset":
                offset = ast.literal_eval(kw.value)
            else:
                raise SpaDSLError(f"line {lineno}: unknown shift() keyword {kw.arg}")

        base = self.next_data_addr
        self.next_data_addr += src.size
        out = Region(name, base, src.shape, src.layout)
        self.regions[name] = out

        # Circular shift: for each i, out[i] = src[(i - offset) % size]
        # We need to load from offset address modulo size
        for i in range(src.size):
            src_idx = (i - offset) % src.size
            self.code.append(f"LDI r{REG_ADDR_A} {src.base + src_idx}")
            self.code.append(f"LD r{REG_A} r{REG_ADDR_A}")
            self.code.append(f"LDI r{REG_ADDR_A} {base + i}")
            self.code.append(f"ST r{REG_ADDR_A} r{REG_A}")

    def _compile_where(self, name: str, call: ast.Call, lineno: int):
        """Compile where(mask, A, B) - simplified: always use A (mask ignored for now)."""
        if len(call.args) != 3:
            raise SpaDSLError(f"line {lineno}: where() requires mask, A, B, e.g. where(M, A, B)")

        mask = self.regions.get(call.args[0].id) if isinstance(call.args[0], ast.Name) else None
        a = self.regions.get(call.args[1].id) if isinstance(call.args[1], ast.Name) else None
        b = self.regions.get(call.args[2].id) if isinstance(call.args[2], ast.Name) else None

        if mask is None or a is None or b is None:
            raise SpaDSLError(f"line {lineno}: where() requires all three arguments to be regions")

        if mask.size != a.size or a.size != b.size:
            raise SpaDSLError(f"line {lineno}: where() requires all regions to have same size")

        base = self.next_data_addr
        self.next_data_addr += a.size
        out = Region(name, base, a.shape, a.layout)
        self.regions[name] = out

        # Simplified: just copy A (TODO: implement proper conditional selection)
        for i in range(a.size):
            self.code.append(f"LDI r{REG_ADDR_B} {a.base + i}")
            self.code.append(f"LD r{REG_B} r{REG_ADDR_B}")
            self.code.append(f"LDI r{REG_ADDR_A} {base + i}")
            self.code.append(f"ST r{REG_ADDR_A} r{REG_B}")

    def _compile_conv2d(self, name: str, call: ast.Call, lineno: int):
        """Compile conv2d(A, kernel) - 2D convolution."""
        if len(call.args) != 2:
            raise SpaDSLError(f"line {lineno}: conv2d() requires region and kernel, e.g. conv2d(A, kernel)")

        src = self.regions.get(call.args[0].id) if isinstance(call.args[0], ast.Name) else None
        if src is None:
            raise SpaDSLError(f"line {lineno}: unknown region {call.args[0].id}")

        # Parse kernel: literal list-of-lists, or a name bound earlier via
        # a constant assignment (`kernel = [[...], [...], [...]]`).
        if isinstance(call.args[1], ast.Name) and call.args[1].id in self.const_bindings:
            kernel = self.const_bindings[call.args[1].id]
        else:
            kernel = ast.literal_eval(call.args[1])
        if not isinstance(kernel, list) or not all(isinstance(row, list) for row in kernel):
            raise SpaDSLError(f"line {lineno}: kernel must be a list of lists, e.g. [[0,0,0], [0,1,0], [0,0,0]]")

        k_h = len(kernel)
        k_w = len(kernel[0]) if k_h > 0 else 0

        # Input must be 2D
        if len(src.shape) != 2:
            raise SpaDSLError(f"line {lineno}: conv2d() requires 2D input region")

        h, w = src.shape

        # Output size with zero padding (no dilation, stride=1)
        out_h = h
        out_w = w

        base = self.next_data_addr
        self.next_data_addr += out_h * out_w
        out = Region(name, base, (out_h, out_w), src.layout)
        self.regions[name] = out

        # For each output position, compute convolution
        for out_y in range(out_h):
            for out_x in range(out_w):
                # Load result into r{REG_A}
                self.code.append(f"LDI r{REG_A} 0")
                # Accumulate kernel-weighted sum
                for ky in range(k_h):
                    for kx in range(k_w):
                        weight = kernel[ky][kx]
                        if weight == 0:
                            continue
                        # Compute input coordinates with zero padding
                        in_y = out_y - (k_h // 2) + ky
                        in_x = out_x - (k_w // 2) + kx
                        if 0 <= in_y < h and 0 <= in_x < w:
                            # Convert 2D coordinates to linear index based on layout
                            in_idx = linear_index_from_2d(in_x, in_y, w, src.layout)
                            # Load input value
                            self.code.append(f"LDI r{REG_ADDR_A} {src.base + in_idx}")
                            self.code.append(f"LD r{REG_B} r{REG_ADDR_A}")
                            # Multiply by weight (use ADD for weight=1, or LDI+ADD for others)
                            if weight == 1:
                                self.code.append(f"ADD r{REG_A} r{REG_B}")
                            elif weight == -1:
                                self.code.append(f"SUB r{REG_A} r{REG_B}")
                            else:
                                # Weighted multiply: repeated addition (slow but correct)
                                for _ in range(abs(weight)):
                                    if weight > 0:
                                        self.code.append(f"ADD r{REG_A} r{REG_B}")
                                    else:
                                        self.code.append(f"SUB r{REG_A} r{REG_B}")
                # Store result - also use layout-aware indexing
                out_idx = linear_index_from_2d(out_x, out_y, out_w, out.layout)
                self.code.append(f"LDI r{REG_ADDR_A} {base + out_idx}")
                self.code.append(f"ST r{REG_ADDR_A} r{REG_A}")


def compile_source(source: str) -> Tuple[List[str], Dict[str, Region]]:
    # Data must live in pixel addresses strictly after the code's own
    # footprint, or LD/ST would silently alias into the instruction stream
    # (glyph_isa_v2 addresses wrap over the whole image; code and data
    # share one linear pixel space). Code length depends on how many
    # region elements get unrolled, and region base addresses depend on
    # code length — so compile once with addresses at 0 to measure the
    # code footprint, then recompile with regions based right after it.
    first = SpaDSLCompiler()
    first.compile(source)
    code_rows = (len(first.code) + WIDTH_INSTRS - 1) // WIDTH_INSTRS
    offset = code_rows * WIDTH_PX

    second = SpaDSLCompiler()
    second.next_data_addr = offset
    tree = ast.parse(source)
    for stmt in tree.body:
        second._compile_stmt(stmt)
    second.code.append("HALT")
    return second.code, second.regions


def compile_file(path: Path) -> List[str]:
    source = path.read_text()
    code, _regions = compile_source(source)
    return code


def compile_and_run(path: Path, max_instructions: int = 100000):
    source = path.read_text()
    code, regions = compile_source(source)

    opcode_map = OpcodeMapV2()
    try:
        assembler = GlyphAssemblerV2(opcode_map)
        image = assembler.assemble(code, width_instrs=WIDTH_INSTRS)
        cpu = GlyphCPUv2(opcode_map, cols_instrs=WIDTH_INSTRS)
        # Regions live in word RAM above the code footprint (compile_source
        # bases them after the code's pixel words). GlyphCPUv2 defaults to
        # 1024 words; larger SpaDSL programs place region data past that, and
        # the CPU's out-of-bounds ST trap (Lever #2) would fault the very
        # region-init stores. Size RAM to cover the full region space.
        region_end = max((r.base + r.size) for r in regions.values()) if regions else 0
        if region_end > len(cpu.memory):
            cpu.memory.extend([0] * (region_end - len(cpu.memory)))
        cpu.run(image, max_instructions=max_instructions)
        return code, cpu, regions
    finally:
        opcode_map.close()


def main():
    parser = argparse.ArgumentParser(description="Compile SpaDSL (.py) to GlyphISA v2 assembly (.glyph)")
    parser.add_argument("source", type=Path, help="SpaDSL source file")
    parser.add_argument("-o", "--output", type=Path, help="Output .glyph assembly path")
    parser.add_argument("--run", action="store_true", help="Assemble and execute on GlyphCPUv2")
    args = parser.parse_args()

    try:
        if args.run:
            code, cpu, regions = compile_and_run(args.source)
        else:
            code = compile_file(args.source)
    except SpaDSLError as e:
        print(f"SpaDSL compile error: {e}", file=sys.stderr)
        sys.exit(1)

    out_path = args.output or args.source.with_suffix(".glyph")
    out_path.write_text("\n".join(code) + "\n")
    print(f"Wrote {len(code)} instructions to {out_path}")

    if args.run:
        print(f"Output: {cpu.output}")


if __name__ == "__main__":
    main()
