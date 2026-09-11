# Glyph ISA (`glyph-isa`)

[![CI](https://github.com/tdw419/glyph-isa/actions/workflows/ci.yml/badge.svg)](https://github.com/tdw419/glyph-isa/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE-MIT)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE-APACHE)
[![Spec: GLS-1.0](https://img.shields.io/badge/Spec-GLS--1.0-green.svg)](docs/spec/GLYPH_ISA_SPEC_v1.0.md)

**The Spatial-Atomic Programming Language Substrate.**

Glyph (`.glyph`) is a programming language where **the pixels are the program and the pixels are the memory**.

- **Visual Containers**: Executables ship directly as `.glyph.png` raster images. Loading a program is an image decode directly into GPU memory, enabling sub-millisecond cold boots.
- **Proof-Carrying Code (PCC)**: Code cannot enter executable spatial memory without passing machine-checked admission invariants enforced by an oracle gate.
- **Unified 2D Geometry**: Word RAM, instruction cursors, and agent memory boxes map onto a 2D Hilbert-space coordinate curve ($N=128$).
- **Multi-Tenant Agent Arenas**: Autonomous AI agents execute in hardware-isolated spatial boxes (`BOX0..BOXn`) with non-blocking mailbox IPC.

---

## Formal Specification

The complete formal language specification is [GLS-1.0](docs/spec/GLYPH_ISA_SPEC_v1.0.md), covering:
- 4-pixel instruction encoding (`INSTR_WIDTH=4`)
- Opcode semantics and the 30 pinned color triplets
- Register conventions (`r0` as comparison flag target; `r26`/`r27` callee-saved; `r31` hardware stack)
- Spatial memory translation and the Two-Table Rule (PTE_RAM vs PTE_HILB)
- Microkernel MMIO mapping and mailbox IPC protocol

---

## Installation

### Prerequisites
- Python 3.11 or 3.12
- Linux / macOS / Windows

```bash
git clone https://github.com/tdw419/glyph-isa.git
cd glyph-isa
pip install -r requirements.txt
```

---

## Quick Start: The `glyphc` Tool

`glyphc` is the unified compiler, runner, and disassembler for Glyph ISA containers.

### 1. Hello World (`examples/01_hello.glyph`)

```assembly
; 01_hello.glyph — Minimal print and halt
LDI r10 42
PRT r10
HALT
```

**Bake to pixel container:**
```bash
./glyphc build examples/01_hello.glyph -o hello.glyph.png
```
*Output:*
```text
✓ Baked 'examples/01_hello.glyph' -> 'hello.glyph.png' (256x19 px, 64 cols)
```

**Execute the container:**
```bash
./glyphc run hello.glyph.png
```
*Output:*
```text
OUTPUT: 42
[HALTED] in 3 steps | r0=0 r10=42
```

**Disassemble the pixel container:**
```bash
./glyphc disasm hello.glyph.png
```
*Output:*
```text
[00,00] LDI r10 42
[01,00] PRT r10
[02,00] HALT
```

---

### 2. Decrement Loop & Flag Branching (`examples/02_counter.glyph`)

Demonstrates GLS-1.0 register conventions: in Glyph ISA, **`r0` is the comparison flag target, NOT a hardwired zero**. `CMP` writes `1` to `r0` if equal, `0` otherwise; `JZ` branches when `r0 != 0`.

```assembly
; 02_counter.glyph — Decrement loop
LDI r1 10
LDI r2 1
LDI r3 0

:loop
SUB r1 r2
CMP r1 r3
JZ :done
JMP :loop

:done
PRT r1
HALT
```

**Build & Run:**
```bash
./glyphc build examples/02_counter.glyph -o counter.glyph.png
./glyphc run counter.glyph.png
```
*Output:*
```text
✓ Baked 'examples/02_counter.glyph' -> 'counter.glyph.png' (256x19 px, 64 cols)
OUTPUT: 0
[HALTED] in 44 steps | r0=1 r10=0
```

---

### 3. Fibonacci & Hardware Stack (`examples/03_fibonacci.glyph`)

Demonstrates recursive subroutine calls (`CALL` / `RET`) and hardware stack frame allocation using `r31`. The stack operates directly on scanline image pixel memory.

```assembly
; 03_fibonacci.glyph — Recursive Fibonacci computation
LDI r31 1023        ; initialize stack pointer in image pixel memory

LDI r10 7           ; compute fib(7)
CALL :fib
PRT r10             ; fib(7) = 13
HALT

:fib
LDI r3 1
CMP r10 r3
JZ :base_case
LDI r3 0
CMP r10 r3
JZ :base_case

PUSH r10            ; save N
LDI r2 1
SUB r10 r2          ; N - 1
CALL :fib
LDI r11 0
ADD r11 r10         ; r11 = fib(N - 1)

POP r10             ; restore N
PUSH r11            ; save fib(N - 1)
LDI r2 2
SUB r10 r2          ; N - 2
CALL :fib           ; r10 = fib(N - 2)

POP r11             ; restore fib(N - 1)
ADD r10 r11         ; r10 = fib(N - 1) + fib(N - 2)
RET

:base_case
RET
```

**Build & Run:**
```bash
./glyphc build examples/03_fibonacci.glyph -o fib.glyph.png
./glyphc run fib.glyph.png
```
*Output:*
```text
✓ Baked 'examples/03_fibonacci.glyph' -> 'fib.glyph.png' (256x19 px, 64 cols)
OUTPUT: 13
[HALTED] in 513 steps | r0=1 r10=13
```

---

### 4. Agent Mailbox IPC (`examples/04_mailbox.glyph`)

Demonstrates multi-tenant agent IPC using the microkernel mailbox protocol (`SYS 6`). Packets are structured as `cksum[31:24] | op[15:8] | payload[7:0]`.

```assembly
; 04_mailbox.glyph — Agent Mailbox IPC protocol
LDI r10 0x140101    ; op=1, payload=1, cksum=0x14
LDI r17 6           ; SYS 6: post packet to mailbox
SYSCALL r10
HALT
```

**Build & Run:**
```bash
./glyphc build examples/04_mailbox.glyph -o mailbox.glyph.png
./glyphc run mailbox.glyph.png
```
*Output:*
```text
✓ Baked 'examples/04_mailbox.glyph' -> 'mailbox.glyph.png' (256x19 px, 64 cols)
[HALTED] in 4 steps | r0=0 r10=4294967295
```

---

## Container Verification

Validate that any `.glyph` file or `.glyph.png` image satisfies geometric invariants:

```bash
./glyphc verify examples/01_hello.glyph
./glyphc verify hello.glyph.png
```
*Output:*
```text
✓ Source 'examples/01_hello.glyph' syntactically valid (bakes to 256x19 container)
✓ Container 'hello.glyph.png' valid: 256x19 px (64 cols, 19 rows)
```

---

## Running the Test Suite

The test suite validates spatial misalignment faults, opcode collisions, Turing completeness, syscall ABIs, and Hilbert curve memory paging:

```bash
pytest tests/
```

*Expected result:*
```text
============================== 36 passed in 9.0s ==============================
```

---

## License

Dual-licensed under either of:
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE))
- **MIT License** ([LICENSE-MIT](LICENSE-MIT))

at your option.
