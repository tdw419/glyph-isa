# Glyph ISA Specification v1.0 (GLS-1.0)

Status: DRAFT for review (GL-2 of `systems/GLYPH_OSS_ROADMAP.md`)
Normative source: `tools/glyph_isa_v2.py` (engine), `tools/glyph_gpt/baker.py`
(assembler/baker), `tools/glyph_gpt/autoatlas.py` (admission oracle),
`tools/rv64i_to_glyph.py` (transpiler). Where this document and the code
disagree, the code is authoritative until a tagged release freezes GLS-1.0.

Audience contract: this document alone should be enough to write an
independent interpreter or assembler for Glyph images. If you find yourself
needing to read the Python source, that is a bug in this spec — file an issue.

---

## 1. Foundational Architecture

Glyph is a spatial assembly language: a program is a raster image, and
execution is a cursor walking that image. There is no side file, no loader
format, no embedded bytecode section — **the pixels are the program and the
pixels are the memory**.

### 1.1 Execution model

- The machine state is an RGB image (`H x W x 3`, uint8, BGR or RGB per the
  container convention of the tool that produced it).
- The CPU holds a program counter as a **2D pixel coordinate** `(x, y)`.
- One instruction occupies an `INSTR_WIDTH x 1` strip; `INSTR_WIDTH = 4`
  pixels. `PC.x` MUST be a multiple of 4 — a misaligned PC is a
  **SpatialMisalignmentFault**.
- At end-of-row the PC wraps to `(0, y+1)`. There is no fall-through past the
  image edge: running off the image halts the machine.
- Execution modes: `MODE_SUPER = 0` (kernel; no box bounds checks) and
  `MODE_USER = 1` (box-bounded). Mode is set at reset to SUPER; see §5.

### 1.2 Opcodes as colors

Every opcode is identified by the RGB color of its instruction strip:

- All 30 opcodes have **pinned colors** (see §3.3). These are immutable across
  builds and platforms so CPU, Rust, and GPU decoders can hardcode color literals
  without external database lookups.
- A reserved palette region (r,g,b all <= `RESERVED_MAX`) never carries an
  opcode — it is metadata/padding.

### 1.3 The Two-Table Rule (memory duality)

A running Glyph machine has two synchronized views of memory:

1. **Word RAM** — a flat array of 32-bit words (16,384 words in the standard
   harness). Scalar `LD/ST` address this array by word index.
2. **The image** — the same bytes laid out as pixels. Kernel data structures
   that live "in image" (admitted tiles, static tables) are written to the
   image at bake time; the word RAM is seeded from the image at load.

The rule: **RAM is the running copy; the image is the disk.** A kernel FS
store mutates the live ndarray so the image (disk) reflects it. Anything that
claims to persist must be visible in the image, not just in RAM.

### 1.4 Visual containers

A program ships as a `.glyph.png` (lossless). The container is not an
envelope around the program — it IS the program plus its initial memory
image. Loading it is a memcpy, not an ELF parse; boot-to-first-instruction
is one image decode.

---

## 2. Registers & Calling Convention

32 registers, `r0`..`r31`, 32-bit.

### 2.1 Register roles

| Register | Role | Notes |
|----------|------|-------|
| r0 | CMP flag target | **NOT hardwired-zero.** JZ reads its equality flag from r0. Treat as scratch that CMP clobbers. |
| r1..r9 | general scratch | caller-saved by convention |
| r10, r11 | argument registers a0, a1 | `SYSCALL` marshals r10/r11 into SYS_A0/SYS_A1 MMIO words |
| r17 | syscall number a7 | read by the SUPER-mode dispatcher at KSYS_PC |
| r26, r27 | callee-saved s10/s11 | MUST be preserved across calls; kernel RMW sequences must PUSH/POP them (Defect-11 rule) |
| r28..r30 | kernel scratch | used by transpiler emitters and kernel stubs |
| r31 | hardware stack pointer | CALL pushes the return address; RET pops it. Grows DOWNWARD. Seeded once by `LDI r31 <stack_tok>` in the prologue. |

### 2.2 Calling convention

- **Caller-saved:** r1..r15 (minus explicit args). A caller must not assume
  they survive a CALL.
- **Callee-saved:** r16..r27. A callee must preserve them (PUSH/POP r26/r27
  around any sequence that borrows them).
- **r28..r30** are kernel-reserved scratch; user code should not rely on them
  across any observable boundary.
- Arguments: r10 (a0), r11 (a1); return value in r10.
- Syscalls: r17 = syscall number, r10/r11 = args, result back in r10.

### 2.3 Stack

The hardware stack is one word-per-slot array walked by r31, downward. CALL
writes the return address (packed pixel PC, see §3.4) and decrements; RET
reads and increments. Kernel context switches may additionally spill
ra/s0-style state to the BOX2 window (§5.2).

---

## 3. Instruction Set (ISA v2)

### 3.1 Assembly syntax

```
; comment
LABEL:               ; optional label on its own line or prefixing an op
<OPCODE> <operand>*  ; e.g.  LDI r10 42
```

Directives in practice (baker): `.box`, `.origin`, `.lane` region markers and
tile framing used by the assembler; they are metadata, never executed.

### 3.2 Opcode set

Control:  `HALT JMP JZ JMPR CALL CALLR KJMP SYSCALL SYSRET`
Data:     `LDI LD ST PRT`
Arithmetic/logic: `ADD SUB CMP AND OR XOR SHL SHR ROTR`
Stack:    `PUSH POP`
GPU-parallel: `PARALLEL_LD PARALLEL_ST PARALLEL_ADD PARALLEL_SUB PARALLEL_REDUCE_SUM`

Semantics (operational, 32-bit wraparound):

- `LDI rD imm` — rD ← sign-extended imm
- `ADD/SUB/AND/OR/XOR rD rS` — rD ← rD op rS
- `SHL/SHR rD rS` — rD ← rD shifted by rS (mod 32)
- `ROTR rD rS` — rotate right
- `CMP rA rB` — compare; equality flag written to **r0** (see §2.1)
- `LD rD addr` / `ST addr rS` — word RAM load/store (see §4 for address
  translation); USER mode is box-bounded
- `JMP label` / `JZ label` — absolute jump / jump if r0 flag set
- `JMPR rS` — indirect jump; also the one-shot user-mode entry primitive
- `CALL label` / `RET` — hardware stack call/return via r31
- `CALLR rS` — indirect call
- `KJMP target` — privilege-boundary kernel jump (SUPER side)
- `SYSCALL rS` — trap to SUPER-mode dispatcher (§5.3); `SYSRET` returns
- `PRT rS` — host-visible print of a register
- `HALT` — stop the machine

PARALLEL_* opcodes are the spatial-execution surface: they are defined to
decompose per-lane on the GPU port; a scalar interpreter executes them on the
single cursor (same result, no parallel speedup).

### 3.3 Pinned Opcode Palette (GLS-1.0)

As of GLS-1.0, all 30 opcodes are permanently pinned to immutable RGB triples.
Host engines, toolchains, and WebGPU decoders hardcode these values without
external database lookups:

| Opcode | RGB | | Opcode | RGB | | Opcode | RGB |
|--------|-----|-|--------|-----|-|--------|-----|
| HALT | 255, 99, 71 | | PUSH | 34, 139, 34 | | KJMP | 46, 139, 87 |
| LDI | 236, 80, 80 | | POP | 139, 69, 19 | | SYSCALL | 255, 69, 0 |
| ADD | 80, 236, 120 | | CALL | 75, 0, 130 | | SYSRET | 255, 99, 72 |
| SUB | 151, 244, 80 | | RET | 255, 215, 0 | | PARALLEL_LD | 147, 51, 234 |
| CMP | 80, 131, 175 | | JMPR | 60, 179, 113 | | PARALLEL_ST | 255, 20, 147 |
| JMP | 178, 34, 34 | | CALLR | 205, 92, 92 | | PARALLEL_ADD | 0, 255, 127 |
| JZ | 242, 230, 222 | | AND | 100, 149, 237 | | PARALLEL_SUB | 255, 140, 0 |
| PRT | 247, 83, 80 | | OR | 255, 165, 0 | | PARALLEL_REDUCE_SUM | 0, 191, 255 |
| LD | 236, 80, 81 | | XOR | 238, 130, 238 | | | |
| ST | 140, 216, 146 | | SHL | 0, 206, 209 | | | |
| | | | SHR | 218, 112, 214 | | | |
| | | | ROTR | 72, 209, 204 | | | |

### 3.4 Packed pixel PC

A PC is carried in words as `(row << 16) | col` where col is the pixel x of
an instruction-aligned strip. Faults, ticks, and the hardware stack all use
this packing.

---

## 4. Spatial Memory Architecture

### 4.1 Address space

- Word RAM: 16,384 words (addresses are byte addresses; word index =
  `addr >> 2`).
- Words 0..1023: legacy scratch/mailboxes/status (GH-2/3/5 era).
- Words 1024..1535: static tables (e.g. FS alias window at 2 px/word).
- Words 1536..1792: **architectural page table** (GH-17).
- 0x8000 upward: **BOX MMIO** (§5).

### 4.2 Page tables (GH-17/GH-25)

- `PAGE_TABLE_ADDR` (MMIO +0x4C, word 8211) holds the page table base word;
  0 disables translation.
- 256 words (1024 bytes) per page. Page-table entry bits:
  - `PTE_V 0x1` valid
  - `PTE_W 0x2` writable
  - `PTE_U 0x4` user-accessible
  - `PTE_PIX 0x8` frame is spatial pixel-backed in the image
  - `PTE_HILB 0x10` pfn field is a packed 2D Hilbert frame origin `(row<<8)|col`
- With `PTE_HILB`, the frame word index is
  `xy2d(col, row) * PAGE_WORDS + offset` using the **Hacker's Delight xy2d**
  curve over a `HILB_SIDE = 64` x 64 frame grid. This is the GH-25 resident
  memory surface: memory larger than a box pages through Hilbert-ordered
  frames so locality on the curve is locality on the canvas.

### 4.3 Access rules

- SUPER mode: full access; MMIO is protected even from SUPER FS stores.
- USER mode: every data access is checked against the task's box windows
  (§5.2) and, if paging is on, walks the page table. Violations fault to the
  kernel fault handler with faulting address and PC written to MMIO.

---

## 5. Kernel Interface (MMIO, Boxes, Syscalls)

### 5.1 MMIO base

`BOX_MMIO_BASE = 0x8000` (byte address; word 2048). The first 256 bytes are
architectural:

| Offset | Name | Purpose |
|--------|------|---------|
| +0x00 | MODE_LATCH | kernel writes 1 (USER); engine one-shots on next JMPR/CALLR |
| +0x04 | KFAULT_PC | packed PC of fault handler; 0 = disabled |
| +0x08 | KSYS_PC | packed PC of syscall dispatcher (E-K2) |
| +0x0C/+0x10 | BOX0_LO/HI | permitted byte range [lo,hi) |
| +0x14/+0x18 | BOX1_LO/HI | second permitted range |
| +0x1C | FAULT_ADDR | engine writes: faulting byte address |
| +0x20 | FAULT_PC | engine writes: packed PC of offending store |
| +0x24 | SYSCALL_PC | saved resume PC across a syscall |
| +0x28/+0x2C | BOX2_LO/HI | third range (kernel-stack spill page) |
| +0x30 | SYS_N | syscall number (from r17) |
| +0x34 | SYS_A0 | arg0 / result (from/to r10) |
| +0x38 | SYS_A1 | arg1 (from r11) |
| +0x3C | KTICK_PC | packed PC of tick handler; 0 = disabled |
| +0x40 | TIMER_COUNT | countdown steps remaining |
| +0x44 | TIMER_RELOAD | reload value on expiry |
| +0x48 | TICK_PC | engine writes interrupted packed PC at tick |
| +0x4C | PAGE_TABLE | page table base word; 0 = disabled |

### 5.2 Boxes

A **box** is the unit of isolation: the union of the BOX0/BOX1/BOX2 windows
plus a task context. USER-mode stores outside the windows fault. The kernel
is just SUPER-mode glyph code — same ISA, no special instructions — reachable
via the packed-PC handler vectors above.

### 5.3 Syscall protocol

1. USER task loads r17 = syscall number, r10/r11 = args.
2. `SYSCALL r10` traps: engine marshals r17/r10/r11 into SYS_N/SYS_A0/SYS_A1,
   saves the resume PC in SYSCALL_PC, and jumps to KSYS_PC in SUPER mode.
3. The dispatcher (ordinary glyph code) reads the MMIO words, does the work,
   writes the result to SYS_A0.
4. `SYSRET` restores USER mode and resumes at SYSCALL_PC; result lands in r10.

### 5.4 Preemption (GH-16)

Setting KTICK_PC and TIMER_RELOAD arms preemption: every TIMER_COUNT steps
the engine interrupts the task, writes the interrupted packed PC to TICK_PC,
and jumps to KTICK_PC in SUPER mode. The tick handler saves context, switches
boxes (or runs kernel work), and re-arms. Tick boundaries are the only
points at which machine state is guaranteed committed — observers should
sample at ticks (this is what the live-surface dumps rely on).

### 5.5 Mailboxes (GH-18/GH-22)

Word-format for agent/kernel messages:
`cksum[31:24] | op[15:8] | payload[7:0]`.
BOX2's tile-ABI window reserves argv word 750 and result word 754. The
agent-facing write path (geos_emit) may target only mailbox words
{700..767}; anything outside is an aperture violation and is rejected
host-side before any image is touched.

---

## 6. Proof-Carrying Code & the Admission Gate

Glyph's safety story is structural: **code cannot enter a running image
without passing an oracle.**

- `autoatlas.ingest` is the sole write path for new code into an image:
  it raises the tile text to an IR, runs the StaticVerifier (bounds, label
  integrity, jump targets), renders the pixel words, and stamps them into
  the image at a relocated window.
- `autoatlas.admit_syscall` is the sole path for new syscall-capable tiles:
  IR verify → oracle word-exact check against a golden register contract →
  table stamp. Rejection at any stage leaves the image untouched.
- There is no runtime JIT, no self-modifying user code path, and no loader
  that bypasses the gate. The agent emit path (§5.5) writes only mailbox
  payloads — never instructions.
- Every admission attempt is receipted (attempt, source, verdict); receipts
  are the audit trail for what code is allowed to exist in the machine.

---

## 7. Compatibility Notes (informative)

- r0 is not zero: transpiled RV code must never read r0 expecting zero
  (the x0-mapping pitfall that motivated this note cost a real bug).
- Opcode color stability: In early prototypes, the original ten opcodes
  derived their colors from an internal SQLite wordbase.db lookup. In GLS-1.0,
  all 30 opcodes are permanently frozen into `PINNED_COLORS`, eliminating the
  database dependency and preventing non-deterministic color variations across
  environments.
- `LD/ST` appear multiple times in historical opcode tables with the same
  color word (`load`/`store`); this is a dictionary-literal artifact, not
  distinct encodings.
- The 2x2-pixel instruction encodings discussed in early notes were
  superseded by the 4-pixel strip (INSTR_WIDTH=4); GLS-1.0 specifies only
  the 4-pixel strip.

---

## Appendix A: Minimal program

```
; hello.glyph — print 42 and halt
LDI r10 42
PRT r10
HALT
```

Assembled and baked to a `.glyph.png`, this loads as an image, executes
three strips, prints `42` to the host, and halts.

## Appendix B: Glossary

- **strip** — one 4-pixel instruction slot
- **packed PC** — `(row << 16) | col` word form of a program counter
- **box** — a USER task's permitted address windows + context
- **admission** — oracle-checked insertion of code into an image
- **tick boundary** — a preemption point; the only committed-state guarantee
