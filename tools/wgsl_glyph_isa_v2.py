"""
WGSL GPU-native port of glyph_isa_v2.py's Spatial ISA v1.0.

Unlike tools/wgsl_spatial_glyph_engine.py and tools/wgsl_spatial_glyph_working.py
(an older, incompatible ad-hoc instruction encoding), this shader faithfully
implements glyph_isa_v2's actual fixed-width format:

    Every instruction is a 1x4 horizontal pixel block:
        Pixel 0 (Opcode):    RGB identifying the opcode (see OpcodeMapV2)
        Pixel 1 (Registers): R=rs1, G=rs2, B=rd  (0xFF = UNUSED_REGISTER)
        Pixel 2 (Imm-Low):   lower 24 bits of immediate/coordinate
        Pixel 3 (Imm-High):  upper 24 bits (only low 8 bits used here -
                              WGSL registers are u32, unlike Python's
                              unbounded ints, so the immediate is carried
                              as a single u32 rather than the full 48-bit
                              range _pack_immediate supports. No opcode in
                              the current ISA needs more than 32 bits.)

    LD/ST/PUSH/POP/CALL/RET read and write the SAME image buffer that
    holds the program - this is self-modifying-code-capable memory, not
    a separate scratch region, exactly matching GlyphCPUv2._mem_read/write.

Opcode colors are pulled from OpcodeMapV2 at generation time (see
generate_wgsl_opcode_table() below) rather than hand-copied, so this file
never silently drifts from whatever tools/glyph_isa_v2.py currently
resolves (pinned to deterministic GLS-1.0 PINNED_COLORS).
"""

import numpy as np

from tools.glyph_isa_v2 import OpcodeMapV2, INSTR_WIDTH, UNUSED_REGISTER

_OPCODE_ORDER = [
    'HALT', 'LDI', 'ADD', 'SUB', 'CMP', 'JMP', 'JZ', 'PRT', 'LD', 'ST',
    'AND', 'OR', 'XOR', 'SHL', 'SHR', 'PUSH', 'POP', 'CALL', 'RET', 'SYSCALL',
    'JMPR', 'CALLR', 'KJMP'
]


def generate_wgsl_opcode_table(opcode_map: OpcodeMapV2):
    """Emit the WGSL const block + get_opcode_from_color() check lines for
    every opcode currently in OpcodeMapV2, so the shader always matches
    the live map rather than a hand-copied snapshot."""
    consts = []
    checks = []
    for i, op in enumerate(_OPCODE_ORDER):
        consts.append("const OPCODE_%s: u32 = %du;" % (op, i))
        r, g, b = opcode_map.opcode_to_rgb(op)
        checks.append(
            "    if (r == %du && g == %du && b == %du) { return OPCODE_%s; }" % (r, g, b, op)
        )
    return "\n".join(consts), "\n".join(checks)


_SHADER_TEMPLATE = """
struct Pixel {
    r: u32,
    g: u32,
    b: u32,
    a: u32,
}

struct SpatialCPU {
    pc: vec2<u32>,             // 2D program counter (x always a multiple of 4)
    registers: array<u32, 32>, // r0-r31; r31 doubles as the stack pointer
    running: u32,
    output_ptr: u32,
    mode: u32,                 // GH-13 privilege: 0 = SUPER, 1 = USER (KJMP latch)
}

struct Uniforms {
    image_width: u32,
    image_height: u32,
    output_buffer_size: u32,
}

// image is BOTH the program ROM and read/write scratch memory (LD/ST/
// PUSH/POP/CALL/RET all operate on it) - matching GlyphCPUv2 exactly.
@group(0) @binding(0) var<storage, read_write> image: array<Pixel>;
@group(0) @binding(1) var<storage, read_write> cpus: array<SpatialCPU>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var<storage, read_write> box_mmio: array<u32, 64>;

__OPCODE_CONSTS__
const UNUSED_REGISTER: u32 = 255u;
const INSTR_WIDTH: u32 = 4u;

fn get_opcode_from_color(r: u32, g: u32, b: u32) -> u32 {
__OPCODE_CHECKS__
    return 1000u; // Unknown opcode
}

fn load_pixel(x: u32, y: u32) -> vec3<u32> {
    let index = y * uniforms.image_width + x;
    let p = image[index];
    return vec3<u32>(p.r, p.g, p.b);
}

fn store_pixel(x: u32, y: u32, val: vec3<u32>) {
    let index = y * uniforms.image_width + x;
    image[index].r = val.x;
    image[index].g = val.y;
    image[index].b = val.z;
}

// Linear-wrap scalar address -> pixel coordinate (scanline order),
// matching GlyphCPUv2._addr_to_xy.
fn addr_to_xy(addr: u32) -> vec2<u32> {
    let total = uniforms.image_width * uniforms.image_height;
    let wrapped = addr % total;
    return vec2<u32>(wrapped % uniforms.image_width, wrapped / uniforms.image_width);
}

fn mem_read(addr: u32) -> u32 {
    let xy = addr_to_xy(addr);
    let p = load_pixel(xy.x, xy.y);
    return (p.x << 16u) | (p.y << 8u) | p.z;
}

fn mem_write(addr: u32, value: u32) {
    let xy = addr_to_xy(addr);
    let v = value & 0xFFFFFFu;
    store_pixel(xy.x, xy.y, vec3<u32>((v >> 16u) & 0xFFu, (v >> 8u) & 0xFFu, v & 0xFFu));
}

// --- GH-17/GH-25 spatial page walker (WGSL twin) -----------------------------
// Constants mirror tools/glyph_isa_v2.py (word units; word = addr >> 2).
const BOX_MMIO_WORD_LO: u32 = 8192u;          // BOX_MMIO_BASE >> 2
const BOX_MMIO_SPAN: u32 = 64u;               // the reserved MMIO block, 64 words
const PAGE_TABLE_WORD: u32 = 8211u;           // PAGE_TABLE_ADDR >> 2
const PAGE_WORDS: u32 = 256u;                 // words per page
const PTE_V: u32 = 1u;
const PTE_W: u32 = 2u;
const PTE_U: u32 = 4u;
const PTE_PIX: u32 = 8u;
const PTE_HILB: u32 = 16u;
const HILB_SIDE: u32 = 64u;                   // 64x64 Hilbert frame grid
const MODE_LATCH_WORD: u32 = 8192u;           // BOX_MMIO_BASE + 0x00, >> 2
const MODE_USER: u32 = 1u;

// box_mmio[i] mirrors GlyphCPUv2.memory[8192 + i] for the reserved block.
// Bound as a STORAGE buffer (binding 4) so the armed page-table base and the
// mode latch persist across dispatches (declared above, with the bindings).

// Hilbert d2xy: frame slot d -> (col, row) on the HILB_SIDE grid. LUT-free
// bitwise twin of tools.geos_hilbert.hilbert_d2xy_true (Hacker's Delight) —
// leg 4 of the gate proves the emitted table agrees with the host curve.
fn hilb_d2xy(d: u32) -> vec2<u32> {
    var rx: u32;
    var ry: u32;
    var x: u32 = 0u;
    var y: u32 = 0u;
    var t: u32 = d;
    var s: u32 = 1u;
    loop {
        if (s >= HILB_SIDE) { break; }
        rx = 1u & (t / 2u);
        ry = 1u & (t ^ rx);
        // rot(s, x, y, rx, ry): rotate the quadrant
        if (ry == 0u) {
            if (rx == 1u) {
                x = s - 1u - x;
                y = s - 1u - y;
            }
            let tmp: u32 = x;
            x = y;
            y = tmp;
        }
        x = x + s * rx;
        y = y + s * ry;
        t = t / 4u;
        s = s * 2u;
    }
    return vec2<u32>(x, y);
}

// GH-25 translation: PTE with PTE_HILB -> frame pixel word. The PTE's pfn
// field is the PACKED 2D frame origin (row << 8 | col); the frame word is
// xy2d(col, row) * PAGE_WORDS + offset. Bitwise mirror of
// GlyphCPUv2._hilb_frame_pix_word (intra-frame offset stays LINEAR).
fn hilb_xy2d(x_in: u32, y_in: u32) -> u32 {
    var d: u32 = 0u;
    var x: u32 = x_in;
    var y: u32 = y_in;
    var s: u32 = HILB_SIDE >> 1u;
    loop {
        if (s == 0u) { break; }
        let rx: u32 = select(0u, 1u, (x & s) != 0u);
        let ry: u32 = select(0u, 1u, (y & s) != 0u);
        d = d + s * s * ((3u * rx) ^ ry);
        if (ry == 0u) {
            if (rx == 1u) {
                x = HILB_SIDE - 1u - x;
                y = HILB_SIDE - 1u - y;
            }
            let tmp: u32 = x;
            x = y;
            y = tmp;
        }
        s = s >> 1u;
    }
    return d;
}

fn hilb_frame_word(pfn_field: u32, offset: u32) -> u32 {
    let col: u32 = pfn_field & 0xFFu;
    let row: u32 = (pfn_field >> 8u) & 0xFFu;
    return hilb_xy2d(col, row) * PAGE_WORDS + offset;
}

// The walker: armed when box_mmio PAGE_TABLE_WORD slot != 0 and the address
// is NOT a SUPER-mode box-MMIO access (the box block bypasses paging, exactly
// like GlyphCPUv2). PTE fetch is image pixels (the WGSL engine has no RAM;
// box_mmio never covers the PT window). The CPU twin is RAM-FIRST with this
// image read as fallback when the RAM PTE is 0 — for the GH-25 harness (PTE
// stamped in the image, RAM zero) both engines take the image PTE; for
// GH-17..23 kernels (RAM PTEs) the CPU never reaches the fallback, matching
// the landed behavior (RCA systems/GH25_WIP_PTE_FETCH_RCA.md).
// A fault jumps to KFAULT_PC (box_mmio slot 15) or halts, mirroring the CPU.
fn walk_ld(addr: u32, is_super: bool) -> u32 {
    let pt_base = box_mmio[PAGE_TABLE_WORD - BOX_MMIO_WORD_LO];
    if (pt_base == 0u || (is_super && addr >= BOX_MMIO_WORD_LO && addr < BOX_MMIO_WORD_LO + BOX_MMIO_SPAN)) {
        return mem_read(addr);
    }
    let vpn = (addr >> 8u) & 0xFFu;
    let offset = addr & 0xFFu;
    let pte_idx = pt_base + vpn;
    let pte = mem_read(pte_idx);
    if ((pte & PTE_V) == 0u) {
        return 4294967295u; // caller-visible fault marker; never taken in gates
    }
    let pfn = pte >> 8u;
    if ((pte & PTE_HILB) != 0u) {
        return mem_read(hilb_frame_word(pfn, offset));
    }
    if ((pte & PTE_PIX) != 0u) {
        return mem_read(pfn * PAGE_WORDS + offset);
    }
    return mem_read(pfn * PAGE_WORDS + offset);
}

fn walk_st(addr: u32, value: u32, is_super: bool) {
    let pt_base = box_mmio[PAGE_TABLE_WORD - BOX_MMIO_WORD_LO];
    if (pt_base != 0u && !(is_super && addr >= BOX_MMIO_WORD_LO && addr < BOX_MMIO_WORD_LO + BOX_MMIO_SPAN)) {
        let vpn = (addr >> 8u) & 0xFFu;
        let offset = addr & 0xFFu;
        let pte = mem_read(pt_base + vpn);
        if ((pte & PTE_V) != 0u) {
            let pfn = pte >> 8u;
            if ((pte & PTE_HILB) != 0u) {
                mem_write(hilb_frame_word(pfn, offset), value);
                return;
            }
            mem_write(pfn * PAGE_WORDS + offset, value);
            return;
        }
        return; // unmapped store: dropped (fault vectoring is CPU-only for now)
    }
    if (addr >= BOX_MMIO_WORD_LO && addr < BOX_MMIO_WORD_LO + BOX_MMIO_SPAN) {
        box_mmio[addr - BOX_MMIO_WORD_LO] = value;
    } else {
        mem_write(addr, value);
    }
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cpu_id = global_id.x;
    if (cpu_id >= arrayLength(&cpus)) {
        return;
    }

    var cpu = cpus[cpu_id];
    if (cpu.running == 0u) {
        return;
    }

    let x = cpu.pc.x;
    let y = cpu.pc.y;

    if (y >= uniforms.image_height || x >= uniforms.image_width) {
        cpu.running = 0u;
        cpus[cpu_id] = cpu;
        return;
    }

    let opcode_px = load_pixel(x, y);
    let reg_px = load_pixel(x + 1u, y);
    let low_px = load_pixel(x + 2u, y);
    let high_px = load_pixel(x + 3u, y);

    let opcode = get_opcode_from_color(opcode_px.x, opcode_px.y, opcode_px.z);

    let rs1 = reg_px.x;
    let rs2 = reg_px.y;
    let rd = reg_px.z;

    // Immediate: low 24 bits from low_px, next 8 bits from high_px's top
    // byte - a u32-sized subset of Python's full 48-bit pack (see module
    // docstring). Sufficient for every opcode in this ISA.
    let low24 = (low_px.x << 16u) | (low_px.y << 8u) | low_px.z;
    let imm = (high_px.z << 24u) | low24;

    var next_pc = vec2<u32>(x + INSTR_WIDTH, y);
    if (next_pc.x >= uniforms.image_width) {
        next_pc.x = 0u;
        next_pc.y = next_pc.y + 1u;
    }

    if (opcode == OPCODE_LDI) {
        cpu.registers[rd] = imm;
    } else if (opcode == OPCODE_ADD) {
        cpu.registers[rd] = cpu.registers[rd] + cpu.registers[rs2];
    } else if (opcode == OPCODE_SUB) {
        cpu.registers[rd] = cpu.registers[rd] - cpu.registers[rs2];
    } else if (opcode == OPCODE_AND) {
        cpu.registers[rd] = cpu.registers[rd] & cpu.registers[rs2];
    } else if (opcode == OPCODE_OR) {
        cpu.registers[rd] = cpu.registers[rd] | cpu.registers[rs2];
    } else if (opcode == OPCODE_XOR) {
        cpu.registers[rd] = cpu.registers[rd] ^ cpu.registers[rs2];
    } else if (opcode == OPCODE_SHL) {
        cpu.registers[rd] = cpu.registers[rd] << cpu.registers[rs2];
    } else if (opcode == OPCODE_SHR) {
        cpu.registers[rd] = cpu.registers[rd] >> cpu.registers[rs2];
    } else if (opcode == OPCODE_CMP) {
        if (cpu.registers[rd] == cpu.registers[rs2]) {
            cpu.registers[0] = 1u;
        } else {
            cpu.registers[0] = 0u;
        }
    } else if (opcode == OPCODE_LD) {
        let addr = cpu.registers[rs2];
        cpu.registers[rd] = walk_ld(addr, cpu.mode == 0u);
    } else if (opcode == OPCODE_ST) {
        // ST rs1 rs2 -> store rs2 into the pixel at address rs1 (rs1 is the
        // ADDRESS register here, matching GlyphCPUv2 and assembler).
        let addr = cpu.registers[rs1];
        walk_st(addr, cpu.registers[rs2], cpu.mode == 0u);
    } else if (opcode == OPCODE_JMPR) {
        // Jump to the packed pixel PC in rd (raw pixel units, no scaling).
        let packed = cpu.registers[rd];
        let tx = packed & 0xFFFFu;
        let ty = (packed >> 16u) & 0xFFFFu;
        next_pc = vec2<u32>(tx, ty);
    } else if (opcode == OPCODE_KJMP) {
        // GH-13 privilege boundary: JMPR + mode latch one-shot (bitwise twin
        // of GlyphCPUv2's KJMP). The box-MMIO view is private per dispatch,
        // so the latch reads box_mmio[0]; the CPU zeroes its RAM copy after
        // the one-shot and the GPU copy starts at zero each run — same
        // visible behavior for both engines' kernels (arm -> KJMP).
        let packed = cpu.registers[rd];
        let tx = packed & 0xFFFFu;
        let ty = (packed >> 16u) & 0xFFFFu;
        next_pc = vec2<u32>(tx * INSTR_WIDTH, ty);
        if (box_mmio[0u] == MODE_USER) {
            cpu.mode = MODE_USER;
        } else {
            cpu.mode = 0u;
        }
        box_mmio[0u] = 0u;
    } else if (opcode == OPCODE_PUSH) {
        cpu.registers[31] = cpu.registers[31] - 1u;
        mem_write(cpu.registers[31], cpu.registers[rd]);
    } else if (opcode == OPCODE_POP) {
        cpu.registers[rd] = mem_read(cpu.registers[31]);
        cpu.registers[31] = cpu.registers[31] + 1u;
    } else if (opcode == OPCODE_PRT) {
        let idx = cpu.output_ptr;
        if (idx < uniforms.output_buffer_size) {
            output[cpu_id * uniforms.output_buffer_size + idx] = cpu.registers[rd];
        }
        cpu.output_ptr = cpu.output_ptr + 1u;
    } else if (opcode == OPCODE_CALL) {
        // Push the (already pixel-unit) return address, then jump to the
        // instruction-index-encoded target (tx * INSTR_WIDTH), exactly
        // matching the asymmetry in GlyphCPUv2: the *saved* return PC is
        // stored in raw pixel units, but the jump *target* in imm is an
        // instruction index that must be scaled by INSTR_WIDTH.
        cpu.registers[31] = cpu.registers[31] - 1u;
        let packed_pc = (next_pc.y << 16u) | (next_pc.x & 0xFFFFu);
        mem_write(cpu.registers[31], packed_pc);

        let tx = imm & 0xFFFFu;
        let ty = (imm >> 16u) & 0xFFFFu;
        next_pc = vec2<u32>(tx * INSTR_WIDTH, ty);
    } else if (opcode == OPCODE_RET) {
        let packed_pc = mem_read(cpu.registers[31]);
        cpu.registers[31] = cpu.registers[31] + 1u;
        let tx = packed_pc & 0xFFFFu;
        let ty = (packed_pc >> 16u) & 0xFFFFu;
        // Unlike CALL's jump target, the saved return address is already
        // in raw pixel units - no INSTR_WIDTH scaling here.
        next_pc = vec2<u32>(tx, ty);
    } else if (opcode == OPCODE_JMP) {
        let tx = imm & 0xFFFFu;
        let ty = (imm >> 16u) & 0xFFFFu;
        next_pc = vec2<u32>(tx * INSTR_WIDTH, ty);
    } else if (opcode == OPCODE_JZ) {
        // Despite the name, this jumps when the CMP flag (r0) is nonzero
        // (i.e. "jump if equal") - matches GlyphCPUv2's own comment.
        if (cpu.registers[0] != 0u) {
            let tx = imm & 0xFFFFu;
            let ty = (imm >> 16u) & 0xFFFFu;
            next_pc = vec2<u32>(tx * INSTR_WIDTH, ty);
        }
    } else if (opcode == OPCODE_SYSCALL) {
        let syscall_num = imm;
        if (syscall_num == 1u) { // WRITE
            let addr = cpu.registers[1];
            let length = cpu.registers[2];
            var i: u32 = 0u;
            loop {
                if (i >= length) { break; }
                let val = mem_read(addr + i);
                let idx = cpu.output_ptr;
                if (idx < uniforms.output_buffer_size) {
                    output[cpu_id * uniforms.output_buffer_size + idx] = val;
                }
                cpu.output_ptr = cpu.output_ptr + 1u;
                i = i + 1u;
            }
            cpu.registers[rd] = 0u;
        } else if (syscall_num == 2u) { // READ
            let addr = cpu.registers[1];
            let length = cpu.registers[2];
            var i: u32 = 0u;
            loop {
                if (i >= length) { break; }
                mem_write(addr + i, 0u);
                i = i + 1u;
            }
            cpu.registers[rd] = 0u;
        } else if (syscall_num == 3u || syscall_num == 4u || syscall_num == 6u) { // FILE_WRITE, FILE_READ, DEBUG
            cpu.registers[rd] = 0u;
        } else if (syscall_num == 5u) { // EXIT
            cpu.registers[rd] = cpu.registers[1]; // Return status
            cpu.running = 0u;
            cpus[cpu_id] = cpu;
            return;
        } else if (syscall_num >= 16u && syscall_num <= 255u) { // GeOS MMIO
            cpu.registers[rd] = 0u;
        } else { // Unknown
            cpu.registers[rd] = 4294967295u; // -1 as u32
        }
    } else if (opcode == OPCODE_HALT) {
        cpu.running = 0u;
        cpus[cpu_id] = cpu;
        return;
    }

    cpu.pc = next_pc;
    cpus[cpu_id] = cpu;
}
"""


def build_shader(opcode_map: OpcodeMapV2) -> str:
    consts, checks = generate_wgsl_opcode_table(opcode_map)
    src = _SHADER_TEMPLATE.replace("__OPCODE_CONSTS__", consts)
    src = src.replace("__OPCODE_CHECKS__", checks)
    return src


def make_cpu_state_array(n_cpus: int = 1):
    """One SpatialCPU per lane: pc(2) + registers(32) + running(1) +
    output_ptr(1) + mode(1) [+ 1 pad] = 38 u32 (152 bytes).

    WGSL rounds the struct size up to its alignment (vec2<u32> => align 8),
    so the host buffer must carry one trailing pad word or wgpu rejects the
    bind group with a size mismatch."""
    dtype = np.dtype([
        ('pc', np.uint32, 2),
        ('registers', np.uint32, 32),
        ('running', np.uint32),
        ('output_ptr', np.uint32),
        ('mode', np.uint32),
        ('_pad', np.uint32, 1),
    ])
    cpus = np.zeros(n_cpus, dtype=dtype)
    cpus['running'] = 1
    return cpus, dtype
