#!/usr/bin/env python3
"""
Glyph Stratum Spatial ISA v1.0 — fixed-width, collision-safe pixel CPU.

Every instruction is a 1x4 horizontal pixel block:
    Pixel 0 (Opcode):    semantic RGB derived from wordbase.db
    Pixel 1 (Registers): R=rs1, G=rs2, B=rd  (0xFF = UNUSED_REGISTER)
    Pixel 2 (Imm-Low):   lower 24 bits of immediate/coordinate (RGB)
    Pixel 3 (Imm-High):  upper bits / flags / padding (black = unused)

PC.x must always be a multiple of 4 (INSTR_WIDTH). Unaligned jump
targets raise SpatialMisalignmentFault.

RGB (0,0,0)-(4,4,4) is the reserved System Palette: no opcode may map
there. Collisions are resolved by reassigning the opcode to the next
free color, not by clamping (clamping would merge two opcodes).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


INSTR_WIDTH = 4
RESERVED_MAX = 4  # (0,0,0)..(4,4,4) reserved
UNUSED_REGISTER = 0xFF

# --- xv6-nano isolation (E-K1): reserved MMIO-style word block ---------------
# GlyphCPUv2's data memory (self.memory) is a flat word array. The kernel
# programs the per-task privilege box by storing to these fixed BYTE
# addresses from SUPER mode; the engine reads them every step so a write
# takes effect immediately. Word index = addr >> 2. The block sits at
# byte 0x8000 (word 0x2000) -- above the kernel's stack top (sp starts
# 0x4000, grows down) and clear of PTR_TABLE_BASE (0x2000 + a small table),
# yet inside the GPU oracle SpatialRV64ICore's 36864-byte space so the
# no-fault leg's identical MMIO stores land harmlessly there too.
# See systems/XV6_NANO_ISOLATION_ROADMAP.md (E1-E5).
BOX_MMIO_BASE = 0x8000
MODE_LATCH_ADDR = BOX_MMIO_BASE + 0x00   # kernel writes 1 (USER); engine one-shots it on the next JMPR/CALLR
KFAULT_PC_ADDR  = BOX_MMIO_BASE + 0x04   # packed pixel PC (row<<16)|col of the kernel fault handler; 0 = disabled
KSYS_PC_ADDR    = BOX_MMIO_BASE + 0x08   # E-K2: packed pixel PC of the syscall handler; unused in E-K1
BOX0_LO_ADDR    = BOX_MMIO_BASE + 0x0C   # byte range [lo, hi) the user task may store into (its arena slot)
BOX0_HI_ADDR    = BOX_MMIO_BASE + 0x10
BOX1_LO_ADDR    = BOX_MMIO_BASE + 0x14   # second permitted range (its own proc[] entry / context struct)
BOX1_HI_ADDR    = BOX_MMIO_BASE + 0x18
FAULT_ADDR_ADDR = BOX_MMIO_BASE + 0x1C   # engine writes: faulting byte address
FAULT_PC_ADDR   = BOX_MMIO_BASE + 0x20   # engine writes: packed pixel PC of the offending store
SYSCALL_PC_ADDR = BOX_MMIO_BASE + 0x24   # E-K2: saved resume PC
BOX2_LO_ADDR    = BOX_MMIO_BASE + 0x28   # third permitted range (its own kernel-stack page: a
BOX2_HI_ADDR    = BOX_MMIO_BASE + 0x2C   # yielding task's switch_to prologue spills ra/s0 here)
SYS_N_ADDR      = BOX_MMIO_BASE + 0x30   # E-K2: SYSCALL marshals a7/a0/a1 (glyph r17/r10/r11) into
SYS_A0_ADDR     = BOX_MMIO_BASE + 0x34   # these words for the SUPER-mode C dispatcher to read;
SYS_A1_ADDR     = BOX_MMIO_BASE + 0x38   # the dispatcher writes its result back into SYS_A0.
KTICK_PC_ADDR     = BOX_MMIO_BASE + 0x3C # GH-16: packed pixel PC of kernel tick handler; 0 = disabled
TIMER_COUNT_ADDR  = BOX_MMIO_BASE + 0x40 # GH-16: countdown steps remaining
TIMER_RELOAD_ADDR = BOX_MMIO_BASE + 0x44 # GH-16: reload value when timer expires
TICK_PC_ADDR      = BOX_MMIO_BASE + 0x48 # GH-16: engine writes: interrupted packed pixel PC
PAGE_TABLE_ADDR   = BOX_MMIO_BASE + 0x4C # GH-17: base word of page table in RAM; 0 = disabled
PAGE_TABLE_WORD   = PAGE_TABLE_ADDR >> 2 # 8211
PAGE_WORDS        = 256                  # GH-17: 256 words (1024 bytes) per page
PAGE_TABLE_BASE_WORD = 1536              # GH-17: architectural page table base (words 1536..1792)

PTE_V   = 0x1   # Valid
PTE_W   = 0x2   # Writable
PTE_U   = 0x4   # User-accessible
PTE_PIX = 0x8   # Spatial pixel-backed frame in image
PTE_HILB = 0x10  # GH-25: pfn field = packed 2D Hilbert frame origin (row<<8|col)
HILB_SIDE = 64  # GH-25: frame grid is 64x64 frame slots (Hilbert curve side)

# --- xv6-nano isolation (GO-2): the box as a 2D tile ------------------------
# systems/GPU_OS_ROADMAP.md GO-2 (folds in isolation-roadmap E-K3). A task's
# permitted memory can be expressed as a rectangle on the pixel grid: a byte
# address `a` maps to word `w = a >> 2`, then to grid coordinates
# (row, col) = (w // W_MEM, w % W_MEM). W_MEM (grid width, in 32-bit words) is
# a fixed constant shared byte-for-byte with the WGSL SpatialRV64ICore engine
# (const W_MEM in tools/SPATIAL_RV64I.wgsl) and the fixture (#define W_MEM in
# tests/fixtures/xv6_nano.c). The tile words sit above the GPU engine's GPR
# snapshot block (which ends at 0x8150) so the two engines never collide.
W_MEM = 32                              # 128 bytes / grid row
TILE_ROW_ADDR   = BOX_MMIO_BASE + 0x160  # kernel-armed tile origin (grid row / col) ...
TILE_COL_ADDR   = BOX_MMIO_BASE + 0x164
TILE_H_ADDR     = BOX_MMIO_BASE + 0x168  # ... and extent. TILE_H == 0 -> tile unset / inert.
TILE_W_ADDR     = BOX_MMIO_BASE + 0x16C

# --- xv6-nano interactive shell input (GO-3): a host-fed MMIO input ring ----
# systems/GPU_OS_ROADMAP.md GO-3. The test harness writes INPUT_LEN and the
# scripted byte stream at INPUT_DATA before the run (the same way it seeds
# KFAULT_PC); the SUPER-mode C syscall_dispatch services SYS_read (syscall #2)
# by copying one line out of the ring into the caller's buffer and advancing
# INPUT_CURSOR. Neither engine needs to know these words -- the dispatcher runs
# in SUPER and its loads/stores are never box-checked. Placed above the GO-2
# tile words (end 0x816C) and the GPU engine's 32-GPR snapshot (end 0x8150);
# the 64-byte data region ends at 0x81C0 (word 8304), inside both the
# 16384-word Glyph memory and the 36864-byte SpatialRV64ICore GO test core.
INPUT_LEN_ADDR    = BOX_MMIO_BASE + 0x170  # total bytes the harness loaded
INPUT_CURSOR_ADDR = BOX_MMIO_BASE + 0x174  # bytes consumed so far (dispatcher advances)
INPUT_DATA_ADDR   = BOX_MMIO_BASE + 0x180  # start of the byte stream
INPUT_DATA_CAP    = 64                     # max scripted bytes -> ends 0x81C0

MODE_SUPER = 0
MODE_USER = 1


class SpatialMisalignmentFault(Exception):
    pass


class OpcodeMapV2:
    """GLS-1.0 Pinned Opcode Map.
    
    All 30 opcodes in Glyph Stratum Spatial ISA v1.0 are permanently pinned
    to immutable RGB triples. No external database lookup is required.
    """
    OPCODES = {
        'HALT': 'halt',
        'LDI': 'load_immediate',
        'ADD': 'add',
        'SUB': 'subtract',
        'CMP': 'compare',
        'JMP': 'jump',
        'JZ': 'jump_if_zero',
        'PRT': 'print',
        'LD': 'load',
        'ST': 'store',
        'AND': 'bitwise_and',
        'OR': 'bitwise_or',
        'XOR': 'bitwise_xor',
        'SHL': 'shift_left',
        'SHR': 'shift_right',
        'ROTR': 'rotate_right',
        'PUSH': 'push',
        'POP': 'pop',
        'CALL': 'call',
        'RET': 'return',
        'JMPR': 'jump_register',
        'CALLR': 'call_register',
        'KJMP': 'kernel_jump',
        'SYSCALL': 'syscall',
        'SYSRET': 'sysret',
        'PARALLEL_LD': 'parallel_load',
        'PARALLEL_ST': 'parallel_store',
        'PARALLEL_ADD': 'parallel_add',
        'PARALLEL_SUB': 'parallel_subtract',
        'PARALLEL_REDUCE_SUM': 'parallel_reduce_sum',
    }

    PINNED_COLORS: Dict[str, Tuple[int, int, int]] = {
        'HALT': (255, 99, 71),
        'LDI': (236, 80, 80),
        'ADD': (80, 236, 120),
        'SUB': (151, 244, 80),
        'CMP': (80, 131, 175),
        'JMP': (178, 34, 34),
        'JZ': (242, 230, 222),
        'PRT': (247, 83, 80),
        'LD': (236, 80, 81),
        'ST': (140, 216, 146),
        'AND': (100, 149, 237),
        'OR': (255, 165, 0),
        'XOR': (238, 130, 238),
        'SHL': (0, 206, 209),
        'SHR': (218, 112, 214),
        'ROTR': (72, 209, 204),
        'PUSH': (34, 139, 34),
        'POP': (139, 69, 19),
        'CALL': (75, 0, 130),
        'RET': (255, 215, 0),
        'JMPR': (60, 179, 113),
        'CALLR': (205, 92, 92),
        'KJMP': (46, 139, 87),
        'SYSCALL': (255, 69, 0),
        'SYSRET': (255, 99, 72),
        'PARALLEL_LD': (147, 51, 234),
        'PARALLEL_ST': (255, 20, 147),
        'PARALLEL_ADD': (0, 255, 127),
        'PARALLEL_SUB': (255, 140, 0),
        'PARALLEL_REDUCE_SUM': (0, 191, 255),
    }

    # Backward compatibility alias
    FIXED_COLORS = PINNED_COLORS

    @staticmethod
    def _is_reserved(rgb: Tuple[int, int, int]) -> bool:
        r, g, b = rgb
        return r <= RESERVED_MAX and g <= RESERVED_MAX and b <= RESERVED_MAX

    def _next_free_color(self, start: Tuple[int, int, int]) -> Tuple[int, int, int]:
        r, g, b = start
        packed = (r << 16) | (g << 8) | b
        while True:
            packed = (packed + 1) % (256 ** 3)
            candidate = ((packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF)
            if self._is_reserved(candidate):
                continue
            if candidate in self._rgb_to_opcode:
                continue
            return candidate

    def _color_for_word(self, word: str, opcode: str) -> Tuple[int, int, int]:
        if opcode in self.PINNED_COLORS:
            return self.PINNED_COLORS[opcode]
        hash_val = sum(ord(c) * (i + 1) for i, c in enumerate(opcode))
        return ((hash_val * 7) % 256, (hash_val * 13) % 256, (hash_val * 17) % 256)

    def _build_maps(self):
        for opcode in self.OPCODES:
            rgb = self._color_for_word(self.OPCODES[opcode], opcode)
            if self._is_reserved(rgb) or rgb in self._rgb_to_opcode:
                rgb = self._next_free_color(rgb)
            self._opcode_to_rgb[opcode] = rgb
            self._rgb_to_opcode[rgb] = opcode

    def __init__(self, wordbase_path: Optional[Path] = None):
        self._opcode_to_rgb: Dict[str, Tuple[int, int, int]] = {}
        self._rgb_to_opcode: Dict[Tuple[int, int, int], str] = {}
        self._build_maps()

    def opcode_to_rgb(self, opcode: str) -> Optional[Tuple[int, int, int]]:
        return self._opcode_to_rgb.get(opcode)

    def rgb_to_opcode(self, rgb: Tuple[int, int, int]) -> Optional[str]:
        return self._rgb_to_opcode.get((int(rgb[0]), int(rgb[1]), int(rgb[2])))

    def close(self):
        pass


def _pack_immediate(value: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Pack a signed/unsigned int into Imm-Low (24 bits) + Imm-High (24 bits)."""
    uval = value & ((1 << 48) - 1)
    low24 = uval & 0xFFFFFF
    high24 = (uval >> 24) & 0xFFFFFF
    low_px = ((low24 >> 16) & 0xFF, (low24 >> 8) & 0xFF, low24 & 0xFF)
    high_px = ((high24 >> 16) & 0xFF, (high24 >> 8) & 0xFF, high24 & 0xFF)
    return low_px, high_px


def _unpack_immediate(low_px, high_px) -> int:
    low24 = (int(low_px[0]) << 16) | (int(low_px[1]) << 8) | int(low_px[2])
    high24 = (int(high_px[0]) << 16) | (int(high_px[1]) << 8) | int(high_px[2])
    return (high24 << 24) | low24


class GlyphAssemblerV2:
    """Assemble a tiny assembly dialect into a fixed-width 4-pixel-per-instruction image."""

    def __init__(self, opcode_map: OpcodeMapV2):
        self.opcode_map = opcode_map

    def assemble(self, lines: List[str], width_instrs: int = 8) -> np.ndarray:
        instrs = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            instrs.append(parts)

        n = len(instrs)
        cols = width_instrs
        rows = (n + cols - 1) // cols
        width_px = cols * INSTR_WIDTH
        image = np.zeros((max(rows, 1), width_px, 3), dtype=np.uint8)

        for idx, parts in enumerate(instrs):
            row = idx // cols
            col = idx % cols
            base_x = col * INSTR_WIDTH

            opcode = parts[0]
            args = parts[1:]
            image[row, base_x] = self.opcode_map.opcode_to_rgb(opcode)

            rs1 = rs2 = rd = UNUSED_REGISTER
            imm = 0

            if opcode == 'LDI':
                rd = int(args[0][1:])
                # Parse immediate, supporting hex (0x) and decimal
                imm_str = args[1]
                if imm_str.startswith('0x') or imm_str.startswith('0X'):
                    imm = int(imm_str, 16)
                else:
                    imm = int(imm_str)
            elif opcode in ('ADD', 'SUB', 'CMP', 'LD', 'AND', 'OR', 'XOR', 'SHL', 'SHR', 'ROTR'):
                rd = int(args[0][1:])
                rs2 = int(args[1][1:])
            elif opcode == 'ST':
                # ST <addr_reg> <value_reg> -> memory[addr_reg] = value_reg.
                # Encoded as rs1/rs2 (not rd/rs2) to match the live CPU
                # dispatch branch, which reads the address from rs1. Prior
                # encoding put the address in rd, which the CPU never read,
                # so ST always crashed with UNUSED_REGISTER (0xFF) as rs1.
                rs1 = int(args[0][1:])
                rs2 = int(args[1][1:])
            elif opcode in ('JMPR', 'CALLR', 'KJMP'):
                # Single operand: the register holding the packed
                # (row<<16)|col jump/call target — the same packing JMP's
                # immediate uses, just read from a register instead of
                # baked in at assemble time. This is what makes dispatch
                # dynamic: the target can be data (e.g. a WCB's stored
                # entry point) rather than a fixed label.
                rd = int(args[0][1:])
            elif opcode in ('PRT', 'PUSH', 'POP', 'SYSCALL'):
                rd = int(args[0][1:])
                # Parse immediate for SYSCALL
                if opcode == 'SYSCALL' and len(args) > 1:
                    imm_str = args[1]
                    if imm_str.startswith('0x') or imm_str.startswith('0X'):
                        imm = int(imm_str, 16)
                    else:
                        imm = int(imm_str)
            elif opcode in ('JMP', 'JZ', 'CALL'):
                x, y = args[0].split(',')
                imm = (int(y) << 16) | int(x)  # pack coord into imm
            elif opcode in ('HALT', 'RET', 'SYSRET'):
                pass
            elif opcode == 'PARALLEL_LD':
                # PARALLEL_LD rd addr count - load count values starting at addr into rd
                rd = int(args[0][1:])
                imm_str = args[1]
                if imm_str.startswith('0x') or imm_str.startswith('0X'):
                    addr = int(imm_str, 16)
                else:
                    addr = int(imm_str)
                # count can be immediate or register
                if len(args) > 2:
                    if args[2].startswith('r'):
                        rs2 = int(args[2][1:])
                        imm = addr  # Just addr in imm, count in rs2
                    else:
                        # Immediate count: pack both addr and count into imm
                        count = int(args[2])
                        imm = (count << 24) | addr  # Pack count in high bits, addr in low bits
                        rs2 = 0  # Clear rs2 since count is in imm
            elif opcode == 'PARALLEL_ST':
                # PARALLEL_ST addr_reg rs count - store count values from rs starting at address in addr_reg
                if args[0].startswith('r'):
                    # Address is in a register
                    addr_reg = int(args[0][1:])
                    rs = int(args[1][1:])
                    if len(args) > 2:
                        if args[2].startswith('r'):
                            rs2 = int(args[2][1:])  # count in register
                            imm = 0  # dummy
                        else:
                            # Immediate count
                            count = int(args[2])
                            imm = count
                            rs2 = 0
                    # Store addr_reg in rs1 temporarily for the CPU to resolve
                    rs1 = addr_reg
                    # Make rd = rs (source register)
                    rd = rs
                else:
                    # Immediate address (fallback)
                    imm_str = args[0]
                    if imm_str.startswith('0x') or imm_str.startswith('0X'):
                        imm = int(imm_str, 16)
                    else:
                        imm = int(imm_str)
                    rs2 = int(args[1][1:])
                    rd = int(args[2][1:])
            elif opcode in ('PARALLEL_ADD', 'PARALLEL_SUB'):
                # PARALLEL_ADD rd rs1 rs2 count - elementwise add/sub of count values
                rd = int(args[0][1:])
                rs1 = int(args[1][1:])
                rs2 = int(args[2][1:])
                # count is in immediate
                if len(args) > 3:
                    imm_str = args[3]
                    if imm_str.startswith('0x') or imm_str.startswith('0X'):
                        imm = int(imm_str, 16)
                    else:
                        imm = int(imm_str)
            elif opcode == 'PARALLEL_REDUCE_SUM':
                # PARALLEL_REDUCE_SUM rd addr count - sum count values starting at addr into rd
                rd = int(args[0][1:])
                imm_str = args[1]
                if imm_str.startswith('0x') or imm_str.startswith('0X'):
                    imm = int(imm_str, 16)
                else:
                    imm = int(imm_str)
                # count can be immediate or register
                if len(args) > 2:
                    if args[2].startswith('r'):
                        rs2 = int(args[2][1:])
                    else:
                        # Immediate count: store count in imm, CPU handles it
                        count = int(args[2])
                        # Pack both addr and count into imm: low 24 bits = addr, high 24 bits = count
                        # But for simplicity, we'll use a different encoding
                        imm = (count << 24) | imm  # Pack count in high bits
                        rs2 = 0

            image[row, base_x + 1] = (rs1, rs2, rd)
            low_px, high_px = _pack_immediate(imm)
            image[row, base_x + 2] = low_px
            image[row, base_x + 3] = high_px

        return image


class GlyphCPUv2:
    """Fixed-width spatial CPU: PC.x always a multiple of INSTR_WIDTH."""

    def __init__(self, opcode_map: OpcodeMapV2, cols_instrs: int, fs_pix_enabled: bool = False):
        self.opcode_map = opcode_map
        self.cols_instrs = cols_instrs
        self.fs_pix_enabled = fs_pix_enabled
        self.registers = [0] * 32
        self.memory = [0] * 1024
        self.pc = (0, 0)
        self.running = False
        self.output = []
        self.current_imm = 0  # Current instruction's immediate value (for syscalls)
        # xv6-nano isolation (E-K1). Inert until a program stores a nonzero
        # KFAULT_PC / MODE_LATCH into the reserved MMIO words -- every existing
        # image runs entirely in MODE_SUPER with no box check.
        self.mode = MODE_SUPER
        self.fault_addr = 0
        self.fault_pc = 0
        self.faulted = False
        self._syscall_regs = None  # register-file snapshot across a SYSCALL->SYSRET trap

    def _check_alignment(self, x: int):
        if x % INSTR_WIDTH != 0:
            raise SpatialMisalignmentFault(f"PC.x={x} is not aligned to {INSTR_WIDTH}")

    def _addr_to_xy(self, image: np.ndarray, addr: int) -> Tuple[int, int]:
        """Linear-wrap: scalar address -> pixel coordinate (scanline order)."""
        height, width, _ = image.shape
        addr %= width * height
        return addr % width, addr // width

    @staticmethod
    def _hilb_frame_pix_word(pfn_field: int, offset: int) -> int:
        """GH-25: packed 2D frame origin -> image pixel word via the verified
        Hacker's Delight xy2d curve (mirrors tools.geos_hilbert and the WGSL
        twin's hilb_frame_word; bitwise so both engines cannot drift)."""
        col = pfn_field & 0xFF
        row = (pfn_field >> 8) & 0xFF
        side = HILB_SIDE
        d = 0
        s = side >> 1
        x, y = col, row
        while s > 0:
            rx = 1 if (x & s) else 0
            ry = 1 if (y & s) else 0
            d += s * s * ((3 * rx) ^ ry)
            if ry == 0:
                if rx == 1:
                    x = side - 1 - x
                    y = side - 1 - y
                x, y = y, x
            s >>= 1
        return d * PAGE_WORDS + offset

    # --- GH-8b pixel-aliased FS window ------------------------------------
    # Word addresses in [1024, 1280) alias image pixels: 2 pixels per 32-bit
    # word (pixel W*2 = bits 23..0, pixel W*2+1 = bits 31..24), so kernel
    # LD/ST on FSTAB (1024..) / file data (1044..) mutate the SAVED ndarray
    # — the image is the disk, bit-exact through PNG round-trips. The zone
    # sits ABOVE all legacy RAM scratch (<1024: GH-2/3/5 mailboxes, status
    # words) and BELOW BOX MMIO (8192+), so GH-1..7 kernels are untouched.
    _FS_PIX_LO_WORD = 1024
    _FS_PIX_HI_WORD = 1280

    def _fs_pix_read(self, image: np.ndarray, word: int) -> int:
        h, w, _ = image.shape
        lin = (word * 2) % (w * h)
        x, y = lin % w, lin // w
        lo = (int(image[y, x][0]) << 16) | (int(image[y, x][1]) << 8) | int(image[y, x][2])
        lin2 = (word * 2 + 1) % (w * h)
        hi = int(image[lin2 // w, lin2 % w][2])
        return ((lo | (hi << 24)) & 0xFFFFFFFF)

    def _fs_pix_write(self, image: np.ndarray, word: int, value: int):
        h, w, _ = image.shape
        value &= 0xFFFFFFFF
        lin = (word * 2) % (w * h)
        image[lin // w, lin % w] = ((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)
        lin2 = (word * 2 + 1) % (w * h)
        image[lin2 // w, lin2 % w] = (0, 0, (value >> 24) & 0xFF)
        # write-through mirror: keeps legacy receipts (memory[] reads of
        # window words) coherent; the PIXELS remain the persisted truth --
        # on reboot memory[] is zero but the image still carries the data.
        if word < len(self.memory):
            self.memory[word] = value

    def _mem_read(self, image: np.ndarray, addr: int) -> int:
        if self.fs_pix_enabled and self._FS_PIX_LO_WORD <= addr < self._FS_PIX_HI_WORD:
            return self._fs_pix_read(image, addr)
        x, y = self._addr_to_xy(image, addr)
        r, g, b = image[y, x]
        return (int(r) << 16) | (int(g) << 8) | int(b)

    def _mem_write(self, image: np.ndarray, addr: int, value: int):
        if self.fs_pix_enabled and self._FS_PIX_LO_WORD <= addr < self._FS_PIX_HI_WORD:
            self._fs_pix_write(image, addr, value)
            return
        x, y = self._addr_to_xy(image, addr)
        value &= 0xFFFFFF
        image[y, x] = ((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)

    _ISO_TOP_WORD = TILE_W_ADDR >> 2

    @property
    def _iso_enabled(self) -> bool:
        """The isolation MMIO block only exists when data memory is large
        enough to hold it (the xv6-nano harness sizes it to 16384 words).
        Smaller images -- the SHA-256 kernel, unit fixtures -- never see the
        box check or the mode latch."""
        return len(self.memory) > self._ISO_TOP_WORD

    def _addr_in_box(self, byte_addr: int) -> bool:
        """E-K1: is `byte_addr` inside either kernel-programmed range? An
        unset range (HI == 0) never matches, so a partially-configured box
        still confines. Word stores are checked at their base byte."""
        lo0 = self.memory[BOX0_LO_ADDR >> 2]
        hi0 = self.memory[BOX0_HI_ADDR >> 2]
        lo1 = self.memory[BOX1_LO_ADDR >> 2]
        hi1 = self.memory[BOX1_HI_ADDR >> 2]
        lo2 = self.memory[BOX2_LO_ADDR >> 2]
        hi2 = self.memory[BOX2_HI_ADDR >> 2]
        if ((hi0 and lo0 <= byte_addr < hi0)
                or (hi1 and lo1 <= byte_addr < hi1)
                or (hi2 and lo2 <= byte_addr < hi2)):
            return True
        # GO-2: the 2D tile predicate. When a tile is armed (TILE_H != 0) the
        # permitted region also includes the rectangle [TILE_ROW, TILE_ROW +
        # TILE_H) x [TILE_COL, TILE_COL + TILE_W) in (row, col) grid
        # coordinates derived from the byte address (row = (addr >> 2) //
        # W_MEM, col = (addr >> 2) % W_MEM). An unset tile is inert.
        h = self.memory[TILE_H_ADDR >> 2]
        if h:
            w = self.memory[TILE_W_ADDR >> 2]
            trow = self.memory[TILE_ROW_ADDR >> 2]
            tcol = self.memory[TILE_COL_ADDR >> 2]
            word = byte_addr >> 2
            row, col = word // W_MEM, word % W_MEM
            if trow <= row < trow + h and tcol <= col < tcol + w:
                return True
        return False

    def step(self, image: np.ndarray) -> bool:
        was_user = (self.mode == MODE_USER)
        x, y = self.pc
        self._check_alignment(x)
        height, width, _ = image.shape
        if y >= height or x >= width:
            self.running = False
            return False

        opcode_px = tuple(image[y, x])
        reg_px = tuple(image[y, x + 1])
        low_px = tuple(image[y, x + 2])
        high_px = tuple(image[y, x + 3])

        opcode = self.opcode_map.rgb_to_opcode(opcode_px)
        if opcode is None:
            self.running = False
            return False

        rs1, rs2, rd = reg_px
        imm = _unpack_immediate(low_px, high_px)
        self.current_imm = imm  # Store for syscalls that need it

        # Fall-through PC advance wraps to the next row at the instruction-row
        # width, mirroring GlyphAssemblerV2's row-major layout. Only explicit
        # JMP/JZ/CALL carry an absolute (col,row) target; without this wrap,
        # straight-line code that crosses a row boundary walks off the array.
        row_width_px = self.cols_instrs * INSTR_WIDTH
        next_x = x + INSTR_WIDTH
        if next_x >= row_width_px:
            next_pc = (0, y + 1)
        else:
            next_pc = (next_x, y)

        if opcode == 'LDI':
            self.registers[rd] = imm & 0xFFFFFFFF
        elif opcode == 'ADD':
            self.registers[rd] = (self.registers[rd] + self.registers[rs2]) & 0xFFFFFFFF
        elif opcode == 'SUB':
            self.registers[rd] = (self.registers[rd] - self.registers[rs2]) & 0xFFFFFFFF
        elif opcode == 'AND':
            self.registers[rd] &= self.registers[rs2]
        elif opcode == 'OR':
            self.registers[rd] |= self.registers[rs2]
        elif opcode == 'XOR':
            self.registers[rd] ^= self.registers[rs2]
        elif opcode == 'SHL':
            # RISC-V-faithful: shift amount is taken mod 32 (sll/srl/sra use
            # rs2[4:0]); results wrap at 32 bits. Without the mod, Python's
            # unbounded ints make 1 << N a multi-million-digit bomb.
            self.registers[rd] = (self.registers[rd] << (self.registers[rs2] & 31)) & 0xFFFFFFFF
        elif opcode == 'SHR':
            self.registers[rd] = (self.registers[rd] & 0xFFFFFFFF) >> (self.registers[rs2] & 31)
        elif opcode == 'ROTR':
            # 32-bit rotate-right: rd = rotr32(rd, rs2 & 31)
            n = self.registers[rs2] & 31
            v = self.registers[rd] & 0xFFFFFFFF
            self.registers[rd] = ((v >> n) | (v << (32 - n))) & 0xFFFFFFFF if n else v
        elif opcode == 'CMP':
            self.registers[0] = 1 if self.registers[rd] == self.registers[rs2] else 0
        elif opcode == 'LD':
            # LD/ST address the 32-bit word-array RAM (self.memory) — the
            # storage the SHA-256 kernel seeds K/W tables into. The pixel
            # image is program ROM; PUSH/POP/CALL/RET stack via _mem_*.
            # GH-8b: FS-window words (780..1023) alias image pixels instead,
            # so kernel FS stores mutate the saved ndarray (image = disk).
            # GH-17: Spatial Paging translations when PAGE_TABLE_ADDR != 0.
            addr = self.registers[rs2]
            pt_base = self.memory[PAGE_TABLE_ADDR >> 2] if len(self.memory) > (PAGE_TABLE_ADDR >> 2) else 0
            if pt_base != 0 and not (self.mode == MODE_SUPER and (BOX_MMIO_BASE >> 2) <= addr < (BOX_MMIO_BASE >> 2) + 256):
                vaddr = addr
                vpn = (vaddr >> 8) & 0xFF
                offset = vaddr & 0xFF
                pte_idx = pt_base + vpn
                # GH-25 (RCA systems/GH25_WIP_PTE_FETCH_RCA.md): RAM is the
                # LIVE table; image is fallback when RAM PTE == 0 (the GH-25
                # harness stamps its Hilbert PTE into image pixels only).
                # RAM-first ordering keeps bake-time text garbage under the
                # PT window from shadowing valid RAM PTEs.
                pte = 0
                if 0 <= vaddr < 65536 and pte_idx < len(self.memory):
                    pte = self.memory[pte_idx]
                    if pte == 0:
                        h_img, w_img, _ = image.shape
                        if pte_idx < w_img * h_img:
                            pte = self._mem_read(image, pte_idx)
                if (not (pte & PTE_V)) or (self.mode == MODE_USER and not (pte & PTE_U)):
                    self.fault_addr = (vaddr << 2) & 0xFFFFFFFF
                    self.fault_pc = (y << 16) | (x & 0xFFFF)
                    self.faulted = True
                    if len(self.memory) > (FAULT_PC_ADDR >> 2):
                        self.memory[FAULT_ADDR_ADDR >> 2] = self.fault_addr
                        self.memory[FAULT_PC_ADDR >> 2] = self.fault_pc
                    self.mode = MODE_SUPER
                    kf = self.memory[KFAULT_PC_ADDR >> 2] if len(self.memory) > (KFAULT_PC_ADDR >> 2) else 0
                    if kf != 0:
                        tx, ty = kf & 0xFFFF, (kf >> 16) & 0xFFFF
                        target_x = tx * INSTR_WIDTH
                        self._check_alignment(target_x)
                        next_pc = (target_x, ty)
                    else:
                        self.running = False
                        return False
                else:
                    pfn = pte >> 8
                    if pte & PTE_HILB:
                        # GH-25: pfn field = packed 2D frame origin (row<<8|col)
                        # on a HILB_SIDE x HILB_SIDE frame grid; the frame word
                        # is xy2d(col,row)*PAGE_WORDS + offset (intra-frame
                        # offset stays LINEAR). Bit-exact with the WGSL twin.
                        pix_word = self._hilb_frame_pix_word(pfn, offset)
                        self.registers[rd] = self._mem_read(image, pix_word)
                    elif pte & PTE_PIX:
                        pix_word = pfn * PAGE_WORDS + offset
                        self.registers[rd] = self._mem_read(image, pix_word)
                    else:
                        paddr = pfn * PAGE_WORDS + offset
                        if paddr < len(self.memory):
                            self.registers[rd] = self.memory[paddr] & 0xFFFFFFFF
                        else:
                            self.registers[rd] = 0
            elif self.fs_pix_enabled and self._FS_PIX_LO_WORD <= addr < self._FS_PIX_HI_WORD:
                self.registers[rd] = self._fs_pix_read(image, addr)
            elif 0 <= addr < len(self.memory):
                self.registers[rd] = self.memory[addr] & 0xFFFFFFFF
        elif opcode == 'ST':
            # ST <addr_reg> <value_reg>: address in rs1, matching the
            # assembler's rs1/rs2 packing for ST. `addr` is a word index --
            # the byte_to_word_mem transpiler lowering already did `>>2`.
            addr = self.registers[rs1]
            val = self.registers[rs2]
            pt_base = self.memory[PAGE_TABLE_ADDR >> 2] if len(self.memory) > (PAGE_TABLE_ADDR >> 2) else 0
            if pt_base != 0 and not (self.mode == MODE_SUPER and (BOX_MMIO_BASE >> 2) <= addr < (BOX_MMIO_BASE >> 2) + 256):
                vaddr = addr
                vpn = (vaddr >> 8) & 0xFF
                offset = vaddr & 0xFF
                pte_idx = pt_base + vpn
                # GH-25: RAM-first, image-fallback (see the LD twin above).
                pte = 0
                if 0 <= vaddr < 65536 and pte_idx < len(self.memory):
                    pte = self.memory[pte_idx]
                    if pte == 0:
                        h_img, w_img, _ = image.shape
                        if pte_idx < w_img * h_img:
                            pte = self._mem_read(image, pte_idx)
                if (not (pte & PTE_V)) or (not (pte & PTE_W)) or (self.mode == MODE_USER and not (pte & PTE_U)):
                    self.fault_addr = (vaddr << 2) & 0xFFFFFFFF
                    self.fault_pc = (y << 16) | (x & 0xFFFF)
                    self.faulted = True
                    if len(self.memory) > (FAULT_PC_ADDR >> 2):
                        self.memory[FAULT_ADDR_ADDR >> 2] = self.fault_addr
                        self.memory[FAULT_PC_ADDR >> 2] = self.fault_pc
                    self.mode = MODE_SUPER
                    kf = self.memory[KFAULT_PC_ADDR >> 2] if len(self.memory) > (KFAULT_PC_ADDR >> 2) else 0
                    if kf != 0:
                        tx, ty = kf & 0xFFFF, (kf >> 16) & 0xFFFF
                        target_x = tx * INSTR_WIDTH
                        self._check_alignment(target_x)
                        next_pc = (target_x, ty)
                    else:
                        self.running = False
                        return False
                else:
                    pfn = pte >> 8
                    if pte & PTE_HILB:
                        # GH-25: pfn field = packed 2D frame origin (row<<8|col);
                        # the frame word is xy2d(col,row)*PAGE_WORDS + offset
                        # (intra-frame offset stays LINEAR). Bit-exact with the
                        # WGSL twin.
                        pix_word = self._hilb_frame_pix_word(pfn, offset)
                        self._mem_write(image, pix_word, val)
                    elif pte & PTE_PIX:
                        pix_word = pfn * PAGE_WORDS + offset
                        self._mem_write(image, pix_word, val)
                    else:
                        paddr = pfn * PAGE_WORDS + offset
                        if paddr >= len(self.memory):
                            self.memory.extend([0] * (paddr + PAGE_WORDS - len(self.memory)))
                        self.memory[paddr] = val & 0xFFFFFFFF
            elif self.mode == MODE_USER and not self._addr_in_box(addr << 2):
                # E-K1 trap: an out-of-box user store. Record the fault, do
                # NOT perform the store, drop to SUPER and vector to the
                # kernel fault handler (KFAULT_PC, packed pixel PC). Stepping
                # continues -- the kernel reaps the offending proc.
                self.fault_addr = (addr << 2) & 0xFFFFFFFF
                self.fault_pc = (y << 16) | (x & 0xFFFF)
                self.faulted = True
                self.memory[FAULT_ADDR_ADDR >> 2] = self.fault_addr
                self.memory[FAULT_PC_ADDR >> 2] = self.fault_pc
                self.mode = MODE_SUPER
                kf = self.memory[KFAULT_PC_ADDR >> 2]
                tx, ty = kf & 0xFFFF, (kf >> 16) & 0xFFFF
                target_x = tx * INSTR_WIDTH
                self._check_alignment(target_x)
                next_pc = (target_x, ty)
            elif self.fs_pix_enabled and self._FS_PIX_LO_WORD <= addr < self._FS_PIX_HI_WORD:
                # GH-8b: kernel (SUPER-mode) FS stores land in image pixels.
                self._fs_pix_write(image, addr, self.registers[rs2])
            elif 0 <= addr < len(self.memory):
                self.memory[addr] = self.registers[rs2] & 0xFFFFFFFF
            else:
                # Lever #2: out-of-bounds store -- addr < 0 or
                # addr >= len(self.memory). Previously silent-dropped, which
                # let a bad pointer / buffer overrun report "clean execution
                # to HALT" while leaving state unwritten. Now an explicit
                # fault, bounds checked against the *actual* RAM size (not a
                # hardcoded ceiling), so the isolation harness's 16384-word
                # image and the 1024-word unit fixtures are each judged
                # against their own len(self.memory).
                self.fault_addr = (addr << 2) & 0xFFFFFFFF
                self.fault_pc = (y << 16) | (x & 0xFFFF)
                self.faulted = True
                if self.mode == MODE_USER and self._iso_enabled:
                    # In USER mode an in-box address that still lands past RAM
                    # is a misconfigured box: reuse the E-K1 kernel vector so
                    # the kernel reaps the offending proc and stepping goes on.
                    self.memory[FAULT_ADDR_ADDR >> 2] = self.fault_addr
                    self.memory[FAULT_PC_ADDR >> 2] = self.fault_pc
                    self.mode = MODE_SUPER
                    kf = self.memory[KFAULT_PC_ADDR >> 2]
                    tx, ty = kf & 0xFFFF, (kf >> 16) & 0xFFFF
                    target_x = tx * INSTR_WIDTH
                    self._check_alignment(target_x)
                    next_pc = (target_x, ty)
                else:
                    # Standalone / kernel mode: stop immediately and loudly.
                    # Callers must treat `faulted` as distinct from a clean
                    # HALT (see run_generated).
                    self.running = False
        elif opcode == 'PUSH':
            self.registers[31] -= 1
            self._mem_write(image, self.registers[31], self.registers[rd])
        elif opcode == 'POP':
            self.registers[rd] = self._mem_read(image, self.registers[31])
            self.registers[31] += 1
        elif opcode == 'PRT':
            val = self.registers[rd]
            self.output.append(val)
            print(f"OUTPUT: r{rd} = {val}")
        elif opcode == 'CALL':
            # Save next_pc (packed) to stack and jump
            self.registers[31] -= 1
            packed_pc = (next_pc[1] << 16) | (next_pc[0] & 0xFFFF)
            self._mem_write(image, self.registers[31], packed_pc)
            
            tx, ty = imm & 0xFFFF, (imm >> 16) & 0xFFFF
            target_x = tx * INSTR_WIDTH
            self._check_alignment(target_x)
            next_pc = (target_x, ty)
        elif opcode == 'RET':
            # Pop packed_pc from stack and jump. A zero return address means
            # "return from the outermost _start": halt instead of wrapping to
            # glyph address 0 and re-running the program forever. (Caught by
            # the rv64i_to_glyph negative-offset differential test: GCC's
            # _start tail-calls ret with an empty call stack.)
            packed_pc = self._mem_read(image, self.registers[31])
            if packed_pc == 0:
                self.running = False
                next_pc = self.pc
            else:
                self.registers[31] += 1
                tx, ty = packed_pc & 0xFFFF, (packed_pc >> 16) & 0xFFFF
                next_pc = (tx, ty)
        elif opcode == 'SYSCALL':
            # Invoke hypervisor syscall - number in immediate, result in rd.
            # syscall_num IS imm (that's how SYSCALL is encoded), so it's not
            # passed again separately - self.current_imm (set above) already
            # gives handlers access to it without changing this call's arity,
            # which would break every existing bridge_handle(syscall_num, image)
            # monkey-patch in spatial_hypervisor_bridge.py.
            ksys = self.memory[KSYS_PC_ADDR >> 2] if self._iso_enabled else 0
            if ksys:
                # E-K2: trap to the kernel's SUPER-mode syscall dispatcher.
                # Marshal a7/a0/a1 (RV -> glyph regs r17/r10/r11) into the
                # reserved words the C dispatcher reads; save the resume PC;
                # drop to SUPER; jump KSYS_PC (loader-seeded, packed pixel
                # PC). SYSRET (mret) returns. See XV6_NANO_ISOLATION_ROADMAP.
                # The dispatcher is a plain C function -- it clobbers
                # caller-saved regs the task expects to survive the (inlined)
                # `ebreak`. A real trap saves the register file; so do we.
                self._syscall_regs = self.registers[:]
                self.memory[SYS_N_ADDR >> 2] = self.registers[17] & 0xFFFFFFFF
                self.memory[SYS_A0_ADDR >> 2] = self.registers[10] & 0xFFFFFFFF
                self.memory[SYS_A1_ADDR >> 2] = self.registers[11] & 0xFFFFFFFF
                # Save the resume point as (row << 16) | col -- the same
                # packing KJMP/JMPR targets use, so SYSRET recovers it with
                # `col * INSTR_WIDTH`.
                self.memory[SYSCALL_PC_ADDR >> 2] = (
                    (next_pc[1] << 16) | ((next_pc[0] // INSTR_WIDTH) & 0xFFFF))
                self.mode = MODE_SUPER
                tx, ty = ksys & 0xFFFF, (ksys >> 16) & 0xFFFF
                target_x = tx * INSTR_WIDTH
                self._check_alignment(target_x)
                next_pc = (target_x, ty)
            else:
                syscall_num = imm
                self.registers[rd] = self._handle_syscall(syscall_num, image)
        elif opcode == 'SYSRET':
            # E-K2: return from the syscall dispatcher (lowered from RV `mret`).
            # Restore the pre-trap register file, then deliver the
            # dispatcher's result (it wrote SYS_A0) into a0 = r10; re-enter
            # USER; resume at the saved post-SYSCALL PC.
            if self._syscall_regs is not None:
                self.registers[:] = self._syscall_regs
                self._syscall_regs = None
            self.registers[10] = self.memory[SYS_A0_ADDR >> 2] & 0xFFFFFFFF
            self.mode = MODE_USER
            sp = self.memory[SYSCALL_PC_ADDR >> 2]
            tx, ty = sp & 0xFFFF, (sp >> 16) & 0xFFFF
            target_x = tx * INSTR_WIDTH
            self._check_alignment(target_x)
            next_pc = (target_x, ty)
        elif opcode == 'JMP':
            tx, ty = imm & 0xFFFF, (imm >> 16) & 0xFFFF
            target_x = tx * INSTR_WIDTH
            self._check_alignment(target_x)
            next_pc = (target_x, ty)
        elif opcode == 'JZ':
            # CMP sets r0=1 on equality; JZ jumps when that flag is set.
            if self.registers[0] != 0:
                tx, ty = imm & 0xFFFF, (imm >> 16) & 0xFFFF
                target_x = tx * INSTR_WIDTH
                self._check_alignment(target_x)
                next_pc = (target_x, ty)
        elif opcode == 'JMPR':
            packed = self.registers[rd]
            tx, ty = packed & 0xFFFF, (packed >> 16) & 0xFFFF
            target_x = tx * INSTR_WIDTH
            self._check_alignment(target_x)
            next_pc = (target_x, ty)
        elif opcode == 'KJMP':
            # switch_to's terminal, data-sourced `ret` -- the privilege
            # boundary. Behaves like JMPR, then: drop to SUPER (returning to
            # the scheduler/kernel), and if the kernel armed MODE_LATCH just
            # before this switch, re-enter USER for the task we're jumping
            # into. E-K1; see systems/XV6_NANO_ISOLATION_ROADMAP.md.
            packed = self.registers[rd]
            tx, ty = packed & 0xFFFF, (packed >> 16) & 0xFFFF
            target_x = tx * INSTR_WIDTH
            self._check_alignment(target_x)
            next_pc = (target_x, ty)
            if self._iso_enabled:
                self.mode = MODE_SUPER
                if self.memory[MODE_LATCH_ADDR >> 2] == MODE_USER:
                    self.mode = MODE_USER
                    self.memory[MODE_LATCH_ADDR >> 2] = 0
        elif opcode == 'CALLR':
            self.registers[31] -= 1
            packed_pc = (next_pc[1] << 16) | (next_pc[0] & 0xFFFF)
            self._mem_write(image, self.registers[31], packed_pc)
            packed = self.registers[rd]
            tx, ty = packed & 0xFFFF, (packed >> 16) & 0xFFFF
            target_x = tx * INSTR_WIDTH
            self._check_alignment(target_x)
            next_pc = (target_x, ty)
        elif opcode == 'PARALLEL_LD':
            # PARALLEL_LD rd addr count - load count values starting at addr into rd
            # Word-RAM addressing (matches scalar LD/ST and the WGSL twin's
            # cpus[...].memory): SpaDSL regions are initialized with scalar ST
            # into self.memory, so the parallel path MUST read the same store.
            # Previously this read image pixels, so region data written by ST
            # was invisible here (stale pixel garbage -> wrong reduce results).
            addr = imm & 0xFFFFFF  # low 24 bits = base address
            count = rs2 if rs2 > 0 else (imm >> 24)  # count in rs2 or high bits of imm
            for i in range(count):
                val = self.memory[addr + i] if 0 <= addr + i < len(self.memory) else 0
                # Store in consecutive registers starting at rd
                if rd + i < 32:  # Register bounds check
                    self.registers[rd + i] = val
        elif opcode == 'PARALLEL_ST':
            # PARALLEL_ST addr_reg rs count - store count values from rs starting at address in addr_reg
            # Word-RAM addressing (see PARALLEL_LD note).
            # rs1 contains the address register, rd contains the source register, imm contains count
            addr = self.registers[rs1]  # Get address from register
            count = imm if imm > 0 else rs2  # count in immediate or rs2
            rs_base = rd  # Source register base
            for i in range(count):
                a = addr + i
                if 0 <= a < len(self.memory):
                    self.memory[a] = self.registers[rs_base + i] & 0xFFFFFFFF if rs_base + i < 32 else 0
                    # GH-8b write-through pixel mirror (RCA GH9LOADER_PATCH_
                    # IMAGE_MIRROR, 2026-09-10): the Python engine FETCHES
                    # instructions from IMAGE pixels (step(): image[y,x]) --
                    # pixels are the persisted instruction truth. The GH-9
                    # loader kernel patches its exec window with this opcode;
                    # 087f29b moved PARALLEL_ST to RAM-only, so the patch
                    # landed in RAM the fetch never sees (window stayed HALT
                    # padding -> every injected program died on instr 0,
                    # result 0: test_gh9_loader / GH-11 / GH-15-step3 e2e
                    # / test_gh11_drives_gh9 reds). Mirror the store into
                    # the image at the same linear pixel index when in
                    # bounds -- identical to _fs_pix_write's write-through
                    # mirror (memory[] coherent, pixels the truth). SpaDSL
                    # region reads (PARALLEL_LD) still hit RAM, so 087f29b's
                    # region fix is preserved; addresses beyond the image
                    # simply skip the mirror.
                    ih, iw, _ = image.shape
                    if 0 <= a < ih * iw:
                        v = self.memory[a]
                        image[a // iw, a % iw] = (
                            (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF)
        elif opcode == 'PARALLEL_ADD':
            # PARALLEL_ADD rd rs1 rs2 count - elementwise add of count values
            # r[rd + i] = r[rs1 + i] + r[rs2 + i] for i in 0..count-1
            count = imm if imm > 0 else 1  # count in immediate
            for i in range(count):
                src1_idx = rs1 + i
                src2_idx = rs2 + i
                dst_idx = rd + i
                if dst_idx < 32 and src1_idx < 32 and src2_idx < 32:  # Register bounds check
                    self.registers[dst_idx] = self.registers[src1_idx] + self.registers[src2_idx]
        elif opcode == 'PARALLEL_SUB':
            # PARALLEL_SUB rd rs1 rs2 count - elementwise sub of count values  
            count = imm if imm > 0 else 1  # count in immediate
            for i in range(count):
                if rd + i < 32 and rs1 + i < 32 and rs2 + i < 32:  # Register bounds check
                    self.registers[rd + i] = self.registers[rs1 + i] - self.registers[rs2 + i]
        elif opcode == 'PARALLEL_REDUCE_SUM':
            # PARALLEL_REDUCE_SUM rd addr count - sum count values starting at addr into rd
            # Word-RAM addressing (see PARALLEL_LD note): must read the same
            # store scalar ST wrote region data into.
            addr = imm & 0xFFFFFF  # low 24 bits = base address
            count = rs2 if rs2 > 0 else (imm >> 24)  # count in rs2 or high bits of imm
            total = 0
            for i in range(count):
                a = addr + i
                if 0 <= a < len(self.memory):
                    total += self.memory[a]
            self.registers[rd] = total & 0xFFFFFFFF
        elif opcode == 'HALT':
            self.running = False
            return False

        # GH-16: Preemptive scheduling timer tick
        if self._iso_enabled and was_user and self.mode == MODE_USER and self.running and not self.faulted:
            ktick = self.memory[KTICK_PC_ADDR >> 2]
            if ktick != 0:
                tcount = self.memory[TIMER_COUNT_ADDR >> 2]
                if tcount > 0:
                    tcount -= 1
                    self.memory[TIMER_COUNT_ADDR >> 2] = tcount
                    if tcount == 0:
                        reload_val = self.memory[TIMER_RELOAD_ADDR >> 2]
                        self.memory[TIMER_COUNT_ADDR >> 2] = reload_val
                        interrupted_col = (next_pc[0] // INSTR_WIDTH) & 0xFFFF
                        interrupted_row = next_pc[1] & 0xFFFF
                        self.memory[TICK_PC_ADDR >> 2] = (interrupted_row << 16) | interrupted_col
                        self.mode = MODE_SUPER
                        tx, ty = ktick & 0xFFFF, (ktick >> 16) & 0xFFFF
                        target_x = tx * INSTR_WIDTH
                        self._check_alignment(target_x)
                        next_pc = (target_x, ty)

        self.pc = next_pc
        return True

    def _handle_syscall(self, syscall_num: int, image: np.ndarray, imm: int = 0) -> int:
        """
        Handle hypervisor syscalls from spatial programs.

        Syscall interface compatible with GeOS hypervisor:
        - 0x01: SYSCALL_WRITE - Write to output buffer (arg1=r1=addr, arg2=r2=len)
        - 0x02: SYSCALL_READ  - Read from input buffer (arg1=r1=addr, arg2=r2=len)
        - 0x03: SYSCALL_FILE_WRITE - Write to file (arg1=r1=path_addr, arg2=r2=data_addr, arg3=r3=len)
        - 0x04: SYSCALL_FILE_READ  - Read from file (arg1=r1=path_addr, arg2=r2=data_addr, arg3=r3=len)
        - 0x05: SYSCALL_EXIT   - Exit with status (arg1=r1=status)
        - 0x06: SYSCALL_DEBUG  - Debug print (arg1=r1=value)
        - 0x10-0xFF: Reserved for Geometry OS spatial services

        Returns: syscall result (0=success, -1=error)
        """
        # Map syscall numbers to hypervisor service addresses
        # These match the GeOS hypervisor bridge constants
        SPATIAL_REGISTRY_BASE = 0x8009_0000

        if syscall_num == 0x01:  # SYSCALL_WRITE
            # Write r2 bytes from address r1 to output buffer
            addr = self.registers[1]
            length = self.registers[2]
            for i in range(length):
                val = self._mem_read(image, addr + i)
                self.output.append(val)
            print(f"[SYSCALL] WRITE: {length} bytes from addr {addr}")
            return 0

        elif syscall_num == 0x02:  # SYSCALL_READ
            # Read r2 bytes from input to address r1 (stub - returns zeros)
            addr = self.registers[1]
            length = self.registers[2]
            for i in range(length):
                self._mem_write(image, addr + i, 0)
            print(f"[SYSCALL] READ: {length} bytes to addr {addr} (stub)")
            return 0

        elif syscall_num == 0x03:  # SYSCALL_FILE_WRITE
            # Write file: r1=path_addr, r2=data_addr, r3=len (stub)
            path_addr = self.registers[1]
            data_addr = self.registers[2]
            length = self.registers[3]
            print(f"[SYSCALL] FILE_WRITE: {length} bytes from addr {data_addr} to path at {path_addr} (stub)")
            return 0

        elif syscall_num == 0x04:  # SYSCALL_FILE_READ
            # Read file: r1=path_addr, r2=data_addr, r3=len (stub)
            path_addr = self.registers[1]
            data_addr = self.registers[2]
            length = self.registers[3]
            print(f"[SYSCALL] FILE_READ: {length} bytes from path at {path_addr} to addr {data_addr} (stub)")
            return 0

        elif syscall_num == 0x05:  # SYSCALL_EXIT
            # Exit with status in r1
            status = self.registers[1]
            print(f"[SYSCALL] EXIT: status={status}")
            self.running = False
            return status

        elif syscall_num == 0x06:  # SYSCALL_DEBUG
            # Debug print: r1=value
            value = self.registers[1]
            print(f"[SYSCALL] DEBUG: r1={value}")
            return 0

        elif 0x10 <= syscall_num <= 0xFF:
            # Reserved for GeOS spatial services - bridge to hypervisor
            print(f"[SYSCALL] GEOS_SERVICE 0x{syscall_num:02X}: dispatched to MMIO 0x{SPATIAL_REGISTRY_BASE:08X}")
            return 0

        else:
            print(f"[SYSCALL] UNKNOWN: syscall_num=0x{syscall_num:02X}")
            return -1

    def run(self, image: np.ndarray, max_instructions: int = 1000):
        self.running = True
        n = 0
        while self.running and n < max_instructions:
            self.step(image)
            n += 1
        return n


def demo():
    opcode_map = OpcodeMapV2()
    print("Opcode -> RGB (collision-safe):")
    for op in OpcodeMapV2.OPCODES:
        print(f"  {op:4} -> {opcode_map.opcode_to_rgb(op)}")

    # NOTE: CMP writes its result to r0 (the flag register), so the loop
    # counter must live elsewhere (r5) to avoid clobbering it.
    program = [
        "LDI r5 0",       # 0: counter = 0
        "LDI r1 5",       # 1: limit = 5
        "CMP r5 r1",      # 2: r0 = (counter == limit)
        "JZ 0,1",         # 3: if equal, jump to HALT (instr 8 -> row1,col0)
        "PRT r5",         # 4: print counter
        "LDI r2 1",       # 5: r2 = 1
        "ADD r5 r2",      # 6: counter += 1
        "JMP 2,0",        # 7: jump back to CMP (instr 2 -> row0,col2)
        "HALT",           # 8
    ]

    assembler = GlyphAssemblerV2(opcode_map)
    image = assembler.assemble(program, width_instrs=8)
    print(f"Assembled: {len(program)} instrs -> image {image.shape}")

    cpu = GlyphCPUv2(opcode_map, cols_instrs=8)
    n = cpu.run(image)
    print(f"Halted after {n} steps. Final registers[0:3]={cpu.registers[:3]}")
    print(f"Output: {cpu.output}")

    opcode_map.close()


if __name__ == '__main__':
    demo()
