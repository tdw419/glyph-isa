#!/usr/bin/env python3
"""Host-side RV64I decoder for pre-decoded-op execution (basic-block-threading port).

Mirrors the decode surface of tools/SPATIAL_RV64I.wgsl — decode_and_execute() plus
expand_rvc() — so the GPU executes pre-decoded ops (op/rd/rs1/rs2/imm/aux, with
sign-extension and RVC expansion done once here) instead of re-extracting bitfields
and sign-extending on every instruction, every step.

The layout of DecodedOp (9 u32s, indexed by PHYSICAL halfword = byte_addr/2):
    [0] op    — opcode enum (see OP_* below); 0xFFFFFFFF = not decoded (runtime fallback)
    [1] rd
    [2] rs1
    [3] rs2
    [4] imm   — pre-sign-extended to 32 bits (or shamt / LUI 20-bit<<12); execute side sexts to 64
    [5] aux   — per-op auxiliary (funct3 for loads/stores/branches, csr_addr, amo_op, dword flag)
    [6] raw   — the expanded 32-bit instruction word, for validation against self-modifying code
    [7] len   — 2 or 4 (byte length; mirrors fetch()'s instr_len)
    [8] epoch — decode epoch tag; mismatch triggers runtime fallback (GPU-side store invalidation)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Opcode enum (must match the WGSL switch in SPATIAL_RV64I.wgsl execute_decoded)
# ---------------------------------------------------------------------------
OP_INVALID = 0xFFFFFFFF

# OP-IMM (0x13)
OP_ADDI, OP_SLTI, OP_SLTIU, OP_XORI, OP_ORI, OP_ANDI, OP_SLLI, OP_SRLI, OP_SRAI = range(9)
# Branch (0x63)
OP_BEQ, OP_BNE, OP_BLT, OP_BGE, OP_BLTU, OP_BGEU = range(9, 15)
# Load (0x03): aux = funct3
OP_LB, OP_LH, OP_LW, OP_LBU, OP_LHU, OP_LD, OP_LWU = range(15, 22)
# Store (0x23): aux = funct3
OP_SB, OP_SH, OP_SW, OP_SD = range(22, 26)
# M extension 64-bit (0x33, funct7=0x01)
OP_MUL, OP_MULH, OP_MULHSU, OP_MULHU, OP_DIV, OP_DIVU, OP_REM, OP_REMU = range(26, 34)
# R-type 64-bit (0x33)
OP_ADD, OP_SUB, OP_SLL, OP_SLT, OP_SLTU, OP_XOR, OP_SRL, OP_SRA, OP_OR, OP_AND = range(34, 44)
# M extension W (0x3B, funct7=0x01)
OP_MULW, OP_DIVW, OP_DIVUW, OP_REMW, OP_REMUW = range(44, 49)
# R-type W (0x3B)
OP_ADDW, OP_SUBW, OP_SLLW, OP_SRLW, OP_SRAW = range(49, 54)
# I-type W (0x1B)
OP_ADDIW, OP_SLLIW, OP_SRLIW, OP_SRAIW = range(54, 58)
# A extension (0x2F): aux bit0 = is_dword; aux[5:1] = amo_op for AMO ops
OP_LR, OP_SC = 58, 59
OP_AMOSWAP, OP_AMOADD, OP_AMOXOR, OP_AMOAND, OP_AMOOR, OP_AMOMIN, OP_AMOMAX, OP_AMOMINU, OP_AMOMAXU = range(60, 69)
# LUI / AUIPC / JAL / JALR / FENCE
OP_LUI, OP_AUIPC, OP_JAL, OP_JALR, OP_FENCE = 69, 70, 71, 72, 73
# SYSTEM (0x73): aux = csr_addr for CSR ops
OP_ECALL, OP_EBREAK, OP_MRET, OP_SRET, OP_WFI, OP_SFENCE_VMA = range(74, 80)
OP_CSRRW, OP_CSRRS, OP_CSRRC, OP_CSRRWI, OP_CSRRSI, OP_CSRRCI = range(80, 86)

OP_COUNT = 86

# ---------------------------------------------------------------------------
# Sign-extension helpers (mirror SPATIAL_RV64I.wgsl)
# ---------------------------------------------------------------------------
def sext12(v: int) -> int:
    if v & 0x800:
        return v | 0xFFFFF000
    return v

def sext13(v: int) -> int:
    if v & 0x1000:
        return v | 0xFFFFE000
    return v

def sext21(v: int) -> int:
    if v & 0x100000:
        return v | 0xFFE00000
    return v

def rvc_sext6(v: int) -> int:
    if v & 0x20:
        return (v | 0xFFFFFFC0) & 0xFFFFFFFF
    return v & 0xFFFFFFFF

def rvc_sext10(v: int) -> int:
    if v & 0x200:
        return (v | 0xFFFFFC00) & 0xFFFFFFFF
    return v & 0xFFFFFFFF

def rvc_sext18(v: int) -> int:
    if v & 0x20000:
        return (v | 0xFFFC0000) & 0xFFFFFFFF
    return v & 0xFFFFFFFF

def w32(v: int) -> int:
    """Wrap a Python int to u32 semantics (WGSL shifts/masks wrap at 32 bits)."""
    return v & 0xFFFFFFFF

def rvc_cj_offset(c: int) -> int:
    b11 = (c >> 12) & 1
    b4 = (c >> 11) & 1
    b98 = (c >> 9) & 3
    b10 = (c >> 8) & 1
    b6 = (c >> 7) & 1
    b7 = (c >> 6) & 1
    b31 = (c >> 3) & 7
    b5 = (c >> 2) & 1
    imm = (b11 << 11) | (b10 << 10) | (b98 << 8) | (b7 << 7) | (b6 << 6) | (b5 << 5) | (b4 << 4) | (b31 << 1)
    if b11:
        imm |= 0xFFFFF000
    return imm

def rvc_cb_offset(c: int) -> int:
    b8 = (c >> 12) & 1
    b43 = (c >> 10) & 3
    b76 = (c >> 5) & 3
    b21 = (c >> 3) & 3
    b5 = (c >> 2) & 1
    imm = (b8 << 8) | (b76 << 6) | (b5 << 5) | (b43 << 3) | (b21 << 1)
    if b8:
        imm |= 0xFFFFFE00
    return imm

def rvc_encode_jal(rd: int, imm: int) -> int:
    imm20 = (imm >> 20) & 1
    imm101 = (imm >> 1) & 0x3FF
    imm11 = (imm >> 11) & 1
    imm1912 = (imm >> 12) & 0xFF
    return (imm20 << 31) | (imm101 << 21) | (imm11 << 20) | (imm1912 << 12) | (rd << 7) | 0x6F

def rvc_encode_branch(funct3: int, rs1: int, rs2: int, imm: int) -> int:
    imm12 = (imm >> 12) & 1
    imm11 = (imm >> 11) & 1
    imm105 = (imm >> 5) & 0x3F
    imm41 = (imm >> 1) & 0xF
    return (imm12 << 31) | (imm105 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm41 << 8) | (imm11 << 7) | 0x63

def expand_rvc(c: int) -> int:
    """Mirror of SPATIAL_RV64I.wgsl expand_rvc(). Returns expanded 32-bit word."""
    op = c & 0x3
    funct3 = (c >> 13) & 0x7
    rd_prime = 8 + ((c >> 2) & 0x7)
    rs1_prime = 8 + ((c >> 7) & 0x7)
    rs2_prime = 8 + ((c >> 2) & 0x7)
    rd_rs1_full = (c >> 7) & 0x1F
    rs2_full = (c >> 2) & 0x1F

    if op == 0:
        if funct3 == 0:
            uimm = (((c >> 11) & 0x3) << 4) | (((c >> 7) & 0xF) << 6) | (((c >> 6) & 0x1) << 2) | (((c >> 5) & 0x1) << 3)
            if uimm == 0:
                return 0
            return w32((uimm << 20) | (2 << 15) | (0 << 12) | (rd_prime << 7) | 0x13)
        elif funct3 == 2:
            off = (((c >> 6) & 0x1) << 2) | (((c >> 10) & 0x7) << 3) | (((c >> 5) & 0x1) << 6)
            return w32((off << 20) | (rs1_prime << 15) | (2 << 12) | (rd_prime << 7) | 0x03)
        elif funct3 == 3:
            off = (((c >> 10) & 0x7) << 3) | (((c >> 5) & 0x3) << 6)
            return w32((off << 20) | (rs1_prime << 15) | (3 << 12) | (rd_prime << 7) | 0x03)
        elif funct3 == 6:
            off = (((c >> 6) & 0x1) << 2) | (((c >> 10) & 0x7) << 3) | (((c >> 5) & 0x1) << 6)
            imm5 = off & 0x1F
            imm7 = (off >> 5) & 0x7F
            return (imm7 << 25) | (rs2_prime << 20) | (rs1_prime << 15) | (2 << 12) | (imm5 << 7) | 0x23
        elif funct3 == 7:
            off = (((c >> 10) & 0x7) << 3) | (((c >> 5) & 0x3) << 6)
            imm5 = off & 0x1F
            imm7 = (off >> 5) & 0x7F
            return (imm7 << 25) | (rs2_prime << 20) | (rs1_prime << 15) | (3 << 12) | (imm5 << 7) | 0x23
        return 0
    elif op == 1:
        if funct3 == 0:
            imm = rvc_sext6((((c >> 12) & 1) << 5) | ((c >> 2) & 0x1F))
            return w32((imm << 20) | (rd_rs1_full << 15) | (0 << 12) | (rd_rs1_full << 7) | 0x13)
        elif funct3 == 1:
            imm = rvc_sext6((((c >> 12) & 1) << 5) | ((c >> 2) & 0x1F))
            return w32((imm << 20) | (rd_rs1_full << 15) | (0 << 12) | (rd_rs1_full << 7) | 0x1B)
        elif funct3 == 2:
            imm = rvc_sext6((((c >> 12) & 1) << 5) | ((c >> 2) & 0x1F))
            return w32((imm << 20) | (0 << 15) | (0 << 12) | (rd_rs1_full << 7) | 0x13)
        elif funct3 == 3:
            if rd_rs1_full == 2:
                imm = rvc_sext10((((c >> 12) & 1) << 9) | (((c >> 3) & 0x3) << 7) | (((c >> 5) & 1) << 6) | (((c >> 2) & 1) << 5) | (((c >> 6) & 1) << 4))
                return w32((imm << 20) | (2 << 15) | (0 << 12) | (2 << 7) | 0x13)
            else:
                imm = rvc_sext18((((c >> 12) & 1) << 17) | (((c >> 2) & 0x1F) << 12))
                return w32((imm & 0xFFFFF000) | (rd_rs1_full << 7) | 0x37)
        elif funct3 == 4:
            funct2_hi = (c >> 10) & 0x3
            if funct2_hi == 0 or funct2_hi == 1:
                shamt = (((c >> 12) & 1) << 5) | ((c >> 2) & 0x1F)
                funct6 = 0x10 if funct2_hi == 1 else 0
                return w32((funct6 << 26) | (shamt << 20) | (rs1_prime << 15) | (5 << 12) | (rs1_prime << 7) | 0x13)
            elif funct2_hi == 2:
                imm = rvc_sext6((((c >> 12) & 1) << 5) | ((c >> 2) & 0x1F))
                return w32((imm << 20) | (rs1_prime << 15) | (7 << 12) | (rs1_prime << 7) | 0x13)
            else:
                is_word = ((c >> 12) & 1) != 0
                funct2_lo = (c >> 5) & 0x3
                if is_word:
                    if funct2_lo == 0:
                        return (0x20 << 25) | (rs2_prime << 20) | (rs1_prime << 15) | (0 << 12) | (rs1_prime << 7) | 0x3B
                    elif funct2_lo == 1:
                        return (0 << 25) | (rs2_prime << 20) | (rs1_prime << 15) | (0 << 12) | (rs1_prime << 7) | 0x3B
                    return 0
                else:
                    if funct2_lo == 0:
                        return (0x20 << 25) | (rs2_prime << 20) | (rs1_prime << 15) | (0 << 12) | (rs1_prime << 7) | 0x33
                    elif funct2_lo == 1:
                        return (rs2_prime << 20) | (rs1_prime << 15) | (4 << 12) | (rs1_prime << 7) | 0x33
                    elif funct2_lo == 2:
                        return (rs2_prime << 20) | (rs1_prime << 15) | (6 << 12) | (rs1_prime << 7) | 0x33
                    else:
                        return (rs2_prime << 20) | (rs1_prime << 15) | (7 << 12) | (rs1_prime << 7) | 0x33
        elif funct3 == 5:
            return rvc_encode_jal(0, rvc_cj_offset(c))
        elif funct3 == 6:
            return rvc_encode_branch(0, rs1_prime, 0, rvc_cb_offset(c))
        elif funct3 == 7:
            return rvc_encode_branch(1, rs1_prime, 0, rvc_cb_offset(c))
        return 0
    elif op == 2:
        if funct3 == 0:
            shamt = (((c >> 12) & 1) << 5) | ((c >> 2) & 0x1F)
            return w32((shamt << 20) | (rd_rs1_full << 15) | (1 << 12) | (rd_rs1_full << 7) | 0x13)
        elif funct3 == 2:
            off = (((c >> 4) & 0x7) << 2) | (((c >> 12) & 1) << 5) | (((c >> 2) & 0x3) << 6)
            return w32((off << 20) | (2 << 15) | (2 << 12) | (rd_rs1_full << 7) | 0x03)
        elif funct3 == 3:
            off = (((c >> 5) & 0x3) << 3) | (((c >> 12) & 1) << 5) | (((c >> 2) & 0x7) << 6)
            return w32((off << 20) | (2 << 15) | (3 << 12) | (rd_rs1_full << 7) | 0x03)
        elif funct3 == 4:
            bit12 = (c >> 12) & 1
            if bit12 == 0:
                if rs2_full == 0:
                    return (0 << 20) | (rd_rs1_full << 15) | (0 << 12) | (0 << 7) | 0x67
                else:
                    return (rs2_full << 20) | (0 << 15) | (0 << 12) | (rd_rs1_full << 7) | 0x33
            else:
                if rd_rs1_full == 0 and rs2_full == 0:
                    return (1 << 20) | 0x73
                elif rs2_full == 0:
                    return (0 << 20) | (rd_rs1_full << 15) | (0 << 12) | (1 << 7) | 0x67
                else:
                    return (rs2_full << 20) | (rd_rs1_full << 15) | (0 << 12) | (rd_rs1_full << 7) | 0x33
        elif funct3 == 6:
            off = (((c >> 9) & 0xF) << 2) | (((c >> 7) & 0x3) << 6)
            imm5 = off & 0x1F
            imm7 = (off >> 5) & 0x7F
            return (imm7 << 25) | (rs2_full << 20) | (2 << 15) | (2 << 12) | (imm5 << 7) | 0x23
        elif funct3 == 7:
            off = (((c >> 10) & 0x7) << 3) | (((c >> 7) & 0x7) << 6)
            imm5 = off & 0x1F
            imm7 = (off >> 5) & 0x7F
            return (imm7 << 25) | (rs2_full << 20) | (2 << 15) | (3 << 12) | (imm5 << 7) | 0x23
        return 0
    return 0


def decode_instruction(instr: int):
    """Decode one expanded 32-bit instruction.

    Returns (op, rd, rs1, rs2, imm, aux) or None if invalid/unsupported
    (caller marks the slot OP_INVALID -> runtime fallback).
    """
    opcode = instr & 0x7F
    rd = (instr >> 7) & 0x1F
    funct3 = (instr >> 12) & 0x7
    rs1 = (instr >> 15) & 0x1F
    rs2 = (instr >> 20) & 0x1F
    funct7 = (instr >> 25) & 0x7F
    funct6 = (instr >> 26) & 0x3F

    if opcode == 0x13:
        imm = sext12(instr >> 20)
        shamt = (instr >> 20) & 0x3F
        if funct3 == 0:
            return OP_ADDI, rd, rs1, 0, imm, 0
        if funct3 == 2:
            return OP_SLTI, rd, rs1, 0, imm, 0
        if funct3 == 3:
            return OP_SLTIU, rd, rs1, 0, imm, 0
        if funct3 == 4:
            return OP_XORI, rd, rs1, 0, imm, 0
        if funct3 == 6:
            return OP_ORI, rd, rs1, 0, imm, 0
        if funct3 == 7:
            return OP_ANDI, rd, rs1, 0, imm, 0
        if funct3 == 1 and funct6 == 0:
            return OP_SLLI, rd, rs1, 0, shamt, 0
        if funct3 == 5 and funct6 == 0:
            return OP_SRLI, rd, rs1, 0, shamt, 0
        if funct3 == 5 and funct6 == 0x10:
            return OP_SRAI, rd, rs1, 0, shamt, 0
        return None

    if opcode == 0x63:
        imm12 = ((instr >> 31) << 12) | (((instr >> 7) & 1) << 11) | (((instr >> 25) & 0x3F) << 5) | (((instr >> 8) & 0xF) << 1)
        imm = sext13(imm12)
        mapping = {0: OP_BEQ, 1: OP_BNE, 4: OP_BLT, 5: OP_BGE, 6: OP_BLTU, 7: OP_BGEU}
        if funct3 in mapping:
            return mapping[funct3], 0, rs1, rs2, imm, 0
        return None

    if opcode == 0x03:
        imm = sext12(instr >> 20)
        mapping = {0: OP_LB, 1: OP_LH, 2: OP_LW, 4: OP_LBU, 5: OP_LHU, 3: OP_LD, 6: OP_LWU}
        if funct3 in mapping:
            return mapping[funct3], rd, rs1, 0, imm, funct3
        return None

    if opcode == 0x23:
        imm5 = (instr >> 7) & 0x1F
        imm7 = (instr >> 25) & 0x7F
        imm = sext12((imm7 << 5) | imm5)
        mapping = {0: OP_SB, 1: OP_SH, 2: OP_SW, 3: OP_SD}
        if funct3 in mapping:
            return mapping[funct3], 0, rs1, rs2, imm, funct3
        return None

    if opcode == 0x33 and funct7 == 0x01:
        mapping = {0: OP_MUL, 1: OP_MULH, 2: OP_MULHSU, 3: OP_MULHU, 4: OP_DIV, 5: OP_DIVU, 6: OP_REM, 7: OP_REMU}
        if funct3 in mapping:
            return mapping[funct3], rd, rs1, rs2, 0, 0
        return None

    if opcode == 0x33:
        if funct3 == 0 and funct7 == 0:
            return OP_ADD, rd, rs1, rs2, 0, 0
        if funct3 == 0 and funct7 == 0x20:
            return OP_SUB, rd, rs1, rs2, 0, 0
        if funct3 == 1 and funct7 == 0:
            return OP_SLL, rd, rs1, rs2, 0, 0
        if funct3 == 2 and funct7 == 0:
            return OP_SLT, rd, rs1, rs2, 0, 0
        if funct3 == 3 and funct7 == 0:
            return OP_SLTU, rd, rs1, rs2, 0, 0
        if funct3 == 4 and funct7 == 0:
            return OP_XOR, rd, rs1, rs2, 0, 0
        if funct3 == 5 and funct7 == 0:
            return OP_SRL, rd, rs1, rs2, 0, 0
        if funct3 == 5 and funct7 == 0x20:
            return OP_SRA, rd, rs1, rs2, 0, 0
        if funct3 == 6 and funct7 == 0:
            return OP_OR, rd, rs1, rs2, 0, 0
        if funct3 == 7 and funct7 == 0:
            return OP_AND, rd, rs1, rs2, 0, 0
        return None

    if opcode == 0x3B and funct7 == 0x01:
        mapping = {0: OP_MULW, 4: OP_DIVW, 5: OP_DIVUW, 6: OP_REMW, 7: OP_REMUW}
        if funct3 in mapping:
            return mapping[funct3], rd, rs1, rs2, 0, 0
        return None

    if opcode == 0x3B:
        if funct3 == 0 and funct7 == 0:
            return OP_ADDW, rd, rs1, rs2, 0, 0
        if funct3 == 0 and funct7 == 0x20:
            return OP_SUBW, rd, rs1, rs2, 0, 0
        if funct3 == 1 and funct7 == 0:
            return OP_SLLW, rd, rs1, rs2, 0, 0
        if funct3 == 5 and funct7 == 0:
            return OP_SRLW, rd, rs1, rs2, 0, 0
        if funct3 == 5 and funct7 == 0x20:
            return OP_SRAW, rd, rs1, rs2, 0, 0
        return None

    if opcode == 0x1B:
        imm = sext12(instr >> 20)
        shamt5 = (instr >> 20) & 0x1F
        top7 = instr >> 25
        if funct3 == 0:
            return OP_ADDIW, rd, rs1, 0, imm, 0
        if funct3 == 1 and top7 == 0:
            return OP_SLLIW, rd, rs1, 0, shamt5, 0
        if funct3 == 5 and top7 == 0:
            return OP_SRLIW, rd, rs1, 0, shamt5, 0
        if funct3 == 5 and top7 == 0x20:
            return OP_SRAIW, rd, rs1, 0, shamt5, 0
        return None

    if opcode == 0x2F:
        if funct3 != 2 and funct3 != 3:
            return None
        is_dword = 1 if funct3 == 3 else 0
        amo_op = funct7 >> 2
        if amo_op == 0x02:
            return OP_LR, rd, rs1, 0, 0, is_dword
        if amo_op == 0x03:
            return OP_SC, rd, rs1, rs2, 0, is_dword
        mapping = {0x00: OP_AMOADD, 0x01: OP_AMOSWAP, 0x04: OP_AMOXOR, 0x08: OP_AMOOR,
                   0x0C: OP_AMOAND, 0x10: OP_AMOMIN, 0x14: OP_AMOMAX, 0x18: OP_AMOMINU, 0x1C: OP_AMOMAXU}
        if amo_op in mapping:
            return mapping[amo_op], rd, rs1, rs2, 0, is_dword
        return None

    if opcode == 0x37:
        return OP_LUI, rd, 0, 0, instr & 0xFFFFF000, 0

    if opcode == 0x17:
        return OP_AUIPC, rd, 0, 0, instr & 0xFFFFF000, 0

    if opcode == 0x67:
        imm = sext12(instr >> 20)
        if funct3 == 0:
            return OP_JALR, rd, rs1, 0, imm, 0
        return None

    if opcode == 0x0F:
        if funct3 == 0 or funct3 == 1:
            return OP_FENCE, 0, 0, 0, 0, 0
        return None

    if opcode == 0x73:
        funct12 = instr >> 20
        if funct3 == 0:
            if funct12 == 0:
                return OP_ECALL, 0, 0, 0, 0, 0
            if funct12 == 1:
                return OP_EBREAK, 0, 0, 0, 0, 0
            if funct12 == 0x302:
                return OP_MRET, 0, 0, 0, 0, 0
            if funct12 == 0x102:
                return OP_SRET, 0, 0, 0, 0, 0
            if funct12 == 0x105:
                return OP_WFI, 0, 0, 0, 0, 0
            if (funct12 >> 5) == 0x09:
                return OP_SFENCE_VMA, 0, rs1, rs2, 0, 0
            return None
        csr_addr = instr >> 20
        mapping = {1: OP_CSRRW, 2: OP_CSRRS, 3: OP_CSRRC, 5: OP_CSRRWI, 6: OP_CSRRSI, 7: OP_CSRRCI}
        if funct3 in mapping:
            return mapping[funct3], rd, rs1, 0, 0, csr_addr
        return None

    if opcode == 0x6F:
        imm20 = ((instr >> 31) << 20) | (((instr >> 12) & 0xFF) << 12) | (((instr >> 20) & 1) << 11) | (((instr >> 21) & 0x3FF) << 1)
        return OP_JAL, rd, 0, 0, sext21(imm20), 0

    return None


def decode_image(linear_words, byte_offset=0, byte_len=None, epoch=0):
    """Decode a linear (pre-Hilbert) memory image into a DecodedOp array.

    Per-slot independent decode: every halfword slot is decoded from its own
    bytes as if an instruction started there (slot = potential instruction
    start at byte 2*slot). No linear-sweep state, so data regions (PE headers,
    jump tables, embedded constants) can never misalign the decode of real
    code that follows them — each fetched address resolves to the op for the
    exact bytes at that address. Slots that are actually mid-instruction are
    never fetched as instruction starts by valid code, so their (garbage) ops
    are never used.

    linear_words: np.uint32 array of the memory image in linear order.
    byte_offset:  linear byte offset where the decoded range STARTS.
    byte_len:     number of bytes to decode FROM byte_offset (default: to the end).
    epoch:        current decode epoch (for GPU-side store invalidation).

    Returns np.ndarray shape (n_halfwords, 9), uint32 — one DecodedOp per
    halfword slot, indexed by GLOBAL slot (slot = byte_addr/2, same indexing
    the shader uses with phys/2). Slots before byte_offset or past the range
    are left as OP_INVALID.
    """
    total_words = linear_words.shape[0]
    total_bytes = total_words * 4
    start = byte_offset
    end = total_bytes if byte_len is None else byte_offset + byte_len
    end = max(start, min(end, total_bytes))

    n_slots = (end + 1) // 2  # global slot count covering bytes [0, end)
    ops = np.full((n_slots, 9), OP_INVALID, dtype=np.uint32)

    # Precompute the raw halfword stream so each slot decodes independently.
    # halfword h (byte 2h) lives in linear word 2h//4, shifted by (2h % 4)*8.
    # Only the [start, end) range is materialized; the shader indexes decoded_ops
    # globally, so returned slots keep their global index.
    start_hw = start >> 1
    end_hw = (end + 1) // 2
    n_hw = end_hw - start_hw
    halfwords = np.zeros(n_hw, dtype=np.uint32)
    for i in range(n_hw):
        h = start_hw + i
        b = 2 * h
        w = b >> 2
        if w >= total_words:
            break
        word = int(linear_words[w])
        halfwords[i] = (word >> ((b & 2) * 8)) & 0xFFFF

    for i in range(n_hw):
        slot = start_hw + i
        b = 2 * slot
        if b + 1 >= end:
            break
        half0 = int(halfwords[i])
        if (half0 & 0x3) == 0x3:
            # 32-bit instruction start at this slot: bytes 2*slot .. 2*slot+3
            if b + 3 >= end:
                break
            half1 = int(halfwords[i + 1]) if i + 1 < n_hw else 0
            instr = (half1 << 16) | half0
            decoded = decode_instruction(instr)
            if decoded is not None:
                op, rd, rs1, rs2, imm, aux = decoded
                ops[slot] = [op & 0xFFFFFFFF, rd, rs1, rs2, imm & 0xFFFFFFFF, aux, instr, 4, epoch]
            else:
                ops[slot, 0] = OP_INVALID
                ops[slot, 6] = instr
                ops[slot, 7] = 4
                ops[slot, 8] = epoch
        else:
            # 16-bit RVC instruction start at this slot. raw stores the RAW halfword
            # (not the expanded word) so the shader can validate against memory with a
            # single u16 compare — no expand_rvc needed on the fast path.
            instr = expand_rvc(half0)
            if instr != 0:
                decoded = decode_instruction(instr)
                if decoded is not None:
                    op, rd, rs1, rs2, imm, aux = decoded
                    ops[slot] = [op & 0xFFFFFFFF, rd, rs1, rs2, imm & 0xFFFFFFFF, aux, half0, 2, epoch]
                else:
                    ops[slot, 0] = OP_INVALID
                    ops[slot, 6] = half0
                    ops[slot, 7] = 2
                    ops[slot, 8] = epoch
            else:
                ops[slot, 0] = OP_INVALID
                ops[slot, 7] = 2
                ops[slot, 8] = epoch

    return ops
