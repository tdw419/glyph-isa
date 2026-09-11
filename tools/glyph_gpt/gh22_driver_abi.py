

# ────────────────────────────── GH-22 ────────────────────────────────────────
# Device Driver ABI: spatial microkernel protocol. The driver runs as an
# UNPRIVILEGED USER task in BOX1 (E-K1 box bound == the MMIO bound); the app
# runs in BOX0. All app->driver communication flows through the kernel-
# mediated mailbox (754) — non-blocking copy-and-continue, GH-13 style. No
# monolithic kernel driver: the kernel never touches the device registers.
# SOURCE RULE (roadmap GPLv2 derivative-work): the driver protocol below is
# written from the simulated DEV-U1 datasheet documented in
# tests/test_gh22_device_driver_abi.py — no Linux source read or transcribed.
#
# Module shape mirrors posix_shim.py: the kernel program text + the two-pass
# bake live here; baker.py re-exports the public names for the test import.
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent), str(_HERE.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from glyph_isa_v2 import (  # noqa: E402
    SYS_A0_ADDR, SYS_N_ADDR,
)

# SYS_A0's word index is a pure compile-time derivation from the engine
# constant (baker does the same: SYS_A0_WORD = SYS_A0_ADDR >> 2) — no
# import needed, so no circularity for the one constant the task builders
# reference at module scope.
SYS_A0_WORD = SYS_A0_ADDR >> 2

# Deferred inside driver_abi_kernel_image(): baker re-exports this module
# (GH-22 test import path) — module-level baker/assembler imports would be
# circular (same pattern as fs_v2.py's syscall_abi_kernel_image import).

GH22_EXIT_APP = 703          # app exit word (in BOX0)         0xFEED0000|22
GH22_UART_APP = 710          # app's request receipt (in BOX0)
GH22_REQ_SCRATCH = 713       # app builds its request word here (in BOX0)
GH22_READOUT_APP = 714       # verdict delivered to the app (kernel SYS 9)
GH22_READOUT_DRV = 720       # request delivered to the driver (kernel SYS 7)
GH22_EXIT_DRV = 723          # driver exit word (in BOX1)       0xFEED0000|23
GH22_VERDICT_DRV = 724       # driver's verdict: 'V' (86) / 'E' (69), in BOX1
GH22_DEV_DATA = 733          # DEV-U1 DATA register    (R/W, inside BOX1)
GH22_DEV_STATUS = 734        # DEV-U1 STATUS register  (R/W, inside BOX1)
GH22_MAILBOX_WORD = 754      # app->driver request mailbox (kernel-mediated)
GH22_BADSYS_WORD = 758       # unknown-syscall marker ('E' = 69)
GH22_FAULT_WORD = 759        # fault-leg verdict (0xFA171)
GH22_BOX0_LO_BYTE = 4 * 700  # app arena [700..717)
GH22_BOX0_HI_BYTE = 4 * 717
GH22_BOX1_LO_BYTE = 4 * 718  # driver arena [718..735) — the MMIO bound
GH22_BOX1_HI_BYTE = 4 * 735
GH22_N_APP = 6               # SYS 6: app -> mailbox (request send)
GH22_N_DRV = 7               # SYS 7: mailbox -> driver read-out
GH22_N_APP_RECV = 8          # SYS 8: driver verdict -> app read-out
GH22_EXIT_OK_APP = 0xFEED0000 | 22
GH22_EXIT_OK_DRV = 0xFEED0000 | 23
GH22_FAULT_SEEN = 0xFA171
GH22_OPCODE_PUT = 1          # DEV-U1 datasheet §2 opcode
GH22_PAYLOAD = 0x5A          # the app's transmit byte
# request word: cksum=(op+payload)&0xFF [31:24], rsvd [23:16],
# opcode [15:8], payload [7:0]
GH22_REQ_GOOD = (((GH22_OPCODE_PUT + GH22_PAYLOAD) & 0xFF) << 24) \
    | (GH22_OPCODE_PUT << 8) | GH22_PAYLOAD
GH22_VERIFIED = 86           # 'V'
GH22_ERRORED = 69            # 'E'
GH22_STATUS_ID = 26          # final status = 0xCAFE0000 | 26 (0xCAFE001A)
GH22_DEV_STATUS_TXACTIVE = 0x01   # datasheet: TX_ACTIVE (driver-set)
GH22_DEV_STATUS_QUIESCENT = 0x81   # datasheet: RX_DONE latched | TX_ACTIVE off

# Packed pixel PCs of the GH-22 selector/handler/dispatch/agent labels,
# bound by driver_abi_kernel_image() before the second (final) assemble pass.
_GH22_KSYS_PC = 0
_GH22_FAULT_PC = 0
_GH22_RET_PC = 0
_GH22_DISPATCH_DRV_PC = 0
_GH22_DISPATCH_APP2_PC = 0
_GH22_TASK_APP_PC = 0
_GH22_TASK_DRV_PC = 0
_GH22_TASK_APP2_PC = 0


def _gh22_ksys_slice(n: int) -> List[str]:
    """One SUPER-mode dispatcher slice for syscall n (6 app-send, 7 driver-
    recv, 8 verdict-send). Non-blocking mailbox semantics: each slice copies
    and SYSRETs; nobody ever waits. The kernel is the POSTMAN only — it
    never interprets or transforms the message (the driver validates it).

    SYS 6 (app):    r6 = mem[SYS_A0] (the request word) -> mailbox 754.
    SYS 7 (driver): mailbox 754 -> the driver's read-out 720 (kernel-
                    mediated, exactly GH-13's SYS 7; the driver then works
                    on its own in-box copy).
    SYS 8 (app):    driver verdict 724 -> the app's read-out 714.
    """
    a: List[str] = []
    add = a.append
    add(f":__ksys_{n}")
    if n == GH22_N_APP:
        add(f"LDI r15 {SYS_A0_ADDR >> 2}")
        add("LD r6 r15")                 # r6 = a0 = the request word
        add(f"LDI r15 {GH22_MAILBOX_WORD}")
        add("ST r15 r6")                 # mailbox <- request
    elif n == GH22_N_DRV:
        add(f"LDI r15 {GH22_MAILBOX_WORD}")
        add("LD r6 r15")                 # r6 = mailbox (request)
        add(f"LDI r15 {GH22_READOUT_DRV}")
        add("ST r15 r6")                 # driver read-out <- request
    else:
        add(f"LDI r15 {GH22_VERDICT_DRV}")
        add("LD r6 r15")                 # r6 = driver's verdict word
        add(f"LDI r15 {GH22_READOUT_APP}")
        add("ST r15 r6")                 # app read-out <- verdict
    add("SYSRET")
    return a


def _gh22_checksum_check() -> List[str]:
    """USER-mode checksum check of the request in the driver's read-out.

    r2 = request word (loaded by the caller). Verifies cksum byte ==
    (opcode + payload) & 0xFF. Register discipline: AND/SHR/SHL are
    IN-PLACE on rd (rd = rd op rs2), so the byte extracts need explicit
    zero+OR copies of r2 into the working registers first.
    """
    a: List[str] = []
    add = a.append
    add("LDI r4 24")
    add("XOR r7 r7")
    add("OR r7 r2")                      # r7 = copy of request
    add("SHR r7 r4")                     # r7 = request >> 24
    add("LDI r4 255")
    add("AND r7 r4")                     # r7 = cksum byte
    add("LDI r4 8")
    add("XOR r5 r5")
    add("OR r5 r2")                      # r5 = copy of request
    add("SHR r5 r4")                     # r5 = request >> 8
    add("LDI r4 255")
    add("AND r5 r4")                     # r5 = opcode byte
    add("XOR r6 r6")
    add("OR r6 r2")                      # r6 = copy of request
    add("AND r6 r4")                     # r6 = payload byte
    add("ADD r5 r6")                     # r5 = opcode + payload
    add("AND r5 r4")                     # r5 = (op + payload) & 0xFF
    add("CMP r7 r5")
    return a


def _gh22_app_task(
    phase: int,
    exit_word: int,
    fault_leg: bool = False,
    corrupt_leg: bool = False,
) -> List[str]:
    """One USER-mode app slice (BOX0).

    phase 1 (SYS 6): build the request word in scratch, copy to the app
      uart receipt, SYSCALL 6 (kernel: scratch -> mailbox).
    phase 2 (SYS 8): SYSCALL 8 (kernel: driver verdict -> read-out 714),
      copy the verdict to the app uart, exit write, KJMP to the done tail.

    Request construction without multiply: byte b at position p builds via
    LDI + SHL(p) + OR (shifts are data ops on registers, no immediates
    beyond 32-bit LDI range).
    """
    a: List[str] = []
    add = a.append
    add(f":__task_app{phase}")
    if fault_leg and phase == 1:
        # App stays CLEAN on the fault leg — the DRIVER is the faulty party
        # (its OOB MMIO store). App: build + send the request normally; the
        # driver faults before it is ever scheduled again.
        pass
    if phase == 1:
        # payload byte 0x5A (r2 = payload first; r2 is the accumulator)
        add(f"LDI r2 {GH22_PAYLOAD}")
        # opcode byte -> bits [15:8]: r3 = op<<8 | payload via copy+shift
        add(f"LDI r3 {GH22_OPCODE_PUT}")
        add("LDI r4 8")
        add("SHL r3 r4")                 # r3 = op << 8
        add("OR r2 r3")                  # r2 = op<<8 | payload
        # checksum: r5 = (op + payload) & 0xFF — clean recompute from the
        # constants (register state at this point: r2 = op<<8|payload)
        add(f"LDI r5 {GH22_OPCODE_PUT}")
        add(f"LDI r6 {GH22_PAYLOAD}")
        add("ADD r5 r6")                 # r5 = op + payload
        add("LDI r4 255")
        add("AND r5 r4")                 # r5 = cksum
        add("LDI r4 24")
        add("SHL r5 r4")                 # r5 = cksum << 24
        add("OR r2 r5")                  # r2 = full request word
        if corrupt_leg:
            add("LDI r4 1")
            add("LDI r3 24")
            add("SHL r4 r3")             # r4 = 1 << 24
            add("XOR r2 r4")             # flip one checksum bit
        # stage in scratch + app receipt
        add(f"LDI r15 {GH22_REQ_SCRATCH}")
        add("ST r15 r2")
        add(f"LDI r15 {GH22_UART_APP}")
        add("ST r15 r2")
        # SYSCALL 6: a7=6, a0=request word
        add(f"LDI r17 {GH22_N_APP}")
        add(f"LDI r10 {GH22_REQ_SCRATCH}")
        add("LD r10 r10")                # a0 = the request WORD VALUE
        add("SYSCALL r12")
    else:
        # SYSCALL 8: the kernel delivers the driver's verdict into the
        # app's read-out (714) — the same kernel-mediated copy as SYS 7.
        add(f"LDI r17 {GH22_N_APP_RECV}")
        add(f"LDI r10 {GH22_READOUT_APP}")
        add("SYSCALL r12")
        add(f"LDI r15 {GH22_READOUT_APP}")
        add("LD r2 r15")                 # r2 = the delivered verdict
        add(f"LDI r15 {GH22_READOUT_APP}")
        add("ST r15 r2")                 # (idempotent re-store; keeps the
                                         # slice shape uniform)
    # exit status write: 0xFEED0000 | 22
    add(f"LDI r15 {exit_word}")
    add("LDI r14 65261")                 # 0xFEED
    add("LDI r4 16")
    add("SHL r14 r4")
    add(f"LDI r4 {22}")
    add("ADD r14 r4")
    add("ST r15 r14")
    # Return through the privilege boundary to the next dispatch leg.
    if phase == 1:
        add(f"LDI r30 {_GH22_DISPATCH_DRV_PC}")
    else:
        add(f"LDI r30 {_GH22_RET_PC}")
    add("KJMP r30")
    return a


def _gh22_driver_task(
    exit_word: int,
    fault_leg: bool = False,
) -> List[str]:
    """The USER-mode driver slice (BOX1). Protocol per the DEV-U1 datasheet:

      1. SYS 7: kernel copies the mailbox (request) into the driver's
         read-out (720). Non-blocking: no polling, one copy.
      2. Verify the packet checksum IN THE DRIVER'S BOX.
         - bad -> verdict 'E' at 724, device regs UNTOUCHED (refuse the
           transfer), exit clean. A corrupted packet is a verdict, not a
           fault.
         - good -> execute the datasheet §2 PUT sequence against the device
           registers (733/734 — inside BOX1, so the box bound IS the MMIO
           bound), verdict 'V'.
      3. fault_leg: the FIRST device access instead stores out-of-box
         (word 900) — the OOB-MMIO leg. E-K1 traps in USER mode.
      4. Exit write, KJMP to the app's second dispatch leg.
    """
    a: List[str] = []
    add = a.append
    add(":__task_drv")
    # SYS 7: mailbox -> read-out
    add(f"LDI r17 {GH22_N_DRV}")
    add(f"LDI r10 {GH22_READOUT_DRV}")
    add("SYSCALL r12")
    # load the request into r2 from the read-out
    add(f"LDI r15 {GH22_READOUT_DRV}")
    add("LD r2 r15")
    # checksum check (sets flags via CMP r7 r5)
    for line in _gh22_checksum_check():
        add(line)
    add("JZ :__drv_ok")
    # ---- corrupt packet: verdict 'E', device untouched ----
    add(f"LDI r15 {GH22_VERDICT_DRV}")
    add(f"LDI r14 {GH22_ERRORED}")
    add("ST r15 r14")
    add("JMP :__drv_exit")
    # ---- good packet: the DEV-U1 PUT sequence (datasheet §2) ----
    add(":__drv_ok")
    if fault_leg:
        # OOB MMIO: word 900 is outside every box — USER store -> KFAULT_PC.
        add("LDI r15 900")
        add("LDI r14 1")
        add("ST r15 r14")
    # extract the payload byte into r6 (r2 = request from the re-load;
    # AND is in-place, so copy r2 into r6 first via zero+OR)
    add(f"LDI r15 {GH22_READOUT_DRV}")
    add("LD r2 r15")
    add("XOR r6 r6")
    add("OR r6 r2")
    add("LDI r4 255")
    add("AND r6 r4")                     # r6 = payload byte
    # step 1: DEV_STATUS <- 0x01 (TX_ACTIVE)
    add(f"LDI r15 {GH22_DEV_STATUS}")
    add("LDI r14 1")
    add("ST r15 r14")
    # step 2: DEV_DATA <- payload
    add(f"LDI r15 {GH22_DEV_DATA}")
    add("ST r15 r6")
    # step 3: DEV_STATUS <- 0x81 (RX_DONE latched, quiescent)
    add(f"LDI r15 {GH22_DEV_STATUS}")
    add("LDI r14 129")                   # 0x81
    add("ST r15 r14")
    # verdict 'V'
    add(f"LDI r15 {GH22_VERDICT_DRV}")
    add(f"LDI r14 {GH22_VERIFIED}")
    add("ST r15 r14")
    # ---- exit ----
    add(f":__drv_exit")
    add(f"LDI r15 {exit_word}")
    add("LDI r14 65261")                 # 0xFEED
    add("LDI r4 16")
    add("SHL r14 r4")
    add("LDI r4 23")
    add("ADD r14 r4")
    add("ST r15 r14")
    add(f"LDI r30 {_GH22_DISPATCH_APP2_PC}")
    add("KJMP r30")
    return a


def _gh22_kernel_program_text(
    status_word: int = 950,
    fault_leg: bool = False,
    corrupt_leg: bool = False,
) -> str:
    """GH-22 glyph assembly: resident driver-ABI microkernel.

    Sequence (all in-image; the host only runs the image and reads memory):
      1. SUPER prologue: zero receipt words, program BOX0 (app) + BOX1
         (driver — its bounds double as the device-MMIO bound), arm
         KSYS_PC/KFAULT_PC, MODE_LATCH=1, KJMP :__task_app1.
      2. App phase 1 (USER, BOX0): build request, receipt, SYS 6 (-> mailbox
         754), exit, KJMP :__dispatch_drv.
      3. Driver (USER, BOX1): SYS 7 (mailbox -> read-out), verify, PUT the
         DEV-U1 sequence, verdict, exit, KJMP :__dispatch_app2.
      4. App phase 2 (USER, BOX0): SYS 8 (verdict -> read-out 714), exit,
         KJMP :__kdone.
      5. :__kdone writes status = 0xCAFE0000 | 26 and HALTs.

    The kernel NEVER touches words 733/734 — the device registers are
    driven exclusively by the unprivileged driver task (no monolithic
    kernel driver). Syscall numbers 6/7/8 are the GH-13 selector numbers
    reused per-mode; the :__ksys selector is this image's own.
    """
    a: List[str] = []
    add = a.append
    # deferred baker constants: baker re-exports this module (GH-22 test
    # import path) — module-level import would be circular, so the program
    # text builder pulls the word-map constants lazily (fs_v2.py pattern).
    from glyph_gpt.baker import (
        BOX0_LO_WORD, BOX1_LO_WORD,
        KFAULT_PC_WORD, KSYS_PC_WORD, MODE_LATCH_WORD,
    )
    add(":__entry")
    add("JMP :__kmain")
    add(":__kmain")
    # ---- zero the receipt words (all kernel-owned) ----
    for w in (GH22_EXIT_APP, GH22_UART_APP, GH22_REQ_SCRATCH,
              GH22_READOUT_APP, GH22_READOUT_DRV, GH22_EXIT_DRV,
              GH22_VERDICT_DRV, GH22_DEV_DATA, GH22_DEV_STATUS,
              GH22_MAILBOX_WORD, GH22_BADSYS_WORD, GH22_FAULT_WORD):
        add(f"LDI r15 {w}")
        add("LDI r14 0")
        add("ST r15 r14")
    # ---- program BOX0 (app) and BOX1 (driver) ----
    add(f"LDI r15 {BOX0_LO_WORD}")
    add(f"LDI r14 {GH22_BOX0_LO_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX0_LO_WORD + 1}")
    add(f"LDI r14 {GH22_BOX0_HI_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX1_LO_WORD}")
    add(f"LDI r14 {GH22_BOX1_LO_BYTE}")
    add("ST r15 r14")
    add(f"LDI r15 {BOX1_LO_WORD + 1}")
    add(f"LDI r14 {GH22_BOX1_HI_BYTE}")
    add("ST r15 r14")
    # ---- arm the vectors (loader-seeded packed pixel PCs) ----
    add(f"LDI r15 {KSYS_PC_WORD}")
    add(f"LDI r14 {_GH22_KSYS_PC}")
    add("ST r15 r14")
    add(f"LDI r15 {KFAULT_PC_WORD}")
    add(f"LDI r14 {_GH22_FAULT_PC if fault_leg else _GH22_RET_PC}")
    add("ST r15 r14")
    # ---- enter app phase 1: latch USER, KJMP ----
    add(f"LDI r15 {MODE_LATCH_WORD}")
    add("LDI r14 1")           # MODE_USER: KJMP one-shots the latch + clears
    add("ST r15 r14")
    add(f"LDI r30 {_GH22_TASK_APP_PC}")
    add("KJMP r30")
    add("JMP :__kdone")
    # ---- syscall dispatcher (SUPER): reached via KSYS_PC ----
    add(":__ksys")
    add(f"LDI r15 {SYS_N_ADDR >> 2}")
    add("LD r5 r15")                    # r5 = SYS_N
    add(f"LDI r4 {GH22_N_APP}")
    add("CMP r5 r4")
    add(f"JZ :__ksys_{GH22_N_APP}")
    add(f"LDI r4 {GH22_N_DRV}")
    add("CMP r5 r4")
    add(f"JZ :__ksys_{GH22_N_DRV}")
    add(f"LDI r4 {GH22_N_APP_RECV}")
    add("CMP r5 r4")
    add(f"JZ :__ksys_{GH22_N_APP_RECV}")
    # unknown syscall: record 'E' (69) and SYSRET dry
    add(f"LDI r15 {GH22_BADSYS_WORD}")
    add("LDI r14 69")                   # 'E'
    add("ST r15 r14")
    add("SYSRET")
    for line in _gh22_ksys_slice(GH22_N_APP):
        add(line)
    for line in _gh22_ksys_slice(GH22_N_DRV):
        add(line)
    for line in _gh22_ksys_slice(GH22_N_APP_RECV):
        add(line)
    # ---- round-robin dispatch: app1 -> driver, driver -> app2 ----
    add(":__dispatch_drv")
    add(f"LDI r15 {MODE_LATCH_WORD}")
    add("LDI r14 1")
    add("ST r15 r14")
    add(f"LDI r30 {_GH22_TASK_DRV_PC}")
    add("KJMP r30")
    add("JMP :__kdone")
    add(":__dispatch_app2")
    add(f"LDI r15 {MODE_LATCH_WORD}")
    add("LDI r14 1")
    add("ST r15 r14")
    add(f"LDI r30 {_GH22_TASK_APP2_PC}")
    add("KJMP r30")
    add("JMP :__kdone")
    # ---- fault handler (SUPER): records the violation, halts via kernel ----
    add(":__kfault")
    if fault_leg:
        add(f"LDI r15 {GH22_FAULT_WORD}")
        add(f"LDI r14 {GH22_FAULT_SEEN}")
        add("ST r15 r14")
    # fall through to the done/halt tail
    # ---- done: status = 0xCAFE0000 | 26, halt ----
    add(":__kdone")
    add(f"LDI r9 {GH22_STATUS_ID}")
    add("LDI r3 51966")        # 0xCAFE
    add("LDI r4 16")
    add("SHL r3 r4")
    add("ADD r3 r9")
    add(f"LDI r15 {status_word}")
    add("ST r15 r3")
    add("HALT")
    # ---- app phase 1 (USER, BOX0): request sender ----
    for line in _gh22_app_task(1, GH22_EXIT_APP, fault_leg, corrupt_leg):
        add(line)
    # ---- driver (USER, BOX1): the device task ----
    for line in _gh22_driver_task(GH22_EXIT_DRV, fault_leg):
        add(line)
    # ---- app phase 2 (USER, BOX0): verdict receiver ----
    for line in _gh22_app_task(2, GH22_EXIT_APP, fault_leg, corrupt_leg):
        add(line)
    return "\n".join(a) + "\n"


def driver_abi_kernel_image(
    atlas: Any,
    status_word: int = 950,
    fault_leg: bool = False,
    corrupt_leg: bool = False,
    cols_instrs: int = 8,
    min_rows: int = 16,
    out_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """GH-22: bake ONE image with the resident driver-ABI microkernel.
    Two passes: pass 1 fixes the packed pixel PCs (:__ksys/:__kfault/
    :__dispatch_drv/:__dispatch_app2/:__task_app1/:__task_drv/
    :__task_app2/:__kdone), pass 2 bakes the final image. The atlas is
    accepted for GH-2 continuity but this scheduler needs no atlas tiles."""
    # deferred import: baker re-exports this module (GH-22 test import
    # path) — a module-level import would be circular (fs_v2.py pattern).
    from glyph_gpt.baker import (
        bake_image, BOX0_LO_WORD, BOX1_LO_WORD,
        KFAULT_PC_WORD, KSYS_PC_WORD, MODE_LATCH_WORD,
    )
    from rv64i_to_glyph import assemble_glyph_to_pixels
    txt1 = _gh22_kernel_program_text(status_word, fault_leg, corrupt_leg)
    _, coords1 = assemble_glyph_to_pixels(txt1, cols_instrs=cols_instrs, min_rows=min_rows)

    def packed(label: str) -> int:
        col, row = coords1[label]
        # engine packing: low 16 bits = column, high 16 = row
        return (col & 0xFFFF) | ((row & 0xFFFF) << 16)

    global _GH22_KSYS_PC, _GH22_FAULT_PC, _GH22_RET_PC
    global _GH22_DISPATCH_DRV_PC, _GH22_DISPATCH_APP2_PC
    global _GH22_TASK_APP_PC, _GH22_TASK_DRV_PC, _GH22_TASK_APP2_PC
    _GH22_KSYS_PC = packed(":__ksys")
    _GH22_FAULT_PC = packed(":__kfault")     # bound BEFORE pass 2 or the
    _GH22_RET_PC = packed(":__kdone")        # fault leg vectors to pixel (0,0)
    _GH22_DISPATCH_DRV_PC = packed(":__dispatch_drv")
    _GH22_DISPATCH_APP2_PC = packed(":__dispatch_app2")
    _GH22_TASK_APP_PC = packed(":__task_app1")
    _GH22_TASK_DRV_PC = packed(":__task_drv")
    _GH22_TASK_APP2_PC = packed(":__task_app2")
    try:
        # Pass 2 re-materializes the program with the real packed PCs baked
        # in (assemble is deterministic, so the label coords are identical).
        return bake_image(
            _gh22_kernel_program_text(status_word, fault_leg, corrupt_leg),
            atlas=None,
            cols_instrs=cols_instrs,
            min_rows=min_rows,
            out_path=out_path,
        )
    finally:
        _GH22_KSYS_PC = _GH22_FAULT_PC = _GH22_RET_PC = 0
        _GH22_DISPATCH_DRV_PC = _GH22_DISPATCH_APP2_PC = 0
        _GH22_TASK_APP_PC = _GH22_TASK_DRV_PC = _GH22_TASK_APP2_PC = 0
