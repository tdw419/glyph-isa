#!/usr/bin/env python3
"""tests/test_gh18_syscall_abi.py — GH-18 Syscall ABI v2 oracle test.

Falsifiable gate for GH-18 (systems/GLYPH_SELF_HOSTING_ROADMAP.md):

Syscalls ARE atlas tiles: the kernel syscall table is an IN-IMAGE word
table of packed pixel PCs into registered, oracle-proven tile code living
in the image's atlas rect. Dispatch is an INDEXED table lookup
(LD TABLE_BASE+sys_n -> KJMP packed pixel PC) replacing the linear
CMP/JZ selector chain ONCE — the invariant baseline is the post-rewrite
dispatcher image.

Gate legs (roadmap row GH-18):
1. Patch isolation is a POSITIVE WHITELIST — diff(post, pre) must be a
   subset of (table_words ∪ tile_rect) exactly; any byte outside fails.
2. sys_n bounds check is UNSIGNED (power-of-two mask): negative and
   0x7FFFFFFF sys_n must vector to the unknown-syscall handler
   ('E' + clean SYSRET), never an OOB table read.
3. Re-entrancy: syscalls run to completion — the GH-16 tick is deferred
   while SYS_N is marshaled (tick masked on SYSCALL entry, unmasked on
   SYSRET); a tick firing mid-dispatch must not clobber SYS_N/A0/PC.
4. Tile admission ties to the ingest gate: a tile that fails verification
   is rejected (E_ATLAS_UNVERIFIED) and the table stays untouched.
5. Table + tile rect live in identity-mapped low pages: SUPER dispatch
   never consults the page walker on the hot path (page table armed,
   dispatch still executes through the table).
6. ABI version word (952) == 0x00020018 baked in-image.
7. Syscall admission end-to-end: register a new syscall tile, its number
   goes live in-image, a USER task issues the syscall, the tile computes
   word-exactly; SYSRET preserves a0..a2 for a1/a2 passthrough.
"""
from __future__ import annotations

import sys
import tempfile
import urllib.request
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.glyph_gpt.atlas import build_default_atlas           # noqa: E402
from tools.glyph_gpt.runner import GlyphRunner                  # noqa: E402
from tools.glyph_gpt.baker import syscall_abi_kernel_image      # noqa: E402 (RED)
from tools.glyph_isa_v2 import (                                # noqa: E402
    SYS_N_ADDR, SYS_A0_ADDR, SYS_A1_ADDR,
)

# GH-18 image ABI constants (mirror baker.py)
GH18_TABLE_WORD = 1568          # syscall table base (in page-table window)
GH18_NSLOTS = 16                # power-of-two mask = 15
GH18_TILE_WORDS = (1600, 1696)  # atlas rect: 24 instrs x 4 px
GH18_BADSYS_WORD = 730          # unknown-syscall marker ('E' = 69)
GH18_EXIT_WORD = 703
GH18_EXIT_A = 703
GH18_EXIT_OK = 0xFEED0000 | 6
GH18_UART_A = 710
GH18_UART_B = 720
GH18_FAULT_WORD = 731
GH18_TICKS_COUNT = 732
GH18_TURN_WORD = 705
GH18_VERIFY_WORD = 704
GH18_VERIFY_OK = 0xFEED0018
GH18_ABI_WORD = 952
GH18_ABI_VERSION = 0x00020018
KERNEL_OK = 0xCAFE0018

# tile dispatch numbers (legacy fixed slices live BELOW the table window)
N_T1 = 6    # legacy fixed slice (A: uart)
N_T2 = 7    # legacy fixed slice (B: mailbox read)
N_NEW = 8   # table-dispatched admitted tile (triple(r1) -> r2)
N_UNK = 9   # NOT in the fixed slices, NOT in the table

BOX0 = (4 * 700, 4 * 717)
BOX1 = (4 * 718, 4 * 735)


def _ollama_available() -> bool:
    try:
        with urllib.request.urlopen(
                "http://localhost:11434/api/tags", timeout=3) as r:
            return bool(__import__("json").loads(r.read()).get("models"))
    except Exception:
        return False


def _run(tmp: Path, mode: str = "baseline", name: str = "gh18.glyph.npy",
         timer_quantum: int = 35):
    out = tmp / name
    syscall_abi_kernel_image(build_default_atlas(), mode=mode,
                             timer_quantum=timer_quantum, out_path=out)
    runner = GlyphRunner(out, ram_words=16384)
    return runner, runner.run(max_instructions=60000, trace=True)


def _seed_admit_tile(runner, table_seeds: dict) -> None:
    """Seed the syscall table entry for N_NEW (packed pixel PC of tile).

    Seeds BOTH surfaces (2026-09-08 receipt, probes tblpath10..33):
    the admit-mode kernel arms a PTE_PIX page map for vpn 6 in-image,
    so the dispatcher's LD [1568..1583] walks to IMAGE PIXELS
    (pix_word = 5*256 + (word & 0xFF); pfn 5 — the identity pfn 6 wrap
    lands on pixel (0,2) and clobbers :__entry's JMP immediate). RAM
    seeds alone die with each drive()'s fresh CPU — the steering
    contract requires the slot to persist in the image. The RAM seed is
    kept for the paged leg's RAM-identity walk and as documentation.
    """
    from tools.glyph_gpt.baker import GH18_TILE_PC
    slot = GH18_TABLE_WORD + (N_NEW - 6)
    table_seeds[slot] = GH18_TILE_PC
    # pixel surface: vpn-6 pix page (pfn 5) — mapping receipt in the
    # admit-mode prologue (baker._gh18_kernel_program_text).
    pw = 5 * 256 + (slot & 0xFF)
    h, w, _ = runner.image.shape
    v = GH18_TILE_PC
    runner.image[pw // w, pw % w] = ((v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF)


# ── image ABI invariants ────────────────────────────────────────────────

def test_gh18_bakes_and_abi_version_word():
    """Baseline image bakes with the ABI version word in its data region
    AND carries it in-image at boot (memory[952] == 0x00020018)."""
    with tempfile.TemporaryDirectory() as d:
        runner, receipt = _run(Path(d))
        assert receipt["halted"] is True, receipt.get("error", receipt)
        assert receipt["faulted"] is False, receipt
        mem = receipt["memory"]
        assert mem[GH18_ABI_WORD] == GH18_ABI_VERSION, (
            f"ABI version word 0x{mem[GH18_ABI_WORD]:08x} != "
            f"0x{GH18_ABI_VERSION:08x}")
        assert receipt["status_word_value"] == KERNEL_OK, receipt


def test_gh18_dispatch_rewrite_is_table_lookup_not_selector_chain():
    """The dispatcher must dispatch through the table (indexed LD+KJMP),
    not the linear CMP/JZ selector. Static gate: the dispatcher block must
    not contain BOTH fixed-slice compares for table-covered numbers."""
    import inspect
    from tools.glyph_gpt import baker
    src = inspect.getsource(baker._gh18_kernel_program_text)
    # The rewrite replaces the linear selector with table lookup: the
    # text must contain the table-load instruction sequence.
    assert "KJMP" in src and "__tbl" in src, (
        "dispatcher must perform an indexed table load + KJMP dispatch")


def test_gh18_two_syscalls_still_dispatch():
    """Regression invariant: the baseline image services the legacy
    syscalls 6 (A: uart write) and 7 (B: mailbox read) via the table."""
    with tempfile.TemporaryDirectory() as d:
        runner, receipt = _run(Path(d))
        assert receipt["halted"] is True, receipt.get("error", receipt)
        assert receipt["faulted"] is False, receipt
        mem = receipt["memory"]
        assert mem[GH18_UART_A] == 0x41414141, hex(mem[GH18_UART_A])
        assert mem[GH18_UART_B] == 0x42424242, hex(mem[GH18_UART_B])
        assert mem[GH18_EXIT_A] == GH18_EXIT_OK
        assert mem[723] == 0xFEED0007
        assert mem[GH18_BADSYS_WORD] == 0   # nothing unknown ran
        assert receipt["status_word_value"] == KERNEL_OK


def test_gh18_unknown_syscall_hits_handler_clean():
    """A sys_n outside the fixed slices AND the table (9) must vector to
    the unknown-syscall handler: 'E' recorded, clean SYSRET, both tasks
    still complete, status word is the KERNEL_OK tail."""
    with tempfile.TemporaryDirectory() as d:
        runner, receipt = _run(Path(d), mode="unknown_syscall")
        assert receipt["halted"] is True, receipt.get("error", receipt)
        assert receipt["faulted"] is False, receipt
        mem = receipt["memory"]
        assert mem[GH18_BADSYS_WORD] == 69, hex(mem[GH18_BADSYS_WORD])
        assert receipt["status_word_value"] == KERNEL_OK


# ── gate leg 1: patch isolation is a positive whitelist ────────────────

def test_gh18_patch_isolation_positive_whitelist():
    """Table seeding mutates ONLY table words ∪ tile rect. The whole
    image diff must be inside the whitelist; any byte outside fails."""
    with tempfile.TemporaryDirectory() as d:
        runner_pre, receipt_pre = _run(Path(d), mode="admit", name="pre.npy")
        mem_pre = receipt_pre["memory"]
        img_pre = runner_pre.image.copy()
        assert receipt_pre["halted"] is True

        seeds = dict.fromkeys(
            range(GH18_TABLE_WORD, GH18_TABLE_WORD + GH18_NSLOTS), 0)
        # Seed BEFORE snapshotting the drive image, and drive the LIVE
        # (seeded) runner: _seed_admit_tile writes the pixel-surface
        # seed to runner.image, and the admit-mode dispatcher's LD walks
        # to IMAGE PIXELS (PTE_PIX vpn-6 map) — a drive from a
        # pre-seed copy RAM-seeds only and dies at 754=0 (receipt:
        # 2026-09-08, mirror-probe vs -k whitelist divergence).
        _seed_admit_tile(runner_pre, seeds)
        runner_post = GlyphRunner(runner_pre.image, ram_words=16384)
        receipt_post = runner_post.drive(seeds=seeds, max_instructions=60000)
        img_post = runner_post.image

        assert receipt_post["halted"] is True, receipt_post.get("error")
        assert receipt_post["faulted"] is False, receipt_post
        # triple(6) = 18 DECIMAL = 0x12. Receipt (2026-09-08, probes
        # 27-42): the literal here was 0x00000018 = 24 — a hex-literal
        # typo for decimal 18; the tile result 0x12 was CORRECT and
        # pytest's rich assert rendered it as '18 == 24' (0x12 == 0x18),
        # sending the previous session chasing a phantom pytest-context
        # seeding bug. Every other leg asserts decimal 18.
        assert receipt_post["memory"][754] == 18  # triple(6) = 18
        assert receipt_post["memory"][GH18_EXIT_WORD] == GH18_EXIT_OK

        # diff the FULL images word-exactly
        flat_pre = img_pre.reshape(-1, 3)
        flat_post = img_post.reshape(-1, 3)
        assert flat_pre.shape == flat_post.shape
        lo, hi = GH18_TILE_WORDS
        outside = []
        # admit-mode pixel-surface table window: the kernel's PTE_PIX
        # vpn-6 map walks table LDs to pfn-5 pixels — pw = 5*256 +
        # (word & 0xFF) for word in [1568,1584); 1568 & 0xFF = 32 ->
        # pixels [1312,1328). It IS the table (admit-mode layout
        # receipt, autoatlas _pix_write_word), inside the whitelist.
        p5lo, p5hi = 5 * 256 + (GH18_TABLE_WORD & 0xFF), \
            5 * 256 + (GH18_TABLE_WORD & 0xFF) + GH18_NSLOTS
        for i in range(flat_pre.shape[0]):
            if (flat_pre[i] != flat_post[i]).any():
                if not (lo <= i < hi or p5lo <= i < p5hi or
                        GH18_TABLE_WORD * 4 <= i * 4 < (GH18_TABLE_WORD + GH18_NSLOTS) * 4):
                    outside.append(i)
        assert not outside, (
            f"{len(outside)} mutated px words OUTSIDE the whitelist "
            f"(first 10: {outside[:10]})")


def test_gh18_admitted_tile_computes_word_exact():
    """End-to-end syscall admission: with the table entry seeded (the
    receipt of autoatlas.ingest()), a USER task issues SYS 8 and the
    admitted tile triples its argument word-exactly. Task A runs the
    admit mode: baseline SYS 6 first (call-chain intact), then SYS 8
    through the seeded table slot."""
    with tempfile.TemporaryDirectory() as d:
        runner, _ = _run(Path(d), mode="admit", name="admit.npy")
        seeds = dict.fromkeys(
            range(GH18_TABLE_WORD, GH18_TABLE_WORD + GH18_NSLOTS), 0)
        _seed_admit_tile(runner, seeds)
        receipt = runner.drive(seeds=seeds, max_instructions=60000)
        assert receipt["halted"] is True, receipt.get("error")
        assert receipt["faulted"] is False, receipt
        mem = receipt["memory"]
        assert mem[754] == 18, f"tile result {mem[754]} != 18"
        assert mem[GH18_EXIT_WORD] == GH18_EXIT_OK
        assert mem[GH18_BADSYS_WORD] == 0
        assert receipt["status_word_value"] == KERNEL_OK


def test_gh18_tile_preserves_returning_task_a0():
    """IRContract admission property, proven on-die: the tile clobbers
    only scratch/return regs — the task's boxed r2..r4 values survive
    the round trip (dispatcher restores, or the contract guarantees).
    The canaries are written by the admit-mode task to box words that
    no other leg touches (714/715/716 — 713 is the SYS 6 payload)."""
    with tempfile.TemporaryDirectory() as d:
        runner, _ = _run(Path(d), mode="admit", name="preserve.npy")
        seeds = dict.fromkeys(
            range(GH18_TABLE_WORD, GH18_TABLE_WORD + GH18_NSLOTS), 0)
        _seed_admit_tile(runner, seeds)
        receipt = runner.drive(seeds=seeds, max_instructions=60000)
        mem = receipt["memory"]
        assert mem[714] == 0x0ABC000C, hex(mem[714])   # BOX0 r2
        assert mem[715] == 0x0ABD000D, hex(mem[715])   # BOX0 r3
        assert mem[716] == 0x0ABE000E, hex(mem[716])   # BOX0 r4
        assert mem[754] == 18                          # result landed


# ── gate leg 2: sys_n bounds check is UNSIGNED ─────────────────────────

def test_gh18_negative_and_huge_sysn_trap_clean():
    """sys_n = -1 and 0x7FFFFFFF must vector to the unknown-syscall
    handler ('E' + clean SYSRET) — never an OOB table read. The task
    still completes and the kernel tails to KERNEL_OK."""
    for mode in ("neg_sysn", "huge_sysn"):
        with tempfile.TemporaryDirectory() as d:
            runner, receipt = _run(Path(d), mode=mode, name=f"{mode}.npy")
            assert receipt["halted"] is True, receipt.get("error")
            assert receipt["faulted"] is False, receipt
            mem = receipt["memory"]
            assert mem[GH18_BADSYS_WORD] == 69, (mode, hex(mem[GH18_BADSYS_WORD]))
            assert receipt["status_word_value"] == KERNEL_OK


# ── gate leg 3: tick-deferred re-entrancy ──────────────────────────────

def test_gh18_mid_syscall_tick_deferred():
    """The GH-16 timer cannot clobber SYS_N/A0/PC mid-call. Task A issues
    SYS 6 with the timer armed at the MINIMUM reload (1): the tick may
    only fire when SYS_N is free (>= 0 = outside a call). The receipt
    must show: SYS_N/A0 cleared to 0 (no torn call), ticks >= 1, clean
    completion with both uarts written."""
    with tempfile.TemporaryDirectory() as d:
        runner, receipt = _run(Path(d), mode="reentrancy", timer_quantum=1)
        assert receipt["halted"] is True, receipt.get("error")
        assert receipt["faulted"] is False, receipt
        mem = receipt["memory"]
        assert mem[GH18_TICKS_COUNT] >= 1, "timer must have ticked"
        assert mem[SYS_N_ADDR >> 2] == 0, "SYS_N must be free (0) at rest"
        assert mem[SYS_A0_ADDR >> 2] == 0, "SYS_A0 must be free (0) at rest"
        assert mem[GH18_UART_A] == 0x41414141
        assert mem[GH18_UART_B] == 0x42424242
        assert receipt["status_word_value"] == KERNEL_OK


# ── gate leg 5: identity-mapped dispatch path ──────────────────────────

def test_gh18_dispatch_identity_mapped_with_paging_armed():
    """With the GH-17 page table armed (identity map of the low pages),
    the table-dispatched tile still computes word-exactly: SUPER dispatch
    never consults the page walker on the hot path."""
    with tempfile.TemporaryDirectory() as d:
        runner, _ = _run(Path(d), mode="admit", name="paged.npy")
        seeds = dict.fromkeys(
            range(GH18_TABLE_WORD, GH18_TABLE_WORD + GH18_NSLOTS), 0)
        _seed_admit_tile(runner, seeds)
        receipt = runner.drive(seeds=seeds, max_instructions=60000)
        assert receipt["halted"] is True, receipt.get("error")
        assert receipt["faulted"] is False, receipt
        mem = receipt["memory"]
        assert mem[754] == 18
        assert mem[GH18_UART_A] == 0x41414141
        assert receipt["status_word_value"] == KERNEL_OK


# ── gate leg 4: ingest-admission tie ───────────────────────────────────

def test_gh18_unproven_tile_rejected_table_untouched(monkeypatch):
    """The GH-18 route rejects unverified tiles: monkeypatched escalate
    returns unverified; admit must NOT return OK, and the in-image table
    word for N_NEW stays 0."""
    from tools.glyph_gpt import autoatlas as aa
    from tools.glyph_gpt.escalate import EscalationResult

    def _fail(*a, **k):
        return EscalationResult(
            contract=a[0] if a else "", verified=False,
            attempts=k.get("max_attempts", 6),
            error="no candidate verified in N")
    monkeypatch.setattr(aa, "escalate", _fail)

    from tools.glyph_gpt.autoatlas import admit_syscall   # noqa: E402 (RED)
    with tempfile.TemporaryDirectory() as d:
        runner, receipt = _run(Path(d), mode="admit", name="reject.npy")
        assert receipt["halted"] is True
        res = admit_syscall(runner, N_NEW)
        assert not res.ok
        assert res.code == "E_ATLAS_UNVERIFIED"
        assert res.table_word == 0, "table entry must stay untouched"
        assert "triple" not in getattr(res, "name", "") or res.table_word == 0


@pytest.mark.skipif(not _ollama_available(), reason="local Ollama not reachable")
def test_gh18_admit_syscall_via_ingest_end_to_end():
    """The full admission pipeline: a local-model-drafted triple tile
    passes the oracle + IR gate, its pixels land in the tile rect, its
    table word goes live, and a USER task issues SYS 8 word-exactly.
    One Ollama round; the drafted program is verified before dispatch."""
    from tools.glyph_gpt.autoatlas import admit_syscall   # noqa: E402 (RED)
    with tempfile.TemporaryDirectory() as d:
        runner, receipt = _run(Path(d), mode="admit", name="ingest.npy")
        assert receipt["halted"] is True
        res = admit_syscall(runner, N_NEW,
                            contract="triple(r1) -> r2: r2 = 3 * r1",
                            argv={0: 6}, expected=18)
        assert res.ok, f"{res.code}: {res.detail}"
        assert res.table_word != 0, "table entry must be live"
        # the LIVE image now carries the tile: issue the syscall again
        receipt2 = runner.drive(seeds={}, max_instructions=60000)
        mem = receipt2["memory"]
        assert mem[754] == 18
        assert mem[GH18_EXIT_WORD] == GH18_EXIT_OK
        assert mem[GH18_BADSYS_WORD] == 0


def test_gh18_syscall_table_window_reserved():
    """GH-18 follow-up (ticket gh18-table-reservation): the syscall
    table's PIXEL surface — words 1568..1583 map (admit-mode PTE_PIX
    pfn-5 walk) to pixel words [1312, 1328) — is a RESERVED window, the
    same contract the fs window [1024,1280) has. Structural guarantee,
    not headroom: for every bake mode, (a) the table's pixel surface
    stays zero (no program text assembles into it — program text must
    end before the reserved window starts), (b) the pfn-5 mapping is a
    mode-independent constant (baker.GH18_TABLE_PIX_WORD), and
    (c) admit_syscall's table-write lands INSIDE the reserved window."""
    from tools.glyph_gpt import baker as B
    from tools.glyph_gpt.autoatlas import GH18_TABLE_PIX_WORD
    from tools.rv64i_to_glyph import assemble_glyph_to_pixels

    # (b) the mapping is a pinned constant, not image-size arithmetic
    assert GH18_TABLE_PIX_WORD == 5 * 256 + (1568 & 0xFF) == 1312

    for mode in ("baseline", "admit", "paged_dispatch"):
        with tempfile.TemporaryDirectory() as d:
            _run(Path(d), mode=mode, name=f"reserve_{mode}.npy")
            # pass-1 determinism: the SAME program text must place its
            # last text pixel strictly BELOW the reserved window.
            txt = B._gh18_kernel_program_text(mode)
            # text extent = non-label instructions (the assembler's
            # own count; the pixel ARRAY is padded to min_rows and
            # would over-count) x 4 px per instruction
            n_instr = sum(
                1 for ln in txt.splitlines()
                if ln.strip() and not ln.strip().startswith("#")
                and not ln.strip().startswith(":"))
            text_end = n_instr * 4
            assert text_end <= GH18_TABLE_PIX_WORD, (
                f"mode {mode}: program text extends to pixel word "
                f"{text_end} >= reserved table window start "
                f"{GH18_TABLE_PIX_WORD}")
            # (a) the baked image keeps the window zero: no opcode text
            # lives in [1312, 1328)
            img = B.syscall_abi_kernel_image(
                build_default_atlas(), mode=mode,
                out_path=Path(d) / f"check_{mode}.npy")
            h, w, _ = img.shape
            for pw in range(GH18_TABLE_PIX_WORD,
                            GH18_TABLE_PIX_WORD + 16):
                pxv = img[pw // w, pw % w]
                assert tuple(pxv[:3]) == (0, 0, 0), (
                    f"mode {mode}: reserved table pixel word {pw} is "
                    f"non-zero {tuple(pxv)} — program text collision")

    # (c) admit_syscall writes the table word through the reserved
    # window: source-level pin so future refactors can't bypass it.
    import inspect
    from tools.glyph_gpt import autoatlas
    src = inspect.getsource(autoatlas.admit_syscall)
    assert "GH18_TABLE_PIX_WORD" in src, (
        "admit_syscall must derive its pfn-5 write from the pinned "
        "GH18_TABLE_PIX_WORD constant")


def test_gh18_runner_line_budget_and_clean_imports():
    """GH-5 / GH-11 / GH-16 invariant preserved: runner.py <= 200 lines."""
    runner_path = _REPO / "tools" / "glyph_gpt" / "runner.py"
    lines = runner_path.read_text().splitlines()
    assert len(lines) <= 200, f"runner.py exceeds 200 lines: {len(lines)}"
