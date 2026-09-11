"""tools/glyph_ir.py — Unified GlyphIR (GH-15).

One semantic intermediate representation that the RV64I transpiler,
autoatlas and baker all lower through. Pure dataclasses + a static
verifier; zero dev-time imports (stdlib only), per the launcher-lane
convention.

Design invariants (roadmap GH-15, ratified):
- Explicit scratch pool: LoweringConfig(scratch_pool=[...], callstack_reg).
  The verifier enforces scratch_pool ∩ contract.preserves == ∅ and
  callstack_reg ∉ scratch_pool.
- Explicit pointer table ordering for dynamic indirect jump dispatch
  (replaces the PTR_TABLE_BASE convention with a recorded, checkable
  ordering).
- Disentangled geometry: CodeWindow enforces the execution-window word
  capacity independently of Contract.data_bounds (RAM read/write
  boundaries).
- Immediate source relocations: SymbolReloc lets a lowered instruction
  reference a symbol for materialization (hi20/lo12/word_addr/abs32)
  without assembly-text string hacks.
- Uniform memory model: Address(base_reg, offset) or (None, flat_addr).
- Honest verification boundary: the static verifier catches statically
  determinable violations; dynamic pointers remain guarded by the
  hardware E-K1 trap at runtime.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class StaticVerificationError(ValueError):
    """Loud, pre-emission rejection of an IR module."""


# ── GH-15 Step 5: the ONE terminator table + lowering policy constants ──
# Every lane (rv64i_to_glyph, autoatlas, baker) references these; none
# may carry a private copy (tests/test_gh15_step5_dedup.py enforces it).
TERMINATOR_OPS = ("JMP", "JZ", "JNZ", "JMPR", "CALL", "CALLR", "RET",
                  "KJMP", "HALT", "SYSCALL", "SYSRET")
DEFAULT_SCRATCH_POOL = ["r26", "r27", "r28", "r29", "r30"]
DEFAULT_CALLSTACK_REG = "r31"


class RelocType(str, Enum):
    HI20 = "hi20"          # upper 20 bits of the symbol address
    LO12 = "lo12"          # lower 12 bits, sign-adjusted
    WORD_ADDR = "word_addr"  # symbol byte address >> 2 (glyph word index)
    ABS32 = "abs32"        # full 32-bit absolute value


@dataclass(frozen=True)
class SymbolReloc:
    """A symbolic reference to be materialized at emission/link time."""
    symbol: str
    reloc_type: RelocType = RelocType.ABS32
    addend: int = 0

    def resolve(self, symbols: Dict[str, int]) -> int:
        if self.symbol not in symbols:
            raise StaticVerificationError(
                f"unresolved symbol '{self.symbol}'")
        addr = symbols[self.symbol] + self.addend
        if self.reloc_type is RelocType.HI20:
            return (addr >> 12) & 0xFFFFF
        if self.reloc_type is RelocType.LO12:
            lo = addr & 0xFFF
            return lo - 0x1000 if lo >= 0x800 else lo
        if self.reloc_type is RelocType.WORD_ADDR:
            if addr & 0x3:
                raise StaticVerificationError(
                    f"symbol '{self.symbol}' addr 0x{addr:x} not 4-aligned")
            return addr >> 2
        if self.reloc_type is RelocType.ABS32:
            return addr & 0xFFFFFFFF
        raise StaticVerificationError(
            f"unknown reloc type {self.reloc_type}")


@dataclass(frozen=True)
class Address:
    """Uniform memory model for every load/store.

    (base_reg, offset) is register-indirect; (None, flat_addr) is an
    absolute glyph-word address. `base_reg` is a glyph register name
    (e.g. "r31"); flat addresses are glyph WORD addresses.
    """
    base_reg: Optional[str] = None
    offset: int = 0
    flat_addr: Optional[int] = None

    def __post_init__(self):
        if self.base_reg is None and self.flat_addr is None:
            raise StaticVerificationError(
                "Address needs base_reg or flat_addr")
        if self.base_reg is not None and self.flat_addr is not None:
            raise StaticVerificationError(
                "Address cannot have both base_reg and flat_addr")

    @classmethod
    def indirect(cls, base_reg: str, offset: int = 0) -> "Address":
        return cls(base_reg=base_reg, offset=offset)

    @classmethod
    def absolute(cls, flat_addr: int) -> "Address":
        return cls(flat_addr=flat_addr)

    def resolve(self, reg_values: Optional[Dict[str, int]] = None) -> int:
        if self.base_reg is None:
            return self.flat_addr
        if reg_values and self.base_reg in reg_values:
            return reg_values[self.base_reg] + self.offset
        raise StaticVerificationError(
            f"Address base {self.base_reg} not statically known")


@dataclass(frozen=True)
class Operand:
    """One source/dest slot: a physical register, immediate, symbol
    reference or memory reference."""
    phys_reg: Optional[str] = None
    imm: Optional[int] = None
    symbol_ref: Optional[SymbolReloc] = None
    mem_ref: Optional[Address] = None

    def __post_init__(self):
        provided = sum(v is not None for v in
                       (self.phys_reg, self.imm, self.symbol_ref,
                        self.mem_ref))
        if provided != 1:
            raise StaticVerificationError(
                "Operand must carry exactly one of "
                "phys_reg/imm/symbol_ref/mem_ref")

    @classmethod
    def reg(cls, name: str) -> "Operand":
        return cls(phys_reg=name)

    @classmethod
    def const(cls, value: int) -> "Operand":
        return cls(imm=value)

    @classmethod
    def sym(cls, symbol: str,
            reloc_type: RelocType = RelocType.ABS32,
            addend: int = 0) -> "Operand":
        return cls(symbol_ref=SymbolReloc(symbol, reloc_type, addend))

    @classmethod
    def mem(cls, address: Address) -> "Operand":
        return cls(mem_ref=address)


@dataclass(frozen=True)
class IRInstruction:
    """One semantic operation. `op` is the Glyph ISA v2 opcode for
    machine ops; `dests`/`sources` are Operands; `metadata` carries
    per-op extras (e.g. the originating RV pc)."""
    op: str
    dests: Tuple[Operand, ...] = ()
    sources: Tuple[Operand, ...] = ()
    metadata: Dict = field(default_factory=dict)

    def written_regs(self) -> List[str]:
        return [o.phys_reg for o in self.dests if o.phys_reg]

    def read_regs(self) -> List[str]:
        out = []
        for o in self.sources:
            if o.phys_reg:
                out.append(o.phys_reg)
            if o.mem_ref and o.mem_ref.base_reg:
                out.append(o.mem_ref.base_reg)
        return out


@dataclass
class BasicBlock:
    """A straight-line run of instructions with one terminator."""
    label: str
    instructions: List[IRInstruction] = field(default_factory=list)
    terminator: Optional[str] = None            # "JMP" | "JZ" | "JMPR" | ...
    terminator_target: Optional[str] = None     # block label / symbol
    phis: List = field(default_factory=list)    # reserved (no SSA yet)

    _LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")

    def __post_init__(self):
        if not self._LABEL_RE.match(self.label):
            raise StaticVerificationError(
                f"illegal block label '{self.label}'")

    def append(self, ins: IRInstruction) -> None:
        if self.terminator is not None:
            raise StaticVerificationError(
                f"block '{self.label}' already terminated")
        self.instructions.append(ins)


@dataclass
class CodeWindow:
    """The execution window a lowered module must fit into (GH-9 mailbox
    patch window by default). Disentangled from RAM data bounds."""
    origin_symbol: str = "__g9window"
    max_words: int = 96
    cols_instrs: int = 24           # 2D wrap factor (instrs per row)


@dataclass
class DataSection:
    """A contiguous data region placed at a glyph word address."""
    symbol: str
    base_word: int
    words: List[int] = field(default_factory=list)


@dataclass
class PointerTable:
    """Explicit ordering of dynamic-jump targets. entry i is the glyph
    word holding the packed PC of block_targets[i] (an RV byte address
    or symbol)."""
    base_word: int
    block_targets: List[str] = field(default_factory=list)


@dataclass
class IRContract:
    """What this module promises its caller: which registers it
    preserves and which RAM words it may read/write."""
    preserves: List[str] = field(default_factory=list)
    data_bounds: Tuple[int, int] = (0, 1 << 30)   # inclusive glyph words


@dataclass
class LoweringConfig:
    """Register-budget policy for a lowering pass."""
    scratch_pool: List[str] = field(
        default_factory=lambda: ["r26", "r27", "r28", "r29", "r30"])
    callstack_reg: str = "r31"


@dataclass
class GlyphIRModule:
    """The whole unit of lowering: blocks + data + window + contract."""
    name: str
    blocks: List[BasicBlock] = field(default_factory=list)
    data_sections: List[DataSection] = field(default_factory=list)
    pointer_table: Optional[PointerTable] = None
    code_window: Optional[CodeWindow] = None
    contract: IRContract = field(default_factory=IRContract)
    lowering: LoweringConfig = field(default_factory=LoweringConfig)

    def block_labels(self) -> List[str]:
        return [b.label for b in self.blocks]


class StaticVerifier:
    """Pre-emission static checks. Loud failures, never warnings."""

    # RAM words reserved by the resident kernel / other subsystems. A
    # module's data_bounds may not overlap these unless it owns them.
    RESERVED_RANGES = [
        (750, 760, "GH-9 argv/receipt block"),
        (800, 896, "GH-9 mailbox patch window"),
        (950, 968, "kernel status/verdict words"),
        (1024, 1280, "GH-8b pixel-FS window"),
    ]

    def __init__(self, module: GlyphIRModule):
        self.module = module
        self.errors: List[str] = []

    def _err(self, msg: str) -> None:
        self.errors.append(msg)

    # ── checks ────────────────────────────────────────────────────
    def check_lowering_config(self) -> None:
        cfg, contract = self.module.lowering, self.module.contract
        overlap = sorted(set(cfg.scratch_pool) & set(contract.preserves))
        if overlap:
            self._err(
                f"scratch_pool {overlap} intersects contract.preserves — "
                "a lowering that clobbers a preserved register is not a "
                "warning, it is a different contract")
        if cfg.callstack_reg in cfg.scratch_pool:
            self._err(
                f"callstack_reg {cfg.callstack_reg} in scratch_pool — "
                "call/ret bookkeeping must not be scratch-clobberable")

    def check_block_terminators(self) -> None:
        labels = set(self.module.block_labels())
        if len(labels) != len(self.module.blocks):
            self._err("duplicate block labels")
        for b in self.module.blocks:
            # The line-list raise maps RUNS of straight-line ops to one
            # block; fall-through runs (no terminator) are legal and
            # simply continue into the next block, so only the unknown
            # TARGET check applies here.
            if (b.terminator in ("JMP", "JZ")
                    and b.terminator_target
                    and b.terminator_target.lstrip(":") not in labels):
                self._err(
                    f"block '{b.label}' targets unknown block "
                    f"'{b.terminator_target}'")

    def check_code_window(self) -> None:
        w = self.module.code_window
        if w is None:
            return
        total = 0
        for b in self.module.blocks:
            total += len(b.instructions)
            if b.terminator:
                total += 1
        if total > w.max_words:
            self._err(
                f"module '{self.module.name}' needs {total} words but the "
                f"'{w.origin_symbol}' window holds {w.max_words}")

    def check_data_bounds(self) -> None:
        lo, hi = self.module.contract.data_bounds
        for ds in self.module.data_sections:
            end = ds.base_word + len(ds.words)
            if ds.base_word < lo or end > hi:
                self._err(
                    f"data section '{ds.symbol}' [{ds.base_word},{end}) "
                    f"escapes contract.data_bounds [{lo},{hi})")
            for rlo, rhi, who in self.RESERVED_RANGES:
                if ds.base_word < rhi and rlo < end:
                    self._err(
                        f"data section '{ds.symbol}' [{ds.base_word},{end}) "
                        f"collides with reserved {who} [{rlo},{rhi})")
        if self.module.pointer_table:
            pt = self.module.pointer_table
            end = pt.base_word + len(pt.block_targets)
            for rlo, rhi, who in self.RESERVED_RANGES:
                if pt.base_word < rhi and rlo < end:
                    self._err(
                        f"pointer table [{pt.base_word},{end}) collides "
                        f"with reserved {who} [{rlo},{rhi})")

    def check_symbols(self, symbols: Dict[str, int]) -> None:
        for b in self.module.blocks:
            for ins in b.instructions:
                for o in list(ins.sources) + list(ins.dests):
                    if o.symbol_ref:
                        try:
                            o.symbol_ref.resolve(symbols)
                        except StaticVerificationError as e:
                            self._err(
                                f"block '{b.label}': {e}")

    # ── entry ─────────────────────────────────────────────────────
    def verify(self, symbols: Optional[Dict[str, int]] = None) -> None:
        self.check_lowering_config()
        self.check_block_terminators()
        self.check_code_window()
        self.check_data_bounds()
        if symbols:
            self.check_symbols(symbols)
        if self.errors:
            raise StaticVerificationError(
                f"GlyphIR module '{self.module.name}' rejected:\n  - "
                + "\n  - ".join(self.errors))


# ── GH-15 Step 5: the ONE glyph-text -> GlyphIRModule raiser ──────────
# Steps 2-4 each landed a near-identical raiser (rv64i_to_glyph
# ._raise_to_ir, autoatlas.raise_tile_text_to_ir, baker
# ._raise_lines_to_ir). This shared raiser owns the parsing core:
# label/block splitting, terminator recognition from TERMINATOR_OPS,
# operand classification, duplicate-label rejection, fused entry block.
# Lanes differ ONLY in RaisePolicy (module shape), never in parsing.

_LABEL_LINE_RE = re.compile(r":([A-Za-z0-9_]+)\s*(.*)")
_REG_TOK_RE = re.compile(r"r[0-9]+")
_NUM_TOK_RE = re.compile(r"0x[0-9a-fA-F]+|-?[0-9]+")


@dataclass(frozen=True)
class RaisePolicy:
    """Per-lane module-shape policy for raise_lines_to_ir."""
    entry_label: str = "__entry"          # first block: <name><entry_label>
    fuse_entry: bool = True               # first ':label' fuses into entry
    cont_prefix: str = "__blk"            # continuations: <name><cont_prefix><NNN>
    strict_operands: bool = False         # loud reject on junk operands
    window: str = "program"               # "program" | "gh9" | "emitted"
    cols_instrs: int = 8
    preserves: Tuple[str, ...] = ()
    data_bounds: Tuple[int, int] = (0, 749)
    extra_metadata: Dict = field(default_factory=dict)


def raise_lines_to_ir(
    lines,
    name: str = "prog",
    policy: Optional[RaisePolicy] = None,
    window_max_words: Optional[int] = None,
    data_sections: Optional[List[DataSection]] = None,
    pointer_table: Optional[PointerTable] = None,
) -> GlyphIRModule:
    """Raise glyph assembly text (list of lines) into a GlyphIRModule.

    One block per label, fused runs into terminator-delimited blocks;
    a jump-target label ALWAYS exists as a block (empty label blocks
    preserved). First label becomes f'{name}{entry_label}'; subsequent
    labels keep their own name and later labels after a terminator get
    f'{name}{cont_prefix}{NNN}' continuations. Raises
    StaticVerificationError on malformed text (duplicate label,
    unparseable operand when policy.strict_operands).
    """
    pol = policy or RaisePolicy()
    blocks: List[BasicBlock] = []
    labels: Dict[str, int] = {}
    entry_label = f"{name}{pol.entry_label}"
    cur = BasicBlock(label=entry_label)
    first = True          # nothing flushed yet
    seen_code = False     # any instruction/terminator line processed
    seen_labels: set = set()

    def _flush() -> None:
        if not first and cur is not None:
            blocks.append(cur)

    def _new_cont() -> BasicBlock:
        return BasicBlock(
            label=f"{name}{pol.cont_prefix}{len(blocks):03d}")

    for raw in lines:
        s = raw.split(";", 1)[0].strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith(":"):
            m = _LABEL_LINE_RE.match(s)
            if m is None:
                raise StaticVerificationError(
                    f"module '{name}': illegal label line '{s}'")
            lbl, rest = m.group(1), m.group(2).strip()
            if (pol.fuse_entry and not seen_code and not blocks
                    and cur is not None and cur.label == entry_label):
                # leading label (no code before it) fuses into entry
                lbl = entry_label
            if lbl in seen_labels:
                raise StaticVerificationError(
                    f"duplicate label ':{lbl}' in module '{name}'")
            seen_labels.add(lbl)
            _flush()
            first = False
            cur = BasicBlock(label=lbl)
            labels[lbl] = len(blocks)
            s = rest
            if not s:
                continue
        parts = s.split()
        op = parts[0].upper()
        seen_code = True
        if op in TERMINATOR_OPS:
            if cur is None:
                # code directly after a terminator without a label:
                # materialize the continuation block now
                cur = BasicBlock(
                    label=f"{name}{pol.cont_prefix}{len(blocks):03d}")
                first = False
            cur.terminator = op
            cur.terminator_target = (
                parts[1].lstrip(":") if len(parts) > 1 else None)
            blocks.append(cur)
            first = False
            # Continuation block is created lazily: materialize it only
            # when another instruction actually follows, so a program
            # ending on its terminator doesn't grow a trailing empty
            # block (Step 5 gate: terms[-1] must be the HALT).
            cur = None
            continue
        if cur is None:
            cur = BasicBlock(
                label=f"{name}{pol.cont_prefix}{len(blocks):03d}")
            first = False
        srcs = []
        for tok in parts[1:]:
            if _REG_TOK_RE.fullmatch(tok):
                srcs.append(Operand.reg(tok))
            elif _NUM_TOK_RE.fullmatch(tok):
                srcs.append(Operand.const(int(tok, 0)))
            elif pol.strict_operands:
                raise StaticVerificationError(
                    f"module '{name}': unparseable operand '{tok}' "
                    f"in '{s}'")
        dests = ([Operand.reg(parts[1])]
                 if len(parts) > 1 and parts[1].startswith("r") else [])
        meta = dict(pol.extra_metadata)
        cur.append(IRInstruction(op=op, dests=tuple(dests),
                                 sources=tuple(srcs), metadata=meta))
    if cur is not None and (cur.instructions or cur.terminator
                            or not first):
        blocks.append(cur)

    n_instr = sum(len(b.instructions) + (1 if b.terminator else 0)
                  for b in blocks)
    if window_max_words is None:
        window_max_words = max(n_instr, 1)
    origin_symbol = {"program": "baked-image", "gh9": "__g9window",
                     "emitted": "emitted-image"}.get(pol.window,
                                                     pol.window)
    return GlyphIRModule(
        name=name,
        blocks=blocks,
        data_sections=(data_sections if data_sections is not None
                       else [DataSection(symbol=f"{name}_text",
                                         base_word=0, words=[])]),
        pointer_table=pointer_table,
        code_window=CodeWindow(origin_symbol=origin_symbol,
                               max_words=window_max_words,
                               cols_instrs=pol.cols_instrs),
        contract=IRContract(preserves=list(pol.preserves),
                            data_bounds=pol.data_bounds),
        lowering=LoweringConfig(
            scratch_pool=list(DEFAULT_SCRATCH_POOL),
            callstack_reg=DEFAULT_CALLSTACK_REG),
    )
