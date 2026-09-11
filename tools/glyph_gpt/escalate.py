#!/usr/bin/env python3
"""escalate.py — GH-12 escalation router: atlas-miss → local Ollama draft → oracle gate.

The loop: a routine the RoutineAtlas can't emit is drafted by the local
model (qwen2.5-coder:14b via the Ollama REST API — no auth expiry, no
per-call cost, per Jericho's unattended-loop policy), assembled and
executed by oracle.run_oracle(), and retried with the exact register
mismatch fed back into the next prompt. Candidates that pass all N
contract checks are returned; candidates that fail NEVER reach the
atlas (the caller decides what to do with a verified-only result).

No frontier tokens: Claude/cloud is only for design forks, never for
routine drafting (GH-12 spec, GLYPH_SELF_HOSTING_ROADMAP.md:59).
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_TOOLS_DIR = Path(__file__).resolve().parent.parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from glyph_gpt.oracle import OracleResult, run_oracle  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5-coder:14b"

ISA_PRIMER = """\
You write programs in Glyph ISA v2 assembly. The COMPLETE opcode set:
- LDI rX imm        ; rd = imm (decimal or 0x hex). The ONLY way to load constants.
- ADD rd, rs2       ; rd = rd + rs2  (ALL ALU ops are 2-operand, in-place:
- SUB rd, rs2       ; rd = rd - rs2   NO 3-operand form, NO immediate ALU)
- AND rd, rs2       ; rd = rd & rs2   (so ANDI/ADDI/ORI do NOT exist —
- OR rd, rs2        ; rd = rd | rs2    load constants with LDI into a temp)
- XOR rd, rs2       ; rd = rd ^ rs2
- SHL rd, rs2       ; rd = rd << (rs2 & 31)
- SHR rd, rs2       ; rd = rd >> (rs2 & 31)
- ROTR rd, rs2      ; 32-bit rotate right
- CMP rd, rs2       ; sets r0 = 1 if equal else 0 (r0 is the compare flag)
- JZ :label         ; jump to label if r0 != 0   (THERE IS NO JNZ — the ONLY
                    ; conditional branch is JZ reading the CMP flag in r0.
                    ; Loop pattern: CMP r_counter, r_zero / JZ :exit / ... / JMP :loop)
- JMP :label        ; unconditional jump
- LD rd, raddr      ; rd = RAM[value of raddr]  (word-indexed)
- ST raddr, rs      ; RAM[value of raddr] = rs
- CALL :label / RET ; call / return
- HALT              ; REQUIRED as the last instruction.
Labels are lines starting with ':', e.g. ':loop'. Comments start with ';'.
Register plan for bit-manipulation tasks (use EXACTLY these roles):
  r1 = input, r2 = output, r3 = constant 1 (mask), r4 = constant 0 (zero),
  r5 = scratch copy, r6 = counter, r7 = constant 1.
The OUTPUT register named in the task (r2, r3, …) is YOURS to write but
must never do double duty as a constant holder — if the task says the
result goes in r3, load your 'constant 1' into a different scratch.
Worked example — count the bits of r1 into r2 (VERIFY this against the
task; do not blindly copy when the contract differs):
    LDI r3 1
    LDI r4 0
    LDI r6 32
    LDI r7 1
    XOR r2 r2
:loop
    CMP r6 r4
    JZ :done
    XOR r5 r5
    ADD r5 r1
    AND r5 r3
    CMP r5 r4
    JZ :skip
    ADD r2 r7
:skip
    SHR r1 r3
    SUB r6 r7
    JMP :loop
:done
    HALT
CRITICAL: every ALU op writes its result INTO rd, destroying rd's old
value — that is why the example copies r1 into r5 before masking. r7/r3
hold constants so they are never clobbered. An AND/CMP whose rd holds a
stale value reads that stale value as an operand — XOR the scratch
register with itself first if you need it to start at 0.
ANTI-PATTERN: 'AND r5 r1' when r5 is 0 yields 0 forever (0 AND anything
is 0). To test the low bit of r1, use exactly this idiom:
  XOR r5 r5 / ADD r5 r1 / AND r5 r3   (r3 holds the constant 1)
TERMINATION: the program MUST reach HALT within a few hundred steps.
For pure functions prefer straight-line code (no loop at all). If you
do loop, the loop body must contain a decrement of a counter register
toward a register holding 0, and a CMP + JZ that exits when it hits 0 —
copy the counter pattern from the worked example exactly.
LDI IMMEDIATE LIMIT: immediates must fit in 24 bits (max 16777215 =
0xFFFFFF). To build 0xFFFFFFFF: LDI rA 65535 / LDI rB 16 / SHL rA rB /
LDI rC 65535 / OR rA rC. NEVER write 'LDI rX 4294967295' — the value
wraps and your mask silently becomes 0xFFFFFF.
There is NO less-than branch. For max(a,b): d = a - b (SUB), then
LDI r4 31 / SHR d, r4 puts d's sign bit in d (1 iff a < b for the
vectors in range); branch on it with CMP d, rzero-flag / JZ.
SIGNED MIN(a,b): same subtraction trick — d = a - b, SHR d by 31 to
isolate the sign bit (d must HOLD a-b first: XOR d d / ADD d ra /
SUB d rb, then put 31 in a scratch and SHR d by that scratch); if the
sign bit is 1 then a < b and min is a, else min is b. Copy BOTH inputs
to scratch registers before subtracting (ALU ops destroy rd).
AVERAGE WITHOUT OVERFLOW: avg = (a AND b) + ((a XOR b) >> 1). NEVER
compute (a+b)/2 directly — the sum wraps at 32 bits and halves wrong.
CRITICAL — THE OUTPUT REGISTER STARTS DIRTY: zero it and accumulate
into it (XOR r3 r3, then ADD), and NEVER initialize it with LDI 1 —
an extra 1 left in the output corrupts every result. Copy a and b
into scratch registers first (XOR rS rS / ADD rS rA …).
AND is in-place too: to get a AND b into a scratch, copy a into it
FIRST (XOR r8 r8 / ADD r8 ra / AND r8 rb) — 'AND r8 rb' with r8 still
0 yields 0 forever (same anti-pattern as the low-bit test). To shift
by 1: 'LDI r7 1' then 'SHR r6 r7'.
OPERAND ORDER, ALL 2-OP FORMS: 'OP rd rs' computes rd = rd OP rs — rd
is BOTH destination AND first operand. 'AND r3 r4' when r3 holds a
constant computes constant AND r4, never 'r3 = r4 AND something'.
Destinations must be LOADED with a real operand (copy or zero) before
the op that writes them.
ROTATES: ROTR rd, rs2 rotates rd right by (rs2 & 31) bits IN ONE
instruction — no loop needed. ROTR IS IN-PLACE ON rd: copy the input
into rd first (XOR rd rd / ADD rd r_input), put the rotate COUNT in a
REGISTER ('LDI r5 8'), then 'ROTR rd r5'. NEVER write 'ROTR rd 8'
(immediates are only for LDI) and never rotate a register you have not
loaded with the value.
ABSOLUTE VALUE of r1: extract the sign bit — copy r1 into a scratch
first (XOR r5 r5 / ADD r5 r1), LDI a scratch 31, SHR the COPY by it,
AND with 1 (SHR shifts rd BY the value of rs2 — rd must already hold
the value; shifting an empty register gives 0; same for AND). If the
sign bit is 0 the answer is r1 itself; if 1, negate: result = 0 - r1
(XOR a scratch to 0, ADD r1, SUB from it). XOR the output register
with itself before any conditional accumulate.
NEGATE IDIOM — copy the value in, THEN subtract it from zero:
  XOR r2 r2 / ADD r2 r1 / SUB r2 r1   is r1 - r1 = 0? NO — trace it:
  the ONLY working negate is 'XOR r2 r2' then 'SUB r2 r1' (r2 = 0 - r1).
  'ADD r2 r1' followed by 'SUB r2 r_const' subtracts the CONSTANT, not
  r1 — that returns r1 unchanged. If you need a 0 register for CMP,
  LDI r4 0 it explicitly; for negation use the 2-line form above.
BRANCHING & LABELS:
- CMP rd rs2 sets r0=1 if equal, 0 if not equal.
- JZ :label takes ONLY the label name (NO register operand!). It jumps to :label iff r0 != 0.
- JMP :label takes ONLY the label name and jumps unconditionally.
- Every label MUST be on its own line (e.g. ':my_label'), NEVER on the same line as an instruction.
- SHIFTING: In 'SHR rd rs2' and 'SHL rd rs2', rd is shifted by the value in register rs2.
  The shift amount MUST be loaded into a register with LDI first (e.g. 'LDI r6 1' then 'SHR r5 r6').
  Never use immediate numbers as shift operands.
Output ONLY the assembly, no prose, no markdown fences.
"""


@dataclass
class EscalationResult:
    contract: str
    verified: bool
    attempts: int = 0
    oracle: Optional[OracleResult] = None
    glyph_text: Optional[str] = None
    history: List[Dict] = field(default_factory=list)   # per-attempt prompt/error log
    error: Optional[str] = None


def _ollama(prompt: str, model: str = OLLAMA_MODEL,
            timeout: int = 120) -> str:
    # num_ctx=8192 (2026-09-08 GH-18 gate receipt): Ollama defaults
    # num_ctx to 2048 — the ISA primer (~1.6k tokens) plus the ABI/task
    # text nearly fills it, and the model silently loses the tail of the
    # prompt (the ABI lines that say WHERE the input lives). Drafts then
    # read garbage iteration counts (r6=2 or 32 for a ×3 contract) while
    # looking syntactically perfect. 8192 keeps the whole prompt plus
    # headroom for the completion; temperature 0 stays deterministic.
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps({"model": model, "prompt": prompt, "stream": False,
                         "options": {"temperature": 0, "num_ctx": 8192}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"]


def _normalize_labels(text: str) -> str:
    """Accept both assembler forms: ':name' (canonical) and 'name:'.

    Also shims two model habits the assembler rejects outright:
    - comma operand separators ('AND r1, r4' -> 'AND r1 r4')
    - named registers (r_tmp -> r4, r_zero -> r5, r_one/r_bit -> r6,
      r_count/r_counter -> r7, r_bitcount/r_result -> r8) — assigned
      deterministically so the same candidate always maps the same way.

    The oracle gate exists to check SEMANTICS; syntax shims belong here,
    not in the retry budget.
    """
    import re
    _NAMED = {
        "r_tmp": "r4", "r_temp": "r4", "r_zero": "r5", "r_one": "r6",
        "r_bit": "r6", "r_count": "r7", "r_counter": "r7",
        "r_bitcount": "r8", "r_result": "r8", "r_inc": "r9",
        "r_increment": "r9", "r_mask": "r4", "r_sum": "r8",
        "r_total": "r8", "r_out": "r2", "r_ret": "r2", "r_val": "r4",
        "r_value": "r4", "r_i": "r7", "r_j": "r10", "r_k": "r11",
    }
    # generic fallback: any remaining r<word> identifier (r_mask2, r_carry,
    # …) maps deterministically into r10..r31 so the same name in one
    # candidate always becomes the same number.
    used = set(_NAMED.values())

    def _alloc(m: "re.Match") -> str:
        h = int(hashlib.md5(m.group(0).encode()).hexdigest(), 16)
        n = 10 + (h % 22)          # r10..r31
        while f"r{n}" in used:     # avoid colliding with fixed mappings
            n = 10 + (n - 9) % 22
        used.add(f"r{n}")
        return f"r{n}"
    # must start with a LETTER after r_ — 'r4' must NOT match (numeric
    # registers are already legal; scrambling them broke every candidate)
    # named map FIRST (deterministic), then generic fallback only for the
    # remainder — reversed order let the generic md5 allocator eat r_zero
    # and hand back a different register per occurrence.
    for name, num in _NAMED.items():
        text = re.sub(r"\b" + name + r"\b", num, text)
    text = re.sub(r"\br_[a-z][a-z_0-9]*\b", _alloc, text)
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if (stripped.endswith(":") and not stripped.startswith(":")
                and " " not in stripped and len(stripped) > 1):
            line = ":" + stripped[:-1]
        for name, num in _NAMED.items():
            line = re.sub(r"\b" + name + r"\b", num, line)
        # strip trailing comments: 'LDI r2 0  ; note' -> 'LDI r2 0'
        line = line.split(";", 1)[0].rstrip()
        # strip operand commas: 'ADD r1, r4' -> 'ADD r1 r4' (not inside labels)
        if not stripped.startswith(":"):
            line = re.sub(r",\s*", " ", line)
            # drop operand-less fragments ('LDI r4' with missing imm) that
            # would die in the assembler with a confusing int('') error
            toks = line.split()
            _arity = {"LDI": 2, "ADD": 2, "SUB": 2, "AND": 2, "OR": 2,
                      "XOR": 2, "SHL": 2, "SHR": 2, "ROTR": 2, "CMP": 2,
                      "LD": 2, "ST": 2, "JMP": 1, "JZ": 1, "CALL": 1}
            if toks and toks[0] in _arity and len(toks) - 1 < _arity[toks[0]]:
                line = "; dropped incomplete: " + line.strip()
            # LD/ST-immediate shim: 'LD rd 750' addresses RAM with an
            # immediate — the ISA only takes a register holding the
            # address. Rewrite through a dedicated scratch (r13) so the
            # candidate executes instead of crashing the oracle with
            # registers[750] (IndexError). 'ST 750 rs' likewise.
            toks = line.split()
            if (len(toks) == 3 and toks[0] == "LD"
                    and toks[1].startswith("r") and toks[2].lstrip("-").isdigit()):
                line = f"LDI r13 {int(toks[2])}\nLD {toks[1]} r13"
            elif (len(toks) == 3 and toks[0] == "ST"
                    and toks[1].lstrip("-").isdigit()
                    and toks[2].startswith("r")):
                line = f"LDI r13 {int(toks[1])}\nST r13 {toks[2]}"
            # 3-operand RISC-style ALU ('AND rd ra rb') -> in-place 2-op
            # equivalent sequence; models default to RISC-V muscle memory.
            toks = line.split()
            if (len(toks) == 4 and toks[0] in
                    ("ADD", "SUB", "AND", "OR", "XOR", "SHL", "SHR")
                    and all(t.startswith("r") for t in toks[1:])):
                rd, ra, rb = toks[1], toks[2], toks[3]
                if toks[0] in ("SHL", "SHR"):
                    line = f"{toks[0]} {ra} {rb}\nXOR {rd} {rd}\nADD {rd} {ra}"
                elif toks[0] == "SUB":
                    line = f"XOR {rd} {rd}\nADD {rd} {ra}\nSUB {rd} {rb}"
                else:
                    line = f"XOR {rd} {rd}\nADD {rd} {ra}\n{toks[0]} {rd} {rb}"
        out.append(line)
    return "\n".join(out)


def _strip_fences(text: str) -> str:
    """Extract assembly from a raw model response.

    Handles: bare asm, a single ```-fenced block, ```glyph-tagged blocks,
    and fenced blocks with trailing prose. The previous split-based logic
    silently discarded a lone fenced block (the most common qwen shape),
    returning '' — every attempt then died as 'empty output'.
    """
    import re
    blocks = re.findall(r"```[a-z]*\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return "\n".join(blocks).strip()
    if "```" in text:  # unterminated fence: take everything after the opening
        return text.split("```", 2)[1].lstrip("\n").strip()
    return text.strip()


def _run_oracle_guarded(glyph_text: str, **kw) -> OracleResult:
    """run_oracle with a crash guard: a malformed candidate (e.g. a
    register index beyond r31 reaching the CPU) must return retry
    feedback, never kill the escalation loop with an exception."""
    try:
        return run_oracle(glyph_text, max_instructions=5000, **kw)
    except (IndexError, KeyError, ValueError, ZeroDivisionError,
            OverflowError) as e:
        return OracleResult(
            passed=False,
            error=(f"oracle crash: {type(e).__name__}: {e} — the program "
                   "used an operand the ISA rejects; check register "
                   "indices (r0..r31) and LD/ST forms"))


def escalate(contract: str, expect_registers: Dict[int, int],
             input_registers: Optional[Dict[int, int]] = None,
             seed_memory: Optional[Dict[int, int]] = None,
             max_attempts: int = 4,
             model: str = OLLAMA_MODEL,
             extra_vectors: Optional[List[Dict]] = None) -> EscalationResult:
    """Draft a routine for `contract` until run_oracle passes all checks.

    contract: one-line behavior spec, e.g. "popcount(r1) -> r2".
    expect_registers: golden register values checked at HALT.
    input_registers: preloaded into registers before step 0 (caller ABI:
                     inputs arrive IN registers, outputs in registers).
    seed_memory: optional preloaded RAM words (argv / input data).
    extra_vectors: additional oracle proofs each candidate must pass
                   ({"seed_memory": {...}, "expect_registers": {...},
                   "input_registers": {...}?}). Gate 1.5 (2026-09-08
                   receipt): a single golden vector can admit a DUAL
                   function — an inverted-polarity popcount passed
                   0xF0F0F0F0 (16 set == 16 clear) then diverged on every
                   other input. Vectors live INSIDE the retry loop so a
                   dual candidate fails and collects targeted feedback.
    """
    result = EscalationResult(contract=contract, verified=False)
    feedback = ""
    feedback_extra = ""
    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt
        prompt = ISA_PRIMER + f"\nTask: {contract}\n"
        if input_registers:
            prompt += ("Inputs: the caller preloads "
                       + ", ".join(f"r{i}=0x{v:08x}" for i, v in input_registers.items())
                       + " before your first instruction runs.\n")
        if seed_memory:
            prompt += (f"Input: RAM seeded as {seed_memory} "
                       f"(word-indexed, 32-bit values).\n")
        if feedback:
            prompt += (f"\nYour previous attempt FAILED verification:\n"
                       f"{feedback}\nFix it and output the full corrected program.\n")
        raw = _ollama(prompt, model=model)
        glyph_text = _normalize_labels(_strip_fences(raw).strip())
        if not glyph_text:
            feedback = "empty output: model returned no assembly"
            result.history.append({"attempt": attempt, "prompt": prompt,
                                   "glyph": glyph_text, "error": feedback,
                                   "passed": False})
            result.oracle = OracleResult(passed=False, error=feedback)
            continue
        res = _run_oracle_guarded(glyph_text, expect_registers=expect_registers,
                                  seed_memory=seed_memory,
                                  input_registers=input_registers)
        if res.passed:
            # gate 1.5: candidate must also pass every extra vector
            for vec in (extra_vectors or []):
                vres = _run_oracle_guarded(glyph_text,
                                           expect_registers=vec["expect_registers"],
                                           seed_memory=vec.get("seed_memory"),
                                           input_registers=vec.get("input_registers"))
                if vres.passed:
                    continue
                res = OracleResult(
                    passed=False,
                    error=(f"extra vector diverged: {vres.error} "
                           f"(passed the primary vector — the program "
                           f"is semantically wrong, see the polarity and "
                           f"loop-bound notes below)"),
                    registers=vres.registers, steps=vres.steps,
                    memory_hash=vres.memory_hash)
                break
        # gate 1.5 surgical hint: when the primary vector passed but an
        # extra one diverged, name the two failure classes that produce
        # exactly this signature (2026-09-08 GH-12 receipt: qwen 6/6
        # counted CLEAR bits over 31 iterations — got 27 = 31-4 — because
        # its 'CMP r5 r3 / JZ :skip' pair skips when the bit IS set).
        if res.passed is False and "extra vector diverged" in (res.error or ""):
            feedback_extra = (
                "\nEXTRA-VECTOR ANALYSIS: your program returned a WRONG "
                "value on a second input while passing the first. The two "
                "classic causes:\n"
                "(1) INVERTED POLARITY — if your loop contains "
                "'CMP r5 r3' with r3=1 followed by 'JZ :skip', you jump "
                "to :skip when the bit IS set, so the ADD counts the ZERO "
                "bits. Count SET bits instead: compare the masked bit "
                "against a register holding ZERO —\n"
                "  LDI r4 0\n"
                "  XOR r5 r5\n"
                "  ADD r5 r1\n"
                "  AND r5 r3      ; r5 = r1 & 1\n"
                "  CMP r5 r4      ; compare bit against ZERO register\n"
                "  JZ :skip       ; skip when bit is CLEAR\n"
                "  ADD r2 r3      ; count the SET bit\n"
                ":skip\n"
                "(2) WRONG LOOP BOUND — exit when the counter register "
                "reaches ZERO ('CMP r6 r4' with LDI r4 0, 'JZ :done'), "
                "and initialize the counter to the FULL width (LDI r6 32). "
                "Exiting when the counter hits 1 (CMP r6 r3) drops the "
                "last iteration.\n"
                "Rewrite the loop with BOTH fixes.")
        result.history.append({"attempt": attempt, "prompt": prompt,
                               "glyph": glyph_text,
                               "error": res.error, "passed": res.passed})
        if res.passed:
            result.verified = True
            result.oracle = res
            result.glyph_text = glyph_text
            return result
        feedback = (res.error or "unknown failure") + feedback_extra
        feedback_extra = ""
        # Semantic feedback: show final register state so the model can see
        # WHICH register went wrong (e.g. it destroyed its input r1 with an
        # in-place AND before testing).
        if res.registers and "contract:" in feedback:
            nonzero = {f"r{i}": hex(v) for i, v in enumerate(res.registers)
                       if v and i > 0}
            feedback += f"\nFinal registers at HALT: {nonzero}"
        # Add syntax-repair hints so the retry targets the actual failure
        # class (named registers, fence tags) instead of blind retrying.
        if "assemble: KeyError" in feedback:
            # Unknown-opcode failure class (2026-09-08 GH-18 gate receipt:
            # qwen drafted 'MUL r2 r1 r3' 6/6 attempts — the ISA has NO
            # multiply, only shift-add; the bare KeyError gave the model
            # zero signal, so every retry repeated the same opcode).
            # Surface the bad mnemonic and the shift-add replacement.
            import re as _re
            m = _re.search(r"KeyError: '(\w+)'", feedback)
            bad = m.group(1) if m else "the op"
            feedback += (f"\nNOTE: '{bad}' is NOT an opcode in Glyph ISA v2 "
                         "(the COMPLETE set is only: LDI ADD SUB AND OR XOR "
                         "SHL SHR ROTR CMP JZ JMP LD ST CALL RET HALT "
                         "SYSRET SYSCALL KJMP JMPR). There is NO multiply: "
                         "multiply by a CONSTANT c with straight-line "
                         "shift-add. For c=3 (y = 3*x) use EXACTLY this "
                         "shape, then do the ABI stores:\n"
                         "  LD r1 r15        (or however the input arrives)\n"
                         "  XOR r2 r2\n"
                         "  ADD r2 r1        ; r2 = x\n"
                         "  LDI r13 1\n"
                         "  SHL r2 r13       ; r2 = 2*x\n"
                         "  ADD r2 r1        ; r2 = 3*x\n"
                         "NO LOOPS for constant multiply — a loop needs a "
                         "terminating condition and x*3 has none (see your "
                         "previous no-halt failure).")
        if "invalid literal for int()" in feedback and "CMP" in glyph_text:
            # CMP-with-immediate failure class (2026-09-08 GH-12 gate
            # receipt: qwen drafted 'CMP r6 0' / 'CMP r5 0' — CMP takes
            # TWO REGISTERS, never an immediate; the bare ValueError gave
            # no signal, so retries alternated between the immediate form
            # and a r4/r7 variant that never exits its loop). Point at the
            # exact line and give the register-compare + XOR-zero pattern.
            import re as _re
            m = _re.search(r"CMP (r\d+) (\d+)", glyph_text)
            bad_line = m.group(0) if m else "CMP rn <imm>"
            feedback += (f"\nNOTE: '{bad_line}' is INVALID — CMP takes TWO "
                         "REGISTERS ('CMP r6 r3'), never an immediate. To "
                         "test against a constant: first put the constant "
                         "in a register with LDI and XOR-zero EVERY "
                         "register you compare or accumulate with:\n"
                         "  XOR r2 r2\n"
                         "  XOR r5 r5\n"
                         "  LDI r3 1\n"
                         "  LDI r6 32        ; loop counter in a REGISTER\n"
                         ":loop\n"
                         "  CMP r6 r3        ; registers only!\n"
                         "  JZ :done\n"
                         "  ...\n"
                         "  SUB r6 r3\n"
                         "  JMP :loop\n"
                         "ALL of r2 r5 r3 r6 must be zeroed/initialized "
                         "BEFORE the loop — the kernel enters with every "
                         "register DIRTY.")
        if "invalid literal" in feedback or "KeyError: 'assembly'" in feedback:
            feedback += ("\nNOTE: registers are r0..r31 with NUMBERS only "
                         "(r_tmp/r_zero are invalid — use r4, r5…). Output "
                         "the code block WITHOUT a ```assembly language tag.")
        elif "empty output" in feedback:
            feedback += "Output the program inside a plain ``` fence."
        result.oracle = res
    result.error = f"no candidate verified in {max_attempts} attempts; last: {feedback}"
    return result
