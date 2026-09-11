/* Two-task cooperative context switch, swtch.S-style. Forces the exact
 * collision flagged after the proc-table primitive: switch_to's terminal
 * `ret` (jalr zero, 0(ra)) ALWAYS executes with ra freshly reloaded from a
 * data pointer (`lw ra, 0(a1)`), never call-preserved -- whether that jump
 * lands on a task's fresh entry point or resumes it mid-function. The
 * transpiler's syntactic RET rule (rd==0 && rs1==ra && imm==0 -> RET, pop
 * the glyph call stack) is wrong for every one of these; it must go through
 * the same computed-jump path proc-table's fn-pointer JALR uses.
 *
 * Ground truth (verify natively on x86 BEFORE trusting): g_log fills
 * [0x1111, 0x2222, 0x3333, 0x4444] in that order, g_log_idx=4, g_done=1.
 */

/* ra, sp, and every callee-saved register (s0-s11 per the RV32 ABI) --
 * GCC at -O1 keeps loop-invariant values (e.g. a cached array base) live in
 * s0-s3 across an ordinary `call`, trusting the callee to preserve them.
 * switch_to IS an ordinary call from each task's point of view, so it must
 * honor that contract even though it's also doing something the ABI never
 * anticipated (jumping to a different stack/pc entirely). Omitting s0-s11
 * here is a real bug this fixture hit on first compile: task A's use of s0
 * silently corrupted task B's cached base pointer across the switch. */
struct context {
    long ra, sp;
    long s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11;
};

/* Raw asm, not a C function body: a C prologue would clobber sp/ra before we
 * get to save them, same reason _start must be pure asm. Mirrors swtch.S. */
__asm__(
    ".global switch_to\n"
    "switch_to:\n"
    "  sw ra,  0(a0)\n"
    "  sw sp,  4(a0)\n"
    "  sw s0,  8(a0)\n"
    "  sw s1, 12(a0)\n"
    "  sw s2, 16(a0)\n"
    "  sw s3, 20(a0)\n"
    "  sw s4, 24(a0)\n"
    "  sw s5, 28(a0)\n"
    "  sw s6, 32(a0)\n"
    "  sw s7, 36(a0)\n"
    "  sw s8, 40(a0)\n"
    "  sw s9, 44(a0)\n"
    "  sw s10,48(a0)\n"
    "  sw s11,52(a0)\n"
    "  lw ra,  0(a1)\n"
    "  lw sp,  4(a1)\n"
    "  lw s0,  8(a1)\n"
    "  lw s1, 12(a1)\n"
    "  lw s2, 16(a1)\n"
    "  lw s3, 20(a1)\n"
    "  lw s4, 24(a1)\n"
    "  lw s5, 28(a1)\n"
    "  lw s6, 32(a1)\n"
    "  lw s7, 36(a1)\n"
    "  lw s8, 40(a1)\n"
    "  lw s9, 44(a1)\n"
    "  lw s10,48(a1)\n"
    "  lw s11,52(a1)\n"
    "  ret\n"
);
extern void switch_to(struct context *old, struct context *new);

struct context ctx_main, ctx_a, ctx_b;
long g_log[8];
long g_log_idx;
long g_done;

long stack_a[64];
long stack_b[64];

__attribute__((noinline)) void task_a(void) {
    g_log[g_log_idx] = 0x1111; g_log_idx = g_log_idx + 1;
    switch_to(&ctx_a, &ctx_b);          /* fresh-start B */
    g_log[g_log_idx] = 0x3333; g_log_idx = g_log_idx + 1;
    switch_to(&ctx_a, &ctx_b);          /* resume B */
    for (;;) { }                        /* must not be reached */
}

__attribute__((noinline)) void task_b(void) {
    g_log[g_log_idx] = 0x2222; g_log_idx = g_log_idx + 1;
    switch_to(&ctx_b, &ctx_a);          /* resume A */
    g_log[g_log_idx] = 0x4444; g_log_idx = g_log_idx + 1;
    g_done = 1;
    switch_to(&ctx_b, &ctx_main);       /* resume run_all, done */
    for (;;) { }                        /* must not be reached */
}

__attribute__((noinline)) void run_all(void) {
    ctx_a.ra = (long)&task_a;
    ctx_a.sp = (long)&stack_a[63];
    ctx_b.ra = (long)&task_b;
    ctx_b.sp = (long)&stack_b[63];
    switch_to(&ctx_main, &ctx_a);       /* fresh-start A; returns here when B is done */
}

void _start(void) {
    __asm__ volatile (
        "li sp, 0x800\n"
        "call run_all\n"
        "ecall\n"
    );
}
