/* G3 scale test: a real central scheduler round-robins 3 tasks via switch_to,
 * each task doing several work units in a genuine `for` loop (not manually
 * unrolled sequential yields like switch_to.c's Primitive 6 fixture). Tests
 * the transpiler + GlyphCPUv2 at size (label/pointer-table resolution over
 * a few hundred instructions) and the RET/resume dataflow fix under a real
 * loop-driven control-flow pattern instead of straight-line code.
 *
 * struct context is padded to 16 longs (64 bytes, power of 2) so indexing
 * ctx_task[id] by a RUNTIME id is a shift, not a multiply -- -nostdlib has
 * no __mulsi3. (task_task[id] init in run_all uses compile-time-constant
 * indices 0/1/2 so wouldn't have needed this, but task_body/scheduler index
 * by a live loop variable and do.)
 */

struct context {
    long ra, sp, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11;
    long _pad0, _pad1;              /* 14 -> 16 longs = 64 bytes */
};

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

#define NTASKS 3
#define WORK_UNITS 3

struct context ctx_sched;
struct context ctx_task[NTASKS];
long stack_task[NTASKS][64];

long g_log[16];
long g_log_idx;
long g_rounds[NTASKS];
long g_done_mask;
long g_total_switches;

__attribute__((noinline)) void task_body(long id) {
    long i;
    for (i = 0; i < WORK_UNITS; i = i + 1) {
        g_log[g_log_idx] = (id << 8) | i;
        g_log_idx = g_log_idx + 1;
        g_rounds[id] = g_rounds[id] + 1;
        switch_to(&ctx_task[id], &ctx_sched);   /* yield back to scheduler */
    }
    g_done_mask = g_done_mask | (1 << id);
    switch_to(&ctx_task[id], &ctx_sched);       /* final yield: announces done */
    for (;;) { }                                /* must not be reached */
}

/* Raw asm tail-jump, not a C call: task_body() NEVER returns (it always
 * exits via switch_to's non-returning jump, or spins in `for(;;)`), but a
 * plain `call task_body` here (what GCC emits for `task_body(0);` in a C
 * function body -- verified: `jal ra,task_body`, not a tail call) still
 * pushes a glyph call-stack frame that nothing will ever pop. After enough
 * accumulated activity, an UNRELATED function's own legitimate `ret` pops
 * one of these orphaned entries instead of its true return address. `j` is
 * a genuine RISC-V jump (rd=zero), never pushes anything. */
__asm__(
    ".global task0_entry\n"
    "task0_entry:\n"
    "  li a0, 0\n"
    "  j task_body\n"
    ".global task1_entry\n"
    "task1_entry:\n"
    "  li a0, 1\n"
    "  j task_body\n"
    ".global task2_entry\n"
    "task2_entry:\n"
    "  li a0, 2\n"
    "  j task_body\n"
);
extern void task0_entry(void);
extern void task1_entry(void);
extern void task2_entry(void);

__attribute__((noinline)) void scheduler(void) {
    long i;
    long active = NTASKS;
    while (active > 0) {
        for (i = 0; i < NTASKS; i = i + 1) {
            if (!(g_done_mask & (1 << i))) {
                g_total_switches = g_total_switches + 1;
                switch_to(&ctx_sched, &ctx_task[i]);   /* runtime-indexed: needs power-of-2 struct */
                if (g_done_mask & (1 << i)) {
                    active = active - 1;
                }
            }
        }
    }
}

__attribute__((noinline)) void run_all(void) {
    ctx_task[0].ra = (long)&task0_entry;
    ctx_task[0].sp = (long)&stack_task[0][63];
    ctx_task[1].ra = (long)&task1_entry;
    ctx_task[1].sp = (long)&stack_task[1][63];
    ctx_task[2].ra = (long)&task2_entry;
    ctx_task[2].sp = (long)&stack_task[2][63];
    scheduler();
}

void _start(void) {
    __asm__ volatile (
        /* 0x800 (used by the smaller fixtures) lands INSIDE stack_task's
         * range here (3 * 256B task stacks push .bss well past it) --
         * verified via readelf before trusting this, same lesson as every
         * prior fixture's address math. */
        "li sp, 0x1000\n"
        "call run_all\n"
        "ecall\n"
    );
}
