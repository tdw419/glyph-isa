/* xv6-nano: cooperative kernel composed from the verified G1-G12 primitives.
 * See systems/XV6_NANO_ROADMAP.md. Scenario-selected at compile time (D5):
 *
 *   SCENARIO 1 (K1): unified image -- kinit freelist, userinit proc table,
 *                    scheduler round-robins 3 tasks, each writes a console
 *                    marker per work unit and sys_yield()s.
 *
 * Every hardware-dependent xv6 part is a convention here: no MMU, no traps,
 * no privilege levels; scheduling is cooperative; "syscalls" are plain C
 * calls (roadmap D2). switch_to is the verified swtch.S-style primitive
 * (G1/G3) -- do not edit the asm, the register set (ra/sp/s0-s11) and the
 * `ret` through a data-loaded ra are load-bearing.
 */
#ifndef SCENARIO
#define SCENARIO 1
#endif

typedef unsigned int uint;

/* --- verified switch_to (G1/G3): context = 16 longs (64B, power-of-2) --- */
struct context {
    long ra, sp, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11;
    long _pad0, _pad1;
};
__asm__(
    ".global switch_to\n"
    "switch_to:\n"
    "  sw ra,  0(a0)\n  sw sp,  4(a0)\n"
    "  sw s0,  8(a0)\n  sw s1, 12(a0)\n  sw s2, 16(a0)\n  sw s3, 20(a0)\n"
    "  sw s4, 24(a0)\n  sw s5, 28(a0)\n  sw s6, 32(a0)\n  sw s7, 36(a0)\n"
    "  sw s8, 40(a0)\n  sw s9, 44(a0)\n  sw s10,48(a0)\n  sw s11,52(a0)\n"
    "  lw ra,  0(a1)\n  lw sp,  4(a1)\n"
    "  lw s0,  8(a1)\n  lw s1, 12(a1)\n  lw s2, 16(a1)\n  lw s3, 20(a1)\n"
    "  lw s4, 24(a1)\n  lw s5, 28(a1)\n  lw s6, 32(a1)\n  lw s7, 36(a1)\n"
    "  lw s8, 40(a1)\n  lw s9, 44(a1)\n  lw s10,48(a1)\n  lw s11,52(a1)\n"
    "  ret\n"
);
extern void switch_to(struct context *old, struct context *new);

/* --- kalloc freelist (G7 shape) --- */
#define NPROC   3
#define PGSIZE  128

struct run { struct run *next; };
static struct run *freelist;
static char kmem_pool[NPROC + 1][PGSIZE] __attribute__((aligned(8)));

static void kinit(void) {
    freelist = 0;
    for (int i = 0; i < NPROC + 1; i++) {
        struct run *r = (struct run *)kmem_pool[i];
        r->next = freelist;
        freelist = r;
    }
}
static void *kalloc(void) {
    struct run *r = freelist;
    if (r) freelist = r->next;
    return (void *)r;
}

/* --- proc table (G4 array-of-structs; scheduler carries an index alongside
 *     the running pointer, so no runtime multiply / __mulsi3) --- */
enum { UNUSED = 0, RUNNABLE, RUNNING, DONE, ZOMBIE, FAULTED };

struct proc {
    struct context context;   /* 64B */
    int state;
    int pid;
    int xcode;                /* K2: sys_exit() code, read by the reaper */
    int kstack;               /* E-K1: base of this proc's kalloc'd stack page (box2) */
    int _pad[12];             /* -> 128B (power of 2) */
};
struct proc proc[NPROC];
struct proc *curproc;
struct context ctx_sched;

/* --- console (G12 style) --- */
char g_console[128];
int  g_clen;
static void cons_putc(char c) { if (g_clen < 128) g_console[g_clen++] = c; }

/* --- oracle globals --- */
int g_total_switches;
int g_iters[NPROC];
int g_xcode[NPROC];          /* K2: reaped exit codes */
int g_reaped_mask;           /* K2: bit i set when proc[i] was reaped */

/* --- kernel services (D2: plain C calls) ---
 * sys_yield/sys_exit MUST inline: they wrap switch_to, whose terminal `ret`
 * is data-sourced and jumps to the scheduler rather than returning normally.
 * A non-inlined wrapper leaves its own glyph call-stack frame unpopped
 * (switch_to's `POP r28` balances only switch_to's frame -- G3 Bug D). One
 * leak per yield accumulates until the scheduler's genuine `ret` pops a
 * stale frame and jumps into garbage. Inlining removes the wrapper frame. */
static inline __attribute__((always_inline))
void sys_yield(void) { switch_to(&curproc->context, &ctx_sched); }

static inline __attribute__((always_inline))
void sys_exit(int code) {
    curproc->xcode = code;
    curproc->state = ZOMBIE;
    switch_to(&curproc->context, &ctx_sched);   /* never returns */
    for (;;) { }
}

static void sys_write(const char *buf, int len) {   /* returns normally: no leak */
    for (int i = 0; i < len; i++) cons_putc(buf[i]);
}

/* --- tasks --- */
#define WORK_UNITS 3

__attribute__((noinline)) void task_body(long id) {
    int i;
    for (i = 0; i < WORK_UNITS; i++) {
        cons_putc((char)('A' + id));
        g_iters[id] = g_iters[id] + 1;
        sys_yield();
    }
    curproc->state = DONE;
    sys_yield();
    for (;;) { }                 /* scheduler never re-runs a DONE proc */
}

/* K2: a task that does `units` work units then sys_exit(code). */
__attribute__((noinline)) void task_exiter(long id) {
    int i;
    for (i = 0; i < 2; i++) {
        cons_putc((char)('A' + id));
        g_iters[id] = g_iters[id] + 1;
        sys_yield();
    }
    sys_exit(0x42);
    for (;;) { }
}

/* K3: spatial memory contract. proc[i]'s box is
 *   [g_arena + i*ARENA_SLOT, g_arena + (i+1)*ARENA_SLOT).
 * box_fill fills its own slot (in-box). box_over fills its
 * slot then writes ONE byte past the whole arena (out of box). Enforcement
 * is a harness-side checker (K3) -- no MMU, no engine fault (D1). */
#define ARENA_SLOT  64
#define ARENA_BYTES (NPROC * ARENA_SLOT)          /* contract limit */
char g_arena[ARENA_BYTES + 8] __attribute__((aligned(8)));  /* +8 guard: the
                    OOB write lands in real storage, just outside the box */
int  g_boxsum[NPROC];        /* oracle: sum of bytes each task wrote */

__attribute__((noinline)) void box_fill(long id) {
    char *base = g_arena + id * ARENA_SLOT;   /* id*64 -> slli, no multiply */
    int s = 0, i;
    for (i = 0; i < ARENA_SLOT; i++) {
        base[i] = (char)(0x41 + id);          /* 'A' / 'B' / 'C' */
        s += (unsigned char)base[i];
    }
    g_boxsum[id] = s;
    curproc->state = DONE;
    sys_yield();
    for (;;) { }
}

__attribute__((noinline)) void box_over(long id) {
    char *base = g_arena + id * ARENA_SLOT;
    int s = 0, i;
    for (i = 0; i < ARENA_SLOT; i++) {
        base[i] = (char)(0x41 + id);
        s += (unsigned char)base[i];
    }
    g_arena[ARENA_BYTES] = 0x7F;               /* one byte past the box */
    g_boxsum[id] = s;
    curproc->state = DONE;
    sys_yield();
    for (;;) { }
}

/* K4: nano-shell. A canned input buffer, a first-word tokenizer, and a
 * cmd_table[] of {name, fn} dispatched through a function pointer -- the G5
 * proc-table `jalr` primitive (the harness seeds PTR_TABLE_BASE for it). */
char g_shell_input[] = "help\necho hi there\nps\n";

static int str_eq(const char *a, const char *b) {
    while (*a && *a == *b) { a++; b++; }
    return *a == *b;                     /* both hit NUL */
}
static char state_char(int st) {
    /* UNUSED RUNNABLE RUNNING DONE ZOMBIE */
    return (st == RUNNING) ? 'R' : (st == DONE) ? 'D'
         : (st == ZOMBIE)  ? 'Z' : (st == RUNNABLE) ? 'r' : '-';
}

__attribute__((noinline)) void cmd_help(const char *rest) {
    (void)rest;
    sys_write("help ps echo\n", 13);
}
__attribute__((noinline)) void cmd_ps(const char *rest) {
    (void)rest;
    for (int i = 0; i < NPROC; i++) cons_putc(state_char(proc[i].state));
    cons_putc('\n');
}
__attribute__((noinline)) void cmd_echo(const char *rest) {
    while (*rest && *rest != '\n') cons_putc(*rest++);   /* rest is the raw line */
    cons_putc('\n');
}

struct cmd { const char *name; void (*fn)(const char *); };
struct cmd cmd_table[3] = {
    { "help", cmd_help },
    { "ps",   cmd_ps   },
    { "echo", cmd_echo },
};

__attribute__((noinline)) void shell_body(long unused) {
    (void)unused;
    char tok[16];
    const char *p = g_shell_input;
    while (*p) {
        int n = 0;
        while (*p && *p != ' ' && *p != '\n' && n < 15) tok[n++] = *p++;
        tok[n] = 0;
        while (*p == ' ') p++;
        const char *rest = p;                /* remainder of the line */
        while (*p && *p != '\n') p++;
        if (*p == '\n') p++;
        for (int i = 0; i < 3; i++) {
            if (str_eq(tok, cmd_table[i].name)) {
                cmd_table[i].fn(rest);        /* jalr through fn pointer (G5) */
                break;
            }
        }
    }
    curproc->state = DONE;
    sys_yield();
    for (;;) { }
}
__asm__(".global shent\nshent:\n  li a0, 0\n  j shell_body\n");
extern void shent(void);

/* raw-asm trampolines: `j <fn>`, never `call` -- the task fns never return,
 * a call would leak a glyph call-stack frame (G3 lesson). */
__asm__(
    ".global t0e\nt0e:\n  li a0, 0\n  j task_body\n"
    ".global t1e\nt1e:\n  li a0, 1\n  j task_body\n"
    ".global t2e\nt2e:\n  li a0, 2\n  j task_body\n"
    ".global t1x\nt1x:\n  li a0, 1\n  j task_exiter\n"
    ".global bw_a\nbw_a:\n  li a0, 0\n  j box_fill\n"
    ".global bw_b\nbw_b:\n  li a0, 1\n  j box_fill\n"
    ".global bw_c\nbw_c:\n  li a0, 2\n  j box_fill\n"
    ".global bw_x\nbw_x:\n  li a0, 2\n  j box_over\n"
);
extern void t0e(void), t1e(void), t2e(void), t1x(void);
extern void bw_a(void), bw_b(void), bw_c(void), bw_x(void);

#if SCENARIO == 6 || SCENARIO == 7 || SCENARIO == 8 || SCENARIO == 9 || SCENARIO == 10
/* --- E-K1/E-K2/GO-2/GO-3/GO-4: GlyphCPUv2- and SpatialRV64ICore-enforced isolation ---
 * systems/XV6_NANO_ISOLATION_ROADMAP.md. GlyphCPUv2 reads this reserved
 * MMIO-style word block every step; byte addresses MUST match the
 * BOX_MMIO_BASE constants in glyph_dispatch/src/glyph/glyph_isa_v2.py.
 * The scheduler (SUPER) programs each task's 3-range box + arms MODE_LATCH
 * before switching in; the engine flips USER on switch_to's terminal KJMP,
 * traps an out-of-box user store back to fault_handler (E-K1), and (E-K2)
 * traps `ebreak` to the SUPER-mode syscall_dispatch at KSYS_PC, `mret`
 * back. KFAULT_PC / KSYS_PC are loader-seeded (the kernel can't name a
 * pixel coordinate).
 */
#define ISO_MODE_LATCH  (*(volatile uint *)0x8000)
#define ISO_BOX0_LO     (*(volatile uint *)0x800C)
#define ISO_BOX0_HI     (*(volatile uint *)0x8010)
#define ISO_BOX1_LO     (*(volatile uint *)0x8014)
#define ISO_BOX1_HI     (*(volatile uint *)0x8018)
#define ISO_FAULT_ADDR  (*(volatile uint *)0x801C)
#define ISO_BOX2_LO     (*(volatile uint *)0x8028)
#define ISO_BOX2_HI     (*(volatile uint *)0x802C)
#define ISO_SYS_N       (*(volatile uint *)0x8030)
#define ISO_SYS_A0      (*(volatile uint *)0x8034)
#define ISO_SYS_A1      (*(volatile uint *)0x8038)
/* GO-2: the box as a 2D tile. Byte addr a -> word a>>2 -> grid (row,col) =
 * (w / W_MEM, w % W_MEM). W_MEM MUST match const W_MEM in
 * tools/SPATIAL_RV64I.wgsl and W_MEM in glyph_dispatch/src/glyph/glyph_isa_v2.py. */
#define W_MEM           32
#define ISO_TILE_ROW    (*(volatile uint *)0x8160)
#define ISO_TILE_COL    (*(volatile uint *)0x8164)
#define ISO_TILE_H      (*(volatile uint *)0x8168)
#define ISO_TILE_W      (*(volatile uint *)0x816C)

int  g_fault_pid;
uint g_fault_addr;

/* Vectored to (in SUPER) by GlyphCPUv2 after an out-of-box user store; the
 * engine has already set mode=SUPER and recorded ISO_FAULT_ADDR. Reap just
 * this proc and hand back to the scheduler -- same exit shape as sys_exit
 * (switch_to away, never return; no glyph call frame of its own to unwind). */
__attribute__((noinline)) void fault_handler(void) {
    g_fault_pid = curproc->pid;
    g_fault_addr = ISO_FAULT_ADDR;
    curproc->state = FAULTED;
    switch_to(&curproc->context, &ctx_sched);
    for (;;) { }
}
#endif  /* SCENARIO == 6 || 7 || 8 || 9 || 10 */

#if SCENARIO == 6
/* SCENARIO 6 tasks: touch ONLY their own arena slot + their own proc[]
 * entry -- no console (g_console is kernel memory; that needs E-K2 SYSCALL). */
__attribute__((noinline)) void s6_inbox(long id) {
    char *base = g_arena + id * ARENA_SLOT;
    int i;
    for (i = 0; i < ARENA_SLOT; i++) base[i] = (char)(0x41 + id);
    curproc->state = DONE;
    sys_yield();
    for (;;) { }
}
__attribute__((noinline)) void s6_over(long id) {
    char *base = g_arena + id * ARENA_SLOT;
    int i;
    for (i = 0; i < ARENA_SLOT; i++) base[i] = (char)(0x41 + id);
    g_arena[ARENA_BYTES] = 0x7F;     /* one byte past the arena -> out of box -> trap */
    curproc->state = DONE;           /* never reached */
    sys_yield();
    for (;;) { }
}
__asm__(
    ".global s6a\ns6a:\n  li a0, 0\n  j s6_inbox\n"
    ".global s6b\ns6b:\n  li a0, 1\n  j s6_inbox\n"
    ".global s6x\ns6x:\n  li a0, 2\n  j s6_over\n"
);
extern void s6a(void), s6b(void), s6x(void), fault_handler(void);
#endif  /* SCENARIO == 6 */

#if SCENARIO == 7 || SCENARIO == 9 || SCENARIO == 10
/* --- E-K2: SYSCALL boundary (shared by SCENARIO 7, GO-3's SCENARIO 9, GO-4's SCENARIO 10).
 * `__syscall` puts n/a/b in a7/a0/a1 then `ebreak` (-> glyph SYSCALL):
 * GlyphCPUv2 / SpatialRV64ICore marshal those regs into ISO_SYS_*, drop to
 * SUPER, jump KSYS_PC. `__sysret` is `mret` (-> glyph SYSRET): mode=USER,
 * result from ISO_SYS_A0 into a0, resume after the ebreak. The dispatcher
 * runs in SUPER so its g_console / caller-buffer writes are unchecked; a
 * user task's OWN direct g_console write still faults. */
extern void fault_handler(void);

static inline __attribute__((always_inline))
int __syscall(int n, const void *a, int b) {
    register int _n asm("a7") = n;
    register const void *_a asm("a0") = a;
    register int _b asm("a1") = b;
    register int _r asm("a0");
    __asm__ volatile ("ebreak" : "=r"(_r) : "r"(_n), "r"(_a), "r"(_b) : "memory");
    return _r;
}
static inline __attribute__((always_inline))
void __sysret(void) { __asm__ volatile ("mret" ::: "memory"); }

#define SYS_write 1
#define SYS_read  2

#if SCENARIO == 9 || SCENARIO == 10
/* GO-3: host-fed MMIO input ring. The harness writes ISO_INPUT_LEN + the
 * scripted bytes at ISO_INPUT_DATA before the run; syscall_dispatch services
 * SYS_read by copying up to the caller's cap bytes out of the ring into the
 * caller's buffer and advancing ISO_INPUT_CURSOR. A read that finds the
 * cursor already at ISO_INPUT_LEN returns 0 (input exhausted). Byte offsets
 * MUST match INPUT_*_ADDR in glyph_dispatch/src/glyph/glyph_isa_v2.py. */
#define ISO_INPUT_LEN     (*(volatile uint *)0x8170)
#define ISO_INPUT_CURSOR  (*(volatile uint *)0x8174)
#define ISO_INPUT_DATA    ((volatile char *)0x8180)
#endif

#if SCENARIO == 10
/* --- GO-4: framebuffer output (systems/GPU_OS_ROADMAP.md). A W_MEM-aligned
 * rectangle of image memory IS the screen: FB row y is grid row base+y, using
 * the first FB_W words of each W_MEM-wide grid row -- a genuine rectangle on
 * the GO-2 pixel grid. SYS_draw (syscall #3): __syscall(SYS_draw,
 * y*FB_W + x, val) -> the SUPER-mode dispatcher recovers x = idx % FB_W,
 * y = idx / FB_W and writes g_fb[y*W_MEM + x] = val iff x < FB_W && y < FB_H
 * (else returns (uint)-1, out-of-bounds rejected). A linear cell index (not a
 * packed (y<<16)|x) keeps the syscall arg small -- a wide shifted immediate
 * diverged on the RV64 GPU core. The
 * dispatcher runs in SUPER so the store is never box-checked; a USER task
 * cannot touch g_fb directly (E-K2). The harness reads g_fb back by ELF
 * symbol -- no MMIO "screen base" word is needed for GO-4. */
#define SYS_draw 3
#define FB_W 8
#define FB_H 6
unsigned g_fb[W_MEM * FB_H] __attribute__((aligned(128)));
#endif

__attribute__((noinline)) void syscall_dispatch(void) {
    uint n = ISO_SYS_N, b = ISO_SYS_A1;
    const char *p = (const char *)ISO_SYS_A0;
    if (n == SYS_write) {
        for (uint i = 0; i < b; i++) cons_putc(p[i]);
        ISO_SYS_A0 = b;                 /* return value */
    }
#if SCENARIO == 9 || SCENARIO == 10
    else if (n == SYS_read) {
        char *dst = (char *)ISO_SYS_A0;
        uint cap = b, cur = ISO_INPUT_CURSOR, tot = ISO_INPUT_LEN, cnt = 0;
        while (cnt + 1 < cap && cur < tot) dst[cnt++] = ISO_INPUT_DATA[cur++];
        dst[cnt] = 0;                   /* NUL-terminate for the shell tokenizer */
        ISO_INPUT_CURSOR = cur;
        ISO_SYS_A0 = cnt;               /* bytes delivered; 0 = input exhausted */
    }
#endif
#if SCENARIO == 10
    else if (n == SYS_draw) {           /* GO-4: draw one framebuffer cell */
        uint idx = ISO_SYS_A0, val = b;
        uint x = idx % FB_W, y = idx / FB_W;
        if (x < FB_W && y < FB_H) { g_fb[y * W_MEM + x] = val; ISO_SYS_A0 = 0; }
        else ISO_SYS_A0 = (uint)-1;     /* out of the screen rect -> rejected */
    }
#endif
    else {
        ISO_SYS_A0 = (uint)-1;
    }
    __sysret();                         /* never returns */
    for (;;) { }
}
extern void syscall_dispatch(void);
#endif  /* SCENARIO == 7 || 9 */

#if SCENARIO == 7
__attribute__((noinline)) void s7_task(long id) {
    char *base = g_arena + id * ARENA_SLOT;
    base[0] = 'X';                              /* in-box: fine */
    __syscall(SYS_write, "hi\n", 3);            /* kernel writes "hi\n" to g_console */
    g_console[0] = '!';                         /* direct kernel-mem write -> FAULT */
    curproc->state = DONE;                      /* never reached */
    sys_yield();
    for (;;) { }
}
__asm__(".global s7e\ns7e:\n  li a0, 0\n  j s7_task\n");
extern void s7e(void);
#endif  /* SCENARIO == 7 */

#if SCENARIO == 9 || SCENARIO == 10
/* --- GO-3/GO-4: nano-shell reading host-fed input via SYS_read ------------------
 * systems/GPU_OS_ROADMAP.md GO-3. s9_shell loops: SYS_read a chunk of the
 * host-fed input (dispatcher NUL-terminates it; returns 0 -> exhausted ->
 * exit), then run K4's first-word tokenizer over it, dispatching each line
 * through cmd9_table[] (the G5 fn-pointer jalr -- _run_glyph seeds
 * PTR_TABLE_BASE, the GPU engine needs no seeding). A USER task cannot
 * touch g_console directly (E-K2), so every command routes its output through
 * __syscall(SYS_write, ...). str_eq / state_char are shared with K4's shell. */
__attribute__((noinline)) void cmd9_help(const char *rest) {
    (void)rest;
    __syscall(SYS_write, "help ps echo\n", 13);
}
__attribute__((noinline)) void cmd9_ps(const char *rest) {
    (void)rest;
    char buf[NPROC + 1];
    int i;
    for (i = 0; i < NPROC; i++) buf[i] = state_char(proc[i].state);
    buf[NPROC] = '\n';
    __syscall(SYS_write, buf, NPROC + 1);
}
__attribute__((noinline)) void cmd9_echo(const char *rest) {
    int n = 0;
    while (rest[n] && rest[n] != '\n') n++;   /* rest is the NUL-terminated line */
    if (n) __syscall(SYS_write, rest, n);
    __syscall(SYS_write, "\n", 1);
}

/* one host-fed buffer of scripted input; a global (NUL-terminated by the
 * dispatcher), not the 128-byte kstack -- so the SUPER-mode SYS_read fill
 * never crowds the shell's frame. Same shape as K4's g_shell_input. */
static char g_s9_input[64];

#if SCENARIO == 10
/* GO-4: `plot` draws a deterministic pattern over the whole FB rect via
 * SYS_draw -- every cell gets 0x40 + ((x + y) & 0x0F). Exercises all
 * FB_W*FB_H cells and the x<FB_W && y<FB_H bounds check. */
__attribute__((noinline)) void cmd9_plot(const char *rest) {
    (void)rest;
    uint x, y;
    for (y = 0; y < FB_H; y++)
        for (x = 0; x < FB_W; x++)
            __syscall(SYS_draw, (const void *)(y * FB_W + x),
                      (int)(0x40 + ((x + y) & 0x0F)));
}
#define NCMD 4
#else
#define NCMD 3
#endif

struct cmd9 { const char *name; void (*fn)(const char *); };
struct cmd9 cmd9_table[NCMD] = {
    { "help", cmd9_help },
    { "ps",   cmd9_ps   },
    { "echo", cmd9_echo },
#if SCENARIO == 10
    { "plot", cmd9_plot },
#endif
};

__attribute__((noinline)) void s9_shell(long unused) {
    (void)unused;
    char tok[16];
    for (;;) {
        int n = __syscall(SYS_read, g_s9_input, sizeof g_s9_input);
        if (n == 0) break;                   /* input exhausted */
        const char *p = g_s9_input;          /* NUL-terminated by the dispatcher */
        while (*p) {
            int t = 0;
            while (*p && *p != ' ' && *p != '\n' && t < 15) tok[t++] = *p++;
            tok[t] = 0;
            while (*p == ' ') p++;
            const char *rest = p;            /* remainder of this line */
            while (*p && *p != '\n') p++;
            if (*p == '\n') p++;
            for (int i = 0; i < NCMD; i++) {
                if (str_eq(tok, cmd9_table[i].name)) {
                    cmd9_table[i].fn(rest);  /* jalr through fn pointer (G5) */
                    break;
                }
            }
        }
    }
    curproc->state = DONE;
    sys_yield();
    for (;;) { }
}
__asm__(".global s9e\ns9e:\n  li a0, 0\n  j s9_shell\n");
extern void s9e(void);
#endif  /* SCENARIO == 9 || 10 */

#if SCENARIO == 8
/* --- GO-2: a task confined to a 2D tile (systems/GPU_OS_ROADMAP.md). The
 * scheduler arms TILE_ROW/COL/H/W (plus BOX1/BOX2 for the proc entry + stack
 * a yielding task still needs) instead of BOX0. The task writes every cell of
 * its tile (all in-bounds -> no trap), then one cell of the row just below
 * the tile -> out of tile -> engine trap -> fault_handler reaps it. Same
 * predicate on both engines; SCENARIO 8 is proven bit-identical GPU/Glyph. */
extern void fault_handler(void);

#define S8_TROW 4
#define S8_TCOL 8
#define S8_TH   4
#define S8_TW   8
#define GRID_ROWS 16
unsigned g_grid[W_MEM * GRID_ROWS] __attribute__((aligned(128)));

__attribute__((noinline)) void s8_tile(long id) {
    int r, c;
    /* strided in-tile writes: every cell of [S8_TROW, S8_TROW+S8_TH) x
     * [S8_TCOL, S8_TCOL+S8_TW) -- all inside the armed tile, must not trap */
    for (r = 0; r < S8_TH; r++)
        for (c = 0; c < S8_TW; c++)
            g_grid[(S8_TROW + r) * W_MEM + (S8_TCOL + c)] = (unsigned)(0x41 + id);
    /* first cell of the row just below the tile -> out of tile -> trap */
    g_grid[(S8_TROW + S8_TH) * W_MEM + S8_TCOL] = 0x7F;
    curproc->state = DONE;              /* never reached */
    sys_yield();
    for (;;) { }
}
__asm__(".global s8e\ns8e:\n  li a0, 0\n  j s8_tile\n");
extern void s8e(void);
#endif  /* SCENARIO == 8 */

__attribute__((noinline)) void scheduler(void) {
    struct proc *p;
    int i, any;
    for (;;) {
        any = 0;
        for (i = 0, p = proc; i < NPROC; i++, p++) {
            if (p->state == RUNNABLE) {
                any = 1;
                p->state = RUNNING;
                curproc = p;
                g_total_switches = g_total_switches + 1;
#if SCENARIO == 6 || SCENARIO == 7 || SCENARIO == 8 || SCENARIO == 9 || SCENARIO == 10
                ISO_BOX1_LO = (uint)&proc[i];
                ISO_BOX1_HI = (uint)&proc[i] + sizeof(struct proc);
                ISO_BOX2_LO = (uint)proc[i].kstack;                 /* its kalloc'd stack page */
                ISO_BOX2_HI = (uint)proc[i].kstack + PGSIZE;
                ISO_MODE_LATCH = 1;             /* engine flips USER on switch_to's KJMP */
#endif
#if SCENARIO == 6 || SCENARIO == 7
                ISO_BOX0_LO = (uint)(g_arena + i * ARENA_SLOT);
                ISO_BOX0_HI = (uint)(g_arena + (i + 1) * ARENA_SLOT);
#endif
#if SCENARIO == 8
                /* GO-2: this task's memory IS a rectangle on the grid. g_grid
                 * is 128-byte (W_MEM-word) aligned, so its base sits at grid
                 * col 0 of some row `base_row`; the tile is offset from there. */
                ISO_TILE_ROW = (((uint)g_grid) >> 2) / W_MEM + S8_TROW;
                ISO_TILE_COL = S8_TCOL;
                ISO_TILE_H   = S8_TH;
                ISO_TILE_W   = S8_TW;
#endif
                switch_to(&ctx_sched, &p->context);
                if (p->state == RUNNING)             /* yielded -> requeue */
                    p->state = RUNNABLE;
                else if (p->state == ZOMBIE &&       /* K2: reap once */
                         !(g_reaped_mask & (1 << i))) {
                    g_xcode[i] = p->xcode;
                    g_reaped_mask = g_reaped_mask | (1 << i);
                }
            }
        }
        if (!any) break;
    }
}

/* compile-time-constant proc indices only -> fixed offsets, no multiply */
__attribute__((noinline)) void userinit(void) {
    char *s0 = (char *)kalloc();
    char *s1 = (char *)kalloc();
    char *s2 = (char *)kalloc();
    proc[0].pid = 1; proc[0].state = RUNNABLE;
    proc[1].pid = 2; proc[1].state = RUNNABLE;
    proc[2].pid = 3; proc[2].state = RUNNABLE;
    proc[0].context.sp = (long)(s0 + PGSIZE - 16);
    proc[1].context.sp = (long)(s1 + PGSIZE - 16);
    proc[2].context.sp = (long)(s2 + PGSIZE - 16);
#if SCENARIO == 6 || SCENARIO == 7 || SCENARIO == 8 || SCENARIO == 9 || SCENARIO == 10
    proc[0].kstack = (int)s0;   /* box2: the switch_to prologue spills ra/s0 here */
    proc[1].kstack = (int)s1;
    proc[2].kstack = (int)s2;
#endif
#if SCENARIO == 5
    proc[0].context.ra = (long)&shent;   /* proc[0] is the shell */
    proc[1].state = UNUSED;              /* cmd_ps reports "R--" */
    proc[2].state = UNUSED;
#elif SCENARIO == 2
    proc[0].context.ra = (long)&t0e;
    proc[1].context.ra = (long)&t1x;   /* proc[1] exits early via sys_exit */
    proc[2].context.ra = (long)&t2e;
#elif SCENARIO == 3
    proc[0].context.ra = (long)&bw_a;    /* all three fill their own box */
    proc[1].context.ra = (long)&bw_b;
    proc[2].context.ra = (long)&bw_c;
#elif SCENARIO == 4
    proc[0].context.ra = (long)&bw_a;
    proc[1].context.ra = (long)&bw_b;
    proc[2].context.ra = (long)&bw_x;   /* proc[2] writes one byte past the box */
#elif SCENARIO == 6
    proc[0].context.ra = (long)&s6a;    /* in-box: fills its arena slot, yields */
    proc[1].context.ra = (long)&s6b;
    proc[2].context.ra = (long)&s6x;    /* out of box: stores past the arena -> engine trap */
#elif SCENARIO == 7
    proc[0].context.ra = (long)&s7e;    /* syscall write (ok) then direct g_console write (faults) */
    proc[1].state = UNUSED;
    proc[2].state = UNUSED;
#elif SCENARIO == 8
    proc[0].context.ra = (long)&s8e;    /* GO-2: fills its tile in-bounds, then one out-of-tile store faults */
    proc[1].state = UNUSED;
    proc[2].state = UNUSED;
#elif SCENARIO == 9
    proc[0].context.ra = (long)&s9e;    /* GO-3: shell reads host-fed input via SYS_read, runs commands */
    proc[1].state = UNUSED;             /* cmd9_ps reports "R--" */
    proc[2].state = UNUSED;
#elif SCENARIO == 10
    proc[0].context.ra = (long)&s9e;    /* GO-4: shell reads "plot\n", cmd9_plot draws g_fb via SYS_draw */
    proc[1].state = UNUSED;
    proc[2].state = UNUSED;
#else
    proc[0].context.ra = (long)&t0e;
    proc[1].context.ra = (long)&t1e;
    proc[2].context.ra = (long)&t2e;
#endif
}

__attribute__((noinline)) void kmain(void) {
    kinit();
    userinit();
    scheduler();
}

void _start(void) {
    __asm__ volatile (
        ".option push\n"
        ".option norelax\n"
        "lui gp, %hi(__global_pointer$)\n"
        "addi gp, gp, %lo(__global_pointer$)\n"
        ".option pop\n"
        "li sp, 0x4000\n"
        "call kmain\n"
        "ecall\n"
    );
}
