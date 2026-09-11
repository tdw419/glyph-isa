/* G4: xv6 fs.c-shaped in-memory inode table. New pattern vs. fifo/slab:
 * NESTED indexing -- itable[inum].addrs[slot] combines a struct-array index
 * (inum, a runtime value: a function parameter, not a sequential loop
 * induction variable a compiler could strength-reduce) with an inner array
 * field access (addrs[slot]). struct inode is padded to 32 bytes (power of
 * 2) so indexing by a genuinely-random-access inum can't need __mulsi3
 * (-nostdlib has none) -- unlike round_robin's ctx_task[], GCC can't turn
 * "index by an arbitrary function argument" into a strength-reduced running
 * pointer the way it can a sequential loop counter.
 */

struct inode {
    unsigned int type;   /* 0 = free */
    unsigned int inum;
    unsigned int size;
    unsigned int addrs[4];
    unsigned int _pad;    /* 4+4+4+16+4 = 32 bytes */
};

#define NINODE 4

struct inode itable[NINODE];

long g_ialloc_log[4];   /* which inum each ialloc() call returned */
long g_ialloc_idx;

__attribute__((noinline)) unsigned int ialloc(unsigned int type) {
    unsigned int i;
    for (i = 0; i < NINODE; i = i + 1) {
        if (itable[i].type == 0) {
            itable[i].type = type;
            itable[i].inum = i;
            itable[i].size = 0;
            return i;
        }
    }
    return 0xFFFFFFFF;
}

__attribute__((noinline)) void ifree(unsigned int inum) {
    itable[inum].type = 0;
}

__attribute__((noinline)) void iwrite_block(unsigned int inum, unsigned int slot,
                                             unsigned int blockaddr) {
    itable[inum].addrs[slot] = blockaddr;
    itable[inum].size = itable[inum].size + 1;
}

__attribute__((noinline)) unsigned int iread_block(unsigned int inum, unsigned int slot) {
    return itable[inum].addrs[slot];
}

long g_r0, g_r1, g_r2, g_r3;   /* readback results */

__attribute__((noinline)) void run_all(void) {
    unsigned int a, b, c, d;

    a = ialloc(1);              /* -> inum 0 */
    b = ialloc(1);              /* -> inum 1 */
    c = ialloc(2);               /* -> inum 2 */
    g_ialloc_log[g_ialloc_idx] = a; g_ialloc_idx = g_ialloc_idx + 1;
    g_ialloc_log[g_ialloc_idx] = b; g_ialloc_idx = g_ialloc_idx + 1;
    g_ialloc_log[g_ialloc_idx] = c; g_ialloc_idx = g_ialloc_idx + 1;

    iwrite_block(a, 0, 0x1000);
    iwrite_block(a, 1, 0x1004);
    iwrite_block(b, 0, 0x2000);
    iwrite_block(c, 3, 0x3000);   /* highest slot, on the LAST allocated inode */

    ifree(b);                     /* free inum 1 */
    d = ialloc(3);                 /* should reuse inum 1 (first free slot) */
    g_ialloc_log[g_ialloc_idx] = d; g_ialloc_idx = g_ialloc_idx + 1;
    iwrite_block(d, 2, 0x4000);

    g_r0 = iread_block(a, 0);      /* 0x1000 */
    g_r1 = iread_block(a, 1);      /* 0x1004 */
    g_r2 = iread_block(c, 3);      /* 0x3000, unaffected by b's reuse */
    g_r3 = iread_block(d, 2);      /* 0x4000, reused slot d==b's old inum */
}

void _start(void) {
    __asm__ volatile (
        "li sp, 0x1000\n"
        "call run_all\n"
        "ecall\n"
    );
}
