/*
 * bump_alloc.c — Minimal nostdlib C bump allocator fixture.
 *
 * Designed for differential execution testing between GPU SpatialRV64ICore
 * and GlyphCPUv2 (via rv64i_to_glyph.py transpilation).
 */

#define ARENA_START 0x100

static unsigned int arena_offset = ARENA_START;

__attribute__((noinline)) unsigned int bump_alloc(unsigned int size) {
    /* Align size to 4-byte boundary */
    size = (size + 3) & ~3;
    unsigned int ptr = arena_offset;
    arena_offset += size;
    return ptr;
}

__attribute__((noinline)) void bump_reset(void) {
    arena_offset = ARENA_START;
}

__attribute__((noinline)) int main_test(void) {
    bump_reset();
    unsigned int p1 = bump_alloc(8);   /* 8 bytes  -> returns 0x100, next 0x108 */
    unsigned int p2 = bump_alloc(13);  /* 13 bytes (aligned to 16) -> returns 0x108, next 0x118 */
    unsigned int p3 = bump_alloc(4);   /* 4 bytes  -> returns 0x118, next 0x11C */

    /* Write distinctive canary values to allocated memory */
    *(volatile unsigned int*)p1 = 0xAA11BB22;
    *(volatile unsigned int*)(p1 + 4) = 0x33445566;
    *(volatile unsigned int*)p2 = 0xDEADBEEF;
    *(volatile unsigned int*)p3 = 0xCAFEBABE;

    /* Read back values */
    unsigned int v1 = *(volatile unsigned int*)p1;
    unsigned int v2 = *(volatile unsigned int*)p2;
    unsigned int v3 = *(volatile unsigned int*)p3;

    /* Return XOR checksum in a0 */
    return v1 ^ v2 ^ v3;
}

void _start(void) {
    __asm__ volatile (
        "li sp, 0x800\n"
        "call main_test\n"
        "ecall\n"
    );
}
