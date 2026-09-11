/* G6: sub-word (byte) memory access -- checksum_str + copy_str, the actual
 * payload the coverage table's gap blocks (any char*-shaped C: strlen,
 * strcmp, path walking). Forces byte LOADS (checksum_str reading through a
 * pointer) and byte STORES (copy_str writing through a pointer, plus direct
 * literal stores building g_src) across a word boundary: g_src/g_dst are 8
 * bytes = 2 words, and the strings used span indices 0-4, landing bytes in
 * BOTH words and at all four lane offsets (0,1,2,3 mod 4).
 *
 * unsigned char throughout, deliberately: only LBU is implemented (not
 * signed LB), so this avoids sign-extension entirely rather than risk
 * masking a real gap.
 */

long g_checksum;
long g_len;

unsigned char g_src[8];
unsigned char g_dst[8];

__attribute__((noinline)) unsigned int checksum_str(const unsigned char *s) {
    unsigned int sum = 0;
    unsigned int i = 0;
    while (s[i] != 0) {
        sum = sum + s[i];
        i = i + 1;
    }
    g_len = i;
    return sum;
}

__attribute__((noinline)) void copy_str(unsigned char *dst, const unsigned char *src) {
    unsigned int i = 0;
    while (src[i] != 0) {
        dst[i] = src[i];
        i = i + 1;
    }
    dst[i] = 0;
}

__attribute__((noinline)) void run_all(void) {
    g_src[0] = 'a';
    g_src[1] = 'b';
    g_src[2] = 'c';
    g_src[3] = 'd';
    g_src[4] = 'e';
    g_src[5] = 0;

    g_checksum = checksum_str(g_src);   /* 'a'+'b'+'c'+'d'+'e' = 495 */
    copy_str(g_dst, g_src);
}

void _start(void) {
    __asm__ volatile (
        "li sp, 0x1000\n"
        "call run_all\n"
        "ecall\n"
    );
}
