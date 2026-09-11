.section .text
.global _start
_start:
    addi sp, sp, -32
    li a1, 0
    sw a1, 0(sp)
    lw a2, 0(sp)
    beqz a1, skip
    li a3, 111
skip:
    li a4, 222
    li s0, 15
    andi s0, s0, 9
    srli s0, s0, 1
1:
    j 1b
