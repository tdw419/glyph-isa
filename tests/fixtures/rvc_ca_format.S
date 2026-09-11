.section .text
.global _start
_start:
    li s0, 0x100
    li s1, 0x1
    or s0, s0, s1
    li a0, 0x40
    li a1, 0x11
    xor a0, a0, a1
    li a2, 0xff
    li a3, 0x0f
    and a2, a2, a3
    li a4, 10
    li a5, 3
    sub a4, a4, a5
1:
    j 1b
