.section .text
.global _start
_start:
    li a0, 0x10000000
    li a1, 72
    sb a1, 0(a0)
    li a1, 105
    sb a1, 0(a0)
    li a1, 10
    sb a1, 0(a0)
    addi t0, x0, 5
    addi t1, x0, 3
    add t2, t0, t1
1:
    j 1b
