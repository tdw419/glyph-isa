import pytest
import numpy as np
from tools.glyph_isa_v2 import OpcodeMapV2, SpatialMisalignmentFault, GlyphCPUv2, GlyphAssemblerV2

def test_opcode_map_collision_avoidance():
    # Create an OpcodeMapV2 and mock its _color_for_word to force collisions
    op_map = OpcodeMapV2()
    # Force 'HALT' to return a reserved color
    op_map._opcode_to_rgb.clear()
    op_map._rgb_to_opcode.clear()
    
    # Mocking
    original_color_for_word = op_map._color_for_word
    def mock_color_for_word(word, opcode):
        if opcode == 'HALT':
            return (0, 0, 0) # Reserved
        if opcode == 'LDI':
            return (0, 0, 1) # Reserved
        if opcode == 'ADD':
            return (5, 5, 5) # Valid
        if opcode == 'SUB':
            return (5, 5, 5) # Collision with ADD!
        return original_color_for_word(word, opcode)
    
    op_map._color_for_word = mock_color_for_word
    op_map._build_maps()
    
    # Assert HALT and LDI got shifted out of reserved space
    assert op_map.opcode_to_rgb('HALT') != (0, 0, 0)
    assert not op_map._is_reserved(op_map.opcode_to_rgb('HALT'))
    
    assert op_map.opcode_to_rgb('LDI') != (0, 0, 1)
    assert not op_map._is_reserved(op_map.opcode_to_rgb('LDI'))
    
    # Assert ADD is (5, 5, 5) and SUB got shifted
    assert op_map.opcode_to_rgb('ADD') == (5, 5, 5)
    assert op_map.opcode_to_rgb('SUB') != (5, 5, 5)
    
    # Assert no two opcodes share the same color
    colors = set(op_map._opcode_to_rgb.values())
    assert len(colors) == len(op_map.OPCODES)
    
    op_map.close()

def test_spatial_misalignment_fault():
    op_map = OpcodeMapV2()
    cpu = GlyphCPUv2(op_map, cols_instrs=8)
    image = np.zeros((1, 8, 3), dtype=np.uint8)
    
    # Valid alignment
    cpu.pc = (4, 0)
    # won't raise fault, but might halt on empty image
    cpu.step(image)
    
    # Invalid alignment
    cpu.pc = (3, 0)
    with pytest.raises(SpatialMisalignmentFault):
        cpu.step(image)
        
    op_map.close()

def test_turing_complete_features():
    op_map = OpcodeMapV2()
    assembler = GlyphAssemblerV2(op_map)
    cpu = GlyphCPUv2(op_map, cols_instrs=8)
    
    # Assembly width_instrs = 8.
    # Col 0..7 is y=0. Col 0..7 is y=1.
    # We will write a program that pushes to the stack, calls a subroutine,
    # does bitwise operations, returns, and halts.
    program = [
        "LDI r1 5",       # 0: (0,0)  r1 = 5
        "LDI r2 3",       # 1: (4,0)  r2 = 3
        "AND r1 r2",      # 2: (8,0)  r1 = 5 & 3 = 1
        "LDI r2 2",       # 3: (12,0) r2 = 2
        "SHL r1 r2",      # 4: (16,0) r1 = 1 << 2 = 4
        "PUSH r1",        # 5: (20,0) stack pushes 4
        "CALL 0,1",       # 6: (24,0) call subroutine at (0,1). Pushes (28,0) as return address.
        "POP r4",         # 7: (28,0) pop from stack into r4. Should be 8 (from subroutine).
        "HALT",           # 8: (32,0)->Wait, 8 is actually (0,1). So the CALL 0,1 jumps to HALT!
    ]
    # Let's adjust coordinates carefully!
    # width_instrs = 8
    # idx 0: x=0, y=0
    # idx 1: x=4, y=0
    # idx 2: x=8, y=0
    # idx 3: x=12, y=0
    # idx 4: x=16, y=0
    # idx 5: x=20, y=0
    # idx 6: x=24, y=0 (CALL)
    # idx 7: x=28, y=0 (POP r4)
    # idx 8: x=0, y=1 (HALT)  <-- End of main program
    # idx 9: x=4, y=1 (Subroutine starts here)
    program = [
        "LDI r1 5",       # 0: r1=5
        "LDI r2 3",       # 1: r2=3
        "AND r1 r2",      # 2: r1=1
        "LDI r2 2",       # 3: r2=2
        "SHL r1 r2",      # 4: r1=4
        "PUSH r1",        # 5: Stack=[4]
        "CALL 1,1",       # 6: Push return PC (x=28, y=0). Jump to x=4, y=1 (idx 9)
        "POP r4",         # 7: Pop modified value from stack into r4
        "HALT",           # 8: Main program halts here
        # Subroutine at (1,1) - idx 9
        "POP r6",         # 9: Pop return address into r6
        "POP r5",         # 10: Pop 4 into r5
        "LDI r7 1",       # 11: r7=1
        "SHL r5 r7",      # 12: r5 = 4 << 1 = 8
        "PUSH r5",        # 13: Push 8 onto stack
        "PUSH r6",        # 14: Push return address back
        "RET",            # 15: Jump back to (x=28, y=0) which is idx 7
    ] + ["HALT"] * 16     # padding for stack space
    
    image = assembler.assemble(program, width_instrs=8)
    # Ensure SP starts at 0, wrapping downwards.
    cpu.registers[31] = 0
    cpu.run(image)
    
    assert cpu.registers[1] == 4
    assert cpu.registers[4] == 8
    
    op_map.close()
