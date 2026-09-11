# Glyph ISA (`glyph-isa`)

**The Spatial-Atomic Programming Language Substrate.**

Glyph (`.glyph`) is a programming language where **the pixels are the program and the pixels are the memory**.

- **Visual Containers**: Executables ship as `.glyph.png` raster images. Loading a program is an image decode directly into GPU memory, enabling <1ms cold boots.
- **Proof-Carrying Code (PCC)**: Code cannot execute without passing machine-checked admission invariants.
- **Unified 2D Geometry**: Memory addresses, instructions, and agent boxes map onto a 2D Hilbert-space coordinate curve ($N=128$).
- **Multi-Tenant Agent Arenas**: Autonomous AI agents execute in hardware-bounded spatial memory boxes (`BOX0..BOXn`) with non-blocking mailbox IPC.

## Specification
The official language specification is [GLS-1.0](docs/spec/GLYPH_ISA_SPEC_v1.0.md).

## Quick Start
```bash
# Run tests
pytest tests/
```

## License
Dual-licensed under either:
- **MIT License** ([LICENSE-MIT](LICENSE-MIT))
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE))
at your option.
