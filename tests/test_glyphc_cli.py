"""tests/test_glyphc_cli.py — Unit and integration tests for glyphc CLI tool."""

import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest
from tools.glyphc import main


def test_glyphc_build_and_run_hello(capsys):
    hello_src = _REPO / 'examples' / '01_hello.glyph'
    assert hello_src.exists()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = Path(tmpdir) / 'hello.glyph.png'
        # Build
        ret = main(['build', str(hello_src), '-o', str(out_png)])
        assert ret == 0
        assert out_png.exists()

        # Run
        ret = main(['run', str(out_png)])
        assert ret == 0
        captured = capsys.readouterr()
        assert 'OUTPUT: 42' in captured.out
        assert '[HALTED]' in captured.out


def test_glyphc_counter(capsys):
    counter_src = _REPO / 'examples' / '02_counter.glyph'

    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = Path(tmpdir) / 'counter.glyph.png'
        assert main(['build', str(counter_src), '-o', str(out_png)]) == 0
        assert main(['run', str(out_png)]) == 0
        captured = capsys.readouterr()
        assert 'OUTPUT: 0' in captured.out
        assert 'r0=1' in captured.out


def test_glyphc_fibonacci(capsys):
    fib_src = _REPO / 'examples' / '03_fibonacci.glyph'

    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = Path(tmpdir) / 'fib.glyph.png'
        assert main(['build', str(fib_src), '-o', str(out_png)]) == 0
        assert main(['run', str(out_png)]) == 0
        captured = capsys.readouterr()
        assert 'OUTPUT: 13' in captured.out


def test_glyphc_disasm_and_verify(capsys):
    hello_src = _REPO / 'examples' / '01_hello.glyph'

    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = Path(tmpdir) / 'hello.glyph.png'
        assert main(['build', str(hello_src), '-o', str(out_png)]) == 0

        # Verify
        assert main(['verify', str(hello_src)]) == 0
        assert main(['verify', str(out_png)]) == 0

        # Disasm
        assert main(['disasm', str(out_png)]) == 0
        captured = capsys.readouterr()
        assert 'LDI r10 42' in captured.out
        assert 'PRT r10' in captured.out
        assert 'HALT' in captured.out
