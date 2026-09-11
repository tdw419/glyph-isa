#!/usr/bin/env python3
"""runner.py — standalone spatial image launcher (GH-5/GH-11; GH-26.1
adds publish_dir). Pure launcher, zero dev imports: loads a spatial
image (.png/.npy/.npz), executes on GlyphCPUv2 or WGSL."""
from __future__ import annotations
import contextlib, hashlib, io, json, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from glyph_isa_v2 import GlyphCPUv2, OpcodeMapV2  # noqa: E402


class GlyphRunner:
    """Minimal standalone launcher for baked Geometry OS spatial images."""

    def __init__(self, image_or_path: Union[str, Path, np.ndarray],
                 cols_instrs: Optional[int] = None,
                 ram_words: Optional[int] = None) -> None:
        if isinstance(image_or_path, np.ndarray):
            self.image = image_or_path.copy()
        else:
            p = Path(image_or_path)
            if p.suffix in (".npy", ".npz"):
                d = np.load(p)
                self.image = d if p.suffix == ".npy" else \
                    (d["image"] if "image" in d else d[list(d.keys())[0]])
            else:
                self.image = np.array(Image.open(p).convert("RGB"))
        # Every instruction is 4 horizontal pixels wide
        self.cols_instrs = cols_instrs or (self.image.shape[1] // 4)
        # ram_words opt-in: arms iso MMIO (needs len > TILE_W>>2); None = 1024 default (GH-1 parity)
        self.ram_words = ram_words

    @classmethod
    def launcher_info(cls) -> Dict[str, Any]:
        n = len(Path(__file__).resolve().read_text().splitlines())
        return {"version": "gh11", "backends": ["cpu", "wgsl"], "max_lines": 200, "lines": n}

    def get_cpu(self) -> GlyphCPUv2:
        """Fresh GlyphCPUv2 (ram_words opt-in arms _iso_enabled)."""
        cpu = GlyphCPUv2(OpcodeMapV2(), cols_instrs=self.cols_instrs, fs_pix_enabled=True)
        if self.ram_words is not None:
            cpu.memory = [0] * self.ram_words
        return cpu

    def md5(self) -> str:
        """MD5 of the resident image container bytes."""
        return hashlib.md5(self.image.tobytes()).hexdigest()

    def run(self, max_instructions: int = 5000, trace: bool = False) -> Dict[str, Any]:
        """Execute image to HALT/fault with receipt. trace=True records
        (instr index, mode) per step in receipt["step_trace"] (GH-1 bit-exact)."""
        receipt: Dict[str, Any] = {"assembled": True, "executed": False, "halted": False, "faulted": False}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                cpu = self.get_cpu()
                if trace:
                    steps = 0
                    trace_list: List[Tuple[int, str]] = []
                    cpu.running = True   # run() sets this; the step loop mirrors it
                    while cpu.running and steps < max_instructions:
                        trace_list.append((cpu.pc[1] * cpu.cols_instrs + cpu.pc[0] // 4,
                                           "USER" if cpu.mode else "SUPER"))
                        cpu.step(self.image); steps += 1  # LIVE image (GH-8b)
                    receipt["step_trace"] = trace_list
                else:
                    steps = cpu.run(self.image, max_instructions=max_instructions)
            self._fill_receipt(receipt, cpu, steps)
        except Exception as e:
            receipt["error"] = f"{type(e).__name__}: {e}"
            if trace:
                import traceback as _tb; receipt["traceback"] = _tb.format_exc()
        return receipt

    def drive(self, seeds: Optional[Dict[int, int]] = None, max_instructions: int = 60000,
              publish_dir: Optional[Path] = None) -> Dict[str, Any]:
        """GH-11 session: host seeds exec-state, kernel runs to HALT/fault
        on the LIVE image. GH-26.1: publish_dir dumps npy+meta per tick."""
        receipt: Dict[str, Any] = {"assembled": True, "executed": False, "halted": False, "faulted": False}
        pub = Path(publish_dir) if publish_dir is not None else None
        if pub is not None:
            pub.mkdir(parents=True, exist_ok=True)  # noqa
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                cpu = self.get_cpu()
                for w, v in (seeds or {}).items():
                    cpu.memory[int(w)] = int(v) & 0xFFFFFFFF
                if pub is None:
                    steps = cpu.run(self.image, max_instructions=max_instructions)
                else:  # publish path: step manually, dump per tick boundary
                    md5, steps = self.md5(), 0
                    cpu.running = True  # run() would set this; we step manually
                    last_tick = cpu.memory[8210]  # TICK_PC (GH-16)
                    while cpu.running and steps < max_instructions:
                        cpu.step(self.image); steps += 1
                        if cpu.mode == 0 and cpu.memory[8210] != last_tick:
                            last_tick = cpu.memory[8210]
                            self._publish(pub, cpu, md5, steps, tag=f"_tick{steps:06d}")
                    self._publish(pub, cpu, md5, steps)  # canonical final
            self._fill_receipt(receipt, cpu, steps)
        except Exception as e:
            receipt["error"] = f"{type(e).__name__}: {e}"
        return receipt

    def _publish(self, pub: Path, cpu: GlyphCPUv2, md5: str, steps: int, tag: str = "") -> None:
        """GH-26.1: one committed-state snapshot: npy + sidecar meta."""
        mem = np.array([int(m) & 0xFFFFFFFF for m in cpu.memory], dtype=np.uint32)
        np.save(pub / f"kernel_memory{tag}.npy", mem)
        meta = {"tick": int(cpu.memory[732]), "step": int(steps), "source_md5": md5,
                "faulted": bool(getattr(cpu, "faulted", False)), "canonical": not tag}
        (pub / ("surface.meta.json" if not tag else f"kernel_memory{tag}.meta.json")).write_text(json.dumps(meta))

    def _fill_receipt(self, receipt: Dict[str, Any], cpu: GlyphCPUv2, steps: int) -> None:
        receipt["executed"] = True
        receipt["faulted"] = bool(getattr(cpu, "faulted", False))
        if receipt["faulted"]:
            receipt["fault_addr"] = int(cpu.fault_addr) & 0xFFFFFFFF
        receipt["halted"] = not cpu.running
        receipt["steps"] = int(steps)
        receipt["registers_full"] = [int(r) & 0xFFFFFFFF for r in cpu.registers]
        receipt["registers"], receipt["output"] = receipt["registers_full"][:8], list(getattr(cpu, "output", []))
        receipt["memory"] = [int(m) & 0xFFFFFFFF for m in cpu.memory]
        receipt["status_word_value"] = receipt["memory"][950]  # GH-8b write-through mirror

    def run_wgsl(self, max_steps: int = 5000) -> Dict[str, Any]:
        import wgpu, wgpu.utils
        from tools.wgsl_glyph_isa_v2 import build_shader, make_cpu_state_array
        receipt: Dict[str, Any] = {"assembled": True, "executed": False, "halted": False, "faulted": False}
        try:
            device = wgpu.utils.get_default_device()
            queue = device.queue
            n_pixels = self.image.shape[0] * self.image.shape[1]
            rgba = np.zeros((n_pixels, 4), dtype=np.uint32)
            rgba[:, 0:3] = self.image.reshape(n_pixels, 3)
            usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
            cpu_state, cpu_dtype = make_cpu_state_array(1)
            mmio = np.zeros(64, dtype=np.uint32)  # GH-25 mirror
            dt = np.dtype([('image_width', np.uint32), ('image_height', np.uint32), ('output_buffer_size', np.uint32)])
            bufs = {}
            for name, arr, extra in (
                    ("img", rgba, 0), ("cpu", cpu_state, 0), ("out", np.zeros(256, np.uint32), 0),
                    ("mmio", mmio, 0), ("u", np.array([(self.image.shape[1], self.image.shape[0], 64)], dt),
                                        wgpu.BufferUsage.UNIFORM)):
                b = device.create_buffer(size=arr.nbytes, usage=usage | extra)
                queue.write_buffer(b, 0, arr.tobytes())
                bufs[name] = b
            bgl = device.create_bind_group_layout(entries=[
                {'binding': i, 'visibility': wgpu.ShaderStage.COMPUTE,
                 'buffer': {'type': 'storage' if i != 3 else 'uniform'}} for i in range(5)])
            bg = device.create_bind_group(layout=bgl, entries=[
                {'binding': 0, 'resource': {'buffer': bufs["img"], 'offset': 0, 'size': rgba.nbytes}},
                {'binding': 1, 'resource': {'buffer': bufs["cpu"], 'offset': 0, 'size': cpu_state.nbytes}},
                {'binding': 2, 'resource': {'buffer': bufs["out"], 'offset': 0, 'size': 256}},
                {'binding': 3, 'resource': {'buffer': bufs["u"], 'offset': 0, 'size': bufs["u"].size}},
                {'binding': 4, 'resource': {'buffer': bufs["mmio"], 'offset': 0, 'size': mmio.nbytes}}])
            sh = device.create_shader_module(code=build_shader(OpcodeMapV2()))
            pipe = device.create_compute_pipeline(
                layout=device.create_pipeline_layout(bind_group_layouts=[bgl]),
                compute={'module': sh, 'entry_point': 'main'})
            rb = None
            for steps in range(1, max_steps + 1):
                enc = device.create_command_encoder()
                p = enc.begin_compute_pass()
                p.set_pipeline(pipe); p.set_bind_group(0, bg)
                p.dispatch_workgroups(1); p.end()
                queue.submit([enc.finish()])
                rb = np.frombuffer(queue.read_buffer(bufs["cpu"]), dtype=cpu_dtype)[0]
                if rb['running'] == 0:
                    break
            receipt["executed"] = True
            receipt["halted"] = bool(rb['running'] == 0)
            receipt["steps"] = steps
            receipt["registers_full"] = [int(r) for r in rb['registers']]
            receipt["registers"] = receipt["registers_full"][:8]
        except Exception as e:
            receipt["error"] = f"{type(e).__name__}: {e}"
        return receipt

    def __call__(self, max_instructions: int = 5000, backend: str = "cpu") -> Dict[str, Any]:
        if backend == "wgsl":
            return self.run_wgsl(max_steps=max_instructions)
        return self.run(max_instructions=max_instructions)

def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 runner.py <img> [--backend=cpu|wgsl]")
    backend = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--backend=")), "cpu")
    r = GlyphRunner(sys.argv[1])(backend=backend)
    print(f"Executed: halted={r.get('halted')} steps={r.get('steps')}")


if __name__ == "__main__": main()