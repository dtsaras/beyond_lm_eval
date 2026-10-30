"""Offloaded hidden states behave like in-memory ones under in-place ops.

A read-only memmap (mode="r") gives a tensor whose in-place update segfaults
the process; copy-on-write (mode="c") makes the update private and leaves the
offload file unchanged. Run in a subprocess so a crash is a failure, not a
dead test runner.
"""
import subprocess
import sys
import textwrap


def test_in_place_update_of_an_offloaded_tensor_is_private(tmp_path):
    code = textwrap.dedent(f"""
        import numpy as np, torch
        from blme.tasks.geometry.utils import _offload_bytes, _offload_view
        for t in (torch.arange(6, dtype=torch.float32).reshape(2, 3),
                  torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)):
            raw, tag = _offload_bytes(t)
            path = r"{tmp_path}" + f"/x_{{tag}}.bin"
            open(path, "wb").write(raw)
            v = _offload_view(path, tag, tuple(t.shape))
            assert torch.equal(v, t) and v.dtype == t.dtype
            v.mul_(2)
            assert torch.equal(v, 2 * t)
            assert open(path, "rb").read() == raw, "the offload file changed"
        print("ok")
    """)
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", (r.returncode, r.stderr[-2000:])
