"""Call functions in the vendored repos (vendor/<repo>/...) through the isolated runner."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor"


class VendorError(RuntimeError):
    pass


def clean_env() -> dict:
    """Environment for subprocesses: drop ORCA (and other ~/.local app) lib dirs from LD_LIBRARY_PATH. Their bundled
    libstdc++ is older than the one system obabel needs ("CXXABI_1.3.15 not found")."""
    env = dict(os.environ)
    # the venv's own bin first, so the pip `obabel` (openbabel-wheel) is used even when the venv is not activated, and a
    # system-wide Open Babel is never required
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    ld = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p and "/.local/orca" not in p.lower()]
    if ld:
        env["LD_LIBRARY_PATH"] = ":".join(ld)
    else:
        env.pop("LD_LIBRARY_PATH", None)
    return env


def call(repo_code_dir: str, module: str, function: str, cwd: str | Path, timeout: int = 3600, **kwargs):
    """Run vendor/<repo_code_dir>/<module>.<function>(**kwargs) with cwd=<cwd>; return its (JSON-able) result."""
    code_dir = VENDOR / repo_code_dir
    if not code_dir.is_dir():
        raise VendorError(f"missing vendored code dir {code_dir}")
    Path(cwd).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        res = Path(td) / "result.json"
        proc = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).with_name("_runner.py")), str(code_dir), module, function,
             str(res), json.dumps(kwargs)],
            cwd=str(cwd), capture_output=True, text=True, timeout=timeout, env=clean_env())
        if not res.is_file():
            raise VendorError(f"{module}.{function} produced no result (exit {proc.returncode}):\n{proc.stderr[-1500:]}")
        payload = json.loads(res.read_text())
    if not payload["ok"]:
        raise VendorError(f"{module}.{function} failed: {payload['error']}\n{payload['trace'][-1200:]}")
    return payload["result"]
