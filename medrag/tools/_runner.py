"""Run one function from a vendored repo in an isolated interpreter.

Vendored modules share names (modrag_protein_functions, vina_dock, ...) and some resolve paths relative to the
working directory, so each call gets its own process, sys.path entry and cwd. Result is written as JSON to a file
(stdout is left for the tool's own prints).
usage: _runner.py <code_dir> <module> <function> <result.json> <kwargs-json>   (cwd set by the caller)
"""
import importlib
import json
import sys

code_dir, module, function, result_path, kwargs = sys.argv[1:6]
sys.path.insert(0, code_dir)
try:
    mod = importlib.import_module(module)
    if hasattr(mod, "find_vina_bin"):
        # the vendored lookup tries the macOS binary first; it exists in the wheel but cannot run on Linux
        import os
        import dockstring

        def _linux_vina(explicit=None):
            path = explicit or os.path.join(os.path.dirname(dockstring.__file__), "resources", "bin", "vina_linux")
            if not os.path.exists(path):
                raise mod.DockError(f"vina binary not found: {path}")
            return path
        mod.find_vina_bin = _linux_vina
    out = getattr(mod, function)(**json.loads(kwargs))
    payload = {"ok": True, "result": out}
except Exception as exc:  # reported to the caller, not swallowed
    import traceback
    payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
with open(result_path, "w") as fh:
    json.dump(payload, fh, default=str)
