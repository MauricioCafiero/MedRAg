"""medrag: local, self-checking hit-to-lead agent."""
from medrag import env  # noqa: F401  (first: e3nn/torch.load guard, see env.py)

import sys as _sys

if __import__("os").environ.get("MEDRAG_TRACE_FILE"):           # development aid: list the vendored files a run really uses
    import atexit as _atexit
    _opened = set()
    _sys.addaudithook(lambda ev, args: _opened.add(str(args[0])) if ev == "open" and args and isinstance(args[0], (str, bytes)) and "/medrag/lib/" in str(args[0]) else None)

    def _dump():
        files = {m.__file__ for m in list(_sys.modules.values()) if getattr(m, "__file__", None) and "/medrag/lib/" in str(m.__file__)}
        with open(__import__("os").environ["MEDRAG_TRACE_FILE"], "a") as fh:
            for f in sorted(files | _opened):
                fh.write(f + "\n")
    _atexit.register(_dump)
