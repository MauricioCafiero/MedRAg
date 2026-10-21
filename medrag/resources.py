"""Memory / weight preflight: know what will fit before loading a model.

The MLIP is the only large local model (the LLM is remote), so the budget is
checked against the checkpoint that is actually on disk.
"""
from __future__ import annotations

import os
from pathlib import Path

import psutil

# Rough resident-RAM multiplier over the on-disk checkpoint size once loaded
# and running a few-hundred-atom system on CPU. An estimate, not a measurement;
# calibrate_checks() reports the measured RSS after its first real call.
RUNTIME_FACTOR = 1.6
HEADROOM_GB = 2.0          # always leave this much free for the OS, docking and the CLI

FAIRCHEM_CACHE = Path(os.environ.get("FAIRCHEM_CACHE_DIR", Path.home() / ".cache" / "fairchem"))
MACE_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mace"

# method -> (glob under cache, label)
MLIP_WEIGHTS = {
    "uma": (FAIRCHEM_CACHE, "models--facebook--UMA/snapshots/*/checkpoints/uma-s-1p*.pt"),
    "maceoff": (MACE_CACHE, "MACE-OFF23_*.model"),
    "polar1-s": (MACE_CACHE, "MACEPOLAR1Smodel*"),      # names as in cafquantum weights.py
    "polar1-m": (MACE_CACHE, "MACEPOLAR1Mmodel*"),
}


def find_weights(method: str) -> Path | None:
    root, pattern = MLIP_WEIGHTS[method]
    hits = sorted(root.glob(pattern))
    return hits[-1].resolve() if hits else None


def mem_gb() -> tuple[float, float]:
    vm = psutil.virtual_memory()
    return vm.available / 2**30, vm.total / 2**30


def est_runtime_gb(method: str) -> float | None:
    w = find_weights(method)
    return None if w is None else w.stat().st_size / 2**30 * RUNTIME_FACTOR


def top_memory_users(n: int = 3) -> list[tuple[str, float]]:
    rows = []
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            rows.append((p.info["name"], p.info["memory_info"].rss / 2**30))
        except (psutil.Error, TypeError):
            continue
    agg: dict[str, float] = {}
    for name, gb in rows:
        agg[name] = agg.get(name, 0.0) + gb
    return sorted(agg.items(), key=lambda kv: -kv[1])[:n]


def fits(method: str) -> tuple[bool, str]:
    """(ok, reason): will this MLIP fit in currently available RAM with headroom?"""
    need = est_runtime_gb(method)
    if need is None:
        return False, f"no {method} weights found under {MLIP_WEIGHTS[method][0]}"
    avail, _ = mem_gb()
    ok = avail - need >= HEADROOM_GB
    return ok, (f"{method}: ~{need:.1f} GB needed (+{HEADROOM_GB:.0f} GB headroom), "
                f"{avail:.1f} GB available -> {'fits' if ok else 'DOES NOT FIT'}")


def report() -> str:
    avail, total = mem_gb()
    lines = [f"RAM: {avail:.1f} GB available of {total:.1f} GB; CPUs: {os.cpu_count()}; GPU: none assumed (CPU torch)",
             "Top memory users: " + ", ".join(f"{n} {g:.1f} GB" for n, g in top_memory_users())]
    for m in MLIP_WEIGHTS:
        w = find_weights(m)
        lines.append(f"{m}: weights {'OK  ' + str(w) if w else 'MISSING'}")
        if w:
            lines.append("  " + fits(m)[1])
    free_disk = psutil.disk_usage(str(Path.home())).free / 2**30
    lines.append(f"Disk free: {free_disk:.1f} GB")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
