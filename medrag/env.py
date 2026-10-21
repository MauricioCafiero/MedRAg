"""Environment guards, imported first by medrag/__init__.py (same role as cafquantum's cafquant/env.py).

Must run before anything imports e3nn (fairchem and mace both do): e3nn 0.4.4, hard-pinned by mace-torch 0.3.16,
loads o3/constants.pt with a plain torch.load, which under torch>=2.6's weights_only default raises
WeightsUnpicklerError unless "slice" is allow-listed.
Unlike cafquant/env.py this does not force OMP/MKL threads to 1: that is a macOS libomp/tblite workaround, and on this
Linux box single-threaded MACE is about 2x slower (measured: 2.47 s vs 1.25 s per call on a 60-atom ligand).
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# OpenMM's CPU platform sums forces in a thread-dependent order, so PDBFixer's hydrogen minimisation lands on slightly
# different coordinates on every run (94 hydrogens, up to 0.08 A) even with the random start seeded (system.H_PLACEMENT_SEED).
# One thread makes the carved site bit-for-bit reproducible; the step takes ~2 s, so nothing is lost.
os.environ.setdefault("OPENMM_CPU_THREADS", "1")


def patch_torch_for_e3nn() -> None:
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([slice])
    except Exception:
        pass


patch_torch_for_e3nn()
