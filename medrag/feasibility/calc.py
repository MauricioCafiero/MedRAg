"""ASE calculator factory with a per-process cache (loading UMA is the expensive part)."""
from __future__ import annotations

import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_CACHE: dict[str, object] = {}
LAST_LOAD_S: dict[str, float] = {}


def _patch_e3nn():
    # e3nn 0.4.4 + torch>=2.6 weights_only default (same workaround as cafquantum)
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([slice])
    except Exception:
        pass


def _patch_polar_gto_kwarg():
    """From cafquantum (code/cafquant/mlips.py): accept-and-drop the retired force_pbc_evaluator kwarg if a newer
    graph_longrange than the one mace 0.3.16 pairs with is installed. Idempotent."""
    import inspect
    from graph_longrange.features import GTOElectrostaticFeatures
    orig = GTOElectrostaticFeatures.precompute_geometry
    if "force_pbc_evaluator" in inspect.signature(orig).parameters:
        return

    def patched(self, *args, force_pbc_evaluator=False, **kwargs):
        return orig(self, *args, **kwargs)

    GTOElectrostaticFeatures.precompute_geometry = patched


def _patch_polar_memory():
    """mace 0.3.16 SparseUvuTensorProduct (l x l -> 0 paths) builds a [atoms, mul, mul] pair matrix
    (einsum 'bud,bvd->buv', 512x512 per atom for MACE-POLAR-1) and only then contracts it with the weights. Contracting
    the weights first is mathematically identical and needs just [atoms, mul, d]. Idempotent."""
    import math
    import torch
    from mace.modules import field_blocks
    cls = field_blocks.SparseUvuTensorProduct
    if getattr(cls, "_medrag_patched", False):
        return

    def forward(self, x1, x2):
        if x1.ndim != 2 or x2.ndim != 2:
            raise ValueError("SparseUvuTensorProduct expects flattened 2D tensors with shape [batch, irreps.dim]")
        batch = x1.shape[0]
        out = x1.new_zeros((batch, self.irreps_out.dim))

        def to_mul_ir(block, mul, dim):
            if self.layout == "mul_ir":
                return block.view(batch, mul, dim)
            return block.view(batch, dim, mul).transpose(1, 2)

        def from_mul_ir(t):
            if self.layout == "mul_ir":
                return t.reshape(batch, -1)
            return t.transpose(1, 2).reshape(batch, -1)

        for (in1_start, in1_stop, in2_start, in2_stop, out_start, out_stop, w_start, w_stop, path_weight,
             mode_code, mul1, mul2, d1) in self._path_meta:
            in1_block = x1[:, in1_start:in1_stop]
            in2_block = x2[:, in2_start:in2_stop]
            out_block = out[:, out_start:out_stop]
            if mode_code == 0:
                x1v = to_mul_ir(in1_block, mul1, d1)
                x2v = to_mul_ir(in2_block, mul2, d1)
                w = self.weight[w_start:w_stop].view(mul1, mul2)
                mixed = (x1v * torch.einsum("uv,bvd->bud", w, x2v)).sum(-1) / math.sqrt(float(d1))   # changed
                out[:, out_start:out_stop] = out_block + path_weight * mixed
            else:
                x1v = to_mul_ir(in1_block, mul1, d1)
                scalars = in2_block.view(batch, mul2)
                w = self.weight[w_start:w_stop].view(mul1, mul2)
                mixed = torch.einsum("bv,uv->bu", scalars, w)
                contrib = path_weight * (x1v * mixed.unsqueeze(-1) / math.sqrt(float(d1)))
                out_block_mi = to_mul_ir(out_block, mul1, d1)
                out[:, out_start:out_stop] = from_mul_ir(out_block_mi + contrib)
        return out * self.output_mask

    cls.forward = forward
    cls._medrag_patched = True


class Centered:
    """ASE calculator wrapper for MACE models: evaluates a copy of the atoms translated to their centroid.

    mace sizes its neighbour-search box for non-periodic input as max(|xyz|) * 5 * cutoff, so a structure sitting at PDB
    coordinates (tens of A from the origin) gets a box thousands of A across and the process runs out of RAM (a 60-atom
    ligand alone peaked at 5 GB). Energies/forces are translation invariant; the stored coordinates are left alone."""

    def __new__(cls, inner):
        from ase.calculators.calculator import Calculator, all_changes

        class _Centered(Calculator):
            implemented_properties = ["energy", "forces"]

            def __init__(self):
                super().__init__()
                self.inner = inner

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                a = self.atoms.copy()
                a.set_constraint()
                a.positions -= a.positions.mean(axis=0)
                a.info.update(self.atoms.info)
                a.calc = self.inner
                self.results = {"energy": a.get_potential_energy(), "forces": a.get_forces(apply_constraint=False)}

        return _Centered()


def get_calculator(method: str = "uma", device: str = "cpu", dtype: str | None = None):
    """UMA / MACE-POLAR-1 s|m (omol-style: charge + spin read from atoms.info) or MACE-OFF23 (small organic molecules only).

    dtype: MACE precision, 'float64' (cafquantum's setting) or 'float32' (UMADock's mace_dtype option; about half the memory).
    Default: the MEDRAG_MACE_DTYPE environment variable, else float64. UMA ignores it."""
    dtype = dtype or os.environ.get("MEDRAG_MACE_DTYPE", "float64")
    key = f"{method}:{dtype}" if method != "uma" else method
    if key in _CACHE:
        return _CACHE[key]
    from medrag import resources
    t0 = time.time()
    _patch_e3nn()
    if method in ("polar1-s", "polar1-m"):
        # adapted from cafquantum make_calculator: download first (cached in ~/.cache/mace), then the RAM check
        _patch_polar_gto_kwarg()
        _patch_polar_memory()
        from mace.calculators import MACECalculator
        from mace.calculators.foundations_models import download_mace_polar_checkpoint
        path = download_mace_polar_checkpoint({"polar1-s": "polar-1-s", "polar1-m": "polar-1-m"}[method])
        ok, why = resources.fits(method)
        if not ok:
            raise MemoryError(why)
        calc = Centered(MACECalculator(model_paths=[path], device=device, default_dtype=dtype, charges_key="charge"))
        LAST_LOAD_S[method] = time.time() - t0
        _CACHE[key] = calc
        return calc
    ok, why = resources.fits(method)
    if not ok:
        raise MemoryError(why)
    if method == "uma":
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.calculate import pretrained_mlip
        try:
            predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device=device)
        except Exception:
            ckpt = resources.find_weights("uma")
            predictor = pretrained_mlip.load_predict_unit(str(ckpt), device=device)
        calc = FAIRChemCalculator(predictor, task_name="omol")
    elif method == "maceoff":
        from mace.calculators.foundations_models import mace_off
        calc = Centered(mace_off(model="medium", device=device, default_dtype=dtype))
    else:
        raise ValueError(f"unknown MLIP {method!r}")
    LAST_LOAD_S[method] = time.time() - t0
    _CACHE[key] = calc
    return calc
