"""JSON-friendly entry point onto vendor/GenMask (CafChemSubs.gen_mask); run through medrag.tools.vendor.call.
Adapted from vendor/GenMask/code/gen_mask_cli.py (same calls, no CSV/image handling)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "vendor/GenMask/code"))
import CafChemSubs as cs

DEFAULT_MODEL = "cafierom/bert-base-cased-ChemTok-ZN250K-V1"


def analogues(smiles: str, percent: float = 0.10, model: str = DEFAULT_MODEL) -> list:
    """Mask/unmask `smiles`; return [{smiles, entropy, qed, tanimoto_to_parent}] (valid, unique analogues)."""
    cs.genmask_model_setup(0, model)
    final_smiles, entropy, qeds, _mols, out_text, _img = cs.gen_mask(smiles, percent)
    if out_text == "Invalid SMILES string":
        raise ValueError(f"gen_mask failed for {smiles!r}")
    sims = []
    if final_smiles:
        # mm_cutoff > 1 silences the function's own logging (Tanimoto is in [0, 1]), as gen_mask_cli does
        _, known = cs.calculate_similarities(smiles, final_smiles, mm_cutoff=1.1)
        sims = [float(x) for x in known[0]]
    return [dict(smiles=s, entropy=float(e), qed=float(q), tanimoto_to_parent=t)
            for s, e, q, t in zip(final_smiles, entropy, qeds, sims)]
