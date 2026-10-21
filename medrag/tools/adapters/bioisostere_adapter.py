"""JSON-friendly entry point onto medrag/lib/bioisostere_assist: the query_lead_cli.py ranking loop without the docking,
CSV and image parts (docking is a separate pipeline step)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / "lib/bioisostere_assist"
sys.path.insert(0, str(HERE))
import CafChemShape as shape      # first: blocks the incompatible openbabel bindings before oddt is imported
import fragment as frag
import library as lib
from rdkit import Chem

MAX_USEFUL_SIMILARITY = 0.999     # as in query_lead_cli: a match this close just rebuilds the lead


def analogues(lead: str, top: int = 10, library: str = str(HERE / "library_100k_usr.pkl")) -> list:
    """Shape-bioisostere swaps of the lead's substituents: [{smiles, similarity, lead_fragment, library_fragment}]."""
    method, params, fragments = lib.load_library(library)
    for s in [s for s in fragments if (m := Chem.MolFromSmiles(s)) is not None and frag.has_radical(m)]:
        del fragments[s]
    out = []
    for lead_frag, lead_core in frag.fragments_with_cores_from_smiles(lead, params["max_frag_atoms"]):
        d = lib.shape_for_fragment(lead_frag, method, params["n_confs"], params["energy_window"], params["prune_rms_thresh"])
        if d is None:
            continue
        ranked = [(shape.best_similarity(d, lib_d), lib_smi) for lib_smi, (lib_d, _label) in fragments.items()]
        ranked = sorted((r for r in ranked if r[0] < MAX_USEFUL_SIMILARITY), reverse=True)
        for sim, lib_smi in ranked[:top]:
            smi = frag.build_analogue(lead_core, lib_smi)
            if smi:
                out.append(dict(smiles=smi, similarity=float(sim), lead_fragment=lead_frag, library_fragment=lib_smi))
    return out
