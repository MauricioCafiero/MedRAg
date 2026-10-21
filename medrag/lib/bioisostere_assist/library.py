"""Shared library-building/loading glue for bioisostere_assist.

Turns a fragment SMILES into a shape descriptor (cap + embed + ODDT shape),
and persists/reloads a whole library of them as a pickle keyed by fragment
SMILES -- storing the full per-conformer descriptor list, not just fragment
identity, so a saved library can be queried against a new lead without
re-embedding/re-shaping the whole pool from scratch.
"""
import pickle

from rdkit import Chem

import CafChemShape as shape
import fragment as frag


def shape_for_fragment(frag_smiles, method, n_confs=shape.N_CONFORMERS,
                       energy_window=shape.ENERGY_WINDOW,
                       prune_rms_thresh=shape.PRUNE_RMS_THRESH):
    """Caps the fragment's dummy atom with methyl, embeds a conformer
    ensemble, and computes a shape descriptor for each surviving conformer.

    Returns a list of descriptors, or None on failure.
    """
    frag_mol = Chem.MolFromSmiles(frag_smiles)
    capped = frag.cap_dummy_with_methyl(frag_mol)
    if capped is None:
        return None
    try:
        confs = shape.embed_conformers(capped, n_confs=n_confs,
                                       energy_window=energy_window,
                                       prune_rms_thresh=prune_rms_thresh)
        return shape.shape_descriptors(confs, method)
    except Exception:
        return None


def save_library(path, method, params, fragments):
    """fragments: dict of fragment_smiles -> (descriptors, source_label)."""
    payload = {
        "method": method,
        "params": params,
        "fragments": {
            smi: {"source": label, "descriptors": d}
            for smi, (d, label) in fragments.items()
        },
    }
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)


def load_library(path):
    """Returns (method, params, fragments), where fragments is a dict of
    fragment_smiles -> (descriptors, source_label) -- the same shape
    save_library takes, so it round-trips directly into ranking code.
    """
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    fragments = {
        smi: (entry["descriptors"], entry["source"])
        for smi, entry in payload["fragments"].items()
    }
    return payload["method"], payload["params"], fragments
