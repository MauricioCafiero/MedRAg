"""3D shape-similarity helpers, adapted from sim_assist's similarity_functions.py
(same author's CafChem project) -- trimmed to just the embedding + ODDT shape
pieces, dropping the Ollama-agent report formatting, and extended to keep a
whole conformer ensemble per fragment rather than a single best one (see
embed_conformers).

No OpenEye involved anywhere: RDKit for conformers, ODDT for shape
descriptors, both open-source. Requires the dedicated `rdkit_env` venv, built
from the vendored oddt-0.7.tar.gz (oddt is unmaintained; PyPI/GitHub could
vanish, hence vendoring -- see openeye/vendor/ and sim_assist/README.md).
"""
import sys
import numpy as np
# oddt (unmaintained, targets older NumPy) calls np.in1d, removed in NumPy 2.x.
# Shim it back onto np.isin before oddt's atom-typing code ever touches it.
if not hasattr(np, 'in1d'):
    np.in1d = np.isin

# openbabel-wheel (installed for CafChemDock.py's `obabel` CLI) also installs
# openbabel's Python bindings, whose API is incompatible with oddt's optional
# openbabel-backed toolkit (oddt.toolkits.ob raises AttributeError at import
# time, not the ImportError oddt's own try/except expects). We only ever use
# oddt's RDKit toolkit, so block the import outright -- this is the standard
# "make `import openbabel` raise ImportError" trick -- and let oddt's
# existing fallback pick rdk instead.
sys.modules.setdefault('openbabel', None)

from rdkit import Chem
from rdkit.Chem import AllChem

import oddt
from oddt.shape import usr, usr_cat, electroshape, usr_similarity

N_CONFORMERS = 10
ENERGY_WINDOW = 10.0    # kcal/mol above the lowest-energy conformer; above this, discard as strained
PRUNE_RMS_THRESH = 0.5  # Angstrom; conformers this close to one already kept are duplicates, skipped at embed time

# method name -> ODDT shape-descriptor function. All three return vectors
# compatible with the same usr_similarity() comparison (verified against
# sim_assist's own usage, which calls it uniformly regardless of method).
SHAPE_METHODS = {
    "usr": usr,
    "usr_cat": usr_cat,
    "electroshape": electroshape,
}


def embed_conformers(mol_or_smiles, n_confs=N_CONFORMERS, seed=0xf00d,
                     energy_window=ENERGY_WINDOW, prune_rms_thresh=PRUNE_RMS_THRESH):
    """Embeds up to n_confs conformers (ETKDGv3, RMSD-pruned during embedding
    so near-duplicates never survive), MMFF94-optimizes each, then discards
    any conformer more than energy_window kcal/mol above the lowest-energy
    one (highly strained conformers) -- the rest are kept, not just the best.

    Accepts a SMILES string or an existing RDKit Mol (Hs optional -- explicit
    Hs are added here).

    Returns an RDKit Mol containing only the surviving conformers (at least
    one). Raises ValueError if embedding fails entirely.
    """
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
        if mol is None:
            raise ValueError(f"could not parse SMILES: {mol_or_smiles}")
    else:
        mol = Chem.Mol(mol_or_smiles)
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 0
    params.pruneRmsThresh = prune_rms_thresh
    cids = list(AllChem.EmbedMultipleConfs(mol, n_confs, params))
    if not cids:
        params.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(mol, n_confs, params))
    if not cids:
        raise ValueError("could not embed any conformer")

    energies = {}
    for cid in cids:
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol)
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            if ff is None:
                continue
            ff.Minimize(maxIts=2000)
            energies[cid] = ff.CalcEnergy()
        except Exception:
            continue

    if not energies:
        # MMFF parameters unavailable (uncommon elements) -- can't judge
        # strain, so keep everything unfiltered rather than discard blindly.
        return mol

    min_e = min(energies.values())
    drop_cids = [cid for cid in cids
                if cid not in energies or energies[cid] - min_e > energy_window]
    for cid in drop_cids:
        mol.RemoveConformer(cid)

    return mol


def shape_descriptors(mol, method="usr"):
    """mol: an RDKit Mol with one or more conformers (e.g. from
    embed_conformers). method: 'usr' (pure shape), 'usr_cat' (shape +
    pharmacophore), or 'electroshape' (shape + chirality + electrostatics).

    Returns a list of descriptors, one per conformer in mol.
    """
    if method not in SHAPE_METHODS:
        raise ValueError(f"unknown shape method: {method}; choose one of {list(SHAPE_METHODS)}")
    fn = SHAPE_METHODS[method]
    descriptors = []
    for conf in mol.GetConformers():
        single = Chem.Mol(mol, confId=conf.GetId())
        descriptors.append(fn(oddt.toolkit.Molecule(single)))
    return descriptors


def best_similarity(descriptors_a, descriptors_b):
    """Max pairwise similarity across every (conformer of A, conformer of B)
    combination -- "do these two fragments have any pair of conformations
    that overlay well," matching how a BROOD-style search actually works,
    rather than comparing only each fragment's single default shape.
    """
    return max(usr_similarity(da, db) for da in descriptors_a for db in descriptors_b)
