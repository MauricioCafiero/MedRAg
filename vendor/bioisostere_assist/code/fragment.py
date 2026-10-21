"""RDKit-only fragmentation for bioisostere search: single-bond cuts
(rdMMPA -- the same algorithm mmpdb wraps) producing dummy-tagged
substituent-like pieces, then capping the open valence with a methyl so the
fragment is a complete, embeddable molecule.

No OpenEye/BROOD involved. Unlike a BROOD-native fragment database, the
methyl cap here is permanent -- there's no proprietary format requiring the
dummy atom's identity to survive, so the capped fragment IS the shape
representation used downstream (see CafChemShape.py).
"""
from rdkit import Chem
from rdkit.Chem import rdMMPA

MAX_FRAG_HEAVY_ATOMS = 8  # keep "substituent-like" pieces, not the big remainder


def has_radical(mol):
    """True if any atom carries an unpaired electron. ZINC contains a small
    number of genuine radical species (8 of the 100k druglike seeds, e.g.
    [CH2]C(C)(C)c1ccccc1), and their fragments rank WELL on shape similarity
    -- a radical is geometrically identical to its closed-shell twin, and USR
    descriptors only see geometry -- so they have to be rejected explicitly
    rather than filtered out by the similarity ranking.
    """
    return any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())


def cap_dummy_with_methyl(mol):
    """Replace every dummy atom (atomic num 0) with carbon. Returns a new,
    sanitized RDKit Mol, or None on failure.
    """
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(6)
            atom.SetNoImplicit(False)
            atom.SetIsotope(0)
    capped = rw.GetMol()
    try:
        Chem.SanitizeMol(capped)
    except Exception:
        return None
    return capped


def fragments_from_smiles(smiles, source_label, max_frag_heavy_atoms=MAX_FRAG_HEAVY_ATOMS):
    """Single-bond-cut fragmentation (rdMMPA, maxCuts=1). Returns a list of
    (fragment_smiles_with_dummy_tag, source_label) for pieces small enough to
    be substituent-like (<= max_frag_heavy_atoms heavy atoms, including the
    dummy) and containing exactly one attachment point.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    cuts = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False)

    seen = set()
    out = []
    for core, chains in cuts:
        for piece in chains.split(".") + ([core] if core else []):
            piece = piece.strip()
            if not piece or piece in seen:
                continue
            seen.add(piece)
            pmol = Chem.MolFromSmiles(piece)
            if pmol is None or has_radical(pmol):
                continue
            n_dummies = sum(1 for a in pmol.GetAtoms() if a.GetAtomicNum() == 0)
            n_heavy = pmol.GetNumAtoms()
            if n_dummies == 1 and n_heavy <= max_frag_heavy_atoms:
                out.append((piece, source_label))
    return out


def fragments_with_cores_from_smiles(smiles, max_frag_heavy_atoms=MAX_FRAG_HEAVY_ATOMS):
    """Single-bond-cut fragmentation (rdMMPA, maxCuts=1), returning
    (substituent_smiles, core_smiles) pairs -- the core is the complementary
    piece from the SAME cut, carrying a matching numbered dummy atom
    (`[*:1]`), so a replacement substituent can be grafted onto the core via
    Chem.molzip(core_mol, replacement_mol) to build a new analogue molecule.

    Used for a query lead (where the reconstructed core is needed); the
    pooled library itself only needs bare substituents (fragments_from_smiles).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    cuts = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False)

    seen = set()
    out = []
    for core, chains in cuts:
        if core:
            continue  # maxCuts=1 always yields an empty core; guard anyway
        pieces = [p.strip() for p in chains.split(".") if p.strip()]
        if len(pieces) != 2:
            continue
        for i in (0, 1):
            sub_smi, core_smi = pieces[i], pieces[1 - i]
            if sub_smi in seen:
                continue
            pmol = Chem.MolFromSmiles(sub_smi)
            if pmol is None or has_radical(pmol):
                continue
            n_dummies = sum(1 for a in pmol.GetAtoms() if a.GetAtomicNum() == 0)
            n_heavy = pmol.GetNumAtoms()
            if n_dummies == 1 and n_heavy <= max_frag_heavy_atoms:
                seen.add(sub_smi)
                out.append((sub_smi, core_smi))
    return out


def build_analogue(core_smiles, replacement_frag_smiles):
    """Grafts replacement_frag_smiles onto core_smiles at their matching
    numbered dummy atoms (Chem.molzip). Returns the analogue's canonical
    SMILES, or None on failure.
    """
    core_mol = Chem.MolFromSmiles(core_smiles)
    frag_mol = Chem.MolFromSmiles(replacement_frag_smiles)
    if core_mol is None or frag_mol is None:
        return None
    try:
        analogue = Chem.molzip(core_mol, frag_mol)
        Chem.SanitizeMol(analogue)
    except Exception:
        return None
    return Chem.MolToSmiles(analogue)


def build_fragment_pool(seed_smiles_with_labels, max_frag_heavy_atoms=MAX_FRAG_HEAVY_ATOMS):
    """seed_smiles_with_labels: iterable of (smiles, label) pairs.
    Returns a deduped list of (fragment_smiles, source_label) -- first
    occurrence's label wins on duplicates.
    """
    pool = []
    for smi, label in seed_smiles_with_labels:
        pool.extend(fragments_from_smiles(smi, label, max_frag_heavy_atoms))
    seen = {}
    for smi, label in pool:
        seen.setdefault(smi, label)
    return list(seen.items())
