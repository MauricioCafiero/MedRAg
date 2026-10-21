"""Receptor/pose file helpers shared by the dock and validate steps."""
from __future__ import annotations

import re
from pathlib import Path

from rdkit import Chem

from medrag.feasibility.system import graph_only, sym_rmsd


def write_clean_receptor(pdb_in: str, pdb_out: str) -> str:
    """Protein-only copy of a PDB: ATOM records only, no waters, ligands, cofactors or ions.
    Docking into a file that still holds the bound ligand blocks the site (a redock of rosuvastatin into 1HWL
    scored -5.8 with the ligand present, -8.2 without), so this is always done explicitly."""
    keep = [l for l in Path(pdb_in).read_text().splitlines() if l.startswith("ATOM")]
    if not keep:
        raise ValueError(f"no ATOM records in {pdb_in}")
    Path(pdb_out).write_text("\n".join(keep) + "\nEND\n")
    return pdb_out


def read_poses(sdf_path: str) -> list[tuple[str, float | None]]:
    """[(mol_block, vina_score)] for each pose in an obabel-written Vina SDF, best first."""
    text = Path(sdf_path).read_text()
    out = []
    for chunk in text.split("$$$$"):
        if not chunk.strip():
            continue
        m = re.search(r"VINA RESULT:\s+(-?\d+\.\d+)", chunk)
        out.append((chunk.strip("\n") + "\n", float(m.group(1)) if m else None))
    return out


def crystal_ligand_block(pdb_path: str, resname: str, chain: str) -> str:
    """Heavy-atom mol block of one co-crystallized ligand occurrence (bond orders come later, from a SMILES template)."""
    lines = [l for l in Path(pdb_path).read_text().splitlines()
             if l.startswith("HETATM") and l[17:20].strip() == resname and l[21] == chain
             and (l[76:78].strip() or l[12:16].strip()[0]) != "H"]
    if not lines:
        raise ValueError(f"no HETATM {resname} chain {chain} in {pdb_path}")
    mol = Chem.MolFromPDBBlock("\n".join(lines) + "\nEND\n", removeHs=False, sanitize=False, proximityBonding=True)
    return Chem.MolToMolBlock(mol, kekulize=False)


def heavy_rmsd(block_a: str, block_b: str, smiles: str | None = None) -> float:
    """Symmetry-aware, in-place (no alignment) heavy-atom RMSD between two poses of the same molecule given as mol
    blocks (atom order may differ). Matching uses the bare heavy-atom graph, so bond orders, charges and where a
    carboxylate's minus sits are irrelevant. `smiles` is accepted for call-site symmetry and not needed."""
    graphs, xyz = [], []
    for b in (block_a, block_b):
        m = Chem.MolFromMolBlock(b, removeHs=True, sanitize=False)
        g = graph_only(m)
        graphs.append(g)
        xyz.append(g.GetConformer().GetPositions())
    order = graphs[1].GetSubstructMatch(graphs[0])          # order[i] = atom of B matching atom i of A
    if len(order) != graphs[0].GetNumAtoms():
        raise ValueError("the two poses are not the same molecule (heavy-atom graphs differ)")
    return sym_rmsd(graphs[0], xyz[0], xyz[1][list(order)])
