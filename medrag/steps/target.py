"""set_target step: fix the receptor, binding site and reference ligand a run works against."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem

from medrag import ui
from medrag.state import RunState
from medrag.tools import poses


def set_target(state: RunState, pdb: str, resname: str, chain: str, name: str = "", ligand_smiles: str | None = None) -> dict:
    """pdb: raw crystal structure; resname/chain: the co-crystallized ligand that defines the site (its centroid is the Vina box
    centre). A protein-only copy is written for docking, since a ligand left in the file blocks the site."""
    t0 = ui.step_start("target", f"{Path(pdb).name}: ligand {resname} chain {chain} defines the site")
    d = state.dir / "target"
    d.mkdir(exist_ok=True)
    receptor = poses.write_clean_receptor(pdb, str(d / "receptor.pdb"))
    block = poses.crystal_ligand_block(pdb, resname, chain)
    (d / "crystal_ligand.mol").write_text(block)
    xyz = Chem.MolFromMolBlock(block, removeHs=False, sanitize=False).GetConformer().GetPositions()
    center = [round(float(c), 3) for c in xyz.mean(axis=0)]
    state.set_target(name=name or Path(pdb).stem, pdb=str(Path(pdb).resolve()), resname=resname, chain=chain,
                     receptor=str(Path(receptor).resolve()), crystal_ligand=str((d / "crystal_ligand.mol").resolve()),
                     center=center, ligand_smiles=ligand_smiles)
    state.log_step("set_target", pdb=pdb, resname=resname, chain=chain, center=center)
    ui.step_done("target", f"docking box centred on the {resname} ligand at {tuple(center)}; protein-only receptor written", t0)
    return state.target
