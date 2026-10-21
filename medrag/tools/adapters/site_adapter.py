"""JSON-friendly entry point onto vendor/FAO_MOLPROP_CLI (docking_module.find_contacts): the residues a ligand pose contacts
(H-bonds, hydrophobic contacts, pi-stacking, pi-cation, halogen bonds, salt bridges; ODDT). Same calls as that repo's
contacted_residues(), but taking a receptor PDB and a ligand SDF directly instead of a dockstring target."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "vendor/FAO_MOLPROP_CLI/code"))
import docking_module as dm
import oddt


def contacts(receptor_pdb: str, ligand_sdf: str) -> dict:
    """{residues: ['GLU146', ...], text: find_contacts report}. Residue ids are resname + resnum, as in the vendored code."""
    pro = next(oddt.toolkit.readfile("pdb", receptor_pdb))
    pro.protein = True
    lig = next(oddt.toolkit.readfile("sdf", ligand_sdf))
    text = dm.find_contacts(pro, lig)
    return dict(residues=sorted(dm._parse_residue_list(text)), text=text)


def _crop_receptor(receptor_pdb: str, ligand_sdf: str, out_pdb: str, radius: float = 8.0) -> str:
    """Receptor residues with any atom within `radius` A of the ligand, written to out_pdb. Contacts (H-bond, hydrophobic, pi,
    halogen, salt bridge) reach at most ~5 A, so nothing is lost, and ODDT no longer works through the whole protein for every
    pose (HMGCR's tetramer is 11,764 atoms: minutes per pose uncropped). Whole residues are kept so rings stay complete."""
    import numpy as np
    from rdkit import Chem
    from scipy.spatial import cKDTree
    lig = Chem.MolFromMolFile(ligand_sdf, removeHs=False, sanitize=False)
    tree = cKDTree(lig.GetConformer().GetPositions())
    lines = [l for l in open(receptor_pdb) if l.startswith(("ATOM", "HETATM"))]
    xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in lines])
    near = tree.query(xyz)[0] <= radius
    keep = {l[21:27] for l, n in zip(lines, near) if n}                  # chain + resseq + icode
    open(out_pdb, "w").write("".join(l for l in lines if l[21:27] in keep) + "END\n")
    return out_pdb


def contacts_many(receptor_pdb: str, ligand_sdfs: list, crop: float = 8.0) -> list:
    """contacts() for several ligand files, each against the receptor cropped to `crop` A around it. A failure gives {error}."""
    import tempfile
    out = []
    with tempfile.TemporaryDirectory() as td:
        for k, path in enumerate(ligand_sdfs):
            try:
                pdb = _crop_receptor(receptor_pdb, path, f"{td}/rec_{k}.pdb", crop) if crop else receptor_pdb
                pro = next(oddt.toolkit.readfile("pdb", pdb))
                pro.protein = True
                lig = next(oddt.toolkit.readfile("sdf", path))
                out.append(dict(residues=sorted(dm._parse_residue_list(dm.find_contacts(pro, lig)))))
            except Exception as e:
                out.append(dict(error=f"{type(e).__name__}: {e}"[:200]))
    return out
