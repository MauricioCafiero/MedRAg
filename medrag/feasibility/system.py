"""Build the MLIP-ready system for one docked pose: a capped pocket cut-out
plus the ligand (with explicit hydrogens), as ASE Atoms.

The pocket cut-out is UMADock's (vendor/UMADock/code/UMADock/prep_binding_site.py): PDBFixer protonation at pH,
whole residues near the ligand, ACE/NME peptide caps on cut chain ends, net charge from the ff14SB partial-charge
sum, and only the C-alpha of each residue constrained. The one addition is selecting residues from the docked
ligand's coordinates (UMADock looks the ligand up by residue name in the PDB) and across all chains (the HMGCR
active site is built from two monomers).
"""
from __future__ import annotations

import importlib.util
import io
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from ase import Atoms
from openmm import app, unit
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import cKDTree

_PREP = Path(__file__).resolve().parents[2] / "vendor/UMADock/code/UMADock/prep_binding_site.py"
_spec = importlib.util.spec_from_file_location("umadock_prep_binding_site", _PREP)
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def ligand_with_h(pose_sdf_block: str, smiles: str) -> Chem.Mol:
    """Docked heavy-atom pose + the SMILES it was docked from -> RDKit mol with explicit H and 3D coords.
    Vina/PDBQT poses carry only polar hydrogens, so bond orders come from the SMILES template and the
    remaining hydrogens are placed geometrically."""
    pose = Chem.MolFromMolBlock(pose_sdf_block, removeHs=False, sanitize=False)
    if pose is None:
        raise ValueError("could not parse pose SDF block")
    rw = Chem.RWMol(pose)                      # drop the pose's own polar H by hand: removeHs(sanitize=False) can leave
    for i in sorted((a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 1), reverse=True):   # them, and the
        rw.RemoveAtom(i)                       # template's explicit H count then double-counts (N valence 6)
    pose = rw.GetMol()
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"bad SMILES {smiles!r}")
    # match the SMILES graph (bond orders ignored) onto the heavy-atom pose and copy its coordinates onto the template:
    # atoms read from a mol block keep a no-implicit-H flag that AssignBondOrdersFromTemplate carries over
    pat = graph_only(template)
    pose_g = graph_only(pose)
    match = pose_g.GetSubstructMatch(pat) if pose.GetNumAtoms() == pat.GetNumAtoms() else ()
    if not match:
        raise ValueError("pose heavy atoms do not match the SMILES graph")
    xyz = pose.GetConformer().GetPositions()
    conf = Chem.Conformer(template.GetNumAtoms())
    for t_idx, p_idx in enumerate(match):       # match[i] = pose atom for pattern (= template) atom i
        conf.SetAtomPosition(t_idx, xyz[p_idx].tolist())
    fixed = Chem.Mol(template)
    fixed.RemoveAllConformers()
    fixed.AddConformer(conf, assignId=True)
    return Chem.AddHs(fixed, addCoords=True)


def graph_only(mol: Chem.Mol) -> Chem.Mol:
    """Heavy-atom copy with every bond single and no formal charges, for symmetry-aware RMSD. Tautomer/charge
    placement (which carboxylate oxygen carries the minus) must not decide whether two poses 'match'."""
    m = Chem.RWMol(Chem.RemoveHs(mol, sanitize=False))
    for b in m.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in m.GetAtoms():
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
    return m.GetMol()


def sym_rmsd(graph: Chem.Mol, xyz_a: np.ndarray, xyz_b: np.ndarray, max_perms: int = 5000) -> float:
    """Heavy-atom RMSD, no alignment, minimised over the graph's automorphisms (equivalent atoms: methyls, phenyl
    flips, carboxylate oxygens...). Both coordinate arrays must be in `graph`'s atom order."""
    perms = graph.GetSubstructMatches(graph, uniquify=False, useChirality=False, maxMatches=max_perms)
    return float(min(np.sqrt(((xyz_a - xyz_b[list(p)]) ** 2).sum(axis=1).mean()) for p in perms))


def mol_to_atoms(mol: Chem.Mol) -> Atoms:
    conf = mol.GetConformer()
    return Atoms([a.GetSymbol() for a in mol.GetAtoms()], positions=conf.GetPositions())


@dataclass
class PoseSystem:
    complex: Atoms                    # pocket atoms first, ligand last
    n_pocket: int
    n_lig: int
    fixed: list[int]                  # constrained indices (caps, backbone, far residues)
    labels: list[str]                 # per complex atom, e.g. "ASP45A:OD1" / "LIG:O3"
    lig_mol: Chem.Mol
    charge: int
    warnings: list[str] = field(default_factory=list)
    caps: list[int] = field(default_factory=list)    # ACE/NME cap atom indices (relaxed first, see checks.relax_caps)

    @property
    def lig_slice(self) -> slice:
        return slice(self.n_pocket, self.n_pocket + self.n_lig)

    def ligand_atoms(self, from_complex: Atoms | None = None) -> Atoms:
        src = from_complex if from_complex is not None else self.complex
        lig = src[self.lig_slice].copy()
        lig.info["charge"] = Chem.GetFormalCharge(self.lig_mol)
        lig.info["spin"] = 1
        lig.constraints = []
        return lig

    def frozen_edge(self, radius: float) -> list[int]:
        """Pocket atoms to freeze during relaxation/MD: every residue (contiguous run of atoms sharing a residue label,
        ACE/NME caps count as their own residues) with no atom within `radius` A of any ligand atom. The cluster edge
        carries forces from the cut-out, which otherwise stop the relaxation from ever converging."""
        P = self.n_pocket
        pos = self.complex.positions
        dist = np.linalg.norm(pos[:P, None, :] - pos[None, P:, :], axis=2).min(axis=1)
        frozen, start = [], 0
        for i in range(1, P + 1):
            if i == P or self.labels[i].split(":")[0] != self.labels[start].split(":")[0]:
                if dist[start:i].min() > radius:
                    frozen.extend(range(start, i))
                start = i
        return frozen

    def lig_heavy(self) -> list[int]:
        """Indices (within the ligand) of non-hydrogen atoms."""
        return [i for i, a in enumerate(self.lig_mol.GetAtoms()) if a.GetAtomicNum() > 1]


@lru_cache(maxsize=4)
def _repaired_receptor(receptor_pdb: str, ph: float):
    """PDBFixer-repaired, protonated receptor (UMADock step 1). Only ATOM records and structural metals are kept
    (waters and crystal ligands out), altloc B+ dropped. Cached: the whole protein is fixed once per receptor."""
    keep = []
    for line in open(receptor_pdb):
        if line[16:17] not in (" ", "A") and line.startswith(("ATOM", "HETATM")):
            continue
        if line.startswith("ATOM") or (line.startswith("HETATM") and line[17:20].strip() in prep.SUPPORTED_METAL_RESNAMES):
            keep.append(line)
    fixer = PDBFixer(pdbfile=io.StringIO("".join(keep)))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    # OpenMM's Modeller.addHydrogens starts every new hydrogen at a random position (python's global `random`) and then
    # minimises, so methyl / NH3+ / OH rotamers land in a different local minimum on every run: the same pocket came out
    # with 94 hydrogens placed differently and single-point energies 5-13 eV apart. PDBFixer.addMissingHydrogens has no
    # seed argument, so seed the global generator for this call only and restore it afterwards.
    state = random.getstate()
    random.seed(H_PLACEMENT_SEED)
    try:
        fixer.addMissingAtoms(seed=H_PLACEMENT_SEED)
        fixer.addMissingHydrogens(pH=ph)
    finally:
        random.setstate(state)
    return fixer.topology, fixer.positions.value_in_unit(unit.angstrom)


# --- cap placement from the real neighbours -------------------------------------------------------------------------
# UMADock's _place_ace / _place_nme put a cap on the N-CA (or C-CA) line extended past the cut, i.e. at 180 deg, while the
# real neighbour sits at ~120 deg. The approximation is harmless for most residues but on a proline the cap's methyl/oxygen
# land on the ring CD hydrogens (0.5 A), and caps over a one-residue gap sit on the same spot. When the neighbouring residue
# is in the structure its atoms ARE the cap: ACE C/O/CH3 = the previous residue's C/O/CA, NME N/CH3 = the next residue's N/CA.
# Positions are keyed by the atom position UMADock passes in; anything not found falls back to UMADock's own placement.
_NATIVE_CAPS: dict[str, dict] = {"ace": {}, "nme": {}}
_umadock_place_ace, _umadock_place_nme = prep._place_ace, prep._place_nme


def _pkey(v) -> tuple:
    return tuple(np.round(np.asarray(v, dtype=float), 3))


def _place_ace_native(n_pos, ca_pos):
    return _NATIVE_CAPS["ace"].get(_pkey(n_pos)) or _umadock_place_ace(n_pos, ca_pos)


def _place_nme_native(c_pos, ca_pos):
    return _NATIVE_CAPS["nme"].get(_pkey(c_pos)) or _umadock_place_nme(c_pos, ca_pos)


prep._place_ace, prep._place_nme = _place_ace_native, _place_nme_native


def _register_native_caps(topology, pos_ang, selected) -> int:
    """Fill _NATIVE_CAPS for every selected residue whose chain neighbour is in the structure. Returns how many caps."""
    have = _protein_residues(topology)
    _NATIVE_CAPS["ace"].clear()
    _NATIVE_CAPS["nme"].clear()
    pos = lambda res, name: next((np.array(prep._heavy_pos_ang(a, pos_ang)) for a in res.atoms() if a.name == name), None)
    n = 0
    for r in selected:
        if not r.id.strip().lstrip("-").isdigit():
            continue
        c, i = r.chain.id, int(r.id)
        prev, nxt = have.get((c, i - 1)), have.get((c, i + 1))
        if prev is not None and pos(r, "N") is not None:                      # ACE on this residue's N
            ch3, cc, o = pos(prev, "CA"), pos(prev, "C"), pos(prev, "O")
            if ch3 is not None and cc is not None and o is not None:
                _NATIVE_CAPS["ace"][_pkey(pos(r, "N"))] = {"CH3": ch3, "C": cc, "O": o}
                n += 1
        if nxt is not None and pos(r, "C") is not None:                       # NME on this residue's C
            nn, ch3 = pos(nxt, "N"), pos(nxt, "CA")
            if nn is not None and ch3 is not None:
                _NATIVE_CAPS["nme"][_pkey(pos(r, "C"))] = {"N": nn, "CH3": ch3}
                n += 1
    return n


H_PLACEMENT_SEED = 0     # fixed so the carved site (and every MLIP number computed on it) is reproducible
MAX_AUTO_GAP = 1         # a gap of this many missing residues is filled: the NME and ACE caps would sit on the same residue
COLLISION_A = 0.9        # two atoms of the cut-out closer than this are physically impossible (an X-H bond is >= 0.96 A)


class SiteBuildError(RuntimeError):
    """The carved binding site has impossible geometry (atoms closer than COLLISION_A); it is never handed to an MLIP."""


def _protein_residues(topology) -> dict:
    return {(r.chain.id, int(r.id)): r for r in topology.residues()
            if r.name in prep.PROTEIN_RES and r.id.strip().lstrip("-").isdigit()}


def _gaps(selected, have: dict, max_gap: int):
    """[(chain, first_id, last_id, [missing residues])] between consecutive selected residues of one chain, gap size
    1..max_gap, only when every missing residue exists in the structure."""
    chosen = sorted({(r.chain.id, int(r.id)) for r in selected if r.id.strip().lstrip("-").isdigit()})
    out = []
    for (c1, a), (c2, b) in zip(chosen, chosen[1:]):
        if c1 == c2 and 1 <= b - a - 1 <= max_gap and all((c1, k) in have for k in range(a + 1, b)):
            out.append((c1, a, b, [have[(c1, k)] for k in range(a + 1, b)]))
    return out


def _residues_near(topology, pos_ang, lig_pts: np.ndarray, cutoff: float, fill_gaps: bool = True):
    """UMADock's _select_residues, but from ligand coordinates and across all chains.

    fill_gaps: also take the residue of a one-residue gap between two selected residues of one chain (MAX_AUTO_GAP). The
    two fragments would be capped separately, and with native-position caps (see _register_native_caps) the NME cap of the
    first and the ACE cap of the second sit on that same residue, atom on atom (2A3R: a cap H 0.01 A from a backbone C,
    which stopped the pocket relaxation from ever converging). Joining the fragments removes both caps. Longer gaps need no
    joining: their caps are different real residues, >= 3.8 A apart. Pass False to reproduce UMADock's selection."""
    tree = cKDTree(lig_pts)
    selected = []
    for res in topology.residues():
        if res.name not in prep.PROTEIN_RES:
            continue
        pts = [p for a in res.atoms() if (p := prep._heavy_pos_ang(a, pos_ang)) is not None]
        if pts and tree.query(np.array(pts))[0].min() <= cutoff:
            selected.append(res)
    if fill_gaps and selected:
        for _c, _a, _b, missing in _gaps(selected, _protein_residues(topology), MAX_AUTO_GAP):
            selected.extend(missing)
    return selected


# X-H bond lengths (A). PDBFixer/OpenMM place every added hydrogen ~1.0 A from its parent: fine for O-H, N-H and C-H, but a
# thiol S-H comes out 0.34 A too short (1.0 vs 1.34), which the MLIP sees as a ~24 eV/A force on the cysteine S and H
# (HMGCR Cys18). Every X-H bond of the carved pocket is reset to its standard length along the existing bond direction.
XH_LENGTH = {"C": 1.09, "N": 1.01, "O": 0.96, "S": 1.34}


def normalise_xh_bonds(topology, xyz: np.ndarray) -> int:
    """Set every heavy-atom-H bond of `topology` (the carved pocket) to XH_LENGTH, in place on `xyz`. Returns how many
    bonds changed by more than 0.02 A."""
    atoms = list(topology.atoms())
    changed = 0
    for b in topology.bonds():
        a1, a2 = b.atom1, b.atom2
        if a1.element is None or a2.element is None:
            continue
        h, x = (a1, a2) if a1.element.symbol == "H" else (a2, a1) if a2.element.symbol == "H" else (None, None)
        if h is None or x.element.symbol not in XH_LENGTH:
            continue
        v = xyz[h.index] - xyz[x.index]
        d = float(np.linalg.norm(v))
        if d < 1e-6:
            continue
        new = XH_LENGTH[x.element.symbol]
        if abs(d - new) > 0.02:
            changed += 1
        xyz[h.index] = xyz[x.index] + v * (new / d)
    return changed


def pocket_collisions(pocket_xyz: np.ndarray, max_pairs: int = 20) -> list[tuple[int, int, float]]:
    """Atom pairs of the carved pocket closer than COLLISION_A: (i, j, distance), closest first."""
    pairs = cKDTree(pocket_xyz).query_pairs(COLLISION_A)
    out = sorted(((i, j, float(np.linalg.norm(pocket_xyz[i] - pocket_xyz[j]))) for i, j in pairs), key=lambda t: t[2])
    return out[:max_pairs]


def build_system(receptor_pdb: str, lig_mol: Chem.Mol, cutoff: float = 4.0, ph: float = 7.0,
                 fill_gaps: bool = True) -> PoseSystem:
    """Carve the binding site around `lig_mol` and verify it.

    Caps are placed from the real neighbouring residues and one-residue gaps are joined, so the site is built right the
    first time. It is still checked every time: if any two atoms of the pocket are closer than COLLISION_A, SiteBuildError
    is raised (a broken site is never handed to an MLIP). With fill_gaps=False (UMADock's exact selection) collisions are
    only reported in PoseSystem.warnings, which is how the old behaviour can be reproduced."""
    topology, pos_ang = _repaired_receptor(str(receptor_pdb), ph)
    lig_xyz = lig_mol.GetConformer().GetPositions()
    heavy = np.array([a.GetAtomicNum() > 1 for a in lig_mol.GetAtoms()])
    lig_pts = lig_xyz[heavy]

    selected = _residues_near(topology, pos_ang, lig_pts, cutoff, fill_gaps)
    if not selected:
        raise ValueError("no receptor residues within the cutoff of the ligand: the pose is not in this receptor")
    metals = prep._select_metals(topology, pos_ang, lig_pts, cutoff)
    system, pocket_xyz = _assemble(topology, pos_ang, selected, metals, lig_mol, cutoff)
    bad = pocket_collisions(pocket_xyz)
    if bad:
        desc = ", ".join(f"{system.labels[i]}-{system.labels[j]} {d:.2f} A" for i, j, d in bad[:3])
        if fill_gaps:
            raise SiteBuildError(f"carved site has {len(bad)} impossible atom pair(s): {desc}")
        system.warnings.append(f"UNCHECKED cut-out collisions ({len(bad)} pairs, fill_gaps=False): {desc}")
    else:
        system.warnings.append(f"site check: no atom pair closer than {COLLISION_A} A")
    return system


def _assemble(topology, pos_ang, selected, metals, lig_mol, cutoff) -> tuple[PoseSystem, np.ndarray]:
    lig_xyz = lig_mol.GetConformer().GetPositions()
    _register_native_caps(topology, pos_ang, selected)
    cluster_top, cluster_pos_nm, _, _ = prep.build_cluster(topology, pos_ang, selected)
    charge, modeller = prep._net_charge(cluster_top, cluster_pos_nm)
    charge += sum(prep.METAL_CHARGE[r.name] for r in metals)
    if metals:
        mtop = app.Topology()
        mchain = mtop.addChain()
        mpos = []
        for res in metals:
            for a in res.atoms():
                if a.element is not None:
                    mtop.addAtom(a.name, a.element, mtop.addResidue(res.name, mchain))
                    mpos.append([float(x) / 10.0 for x in pos_ang[a.index]])
        modeller.add(mtop, unit.Quantity(np.array(mpos), unit.nanometer))

    top = modeller.topology
    pocket_xyz = np.array(modeller.positions.value_in_unit(unit.angstrom), dtype=float)
    normalise_xh_bonds(top, pocket_xyz)
    pocket_atoms = list(top.atoms())
    n_pocket = len(pocket_atoms)
    labels = [f"{a.residue.name}{a.residue.id}:{a.name}" for a in pocket_atoms]
    fixed = [i for i, a in enumerate(pocket_atoms) if a.name == "CA" and a.residue.name in prep.PROTEIN_RES]
    caps = [i for i, a in enumerate(pocket_atoms) if a.residue.name in ("ACE", "NME")]

    lig_sym = [a.GetSymbol() for a in lig_mol.GetAtoms()]
    labels += [f"LIG:{s}{i}" for i, s in enumerate(lig_sym)]
    atoms = Atoms([a.element.symbol for a in pocket_atoms] + lig_sym, positions=np.vstack([pocket_xyz, lig_xyz]))
    charge += Chem.GetFormalCharge(lig_mol)
    atoms.info["charge"] = int(charge)
    atoms.info["spin"] = 1
    warns = [f"cut-out: {len(selected)} residues within {cutoff} A, {len(metals)} metal ion(s); "
             "only C-alpha atoms are constrained"]
    return PoseSystem(complex=atoms, n_pocket=n_pocket, n_lig=len(lig_sym), fixed=fixed, labels=labels,
                      lig_mol=lig_mol, charge=int(charge), warnings=warns, caps=caps), pocket_xyz
