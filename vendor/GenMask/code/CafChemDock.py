"""
CafChemDock: minimal AutoDock Vina docking at a known binding-site box center.

A self-contained reimplementation of the centroid-docking path used by the
dock_assist repo's vina_dock.py (blind pocket detection is intentionally left
out -- this module always docks at a caller-supplied box center). Kept
independent on purpose: GenMaskFill has no import-time or runtime dependency
on the dock_assist repo, only on packages installed in its own venv
(dockstring for the vendored Vina binary, RDKit for ligand 3D embedding, and
the `obabel` CLI installed by the openbabel-wheel package).
"""
import os
import re
import shutil
import subprocess
import tempfile


class DockError(Exception):
    """Recoverable docking failure (bad SMILES, Vina crash, obabel error, ...)."""


def find_vina_bin(explicit=None):
    """Locate the Vina binary: an explicit path, else dockstring's vendored copy."""
    if explicit:
        if not os.path.exists(explicit):
            raise DockError(f"vina binary not found: {explicit}")
        return explicit
    try:
        import dockstring
    except ImportError:
        raise DockError("could not import dockstring to locate the vendored Vina binary; "
                        "pass vina_bin explicitly.")
    candidates = [
        os.path.join(os.path.dirname(dockstring.__file__), "resources", "bin", "vina_mac_catalina"),
        os.path.join(os.path.dirname(dockstring.__file__), "resources", "bin", "vina_linux"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise DockError(f"no vendored Vina binary found alongside dockstring ({candidates}).")


def require(tool):
    if shutil.which(tool) is None:
        raise DockError(f"required tool '{tool}' not found on PATH.")


def _convert(cmd, label):
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise DockError(f"{label} failed (obabel exit {res.returncode})\n"
                        f"  cmd: {' '.join(cmd)}\n  stderr: {res.stderr.strip()}")
    return res.stdout.strip()


def _build_ligand_pdbqt(smiles, sdf_path, pdbqt_path):
    """SMILES -> 3D conformer (RDKit ETKDG + MMFF) -> SDF -> PDBQT (Open Babel)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise DockError(f"could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=0xF1A9) != 0:
        raise DockError(f"RDKit 3D embedding failed for SMILES: {smiles}")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol)  # fallback if MMFF params missing
    w = Chem.SDWriter(sdf_path)
    w.write(mol)
    w.close()
    _convert(["obabel", "-isdf", sdf_path, "-opdbqt", "-O", pdbqt_path],
             f"ligand SDF->PDBQT ({smiles})")


def _build_receptor_pdbqt(pdb_path, pdbqt_path):
    """receptor PDB -> rigid PDBQT (Open Babel). -xr = no torsion tree (Vina requires a rigid receptor)."""
    _convert(["obabel", "-ipdb", pdb_path, "-h", "-opdbqt", "-xr", "-O", pdbqt_path],
             "receptor PDB->PDBQT")


def _run_vina(vina_bin, rec_pdbqt, lig_pdbqt, center, size,
             exhaustiveness, num_modes, seed, cpu, out_pdbqt, log_path):
    cmd = [
        vina_bin,
        "--receptor", rec_pdbqt,
        "--ligand", lig_pdbqt,
        "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
        "--size_x", str(size[0]), "--size_y", str(size[1]), "--size_z", str(size[2]),
        "--out", out_pdbqt,
        "--log", log_path,
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
    ]
    if cpu:
        cmd += ["--cpu", str(cpu)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise DockError(f"Vina failed (exit {res.returncode})\n  stderr: {res.stderr.strip()}")


def _parse_vina_log(log_path):
    """Return the best (first, most favorable) affinity in a Vina log, or None."""
    if not os.path.exists(log_path):
        return None
    row_re = re.compile(r"^\s*\d+\s+(-?\d+\.\d+)\s+")
    with open(log_path) as fh:
        for line in fh:
            m = row_re.match(line)
            if m:
                return float(m.group(1))
    return None


def dock_at_centroid(receptor_pdb, smiles_list, center, box_size=22.0,
                     exhaustiveness=8, num_modes=9, seed=0, cpu=0,
                     vina_bin=None, overwrite_receptor=False):
    """Dock each SMILES into `receptor_pdb` at a fixed box center.

    Args:
        receptor_pdb: path to a receptor PDB file.
        smiles_list: iterable of ligand SMILES strings (duplicates collapse
            naturally since the return value is keyed by SMILES).
        center: (x, y, z) box center in Angstroms -- e.g. a co-crystallized
            ligand's centroid.
        box_size: cubic box edge length in Angstroms (default 22).
        exhaustiveness, num_modes, seed, cpu: Vina search parameters.
        vina_bin: explicit path to a Vina binary (default: dockstring's
            vendored copy).
        overwrite_receptor: rebuild the cached receptor PDBQT even if one
            already exists next to receptor_pdb.

    Returns:
        dict mapping each input SMILES to its best (most negative = most
        favorable) Vina affinity in kcal/mol, or None if that molecule failed
        to dock (bad SMILES, embedding failure, Vina crash -- logged to
        stdout, does not abort the rest of the batch).
    """
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]
    if not os.path.exists(receptor_pdb):
        raise DockError(f"receptor not found: {receptor_pdb}")
    require("obabel")
    vina_bin = find_vina_bin(vina_bin)
    receptor_pdb = os.path.abspath(receptor_pdb)
    stem = os.path.splitext(receptor_pdb)[0]
    rec_pdbqt = stem + ".pdbqt"
    if overwrite_receptor or not os.path.exists(rec_pdbqt):
        _build_receptor_pdbqt(receptor_pdb, rec_pdbqt)

    center = [float(c) for c in center]
    size = [float(box_size)] * 3

    scores = {}
    work = tempfile.mkdtemp(prefix="cafchemdock_")
    try:
        for i, smi in enumerate(smiles_list):
            if smi in scores:
                continue
            try:
                lig_sdf = os.path.join(work, f"lig_{i}.sdf")
                lig_pdbqt = os.path.join(work, f"lig_{i}.pdbqt")
                _build_ligand_pdbqt(smi, lig_sdf, lig_pdbqt)
                poses_pdbqt = os.path.join(work, f"poses_{i}.pdbqt")
                log_path = os.path.join(work, f"vina_{i}.log")
                _run_vina(vina_bin, rec_pdbqt, lig_pdbqt, center, size,
                         exhaustiveness, num_modes, seed, cpu, poses_pdbqt, log_path)
                aff = _parse_vina_log(log_path)
                if aff is None:
                    print(f"CafChemDock: no score parsed for {smi}")
                scores[smi] = aff
            except DockError as e:
                print(f"CafChemDock: docking failed for {smi}: {e}")
                scores[smi] = None
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return scores
