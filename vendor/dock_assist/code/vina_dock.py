#!/usr/bin/env python3
"""Standalone AutoDock Vina docking for a user-provided receptor PDB + a SMILES ligand.

Uses only what is already installed in this env:
  - the Vina 1.1.2 binary vendored inside dockstring (called as a subprocess),
  - Open Babel (obabel CLI) for PDB/PDBQT conversion,
  - RDKit for ligand 3D embedding.

This is a standalone test harness, deliberately separate from the main
molopt.py / docking_module.py pipeline. It is NOT wired into the optimizer.

Flow:
  receptor.pdb  --obabel-->  receptor.pdbqt
  SMILES        --RDKit 3D--> ligand.sdf --obabel--> ligand.pdbqt
  vina --receptor receptor.pdbqt --ligand ligand.pdbqt --box ... --out poses.pdbqt
  poses.pdbqt (best model) --obabel--> best_pose.sdf   (for inspection)

Example:
  python3 vina_dock.py --receptor dude_receptor_ADRB2.pdb --smiles 'c1ccc(O)cc1' \
      --center -1.0 2.0 3.0 --size 20 20 20
  # for a receptor with no known binding site, use --blind to detect pockets instead
"""
import argparse
import contextlib
import glob
import io
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))

# When True, blind_dock docks the top-2 detected pockets and, if the best-score
# pocket's pose is NOT near a co-crystallized ligand, switches to the other
# pocket when IT is near one (proximity preferred over Vina score). Default off
# so the agent-facing tool behaves unchanged unless a developer opts in here.
#
# This is a module global -- NOT a blind_dock_agent kwarg -- on purpose: the
# LLM tool-caller would otherwise hallucinate the optional parameter (see the
# agent-tool-signature gotcha). A developer flips this in code; the model never
# sees it in the tool schema.
FALLBACK_TO_SECOND_POCKET = False
# Min atom-to-atom distance (Angstroms) for a docked pose to count as "near" a
# co-crystallized ligand. Mirrors NEARBY_DISTANCE_CUTOFF in modrag_protein_functions
# so "near a known ligand" means the same here and in the post-hoc check tool.
NEAR_LIGAND_CUTOFF = 5.0
# Cubic box edge length (Angstroms) used by dock_at_centroid / the agent-facing
# dock_at_centroid_agent. A module global -- NOT an agent-facing kwarg -- on
# purpose: the LLM tool-caller would otherwise hallucinate the optional
# parameter (see the agent-tool-signature gotcha). 22 A is a bit tighter than
# blind_dock's 28 A blind box because the site is now known, but still large
# enough to explore around the chosen centroid.
CENTROID_DOCK_BOX = 22.0


class DockError(Exception):
    """Recoverable docking failure (bad SMILES, Vina crash, obabel error, ...).

    Raised by the helpers instead of sys.exit so the agent-facing blind_dock()
    can fail one molecule and continue with the rest. The CLI main() catches
    this and converts it to a clean sys.exit with the message.
    """


# --- tool discovery --------------------------------------------------------

def find_vina_bin(explicit=None):
    """Locate the Vina binary: an explicit path, else dockstring's vendored copy."""
    if explicit:
        if not os.path.exists(explicit):
            raise DockError(f"--vina-bin not found: {explicit}")
        return explicit
    try:
        import dockstring
    except ImportError:
        raise DockError("could not import dockstring to locate the vendored "
                        "Vina binary; pass --vina-bin /path/to/vina explicitly.")
    candidates = [
        os.path.join(os.path.dirname(dockstring.__file__), "resources", "bin", "vina_mac_catalina"),
        os.path.join(os.path.dirname(dockstring.__file__), "resources", "bin", "vina_linux"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise DockError(f"no vendored Vina binary found under {os.path.dirname(dockstring.__file__)}")


def require(tool):
    if shutil.which(tool) is None:
        raise DockError(f"required tool '{tool}' not found on PATH.")


# --- prep steps ------------------------------------------------------------

def convert(cmd, label):
    """Run an obabel conversion, raising DockError on failure."""
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise DockError(f"{label} failed (obabel exit {res.returncode})\n"
                        f"  cmd: {' '.join(cmd)}\n  stderr: {res.stderr.strip()}")
    return res.stdout.strip()


def build_ligand_3d(smiles, sdf_path):
    """SMILES -> 3D conformer (RDKit ETKDG + MMFF) -> SDF.

    Shared by both the Open Babel ligand prep (build_ligand_pdbqt) and the
    meeko ligand prep used by the AD4Zn metal path (meeko wants an SDF with
    explicit hydrogens).
    """
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


def build_ligand_pdbqt(smiles, sdf_path, pdbqt_path):
    """SMILES -> 3D conformer (RDKit ETKDG + MMFF) -> SDF -> PDBQT (Open Babel)."""
    build_ligand_3d(smiles, sdf_path)
    convert(["obabel", "-isdf", sdf_path, "-opdbqt", "-O", pdbqt_path],
            f"ligand SDF->PDBQT ({smiles})")
    n = sum(1 for _ in open(pdbqt_path))
    if not any(l.startswith(("ROOT", "BRANCH")) for l in open(pdbqt_path)):
        print(f"vina_dock: WARNING — ligand PDBQT has no torsion-tree records "
              f"({n} lines); Vina may treat it as rigid.", file=sys.stderr)
    return n


def build_receptor_pdbqt(pdb_path, pdbqt_path):
    """receptor PDB -> rigid PDBQT (Open Babel).

    -h adds hydrogens (Vina only uses polar H, but extra H is harmless);
    -xr writes a RIGID molecule (no ROOT/BRANCH torsion tree). Without -xr,
    Open Babel writes the receptor as a flexible ligand with ~hundreds of
    BRANCH records, which Vina 1.1.2 rejects with "Unknown or inappropriate
    tag" (torsion-tree records are only valid in the --flex side-chain file).
    """
    convert(["obabel", "-ipdb", pdb_path, "-h", "-opdbqt", "-xr", "-O", pdbqt_path],
            "receptor PDB->PDBQT")
    n = sum(1 for l in open(pdbqt_path) if l.startswith("ATOM"))
    if any(l.startswith(("ROOT", "BRANCH")) for l in open(pdbqt_path)):
        print("vina_dock: WARNING — receptor PDBQT still has torsion-tree records; "
              "Vina may fail to parse it.", file=sys.stderr)
    return n


# --- blind pocket detection -----------------------------------------------

def parse_pdb_heavy_atoms(pdb_path):
    """Return (N,3) array of receptor heavy-atom coordinates (ATOM records, no H).

    HETATM (waters/ions/cofactors) are excluded so they don't fill in cavities.
    Element is read from columns 77-78, falling back to the atom name.
    """
    import numpy as np
    coords = []
    for line in open(pdb_path):
        if not line.startswith("ATOM"):
            continue
        el = line[76:78].strip()
        if not el:
            name = line[12:16].strip()
            el = next((c for c in name if c.isalpha()), "C")
        if el.upper() == "H":
            continue
        try:
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
    if not coords:
        raise DockError(f"no ATOM heavy-atom coordinates parsed from {pdb_path}")
    return np.array(coords)


def find_pockets(pdb_path, spacing=1.0, r_shell=8.0, d_min=2.5, d_max=8.0,
                 top_frac=0.06, eps=2.5, min_samples=25):
    """Detect putative binding pockets via a buriedness grid scan (no external tool).

    Idea: grid the receptor bounding box; for each empty-but-near-protein voxel
    (nearest-atom distance in [d_min, d_max]) score "buriedness" = count of
    receptor atoms within r_shell. Keep the top `top_frac` by buriedness, cluster
    them with DBSCAN, and rank clusters by buriedness * log1p(n_voxels).

    Two failure modes shaped these defaults:
      (1) Mean buriedness alone favors tiny deep crevices (a 4-voxel pit beats a
          400-voxel cavity), so the true site — a sizable cleft with only
          *moderate* buriedness — ranked #6-#32 on the DUD-E set and was only
          captured because the 28 A boxes happened to overlap it. Multiplying by
          log1p(cluster size) rewards substantial cavities while damping size
          enough that giant flat surface patches don't dominate (log << sqrt).
      (2) A permissive top_frac (0.10) keeps enough moderate-buriedness voxels to
          define the true cleft, but also lets diffuse near-surface voxels
          survive, which DBSCAN then merges into one giant blob (10k+ voxels)
          whose center is far from any binding site and drowns the true pocket.
          Raising min_samples to ~25 fragments that low-density surface blob
          (diffuse regions can't sustain 25 neighbours in a 2.5 A ball) while
          the dense, focused true pockets remain intact.

    Validated on the five DUD-E receptors (HMGCR/ADRB1/ADRB2/MAOB/DRD2) against
    the dockstring box centers: with top_frac=0.06, min_samples=25 and the
    b*log1p(n) score, the true site is the #1 pocket on ALL FIVE, with the
    detected center 3-9 A from the dockstring center. The all-#1 result holds
    across a robust plateau (top_frac in [0.04, 0.08], min_samples in [20, 40]).

    Cross-validated on SULT1A3 (PDB 2A3R, a homodimer crystallised with dopamine
    + PAP, not used to tune these params): the two substrate pockets are detected
    as the top-2 pockets (each ~7.5 A from a bound-dopamine centroid), and a blind
    dock of dopamine lands in the correct active site within ~5 A (atom-atom) of
    the crystallographic dopamine.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from sklearn.cluster import DBSCAN

    coords = parse_pdb_heavy_atoms(pdb_path)
    if len(coords) < 100:
        raise DockError(f"--blind: only {len(coords)} heavy atoms; "
                        "pocket detection is unreliable on such a small receptor.")
    tree = cKDTree(coords)
    lo = coords.min(axis=0) - 2.0
    hi = coords.max(axis=0) + 2.0
    axes = [np.arange(lo[i], hi[i] + spacing, spacing) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    grid = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    d_near, _ = tree.query(grid, k=1)
    buried = tree.query_ball_point(grid, r=r_shell, return_length=True)

    band = (d_near >= d_min) & (d_near <= d_max)
    surv = grid[band]
    surv_buried = buried[band]
    if len(surv) < 50:
        raise DockError("--blind: no near-protein empty voxels found; "
                        "check the receptor or widen --blind-band.")

    thr = np.quantile(surv_buried, 1 - top_frac)
    keep = np.where(surv_buried >= thr)[0]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(surv[keep]).labels_
    pockets = []
    for lab in set(labels):
        if lab < 0:
            continue
        idx = np.where(labels == lab)[0]
        gi = keep[idx]
        pts = surv[gi]
        mean_buried = float(surv_buried[gi].mean())
        n = int(len(gi))
        pockets.append({
            "center": pts.mean(axis=0),
            "n_voxels": n,
            "buriedness": mean_buried,
            "score": mean_buried * np.log1p(n),  # depth * log(1 + volume)
        })
    pockets.sort(key=lambda p: p["score"], reverse=True)
    return pockets


# --- run + parse -----------------------------------------------------------

def run_vina(vina_bin, rec_pdbqt, lig_pdbqt, center, size, args, out_pdbqt, log_path,
             no_log=False):
    """Run standard-receptor Vina and parse-ready-log it.

    no_log=True is for Vina >= 1.2 binaries, which dropped the --log flag:
    the mode table arrives on stdout and is written to log_path instead
    (same format, so parse_vina_log works for both).
    """
    cmd = [
        vina_bin,
        "--receptor", rec_pdbqt,
        "--ligand", lig_pdbqt,
        "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
        "--size_x", str(size[0]), "--size_y", str(size[1]), "--size_z", str(size[2]),
        "--out", out_pdbqt,
        "--exhaustiveness", str(args.exhaustiveness),
        "--num_modes", str(args.num_modes),
    ]
    if not no_log:
        cmd += ["--log", log_path]
    if args.cpu:
        cmd += ["--cpu", str(args.cpu)]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    print(f"vina_dock: running Vina...\n  {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise DockError(f"Vina failed (exit {res.returncode})\n"
                        f"  stderr: {res.stderr.strip()}\n  stdout: {res.stdout.strip()}")
    if no_log:
        with open(log_path, "w") as fh:
            fh.write(res.stdout)
    if res.stderr.strip():
        # Vina prints progress/notes on stderr; surface it for transparency.
        print(f"vina_dock: Vina stderr:\n{res.stderr.strip()}")
    return res.stdout


def parse_vina_log(log_path):
    """Return (best_affinity, n_modes) from a Vina log, or (None, 0) if unparseable.

    Vina 1.1.2 writes a mode table whose data rows look like:
        1        -8.1      0.000      0.000
    The first such row is the best pose.
    """
    if not os.path.exists(log_path):
        return None, 0
    row_re = re.compile(r"^\s*\d+\s+(-?\d+\.\d+)\s+")
    affinities = []
    with open(log_path) as fh:
        for line in fh:
            m = row_re.match(line)
            if m:
                affinities.append(float(m.group(1)))
    if not affinities:
        return None, 0
    return affinities[0], len(affinities)


# --- proximity fallback helpers -------------------------------------------
# These mirror the HETNAM/HETATM scan in
# modrag_protein_functions.check_nearby_molecules so that "near a known ligand"
# means the same thing inside blind_dock (the fallback gate) and in the
# separate post-hoc check tool the agent calls afterward.

def _crystal_ligand_coords(pdb_path):
    """Return a list of coordinate lists, one per co-crystallized ligand
    occurrence in the PDB, or [] if the structure has no such ligand.

    Only ligands with a HETNAM record are kept, so waters and ions (which lack
    HETNAM entries) are excluded -- matching check_nearby_molecules. Each
    (symbol, chain) occurrence is its own entry.
    """
    hetnam = set()
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("HETNAM"):
                sym = line[11:15].strip()
                if sym:
                    hetnam.add(sym)
    if not hetnam:
        return []
    occs = []
    cur = None
    cur_key = None
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            sym = line[17:20].strip()
            if sym not in hetnam:
                continue
            chain = line[21].strip()
            key = (sym, chain)
            if key != cur_key:
                cur_key = key
                cur = []
                occs.append(cur)
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                cur.append((x, y, z))
            except ValueError:
                continue
    return [c for c in occs if c]


def _pose_min_distance(pdbqt_path, crystal_occs):
    """Min atom-to-atom distance between the FIRST (best) pose in a Vina poses
    PDBQT and any co-crystallized ligand occurrence. None if no atoms read.

    Only the first MODEL is read so the gate reflects the single best pose we
    would actually keep (the SDF stores the top poses), matching
    check_nearby_molecules which only inspects the first SDF conformer.
    """
    if not crystal_occs:
        return None
    pose = []
    seen_model = False
    with open(pdbqt_path) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if line.startswith("MODEL"):
                if seen_model:
                    break  # stop before the second model
                seen_model = True
                continue
            if line.startswith(("ATOM", "HETATM")):
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    pose.append((x, y, z))
                except ValueError:
                    continue
    if not pose:
        return None
    best = None
    for lx, ly, lz in pose:
        for occ in crystal_occs:
            for mx, my, mz in occ:
                d = math.sqrt((lx - mx) ** 2 + (ly - my) ** 2 + (lz - mz) ** 2)
                if best is None or d < best:
                    best = d
    return best


# --- agent-facing blind-docking API ----------------------------------------

def blind_dock(receptor_pdb, smiles_list, npockets=1, exhaustiveness=8,
               num_modes=3, seed=0, cpu=0,
               blind_box=28.0, blind_spacing=1.0, blind_r=8.0,
               blind_band=(2.5, 8.0), blind_top_frac=0.06,
               blind_min_samples=25, blind_eps=2.5,
               overwrite_receptor=False, verbose=False):
    """Blind-dock a list of ligands into a receptor of unknown binding site.

    A standalone, side-effect-explicit tool for an agent: it detects putative
    binding pockets with find_pockets() (once, reused for every ligand), docks
    each SMILES into the top `npockets` pockets with Vina, and returns a
    multi-line report. The true binding site is the #1 pocket on the validated
    DUD-E set + SULT1A3, so npockets=1 (the default) is fast and usually enough;
    raise it to 3 for a safety net on a novel receptor.

    Persisted artefacts (next to the input PDB):
      - <stem>.pdbqt      rigid receptor (built once, reused across calls unless
                          overwrite_receptor=True)
      - <stem>_<i>.sdf    top-`num_modes` poses (capped to 3) for molecule i,
                          from that molecule's best-scoring pocket
    Per-run intermediates (ligand PDBQT, per-pocket poses/logs) go to a temp
    dir that is cleaned up at the end.

    Args:
        receptor_pdb: path to a receptor PDB file (binding site unknown).
        smiles_list: iterable of ligand SMILES strings.
        npockets: number of top-ranked pockets to dock each ligand into (1).
        exhaustiveness, num_modes, seed, cpu: Vina params. num_modes caps the
            poses written to each SDF at 3 (Vina outputs num_modes poses; the
            SDF keeps the top 3).
        blind_*: forwarded to find_pockets() (defaults are the validated ones).
        overwrite_receptor: rebuild <stem>.pdbqt even if it already exists.
        verbose: if True, surface Vina/obabel progress on stdout/stderr;
            default False (quiet; failures are captured in the report).

    Returns:
        A multi-line string report (header with receptor + receptor-PDBQT path
        + pocket centers, one block per molecule with score + pocket used +
        pose-SDF path, and an overall best-molecule line). Failed molecules are
        marked and do not abort the rest.

    Raises:
        DockError only for setup failures that affect every molecule (receptor
        missing, obabel/Vina not found, no pockets detected). Per-molecule
        failures are caught and reported, not raised.
    """
    print("vina_dock: blind docking...")
    print('==============================================')

    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]

    if not os.path.exists(receptor_pdb):
        raise DockError(f"receptor not found: {receptor_pdb}")
    require("obabel")
    vina_bin = find_vina_bin(None)  # always use dockstring's vendored copy
    receptor_pdb = os.path.abspath(receptor_pdb)
    stem = os.path.splitext(receptor_pdb)[0]
    rec_pdbqt = stem + ".pdbqt"

    # Quiet mode: swallow the helpers' progress/warning prints so the returned
    # report is the sole, clean output. Failures still surface via DockError
    # messages (convert/run_vina embed the relevant stderr).
    with contextlib.ExitStack() as stack:
        if not verbose:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            # RDKit writes SMILES-parse warnings to C-level stderr (fd 2), which
            # bypasses redirect_stderr. Mute its app logger for the call so a
            # bad SMILES is reported only via the returned DockError message.
            try:
                from rdkit import RDLogger
                RDLogger.DisableLog("rdApp.*")
                stack.callback(RDLogger.EnableLog, "rdApp.*")
            except Exception:
                pass

        # 1) rigid receptor PDBQT (build once, reuse across calls)
        if overwrite_receptor or not os.path.exists(rec_pdbqt):
            build_receptor_pdbqt(receptor_pdb, rec_pdbqt)

        # 2) detect pockets once, reuse for every ligand
        pockets = find_pockets(receptor_pdb, spacing=blind_spacing, r_shell=blind_r,
                               d_min=blind_band[0], d_max=blind_band[1],
                               top_frac=blind_top_frac, eps=blind_eps,
                               min_samples=blind_min_samples)
        if not pockets:
            raise DockError("no pockets detected in receptor")
        if FALLBACK_TO_SECOND_POCKET and npockets < 2:
            # need >=2 pockets docked so the off-target fallback has somewhere
            # to fall back to; if fewer than 2 are detected, fallback is a no-op
            npockets = 2
        pockets = pockets[:max(1, npockets)]
        # Co-crystallized ligand coords (HETNAM-keyed HETATM) for the proximity
        # fallback; empty if the structure has no known ligand to compare to,
        # in which case the fallback silently degrades to best-score selection.
        crystal_occs = (_crystal_ligand_coords(receptor_pdb)
                        if FALLBACK_TO_SECOND_POCKET else [])
        box_size = [blind_box] * 3

        # 3) dock each ligand into the top-N pockets, keep best score
        work = tempfile.mkdtemp(prefix="blind_dock_")
        vina_args = SimpleNamespace(exhaustiveness=exhaustiveness,
                                    num_modes=max(num_modes, 3), cpu=cpu, seed=seed)
        n_out_poses = min(3, vina_args.num_modes)
        results = []  # (idx, smi, affinity, best_pocket, sdf_path, status, detail)
        for idx, smi in enumerate(smiles_list):
            sdf_path = f"{stem}_{idx}.sdf"
            try:
                lig_sdf = os.path.join(work, f"ligand_{idx}.sdf")
                lig_pdbqt = os.path.join(work, f"ligand_{idx}.pdbqt")
                build_ligand_pdbqt(smi, lig_sdf, lig_pdbqt)
                pocket_res = []
                for j, p in enumerate(pockets):
                    poses_pdbqt = os.path.join(work, f"poses_{idx}_p{j+1}.pdbqt")
                    log_path = os.path.join(work, f"vina_{idx}_p{j+1}.log")
                    run_vina(vina_bin, rec_pdbqt, lig_pdbqt, p["center"],
                             box_size, vina_args, poses_pdbqt, log_path)
                    aff, nmodes = parse_vina_log(log_path)
                    pocket_res.append((j + 1, aff, nmodes, poses_pdbqt))
                valid = [r for r in pocket_res if r[1] is not None]
                if not valid:
                    results.append((idx, smi, None, None, sdf_path,
                                    "failed", "no score parsed from any Vina log"))
                    continue
                # primary = best Vina affinity (the pre-fallback choice)
                primary = min(valid, key=lambda r: r[1])
                chosen = primary
                detail = f"{primary[2]} modes"
                if (FALLBACK_TO_SECOND_POCKET and crystal_occs
                        and len(valid) > 1):
                    pd = _pose_min_distance(primary[3], crystal_occs)
                    if pd is None or pd > NEAR_LIGAND_CUTOFF:
                        # best-score pocket is off-target; look for a pocket
                        # whose pose IS near a co-crystallized ligand, and prefer
                        # the closest (proximity over Vina score).
                        near = []
                        for r in valid:
                            if r is primary:
                                continue
                            d = _pose_min_distance(r[3], crystal_occs)
                            if d is not None and d <= NEAR_LIGAND_CUTOFF:
                                near.append((d, r))
                        if near:
                            near.sort(key=lambda x: x[0])
                            chosen = near[0][1]
                            pd_s = f"{pd:.1f} A" if pd is not None else "n/a"
                            detail = (f"{chosen[2]} modes; switched from "
                                      f"pocket #{primary[0]} (off-target, {pd_s} "
                                      f"from crystal ligand) to pocket "
                                      f"#{chosen[0]} (near, {near[0][0]:.1f} A)")
                cj, caff, cnm, cpp = chosen
                convert(["obabel", "-ipdbqt", cpp, "-osdf", "-O", sdf_path,
                         "-l", str(n_out_poses)],
                        f"pose PDBQT->SDF (mol {idx}, top {n_out_poses})")
                results.append((idx, smi, caff, cj, sdf_path, "ok", detail))
            except DockError as e:
                results.append((idx, smi, None, None, sdf_path, "failed", str(e)))
            except Exception as e:  # defensive: never let one molecule kill the run
                results.append((idx, smi, None, None, sdf_path, "error",
                                f"{type(e).__name__}: {e}"))

    shutil.rmtree(work, ignore_errors=True)

    # 4) build the report
    lines = []
    lines.append("Blind docking report")
    lines.append(f"  receptor:     {receptor_pdb}")
    lines.append(f"  receptor PDBQT: {rec_pdbqt}")
    lines.append(f"  vina binary: {vina_bin}")
    if FALLBACK_TO_SECOND_POCKET:
        if crystal_occs:
            lines.append(f"  proximity fallback: enabled "
                         f"({len(crystal_occs)} crystal ligand occurrence(s), "
                         f"cutoff {NEAR_LIGAND_CUTOFF:.1f} A)")
        else:
            lines.append("  proximity fallback: enabled but no co-crystallized "
                         "ligand found; falling back to best score")
    lines.append(f"  pockets docked (top {len(pockets)}):")
    for i, p in enumerate(pockets):
        c = p["center"]
        lines.append(f"    #{i+1} ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) "
                     f"voxels={p['n_voxels']} buriedness={p['buriedness']:.0f} "
                     f"score={p['score']:.0f}")
    lines.append("")
    lines.append(f"Molecules ({len(results)}):")
    for idx, smi, aff, pj, sdf, status, detail in results:
        lines.append(f"  [{idx}] {smi}")
        if status == "ok":
            lines.append(f"        score: {aff:.2f} kcal/mol   (best pocket #{pj}, {detail})")
            lines.append(f"        poses SDF (top {n_out_poses}): {sdf}")
        else:
            lines.append(f"        {status}: {detail}")
    lines.append("")
    ok = [r for r in results if r[5] == "ok"]
    if ok:
        bidx, bsmi, baff, bpj, bsdf, _, _ = min(ok, key=lambda r: r[2])
        lines.append(f"Best molecule: [{bidx}] {bsmi}  "
                     f"score={baff:.2f} kcal/mol  SDF={bsdf}")
    else:
        lines.append("Best molecule: none docked successfully")
    return "\n".join(lines)


def blind_dock_agent(receptor_pdb: str, smiles_list: list[str]) -> str:
    """
    Dock ligands in a protein using ligand smiles and a receptor PDB file.

    Args:
        receptor_pdb: path to a receptor PDB file (binding site unknown).
        smiles_list: iterable list of ligand SMILES strings.
    Returns:
        A multi-line string report (header with receptor + receptor-PDBQT path
        + pocket centers, one block per molecule with score
    """
    return blind_dock(receptor_pdb, smiles_list, npockets=1)


def dock_at_centroid(receptor_pdb, smiles_list, center, box_size=22.0,
                     exhaustiveness=8, num_modes=3, seed=0, cpu=0,
                     overwrite_receptor=False, verbose=False):
    """Dock a list of ligands into a receptor at a KNOWN box center.

    Unlike blind_dock (which detects pockets), this docks into a single box
    centered on `center` -- e.g. the centroid of a co-crystallized ligand
    returned by modrag_protein_functions.find_ligand_centroids, or any other
    user-supplied site. Use this when the binding site is already known.

    Persisted artefacts (next to the input PDB):
      - <stem>.pdbqt      rigid receptor (built once, reused across calls
                          unless overwrite_receptor=True)
      - <stem>_c<i>.sdf   top poses for molecule i ("c" = centroid dock, to
                          avoid clobbering <stem>_<i>.sdf from blind_dock)
    Per-run intermediates (ligand PDBQT, poses/logs) go to a temp dir that is
    cleaned up at the end.

    Args:
        receptor_pdb: path to a receptor PDB file.
        smiles_list: iterable of ligand SMILES strings.
        center: (x, y, z) box center in Angstroms.
        box_size: cubic box edge length in Angstroms (22); the box is the same
            size on all three axes.
        exhaustiveness, num_modes, seed, cpu: Vina params. num_modes caps the
            poses written to each SDF at 3 (Vina outputs num_modes poses; the
            SDF keeps the top 3).
        overwrite_receptor: rebuild <stem>.pdbqt even if it already exists.
        verbose: if True, surface Vina/obabel progress on stdout/stderr;
            default False (quiet; failures are captured in the report).

    Returns:
        A multi-line string report (header with receptor + center + box +
        receptor-PDBQT path, one block per molecule with score + pose-SDF
        path, and an overall best-molecule line). Failed molecules are
        marked and do not abort the rest.

    Raises:
        DockError only for setup failures that affect every molecule (receptor
        missing, obabel/Vina not found). Per-molecule failures are caught and
        reported, not raised.
    """
    print("vina_dock: docking at centroid...")
    print('==============================================')

    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]

    if not os.path.exists(receptor_pdb):
        raise DockError(f"receptor not found: {receptor_pdb}")
    require("obabel")
    vina_bin = find_vina_bin(None)  # always use dockstring's vendored copy
    receptor_pdb = os.path.abspath(receptor_pdb)
    stem = os.path.splitext(receptor_pdb)[0]
    rec_pdbqt = stem + ".pdbqt"
    center = [float(c) for c in center]
    box = [float(box_size)] * 3

    # Quiet mode: swallow the helpers' progress/warning prints so the returned
    # report is the sole, clean output (same approach as blind_dock).
    with contextlib.ExitStack() as stack:
        if not verbose:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            try:
                from rdkit import RDLogger
                RDLogger.DisableLog("rdApp.*")
                stack.callback(RDLogger.EnableLog, "rdApp.*")
            except Exception:
                pass

        # 1) rigid receptor PDBQT (build once, reuse across calls)
        if overwrite_receptor or not os.path.exists(rec_pdbqt):
            build_receptor_pdbqt(receptor_pdb, rec_pdbqt)

        # 2) dock each ligand into the single given box, keep the score
        work = tempfile.mkdtemp(prefix="dock_at_centroid_")
        vina_args = SimpleNamespace(exhaustiveness=exhaustiveness,
                                    num_modes=max(num_modes, 3), cpu=cpu, seed=seed)
        n_out_poses = min(3, vina_args.num_modes)
        results = []  # (idx, smi, affinity, sdf_path, status, detail)
        for idx, smi in enumerate(smiles_list):
            sdf_path = f"{stem}_c{idx}.sdf"
            try:
                lig_sdf = os.path.join(work, f"ligand_{idx}.sdf")
                lig_pdbqt = os.path.join(work, f"ligand_{idx}.pdbqt")
                build_ligand_pdbqt(smi, lig_sdf, lig_pdbqt)
                poses_pdbqt = os.path.join(work, f"poses_{idx}.pdbqt")
                log_path = os.path.join(work, f"vina_{idx}.log")
                run_vina(vina_bin, rec_pdbqt, lig_pdbqt, center, box,
                         vina_args, poses_pdbqt, log_path)
                aff, nmodes = parse_vina_log(log_path)
                if aff is None:
                    results.append((idx, smi, None, sdf_path, "failed",
                                    "no score parsed from Vina log"))
                    continue
                convert(["obabel", "-ipdbqt", poses_pdbqt, "-osdf", "-O", sdf_path,
                         "-l", str(n_out_poses)],
                        f"pose PDBQT->SDF (mol {idx}, top {n_out_poses})")
                results.append((idx, smi, aff, sdf_path, "ok", f"{nmodes} modes"))
            except DockError as e:
                results.append((idx, smi, None, sdf_path, "failed", str(e)))
            except Exception as e:  # defensive: never let one molecule kill the run
                results.append((idx, smi, None, sdf_path, "error",
                                f"{type(e).__name__}: {e}"))

    shutil.rmtree(work, ignore_errors=True)

    # 3) build the report
    lines = []
    lines.append("Centroid docking report")
    lines.append(f"  receptor:         {receptor_pdb}")
    lines.append(f"  receptor PDBQT:   {rec_pdbqt}")
    lines.append(f"  vina binary:      {vina_bin}")
    lines.append(f"  box center:       ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    lines.append(f"  box size:         {box[0]:.1f} A (cubic)")
    lines.append("")
    lines.append(f"Molecules ({len(results)}):")
    for idx, smi, aff, sdf, status, detail in results:
        lines.append(f"  [{idx}] {smi}")
        if status == "ok":
            lines.append(f"        score: {aff:.2f} kcal/mol   ({detail})")
            lines.append(f"        poses SDF (top {n_out_poses}): {sdf}")
        else:
            lines.append(f"        {status}: {detail}")
    lines.append("")
    ok = [r for r in results if r[4] == "ok"]
    if ok:
        bidx, bsmi, baff, bsdf, _, _ = min(ok, key=lambda r: r[2])
        lines.append(f"Best molecule: [{bidx}] {bsmi}  "
                     f"score={baff:.2f} kcal/mol  SDF={bsdf}")
    else:
        lines.append("Best molecule: none docked successfully")
    return "\n".join(lines)


def dock_at_centroid_agent(receptor_pdb: str, smiles_list: list[str],
                           center_x: float, center_y: float, center_z: float) -> str:
    """
    Dock ligands into a protein at a KNOWN binding-site center (e.g. the
    centroid of a co-crystallized ligand returned by find_ligand_centroids,
    or any chosen x,y,z). Use this instead of blind_dock_agent when the
    binding site / a reference ligand centroid is already known.

    Args:
        receptor_pdb: path to a receptor PDB file.
        smiles_list: iterable list of ligand SMILES strings.
        center_x: box-center X coordinate in Angstroms.
        center_y: box-center Y coordinate in Angstroms.
        center_z: box-center Z coordinate in Angstroms.
    Returns:
        A multi-line string report (header with receptor + center + box +
        one block per molecule with score and pose-SDF path).
    """
    return dock_at_centroid(receptor_pdb, smiles_list,
                            center=(center_x, center_y, center_z),
                            box_size=CENTROID_DOCK_BOX)


# --- metal-site docking ------------------------------------------------------
# Zinc docking via the AutoDock4Zn force field (AD4Zn), plus an approximate
# metal-kept-in-receptor path for the remaining biologically common metal ions.
#
# How the engine is chosen (decided in code, never exposed to the LLM -- see
# the agent-tool-signature gotcha comment near FALLBACK_TO_SECOND_POCKET):
#   Zn2+  -> full AD4Zn pipeline, validated end-to-end on PDB 1s63 where an
#            inhibitor redocks at -12.2 kcal/mol with its chelating N at 1.91 A
#            from the zinc (tutorial reference: ~-13.5). The zinc chemistry
#            lives in tz-pseudo-atoms + calibrated AutoGrid pair terms
#            (Santos-Martins et al., JCIM 2014), NOT in the Vina scoring pass,
#            which reads the maps via `--scoring ad4`.
#   Mg/Ca/Mn/Fe -> tier 0 "ad4map": same AutoGrid-maps machinery, but the
#            receptor keeps no TZ atoms and the metal is typed from the
#            stock AD4Zn.dat parameters (no new parameters). Ions are
#            appended to the protein-only receptor by hand with formal
#            charge (+2) -- obabel types Cu as C / Na+ as NA and meeko only
#            templates Zn, so neither prep tool can do this step.
#   Cu/Co/Ni/Cd/Hg/Na/K -> tier 1 "ad4merz": same maps machinery with
#            AD4Merz.dat (Li & Merz 12-6-4 TIP3P self-parameters, JCTC 2014;
#            Joung & Cheatham monovalent, JPCB 2008). Without the r^-4 term
#            or directional fitting these contacts run long; like tier 0
#            this is calibrated-free and the coordination check is honest.
#   other -> plain Vina with the metal ion kept in the receptor. Vina treats a
#            metal atom as a generic vdW sphere (measured: a C and an N ligand
#            atom both settle ~3.0 A from a Zn), so scores are APPROXIMATE;
#            the coordination post-check on each pose is the honest readout.
#
# External tools (all user-space, no system installer):
#   conda env "adfr-suite" (osx-64, Rosetta): pythonsh + autogrid4
#   (4.2.7.x.2019-07-11 -- the exact version the AD4Zn tutorial requires;
#   its libgomp was patched in-place against the env's own lib/ copy).
#   conda env "vina" (native arm64): vina 1.2.7 CLI (--scoring ad4 + --maps
#   are NOT supported by the dockstring-vendored Vina 1.1.2).
#   dock-env: meeko (mk_prepare_* receptor/ligand PDBQTs for the AD4Zn path).
#   dock_assist/ad4zn/: zinc_pseudo.py, prepare_gpf4zn.py, AD4Zn.dat fetched
#   from the official AutoDock-Vina repo; AD4Merz.dat generated in-repo from
#   AD4Zn.dat (literature metal parameters, see its header) for tier 1.
METAL_ELEMENTS = {"ZN", "MG", "CA", "FE", "MN", "CU", "CO", "NI", "HG", "CD",
                  "NA", "K"}
ADFR_PYTHONSH = os.path.expanduser(os.environ.get(
    "ADFR_PYTHONSH", "~/miniforge3/envs/adfr-suite/bin/pythonsh"))
AUTOGRID_BIN = os.path.expanduser(os.environ.get(
    "AUTOGRID_BIN", "~/miniforge3/envs/adfr-suite/bin/autogrid4"))
VINA_12_BIN = os.path.expanduser(os.environ.get(
    "VINA_12_BIN", "~/miniforge3/envs/vina/bin/vina"))
AD4ZN_DIR = os.path.expanduser(os.environ.get(
    "AD4ZN_DIR", os.path.join(HERE, "..", "ad4zn")))
AD4ZN_DAT = os.path.join(AD4ZN_DIR, "AD4Zn.dat")
AD4MERZ_DAT = os.path.join(AD4ZN_DIR, "AD4Merz.dat")
# Tiered parameter coverage for the non-zinc metals (engine decided in code):
#   Tier 0 (ad4map):  metals that stock AD4Zn.dat already parameterizes --
#   Mg, Ca, Mn, Fe -- docked through AutoGrid maps (metal LJ + charge +
#   solvation from the .dat) instead of Vina's generic vdW sphere.
#   Tier 1 (ad4merz): metals ABSENT from stock AD4Zn.dat get literature
#   self-parameters folded into AD4Merz.dat (see its header): Li & Merz
#   12-6-4 set, TIP3P (Li & Merz, JCTC 2014) for Cu/Co/Ni/Cd/Hg, Joung &
#   Cheatham TIP3P (JPCB 2008) for Na/K. The r^-4 induced-dipole term is not
#   representable in AutoGrid pairwise form, so wells run shallower and
#   contacts longer than crystallographic coordination values (the AD4Zn
#   authors hit the same wall, which is why Zn gets directional TZ terms).
# [measured] AutoGrid's 12-6 pair well is NOT eps -- it is eps DIVIDED BY 6,
# verified with single-atom autogrid probes on both stock and nbp-override
# pairs: stock Zn-OA minimum -0.054 kcal/mol at R=(Rii_a+Rii_b)/2=2.34 A fits
# sqrt(0.550*0.200)/6=0.0553; stock C-OA minimum -0.0300 at 3.60 A fits
# sqrt(0.150*0.200)/6=0.0289; nbp_r_eps depth is exactly linear in the given
# eps (eps 1/4/16 -> depths 0.165/0.664/2.658 = eps/6) and lands the minimum
# exactly at the given r. So the Scripps-fitted zinc numbers are calibrated
# WITH these semantics: nbp `OA Zn 2.1 3.8453` is a -0.64 well, and the deep
# zinc binding comes from the TZ terms (`NA TZ 0.25 23.2135` = a -3.87 well
# per donor at the pseudo-atom). The tier engines have no TZ atoms, so their
# donor site terms must carry the full pull themselves: NBP_DONOR_BASE holds
# the INTENDED effective well depths (the writer multiplies by 6, see
# _build_ad4map_gpf), anchored on zinc's effective OA-direct + TZ + electro-
# statics sum (~-5 to -6 per donor for Zn2+); literature 12-6-4 epsilons are
# absorbed into these effective terms. Not validated beyond the 1s63
# mutation redock, so the coordination post-check stays the honest readout.
AD4MAP_ELEMENTS = {"MG", "CA", "MN", "FE"}                    # tier 0 -> AD4Zn.dat
AD4MERZ_ELEMENTS = {"CU", "CO", "NI", "CD", "HG", "NA", "K"}  # tier 1 -> AD4Merz.dat
# PDBQT atom type (as obabel writes it) + formal charge for ions kept in
# map-path receptors. obabel types Mg/Fe/Cu/Na/... correctly but writes
# charge 0.000 (measured) and meeko only templates Zn, so the tier 0/1
# receptor is the metal-keeping obabel PDBQT with the charge column of its
# metal lines rewritten to the formal value AutoGrid's electrostatics map
# needs. Charge = usual biological oxidation state (Fe assumed Fe2+).
METAL_ATOM_TYPES = {
    "ZN": ("Zn", 2.0), "MG": ("Mg", 2.0), "CA": ("Ca", 2.0),
    "MN": ("Mn", 2.0), "FE": ("Fe", 2.0), "CU": ("Cu", 2.0),
    "CO": ("Co", 2.0), "NI": ("Ni", 2.0), "CD": ("Cd", 2.0),
    "HG": ("Hg", 2.0), "NA": ("Na", 1.0), "K":  ("K", 1.0),
}
# Crystal-chemical metal-donor contact distances (literature mean
# metal-oxygen coordination distances, one decimal) used as the pair minimum
# in the map engines' nbp_r_eps site terms (see comment above).
# Zn 2.10 is the Scripps zinc-fit value itself.
METAL_SITE_R = {
    "ZN": 2.10, "MG": 2.07, "CA": 2.35, "MN": 2.19, "FE": 2.10,
    "CU": 2.15, "CO": 2.09, "NI": 2.06, "CD": 2.28, "HG": 2.20,
    "NA": 2.36, "K":  2.81,
}
# (donor PDBQT type, contact offset from METAL_SITE_R, effective pair well)
# -- effective well = the depth the pair minimum should reach in kcal/mol;
# _build_ad4map_gpf writes nbp_r_eps eps = 6 x this (AutoGrid divides nbp eps
# by 6, [measured] see comment above). OA is anchored on the zinc fit's
# effective donor pull (~-3.85 at its 2.10 A contact); the others are
# softness-scaled estimates pending a redocking benchmark: S softer and
# longer, aromatic/nitrile N medium, alkyl-amine N slightly stronger than
# aromatic N. The charge-dependence of the metal pull is NOT folded into the
# wells: it arrives through the electrostatics map (Mg2+ doubles Na+'s
# potential), and the 12-6 dispersion/repulsion itself is nearly
# charge-independent. Monovalent wells therefore equal the divalent ones.
NBP_DONOR_BASE = (("OA", 0.00, 3.8453), ("SA", 0.15, 2.50),
                  ("NA", -0.10, 1.50), ("N",  -0.10, 2.00))
# Grid box edge (Angstroms) for metal-site docking, and the AutoGrid spacing.
# autoGrid's default 0.375 A spacing; npts derives from box/spacing.
METAL_DOCK_BOX = 22.0
AUTOGRID_SPACING = 0.375
# Snap radius when mapping a caller-supplied center to the nearest metal ion.
METAL_SNAP_CUTOFF = 3.0
# Distance (Angstroms) from a pose donor atom to the metal for the pose to
# count as coordinating: crystallographic Zn-N bonds are ~2.0-2.2 A, N/O
# second-shell contacts go to ~4 A. 2.5 is a coordination-bond read.
COORDINATION_CUTOFF = 2.5
# PDBQT atom types counted as coordination donors (polar N/O/S; HD excluded).
DONOR_PDBQT_TYPES = {"N", "NA", "NS", "NC", "O", "OA", "OS", "OQ", "S", "SA"}


def _run(cmd, label, cwd=None):
    """Run a subprocess, raising DockError on failure (generic convert())."""
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if res.returncode != 0:
        raise DockError(f"{label} failed (exit {res.returncode})\n"
                        f"  cmd: {' '.join(cmd)}\n  stderr: {res.stderr.strip()}")
    return res.stdout


def detect_metal_sites(pdb_path):
    """Find metal-ion sites in a receptor PDB with their coordinating residues.

    A pure-PDB scan: any HETATM whose element (or, as fallback, residue name)
    is in METAL_ELEMENTS is one site. Contacts are PDB atoms (ATOM records and
    other HETATMs) within 2.5 A of the ion -- typical direct coordination
    distances are ~2.0-2.4 A, so 2.5 keeps the first coordination sphere.

    Returns a list of dicts: {element, resname, chain, resseq, x, y, z,
    contacts} with contacts a list of "RESNAME chain+resseq" strings,
    sorted, one per coordinating residue.
    """
    if not os.path.exists(pdb_path):
        raise DockError(f"receptor not found: {pdb_path}")

    def _coord(line):
        return [float(line[30:38]), float(line[38:46]), float(line[46:54])]

    sites = {}  # (element, resname, chain, resseq) -> coords
    atoms = []  # (xyz, resname, chain, resseq, site_key) for contact search
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            resname = line[17:20].strip().upper()
            el = line[76:78].strip().upper() or resname
            key = (el, resname, line[21].strip(), line[22:26].strip())
            xyz = _coord(line)
            if (line.startswith("HETATM") and el in METAL_ELEMENTS
                    and resname != "HOH"):
                sites.setdefault(key, []).append(xyz)
                atoms.append((xyz, resname, key[2], key[3], key))
            else:
                atoms.append((xyz, resname, key[2], key[3], None))
    out = []
    for (el, resname, chain, resseq), xyzs in sorted(sites.items()):
        # ion center: single-atom ions give one xyz; a multi-atom metal residue
        # (rare) uses its atom-mean -- only for the reported/docked center.
        ox = oy = oz = 0.0
        for x, y, z in xyzs:
            ox += x; oy += y; oz += z
        center = [ox / len(xyzs), oy / len(xyzs), oz / len(xyzs)]
        contacts = set()
        for (axyz, ares, acha, aseq, akey) in atoms:
            if akey == (el, resname, chain, resseq):
                continue
            d = math.sqrt((axyz[0] - center[0]) ** 2 +
                          (axyz[1] - center[1]) ** 2 +
                          (axyz[2] - center[2]) ** 2)
            if d <= COORDINATION_CUTOFF:
                contacts.add(f"{ares} {acha}{aseq}")
        out.append({"element": el, "resname": resname, "chain": chain,
                    "resseq": resseq, "xyz": center,
                    "contacts": sorted(contacts)})
    return out


def detect_metal_sites_agent(receptor_pdb: str) -> str:
    """
    List the metal ions (Zn, Mg, Ca, Fe, Mn, Cu, ...) present in a receptor
    PDB file, with each ion's element, chain, residue number, x,y,z
    coordinates and its coordinating residues (contacts within 2.5 A). Call
    this on a receptor before dock_metal_site_agent: the returned (x, y, z)
    of the metal ion matching the user's site of interest is what to pass as
    the box center there.

    Args:
        receptor_pdb: path to a receptor PDB file.
    Returns:
        A report listing each metal site; "none" if the structure has no
        metal ions.
    """
    sites = detect_metal_sites(receptor_pdb)
    if not sites:
        return f"no metal ions found in {os.path.abspath(receptor_pdb)}"
    lines = [f"Metal sites in {os.path.abspath(receptor_pdb)}:"]
    for s in sites:
        lines.append(f"  {s['element']} ({s['resname']} {s['chain']}{s['resseq']}) "
                     f"at ({s['xyz'][0]:.3f}, {s['xyz'][1]:.3f}, {s['xyz'][2]:.3f})")
        lines.append(f"    coordinating residues: "
                     f"{', '.join(s['contacts']) if s['contacts'] else 'none within 2.5 A'}")
    return "\n".join(lines)


def _nearest_metal(sites, center, max_dist=METAL_SNAP_CUTOFF):
    """Nearest detected metal ion within `max_dist` of `center`, else None."""
    best = None
    for s in sites:
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(s["xyz"], center)))
        if best is None or d < best[0]:
            best = (d, s)
    return best[1] if best and best[0] <= max_dist else None


def build_receptor_pdbqt_with_metals(pdb_path, pdbqt_path):
    """Receptor PDB -> rigid PDBQT that KEEPS metal ions.

    Like build_receptor_pdbqt (obabel -h -xr) but the input is first filtered
    to keep metal-ion HETATMs while still dropping waters/cofactors/ligands,
    so a metal site is represented in the docking grid instead of emptied out.
    Used by the generic-metal path (non-zinc). Writes the filtered PDB next to
    the target .pdbqt and removes it afterwards.
    """
    filtered = pdbqt_path + ".filtered.pdb"
    n = 0
    with open(pdb_path) as src, open(filtered, "w") as dst:
        for line in src:
            if line.startswith("ATOM"):
                dst.write(line); n += 1
            elif line.startswith("HETATM"):
                el = line[76:78].strip().upper() or line[17:20].strip().upper()
                if el in METAL_ELEMENTS and line[17:20].strip().upper() != "HOH":
                    dst.write(line); n += 1
    try:
        convert(["obabel", "-ipdb", filtered, "-h", "-opdbqt", "-xr",
             "-O", pdbqt_path], "receptor (with metals) PDB->PDBQT")
    finally:
        try:
            os.remove(filtered)
        except OSError:
            pass
    return n


def _prepare_map_receptor(pdb_path, out_pdbqt):
    """Rigid receptor PDBQT with formal-charge metal ions (tier 0/1 paths).

    meeko only templates Zn (prepare fails on Cu/Na ions) and the obabel the
    pipeline resolves (openbabel-wheel 3.1.0) cannot produce a charged metal
    receptor either: default flags write charge 0.000 everywhere, and with
    --partialcharge gasteiger a molecule containing a metal ion converts to
    nothing ("0 molecules converted", exit 0, empty output; adfr-suite's
    2.4.1 charges by default but the tier paths must not depend on which
    obabel binary is first on PATH) -- both [measured]. So: meeko preps the
    protein (metal HETATMs filtered out first -- the same prep the validated
    zinc path uses, with real Gasteiger charges), and each metal ion is
    appended as a hand-built receptor line with its formal charge from
    METAL_ATOM_TYPES. AutoGrid's electrostatics map reads the PDBQT charge
    (a neutral ion would make the metal map vdW-only, and autogrid refuses
    an all-zero-charged receptor outright).
    """
    protein = out_pdbqt + ".proteinonly.pdb"
    with open(pdb_path) as src, open(protein, "w") as dst:
        for line in src:
            if line.startswith("ATOM"):
                dst.write(line)
    try:
        _meeko_receptor_pdbqt(protein, out_pdbqt[:-6])
        # drop meeko's stray side files, keep only the cached pdbqt
        out_root = out_pdbqt[:-6]
        for junk in glob.glob(out_root + "-*") + [out_root + ".pdb",
                                                  out_root + ".h-pdb.pdb"]:
            try:
                os.remove(junk)
            except OSError:
                pass
    finally:
        try:
            os.remove(protein)
        except OSError:
            pass
    lines = open(out_pdbqt).read().splitlines()
    if not lines:
        raise DockError(f"meeko produced an empty receptor from {pdb_path}")
    serial = max(int(l[6:11]) for l in lines
                 if l.startswith("ATOM") and l[6:11].strip())
    n_metals = 0
    with open(pdb_path) as src:
        for dl in src:
            if not dl.startswith("HETATM"):
                continue
            el = dl[76:78].strip().upper() or dl[17:20].strip().upper()
            if el not in METAL_ATOM_TYPES or dl[17:20].strip().upper() == "HOH":
                continue
            typ, chg = METAL_ATOM_TYPES[el]
            serial += 1
            # hand-built line, columns verified against meeko's Zn layout:
            # xyx 30:54, charge 70:76, type 77:79 (coordination scan reads
            # nothing; AutoGrid reads charge and type only)
            lines.append(
                f"ATOM  {serial:5d}  {typ:<2} "
                f" {typ:<2}  {dl[21]}{int(dl[22:26]):>4d}    "
                f"{float(dl[30:38]):8.3f}{float(dl[38:46]):8.3f}"
                f"{float(dl[46:54]):8.3f}{'1.00':>6}{'0.00':>6}    "
                f"{chg:+.3f} {typ}")
            n_metals += 1
    if not n_metals:
        raise DockError("no recognized metal ions left in prepared map "
                        f"receptor: {pdb_path}")
    with open(out_pdbqt, "w") as dst:
        dst.write("\n".join(lines) + "\n")
    return out_pdbqt


def _build_ad4map_gpf(rec_pdbqt, lig_pdbqt, gpf_path, center, box_edge,
                      params_name, metal_type):
    """Write an AutoGrid4 GPF for one ligand at one metal site.

    prepare_gpf4zn.py cannot be reused: it hardcodes parameter_file=AD4Zn.dat
    and appends the calibrated Zn/TZ nbp_r_eps lines regardless of metal. This
    instead re-creates the exact GPF structure autogrid+Vina accepted on the
    zinc path; the metal contributes via its atom_par entry (vdW, charge,
    desolvation vol/par) in `params_name`, which must sit in AutoGrid's cwd
    under that basename.

    On top of atom_par, donor site terms are emitted (NBP_DONOR_BASE /
    METAL_SITE_R): plain 12-6 metal-donor pairs are too weak to coordinate
    (see the tier comment block above), so the donor pair wells sit at each
    metal's crystal-chemical contact distance. NBP_DONOR_BASE holds the
    INTENDED effective well depth; AutoGrid's nbp pair well is nbp_eps/6
    [measured], so the written epsilon is 6 x the intended well. The
    charge-dependence of the pull is not in the wells -- it comes via the
    electrostatics map. `metal_type` is the case-exact
    PDBQT type obabel wrote for the ion (e.g. "Cu", "Na").
    """
    def _types(path):
        seen = []
        with open(path) as fh:
            for line in fh:
                if line.startswith(("ATOM", "HETATM")):
                    t = line[77:79].strip()
                    if t and t not in seen:
                        seen.append(t)
        return seen
    rec_types = _types(rec_pdbqt)
    lig_types = _types(lig_pdbqt)
    if not lig_types:
        raise DockError(f"no atom types parsed from ligand PDBQT: {lig_pdbqt}")
    base = os.path.basename(rec_pdbqt)[:-6]  # strip .pdbqt
    npts = int(round(box_edge / AUTOGRID_SPACING))
    lines = [
        f"npts {npts} {npts} {npts}                        # num.grid points in xyz",
        f"parameter_file {params_name}             # force field parameter file",
        f"gridfld {base}.maps.fld          # grid_data_file",
        f"spacing {AUTOGRID_SPACING}                        # spacing(A)",
        f"receptor_types {' '.join(rec_types)} # receptor atom types",
        f"ligand_types {' '.join(lig_types)} # ligand atom types",
        f"receptor {os.path.basename(rec_pdbqt)}            # macromolecule",
        f"gridcenter {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}           # xyz-coordinates",
        "smooth 0.5                           # store minimum energy w/in rad(A)",
    ]
    lines += [f"map {base}.{t}.map                 # atom-specific affinity map"
              for t in lig_types]
    lines += [
        f"elecmap {base}.e.map             # electrostatic potential map",
        f"dsolvmap {base}.d.map            # desolvation potential map",
        "dielectric -0.1465                   # <0, AD4 distance-dep.diel;>0, constant",
    ]
    mt = metal_type.upper()
    if mt not in METAL_SITE_R:
        raise DockError(f"no site terms for metal type {metal_type!r} "
                        f"({mt=}) -- add it to METAL_SITE_R")
    r_site = METAL_SITE_R[mt]
    for donor, dr, well in NBP_DONOR_BASE:
        if donor in lig_types:  # only emit pairs the ligand actually uses
            eps = 6 * well  # nbp pair well == nbp_eps/6 [measured]
            lines.append(f"nbp_r_eps {r_site + dr:.2f} {eps:.4f} "
                         f"12 6 {donor} {metal_type}")
    with open(gpf_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return gpf_path


def _meeko_script(name):
    """Path to a meeko CLI script: the running interpreter's bin dir first
    (works when dock-env/bin/python is invoked directly, where PATH may not
    include the venv), then PATH."""
    sib = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), name)
    if os.path.exists(sib):
        return sib
    s = shutil.which(name)
    return s


def _meeko_receptor_pdbqt(pdb_path, out_root):
    """PDB -> hydrogen-added, Gasteiger-charged receptor PDBQT via meeko.

    The AD4Zn tutorial's receptor prep (meeko handles residue-bond templates
    the way zinc_pseudo/prepare_gpf4zn expect). `out_root` is the output path
    WITHOUT extension; `-p` writes <out_root>.pdbqt.
    """
    script = _meeko_script("mk_prepare_receptor.py")
    if script is None:
        raise DockError("meeko not installed in this python env (no "
                        "mk_prepare_receptor.py); pip install meeko.")
    _run([sys.executable, script, "-i", pdb_path, "-o", out_root, "-p"],
         "meeko receptor prep")
    out = out_root + ".pdbqt"
    if not os.path.exists(out):
        raise DockError(f"meeko receptor prep produced no PDBQT: {out}\n"
                        f"  (meeko refuses receptors with unresolvable residues)")
    return out


def _meeko_ligand_pdbqt(sdf_path, pdbqt_path):
    """SDF (with explicit Hs) -> charged ligand PDBQT via meeko.

    The AD4Zn path preps the ligand with meeko rather than Open Babel, matching
    the tutorial: with --scoring ad4, Vina interpolates ligand Gasteiger charges
    against the electrostatics/desolvation maps, so charge typing matters.
    """
    script = _meeko_script("mk_prepare_ligand.py")
    if script is None:
        raise DockError("meeko not installed in this python env (no "
                        "mk_prepare_ligand.py); pip install meeko.")
    _run([sys.executable, script, "-i", sdf_path, "-o", pdbqt_path],
         "meeko ligand prep")
    if not os.path.exists(pdbqt_path):
        raise DockError(f"meeko ligand prep produced no PDBQT: {pdbqt_path}")


def _add_zinc_pseudos(receptor_pdbqt, out_pdbqt=None):
    """Add TZ (tetrahedral-zinc directional) pseudo atoms around each ZN ion
    via the official zinc_pseudo.py; returns the TZ-augmented PDBQT path. The
    default output name appends _TZ (matching the script's own convention)."""
    script = os.path.join(AD4ZN_DIR, "zinc_pseudo.py")
    if not os.path.exists(script):
        raise DockError(f"zinc_pseudo.py not found: {script} (see ad4zn/)")
    out = out_pdbqt or receptor_pdbqt.replace(".pdbqt", "_TZ.pdbqt")
    _run([ADFR_PYTHONSH, script, "-r", receptor_pdbqt, "-o", out],
         "zinc_pseudo.py")
    if not os.path.exists(out):
        raise DockError("zinc_pseudo.py produced no output: " + out)
    return out


def _build_ad4zn_gpf(tz_pdbqt, lig_pdbqt, gpf_path, center, box_edge, cwd=None):
    """Write the AD4Zn grid-parameter file for one ligand at one site.

    prepare_gpf4zn derives the ligand atom types and the calibrated Zn/TZ
    nbp_r_eps terms from the ligand + receptor, so the GPF (and the maps)
    are per-ligand. gridcenter is pinned to the METAL ION -- AutoGrid's
    'auto' centers on the RECEPTOR bounding box instead, which put a
    validated test run in empty space (score +1.5 instead of -12.2).

    The script echoes the receptor path RELATIVELY into the GPF (both the
    `receptor` and `gridfld` lines), so cwd must be the directory holding
    that receptor for prepare_gpf4zn itself and for autogrid afterwards.
    """
    script = os.path.join(AD4ZN_DIR, "prepare_gpf4zn.py")
    if not os.path.exists(script):
        raise DockError(f"prepare_gpf4zn.py not found: {script} (see ad4zn/)")
    if not os.path.exists(AD4ZN_DAT):
        raise DockError(f"AD4Zn.dat not found: {AD4ZN_DAT} (see ad4zn/)")
    npts = int(round(box_edge / AUTOGRID_SPACING))
    gc = f"{center[0]:.3f},{center[1]:.3f},{center[2]:.3f}"
    _run([ADFR_PYTHONSH, script, "-l", lig_pdbqt, "-r", tz_pdbqt,
          "-o", gpf_path,
          "-p", f"npts={npts},{npts},{npts}",
          "-p", f"gridcenter={gc}",
          "-p", f"parameter_file={AD4ZN_DAT}"],
         "prepare_gpf4zn.py", cwd=cwd)
    if not os.path.exists(gpf_path):
        raise DockError("prepare_gpf4zn.py produced no GPF: " + gpf_path)


def _run_autogrid(gpf_path, glg_path, cwd=None):
    """Compute AutoGrid maps for a prepared GPF (AD4Zn terms included).

    cwd should be the working dir when the GPF references the receptor by a
    relative name; AutoGrid writes the maps relative to its cwd.
    """
    if not os.path.exists(AUTOGRID_BIN):
        raise DockError(f"autogrid4 not found: {AUTOGRID_BIN} "
                        f"(conda: CONDA_SUBDIR=osx-64 conda create -n adfr-suite "
                        f"-c hcc adfr-suite; see module docstring)")
    _run([AUTOGRID_BIN, "-p", gpf_path, "-l", glg_path], "autogrid4", cwd=cwd)


def run_vina_ad4_maps(vina_bin, maps_prefix, lig_pdbqt, args, out_pdbqt, log_path):
    """Dock ONE ligand against precomputed AutoGrid maps with Vina's ad4 scoring.

    --maps + --scoring ad4 need Vina >= 1.2 (the dockstring-vendored 1.1.2
    binary has neither flag). All forcefield information (incl. AD4Zn) is in
    the maps, so the box flags are not used here.
    """
    cmd = [
        vina_bin,
        "--ligand", lig_pdbqt,
        "--maps", maps_prefix,
        "--scoring", "ad4",
        "--out", out_pdbqt,
        "--exhaustiveness", str(args.exhaustiveness),
        "--num_modes", str(args.num_modes),
    ]
    if args.cpu:
        cmd += ["--cpu", str(args.cpu)]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    print(f"vina_dock: running Vina (ad4 maps)...\n  {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise DockError(f"Vina (ad4 maps) failed (exit {res.returncode})\n"
                        f"  stderr: {res.stderr.strip()}\n  stdout: {res.stdout.strip()}")
    # Vina 1.2 has no --log flag; the mode table comes on stdout. Save it so
    # parse_vina_log() sees the same format it would from a 1.1.2 log file.
    with open(log_path, "w") as fh:
        fh.write(res.stdout)
    return res.stdout


def _pose_donor_metal_distance(pdbqt_path, metal_xyz):
    """Min distance from pose donor atoms (first MODEL only) to a metal ion.

    Atom types in DONOR_PDBQT_TYPES count as donors (polar N/O/S). Returns
    None when the pose file has no donor atoms at all.
    """
    best = None
    seen_model = False
    with open(pdbqt_path) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if line.startswith("MODEL"):
                if seen_model:
                    break
                seen_model = True
                continue
            if line.startswith(("ATOM", "HETATM")):
                t = line[77:79].strip()
                if t not in DONOR_PDBQT_TYPES:
                    continue
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except ValueError:
                    continue
                d = math.sqrt((x - metal_xyz[0]) ** 2 + (y - metal_xyz[1]) ** 2 +
                              (z - metal_xyz[2]) ** 2)
                if best is None or d < best:
                    best = d
    return best


def dock_metal_site(receptor_pdb, smiles_list, center, box_size=METAL_DOCK_BOX,
                    exhaustiveness=8, num_modes=3, seed=0, cpu=0,
                    overwrite_receptor=False, verbose=False, engine=None):
    """Dock ligands at a metal site, engine chosen by the metal ion present.

    `center` should be a metal ion's (x, y, z) from detect_metal_sites(); the
    code snaps it to the nearest detected ion within METAL_SNAP_CUTOFF (3 A)
    and picks the docking engine itself:
      Zn2+  -> AD4Zn: meeko receptor PDBQT -> zinc_pseudo TZ atoms ->
               per-ligand prepare_gpf4zn + autogrid maps -> Vina 1.2.7
               --scoring ad4 --maps. Zinc-calibrated, directional.
      Mg/Ca/Mn/Fe -> tier 0 'ad4map': AutoGrid maps with the stock metal
               parameters already in AD4Zn.dat.
      Cu/Co/Ni/Cd/Hg/Na/K -> tier 1 'ad4merz': AutoGrid maps with the
               literature self-parameters in AD4Merz.dat (Li & Merz 12-6-4
               TIP3P, Joung & Cheatham monovalent). No directional terms, so
               contacts run long -- the coordination check stays the readout.
      other -> receptor PDBQT rebuilt WITH the metal ions (waters/cofactors
               still dropped), docked with Vina 1.2.7 generic scoring.
               Approximate scores; poses are checked for coordination.
    Every molecule's report line carries its best pose's donor-to-metal
    distance (<= 2.5 A marks an actual coordination bond).

    The `engine` kwarg is a DEV-ONLY override ('ad4zn', 'ad4map', 'ad4merz',
    'vina-metal'; default None = auto by detected element). Never expose it
    to the LLM-facing agent wrapper -- tool-callers hallucinate optional
    kwargs (see the agent-tool-signature gotcha comment).

    Persisted artefacts (next to the input PDB):
      <stem>_mk.pdbqt / <stem>_mkTZ.pdbqt   (zinc path, cached receptor)
      <stem>_map.pdbqt                      (tier 0/1 map path, cached)
      <stem>_metal.pdbqt                    (generic-metal path)
      <stem>_m<i>.sdf                       top poses for molecule i

    Raises:
        DockError for setup failures affecting every molecule (no metal ion
        near `center`, ADFR/meeko tooling missing, autogrid failure).
        Per-molecule failures are caught and reported, not raised.
    """
    print("vina_dock: docking at metal site...")
    print('==============================================')

    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]

    if not os.path.exists(receptor_pdb):
        raise DockError(f"receptor not found: {receptor_pdb}")
    receptor_pdb = os.path.abspath(receptor_pdb)
    stem = os.path.splitext(receptor_pdb)[0]
    center = [float(c) for c in center]

    sites = detect_metal_sites(receptor_pdb)
    if not sites:
        raise DockError(f"no metal ions detected in {receptor_pdb}; "
                        "use dock_at_centroid instead")
    metal = _nearest_metal(sites, center)
    if metal is None:
        raise DockError(f"no metal ion within {METAL_SNAP_CUTOFF:.1f} A of "
                        f"{center}; call detect_metal_sites_agent for exact "
                        f"metal coordinates")
    mxyz = metal["xyz"]
    mtag = f"{metal['element']} ({metal['resname']} {metal['chain']}{metal['resseq']})"
    el = metal["element"].upper()
    engine_override = engine
    tier_dat = None
    if el == "ZN":
        engine = "ad4zn"
    elif el in AD4MAP_ELEMENTS:
        engine, tier_dat = "ad4map", AD4ZN_DAT
    elif el in AD4MERZ_ELEMENTS:
        engine, tier_dat = "ad4merz", AD4MERZ_DAT
    else:
        engine = "vina-metal"
    if engine_override and engine_override != "auto":
        if engine_override not in ("ad4zn", "ad4map", "ad4merz", "vina-metal"):
            raise DockError(f"unknown engine override: {engine_override}")
        engine = engine_override
        tier_dat = {"ad4map": AD4ZN_DAT, "ad4merz": AD4MERZ_DAT}.get(engine)
    use_ad4zn = engine == "ad4zn"
    use_maps = use_ad4zn or tier_dat is not None
    box = [float(box_size)] * 3

    # Quiet mode swallowing, identical to blind_dock/dock_at_centroid.
    with contextlib.ExitStack() as stack:
        if not verbose:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            try:
                from rdkit import RDLogger
                RDLogger.DisableLog("rdApp.*")
                stack.callback(RDLogger.EnableLog, "rdApp.*")
            except Exception:
                pass
            # subprocess children also write; route their output to the
            # swallowed stdio by keeping them on captured pipes (default).

        if use_ad4zn:
            if not os.path.exists(ADFR_PYTHONSH):
                raise DockError(f"pythonsh not found: {ADFR_PYTHONSH} "
                                f"(set ADFR_PYTHONSH or create the adfr-suite env)")
            if _meeko_script("mk_prepare_receptor.py") is None:
                raise DockError("meeko (mk_prepare_receptor.py) not found; "
                                "pip install meeko")
            mk_root = stem + "_mk"
            if overwrite_receptor or not os.path.exists(mk_root + ".pdbqt"):
                _meeko_receptor_pdbqt(receptor_pdb, mk_root)
                # drop meeko's stray side files, keep only the cached pdbqt
                for junk in glob.glob(mk_root + "-*") + [mk_root + ".pdb",
                                                         mk_root + ".h-pdb.pdb"]:
                    try:
                        os.remove(junk)
                    except OSError:
                        pass
            tz_pdbqt = stem + "_mkTZ.pdbqt"
            if overwrite_receptor or not os.path.exists(tz_pdbqt):
                _add_zinc_pseudos(mk_root + ".pdbqt", out_pdbqt=tz_pdbqt)
            rec_pdbqt = tz_pdbqt
            vina_bin = VINA_12_BIN
            if not os.path.exists(vina_bin):
                raise DockError(f"Vina >= 1.2 binary not found: {vina_bin} "
                                f"(set VINA_12_BIN; --scoring ad4 needs Vina 1.2+)")
            npts = int(round(box_size / AUTOGRID_SPACING))
        elif tier_dat is not None:
            if _meeko_script("mk_prepare_receptor.py") is None:
                raise DockError("meeko (mk_prepare_receptor.py) not found; "
                                "pip install meeko")
            if _meeko_script("mk_prepare_ligand.py") is None:
                raise DockError("meeko (mk_prepare_ligand.py) not found; "
                                "pip install meeko")
            rec_pdbqt = stem + "_map.pdbqt"
            if overwrite_receptor or not os.path.exists(rec_pdbqt):
                _prepare_map_receptor(receptor_pdb, rec_pdbqt)
            vina_bin = VINA_12_BIN
            if not os.path.exists(vina_bin):
                raise DockError(f"Vina >= 1.2 binary not found: {vina_bin} "
                                f"(set VINA_12_BIN; --scoring ad4 needs Vina 1.2+)")
            npts = int(round(box_size / AUTOGRID_SPACING))
        else:
            rec_pdbqt = stem + "_metal.pdbqt"
            if overwrite_receptor or not os.path.exists(rec_pdbqt):
                build_receptor_pdbqt_with_metals(receptor_pdb, rec_pdbqt)
            vina_bin = VINA_12_BIN  # 1.2.x reads non-organic types best
            if not os.path.exists(vina_bin):
                raise DockError(f"Vina >= 1.2 binary not found: {vina_bin} "
                                f"(set VINA_12_BIN)")
            npts = None

        work = tempfile.mkdtemp(prefix="dock_metal_")
        vina_args = SimpleNamespace(exhaustiveness=exhaustiveness,
                                    num_modes=max(num_modes, 3), cpu=cpu, seed=seed)
        n_out_poses = min(3, vina_args.num_modes)

        # Deterministic map naming: the receptor is copied into the temp dir
        # as rec.pdbqt, so the GPF's gridfld is rec.maps.fld and AutoGrid
        # drops maps at work/rec.<TYPE>.map (gridcenter itself is pinned to
        # the metal ion below -- autogrid's 'auto' would center on the
        # RECEPTOR bounding box, which put a validated test run in empty
        # space, score +1.5 instead of -12.2). The parameter file is copied
        # alongside under its basename: both prepare_gpf4zn and my GPF writer
        # basename it in the GPF, so AutoGrid resolves it relative to cwd.
        maps_prefix = None
        if use_maps:
            work_rec = os.path.join(work, "rec.pdbqt")
            shutil.copy(rec_pdbqt, work_rec)
            params_dat = AD4ZN_DAT if use_ad4zn else tier_dat
            shutil.copy(params_dat, os.path.join(work, os.path.basename(params_dat)))
            maps_prefix = os.path.join(work, "rec")

        results = []  # (idx, smi, affinity, sdf_path, status, detail, coord_d)
        for idx, smi in enumerate(smiles_list):
            sdf_path = f"{stem}_m{idx}.sdf"
            try:
                lig_sdf = os.path.join(work, f"ligand_{idx}.sdf")
                build_ligand_3d(smi, lig_sdf)
                if use_maps:
                    lig_pdbqt = os.path.join(work, f"ligand_{idx}.pdbqt")
                    _meeko_ligand_pdbqt(lig_sdf, lig_pdbqt)
                    gpf_path = os.path.join(work, f"grid_{idx}.gpf")
                    if use_ad4zn:
                        _build_ad4zn_gpf(work_rec, lig_pdbqt, gpf_path, mxyz,
                                         box_size, cwd=work)
                    else:
                        _build_ad4map_gpf(work_rec, lig_pdbqt, gpf_path, mxyz,
                                          box_size, os.path.basename(params_dat),
                                          METAL_ATOM_TYPES[metal["element"]][0])
                    _run_autogrid(gpf_path, os.path.join(work, f"grid_{idx}.glg"),
                                  cwd=work)
                    poses_pdbqt = os.path.join(work, f"poses_{idx}.pdbqt")
                    log_path = os.path.join(work, f"vina_{idx}.log")
                    run_vina_ad4_maps(vina_bin, maps_prefix, lig_pdbqt,
                                      vina_args, poses_pdbqt, log_path)
                else:
                    lig_pdbqt = os.path.join(work, f"ligand_{idx}.pdbqt")
                    convert(["obabel", "-isdf", lig_sdf, "-opdbqt", "-O", lig_pdbqt],
                            f"ligand SDF->PDBQT ({smi})")
                    poses_pdbqt = os.path.join(work, f"poses_{idx}.pdbqt")
                    log_path = os.path.join(work, f"vina_{idx}.log")
                    run_vina(vina_bin, rec_pdbqt, lig_pdbqt, mxyz, box,
                             vina_args, poses_pdbqt, log_path, no_log=True)
                aff, nmodes = parse_vina_log(log_path)
                if aff is None:
                    results.append((idx, smi, None, sdf_path, "failed",
                                    "no score parsed from Vina log", None))
                    continue
                convert(["obabel", "-ipdbqt", poses_pdbqt, "-osdf", "-O", sdf_path,
                         "-l", str(n_out_poses)],
                        f"pose PDBQT->SDF (mol {idx}, top {n_out_poses})")
                cd = _pose_donor_metal_distance(poses_pdbqt, mxyz)
                if cd is None:
                    cstr = "no donor atoms in pose"
                elif cd <= COORDINATION_CUTOFF:
                    cstr = f"COORDINATES {metal['element']} at {cd:.2f} A"
                else:
                    cstr = f"no coordination ({cd:.2f} A)"
                results.append((idx, smi, aff, sdf_path, "ok",
                                f"{nmodes} modes; {cstr}", cd))
            except DockError as e:
                results.append((idx, smi, None, sdf_path, "failed", str(e), None))
            except Exception as e:  # defensive: never let one molecule kill the run
                results.append((idx, smi, None, sdf_path, "error",
                                f"{type(e).__name__}: {e}", None))

    shutil.rmtree(work, ignore_errors=True)

    # report
    lines = []
    lines.append("Metal-site docking report")
    lines.append(f"  receptor:       {receptor_pdb}")
    lines.append(f"  metal site:     {mtag} at "
                 f"({mxyz[0]:.3f}, {mxyz[1]:.3f}, {mxyz[2]:.3f})")
    engine_note = {
        "ad4zn": "(AutoDock4Zn force field via AutoGrid maps)",
        "ad4map": "(AutoGrid ad4 maps; stock AD4Zn.dat metal parameters + site terms)",
        "ad4merz": "(AutoGrid ad4 maps; Merz 12-6-4 / Joung-Cheatham metal parameters + site terms)",
        "vina-metal": "(metal kept in receptor; scores approximate)",
    }[engine]
    lines.append(f"  engine:         {engine} {engine_note}")
    lines.append(f"  vina binary:    {vina_bin}")
    lines.append(f"  box:            ({mxyz[0]:.3f}, {mxyz[1]:.3f}, {mxyz[2]:.3f}) "
                 f"edge {box[0]:.1f} A"
                 + (f" (autogrid npts {npts}, spacing {AUTOGRID_SPACING} A)" if npts else ""))
    lines.append("")
    lines.append(f"Molecules ({len(results)}):")
    for idx, smi, aff, sdf, status, detail, cd in results:
        lines.append(f"  [{idx}] {smi}")
        if status == "ok":
            lines.append(f"        score: {aff:.2f} kcal/mol   ({detail})")
            lines.append(f"        poses SDF (top {n_out_poses}): {sdf}")
        else:
            lines.append(f"        {status}: {detail}")
    lines.append("")
    ok = [r for r in results if r[4] == "ok"]
    if ok:
        coord = [r for r in ok if r[6] is not None and r[6] <= COORDINATION_CUTOFF]
        bidx, bsmi, baff, bsdf, _, _, bcd = min(ok, key=lambda r: r[2])
        lines.append(f"Best molecule: [{bidx}] {bsmi}  "
                     f"score={baff:.2f} kcal/mol  SDF={bsdf}")
        if coord:
            cbest = min(coord, key=lambda r: r[6])
            lines.append(f"Coordinating poses: {len(coord)}/{len(ok)} "
                         f"(best {cbest[6]:.2f} A, molecule [{cbest[0]}])")
        else:
            lines.append("Coordinating poses: none within "
                         f"{COORDINATION_CUTOFF:.1f} A")
    else:
        lines.append("Best molecule: none docked successfully")
    return "\n".join(lines)


def dock_metal_site_agent(receptor_pdb: str, smiles_list: list[str],
                          center_x: float, center_y: float, center_z: float) -> str:
    """
    Dock ligands at a METAL ion site in a receptor, engine chosen by the
    metal ion: Zn2+ gets the zinc-specific AutoDock4Zn force field,
    Mg/Ca/Mn/Fe get AutoGrid maps with stock AutoDock metal parameters,
    Cu/Co/Ni/Cd/Hg/Na/K get AutoGrid maps with literature ion parameters,
    and any other metal keeps the ion in the receptor with generic Vina
    scoring. Every pose is checked for metal coordination (donor atom within
    2.5 A; the honest readout for the non-zinc engines). The box center must
    be the metal ion's x,y,z coordinates from detect_metal_sites_agent (the
    ion is snapped up to 3 A away). Use this instead of
    dock_at_centroid_agent when the binding site is a metal ion (e.g. HHH
    zinc proteases, Mg/Ca binding sites).

    Args:
        receptor_pdb: path to a receptor PDB file.
        smiles_list: iterable list of ligand SMILES strings.
        center_x: metal-center X coordinate from detect_metal_sites_agent.
        center_y: metal-center Y coordinate from detect_metal_sites_agent.
        center_z: metal-center Z coordinate from detect_metal_sites_agent.
    Returns:
        A multi-line report (metal site + engine chosen, one block per
        molecule with score, coordination distance and pose-SDF path).
    """
    return dock_metal_site(receptor_pdb, smiles_list,
                           center=(center_x, center_y, center_z),
                           box_size=METAL_DOCK_BOX)


# --- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--receptor", required=True, help="Receptor as a PDB file.")
    ap.add_argument("--smiles", required=True, help="Ligand as a SMILES string.")
    ap.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="Search-box center in Angstroms. Required unless --blind is "
                         "given (blind mode detects the box).")
    ap.add_argument("--size", nargs="+", type=float, default=[20, 20, 20],
                    metavar="ANG",
                    help="Search-box sizes in Angstroms (1 or 3 values; default 20 20 20). "
                         "Vina caps each dimension at 30.")
    ap.add_argument("--exhaustiveness", type=int, default=8,
                    help="Vina global search effort (default 8).")
    ap.add_argument("--num-modes", type=int, default=9, dest="num_modes",
                    help="Number of output poses (default 9).")
    ap.add_argument("--cpu", type=int, default=0,
                    help="CPUs to use (default: let Vina autodetect).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducibility (default: Vina's default).")
    ap.add_argument("--work-dir", default=None,
                    help="Where to write intermediates + outputs "
                         "(default: ./vina_run_<timestamp>).")
    ap.add_argument("--keep", action="store_true",
                    help="Keep intermediates (receptor/ligand PDBQT, SDF). "
                         "By default only the Vina output + best_pose.sdf are kept.")
    ap.add_argument("--vina-bin", default=None,
                    help="Path to a Vina binary (default: dockstring's vendored binary).")
    # --- blind (binding-site-unknown) mode ---
    ap.add_argument("--blind", action="store_true",
                    help="Blind mode: detect putative binding pockets with a buriedness "
                         "grid scan (pure Python, scipy) and dock the top N, keeping the "
                         "best score. Use for a novel receptor with no known binding site. "
                         "Mutually exclusive with --center. For known dockstring targets, "
                         "just use dockstring directly instead.")
    ap.add_argument("--blind-npockets", type=int, default=3, dest="blind_npockets",
                    help="Number of top-ranked pockets to dock in --blind mode (default 3; "
                         "validated to contain the true site for the five DUD-E receptors).")
    ap.add_argument("--blind-box", type=float, default=28.0, dest="blind_box",
                    help="Focused box size (Angstroms, cubic) placed on each detected "
                         "pocket in --blind mode (default 28).")
    ap.add_argument("--blind-spacing", type=float, default=1.0, dest="blind_spacing",
                    help="Grid spacing in Angstroms for the pocket scan (default 1.0).")
    ap.add_argument("--blind-r", type=float, default=8.0, dest="blind_r",
                    help="Buriedness shell radius in Angstroms (default 8.0).")
    ap.add_argument("--blind-band", nargs=2, type=float, default=[2.5, 8.0],
                    metavar=("DMIN", "DMAX"), dest="blind_band",
                    help="Nearest-atom distance band (Angstroms) defining empty-but-near-"
                         "protein voxels (default 2.5 8.0).")
    ap.add_argument("--blind-top-frac", type=float, default=0.06, dest="blind_top_frac",
                    help="Fraction of highest-buriedness voxels kept before clustering "
                         "(default 0.06). Higher keeps more moderate-buriedness cleft voxels "
                         "(good) but also diffuse surface voxels that merge into a giant blob "
                         "(bad unless --blind-min-samples is raised).")
    ap.add_argument("--blind-min-samples", type=int, default=25, dest="blind_min_samples",
                    help="DBSCAN min neighbours (within --blind-eps) for a voxel to start a "
                         "cluster (default 25). Higher fragments the diffuse low-density "
                         "surface blob that drowns true pockets; the dense true cleft survives. "
                         "Validated plateau with --blind-top-frac in [0.04, 0.08] and this in "
                         "[20, 40] (true site = #1 pocket on all five DUD-E receptors).")
    ap.add_argument("--blind-eps", type=float, default=2.5, dest="blind_eps",
                    help="DBSCAN neighbourhood radius in Angstroms (default 2.5).")
    args = ap.parse_args()

    if args.blind and args.center is not None:
        ap.error("--blind is mutually exclusive with --center (blind mode detects the box).")
    if not args.blind and args.center is None:
        ap.error("--center is required unless --blind is given "
                 "(blind mode detects the box; a real run must target a specific site).")

    if not os.path.exists(args.receptor):
        sys.exit(f"vina_dock: receptor not found: {args.receptor}")

    require("obabel")
    vina_bin = find_vina_bin(args.vina_bin)
    print(f"vina_dock: Vina binary: {vina_bin}")

    work = args.work_dir or os.path.join(HERE, f"vina_run_{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(work, exist_ok=True)

    rec_pdb = os.path.abspath(args.receptor)
    rec_pdbqt = os.path.join(work, "receptor.pdbqt")
    lig_sdf = os.path.join(work, "ligand.sdf")
    lig_pdbqt = os.path.join(work, "ligand.pdbqt")
    best_sdf = os.path.join(work, "best_pose.sdf")

    # Resolve the box(es) to dock: either one explicit box, or top-N detected pockets.
    if args.blind:
        print("vina_dock: --blind: scanning receptor for putative binding pockets...")
        pockets = find_pockets(rec_pdb, spacing=args.blind_spacing, r_shell=args.blind_r,
                               d_min=args.blind_band[0], d_max=args.blind_band[1],
                               top_frac=args.blind_top_frac, eps=args.blind_eps,
                               min_samples=args.blind_min_samples)
        if not pockets:
            sys.exit("vina_dock: --blind: no pockets detected; supply --center instead.")
        pockets = pockets[:args.blind_npockets]
        print(f"  detected {len(pockets)} pocket(s) (docking top {len(pockets)}):")
        boxes = []
        for i, p in enumerate(pockets):
            c = p["center"]
            print(f"    #{i+1} center=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) "
                  f"voxels={p['n_voxels']} buriedness={p['buriedness']:.0f} "
                  f"score={p['score']:.0f}")
            boxes.append((f"pocket_{i+1}", c, [args.blind_box] * 3))
    else:
        if len(args.size) == 1:
            size = [args.size[0]] * 3
        elif len(args.size) == 3:
            size = args.size
        else:
            ap.error("--size needs 1 or 3 values, got %d" % len(args.size))
        if size[0] * size[1] * size[2] > 27000:
            print("vina_dock: WARNING — search-space volume > 27000 A^3; Vina will accept it "
                  "but warns that such a large box needs higher --exhaustiveness to be sampled "
                  "adequately (see Vina FAQ).", file=sys.stderr)
        center = args.center
        print(f"vina_dock: box center = {center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f} "
              f"(size {size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} A)")
        boxes = [("box", center, size)]

    print("vina_dock: preparing receptor PDBQT...")
    n_rec = build_receptor_pdbqt(rec_pdb, rec_pdbqt)
    print(f"  receptor.pdbqt: {n_rec} atoms")

    print("vina_dock: preparing ligand PDBQT...")
    n_lig = build_ligand_pdbqt(args.smiles, lig_sdf, lig_pdbqt)
    print(f"  ligand.pdbqt: {n_lig} lines")

    # Dock each box; keep the lowest-affinity (best) result.
    results = []  # (label, center, size, poses_path, log_path, affinity, n_modes)
    best_idx = None
    for label, center, size in boxes:
        poses_pdbqt = os.path.join(work, f"poses_{label}.pdbqt") if len(boxes) > 1 \
            else os.path.join(work, "poses.pdbqt")
        log_path = os.path.join(work, f"vina_{label}.log") if len(boxes) > 1 \
            else os.path.join(work, "vina.log")
        print(f"\nvina_dock: docking {label} at "
              f"({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) "
              f"size {size[0]:.0f}x{size[1]:.0f}x{size[2]:.0f}")
        run_vina(vina_bin, rec_pdbqt, lig_pdbqt, center, size, args, poses_pdbqt, log_path)
        aff, n_modes = parse_vina_log(log_path)
        results.append((label, center, size, poses_pdbqt, log_path, aff, n_modes))
        if aff is not None and (best_idx is None or aff < results[best_idx][5]):
            best_idx = len(results) - 1

    print("\n" + "=" * 60)
    if best_idx is None:
        print("vina_dock: could not parse a score from any Vina log — see logs in:")
        print(f"  {work}")
    else:
        bl, bc, bs, bp, blog, baff, bnm = results[best_idx]
        if len(boxes) > 1:
            print(f"Best pocket: {bl} (center {bc[0]:.2f},{bc[1]:.2f},{bc[2]:.2f})")
        print(f"Best affinity: {baff:.2f} kcal/mol   ({bnm} modes)")
        print(f"Poses (PDBQT): {bp}")
        print(f"Vina log:      {blog}")
        # Convert the best pose (first model) to SDF for easy inspection.
        convert(["obabel", "-ipdbqt", bp, "-osdf", "-O", best_sdf, "-l", "1"],
                "best pose PDBQT->SDF")
        print(f"Best pose (SDF): {best_sdf}")
    if len(boxes) > 1:
        print("\nAll pockets docked (label: affinity kcal/mol):")
        for label, _, _, poses_pdbqt, log_path, aff, n_modes in results:
            print(f"  {label}: {aff if aff is not None else 'n/a'}")
    print("=" * 60)

    if not args.keep:
        for p in (rec_pdbqt, lig_sdf, lig_pdbqt):
            try:
                os.remove(p)
            except OSError:
                pass
        print("vina_dock: removed intermediates (receptor/ligand PDBQT, SDF). "
              "Use --keep to retain them.")


if __name__ == "__main__":
    try:
        main()
    except DockError as e:
        sys.exit(f"vina_dock: {e}")