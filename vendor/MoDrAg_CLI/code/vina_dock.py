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


def build_ligand_pdbqt(smiles, sdf_path, pdbqt_path):
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

def run_vina(vina_bin, rec_pdbqt, lig_pdbqt, center, size, args, out_pdbqt, log_path):
    cmd = [
        vina_bin,
        "--receptor", rec_pdbqt,
        "--ligand", lig_pdbqt,
        "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
        "--size_x", str(size[0]), "--size_y", str(size[1]), "--size_z", str(size[2]),
        "--out", out_pdbqt,
        "--log", log_path,
        "--exhaustiveness", str(args.exhaustiveness),
        "--num_modes", str(args.num_modes),
    ]
    if args.cpu:
        cmd += ["--cpu", str(args.cpu)]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    print(f"vina_dock: running Vina...\n  {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise DockError(f"Vina failed (exit {res.returncode})\n"
                        f"  stderr: {res.stderr.strip()}\n  stdout: {res.stdout.strip()}")
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
    # Next step: confirm the predicted pose actually sits in the real binding
    # site. check_nearby_molecules compares the docked pose against any
    # co-crystallized ligand in the receptor PDB; if a crystal ligand is
    # present, run it on the best molecule's pose SDF.
    lines.append("")
    lines.append("Next step: verify the docked pose with the "
                 "check_nearby_molecules tool. Pass the receptor PDB as "
                 f"`pdb_filepath` and the best-molecule pose SDF as "
                 f"`ligand_filepath` to confirm the pose landed in the correct "
                 f"binding site.")
    return "\n".join(lines)


def blind_dock_agent(receptor_pdb: str, smiles_list: list[str]) -> str:
    """
    Blind-dock ligands into a receptor of UNKNOWN binding site, given a
    receptor PDB file. Use this when the protein is NOT in docking_node's
    dockstring target list, or when only a PDB structure (not a named target)
    is available. If the user gives a protein name or PDB ID but no file,
    fetch the PDB with get_pdb_file first, then pass the saved PDB path here.
    Do NOT use this when the protein is in docking_node's list -- prefer
    docking_node (faster, validated).

    Args:
        receptor_pdb: path to a receptor PDB file (binding site unknown).
        smiles_list: iterable list of ligand SMILES strings.
    Returns:
        A multi-line string report (header with receptor + receptor-PDBQT path
        + pocket centers, one block per molecule with score
    """
    return blind_dock(receptor_pdb, smiles_list, npockets=1)


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