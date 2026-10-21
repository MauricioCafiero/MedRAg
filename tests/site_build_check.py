"""Carve a binding site around many ligand placements and check that every one comes out clean (no atom pair closer than
system.COLLISION_A) or fails loudly with SiteBuildError. No MLIP involved. usage: site_build_check.py"""
import glob, json, sys, time
from collections import Counter
import numpy as np
from medrag.tools import poses
from medrag.feasibility import system as S

SULT = dict(pdb="runs/sult_test/pdb_files/SULT1A3_2A3R.pdb", rec="runs/sult_e2e/target/receptor.pdb")
HMGCR = dict(pdb="runs/dock_test/pdb_files/HMGCR_1HWL.pdb", rec="runs/dock_test/pdb_files/HMGCR_1HWL_protonated.pdb")
FBI = "CC(C)c1nc(N(C)S(C)(=O)=O)nc(-c2ccc(F)cc2)c1/C=C/C(O)CC(O)CC(=O)[O-]"
DOPA = "[NH3+]CCc1ccc(O)c(O)c1"
cases = []
for ch in "AB": cases.append((f"2A3R LDP {ch}", SULT["pdb"], SULT["pdb"], poses.crystal_ligand_block(SULT["pdb"], "LDP", ch), DOPA))
for ch in "ABCD": cases.append((f"1HWL FBI {ch}", HMGCR["rec"], HMGCR["pdb"], poses.crystal_ligand_block(HMGCR["pdb"], "FBI", ch), FBI))
for f in glob.glob("runs/sult_e2e/dock/poses/*.sdf"):
    st = json.load(open("runs/sult_e2e/state.json"))
    break
smi_by_file = {r["dock"]["sdf"].split("/")[-1]: r["dock"]["species"] for r in st["candidates"].values() if r.get("dock", {}).get("status") == "ok"}
for f in sorted(glob.glob("runs/sult_e2e/dock/poses/*.sdf")):
    for j, (blk, sc) in enumerate(poses.read_poses(f)):
        cases.append((f"sult pose {f.split('/')[-1][:6]}#{j+1}", SULT["rec"], None, blk, smi_by_file[f.split("/")[-1]]))
for j, (blk, sc) in enumerate(poses.read_poses("runs/dock_clean2/pdb_files/HMGCR_atoms_c0.sdf")):
    cases.append((f"hmgcr redock #{j+1}", "runs/dock_clean2/pdb_files/HMGCR_atoms.pdb", None, blk, FBI))

res = Counter(); rows = []
for name, rec, _, blk, smi in cases:
    if not __import__("os").path.exists(rec): res["missing receptor"] += 1; continue
    try:
        lig = S.ligand_with_h(blk, smi)
    except Exception as e:
        res["ligand build failed"] += 1; rows.append((name, f"ligand: {e}")); continue
    # baseline: UMADock's exact selection
    try: base = S.build_system(rec, lig, fill_gaps=False); nb = sum("UNCHECKED" in w for w in base.warnings)
    except Exception as e: nb = None
    try:
        t = time.time(); sysm = S.build_system(rec, lig)
        added = []
        bad = S.pocket_collisions(sysm.complex.positions[:sysm.n_pocket])
        res["clean" if not bad else "RETURNED BROKEN"] += 1
        res["repair loop used"] += bool(added); res["umadock selection had collisions"] += bool(nb)
        rows.append((name, f"{len(sysm.complex)} atoms, caps {len(sysm.caps)}, umadock-collisions={'yes' if nb else 'no'}, {added[0].split('; ')[-1] if added else 'no repair'}"))
    except S.SiteBuildError as e:
        res["SiteBuildError (loud)"] += 1; rows.append((name, f"SiteBuildError: {str(e)[:120]}"))
    except Exception as e:
        res["other error"] += 1; rows.append((name, f"{type(e).__name__}: {str(e)[:120]}"))
print(dict(res), "of", len(cases), "cases")
for r in rows:
    if "no repair" not in r[1] or "umadock-collisions=yes" in r[1] or "Error" in r[1] or "ligand" in r[1]: print(" ", r)
