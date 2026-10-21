"""Run the routine feasibility checks on the crystal pose and every redocked pose of dopamine / paracetamol in SULT1A3.
usage: feas_poses.py [mlip]      results: runs/feas_sult_poses/<name>/result.json (finished poses are skipped on rerun;
unfinished ones resume from their relax checkpoints)."""
import json, sys, time
from pathlib import Path
from medrag.tools import poses
from medrag.feasibility.system import ligand_with_h, build_system
from medrag.feasibility import checks, calc
from medrag.feasibility.checks import PoseContext
from medrag.feasibility.thresholds import Thresholds

mlip = sys.argv[1] if len(sys.argv) > 1 else "polar1-s"
PDB = "runs/sult_test/pdb_files/SULT1A3_2A3R.pdb"
RECEPTOR = "runs/sult_dock/pdb_files/SULT1A3_atoms.pdb"
OUT = Path("runs/feas_sult_poses" + ("" if mlip == "polar1-s" else f"_{mlip}"))   # polar1-s keeps the original dir
DOPA, PARA = "[NH3+]CCc1ccc(O)c(O)c1", "CC(=O)Nc1ccc(O)cc1"
jobs = [("crystal_dopamine", DOPA, poses.crystal_ligand_block(PDB, "LDP", "A"))]
for name, smi, f in (("dopamine", DOPA, "c0"), ("paracetamol", PARA, "c1")):
    for i, (blk, sc) in enumerate(poses.read_poses(f"runs/sult_dock/pdb_files/SULT1A3_atoms_{f}.sdf")):
        jobs.append((f"{name}_pose{i+1}", smi, blk))
order = ["crystal_dopamine", "dopamine_pose3", "dopamine_pose1", "dopamine_pose2", "paracetamol_pose3", "paracetamol_pose1", "paracetamol_pose2"]
jobs.sort(key=lambda j: order.index(j[0]))
c = calc.get_calculator(mlip)
for name, smi, blk in jobs:
    wd = OUT / name
    if (wd / "result.json").exists():
        print(f"{name}: done, skipping", flush=True); continue
    t0 = time.time()
    sysm = build_system(RECEPTOR, ligand_with_h(blk, smi))
    ctx = PoseContext(system=sysm, calc=c, thresholds=Thresholds(), workdir=str(wd))
    verdict, reasons = checks.run_checks(ctx)
    res = {k: dict(value=r.value, unit=r.unit, status=r.status, detail=r.detail, cost_s=r.cost_s, extra=r.extra) for k, r in ctx.results.items()}
    (wd / "result.json").write_text(json.dumps(dict(name=name, mlip=mlip, atoms=len(sysm.complex), caps=len(sysm.caps), charge=sysm.charge,
        prep=ctx.prep, verdict=verdict, reasons=reasons, checks=res, wall_s=time.time() - t0), indent=1, default=str))
    (wd / "ligand_docked.sdf").write_text(ctx.ligand_block()); (wd / "ligand_relaxed.sdf").write_text(ctx.ligand_block(relaxed=True))
    print(f"{name}: {verdict} in {time.time()-t0:.0f}s | " + " | ".join(f"{k} {r.value:.2f} {r.status}" for k, r in ctx.results.items()), flush=True)
