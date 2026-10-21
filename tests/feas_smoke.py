"""Smoke test of the feasibility checks on a real system.
usage: feas_smoke.py [crystal|docked1] [uma|polar1-s|polar1-m|maceoff] [hmgcr|sult1a3] [tag]
hmgcr:   1HWL, rosuvastatin (FBI chain A); also redocks and reports RMSD to the crystal pose.
sult1a3: 2A3R, L-dopamine (LDP chain A) - the small site UMADock/dock_assist use (275-atom cut-out at 4 A)."""
import sys, time
from pathlib import Path
import psutil
from medrag.tools import poses
from medrag.feasibility.system import ligand_with_h, build_system
from medrag.feasibility import checks, calc
from medrag.feasibility.checks import PoseContext
from medrag.feasibility.thresholds import Thresholds

which = sys.argv[1] if len(sys.argv) > 1 else "docked1"
mlip = sys.argv[2] if len(sys.argv) > 2 else "uma"
target = sys.argv[3] if len(sys.argv) > 3 else "hmgcr"
tag = sys.argv[4] if len(sys.argv) > 4 else ""      # new workdir for a fresh run (an existing workdir is resumed)

TARGETS = {
    "hmgcr": dict(
        smi="CC(C)c1nc(N(C)S(C)(=O)=O)nc(-c2ccc(F)cc2)c1/C=C/C(O)CC(O)CC(=O)[O-]",
        receptor="runs/dock_test/pdb_files/HMGCR_1HWL_protonated.pdb",   # has H; build_system ignores HETATM
        crystal="runs/dock_test/pdb_files/HMGCR_1HWL.pdb", resname="FBI", chain="A",
        docked="runs/dock_clean2/pdb_files/HMGCR_atoms_c0.sdf",
        workdir=f"runs/feas_{which}_{mlip}_staged{tag}"),                  # no tag = the original (old cut-out) run dir: pass a tag for a fresh run
    "sult1a3": dict(
        smi="[NH3+]CCc1ccc(O)c(O)c1",                                      # L-dopamine at pH 7
        receptor="runs/sult_test/pdb_files/SULT1A3_2A3R.pdb",              # raw: build_system re-protonates (PDBFixer)
        crystal="runs/sult_test/pdb_files/SULT1A3_2A3R.pdb", resname="LDP", chain="A",
        docked=None,
        workdir=f"runs/feas_sult1a3_{which}_{mlip}{tag}"),
}
T = TARGETS[target]
SMI = T["smi"]

xtal = poses.crystal_ligand_block(T["crystal"], T["resname"], T["chain"])
docked = poses.read_poses(T["docked"]) if T["docked"] else []
for i, (blk, sc) in enumerate(docked):
    print(f"pose {i+1}: vina {sc}  RMSD to crystal {T['resname']} chain {T['chain']} = {poses.heavy_rmsd(blk, xtal, SMI):.2f} A")

t0 = time.time(); c = calc.get_calculator(mlip); print(f"{mlip} load {time.time()-t0:.1f}s")
print(f"process RSS after load: {psutil.Process().memory_info().rss/2**30:.2f} GB")

blk = xtal if which == "crystal" else docked[0][0]
sysm = build_system(T["receptor"], ligand_with_h(blk, SMI))
print(f"{target} {which}: pocket {sysm.n_pocket} atoms, ligand {sysm.n_lig}, fixed {len(sysm.fixed)}, caps {len(sysm.caps)}, charge {sysm.charge}")
ctx = PoseContext(system=sysm, calc=c, thresholds=Thresholds(), workdir=T["workdir"])
t0 = time.time(); a = ctx.fresh(); a.get_potential_energy(); print(f"one single point: {time.time()-t0:.1f}s")
print("workdir:", ctx.workdir, "| relax fmax", ctx.relax_fmax, "steps", ctx.relax_steps, flush=True)
info = checks.relax_caps(ctx); print("stage 1 (caps only):", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()}, flush=True)
for fn in (checks.clash_count, checks.force_residuals, checks.relax_displacement, checks.ligand_strain, checks.contact_persistence):
    r = fn(ctx); print(r.line(), f"({r.cost_s:.0f}s)", flush=True)
print("verdict:", checks.verdict(ctx.results))
print(f"RSS at end {psutil.Process().memory_info().rss/2**30:.2f} GB")
