"""Build a real report from the 1HWL redock: dock step (poses, Vina score, RMSD to the crystal pose)."""
from pathlib import Path
from medrag.tools import poses
from medrag.report import Report

SMI = "CC(C)c1nc(N(C)S(C)(=O)=O)nc(-c2ccc(F)cc2)c1/C=C/C(O)CC(O)CC(=O)[O-]"
RUN = Path("runs/dock_clean2/pdb_files")
OUT = Path("runs/report_demo"); OUT.mkdir(parents=True, exist_ok=True)
docked = poses.read_poses(str(RUN / "HMGCR_atoms_c0.sdf"))
xtal = poses.crystal_ligand_block("runs/dock_test/pdb_files/HMGCR_1HWL.pdb", "FBI", "A")
(OUT / "crystal_FBI.sdf").write_text(xtal)
rows, plist = [], []
for i, (blk, sc) in enumerate(docked):
    r = poses.heavy_rmsd(blk, xtal, SMI)
    rows.append({"pose": i + 1, "vina (kcal/mol)": sc, "RMSD to crystal (A)": round(r, 2)})
    plist.append({"name": f"pose {i+1}", "sdf": blk, "caption": f"vina {sc}, RMSD {r:.2f} A"})
rep = Report(OUT, title="HMGCR (1HWL) redock demo", receptor=str(RUN / "HMGCR_atoms.pdb"),
             subtitle="rosuvastatin redocked into the FBI site; grey = crystal pose", reference_sdf=str(OUT / "crystal_FBI.sdf"))
rep.add_step("dock", "Docking", f"{len(docked)} poses from Vina at the FBI site (protein-only receptor).", table=rows, poses=plist)
print(rep.write())
