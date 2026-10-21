"""Derive target-specific feasibility cut-offs from a run of tests/feas_poses.py and show how each check separates
the crystal / right-site poses from the wrong-site ones.
usage: calibrate.py <results_dir> [out.json]     (results_dir holds <pose>/result.json; 'crystal*' is the reference,
poses named in RIGHT are the correct-site redocks, the rest are decoys)."""
import json, sys
from pathlib import Path
from medrag.feasibility.thresholds import Thresholds, DEFAULTS, LOWER_IS_WORSE

RIGHT = {"dopamine_pose3", "paracetamol_pose3"}      # verified by RMSD / centroid distance to the crystal ligand

d = Path(sys.argv[1])
res = {p.parent.name: json.loads(p.read_text()) for p in sorted(d.glob("*/result.json"))}
ref_name = next(n for n in res if n.startswith("crystal"))
val = lambda n, k: res[n]["checks"][k]["value"] if k in res[n]["checks"] else None
ref = {k: val(ref_name, k) for k in DEFAULTS}
ref = {k: v for k, v in ref.items() if v is not None}

th = Thresholds.from_reference(ref)
out = sys.argv[2] if len(sys.argv) > 2 else str(d / "thresholds.json")
th.save(out)

checks = [k for k in DEFAULTS if k in ref]
print(f"{'pose':20s} {'site':6s} " + " ".join(f"{k[:14]:>14s}" for k in checks))
for n in res:
    site = "xtal" if n == ref_name else "right" if n in RIGHT else "wrong"
    print(f"{n:20s} {site:6s} " + " ".join(f"{val(n, k):14.2f}" for k in checks))
print("\ncut-offs (warn, fail): default -> crystal-calibrated")
for k in checks:
    print(f"  {k:20s} {DEFAULTS[k]} -> ({th.cuts[k][0]:.3g}, {th.cuts[k][1]:.3g})")
dec = [n for n in res if n not in RIGHT and n != ref_name]
print("\ndecoys failing at the calibrated cuts:",
      {k: sum(th.grade(k, val(n, k)) == "fail" for n in dec) for k in checks}, "of", len(dec))
print("saved", out)
