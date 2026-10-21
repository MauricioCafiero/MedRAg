"""site step: is a docked pose in the expected binding site? Expected = the residues the crystal ligand contacts (H-bonds,
hydrophobic, pi, halogen, salt bridges; ODDT through medrag/lib/FAO_MOLPROP_CLI's find_contacts). Each docked pose is scored by
recall = fraction of those residues it also contacts. On SULT1A3 this separated the right site (recall 1.00 / 0.71) from the
wrong one (0.14-0.29) where Vina scores and the MLIP feasibility checks did not (n = 6 poses; the 0.5 cut sits in that gap)."""
from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from medrag.feasibility.system import ligand_with_h
from medrag import ui
from medrag.state import RunState
from medrag.tools import poses, vendor

ADAPTERS = str(Path(__file__).resolve().parents[1] / "tools/adapters")


def _write_sdfs(d: Path, tag: str, blocks_smiles: list) -> list[str]:
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for j, (blk, smi) in enumerate(blocks_smiles):
        p = d / f"{tag}_{j}.sdf"
        w = Chem.SDWriter(str(p))
        w.write(ligand_with_h(blk, smi))          # ODDT's H-bond typing needs the full-H ligand
        w.close()
        paths.append(str(p))
    return paths


def expected_residues(state: RunState, crystal_smiles: str | None = None) -> list[str]:
    """Residues the crystal ligand contacts; cached in state.target['expected_residues']."""
    t = state.target
    if t.get("expected_residues"):
        return t["expected_residues"]
    smi = crystal_smiles or t.get("ligand_smiles") or state.data["seed"]
    blk = poses.crystal_ligand_block(t["pdb"], t["resname"], t["chain"])
    cwd = (state.dir / "site").resolve()
    (res,) = vendor.call(ADAPTERS, "site_adapter", "contacts_many", cwd=cwd, receptor_pdb=t["receptor"],
                         ligand_sdfs=_write_sdfs(cwd, "crystal", [(blk, smi)]))
    if "error" in res or not res["residues"]:
        raise RuntimeError(f"could not get the crystal ligand's contacts: {res}")
    state.set_target(expected_residues=res["residues"], ligand_smiles=smi)
    return res["residues"]


def assess_poses(state: RunState, recall_cut: float = 0.5, crystal_smiles: str | None = None) -> dict:
    """Score every docked pose by recall of the expected residues; in_site becomes the poses with recall >= recall_cut (the
    centroid-based list is kept as in_site_centroid)."""
    expected = expected_residues(state, crystal_smiles)
    cwd = (state.dir / "site").resolve()
    n = n_in = 0
    # docked candidates a later filter run failed are never validated, so they are not scored
    eligible = [(smi, rec) for smi, rec in state.candidates.items() if rec.get("dock", {}).get("status") == "ok"
                and not (smi != state.data["seed"] and rec.get("filter", {}).get("status") == "fail")]
    t0 = ui.step_start("site", f"{len(eligible)} docked molecules vs the {len(expected)} residues the crystal ligand contacts")
    prog_cm = ui.progress("site check", len(eligible))
    prog = prog_cm.__enter__()
    for smi, rec in eligible:
        d = rec["dock"]
        prog.advance(d["species"])
        if all("site_recall" in p for p in d["poses"]) and d.get("site_expected") == [expected, recall_cut]:
            n_in += bool(d["in_site"])                  # already scored against these expected residues: resume
            n += 1
            continue
        tag = Path(d["sdf"]).stem
        try:
            blocks = [(blk, d["species"]) for blk, _ in poses.read_poses(d["sdf"])]
            res = vendor.call(ADAPTERS, "site_adapter", "contacts_many", cwd=cwd, receptor_pdb=state.target["receptor"],
                              ligand_sdfs=_write_sdfs(cwd, tag, blocks))
        except Exception as e:                # one bad molecule must not stop the step: it is simply not in the site
            d.setdefault("in_site_centroid", d.get("in_site", []))
            d["in_site"] = []
            d["site_error"] = f"{type(e).__name__}: {e}"[:200]
            state.save()
            continue
        d.setdefault("in_site_centroid", d.get("in_site", []))
        for p, r in zip(d["poses"], res):
            got = set(r.get("residues", []))
            p["site_recall"] = round(len(got & set(expected)) / len(expected), 2) if "error" not in r else None
            p["contacted"] = sorted(got)
        d["in_site"] = [j for j, p in enumerate(d["poses"]) if (p["site_recall"] or 0) >= recall_cut]
        d["site_expected"] = [expected, recall_cut]
        n += 1
        n_in += bool(d["in_site"])
        state.save()
    prog_cm.__exit__(None, None, None)
    state.log_step("site", expected=expected, recall_cut=recall_cut, poses_assessed=n, candidates_in_site=n_in)
    ui.step_done("site", f"{n_in} of {n} docked molecules have a pose reproducing at least {recall_cut:.0%} of the crystal contacts", t0)
    return dict(expected=expected, candidates=n, in_site=n_in)
