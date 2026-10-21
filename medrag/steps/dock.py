"""dock step: AutoDock Vina into the crystal-ligand site, through the vendored dock_assist.dock_at_centroid (isolated
process). Candidates go in small chunks and the state is saved after each chunk, so an interrupted run resumes.
The seed is docked too, as the reference. Poses are kept per candidate (top 3) and judged by Vina score and by whether
they sit in the crystal site (centroid within `site_radius` A of the crystal ligand's centroid)."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
from rdkit import Chem

from medrag import ui
from medrag.state import RunState
from medrag.tools import poses, vendor

CHUNK = 6


def _centroid(block: str) -> np.ndarray:
    return Chem.MolFromMolBlock(block, removeHs=False, sanitize=False).GetConformer().GetPositions().mean(axis=0)


def dock_candidates(state: RunState, exhaustiveness: int = 8, site_radius: float = 3.0, only: list[str] | None = None,
                    max_dock: int | None = None) -> dict:
    """Dock the seed and every non-failed candidate (`species` SMILES). Returns {docked, failed}.

    max_dock: dock at most this many candidates in total (already docked ones count), best first: pass before warn, then by
    QED. Vina takes ~40 s per molecule on a pocket the size of HMGCR's."""
    cwd = (state.dir / "dock").resolve()
    (cwd / "pdb_files").mkdir(parents=True, exist_ok=True)
    (cwd / "poses").mkdir(exist_ok=True)
    rec = cwd / "pdb_files" / "receptor.pdb"
    shutil.copy(state.target["receptor"], rec)
    center = np.array(state.target["center"])

    todo = []
    for smi, r in state.candidates.items():
        f = r.get("filter", {})
        is_seed = smi == state.data["seed"]
        if (is_seed or f.get("status") in ("pass", "warn")) and r.get("dock", {}).get("status") != "ok" and (only is None or smi in only):
            todo.append((smi, r.get("species") or smi))
    rank = lambda smi: ({"pass": 0, "warn": 1}.get(state.candidates[smi].get("filter", {}).get("status"), 2),
                        -state.candidates[smi].get("filter", {}).get("props", {}).get("qed", 0.0))
    todo.sort(key=lambda t: (t[0] != state.data["seed"], rank(t[0])))   # seed first, then best filter status / QED
    if max_dock is not None:
        n_done = sum(r.get("dock", {}).get("status") == "ok" for r in state.candidates.values())
        todo = todo[:max(0, max_dock - n_done)]
    n_ok = n_fail = 0
    t0 = ui.step_start("dock", f"{len(todo)} molecules into the site (Vina, exhaustiveness {exhaustiveness}, chunks of {CHUNK})")
    with ui.progress("docking", len(todo)) as prog:
        for k in range(0, len(todo), CHUNK):
            chunk = todo[k:k + CHUNK]
            try:
                vendor.call("dock_assist/code", "vina_dock", "dock_at_centroid", cwd=cwd, receptor_pdb=str(rec),
                            smiles_list=[sp for _, sp in chunk], center=list(center), exhaustiveness=exhaustiveness, num_modes=9)
            except vendor.VendorError:
                raise                      # per-ligand failures come back in the report; an exception means setup is broken
            for j, (smi, sp) in enumerate(chunk):
                sdf = cwd / "pdb_files" / f"receptor_c{j}.sdf"
                rec_d = state.candidates[smi]
                if not sdf.is_file():
                    rec_d["dock"] = dict(status="failed", detail="no pose written")
                    n_fail += 1
                    continue
                dest = cwd / "poses" / f"{hashlib.md5(smi.encode()).hexdigest()[:12]}.sdf"   # stable across processes
                shutil.move(str(sdf), dest)
                plist = []
                for blk, score in poses.read_poses(str(dest)):
                    plist.append(dict(score=score, centroid_dist=round(float(np.linalg.norm(_centroid(blk) - center)), 2)))
                rec_d["dock"] = dict(status="ok", sdf=str(dest), species=sp, poses=plist, best_score=plist[0]["score"],
                                     in_site=[i for i, p in enumerate(plist) if p["centroid_dist"] <= site_radius])
                n_ok += 1
            state.save()
            prog.advance(chunk[-1][1], len(chunk))
    state.log_step("dock", docked=n_ok, failed=n_fail, exhaustiveness=exhaustiveness, site_radius=site_radius)
    scores = [r["dock"]["best_score"] for r in state.candidates.values() if r.get("dock", {}).get("status") == "ok"]
    ui.step_done("dock", f"{n_ok} docked, {n_fail} failed" + (f" | {len(scores)} docked in total, best Vina {min(scores)} kcal/mol" if scores else ""), t0)
    return dict(docked=n_ok, failed=n_fail)
