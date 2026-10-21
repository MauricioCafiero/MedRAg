"""validate step: MLIP pose feasibility checks (medrag.feasibility.checks.run_checks) on the best in-site pose of the top-scoring
docked candidates. This is a sanity filter (strain, clashes, forces, contacts), not a rescoring or a pose-correctness test:
on SULT1A3 it did not separate the right site from the wrong one. Each pose has its own workdir with checkpoints and a
result.json, so a rerun skips finished poses and resumes the unfinished one."""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

from medrag.feasibility import calc, checks
from medrag.feasibility.checks import CheckResult, PoseContext, verdict
from medrag.feasibility.system import build_system, ligand_with_h
from medrag.feasibility.thresholds import Thresholds
from medrag import ui
from medrag.state import RunState
from medrag.tools import poses


def pick_pose(rec: dict) -> int | None:
    """Index of the best-scoring pose that sits in the crystal site, else None."""
    d = rec.get("dock", {})
    return d["in_site"][0] if d.get("status") == "ok" and d.get("in_site") else None


def validate_candidates(state: RunState, mlip: str = "polar1-s", top: int = 5, thresholds: str | None = None) -> dict:
    """Run the checks on the seed (if its best pose is in the site) and on the `top` best-scoring candidates with an in-site pose."""
    ranked = sorted((r for r in state.candidates.values() if pick_pose(r) is not None),
                    key=lambda r: r["dock"]["poses"][pick_pose(r)]["score"])
    seed = state.data["seed"]
    chosen = [r for r in ranked if r["smiles"] == seed] + [r for r in ranked if r["smiles"] != seed][:top]
    th = Thresholds.load(thresholds) if thresholds else Thresholds()
    todo = [r for r in chosen if r.get("validate", {}).get("status") != "ok"]
    t0 = ui.step_start("validate", f"{len(todo)} molecules (the seed + the best in-site candidates), {mlip}, about 10 min each on a CPU")
    c = calc.get_calculator(mlip)
    summary = {}
    for k, rec in enumerate(todo, 1):
        ui.note(f"[{k}/{len(todo)}] relaxing the pocket and measuring strain for {rec['dock']['species'][:60]}")
        j = pick_pose(rec)
        blk = poses.read_poses(rec["dock"]["sdf"])[j][0]
        wd = state.dir / "validate" / f"{hashlib.md5(rec['smiles'].encode()).hexdigest()[:10]}_{re.sub(r'[^A-Za-z0-9]', '', rec['smiles'])[:16]}"   # keyed by molecule, not rank
        t0 = time.time()
        try:
            sysm = build_system(state.target["receptor"], ligand_with_h(blk, rec["dock"]["species"]))
            ctx = PoseContext(system=sysm, calc=c, thresholds=th, workdir=str(wd))
            verdict, reasons = checks.run_checks(ctx)
            (wd / "ligand_docked.sdf").write_text(ctx.ligand_block())
            (wd / "ligand_relaxed.sdf").write_text(ctx.ligand_block(relaxed=True))
            rec["validate"] = dict(
                status="ok", mlip=mlip, pose=j, verdict=verdict, reasons=reasons, workdir=str(wd), wall_s=round(time.time() - t0),
                checks={k: dict(value=r.value, unit=r.unit, status=r.status, detail=r.detail) for k, r in ctx.results.items()})
        except Exception as e:                         # one bad pose must not lose the rest of the batch
            rec["validate"] = dict(status="error", detail=f"{type(e).__name__}: {e}"[:400], pose=j)
        state.save()
        summary[rec["smiles"]] = rec["validate"].get("verdict", "error")
        v = rec["validate"]
        if v.get("status") == "ok":
            ui.verdict_line(rec["dock"]["species"], v["verdict"], v["checks"]["ligand_strain"]["value"],
                            rec["dock"]["poses"][v["pose"]]["score"], k, len(todo))
        else:
            ui.warn(f"{rec['dock']['species'][:50]}: {v.get('detail', 'error')[:100]}")
    regrade_validation(state, th)
    state.log_step("validate", mlip=mlip, top=top, thresholds=thresholds or "defaults", results=summary)
    final = [r["validate"]["verdict"] for r in state.candidates.values() if r.get("validate", {}).get("status") == "ok"]
    ui.step_done("validate", f"{len(final)} validated: " + ", ".join(f"{final.count(v)} {v}" for v in ("accept", "demote", "reject") if v in final)
                 + (f" | strain graded relative to the seed ({state.data['validate_seed_strain']:.1f} kcal/mol)"
                    if state.data.get("validate_seed_strain") is not None else ""), t0)
    return summary


def regrade_validation(state: RunState, thresholds: Thresholds | None = None, margin: float = 3.0) -> dict:
    """Grade every validated candidate from its stored check values, with the strain cut-off made relative to the seed
    (Thresholds.relative_to_seed) when the seed was validated. No MLIP work: safe to run again whenever results are added."""
    th = thresholds or Thresholds()
    seed = state.candidates.get(state.data["seed"], {}).get("validate", {})
    seed_strain = seed.get("checks", {}).get("ligand_strain", {}).get("value") if seed.get("status") == "ok" else None
    if seed_strain is not None:
        th = th.relative_to_seed(seed_strain, margin)
        state.data["validate_seed_strain"] = seed_strain
    changed = 0
    for rec in state.candidates.values():
        v = rec.get("validate")
        if not v or v.get("status") != "ok":
            continue
        results = {}
        for name, c in v["checks"].items():
            c.setdefault("status_absolute", c["status"])         # keep the grade under the plain defaults
            new = th.grade(name, c["value"]) if name in th.cuts else c["status"]
            changed += new != c["status"]
            c["status"] = new
            results[name] = CheckResult(name, c["value"], c["unit"], new, th.cuts.get(name, ()), c["detail"])
        v["verdict"], v["reasons"] = verdict(results)
        v["graded_vs_seed"] = seed_strain is not None
    state.save()
    return dict(seed_strain=seed_strain, regraded_checks=changed)
