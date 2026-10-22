"""Pipeline driver: set_target -> generate -> filter -> dock -> validate -> report on one run state (runs/<run_id>/).
Every step saves its results into state.json as it goes; finished steps are skipped on rerun, and dock / validate resume
per candidate. The report (runs/<run_id>/report.html) is rebuilt from the state after each run.

usage: python -m medrag.pipeline RUN_ID --pdb 2A3R.pdb --resname LDP --chain A --seed '[NH3+]CCc1ccc(O)c(O)c1' \\
           [--steps target,generate,filter,dock,site,validate,boltz,report] [--mlip polar1-s] [--validate-top 5]"""
from __future__ import annotations

import argparse
from pathlib import Path

from medrag import ui
from medrag.report import Report, open_in_browser
from medrag.state import RunState
from medrag.steps import report_steps
from medrag.steps.discover import find_structure
from medrag.steps.boltz import boltz_finalists
from medrag.steps.dock import dock_candidates
from medrag.steps.filter import filter_candidates
from medrag.steps.generate import generate
from medrag.steps.site import assess_poses
from medrag.steps.target import set_target
from medrag.steps.validate import regrade_validation, validate_candidates

ALL = ("target", "generate", "filter", "dock", "site", "validate", "boltz", "report")


def build_report(state: RunState, open_browser: bool = False) -> str:
    """Write runs/<id>/report.html (and open it in the browser if asked); returns its path."""
    t0 = ui.step_start("report", "PyMOL stills + 3Dmol views (stills are cached between runs)")
    t = state.target
    rep = Report(state.dir, title=f"{t.get('name', state.data['run_id'])} · hit-to-lead", receptor=t.get("receptor"),
                 subtitle=f"Seed molecule: {state.data['seed']}", reference_sdf=t.get("crystal_ligand"))
    chips, stats = report_steps.header(state)
    rep.set_header(chips=chips, stats=stats)
    report_steps.add_generate_section(rep, state)
    if any("filter" in r for r in state.candidates.values()):
        report_steps.add_filter_section(rep, state)
    if any("dock" in r for r in state.candidates.values()):
        report_steps.add_dock_section(rep, state)
    if any("validate" in r for r in state.candidates.values()):
        report_steps.add_validate_section(rep, state)
    if any(r.get("boltz", {}).get("status") == "ok" for r in state.candidates.values()):
        report_steps.add_boltz_section(rep, state)
    path = str(Path(rep.write()).resolve())
    opened = open_browser and open_in_browser(path)
    ui.step_done("report", path + ("  (opened in your browser)" if opened else "  (open it in a browser)" if open_browser else ""), t0)
    return path


def run(run_id: str, pdb: str | None = None, resname: str | None = None, chain: str | None = None, seed: str | None = None,
        steps=ALL, mlip: str = "polar1-s", validate_top: int = 5, root: str = "runs", **kw) -> RunState:
    st = RunState(run_id, root)
    ui.set_run(run_id)
    if kw.get("protein") and not st.target.get("pdb"):          # protein name only: find the structure, ligand and seed
        find_structure(st, kw["protein"], seed=seed)
    seed = seed or st.data["seed"]
    for s in steps:
        if s == "target":
            if pdb:
                set_target(st, pdb, resname, chain, name=kw.get("name", ""), ligand_smiles=kw.get("crystal_smiles") or seed)
        elif s == "generate":
            if not st.step_done("generate"):
                generate(st, seed, **{k: kw[k] for k in ("sources", "percent", "bioisostere_top", "chembl_id") if kw.get(k)})
        elif s == "filter":
            filter_candidates(st)
        elif s == "dock":
            dock_candidates(st, max_dock=kw.get("max_dock"))
        elif s == "site":
            assess_poses(st, recall_cut=kw.get("recall_cut", 0.5), crystal_smiles=kw.get("crystal_smiles"))
        elif s == "validate":
            validate_candidates(st, mlip=mlip, top=validate_top, thresholds=kw.get("thresholds"))
        elif s == "boltz":
            if kw.get("boltz"):                                  # opt-in: it spends Modal credit
                boltz_finalists(st, backend=kw["boltz"])
        elif s == "regrade":
            r = regrade_validation(st)
            ui.note(f"re-graded validation against the seed's strain {r['seed_strain']}: {r['regraded_checks']} check(s) changed")
        elif s == "report":
            path = build_report(st, open_browser=kw.get("open_browser", True))
            if not ui.console.is_terminal:
                print("report:", path, flush=True)          # plain line for log watchers
        else:
            raise ValueError(f"unknown step {s!r}")
    return st


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id")
    ap.add_argument("--protein", help="protein or gene name (e.g. HMGCR): the structure, bound ligand, seed and ChEMBL data set are "
                    "found automatically (replaces --pdb/--resname/--chain/--seed)")
    ap.add_argument("--pdb"), ap.add_argument("--resname"), ap.add_argument("--chain"), ap.add_argument("--seed")
    ap.add_argument("--steps", default=",".join(ALL))
    ap.add_argument("--mlip", default="polar1-s")
    ap.add_argument("--validate-top", type=int, default=5)
    ap.add_argument("--thresholds")
    ap.add_argument("--gpt", action="store_true", help="add the mini-GPT as a third generate source; its data set is found automatically "
                    "(the target's ChEMBL IC50 set if big enough, else PubChem molecules similar to the seed)")
    ap.add_argument("--gpt-chembl", help="like --gpt but with a ChEMBL target id you choose")
    ap.add_argument("--boltz", choices=["modal"], help="co-fold the validated finalists with Boltz-2 on a Modal GPU (needs `modal setup`; costs a few cents)")
    ap.add_argument("--max-dock", type=int, help="dock at most this many candidates (best filter status / QED first)")
    ap.add_argument("--no-open", action="store_true", help="do not open the HTML report in the browser when it is built")
    ap.add_argument("--crystal-smiles", help="SMILES of the co-crystallized ligand (default: the seed)")
    ap.add_argument("--recall-cut", type=float, default=0.5, help="min fraction of the crystal ligand's contact residues a pose must contact")
    a = ap.parse_args()
    run(a.run_id, a.pdb, a.resname, a.chain, a.seed, steps=a.steps.split(","), mlip=a.mlip, validate_top=a.validate_top,
        thresholds=a.thresholds, crystal_smiles=a.crystal_smiles, recall_cut=a.recall_cut, open_browser=not a.no_open, max_dock=a.max_dock, protein=a.protein, boltz=a.boltz,
        **({"sources": ("genmask", "bioisostere", "gpt"), "chembl_id": a.gpt_chembl} if (a.gpt or a.gpt_chembl) else {}))
