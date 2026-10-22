"""CafChem MedRAg agent: a MoDrAg-style command-line assistant that drives the hit-to-lead pipeline
(set_target -> generate -> filter -> dock -> site -> validate -> [boltz] -> report) through tool calls.

    python -m medrag.agent --run sult_e2e [--backend ollama|anthropic] [--model NAME] [--print]
    python -m medrag.agent --run sult_e2e --once "Filter the candidates and show me the best five."

All results live in runs/<run>/state.json (see medrag.state), so a session can be closed and resumed with the same --run.
Tool signatures are kept minimal on purpose: run-wide settings (MLIP, thresholds, cut-offs) are fixed in code, never tool
arguments the model could invent."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from medrag import pipeline, ui
from medrag.report import open_in_browser
from medrag.llm import DEFAULT_MODELS, make_backend, run_turn
from medrag.state import RunState
from medrag.steps.boltz import boltz_finalists as _boltz_finalists
from medrag.steps.dock import dock_candidates as _dock_candidates
from medrag.steps.filter import filter_candidates as _filter_candidates
from medrag.steps.discover import find_structure as _find_structure
from medrag.steps.generate import find_chembl_target as _find_chembl_target
from medrag.steps.generate import generate as _generate
from medrag.steps.site import assess_poses as _assess_poses
from medrag.steps.target import set_target as _set_target

ROOT = Path(__file__).resolve().parents[1]
MLIP = "polar1-s"            # MACE-POLAR-1 small: faster than UMA with no worse discrimination (SULT1A3 test)
STATE: RunState | None = None
QUIET = False                # --quiet hides tool calls and results (the steps' own status lines still print)
OPEN_BROWSER = True          # --no-open turns this off
_OPENED = False              # open the report tab once per session; later refreshes just overwrite the file (reload the tab)
STATE_CHANGING = {"set_target", "generate_analogues", "generate_with_gpt", "filter_candidates", "dock_candidates", "check_binding_site", "run_boltz"}


def _st() -> RunState:
    if STATE is None:
        raise RuntimeError("no run is open")
    return STATE


def _table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join("" if c is None else (f"{c:.3g}" if isinstance(c, float) else str(c)) for c in r) + " |" for r in rows]
    return "\n".join(out)


# --- tools ---------------------------------------------------------------------------------------------

def run_status() -> str:
    """Show what the current run contains: target, seed, which steps are done, and how many candidates are at each stage."""
    s = _st()
    c = s.candidates.values()
    lines = [f"run {s.data['run_id']}  seed: {s.data['seed']}",
             f"target: {s.target.get('name')} ({s.target.get('resname')} chain {s.target.get('chain')}), "
             f"expected site residues: {s.target.get('expected_residues')}",
             "steps done: " + (", ".join(x["step"] for x in s.data["steps"]) or "none"),
             f"candidates: {len(s.candidates)}; filtered: {sum('filter' in r for r in c)}; "
             f"docked: {sum(r.get('dock', {}).get('status') == 'ok' for r in c)}; "
             f"in site: {sum(bool(r.get('dock', {}).get('in_site')) for r in c)}; "
             f"validated: {sum(r.get('validate', {}).get('status') == 'ok' for r in c)}; "
             f"Boltz co-folded: {sum(r.get('boltz', {}).get('status') == 'ok' for r in c)}"]
    return "\n".join(lines)


def find_target(protein_name: str, seed_smiles: str = "") -> str:
    """Start from a protein name alone. Finds the human UniProt entry, searches the PDB for structures of that protein, picks one
    with a good-resolution, drug-like co-crystallized ligand, and sets the binding site from where that ligand sits. It also
    checks ChEMBL for a data set the GPT generator could learn from. The crystal ligand only identifies the site: by default it
    is also used as the seed (at pH 7.4), but the user can give their own seed molecule. Use this first when the user names only
    a protein.

    Args:
      protein_name: gene or protein name, e.g. HMGCR or SULT1A3
      seed_smiles: optional SMILES of the molecule to optimise (leave empty to start from the crystal ligand)
    """
    r = _find_structure(_st(), protein_name, seed=seed_smiles or None)
    c = r["chembl"]
    return (f"Binding site set from PDB {r['pdb_id']} ({r['resolution']} A), bound ligand {r['resname']} chain {r['chain']}; "
            f"seed {r['seed']}{' (given by the user)' if seed_smiles else ' (the crystal ligand)'}. UniProt {r['uniprot']}. "
            f"ChEMBL: {c['best_target']} has {c['n_ic50']} IC50 molecules "
            f"({'usable for generate_with_gpt' if c['usable'] else 'too few, generate_with_gpt would use PubChem neighbours of the seed'}).")


def set_target(pdb_path: str, ligand_resname: str, chain: str, seed_smiles: str) -> str:
    """Set the protein target. The co-crystallized ligand (3-letter residue name and chain in the PDB file) defines the
    binding site; the seed is the molecule to start the optimisation from (usually the same ligand).

    Args:
      pdb_path: path to the crystal structure PDB file
      ligand_resname: three-letter residue name of the co-crystallized ligand, e.g. LDP
      chain: chain id of that ligand, e.g. A
      seed_smiles: SMILES of the seed molecule, with the charge state it has at pH 7.4
    """
    s = _st()
    t = _set_target(s, pdb_path, ligand_resname, chain, ligand_smiles=seed_smiles)
    s.data["seed"] = seed_smiles
    s.save()
    return f"Target set. Box centre {t['center']}, receptor {t['receptor']}"


def generate_analogues() -> str:
    """Propose analogues of the seed with GenMask (masked-language-model SMILES infilling) and bioisostere_assist
    (shape-matched substituent swaps). Adds them to the run; run filter_candidates next."""
    s = _st()
    r = _generate(s, s.data["seed"])
    return f"Generated: {r['new']} new candidates ({r['total']} total). Errors: {r['errors'] or 'none'}"


def find_chembl_target(protein_name: str) -> str:
    """Look up a protein's human UniProt entry and its ChEMBL target ids, with the number of IC50 bioactivities in ChEMBL for
    each. Needed before generate_with_gpt, which can only train on targets that have IC50 data.

    Args:
      protein_name: gene or protein name, e.g. HMGCR
    """
    r = _find_chembl_target(_st(), protein_name)
    pairs = list(zip(r["chembl"][0], r["counts"][0])) if r.get("chembl") and r["chembl"][0] else []
    return f"UniProt {r['uniprot']}; ChEMBL targets with IC50 bioactivity counts: {pairs or 'none found'}"


def generate_with_gpt(chembl_id: str = "") -> str:
    """Generate new molecules with the mini GPT. Leave chembl_id empty: the tool finds a data set itself, using the target's
    ChEMBL IC50 bioactives when there are at least 50, and otherwise molecules similar to the seed from PubChem. It then
    fine-tunes the GPT on that set (a few minutes) and samples from it; molecules already in the training set are dropped.
    These are new compounds in the neighbourhood of the data set, not analogues of the seed.

    Args:
      chembl_id: optional ChEMBL target id to use instead (see find_chembl_target); leave empty for the automatic choice
    """
    s = _st()
    r = _generate(s, s.data["seed"], sources=("gpt",), chembl_id=chembl_id or None)
    return f"GPT generated: {r['new']} new candidates ({r['total']} total). Errors: {r['errors'] or 'none'}"


def filter_candidates() -> str:
    """Protonate candidates at pH 7.4 and grade them on drug-likeness (QED, Lipinski), structural alerts (PAINS) and
    ADMET-AI predictions (hERG, AMES, DILI, carcinogenicity, skin reaction, LD50, CYP inhibition, bioavailability, BBB).
    Each candidate gets pass / warn / fail; candidates that fail are not docked."""
    return f"Filter result (counts): {_filter_candidates(_st())}"


def dock_candidates() -> str:
    """Dock the seed and every candidate that did not fail the filter with AutoDock Vina into the crystal-ligand site.
    Takes a few minutes for a few dozen molecules. Vina scores alone do not tell you whether a pose is in the right site:
    run check_binding_site next."""
    r = _dock_candidates(_st())
    return f"Docked {r['docked']} molecules ({r['failed']} failed)."


def check_binding_site() -> str:
    """Check, for every docked pose, which residues it contacts and what fraction of the crystal ligand's contact residues
    it reproduces (recall). Poses with recall >= 0.5 count as being in the expected binding site, and only those are
    validated. Run after dock_candidates."""
    r = _assess_poses(_st())
    return f"Expected residues {r['expected']}. {r['in_site']} of {r['candidates']} docked candidates have a pose in the site."


def show_candidates(stage: str = "docked", n: int = 10) -> str:
    """Show candidates as a table, best first.

    Args:
      stage: one of 'filtered' (all candidates with their filter status), 'docked' (ranked by Vina score of the best in-site pose, else the best pose), 'validated' (MLIP feasibility results) or 'boltz' (Boltz-2 co-folding results)
      n: maximum number of rows
    """
    s = _st()
    rows = []
    if stage == "filtered":
        for r in s.candidates.values():
            f = r.get("filter")
            if f and "props" in f:
                rows.append([r["species"], f["status"], f["props"]["qed"], f["props"]["tanimoto_to_seed"], "; ".join(f["reasons"])])
        rows.sort(key=lambda x: ({"pass": 0, "warn": 1, "fail": 2}[x[1]], -x[2]))
        return _table(rows[:n], ["species", "status", "qed", "sim to seed", "flags"])
    if stage == "docked":
        for r in s.candidates.values():
            d = r.get("dock")
            if d and d.get("status") == "ok":
                j = d["in_site"][0] if d["in_site"] else 0
                p = d["poses"][j]
                rows.append([d["species"], p["score"], j + 1, p.get("site_recall"), "yes" if d["in_site"] else "NO", r["filter"]["status"] if "filter" in r else "seed"])
        rows.sort(key=lambda x: (x[4] != "yes", x[1]))
        return _table(rows[:n], ["species", "vina", "pose", "site recall", "in site", "filter status now"])
    if stage == "validated":
        for r in s.candidates.values():
            v = r.get("validate")
            if v and v.get("status") == "ok":
                c = v["checks"]
                rows.append([r["dock"]["species"], v["verdict"], c["ligand_strain"]["value"], c["force_residuals"]["value"],
                             c["contact_persistence"]["value"], "; ".join(v["reasons"])[:120]])
        return _table(rows[:n], ["species", "verdict", "strain kcal/mol", "force eV/A", "contacts", "flags"]) if rows else "Nothing validated yet."
    if stage == "boltz":
        for r in s.candidates.values():
            b = r.get("boltz")
            if b and b.get("status") == "ok":
                rows.append([r["dock"]["species"], "SEED (the known crystal ligand)" if r["smiles"] == s.data["seed"] else "candidate", b["pic50"], b["ic50_uM"],
                             b["dG_kcal"], b["binder_prob"], b["confidence"], r.get("validate", {}).get("verdict", "")])
        rows.sort(key=lambda x: -x[5])
        return _table(rows[:n], ["species", "role", "pIC50", "IC50 uM", "dG kcal/mol", "binder prob", "confidence", "MLIP verdict"]) if rows else "No Boltz results yet."
    return "stage must be 'filtered', 'docked', 'validated' or 'boltz'"


def start_validation(top: int = 3) -> str:
    """Start MLIP feasibility validation in the BACKGROUND for the seed and the top-scoring in-site candidates. It takes
    roughly 8-10 minutes per molecule, so this returns immediately; call validation_status to follow it. Validation is a
    sanity filter (strain, clashes, forces, lost contacts), not a rescoring and not a test of whether the pose is correct.

    Args:
      top: how many of the best-scoring in-site candidates to validate besides the seed
    """
    s = _st()
    log = s.dir / "validate.log"
    cmd = [sys.executable, "-u", "-m", "medrag.pipeline", s.data["run_id"], "--steps", "validate,report",
           "--validate-top", str(int(top)), "--mlip", MLIP, "--no-open"]
    prefix = ["systemd-run", "--user", "--scope", "-p", "MemoryMax=6G", "-p", "MemorySwapMax=0"] if shutil.which("systemd-run") else []
    if shutil.which("systemd-inhibit"):      # validation takes an hour or more: a suspended machine would freeze or kill it
        prefix = ["systemd-inhibit", "--what=sleep:idle", "--who=medrag", "--why=MedRAg validation job", "--mode=block"] + prefix
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    subprocess.Popen(prefix + cmd, cwd=ROOT, env=env, stdout=open(log, "ab"), stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    return f"Validation started in the background (log: {log}). Poll with validation_status."


def validation_status() -> str:
    """Report whether the background validation job is still running and how many molecules have results so far."""
    s = _st()
    s.data = json.loads(s.path.read_text())          # the job writes the state file; re-read it
    done = sum(r.get("validate", {}).get("status") == "ok" for r in s.candidates.values())
    err = [r["dock"]["species"] for r in s.candidates.values() if r.get("validate", {}).get("status") == "error"]
    running = subprocess.run(["pgrep", "-f", f"medrag.pipeline {s.data['run_id']} --steps validate"], capture_output=True).returncode == 0
    tail = (s.dir / "validate.log").read_text()[-400:] if (s.dir / "validate.log").exists() else ""
    note = ""
    if not running and done:                      # the job is finished: bring the report up to date and show it
        path, opened = refresh_report()
        note = f"\nThe validation job has finished; the report was refreshed ({'opened in the browser' if opened else 'reload the tab'}): {path}"
    return f"running: {running}; validated so far: {done}; errors: {err or 'none'}{note}\nlog tail: {tail}"


def run_boltz() -> str:
    """Co-fold the validated finalists (the seed and the molecules that went through validate) with Boltz-2 on a Modal GPU,
    which predicts each protein-ligand complex and its affinity (binder probability, pIC50, IC50, dG). It is an independent
    second opinion on the finalists, not a rescoring of the docking. It costs real money on the user's Modal account (a few
    cents per molecule, about 2-3 minutes in total), so call it ONLY when the user asks for Boltz or co-folding. Molecules that
    already have a result are skipped, so repeating it is free. Run it after validation has finished."""
    s = _st()
    if not any(r.get("validate") for r in s.candidates.values()):
        return "Nothing has been validated yet: Boltz runs on the validated finalists. Run start_validation first."
    r = _boltz_finalists(s, backend="modal")
    done = sum(x.get("boltz", {}).get("status") == "ok" for x in s.candidates.values())
    return f"Boltz-2 results exist for {done} finalists. Show them with show_candidates('boltz')." if done else \
        "Boltz did not run (no Modal login: the user needs to run `modal setup` once) or every call failed; see the step output."


def refresh_report() -> tuple[str, bool]:
    """Rebuild the HTML report; open it in the browser the first time this session. Returns (path, opened_now)."""
    global _OPENED
    first = OPEN_BROWSER and not _OPENED
    path = pipeline.build_report(_st(), open_browser=first)
    _OPENED = _OPENED or first
    return path, first


def build_report() -> str:
    """Rebuild the HTML report for the run (tables, PyMOL stills and rotatable 3Dmol views for every finished step) and show
    it in the user's browser. Returns the file path."""
    path, opened = refresh_report()
    return f"Report written to {path}. " + ("Opened in the browser." if opened else "Already open in the browser: reload the tab.")


TOOLS = [run_status, find_target, set_target, generate_analogues, find_chembl_target, generate_with_gpt, filter_candidates, dock_candidates, check_binding_site,
         show_candidates, start_validation, validation_status, run_boltz, build_report]

SYSTEM = f"""You are MedRAg, CafChem's medicinal chemistry agent, a hit-to-lead drug design assistant. You drive one pipeline with these tools:
{', '.join(t.__name__ for t in TOOLS)}.
Order: find_target (when the user gives only a protein name) or set_target (when they give a PDB file) -> generate_analogues (optionally also find_chembl_target + generate_with_gpt) -> filter_candidates -> dock_candidates -> check_binding_site -> start_validation
-> build_report (the report opens in the user's browser). Call run_status first if you are unsure what has been done.

Ground rules:
- Report numbers only from tool output; never invent scores, residues or SMILES. If a tool errors, say so and what you tried.
- Vina scores on this kind of small ligand do not separate the right site from the wrong one: judge a pose by the
  check_binding_site recall (fraction of the crystal ligand's contact residues reproduced), and mention it next to any score.
- The MLIP validation is a sanity filter for strain, clashes, forces and lost contacts. It is not a rescoring and not a
  correctness test; do not present its verdicts as binding predictions.
- A candidate keeps its docking result even if a later filter run changes its filter status, so a 'fail' can appear next to a docked pose; say so, do not call it undocked.
- Filter cut-offs are provisional. Say so when a filter outcome drives a conclusion.
- run_boltz spends the user's Modal credit: call it only when the user asks for Boltz or co-folding, and never twice in a row. Its results are a second opinion (binder probability, pIC50), not a ranking. Identify the seed only from the 'role' column of show_candidates('boltz'); never guess which row is the seed. Quote the seed's own Boltz scores when comparing.
- start_validation is slow (~8-10 min per molecule) and runs in the background; do not start it twice, poll validation_status.
- Be concise."""


PLANNER_SYSTEM = """You break a user's request to a drug-design pipeline assistant into the concrete tasks it must accomplish.
Reply with JSON only: {"tasks": ["...", "..."]}. Each task is one checkable thing the user asked for (an action to take, a
number to report, a question to answer). Use only what the request asks for; do not add extra steps or quality goals. A
simple request gives one or two tasks."""

REVIEWER_SYSTEM = """You check whether an assistant's reply accomplished every task from the user's original request.
You get the request, the task list, the assistant's reply and a log of the tools it actually called (with result snippets).
For each task decide: "done" (the reply accomplishes it AND the tool log supports any numbers or actions claimed),
"blocked" (could not be done and the reply honestly says why, or it is a slow background job that was started and reported as
running), or "missing" (not addressed, only claimed without a supporting tool call, or contradicted by the log).
Reply with JSON only: {"tasks": [{"task": "...", "status": "done|blocked|missing", "note": "<=15 words"}]}."""


def _json_from(text: str) -> dict:
    """First {...} block of a model reply, parsed; {} if there is none."""
    i, j = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[i:j + 1]) if 0 <= i < j else {}
    except json.JSONDecodeError:
        return {}


def plan(backend, request: str) -> list[str]:
    """Planner LLM: the request as a task checklist. Falls back to the request itself if the reply is unusable."""
    r = backend.step(PLANNER_SYSTEM, [{"role": "user", "content": request}], [])
    tasks = [str(t) for t in _json_from(r.text).get("tasks", []) if str(t).strip()]
    return tasks or [request]


def _tool_log(messages: list[dict]) -> str:
    """Compact record of the tool calls and result snippets in `messages` (this turn only)."""
    lines = []
    for m in messages:
        for c in m.get("tool_calls", []) or []:
            lines.append(f"CALL {c['name']}({json.dumps(c['args'])[:150]})")
        if m["role"] == "tool":
            lines.append(f"  -> {m['content'][:300]}")
    return "\n".join(lines) or "(no tools were called)"


def review(backend, request: str, tasks: list[str], reply: str, turn_messages: list[dict]) -> list[dict]:
    """Reviewer LLM: status of every planned task. If its reply cannot be parsed the tasks are marked 'unchecked'."""
    body = (f"REQUEST:\n{request}\n\nTASKS:\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(tasks))
            + f"\n\nASSISTANT REPLY:\n{reply}\n\nTOOL LOG:\n{_tool_log(turn_messages)}")
    r = backend.step(REVIEWER_SYSTEM, [{"role": "user", "content": body}], [])
    got = _json_from(r.text).get("tasks", [])
    if not got:
        return [dict(task=t, status="unchecked", note="reviewer reply unusable") for t in tasks]
    return [dict(task=str(g.get("task", "")), status=str(g.get("status", "unchecked")).lower(), note=str(g.get("note", ""))) for g in got]


def _checklist_md(items: list[dict]) -> str:
    mark = {"done": "✅", "blocked": "⏸️", "missing": "❌", "unchecked": "❔"}
    return "\n".join(f"- {mark.get(i['status'], '❔')} {i['task']}" + (f" — {i['note']}" if i.get("note") else "") for i in items)


def _auto_report(turn_messages: list[dict], on_event) -> str:
    """If this turn changed results (and the model did not already build the report), refresh the HTML report so the browser
    view is current. A failure here never loses the answer."""
    called = [c["name"] for m in turn_messages for c in (m.get("tool_calls") or [])]
    if not STATE_CHANGING.intersection(called) or "build_report" in called:
        return ""
    try:
        path, opened = refresh_report()
    except Exception as exc:
        on_event("report", f"refresh failed: {exc!r}")
        return f"\n\n(Report refresh failed: {exc})"
    return f"\n\n📄 Report {'opened in your browser' if opened else 'refreshed: reload the tab'}: {path}"


def chat(backend, messages: list[dict], prompt: str, verbose: bool, review_backend=None, max_rounds: int = 2) -> str:
    """One user turn. With review_backend: plan the request into tasks, let the agent answer the original message, have the
    reviewer check the reply against the tasks, and send anything missing back to the agent (at most `max_rounds` times)."""
    def on_event(kind, text):
        if QUIET:
            return
        if kind == "plan":
            ui.note("plan: " + text)
        elif kind == "call":
            ui.tool_call(text.replace("({})", "()"))
        elif kind == "result":
            ui.tool_result(text)
        elif kind == "review":
            st = [x[0] for x in json.loads(text)]
            ui.note("review: " + ", ".join(f"{st.count(k)} {k}" for k in ("done", "blocked", "missing", "unchecked") if k in st))
        elif kind == "thinking" and verbose:
            ui.note("thinking: " + text[:300].replace("\n", " "))
        elif kind == "report":
            ui.note(text)

    tasks = plan(review_backend, prompt) if review_backend else []
    if tasks:
        on_event("plan", " | ".join(tasks))
    start = len(messages)
    messages.append({"role": "user", "content": prompt})
    reply = run_turn(backend, SYSTEM, messages, TOOLS, on_event=on_event)
    if not review_backend:
        return reply + _auto_report(messages[start:], on_event)
    for rnd in range(max_rounds + 1):
        items = review(review_backend, prompt, tasks, reply, messages[start:])
        on_event("review", json.dumps([(i["status"], i["task"][:60]) for i in items]))
        missing = [i for i in items if i["status"] in ("missing", "unchecked")]
        if not missing or rnd == max_rounds:
            break
        messages.append({"role": "user", "content": "A reviewer checked your reply against the user's request and found these tasks "
                         "not accomplished:\n" + "\n".join(f"- {i['task']}: {i['note']}" for i in missing)
                         + "\nDo them now using the tools (or say plainly why you cannot), then give the complete answer."})
        reply = run_turn(backend, SYSTEM, messages, TOOLS, on_event=on_event)
    return reply + _auto_report(messages[start:], on_event) + "\n\n---\n**Checklist**\n" + _checklist_md(items)


def main(argv=None):
    global STATE, OPEN_BROWSER, QUIET
    ap = argparse.ArgumentParser(prog="python -m medrag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run id (runs/<id>/); an existing run is resumed")
    ap.add_argument("--backend", choices=["ollama", "anthropic"], default="ollama")
    ap.add_argument("--model", help=f"default: {DEFAULT_MODELS}")
    ap.add_argument("--print", dest="verbose", action="store_true", help="also show the model's thinking")
    ap.add_argument("--quiet", action="store_true", help="hide tool calls and results (step status lines still show)")
    ap.add_argument("--once", help="run a single prompt and exit")
    ap.add_argument("--no-open", action="store_true", help="do not open the HTML report in the browser")
    ap.add_argument("--no-review", action="store_true", help="skip the planner/reviewer pass (2 extra LLM calls per turn)")
    ap.add_argument("--planner-backend", choices=["ollama", "anthropic"], help="backend for the planner/reviewer (default: same as --backend)")
    ap.add_argument("--planner-model", help="model for the planner/reviewer (default: same as --model)")
    a = ap.parse_args(argv)
    OPEN_BROWSER = not a.no_open
    QUIET = a.quiet

    STATE = RunState(a.run)
    ui.set_run(a.run)
    backend = make_backend(a.backend, a.model)
    reviewer = None
    if not a.no_review:
        reviewer = backend if not (a.planner_backend or a.planner_model) else make_backend(a.planner_backend or a.backend, a.planner_model)
    chat_log = STATE.dir / "chat.jsonl"
    messages: list[dict] = []
    if chat_log.exists():                       # resume the conversation too, not only the pipeline state
        messages = [json.loads(l) for l in chat_log.read_text().splitlines() if l.strip()]
    n_saved = len(messages)

    def save():
        nonlocal n_saved
        with open(chat_log, "a") as fh:
            for m in messages[n_saved:]:
                fh.write(json.dumps(m, default=str) + "\n")
        n_saved = len(messages)

    from rich.console import Console
    from rich.markdown import Markdown
    console = Console(width=100)

    def turn(prompt):
        t0 = time.time()
        reply = chat(backend, messages, prompt, a.verbose, review_backend=reviewer)
        save()
        ui.console.print(f"\n[{ui.MAGENTA}]Response {(time.time() - t0) / 60:.2f}m >[/]")
        console.print(Markdown(reply))

    def finish():
        """Leave the report on screen: open it if this session never did and there is one."""
        rp = STATE.dir / "report.html"
        if OPEN_BROWSER and not _OPENED and rp.exists():
            ui.step_done("report", f"{rp.resolve()}  (opened in your browser)" if open_in_browser(rp) else str(rp.resolve()))

    if a.once:
        turn(a.once)
        finish()
        return
    ui.banner(a.run, a.backend, backend.model)
    if STATE.data["seed"] or STATE.target:
        ui.console.print(run_status(), style=ui.DIM)
    else:
        ui.console.print("New run: ask me to set the target (a PDB file, the co-crystallized ligand and a seed SMILES).", style=ui.DIM)
    ui.console.print("Type `quit` to exit.", style=ui.ORANGE)
    while True:
        try:
            prompt = input("\n\033[1;36mWhat next? > \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt in ("quit", "exit"):
            break
        if prompt:
            turn(prompt)
    finish()
    ui.console.print("Goodbye!", style=ui.MAGENTA)


if __name__ == "__main__":
    main()
