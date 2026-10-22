"""boltz step: Boltz-2 co-folding + binding affinity for the finalists (the molecules that went through validate), on a Modal GPU.

Boltz predicts the whole protein-ligand complex from sequence and SMILES, with no docking box, and scores affinity with its own head.
It is an independent second opinion on the docked candidates, not a replacement for them. Per finalist the results text (affinity json,
confidence json) and the structure (cif) are kept in runs/<id>/boltz/<molecule>/ and shown in the report. Needs a Modal login
(`modal setup`); the first finalist builds the alignment and the others reuse it. A rerun skips finished molecules."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from medrag import ui
from medrag.state import RunState

_AA = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
       "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
       "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C", "MSE": "M"}


def protein_sequence(receptor_pdb: str, chain: str) -> str:
    """One-letter sequence of the residues present in the receptor file for `chain` (the first protein chain if `chain` is absent)."""
    seqs: dict[str, list[str]] = {}
    seen = set()
    for l in Path(receptor_pdb).read_text().splitlines():
        if l.startswith("ATOM") and l[12:16].strip() == "CA":
            key = (l[21], l[22:27])
            if key not in seen:
                seen.add(key)
                seqs.setdefault(l[21], []).append(_AA.get(l[17:20].strip(), "X"))
    return "".join(seqs.get(chain) or next(iter(seqs.values())))


def modal_login_ok() -> bool:
    try:
        import modal
        return bool(modal.config.config.get("token_id"))
    except Exception:
        return False


def finalists(state: RunState) -> list[dict]:
    """Every molecule that completed validate (the seed is included as the reference)."""
    return [r for r in state.candidates.values() if r.get("validate", {}).get("status") == "ok"]


def _parse(files: dict[str, bytes], name: str, dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    for rel, data in files.items():
        if not rel.startswith("_msa/") and not rel.endswith(".npz"):          # keep the text results and the structure
            (dest / rel).write_bytes(data)
    aff = json.loads((dest / f"affinity_{name}.json").read_text())
    conf = json.loads((dest / f"confidence_{name}_model_0.json").read_text())
    y = aff["affinity_pred_value"]                                           # log10(IC50 in uM); Boltz docs: pIC50 = 6 - y
    pic50 = 6 - y
    return dict(status="ok", pic50=round(pic50, 2), ic50_uM=round(10 ** y, 3), dG_kcal=round(-1.364 * pic50, 2),
                binder_prob=round(aff["affinity_probability_binary"], 2), confidence=round(conf["confidence_score"], 3),
                iptm=round(conf["iptm"], 3), cif=str(dest / f"{name}_model_0.cif"), affinity_json=str(dest / f"affinity_{name}.json"))


def boltz_finalists(state: RunState, backend: str = "modal", gpu: str = "A10G") -> dict:
    if backend != "modal":
        raise ValueError("only the modal backend is available so far")
    todo = [r for r in finalists(state) if r.get("boltz", {}).get("status") != "ok"]
    t0 = ui.step_start("boltz", f"{len(todo)} finalists co-folded with Boltz-2 on a Modal {gpu} GPU (about 2 min each, a few cents)")
    if not todo:
        ui.step_done("boltz", "nothing to do", t0)
        return {}
    if not modal_login_ok():
        ui.warn("not logged in to Modal: run `modal setup` once (opens a browser), then rerun with --steps boltz,report")
        ui.step_done("boltz", "skipped (no Modal login)", t0)
        return {}
    from medrag.boltz_modal import app, cofold
    t = state.target
    seq = protein_sequence(t["receptor"], t.get("chain", "A"))
    d = state.dir / "boltz"
    d.mkdir(exist_ok=True)
    msa = d / "msa.csv"
    msa_ref = "./msa.csv"

    def job(rec, with_msa):
        name = "m" + hashlib.md5(rec["smiles"].encode()).hexdigest()[:10]
        prot = f"      id: A\n" + (f"      msa: {msa_ref}\n" if with_msa else "") + f"      sequence: {seq}\n"
        yaml_text = (f"version: 1\nsequences:\n  - protein:\n{prot}  - ligand:\n      id: B\n      smiles: '{rec['smiles']}'\n"
                     "properties:\n  - affinity:\n      binder: B\n")
        return yaml_text, ({msa_ref: msa.read_bytes()} if with_msa else {}), name

    summary = {}
    with app.run():
        if not msa.is_file():                                               # the first finalist builds the alignment, the rest reuse it
            first = todo[0]
            ui.note(f"building the alignment and co-folding {first['smiles'][:50]} ...")
            y, f, n = job(first, False)
            out = cofold.remote(y, f, n)
            msa.write_bytes(next(v for k, v in out.items() if k.startswith("_msa/")))
            first["boltz"] = _parse(out, n, d / n)
            state.save()
            todo = todo[1:]
        jobs = [job(r, True) for r in todo]
        for rec, res in zip(todo, cofold.starmap(jobs, return_exceptions=True)):
            if isinstance(res, Exception):
                rec["boltz"] = dict(status="error", detail=f"{type(res).__name__}: {res}"[:300])
                ui.warn(f"{rec['smiles'][:50]}: {rec['boltz']['detail'][:100]}")
            else:
                rec["boltz"] = _parse(res, job(rec, True)[2], d / job(rec, True)[2])
            state.save()
    for r in finalists(state):
        b = r.get("boltz", {})
        if b.get("status") == "ok":
            summary[r["smiles"]] = b["pic50"]
            ui.note(f"{r['smiles'][:50]}: pIC50 {b['pic50']}, binder {b['binder_prob']}, confidence {b['confidence']}")
    state.log_step("boltz", backend=backend, gpu=gpu, results=summary)
    ui.step_done("boltz", f"{len(summary)} finalists co-folded", t0)
    return summary
