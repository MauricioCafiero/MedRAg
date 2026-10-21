"""generate step: propose analogues of a seed molecule. Sources are the vendored tools, run in isolated processes:
  genmask      - mask/unmask SMILES with the ChemTok BERT model (medrag/lib/GenMask)
  bioisostere  - shape-similar substituent swaps from the prebuilt 100k library (medrag/lib/bioisostere_assist)
  gpt          - MoDrAg's mini GPT fine-tuned on a ChEMBL target's IC50 bioactives (needs a ChEMBL id; new molecules for the
                 target, not analogues of the seed)"""
from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from medrag import ui
from medrag.state import RunState
from medrag.tools import vendor

ADAPTERS = str(Path(__file__).resolve().parents[1] / "tools/adapters")


def _canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None


def gpt_workdir(state: RunState) -> Path:
    """Per-run tree for the vendored GPT code, which reads/writes ../data, ../scratch and ../images next to its cwd. The foundation
    weights and vocab are symlinked, so the fine-tuned model and the bioactives cache land in the run, not in medrag/lib/."""
    root = (state.dir / "gpt").resolve()
    for d in ("code", "scratch", "images", "data"):
        (root / d).mkdir(parents=True, exist_ok=True)
    for f in ("GPT_ZN305_mini.pt", "vocab.txt", "vocab_305K.txt", "ZN305K_smiles.csv"):
        link = root / "data" / f
        if not link.exists():
            link.symlink_to(vendor.VENDOR / "MoDrAg_CLI/data" / f)
    return root / "code"


def _uniprot_from_pdb(state: RunState) -> str:
    """UniProt accession of the target chain from the PDB file's DBREF records ('' if there is none)."""
    if state.target.get("uniprot"):                    # set by discovery: the current primary accession
        return state.target["uniprot"]
    chain, found = state.target.get("chain", ""), []
    try:
        for line in open(state.target["pdb"]):
            if line.startswith("DBREF"):
                f = line.split()
                if "UNP" in f and f.index("UNP") + 1 < len(f):
                    found.append((f[2] if len(f) > 2 else "", f[f.index("UNP") + 1]))
    except OSError:
        return ""
    return next((acc for ch, acc in found if ch == chain), found[0][1] if found else "")


def find_chembl_target(state: RunState, protein_name: str) -> dict:
    """ChEMBL target ids and IC50 bioactive counts for a protein (MoDrAg uniprot_node + listbioactives_node)."""
    return vendor.call(ADAPTERS, "gpt_adapter", "chembl_target", cwd=gpt_workdir(state), protein_name=protein_name, timeout=600)


def generate(state: RunState, seed: str, sources=("genmask", "bioisostere"), percent: float = 0.10,
             bioisostere_top: int = 10, chembl_id: str | None = None) -> dict:
    """Add analogues of `seed` to the run state. Returns {source: n_new_candidates}. Failures of one source are recorded,
    not fatal."""
    cwd = state.dir / "generate"
    seed_c = _canon(seed)
    if seed_c is None:
        raise ValueError(f"bad seed SMILES {seed!r}")
    state.data["seed"] = seed_c
    state.add_candidate(seed_c, "seed")
    counts, errors = {}, {}
    t0 = ui.step_start("generate", f"seed {seed_c} | sources: {', '.join(sources)}")
    for src in sources:
        if src == "gpt":
            ui.note("mini GPT: choosing a data set, then fine-tuning (a few minutes)")
        try:
            if src == "genmask":
                rows = vendor.call(ADAPTERS, "genmask_adapter", "analogues", cwd=cwd, smiles=seed, percent=percent)
                meta = lambda r: dict(entropy=r["entropy"], qed=r["qed"], tanimoto_to_parent=r["tanimoto_to_parent"])
            elif src == "bioisostere":
                rows = vendor.call(ADAPTERS, "bioisostere_adapter", "analogues", cwd=cwd, lead=seed, top=bioisostere_top)
                meta = lambda r: dict(similarity=r["similarity"], lead_fragment=r["lead_fragment"],
                                      library_fragment=r["library_fragment"])
            elif src == "gpt":
                gid = chembl_id
                if gid:
                    ui.note(f"GPT training set: ChEMBL {gid} (given)")
                else:
                    ts = vendor.call(ADAPTERS, "gpt_adapter", "training_set", cwd=gpt_workdir(state), uniprot=_uniprot_from_pdb(state),
                                     seed_smiles=seed, timeout=900)
                    best = f"best ChEMBL target {ts['tried'][0][0]} has {ts['tried'][0][1]} IC50 molecules" if ts["tried"] else "no ChEMBL target found"
                    if ts["kind"] == "none":
                        raise ValueError(f"no training set found ({best}; PubChem had no neighbours of the seed)")
                    gid = ts["id"]
                    ui.note(f"GPT training set: {ts['n']} ChEMBL IC50 molecules of {gid}" if ts["kind"] == "chembl"
                            else f"GPT training set: {ts['n']} PubChem molecules similar to the seed ({best}; fewer than 50, so not enough to learn from)")
                out = vendor.call(ADAPTERS, "gpt_adapter", "generate", cwd=gpt_workdir(state), chembl_id=gid, timeout=3600)
                rows = out["molecules"]
                if out["dropped"]:
                    ui.note(f"gpt: dropped {out['dropped']} generated molecules that were in the training set (not new)")
                meta = lambda r: dict(training_set=gid)
            else:
                raise ValueError(f"unknown source {src!r}")
        except Exception as e:                       # one broken source must not lose the others
            errors[src] = f"{type(e).__name__}: {e}"
            ui.warn(f"{src} failed: {errors[src][:120]}")
            continue
        before = len(state.candidates)
        for r in rows:
            c = _canon(r["smiles"])
            if c is not None:
                state.add_candidate(c, src, parent=seed_c, **meta(r))
        counts[src] = len(state.candidates) - before
        ui.note(f"{src}: {len(rows)} proposed, {counts[src]} new unique molecules")
    state.log_step("generate", seed=seed_c, new=counts, errors=errors, total=len(state.candidates))
    ui.step_done("generate", f"{len(state.candidates) - 1} candidate molecules ({', '.join(f'{k} {v}' for k, v in counts.items())})", t0)
    return dict(new=counts, errors=errors, total=len(state.candidates))
