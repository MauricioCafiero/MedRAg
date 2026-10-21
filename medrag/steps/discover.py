"""discover step: from a protein name alone to a ready target. With MoDrAg's tools it finds the human UniProt entry, searches RCSB
for structures, keeps those that are really that protein, lists their bound drug-like ligands (dock_assist's QED helper), takes the
best (good resolution, drug-like) whose RCSB chemical-component SMILES matches the crystal ligand, sets the run's target and seed
from it, and checks ChEMBL for a data set the mini GPT could learn from."""
from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from medrag import ui
from medrag.state import RunState
from medrag.steps.generate import ADAPTERS, gpt_workdir
from medrag.steps.target import set_target
from medrag.tools import vendor

GOOD_RESOLUTION = 2.8       # A; entries at or better than this are preferred over a nicer-looking ligand in a worse structure


def _workdir(state: RunState) -> Path:
    """<run>/discover/{code,pdb_files,scratch,images}: MoDrAg's tools read and write ../pdb_files etc. next to their cwd."""
    root = (state.dir / "discover").resolve()
    for d in ("code", "pdb_files", "scratch", "images"):
        (root / d).mkdir(parents=True, exist_ok=True)
    return root / "code"


def find_structure(state: RunState, protein_name: str, seed: str | None = None, max_pdbs: int = 10) -> dict:
    """Set the run's target (PDB file, co-crystallized ligand) and seed from a protein name. `seed` overrides the default seed,
    which is the crystal ligand at pH 7.4. Raises ValueError with the reason if nothing usable is found."""
    from medrag.steps.filter import protonate           # imported here: it pulls in cafprot and the ADMET machinery
    t0 = ui.step_start("discover", f"{protein_name}: UniProt entry, RCSB structures, bound ligands, ChEMBL data")
    cwd = _workdir(state)
    res = vendor.call(ADAPTERS, "discover_adapter", "candidates", cwd=cwd, protein_name=protein_name, max_pdbs=max_pdbs, timeout=1200)
    if not res["uniprot"]:
        raise ValueError(f"no human UniProt entry found for {protein_name!r}")
    mine = [p for p in res["pdbs"] if p["same_protein"]]
    ui.note(f"UniProt {res['uniprot']}; {len(res['pdbs'])} PDB entries checked, {len(mine)} are this protein")
    options = []
    for p in mine:
        for lig in vendor.call(ADAPTERS, "ligand_adapter", "ligands", cwd=cwd, pdb_path=p["path"]):
            options.append(dict(p, **lig))
    if not options:
        raise ValueError(f"none of the {len(mine)} {protein_name} structures has a bound drug-like ligand")
    good = lambda o: (o["resolution"] or 99.0) <= GOOD_RESOLUTION
    options.sort(key=lambda o: (not good(o), -(o["qed"] or 0.0), o["resolution"] or 99.0))
    ui.note(f"{len(options)} bound ligands in {len({o['pdb_id'] for o in options})} structures; best by resolution then QED: "
            + ", ".join(f"{o['pdb_id']}/{o['resname']} ({o['resolution'] or '?'} A, QED {o['qed']:.2f})" for o in options[:3]))
    chosen = crystal_smiles = None
    for o in options[:8]:
        smi = vendor.call(ADAPTERS, "discover_adapter", "ligand_smiles", cwd=cwd, resname=o["resname"], timeout=120)
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is not None and m.GetNumHeavyAtoms() == o["heavy"]:       # the SMILES describes the whole ligand seen in the crystal
            chosen, crystal_smiles = o, smi
            break
    if chosen is None:
        raise ValueError("no candidate ligand has a chemical-component SMILES matching its crystal atoms")
    species = protonate(crystal_smiles)
    ts = vendor.call(ADAPTERS, "gpt_adapter", "training_set", cwd=gpt_workdir(state), uniprot=res["uniprot"], seed_smiles="", timeout=900)
    best = ts["tried"][0] if ts["tried"] else None
    chembl = dict(uniprot=res["uniprot"], best_target=best[0] if best else None, n_ic50=best[1] if best else 0,
                  usable=ts["kind"] == "chembl", tried=ts["tried"])
    ui.note(f"ChEMBL: {best[0]} has {best[1]} IC50 molecules ({'enough for the GPT' if chembl['usable'] else 'too few; the GPT would use PubChem neighbours of the seed'})"
            if best else "ChEMBL: no target with bioactivity data found")
    set_target(state, chosen["path"], chosen["resname"], chosen["chain"], name=protein_name, ligand_smiles=crystal_smiles)
    state.set_target(pdb_id=chosen["pdb_id"], resolution=chosen["resolution"], uniprot=res["uniprot"], chembl=chembl,
                     alternatives=[dict(pdb_id=o["pdb_id"], resname=o["resname"], chain=o["chain"], resolution=o["resolution"], qed=o["qed"])
                                   for o in list({(o["pdb_id"], o["resname"]): o for o in reversed(options)}.values())[::-1]][:8])
    state.data["seed"] = seed or species
    state.save()
    state.log_step("discover", protein=protein_name, uniprot=res["uniprot"], pdb_id=chosen["pdb_id"], ligand=chosen["resname"],
                   resolution=chosen["resolution"], seed=state.data["seed"], chembl=chembl["best_target"], n_ic50=chembl["n_ic50"])
    ui.step_done("discover", f"{chosen['pdb_id']} ({chosen['resolution'] or '?'} A) with bound {chosen['resname']} chain {chosen['chain']}; seed {state.data['seed']}", t0)
    return dict(pdb_id=chosen["pdb_id"], resname=chosen["resname"], chain=chosen["chain"], resolution=chosen["resolution"],
                seed=state.data["seed"], uniprot=res["uniprot"], chembl=chembl)
