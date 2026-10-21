"""JSON-friendly entry points onto MoDrAg's ChEMBL tools and mini-GPT fine-tune (medrag/lib/MoDrAg_CLI). The vendored code
reads and writes ../data, ../scratch and ../images relative to its cwd, so medrag.steps.generate gives each run its own
gpt/ tree (gpt/code as cwd, gpt/data holding symlinks to the foundation weights and vocab) and nothing lands in medrag/lib/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib/MoDrAg_CLI"))
import torch  # noqa: F401,E402  (MoDrAg imports torch before the native chem libs)


def chembl_target(protein_name: str) -> dict:
    """UniProt (human) -> ChEMBL target ids with their bioactive counts, using MoDrAg's uniprot_node / listbioactives_node.
    Both return (data, text, data) in an order their annotations do not state, so they are unpacked by what the code returns."""
    import pandas as pd
    from modrag_protein_functions import listbioactives_node, uniprot_node
    id_lists, up_text, _ = uniprot_node([protein_name], human_flag=True)
    ids = id_lists[0] if id_lists else []
    if not ids:
        return dict(protein=protein_name, uniprot=[], chembl=[], counts=[], text=up_text)
    # the search is free text: keep the entries whose gene name is the protein asked for (e.g. SULT1A3, not SULT1A4)
    df = pd.read_csv(f"../scratch/{protein_name}_uniprot_ids.tsv", sep="\t")
    exact = [r["Entry"] for _, r in df.iterrows() if r["Entry"] in ids and protein_name.upper() in str(r["Gene Names"]).upper().split()]
    pick = exact[:1] or ids[:1]
    counts, text, chembl_ids = listbioactives_node(pick)
    return dict(protein=protein_name, uniprot=pick, counts=counts, chembl=chembl_ids, text=text)


MIN_BIOACTIVES = 50     # fewer ChEMBL IC50 molecules than this is not enough to fine-tune on


def training_set(uniprot: str = "", seed_smiles: str = "", min_bioactives: int = MIN_BIOACTIVES) -> dict:
    """Choose the data set the mini GPT is fine-tuned on, with MoDrAg's own tools:
    1. the target's ChEMBL IC50 bioactives: uniprot_node resolves the accession (a PDB DBREF often carries a secondary one),
       listbioactives_node counts IC50 molecules per ChEMBL target; the best-populated target wins if it has >= min_bioactives;
    2. otherwise molecules similar to the seed from PubChem (related_node), saved as a SMILES-only data set (the GPT needs no
       activity values). These are neighbours of the seed, not known actives of the target.
    Returns {kind: 'chembl'|'similar'|'none', id, n, uniprot, tried: [[chembl_id, n_ic50], ...]}."""
    import hashlib

    import pandas as pd
    from modrag_molecule_functions import related_node
    from modrag_protein_functions import listbioactives_node, uniprot_node

    primary, pairs = uniprot, []
    if uniprot:
        ids, _, _ = uniprot_node([uniprot])
        primary = ids[0][0] if ids and ids[0] else uniprot
        counts, _, chembl = listbioactives_node([primary])
        if chembl and chembl[0]:
            pairs = sorted(([c, int(n)] for c, n in zip(chembl[0], counts[0])), key=lambda p: -p[1])
        if pairs and pairs[0][1] >= min_bioactives:
            return dict(kind="chembl", id=pairs[0][0], n=pairs[0][1], uniprot=primary, tried=pairs)
    if seed_smiles:
        lists, _, _ = related_node([seed_smiles])
        smiles = sorted({s for s in (lists[0] if lists else []) if s})
        if len(smiles) >= 10:
            gid = "SIM_" + hashlib.md5(seed_smiles.encode()).hexdigest()[:8].upper()     # gpt_node upper-cases the id
            pd.DataFrame({"SMILES": smiles}).to_csv(f"../scratch/{gid}_bioactives.csv", index=False)
            return dict(kind="similar", id=gid, n=len(smiles), uniprot=primary, tried=pairs)
    return dict(kind="none", id="", n=0, uniprot=primary, tried=pairs)


def generate(chembl_id: str) -> dict:
    """Fine-tune the mini GPT on the data set `chembl_id` (a ChEMBL target, or a SIM_ id from training_set) and sample new
    molecules (MoDrAg gpt_node). Molecules that are in the training set are dropped: they are not new."""
    import pandas as pd
    from gpt_node import gpt_node
    from rdkit import Chem
    smiles, text, _images = gpt_node(chembl_id)
    if not smiles:
        raise RuntimeError(f"gpt_node produced no molecules for {chembl_id}")
    canon = lambda x: Chem.MolToSmiles(m) if (m := Chem.MolFromSmiles(x)) is not None else None
    train = set()
    try:
        train = {canon(x) for x in pd.read_csv(f"../scratch/{chembl_id.upper()}_bioactives.csv")["SMILES"].dropna()}
    except Exception:
        pass
    fresh = [x for x in smiles if canon(x) not in train]
    return dict(molecules=[dict(smiles=x) for x in fresh], dropped=len(smiles) - len(fresh), n_generated=len(smiles))
