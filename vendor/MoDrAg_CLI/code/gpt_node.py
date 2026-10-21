"""
MoDrAg GPT node: fine-tunes the mini SMILES GPT on a ChEMBL target's
bioactives and generates novel molecules. MoDrAg-facing entry point wired
into modrag.py. getbioactives_node is imported from modrag_protein_functions;
bioactives are read from ../scratch/{chembl_id}_bioactives.csv (and fetched
first if absent, mirroring predict_node).
"""

import os

import pandas as pd

from finetune_gpt import finetune_gpt
from modrag_protein_functions import getbioactives_node

# Where the CLI looks for the rendered grid so it can surface it to the user
# (modrag.py watches this path's mtime after each turn).
CHAT_IMAGE_PATH = "../images/chat_image.png"


def gpt_node(chembl_id: str) -> (list[str], str, list):
    """Fine-tunes a mini GPT on the bioactive molecules of a ChEMBL target and
    generates novel drug-like molecules. Use this to propose new candidate
    molecules for a target you have a ChEMBL ID for.

      Args:
        chembl_id: a ChEMBL target ID, e.g. "CHEMBL213"

      Returns:
        smiles_list: generated SMILES strings
        gpt_string: summary of the generated molecules
        gpt_images: a list containing the grid image of the generated molecules
    """
    print("GPT tool")
    print("=" * 51)

    # Coerce common argument shapes into a single ChEMBL id string. The LLM
    # agent calls this node as gpt_node(**tc.function.arguments) and often
    # passes chembl_id as a list (["CHEMBL220"]), an int (220), or None
    # instead of a bare string. Without this, chembl_id.upper() below throws
    # AttributeError instantly -- and it sits outside the try/except below,
    # so the node's safety net never catches it and the whole turn crashes.
    # Mirrors the bare-string coercion added to smiles_node (commit 6cdb59c).
    if isinstance(chembl_id, (list, tuple)):
        chembl_id = chembl_id[0] if chembl_id else ""
    if chembl_id is None:
        chembl_id = ""
    chembl_id = str(chembl_id).strip().strip('"').strip("'").upper()

    if not chembl_id:
        print("gpt_node failed: no chembl_id argument provided")
        return [], "", [None]

    # Fetch the bioactives CSV if it hasn't been cached by a prior call.
    bioactives_csv = f"../scratch/{chembl_id}_bioactives.csv"
    if not os.path.exists(bioactives_csv):
        getbioactives_node([chembl_id])

    try:
        df = pd.read_csv(bioactives_csv)
        smiles_list, gpt_string, img = finetune_gpt(df, chembl_id)
    except Exception as exc:  # pragma: no cover - agent-facing safety net
        print(f"gpt_node failed: {exc}")
        return [], "", [None]

    # Surface the grid through the CLI's chat image path so modrag.py detects it.
    if img is not None:
        try:
            os.makedirs(os.path.dirname(CHAT_IMAGE_PATH), exist_ok=True)
            img.save(CHAT_IMAGE_PATH)
        except Exception as exc:  # pragma: no cover - image is best-effort
            print(f"Could not save GPT grid image: {exc}")

    return smiles_list, gpt_string, [img]