"""
MoDrAg GPT mini-fine-tune (PyTorch) — laptop-friendly rewrite of the original
TensorFlow/DeepChem prototype that previously lived at code/finetune_gpt.py.

Fine-tunes the 1-block *mini* foundation (../data/GPT_ZN305_mini.pt) on a set of
ChEMBL bioactives by appending a single new transformer block on top, training
it with the foundation frozen, then unfreezing everything and training a little
more. The result is a 2-block model (1 foundation + 1 new) — about half the size
of the 4-block Tyrosinase finetune in the parent CafChem repo — so the whole
pipeline runs on a small laptop.

This module reuses the toolkit in CafChemGPT.py (tokenize, train, generate) and
is intentionally separate from it; CafChemGPT.py is left unchanged. The
weight-copy logic in make_mini_finetune_gpt mirrors CafChemGPT.make_finetune_gpt
but loads the mini foundation instead of the hardcoded 2-block one.

Called from MoDrAg via gpt_node (see gpt_node.py):
    gpt_node(chembl_id)  ->  finetune_gpt(df, chembl_id)
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

# Module-level print flag - set from modrag.py (NODE_INTEGRATION_SKILLS.md).
print_flag = False

# CafChemGPT.py lives alongside this file in code/, so the plain import
# resolves with no sys.path surgery.
from CafChemGPT import (  # noqa: E402
    get_device, load_gpt, GPT, train_gpt, save_gpt, unfreeze_gpt,
    test_vocab, trim_vocab, make_datasets, make_prompts, gen_mols,
    _strip_salts, _draw_grid,
)

# ---- config ---------------------------------------------------------------
# Paths are relative to code/ (MoDrAg's cwd): data/ and scratch/ live at the
# repo root, hence the ../ prefix.
MINI_FOUNDATION_FILE = "../data/GPT_ZN305_mini"   # .pt appended by save_gpt
NUM_NEW_BLOCKS = 1                            # 1 foundation block + 1 new
BATCH_SIZE = 128
LR = 1e-3
FROZEN_EPOCHS = 25
UNFROZEN_EPOCHS = 25
N_PROMPTS = 50
PROMPT_LEN = 2
TEMP = 1.5
USE_RAMP = True
MAX_DATASET = 2000                            # cap bioactives, as legacy did


def make_mini_finetune_gpt(num_new_blocks=NUM_NEW_BLOCKS, freeze_old_layers=True,
                           foundation_file=MINI_FOUNDATION_FILE, device=None):
    """Load the 1-block mini foundation and append `num_new_blocks` freshly
    initialized transformer blocks on top, optionally freezing the foundation
    block + embedding.

    Mirrors CafChemGPT.make_finetune_gpt but loads the mini foundation (passed in
    here) instead of the hardcoded 2-block FOUNDATION_FILE — CafChemGPT.py is
    left unchanged, so the weight-copy logic is duplicated here.

        Args:
            num_new_blocks: new transformer blocks to add on top
            freeze_old_layers: freeze the foundation block + embedding
            foundation_file: mini foundation path (no .pt)
            device: torch device (default = get_device())
        Returns:
            gpt_ft: fine-tuning model
    """
    if device is None:
        device = get_device()

    ckpt = torch.load(foundation_file + ".pt", map_location=device,
                      weights_only=False)
    fcfg = ckpt["config"]
    foundation = load_gpt(foundation_file, fcfg["num_blocks"],
                          fcfg["max_length"], fcfg["vocab_size"], device=device)
    base_cfg = foundation.config()
    total_blocks = base_cfg["num_blocks"] + num_new_blocks

    gpt_ft = GPT(
        total_blocks, base_cfg["max_length"], base_cfg["vocab_size"],
        embed_dim=base_cfg["embed_dim"], num_heads=base_cfg["num_heads"],
        key_dim=base_cfg.get("key_dim"), ff_dim=base_cfg["ff_dim"],
        dropout_rate=base_cfg["dropout_rate"],
    ).to(device)

    # Copy embedding + first N blocks + head from the foundation; new blocks
    # keep their fresh init.
    new_state = gpt_ft.state_dict()
    old_state = foundation.state_dict()
    copied = []
    for key, val in old_state.items():
        if key in new_state and new_state[key].shape == val.shape:
            new_state[key] = val
            copied.append(key)
    gpt_ft.load_state_dict(new_state)
    if print_flag:
        print(f"Copied {len(copied)} parameter tensors from the mini foundation.")

    if freeze_old_layers:
        for p in gpt_ft.embedding.parameters():
            p.requires_grad = False
        for i in range(base_cfg["num_blocks"]):
            for p in gpt_ft.blocks[i].parameters():
                p.requires_grad = False
            if print_flag:
                print(f"setting transformer block {i} untrainable.")
        for p in gpt_ft.blocks[base_cfg["num_blocks"]:].parameters():
            p.requires_grad = True
        for p in gpt_ft.head.parameters():
            p.requires_grad = True
        if print_flag:
            print(f"setting {num_new_blocks} new block(s) + head trainable.")

    gpt_ft.summary()
    return gpt_ft


def finetune_gpt(df, chembl_id, foundation_file=MINI_FOUNDATION_FILE,
                 frozen_epochs=FROZEN_EPOCHS, unfrozen_epochs=UNFROZEN_EPOCHS):
    """Fine-tune the mini foundation on the ChEMBL bioactives in `df`, generate
    novel molecules, cache them to gen_smiles_{chembl_id}.csv, and return
    (smiles_list, out_text, img).

    PyTorch rewrite of legacy_code/finetune_gpt.finetune_gpt. Uses the
    CafChemGPT toolkit (tokenize, train, generate) and the 1-block mini
    foundation. The fine-tuned model is saved to
    ../data/GPT_{chembl_id}_mini_finetuned.pt.

        Args:
            df: DataFrame with a "SMILES" column (ChEMBL bioactives)
            chembl_id: ChEMBL id (used for cache + output filenames)
            foundation_file: mini foundation path (no .pt)
            frozen_epochs: epochs with the foundation block frozen
            unfrozen_epochs: epochs with everything trainable
        Returns:
            final_smiles: list of generated SMILES strings
            out_text: human-readable summary string
            img: grid image of the generated molecules
    """
    print("GPT finetune")
    print("=" * 51)

    cache = f"../scratch/gen_smiles_{chembl_id}.csv"
    if os.path.exists(cache):
        cached = pd.read_csv(cache)
        final_smiles = cached["SMILES"].to_list()
        # Silence RDKit's C-level stderr parse errors (cached SMILES are already
        # validated on first generation; if one is bad, _draw_grid just shows a
        # blank cell, no need to spam the console). Mirrors CafChemGPT._silence_rdkit.
        from CafChemGPT import _silence_rdkit
        with _silence_rdkit():
            final_mols = [Chem.MolFromSmiles(s) for s in final_smiles]
        img = _draw_grid(final_mols, final_smiles)
        if print_flag:
            print(f"Loaded cached {cache} ({len(final_smiles)} molecules).")
    else:
        # Prepare dataset from bioactives ====================================
        if len(df) > MAX_DATASET:
            df = df.sample(n=MAX_DATASET, random_state=42)
        Xa = [_strip_salts(s) for s in df["SMILES"]]
        data_csv = f"../scratch/{chembl_id}_ft.csv"
        pd.DataFrame({"SMILES": Xa}).to_csv(data_csv, index=False)

        novel = test_vocab(data_csv, "SMILES")
        trim_vocab(data_csv, novel)
        trimmed_csv = f"../scratch/{chembl_id}_ft_trimmed.csv"
        fx, fy, VOCAB_SIZE, tokenizer, max_length = make_datasets(trimmed_csv, "SMILES")
        if print_flag:
            print(f"fx {fx.shape} | fy {fy.shape} | VOCAB {VOCAB_SIZE} | max_len {max_length}")

        # Build + train the fine-tune model (frozen, then unfrozen) ==========
        gpt_ft = make_mini_finetune_gpt(NUM_NEW_BLOCKS, freeze_old_layers=True,
                                        foundation_file=foundation_file)
        gpt_ft = train_gpt(gpt_ft, fx, fy, epochs=frozen_epochs,
                          batch_size=BATCH_SIZE, lr=LR)
        gpt_ft = unfreeze_gpt(gpt_ft)
        gpt_ft = train_gpt(gpt_ft, fx, fy, epochs=unfrozen_epochs,
                          batch_size=BATCH_SIZE, lr=LR)
        save_gpt(gpt_ft, f"../data/GPT_{chembl_id}_mini_finetuned")

        # Generate ===========================================================
        prompts = make_prompts(N_PROMPTS, PROMPT_LEN)
        img, final_smiles = gen_mols(prompts, USE_RAMP, gpt_ft, tokenizer,
                                     TEMP, VOCAB_SIZE)
        pd.DataFrame({"SMILES": final_smiles}).to_csv(cache, index=False)

    if print_flag:
        print(f"Generated {len(final_smiles)} unique molecules.")
    out_text = (f"The novel molecules generated by a mini GPT fine-tuned on "
                f"{chembl_id} are:\n")
    for smile in final_smiles:
        out_text += f"{smile}\n"
    return final_smiles, out_text, img