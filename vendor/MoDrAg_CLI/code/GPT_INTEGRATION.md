# Workbench — laptop-friendly mini GPT fine-tune

A self-contained slice of this repo that you can **port into the separate agentic
med-chem workbench**. It fine-tunes the 1-block *mini* foundation on a ChEMBL
target's bioactives and generates novel molecules — small enough to run on an
8 GB Mac (1 foundation block + 1 new block = a 2-block fine-tune, ~half the
4-block Tyrosinase fine-tune in the parent repo).

This is the PyTorch rewrite of `legacy_code/` (the old TensorFlow + DeepChem
prototype). `legacy_code/` stays as reference and is not used at runtime.

## What's in here

| File | Role |
|------|------|
| `gpt_node.py` | Entry point called by the workbench. `gpt_node(chembl_id)` → reads `{chembl_id}_bioactives.csv` (fetched by the workbench's `getbioactives_node`), calls `finetune_gpt`, returns `(smiles_list, gpt_string, [img])`. |
| `finetune_gpt.py` | `finetune_gpt(df, chembl_id)`: cap to 2000, tokenize, fine-tune the mini foundation (frozen then unfrozen, 25 + 25 epochs), save `data/GPT_{chembl_id}_mini_finetuned.pt`, generate 50 molecules, cache to `gen_smiles_{chembl_id}.csv`. |
| `CafChemGPT.py` | **Snapshot copy** of `code/CafChemGPT.py` (the PyTorch toolkit: model, `train_gpt`, inference, dataset utils). In this repo `code/CafChemGPT.py` remains the source of truth; this copy is bundled so the workbench repo has everything it needs with no `code/` dir. |
| `smiles_tokenizer.py` | **Snapshot copy** of `code/smiles_tokenizer.py` (DeepChem-style SMILES tokenizer, no DeepChem dependency). |
| `data/GPT_ZN305_mini.pt` | The 1-block mini foundation (~0.7M params), trained on `ZN305K_smiles.csv`. Produce/refresh it with `notebooks/Colab_Mini_Foundation_Train.ipynb`. |
| `data/vocab_305K.txt` | 100-token vocabulary used for training + inference. |
| `data/vocab.txt` | Larger 591-token vocabulary (used by the `test_vocab` compatibility check). |

`gpt_node.py` and `finetune_gpt.py` are the new code; the two `.py` copies and
the `data/` assets are the dependencies gathered here so porting is a single
folder copy.

## Porting into the workbench repo

The legacy workbench repo already ships `vocab.txt`, `vocab_305K.txt`, and
`ZN305K_smiles.csv` at its **root** (the old TF prototype read them cwd-relative).
This new code expects them under a `data/` folder (matching this repo's layout),
so on the workbench side:

1. Copy `workbench/gpt_node.py`, `workbench/finetune_gpt.py`,
   `workbench/CafChemGPT.py`, and `workbench/smiles_tokenizer.py` into the
   workbench repo.
2. Create a `data/` folder in the workbench repo and move the existing
   `vocab_305K.txt`, `vocab.txt`, and `ZN305K_smiles.csv` into it, then add
   `GPT_ZN305_mini.pt` (from `workbench/data/` here).
3. Run `gpt_node(chembl_id)` from the workbench repo root. The toolkit resolves
   `data/...` paths cwd-relative, so the `data/` folder must be under the cwd.

`finetune_gpt.py` imports `CafChemGPT` from `code/` when that dir exists (this
repo) and otherwise from its own folder (the ported workbench repo), so no
import rewiring is needed.

## Notes

- `data/GPT_ZN305_mini.pt` is a **snapshot** — retrain with the Colab notebook if
  `code/CafChemGPT.py` or `code/smiles_tokenizer.py` change in ways that break
  checkpoint compatibility (architecture, tokenizer vocab, or the
  `[PAD]/[unused]/[CLS]/[SEP]` token order).
- `getbioactives_node` is provided by the workbench runtime, not defined here
  (same arrangement as `legacy_code/old_prot.py`); `gpt_node.py` imports cleanly
  on its own and only fails if called without it in scope.
- Per-target fine-tune outputs (`data/GPT_*_mini_finetuned.pt`,
  `gen_smiles_*.csv`, `data/*_ft.csv`) are transient and gitignored in the
  parent repo.