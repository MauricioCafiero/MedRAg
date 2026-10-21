# GenMask

Generate novel molecule and protein analogues by masking part of a sequence and
unmasking it with a Hugging Face masked-language model.

Given a SMILES string or an amino-acid sequence, GenMask masks a fraction of its
tokens/residues, then fills them back in most-confident-position first, keeping every
candidate whose probability clears a cutoff (not just the single best guess). Repeating
this with different masking patterns (start, end, and several random draws) produces a
diverse set of plausible analogues in one run.

## Contents

- **Small molecules** — mask/unmask SMILES tokens with a BERT-style model, score
  results with QED (drug-likeness) and Tanimoto similarity to the parent molecule.
- **Proteins** — mask/unmask residues with an ESM2 model, score results with percent
  sequence identity to the parent. Sequences can come from a raw string or be pulled
  directly from a PDB entry and chain.
- **Docking + matched-molecular-pair (MMP) analysis** — dock the parent and every
  analogue against a receptor with AutoDock Vina, then use `mmpdb` to find which
  structural transformations actually help or hurt the predicted binding score.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export HF_TOKEN=your_huggingface_token   # required for gated/private HF models
```

`pip install -r requirements.txt` also installs the `mmpdb` CLI and the
`obabel` CLI (via `openbabel-wheel`) into `.venv/bin` — both are needed by
[Docking + matched-molecular-pair analysis](#docking--matched-molecular-pair-analysis)
and require no separate system install.

## Usage

### Small molecules (SMILES)

```bash
python code/gen_mask_cli.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
python code/gen_mask_cli.py --csv molecules.csv
```

The CSV mode looks for a `smiles`/`SMILES`/`Smiles` column (case-insensitive).

| Flag | Default | Description |
|---|---|---|
| `--smiles` | — | A single SMILES string |
| `--csv` | — | CSV file with a SMILES column |
| `--percent` | `0.10` | Fraction of tokens masked per pass |
| `--model` | `cafierom/bert-base-cased-ChemTok-ZN250K-V1` | HF model name |
| `--out` | `outputs/results.csv` | Output CSV path |
| `--no-image` | off | Skip saving a grid-image PNG per input molecule |
| `--no-tanimoto` | off | Skip the `tanimoto_to_parent` similarity column |

Output columns: `input_smiles`, `analogue_smiles`, `entropy`, `qed`, `tanimoto_to_parent`.

### Docking + matched-molecular-pair analysis

```bash
python code/dock_mmp_cli.py --results outputs/results.csv \
    --receptor receptor.pdb --center 12.3 45.6 -7.8
```

Takes a GenMask results CSV, pools every distinct molecule across
`input_smiles` and `analogue_smiles` (deduplicated), docks each one with
AutoDock Vina at a fixed binding-site box (`code/CafChemDock.py` — a
self-contained centroid-docking module; it does not detect binding sites,
you must supply one), then runs the pooled set through `mmpdb`'s
fragment/index/proprulecat pipeline so that matched pairs are found across
the *whole* pool, not just each GenMask parent-child pair in isolation. This
gives more statistical support per transformation than tracking lineage
alone would.

Vina affinities are negative, and **more negative is more favorable** — so a
transformation with a negative mean delta is predicted to improve binding.

| Flag | Default | Description |
|---|---|---|
| `--results` | `outputs/results.csv` | GenMask results CSV to pool and dock |
| `--receptor` | — | Receptor PDB file (required) |
| `--center` | — | Docking box center `X Y Z` in Angstroms (required) — e.g. a co-crystallized ligand's centroid |
| `--box-size` | `22.0` | Cubic box edge length in Angstroms |
| `--exhaustiveness` | `8` | Vina search effort |
| `--num-modes` | `9` | Vina output poses considered |
| `--min-pairs` | `2` | Only report transformations backed by at least this many matched pairs |
| `--out-dir` | `outputs/dock_mmp` | Where intermediates + outputs are written |

Outputs (in `--out-dir`): `results_scored.csv` (the input CSV plus
`input_docking_score`, `analogue_docking_score`, `delta_docking_score`) and
`transformations.csv` (one row per structural transformation, with the
number of supporting pairs and the mean/std/p-value of its effect on
docking score, straight from `mmpdb proprulecat`).

Requires `obabel` on `PATH` (installed into `.venv/bin` by the
`openbabel-wheel` dependency) and a Vina binary (uses the one vendored
inside `dockstring` by default).

### Proteins

```bash
python code/protein_mask_cli.py --sequence "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEV..."
python code/protein_mask_cli.py --pdb 4ZGM --chain B
python code/protein_mask_cli.py --csv sequences.csv
```

The CSV mode looks for a `sequence`/`Sequence`/`SEQUENCE` column (case-insensitive).

| Flag | Default | Description |
|---|---|---|
| `--sequence` | — | A single amino-acid sequence |
| `--csv` | — | CSV file with a sequence column |
| `--pdb` | — | PDB ID to fetch a chain's sequence from (use with `--chain`) |
| `--chain` | — | Chain letter to extract, e.g. `B` (required with `--pdb`) |
| `--percent` | `0.10` | Fraction of residues masked per pass |
| `--prob-cutoff` | `0.05` | Minimum candidate probability to keep a beam |
| `--model` | `facebook/esm2_t12_35M_UR50D` | HF ESM checkpoint |
| `--out` | `outputs/protein_results.csv` | Output CSV path |

Output columns: `input_sequence`, `generated_sequence`, `entropy`, `percent_identity_to_parent`.

### Three-letter amino acid codes

`gen_mask` and `gen_from_multimask` expect single-letter sequences. If your sequence
is in three-letter form, convert it with the helpers in `CafChemProteinMaskEmbed.py`
(not wired into the CLI — use them in a script or interpreter):

```python
import CafChemProteinMaskEmbed as pm

seq = pm.three_letter_seq_to_one("Ala-Gly-Leu")   # -> "AGL"
# ... pass seq to pm.gen_mask(seq, percent_masked) ...
three_letter = pm.one_letter_seq_to_three("AGL")  # -> "ALA-GLY-LEU"
```

`three_letter_seq_to_one` accepts codes separated by whitespace, hyphens,
underscores, commas, or slashes, or run together with no separator (e.g.
`"AlaGlyLeu"`) as long as the total length is a multiple of 3.

## Examples

### Small molecules

```bash
python code/gen_mask_cli.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
```

```
First masking generated 2 SMILES
Last masking generated 6 SMILES
Random masking generated 2 SMILES
Random masking generated 0 SMILES
Random masking generated 0 SMILES
Random masking generated 2 SMILES
Total SMILES generated: 12
Total unique SMILES generated: 9

analogue 1: N#CCC(=O)Oc1ccccc1C(=O)O with QED: 0.591
analogue 2: [NH3+]CC(=O)Oc1ccccc1C(=O)O with QED: 0.505
analogue 3: CC(=O)Oc1ccccc1C(=O)[O-] with QED: 0.472
analogue 4: CC(=O)Oc1ccccc1C(C)=O with QED: 0.394
analogue 5: CC(=O)Oc1ccccc1C(=O)O with QED: 0.550
analogue 6: C=Cc1ccccc1OC(C)=O with QED: 0.492
analogue 7: CC(=O)Oc1ccccc1C=O with QED: 0.377
analogue 8: COC(=O)c1ccccc1OC(C)=O with QED: 0.527
analogue 9: CCOC(=O)c1ccccc1OC(C)=O with QED: 0.561

Saved image: outputs/molecule_0.png
Saved 9 analogues to outputs/results.csv
```

`outputs/molecule_0.png`:

![Aspirin analogues generated by gen_mask_cli.py, each labeled with its QED score](docs/images/aspirin_analogues.png)

`outputs/results.csv`:

| analogue_smiles | entropy | qed | tanimoto_to_parent |
|---|---|---|---|
| N#CCC(=O)Oc1ccccc1C(=O)O | 0.274 | 0.591 | 0.652 |
| [NH3+]CC(=O)Oc1ccccc1C(=O)O | 0.361 | 0.505 | 0.698 |
| CC(=O)Oc1ccccc1C(=O)[O-] | 0.506 | 0.472 | 0.750 |
| CC(=O)Oc1ccccc1C(C)=O | 0.513 | 0.394 | 0.750 |
| C=Cc1ccccc1OC(C)=O | 0.560 | 0.492 | 0.545 |
| CC(=O)Oc1ccccc1C=O | 0.513 | 0.377 | 0.581 |
| COC(=O)c1ccccc1OC(C)=O | 0.145 | 0.527 | 0.698 |
| CCOC(=O)c1ccccc1OC(C)=O | 0.289 | 0.561 | 0.652 |

### Docking + MMP analysis: paracetamol against SULT1A3

Seed molecule: paracetamol (`CC(=O)Nc1ccc(O)cc1`). Receptor: PDB `2A3R`
(SULT1A3), docked at the co-crystallized dopamine centroid.

```bash
python code/gen_mask_cli.py --smiles "CC(=O)Nc1ccc(O)cc1" --percent 0.30 \
    --out outputs/paracetamol_results.csv --no-image
# -> 53 unique analogues

python code/dock_mmp_cli.py --results outputs/paracetamol_results.csv \
    --receptor sult1a3_2A3R.pdb --center 50.536 117.105 -1.369 \
    --out-dir outputs/dock_mmp_paracetamol
```

```
Pooled 53 unique molecules from 53 rows.
Docking pooled molecules with Vina...
Docked 53/53 molecules successfully.
Running mmpdb fragment / index / proprulecat...

Per-molecule scores:  outputs/dock_mmp_paracetamol/results_scored.csv
Transformation rules: outputs/dock_mmp_paracetamol/transformations.csv
```

The 53-molecule pool happens to sample the same aromatic-ring position with
three different substituents (methyl, chloro, fluoro) often enough for
`mmpdb` to report all three pairwise comparisons with reasonable support
(remember: more negative = more favorable predicted binding):

| Transformation | Pairs | Mean Δ docking score (kcal/mol) | Std | p-value |
|---|---|---|---|---|
| `[*:1]Cl` → `[*:1]F` | 10 | −0.37 | 0.19 | 0.0002 |
| `[*:1]C` → `[*:1]F`  | 13 | −0.31 | 0.34 | 0.0071 |
| `[*:1]Cl` → `[*:1]C` | 10 | −0.06 | 0.23 | 0.425 (n.s.) |

Reading it: swapping that ring substituent for fluorine is predicted to
improve SULT1A3 affinity over either chlorine or methyl, consistently enough
across pairs to be statistically significant; chlorine vs. methyl shows no
significant difference. This is a single-receptor-conformation, single-seed
illustration of the pipeline, not a validated SAR claim — treat it as a
worked example of the mechanism, not a result to act on.

### Proteins

```bash
python code/protein_mask_cli.py --pdb 4ZGM --chain B --percent 0.10
```

```
Fetched 4ZGM chain B: HXEGTFTSDVSSYLEGQAAKEFIAWLVRGRG
First masking generated 9 sequences
Last masking generated 81 sequences
Random masking generated 27 sequences
Random masking generated 9 sequences
Random masking generated 27 sequences
Random masking generated 9 sequences
Total sequences generated: 162

analogue 1: MGSGTFTSDVSSYLEGQAAKEFIAWLVRGRG with 90.3% identity to parent
analogue 2: MLSGTFTSDVSSYLEGQAAKEFIAWLVRGRG with 90.3% identity to parent
analogue 3: MGAGTFTSDVSSYLEGQAAKEFIAWLVRGRG with 90.3% identity to parent
...

Saved 162 analogues to outputs/protein_results.csv
```

## Repository layout

```
code/       library modules and CLI entry points
outputs/    generated CSVs and images (not tracked in git)
docs/       README assets (example images)
archive/    earlier exploratory notebooks
```

## Models

- **SMILES**: `cafierom/bert-base-cased-ChemTok-ZN250K-V1`, a BERT model trained on
  tokenized SMILES strings.
- **Proteins**: any Hugging Face ESM2 checkpoint (e.g. `facebook/esm2_t12_35M_UR50D`
  through `facebook/esm2_t33_650M_UR50D`) — larger checkpoints give better predictions
  at the cost of slower runs and bigger downloads.
