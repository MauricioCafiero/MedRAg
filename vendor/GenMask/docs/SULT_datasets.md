# SULT1A1 / SULT1A3 binder and decoy datasets

Benchmark sets of known binders (actives) and property-matched decoys for the human
phenol sulfotransferases **SULT1A1** and **SULT1A3**, built by
[`code/build_sult_dataset.py`](../code/build_sult_dataset.py).

## Why this script exists

There is no off-the-shelf benchmark for these targets:

- **DUD-E**, **DEKOIS 2.0** and **LIT-PCBA** contain no sulfotransferase.
- **ChEMBL** holds only 42 unique molecules for human SULT1A1 (`CHEMBL1743291`) and
  8 for SULT1A3 (`CHEMBL1743293`), most of them `Km`/`Vmax` substrate measurements
  rather than inhibition data.
- **BindingDB** adds 4 compounds for SULT1A1 (`P50225`) and none for SULT1A3.
- **PubChem** has no qHTS campaign against either isoform — searching by gene symbol
  returns ChEMBL re-exports plus genome-wide siRNA screens that are irrelevant here.

The usable core is therefore a literature-curated set, which the script pools with
whatever the databases do hold, and pairs with decoys generated locally.

## Sources

| Source | What it contributes | How it is fetched |
|---|---|---|
| Martiny et al., *PLoS ONE* 2013, 8(9):e73587 ([doi:10.1371/journal.pone.0073587](https://doi.org/10.1371/journal.pone.0073587)) | Structurally diverse binders (substrates **or** inhibitors) curated from BRENDA, Aureus, TOXNET, PubChem and the literature, clustered at FCFP_4 Tanimoto 0.6: 60 for SULT1A1 (S1), 50 for SULT1A3 (S2), 33 for SULT1E1 (S3). The pre-clustering sets were 157 / 117 / 80 compounds. | Supplementary SDFs via the Europe PMC REST API (`PMC3765257`) |
| ChEMBL | Activities for `CHEMBL1743291` (1A1), `CHEMBL1743293` (1A3), `CHEMBL2346` (1E1), with the measured value kept in the `activity` column | ChEMBL REST API |
| BindingDB | Ligands by UniProt accession (`P50225`, `P0DMM9`, `P49888`) | BindingDB REST API |
| ZINC20 | Decoy candidates: 2D tranches spanning MW ≤ 450 Da and cLogP ≤ 5 | `files.docking.org` |

Sources are pooled and deduplicated by InChIKey; the `source` column records every
source a molecule was found in (e.g. `Martiny2013|ChEMBL`).

## How decoys are chosen

DUD-E style — a decoy should be *hard to tell apart by physical properties* but
*structurally unrelated*, so that a screening method cannot win on bulk properties alone:

1. Every molecule is standardized: largest fragment, normalized functional groups,
   neutralized, explicit hydrogens removed, canonical SMILES.
2. For each active, ZINC candidates are required to match on **MW ±25 Da, cLogP ±1.0,
   HBD ±1, HBA ±2, rotatable bonds ±2, and identical net charge**. The window widens
   twice (to ±50/±1.5/… and ±100/±2.0/…) when an active is too exotic to match tightly.
3. A candidate is rejected if its **ECFP4 (Morgan r=2, 2048-bit) Tanimoto reaches 0.35
   against *any* active** — not just the one it was matched to.
4. Each decoy is used once across the whole set.

## What was built

`python code/build_sult_dataset.py --target both --decoys-per-active 50`, run 2026-09-15:

| | SULT1A1 | SULT1A3 |
|---|---|---|
| Actives (unique by InChIKey) | 99 | 58 |
| — from Martiny 2013 | 60 | 50 |
| — from ChEMBL | 42 | 8 |
| — from BindingDB | 4 | 0 |
| Decoys | 4,900 | 2,850 |
| Decoys per active | 49.5 | 49.1 |

(Source counts overlap — 33 molecules are binders of both isoforms, and several appear
in more than one database.) The background pool was 572,738 standardized ZINC molecules.

One active per set gets no decoys: the tri-iodinated thyronines (MW > 600) fall outside
the tranche window and have no property match in ZINC.

**Match quality:**

| | SULT1A1 actives / decoys | SULT1A3 actives / decoys |
|---|---|---|
| MW | 246.8 ± 96.8 / 243.5 ± 81.6 | 254.2 ± 87.5 / 249.5 ± 75.0 |
| cLogP | 2.68 ± 1.26 / 2.52 ± 1.30 | 2.30 ± 1.30 / 2.12 ± 1.39 |
| HBD | 1.83 ± 0.95 / 1.24 ± 0.85 | 2.48 ± 1.12 / 1.76 ± 0.97 |
| HBA | 3.24 ± 1.49 / 2.71 ± 1.43 | 3.67 ± 1.45 / 3.05 ± 1.40 |

Decoy → nearest-active ECFP4 Tanimoto: median 0.21, p95 0.31, max 0.35 (the cutoff).
Active → nearest-other-active: median 0.47 (1A1) / 0.39 (1A3), i.e. the actives are
genuinely diverse rather than one congeneric series.

**Known residual bias:** decoys average ~0.6 fewer H-bond donors and acceptors than the
actives. SULT binders are unusually polyphenol-rich and ZINC simply holds fewer such
molecules inside the matching window, so a classifier could learn "count the OH groups"
rather than anything about binding. Check feature importances before believing a model.

## Output files

Written to `--out-dir` (default `data/`), one set per target:

| File | Columns |
|---|---|
| `<target>_actives.csv` | `smiles`, `source`, `source_id`, `activity`, `inchikey`, `label` (=1), `target` |
| `<target>_decoys.csv` | `smiles`, `zinc_id`, `tranche`, `matched_active`, `label` (=0) |
| `<target>_benchmark.csv` | `smiles`, `id`, `label` — the flat file for scoring/enrichment |

## Usage

```bash
python code/build_sult_dataset.py --target both --decoys-per-active 50
```

| Flag | Default | Description |
|---|---|---|
| `--target` | `both` | `sult1a1`, `sult1a3`, `sult1e1`, or `both` (the two 1A isoforms) |
| `--decoys-per-active` | `50` | Decoys requested per active |
| `--tanimoto-cutoff` | `0.35` | Reject a decoy at/above this ECFP4 similarity to any active |
| `--drop-sulfonatable` | off | Also reject decoys bearing a phenol, aliphatic hydroxyl, aryl amine or N-hydroxy group |
| `--per-tranche` | `8000` | Molecules sampled per ZINC tranche file (background ≈ 90 × this) |
| `--seed` | `0` | Decoy sampling seed |
| `--out-dir` / `--cache-dir` | `data` / `data/cache` | Outputs and download cache |

The first run downloads ~115 MB of ZINC tranches and caches an annotated background CSV
(`data/cache/zinc_background_props_8000.csv`, 57 MB), so later runs with different
cutoffs take about a minute. The bulk ZINC download is gitignored; the small
`data/cache/chembl_*.json` and `bindingdb_*.json` source pulls are tracked, and the
script reads them instead of re-fetching (ChEMBL's API is frequently down — pass
`--refresh-sources` to force a re-fetch).

## A note on structure parsing

The Martiny supplementary SDFs draw nitro groups as neutral tetravalent nitrogen,
`N(-O)(=O)`, which RDKit rejects. A naive repair produces `[NH+]([O-])O` — a protonated
N-hydroxy, not a nitro — which silently corrupts 4-nitrocatechol, nimesulide and DCNP
(2,6-dichloro-4-nitrophenol, the canonical SULT1A1 inhibitor). `from_molblock()` charges
the *single-bonded* terminal oxygen instead, and also charges the tetravalent ammonium
nitrogens drawn neutral in the same files. All 143 supplementary records parse; a plain
`SDMolSupplier` recovers only 132.

## Caveats

- **Substrates and inhibitors are mixed.** Most SULT1A "binders" in the literature are
  substrates characterized by `Km`, not inhibitors with an `IC50`. For an inhibitor-only
  set, filter `<target>_actives.csv` on the `activity` column.
- **Latent actives.** SULT1A1 is promiscuous toward small phenols, so a property-matched
  decoy carrying a free phenol may well be a real substrate. `--drop-sulfonatable`
  removes those candidates; it lowers false-negative contamination but makes the
  benchmark easier and biases decoys away from the actives' chemotype. Which one you
  want depends on whether you are measuring enrichment honestly or building training data.
- **DUD-E-style bias.** Property-matched/topologically-dissimilar decoys are known to
  flatter docking and to be separable by machine learning on artefacts rather than
  binding. Treat enrichment numbers as a protocol sanity check, not as validation.
- **The actives are a clustered subset.** Martiny et al. published the 60/50 diverse
  representatives, not their full 157/117 sets, so the actives are chemotype-diverse but
  not exhaustive.
- **SULT1A1 and SULT1A3 overlap.** 33 molecules are binders of both in this pool — they
  share 93% sequence identity overall and 73% across the binding pocket.

## Structures for the docking step

`code/dock_mmp_cli.py` needs a receptor and a box center. Both isoforms crystallize with
the spent cofactor PAP (`A3P`) bound, which fills part of the site — keep it in the
receptor and center the box on the *substrate*, not on PAP.

| PDB | Isoform | Resolution | Bound ligands |
|---|---|---|---|
| `1LS6` | SULT1A1 | 1.9 Å | PAP + p-nitrophenol |
| `3U3O` | SULT1A1 | 2.0 Å | PAP + 7-hydroxycoumarin |
| `2D06` | SULT1A1 | 2.3 Å | PAP + estradiol |
| `3U3K` | SULT1A1 | 2.4 Å | PAP + 2-naphthol |
| `2A3R` | SULT1A3 | 2.6 Å | PAP + dopamine |
| `1CJM` | SULT1A3 | 2.4 Å | sulfate only (apo-like site) |

`2A3R` is the receptor used in the paracetamol worked example in the README.
