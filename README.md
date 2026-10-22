# CafChem MedRAg

*medicinal chemistry agent*

MedRAg is a local, self-checking hit-to-lead agent. It takes a protein structure with a co-crystallized ligand and a seed molecule,
proposes analogues, filters them, docks them, checks that the poses are in the right binding site, runs MLIP pose-feasibility
checks, and writes an HTML report. An LLM agent (Ollama cloud or Anthropic) can drive every step through tool calls.

```
set_target -> generate -> filter -> dock -> site -> validate -> [boltz] -> report
```

Built on the author's own repos: the pieces it uses were copied, unchanged, into `medrag/lib/` (see
[Code borrowed from other repos](#code-borrowed-from-other-repos)) and are called through thin wrappers.

## Quickstart (about 15 minutes, most of it downloads)

Needs Linux, `git`, [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`) and internet.
`uv` fetches Python 3.12 itself. Everything below runs inside one virtual environment.

```bash
git clone <this repo> medrag && cd medrag
bash setup.sh                    # builds .venv: CPU torch 2.8.0, mace-torch, fairchem, rdkit, openmm, pymol, ...
source .venv/bin/activate

# the simplest start: just a protein name
python -m medrag.pipeline demo --protein SULT1A3 --gpt --steps generate,filter,dock,site,report --max-dock 5
```

From the name alone, MedRAg finds the UniProt entry, searches the PDB for structures of that protein, picks one with a good-resolution,
drug-like bound ligand (that ligand only identifies the binding site), checks ChEMBL for a data set, then generates analogues,
filters and docks the best five, checks their binding site, and opens `runs/demo/report.html` in your browser. By default the
crystal ligand is also the seed; give your own with `--seed 'SMILES'`. Add `validate` to `--steps` (and `--validate-top 3`) for the
MLIP feasibility check, which takes about 10 minutes per molecule on a CPU. Rerunning the same command resumes.

If you already have a structure, skip the search:

```bash
curl -O https://files.rcsb.org/download/2A3R.pdb
python -m medrag.pipeline demo2 --pdb 2A3R.pdb --resname LDP --chain A --seed '[NH3+]CCc1ccc(O)c(O)c1' \
    --steps target,generate,filter,dock,site,report --max-dock 5
```

**What gets downloaded on first use** (into the usual per-user caches, not the repo): the MACE-POLAR-1 weights (~50 MB, public),
the GenMask language model (~430 MB, public Hugging Face repo) and, only if you choose `--mlip uma`, UMA (~4 GB, a gated Hugging
Face model: request access and `huggingface-cli login` first). Default is MACE-POLAR-1 small, which needs no account.

**Accounts and keys: what needs one**

| You want to | Needs |
|---|---|
| Run the pipeline from the command line (`python -m medrag.pipeline`), including the report | nothing: no account, no key |
| Drive it with the LLM agent (`python -m medrag`) | an `OLLAMA_API_KEY` (Ollama cloud, the default) or an `ANTHROPIC_API_KEY` |
| Use UMA as the MLIP (`--mlip uma`) | a Hugging Face account with access to the gated UMA model, then `huggingface-cli login` |
| Add Boltz-2 co-folding (`--boltz modal`) | a [Modal](https://modal.com) account, then `modal setup` once (Modal bills your account; a molecule costs a few cents) |

**Hardware.** CPU only, no GPU needed. Memory is the limit: the MLIP step peaks at about 4 GB for a 300-atom pocket and about 7.5 GB
for a 480-atom one (HMGCR) because we need forces for relaxation. 16 GB RAM is comfortable; 8 GB handles only small sites.

**Platforms.** Developed and tested on Linux x86-64 (including a from-scratch rebuild with `setup.sh`). Not tested on macOS or
Windows; `setup.sh` takes CPU torch from the PyTorch CPU wheel index, which may need adjusting on a Mac.

## Running it

All commands assume the venv is active (`source .venv/bin/activate`) and are run from the repo root. State for a run lives in
`runs/<id>/state.json`; finished steps are skipped on a rerun, and dock and validate resume per molecule.

```bash
python -m medrag.pipeline sult_demo --pdb 2A3R.pdb --resname LDP --chain A \
    --seed '[NH3+]CCc1ccc(O)c(O)c1' --steps target,generate,filter,dock,site,validate,report --validate-top 3
```

`--crystal-smiles` (default: the seed) gives the co-crystallized ligand's structure; `--recall-cut` (default 0.5) sets the
binding-site test below; `--max-dock N` docks only the best N; `--gpt` adds the mini-GPT as a third generator (`--gpt-chembl CHEMBL402` to choose its ChEMBL data set yourself); `--no-open` stops the report opening in the browser. `--boltz modal` (with `boltz` in `--steps`, after `validate`) co-folds the validated finalists with Boltz-2 on a Modal GPU: run `modal setup` once to log in; about 2 min and a few cents per molecule (the weights are baked into the image, so there is no storage charge).

The agent (MoDrAg-style REPL; the same run directory is resumed, including the chat). It needs an LLM key:

```bash
export OLLAMA_API_KEY=...        # ollama cloud backend (nothing runs locally), default model glm-5.3
export ANTHROPIC_API_KEY=...     # anthropic backend
python -m medrag --run sult_demo --backend ollama
python -m medrag.agent --run sult_demo --backend anthropic
python -m medrag.agent --run sult_demo --once "Filter the candidates and show me the best five."
```
The agent also has a `run_boltz` tool (Boltz-2 on Modal, see below); it spends Modal credit, so it only runs when you ask for Boltz or co-folding, and it skips molecules that already have results. For example: `--once "Co-fold the finalists with Boltz and show me the results ranked by binder probability."`

Each turn is planned and reviewed: one LLM call breaks the request into tasks, the agent answers the original message, and the
reply is checked against the task list and the tool-call log. Missing tasks go back to the agent (two rounds at most) and the
answer ends with a checklist. `--no-review` skips this; `--planner-backend/--planner-model` use a different model for it.

The HTML report (`runs/<id>/report.html`) opens in the browser the first time the agent or CLI builds it and is refreshed after
every turn that changes results. It is a dark single page with a floating title bar that keeps every section link in view (plus a
back-to-top button), a hero header with the run's key facts, summary tiles (candidates, filter, docked, in the site, validated) and one
numbered card per step. Long tables scroll inside a fixed-height window with a sticky header, sortable columns and colored status
pills; the PyMOL stills and 2D molecule grids sit in sideways-scrolling strips. Dock and validate also have a rotatable 3Dmol view:
the protein as a ribbon, the pocket residues within 4 A as thin sticks, the ligand as thick sticks, and the reference pose (crystal
ligand, or the docked pose behind a relaxed one) as grey lines. 3Dmol loads from a CDN, so that view needs internet.

## The steps

| step | what it does | code |
|---|---|---|
| discover | from a protein name: UniProt entry, RCSB structures (searched by accession), bound drug-like ligands (dock_assist), best by resolution then QED; ligand SMILES from the RCSB chemical component; ChEMBL data check. Optional: `--protein NAME` | `steps/discover.py` |
| target | protein-only receptor copy, Vina box at the crystal ligand's centroid (the crystal ligand only identifies the site; the seed can be any molecule) | `steps/target.py` |
| generate | GenMask (BERT mask/unmask), bioisostere_assist (shape-matched swaps), and optionally (`--gpt`) MoDrAg's mini GPT, fine-tuned on a data set it finds itself: the target's ChEMBL IC50 molecules if there are at least 50, otherwise PubChem molecules similar to the seed; molecules already in the training set are dropped (new compounds, not seed analogues; 0.5 to 2.5 min on CPU) | `steps/generate.py` |
| filter | pH 7.4 protonation (cafprot), unrealistic-motif check (cyclic allenes, small-ring alkynes), QED, Lipinski, PAINS, ADMET-AI (hERG, AMES, DILI, carcinogenicity, skin, LD50, CYP, bioavailability, BBB); pass / warn / fail. ADMET cut-offs are no tighter than the seed's own values (see below) | `steps/filter.py` |
| dock | AutoDock Vina through dock_assist, in chunks, top 3 poses each | `steps/dock.py` |
| site | contact residues of the crystal ligand are the expected set; each pose is scored by recall of them | `steps/site.py` |
| validate | MLIP feasibility checks on the best in-site pose of the top candidates | `steps/validate.py`, `feasibility/` |
| boltz (optional) | Boltz-2 co-folding and affinity for the validated finalists, on a Modal GPU only (no local option yet) | `steps/boltz.py`, `boltz_modal.py` |
| report | HTML | `report.py`, `steps/report_steps.py` |

**Site check.** Vina scores did not separate the right site from the wrong one on SULT1A3. Recall of the crystal ligand's
contact residues did: right-site poses scored 0.71 to 1.0, wrong-site poses 0 to 0.29, and it agreed with a centroid-distance
test on all 105 poses of the end-to-end run. Only in-site poses (recall >= 0.5) go on to validate.

**Validate.** The pocket (residues within 4 A of the ligand) is carved with UMADock's cut-out, then made safe to hand to an MLIP:
caps are placed at the real neighbouring residues' atom positions, one-residue gaps are joined, hydrogen placement is seeded and
single-threaded so the same input always gives the same site, every X-H bond is reset to its standard length, and the site is checked
for impossible atom pairs and refused if any remain. It is then relaxed with the MLIP in stages (caps first, edge frozen, fmax
0.3 eV/A), followed by: clashes, residual forces near the ligand, ligand displacement, ligand strain (docked geometry under a
harmonic heavy-atom restraint against the lowest vacuum minimum of the same species; proton transfer inside the pocket is followed,
not suppressed) and contact persistence. Strain is graded relative to the seed: warn above seed + 3 kcal/mol, fail above seed + 6,
never tighter than the defaults. Default MLIP is MACE-POLAR-1 small (`polar1-s`); UMA (`uma`) was no better on SULT1A3 and about
1.7x slower. Dynamics (`final_stability`) is a finalist-only stability check and is not part of the routine run. Every relaxation is
checkpointed and resumable.

**Boltz (optional, Modal only).** `--boltz modal` sends each validated finalist (the seed included) to a Modal A10G GPU running Boltz-2, which
predicts the protein-ligand complex from sequence and SMILES and scores affinity (binder probability, pIC50, IC50, dG). It is a second
opinion, not a rescoring of the docking, and its structure is shown in Boltz's own frame, not on the crystal receptor. It runs on your own
Modal account: install Modal (`setup.sh` does) and log in once with `modal setup`. There is no volume to create. The first run builds the
image, which includes the Boltz weights (a one-time download of a few minutes that Modal caches); later runs reuse it. A molecule takes about
2 minutes (a SULT1A3 test cost $0.04), and each call is capped at 10 minutes. The first molecule builds the sequence alignment on the public
ColabFold server, so the protein sequence is sent there; the rest reuse it.

## What the results do and do not show

- Validate is a sanity filter for strain, clashes, forces and lost contacts. On SULT1A3 it did not separate right-site from
  wrong-site poses, and it is not a rescoring or a binding prediction.
- Strain for charged ligands is sensitive to the restraint constant and to the vacuum reference (crystal dopamine: 16.5 kcal/mol
  with MACE-POLAR-1, 6.2 with UMA). Compare strain only within one MLIP and one setting.
- Both ADMET and strain cut-offs are judged relative to the seed, because absolute cut-offs rejected whole series including the lead
  (HMGCR: rosuvastatin itself has DILI 0.95 and 29 kcal/mol strain). So a candidate is flagged only when clearly worse than the lead.
- All cut-offs (filter and validate) are provisional. `calibrate.py` derives validate thresholds from a crystal pose, but calibrating
  from the crystal alone only loosens the cut-offs until the crystal passes.
- The tests so far are two targets: SULT1A3 (2A3R, the small development example) and HMGCR (1HWL, rosuvastatin), a handful of
  ligands and 7 poses for the validate comparison. On HMGCR nearly every docked pose sits in the site, so the site check does no
  harm there but does not discriminate much.
- Skin-reaction cut-offs were set after seeing that ADMET-AI scores every catechol 0.7 to 0.9; they are tuned to that series.
- The GPT's data set is only as good as what exists. HMGCR has 292 ChEMBL IC50 molecules, so it learns from real actives. SULT1A3 has none (the whole sulfotransferase family has at most 3), so the GPT falls back to PubChem neighbours of the seed: its output is then new molecules in the seed's chemical neighbourhood, not target-specific actives, and the run says so.
- The filter marks candidates but never deletes them, and a candidate keeps its docking result if a later filter run changes its status.

## Layout

```
medrag/            pipeline.py (driver), agent.py, llm.py (ollama + anthropic backends), state.py, report*.py
medrag/steps/      one module per step
medrag/feasibility/  system.py (pocket cut-out), calc.py (MLIPs), checks.py, thresholds.py
medrag/tools/      vendor.py (isolated calls into medrag/lib/), adapters/ (JSON entry points onto the borrowed tools)
medrag/lib/        code copied from the author's other repos (below), one folder per source repo
tests/              development scripts that read local run data (runs/...): not runnable from a fresh clone
calibrate.py        validate thresholds from a feas_poses run (development tool)
runs/<id>/          created on first run: state.json, report.html, per-step working directories
```

## Code borrowed from other repos

All of it is the author's own, copied unchanged (line endings aside) from the commit shown into `medrag/lib/<repo>/`, with that
repo's `LICENSE` beside it. Everything not listed here is new in this repo.

| Repo (github.com/MauricioCafiero/...) | Commit | Files used | What MedRAg uses it for |
|---|---|---|---|
| [GenMask](https://github.com/MauricioCafiero/GenMask) | `55a4165` | `CafChemSubs.py` | analogue generation by masking and refilling SMILES |
| [bioisostere_assist](https://github.com/MauricioCafiero/bioisostere_assist) | `744deb5` | `CafChemShape.py`, `fragment.py`, `library.py`, `library_100k_usr.pkl`, `oddt-0.7.tar.gz` | shape-matched substituent swaps from the prebuilt 100k library; the `oddt` source for `setup.sh` |
| [MoDrAg_CLI](https://github.com/MauricioCafiero/MoDrAg_CLI) | `f71d8d9` | `admet_node.py`, `modrag_protein_functions.py`, `modrag_molecule_functions.py`, `gpt_node.py`, `finetune_gpt.py`, `CafChemGPT.py`, `smiles_tokenizer.py`, and `data/` (mini-GPT weights, vocabularies, 305K SMILES) | ADMET-AI predictions; UniProt, ChEMBL and RCSB lookups; PubChem neighbours; the mini GPT |
| [dock_assist](https://github.com/MauricioCafiero/dock_assist) | `8895041` | `vina_dock.py`, `modrag_protein_functions.py` | AutoDock Vina docking; listing a structure's bound ligands |
| [UMADock](https://github.com/MauricioCafiero/UMADock) | `8b3b34a` | `prep_binding_site.py` | cutting the binding site out of the protein for the MLIP checks |
| [cafprot](https://github.com/MauricioCafiero/cafprot) | `d3da76e` | `cafprot.py` | charge state of each molecule at pH 7.4 |
| [FAO_MOLPROP_CLI](https://github.com/MauricioCafiero/FAO_MOLPROP_CLI) | `46e8672` | `docking_module.py` | residue-contact analysis behind the binding-site check |

The Boltz step uses [Boltz-2](https://github.com/jwohlwend/boltz) (MIT), installed in the Modal image rather than copied here; the affinity
conversion (pIC50 = 6 - y, dG = -1.364 x pIC50) follows the author's [boltz_local](https://github.com/MauricioCafiero/boltz_local).

## License

MIT (see `LICENSE`). The borrowed files keep the licenses of the repos above (copies in `medrag/lib/<repo>/LICENSE`).
