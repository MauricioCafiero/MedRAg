# FAO_MOLPROP_CLI

Adversarial molecule optimization on the command line. An **Ollama** main model
designs molecules to bind a protein target, calling chemistry + docking tools to
propose and score them; an **adversary** model (OpenAI or Anthropic) critiques
each proposal; the main model refines until it signals `Done`. Every step is
written to a timestamped Markdown report under `results/`, with a JSON messages
sidecar written alongside it so a stopped run can be resumed.

`molopt.py` is a single-file CLI port of `Ollama_MolOpt.ipynb`, which is kept in
the repo as the reference/source of truth for the prompts.

## What it does

Headless loop, one target at a time:

1. The main (Ollama) model is seeded with a starting molecule/score list
   (`code/adversarial_set.md`) and reasoning instructions.
2. It reasons about the data and calls **chemistry tools** (grow a scaffold,
   swap groups, dock, check drug-likeness, …) to explore chemical space.
3. Each proposal is sent to the **adversary**, which finds flaws in the
   reasoning / estimated docking scores and suggests modifications.
4. The main model revises in light of the critique. The loop ends when the main
   model replies `Done` (robustly detected: a trailing standalone `Done` after a
   final proposal counts) or when `--max-turns` is hit.

Docking is done with **dockstring** against a receptor (default target HMGCR);
the prepared DUD-E receptor `.pdb` files live at the repo root. Docking works
for any of dockstring's 58 targets; the `dock_and_get_interacting_residues`
residue-contact report additionally needs a prepared receptor PDB on disk.
Targets with a receptor PDB available are **HMGCR, ADRB1, ADRB2, MAOB, DRD2**
(mapped in `RECEPTOR_FILES` in `code/docking_module.py`). Docking is the
runtime bottleneck, not the LLM.

## Architecture

- **Main model** — Ollama, function-calling. Hosted default `https://ollama.com`
  with a bearer token, or a local daemon via `--ollama-host
  http://localhost:11434`. `--think`/`--no-think` auto-enables thinking for
  models that support it (the built-in set all do; `NO_THINK_MODELS` is empty).
- **Adversary** — `--adversary openai|anthropic`. OpenAI uses the Responses API
  (`client.responses.create`); Anthropic uses the Messages API
  (`client.messages.create`). SDKs are lazy-imported so a missing dep doesn't
  block the other. The Anthropic client passes an explicit
  `base_url="https://api.anthropic.com"` (overridable via `--anthropic-base-url`)
  so it bypasses any ambient `ANTHROPIC_BASE_URL` pointing at a local proxy.
- **Chemistry tools** — reused unchanged from `code/`:
  - `MolPropOp.py`: `grow_cycle`, `replace_groups`, `make_random_list`,
    `related`, `lipinski`
  - `docking_module.py`: `dock_and_get_interacting_residues`,
    `calculate_SAS_and_NP`, plus the shared mutable `scoring_args`
    (`[cpu_count, protein]`) and the NumPy-2 `np.in1d` shim.
- **Robustness** — auto-retry on Ollama connection errors (`--max-retries`);
  tool-call exceptions are fed back to the model for self-correction;
  `--max-tool-calls` caps per-turn tool rounds (once hit, a text-only summary is
  forced — a stuck model can't loop forever); invalid substituents are surfaced
  to the model as `invalid SMILES, skipped` entries so it stops retrying them.

## Repository layout

```
code/
  molopt.py                   # the CLI: Ollama main model + OpenAI/Anthropic adversary
  molopt_oa.py                # variant: OpenAI <-> Anthropic <-> Gemini adversaries (--start selects proposer)
  run_replicates.py           # run N replicates of each adversary set; writes a manifest
  analyze_replicates.py       # compare final compounds across sets/replicates (CSV + stats + plots)
  test_models.py               # smoke-test candidate Ollama cloud models
  verify_results.py           # verify a run's final molecules and store them in sqlite
  vina_dock.py                 # standalone Vina docking harness incl. blind pocket detection
  _gemini_smoke.py            # one-shot smoke test for GeminiActor tool-calling
  adversarial_set.md          # starting molecule/score list fed to the model
  MolPropOp.py                # molecular-property operations (grow/replace/...)
  docking_module.py           # dockstring docking + SAS/NP scoring
  mock_tools.py                # --mock-tools synthetic-score stand-in (smoke-test w/o real docking)
Ollama_MolOpt.ipynb           # original notebook (reference, not deleted)
new_models_to_add.txt         # candidate cloud model names (with -cloud suffix)
.env.example                  # copy to .env and fill in API keys
requirements.txt              # full deps (base + chem stack + LLM SDKs)
requirements-base.txt         # minimal build deps (setuptools/wheel/six)
requirements-oddt.txt         # optional: oddt (heavier scientific stack)
suppressing_rdkit_smiles_errors.md  # drop-in fix to silence RDKit parse logs
HMGCR_dude_receptor_2.pdb      # HMGCR receptor used by the code (selected via RECEPTOR_FILES)
HMGCR_dude_receptor.pdb        # older copy, kept as a backup; not referenced by the code
dude_receptor_ADRB1.pdb        # ADRB1 receptor for residue-contact analysis
dude_receptor_ADRB2.pdb        # ADRB2 receptor for residue-contact analysis
MAOB-Dud-e-receptor.pdb        # MAOB receptor for residue-contact analysis
DRD2_target.pdb                # DRD2 receptor (converted from DRD2_target.pdbqt via Open Babel)
molecules.sqlite               # accumulated verified molecules + recomputed metrics (created on first run)
results/                       # timestamped session reports (.md) + JSON message sidecars (.json)
results/batches/               # replicate-batch output; a separate private repo, gitignored here
```

## Install

Install in this **specific order** — it matters because `oddt` has been
unmaintained for years and breaks the rest of the stack if installed normally.
Install the build tools first, then `oddt` *without build isolation* (so it
reuses the setuptools/wheel just installed instead of trying to fetch its own,
ancient build deps), then everything else last.

```bash
python3 -m venv fao-env
source fao-env/bin/activate

# 1) build tools first (oddt's --no-build-isolation install needs them present)
pip install -r requirements-base.txt        # setuptools, wheel, six

# 2) oddt, installed WITHOUT build isolation (unmaintained; would otherwise fail
#    to build and/or drag in incompatible deps). Required by docking_module.py.
pip install --no-build-isolation -r requirements-oddt.txt   # oddt

# 3) all other packages last (rdkit, dockstring, pyscf, openbabel-wheel,
#    ollama, openai, anthropic, ...)
pip install -r requirements.txt
```

> **Activate the venv before running anything** (`source fao-env/bin/activate`), not just
> `fao-env/bin/python ...`. The openbabel Python bindings and the `obabel` CLI that dockstring
> shells out to both need `fao-env/bin` on PATH; without activation `import oddt` crashes with
> `AttributeError: ...OBElementTable` and docking fails with `FileNotFoundError: obabel`.

> **Replicate analysis** (`analyze_replicates.py`) also needs matplotlib, which is *not* in
> `requirements.txt` (it's optional, analysis-only): `pip install matplotlib` in the activated
> venv. Run a batch with `code/run_replicates.py`, then `python code/analyze_replicates.py --batch-dir
> results/batches/<batch_id>`. Add `--skip-docking` to produce all CSVs/plots without the
> CPU-heavy dockstring recomputation.

> **Why the order matters:** `oddt` is unmaintained and its build is fragile. If
> you `pip install -r requirements.txt` first and then `oddt`, or let `oddt`
> build in isolation, it can pull an incompatible NumPy / fail to compile, which
> breaks `docking_module.py`. Installing build tools → `oddt` with
> `--no-build-isolation` → the rest avoids that. `docking_module.py` also
> applies an `np.in1d -> np.isin` shim at import time for NumPy 2.x, since oddt
> 0.7 otherwise crashes on NumPy 2.
>
> Note: `openbabel-wheel` and `rdkit` are pip-installable wheels on macOS/manylinux.

## Configure

Copy `.env.example` to `.env` and fill in keys, **or** export them in your shell
(`source ~/.zshrc` if they live there). Real environment variables always override
`.env`, which is only read from the directory you run `molopt.py` from.

| Variable          | Used by                          | Required?                          |
|-------------------|----------------------------------|------------------------------------|
| `OLLAMA_API_KEY`  | main model (hosted endpoint)     | yes for `https://ollama.com`; no for local daemon |
| `OPENAI_API_KEY`  | `--adversary openai`             | only if using the OpenAI adversary |
| `ANTHROPIC_API_KEY` | `--adversary anthropic`        | only if using the Anthropic adversary |

## Usage

```bash
# list the built-in Ollama model set, then run with the default model
python3 code/molopt.py --list-models
python3 code/molopt.py --protein HMGCR

# specific cloud model (API name, no '-cloud' suffix) with think forced on
python3 code/molopt.py --model gemma4:31b --think

# Anthropic adversary instead of the default OpenAI one
python3 code/molopt.py --model qwen3.5:397b \
  --adversary anthropic --adversary-model claude-haiku-4-5-20251001

# quick chemistry-stack check with no LLM keys
python3 code/molopt.py --self-test
```

`molopt_oa.py` is the OpenAI↔Anthropic variant (no Ollama): `--start openai|anthropic`
picks which provider leads as the tool-calling proposer; the other provider is the
critique-only adversary. Otherwise the loop and outputs are the same as `molopt.py`.

```bash
python3 code/molopt_oa.py --protein HMGCR --start openai \
  --openai-model gpt-5.2 --anthropic-model claude-haiku-4-5-20251001
python3 code/molopt_oa.py --self-test
```

See `python3 code/molopt.py --help` / `python3 code/molopt_oa.py --help` for the full option lists.

### Key flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--protein` | `HMGCR` | Dockstring target. Docking works for any of dockstring's 58 targets; residue-contact analysis needs a receptor PDB on disk (HMGCR, ADRB1, ADRB2, MAOB). |
| `--model` | `deepseek-v4-pro` | Ollama main model (API name, no `-cloud` suffix). |
| `--adversary` | `openai` | `openai` (default `gpt-5.2`) or `anthropic` (default `claude-haiku-4-5-20251001`). |
| `--max-turns` | `20` | Safety cap on adversary↔main turns. |
| `--max-tool-calls` | `12` | Per-turn cap on tool-calling rounds; forces a text summary once hit. |
| `--ollama-host` | `https://ollama.com` | Use `http://localhost:11434` for a local daemon. |
| `--context-file` | `code/adversarial_set.md` | Starting molecule/score list. Ignored when `--resume` is given. |
| `--resume` | off | Resume a prior run from its JSON sidecar; skips context seeding + the initial model turn and continues from the last assistant message. |
| `--results-dir` | `./results` | Where timestamped reports + sidecars are written. |
| `--trace` | off | Write the full thinking + tool-call trace into the results `.md`. |
| `--rdkit-verbose` | off | Re-enable RDKit `SMILES Parse Error` stderr (silenced by default). |
| `--quiet` | off | Suppress thinking/content/tool prints to stdout. |
| `--self-test` | — | Run the chemistry tools directly with no LLM keys, then exit. |
| `--list-models` | — | Print the built-in model list and exit. |

## Models

The built-in Ollama set (see `--list-models`) is the cloud catalog with the
`-cloud`/`:cloud` suffix dropped — that suffix is a catalog label; the **chat API
name drops it** (e.g. `gemma4:31b-cloud` → `gemma4:31b`, `glm-5.2:cloud` →
`glm-5.2`). `new_models_to_add.txt` lists the catalog names; `molopt.py` uses the
stripped API names. All five current models support thinking + tool-calling
simultaneously (verified by `test_models.py`), so `NO_THINK_MODELS` is empty; the
old model list and old no-think set are kept as comments for backwards
compatibility.

## Outputs

Each run writes `results/<model>_<protein>_<timestamp>.md` containing the turn
headers, the main model's proposals, and the adversary feedback. By default only
terse section headers + text go to the `.md`; the full thinking/tool-call trace
goes to stdout (use `--trace` to include it in the `.md` too). The final line
reports the terminal status: `# Session end: Done` (model signalled Done) or a
`MAX_TURNS` message.

### Example: GLM-5.2 + Claude adversary optimizing HMGCR

A full adversarial run (the command used for the example below):

```bash
python3 -u molopt.py --protein HMGCR --model glm-5.2 \
  --adversary anthropic --adversary-model claude-haiku-4-5-20251001 \
  --max-turns 2 --max-tool-calls 3
```

`--max-tool-calls 3` is the practical ceiling: each `grow_cycle` docks ~45
molecules, so the default `12` would mean ~540 dockings per turn (~75 min/turn).
`3` keeps a turn to ~135 dockings (~15-20 min) while still letting the model
explore before the adversary critiques. (Once the cap is hit the model is
nudged to emit a text summary — *without* saying `Done` — so the adversary
loop still runs.)

The run completed two full adversary↔main cycles (initial model turn, then
Turn 1 and Turn 2), each adversary critique driving a refinement:

| Phase | What happened |
|-------|---------------|
| Initial turn | Model explores the coumarin-flavone scaffold via `grow_cycle`; reaches ~-9.3 with bis-phenyl carboxylates (LogP ~5.6, **fails Lipinski**) |
| Adversary 1 | "Analysis of Proposed Molecules" — flags the over-lipophilic phenyl-rich hits, suggests heteroaromatic replacements |
| Turn 1 | Model applies the suggestion: replaces pendant phenyl with 3-pyridinyl |
| Adversary 2 | "Critical Analysis of Proposed Molecules" — confirms the pyridine gain, pushes for F substitution + validation |
| Turn 2 | Model lands the breakthrough: **-9.5** with LogP **4.44** (now **passes Lipinski**) |

Best molecules (dockstring units, more negative is better; Mac scores — see
the macOS caveat below):

| # | SMILES | Score | QED | LogP | Lipinski | SAS |
|---|-------|:---:|:---:|:---:|:---:|:---:|
| 1 (di-pyridine + F) | `O=c1cc(-c2c(c7cccnc7)cc(F)cc2)oc2c(c7cccnc7)ccc(C(C(=O)[O-]))c12` | **-9.5** | ~0.40 | ~4.0 | ✅ | pending |
| 2 (di-pyridine + Me) | `O=c1cc(-c2c(c7cccnc7)cc(C(C))cc2)oc2c(c7cccnc7)ccc(C(C(=O)[O-]))c12` | **-9.5** | 0.371 | 4.44 | ✅ | 2.98 |
| 3 (phenyl + Me) | `O=c1cc(-c2c(c7ccccc7)cc(C(C))cc2)oc2cccc(C(C(=O)[O-]))c12` | **-9.0** | 0.523 | 3.98 | ✅ | 2.66 |

The SAR the loop converged on: keep the chromone carboxylate anchor
(`C(C(=O)[O-])`), and **replace pendant phenyls with 3-pyridinyl** — the
nitrogen adds H-bond acceptors and polar surface area, improving *both* binding
(-9.3 → -9.5) and drug-likeness (LogP 5.65 → 4.44). Position matters:
3-pyridinyl (-9.5) >> 4-pyridinyl (-8.9) >> 2-pyridinyl (-8.3).

Results for this run: `results/glm-5.2_HMGCR_2026-06-28_12-44-57.md`
(+ `.json` messages sidecar — see [Resuming a run](#resuming-a-run)).

### Example: DeepSeek-v4-pro + Claude adversary optimizing HMGCR

```bash
python3 -u molopt.py --protein HMGCR --model deepseek-v4-pro \
  --adversary anthropic --adversary-model claude-haiku-4-5-20251001 \
  --max-turns 2 --max-tool-calls 3
```

This run was **started fresh, killed during Turn 1, then resumed** from its JSON
sidecar (`--resume results/deepseek-v4-pro_HMGCR_2026-06-29_10-29-38.json`). The
resume completed Turn 1 and Turn 2, ending with `MAX_TURNS_REACHED` and a full
final summary.

| Phase | What happened |
|-------|---------------|
| Initial turn | Model explores flavone scaffolds and finds a **−9.0** nitrovinyl + carboxylate lead, but with QED 0.514 and **2 undesirable moieties** |
| Adversary 1 | Flags steric/charge risks and warns that estimated scores are untested; pushes for neutral analogs, better drug-likeness, and validation |
| Turn 1 | Model empirically checks Lipinski/SAS/NP and tests third substituents; discovers **C(=O)N at position 8** improves the lead to **−9.3** |
| Adversary 2 | Confirms position 8 is optimal but re-raises the binding-site concern and nitrovinyl liability; asks for Rosuvastatin comparison |
| Turn 2 | Model docks Rosuvastatin, confirms the molecules hit a **different (likely allosteric) pocket**, and finalizes 5 candidates |

Best molecules (verification values from `verify_results.py`):

| # | SMILES | Score | QED | LogP | SAS | NP |
|---|-------|:---:|:---:|:---:|:---:|:---:|
| 1 (CF₃ + COO⁻ + C(=O)N @pos8) | `NC(=O)c1cccc2oc(-c3cccc(CC(=O)[O-])c3C(F)(F)F)cc(=O)c12` | **−9.10** | 0.731 | 1.87 | 3.03 | −0.00 |
| 2 (CF₃ + COO⁻ + C(=O)N @pos5) | `NC(=O)c1cccc2c(=O)cc(-c3cccc(CC(=O)[O-])c3C(F)(F)F)oc12` | −9.00 | 0.731 | 1.87 | 3.01 | −0.12 |
| 3 (SO₂NH₂ + COO⁻) | `NS(=O)(=O)c1c(CC(=O)[O-])cccc1-c1cc(=O)c2ccccc2o1` | −8.90 | 0.718 | 0.40 | 2.84 | −0.28 |
| 5 (NO₂vinyl + COO⁻ + C(=O)N @pos8) | `NC(=O)c1cccc2oc(-c3cccc(CC(=O)[O-])c3C=C[N+](=O)[O-])cc(=O)c12` | **−9.30** | 0.485 | 1.10 | 3.25 | 0.21 |

Molecule **#1** is the standout: potent (−9.1), drug-like (QED 0.731, 0 undesirable moieties, SAS 3.03), and synthetically accessible. Molecule #5 is the most potent overall (−9.3) but carries the nitrovinyl liability.

The key SAR: keep the **ortho CF₃ + carboxylate** on the pendant phenyl and add a
**carboxamide at position 8** of the flavone core for an extra H-bond to MET640.
Position 3 is sterically disfavoured. The binding pocket overlaps with
Rosuvastatin only at **ASN639**, suggesting an allosteric site rather than the
orthosteric HMGCR active site.

Result files:
- Original run: `results/deepseek-v4-pro_HMGCR_2026-06-29_10-29-38.md`
- Resumed run: `results/deepseek-v4-pro_HMGCR_2026-06-29_11-44-49.md`

### Example: Kimi-k2.7-code + Claude adversary optimizing HMGCR

```bash
python3 -u molopt.py --protein HMGCR --model kimi-k2.7-code \
  --adversary anthropic --adversary-model claude-haiku-4-5-20251001 \
  --max-turns 2 --max-tool-calls 3
```

This run completed two full adversary↔main cycles, ending with
`MAX_TURNS_REACHED` and a full final summary.

| Phase | What happened |
|-------|---------------|
| Initial turn | Model starts from the provided flavone leads and finds **−9.0** with a 6-hydroxy-2,4-difluorophenyl flavone carboxylate; proposes 5 candidates |
| Adversary 1 | Challenges whether the −9.0 estimate is real, asks for Lipinski/QED/SAS validation, suggests positional isomers and alternative charged anchors |
| Turn 1 | Model validates: the −9.0 is real; discovers 2,4-diF-phenyl is optimal among fluorophenyls; rules out 6-OMe (−7.7) and alternative carboxylate positions (−8.2/−8.4) |
| Adversary 2 | Pushes for larger aryl groups (naphthyl/biphenyl), sulfonate/phosphonate anchors, and final drug-likeness assessment |
| Turn 2 | Model finds a **2-naphthyl** scaffold is superior; the 2-naphthyl sulfonate reaches **−9.4**, while the 2-naphthyl carboxylate is close behind at **−9.3** |

Best molecules (verification values from `verify_results.py`):

| # | SMILES | Score | QED | LogP | SAS | NP |
|---|-------|:---:|:---:|:---:|:---:|:---:|
| 1 (2-naphthyl sulfonate) | `O=c1cc(-c2ccc3ccccc3c2)oc2cccc(CS(=O)(=O)[O-])c12` | **−9.40** | 0.517 | 3.66 | 2.63 | 0.15 |
| 2 (2-naphthyl carboxylate) | `O=C([O-])Cc1cccc2oc(-c3ccc4ccccc4c3)cc(=O)c12` | −9.30 | 0.579 | 2.91 | 2.57 | 0.29 |
| 3 (2,4-diF-phenyl + 6-OH) | `O=C([O-])Cc1cc(O)cc2oc(-c3ccc(F)cc3F)cc(=O)c12` | −9.00 | **0.789** | 1.74 | 2.92 | 0.39 |
| 4 (2,4-diF-phenyl) | `O=C([O-])Cc1cccc2oc(-c3ccc(F)cc3F)cc(=O)c12` | −8.90 | 0.741 | 2.03 | 2.73 | −0.27 |
| 5 (para-F-phenyl) | `O=C([O-])Cc1cccc2oc(-c2ccc(F)cc2)cc(=O)c12` | −8.80 | 0.740 | 1.89 | 2.59 | −0.02 |

The key SAR: a **2-naphthyl** group at the flavone 2-position gives better
shape complementarity than phenyl, boosting scores by ~0.4–0.5 kcal/mol. The
**sulfonate** anchor scores slightly higher than carboxylate (−9.4 vs −9.3) but
has lower QED (0.517 vs 0.579). If drug-likeness is the priority, **molecule #3**
is the best balanced: QED 0.789, SAS 2.92, and still −9.0. The 2,4-difluorophenyl
+ 6-OH pattern had already reached −9.0 in Turn 1 and remains the most
ligand-efficient option.

Result file: `results/kimi-k2.7-code_HMGCR_2026-06-29_12-18-58.md`

### Resuming a run

Every run also writes a JSON messages sidecar next to the `.md` report (same
timestamp stem, `.json`), updated incrementally after each turn (atomically, so
a killed run still leaves a usable sidecar of the last completed turn). Use
`--resume <sidecar.json>` to pick up a stopped run from where it left off
instead of re-seeding from `code/adversarial_set.md`:

```bash
# resume the example run from its sidecar, for one more refinement turn
python3 -u molopt.py --resume results/glm-5.2_HMGCR_2026-06-28_12-44-57.json \
  --protein HMGCR --model glm-5.2 --adversary anthropic \
  --adversary-model claude-haiku-4-5-20251001 --max-turns 1
```

On resume the conversation is seeded from the sidecar (the model already knows
what it proposed and scored), context-file seeding and the initial model turn
are skipped, and refinement continues from the last assistant message. A fresh
timestamped `.md` + sidecar are written for the resumed session; the source
sidecar is never modified.

## Verifying results

`verify_results.py` parses the final model-response block from a `molopt.py`
`.md` report, extracts every RDKit-valid SMILES, and recomputes the five project
metrics by **reusing the existing helpers** (`docking_module.scoring_function`,
`MolPropOp.lipinski`, `docking_module.calculate_SAS_and_NP`). Novel molecules
are inserted into a local sqlite DB (`molecules.sqlite`), keyed by canonical
SMILES + InChIKey; molecules already present are skipped so the DB can be
incrementally populated across many runs.

```bash
# verify the most recent run (auto-picks newest results/*.md)
python3 code/verify_results.py

# verify a specific run
python3 code/verify_results.py results/glm-5.2_HMGCR_2026-06-28_12-44-57.md

# dry-run: just list the SMILES that would be extracted, no docking/DB writes
python3 code/verify_results.py results/<run>.md --dry-run

# change DB path or minimum heavy-atom filter (default 5, filters fragment noise)
python3 code/verify_results.py results/<run>.md --db molecules.sqlite --min-heavy-atoms 5
```

The protein target is read from the `.md` header (`# protein: ...`), and the
docking target is configured the same way `molopt.py` does at runtime (mutating
the shared `scoring_args` list). For a partial run whose last section is the
`# Initial model response:`, the verifier falls back to that block, so even a
killed run's initial proposals can be captured.

## Replicate comparison (`run_replicates.py` + `analyze_replicates.py`)

To compare how the **adversary pairing** affects the designed molecules, run several
replicates of each adversary set, then analyze the final compounds by property.

The four adversary sets (each is one (proposer, adversary) pairing; the proposer has
the chemistry tools, the adversary critiques):

| set label              | script        | proposer (tools) | adversary (critique) |
|------------------------|---------------|------------------|----------------------|
| `openai_vs_anthropic`  | `molopt_oa.py` `--start openai`    | OpenAI    | Anthropic |
| `anthropic_vs_openai`  | `molopt_oa.py` `--start anthropic` | Anthropic | OpenAI    |
| `ollama_vs_openai`     | `molopt.py` `--adversary openai`    | Ollama    | OpenAI    |
| `ollama_vs_anthropic`  | `molopt.py` `--adversary anthropic` | Ollama    | Anthropic |

`run_replicates.py` runs `--replicates` sessions of each set **sequentially** (docking
is CPU-bound, so no parallelism). Each replicate gets its own `--results-dir` under
`results/batches/<batch_id>/<set>/rep<N>/` (one run → one `.md` + one `.json` sidecar),
and a `manifest.json` at the batch root records every job. It is resumable: a replicate
with a terminal sidecar (`Done` / `max_turns_reached`) is skipped on re-launch, so a
killed batch loses no completed work (`--force` re-runs all). It refuses to run unless
`obabel` is on PATH — i.e. the venv is activated (see the note in Install).

```bash
source fao-env/bin/activate          # required (openbabel bindings + obabel CLI)

# quick: 1 rep of one set, see the commands only
python code/run_replicates.py --dry-run --replicates 1 --sets openai_vs_anthropic

# real batch: 3 reps of all 4 sets. Launch detached so a long batch survives the shell:
fao-env/bin/python -c "import subprocess; subprocess.Popen(['fao-env/bin/python','code/run_replicates.py','--replicates','3'], stdout=open('run_replicates.log','ab'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)"
tail -f run_replicates.log
```

Then analyze. `analyze_replicates.py` reads the manifest, extracts the final-compound
SMILES from each completed run's last model-response block (reusing `verify_results.py`'s
extractor), recomputes the five metrics (docking, QED, aLogP, SAS, NP) via the project
helpers, and writes to `results/batches/<batch_id>/analysis/`:

- `compounds_<batch>.csv` — one row per proposed molecule, tagged with
  `set_label, replicate, proposer_model, adversary_model, protein` + the 5 metrics.
- `summary_<batch>.csv` — per-set aggregate stats (n_compounds, n_unique, docking
  mean/median/best, QED/aLogP/SAS/NP means).
- `best_per_replicate_<batch>.csv` — best-by-docking molecule per (set, replicate).
- `dock_dist_by_set.png`, `best_dock_by_replicate.png`, `qed_vs_dock.png`,
  `property_dist_by_set.png`.

Unlike `verify_results.py` (which dedups globally and skips known molecules), the
analyzer calls the extractor per-run independently, so the **same SMILES appearing in
multiple replicates is kept** — that overlap is part of the comparison.

```bash
python code/analyze_replicates.py --batch-dir results/batches/<batch_id>
# CPU-light check: RDKit-only metrics + all CSVs/plots, no dockstring recomputation:
python code/analyze_replicates.py --batch-dir results/batches/<batch_id> --skip-docking
```

`--skip-docking` leaves the `docking` column blank and skips the docking-dependent
plots; the QED/aLogP/SAS/NP metrics (RDKit-only) and the property/QED plots are still
produced. Requires `matplotlib` (see the note in Install).

## Smoke-testing models

`test_models.py` is a fast standalone check (no chem-stack import) that, per
model: resolves the API name (stripped first, raw fallback), tests tool-use,
tests `think=True` + tools together, and runs one short adversary turn. Keys are
resolved the same way as `molopt.py` (CLI flags → env → `.env`).

```bash
source ~/.zshrc   # for the API keys
python3 -u code/test_models.py                       # all models in new_models_to_add.txt
python3 -u code/test_models.py --only glm-5.2:cloud  # just one
```

## Notes & caveats

- **macOS docking**: dockstring prints a `DockstringWarning` that Mac scores may
  not match the Linux baselines in the DOCKSTRING paper. Scores are usable for
  relative ranking within a run; don't compare them directly to paper baselines.
- **RDKit stderr**: invalid-SMILES `SMILES Parse Error` lines are silenced by
  default (`RDLogger.DisableLog('rdApp.*')`); failures are surfaced to the model
  as `invalid SMILES, skipped` instead. See
  [`suppressing_rdkit_smiles_errors.md`](suppressing_rdkit_smiles_errors.md) for a
  drop-in copy of that fix for other repos. `--rdkit-verbose` re-enables the log.
- **`ANTHROPIC_BASE_URL` gotcha**: the Anthropic adversary routes to
  `api.anthropic.com` explicitly so a locally-set `ANTHROPIC_BASE_URL` proxy
  (which 404s on `claude-*` names) is bypassed. Use `--anthropic-base-url` to
  point at your own proxy if you actually want one.
- **Keys per shell**: each fresh shell does not auto-load `.zshrc` exports, so
  `source ~/.zshrc` (or use `.env`) before launching.

## Standalone Vina docking (`vina_dock.py`)

`vina_dock.py` (repo root) is a **standalone** AutoDock Vina harness, deliberately
separate from the `molopt.py` / `docking_module.py` pipeline (and not wired into
the optimizer). It docks **any user-provided receptor PDB** (not just dockstring's
58 targets) against a **SMILES** ligand, using only tools already in this env:

- the **Vina 1.1.2 binary vendored inside dockstring** (called as a subprocess),
- **Open Babel** for PDB↔PDBQT conversion (receptor needs `-xr` for a rigid
  PDBQT, otherwise Vina rejects the torsion-tree records),
- **RDKit** for ligand 3D embedding,
- **scipy** (`cKDTree`) for the blind pocket detector.

### Agent API (Python)

The primary entry point for use as an agent tool is the `blind_dock()` function:

```python
from vina_dock import blind_dock
report = blind_dock("my_receptor.pdb", ["c1ccc(O)cc1", "CCO"], npockets=1)
print(report)
```

`blind_dock(receptor_pdb, smiles_list, npockets=1, ...) -> str` detects pockets
once and reuses them for every ligand, docks each SMILES into the top `npockets`
(default 1 — the validated #1 site; raise to 3 for a safety net on a novel
receptor), and returns a multi-line report (receptor + receptor-PDBQT path +
pocket centers; per-molecule score + pocket used + pose-SDF path; overall best
molecule). **Per-molecule failures are caught and reported, not raised** — one
bad SMILES never aborts the batch. Only setup failures (missing receptor, no
obabel/Vina, no pockets detected) raise `DockError`.

Persisted artefacts (next to the input PDB): `<stem>.pdbqt` (rigid receptor,
built once) and `<stem>_<i>.sdf` (top-3 poses for molecule i, from its
best-scoring pocket). Per-run intermediates go to a temp dir cleaned at the end.

### CLI

```
# explicit site
python3 code/vina_dock.py --receptor my_receptor.pdb --smiles 'c1ccc(O)cc1' \
    --center 9.25 6.17 -7.0 --size 25 25 25

# blind (binding site unknown)
python3 code/vina_dock.py --receptor my_receptor.pdb --smiles 'c1ccc(O)cc1' --blind
```

CLI failures raise `DockError`, caught at the entry point into a clean
`sys.exit("vina_dock: <msg>")`. CLI run intermediates go to
`vina_run_<timestamp>/` (gitignored).

### Blind pocket detection

`--blind` / the `blind_dock()` API scan the receptor for low-atom-density
cavities: a scipy `cKDTree` buriedness grid (atoms within an 8 Å shell), keep the
top `top_frac` (0.06) most-buried voxels, DBSCAN-cluster, and rank pockets by
`buriedness · log1p(n_voxels)` with `min_samples=25` to fragment diffuse
surface blobs. No `fpocket`/`p2rank` install needed. Validated 5/5 on the DUD-E
receptors — the known dockstring site is the **#1** detected pocket on all five
(5.4–7.5 Å from the dockstring box center), and cross-validated on SULT1A3
(PDB 2A3R): both substrate sites recovered as top-2, blind-docked dopamine
6.6 Å from the crystallographic ligand.

For **known dockstring targets**, just use `molopt.py`/dockstring directly —
`vina_dock.py` is for receptors outside dockstring's set.

## TODO / later polish

- Suppress the per-molecule dockstring `DockstringWarning` (Mac-vs-Linux) that
  floods `run.log` — add
  `warnings.filterwarnings('ignore', category=DockstringWarning)` near the
  RDKit suppress in `molopt.py` (or `docking_module.py`).
- A few `SMILES Parse Error` lines still leak to stderr per run (oddt-internal
  parse edge; `RDLogger.DisableLog('rdApp.*')` already catches most). Low volume
  (~1-3 per ~hundreds of parses), not worth fixing now but noted here.