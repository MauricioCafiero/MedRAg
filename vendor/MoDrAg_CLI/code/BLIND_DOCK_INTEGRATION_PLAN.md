# Integrating `dock_assist` blind docking into MoDrAg_CLI

Plan for folding the blind docking code from
[github.com/MauricioCafiero/dock_assist](https://github.com/MauricioCafiero/dock_assist)
back into this (modrag) repo. The blind docking code runs on the `dockstring`
(vendored AutoDock Vina) and `openbabel-wheel` libraries already present here.

## What dock_assist contributes

| dock_assist file | What it is | Collisions with modrag |
|---|---|---|
| `code/vina_dock.py` (793 lines) | **The real new work** — blind pocket detection (`find_pockets`), Vina/obabel plumbing, `blind_dock` / `blind_dock_agent`. Stdlib-only at module top; numpy/scipy/sklearn/rdkit/dockstring imported lazily inside functions. | **None** — no name overlaps with modrag. Essentially drop-in. |
| `code/modrag_protein_functions.py` | A **trimmed, divergent copy** of modrag's protein module — only 4 functions. | **3 collisions** (breaking signature/arity changes) + 1 new function. |
| `code/dock_assist.py` | A clone of `modrag.py` (same Ollama + rich + tool-dispatch shape). | Reference for wiring only; folded into the existing `modrag.py`. |
| `code/unit_test.py` | Trivial one-off. | Discarded. |

Core insight: **`vina_dock.py` is the gift; `modrag_protein_functions.py` is the hazard.**
Do not copy the latter verbatim.

## 1. New dependencies

Add to `requirements.txt`:

```
scipy
numpy
```

Everything else dock_assist needs is **already** in modrag's `requirements.txt`
(`rdkit`, `scikit-learn`, `pubchempy`, `requests`, `rcsb-api`, `dockstring`,
`openbabel-wheel`, `rich`, `ollama`). `scipy` is the only genuinely new *direct*
dep (imported directly in `find_pockets` via `scipy.spatial.cKDTree`); `numpy`
is also imported directly and was only present transitively before.

External binaries (no system installs — both come from pip packages already listed):
- `obabel` CLI → from `openbabel-wheel`.
- AutoDock Vina 1.1.2 binary → vendored inside `dockstring`, located at runtime by `find_vina_bin`.

## 2. Add `vina_dock.py` as a new module — drop in

`code/vina_dock.py` copied verbatim. No renames required — none of its names
(`blind_dock`, `blind_dock_agent`, `find_pockets`, `build_receptor_pdbqt`,
`build_ligand_pdbqt`, `run_vina`, `parse_vina_log`, `find_vina_bin`,
`parse_pdb_heavy_atoms`, `_crystal_ligand_coords`, `_pose_min_distance`,
`DockError`, etc.) collide with modrag.

Notes:
- Module global `FALLBACK_TO_SECOND_POCKET` (default `False` in the file) is set
  by `modrag.py` after import, exactly as dock_assist does.
- `NEAR_LIGAND_CUTOFF = 5.0` mirrors the new `NEARBY_DISTANCE_CUTOFF = 5.0`.
- The file is fully standalone: it does **not** import `modrag_protein_functions`
  and does **not** read `print_flag` (uses `print` directly).
- `blind_dock_agent(receptor_pdb, smiles_list)` has a clean Args/Returns docstring,
  so the Ollama client builds a tool schema from it automatically.

## 3. The three divergent functions — do NOT overwrite; rename + merge

dock_assist's `modrag_protein_functions.py` redefines three functions that
already exist in modrag with incompatible signatures/returns. Copying them in
verbatim would break modrag's callers.

### 3a. `get_protein_from_pdb` — port behavior as a NEW function `get_pdb_file`
- **modrag:** `get_protein_from_pdb(pdb_id) -> str` (raw PDB text, no file write).
  Used by `pdb_node` — left untouched.
- **dock_assist:** `get_protein_from_pdb(pdb_id, protein_name) -> str`
  (downloads + caches to `pdb_files/{protein_name}_{pdb_id}.pdb`, glob-based reuse).
- **Decision:** Add `get_pdb_file(pdb_id, protein_name)` (the dock_assist version)
  as a new function. Keep the existing `get_protein_from_pdb(pdb_id)` as-is.
  The blind-docking flow calls `get_pdb_file` to land a cached receptor on disk;
  the file path it returns is what `blind_dock_agent` consumes.

### 3b. `find_PDBID_node` — keep modrag's tuple return
- **modrag:** `find_PDBID_node(...) -> (list[str], str, None)`
- **dock_assist:** `find_PDBID_node(...) -> str` (drops the list)
- **Decision:** Keep modrag's version (dispatcher `str()`-ifies the tuple fine).
  The blind-docking flow does not need the divergent return.

### 3c. `smiles_node` — keep modrag's tuple return
- **modrag** (in `modrag_molecule_functions.py`): `smiles_node(...) -> (list, str, None)`
- **dock_assist** (in its protein module): `smiles_node(...) -> str`
- **Decision:** Keep modrag's version.

### 3d. `check_nearby_molecules` — genuinely new, add as-is
- `check_nearby_molecules(pdb_filepath, ligand_filepath) -> str` has no modrag
  equivalent. Added to `code/modrag_protein_functions.py` with module constant
  `NEARBY_DISTANCE_CUTOFF = 5.0`.

## 4. Wiring into `modrag.py` (follow `NODE_INTEGRATION_SKILLS.md`)

modrag's tool dispatcher does `result = fn(**args)` then
`messages.append({'role': 'tool', ..., 'content': str(result)})` — i.e. it
`str()`-ifies whatever a tool returns. So the new functions returning plain
strings (like the existing tuple-returning nodes) integrate without any
return-shape wrapping.

Four registration touch-points in `code/modrag.py`:
1. **Imports** — `from vina_dock import blind_dock_agent` and
   `from modrag_protein_functions import check_nearby_molecules, get_pdb_file`;
   `import vina_dock`; set `vina_dock.FALLBACK_TO_SECOND_POCKET` and
   `modrag_protein_functions.print_flag` after import.
2. **`tools` list** — add `check_nearby_molecules`, `get_pdb_file`,
   `blind_dock_agent`.
3. **`available_functions` dict** — map the three names to callables.
4. **`client.chat(tools=...)`** — passes the `tools` list; nothing extra once (2) is done.

## 5. Runtime file hygiene

dock_assist creates `pdb_files/` at runtime and deletes leftover `*.sdf` on
startup (keeps `*.pdb`). Mirrored by adding `pdb_files/` to `.gitignore`.

## 6. Discrepancies resolved while porting
- dock_assist's committed code set `use_second_pocket_fallback = True` while its
  README said off. In modrag the fallback is enabled via `modrag.py`
  (`vina_dock.FALLBACK_TO_SECOND_POCKET = True`) to match the committed behavior.

## Suggested order of work
1. Add `scipy`, `numpy` to `requirements.txt`; ensure installed in `modrag-env`.
2. Copy `vina_dock.py` → `code/vina_dock.py`.
3. Add `check_nearby_molecules` + `NEARBY_DISTANCE_CUTOFF` to
   `code/modrag_protein_functions.py`.
4. Add `get_pdb_file(pdb_id, protein_name)` to `code/modrag_protein_functions.py`;
   leave existing `get_protein_from_pdb`, `find_PDBID_node`, `smiles_node` untouched.
5. Wire the three new tools into `modrag.py`.
6. Update `.gitignore` for `pdb_files/`.
7. Smoke-test the import + a `--print` run on a known receptor (e.g. 2A3R / SULT1A3).

## Net effect
One new module (`vina_dock.py`), one new direct dependency (`scipy`, +`numpy` made
explicit), two new functions in the protein module (`check_nearby_molecules`,
`get_pdb_file`), three new LLM tools registered in `modrag.py`, and **zero breaks**
to existing modrag functions/signatures. The divergent `smiles_node` /
`find_PDBID_node` from dock_assist are discarded in favor of modrag's existing
tuple-returning versions; only `get_protein_from_pdb`'s caching behavior is ported
(as a new `get_pdb_file`).