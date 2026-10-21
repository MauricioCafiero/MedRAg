# Session memory for MoDrAg_CLI

Design for a `memory` / `recall` keyword pair that saves a summary of the
current session (PDB + docking-pose SDF + CSV file names) to a persistent
**vault** at the repo root, moves the important docking-pose SDFs into the
vault, and can load a prior session's summary back into the conversation so
the model has its context.

Implemented in `code/modrag_memory.py` (stdlib only) and wired into
`code/modrag.py`.

## What was added

| File | Change |
|---|---|
| `code/modrag_memory.py` | **New module.** `save_session`, `recall_session`, `list_sessions`, `is_important_sdf`, plus internal `_collect_files`, `_summarize_messages`, `_topic_from_messages`. Stdlib only. |
| `code/modrag.py` | Import `modrag_memory`, propagate `print_flag`, track `session_start_time` (reset in `start_chat` and after each save), add `handle_memory_prompt()` + `VAULT_DIR`, and short-circuit the keyword prompts in both input branches (first prompt and loop). |
| `code/tool_tests.py` | Test 9: self-contained save/recall round-trip in a temp vault (no real vault touched). |
| `README.md` | "Session Memory" usage section. |

No third-party dependencies. No existing tool signatures or `tools`/`available_functions` lists changed — memory is a prompt keyword, not a tool the model calls, so it is deterministic.

## Trigger mechanism

Keyword-only, handled in `modrag.py` *before* the prompt is sent to Ollama
(alongside the existing `quit` check). This makes behavior deterministic and
avoids a model round-trip.

| Keyword | Action |
|---|---|
| `memory` / `save memory` / `remember` | `save_session(...)`; reset `session_start_time` so the next memory covers only work after now. |
| `recall` | `list_sessions(...)` — print `vault/INDEX.md`. |
| `recall last` | `recall_session(which='last')` — load the most recent summary into `messages` as context. |
| `recall <date>` (e.g. `recall 2026-07-19`) | `recall_session(which=<date>)` — load the latest folder whose name starts with that date. |

`handle_memory_prompt` returns `(handled, output)`. When `handled` is True,
`modrag.py` prints the output via `rich` Markdown and skips `chat_turn`.
For `recall`, the summary is *also* appended to `messages` as a user-role
context message, so the next real question the user asks is answered with
that prior session in context.

## Session scoping

`session_start_time = time.time()` is captured at module load and reset in
`start_chat()`. `_collect_files` walks `pdb_files/` and `../scratch/` and
keeps only files whose `mtime >= session_start_time`, so a save only ever
captures artifacts from *this* session's window. After a save, the window
is reset to `time.time()`.

## "Important" SDF rule

`is_important_sdf(path)`:
- **Important (moved to vault):** `best_pose.sdf`, and any `<stem>_<idx>.sdf`
  (the `blind_dock` pose outputs written next to the receptor PDB).
- **Not important (left in place):** `ligand_*.sdf`, `receptor*.sdf`
  (intermediates), and anything else.

Pose outputs are the real docking results; intermediates live in the temp
`work` dir that `vina_dock` already removes. PDB and CSV files are listed in
the summary but **never moved or copied** — PDB persistence (the `pdb_files/`
cache reused by `get_pdb_file`) is untouched.

## Vault layout (append-only)

```
vault/
  INDEX.md                          # one line appended per save:
                                     # - <ts> | <topic> | PDB=n | SDF=n | CSV=n | <folder>
  2026-07-19_1714_PCSK9/
    session_summary.md
    poses/PCSK9_6U2N_0.sdf
```

- Each save creates a **new** timestamped folder
  `YYYY-MM-DD_HHMM_<topic>`. If two saves land in the same minute a `_2`,
  `_3` suffix is added. Nothing is ever overwritten.
- `<topic>` is filesystem-safe: the stem of the first PDB touched this
  session, else the first `PROT_PDBID`-like token found in a user prompt,
  else `session`.
- `INDEX.md` is append-only, so `recall` listing is a cheap file read
  rather than a directory scan.

## Summary contents

`session_summary.md` contains:
1. Header with save timestamp and topic stem.
2. **Files this session** grouped into: PDB files (named, not moved),
   docking-pose SDFs (moved to vault), other SDFs (not moved), CSVs (named,
   not moved).
3. **Session transcript** — an excerpt built from `messages`: user prompts,
   assistant final content, and trimmed tool results (so docking reports /
   scores are captured). Both dict entries (user/system/tool) and the
   ollama `Message` object appended for assistant turns are handled
   defensively via `getattr`.

## Wiring details in `modrag.py`

- Top of file: `import modrag_memory`.
- `print_flag` propagation block: `modrag_memory.print_flag = print_flag`,
  and `session_start_time = time.time()`.
- `start_chat()`: declare `global session_start_time` and reset it.
- After `chat_turn` definition: `VAULT_DIR = '../vault'` and
  `handle_memory_prompt(prompt)` (returns `(handled, output)`).
- Both input branches: call `handle_memory_prompt(next_prompt)` first; if
  `handled`, print the output (and `continue` in the loop) and skip
  `chat_turn`.

## Test

`tool_tests.py` Test 9 runs a self-contained save → move → recall round-trip
in a temp vault, patching `modrag_memory.PDB_DIR` / `SCRATCH_DIR` to the temp
tree so the real `pdb_files/` / `scratch/` / `vault/` are never touched. It
asserts: pose SDF moved, ligand intermediate + PDB left in place, summary +
index written, `recall last` returns the score, unknown date returns
`None`.

A standalone smoke test (run during development) additionally verified the
append-only behavior: two saves produce two distinct folders.