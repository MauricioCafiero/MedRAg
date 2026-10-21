"""
modrag_memory.py — session memory for Modrag.

Two keyword-driven entry points (wired up in modrag.py, NOT called by the
model as a tool):

  save_session(messages, start_time, vault_dir)
      Write a Markdown summary of the current session to a new timestamped
      folder under <vault_dir>, and MOVE the "important" docking-pose SDF
      files (the `<stem>_<idx>.sdf` pose outputs and `best_pose.sdf`) into
      that folder's `poses/` subdirectory. PDB and CSV files are named in the
      summary but left in place (persistence untouched).

  recall_session(vault_dir, which) / list_sessions(vault_dir)
      Read a previously saved session summary back, so the model can be given
      the context of an earlier session at the start of a new one.

Vault layout (append-only, never overwritten):

  vault/
    INDEX.md
    2026-07-19_1530_PCSK9/
      session_summary.md
      poses/PCSK9_6U2N_0.sdf

Stdlib only — no third-party deps, so this works regardless of which
modrag modules are imported.
"""

import os
import re
import time
import shutil
from datetime import datetime

# Module-level print flag - set from modrag.py (matches the other modules)
print_flag = False

# Directories scanned for session artifacts, relative to the modrag CWD
# (modrag runs from code/, so all of these resolve to the repo root:
# ../pdb_files, ../scratch — matching ../images and ../vault used elsewhere).
PDB_DIR = '../pdb_files'
SCRATCH_DIR = '../scratch'

# Pose-output SDF naming: blind_dock writes `<stem>_<idx>.sdf` next to the
# receptor PDB; the vina_dock CLI writes `best_pose.sdf`. Both are "important".
# Intermediates (`ligand_*.sdf`, `receptor*.sdf`) are NOT important and are
# skipped.
_POSE_SDF_RE = re.compile(r'^(.+)_\d+\.sdf$')


def is_important_sdf(path):
  '''
    Returns True if the given SDF file is a docking *result* (a pose output)
    rather than a throwaway intermediate.

      Args:
        path: path to an .sdf file
      Returns:
        bool: True if this SDF should be moved to the vault
  '''
  base = os.path.basename(path)
  if base == 'best_pose.sdf':
    return True
  if base.startswith('ligand') or base.startswith('receptor'):
    return False
  return bool(_POSE_SDF_RE.match(base))


def _collect_files(start_time, search_dirs):
  '''
    Walks search_dirs and returns files whose mtime >= start_time, grouped by
    role: {'pdb': [...], 'sdf_important': [...], 'sdf_other': [...], 'csv': [...]}.
    Missing dirs are silently skipped.
  '''
  found = {'pdb': [], 'sdf_important': [], 'sdf_other': [], 'csv': []}
  for d in search_dirs:
    if not os.path.isdir(d):
      continue
    for root, _dirs, files in os.walk(d):
      for name in files:
        full = os.path.join(root, name)
        try:
          if os.path.getmtime(full) < start_time:
            continue
        except OSError:
          continue
        if name.endswith('.pdb') or name.endswith('.pdbqt'):
          found['pdb'].append(full)
        elif name.endswith('.sdf'):
          (found['sdf_important'] if is_important_sdf(full)
           else found['sdf_other']).append(full)
        elif name.endswith('.csv'):
          found['csv'].append(full)
  for k in found:
    found[k].sort()
  return found


def _topic_from_messages(messages, found):
  '''
    Picks a short, filesystem-safe topic stem for the vault folder name:
    the first receptor/protein name seen in the conversation, else the stem
    of the first PDB file, else 'session'.
  '''
  # First PDB filename stem, e.g. PCSK9_6U2N -> PCSK9_6U2N
  if found['pdb']:
    stem = os.path.splitext(os.path.basename(found['pdb'][0]))[0]
    # drop a trailing _<pdbid> isn't worth it; keep it readable
    return _safe_name(stem)
  # Fallback: scan user prompts for something protein-like
  for m in messages:
    if isinstance(m, dict) and m.get('role') == 'user' and m.get('content'):
      txt = str(m['content'])
      match = re.search(r'\b([A-Z0-9]{3,}_\d[A-Z0-9]{3})\b', txt)
      if match:
        return _safe_name(match.group(1))
  return 'session'


def _safe_name(s):
  return re.sub(r'[^A-Za-z0-9_-]', '_', s)[:40] or 'session'


def _summarize_messages(messages):
  '''
    Builds a human-readable transcript excerpt from the message list for the
    session summary. Handles both dict entries (user/system/tool) and the
    ollama Message object appended for assistant turns.
  '''
  lines = []
  for m in messages:
    # dict entries
    if isinstance(m, dict):
      role = m.get('role', '?')
      if role == 'system':
        continue
      if role == 'tool':
        # tool result: keep a trimmed copy so scores/reports are captured
        content = str(m.get('content', ''))
        name = m.get('tool_name', 'tool')
        lines.append(f'### Tool result ({name})\n')
        lines.append(_trim(content, 1500))
        lines.append('')
        continue
      content = str(m.get('content', '')).strip()
      if content:
        lines.append(f'### {role.capitalize()}\n')
        lines.append(_trim(content, 1000))
        lines.append('')
      continue
    # ollama Message object (assistant turn) — use getattr for safety
    role = getattr(m, 'role', None) or 'assistant'
    content = str(getattr(m, 'content', '') or '').strip()
    tool_calls = getattr(m, 'tool_calls', None) or []
    if tool_calls:
      names = []
      for tc in tool_calls:
        fn = getattr(tc, 'function', None)
        nm = getattr(fn, 'name', None) if fn is not None else None
        names.append(nm or '?')
      lines.append(f'### {role.capitalize()} (called tools: {", ".join(names)})')
    elif content:
      lines.append(f'### {role.capitalize()}')
    else:
      continue
    if content:
      lines.append('')
      lines.append(_trim(content, 1000))
      lines.append('')
  return '\n'.join(lines).strip()


def _trim(text, limit):
  text = text.strip()
  if len(text) <= limit:
    return text
  return text[:limit] + '\n...[truncated]'


def save_session(messages, start_time, vault_dir='../vault'):
  '''
    Writes a session memory: a timestamped folder under vault_dir containing
    session_summary.md and a poses/ folder with the important (pose-output)
    SDFs moved into it. PDB/CSV files are listed by name but left in place.

      Args:
        messages: the modrag global messages list (system + turns)
        start_time: time.time() captured when this session's work began
        vault_dir: where to write the vault (repo-root vault/ by default)
      Returns:
        a short status string describing what was saved and where
  '''
  os.makedirs(vault_dir, exist_ok=True)
  found = _collect_files(start_time, [PDB_DIR, SCRATCH_DIR])

  timestamp = time.strftime('%Y-%m-%d_%H%M')
  topic = _topic_from_messages(messages, found)
  folder = os.path.join(vault_dir, f'{timestamp}_{topic}')
  # avoid clobbering if two saves land in the same minute
  i = 2
  while os.path.exists(folder):
    folder = os.path.join(vault_dir, f'{timestamp}_{topic}_{i}')
    i += 1
  poses_dir = os.path.join(folder, 'poses')
  os.makedirs(poses_dir, exist_ok=True)

  # Move important SDFs into poses/
  moved = []
  for sdf in found['sdf_important']:
    dst = os.path.join(poses_dir, os.path.basename(sdf))
    try:
      shutil.move(sdf, dst)
      moved.append(dst)
    except OSError as e:
      if print_flag:
        print(f'[memory] could not move {sdf}: {e}')

  # Build the summary
  summary_lines = []
  summary_lines.append(f'# Modrag session memory — {timestamp.replace("_", " ")}')
  summary_lines.append('')
  summary_lines.append(f'- Saved: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
  summary_lines.append(f'- Topic stem: {topic}')
  summary_lines.append('')

  summary_lines.append('## Files this session')
  summary_lines.append('')
  summary_lines.append('### PDB files (named, not moved)')
  if found['pdb']:
    for p in found['pdb']:
      summary_lines.append(f'- {os.path.basename(p)}  (`{p}`)')
  else:
    summary_lines.append('- (none)')
  summary_lines.append('')
  summary_lines.append('### Docking-pose SDFs (moved to vault)')
  if moved:
    for p in moved:
      summary_lines.append(f'- {os.path.basename(p)}')
  else:
    summary_lines.append('- (none)')
  summary_lines.append('')
  summary_lines.append('### Other SDFs (not moved)')
  if found['sdf_other']:
    for p in found['sdf_other']:
      summary_lines.append(f'- {os.path.basename(p)}  (`{p}`)')
  else:
    summary_lines.append('- (none)')
  summary_lines.append('')
  summary_lines.append('### CSVs (named, not moved)')
  if found['csv']:
    for p in found['csv']:
      summary_lines.append(f'- {os.path.basename(p)}  (`{p}`)')
  else:
    summary_lines.append('- (none)')
  summary_lines.append('')
  summary_lines.append('## Session transcript')
  summary_lines.append('')
  summary_lines.append(_summarize_messages(messages) or '(no conversation captured)')

  summary_path = os.path.join(folder, 'session_summary.md')
  with open(summary_path, 'w') as f:
    f.write('\n'.join(summary_lines) + '\n')

  # Append a one-line entry to the append-only index
  index_path = os.path.join(vault_dir, 'INDEX.md')
  index_line = (f'- {timestamp} | {topic} | '
                f'PDB={len(found["pdb"])} | SDF={len(moved)} | '
                f'CSV={len(found["csv"])} | {os.path.basename(folder)}')
  with open(index_path, 'a') as f:
    f.write(index_line + '\n')

  if print_flag:
    print(f'[memory] saved to {folder}')

  return (f'Session memory saved to {folder} '
          f'({len(moved)} pose SDF(s) moved, '
          f'{len(found["pdb"])} PDB file(s) named, '
          f'{len(found["csv"])} CSV file(s) named).')


def list_sessions(vault_dir='../vault'):
  '''
    Returns a formatted string listing every saved session, read from the
    vault INDEX.md (cheap; no directory scan). Empty string if none.
  '''
  index_path = os.path.join(vault_dir, 'INDEX.md')
  if not os.path.exists(index_path):
    return f'(no saved sessions in {vault_dir} yet)'
  with open(index_path, 'r') as f:
    lines = [ln.rstrip('\n') for ln in f if ln.strip()]
  if not lines:
    return f'(no saved sessions in {vault_dir} yet)'
  out = ['Saved sessions (most recent last):', '']
  out.extend(lines)
  out.append('')
  out.append('Use `recall <date>` to load one (e.g. `recall 2026-07-19`), '
             'or `recall last` for the most recent.')
  return '\n'.join(out)


def recall_session(vault_dir='../vault', which='last'):
  '''
    Reads a saved session summary back into memory so it can be fed to the
    model as context.

      Args:
        vault_dir: the vault directory
        which: 'last' for the most recent session, or a date prefix like
               '2026-07-19' (matches any folder starting with that date)
      Returns:
        the summary text, prefixed with a header identifying the session.
        None if no matching session was found.
  '''
  if not os.path.isdir(vault_dir):
    return None
  folders = [d for d in os.listdir(vault_dir)
            if os.path.isdir(os.path.join(vault_dir, d))]
  folders.sort()
  if not folders:
    return None

  if which == 'last':
    target = folders[-1]
  else:
    # match by date prefix (the YYYY-MM-DD part of the folder name)
    matches = [d for d in folders if d.startswith(which)]
    if not matches:
      return None
    target = matches[-1]

  summary_path = os.path.join(vault_dir, target, 'session_summary.md')
  if not os.path.exists(summary_path):
    return None
  with open(summary_path, 'r') as f:
    body = f.read()
  header = f'[Recalled session {target}]\n\n'
  return header + body