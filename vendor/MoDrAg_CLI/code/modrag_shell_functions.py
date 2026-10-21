"""
MoDrAg shell node: a tiny, cross-platform, sandboxed shell for the agent to
explore the project's output files (bioactive CSVs in ../scratch, receptor
PDBs in ../pdb_files, saved sessions in ../vault, images, ...).

Commands are NOT passed to a real shell (bash/zsh/PowerShell) - they are
parsed here and implemented with Python's os/fnmatch/glob. That gives the
same syntax and behavior on every OS, with no shell-injection surface, and
no destructive commands (there is no rm/mv/cp). Everything is restricted to
the project folder (the parent of code/): paths outside it are refused.
The working directory persists across calls within a session via `cd`.
"""

import os
import time
import fnmatch
import shlex
from glob import glob

# Module-level print flag - set from modrag.py
print_flag = True

# Sandbox root: the repo root (parent of code/). All paths must stay inside.
SANDBOX_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Persistent working directory across shell_node calls in a session. Starts
# in code/ so relative paths ('../scratch', '../pdb_files') match the
# convention used by every other node and the README.
_cwd = os.path.dirname(os.path.abspath(__file__))

# Output caps, so a big directory or file cannot flood the chat context
MAX_LS_ENTRIES = 500
MAX_FIND_HITS = 300
MAX_CAT_BYTES = 20000
MAX_GREP_HITS = 100

SUPPORTED = ('pwd, cd, ls [-l -t -r -a], mkdir [-p], find [-name -type '
             '-maxdepth], cat, head [-n], tail [-n], wc, grep [-i -n]')


def _err(msg: str) -> str:
    return f'shell: {msg}'


def _resolve(raw: str):
    """Resolve a path relative to the current dir; return None if it would
    escape the project folder."""
    p = os.path.abspath(os.path.join(_cwd, os.path.expanduser(str(raw))))
    if not (p == SANDBOX_ROOT or p.startswith(SANDBOX_ROOT + os.sep)):
        return None
    return p


def _expand_args(args):
    """Expand wildcard arguments (e.g. ../scratch/*.csv) against the current
    dir. Args without wildcards pass through unchanged."""
    out = []
    for a in args:
        if '*' in a or '?' in a:
            hits = sorted(glob(os.path.join(_cwd, a))) or [a]
            out.extend(hits)
        else:
            out.append(a)
    return out


def _split_flags(args):
    """Split combined single-letter POSIX flags ('-lt' -> {'l','t'}) from the
    positional arguments. Long options (used by find) are handled per-command."""
    flags, rest = set(), []
    for a in args:
        a = str(a)
        if a.startswith('-') and len(a) > 1 and not a.startswith('--'):
            flags.update(a[1:])
        else:
            rest.append(a)
    return flags, rest


def _cmd_pwd(args):
    return _cwd


def _cmd_cd(args):
    global _cwd
    target = args[0] if args else SANDBOX_ROOT
    p = _resolve(target)
    if p is None:
        return _err(f'access outside the project folder denied: {target}')
    if not os.path.isdir(p):
        return _err(f'{target}: Not a directory')
    _cwd = p
    return _cwd


def _cmd_ls(args):
    flags, paths = _split_flags(args)
    paths = _expand_args(paths) or ['.']
    out = []
    for raw in paths:
        p = _resolve(raw)
        if p is None:
            out.append(_err(f'access outside the project folder denied: {raw}'))
            continue
        if not os.path.exists(p):
            out.append(f'ls: {raw}: No such file or directory')
            continue
        entries = []
        if os.path.isdir(p):
            names = os.listdir(p)
            if 'a' not in flags:
                names = [n for n in names if not n.startswith('.')]
            if len(names) > MAX_LS_ENTRIES:
                out.append(f'ls: showing first {MAX_LS_ENTRIES} of '
                           f'{len(names)} entries in {raw}')
                names = names[:MAX_LS_ENTRIES]
            for n in names:
                fp = os.path.join(p, n)
                try:
                    st = os.stat(fp)
                    entries.append((n, os.path.isdir(fp), st.st_size, st.st_mtime))
                except OSError:
                    entries.append((n, os.path.isdir(fp), 0, 0))
        else:
            st = os.stat(p)
            entries.append((os.path.basename(p), False, st.st_size, st.st_mtime))
        if 't' in flags:
            entries.sort(key=lambda e: e[3], reverse=('r' not in flags))
        else:
            entries.sort(key=lambda e: e[0].lower(), reverse=('r' in flags))
        if len(paths) > 1:
            out.append(f'{raw}:')
        for n, is_dir, size, mtime in entries:
            if 'l' in flags:
                kind = 'd' if is_dir else '-'
                ts = time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))
                out.append(f'{kind} {size:>10}  {ts}  {n}')
            else:
                out.append(n)
    return '\n'.join(out)


def _cmd_mkdir(args):
    flags, paths = _split_flags(args)
    out = []
    for raw in paths:
        p = _resolve(raw)
        if p is None:
            out.append(_err(f'access outside the project folder denied: {raw}'))
            continue
        try:
            os.makedirs(p, exist_ok=('p' in flags))
            out.append(f'created directory: {raw}')
        except OSError as exc:
            out.append(_err(f'mkdir {raw}: {exc}'))
    return '\n'.join(out)


def _cmd_find(args):
    paths, name, ftype, maxdepth = [], None, None, None
    i = 0
    while i < len(args):
        a = str(args[i])
        if a == '-name' and i + 1 < len(args):
            name = str(args[i + 1]); i += 2
        elif a == '-type' and i + 1 < len(args):
            ftype = str(args[i + 1]); i += 2
        elif a == '-maxdepth' and i + 1 < len(args):
            try:
                maxdepth = int(args[i + 1])
            except ValueError:
                return _err(f'find: bad -maxdepth value: {args[i + 1]}')
            i += 2
        elif a.startswith('-'):
            return _err(f'find: unsupported option {a} '
                        f'(supported: -name, -type f|d, -maxdepth)')
        else:
            paths.append(a); i += 1

    if paths:
        paths = _expand_args(paths)
    else:
        paths = ['.']

    hits, out = 0, []
    for raw in paths:
        top = _resolve(raw)
        if top is None:
            out.append(_err(f'access outside the project folder denied: {raw}'))
            continue
        if not os.path.isdir(top):
            out.append(f'find: {raw}: No such directory')
            continue
        for dirpath, dirnames, filenames in os.walk(top):
            rel = os.path.relpath(dirpath, top)
            level = 0 if rel == '.' else rel.count(os.sep) + 1
            if maxdepth is not None and level >= maxdepth:
                dirnames[:] = []
            if maxdepth is not None and level >= maxdepth:
                continue
            for n in dirnames + filenames:
                is_dir = n in dirnames
                if ftype == 'f' and is_dir:
                    continue
                if ftype == 'd' and not is_dir:
                    continue
                if name is not None and not fnmatch.fnmatch(n, name):
                    continue
                relname = os.path.normpath(os.path.join(raw, rel, n))
                out.append(relname)
                hits += 1
                if hits >= MAX_FIND_HITS:
                    out.append(f'find: stopping at {MAX_FIND_HITS} results')
                    return '\n'.join(out)
    if not out:
        out.append('find: no matches')
    return '\n'.join(out)


def _readable_path(raw, out):
    """Shared check+resolve for file-reading commands; returns the path or
    None (after appending an error line to `out`)."""
    p = _resolve(raw)
    if p is None:
        out.append(_err(f'access outside the project folder denied: {raw}'))
        return None
    if not os.path.isfile(p):
        out.append(f'{raw}: No such file')
        return None
    return p


def _read_text(p):
    """Read a text file capped at MAX_CAT_BYTES; returns (text, truncated,
    was_binary) or None if unreadable."""
    try:
        with open(p, 'rb') as fh:
            blob = fh.read(MAX_CAT_BYTES + 1)
    except OSError:
        return None
    if b'\x00' in blob:
        return None
    truncated = len(blob) > MAX_CAT_BYTES
    return blob[:MAX_CAT_BYTES].decode('utf-8', errors='replace'), truncated, False


def _cmd_cat(args):
    out = []
    for raw in _expand_args(args):
        p = _readable_path(raw, out)
        if p is None:
            continue
        got = _read_text(p)
        if got is None:
            out.append(f'{raw}: not a text file (skipped)')
            continue
        text, truncated, _ = got
        out.append(text)
        if truncated:
            out.append(f'(truncated to {MAX_CAT_BYTES} bytes)')
    return '\n'.join(out)


def _cmd_head(args):
    return _bounded_lines(args, head=True)


def _cmd_tail(args):
    return _bounded_lines(args, head=False)


def _bounded_lines(args, head):
    n = 20 if head else 10
    files = []
    i = 0
    while i < len(args):
        a = str(args[i])
        if a in ('-n', '--lines') and i + 1 < len(args):
            try:
                n = int(args[i + 1])
            except ValueError:
                return _err(f'bad line count: {args[i + 1]}')
            i += 2
        elif a.startswith('-') and len(a) > 1:
            return _err(f'unsupported option {a} (use -n N)')
        else:
            files.append(a); i += 1

    out = []
    for raw in _expand_args(files):
        p = _readable_path(raw, out)
        if p is None:
            continue
        got = _read_text(p)
        if got is None:
            out.append(f'{raw}: not a text file (skipped)')
            continue
        text, _, _ = got
        lines = text.splitlines()
        take = lines[:n] if head else lines[-n:]
        if len(files) > 1:
            out.append(f'==> {raw} <==')
        out.extend(take)
    return '\n'.join(out)


def _cmd_wc(args):
    flags, files = _split_flags(args)
    out = []
    for raw in _expand_args(files):
        p = _readable_path(raw, out)
        if p is None:
            continue
        try:
            with open(p, 'rb') as fh:
                blob = fh.read()
        except OSError as exc:
            out.append(_err(f'wc {raw}: {exc}'))
            continue
        lines = blob.count(b'\n')
        words = len(blob.split())
        if 'l' in flags:
            out.append(f'{lines:>8} {raw}')
        elif 'w' in flags:
            out.append(f'{words:>8} {raw}')
        elif 'c' in flags:
            out.append(f'{len(blob):>8} {raw}')
        else:
            out.append(f'{lines:>8} {words:>8} {len(blob):>8} {raw}')
    return '\n'.join(out)


def _cmd_grep(args):
    flags_set, rest = set(), []
    for a in args:
        a = str(a)
        if a.startswith('-') and len(a) > 1 and not a.startswith('--'):
            flags_set.update(a[1:])
        else:
            rest.append(a)
    if len(rest) < 1:
        return _err('usage: grep [-i] [-n] PATTERN file...')
    pattern = rest[0]
    files = _expand_args(rest[1:])
    if not files:
        return _err('usage: grep [-i] [-n] PATTERN file...')

    needle = pattern.lower() if 'i' in flags_set else pattern
    out = []
    hits = 0
    for raw in files:
        p = _readable_path(raw, out)
        if p is None:
            continue
        got = _read_text(p)
        if got is None:
            out.append(f'{raw}: not a text file (skipped)')
            continue
        text, _, _ = got
        for lineno, line in enumerate(text.splitlines(), start=1):
            hay = line.lower() if 'i' in flags_set else line
            if needle in hay:
                prefix = f'{raw}:{lineno}: ' if 'n' in flags_set else f'{raw}: '
                out.append(prefix + line.strip())
                hits += 1
                if hits >= MAX_GREP_HITS:
                    out.append(f'grep: stopping at {MAX_GREP_HITS} matches')
                    return '\n'.join(out)
    if not out:
        out.append('grep: no matches')
    return '\n'.join(out)


COMMANDS = {
    'pwd': _cmd_pwd, 'cd': _cmd_cd, 'ls': _cmd_ls, 'mkdir': _cmd_mkdir,
    'find': _cmd_find, 'cat': _cmd_cat, 'head': _cmd_head, 'tail': _cmd_tail,
    'wc': _cmd_wc, 'grep': _cmd_grep,
}


def shell_node(command: str) -> str:
    """Runs a simple, safe, cross-platform shell command to explore and manage
    the project's output files: bioactive molecule CSVs in ../scratch,
    receptor PDB files in ../pdb_files, saved poses and session summaries in
    ../vault, generated images, and so on. Commands use the same POSIX-style
    syntax on every operating system (no real shell is executed, and it
    cannot access anything outside the project folder).

      Supported commands:
        pwd                          show the current directory (code/ at start)
        cd <dir>                     change directory (persists between calls; .. works)
        ls [-l] [-t] [-r] [-a] PATH  list directory contents; -l long form with sizes
                                     and dates, -t sort by modification time,
                                     -r reverse, -a include hidden; e.g. ls -lt ../scratch
        mkdir [-p] DIR [DIR...]      create directory(-ies); -p creates parents
        find [PATH] [-name PATTERN] [-type f|d] [-maxdepth N]
                                     recursive file search, e.g.
                                     find ../scratch -name "*.csv"
        cat FILE...                  print file contents (text only, size-capped)
        head [-n N] FILE...          first lines of file(s)
        tail [-n N] FILE...          last lines of file(s)
        wc [-l|-w|-c] FILE...        count lines, words, or bytes
        grep [-i] [-n] PATTERN FILE...   search text in files, e.g.
                                     grep -i IC50 ../scratch/CHEMBL213_bioactives.csv

      Use this tool to discover data files (e.g. which bioactives CSVs were
      already downloaded, which receptor PDBs are cached, which docking pose
      SDFs exist) and to inspect their contents before running other tools.
      Wildcards like *.csv are expanded. There are intentionally no rm, mv or
      cp commands.

      Args:
        command: the command string to run, e.g. "ls -lt ../scratch" or
                 "find ../pdb_files -name \"*.csv\""

      Returns:
        the command output as a string (errors are in-line, prefixed "shell:")
    """
    print("shell tool")
    print('===================================================')

    # Coerce common argument shapes into one command string. The LLM calls
    # this node as shell_node(**tc.function.arguments) and may pass the
    # command as a list of words/tokens, or None. Mirrors the coercion in
    # smiles_node / gpt_node.
    if command is None:
        return _err('no command provided (supported: ' + SUPPORTED + ')')
    if isinstance(command, (list, tuple)):
        command = ' '.join(str(c) for c in command)
    command = str(command).strip()
    # Unwrap the whole command only if BOTH ends carry the same quote char
    # (the LLM sometimes wraps the entire command in quotes). Do not strip
    # bare quotes at one end: a command ending in a quoted argument, e.g.
    # 'find ../scratch -name "*.csv"', would lose its closing quote and fail
    # to parse.
    if len(command) >= 2 and command[0] == command[-1] and command[0] in ('"', "'"):
        command = command[1:-1]
    if not command:
        return _err('no command provided (supported: ' + SUPPORTED + ')')
    if any(c in command for c in ';' + '&|><`$\n'):
        return _err('chaining/redirection characters are not supported - '
                    'run one command per call')

    if print_flag:
        print(f'[{os.path.relpath(_cwd, SANDBOX_ROOT) or "."}] {command}')

    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return _err(f'could not parse command: {exc}')
    if not tokens:
        return _err('no command provided (supported: ' + SUPPORTED + ')')

    cmd, args = tokens[0], tokens[1:]
    handler = COMMANDS.get(cmd)
    if handler is None:
        return _err(f'{cmd}: command not supported. Supported: {SUPPORTED}')

    try:
        return handler(args)
    except Exception as exc:
        return _err(f'{cmd} failed: {exc}')