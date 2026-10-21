"""pH-aware protonation for ligands (SMILES) and receptors (PDB).

Everything in the surrounding project family currently adds hydrogens by
naive rules -- RDKit's AddHs() (which honours whatever charges the input
SMILES happened to carry) or `obabel -h` (valence-based, not pKa-based).
Neither asks what charge state a molecule or a residue actually has at
physiological pH. This module answers that question in one place:

    normalize_smiles(smiles, ph)   ligand side, via Dimorphite-DL
    protonate_receptor(pdb, ph)    receptor side, via pdb2pqr + PROPKA

Not a package -- there is nothing to pip install. Copy cafprot.py next to
the code that needs it, or put this repo on sys.path.

Both functions fall back to returning their input unchanged, with a warning,
rather than raising: one bad molecule should never abort a batch.

PYTHON 3.14: PROPKA 3.5.1 (the newest release) crashes there -- it reads
instance __annotations__ at runtime, which PEP 649 made lazy. pdb2pqr itself
still runs, and will even accept --with-ph while doing no pKa prediction at
all, so a 3.14 run can look successful while ignoring the pH entirely. A
shim (see _PROPKA_SHIM) restores the annotations inside the pdb2pqr
subprocess, so pH-aware receptor protonation works on 3.14 too, producing
output byte-identical to 3.11's native PROPKA.
"""
import os
import shutil
import subprocess
import sys
import warnings

PH_LIGAND = 7.4      # blood plasma
PH_RECEPTOR = 7.0    # pdb2pqr's own default, closer to intracellular


def normalize_smiles(smiles, ph=PH_LIGAND, canonicalize_tautomer=False,
                     warn_if_ambiguous=True):
    """The dominant protonation state of `smiles` at `ph`, as canonical SMILES.

    Warns when the answer is ambiguous -- i.e. when Dimorphite-DL considers
    more than one state plausible at this pH, which is where its class-mean
    fallback can pick the wrong one (phenol and azole N-H are the usual
    offenders; see protonation_states() and the README). Pass
    warn_if_ambiguous=False to silence that for bulk runs.

    Returns `smiles` unchanged (with a warning) if Dimorphite-DL is missing or
    cannot handle the input. Set canonicalize_tautomer=True to additionally
    collapse tautomers via RDKit -- a related but distinct problem, off by
    default because it changes structures protonation alone would leave alone.
    """
    from rdkit import Chem

    try:
        from dimorphite_dl import protonate_smiles
        # First at the library's own precision: if it returns a single state,
        # the group's whole pKa range sits on one side of this pH and the
        # answer is unambiguous. Measured on a panel of 9 molecules with known
        # pKa, every single-state case was correct and every error was a
        # multi-state one -- so the count is a usable confidence signal.
        plausible = protonate_smiles(smiles, ph_min=ph, ph_max=ph, precision=1.0)
        if len(plausible) == 1:
            states = plausible
        else:
            # Ambiguous: the range straddles this pH. Fall back to the class
            # mean for a deterministic answer, but say so -- the correct form
            # is in `plausible`, and only chemistry knowledge picks it.
            states = protonate_smiles(smiles, ph_min=ph, ph_max=ph, precision=0.0)
            if warn_if_ambiguous:
                warnings.warn(
                    f"protonation of {smiles!r} at pH {ph} is ambiguous: {list(plausible)}. "
                    f"Returning the class-mean choice; see protonation_states()."
                )
    except Exception as exc:
        warnings.warn(f"protonation failed for {smiles!r} ({exc}); using input unchanged")
        return smiles

    if not states:
        warnings.warn(f"no protonation state returned for {smiles!r}; using input unchanged")
        return smiles

    mol = Chem.MolFromSmiles(states[0])
    if mol is None:
        warnings.warn(f"unparseable protonated form {states[0]!r} for {smiles!r}; using input unchanged")
        return smiles

    if canonicalize_tautomer:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)

    return Chem.MolToSmiles(mol)


# PROPKA 3.5.1 (the newest release there is) reads `self.__annotations__` to
# type its parameter file. PEP 649 made annotations lazy in 3.14, so instances
# no longer expose that attribute and PROPKA dies on import of its parameters.
# The annotations still exist -- annotationlib can materialise them -- but
# reattaching them to the CLASS doesn't help, because 3.14 redirects that
# assignment into __annotations_cache__ where instance lookup won't find it.
# Putting the dict in each instance's __dict__ does work. Applied inside the
# pdb2pqr subprocess, since that is where PROPKA actually runs.
_PROPKA_SHIM = """
import annotationlib, propka.parameters as pp
_ann = dict(annotationlib.get_annotations(pp.Parameters))
_orig_init = pp.Parameters.__init__
def _init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self.__dict__['__annotations__'] = _ann
pp.Parameters.__init__ = _init
from pdb2pqr.main import main
main()
"""


def _propka_native():
    """True if PROPKA works as shipped, with no shim (Python <= 3.13)."""
    try:
        import propka.parameters
        propka.parameters.Parameters().__annotations__
        return True
    except Exception:
        return False


def protonation_states(smiles, ph=PH_LIGAND, precision=1.0):
    """Every protonation state Dimorphite-DL considers plausible at `ph`.

    normalize_smiles() asks for one answer (precision=0, i.e. each moiety's
    pKa range collapsed to its mean). This exposes what that discards: the
    library associates each ionizable group with a range [mu - n*sigma,
    mu + n*sigma] and returns BOTH forms when that range straddles the pH,
    which is its way of saying "either is plausible here".

    Worth calling for any group in a chemically broad class -- the Phenol
    SMARTS also matches N-OH and O-OH, and the aromatic N-H class spans
    tetrazole to pyrrole -- where the class mean is a poor stand-in for the
    specific molecule. Returns [smiles] unchanged if Dimorphite-DL fails.
    """
    try:
        from dimorphite_dl import protonate_smiles
        states = protonate_smiles(smiles, ph_min=ph, ph_max=ph, precision=precision)
    except Exception as exc:
        warnings.warn(f"protonation failed for {smiles!r} ({exc}); using input unchanged")
        return [smiles]
    return list(states) if states else [smiles]


def propka_available():
    """Whether a requested pH will actually mean anything here -- i.e. whether
    PROPKA can run, natively or via the 3.14 shim.
    """
    if _propka_native():
        return True
    try:
        import annotationlib  # 3.14+, the interpreter the shim is for
        import propka.parameters  # noqa: F401
        return True
    except Exception:
        return False


def protonate_receptor(pdb_path, ph=PH_RECEPTOR, overwrite=False):
    """A properly protonated copy of `pdb_path`, written as
    `<stem>_protonated.pdb` next to the input, whose path is returned.

    Cached: an existing output is reused unless overwrite=True. Returns the
    input path unchanged (with a warning) if pdb2pqr is missing or fails.

    Without PROPKA the pH CANNOT be applied; rather than return a file that
    silently ignores it, this warns explicitly and still returns the
    hydrogen-added structure, which remains better than obabel's valence-only
    guess.
    """
    out_pdb = f"{os.path.splitext(pdb_path)[0]}_protonated.pdb"
    if os.path.exists(out_pdb) and not overwrite:
        return out_pdb

    # Console scripts live beside the interpreter, which matters when a venv's
    # python is invoked directly (venv/bin/python ...) without activation --
    # then the venv's bin/ is not on PATH and a plain which() finds nothing.
    if _propka_native():
        # Console scripts live beside the interpreter, which matters when a
        # venv's python is invoked directly (venv/bin/python ...) without
        # activation -- then the venv's bin/ is not on PATH and a plain
        # which() finds nothing.
        launcher = None
        for path in (None, os.path.dirname(sys.executable)):
            launcher = shutil.which("pdb2pqr", path=path) or shutil.which("pdb2pqr30", path=path)
            if launcher:
                break
        if launcher is None:
            warnings.warn("pdb2pqr not found on PATH; using the unprotonated input")
            return pdb_path
        cmd = [launcher]
    else:
        # Same pdb2pqr entry point, reached through the PEP 649 shim above.
        cmd = [sys.executable, "-c", _PROPKA_SHIM]

    cmd += ["--ff=AMBER", f"--pdb-output={out_pdb}"]
    if propka_available():
        cmd += [f"--with-ph={ph}", "--titration-state-method=propka"]
    else:
        warnings.warn(
            f"PROPKA cannot run here, so pH {ph} was NOT applied to {pdb_path}; hydrogens "
            f"were added without pKa prediction."
        )
    cmd += [pdb_path, f"{os.path.splitext(pdb_path)[0]}_protonated.pqr"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(out_pdb):
        warnings.warn(
            f"pdb2pqr failed on {pdb_path} ({result.stderr.strip().splitlines()[-1:] or 'no stderr'}); "
            f"using the unprotonated input"
        )
        return pdb_path
    return out_pdb
