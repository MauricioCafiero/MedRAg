"""Local API tests for cafprot. No pytest needed: `python tests/test_cafprot.py`
(pytest also collects them if you prefer).

Deliberately does not touch any other repo -- the only input is the small
4-residue fragment in tests/mini_peptide.pdb.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cafprot

HERE = os.path.dirname(os.path.abspath(__file__))
MINI_PDB = os.path.join(HERE, "mini_peptide.pdb")


def test_acid_is_deprotonated():
    """Aspirin's carboxylic acid (pKa ~3.5) is ionised at 7.4."""
    assert cafprot.normalize_smiles("CC(=O)Oc1ccccc1C(=O)O") == "CC(=O)Oc1ccccc1C(=O)[O-]"


def test_base_is_protonated():
    """An aliphatic amine (pKa ~10.7) is protonated at 7.4."""
    assert cafprot.normalize_smiles("CCN") == "CC[NH3+]"


def test_ph_changes_the_answer():
    """The pH argument actually does something: the same acid is neutral at 1."""
    acid = "CC(=O)Oc1ccccc1C(=O)O"
    assert cafprot.normalize_smiles(acid, ph=1.0) != cafprot.normalize_smiles(acid, ph=7.4)


def test_already_correct_input_is_stable():
    """Normalising twice changes nothing the second time."""
    once = cafprot.normalize_smiles("CCN")
    assert cafprot.normalize_smiles(once) == once


def test_bad_smiles_falls_back_to_input():
    """A garbage input warns and comes back untouched, never raises."""
    bad = "this is not a smiles"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert cafprot.normalize_smiles(bad) == bad
        assert caught, "expected a warning on unparseable input"


def test_tautomer_flag_is_off_by_default():
    """The tautomer option is opt-in and only changes things when asked."""
    enol = "C=C(O)C"
    assert cafprot.normalize_smiles(enol, canonicalize_tautomer=True) != enol


def test_phenol_single_state_follows_the_class_mean():
    """Regression pin, NOT an endorsement: phenol (pKa ~10) comes back ionised
    at 7.4 because normalize_smiles asks for precision=0, collapsing the
    Phenol class to its mean of 7.07 -- that SMARTS also covers N-OH, O-OH and
    electron-poor phenols. Pinned so a version bump that shifts it is noticed.
    """
    assert cafprot.normalize_smiles("Oc1ccccc1") == "[O-]c1ccccc1"


def test_ambiguous_input_warns_and_unambiguous_does_not():
    """A molecule whose pKa range straddles the pH is flagged; a clear-cut one
    is not. Phenol is ambiguous at 7.4, acetic acid is not.
    """
    for smiles, should_warn in [("Oc1ccccc1", True), ("CC(=O)O", False)]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cafprot.normalize_smiles(smiles)
            warned = any("ambiguous" in str(w.message) for w in caught)
        assert warned is should_warn, f"{smiles}: expected warn={should_warn}"

    # and it can be silenced for bulk runs
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cafprot.normalize_smiles("Oc1ccccc1", warn_if_ambiguous=False)
        assert not any("ambiguous" in str(w.message) for w in caught)


def test_protonation_states_exposes_the_discarded_uncertainty():
    """At the library's own default precision, phenol is reported as BOTH
    forms -- the ambiguity the single-answer API throws away.
    """
    states = cafprot.protonation_states("Oc1ccccc1", ph=7.4)
    assert len(states) > 1, f"expected the ambiguity to surface, got {states}"
    assert cafprot.normalize_smiles("Oc1ccccc1") in states
    # a well-defined group stays unambiguous even at default precision
    assert len(cafprot.protonation_states("CC(=O)Oc1ccccc1C(=O)O", ph=7.4)) == 1


def test_receptor_protonation_writes_and_caches():
    out = cafprot.protonate_receptor(MINI_PDB, overwrite=True)
    try:
        assert out.endswith("_protonated.pdb"), out
        assert os.path.exists(out)
        assert sum(1 for l in open(out) if l.startswith(("ATOM", "HETATM"))) > 0
        # hydrogens were actually added
        assert sum(1 for l in open(out) if l.startswith("ATOM") and l[76:78].strip() == "H") > 0
        # second call is cached: same path, not re-run
        before = os.path.getmtime(out)
        assert cafprot.protonate_receptor(MINI_PDB) == out
        assert os.path.getmtime(out) == before
    finally:
        # pdb2pqr also drops a .log beside its output
        for ext in ("_protonated.pdb", "_protonated.pqr", "_protonated.log"):
            p = os.path.splitext(MINI_PDB)[0] + ext
            if os.path.exists(p):
                os.remove(p)


def test_missing_receptor_falls_back_to_input():
    """A nonexistent PDB warns and returns the input path, never raises."""
    missing = os.path.join(HERE, "does_not_exist.pdb")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert cafprot.protonate_receptor(missing) == missing
            assert caught, "expected a warning when pdb2pqr fails"
    finally:
        # pdb2pqr writes its log even for an input it cannot read
        stray = os.path.splitext(missing)[0] + "_protonated.log"
        if os.path.exists(stray):
            os.remove(stray)


def test_propka_availability_is_reported():
    """propka_available() must agree with whether a pH can actually be applied."""
    available = cafprot.propka_available()
    assert isinstance(available, bool)
    route = "native" if cafprot._propka_native() else "via 3.14 shim"
    print(f"    (PROPKA usable here: {available}, {route}; Python {sys.version.split()[0]})")


def test_receptor_ph_is_actually_applied():
    """The entire point of the receptor half: pH must change the answer.

    This fails on any interpreter where PROPKA cannot run -- which is exactly
    what the PEP 649 shim in cafprot exists to prevent, and what a plain
    pdb2pqr call hides by exiting 0 while ignoring --with-ph.
    """
    assert cafprot.propka_available(), "PROPKA cannot run here, so pH would be silently ignored"
    import shutil
    import tempfile

    work_dir = tempfile.mkdtemp()
    try:
        hydrogens = {}
        for ph in (1.0, 13.0):
            work = os.path.join(work_dir, f"ph{ph}.pdb")
            shutil.copy(MINI_PDB, work)
            out = cafprot.protonate_receptor(work, ph=ph, overwrite=True)
            hydrogens[ph] = sum(1 for l in open(out)
                                if l.startswith("ATOM") and l[76:78].strip() == "H")
        assert hydrogens[1.0] != hydrogens[13.0], (
            f"pH had no effect ({hydrogens}) -- the carboxylates should titrate")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
