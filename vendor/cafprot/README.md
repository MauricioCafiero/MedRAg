# cafprot

pH-aware protonation for ligands (SMILES) and receptors (PDB), in one place.

The usual ways of adding hydrogens don't consider pH at all: RDKit's `AddHs()`
honours whatever charges the input SMILES happened to carry, and `obabel -h`
works from valence rather than pKa. `cafprot` answers the question they skip —
what charge state does this molecule, or this residue, actually have at a given
pH.

**Not a package.** There is nothing to `pip install`, no `pyproject.toml`, no
version to pin. Copy `cafprot.py` next to the code that needs it, or put this
repo on `sys.path`.

**Not integrated with anything.** No other repo imports it.

## API

```python
import cafprot

cafprot.normalize_smiles("CC(=O)Oc1ccccc1C(=O)O")   # -> 'CC(=O)Oc1ccccc1C(=O)[O-]'
cafprot.normalize_smiles("CCN")                      # -> 'CC[NH3+]'
cafprot.protonation_states("Oc1ccccc1")              # -> ['Oc1ccccc1', '[O-]c1ccccc1']
cafprot.protonate_receptor("receptor.pdb")           # -> 'receptor_protonated.pdb'
cafprot.propka_available()                           # -> True / False
```

| Function | pH default | Returns |
|---|---|---|
| `normalize_smiles(smiles, ph, canonicalize_tautomer=False, warn_if_ambiguous=True)` | `PH_LIGAND` = **7.4** | one state, as canonical SMILES |
| `protonation_states(smiles, ph, precision=1.0)` | `PH_LIGAND` = **7.4** | every plausible state |
| `protonate_receptor(pdb_path, ph, overwrite=False)` | `PH_RECEPTOR` = **7.0** | path to a protonated PDB |

The pH defaults are constants at the top of `cafprot.py`, so the convention is
set in one place rather than per caller.

All three return their input unchanged, with a warning, rather than raising —
one bad molecule should not abort a batch. Receptor output is cached as
`<stem>_protonated.pdb` beside the input and reused unless `overwrite=True`.

Ligand protonation runs at about 1 ms per molecule.

## pH range

Any pH from 0 to 14. Outside that, pdb2pqr rejects the value, and
`protonate_receptor()` warns and returns the input unchanged.

| pH | aspirin (pKa 3.5) | ethylamine (pKa 10.7) | imidazole |
|---|---|---|---|
| 0–2 | neutral | `[NH3+]` | `[nH+]` |
| 4–6 | `[O-]` | `[NH3+]` | `[nH+]` → neutral |
| 7.4 | `[O-]` | `[NH3+]` | `[n-]` |
| 9–14 | `[O-]` | neutral | `[n-]` |

On the receptor side, the ASP/GLU test fragment carries 25 hydrogens at pH 0–2
and 24 from pH 4 up — the carboxylates titrating at pKa ≈ 4.

## Ambiguous protonation

Dimorphite-DL associates each ionizable group with a pKa *range*,
`[µ − nσ, µ + nσ]`, rather than a point value, and reports both the protonated
and deprotonated forms when that range straddles the requested pH.

`normalize_smiles` must return a single answer, so when a molecule is
ambiguous it picks the form implied by the group's mean pKa **and warns**:

```
UserWarning: protonation of 'Oc1ccccc1' at pH 7.4 is ambiguous:
['Oc1ccccc1', '[O-]c1ccccc1']. Returning the class-mean choice;
see protonation_states().
```

Take the warning seriously, because the class mean can disagree with the
specific molecule. The groups are broad: `Phenol` is `[c,n,o:1]-[O:2]-[H]`,
which also matches N–OH and O–OH and electron-poor phenols, so its mean is
7.07 (σ 3.28); the aromatic N–H class `[n:1]-[H]` spans tetrazole to pyrrole,
mean 7.17 (σ 2.95). Phenol (pKa ≈ 10) and imidazole's N–H (pKa ≈ 14.5) are
therefore reported ionised at pH 7.4.

On a 9-molecule panel with known pKa, **every unambiguous answer was correct,
every error was an ambiguous case, and the correct form was present in
`protonation_states()` in all 9.** So the practical rule is: no warning means
trust it; a warning means choose from `protonation_states()` yourself.

An ambiguity flag is not always a shortcoming — 4-nitrophenol has pKa 7.15, so
at pH 7.4 both forms genuinely are populated.

Pass `warn_if_ambiguous=False` to silence the warning in bulk runs.

Receptor protonation goes through PROPKA and is not affected by any of this.

## Setup

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Python support

Tested on 3.11.15 and 3.14.5; both halves work on both, including pH-aware
receptor protonation.

PROPKA 3.5.1 is incompatible with Python 3.14 on its own — it reads instance
`__annotations__`, which PEP 649 made lazy. `cafprot` applies a compatibility
shim inside the pdb2pqr subprocess, and the resulting structures are
byte-identical to those from a native PROPKA run on 3.11.

`propka_available()` reports whether a pH can actually be applied. If it ever
returns False, `protonate_receptor()` still adds hydrogens but warns that the
pH was not applied, rather than returning a file that looks pH-corrected and
isn't.

## Tests

```sh
python tests/test_cafprot.py     # or: pytest tests/
```

13 tests, no pytest required, no network, and no dependency on any other repo —
the only input is the 4-residue fragment in `tests/mini_peptide.pdb`. 13/13 on
both 3.11.15 and 3.14.5.

## License

[MIT](LICENSE)
