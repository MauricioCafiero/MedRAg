"""
Unit tests for smiles_node() in modrag_molecule_functions.

The bug being targeted: smiles_node is invoked by the LLM agent as
`smiles_node(**tc.function.arguments)`, and the model frequently passes a
bare string ("aspirin") instead of a list (["aspirin"]). The node then
iterates the string character-by-character, querying PubChem for 'a', 's',
'p', ... -- single letters that accidentally resolve to elements ('s'->[S],
'p'->[P], 'i'->II, 'n'->N#N). So a request for aspirin returns garbage like
['[S]','[P]','II','II','N#N'] instead of acetylsalicylic acid.

These tests are real unit tests: pubchempy is mocked so they run offline and
deterministically (tool_tests.py Test 2 hits the live PubChem API and only
uses list inputs, so it never exercises this failure mode).

Run:
    modrag-env/bin/python -m unittest smiles_node_test -v
"""

import unittest
from unittest import mock

import modrag_molecule_functions as mmf
from modrag_molecule_functions import smiles_node


class _FakeCompound:
    """Stand-in for a pubchempy Compound exposing only `.smiles`."""
    def __init__(self, smiles):
        self.smiles = smiles


def _patcher(mapping, raises=()):
    """Patch pcp.get_compounds to resolve names per `mapping`.

    `mapping`: name -> SMILES string (success cases).
    `raises`: names that should raise (simulating a network/lookup error).
    Any other query returns [] (not found -> res[0] raises IndexError).
    """
    raises = set(raises)

    def fake_get_compounds(query, namespace, *args, **kwargs):
        if query in raises:
            raise RuntimeError("pubchem lookup failed")
        if query in mapping:
            return [_FakeCompound(mapping[query])]
        return []

    return mock.patch.object(mmf.pcp, "get_compounds", side_effect=fake_get_compounds)


class SmilesNodeTest(unittest.TestCase):

    # --- baseline: a proper list input works ----------------------------
    def test_list_input_resolves(self):
        with _patcher({"aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O", "ethanol": "CCO"}):
            smiles_list, smiles_string, third = smiles_node(["aspirin", "ethanol"])

        self.assertEqual(smiles_list, ["CC(=O)OC1=CC=CC=C1C(=O)O", "CCO"])
        self.assertIn("aspirin: The SMILES string for the molecule is: "
                      "CC(=O)OC1=CC=CC=C1C(=O)O", smiles_string)
        self.assertIsNone(third)

    # --- THE BUG: a bare string must be treated as one name -------------
    def test_bare_string_treated_as_single_name(self):
        with _patcher({"aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O"}):
            smiles_list, smiles_string, _ = smiles_node("aspirin")

        # Currently this iterates the characters of "aspirin" and returns a
        # list of element symbols ([S], [P], II, ...). After the fix it must
        # query PubChem once for "aspirin".
        self.assertEqual(smiles_list, ["CC(=O)OC1=CC=CC=C1C(=O)O"],
                         "a bare string must be treated as a single name, "
                         "not iterated character-by-character")
        self.assertIn("aspirin: The SMILES string for the molecule is: "
                      "CC(=O)OC1=CC=CC=C1C(=O)O", smiles_string)

    # --- a comma-separated string should split into several names ------
    def test_comma_separated_string_splits(self):
        with _patcher({"aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O", "ethanol": "CCO"}):
            smiles_list, smiles_string, _ = smiles_node("aspirin, ethanol")

        self.assertEqual(smiles_list,
                         ["CC(=O)OC1=CC=CC=C1C(=O)O", "CCO"])
        self.assertIn("aspirin: The SMILES string for the molecule is: "
                      "CC(=O)OC1=CC=CC=C1C(=O)O", smiles_string)
        self.assertIn("ethanol: The SMILES string for the molecule is: CCO",
                      smiles_string)

    # --- whitespace-only/whitespace-separated string ---------------------
    def test_whitespace_separated_string_splits(self):
        with _patcher({"aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O", "ethanol": "CCO"}):
            smiles_list, _, _ = smiles_node("  aspirin   ethanol  ")

        self.assertEqual(smiles_list,
                         ["CC(=O)OC1=CC=CC=C1C(=O)O", "CCO"])

    # --- list entries with surrounding whitespace are cleaned ----------
    def test_list_entries_are_stripped(self):
        with _patcher({"aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O"}):
            smiles_list, smiles_string, _ = smiles_node(["  aspirin  "])

        self.assertEqual(smiles_list, ["CC(=O)OC1=CC=CC=C1C(=O)O"])
        self.assertIn("aspirin: The SMILES string for the molecule is: "
                      "CC(=O)OC1=CC=CC=C1C(=O)O", smiles_string)

    # --- empty input ----------------------------------------------------
    def test_empty_string_input(self):
        with _patcher({}):
            smiles_list, smiles_string, _ = smiles_node("")

        self.assertEqual(smiles_list, [])
        self.assertEqual(smiles_string, "")

    def test_empty_list_input(self):
        with _patcher({}):
            smiles_list, smiles_string, _ = smiles_node([])

        self.assertEqual(smiles_list, [])
        self.assertEqual(smiles_string, "")

    # --- a name that genuinely fails still reports Fail -----------------
    def test_unknown_name_reports_fail(self):
        with _patcher({"aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O"}):
            smiles_list, smiles_string, _ = smiles_node(
                ["aspirin", "notarealdrug_xyz"])

        self.assertIn("aspirin: The SMILES string for the molecule is: "
                      "CC(=O)OC1=CC=CC=C1C(=O)O", smiles_string)
        self.assertIn("notarealdrug_xyz: Fail", smiles_string)


if __name__ == "__main__":
    unittest.main(verbosity=2)