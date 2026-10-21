from modrag_molecule_functions import *
from modrag_property_functions import *

# Test 1: name_node() - Convert SMILES to molecule names
print("=" * 60)
print("TEST 1: name_node() - SMILES to Names")
print("=" * 60)
try:
    smiles_input = ['CCO', 'CCN', 'CCC']
    names, name_string, _ = name_node(smiles_input)
    print("Input SMILES:", smiles_input)
    print("Output Names:", names)
    print(name_string)
    print("✓ name_node test passed\n")
except Exception as e:
    print(f"✗ name_node test failed: {e}\n")

# Test 2: smiles_node() - Convert molecule names to SMILES
print("=" * 60)
print("TEST 2: smiles_node() - Names to SMILES")
print("=" * 60)
try:
    names_input = ['ethanol', 'ammonia']
    smiles_list, smiles_string, _ = smiles_node(names_input)
    print("Input Names:", names_input)
    print("Output SMILES:", smiles_list)
    print(smiles_string)
    print("✓ smiles_node test passed\n")
except Exception as e:
    print(f"✗ smiles_node test failed: {e}\n")

# Test 3: related_node() - Find similar molecules
print("=" * 60)
print("TEST 3: related_node() - Find Similar Molecules")
print("=" * 60)
try:
    smiles_input = ['CCO']
    similar_list, related_string, images = related_node(smiles_input)
    print("Input SMILES:", smiles_input)
    print("Number of similar molecules found:", len(similar_list[0]) if similar_list[0] else 0)
    print(related_string[:500], "...")  # Print first 500 chars
    print("✓ related_node test passed\n")
except Exception as e:
    print(f"✗ related_node test failed: {e}\n")

# Test 4: structure_node() - Generate 3D structures
print("=" * 60)
print("TEST 4: structure_node() - Generate 3D Structures")
print("=" * 60)
try:
    smiles_input = ['CCO', 'CC']
    structures, output_string, images = structure_node(smiles_input)
    print("Input SMILES:", smiles_input)
    print("Number of structures generated:", len(structures))
    print(output_string)
    if images and images[0]:
        images[0].save("../outputs/structure_output.png")
        print("Saved structure image as 'structure_output.png'")
    print("✓ structure_node test passed\n")
except Exception as e:
    print(f"✗ structure_node test failed: {e}\n")

# Test 5: canonical_node() - Convert to canonical SMILES
print("=" * 60)
print("TEST 5: canonical_node() - Canonical SMILES")
print("=" * 60)
try:
    smiles_input = 'C1=CC=CC=C1CCO' 
    canonical_smiles = canonical_node(smiles_input)
    print("Input SMILES:", smiles_input)
    print("Canonical SMILES:", canonical_smiles)
    print("✓ canonical_node test passed\n")
except Exception as e:
    print(f"✗ canonical_node test failed: {e}\n")

# Test 6: substitution_node() - Molecular substitution
print("=" * 60)
print("TEST 6: substitution_node() - Molecular Substitution")
print("=" * 60)
try:
    smiles_input = ['c1ccc(O)cc1']  # Benzene
    sub_smiles_list, sub_string, sub_images = substitution_node(smiles_input)
    print("Input SMILES:", smiles_input)
    print("Number of novel molecules generated:", len(sub_smiles_list[0]) if sub_smiles_list[0] else 0)
    print(sub_string[:500], "...")  # Print first 500 chars
    if sub_images and sub_images[0]:
        sub_images[0].save("../outputs/substitution_output.png")
        print("Saved substitution image as 'substitution_output.png'")
    print("✓ substitution_node test passed\n")
except Exception as e:
    print(f"✗ substitution_node test failed: {e}\n")

# Test 7: lipinski_node() - Lipinski properties
print("=" * 60)
print("TEST 7: lipinski_node() - Lipinski Properties")
print("=" * 60)
try:
    smiles_input = ['CCO', 'c1ccccc1']  # Ethanol, Benzene
    lipinski_list, lipinski_string, _ = lipinski_node(smiles_input)
    print("Input SMILES:", smiles_input)
    print("Number of molecules analyzed:", len(lipinski_list))
    print(lipinski_string)
    print("✓ lipinski_node test passed\n")
except Exception as e:
    print(f"✗ lipinski_node test failed: {e}\n")

# Test 8: pharmfeature_node() - Pharmacophore features
print("=" * 60)
print("TEST 8: pharmfeature_node() - Pharmacophore Features")
print("=" * 60)
try:
    test_smiles = ['CC(OC(=O)c1c[nH]c2ccccc12)C1CCCCN1C']  # Benzene as reference
    known_smiles = 'CN1CCOc2c(C(=O)NC3CC4CCC(C3)N4C)cc(Cl)cc21'  # test molecules
    feat_scores, feat_string, _ = pharmfeature_node(known_smiles, test_smiles)
    print("Reference SMILES:", known_smiles)
    print("Test SMILES:", test_smiles)
    print("Feature overlap scores:", feat_scores)
    print(feat_string[:800], "...")  # Print first 500 chars
    print("✓ pharmfeature_node test passed\n")
except Exception as e:
    print(f"✗ pharmfeature_node test failed: {e}\n")

# Test 9: modrag_memory - session save / recall (self-contained temp vault)
print("=" * 60)
print("TEST 9: modrag_memory - save / recall session memory")
print("=" * 60)
try:
    import time, os, shutil, tempfile
    from modrag_memory import (save_session, recall_session, list_sessions,
                               is_important_sdf)
    import modrag_memory as M

    tmp = tempfile.mkdtemp(prefix='vault_test_')
    vault = os.path.join(tmp, 'vault')
    pdbdir = os.path.join(tmp, 'pdb_files')
    scratch = os.path.join(tmp, 'scratch')
    os.makedirs(pdbdir); os.makedirs(scratch)
    open(os.path.join(pdbdir, 'PCSK9_6U2N.pdb'), 'w').close()
    open(os.path.join(pdbdir, 'PCSK9_6U2N_0.sdf'), 'w').write('pose')
    open(os.path.join(pdbdir, 'ligand_0.sdf'), 'w').write('lig')
    open(os.path.join(scratch, 'CHEMBL123_bioactives.csv'), 'w').close()

    M.PDB_DIR = pdbdir
    M.SCRATCH_DIR = scratch

    assert is_important_sdf('x/PCSK9_6U2N_0.sdf') is True
    assert is_important_sdf('x/best_pose.sdf') is True
    assert is_important_sdf('x/ligand_0.sdf') is False

    messages = [
      {'role': 'system', 'content': 'sys'},
      {'role': 'user', 'content': 'Dock against PCSK9_6U2N'},
      {'role': 'tool', 'tool_name': 'blind_dock_agent', 'content': 'affinity = -8.5'},
    ]
    status = save_session(messages, time.time() - 1, vault_dir=vault)
    print(status)

    assert not os.path.exists(os.path.join(pdbdir, 'PCSK9_6U2N_0.sdf'))
    assert os.path.exists(os.path.join(pdbdir, 'ligand_0.sdf'))
    assert os.path.exists(os.path.join(pdbdir, 'PCSK9_6U2N.pdb'))
    assert os.path.exists(os.path.join(vault, 'INDEX.md'))

    recalled = recall_session(vault_dir=vault, which='last')
    assert recalled and 'Recalled session' in recalled and '-8.5' in recalled
    assert recall_session(vault_dir=vault, which='1999-01-01') is None
    print(list_sessions(vault_dir=vault))

    shutil.rmtree(tmp, ignore_errors=True)
    print("✓ modrag_memory test passed\n")
except Exception as e:
    print(f"✗ modrag_memory test failed: {e}\n")

# Test 10: shell_node() - sandboxed cross-platform shell (read-only checks,
# no mkdir here so the test never litters the repo)
print("=" * 60)
print("TEST 10: shell_node() - Sandboxed shell tool")
print("=" * 60)
try:
    from modrag_shell_functions import shell_node

    out = shell_node('pwd')
    assert out.endswith('code'), out  # starts in the code/ directory
    out = shell_node('ls ../scratch')
    assert 'shell:' not in out and out.strip() != '', out
    out = shell_node('ls -lt ../scratch')
    assert '-' in out.splitlines()[0], out  # long form
    out = shell_node('find ../scratch -name "*.csv" -maxdepth 1 -type f')
    assert ('.csv' in out) or ('no matches' in out), out
    out = shell_node('find ../scratch -name "*.csv" -maxdepth 1 -type f')
    assert ('.csv' in out) or ('no matches' in out), out
    # regression: command ENDING in a quoted argument must still parse
    out = shell_node('find ../pdb_files -name "*.pdb"')
    assert ('.pdb' in out) or ('no matches' in out), out
    # sandbox: outside the project must be refused
    assert 'project folder denied' in shell_node('ls /etc')
    assert 'project folder denied' in shell_node('cd ~')
    # no chaining, no destructive commands
    assert 'not supported' in shell_node('chaining redir >/dev/null')
    assert 'not supported' in shell_node('rm ../scratch')
    print("✓ shell_node test passed\n")
except Exception as e:
    print(f"✗ shell_node test failed: {e}\n")

# Test 11: admet_node() - local ADMET-AI prediction (subset / full / custom)
print("=" * 60)
print("TEST 11: admet_node() - ADMET-AI toxicity/ADMET prediction")
print("=" * 60)
try:
    import admet_node as M
    M.print_flag = False
    # subset + per-molecule error isolation (model loads once, ~10s)
    dicts, string, _ = M.admet_node(['CCO', 'not_a_smiles'])
    assert len(dicts[0]) == 22 and dicts[1] is None and 'invalid SMILES' in string
    assert 'AMES' in dicts[0] and 'hERG' in dicts[0]
    # full set
    d2, s2, _ = M.admet_node(['CCO'], endpoints='full')
    assert len(d2[0]) > 40, len(d2[0])
    # custom list + unknown name note
    d3, s3, _ = M.admet_node('CCO', endpoints=['AMES', 'Bogus_Name'])
    assert d3[0]['AMES'] and 'Bogus_Name' in s3
    # percentiles interleaved after their endpoints
    d4, _, _ = M.admet_node(['CCO'], include_percentiles=True)
    pct = [k for k in d4[0] if k.endswith('_percentile')]
    assert len(pct) == 22 and 0 <= float(d4[0]['AMES_drugbank_approved_percentile']) <= 100
    print("✓ admet_node test passed (subset=22, full=%d endpoints)\n" % len(d2[0]))
except Exception as e:
    print(f"✗ admet_node test failed: {e}\n")

# Test 12: similarity_node() - Morgan/Tanimoto similarity + grid image
print("=" * 60)
print("TEST 12: similarity_node() - Morgan/Tanimoto similarity")
print("=" * 60)
try:
    import modrag_property_functions as MP
    MP.print_flag = False
    # aspirin as reference; salicylic acid scores 0.25 (above the 0.15
    # cutoff), ethanol 0.11 (below) - verified with BulkTanimotoSimilarity
    ref = ['CC(=O)Oc1ccccc1C(=O)O']
    queries = ['OC(=O)c1ccc(O)cc1', 'CCO']
    data, sim_out, images = MP.similarity_node(ref, queries, sim_cutoff=0.15)
    assert 'Reference molecule 0 and test molecule 0' in sim_out, sim_out
    assert 'could not be parsed' not in sim_out
    # ethanol is dissimilar to aspirin at the default cutoff
    assert 'test molecule 1 have a similarity' not in sim_out, sim_out
    assert images, "expected a grid image for the similar pairs"
    print(sim_out)
    print("✓ similarity_node test passed\n")
except Exception as e:
    print(f"✗ similarity_node test failed: {e}\n")