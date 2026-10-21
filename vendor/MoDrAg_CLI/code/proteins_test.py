from modrag_molecule_functions import *
from modrag_property_functions import *
from modrag_protein_functions import *

# Test: uniprot_node() - Search UNIPROT for proteins
print("=" * 60)
print("TEST: uniprot_node() - UNIPROT Protein Search")
print("=" * 60)
try:
    protein_names = ['MAOB']
    protein_ids, protein_string, _ = uniprot_node(protein_names, human_flag=True)
    print("Input Protein Names:", protein_names)
    print("Number of results per query:", [len(ids) for ids in protein_ids])
    print(protein_string[:800], "...")  # Print first 800 chars
    print("✓ uniprot_node test passed\n")
except Exception as e:
    print(f"✗ uniprot_node test failed: {e}\n")

# Test: listbioactives_node() - Find bioactive molecules for proteins
print("=" * 60)
print("TEST: listbioactives_node() - List Bioactives")
print("=" * 60)
try:
    # Using a common protein ID (P06213 is insulin receptor)
    up_ids_list = ['P06213']
    bioacts_list, bioacts_string, chembl_ids = listbioactives_node(up_ids_list)
    print("Input UNIPROT IDs:", up_ids_list)
    print("Number of bioactives per protein:", bioacts_list)
    print("ChEMBL IDs found:", chembl_ids)
    print(bioacts_string[:600], "...")  # Print first 600 chars
    print("✓ listbioactives_node test passed\n")
except Exception as e:
    print(f"✗ listbioactives_node test failed: {e}\n")

# Test: getbioactives_node() - Get bioactive molecules for ChEMBL IDs
print("=" * 60)
print("TEST: getbioactives_node() - Get Bioactives")
print("=" * 60)
try:
    # Using a common ChEMBL ID
    chembl_ids_list = ['CHEMBL2039']
    bioactives_list, bioactives_string, bioactives_images = getbioactives_node(chembl_ids_list)
    print("Input ChEMBL IDs:", chembl_ids_list)
    print("Number of bioactive molecules found:", len(bioactives_list[0]) if bioactives_list[0] else 0)
    print(bioactives_string[:600], "...")  # Print first 600 chars
    if bioactives_images and bioactives_images[0]:
        bioactives_images[0].save("../outputs/bioactives_output.png")
        print("Saved bioactives image as 'bioactives_output.png'")
    print("✓ getbioactives_node test passed\n")
except Exception as e:
    print(f"✗ getbioactives_node test failed: {e}\n")

# Test: find_PDBID_node() - Find PDB IDs for protein names
print("=" * 60)
print("TEST: find_PDBID_node() - Find PDB IDs")
print("=" * 60)
try:
    protein_names = ['MAOB']
    pdb_ids, pdb_search_string, _ = find_PDBID_node(protein_names)
    print("Input Protein Names:", protein_names)
    print("Number of PDB IDs found:", [len(ids) for ids in pdb_ids])
    print(pdb_search_string[:600], "...")  # Print first 600 chars
    print("✓ find_PDBID_node test passed\n")
except Exception as e:
    print(f"✗ find_PDBID_node test failed: {e}\n")

# Test: pdb_node() - Get sequences and ligands from PDB IDs
print("=" * 60)
print("TEST: pdb_node() - Get PDB Sequences and Ligands")
print("=" * 60)
try:
    pdb_ids = ['2A3R']  # Insulin PDB ID
    sequences, pdb_string, _ = pdb_node(pdb_ids)
    print("Input PDB IDs:", pdb_ids)
    print("Number of sequences found:", len(sequences))
    print("Chains per PDB:", [len(seqs) for seqs in sequences])
    print(pdb_string[:600], "...")  # Print first 600 chars
    print("✓ pdb_node test passed\n")
except Exception as e:
    print(f"✗ pdb_node test failed: {e}\n")

# Test: target_node() - Find disease targets
print("=" * 60)
print("TEST: target_node() - Find Disease Targets")
print("=" * 60)
try:
    disease_names = ["Parkinson's Disease"]
    targets_list, targets_string, _ = target_node(disease_names)
    print("Input Disease Names:", disease_names)
    print("Number of targets found:", [len(targets) for targets in targets_list])
    print(targets_string[:600], "...")  # Print first 600 chars
    print("✓ target_node test passed\n")
except Exception as e:
    print(f"✗ target_node test failed: {e}\n")

# Test: docking_node() - Dock molecules into proteins
print("=" * 60)
print("TEST: docking_node() - Molecular Docking")
print("=" * 60)
try:
    smiles_list = ['[NH3+]CCc1ccc(O)c(O)c1']  # Ethanol and Ethane
    protein = 'DRD2'  # Dopamine receptor D2
    docking_scores, docking_string, _ = docking_node(smiles_list, protein)
    print("Input SMILES:", smiles_list)
    print("Target Protein:", protein)
    print("Docking Scores:", docking_scores)
    print(docking_string[:600], "...")  # Print first 600 chars
    print("✓ docking_node test passed\n")
except Exception as e:
    print(f"✗ docking_node test failed: {e}\n")
