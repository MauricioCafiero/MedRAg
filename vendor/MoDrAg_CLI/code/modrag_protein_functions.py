import pandas as pd
import requests
import os
import itertools
import json
import glob
import math
from rdkit import Chem
from rdkit.Chem import AllChem, QED, Descriptors
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolsToGridImage
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
import numpy as np

try:
  from chembl_webresource_client.new_client import new_client
  chembl_flag = True
except:
  print("Failed to import chembl_webresource_client. Some functionality may be limited.")
  chembl_flag = False
from rcsbapi.search import TextQuery
from dockstring import load_target

# Module-level print flag - set from modrag.py
print_flag = False

# Distance (Angstrom) under which a co-crystallized molecule is reported as
# "close" to the docked ligand (minimum atom-to-atom distance). Mirrors
# vina_dock.NEAR_LIGAND_CUTOFF so "near a known ligand" means the same thing
# in the post-hoc check and the blind-dock proximity fallback.
NEARBY_DISTANCE_CUTOFF = 5.0

def uniprot_node(protein_names: list[str], human_flag: bool = False) -> (list[str], str):
  '''
    This tool takes in the user requested protein and searches UNIPROT for matches.
    It returns a string scontaining the protein ID, gene name, organism, and protein name.
      Args:
        query_protein: the name of the protein to search for.

      Returns:
        total_ids: a list of UNIPROT IDs for the given protein names.
        protein_string: a string containing the protein ID, gene name, organism, and protein name.

  '''
  print("UNIPROT tool")
  print('===================================================')

  total_ids = []
  protein_string = ''

  for protein_name in protein_names:
    try:
      url = f'https://rest.uniprot.org/uniprotkb/search?query={protein_name}&format=tsv'
      response = requests.get(url).text

      f = open(f"../scratch/{protein_name}_uniprot_ids.tsv", "w")
      f.write(response)
      f.close()

      prot_df_raw = pd.read_csv(f'../scratch/{protein_name}_uniprot_ids.tsv', sep='\t')
      if human_flag:
        prot_df = prot_df_raw[prot_df_raw['Organism'] == "Homo sapiens (Human)"]
        if print_flag:
            print(f"Found {len(prot_df)} Human proteins out of {len(prot_df_raw)} total proteins")
      else:
        prot_df = prot_df_raw

      prot_ids = prot_df['Entry'].tolist()
      genes = prot_df['Gene Names'].tolist()
      organisms = prot_df['Organism'].tolist()
      names = prot_df['Protein names'].tolist()

      sub_ids = []
      for id, gene, organism, name in zip(prot_ids, genes, organisms, names):
        protein_string += f'Protein {protein_name}, ID: {id}, Gene: {gene}, Organism: {organism}, Name: {name}\n'
        sub_ids.append(id)

      protein_string += '==========================================================================================\n'
      total_ids.append(sub_ids)
    except:
      protein_string += f'No proteins found for {protein_name}'
      protein_string += '==========================================================================================\n'
      total_ids.append([])

  return total_ids, protein_string, None

def listbioactives_node(up_ids_list: list[str]) -> (list[int], list[str], str):
  '''
    Accepts a UNIPROT ID and searches for bioactive molecules
      Args: 
        up_ids_list: the UNIPROT IDs of the proteins to search for.
      Returns:
        total_bioacts_list: a list of the number of bioactive molecules for each protein
        total_chembl_ids_list: a list of the ChEMBL IDs for each protein
        bioact_string: a string containing the results of the search.
  '''
  print("List bioactives tool")
  print('===================================================')
  if not chembl_flag:
    print("ChEMBL client not available at this time")
    return [], "ChEMBL client not available.", []

  total_bioacts_list = []
  total_chembl_ids_list = []
  bioact_string = ''

  for up_id in up_ids_list:

    targets = new_client.target
    bioact = new_client.activity

    try:
      target_info = targets.get(target_components__accession=up_id).only("target_chembl_id","organism", "pref_name", "target_type")
      target_info = pd.DataFrame.from_records(target_info)
      if print_flag:
          print(target_info)
      if len(target_info) > 0:
        if print_flag:
            print(f"Found info for Uniprot ID: {up_id}")

      chembl_ids = target_info['target_chembl_id'].tolist()

      chembl_ids = list(set(chembl_ids))
      if print_flag:
          print(f"Found {len(chembl_ids)} unique ChEMBL IDs")

      len_all_bioacts = []
      for chembl_id in chembl_ids:
        bioact_chosen = bioact.filter(target_chembl_id=chembl_id, type="IC50", relation="=").only(
            "molecule_chembl_id",
            "type",
            "standard_units",
            "relation",
            "standard_value",
        )
        len_this_bioacts = len(bioact_chosen)
        len_all_bioacts.append(len_this_bioacts)
        bioact_string += f"For Uniprot {up_id}: length of Bioactivities for ChEMBL ID {chembl_id}: {len_this_bioacts}\n"

      bioact_string += f'================================================================================================\n'
      total_chembl_ids_list.append(chembl_ids)
      total_bioacts_list.append(len_all_bioacts)

    except:
      bioact_string += f'No bioactives found for Uniprot {up_id}\n'
      bioact_string += f'================================================================================================\n'
      total_chembl_ids_list.append([])
      total_bioacts_list.append([])

  return total_bioacts_list, bioact_string, total_chembl_ids_list

def getbioactives_node(chembl_ids_list: list[str]) -> (list[str], str):
  '''
    Accepts a Chembl ID and get all bioactives molecule SMILES and IC50s for that ID
      Args:
        chembl_id: the chembl ID to query
      Returns:
        bioactives_list: a list of the bioactive molecules for each chembl ID
        bioactives_string: a string containing the results of the search.
        bioactives_images: a list of images for each bioactive molecule.
  '''
  print("Get bioactives tool")
  print('===================================================')
  if not chembl_flag:
    print("ChEMBL client not available at this time")
    return [], "ChEMBL client not available.", None

  bioactives_list = []
  bioactives_images = []
  bioactives_string = ''

  for chembl_id in chembl_ids_list:
    try:
      #check if f'{chembl_id}_bioactives.csv' exists
      chembl_id = chembl_id.upper()
      if os.path.exists(f'../scratch/{chembl_id}_bioactives.csv'):
        if print_flag:
            print(f'Found {chembl_id}_bioactives.csv')
        total_bioact_df = pd.read_csv(f'../scratch/{chembl_id}_bioactives.csv')
        if print_flag:
            print(f"number of records: {len(total_bioact_df)}")
      else:

        compounds = new_client.molecule
        bioact = new_client.activity

        bioact_chosen = bioact.filter(target_chembl_id=chembl_id, type="IC50", relation="=").only(
            "molecule_chembl_id",
            "type",
            "standard_units",
            "relation",
            "standard_value",
        )

        chembl_ids = []
        ic50s = []
        for record in bioact_chosen:
            if record["standard_units"] == 'nM':
                chembl_ids.append(record["molecule_chembl_id"])
                ic50s.append(float(record["standard_value"]))

        bioact_dict = {'chembl_ids' : chembl_ids, 'IC50s': ic50s}
        bioact_df = pd.DataFrame.from_dict(bioact_dict)
        bioact_df.drop_duplicates(subset=["chembl_ids"], keep= "last")
        if print_flag:
            print(f"Number of records: {len(bioact_df)}")
            print(bioact_df.shape)

        compounds_provider = compounds.filter(molecule_chembl_id__in=bioact_df["chembl_ids"].to_list()).only(
            "molecule_chembl_id",
            "molecule_structures"
        )

        cids_list = []
        smiles_list = []

        for record in compounds_provider:
            cid = record['molecule_chembl_id']
            cids_list.append(cid)

            if record['molecule_structures']:
                if record['molecule_structures']['canonical_smiles']:
                    smile = record['molecule_structures']['canonical_smiles']
                else:
                    if print_flag:
                        print("No canonical smiles")
                    smile = None
            else:
                if print_flag:
                    print('no structures')
                smile = None
            smiles_list.append(smile)

        new_dict = {'SMILES': smiles_list, 'chembl_ids_2': cids_list}
        new_df = pd.DataFrame.from_dict(new_dict)

        total_bioact_df = pd.merge(bioact_df, new_df, left_on='chembl_ids', right_on='chembl_ids_2')
        if print_flag:
            print(f"number of records: {len(total_bioact_df)}")

        total_bioact_df.drop_duplicates(subset=["chembl_ids"], keep= "last")
        if print_flag:
            print(f"number of records after removing duplicates: {len(total_bioact_df)}")

        total_bioact_df.dropna(axis=0, how='any', inplace=True)
        total_bioact_df.drop(["chembl_ids_2"],axis=1,inplace=True)
        if print_flag:
            print(f"number of records after dropping Null values: {len(total_bioact_df)}")

        total_bioact_df.sort_values(by=["IC50s"],inplace=True)

        if len(total_bioact_df) > 0:
          total_bioact_df.to_csv(f'../scratch/{chembl_id}_bioactives.csv')

      limit = 50
      if len(total_bioact_df) > limit:
        total_bioact_df = total_bioact_df.iloc[:limit]

      bioact_tuple_list = []
      bioactives_string += f'Results for top bioactivity (IC50 value) for molecules in ChEMBL ID: {chembl_id}. \n'
      for smile, ic50 in zip(total_bioact_df['SMILES'], total_bioact_df['IC50s']):
        bioactives_string += f'Molecule SMILES: {smile}, IC50 (nM): {ic50}\n'
        bioact_tuple_list.append((smile, ic50))
      bioactives_string += f'=========================================================================================\n'

      mols = [Chem.MolFromSmiles(smile) for smile in total_bioact_df['SMILES'].to_list()]
      legends = [f'IC50: {ic50}' for ic50 in total_bioact_df['IC50s'].to_list()]
      img = MolsToGridImage(mols, molsPerRow=5, legends=legends, subImgSize=(200,200))
      # Save image to chat location
      if not os.path.exists('../images'):
        os.makedirs('../images')
      img.save('../images/chat_image.png')
      bioactives_images.append(img)
      bioactives_list.append(bioact_tuple_list)
    except: 
      bioactives_list.append([])
      bioactives_string += f'No bioactives found for ChEMBL ID: {chembl_id}\n'
      bioactives_string += f'=========================================================================================\n'
      bioactives_images.append(None)

  return bioactives_list, bioactives_string, bioactives_images

def get_protein_from_pdb(pdb_id):
  '''
    Helper function to get the protein information from the PDB database.
    Args:
      pdb_id: the PDB ID of the protein
    Returns:
      r.text: the PDB information as a string
  '''
  url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
  r = requests.get(url)
  return r.text

def get_pdb_file(pdb_id: str, protein_name: str) -> str:
  '''
    Retrieves a protein PDB file from the PDB database and caches it on disk
    for use as a receptor in blind docking (see vina_dock.blind_dock_agent).
    Reuses an existing cached file for the same PDB ID if one is already
    present in ../pdb_files/ (matched by the PDB-ID suffix of the filename, so
    the leading protein-name part may differ).

      Args:
        pdb_id: the PDB ID of the protein
        protein_name: the name of the protein (used to name the cached file)
      Returns:
        A status string describing whether the file was reused or downloaded,
        and the path it was saved to (../pdb_files/<protein_name>_<pdb_id>.pdb).
  '''
  print('PDB retrieval tool')
  print('===================================================')

  # Check whether a .pdb file for this PDB ID is already present in ../pdb_files/.
  # The leading part of the filename may differ (e.g. "sult1a3" vs "sulfotransferase"),
  # so only the PDB ID is used as the match test.
  pdb_id_upper = pdb_id.upper()
  for existing in glob.glob('../pdb_files/*.pdb'):
    stem = os.path.basename(existing).rsplit('.', 1)[0]
    if stem.upper().endswith(f'_{pdb_id_upper}'):
      print(f'PDB file for PDB ID {pdb_id} already present at {existing}; reusing it.')
      return (f'The PDB file for protein {protein_name} with PDB ID {pdb_id} was already '
              f'present in ../pdb_files/ as {existing} and has been reused.')

  url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
  r = requests.get(url)
  os.makedirs('../pdb_files', exist_ok=True)
  with open(f"../pdb_files/{protein_name}_{pdb_id}.pdb", 'w') as f:
    f.write(r.text)

  return f'The PDB file for protein {protein_name} with PDB ID {pdb_id} has been retrieved and saved as ../pdb_files/{protein_name}_{pdb_id}.pdb'

def check_nearby_molecules(pdb_filepath: str, ligand_filepath: str) -> str:
  '''
    Checks for nearby molecules in the PDB file to assess if the docking has
    located the correct binding site. Should be called to verify blind docking
    in the case where a ligand is present in the crystal structure.

      Args:
        pdb_filepath: the path to the PDB file
        ligand_filepath: the path to the ligand file (SDF of docked poses)
      Returns:
        nearby_molecules: a string containing the results of the check.
  '''
  print(f"Nearby molecules check tool")
  print('===================================================')

  with open(pdb_filepath, 'r') as pdb_file:
      pdb_content = pdb_file.readlines()

  ligand_names = {}
  for line in pdb_content:
    if line.startswith('HETNAM'):
      molecule_symbol = line[11:15].strip()
      molecule_name = line[15:70].strip()
      if not molecule_symbol:
        continue
      if molecule_symbol not in ligand_names:
        ligand_names[molecule_symbol] = molecule_name
      elif molecule_name:
        ligand_names[molecule_symbol] += ' ' + molecule_name

  molecule_dict = {}
  for line in pdb_content:
    if line.startswith('HETATM'):
      molecule_symbol = line[17:20].strip()
      if molecule_symbol not in ligand_names:
        continue
      chain_id = line[21].strip()
      occ_key = (molecule_symbol, chain_id)
      if occ_key not in molecule_dict:
        molecule_dict[occ_key] = {'name': ligand_names[molecule_symbol] or molecule_symbol,
                                  'coords': []}
      x_coord = float(line[30:38])
      y_coord = float(line[38:46])
      z_coord = float(line[46:54])
      molecule_dict[occ_key]['coords'].append((x_coord, y_coord, z_coord))

  ligand_atom_coords = []
  try:
    supplier = Chem.SDMolSupplier(ligand_filepath, removeHs=True)
    poses = [m for m in supplier if m is not None]
  except Exception:
    poses = []
  for mol in poses:
    if not ligand_atom_coords:
      conf = mol.GetConformer()
      ligand_atom_coords = [(conf.GetAtomPosition(j).x,
                             conf.GetAtomPosition(j).y,
                             conf.GetAtomPosition(j).z) for j in range(mol.GetNumAtoms())]

  nearby_molecules = f'Checked for nearby molecules in {pdb_filepath}.\n'
  if not ligand_atom_coords:
    nearby_molecules += 'No ligand could be read from the SDF.\n'
  else:
    distances = []
    for (molecule_symbol, chain_id), info in molecule_dict.items():
      mol_coords = info['coords']
      if not mol_coords or not ligand_atom_coords:
        continue
      best = None
      for lx, ly, lz in ligand_atom_coords:
        for mx, my, mz in mol_coords:
          d = math.sqrt((lx - mx) ** 2 + (ly - my) ** 2 + (lz - mz) ** 2)
          if best is None or d < best:
            best = d
      distances.append((best, molecule_symbol, chain_id, info))
    distances.sort(key=lambda d: d[0])

    if distances:
      # Headline: the closest co-crystallized ligand and its min atom-to-atom
      # distance, regardless of cutoff -- so the caller always gets a number,
      # not just a yes/no "nearby". Tagged CLOSE (within cutoff) or FAR.
      cdist, csym, cchain, cinfo = distances[0]
      cname = cinfo['name']
      tag = 'CLOSE' if cdist <= NEARBY_DISTANCE_CUTOFF else 'FAR'
      nearby_molecules += (f'Closest co-crystallized ligand: {cname} '
                           f'({csym}, chain {cchain}): min atom-to-atom '
                           f'{cdist:.2f} A -- {tag}\n')
      nearby_molecules += (f'Other molecules within {NEARBY_DISTANCE_CUTOFF:.1f} A '
                           f'(min atom-to-atom distance):\n')
      any_close = False
      for dist, molecule_symbol, chain_id, info in distances[1:]:
        if dist > NEARBY_DISTANCE_CUTOFF:
          break  # distances is sorted; nothing further can be close
        any_close = True
        molecule_name = info['name']
        nearby_molecules += (f'  {molecule_name} ({molecule_symbol}, chain '
                             f'{chain_id}): min atom-to-atom {dist:.2f} A -- '
                             f'CLOSE\n')
      if not any_close:
        nearby_molecules += '  (none other within cutoff)\n'
    else:
      nearby_molecules += ('No co-crystallized ligands (HETNAM-keyed HETATM) '
                           'found in the PDB.\n')

  return nearby_molecules

def one_to_three(one_seq):
  '''
    Converts a one-letter amino acid sequence to a three-letter sequence.
    Args:
      one_seq: the one-letter amino acid sequence
    Returns:
      three_seq: the three-letter amino acid sequence
  '''
  rev_aa_hash = {
      'A': 'ALA',
      'R': 'ARG',
      'N': 'ASN',
      'D': 'ASP',
      'C': 'CYS',
      'Q': 'GLN',
      'E': 'GLU',
      'G': 'GLY',
      'H': 'HIS',
      'I': 'ILE',
      'L': 'LEU',
      'K': 'LYS',
      'M': 'MET',
      'F': 'PHE',
      'P': 'PRO',
      'S': 'SER',
      'T': 'THR',
      'W': 'TRP',
      'Y': 'TYR',
      'V': 'VAL'
  }

  try:
    three_seq = rev_aa_hash[one_seq]
  except:
    three_seq = 'X'

  return three_seq

def three_to_one(three_seq):
  '''
  Converts a three-letter amino acid sequence to a one-letter sequence.
  Args:
    three_seq: the three-letter amino acid sequence
  Returns:
    one_seq: the one-letter amino acid sequence
  '''
  aa_hash = {
      'ALA': 'A',
      'ARG': 'R',
      'ASN': 'N',
      'ASP': 'D',
      'CYS': 'C',
      'GLN': 'Q',
      'GLU': 'E',
      'GLY': 'G',
      'HIS': 'H',
      'ILE': 'I',
      'LEU': 'L',
      'LYS': 'K',
      'MET': 'M',
      'PHE': 'F',
      'PRO': 'P',
      'SER': 'S',
      'THR': 'T',
      'TRP': 'W',
      'TYR': 'Y',
      'VAL': 'V'
  }

  one_seq = []
  for residue in three_seq:
    try:
      one_seq.append(aa_hash[residue])
    except:
      one_seq.append('X')

  return one_seq

def pdb_node(test_pdb_list: list[str]) -> (list[str], str):
  '''
    Accepts a PDB ID and queires the protein databank for the sequence of the protein, as well as other
    information such as ligands.
      Args:
        test_pdb_list: the PDB IDs to query
      Returns:
        all_seqs: a list of the sequences for each PDB ID
        total_pdb_string: a string containing the results of the PDB query.
      (collects all ligands but does not return them currently)
  '''

  print(f"pdb tool")
  print('===================================================')

  total_pdb_string = ''
  all_seqs = []
  all_ligands = []

  for test_pdb in test_pdb_list:
    try:
      pdb_str = get_protein_from_pdb(test_pdb)
      chains = {}
      other_molecules = {}

      #print(pdb_str.split('\n')[0])
      for line in pdb_str.split('\n'):
        parts = line.split()
        try:
          if parts[0] == 'SEQRES':
            if parts[2] not in chains:
              chains[parts[2]] = []
            chains[parts[2]].extend(parts[4:])
          if parts[0] == 'HETNAM':
            j = 1
            if parts[1].strip() in ['2','3','4','5','6','7','8','9']:
              j = 2
            if print_flag:
                print(parts[j])
            if parts[j] not in other_molecules:
              other_molecules[parts[j]] = []
            other_molecules[parts[j]].extend(parts[2:])
        except:
          if print_flag:
              print('Blank line')

        chains_ol = {}
        for chain in chains:
          chains_ol[chain] = three_to_one(chains[chain])

      sub_seqs = []
      sub_ligands = []
      total_pdb_string += f"Chains in PDB ID {test_pdb}: {', '.join(chains.keys())} \n"
      for chain in chains_ol:
        total_pdb_string += f"Chain {chain}: {''.join(chains_ol[chain])} \n"
        sub_seqs.append(''.join(chains_ol[chain]))
        if print_flag:
            print(f"Chain {chain}: {''.join(chains_ol[chain])}")
      total_pdb_string += f"Ligands in PDB ID {test_pdb}.\n"
      for mol in other_molecules:
        total_pdb_string += f"Molecule {mol}: {''.join(other_molecules[mol])} \n"
        sub_ligands.append(''.join(other_molecules[mol]))
      total_pdb_string += f'=========================================================================================\n'

      all_seqs.append(sub_seqs)
      all_ligands.append(sub_ligands)
    except:
      total_pdb_string += f'Failed to get data for PDB ID {test_pdb}\n'
      total_pdb_string += f'=========================================================================================\n'
      all_seqs.append([])
      all_ligands.append([])

  return all_seqs, total_pdb_string, None

def find_PDBID_node(test_protein_list: list[str]) -> (list[str], str):
  '''
    Accepts a protein name and searches the protein databank for PDB IDs that match along with the entry titles.
      Args:
        test_protein_list: the protein names to query
      Returns:
        total_ids: a list of the PDB IDs for each protein name
        pdb_string: a string containing the results of the PDB search.
  '''

  print(f"PDB search tool")
  print('===================================================')

  total_ids = []
  pdb_string = ''
  which_pdbs = 0

  for test_protein in test_protein_list:
    try:
      query = TextQuery(value=test_protein)
      results = query()

      def pdb_gen():
        for rid in results:
          yield(rid)

      take10 = itertools.islice(pdb_gen(), which_pdbs, which_pdbs+10, 1)

      local_ids = []
      pdb_string += f'10 PDBs that match the protein {test_protein} are: \n'
      for pdb in take10:
        data = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb}").json()
        title = data['struct']['title']
        pdb_string += f'PDB ID: {pdb}, with title: {title} \n'
        local_ids.append(pdb)
      total_ids.append(local_ids)
    except:
      pdb_string += f'Failed to get PDB IDs for protein {test_protein}\n'
      total_ids.append([])

  return total_ids, pdb_string, None

def target_node(search_descriptors: list[str]):
  '''
  Accepts a disease name and searches Open Targets for associated targets

  Args:
    search_descriptor (str): Disease name

  Returns:
    targets_list (list): List of targets
    targets_string (str): String of targets
    None
  '''
  print("Open Targets tool")
  print('===================================================')
  base_url = "https://api.platform.opentargets.org/api/v4/graphql"

  disease_query_string = """
    query searchEntity($queryString: String!) {
      search(queryString: $queryString){
        total
        hits  {
          id
          entity
          description
        }
      }
    }
  """

  target_query_string = """
    query associatedTargets($efo_id: String!) {
      disease(efoId: $efo_id) {
        id
        name
        associatedTargets {
          count
          rows {
            target {
              id
              approvedSymbol
            }
            score
          }
        }
      }
    }
  """
  total_targets_list = []
  total_targets_string = ''

  for search_descriptor in search_descriptors:

    variables = {"queryString": search_descriptor}
    r = requests.post(base_url, json={"query": disease_query_string, "variables": variables})

    disease_list = []
    targets_list = []

    if r.status_code == 200:
      api_response = json.loads(r.text)
      if len(api_response['data']['search']['hits']) > 0:
        for hit in api_response['data']['search']['hits']:
          if hit['entity'] == 'disease':
            disease_list.append(hit['id'])
    else:
      if print_flag:
          print('Could not find results.')

    if len(disease_list) > 0:
      q = requests.post(base_url, json={"query": target_query_string, "variables": {"efo_id": disease_list[0]}})
      if q.status_code == 200:
        api_response = json.loads(q.text)
        for target in api_response['data']['disease']['associatedTargets']['rows']:
          targets_list.append(target['target']['approvedSymbol'])

    targets_string = f'Possible targets for {search_descriptor} include: \n'
    if len(targets_list) > 0:
      for i, target in enumerate(targets_list):
        targets_string += f'{i+1}. {target}\n'
    else:
      targets_string = f'No targets found for {search_descriptor}'
    
    total_targets_list.append(targets_list)
    total_targets_string += targets_string

  return total_targets_list, total_targets_string, None

def docking_node(smiles_list: list[str], query_protein: str) -> (list[float], str):
  '''
    Docking tool: uses dockstring to dock the molecule into the protein.
    PREFERRED docking path: use this FIRST if the user's protein is in the list
    below. If the protein is NOT in this list, or the user supplies a PDB file /
    unknown binding site, use blind_dock_agent instead (which needs a receptor
    PDB file -- fetch one with get_pdb_file if only a name or PDB ID is given).
    The query proteins can
    be any of the following list: IGF1R,JAK2,KIT,LCK,MAPK14,MAPKAPK2,MET,PTK2,PTPN1,SRC,ABL1,AKT1,
    AKT2,CDK2,CSF1R,EGFR,KDR,MAPK1,FGFR1,ROCK1,MAP2K1,PLK1,HSD11B1,PARP1,PDE5A,PTGS2,ACHE,MAOB,CA2,
    GBA,HMGCR,NOS1,REN,DHFR,ESR1,ESR2,NR3C1,PGR,PPARA,PPARD,PPARG,AR,THRB,ADAM17,F10,F2,BACE1,CASP3,
    MMP13,DPP4,ADRB1,ADRB2,DRD2,DRD3,ADORA2A,CYP2C9,CYP3A4,HSP90AA1

    Args:
      smiles_list: the SMILES strings of the molecules to dock
      protein: the protein to dock into
    Returns:
      docking_scores: a list of docking scores for each molecule
      docking_string: a string containing the results of the docking.
  '''
  print("docking tool")
  print('===================================================')
  cpuCount = os.cpu_count()
  if print_flag:
      print(f"Number of CPUs: {cpuCount}")

  if print_flag:
      print(f'query_protein: {query_protein}')

  scores_list = []
  scores_string = 'Docking below performed with AutoDock Vina on protein structures from the DUDE database.\n'

  for query_smiles in smiles_list:
    try:
      query_smiles = query_smiles.replace('.[Na+]','').replace('.[Na+]','').replace('.[K+]','').replace('[K+].','').replace('.[Cl-]','').replace('[Cl-].','')
      target = load_target(query_protein)
      if print_flag:
          print("===============================================")
          print(f"Docking molecule with {cpuCount} cpu cores.")
      score, aux = target.dock(query_smiles, num_cpus = cpuCount)
      scores_list.append(score)
      mol = aux['ligand']
      if print_flag:
          print(f"Docking score: {score}")
          print("===============================================")
      atoms_list = ""
      template = mol
      molH = Chem.AddHs(mol)
      AllChem.ConstrainedEmbed(molH,template, useTethers=True)
      xyz_string = f"{molH.GetNumAtoms()}\n\n"
      for atom in molH.GetAtoms():
        atoms_list += atom.GetSymbol()
        pos = molH.GetConformer().GetAtomPosition(atom.GetIdx())
        xyz_string += f"{atom.GetSymbol()} {pos[0]} {pos[1]} {pos[2]}\n"
      scores_string += f"Docking score for molecule with SMILES: {query_smiles} is: {score} kcal/mol \n\n"
      scores_string += f"pose XYZ structure for molecule with SMILES: {query_smiles} is: \n"
      lines = xyz_string.split('\n')
      for line in lines[2:]:
        scores_string += f'{line}\n'
      scores_string += f"=========================================================\n"

    except:
      if print_flag:
          print(f"Molecule {query_smiles} could not be docked!")
      scores_string = 'Could not dock!'
      scores_list.append(None)

  return scores_list, scores_string, None

def rdkit_featurize(smiles_list: list, target_list: list):
  '''
    Takes a list of SMILES strings and creates features using the RDKit set of
    descriptors.

      Args:
        smiles_list: List of SMILES strings to featurize
        target_list: List of the ground truth values for each molecule
      Returns:
        X: 2D list of features (rows are molecules, columns are features)
        y: list of target values
        mols: list of RDKit mol objects
        legend: list of SMILES strings (should be identical to input list,
                unless a molecule could not be featurized, in which case that molecule
                is left out)
  '''
  if target_list is None:
    target_list = [0.0] * len(smiles_list)

  X = []
  mols = []
  legend = []
  y = []
  add_flag = True
  for i,smile in enumerate(smiles_list):
    try:
      mol = Chem.MolFromSmiles(smile)
      dictionary_descriptors = Chem.Descriptors.CalcMolDescriptors(mol)
      temp_vec = []
      for key in dictionary_descriptors:
        temp_vec.append(dictionary_descriptors[key])
        add_flag = True
      X.append(temp_vec)
      mols.append(mol)
      legend.append(smile)
      y.append(target_list[i])
    except:
      if print_flag:
        print(f"Could not featurize molecule {i}")

  if print_flag:
    print(f"Total number of molecules: {len(X)}")
  if print_flag:
    print(f"Total number of descriptors per molecule: {len(X[0])}")

  return X, y, legend

def predict_node(smiles_list_in: list[str], chembl_id: str) -> (list[float],str,list):
  '''
    uses the current_bioactives.csv file from the get_bioactives node to fit the
    Light GBM model and predict the IC50 for the current smiles.
      Args:
        smiles_list: the SMILES strings of the molecules to predict
        chembl_id: the chembl ID to query
      Returns:
        preds: a list of predicted IC50 values for the input SMILES
        preds_string: a string containing the predicted IC50 values for the input SMILES
        preds_images: a list of images for each predicted molecule (currently not implemented)
  '''
  print("Predict Tool")
  print('===================================================')

  # if f'{chembl_id}_bioactives.csv' does not exist, call the bioactives node
  if not os.path.exists(f'../scratch/{chembl_id}_bioactives.csv'):
    _, _, _ = getbioactives_node([chembl_id])
  
  try:
    chembl_id = chembl_id.upper()
    df = pd.read_csv(f'../scratch/{chembl_id}_bioactives.csv')
    #if length of the dataframe is over 2000, take a random sample of 2000 points
    if len(df) > 2000:
      df = df.sample(n=2000, random_state=42)

    y_raw = df["IC50s"].to_list()
    smiles_list = df["SMILES"].to_list()
    ions_to_clean = ["[Na+].",".[Na+]","[Cl-].",".[Cl-]","[K+].",".[K+]"]
    Xa = []
    y = []
    for smile, value in zip(smiles_list, y_raw):
      for ion in ions_to_clean:
        smile = smile.replace(ion,"")
      # IC50 <= 0 is physically meaningless (missing/placeholder value); log10 is
      # undefined there and would inject -inf into the training targets, so skip
      # those rows rather than emitting a divide-by-zero warning.
      if value is None or value <= 0:
        if print_flag:
          print(f"Skipping molecule with non-positive IC50 ({value}): {smile}")
        continue
      y.append(np.log10(value))
      Xa.append(smile)

    if print_flag:
      print(f"Number of molecules: {len(Xa)}")

   
    f, y, Xa = rdkit_featurize(Xa, y)
    f = np.array(f)

    nan_indicies = np.isnan(f)
    bad_rows = []
    for i, row in enumerate(nan_indicies):
        for item in row:
            if item == True:
                if i not in bad_rows:
                    if print_flag:
                      print(f"Row {i} has a NaN.")
                    bad_rows.append(i)

    if print_flag:
      print(f"Old dimensions are: {f.shape}.")

    for j,i in enumerate(bad_rows):
        k=i-j
        f = np.delete(f,k,axis=0)
        y = np.delete(y,k,axis=0)
        Xa = np.delete(Xa,k,axis=0)
        if print_flag:
          print(f"Deleting row {k} from arrays.")

    if print_flag:
      print(f"New dimensions are: {f.shape}")
    if f.shape[0] != len(y) or f.shape[0] != len(Xa):
      raise ValueError("Number of rows in X and y do not match.")

    # Create feature names for the model
    feature_names = [f"descriptor_{i}" for i in range(f.shape[1])]
    
    X_train, X_test, y_train, y_test = train_test_split(f, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to DataFrames with feature names
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)

    # n_jobs=1 forces single-threaded LightGBM. Inside the modrag process, torch
    # (Metal/MPS) and the meeko/autodock-vina native libs are already loaded, and
    # LightGBM's default OpenMP parallel region (n_jobs=-1) SIGSEGVs (exit 139) when
    # it spawns OMP threads against that already-initialized runtime — predict_node
    # dies right after "New dimensions are: ...". Disabling OpenMP parallelism
    # avoids the conflict; on ~2k rows x 217 descriptors the fit is sub-second
    # anyway, so there is no real cost. Do NOT set this back to -1. See memory note
    # gpt-finetune-chembl220-crash for the same class of native-lib/Metal crash.
    model = LGBMRegressor(metric='rmse', max_depth = 50, verbose = -1, num_leaves = 31,
                          feature_fraction = 0.8, min_data_in_leaf = 20, n_jobs=1)
    modelname = "LightGBM Regressor"
    model.fit(X_train, y_train)

    train_score = model.score(X_train,y_train)
    if print_flag:
      print(f"score for training set: {train_score:.3f}")

    valid_score = model.score(X_test, y_test)
    if print_flag:
      print(f"score for validation set: {valid_score:.3f}")
  except:
    return [], 'Model training failed, unable to predict.', None

  preds = []
  preds_string = ''

  for smiles in smiles_list_in:
    if print_flag:
      print(f"in predict node, smiles: {smiles}")
    idx = 0
    try:
      for ion in ions_to_clean:
        smiles = smiles.replace(ion,"")
      test_feat, _, _ = rdkit_featurize([smiles], None)
      test_feat = scaler.transform(test_feat)
      # Convert to DataFrame with same feature names used during training
      test_feat = pd.DataFrame(test_feat, columns=feature_names)
      prediction = model.predict(test_feat)
      test_ic50 = 10**(prediction[0])
      if print_flag:
        print(f"Predicted IC50 for {smiles}: {test_ic50}")
      preds_string += f"The predicted IC50 value for {smiles} is : {test_ic50:.3f} nM.\n"
      
      preds.append(test_ic50)
      idx+=1
    except:
      preds.append(None)
      preds_string += f"The prediction for {smiles} failed.\n"

  preds_string += f"The Bioactive data was fitted with the LightGMB model, using RDKit descriptors. The training score \
was {train_score:.3f} and the testing score was {valid_score:.3f}. "

  return preds, preds_string, None