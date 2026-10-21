from modrag_protein_functions import uniprot_node, listbioactives_node, getbioactives_node, predict_node, docking_node
from modrag_molecule_functions import smiles_node

def get_actives_for_protein(query_protein: str):
  '''
  Finds Bioactive molecules for a give protein. Uses Uniprot to find chembl IDs for the 
  protein, and then queries chembl for bioative molecules. 

  Args:
    query_protein: The protein to search for.

  Returns:
    bioactives_for_protein_string: A string containing bioactive molecules.
    bioacts_images: A list of images of the bioactives.
  '''

  bioactives_for_protein_string = ''

  # find UP accession codes for protein
  up_ac_list, ids_string, _ = uniprot_node([query_protein])

  bioactives_for_protein_string += 'Found the following Uniprot ACs: \n'
  for up_ac in up_ac_list[0]:
    bioactives_for_protein_string += up_ac + ', \n'

  # find chembl IDs for each accession code
  bioacts, chembl_string, chembl_ids = listbioactives_node(up_ac_list[0])

  bioactives_for_protein_string += 'Found the following chembl IDs: \n'
   
  #check for chembl IDs with bioactives:
  largest = 0
  for bioact_num, chemblid in zip(bioacts[0], chembl_ids[0]):
    bioactives_for_protein_string += f'{chemblid}: {bioact_num} bioactive molecules. \n'
    if bioact_num > largest:
      largest = bioact_num
      largest_id = chemblid
  if largest == 0:
    return 'No bioactives found for protein', None

  bioactives_for_protein_string += f'Chose the Chembl ID {largest_id} with {largest} bioactive molecules. \n'
  # get list of bioactives for best chembl_id
  bioacts, bioacts_string, bioacts_images = getbioactives_node([largest_id])

  bioactives_for_protein_string += bioacts_string

  return bioacts, bioactives_for_protein_string, bioacts_images

def get_predictions_for_protein(smiles_list: list[str], query_protein: str):
  '''
  
  Uses Uniprot to find chembl IDs for the protein, and then queries chembl for 
  bioactive molecules to train a model and predict the activity of the given smiles.

  Args:
    smiles_list: A list of SMILES strings.
    query_protein: The protein to search for.

  Returns:
    bioactives_for_protein_string: A string containing bioactive molecules.
    bioacts_images: A list of images of the bioactives.
  '''

  predictions_string = ''

  # find UP accession codes for protein
  up_ac_list, ids_string, _ = uniprot_node([query_protein])

  predictions_string += 'Found the following Uniprot ACs: \n'
  for up_ac in up_ac_list[0]:
    predictions_string += up_ac + ', \n'

  # find chembl IDs for each accession code
  bioacts, chembl_string, chembl_ids = listbioactives_node(up_ac_list[0])

  predictions_string += 'Found the following chembl IDs: \n'
   
  #check for chembl IDs with bioactives:
  largest = 0
  for bioact_num, chemblid in zip(bioacts[0], chembl_ids[0]):
    predictions_string += f'{chemblid}: {bioact_num} bioactive molecules. \n'
    if bioact_num > largest:
      largest = bioact_num
      largest_id = chemblid
  if largest == 0:
    return [], 'No bioactives found for protein'

  predictions_string += f'Chose the Chembl ID {largest_id} with {largest} bioactive molecules. \n'
  # train the model on the chembl ID and then predict
  preds, preds_string, _ = predict_node(smiles_list, largest_id)

  predictions_string += preds_string

  return preds, predictions_string, None

def dock_from_names(names_list: list[str], protein: str):
  '''
  Accepts names of molecules and docks them in a given protein.

  Args:
    names_list: A list of names of molecules.
    protein: The protein to dock in.

  Returns:
    dock_from_names_string: A string containing the docking scores for the molecules.
  '''
  dock_from_names_string = ''

  # get SMILES for names:
  smiles_list, smiles_string, _ = smiles_node(names_list)

  for smiles, names in zip(smiles_list, names_list):
    dock_from_names_string += f'The SMILES for {names} is {smiles}. \n'

  #send for docking
  scores, scores_string, _ = docking_node(smiles_list, protein)

  for score, name in zip(scores, names_list):
    dock_from_names_string += f'The docking score for {name} is {score}. \n'

  return scores, dock_from_names_string, None
  
  