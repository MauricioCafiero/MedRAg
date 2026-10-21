from rdkit import Chem
from rdkit.Chem import AllChem, QED
from rdkit.Chem import Draw
import os, re
import pubchempy as pcp
from PIL import Image
from collections import Counter

# Module-level print flag - set from modrag.py
print_flag = True

def canonical_node(smiles: str) -> str:
  '''
    converts a SMILES string to a canonical form. This is useful for ensuring that
    the substituents are added to the correct positions on the ring.
    Args:
      smiles: a SMILES string for a molecule

    Returns:
      new_smiles: a canonical SMILES string for the same molecule
  '''
  new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
  return new_smiles

def name_node(smiles_list: list[str]) -> (list[str], str):
  '''
    Queries Pubchem for the name of the molecule based on the smiles string.
      Args:
        smiles_list: the list of input smiles strings
      Returns:
        names_list: the list of names of the molecules
        name_string: a string of the tool results
  '''
  print("name tool")
  print('===================================================')

  names = []
  name_string = ''
  for smiles in smiles_list:
    try:
        res = pcp.get_compounds(smiles, "smiles")
        name = res[0].iupac_name
        names.append(name)
        name_string += f'{smiles}: IUPAC molecule name: {name}\n'
        if print_flag:
            print(smiles, name)
        syn_list = pcp.get_synonyms(res[0].cid)
        for alt_name in syn_list[0]['Synonym'][:5]:
            name_string += f'{smiles}: alternative or common name: {alt_name}\n'
    except:
        name = "unknown"
        name_string += f'{smiles}: Fail\n'

  return names, name_string, None

def smiles_node(names_list: list[str]) -> (list[str], str):
  '''
    Queries Pubchem for the smiles string of the molecule based on the name.
      Args:
        names_list: the list of molecule names
      Returns:
        smiles_list: the list of smiles strings of the molecules    
        smiles_string: a string of the tool results
  '''
  print("smiles tool")
  print('===================================================')

  # Coerce common argument shapes into a list of names. The LLM agent calls
  # this node as smiles_node(**tc.function.arguments) and frequently passes a
  # bare string ("aspirin") or a comma/whitespace-separated string instead of
  # a list. Without this, `for name in names_list` iterates the string
  # character-by-character and queries PubChem for 'a', 's', 'p', ... which
  # accidentally resolve to elements ('s'->[S], 'p'->[P], 'i'->II, 'n'->N#N),
  # returning garbage SMILES for molecules that should resolve cleanly.
  if isinstance(names_list, str):
      names_list = [s for s in names_list.replace(',', ' ').split() if s]

  smiles_list = []
  smiles_string = ''
  for name in names_list:
    name = name.strip()
    try:
        res = pcp.get_compounds(name, "name")
        smiles = res[0].smiles
        #smiles = smiles.replace('#','~')
        smiles_list.append(smiles)
        smiles_string += f'{name}: The SMILES string for the molecule is: {smiles}\n'
    except:
        smiles = "unknown"
        smiles_string += f'{name}: Fail\n'

  return smiles_list, smiles_string, None

def related_node(smiles_list: list[str]) -> (list[list[str]], str, list):
  '''
    Queries Pubchem for similar molecules based on the smiles string or name
      Args:
        smiles: the input smiles string, OR
        name: the molecule name
      Returns:
        total_similar_list: a list of lists of similar molecules
        related_string: a string of the tool results
        all_images: a list of images of the similar molecules
  '''
  print("related tool")
  print('===================================================')


  total_similar_list = []
  all_images = []
  related_string = ''
  for smiles in smiles_list:
    try:
        res = pcp.get_compounds(smiles, "smiles", searchtype="similarity",listkey_count=50)
        related_string += f'The following molecules are similar to {smiles}: \n'
        if print_flag:
            print('got related molecules with smiles')

        sub_smiles = []

        i = 0
        for compound in res:
            if i == 0:
                if print_flag:
                    print(compound.iupac_name)
                i+=1
            sub_smiles.append(compound.smiles)
            related_string += f'Name: {compound.iupac_name}\n'
            related_string += f'SMILES: {compound.smiles}\n'
            related_string += f'Molecular Weight: {compound.molecular_weight}\n'
            related_string += f'LogP: {compound.xlogp}\n'
            related_string += '===================\n'

        sub_mols = [Chem.MolFromSmiles(smile) for smile in sub_smiles]
        legend = [str(compound.smiles) for compound in res]

        total_similar_list.append(sub_smiles)
        img = Draw.MolsToGridImage(sub_mols, legends=legend, molsPerRow=4, subImgSize=(250, 250))
        # Save image to chat location
        if not os.path.exists('../images'):
          os.makedirs('../images')
        img.save('../images/chat_image.png')
        #pic = img.data
        all_images.append(img)
    except:
        related_string += f'{smiles}: Fail\n'
        total_similar_list.append([])
        all_images.append(None)

  return total_similar_list, related_string, all_images

def structure_node(smiles_list: list[str]) -> (list[str], str, list):
  '''
    Generates the 3D structure of the molecule based on the smiles string.
      Args:
        smiles: the input smiles string
      Returns:
        all_structures: a list of strings of the 3D structure of the molecule
        output_string: a string of the chemical formulae.
        all_images: a list of images of the 3D structure of the molecule
  '''
  print("structure tool")

  all_mols = []
  all_structures = []
  output_string = ''

  for smile in smiles_list:
    mol = Chem.MolFromSmiles(smile)
    molH = Chem.AddHs(mol)
    AllChem.EmbedMolecule(molH)
    AllChem.MMFFOptimizeMolecule(molH)

    structure_string = ""
    all_symbols = []
    for atom in molH.GetAtoms():
      symbol = atom.GetSymbol()
      all_symbols.append(symbol)
      pos = molH.GetConformer().GetAtomPosition(atom.GetIdx())
      structure_string += f'{symbol}  {pos[0]}  {pos[1]}  {pos[2]}\n'
      
    atom_freqs = Counter(all_symbols)
    formula = ''.join([f'{atom}{count}' for atom, count in atom_freqs.items()]) 

    output_string += f'For {smile}: Formula is: {formula}\n'
    all_structures.append(structure_string)
    all_mols.append(molH)
  
  img = Draw.MolsToGridImage(all_mols, molsPerRow=3, subImgSize=(250, 250))
  # Save image to chat location
  if not os.path.exists('../images'):
    os.makedirs('../images')
  img.save('../images/chat_image.png')

  return all_structures, output_string, [img]