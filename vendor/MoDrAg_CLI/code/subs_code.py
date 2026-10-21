from rdkit import Chem
from rdkit.Chem import QED
import re, random

print_flag = False

free_carbon_search_patterns = ['c[0-9]c', r'1c\[n', 'cc', r'c\[nH\]', r'CC']
free_carbon_insert_points = [2, 2, 1, 1, 1]

e_withdraw = [
    'I',              #iodo
    'Br',             #bromo 
    'Cl',             #chloro
    'F',              #fluoro
    'C(Cl)(Cl)(Cl)',  #trichloromethyl
    'C(F)(F)(F)',     #trifluoromethyl
    'C(=O)[O-]',      #carboxylate
    'C(=O)',          #carbonyl
    'C#N',            #nitrile
    '[N+](=O)[O-]',   #nitro
    '[NH3+]']         #ammonium

e_donate = [
    "[O-]",          # Phenoxide (strongest resonance donor)
    "N(C)C",         # Dimethylamino (-NMe2)
    "NC",            # Methylamino (-NHMe)
    "N",             # Amino (-NH2)
    "O",             # Hydroxy (-OH)
    "OC",            # Methoxy (-OMe)
    "NC(=O)C",       # Acetamido (-NHCOCH3)
    "SC",            # Methylthio (-SMe)
    "OC(=O)C",       # Acetoxy (-OCOCH3)
    "C(C)(C)C",      # tert-Butyl (-C(CH3)3)
    "C(C)C",         # Isopropyl (-CHMe2)
    "CC",            # Ethyl (-Et)
    "C",             # Methyl (-Me)
    "c5ccccc5",      # Phenyl (-Ph)
    "C=C"           # Vinyl (-CH=CH2)
    #"[Si](C)(C)C"    # Trimethylsilyl (-SiMe3)
]

linkers = [
    "C",             # Methylene (-CH2-)
    "CC",            # Ethylene (-CH2CH2-)
    "CCC",           # Propylene (-CH2CH2CH2-)
    "C=C",           # Vinylene (-CH=CH-)
    "C#C",           # Acetylene (-C≡C-)
    "CC=C",          # Allylene (-CH2CH=CH-)
    "C=CC",          # Propenylene (-CH=CHCH2-)
    "O",             # Oxygen (-O-)
    "S",             # Sulfur (-S-)
    "N",             # Nitrogen (-NH-)
    "C(=O)",          # Carbonyl (-C(=O)-)
    "C(=O)O",        # Ester (-C(=O)O-)
    "C(=O)N",        # Amide (-C(=O)N-)
    "C(=O)C"       # Ketone (-C(=O)C-)
]

withdraw_with_linkers = [f'{linker}({e})' for linker in linkers for e in e_withdraw] 
donate_with_linkers = [f'{linker}({e})' for linker in linkers for e in e_donate]

def make_random_list(num_items: int) -> tuple[list, list]:
    '''
    selects num_items from the lists e_withdraw, e_donate, withdraw_with_linkers, and 
    donate_with_linkers and returns them as a single list along with the remaining items.
    Args:
      num_items: the number of items to select from the lists

    Returns:
      tuple: (selected_items: list, remaining_items: list)
        - selected_items: a list of num_items items randomly selected from the lists
        - remaining_items: a list of items that were not selected
    '''
    # Combine all substituent lists
    all_substituents = e_withdraw + e_donate + withdraw_with_linkers + donate_with_linkers
    
    # Randomly select num_items (without replacement if num_items <= total available)
    if num_items >= len(all_substituents):
        return all_substituents, []
    else:
        selected = random.sample(all_substituents, num_items)
        remaining = [item for item in all_substituents if item not in selected]
        return selected, remaining


def grow_cycle(best_smiles: str = 'c1ccccc1', substituents: list[str] = e_withdraw):
    '''
    add substituents to free carbons in the molecule. 

    Args:
      best_smiles : the current best molecule, as a SMILES string
      substituents : a list of SMILES strings for substituents to add

    Returns:
      total_list : a list of tuples containing the new SMILES strings
    '''
    if print_flag:
        print('=============================================================================')
        print(f"Starting grow cycle with best score {best_smiles}.")
  
    total_list = []
    for pattern, insert_point in zip(free_carbon_search_patterns, free_carbon_insert_points):
        current_smiles = best_smiles
        for match in re.finditer(f'(?={pattern})', current_smiles):
            for e in substituents:
                new_smiles = f'{current_smiles[:match.start() + insert_point]}({e}){current_smiles[match.start() + insert_point:]}'
                mol = Chem.MolFromSmiles(new_smiles)
                if mol != None:
                    #print(new_smiles)
                    try:
                        total_list.append(new_smiles)
                    except:
                        if print_flag:
                            print(f"Error substituting {new_smiles}")
                    #print(f"{new_smiles}")
                else:
                    if print_flag:
                        print(new_smiles, 'bad')
    if print_flag:
        print('=============================================================================')
    return total_list

def replace_groups(orig_smiles: str = 'c1ccccc1', new_substituents: list[str] = e_donate):
    '''
    replace existing substituents in the molecule with new ones. 

    Args:
      orig_smiles: the current best molecule, as a SMILES string
      new_substituents: a list of SMILES strings for substituents to add

    Returns:
      total_list: a list of tuples containing the new SMILES strings
    '''
    if print_flag:
        print('=============================================================================')
        print(f"Starting replace cycle with {orig_smiles}.")
    best_smiles = orig_smiles
    #look in best_smiles for substituents in the substituents_to_replace list and replace them with substituents in the new_substituents list
    # create substituents_to_replace list by looking for c(\D\D*)c in best_smiles and extracting the D\D* part
    substituents_to_replace = re.findall(r"c\(\D\D*\)c", best_smiles)
    # add to substituents to replace with the C(\D\D*)C motif
    substituents_to_replace += re.findall(r"C\(\D\D*\)C", best_smiles)
    if print_flag:
        print(f"Found substituents to replace: {substituents_to_replace}")
    
    total_list = []
    for old in substituents_to_replace:
        if old in orig_smiles:
            if print_flag:
                print(f"Found {old} in {orig_smiles}")
            for new in new_substituents:
                if 'c(' in old:
                    new = f'c({new})c'
                elif 'C(' in old:
                    new = f'C({new})C'
                new_smiles = orig_smiles.replace(old, new)
                mol = Chem.MolFromSmiles(new_smiles)
                if mol != None:
                    try:
                        total_list.append(new_smiles)
                    except:
                        if print_flag:
                            print(f"Error substituting {new_smiles}")
                    #print(f"{new_smiles}")
                else:
                    if print_flag:
                        print(new_smiles, 'bad')

    if print_flag:
        print('=============================================================================')
    return total_list

