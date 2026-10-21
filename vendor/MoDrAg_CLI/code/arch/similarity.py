import numpy as np, os
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.AllChem import GetMorganGenerator


def similarity_node(ref_mols: list[str], smiles_list: list[str], sim_cutoff: float = 0.15) -> None:
    '''
    A tool to calculate the similarity of molecules in a dataframe based on their SMILES strings.
      Args:
        ref_mols: a list of SMILES strings of reference molecules to compare against; this is typically 
                  only one molecule, but can be a list of molecules to compare against.
        smiles_list: a list of SMILES strings of the molecules to be compared
        sim_cutoff: the minimum similarity value to be considered similar (default is 0.15)
      Returns:
        []: an empty list (this function does not return any data, but saves an image of the similar molecules to the images folder)
        sim_out: a string of the tool results
        all_images: a list of images of the similar molecules
    '''
    print("similarity tool")
    print('===================================================')

    sim_out = 'Calculating similarity of molecules...\n'
    
    # Create and filter reference molecules
    ref_list = []
    for i, smile in enumerate(ref_mols):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            sim_out += f"Reference molecule {i} ({smile}) could not be parsed.\n"
        else:
            ref_list.append(mol)
    
    # Create and filter test molecules
    mols_list = []
    for i, smile in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            sim_out += f"Test molecule {i} ({smile}) could not be parsed.\n"
        else:
            mols_list.append(mol)
    
    # Check if we have valid molecules
    if not ref_list or not mols_list:
        sim_out += "No valid molecules to compare.\n"
        return [], sim_out, []
    
    gen = GetMorganGenerator(radius=2)
    ref_fingerprints = [gen.GetFingerprint(mol) for mol in ref_list]
    fingerprints = [gen.GetFingerprint(mol) for mol in mols_list]

    ref_dim = len(ref_fingerprints)
    sim_dim = len(fingerprints)
    sim_array = np.zeros((ref_dim, sim_dim))
    count_sim = 0
    ave_sim = 0.0
    sim_mols_show = []
    sim_val_show = []
    

    for i,ref_fp in enumerate(ref_fingerprints):
        sim_array[i][:] = DataStructs.BulkTanimotoSimilarity(ref_fp,fingerprints) 

    for i in range(ref_dim):
        for j in range(sim_dim):
            if sim_array[i][j] > sim_cutoff:
                sim_out += f"Reference molecule {i} and test molecule {j} have a similarity of {sim_array[i][j]:.2f}.\n"
                count_sim += 1
                ave_sim += sim_array[i][j]
                sim_mols_show.append(ref_list[i])
                sim_mols_show.append(mols_list[j])
                sim_val_show.append(f"Reference {i} and Test {j}: similarity of {sim_array[i][j]:.2f}")
                sim_val_show.append(f"")
    try:
        ave_sim = ave_sim/count_sim
        print(f"There are {count_sim} out of {ref_dim*sim_dim} molecule pairs with similarity greater than {sim_cutoff} \
        with an average of {ave_sim:.2f}")
    except:
        print("No similarities found!")

    # save image to chat location
    all_images = []
    if sim_mols_show:
        if not os.path.exists('../images'):
            os.makedirs('../images')
        img = Draw.MolsToGridImage(sim_mols_show, legends=sim_val_show, molsPerRow=4, subImgSize=(250, 250))
        img.save('../images/chat_image.png')
        all_images = [img]

    return [], sim_out, all_images