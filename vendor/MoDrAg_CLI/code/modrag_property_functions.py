from rdkit import Chem
from rdkit.Chem import AllChem, QED
from rdkit.Chem import Draw
from rdkit.Chem import rdMolAlign
import os, re
import numpy as np
from rdkit import DataStructs
from rdkit.Chem.AllChem import GetMorganGenerator
from rdkit import RDConfig
from rdkit.Chem.FeatMaps import FeatMaps
from subs_code import *

fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef'))

fmParams = {}
for k in fdef.GetFeatureFamilies():
    fparams = FeatMaps.FeatMapParams()
    fmParams[k] = fparams

# Module-level print flag - set from modrag.py
print_flag = False

def substitution_node(smiles_list: list[str]) -> (list[str], str, list):
  '''
    A simple substitution routine that performs three types of subsitutions:
    1. placing a substituent on a free carbon on a ring
    2. replacing an existing substituent on a ring with a new one
    3. placing a substituent on a free carbon in an SP3 chain.
    Returns a list of novel molecules and their
    QED score (1 is most drug-like, 0 is least drug-like).

      Args:
        smiles: the input smiles string
      Returns:
        new_smiles_list: a list of novel molecules and their QED scores.
        new_smiles_string: a string of the tool results
  '''
  print("substitution tool")
  print('===================================================')

  total_sub_smiles_list = []
  total_sub_smiles_string = ''
  total_sub_images = []

  new_fragments, _ = make_random_list(10)  # Select 10 random fragments from the substituent lists
  for smiles in smiles_list:
    try:
        new_smiles = []
        temp_smiles = grow_cycle(best_smiles=smiles, substituents=new_fragments)
        new_smiles.extend(temp_smiles)
        temp_smiles = replace_groups(orig_smiles=smiles, new_substituents=new_fragments)
        new_smiles.extend(temp_smiles)

        qeds = []
        for new_smile in new_smiles:
            qeds.append(get_qed(new_smile))
        original_qed = get_qed(smiles)

        total_sub_smiles_string += "Substitution tool results: \n"
        total_sub_smiles_string += f"The original molecule SMILES was {smiles} with QED {original_qed}.\n"
        total_sub_smiles_string += "Novel Molecules or Analogues and QED values: \n"
        for i in range(len(new_smiles)):
            total_sub_smiles_string += f"SMILES: {new_smiles[i]}, QED: {qeds[i]:.3f}\n"
        total_sub_smiles_list.append(new_smiles)

        mols = [Chem.MolFromSmiles(smile) for smile in new_smiles]
        img = Draw.MolsToGridImage(mols,legends=new_smiles, molsPerRow=4, subImgSize=(250, 250))
        # Save image to chat location
        if not os.path.exists('../images'):
          os.makedirs('../images')
        img.save('../images/chat_image.png')
        total_sub_images.append(img)
    except:
        total_sub_smiles_list.append([])
        total_sub_smiles_string += f"SMILES: {smiles}, Fail\n"
        total_sub_images.append(None)

  return total_sub_smiles_list, total_sub_smiles_string, total_sub_images

def get_qed(smiles):
  '''
    Helper function to compute QED for a given molecule.
      Args:
        smiles: the input smiles string
      Returns:
        qed: the QED score of the molecule.
  '''
  mol = Chem.MolFromSmiles(smiles)
  qed = Chem.QED.default(mol)

  return qed

def lipinski_node(smiles_list: list[str]) -> (list[float], str):
  '''
    A tool to calculate QED and other lipinski properties of a molecule.
      Args:
        smiles: the input smiles string
      Returns:
        total_lipinski_list: a list of the QED and other lipinski properties of the molecules,
                      including Molecular Weight, LogP, HBA, HBD, Polar Surface Area,
                      Rotatable Bonds, Aromatic Rings and Undesireable Moieties.
        total_lipinski_string: a string of the tool results
  '''
  print("lipinski tool")
  print('===================================================')

  total_lipinski_list = []
  total_lipinski_string = ''

  for smiles in smiles_list:
    for ion in ['.[Na+]', '.[K+]', '.[Cl-]', '.[Br-]', '[Na+].', '[K+].', '[Cl-].', '[Br-].']:
        smiles = smiles.replace(ion, '')
    lipinski_list = []
    try:
        mol = Chem.MolFromSmiles(smiles)
        qed = Chem.QED.default(mol)

        p = Chem.QED.properties(mol)
        mw = p[0]
        logP = p[1]
        hba = p[2]
        hbd = p[3]
        psa = p[4]
        rb = p[5]
        ar = p[6]
        um = p[7]

        lipinski_list.append(qed)
        lipinski_list.append(mw)
        lipinski_list.append(logP)
        lipinski_list.append(hba)
        lipinski_list.append(hbd)
        lipinski_list.append(psa)
        lipinski_list.append(rb)
        lipinski_list.append(ar)
        lipinski_list.append(um)

        total_lipinski_string += f"Properties of SMILES: {smiles}: QED: {qed:.3f}\n"
        total_lipinski_string += f"Molecular Weight: {mw:.3f}, LogP: {logP:.3f}\n"
        total_lipinski_string += f"Hydrogen bond acceptors: {hba}, Hydrogen bond donors: {hbd}\n"
        total_lipinski_string += f"Polar Surface Area: {psa:.3f}, Rotatable Bonds: {rb}\n"
        total_lipinski_string += f"Aromatic Rings: {ar}, Undesireable moieties: {um}\n"
        total_lipinski_string += "===================================================\n"
        total_lipinski_list.append(lipinski_list)
    except:
        total_lipinski_list.append([])
        total_lipinski_string += f"SMILES: {smiles}, Could not get properties\n"

  return total_lipinski_list, total_lipinski_string, None

def pharmfeature_node(known_smiles: str, test_smiles: list[str]) -> (list[float], str):
  '''
    A tool to compare the pharmacophore features of a query molecule against
    a those of a reference molecule and report the pharmacophore features of both and the feature
    score of the query molecule.

      Args:
        known_smiles: the reference smiles string
        test_smiles: the query smiles string
      Returns:
        total_pharmfeature_scores: a list of the pharmacophore feature scores of the query molecules.
        total_pharmfeature_string: a string of the tool results
  '''
  print("pharmfeature tool")
  print('===================================================')

  keep = ('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 'ZnBinder', 'Aromatic', 'LumpedHydrophobe')
  feat_hash = {'Donor': 'Hydrogen bond donors', 'Acceptor': 'Hydrogen bond acceptors',
               'NegIonizable': 'Negatively ionizable groups', 'PosIonizable': 'Positively ionizable groups',
               'ZnBinder': 'Zinc Binders', 'Aromatic': 'Aromatic rings', 'LumpedHydrophobe': 'Hydrophobic/non-polar groups' }


  smiles = [known_smiles, *test_smiles]
  mols = [Chem.MolFromSmiles(x) for x in smiles]

  mols = [Chem.AddHs(m) for m in mols]
  ps = AllChem.ETKDGv3()

  for m in mols:
      AllChem.EmbedMolecule(m,ps)

  total_pharmfeature_scores = []
  total_pharmfeature_string = ''

  #i = 1
  for i in range(1, len(mols)):
    o3d = rdMolAlign.GetO3A(mols[i],mols[0])
    o3d.Align()

    feat_vectors = []
    for m in [mols[0], mols[i]]:
        rawFeats = fdef.GetFeaturesForMol(m)
        feat_vectors.append([f for f in rawFeats if f.GetFamily() in keep])

    feat_maps = [FeatMaps.FeatMap(feats = x,weights=[1]*len(x),params=fmParams) for x in feat_vectors]
    test_score = feat_maps[0].ScoreFeats(feat_maps[1].GetFeatures())/min(feat_maps[1].GetNumFeatures(),feat_maps[0].GetNumFeatures())

    feats_known = {}
    feats_test = {}
    for feat in feat_vectors[0]:
        if feat.GetFamily() not in feats_known.keys():
            feats_known[feat.GetFamily()]  = 1
        else:
            feats_known[feat.GetFamily()] += 1

    for feat in feat_vectors[1]:
        if feat.GetFamily() not in feats_test.keys():
            feats_test[feat.GetFamily()]  = 1
        else:
            feats_test[feat.GetFamily()] += 1

    total_pharmfeature_string += f"PharmFeature tool results for SMILES: {smiles[i]}: \n"
    total_pharmfeature_string += f"The Pharmacophore Feature Overlap Score of the test molecule \
versus the reference molecule is {test_score:.3f}. \n\n"
    total_pharmfeature_scores.append(test_score)

    for feat in feats_known.keys():
        total_pharmfeature_string += f"There are {feats_known[feat]} {feat_hash[feat]} in the reference molecule. \n"

    for feat in feats_test.keys():
        total_pharmfeature_string += f"There are {feats_test[feat]} {feat_hash[feat]} in the test molecule. \n"
    #i += 1
    total_pharmfeature_string += "===================================================\n"

  return total_pharmfeature_scores, total_pharmfeature_string, None

def similarity_node(ref_mols: list[str], smiles_list: list[str], sim_cutoff: float = 0.15) -> None:
    '''
    A tool to calculate the similarity between reference molecules and test molecules based on 
    Morgan fingerprints and Tanimoto similarity. It generates a grid image of similar molecules and saves it to the images folder.
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

    sim_out = 'Calculating Tanimoto similarity of molecules using Morgan fingerprints...\n'

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
        if print_flag:
            print(f"There are {count_sim} out of {ref_dim*sim_dim} molecule pairs with similarity greater than {sim_cutoff} with an average of {ave_sim:.2f}")
    except:
        if print_flag:
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
