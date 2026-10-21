import torch
import pandas as pd
import numpy as np
import transformers
from transformers import AutoTokenizer
from transformers import pipeline
import random
import re
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import py3Dmol

sub_locations_re = ["cc",                          #first unsubstituted carbons encountered
                "c[1-9]cc",                        #unsubstituted carbon 2 of ring
                "ccc[1-9]",                        #unsubstituted carbon 4 of ring
                "c[1-9]c(\([A-Z]+\))?c",           #carbon 2 of ring
                "c[1-9]cc(\([A-Z]+\))?c",          #carbon 3 of ring
                "c[1-9]ccc(\([A-Z]+\))?c",         #carbon 4 of ring
                "c[1-9]cccc(\([A-Z]+\))?c",        #carbon 5 of ring
                "c[1-9]ccccc(\([A-Z]+\))?"]        #carbon 6 of ring

sub_location_names = ["any unsubbed carbon","unsubbed carbon at C2", "unsubbed carbon at C4",
                      "substituent on C2","substituent on C3","substituent on C4","substituent on C5","substituent on C6"]

possible_sub_points = ["cc","c(O)c","c(OC)c"]

new_fragments = ["c(F)c","c(C#N)c","c(I)c","c([N+]([O-])=O)c","c(OC)c","c(Cl)c"]

new_fragment_names = ["Fluoro","Cyano","Iodo","Nitro","Methoxy","Chloro"]

def add_fragment(frag_in: str, name_in: str):
  '''
  accepts a SMILES representation of a new fragment for substitution and a name for the 
  fragment, appends them to the list used in the substitution routine.

    Args:
      frag_in: a SMILES representation of a new fragment for substitution, must be
      of the form: c(substituent)c
      name_in: a name for the fragment
  '''
  global new_fragments
  global new_fragment_names
  new_fragments.append(frag_in)
  new_fragment_names.append(name_in)
  print("Fragment added:")
  print(f"Name: {name_in}     Fragment: {frag_in}")

def sub_fragment(frag_in: str, name_in: str):
  '''
  accepts a SMILES representation of a fragment to remove from the substitution 
  list as well as the corresponding a name to remove. 
  
    Args:
      frag_in: a SMILES representation of a fragment to remove.
      name_in: a name for the fragment to remove
  '''
  global new_fragments
  global new_fragment_names
  new_fragments.remove(frag_in)
  new_fragment_names.remove(name_in)
  print("Fragment removed:")
  print(f"Name: {name_in}     Fragment: {frag_in}") 

def which_fragments():
  '''
  prints the current list of fragments and names for substitution
  '''
  global new_fragments
  global new_fragment_names
  for frag,name in zip(new_fragments,new_fragment_names):
    print(f"Name: {name}     Fragment: {frag}")

def make_sub_string(match):
  '''
  accepts a match object and checks for the existence of a match with the possible 
  substitution point. If a match is found, creates and returns the substitution.

    Args:
      match: a regex object
    
    Returns:
      new_frag: the substituted string, or the original string if the substitution failed
  '''
  global could_not_match
  global sub_point_stored
  global new_fragment_stored
  
  original_frag = match.group()

  if sub_point_stored in original_frag:
      new_frag = original_frag.replace(sub_point_stored,new_fragment_stored)
      return new_frag
  else:
      could_not_match += 1     #make a list of what we can't match?

      return match.group()

def hold_values(sub_point,new_fragment):
  '''
  stores the subsitutiton points and new fragments in global variables to
  be used by the make_sub_string function 
  '''
  global sub_point_stored
  global new_fragment_stored
  sub_point_stored = sub_point 
  new_fragment_stored = new_fragment

def sub_rings(smile_in: str, number_subs: int) -> str:
  '''
  accepts a SMILES string and tries all posible substitutions indicated by the 
  possible_sub_points list and the new_fragments list. Specific cases of the 
  possible_sub_points list are found in the sub_locations_re list as regex. The 
  lists have corresponding name lists.

    Args:
      smile_in: a SMILES string
      number_subs: the number of substitutions to make per molecule
    
    Returns:
      new_smiles: a list of all the generated molecules.
      qeds: a list of the QED value for each molecule
      mols: a list of the mol objects for each molecule.
      new_legends: a list of the substitution made for each molecule.
      img: an image of the molecules with legends.
  '''
  new_smiles = []
  new_legends = []
  global could_not_match
  could_not_match = 0 
  
  for sub_point in possible_sub_points:
    if sub_point == "cc":
      sub_locations = sub_locations_re[:3]
      sub_names = sub_location_names[:3]
    else:
      sub_locations = sub_locations_re[3:]
      sub_names = sub_location_names[3:]
    for specific_frag, frag_name in zip(sub_locations,sub_names):
      for new_fragment in new_fragments:

          res = re.search("c[1-9]c(\([A-Z]+\))?c(\([A-Z]+\))?c(\([A-Z]+\))?c(\([A-Z]+\))?c[1-9]",smile_in)
          if res:
              if sub_point in res.group():
                  hold_values(sub_point,new_fragment)
                  new_mol = re.sub(specific_frag,make_sub_string,smile_in,number_subs)
                  if new_mol != smile_in and new_mol not in new_smiles:
                      new_smiles.append(new_mol)
                      substituent = new_fragment.strip("c()")
                      new_legends.append(f"{frag_name} substitution with {substituent}.")
                  
  print(f"Substituted {len(new_smiles)} molecules")
  
  qeds,mols = calc_qed(new_smiles)
  legends = [f"QED: {qed:.3f}\n"+legend for qed,legend in zip(qeds, new_legends)]

  print(f"Could not match {could_not_match} requests.")
 
  img = Draw.MolsToGridImage(mols, legends=legends, molsPerRow=3, subImgSize=(200,200),useSVG=False,returnPNG=False)

  return new_smiles, qeds, mols, new_legends, img
  
def genmask_model_setup(model_size: int, model_name: str):
  '''
    Accepts a model name and loads the tokenizer and a pipeline for the model
    Args:
        model_size: deprecated, will be removed, can use any integer
        model_name: the name of the HF model, in the form user/model
  '''
  global device
  global tokenizer
  global mask_filler
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tokenizer = AutoTokenizer.from_pretrained(model_name,padding = True, truncation = True)
  mask_filler = pipeline("fill-mask", model_name)
  
def gen_from_multimask(text, print_flag=True, mask_flag="random", percent = 0.10, top_k = 3):
  """
  Takes a SMILES string and tokenizes it. Depending on the mask flag, it then masks the
  requested percentage of tokens in the string either randomly, at the begining (first) or at
  the end (last). The masked string is then sent to the mask filler in a single pass, and the
  masked positions are unmasked most-confident-first (highest top-1 probability first, as in
  the ESM protein unmasking code), expanding into all possible new strings where the top k
  beams at each position are kept if their probability is greater than 0.1. Entropy is also
  calculated for each beam.

    Args:
        text: The SMILES string of the original molecule.

    Returns:
        final_smiles: a list of all the generated molecules.
        total_entropy: a list of the entropy of each generated molecule.
  """
  new_tok_list = []
  single_tok = tokenizer(text, padding=True, truncation=True, max_length=250, return_special_tokens_mask=True)
  length_count = 0
  for token in single_tok["input_ids"]:
    if token != 0:
      length_count += 1

  if mask_flag == "last":
    masked_tokens = [*range(int(length_count*(1.0-percent))-1,length_count-1)]
  elif mask_flag == "first":
    masked_tokens = [*range(0,int(length_count*percent))]
  elif mask_flag == "random":
    masked_tokens = random.sample(range(1, length_count), int(length_count*percent))

  for j,token in enumerate(single_tok["input_ids"]):
    if token != 0:
      if j in masked_tokens:
        new_tok_list.append(103)
      else:
        new_tok_list.append(token)
  masked_smile = tokenizer.decode(new_tok_list,
                    skip_special_tokens=False).replace("[PAD]","").replace("[SEP]","").replace("[CLS]","").replace(" ","")
  result = mask_filler(masked_smile,top_k=top_k)

  # The pipeline reports one sublist per [MASK] occurrence, in left-to-right order, except when
  # there is exactly one mask, where it returns a single flat list instead. Track the position
  # of each mask within the token list so a candidate can be dropped into the right slot
  # regardless of the order in which slots get filled.
  pieces = tokenizer.convert_ids_to_tokens(new_tok_list)
  mask_slots = [idx for idx,tok_id in enumerate(new_tok_list) if tok_id == 103]
  if len(mask_slots) == 1:
    result = [result]

  # Most-likely-first: rank mask slots by their top-1 confidence, highest first, same idea as
  # the confidence-ordered unmasking used for the protein sequences.
  top1_scores = [result[i][0]["score"] for i in range(len(result))]
  fill_order = sorted(range(len(result)), key=lambda i: top1_scores[i], reverse=True)

  beams = [{"pieces": pieces, "entropy": 0.0}]
  for mask_i in fill_order:
    new_beams = []
    slot = mask_slots[mask_i]
    for beam in beams:
      for j in range(top_k):
        p = result[mask_i][j]["score"]
        if p > 0.1:
          new_pieces = beam["pieces"].copy()
          new_pieces[slot] = result[mask_i][j]["token_str"]
          new_beams.append({"pieces": new_pieces, "entropy": beam["entropy"] - p*np.log(p)})
    beams = new_beams

  special_strs = {tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token}
  final_smiles = []
  total_entropy = []
  for beam in beams:
    new_smile = "".join(p for p in beam["pieces"] if p not in special_strs).replace("##","")
    final_smiles.append(new_smile)
    total_entropy.append(beam["entropy"])

  if print_flag:
    print(f"original:    {text}")
    final_smiles.insert(0,text)
    for smile in final_smiles:
      print(f"generated:   {smile}")

  return final_smiles,total_entropy

def validate_smiles(in_smiles, in_entropy):
  """
  Takes a list of SMILES strings checks to see if the compile to valid MOL objects.
  Valid molecules are then converted to canonical SMILES strings and duplicates are
  dropped.

    Args:
        text: The SMILES string of the original molecule.

    Returns:
        unique_smiles: a list of all the unique, valid generated molecules.
        unique_entropies: a list of the entropy of each generated molecule.
  """
  valid_smiles = []
  valid_entropies = []
  unique_smiles = []
  unique_entropies = []

  for smile,entropy in zip(in_smiles,in_entropy):
    try:
      mol = Chem.MolFromSmiles(smile)
      if mol is not None:
        valid_smiles.append(smile)
        valid_entropies.append(entropy)
    except:
      print("Could not convert to mol")

  canon_smiles = [Chem.CanonSmiles(smile) for smile in valid_smiles]

  for smile,entropy in zip(canon_smiles,valid_entropies):
    if smile not in unique_smiles:
      unique_smiles.append(smile)
      unique_entropies.append(entropy)
  try:
    print(f"Total unique SMILES generated: {len(unique_smiles)}")
    print(f"Average entropy: {sum(unique_entropies)/len(unique_entropies)}")
  except:
    print("Could not calculate QED as no molecule generated!")
    
  return unique_smiles,unique_entropies

def calc_qed(smiles, lip_props = False):
  '''
  Takes a list of SMILES strings and calculates the QED value for each molecule.
  A value of 1.0 is perfect drug-likeness, and a value of 0.0 is not drug-like.

    Args: 
      smiles: a list of SMILES strings
      lip_props: Whether or not to calculate the lipinski properties

    Returns:
      qed: a list of the QED values of each generated molecule.
      mols: a list of the molecule objects corresponding to each generated molecule.
      If lip_props is True, then:
        Molecular_weight: a list of the molecular weights of each generated molecule.
        aLogP: a list of the logP values of each generated molecule.
        Hydrogen_bond_acceptors: a list of the number of hydrogen bond acceptors of each generated molecule.
        Hydrogen_bond_donors: a list of the number of hydrogen bond donors of each generated molecule.
        Polar_surface_area: a list of the polar surface area of each generated molecule.
        Rotatable_Bonds: a list of the number of rotatable bonds of each generated molecule.
        Aromatic_Rings: a list of the number of aromatic rings of each generated molecule.
        Undesireable_moieties: a list of the number of undesireable moieties of each generated molecule.
  '''
  mols = [Chem.MolFromSmiles(smile) for smile in smiles]
  qed = [Chem.QED.default(mol) for mol in mols]
  
  
  if lip_props:
    Molecular_weight = []
    aLogP = []
    Hydrogen_bond_acceptors = []
    Hydrogen_bond_donors = []
    Polar_surface_area = []
    Rotatable_Bonds = []
    Aromatic_Rings = []
    Undesireable_moieties = []
    for mol in mols:
      p = Chem.QED.properties(mol)
      Molecular_weight.append(p[0])
      aLogP.append(p[1])
      Hydrogen_bond_acceptors.append(p[2])
      Hydrogen_bond_donors.append(p[3])
      Polar_surface_area.append(p[4])
      Rotatable_Bonds.append(p[5])
      Aromatic_Rings.append(p[6])
      Undesireable_moieties.append(p[7])
    
    return qed, mols, Molecular_weight, aLogP, Hydrogen_bond_acceptors, Hydrogen_bond_donors,\
           Polar_surface_area, Rotatable_Bonds, Aromatic_Rings, Undesireable_moieties
  
  return qed,mols

def gen_mask(smile_in: str, percent_masked: float) -> str:
  """
  The molecule corresponding to the input smiles is masked in different,
  random ways, creating various masked versions of the molelcule.
  A model, cafierom/bert-base-cased-ChemTok-ZN250K-V1,
  is used to generate SMILES strings for analogue molecules by unmasking the
  masked versions. All possibilities created by the generative mask-filling
  are kept as long as the probability is greater than a cut-off, which is set
  to 0.1 but which may be changed.The QED value, or quantitative estimate of druglikeness, a weighted average of
  various ADME properties is also calculated. A value of 1.0 is perfect
  drug-likeness, and a value of 0.0 is not drug-like.

    Args:
        smile: The SMILES string of the original molecule.
        percent_masked: The percentage of tokens to mask.

    Returns:
        final_smiles: a list of all the generated molecules.

        total_entropy: a list of the entropy of each generated molecule.

        qeds: a list of the QED values of each generated molecule.

        mols: a list of the molecule objects corresponding to each generated molecule.

        out_text: a string with all of the SMILES for the generated molecules
        and their QED values.

        pic: An image of the molecules with QED values.
  """
  which_statins = [smile_in]
  percent_to_use = percent_masked
  try:
    main_smiles = []
    main_entropy = []
    for statin in which_statins:
      result, calc_entropy = gen_from_multimask(statin, print_flag=False, mask_flag = "first", percent=percent_to_use)
      for smile,entropy in zip(result,calc_entropy):
        if smile not in main_smiles:
          main_smiles.append(smile)
          main_entropy.append(entropy)
      length = len(main_smiles)
      print(f"First masking generated {length} SMILES")

      result, calc_entropy = gen_from_multimask(statin, print_flag=False, mask_flag = "last", percent=percent_to_use)
      for smile,entropy in zip(result,calc_entropy):
        if smile not in main_smiles:
          main_smiles.append(smile)
          main_entropy.append(entropy)
      print(f"Last masking generated {len(main_smiles)-length} SMILES")
      length = len(main_smiles)

      for _ in range(4):
        result, calc_entropy = gen_from_multimask(statin, print_flag=False, mask_flag = "random", percent=percent_to_use)
        for smile,entropy in zip(result,calc_entropy):
          if smile not in main_smiles:
            main_smiles.append(smile)
            main_entropy.append(entropy)
        print(f"Random masking generated {len(main_smiles)-length} SMILES")
        length = len(main_smiles)

    print(f"Total SMILES generated: {len(main_smiles)}")

    final_smiles,final_entropy = validate_smiles(main_smiles,main_entropy)
    qeds,mols = calc_qed(final_smiles)

    out_text = f"Total SMILES generated for hit: {len(final_smiles)}\n"
    out_text += "===================================================\n"
    i = 1
    for smile, qed in zip(final_smiles,qeds):
      out_text += f"analogue {i}: {smile} with QED: {qed:.3f}\n"
      i += 1

    legends = [f"QED = {qed:.3f}" for qed in qeds]

    img = Draw.MolsToGridImage(mols, legends=legends, molsPerRow=3, subImgSize=(200,200),useSVG=False,returnPNG=False)

  except:
    final_smiles = []
    final_entropy = []
    qeds = []
    mols = []
    out_text = "Invalid SMILES string"
    img = None
  return final_smiles, final_entropy, qeds, mols, out_text, img  

def smileToXYZ(smile: str, filename: str) -> str:
  """
     Add H's to a SMILES string, make it 3D, optimize with MMFF from RDKit
     and return an XYZ string and file.
    
    Args:
        smile: The SMILES string of the molecule
        filename: The name of the file to save the XYZ string.
        
    Returns:
        The XYZ structure string; also saves a file.
  """

  mol = Chem.MolFromSmiles(smile)
  molH = Chem.AddHs(mol)
  AllChem.EmbedMolecule(molH)
  AllChem.MMFFOptimizeMolecule(molH)

  num_atoms = molH.GetNumAtoms()
  xyz_string = f"{num_atoms}\n\n"

  for atom in molH.GetAtoms():
    atom_sym = atom.GetSymbol()
    pos = molH.GetConformer().GetAtomPosition(atom.GetIdx())
    xyz_string += f"{atom_sym} {pos.x} {pos.y} {pos.z}\n"

  with open(filename, 'w') as f:
    f.write(xyz_string)
  print(f"XYZ file saved as {filename}")

  return xyz_string

def visualize_molecule(xyz_string: str):
  '''
    input an XYZ string to vosualize the molecule in 3D
  '''
  viewer = py3Dmol.view(width=800, height=400)
  viewer.addModel(xyz_string, "xyz")  
  viewer.setStyle({"stick": {}, "sphere": {"radius": 0.5}})
  viewer.zoomTo()
  viewer.show()

def calculate_similarities(known: str, to_compare: list, mm_cutoff: float):
    '''
    calculates the Tanimoto similarity between a known and a list of molecules
    using circular fingerprints.

        Args:
            known: SMILES for the known
            to_compare: list of SMILES to compare
        Returns:
            sim_array: an array of similarity values between the molecules in the list
            known_sim: a list of similarities between the list and the known
    '''
    
    mols = [Chem.MolFromSmiles(smile) for smile in to_compare]
    fp = [AllChem.GetMorganFingerprint(m,2) for m in mols]
    
    known_mol = Chem.MolFromSmiles(known)
    known_fp = AllChem.GetMorganFingerprint(known_mol,2)
    
    sim_dim = len(to_compare)
    sim_array = np.zeros((sim_dim,sim_dim))

    for i,loop_fp in enumerate(fp):
        sim_array[i][i+1:] = DataStructs.BulkTanimotoSimilarity(loop_fp,fp[i+1:]) 
        
    for i in range(sim_dim):
        for j in range(i,sim_dim,1):
            if sim_array[i][j] > mm_cutoff:
                print(f"Molecules {i} and {j} have a similarity of {sim_array[i][j]:.2f}.")
    
    print("========================================================")
    
    known_array = np.zeros((1,sim_dim))
    known_array[0,:] = DataStructs.BulkTanimotoSimilarity(known_fp,fp)
    for i in range(sim_dim):
        if known_array[0,i] > mm_cutoff:
            print(f"Molecule {i} and the known have a similarity of {known_array[0,i]:.2f}.")

    return sim_array, known_array