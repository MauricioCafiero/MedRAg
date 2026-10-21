
import os, re
import numpy as np
import pandas as pd
import requests, json
import itertools
import lightgbm as lgb
from lightgbm import LGBMRegressor
import deepchem as dc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from finetune_gpt import *

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

def predict_node(smiles_list_in: list[str], chembl_id: str) -> (list[float],str):
  '''
    uses the current_bioactives.csv file from the get_bioactives node to fit the
    Light GBM model and predict the IC50 for the current smiles.
      Args:
        smiles_list: the SMILES strings of the molecules to predict
        chembl_id: the chembl ID to query
      Returns:
        preds: a list of predicted IC50 values for the input SMILES
        preds_string: a string containing the predicted IC50 values for the input SMILES
  '''
  print("Predict Tool")
  print('===================================================')

  # if f'{chembl_id}_bioactives.csv' does not exist, call the bioactives node
  if not os.path.exists(f'{chembl_id}_bioactives.csv'):
    _, _, _ = getbioactives_node([chembl_id])
  
  try:
    chembl_id = chembl_id.upper()
    df = pd.read_csv(f'{chembl_id}_bioactives.csv')
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
      y.append(np.log10(value))
      Xa.append(smile)

    mols = [Chem.MolFromSmiles(smile) for smile in Xa]
    print(f"Number of molecules: {len(mols)}")

    featurizer=dc.feat.RDKitDescriptors()
    featname="RDKitDescriptors"
    f = featurizer.featurize(mols)

    nan_indicies = np.isnan(f)
    bad_rows = []
    for i, row in enumerate(nan_indicies):
        for item in row:
            if item == True:
                if i not in bad_rows:
                    print(f"Row {i} has a NaN.")
                    bad_rows.append(i)

    print(f"Old dimensions are: {f.shape}.")

    for j,i in enumerate(bad_rows):
        k=i-j
        f = np.delete(f,k,axis=0)
        y = np.delete(y,k,axis=0)
        Xa = np.delete(Xa,k,axis=0)
        print(f"Deleting row {k} from arrays.")

    print(f"New dimensions are: {f.shape}")
    if f.shape[0] != len(y) or f.shape[0] != len(Xa):
      raise ValueError("Number of rows in X and y do not match.")

    X_train, X_test, y_train, y_test = train_test_split(f, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LGBMRegressor(metric='rmse', max_depth = 50, verbose = -1, num_leaves = 31,
                          feature_fraction = 0.8, min_data_in_leaf = 20)
    modelname = "LightGBM Regressor"
    model.fit(X_train, y_train)

    train_score = model.score(X_train,y_train)
    print(f"score for training set: {train_score:.3f}")

    valid_score = model.score(X_test, y_test)
    print(f"score for validation set: {valid_score:.3f}")
  except:
    return [], 'Model training failed, unable to predict.', None

  preds = []
  preds_string = ''

  for smiles in smiles_list_in:
    print(f"in predict node, smiles: {smiles}")
    try:
      for ion in ions_to_clean:
        smiles = smiles.replace(ion,"")
      test_mol = Chem.MolFromSmiles(smiles)
      test_feat = featurizer.featurize([test_mol])
      test_feat = scaler.transform(test_feat)
      prediction = model.predict(test_feat)
      test_ic50 = 10**(prediction[0])
      print(f"Predicted IC50 for {smiles}: {test_ic50}")
      preds_string += f"The predicted IC50 value for {smiles} is : {test_ic50:.3f} nM.\n"
      
      preds.append(test_ic50)
    except:
      preds.append(None)
      preds_string += f"The prediction for {smiles} failed.\n"

  preds_string += f"The Bioactive data was fitted with the LightGMB model, using RDKit descriptors. The training score \
was {train_score:.3f} and the testing score was {valid_score:.3f}. "

  return preds, preds_string, None

def gpt_node(chembl_id: str) -> (list[str], str, Image.Image):
  '''
    Uses a Chembl dataset, previously stored in a CSV file by the get_bioactives node, to
    to finetune a GPT model to generate novel molecules for the target protein.

    Args:
      chembl_id: the ChEMBL ID to query
    returns:
      smiles_list: a list of generated SMILES strings
      gpt_string: a string containing the results of the GPT finetuning and generation.
      img: an image containing the generated molecules.
  '''
  print("GPT node")
  print('===================================================')
  
  # if f'{chembl_id}_bioactives.csv' does not exist, call the bioactives node
  chembl_id = chembl_id.upper()
  if not os.path.exists(f'{chembl_id}_bioactives.csv'):
    _, _, _ = getbioactives_node([chembl_id])

  try:
    df = pd.read_csv(f'{chembl_id}_bioactives.csv')
    smiles_list, gpt_string, img = finetune_gpt(df, chembl_id) 

  except:
    gpt_string = ''
    smiles_list = []
    img = None

  return smiles_list, gpt_string, [img]





