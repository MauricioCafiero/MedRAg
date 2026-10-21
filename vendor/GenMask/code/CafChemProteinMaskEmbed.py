from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM, pipeline
import torch
import random
import re
import py3Dmol
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


def get_protein_from_pdb(pdb_id):
  '''
  Get protein structure from PDB
  Input: PDB ID
  Output: PDB file
  '''
  url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
  r = requests.get(url)
  return r.text

def show_protein(pdb_id):
  '''
  Show protein structure from PDB
  Input: PDB ID
  Output: structure view
  '''
  colors = ['turquoise', 'red', 'yelllow', 'blue', 'green', 'orange', 'purple']
  chains = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

  pdb_str = get_protein_from_pdb(pdb_id)
  pdbview = py3Dmol.view(width=800, height=600)
  pdbview.addModel(pdb_str, 'pdb')
  for i, chain in enumerate(chains):
    try:
      pdbview.setStyle({'chain': chain}, {'cartoon': {'color': colors[i]}})
    except:
      print('No additional chains')
  pdbview.zoomTo()
  return pdbview

def extract_sequence(pdb_id):
  '''
  Extract sequence from PDB file
  Input: PDB ID
  Output: dictionary of chains and sequences in three-letter and one-letter formats
  '''
  pdb_str = get_protein_from_pdb(pdb_id)
  chains = {}

  #print(pdb_str.split('\n')[0])
  for line in pdb_str.split('\n'):
    parts = line.split()
    try:
      if parts[0] == 'SEQRES':
        if parts[2] not in chains:
          chains[parts[2]] = []
        chains[parts[2]].extend(parts[4:])
    except:
      print('Blank line')

    chains_ol = {}
    for chain in chains:
      chains_ol[chain] = three_to_one(chains[chain])

  return chains, chains_ol

def one_to_three(one_seq):
  '''
  Convert one-letter code to three-letter code
  Input: one-letter code
  Output: three-letter code
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
  Convert three-letter code to one-letter code
  Input: three-letter code
  Output: one-letter code
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

def three_letter_seq_to_one(three_letter_seq: str) -> str:
  '''
  Convert a three-letter-code amino acid sequence into the single-letter sequence
  string used elsewhere in this module (gen_mask, gen_from_multimask, embeddings).
  Accepts codes separated by whitespace, hyphens, underscores, commas, or slashes
  (e.g. "Ala-Gly-Leu", "ALA GLY LEU"), or run together with no separator
  (e.g. "AlaGlyLeu") as long as the total length is a multiple of 3. Not wired into
  the CLIs -- convert to single-letter before calling gen_mask, and use
  one_letter_seq_to_three below to convert results back afterward.

    Args:
        three_letter_seq: three-letter-code sequence, case-insensitive.

    Returns:
        A single-letter sequence string. Unrecognized codes become 'X'.
  '''
  tokens = [t for t in re.split(r'[\s\-_,/]+', three_letter_seq.strip()) if t]
  if len(tokens) == 1 and len(tokens[0]) > 3 and len(tokens[0]) % 3 == 0:
    joined = tokens[0]
    tokens = [joined[i:i+3] for i in range(0, len(joined), 3)]
  tokens = [t.upper() for t in tokens]
  return ''.join(three_to_one(tokens))

def one_letter_seq_to_three(seq: str, sep: str = '-') -> str:
  '''
  Convert a single-letter amino acid sequence string back into three-letter-code
  form, joined by sep. Inverse of three_letter_seq_to_one -- use this to convert
  gen_mask's output sequences back to three-letter form if that's what the user
  brought in originally.

    Args:
        seq: single-letter sequence string.
        sep: separator to join the three-letter codes with (default '-').

    Returns:
        A three-letter-code sequence string, e.g. "ALA-GLY-LEU".
  '''
  return sep.join(one_to_three(residue) for residue in seq)

def genmask_model_setup(model_name: str):
  '''
  Accepts an HF checkpoint name (e.g. facebook/esm2_t12_35M_UR50D) and loads the
  tokenizer and a fill-mask pipeline for it, mirroring CafChemSubs.genmask_model_setup.

    Args:
        model_name: the name of the HF ESM checkpoint
  '''
  global device
  global tokenizer
  global mask_filler
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  mask_filler = pipeline("fill-mask", model_name)

def percent_identity(seq_a: str, seq_b: str) -> float:
  '''
  Fraction of matching residues at matching positions between two equal-length
  sequences (masking only substitutes residues, so lengths always match).
  '''
  matches = sum(1 for a, b in zip(seq_a, seq_b) if a == b)
  return matches / len(seq_a)

def gen_from_multimask(seq, print_flag=True, mask_flag="random", percent=0.10, top_k=3, prob_cutoff=0.05):
  """
  Takes a protein sequence and tokenizes it. Depending on the mask flag, it then masks
  the requested percentage of residues either randomly, at the beginning (first) or at
  the end (last). The masked sequence is then sent to the mask filler in a single pass,
  and the masked positions are unmasked most-confident-first (highest top-1 probability
  first), expanding into all possible new sequences where the top k beams at each
  position are kept if their probability is greater than prob_cutoff. Entropy is also
  calculated for each beam. This mirrors CafChemSubs.gen_from_multimask for SMILES.

  Note: ESM's per-position confidence is considerably flatter than the small custom
  BERT used for SMILES, especially once more than a handful of residues are masked in
  the same forward pass (less surrounding context per masked position) -- SMILES'
  0.1 cutoff regularly prunes every candidate at some position and collapses the whole
  beam to zero results, so the default here is lower (0.05). Beam width is also capped
  at 200 (keeping the lowest-entropy/most-confident beams) since masking a chunk of a
  protein sequence puts far more positions in play at once than masking a few SMILES
  tokens, and an uncapped cartesian product over top_k candidates can blow up.

    Args:
        seq: The amino acid sequence of the original protein (one-letter codes).

    Returns:
        final_seqs: a list of all the generated sequences.
        total_entropy: a list of the entropy of each generated sequence.
  """
  ids = tokenizer(seq)["input_ids"]
  length_count = len(ids)

  if mask_flag == "last":
    masked_positions = [*range(int(length_count*(1.0-percent))-1, length_count-1)]
  elif mask_flag == "first":
    masked_positions = [*range(1, 1+int(length_count*percent))]
  elif mask_flag == "random":
    masked_positions = random.sample(range(1, length_count-1), int(length_count*percent))

  new_ids = [tokenizer.mask_token_id if i in masked_positions else t for i,t in enumerate(ids)]
  masked_seq = tokenizer.decode(new_ids, skip_special_tokens=False)\
      .replace(tokenizer.cls_token,"").replace(tokenizer.eos_token,"").replace(tokenizer.pad_token,"").replace(" ","")

  result = mask_filler(masked_seq, top_k=top_k)

  pieces = tokenizer.convert_ids_to_tokens(new_ids)
  mask_slots = [idx for idx,tok_id in enumerate(new_ids) if tok_id == tokenizer.mask_token_id]
  if len(mask_slots) == 1:
    result = [result]

  top1_scores = [result[i][0]["score"] for i in range(len(result))]
  fill_order = sorted(range(len(result)), key=lambda i: top1_scores[i], reverse=True)

  # Masking a chunk of residues at once (unlike the 2-3 tokens typically masked for a
  # SMILES string) can put many positions in play simultaneously; an unpruned cartesian
  # product over top_k candidates per position could reach top_k**num_masks beams. Cap
  # the beam width after each position is folded in, keeping the lowest-entropy (most
  # confident) beams, so this stays tractable regardless of how many residues are masked.
  max_beams = 200

  beams = [{"pieces": pieces, "entropy": 0.0}]
  for mask_i in fill_order:
    new_beams = []
    slot = mask_slots[mask_i]
    for beam in beams:
      for j in range(top_k):
        p = result[mask_i][j]["score"]
        if p > prob_cutoff:
          new_pieces = beam["pieces"].copy()
          new_pieces[slot] = result[mask_i][j]["token_str"]
          new_beams.append({"pieces": new_pieces, "entropy": beam["entropy"] - p*np.log(p)})
    new_beams.sort(key=lambda b: b["entropy"])
    beams = new_beams[:max_beams]

  special_strs = {tokenizer.cls_token, tokenizer.eos_token, tokenizer.pad_token}
  final_seqs = []
  total_entropy = []
  for beam in beams:
    new_seq = "".join(p for p in beam["pieces"] if p not in special_strs)
    final_seqs.append(new_seq)
    total_entropy.append(beam["entropy"])

  if print_flag:
    print(f"original:    {seq}")
    final_seqs.insert(0,seq)
    for s in final_seqs:
      print(f"generated:   {s}")

  return final_seqs,total_entropy

def gen_mask(seq: str, percent_masked: float, prob_cutoff: float = 0.05) -> str:
  """
  The protein sequence corresponding to the input is masked in different, random ways,
  creating various masked versions of the sequence. An ESM masked-language model is
  used to generate new sequences by unmasking the masked versions, most-confident
  position first. All possibilities created by the generative mask-filling are kept as
  long as the probability is greater than prob_cutoff (default 0.05; lower than the
  SMILES default of 0.1 since ESM's per-position confidence runs flatter -- see
  gen_from_multimask). The percent sequence identity to the parent is also calculated
  for each generated sequence. This mirrors CafChemSubs.gen_mask for SMILES.

    Args:
        seq: The amino acid sequence of the original protein.
        percent_masked: The percentage of residues to mask.
        prob_cutoff: Minimum candidate probability to keep a beam (default 0.05).

    Returns:
        final_seqs: a list of all the generated sequences.
        final_entropy: a list of the entropy of each generated sequence.
        identities: a list of the percent identity to the parent for each generated sequence.
        out_text: a string with all of the sequences and their percent identity.
  """
  try:
    main_seqs = []
    main_entropy = []

    result, calc_entropy = gen_from_multimask(seq, print_flag=False, mask_flag="first", percent=percent_masked, prob_cutoff=prob_cutoff)
    for s,e in zip(result,calc_entropy):
      if s not in main_seqs:
        main_seqs.append(s)
        main_entropy.append(e)
    length = len(main_seqs)
    print(f"First masking generated {length} sequences")

    result, calc_entropy = gen_from_multimask(seq, print_flag=False, mask_flag="last", percent=percent_masked, prob_cutoff=prob_cutoff)
    for s,e in zip(result,calc_entropy):
      if s not in main_seqs:
        main_seqs.append(s)
        main_entropy.append(e)
    print(f"Last masking generated {len(main_seqs)-length} sequences")
    length = len(main_seqs)

    for _ in range(4):
      result, calc_entropy = gen_from_multimask(seq, print_flag=False, mask_flag="random", percent=percent_masked, prob_cutoff=prob_cutoff)
      for s,e in zip(result,calc_entropy):
        if s not in main_seqs:
          main_seqs.append(s)
          main_entropy.append(e)
      print(f"Random masking generated {len(main_seqs)-length} sequences")
      length = len(main_seqs)

    print(f"Total sequences generated: {len(main_seqs)}")

    final_seqs = main_seqs
    final_entropy = main_entropy
    identities = [percent_identity(seq, s) for s in final_seqs]

    out_text = f"Total sequences generated for hit: {len(final_seqs)}\n"
    out_text += "===================================================\n"
    for i, (s, ident) in enumerate(zip(final_seqs, identities), 1):
      out_text += f"analogue {i}: {s} with {ident*100:.1f}% identity to parent\n"

  except Exception:
    final_seqs = []
    final_entropy = []
    identities = []
    out_text = "Invalid sequence"

  return final_seqs, final_entropy, identities, out_text

class gen_mask_fill():
  '''
  Class to generate masks and fill them with ESM predictions
  '''
  def __init__(self, checkpoint: str, seq: list, num_to_mask: int):
    '''
    Constructor for mask filling
    Input:
    - checkpoint: path to ESM model
    - seq: sequence to mask
    - num_to_mask: number of residues to mask
    '''
    self.checkpoint = checkpoint
    self.seq = seq
    self.num_to_mask = num_to_mask

  def start_model(self):
    '''
    Start ESM model and tokenizer
    '''
    self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
    self.model = EsmForMaskedLM.from_pretrained((self.checkpoint))

  def mask_tokens(self):
    '''
    Mask tokens in sequence
    Output:
    - seq_ids: sequence of tokens
    - masked_chain: masked sequence
    - masked_chain_ids: masked sequence of tokens
    '''
    self.seq_ids = self.tokenizer(''.join(self.seq))['input_ids']

    masked_chain = []
    self.rdn_ixd = torch.randint(1, len(self.seq)-1, (self.num_to_mask,))
    for i, token in enumerate(self.seq):
      if i in self.rdn_ixd:
        masked_chain.append('<mask>')
      else:
        masked_chain.append(token)

    self.masked_chain = masked_chain
    self.masked_chain_ids = self.tokenizer(''.join(masked_chain))['input_ids']

    return self.seq_ids, self.masked_chain, self.masked_chain_ids

  def unmask(self):
    '''
    Unmask tokens in sequence
    Output:
    - model_preds: predictions from ESM
    '''
    model_out = self.model(**self.tokenizer(text = ''.join(self.masked_chain), return_tensors='pt'))

    model_preds = []
    for row in model_out.logits[0]:
      probs = torch.softmax(row.detach().clone(), dim=0)
      model_preds.append(torch.argmax(probs).detach().clone().item())

    self.model_preds = model_preds
    return self.model_preds

  def new_seq_from_ids(self):
    '''
    Convert tokens to sequence
    Output:
    - new_seq: new sequence
    '''
    raw = self.tokenizer.decode(self.model_preds)
    self.new_seq = raw.replace('<cls>','').replace('<eos>','').replace(' ','')

  def compare_seqs(self):
    '''
    Compare original and new sequences
    Output:
    - chain: original sequence
    - new_seq: new sequence
    '''
    self.new_seq_from_ids()
    self.chain = ''.join(self.seq).replace('<cls>','').replace('<eos>','')
    print(f"Original: {self.chain}")
    print(f"Novel   : {self.new_seq}")

    i = 1
    for char_o, char_n in zip(self.seq,self.new_seq):
      if self.masked_chain_ids[i-1] == 32:
        mask = 'was masked.'
      else:
        mask = 'was not masked.'

      if char_o != char_n:
        print(f"Residue {i} changed {one_to_three(char_o)} --> {one_to_three(char_n)}. This token {mask}")
      i += 1

    return self.chain, self.new_seq
  
  def compare_seqs_naive(self):
    '''
    Compare original and new sequences by % of differences
    Output:
    - chain: original sequence
    - new_seq: new sequence
    '''
    self.new_seq_from_ids()
    self.chain = ''.join(self.seq).replace('<cls>','').replace('<eos>','')
    print(f"Original: {self.chain}")
    print(f"Novel   : {self.new_seq}")

    num_diff = 0
    for char_o, char_n in zip(self.seq,self.new_seq):

      if char_o != char_n:
        num_diff += 1
    
    print(f"Number of differences: {num_diff} out of {len(self.seq)}")
    print(f"Percentage of differences: {num_diff/len(self.seq):.3f}")

    return self.chain, self.new_seq

class embed_proteins():
  '''
  Class to embed proteins using ESM
  '''
  def __init__(self, checkpoint: str, list_seqs: list):
    '''
    Constructor for embedding proteins
    Input:
    - checkpoint: path to ESM model
    - list_seqs: list of sequences to embed
    '''
    self.checkpoint = checkpoint
    self.list_seqs = list_seqs
    self.model_start_flag = 0

    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {self.device}")

  def start_model(self):
    '''
    Start ESM model and tokenizer
    '''
    self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
    self.model = EsmModel.from_pretrained((self.checkpoint))
    self.model_start_flag = 1
    print(f"Model loaded from {self.checkpoint}")

  def embed_seqs(self):
    '''
    Embed sequences
    Output:
    - embeddings: embeddings of sequences
    '''
    if self.model_start_flag == 0:
      self.start_model()

    model_inputs = self.tokenizer(self.list_seqs, padding=True, return_tensors='pt')
    model_input = {k: v.to(self.device) for k, v in model_inputs.items()}

    self.model.to(self.device)
    self.model.eval()

    with torch.no_grad():
      model_out = self.model(**model_input)
      self.embeddings = model_out.last_hidden_state.mean(dim=1)

    self.embeddings = self.embeddings.detach().cpu().numpy()

    return self.embeddings

  def compare_embeddings(self, a: int, b: int):
    '''
    Compare embeddings of two sequences using cosine similarity
    Input:
    - a: index of first sequence
    - b: index of second sequence
    '''
    v1 = self.embeddings[a].reshape(1,-1)
    v2 = self.embeddings[b].reshape(1,-1)

    ov = cosine_similarity(v1,v2)

    print(f"Overlap between protein {a} and {b}: {ov[0][0]:.5f}")