"""
CafChemGPT - SMILES GPT, refactored from TensorFlow/Keras to PyTorch.

The public API mirrors the original TensorFlow module so the existing
notebook/workflow still reads naturally:

    trim_vocab, test_vocab, make_datasets, strip_smiles, mols_from_smiles,
    test_gen, make_gpt, save_gpt, make_finetune_gpt, unfreeze_gpt,
    load_gpt, load_foundation, make_prompts, gen_mols

PyTorch-specific additions:

    get_device  - pick MPS (Apple Silicon) by default, falling back to CUDA then CPU
    train_gpt   - replaces the old Keras compile/fit pattern

Apple Silicon (MPS) is the default device; CUDA is used when available;
CPU is the final fallback.
"""

import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import contextlib
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit import RDLogger
from PIL import Image, ImageDraw as PILImageDraw
from smiles_tokenizer import SmilesTokenizer


@contextlib.contextmanager
def _silence_rdkit():
    """Temporarily mute RDKit's C-level app logger.

    Chem.MolFromSmiles writes SMILES-parse failures (``[hh:mm:ss] ERROR: ...``)
    to fd 2 regardless of whether the caller checks for None. Generated GPT
    SMILES are frequently invalid and already handled by the caller, so the
    stderr spam is redundant. Mirrors vina_dock.py's RDLogger.DisableLog
    pattern. No-op if RDLogger can't be toggled for some reason.
    """
    try:
        RDLogger.DisableLog("rdApp.*")
        yield
    finally:
        try:
            RDLogger.EnableLog("rdApp.*")
        except Exception:
            pass

# Module-level print flag - set from modrag.py (matches the modrag_*_functions
# convention in NODE_INTEGRATION_SKILLS.md). Verbose training/generation
# diagnostics are gated behind it; the node header (printed in gpt_node) is
# always shown.
print_flag = False

# ---------------------------------------------------------------------------
# Device selection: Apple Silicon (MPS) first, then CUDA, then CPU.
# ---------------------------------------------------------------------------

def get_device():
    """Return the best available torch device: MPS > CUDA > CPU."""
    if torch.backends.mps is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Vocabulary / dataset utilities (unchanged in spirit, framework agnostic).
# ---------------------------------------------------------------------------

# Salt fragments that are stripped from SMILES before tokenization.
_SALT_REPLACES = (
    ("[Na+].", ""), ("[Cl-].", ""), (".[Cl-]", ""), (".[Na+]", ""),
    ("[K+].", ""), ("[Br-].", ""), (".[K+]", ""), (".[Br-]", ""),
    ("[I-].", ""), (".[I-]", ""), ("[Ca2+].", ""), (".[Ca2+]", ""),
)


def _strip_salts(smiles: str) -> str:
    for old, new in _SALT_REPLACES:
        smiles = smiles.replace(old, new)
    return smiles


def trim_vocab(filename: str, tokens_to_remove: list, smiles_column="SMILES"):
    '''
    Trims entries from a SMILES list that contain tokens not found in the
    Foundation model's vocabulary list. Also trims entries that are longer
    than the Foundation model's context window.

        Args:
            filename: a CSV file with the dataset to be trimmed
            tokens_to_remove: a set of tokens to remove, from test_vocab
            smiles_column: name of the SMILES column
        Returns:
            None: a new CSV file is saved with the trimmed list.
    '''
    df = pd.read_csv(filename)

    Xa = [_strip_salts(s) for s in df[smiles_column]]

    smiles_removed_tokens = []
    tokens_to_remove_set = set(tokens_to_remove)
    tokenizer = SmilesTokenizer(vocab_file="../data/vocab_305K.txt")
    for smiles in Xa:
        tokens = tokenizer._tokenize(smiles)
        if not any(token in tokens_to_remove_set for token in tokens):
            smiles_removed_tokens.append(smiles)

    smiles_no_long = [s for s in smiles_removed_tokens if len(s) <= 166]

    if print_flag:
        print(f"Removed {len(Xa) - len(smiles_no_long)} entries from the list!")

    new_df = pd.DataFrame({"SMILES": smiles_no_long})
    out = f'{filename.replace(".csv", "")}_trimmed.csv'
    new_df.to_csv(out, index=False)
    if print_flag:
        print(f"New CSV file written: {out}")


def test_vocab(filename: str, smiles_column='SMILES'):
    '''
    Tests the vocabulary of a new dataset against the foundation model
    vocabulary. Rejects if the new dataset has tokens not in the foundation
    model vocabulary, or if the context window is too large.

        Args:
            filename: name of new dataset
            smiles_column: name of the smiles column
        Returns:
            novel_items: list of tokens not in the foundation model vocabulary
    '''
    df = pd.read_csv(filename)

    Xa = [_strip_salts(s) for s in df[smiles_column]]

    tokenizer = SmilesTokenizer(vocab_file="../data/vocab.txt")
    featname = "SMILES Tokenizer"

    fl = list(map(lambda x: tokenizer.encode(x), Xa))

    biggest = 1
    smallest = 200
    for i in range(len(fl)):
        temp = len(fl[i])
        if temp > biggest:
            biggest = temp
        if temp < smallest:
            smallest = temp

    if print_flag:
        print(biggest, smallest)

    string_length = smallest - 1
    max_length = biggest

    fl2 = list(map(lambda x: tokenizer.add_padding_tokens(x, max_length), fl))

    fl2set = set()
    for sublist in fl2:
        fl2set.update(sublist)
    new_vocab_size = len(fl2set)
    if print_flag:
        print("New vocabulary size: ", new_vocab_size)

    with open("../data/vocab_305K.txt", "r") as f:
        raw_lines = f.readlines()
    VOCAB_SIZE = len(raw_lines)
    if print_flag:
        print("Vocabulary size for standard dataset: ", VOCAB_SIZE)

    lines = [line.replace("\n", "") for line in raw_lines]

    novel_items = []
    for item in fl2set:
        item = tokenizer.decode([item])
        item = tokenizer.convert_tokens_to_string(item)
        item = item.replace(" ", "")

        if item not in lines:
            if print_flag:
                print(f"{item} not in standard vocabulary")
            novel_items.append(item)

    if print_flag:
        if len(novel_items) > 0:
            print("This dataset is not compatible with the Foundation model vocabulary")
        else:
            print("This dataset is compatible with the Foundation model vocabulary")

        if max_length > 166:
            print("This dataset's context window is not compatible with the Foundation model.")
        else:
            print("This dataset's context window is compatible with the Foundation model")

    return novel_items


def make_datasets(filename: str, smiles_column='SMILES'):
    '''
    Tokenizes a dataset and returns the input and target arrays.

        Args:
            filename: name of new dataset
            smiles_column: name of the smiles column
        Returns:
            fx: input array
            fy: target array
            VOCAB_SIZE: vocabulary size
            tokenizer: tokenizer object
            max_length: longest SMILES chain
    '''
    df = pd.read_csv(filename)

    Xa = [_strip_salts(s) for s in df[smiles_column]]

    tokenizer = SmilesTokenizer(vocab_file="../data/vocab_305K.txt")
    featname = "SMILES Tokenizer"

    fl = list(map(lambda x: tokenizer.encode(x), Xa))

    biggest = 1
    smallest = 200
    for i in range(len(fl)):
        temp = len(fl[i])
        if temp > biggest:
            biggest = temp
        if temp < smallest:
            smallest = temp

    if print_flag:
        print(biggest, smallest)

    string_length = smallest - 1
    max_length = biggest

    fl2 = list(map(lambda x: tokenizer.add_padding_tokens(x, max_length), fl))

    with open("../data/vocab_305K.txt", "r") as f:
        lines = f.readlines()
    VOCAB_SIZE = len(lines)
    if print_flag:
        print("Vocabulary size for this dataset: ", VOCAB_SIZE)

    x = []
    y = []
    for string in fl2:
        x.append(string[0:max_length - 1])  # input
        y.append(string[1:max_length])      # target (shifted by one)

    x = np.array(x)
    y = np.array(y)
    if print_flag:
        print("Number of features and datapoints, targets: ", x.shape, y.shape)
        print("featurization done with: ", featname)

    return x, y, VOCAB_SIZE, tokenizer, max_length


def strip_smiles(input_string):
    '''
    Cleans un-needed tokens from the SMILES string.

        Args:
            input_string: SMILES string
        Returns:
            output_string: cleaned SMILES string
    '''
    output_string = input_string.replace(" ", "").replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "")
    output_string = output_string.replace("[Na+].", "").replace(".[Na+]", "")
    return output_string


def mols_from_smiles(input_smiles_list):
    '''
    Converts a list of SMILES strings to a list of RDKit molecules.

        Args:
            input_smiles_list: list of SMILES strings
        Returns:
            valid_mols: list of RDKit molecules
            valid_smiles: list of SMILES strings
    '''
    valid_mols = []
    valid_smiles = []

    good_count = 0
    with _silence_rdkit():
        for ti, smile in enumerate(input_smiles_list):
            temp_mol = Chem.MolFromSmiles(smile)
            if temp_mol is not None:
                valid_mols.append(temp_mol)
                valid_smiles.append(smile)
                good_count += 1
            else:
                if print_flag:
                    print(f"SMILES {ti} was not valid!")

    if len(valid_mols) == len(valid_smiles) == good_count:
        if print_flag:
            print(f"Generated a total of {good_count} mol objects")
    else:
        if print_flag:
            print("mismatch!")
    return valid_mols, valid_smiles


# ---------------------------------------------------------------------------
# Model definition (PyTorch).
# ---------------------------------------------------------------------------

# Default hyperparameters mirror the original TF/Keras model
# (ChemSpaceAttn.ipynb, cell 11): 2 heads, each with a 256-wide key (key_dim),
# i.e. q/k/v project embed_dim -> num_heads*key_dim (256 -> 512), NOT a 4-way
# split of embed_dim. The earlier refactor used N_HEADS=4/head_dim=64, which is
# a smaller, different attention -- matching TF here is what closed the gap.
EMBEDDING_DIM = 256
N_HEADS = 2
KEY_DIM = 256
FEED_FORWARD_DIM = 256
DROPOUT_RATE = 0.1

# Foundation model constants (match the original 305K foundation).
FOUNDATION_NUM_BLOCKS = 2
FOUNDATION_MAX_LENGTH = 166
FOUNDATION_VOCAB_SIZE = 100
FOUNDATION_FILE = "../data/GPT_ZN305_pytorch.pt"


def _causal_mask(seq_len: int, device) -> torch.Tensor:
    """Return an additive (L, L) causal mask: 0 attend, -inf masked."""
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
    )
    return mask


class TransformerBlock(nn.Module):
    """Transformer block with multi-head causal self-attention.

    Uses torch scaled_dot_product_attention (SDPA) for a fused, MPS-friendly
    causal attention kernel - far fewer CPU op dispatches than
    nn.MultiheadAttention, which keeps training fast on Apple Silicon even
    under CPU contention. Post-norm residual structure matches the original
    Keras model. attn_scores is returned as None for API parity (callers
    ignore it)."""

    def __init__(self, num_heads, key_dim, embed_dim, ff_dim,
                 dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.head_dim = key_dim           # SDPA head_dim == per-head key/query width
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # TF MultiHeadAttention(num_heads, key_dim): q/k/v project embed_dim ->
        # num_heads*key_dim (NOT an embed_dim->embed_dim split). With N_HEADS=2
        # and KEY_DIM=256 that's 256->512, ~2x the attention params of a 4-head
        # embed_dim//heads split. Matching this is what closed the quality gap.
        qkv_dim = num_heads * key_dim
        self.q_proj = nn.Linear(embed_dim, qkv_dim)
        self.k_proj = nn.Linear(embed_dim, qkv_dim)
        self.v_proj = nn.Linear(embed_dim, qkv_dim)
        self.out_proj = nn.Linear(qkv_dim, embed_dim)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.ln_1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ffn_1 = nn.Linear(embed_dim, ff_dim)
        self.ffn_2 = nn.Linear(ff_dim, embed_dim)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.ln_2 = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        B, L, _ = x.shape
        h, hd = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, L, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, L, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, L, h, hd).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, L, h * hd)
        attn_out = self.dropout_1(self.out_proj(attn))
        out1 = self.ln_1(x + attn_out)
        ffn = self.ffn_2(F.relu(self.ffn_1(out1)))
        ffn = self.dropout_2(ffn)
        return self.ln_2(out1 + ffn), None


class TokenAndPositionEmbedding(nn.Module):
    """Embeds tokens and positions."""

    def __init__(self, max_len, vocab_size, embed_dim):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        # Small init (GPT-2 style, N(0, 0.02)). PyTorch's nn.Embedding default
        # is N(0, 1) -- far too large for transformer embeddings: token+pos
        # feeds straight into attention (no pre-norm), so std-1.0 makes early
        # attention scores blow up, softmax peaks, gradients flow poorly, and
        # the model converges to a worse minimum (stalled ~0.58 vs TF's ~0.52).
        # Keras Embedding defaults to uniform[-0.05, 0.05] (~0.029 std); this
        # matches that regime. Safe for load_gpt / make_finetune_gpt, which
        # overwrite these weights with loaded state afterward.
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, x):
        maxlen = x.size(1)
        positions = self.pos_emb(torch.arange(maxlen, device=x.device))
        return self.token_emb(x) + positions


class GPT(nn.Module):
    """A small decoder-only GPT producing token logits over the vocabulary."""

    def __init__(self, num_blocks, max_length, vocab_size,
                 embed_dim=EMBEDDING_DIM, num_heads=N_HEADS, key_dim=None,
                 ff_dim=FEED_FORWARD_DIM, dropout_rate=DROPOUT_RATE):
        super().__init__()
        # key_dim defaults to embed_dim//num_heads so OLD checkpoints (which
        # predate the key_dim field and used a plain 4-way split) still load.
        # New models pass KEY_DIM explicitly to match the TF architecture.
        if key_dim is None:
            key_dim = embed_dim // num_heads
        self.num_blocks = num_blocks
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.embedding = TokenAndPositionEmbedding(max_length, vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(num_heads, key_dim, embed_dim, ff_dim, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        h = self.embedding(x)
        scores = None
        for block in self.blocks:
            h, scores = block(h)
        logits = self.head(h)
        return logits, scores

    def config(self):
        return {
            "num_blocks": self.num_blocks,
            "max_length": self.max_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        }


def make_gpt(num_blocks: int, max_length: int, VOCAB_SIZE: int,
             embed_dim=EMBEDDING_DIM, num_heads=N_HEADS, key_dim=KEY_DIM,
             ff_dim=FEED_FORWARD_DIM, dropout_rate=DROPOUT_RATE,
             device=None):
    '''
    Creates a GPT with a specified number of transformer blocks.

        Args:
            num_blocks: number of transformer blocks
            max_length: context window
            VOCAB_SIZE: vocabulary size
        Returns:
            gpt: GPT model (on the default device)
    '''
    if device is None:
        device = get_device()
    gpt = GPT(num_blocks, max_length, VOCAB_SIZE,
              embed_dim=embed_dim, num_heads=num_heads, key_dim=key_dim,
              ff_dim=ff_dim, dropout_rate=dropout_rate).to(device)
    gpt.summary()
    return gpt


def save_gpt(gpt, filename: str):
    '''
    Saves a GPT model.

        Args:
            gpt: GPT model
            filename: name of the model (a .pt file is written)
        Returns:
            None; saves model state + config to a .pt file.
    '''
    gpt.summary()

    path = f"{filename}.pt"
    torch.save({"config": gpt.config(), "state_dict": gpt.state_dict()}, path)
    if print_flag:
        print(f"model saved with name: {filename}. (-> {path})")


def _build_from_config(cfg, device=None):
    if device is None:
        device = get_device()
    # key_dim absent in configs saved before the TF-architecture match -> None
    # -> GPT falls back to embed_dim//num_heads, so old checkpoints still load.
    return GPT(
        cfg["num_blocks"], cfg["max_length"], cfg["vocab_size"],
        embed_dim=cfg.get("embed_dim", EMBEDDING_DIM),
        num_heads=cfg.get("num_heads", N_HEADS),
        key_dim=cfg.get("key_dim"),
        ff_dim=cfg.get("ff_dim", FEED_FORWARD_DIM),
        dropout_rate=cfg.get("dropout_rate", DROPOUT_RATE),
    ).to(device)


def load_gpt(filename: str, total_layers: int, max_length: int, VOCAB_SIZE: int,
             device=None):
    '''
    Loads a GPT model.

        Args:
            filename: name of the model (.pt file, without extension)
            total_layers: total number of transformer blocks
            max_length: context window
            VOCAB_SIZE: vocabulary size
        Returns:
            gpt_load: loaded GPT model
    '''
    if device is None:
        device = get_device()
    path = f"{filename}.pt"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {
        "num_blocks": total_layers, "max_length": max_length,
        "vocab_size": VOCAB_SIZE,
    })
    # Honor the caller's requested architecture if a config is unavailable.
    cfg.setdefault("num_blocks", total_layers)
    cfg.setdefault("max_length", max_length)
    cfg.setdefault("vocab_size", VOCAB_SIZE)

    gpt_load = _build_from_config(cfg, device=device)
    gpt_load.load_state_dict(ckpt["state_dict"])
    if print_flag:
        print(f"model loaded with name: {filename}.")
    gpt_load.summary()
    return gpt_load


def load_foundation(device=None):
    '''
    Loads the GPT Foundation model (PyTorch, trained on ZN305K).

        Args:
            None
        Returns:
            gpt_load: loaded GPT model
    '''
    if device is None:
        device = get_device()
    path = FOUNDATION_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Foundation model not found at {path}. Build it first with "
            "build_foundation() / train the foundation on ZN305K_smiles.csv."
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    gpt_load = _build_from_config(ckpt["config"], device=device)
    gpt_load.load_state_dict(ckpt["state_dict"])
    if print_flag:
        print("Foundation model loaded.")
    gpt_load.summary()
    return gpt_load


def make_finetune_gpt(num_new_blocks: int, freeze_old_layers=True,
                      device=None):
    '''
    Creates a fine-tuning model from the saved foundation model: loads the
    foundation weights and appends `num_new_blocks` freshly initialized
    transformer blocks on top.

        Args:
            num_new_blocks: number of new transformer blocks to add
            freeze_old_layers: whether to freeze the foundation layers
        Returns:
            gpt_ft: fine-tuning model
    '''
    if device is None:
        device = get_device()

    foundation = load_foundation(device=device)
    base_cfg = foundation.config()
    total_blocks = base_cfg["num_blocks"] + num_new_blocks

    gpt_ft = GPT(
        total_blocks, base_cfg["max_length"], base_cfg["vocab_size"],
        embed_dim=base_cfg["embed_dim"], num_heads=base_cfg["num_heads"],
        key_dim=base_cfg.get("key_dim"),
        ff_dim=base_cfg["ff_dim"], dropout_rate=base_cfg["dropout_rate"],
    ).to(device)

    # Copy foundation embedding + first N blocks + (optionally) head.
    new_state = gpt_ft.state_dict()
    old_state = foundation.state_dict()
    copied = []
    for key, val in old_state.items():
        if key in new_state and new_state[key].shape == val.shape:
            new_state[key] = val
            copied.append(key)
    # The classification head is part of the foundation; reuse it as the
    # starting point for fine-tuning as well.
    gpt_ft.load_state_dict(new_state)
    if print_flag:
        print(f"Copied {len(copied)} parameter tensors from the foundation model.")

    # Freeze the foundation blocks (and embedding) if requested.
    if freeze_old_layers:
        for p in gpt_ft.embedding.parameters():
            p.requires_grad = False
        for i in range(base_cfg["num_blocks"]):
            for p in gpt_ft.blocks[i].parameters():
                p.requires_grad = False
            if print_flag:
                print(f"setting transformer block {i} untrainable.")
        for p in gpt_ft.blocks[base_cfg["num_blocks"]:].parameters():
            p.requires_grad = True
        for p in gpt_ft.head.parameters():
            p.requires_grad = True
        if print_flag:
            print(f"setting {num_new_blocks} new block(s) + head trainable.")

    gpt_ft.summary()
    return gpt_ft


def unfreeze_gpt(gpt_model):
    '''
    Unfreezes all parameters in a model.

        Args:
            gpt_model: model to unfreeze
        Returns:
            gpt_model: unfrozen model
    '''
    for p in gpt_model.parameters():
        p.requires_grad = True
    if print_flag:
        print("All layers set trainable.")
    gpt_model.summary()
    return gpt_model


# ---------------------------------------------------------------------------
# Training (replaces Keras compile/fit).
# ---------------------------------------------------------------------------

def train_gpt(gpt, fx, fy, epochs=5, batch_size=512, lr=1e-3,
              verbose=True, use_amp=None,
              warmup_steps=0, min_lr_frac=1.0):
    '''
    Trains a GPT model with next-token cross-entropy loss.

        Args:
            gpt: GPT model
            fx: input array (N, L) of token ids
            fy: target array (N, L) of token ids (shifted)
            epochs: number of epochs
            batch_size: mini-batch size
            lr: peak learning rate
            use_amp: mixed-precision autocast. None = auto (bf16 on CUDA, off
                elsewhere — MPS/CPU). True/False forces it on/off. bf16 on an
                A100 is a big speedup and needs no grad scaler.
            warmup_steps: linear LR warmup from ~0 up to `lr` over this many
                optimizer steps, then either hold (min_lr_frac=1.0) or cosine-
                decay. 0 = no warmup (constant lr). Useful with fresh-init
                transformers so early steps don't destabilize the model.
            min_lr_frac: final LR as a fraction of `lr` after cosine decay over
                the steps remaining after warmup. 1.0 = no decay (warmup then
                constant). e.g. 0.1 -> lr decays to 0.1*lr.
        Returns:
            gpt: trained model

    Note: the loss is NOT masked on padding -- this matches the original TF
    training (unmasked SparseCategoricalCrossentropy over all 166 positions).
    ~75% of targets are [PAD], so the model learns the [SEP] -> [PAD] -> [PAD]
    shut-down transition. Masking PAD removes that signal and the model never
    learns to stop, which produces run-on chains at inference (decode drops
    PAD, but the model emits stray atoms instead). So the reported loss is the
    TF-comparable unmasked number, NOT a real-token-only number.
    '''
    device = next(gpt.parameters()).device
    gpt.train()

    # Mixed precision: bf16 autocast on CUDA is free speed (no grad scaler
    # needed). Auto-off on MPS/CPU so local Apple-Silicon runs are unaffected.
    if use_amp is None:
        use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else None

    x = torch.as_tensor(np.asarray(fx), dtype=torch.long, device=device)
    y = torch.as_tensor(np.asarray(fy), dtype=torch.long, device=device)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gpt.parameters()), lr=lr)

    # Optional LR schedule: linear warmup over `warmup_steps`, then cosine
    # decay to `min_lr_frac * lr` over the remaining steps. Defaults
    # (warmup_steps=0, min_lr_frac=1.0) reproduce the original constant-LR
    # behavior, so existing callers (finetune, run_gpt.py) are unaffected.
    total_steps = epochs * len(loader)
    warmup = min(warmup_steps, total_steps)
    decay_steps = max(1, total_steps - warmup)
    use_sched = warmup > 0 or min_lr_frac < 1.0
    if use_sched:
        def lr_lambda(step):
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup  # +1 so the first step isn't lr=0
            prog = (step - warmup) / decay_steps
            prog = min(max(prog, 0.0), 1.0)
            return min_lr_frac + 0.5 * (1.0 - min_lr_frac) * (1.0 + np.cos(np.pi * prog))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    global_step = 0
    for epoch in range(epochs):
        running = 0.0
        n_batches = 0
        t0 = time.perf_counter()
        for xb, yb in loader:
            optimizer.zero_grad()
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    logits, _ = gpt(xb)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        yb.reshape(-1),
                    )
            else:
                logits, _ = gpt(xb)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    yb.reshape(-1),
                )
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running += float(loss.item())
            n_batches += 1
            global_step += 1
            if verbose and print_flag and (n_batches % 50 == 0):
                el = time.perf_counter() - t0
                print(f"  epoch {epoch+1}/{epochs} batch {n_batches}/{len(loader)} "
                      f"loss {running/n_batches:.4f} ({el:.1f}s)", flush=True)
        if verbose and print_flag:
            print(f"epoch {epoch + 1}/{epochs} - loss: {running / max(n_batches, 1):.4f} "
                  f"- lr {optimizer.param_groups[0]['lr']:.2e} "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)
    gpt.eval()
    return gpt


# ---------------------------------------------------------------------------
# Inference / molecule generation.
# ---------------------------------------------------------------------------

def _sample_next_token(logits_last, T_int, rn_seed=None):
    """Sample the next token id from the last-position logits with temperature."""
    if T_int < 0.015:
        return torch.argmax(logits_last, dim=-1).cpu().numpy()
    probs = torch.softmax(logits_last, dim=-1)
    rescaled = torch.pow(probs, 1.0 / T_int)
    rescaled = rescaled / rescaled.sum(dim=-1, keepdim=True)
    return torch.multinomial(rescaled, num_samples=1).squeeze(-1).cpu().numpy()


@torch.no_grad()
def _generate(model, test_array, n_steps, tokenizer, VOCAB_SIZE,
              TEMP, use_ramp, rn_seed=42):
    """Greedy/temperature generation loop shared by test_gen and gen_mols."""
    device = next(model.parameters()).device
    if rn_seed is not None:
        torch.manual_seed(rn_seed)
    model.eval()

    batch_length, prompt_length = test_array.shape
    c_final = n_steps

    sig_start = 0.10
    c_o = int(c_final * sig_start)

    for c in range(0, c_final, 1):
        if use_ramp:
            T_int = TEMP * (1.0 / (1.0 + np.exp(-(c - c_o))))
        else:
            T_int = TEMP

        x_t = torch.as_tensor(test_array, dtype=torch.long, device=device)
        logits, _ = model(x_t)
        logits_last = logits[:, -1, :]

        preds = _sample_next_token(logits_last, T_int, rn_seed)
        preds = preds.reshape(-1).astype(int)
        test_array = np.c_[test_array, preds]
        if print_flag:
            print(test_array.shape)

    gen_molecules = list(map(lambda x: tokenizer.decode(x), test_array))
    gen_molecules = list(map(lambda x: tokenizer.convert_tokens_to_string(x), gen_molecules))
    gen_molecules = list(map(lambda x: strip_smiles(x), gen_molecules))
    return gen_molecules


def _draw_grid(mols, legends, molsPerRow=3, subImgSize=(300, 300)):
    """Draw a grid of molecules, robust to empty lists or undrawable mols.

    RDKit's MolsToGridImage crashes ("no draw context") on an empty list and
    can fail on mols with bad valences/conformers. This helper computes 2D
    coords, drops anything that can't be drawn, and falls back to a labeled
    placeholder image when there is nothing valid to show."""
    clean_mols, clean_legends = [], []
    for mol, leg in zip(mols, legends):
        if mol is None:
            continue
        try:
            mol = Chem.RemoveHs(mol) if any(a.GetNumExplicitHs() for a in mol.GetAtoms()) else mol
            AllChem.Compute2DCoords(mol)
            clean_mols.append(mol)
            clean_legends.append(leg)
        except Exception:
            continue

    if clean_mols:
        try:
            return Draw.MolsToGridImage(
                clean_mols, molsPerRow=molsPerRow, subImgSize=subImgSize,
                legends=clean_legends,
            )
        except Exception as e:
            if print_flag:
                print(f"Grid draw failed ({e}); drawing individually.")

    # Fallback: a placeholder image listing the SMILES (or noting none valid).
    w, h = 600, 400
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    d = PILImageDraw.Draw(img)
    if clean_legends:
        msg = f"{len(clean_legends)} valid molecule(s) but drawing failed:\n" + "\n".join(clean_legends[:8])
    else:
        msg = ("No valid molecules were generated.\n"
               "The model is likely undertrained - raise FOUNDATION_EPOCHS / "
               "FINETUNE_EPOCHS and re-run.")
    d.text((10, 10), msg, fill=(0, 0, 0))
    return img


def test_gen(model, tokenizer, T_int, VOCAB_SIZE, rn_seed=42):
    """Use a GPT model to generate novel molecules (quick 5-molecule test).

    Stripped-down inference with a fixed set of 5 short prompts for a fast
    sanity check.

        Args:
            model: the GPT model to use
            tokenizer: tokenizer to use
            T_int: temperature for inference
            VOCAB_SIZE: vocabulary size
            rn_seed: random seed
        Returns:
            img: image of generated molecules
    """
    test_string = ['C(', 'O=', 'c1', 'NC', 'CO']
    batch_length = len(test_string)
    test_xlist = np.empty([batch_length, 3], dtype=int)

    test_tokenized = list(map(lambda x: tokenizer.encode(x), test_string))
    for i in range(batch_length):
        test_xlist[i][:] = test_tokenized[i][:3]
    test_array = np.array(test_xlist)

    gen_molecules = _generate(
        model, test_array, n_steps=80 - 3, tokenizer=tokenizer,
        VOCAB_SIZE=VOCAB_SIZE, TEMP=T_int, use_ramp=False, rn_seed=rn_seed,
    )

    mols, smiles = mols_from_smiles(gen_molecules)
    img = _draw_grid(mols, smiles)
    return img


def make_prompts(num_prompts: int, prompt_length: int):
    '''
    Builds prompts by sampling SMILES prefixes from the ZN305K dataset.

        Args:
            num_prompts: how many prompts to make
            prompt_length: how many tokens in the prompt
        Returns:
            prompts: a list of prompts
    '''
    df = pd.read_csv("../data/ZN305K_smiles.csv")
    Xa = [_strip_salts(s) for s in df["SMILES"]]

    raw_prompts = random.choices(Xa, k=num_prompts)
    prompts = [smile[:prompt_length] for smile in raw_prompts]
    return prompts


def gen_mols(prompts: list, use_ramp: bool, model, tokenizer, TEMP: float,
             VOCAB_SIZE: int, rn_seed=42):
    '''
    Use a GPT model to generate novel molecules from prompts.

        Args:
            prompts: a list of prompts for inference
            use_ramp: Boolean to use temperature ramp during inference
            model: the GPT model to use
            tokenizer: tokenizer to use
            TEMP: temperature for inference
            VOCAB_SIZE: vocabulary size
            rn_seed: random seed
        Returns:
            img: image of generated molecules
            final_smiles: list of unique SMILES strings
    '''
    test_string = prompts
    batch_length = len(test_string)
    prompt_length = len(test_string[0])
    test_xlist = np.empty([batch_length, prompt_length], dtype=int)

    test_tokenized = list(map(lambda x: tokenizer.encode(x), test_string))
    for i in range(batch_length):
        test_xlist[i][:] = test_tokenized[i][:prompt_length]
    test_array = np.array(test_xlist)

    n_steps = 90 - prompt_length
    gen_molecules = _generate(
        model, test_array, n_steps=n_steps, tokenizer=tokenizer,
        VOCAB_SIZE=VOCAB_SIZE, TEMP=TEMP, use_ramp=use_ramp, rn_seed=rn_seed,
    )

    mols, smiles = mols_from_smiles(gen_molecules)

    final_smiles = []
    final_mols = []
    for smile, mol in zip(smiles, mols):
        if smile not in final_smiles:
            final_smiles.append(smile)
            final_mols.append(mol)

    if print_flag:
        print(f"Generated {len(final_smiles)} unique molecules.")

    img = _draw_grid(final_mols, final_smiles)
    return img, final_smiles


# ---------------------------------------------------------------------------
# Convenience: summary helper so gpt.summary() works like Keras.
# ---------------------------------------------------------------------------

def _summary(self):
    if print_flag:
        print(self)
        n_params = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {n_params:,} | Trainable: {n_train:,}")


nn.Module.summary = _summary