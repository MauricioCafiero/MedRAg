"""
SMILES Tokenizer - A standalone tokenizer for SMILES strings.

Replaces the DeepChem dependency with a minimal implementation
using only re and numpy.
"""

import re
import numpy as np
from typing import List, Union


class SmilesTokenizer:
    """
    A tokenizer for SMILES (Simplified Molecular Input Line Entry System) strings.

    This tokenizer handles SMILES-specific patterns including:
    - Bracketed atoms like [C@H], [N+], [O-], etc.
    - Multi-character elements like Cl, Br
    - Ring closure numbers like %10, %11
    - Special tokens like [CLS], [SEP], [MASK], [pad]

    Attributes:
        vocab: Dictionary mapping tokens to IDs
        ids_to_tokens: Dictionary mapping IDs to tokens
        special_tokens: Set of special tokens
        pad_token: Token used for padding
        cls_token: Token for classification
        sep_token: Token for separation
        mask_token: Token for masking
        unk_token: Token for unknown characters
    """

    def __init__(self, vocab_file: str = "SMILES_VAE/data/vocab.txt"):
        """
        Initialize the tokenizer with a vocabulary file.

        Args:
            vocab_file: Path to the vocabulary file (one token per line)
        """
        self.vocab = {}
        self.ids_to_tokens = {}
        self.special_tokens = {"", "[CLS]", "[SEP]", "[MASK]"}

        # Read vocabulary file
        with open(vocab_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            token = line.strip()
            if token:
                self.vocab[token] = idx
                self.ids_to_tokens[idx] = token

        # Set special token IDs (use defaults if not in vocab)
        self.pad_token = ""
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"

        # Detect unk token name from vocab (support both [unused] and   <think>   )
        self.unk_tokens = []
        if "[unused]" in self.vocab:
            self.unk_tokens.append("[unused]")
        if "   <think>    " in self.vocab:
            self.unk_tokens.append("   <think>    ")
        # Default to   <think>    if neither found
        if not self.unk_tokens:
            self.unk_tokens = ["   <think>    "]

        self.unk_token = self.unk_tokens[0]
        self.special_tokens.update(self.unk_tokens)

        self.pad_token_id = self.vocab.get(self.pad_token, 0)
        self.cls_token_id = self.vocab.get(self.cls_token, 12)
        self.sep_token_id = self.vocab.get(self.sep_token, 13)
        self.mask_token_id = self.vocab.get(self.mask_token, 14)
        self.unk_token_id = self.vocab.get(self.unk_token, 0)
        # Store all unknown token IDs for lookup
        self.unk_token_ids = set(self.vocab.get(t, 0) for t in self.unk_tokens)

        # Compile SMILES tokenization regex patterns (DeepChem/canonical style).
        # Order matters: longer/more specific patterns first.
        #
        # NOTE: do NOT use a generic [A-Z][a-z]? here. That pattern merges an
        # aliphatic atom with a following lowercase aromatic atom into one token
        # (Cc, Nc, Oc, Cn, Sc, Fc ...) which isn't in the vocab, so ~2% of all
        # tokens -- the most common drug-like motifs (toluene, phenol, aniline)
        # -- fell through to [unused], corrupting training and breaking SMILES
        # at generation time. Halogens are handled with Cl?/Br? and every other
        # atom as a single letter, so Cc -> C, c (two tokens, both in vocab).
        self.token_patterns = [
            r'\[[^\]]+\]',     # Bracketed atoms: [C@H], [N+], [O-], [nH]
            r'%\d{2}',          # Two-digit ring closure: %10, %11
            r'Br?|Cl?',        # Two-letter halogens (also matches lone B / C)
            r'N|O|S|P|F|I|H',  # Single-letter uppercase atoms
            r'b|c|n|o|s|p',     # Lowercase aromatic atoms
            r'[@.\\/=#$:~*+\-()\[\]]',  # Bonds, chirality, branches, misc
            r'\d',              # Single-digit ring closure
        ]
        self.token_regex = re.compile('|'.join(self.token_patterns))

    def _tokenize(self, smiles: str) -> List[str]:
        """
        Tokenize a SMILES string into individual tokens.

        Args:
            smiles: SMILES string to tokenize

        Returns:
            List of token strings
        """
        tokens = []
        i = 0
        while i < len(smiles):
            match = self.token_regex.match(smiles, i)
            if match:
                token = match.group(0)
                # Handle bracketed atoms (remove brackets for vocab lookup)
                if token.startswith('[') and token.endswith(']'):
                    tokens.append(token)
                else:
                    tokens.append(token)
                i = match.end()
            else:
                # Skip unknown characters (whitespace, etc.)
                i += 1

        return tokens

    def encode(self, smiles: Union[str, List[str]]) -> np.ndarray:
        """
        Encode a SMILES string or list of SMILES strings to token IDs.

        Args:
            smiles: SMILES string or list of SMILES strings

        Returns:
            Numpy array of token IDs (1D for single string, 2D for list)
        """
        if isinstance(smiles, str):
            # Single SMILES string
            tokens = self._tokenize(smiles)
            # Add CLS at start and SEP at end
            token_ids = [self.cls_token_id]
            for token in tokens:
                token_ids.append(self.vocab.get(token, self.unk_token_id))
            token_ids.append(self.sep_token_id)
            return np.array(token_ids, dtype=np.int32)
        else:
            # List of SMILES strings
            encoded = []
            for s in smiles:
                token_ids = self.encode(s)
                encoded.append(token_ids)
            # Pad to max length
            max_len = max(len(ids) for ids in encoded)
            padded = []
            for ids in encoded:
                if len(ids) < max_len:
                    padded_ids = np.concatenate([
                        ids,
                        np.full(max_len - len(ids), self.pad_token_id, dtype=np.int32)
                    ])
                else:
                    padded_ids = ids
                padded.append(padded_ids)
            return np.array(padded, dtype=np.int32)

    def decode(self, token_ids: Union[List[int], np.ndarray]) -> Union[str, List[str]]:
        """
        Decode token IDs to SMILES string(s).

        Args:
            token_ids: Token IDs or batch of token IDs (1D or 2D array)

        Returns:
            SMILES string or list of SMILES strings
        """
        token_ids = np.asarray(token_ids)

        if token_ids.ndim == 1:
            # Single sequence
            tokens = []
            for token_id in token_ids:
                token = self.ids_to_tokens.get(int(token_id), self.unk_token)
                if token not in {self.pad_token} | set(self.unk_tokens):
                    tokens.append(token)
            return self.convert_tokens_to_string(tokens)
        else:
            # Batch of sequences
            smiles_list = []
            for seq in token_ids:
                smiles_list.append(self.decode(seq))
            return smiles_list

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert a list of tokens to a SMILES string.

        Args:
            tokens: List of token strings

        Returns:
            SMILES string
        """
        # Remove special tokens and join
        filtered_tokens = [
            token for token in tokens
            if token not in {self.cls_token, self.sep_token, self.pad_token}
        ]
        return ''.join(filtered_tokens)

    def convert_ids_to_tokens(self, token_ids: Union[List[int], np.ndarray]) -> List[str]:
        """
        Convert token IDs to token strings.

        Args:
            token_ids: List of token IDs

        Returns:
            List of token strings
        """
        return [self.ids_to_tokens.get(int(token_id), self.unk_token) for token_id in token_ids]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert token strings to token IDs.

        Args:
            tokens: List of token strings

        Returns:
            List of token IDs
        """
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def add_padding_tokens(self, token_ids: Union[List[int], np.ndarray], max_length: int) -> np.ndarray:
        """
        Add padding tokens to a list of token IDs to reach max_length.

        Args:
            token_ids: List of token IDs
            max_length: Desired length after padding

        Returns:
            Numpy array of token IDs padded to max_length
        """
        token_ids = np.asarray(token_ids)
        current_length = len(token_ids)

        if current_length >= max_length:
            return token_ids[:max_length]

        # Create padded array
        padded = np.full(max_length, self.pad_token_id, dtype=np.int32)
        padded[:current_length] = token_ids

        return padded

    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.vocab)

    def __len__(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.vocab)
