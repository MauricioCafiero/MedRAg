#!/usr/bin/env python
"""
CLI for CafChemProteinMaskEmbed.gen_mask: generate analogue protein sequences by
masking and unmasking residues of an input sequence, mirroring gen_mask_cli.py for
SMILES.

Usage:
    python code/protein_mask_cli.py --sequence "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEV..."
    python code/protein_mask_cli.py --csv sequences.csv
    python code/protein_mask_cli.py --pdb 4ZGM --chain B
    python code/protein_mask_cli.py --pdb 4ZGM --chain B --percent 0.05 --model facebook/esm2_t12_35M_UR50D --out outputs/protein_results.csv
"""
import argparse
import os
import sys

import pandas as pd

import CafChemProteinMaskEmbed as pm

DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "outputs", "protein_results.csv")


def find_sequence_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.lower() == "sequence":
            return col
    raise ValueError(
        f"No sequence column found (looked for 'sequence'/'Sequence'/'SEQUENCE'). "
        f"Columns present: {list(df.columns)}"
    )


def sequence_from_pdb(pdb_id: str, chain: str) -> str:
    _, chains_ol = pm.extract_sequence(pdb_id)
    if chain is None:
        raise SystemExit(f"--chain is required with --pdb. Available chains for {pdb_id}: {sorted(chains_ol.keys())}")
    if chain not in chains_ol:
        raise SystemExit(f"Chain '{chain}' not found in {pdb_id}. Available chains: {sorted(chains_ol.keys())}")
    return "".join(chains_ol[chain])


def run_one(seq: str, percent: float, prob_cutoff: float, tag: str):
    final_seqs, final_entropy, identities, out_text = pm.gen_mask(seq, percent, prob_cutoff=prob_cutoff)
    print(out_text)

    return pd.DataFrame({
        "input_sequence": [seq] * len(final_seqs),
        "generated_sequence": final_seqs,
        "entropy": final_entropy,
        "percent_identity_to_parent": [i*100 for i in identities],
    })


def main():
    parser = argparse.ArgumentParser(description="Generate protein analogues via masked-language-model unmasking.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence", help="A single amino acid sequence (one-letter codes).")
    source.add_argument("--csv", help="Path to a CSV file with a sequence/Sequence/SEQUENCE column.")
    source.add_argument("--pdb", help="A PDB ID to fetch and pull a chain's sequence from (use with --chain).")
    parser.add_argument("--chain", help="Chain letter to use with --pdb, e.g. 'B'.")
    parser.add_argument("--percent", type=float, default=0.10, help="Fraction of residues to mask (default: 0.10).")
    parser.add_argument("--prob-cutoff", type=float, default=0.05, help="Minimum candidate probability to keep a beam (default: 0.05; ESM's confidence runs flatter than the SMILES model's, so this is lower than the SMILES CLI's implicit 0.1).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"HF ESM checkpoint (default: {DEFAULT_MODEL}).")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Output CSV path (default: {DEFAULT_OUT}).")
    args = parser.parse_args()

    if args.chain and not args.pdb:
        parser.error("--chain is only used together with --pdb.")

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    pm.genmask_model_setup(args.model)

    if args.sequence:
        result_df = run_one(args.sequence, args.percent, args.prob_cutoff, tag="sequence_0")
    elif args.pdb:
        seq = sequence_from_pdb(args.pdb, args.chain)
        print(f"Fetched {args.pdb} chain {args.chain}: {seq}")
        result_df = run_one(seq, args.percent, args.prob_cutoff, tag=f"{args.pdb}_{args.chain}")
    else:
        df = pd.read_csv(args.csv)
        seq_col = find_sequence_column(df)
        frames = []
        for i, seq in enumerate(df[seq_col]):
            print(f"\n=== [{i+1}/{len(df)}] {seq} ===")
            frames.append(run_one(seq, args.percent, args.prob_cutoff, tag=f"sequence_{i}"))
        result_df = pd.concat(frames, ignore_index=True)

    result_df.to_csv(args.out, index=False)
    print(f"\nSaved {len(result_df)} analogues to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
