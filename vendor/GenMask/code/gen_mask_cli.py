#!/usr/bin/env python
"""
CLI for CafChemSubs.gen_mask: generate analogue molecules by masking and
unmasking tokens of an input SMILES string.

Usage:
    python code/gen_mask_cli.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
    python code/gen_mask_cli.py --csv molecules.csv
    python code/gen_mask_cli.py --csv molecules.csv --percent 0.15 --model cafierom/bert-base-cased-ChemTok-ZN250K-V1 --out outputs/results.csv
"""
import argparse
import os
import sys

import pandas as pd

import CafChemSubs as cs

DEFAULT_MODEL = "cafierom/bert-base-cased-ChemTok-ZN250K-V1"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "outputs", "results.csv")


def find_smiles_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.lower() == "smiles":
            return col
    raise ValueError(
        f"No SMILES column found (looked for 'smiles'/'SMILES'/'Smiles'). "
        f"Columns present: {list(df.columns)}"
    )


def run_one(smile: str, percent: float, no_image: bool, out_dir: str, tag: str, tanimoto: bool):
    final_smiles, final_entropy, qeds, mols, out_text, img = cs.gen_mask(smile, percent)
    print(out_text)

    if not no_image and img is not None:
        img_path = os.path.join(out_dir, f"{tag}.png")
        img.save(img_path)
        print(f"Saved image: {img_path}")

    result = {
        "input_smiles": [smile] * len(final_smiles),
        "analogue_smiles": final_smiles,
        "entropy": final_entropy,
        "qed": qeds,
    }

    if tanimoto and final_smiles:
        # mm_cutoff > 1.0 disables the function's built-in stdout logging, since
        # Tanimoto similarity is bounded in [0, 1]; we only want the raw values here.
        _, known_array = cs.calculate_similarities(smile, final_smiles, mm_cutoff=1.1)
        result["tanimoto_to_parent"] = known_array[0]

    return pd.DataFrame(result)


def main():
    parser = argparse.ArgumentParser(description="Generate molecule analogues via masked-language-model unmasking.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smiles", help="A single SMILES string.")
    source.add_argument("--csv", help="Path to a CSV file with a smiles/SMILES/Smiles column.")
    parser.add_argument("--percent", type=float, default=0.10, help="Fraction of tokens to mask (default: 0.10).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model name (default: {DEFAULT_MODEL}).")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Output CSV path (default: {DEFAULT_OUT}).")
    parser.add_argument("--no-image", action="store_true", help="Skip saving grid-image PNGs.")
    parser.add_argument("--no-tanimoto", action="store_true", help="Skip the tanimoto_to_parent column (Morgan fingerprint similarity to the input molecule).")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    cs.genmask_model_setup(0, args.model)

    tanimoto = not args.no_tanimoto

    if args.smiles:
        result_df = run_one(args.smiles, args.percent, args.no_image, out_dir, tag="molecule_0", tanimoto=tanimoto)
    else:
        df = pd.read_csv(args.csv)
        smiles_col = find_smiles_column(df)
        frames = []
        for i, smile in enumerate(df[smiles_col]):
            print(f"\n=== [{i+1}/{len(df)}] {smile} ===")
            frames.append(run_one(smile, args.percent, args.no_image, out_dir, tag=f"molecule_{i}", tanimoto=tanimoto))
        result_df = pd.concat(frames, ignore_index=True)

    result_df.to_csv(args.out, index=False)
    print(f"\nSaved {len(result_df)} analogues to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
