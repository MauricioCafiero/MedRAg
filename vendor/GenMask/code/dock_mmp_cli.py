#!/usr/bin/env python
"""
CLI: dock every molecule in a GenMask results CSV against a receptor, then
use mmpdb to find matched molecular pairs across the whole pooled set
(parents + analogues, deduplicated) and report which structural
transformations help or hurt the docking score.

Vina affinities are negative and more-negative = more favorable, so a
transformation with a negative mean delta improved (predicted) binding.

Usage:
    python code/dock_mmp_cli.py --receptor sult1a3_2A3R.pdb \\
        --center 50.536 117.105 -1.369 --results outputs/results.csv
"""
import argparse
import csv
import os
import subprocess
import sys

import pandas as pd

import CafChemDock as dock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS = os.path.join(REPO_ROOT, "outputs", "results.csv")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "outputs", "dock_mmp")


def pooled_smiles(df):
    """Every distinct SMILES across input_smiles + analogue_smiles, each given a stable ID."""
    seen = {}
    for col in ("input_smiles", "analogue_smiles"):
        for smi in df[col]:
            if smi not in seen:
                seen[smi] = f"mol{len(seen)}"
    return seen  # smiles -> id


def write_smi_file(id_by_smiles, path):
    with open(path, "w") as fh:
        for smi, mol_id in id_by_smiles.items():
            fh.write(f"{smi} {mol_id}\n")


def write_properties_file(id_by_smiles, scores, property_name, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["ID", property_name])
        for smi, mol_id in id_by_smiles.items():
            score = scores.get(smi)
            w.writerow([mol_id, "*" if score is None else score])


def run_mmpdb(smi_path, props_path, out_dir, property_name, min_pairs):
    fragdb = os.path.join(out_dir, "pool.fragdb")
    mmpdb_path = os.path.join(out_dir, "pool.mmpdb")
    rules_out = os.path.join(out_dir, "transformations.csv")

    subprocess.run(["mmpdb", "fragment", smi_path, "-o", fragdb], check=True)
    subprocess.run(["mmpdb", "index", fragdb, "-o", mmpdb_path,
                    "--properties", props_path], check=True)
    # Note: mmpdb 3.1.4's `proprulecat` unconditionally passes
    # all_properties=True internally, so passing -p/--property always raises
    # "Cannot specify --property and --all-properties" (an upstream bug).
    # Workaround: omit -p -- since this database only ever has one loaded
    # property (property_name), "all properties" resolves to just that one.
    subprocess.run(["mmpdb", "proprulecat", mmpdb_path,
                    "--min-count", str(min_pairs),
                    "-o", rules_out], check=True)
    return rules_out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=DEFAULT_RESULTS,
                    help=f"GenMask results CSV (default: {DEFAULT_RESULTS}).")
    ap.add_argument("--receptor", required=True, help="Receptor PDB file.")
    ap.add_argument("--center", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"),
                    help="Docking box center in Angstroms (e.g. a co-crystallized ligand's centroid).")
    ap.add_argument("--box-size", type=float, default=22.0, dest="box_size",
                    help="Cubic box edge length in Angstroms (default 22).")
    ap.add_argument("--exhaustiveness", type=int, default=8, help="Vina search effort (default 8).")
    ap.add_argument("--num-modes", type=int, default=9, dest="num_modes",
                    help="Number of Vina output poses to consider (default 9).")
    ap.add_argument("--seed", type=int, default=0, help="Vina random seed (default 0).")
    ap.add_argument("--cpu", type=int, default=0, help="CPUs for Vina (default: autodetect).")
    ap.add_argument("--property-name", default="docking_score", dest="property_name",
                    help="Property name mmpdb reports on (default: docking_score).")
    ap.add_argument("--min-pairs", type=int, default=2, dest="min_pairs",
                    help="Only report transformations backed by at least this many pairs (default 2).")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help=f"Where to write intermediates + outputs (default: {DEFAULT_OUT_DIR}).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.results)
    id_by_smiles = pooled_smiles(df)
    print(f"Pooled {len(id_by_smiles)} unique molecules from {len(df)} rows.")

    print("Docking pooled molecules with Vina...")
    scores = dock.dock_at_centroid(
        args.receptor, list(id_by_smiles.keys()), center=args.center,
        box_size=args.box_size, exhaustiveness=args.exhaustiveness,
        num_modes=args.num_modes, seed=args.seed, cpu=args.cpu,
    )
    n_ok = sum(v is not None for v in scores.values())
    print(f"Docked {n_ok}/{len(scores)} molecules successfully.")

    smi_path = os.path.join(args.out_dir, "pool.smi")
    props_path = os.path.join(args.out_dir, "pool_props.csv")
    write_smi_file(id_by_smiles, smi_path)
    write_properties_file(id_by_smiles, scores, args.property_name, props_path)

    print("Running mmpdb fragment / index / proprulecat...")
    rules_out = run_mmpdb(smi_path, props_path, args.out_dir, args.property_name, args.min_pairs)

    df["input_docking_score"] = df["input_smiles"].map(scores)
    df["analogue_docking_score"] = df["analogue_smiles"].map(scores)
    df["delta_docking_score"] = df["analogue_docking_score"] - df["input_docking_score"]
    scored_out = os.path.join(args.out_dir, "results_scored.csv")
    df.to_csv(scored_out, index=False)

    print(f"\nPer-molecule scores:  {scored_out}")
    print(f"Transformation rules: {rules_out}")


if __name__ == "__main__":
    sys.exit(main())
