#!/usr/bin/env python
"""
CLI for bioisostere_assist: build a bioisostere fragment library from a pool
of molecules. Fragments each one at single bonds (RDKit rdMMPA), embeds a
conformer ensemble for every resulting substituent-like fragment, and
computes an ODDT 3D shape descriptor for every surviving conformer.

Saves BOTH a human-readable CSV (fragment identity + source) and a pickle
containing the actual descriptor vectors -- the pickle is what makes this a
reusable library rather than a one-off report: query_lead_cli.py loads it
directly, without re-embedding/re-shaping the whole pool again.

No OpenEye/BROOD anywhere -- RDKit for fragmentation/conformers, ODDT for
shape descriptors, both open-source (see README.md for why oddt is
vendored). The similarity measure is swappable via --method: 'usr' (pure
shape), 'usr_cat' (shape + pharmacophore atom typing), or 'electroshape'
(shape + chirality + electrostatics) -- see CafChemShape.SHAPE_METHODS.

Usage:
    python code/build_library_cli.py --csv data/human_druglike_100k.smi \\
        --method usr --out outputs/library_100k_usr
"""
import argparse
import csv
import os
import sys

import CafChemShape as shape
import fragment as frag
import library as lib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(REPO_ROOT, "data", "human_druglike_100k.smi")
DEFAULT_OUT = os.path.join(REPO_ROOT, "outputs", "library")


def load_seeds(csv_path, n_mols):
    seeds = []
    with open(csv_path) as fh:
        for i, line in enumerate(fh):
            if n_mols is not None and i >= n_mols:
                break
            parts = line.split()
            if not parts:
                continue
            smi = parts[0]
            label = parts[1] if len(parts) > 1 else f"mol{i}"
            seeds.append((smi, label))
    return seeds


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help=f"Whitespace-separated SMILES [ID] file, one molecule per line "
                         f"(default: {DEFAULT_CSV}).")
    ap.add_argument("--n-mols", type=int, default=None, dest="n_mols",
                    help="Limit to the first N seed molecules (default: all).")
    ap.add_argument("--method", default="usr", choices=list(shape.SHAPE_METHODS),
                    help="Shape-similarity measure (default: usr). "
                         "usr = pure shape; usr_cat = shape + pharmacophore atom typing; "
                         "electroshape = shape + chirality + electrostatics.")
    ap.add_argument("--max-frag-atoms", type=int, default=frag.MAX_FRAG_HEAVY_ATOMS,
                    dest="max_frag_atoms",
                    help=f"Max heavy atoms (incl. dummy) for a fragment to be kept as "
                         f"substituent-like (default: {frag.MAX_FRAG_HEAVY_ATOMS}).")
    ap.add_argument("--n-confs", type=int, default=shape.N_CONFORMERS, dest="n_confs",
                    help=f"Conformers to embed per fragment before pruning (default: {shape.N_CONFORMERS}).")
    ap.add_argument("--energy-window", type=float, default=shape.ENERGY_WINDOW, dest="energy_window",
                    help=f"Discard conformers more than this many kcal/mol above the "
                         f"lowest-energy one (default: {shape.ENERGY_WINDOW}).")
    ap.add_argument("--prune-rms-thresh", type=float, default=shape.PRUNE_RMS_THRESH,
                    dest="prune_rms_thresh",
                    help=f"RMSD (Angstrom) below which a newly embedded conformer is "
                         f"treated as a duplicate and dropped (default: {shape.PRUNE_RMS_THRESH}).")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"Output path stem (default: {DEFAULT_OUT}) -- writes "
                         f"<out>.csv (human-readable) and <out>.pkl (the actual library, "
                         f"loaded by query_lead_cli.py).")
    args = ap.parse_args()

    # Python block-buffers stdout when it isn't a tty, so on a long library
    # build every progress line sits unflushed until the process exits --
    # redirected to a log file, that looks identical to a hung run.
    sys.stdout.reconfigure(line_buffering=True)

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading seeds from {args.csv} (n_mols={args.n_mols or 'all'})...")
    seeds = load_seeds(args.csv, args.n_mols)
    print(f"Loaded {len(seeds)} seed molecules")

    pool = frag.build_fragment_pool(seeds, args.max_frag_atoms)
    print(f"Fragmented: {len(pool)} unique substituent-like fragments (<= {args.max_frag_atoms} heavy atoms)")

    descriptors = {}
    n_fail = 0
    for i, (smi, label) in enumerate(pool):
        d = lib.shape_for_fragment(smi, args.method, args.n_confs, args.energy_window, args.prune_rms_thresh)
        if d is not None:
            descriptors[smi] = (d, label)
        else:
            n_fail += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(pool)} fragments processed")
    print(f"Shaped: {len(descriptors)} ok, {n_fail} failed (method={args.method})")

    csv_path = args.out + ".csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fragment_smiles", "source_label", "n_conformers"])
        for smi, (d, label) in descriptors.items():
            w.writerow([smi, label, len(d)])
    print(f"\nWrote fragment manifest to {csv_path}")

    params = {
        "max_frag_atoms": args.max_frag_atoms,
        "n_confs": args.n_confs,
        "energy_window": args.energy_window,
        "prune_rms_thresh": args.prune_rms_thresh,
    }
    pkl_path = args.out + ".pkl"
    lib.save_library(pkl_path, args.method, params, descriptors)
    print(f"Wrote library (with descriptor vectors) to {pkl_path}")


if __name__ == "__main__":
    sys.exit(main())
