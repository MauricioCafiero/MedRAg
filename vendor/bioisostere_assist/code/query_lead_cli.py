#!/usr/bin/env python
"""
CLI for bioisostere_assist: the "new lead comes in" workflow. Loads a
prebuilt library (from build_library_cli.py) -- instant, no recomputation --
fragments the new lead molecule, embeds + shapes only the lead's own
fragments (fast: one molecule, not the whole pool), ranks every one of the
lead's substituent fragments against the entire library, and builds an
actual new analogue molecule for each top match by grafting the matched
library fragment onto the lead's core (Chem.molzip) -- plus a grid image of
the top hits.

Pass --receptor and --center to also dock the lead and every surviving
analogue (AutoDock Vina, CafChemDock.py) and report each analogue's
Δdocking-score relative to the lead. Vina affinities are negative; more
negative is more favorable, so a negative Δ means the analogue is predicted
to bind better than the lead.

Usage:
    python code/query_lead_cli.py --library library/library_100k_usr.pkl \\
        --lead "CC(=O)Nc1ccc(O)cc1" --top 15 --out outputs/lead_matches.csv \\
        --receptor receptors/HMGCR_1HWL.pdb --center 17.357 7.520 14.925
"""
import argparse
import csv
import os
import sys

from rdkit import Chem
from rdkit.Chem import Draw

import CafChemDock as dock
import CafChemShape as shape
import fragment as frag
import library as lib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "outputs", "lead_matches.csv")

# A match at or above this similarity just reconstructs the lead itself (or
# something indistinguishable from it under the shape descriptor) -- never
# a useful "new" analogue, so these are always dropped, not configurable.
MAX_USEFUL_SIMILARITY = 0.999


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", required=True, help="Path to a .pkl built by build_library_cli.py.")
    ap.add_argument("--lead", required=True, help="SMILES of the new lead molecule.")
    ap.add_argument("--max-frag-atoms", type=int, default=None, dest="max_frag_atoms",
                    help="Override the library's own fragment-size cutoff (default: match library).")
    ap.add_argument("--n-confs", type=int, default=None, dest="n_confs",
                    help="Override the library's own conformer count (default: match library).")
    ap.add_argument("--energy-window", type=float, default=None, dest="energy_window",
                    help="Override the library's own energy window (default: match library).")
    ap.add_argument("--prune-rms-thresh", type=float, default=None, dest="prune_rms_thresh",
                    help="Override the library's own RMSD-pruning threshold (default: match library).")
    ap.add_argument("--top", type=int, default=15, help="Top matches per lead fragment to report (default: 15).")
    ap.add_argument("--image-top", type=int, default=12, dest="image_top",
                    help="Number of top-ranked analogues (across all lead fragments) to draw in the "
                         "grid image (default: 12).")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"Output CSV path (default: {DEFAULT_OUT}).")
    ap.add_argument("--receptor", default=None,
                    help="Receptor PDB file -- if given (with --center), also dock the lead and "
                         "every surviving analogue with AutoDock Vina.")
    ap.add_argument("--center", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
                    help="Docking box center in Angstroms (e.g. a co-crystallized ligand's centroid).")
    ap.add_argument("--box-size", type=float, default=22.0, dest="box_size",
                    help="Cubic docking box edge length in Angstroms (default: 22.0).")
    ap.add_argument("--exhaustiveness", type=int, default=8, help="Vina search effort (default: 8).")
    ap.add_argument("--num-modes", type=int, default=9, dest="num_modes",
                    help="Vina output poses considered (default: 9).")
    ap.add_argument("--dock-seed", type=int, default=0, dest="dock_seed", help="Vina random seed (default: 0).")
    ap.add_argument("--cpu", type=int, default=0, help="CPUs for Vina (default: autodetect).")
    args = ap.parse_args()

    # Python block-buffers stdout when it isn't a tty, so on a long docking run
    # every progress line sits unflushed until the process exits -- redirected
    # to a log file, that looks identical to a hung run.
    sys.stdout.reconfigure(line_buffering=True)

    do_dock = args.receptor is not None and args.center is not None

    print(f"Loading library from {args.library}...")
    method, lib_params, library_fragments = lib.load_library(args.library)
    print(f"Library: {len(library_fragments)} fragments, method={method}, built with {lib_params}")

    # Libraries built before radical filtering existed can still contain
    # fragments like [CH2][*:1]; drop them here so an old .pkl stays usable.
    # Filtering at load (not at ranking) means a rejected fragment's slot goes
    # to the next real match instead of leaving a blank analogue in the output.
    radical_frags = [s for s in library_fragments
                     if (m := Chem.MolFromSmiles(s)) is not None and frag.has_radical(m)]
    for s in radical_frags:
        del library_fragments[s]
    if radical_frags:
        print(f"Dropped {len(radical_frags)} radical-bearing fragment(s) from the library: "
              f"{', '.join(radical_frags)}")

    max_frag_atoms = args.max_frag_atoms if args.max_frag_atoms is not None else lib_params["max_frag_atoms"]
    n_confs = args.n_confs if args.n_confs is not None else lib_params["n_confs"]
    energy_window = args.energy_window if args.energy_window is not None else lib_params["energy_window"]
    prune_rms_thresh = args.prune_rms_thresh if args.prune_rms_thresh is not None else lib_params["prune_rms_thresh"]

    lead_pool = frag.fragments_with_cores_from_smiles(args.lead, max_frag_atoms)
    print(f"\nLead fragmented into {len(lead_pool)} substituent-like fragments")

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for lead_frag_smi, lead_core_smi in lead_pool:
        d = lib.shape_for_fragment(lead_frag_smi, method, n_confs, energy_window, prune_rms_thresh)
        if d is None:
            print(f"\n[FAILED to embed/shape] {lead_frag_smi}")
            continue

        ranked = [(shape.best_similarity(d, lib_d), lib_smi, lib_label)
                 for lib_smi, (lib_d, lib_label) in library_fragments.items()]
        ranked = [r for r in ranked if r[0] < MAX_USEFUL_SIMILARITY]
        ranked.sort(reverse=True)

        print(f"\nLead fragment: {lead_frag_smi}")
        print(f"Top {min(args.top, len(ranked))} library matches:")
        for sim, lib_smi, lib_label in ranked[:args.top]:
            analogue_smi = frag.build_analogue(lead_core_smi, lib_smi)
            print(f"  {sim:.3f}  {lib_smi:20s} (from {lib_label})  -> {analogue_smi}")
            rows.append([lead_frag_smi, lib_smi, lib_label, f"{sim:.4f}", analogue_smi])

    dock_scores = {}
    lead_dock_score = None
    if do_dock:
        print(f"\nDocking lead + {len({r[4] for r in rows if r[4]})} unique analogues "
             f"into {args.receptor} at {tuple(args.center)}...")
        to_dock = [args.lead] + [r[4] for r in rows if r[4]]
        dock_scores = dock.dock_at_centroid(
            args.receptor, to_dock, center=args.center, box_size=args.box_size,
            exhaustiveness=args.exhaustiveness, num_modes=args.num_modes,
            seed=args.dock_seed, cpu=args.cpu,
        )
        lead_dock_score = dock_scores.get(args.lead)
        print(f"Lead docking score: {lead_dock_score}")

    header = ["lead_fragment", "library_fragment", "library_source", "similarity", "analogue_smiles"]
    if do_dock:
        header += ["docking_score", "delta_docking_score"]

    out_rows = []
    for row in rows:
        analogue_smi = row[4]
        if do_dock:
            aff = dock_scores.get(analogue_smi) if analogue_smi else None
            delta = (aff - lead_dock_score) if (aff is not None and lead_dock_score is not None) else None
            row = row + [aff, delta]
        out_rows.append(row)

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(out_rows)
    print(f"\nWrote {len(out_rows)} matches to {args.out}")

    image_rows = [r for r in out_rows if r[4]]
    image_rows.sort(key=lambda r: float(r[3]), reverse=True)
    image_rows = image_rows[:args.image_top]

    mols, legends = [], []
    for row in image_rows:
        lead_frag_smi, lib_smi, lib_label, sim, analogue_smi = row[:5]
        m = Chem.MolFromSmiles(analogue_smi)
        if m is None:
            continue
        mols.append(m)
        legend = f"sim={sim} ({lead_frag_smi} -> {lib_smi})"
        if do_dock:
            aff, delta = row[5], row[6]
            legend += f"\ndock={aff} (Δ={delta:+.2f})" if delta is not None else "\ndock=failed"
        legends.append(legend)

    if mols:
        img_path = os.path.splitext(args.out)[0] + ".png"
        img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(300, 300),
                                   legends=legends, returnPNG=False)
        img.save(img_path)
        print(f"Wrote top-{len(mols)} analogue grid image to {img_path}")


if __name__ == "__main__":
    sys.exit(main())
