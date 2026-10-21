"""Bound drug-like ligands of PDB files, found with dock_assist's own find_ligand_centroids (medrag/lib/dock_assist): its text report is
parsed, then crystallization additives, ions, cofactors and ligands far from the protein are dropped. A separate module from
discover_adapter because dock_assist and MoDrAg_CLI both ship a `modrag_protein_functions` and cannot share a process."""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib/dock_assist"))
import modrag_protein_functions as dock_assist_funcs        # noqa: E402  (the dock_assist copy)

# crystallization additives, buffers, ions and cofactors: bound to many proteins, never the molecule being optimised
NOT_A_LEAD = set("""HOH WAT DOD GOL EDO PEG PGE PG4 1PE P6G 2PE PE3 PE4 SO4 PO4 ACT ACY FMT DMS MPD MRD TRS MES EPE CIT TLA BME IPA
CL NA K MG ZN CA MN FE FE2 CU NI CO CD BR IOD NO3 SCN AZI PEO EOH MOH DTT TAR FLC OGA IMD BCT CO3 NH4 OXY SEP TPO PTR MLI
ADP ATP AMP ANP GDP GTP GNP NAD NAP NDP NAI FAD FMN HEM HEC SAM SAH COA ACO PAP A3P UDP UTP CTP TTP""".split())

_ENTRY = re.compile(r"\[\d+\] .+? \((\w+), chain (\w*)\): (\d+) heavy atoms\s+centroid: \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)\s+QED: ([\d.]+|n/a)")


def ligands(pdb_path: str, min_heavy: int = 10, max_heavy: int = 70, max_dist: float = 7.0) -> list:
    """Bound ligand occurrences [{resname, chain, heavy, qed, centroid}] from find_ligand_centroids, best first (QED, then size).
    Dropped: additives/cofactors, ligands outside the 10-70 heavy-atom range, and ligands whose centroid is more than `max_dist` A
    from any protein atom (not in a binding site)."""
    from scipy.spatial import cKDTree
    prot = [[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in open(pdb_path) if l.startswith("ATOM")]
    if not prot:
        return []
    tree = cKDTree(np.array(prot))
    out = []
    for sym, chain, heavy, cx, cy, cz, qed in _ENTRY.findall(dock_assist_funcs.find_ligand_centroids(pdb_path)):
        c, heavy = [float(cx), float(cy), float(cz)], int(heavy)
        if sym in NOT_A_LEAD or not (min_heavy <= heavy <= max_heavy) or tree.query(c)[0] > max_dist:
            continue
        out.append(dict(resname=sym, chain=chain, heavy=heavy, qed=None if qed == "n/a" else float(qed), centroid=c))
    return sorted(out, key=lambda r: (-(r["qed"] or 0.0), -r["heavy"]))
