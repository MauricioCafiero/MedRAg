"""Build binder (active) + property-matched decoy sets for human SULT1A1 / SULT1A3.

Actives are pooled from three public sources and deduplicated by InChIKey:

  * Martiny et al., PLoS ONE 2013, 8(9):e73587 (doi:10.1371/journal.pone.0073587)
    supplementary SDFs -- clustered, structurally diverse binders (substrates or
    inhibitors) curated from BRENDA, Aureus, TOXNET, PubChem and the literature:
    60 for SULT1A1 (S1), 50 for SULT1A3 (S2). Fetched from Europe PMC.
  * ChEMBL activities for the human targets CHEMBL1743291 (1A1) / CHEMBL1743293 (1A3).
  * BindingDB ligands by UniProt accession (P50225 / P0DMM9).

Decoys are chosen DUD-E style: physico-chemically matched to an active
(MW, cLogP, HBD, HBA, rotatable bonds, net charge) but topologically dissimilar
(Morgan/ECFP4 Tanimoto below a cutoff to *every* active), drawn from ZINC20
2D tranches downloaded from files.docking.org.

Usage:
    python code/build_sult_dataset.py --target both --decoys-per-active 50
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.DataStructs import BulkTanimotoSimilarity

RDLogger.DisableLog("rdApp.*")

TARGETS = {
    "sult1a1": {"sdf": "pone.0073587.s001.sdf", "chembl": "CHEMBL1743291", "uniprot": "P50225"},
    "sult1a3": {"sdf": "pone.0073587.s002.sdf", "chembl": "CHEMBL1743293", "uniprot": "P0DMM9"},
    # SULT1E1 (estrogen sulfotransferase) -- the usual off-target, handy as a selectivity set
    "sult1e1": {"sdf": "pone.0073587.s003.sdf", "chembl": "CHEMBL2346", "uniprot": "P49888"},
}
SUPPL_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC3765257/supplementaryFiles"

# ZINC20 2D tranches: <MW bin><logP bin><reactivity><purchasability>.
# MW  A<=200 B<=250 C<=300 D<=325 E<=350 F<=375 G<=400 H<=425 I<=450
# logP A<=-1 B<=0 C<=1 D<=2 E<=2.5 F<=3 G<=3.5 H<=4 I<=4.5 J<=5
# "AA" = anodyne + most-available. This window brackets the SULT1A binders (MW 150-520, cLogP 0-4.5).
MW_BINS = "ABCDEFGHI"   # up to 450 Da
LOGP_BINS = "ABCDEFGHIJ"  # up to cLogP 5

# Groups a sulfotransferase actually conjugates -- optional extra filter on decoys.
SULFONATABLE = [Chem.MolFromSmarts(s) for s in (
    "[OX2H][c]",                      # phenol / catechol
    "[OX2H][CX4]",                    # aliphatic hydroxyl
    "[NX3;H2,H1;!$(NC=O)][c]",        # aryl amine
    "[OX2H][NX3]",                    # N-hydroxy
)]

_uncharger = rdMolStandardize.Uncharger()
_chooser = rdMolStandardize.LargestFragmentChooser()
_normalizer = rdMolStandardize.Normalizer()
_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def standardize(mol: Chem.Mol) -> str | None:
    """Largest fragment, normalized functional groups, neutralized -> canonical SMILES."""
    try:
        mol = _normalizer.normalize(_chooser.choose(mol))
        mol = _uncharger.uncharge(Chem.RemoveHs(mol))
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def from_molblock(block: str) -> Chem.Mol | None:
    """Parse an SDF record, repairing the neutral tetravalent nitro nitrogens used in the
    supplementary files (N(-O)(=O) with both oxygens uncharged), which RDKit rejects."""
    mol = Chem.MolFromMolBlock(block, sanitize=False)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0 or atom.GetExplicitValence() <= 3:
            continue
        # the terminal oxygen on the *single* bond is the one carrying the negative charge
        bonds = sorted(atom.GetBonds(), key=lambda b: b.GetBondTypeAsDouble())
        for bond in bonds:
            other = bond.GetOtherAtom(atom)
            if other.GetSymbol() != "O" or other.GetDegree() != 1 or other.GetFormalCharge() != 0:
                continue
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                bond.SetBondType(Chem.BondType.SINGLE)
            other.SetNoImplicit(True)
            other.SetNumExplicitHs(0)
            other.SetFormalCharge(-1)
            atom.SetFormalCharge(1)
            atom.SetNoImplicit(True)
            break
        else:
            # not a nitro: a tetravalent neutral N is a drawn-as-neutral ammonium
            if atom.GetExplicitValence() == 4:
                atom.SetFormalCharge(1)
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def props(mol: Chem.Mol) -> tuple[float, float, int, int, int, int]:
    return (
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Chem.GetFormalCharge(mol),
    )


# files.docking.org rejects the default python-requests user agent with a 403
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; GenMask-dataset-builder)"}


def get_json(url: str, tries: int = 4) -> dict:
    """REST fetch with a retry -- ChEMBL and BindingDB intermittently answer with an
    HTML error page instead of JSON."""
    for attempt in range(tries):
        try:
            resp = requests.get(url, timeout=180, headers=HEADERS)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt == tries - 1:
                raise
            wait = 5 * (attempt + 1)
            print(f"  {type(exc).__name__} on {url.split('?')[0]}; retrying in {wait}s")
            time.sleep(wait)
    return {}


def cached_download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {url}")
        resp = requests.get(url, timeout=300, headers=HEADERS)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path


def cached_records(path: Path, fetch, refresh: bool = False) -> list[dict]:
    """Return records from the JSON cache, fetching only when it is missing (or when
    --refresh-sources is given). ChEMBL and BindingDB both have frequent outages, and
    these sets are static -- there is nothing to gain from re-fetching them."""
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    try:
        recs = fetch()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(recs, indent=1))
        return recs
    except Exception as exc:
        if path.exists():
            print(f"  {type(exc).__name__}: falling back to cached {path.name}")
            return json.loads(path.read_text())
        print(f"  {type(exc).__name__}: {path.stem} unavailable and not cached; skipping")
        return []


def load_martiny(target: str, cache: Path) -> list[dict]:
    zip_path = cached_download(SUPPL_URL, cache / "martiny2013_suppl.zip")
    name = TARGETS[target]["sdf"]
    with zipfile.ZipFile(zip_path) as zf:
        raw = zf.read(name).decode("utf8", "ignore")
    out = []
    for i, block in enumerate(raw.split("$$$$\n")):
        if not block.strip():
            continue
        mol = from_molblock(block)
        smi = standardize(mol) if mol else None
        if smi:
            out.append({"smiles": smi, "source": "Martiny2013", "source_id": f"{name}:{i + 1}", "activity": ""})
    return out


def load_chembl(target: str, cache: Path, refresh: bool = False) -> list[dict]:
    return cached_records(cache / f"chembl_{target}.json", lambda: _fetch_chembl(target), refresh)


def _fetch_chembl(target: str) -> list[dict]:
    tid = TARGETS[target]["chembl"]
    url = f"https://www.ebi.ac.uk/chembl/api/data/activity?target_chembl_id={tid}&format=json&limit=1000"
    acts = get_json(url)["activities"]
    best: dict[str, dict] = {}
    for a in acts:
        smi = a.get("canonical_smiles")
        if not smi:
            continue
        note = f"{a['standard_type']}={a.get('standard_value')}{a.get('standard_units') or ''}"
        best.setdefault(a["molecule_chembl_id"], {"smiles": smi, "source": "ChEMBL",
                                                  "source_id": a["molecule_chembl_id"], "activity": note})
    out = []
    for rec in best.values():
        mol = Chem.MolFromSmiles(rec["smiles"])
        smi = standardize(mol) if mol else None
        if smi:
            rec["smiles"] = smi
            out.append(rec)
    return out


def load_bindingdb(target: str, cache: Path, refresh: bool = False) -> list[dict]:
    return cached_records(cache / f"bindingdb_{target}.json", lambda: _fetch_bindingdb(target), refresh)


def _fetch_bindingdb(target: str) -> list[dict]:
    acc = TARGETS[target]["uniprot"]
    url = f"https://bindingdb.org/rest/getLigandsByUniprots?uniprot={acc}&code=0&response=application/json"
    rows = get_json(url, tries=2).get("getLindsByUniprotsResponse", {}).get("affinities", [])
    if isinstance(rows, dict):
        rows = [rows]
    out = []
    for r in rows:
        mol = Chem.MolFromSmiles(r["smile"].split(" |")[0])
        smi = standardize(mol) if mol else None
        if smi:
            out.append({"smiles": smi, "source": "BindingDB", "source_id": r.get("monomerid", ""),
                        "activity": f"{r.get('affinity_type')}={r.get('affinity')}nM"})
    return out


def collect_actives(target: str, cache: Path, refresh: bool = False) -> list[dict]:
    seen: dict[str, dict] = {}
    for loader in (load_martiny(target, cache), load_chembl(target, cache, refresh),
                   load_bindingdb(target, cache, refresh)):
        for rec in loader:
            key = Chem.MolToInchiKey(Chem.MolFromSmiles(rec["smiles"]))
            if key in seen:  # keep first source, note the corroborating one
                seen[key]["source"] += "|" + rec["source"]
            else:
                rec["inchikey"] = key
                seen[key] = rec
    return list(seen.values())


def load_background(cache: Path, per_tranche: int, seed: int) -> list[dict]:
    """Download + property-annotate the ZINC tranche window (cached as a CSV).

    A random `per_tranche` sample is taken from each tranche file so the background
    spans the whole MW/logP window instead of being dominated by the biggest tranches.
    """
    prop_csv = cache / f"zinc_background_props_{per_tranche}.csv"
    if prop_csv.exists():
        with prop_csv.open() as fh:
            rows = [dict(r) for r in csv.DictReader(fh)]
        for r in rows:
            r["mw"], r["logp"] = float(r["mw"]), float(r["logp"])
            for k in ("hbd", "hba", "rotb", "charge"):
                r[k] = int(r[k])
        print(f"  reusing {len(rows)} background molecules from {prop_csv}")
        return rows

    rng = random.Random(seed)
    rows, seen = [], set()
    for mw in MW_BINS:
        for lp in LOGP_BINS:
            name = f"{mw}{lp}AA"
            path = cache / "zinc" / f"{name}.smi"
            try:
                cached_download(f"https://files.docking.org/2D/{mw}{lp}/{name}.smi", path)
            except requests.HTTPError:
                continue
            lines = path.read_text().splitlines()[1:]
            if len(lines) > per_tranche:
                lines = rng.sample(lines, per_tranche)
            for line in lines:
                parts = line.split()
                if len(parts) < 2:
                    continue
                mol = Chem.MolFromSmiles(parts[0])
                smi = standardize(mol) if mol else None
                if not smi or smi in seen:
                    continue
                seen.add(smi)
                p = props(Chem.MolFromSmiles(smi))
                rows.append({"smiles": smi, "zinc_id": parts[1], "tranche": name, "mw": p[0], "logp": p[1],
                             "hbd": p[2], "hba": p[3], "rotb": p[4], "charge": p[5]})
            print(f"  {name}: background now {len(rows)} molecules")
    if not rows:
        sys.exit("No background molecules downloaded -- is files.docking.org reachable?")
    prop_csv.parent.mkdir(parents=True, exist_ok=True)
    with prop_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def pick_decoys(actives: list[dict], background: list[dict], n_per_active: int,
                tanimoto_cutoff: float, drop_sulfonatable: bool, seed: int) -> list[dict]:
    rng = random.Random(seed)
    active_fps = [_fpgen.GetFingerprint(Chem.MolFromSmiles(a["smiles"])) for a in actives]
    bg_props = np.array([[b["mw"], b["logp"], b["hbd"], b["hba"], b["rotb"], b["charge"]] for b in background])
    available = np.ones(len(background), dtype=bool)
    decoys: list[dict] = []
    # windows are widened when an active is too exotic to match tightly
    windows = [(25, 1.0, 1, 2, 2), (50, 1.5, 2, 3, 3), (100, 2.0, 3, 4, 4)]

    for active in actives:
        amw, alogp, ahbd, ahba, arotb, acharge = props(Chem.MolFromSmiles(active["smiles"]))
        picked: list[dict] = []
        for dmw, dlogp, dhbd, dhba, drotb in windows:
            mask = (available
                    & (np.abs(bg_props[:, 0] - amw) <= dmw)
                    & (np.abs(bg_props[:, 1] - alogp) <= dlogp)
                    & (np.abs(bg_props[:, 2] - ahbd) <= dhbd)
                    & (np.abs(bg_props[:, 3] - ahba) <= dhba)
                    & (np.abs(bg_props[:, 4] - arotb) <= drotb)
                    & (bg_props[:, 5] == acharge))
            idx = list(np.flatnonzero(mask))
            rng.shuffle(idx)
            for i in idx:
                if len(picked) >= n_per_active:
                    break
                cand = background[i]
                cmol = Chem.MolFromSmiles(cand["smiles"])
                if drop_sulfonatable and any(cmol.HasSubstructMatch(p) for p in SULFONATABLE):
                    available[i] = False
                    continue
                if max(BulkTanimotoSimilarity(_fpgen.GetFingerprint(cmol), active_fps)) >= tanimoto_cutoff:
                    available[i] = False
                    continue
                available[i] = False
                picked.append({"smiles": cand["smiles"], "zinc_id": cand["zinc_id"], "tranche": cand["tranche"],
                               "matched_active": active["source_id"], "label": 0})
            if len(picked) >= n_per_active:
                break
        if len(picked) < n_per_active:
            print(f"  only {len(picked)}/{n_per_active} decoys for {active['source_id']} ({active['smiles']})")
        decoys.extend(picked)
    return decoys


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {len(rows)} rows -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["sult1a1", "sult1a3", "sult1e1", "both"], default="both",
                    help='"both" means the two SULT1A isoforms; sult1e1 is available as a selectivity set')
    ap.add_argument("--decoys-per-active", type=int, default=50)
    ap.add_argument("--tanimoto-cutoff", type=float, default=0.35,
                    help="reject a decoy whose ECFP4 Tanimoto to ANY active reaches this")
    ap.add_argument("--drop-sulfonatable", action="store_true",
                    help="also reject decoys bearing a phenol/alcohol/aryl-amine (a real SULT handle) -- "
                         "fewer latent actives, but an easier and more biased benchmark")
    ap.add_argument("--per-tranche", type=int, default=8000,
                    help="molecules sampled from each ZINC tranche file (background size ~= 90 x this)")
    ap.add_argument("--refresh-sources", action="store_true",
                    help="re-fetch ChEMBL/BindingDB instead of using the cached JSON pulls")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--cache-dir", default="data/cache")
    args = ap.parse_args()

    out_dir, cache = Path(args.out_dir), Path(args.cache_dir)
    targets = ["sult1a1", "sult1a3"] if args.target == "both" else [args.target]

    print("Collecting actives...")
    actives = {t: collect_actives(t, cache, args.refresh_sources) for t in targets}
    for t, recs in actives.items():
        by_source: dict[str, int] = {}
        for r in recs:
            by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        print(f"  {t}: {len(recs)} unique binders {by_source}")

    print("Loading ZINC background...")
    background = load_background(cache, args.per_tranche, args.seed)
    print(f"  {len(background)} background molecules")

    for t in targets:
        print(f"Selecting decoys for {t}...")
        recs = sorted(actives[t], key=lambda r: r["source_id"])
        for r in recs:
            r["label"] = 1
            r["target"] = t
        decoys = pick_decoys(recs, background, args.decoys_per_active,
                             args.tanimoto_cutoff, args.drop_sulfonatable, args.seed)
        write_csv(out_dir / f"{t}_actives.csv", recs)
        write_csv(out_dir / f"{t}_decoys.csv", decoys)
        combined = [{"smiles": r["smiles"], "id": r["source_id"], "label": 1} for r in recs]
        combined += [{"smiles": d["smiles"], "id": d["zinc_id"], "label": 0} for d in decoys]
        write_csv(out_dir / f"{t}_benchmark.csv", combined)


if __name__ == "__main__":
    main()
