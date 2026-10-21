"""Find structures for a protein name with MoDrAg's tools (vendor/MoDrAg_CLI): UniProt entry -> RCSB search -> PDB files,
each checked to really be that protein (DBREF) and annotated with its resolution. Runs with cwd = <run>/discover/code, so
MoDrAg's ../pdb_files and ../scratch land in the run directory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "vendor/MoDrAg_CLI/code"))
import torch  # noqa: F401,E402  (MoDrAg imports torch before the native chem libs)


def _dbref_accessions(pdb_path: str) -> set:
    acc = set()
    for line in open(pdb_path):
        if line.startswith("DBREF"):
            f = line.split()
            if "UNP" in f and f.index("UNP") + 1 < len(f):
                acc.add(f[f.index("UNP") + 1])
    return acc


def _rcsb_entries_by_uniprot(accession: str, rows: int = 60) -> list:
    """PDB entry ids of a UniProt accession that contain bound small molecules, best resolution first (RCSB search API, the same
    service MoDrAg's find_PDBID_node uses; a name search is too loose to find the right protein reliably)."""
    import requests
    ident = "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers"
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
            {"type": "terminal", "service": "text", "parameters": {"attribute": f"{ident}.database_accession", "operator": "exact_match", "value": accession}},
            {"type": "terminal", "service": "text", "parameters": {"attribute": f"{ident}.database_name", "operator": "exact_match", "value": "UniProt"}},
            {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_entry_info.nonpolymer_entity_count", "operator": "greater", "value": 0}}]},
         "return_type": "entry",
         "request_options": {"paginate": {"start": 0, "rows": rows}, "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}]}}
    r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q, timeout=60)
    if r.status_code == 200:
        return [h["identifier"] for h in r.json().get("result_set", [])]
    from rcsbapi.search import AttributeQuery             # MoDrAg's own RCSB library, unsorted but always valid
    return list(AttributeQuery(f"{ident}.database_accession", "exact_match", accession)())


def candidates(protein_name: str, max_pdbs: int = 10) -> dict:
    """{uniprot, gene, pdbs: [{pdb_id, path, resolution, title, same_protein}]} for a protein name."""
    import pandas as pd
    import requests
    from modrag_protein_functions import find_PDBID_node, get_pdb_file, uniprot_node

    id_lists, _, _ = uniprot_node([protein_name], human_flag=True)
    ids = id_lists[0] if id_lists else []
    primary = ""
    if ids:
        df = pd.read_csv(f"../scratch/{protein_name}_uniprot_ids.tsv", sep="\\t")
        exact = [r["Entry"] for _, r in df.iterrows() if r["Entry"] in ids and protein_name.upper() in str(r["Gene Names"]).upper().split()]
        primary = (exact or ids)[0]
    by_accession = _rcsb_entries_by_uniprot(primary) if primary else []
    order = list(by_accession)
    if len(order) < max_pdbs:                         # too few by accession: add MoDrAg's name search (checked via DBREF below)
        for lst in find_PDBID_node([protein_name] + ([primary] if primary else []))[0]:
            order += [p for p in lst if p not in order]
    resolved = {}                                     # accession -> primary accession (secondary ids such as P50224 -> P0DMM9)

    def same_protein(path):
        accs = _dbref_accessions(path)
        if primary in accs:
            return True
        for a in accs - set(resolved):
            r, _, _ = uniprot_node([a])
            resolved[a] = r[0][0] if r and r[0] else a
        return any(resolved.get(a) == primary for a in accs)

    out = []
    for pdb in order[:max_pdbs * 2]:
        if len(out) >= max_pdbs:
            break
        get_pdb_file(pdb, protein_name)
        path = str(Path(f"../pdb_files/{protein_name}_{pdb}.pdb").resolve())
        if not Path(path).is_file():
            continue
        try:
            e = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb}", timeout=30).json()
            res = (e.get("rcsb_entry_info", {}).get("resolution_combined") or [None])[0]
            title = e.get("struct", {}).get("title", "")
        except Exception:
            res, title = None, ""
        out.append(dict(pdb_id=pdb, path=path, resolution=res, title=title, same_protein=pdb in by_accession or (bool(primary) and same_protein(path))))
    return dict(uniprot=primary, protein=protein_name, searched=order, pdbs=out)


def ligand_smiles(resname: str) -> str:
    """Canonical SMILES of a PDB chemical component (RCSB chem-comp record); '' if unavailable."""
    import requests
    d = requests.get(f"https://data.rcsb.org/rest/v1/core/chemcomp/{resname}", timeout=30).json()
    desc = d.get("rcsb_chem_comp_descriptor", {})
    return desc.get("SMILES_stereo") or desc.get("SMILES") or ""
