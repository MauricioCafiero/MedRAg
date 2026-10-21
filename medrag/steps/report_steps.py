"""HTML report sections for the first half of the pipeline (generate, filter), from a RunState."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from medrag.report import Report
from medrag.state import RunState


def add_generate_section(rep: Report, state: RunState, max_grid: int = 24):
    cands = state.candidates
    by_source = Counter(src["source"] for r in cands.values() for src in r["sources"])
    table = [dict(source=k, candidates=v) for k, v in sorted(by_source.items())]
    shown = [r for r in cands.values() if r["smiles"] != state.data["seed"]][:max_grid]
    legend = lambda r: "+".join(sorted({s["source"] for s in r["sources"]}))
    rep.add_step("generate", "Generate", f"Seed {state.data['seed']}: {len(cands) - 1} unique analogues proposed.",
                 table=table, molecules=[dict(smiles=r["smiles"], legend=legend(r)) for r in shown])


def add_filter_section(rep: Report, state: RunState, max_grid: int = 24):
    rows = []
    for smi, r in state.candidates.items():
        f = r.get("filter")
        if not f or "props" not in f:
            continue
        p, a, dv = f["props"], f.get("admet") or {}, f.get("derived") or {}
        g = lambda k: float(a.get(k, "nan"))
        rows.append(dict(species=r["species"], status=f["status"], qed=p["qed"], mw=p["mw"], logp=p["logp"],
                         sim_to_seed=p["tanimoto_to_seed"], hERG=g("hERG"), AMES=g("AMES"), DILI=g("DILI"),
                         carcinogen=g("Carcinogens_Lagunin"), skin=g("Skin_Reaction"),
                         LD50_mg_kg=dv.get("LD50_mg_kg", float("nan")), CYP_inhib=",".join(dv.get("CYP_inhibited", [])) or "-",
                         F30=g("Bioavailability_Ma"), BBB=g("BBB_Martins"), flags="; ".join(f["reasons"])))
    order = {"pass": 0, "warn": 1, "fail": 2}
    rows.sort(key=lambda x: (order[x["status"]], -x["qed"]))
    tally = Counter(x["status"] for x in rows)
    ok = [x for x in rows if x["status"] != "fail"][:max_grid]
    rep.add_step("filter", "Filter",
                 f"Protonated at pH 7.4, then QED / Lipinski / alerts / ADMET-AI. {tally['pass']} pass, {tally['warn']} warn, "
                 f"{tally['fail']} fail (excluding the seed and duplicates). Cut-offs are provisional.",
                 table=rows, molecules=[dict(smiles=x["species"], legend=f"{x['status']} qed {x['qed']:.2f}") for x in ok])


def add_dock_section(rep: Report, state: RunState, top: int = 6):
    rows, shown = [], []
    for smi, r in state.candidates.items():
        d = r.get("dock")
        if not d:
            continue
        if d.get("status") != "ok":
            rows.append(dict(species=r.get("species", smi), vina="n/a", in_site="n/a", note=d.get("detail", d["status"])))
            continue
        p0 = d["poses"][0]
        j0 = d["in_site"][0] if d["in_site"] else None
        rows.append(dict(species=d["species"], vina=d["best_score"], best_pose_dist_A=p0["centroid_dist"],
                         best_pose_recall=p0.get("site_recall", float("nan")),
                         in_site=f"pose {j0 + 1} (recall {d['poses'][j0].get('site_recall', float('nan')):.2f})" if j0 is not None else "no",
                         note="seed" if smi == state.data["seed"] else ""))
        shown.append((d["best_score"], smi, r))
    rows.sort(key=lambda x: (x["vina"] if isinstance(x["vina"], float) else 99))
    from medrag.tools.poses import read_poses
    pose_list = []
    for score, smi, r in sorted(shown, key=lambda t: t[0])[:top]:
        d = r["dock"]
        j = d["in_site"][0] if d["in_site"] else 0
        pose_list.append(dict(name=d["species"][:28], sdf=read_poses(d["sdf"])[j][0],
                              caption=f"vina {d['poses'][j]['score']} ({'in site' if d['in_site'] else 'OFF SITE'})"))
    rep.add_step("dock", "Dock", f"Vina into the crystal-ligand site (centroid {state.target.get('center')}). Vina barely "
                 "discriminates between sites for small ligands, so 'in site' means the pose contacts at least half of the residues the crystal ligand contacts (recall).",
                 table=rows, poses=pose_list, max_images=top)


def add_validate_section(rep: Report, state: RunState):
    rows, pose_list = [], []
    for smi, r in state.candidates.items():
        v = r.get("validate")
        if not v:
            continue
        if v.get("status") != "ok":
            rows.append(dict(species=r.get("species", smi), verdict="error", note=v.get("detail", "")))
            continue
        c = v["checks"]
        rows.append(dict(species=r["dock"]["species"], verdict=v["verdict"], vina=r["dock"]["best_score"],
                         strain_kcal=c.get("ligand_strain", {}).get("value", float("nan")),
                         force_eVA=c.get("force_residuals", {}).get("value", float("nan")),
                         relax_shift_A=c.get("relax_displacement", {}).get("value", float("nan")),
                         contacts=c.get("contact_persistence", {}).get("value", float("nan")), flags="; ".join(v["reasons"])))
        wd = Path(v["workdir"])
        if (wd / "ligand_relaxed.sdf").is_file():
            pose_list.append(dict(name=r["dock"]["species"][:28], sdf=(wd / "ligand_relaxed.sdf").read_text(),
                                  reference_sdf=(wd / "ligand_docked.sdf").read_text(), caption=f"relaxed (grey = docked), {v['verdict']}"))
    rep.add_step("validate", "Validate (MLIP feasibility)",
                 "Best in-site pose per candidate, relaxed in a truncated pocket cluster with an MLIP. A sanity filter for strain, "
                 "clashes, forces and lost contacts - it is not a rescoring and did not separate right-site from wrong-site poses on SULT1A3."
                 + (f" Strain is graded relative to the seed ({state.data['validate_seed_strain']:.1f} kcal/mol): warn above seed + 3, "
                    "fail above seed + 6, and never tighter than the defaults." if state.data.get("validate_seed_strain") is not None else ""),
                 table=rows, poses=pose_list, max_images=len(pose_list))
