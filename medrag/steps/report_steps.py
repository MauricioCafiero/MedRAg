"""HTML report sections for the first half of the pipeline (generate, filter), from a RunState."""
from __future__ import annotations

import html
import math
from collections import Counter
from pathlib import Path

from medrag.report import Report
from medrag.state import RunState


_GEN_TITLES = {"genmask": "GenMask · masked-language-model infilling", "bioisostere": "bioisostere_assist · shape-matched substituent swaps",
               "gpt": "Mini GPT · new molecules from a fine-tuned model"}


def add_generate_section(rep: Report, state: RunState, per_source: int = 12):
    cands = state.candidates
    by_source = Counter(src["source"] for r in cands.values() for src in r["sources"])
    table = [dict(source=k, candidates=v) for k, v in sorted(by_source.items())]
    seed = state.data["seed"]
    mols = []
    for source in ("genmask", "bioisostere", "gpt"):
        rows = []
        for r in cands.values():
            meta = next((x for x in r["sources"] if x["source"] == source), None)
            if meta and r["smiles"] != seed:
                rows.append((r["smiles"], meta))
        if not rows:
            continue
        if source == "bioisostere":
            rows.sort(key=lambda t: -t[1].get("similarity", 0))
        title = f"{_GEN_TITLES[source]} · {len(rows)} molecules" + (f" (showing {per_source})" if len(rows) > per_source else "")
        if source == "gpt":
            title += f" · trained on {rows[0][1].get('training_set') or rows[0][1].get('chembl_id', '?')}"
        for smi, meta in rows[:per_source]:
            legend = (f"Tanimoto {meta['tanimoto_to_parent']:.2f}" if source == "genmask" and "tanimoto_to_parent" in meta
                      else f"shape sim {meta['similarity']:.2f}" if source == "bioisostere" and "similarity" in meta else source)
            mols.append(dict(smiles=smi, legend=legend, group=title))
    rep.add_step("generate", "Generate", f"Seed {seed}: {len(cands) - 1} unique analogues proposed by {len(by_source) - ('seed' in by_source)} generators.",
                 table=table, molecules=mols)


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
    # charts: outcome by generator, QED vs similarity to the seed, most common fail flags
    seed = state.data["seed"]
    per_src: dict = {}
    pts = []
    for smi, r in state.candidates.items():
        f = r.get("filter")
        if not f or "props" not in f or smi == seed:
            continue
        src = next((x["source"] for x in r["sources"] if x["source"] != "seed"), "?")
        c = per_src.setdefault(src, Counter())
        c[f["status"]] += 1
        pts.append((f["props"]["tanimoto_to_seed"], f["props"]["qed"], {"pass": "ok", "warn": "warn", "fail": "bad"}[f["status"]],
                    f"{r['species']} · {f['status']} · QED {f['props']['qed']:.2f}, similarity {f['props']['tanimoto_to_seed']:.2f}"))
    flags = Counter(x for r in state.candidates.values() for x in r.get("filter", {}).get("reasons", []) if x.endswith("fail"))
    chart = ""
    if pts:
        chart = ('<div class="chartgrid"><div><div class="ct">Outcome by generator</div>'
                 + _svg_stacked([(k, v["pass"], v["warn"], v["fail"]) for k, v in sorted(per_src.items())], w=620)
                 + '</div><div><div class="ct">Most common fail flags</div><div class="chips" style="margin-top:6px">'
                 + ("".join(f'<span class="chip"><b>{html.escape(k.replace(" fail", ""))}</b>{v}</span>' for k, v in flags.most_common(8)) or "none")
                 + '</div><div class="hint" style="margin-top:12px">ADMET cut-offs are no tighter than the seed\'s own values, so a flag means '
                 'clearly worse than the lead.</div></div></div>'
                 '<div class="ct" style="margin-top:16px">Drug-likeness vs similarity to the seed (every candidate)</div>'
                 + _svg_scatter(pts, "Tanimoto similarity to the seed", "QED (drug-likeness)", 0.0, 1.0, 0.35, "QED warn below 0.35")
                 + '<div class="hint">green = pass, amber = warn, red = fail. Molecules from the GPT sit at low similarity because they are new compounds, '
                 'not seed analogues.</div>')
    rep.add_step("filter", "Filter",
                 f"Protonated at pH 7.4, then QED / Lipinski / alerts / ADMET-AI. {tally['pass']} pass, {tally['warn']} warn, "
                 f"{tally['fail']} fail (excluding the seed and duplicates). Cut-offs are provisional.",
                 table=rows, molecules=[dict(smiles=x["species"], legend=f"{x['status']} qed {x['qed']:.2f}") for x in ok], chart_html=chart)


_DOT = {"ok": "#3ddc97", "warn": "#ffd166", "bad": "#ff5c7a", "seed": "#6c8cff"}


def _nice_step(span: float, n: int = 6) -> float:
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    return next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)


def _svg_scatter(points: list, xlabel: str, ylabel: str, ymin: float = 0.0, ymax: float = 1.0, hline: float | None = None,
                 hlabel: str = "", w: int = 960, h: int = 300) -> str:
    """Generic scatter, one dot per (x, y, kind, tooltip); kind picks the colour (ok / warn / bad / seed). Optional dashed y line."""
    ml, mr, mt, mb = 60, 24, 18, 46
    xs = [p[0] for p in points]
    lo, hi = min(xs), max(xs)
    pad = (hi - lo) * 0.06 or 0.5
    lo, hi = lo - pad, hi + pad
    sx = lambda v: ml + (v - lo) / (hi - lo) * (w - ml - mr)
    sy = lambda v: mt + (1 - (v - ymin) / (ymax - ymin)) * (h - mt - mb)
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(ylabel)} against {html.escape(xlabel)}">']
    ystep = _nice_step(ymax - ymin, 4)
    yv = ymin
    while yv <= ymax + 1e-9:
        out.append(f'<line x1="{ml}" x2="{w - mr}" y1="{sy(yv):.1f}" y2="{sy(yv):.1f}" stroke="#26335c"/><text x="{ml - 8}" y="{sy(yv) + 4:.1f}" text-anchor="end">{yv:g}</text>')
        yv += ystep
    if hline is not None:
        out.append(f'<line x1="{ml}" x2="{w - mr}" y1="{sy(hline):.1f}" y2="{sy(hline):.1f}" stroke="#ffd166" stroke-dasharray="6 4" stroke-width="1.4"/>'
                   f'<text x="{w - mr}" y="{sy(hline) - 6:.1f}" text-anchor="end" style="fill:#ffd166">{html.escape(hlabel)}</text>')
    step = _nice_step(hi - lo)
    t = math.ceil(lo / step) * step
    while t <= hi:
        out.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{mt}" y2="{h - mb}" stroke="#1c2750"/><text x="{sx(t):.1f}" y="{h - mb + 16}" text-anchor="middle">{t:g}</text>')
        t += step
    out.append(f'<text x="{(ml + w - mr) / 2:.0f}" y="{h - 8}" text-anchor="middle">{html.escape(xlabel)}</text>')
    out.append(f'<text transform="translate(16 {(mt + h - mb) / 2:.0f}) rotate(-90)" text-anchor="middle">{html.escape(ylabel)}</text>')
    for x, y, kind, title in sorted(points, key=lambda p: p[2] == "seed"):
        r, sw = (7, 2.5) if kind == "seed" else (4.2, 0)
        out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{r}" fill="{_DOT[kind]}" fill-opacity="{0.95 if kind == "seed" else 0.62}" '
                   f'stroke="#e6ebf7" stroke-width="{sw}"><title>{html.escape(title)}</title></circle>')
    out.append("</svg>")
    return "".join(out)


def _svg_stacked(rows: list, w: int = 960) -> str:
    """Horizontal stacked bars: rows = [(label, pass, warn, fail)]."""
    ml, mr, mt, rowh = 150, 40, 12, 38
    h = mt + rowh * len(rows) + 30
    xmax = max(p + wv + f for _, p, wv, f in rows) or 1
    sx = lambda v: v / xmax * (w - ml - mr)
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="filter outcome by generator">']
    for i, (label, p, wv, f) in enumerate(rows):
        y, x = mt + i * rowh, ml
        out.append(f'<text x="{ml - 10}" y="{y + 21}" text-anchor="end">{html.escape(label)} <tspan style="fill:#8e9bc0">({p + wv + f})</tspan></text>')
        for n, col, name in ((p, "#3ddc97", "pass"), (wv, "#ffd166", "warn"), (f, "#ff5c7a", "fail")):
            if n:
                out.append(f'<rect x="{x:.1f}" y="{y + 6}" width="{sx(n):.1f}" height="22" fill="{col}" fill-opacity="0.85"><title>{label}: {n} {name}</title></rect>')
                if sx(n) > 24:
                    out.append(f'<text x="{x + sx(n) / 2:.1f}" y="{y + 21}" text-anchor="middle" style="fill:#0b1020;font-weight:700">{n}</text>')
                x += sx(n)
    lx = ml
    for col, name in (("#3ddc97", "pass"), ("#ffd166", "warn"), ("#ff5c7a", "fail")):
        out.append(f'<rect x="{lx}" y="{h - 20}" width="11" height="11" rx="3" fill="{col}"/><text x="{lx + 17}" y="{h - 10}">{name}</text>')
        lx += 70
    out.append("</svg>")
    return "".join(out)


def _svg_strain_bars(items: list, seed_strain: float | None, warn: float, fail: float, w: int = 960) -> str:
    """Horizontal strain bars (kcal/mol), one per validated molecule, coloured by verdict, with the seed / warn / fail lines."""
    ml, mr, mt, rowh = 330, 70, 58, 30
    h = mt + rowh * len(items) + 26
    xmax = max([fail, *[v for _, v, _ in items]]) * 1.08
    sx = lambda v: ml + v / xmax * (w - ml - mr)
    col = {"accept": "#3ddc97", "demote": "#ffd166", "reject": "#ff5c7a"}
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="ligand strain per validated molecule">']
    for k, (xv, c, name, dash) in enumerate(((seed_strain, "#6c8cff", "seed", ""), (warn, "#ffd166", "warn", ' stroke-dasharray="6 4"'),
                                              (fail, "#ff5c7a", "fail", ' stroke-dasharray="6 4"'))):
        if xv is not None:                                         # labels on three different rows so they never overlap
            ty = 14 + 16 * k
            out.append(f'<line x1="{sx(xv):.1f}" x2="{sx(xv):.1f}" y1="{ty + 4}" y2="{h - 22}" stroke="{c}" stroke-width="1.6"{dash}/>'
                       f'<text x="{sx(xv) + 5:.1f}" y="{ty + 2}" style="fill:{c}">{name} {xv:.1f}</text>')
    for i, (label, v, verdict) in enumerate(items):
        y = mt + i * rowh
        out.append(f'<text x="{ml - 10}" y="{y + 17}" text-anchor="end" style="font-family:ui-monospace,Menlo,monospace">{html.escape(label)}</text>'
                   f'<rect x="{ml}" y="{y + 5}" width="{max(sx(v) - ml, 1):.1f}" height="16" rx="4" fill="{col.get(verdict, "#6c8cff")}" fill-opacity="0.85"/>'
                   f'<text x="{sx(v) + 6:.1f}" y="{y + 17}">{v:.1f}</text>')
    out.append(f'<text x="{(ml + w - mr) / 2:.0f}" y="{h - 4}" text-anchor="middle">ligand strain (kcal/mol), docked geometry vs the lowest vacuum minimum</text></svg>')
    return "".join(out)


def add_dock_section(rep: Report, state: RunState, top: int = 6):
    rows, shown, points = [], [], []
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
        for j, p in enumerate(d["poses"]):
            if p.get("site_recall") is not None and p.get("score") is not None:
                kind = "seed" if smi == state.data["seed"] else "ok" if p["site_recall"] >= 0.5 else "bad"
                points.append((p["score"], p["site_recall"], kind, f"{d['species']} · pose {j + 1}: Vina {p['score']}, recall {p['site_recall']:.2f}"))
    rows.sort(key=lambda x: (x["vina"] if isinstance(x["vina"], float) else 99))
    from medrag.tools.poses import read_poses
    pose_list = []
    for score, smi, r in sorted(shown, key=lambda t: t[0])[:top]:
        d = r["dock"]
        j = d["in_site"][0] if d["in_site"] else 0
        pose_list.append(dict(name=d["species"][:28], sdf=read_poses(d["sdf"])[j][0],
                              caption=f"vina {d['poses'][j]['score']} ({'in site' if d['in_site'] else 'OFF SITE'})"))
    chart = (f'<div class="ct">Every docked pose: Vina score vs binding-site recall</div>{_svg_scatter(points, "Vina score (kcal/mol) — more negative is better", "site recall", 0.0, 1.0, 0.5, "in-site cut-off (recall 0.5)")}'
             '<div class="hint">green = pose reproduces ≥50% of the crystal ligand\'s contacts, red = it does not, blue = the seed. '
             'If the dots in one column are both green and red, the Vina score alone cannot tell the right site from the wrong one.</div>') if points else ""
    rep.add_step("dock", "Dock", f"Vina into the crystal-ligand site (centroid {state.target.get('center')}). 'In site' means the pose contacts at least "
                 "half of the residues the crystal ligand contacts (recall). The table and views below show each molecule's best in-site pose.",
                 table=rows, poses=pose_list, max_images=top, chart_html=chart, ligand="cyan", ref_name="crystal ligand")


def add_validate_section(rep: Report, state: RunState):
    rows, pose_list, cards, bars = [], [], [], []
    seed_strain = state.data.get("validate_seed_strain")
    for smi, r in state.candidates.items():
        v = r.get("validate")
        if not v:
            continue
        if v.get("status") != "ok":
            rows.append(dict(species=r.get("species", smi), verdict="error", note=v.get("detail", "")))
            continue
        c = v["checks"]
        g = lambda k: c.get(k, {}).get("value", float("nan"))
        sp = r["dock"]["species"]
        rows.append(dict(species=sp, verdict=v["verdict"], vina=r["dock"]["best_score"], strain_kcal=g("ligand_strain"),
                         force_eVA=g("force_residuals"), relax_shift_A=g("relax_displacement"), contacts=g("contact_persistence"),
                         flags="; ".join(v["reasons"])))
        tone = {"accept": "ok", "demote": "warn", "reject": "bad"}.get(v["verdict"], "")
        delta = f" ({g('ligand_strain') - seed_strain:+.1f} vs seed)" if seed_strain is not None and smi != state.data["seed"] else " (the seed)" if smi == state.data["seed"] else ""
        cards.append(f'<div class="vcard {tone}"><div class="vtop"><span class="pill {tone}">{html.escape(v["verdict"])}</span>'
                     f'<span class="vname" title="{html.escape(sp)}">{html.escape(sp)}</span></div><div class="vm">'
                     f'<div><b>{g("ligand_strain"):.1f}</b><small>strain kcal/mol</small></div><div><b>{g("force_residuals"):.1f}</b><small>force eV/Å</small></div>'
                     f'<div><b>{g("relax_displacement"):.2f}</b><small>shift Å</small></div><div><b>{g("contact_persistence"):.2f}</b><small>contacts kept</small></div></div>'
                     f'<div class="vflag">Vina {r["dock"]["poses"][v["pose"]]["score"]}{html.escape(delta)}</div></div>')
        bars.append((sp[:44] + ("…" if len(sp) > 44 else ""), g("ligand_strain"), v["verdict"]))
        wd = Path(v["workdir"])
        if (wd / "ligand_relaxed.sdf").is_file():
            pose_list.append(dict(name=sp[:28], sdf=(wd / "ligand_relaxed.sdf").read_text(),
                                  reference_sdf=(wd / "ligand_docked.sdf").read_text(), caption=f"relaxed (grey = docked), {v['verdict']}"))
    chart = ""
    if cards:
        warn = max(6.0, seed_strain + 3) if seed_strain is not None else 6.0
        fail = max(12.0, seed_strain + 6) if seed_strain is not None else 12.0
        chart = (f'<div class="vcards">{"".join(cards)}</div><div class="ct">Ligand strain against the seed</div>'
                 f'{_svg_strain_bars(bars, seed_strain, warn, fail)}'
                 '<div class="hint">bars are coloured by verdict; strain is graded relative to the seed, so a bar is only a problem past the amber line.</div>')
    rep.add_step("validate", "Validate (MLIP feasibility)",
                 "Each best in-site pose is relaxed in a truncated pocket cluster with an MLIP, then checked for strain, clashes, residual forces and lost contacts. "
                 "This is a sanity filter, not a rescoring or a binding prediction."
                 + (f" Strain is graded relative to the seed ({seed_strain:.1f} kcal/mol): warn above seed + 3, fail above seed + 6, never tighter than the defaults."
                    if seed_strain is not None else ""),
                 table=rows, poses=pose_list, max_images=len(pose_list), chart_html=chart, ligand="magenta", ref_name="docked pose")


def header(state: RunState) -> tuple[list, list]:
    """(chips, stat tiles) for the top of the report, from whatever the run has done so far."""
    t, c = state.target, list(state.candidates.values())
    chips = [("Run", state.data["run_id"])]
    if t.get("pdb_id"):
        chips.append(("Structure", f"{t['pdb_id']}" + (f" · {t['resolution']} Å" if t.get("resolution") else "")))
    if t.get("resname"):
        chips.append(("Site ligand", f"{t['resname']} chain {t.get('chain', '')}"))
    if t.get("uniprot"):
        chips.append(("UniProt", t["uniprot"]))
    ch = t.get("chembl") or {}
    if ch.get("best_target"):
        chips.append(("ChEMBL", f"{ch['best_target']} · {ch['n_ic50']} IC50 molecules"))
    mlip = next((s.get("mlip") for s in reversed(state.data["steps"]) if s["step"] == "validate"), None)
    if mlip:
        chips.append(("MLIP", mlip))
    stats = []
    if len(c) > 1:
        by_src = Counter(src["source"] for r in c for src in r["sources"] if src["source"] != "seed")
        stats.append(dict(label="Candidates generated", value=len(c) - 1, tone="cyan", sub=" · ".join(f"{k} {v}" for k, v in sorted(by_src.items()))))
    filt = Counter(r["filter"]["status"] for r in c if r.get("filter"))
    if filt:
        stats.append(dict(label="Passed the filter", value=filt.get("pass", 0), tone="green", sub=f"warn {filt.get('warn', 0)} · fail {filt.get('fail', 0)}"))
    docked = [r for r in c if r.get("dock", {}).get("status") == "ok"]
    if docked:
        stats.append(dict(label="Docked", value=len(docked), tone="blue", sub=f"best Vina {min(r['dock']['best_score'] for r in docked)} kcal/mol"))
        n_site = sum(bool(r["dock"].get("in_site")) for r in docked)
        exp = t.get("expected_residues") or []
        if any("site_expected" in r["dock"] for r in docked):
            stats.append(dict(label="In the binding site", value=n_site, tone="magenta",
                              sub=f"pose reproduces ≥50% of the {len(exp)} crystal contacts"))
    val = Counter(r["validate"]["verdict"] for r in c if r.get("validate", {}).get("status") == "ok")
    if val:
        stats.append(dict(label="MLIP validated", value=sum(val.values()), tone="orange",
                          sub=" · ".join(f"{val[k]} {k}" for k in ("accept", "demote", "reject") if k in val)))
    return chips, stats


def add_boltz_section(rep: Report, state: RunState):
    """Boltz-2 co-folding of the finalists: a table and cards from the affinity/confidence text files, and a rotatable view of each cif
    (Boltz's own protein + ligand complex, in its own frame, so it is not overlaid on the crystal receptor used elsewhere)."""
    rows, cards, models = [], [], []
    seed = state.data["seed"]
    for smi, r in state.candidates.items():
        b = r.get("boltz")
        if not b or b.get("status") != "ok":
            continue
        name = r.get("dock", {}).get("species", smi)
        tag = " (the seed)" if smi == seed else ""
        rows.append(dict(species=name + tag, pIC50=b["pic50"], IC50_uM=b["ic50_uM"], dG_kcal=b["dG_kcal"], binder_prob=b["binder_prob"],
                         confidence=b["confidence"], iptm=b["iptm"], md_verdict=r.get("validate", {}).get("verdict", "")))
        tone = "ok" if b["binder_prob"] >= 0.5 else "warn"
        cards.append(f'<div class="vcard {tone}"><div class="vtop"><span class="pill {tone}">binder {b["binder_prob"]:.2f}</span>'
                     f'<span class="vname" title="{html.escape(name)}">{html.escape(name + tag)}</span></div><div class="vm">'
                     f'<div><b>{b["pic50"]:.2f}</b><small>pIC50</small></div><div><b>{b["ic50_uM"]:.1f}</b><small>IC50 µM</small></div>'
                     f'<div><b>{b["dG_kcal"]:.1f}</b><small>ΔG kcal/mol</small></div><div><b>{b["confidence"]:.2f}</b><small>confidence</small></div></div></div>')
        cif = Path(b["cif"])
        if cif.is_file():
            models.append(dict(name=name[:28], cif=cif.read_text()))
    if not rows:
        return
    chart = f'<div class="vcards">{"".join(cards)}</div>'
    if models:
        import json
        data = json.dumps(models).replace("</", "<\\/")
        chart += ('<h3>Rotatable view (Boltz complex)</h3><div class="viewwrap"><div class="tabs" id="bz_tabs"></div>'
                  '<div class="viewer" id="bz_view"></div><div class="legend"><span><i style="background:#7b8cff"></i>protein ribbon</span>'
                  '<span><i style="background:#ffd166"></i>predicted ligand (thick sticks)</span><span>drag to rotate, scroll to zoom</span></div></div>'
                  f'<script>window.addEventListener("load", function () {{ const M = {data}; let v = null;'
                  'function show(i) { if (!v) v = $3Dmol.createViewer(document.getElementById("bz_view"), { backgroundColor: "#0b1020" });'
                  'v.removeAllModels(); const m = v.addModel(M[i].cif, "cif");'
                  'v.setStyle({}, { cartoon: { color: "#7b8cff", opacity: 0.5, arrows: true } });'
                  'v.setStyle({ chain: "B" }, { stick: { radius: 0.3, colorscheme: "yellowCarbon" } });'
                  'v.zoomTo({ chain: "B" }); v.zoom(0.6); v.render();'
                  'document.querySelectorAll("#bz_tabs button").forEach((b, k) => b.classList.toggle("active", k === i)); }'
                  'const t = document.getElementById("bz_tabs");'
                  'M.forEach((m, i) => { const b = document.createElement("button"); b.textContent = m.name; b.onclick = () => show(i); t.appendChild(b); });'
                  'show(0); });</script>')
    rep.add_step("boltz", "Boltz-2 co-folding (finalists)",
                 "Boltz-2 predicts the whole protein-ligand complex from the sequence and SMILES, with no docking box, and scores affinity with its own head. "
                 "It is an independent second opinion on the finalists, not a rescoring of the docking. Binder probability is Boltz's chance the molecule binds at all; "
                 "pIC50 is only meaningful for comparing molecules that already bind. The view shows Boltz's own complex, which is not overlaid on the crystal receptor above.",
                 table=rows, chart_html=chart)
