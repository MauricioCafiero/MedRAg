"""CafChem MedRAg HTML report: one self-contained page per run, dark theme, a floating title bar that always shows every section link,
summary tiles, and one numbered card per pipeline step with sortable scrollable tables, scrollable image strips (PyMOL stills, 2D
molecule grids) and a rotatable 3Dmol viewer (protein as a ribbon, nearby residues as thin sticks, the ligand as thick sticks).

Usage:
    rep = Report(run_dir, title="HMGCR hit-to-lead", receptor="path/to/receptor.pdb", reference_sdf="crystal_ligand.mol")
    rep.set_header(subtitle="...", chips=[("PDB", "1HW8")], stats=[dict(label="Candidates", value=152, sub="...", tone="cyan")])
    rep.add_step("dock", "Dock", "summary text", table=[{...}, ...], poses=[{"name", "sdf", "caption", "reference_sdf"}], molecules=[...])
    rep.write()          # -> <run_dir>/report.html  (single file; images are embedded, 3Dmol.js comes from a CDN)

PyMOL runs headless in a subprocess (report_pymol.py) because it keeps global state.
"""
from __future__ import annotations

import base64
import html
import json
import math
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

CDN = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.0/3Dmol-min.js"
STYLE_VERSION = "dark-v2"            # part of the PyMOL still cache key: bump to re-render every still
BG_RGB = (0.043, 0.063, 0.125)       # #0b1020
PILLS = {"pass": "ok", "accept": "ok", "yes": "ok", "warn": "warn", "demote": "warn", "fail": "bad", "reject": "bad", "no": "bad",
         "seed": "seed", "error": "bad"}


def render_pose_png(receptor: str, ligand_file: str, out_png: str, reference: str | None = None,
                    label: str = "", width: int = 700, height: int = 520, timeout: int = 180, ligand_color: str = "cyan") -> bool:
    """PyMOL still of one ligand in its pocket. Returns False (and leaves no image) if PyMOL fails."""
    spec = Path(out_png).with_suffix(".json")
    spec.write_text(json.dumps(dict(receptor=str(receptor), ligand=str(ligand_file), out=str(out_png),
                                    reference=reference, label=label, width=width, height=height, ligand_color=ligand_color)))
    try:
        subprocess.run([sys.executable, str(Path(__file__).with_name("report_pymol.py")), str(spec)],
                       check=True, capture_output=True, timeout=timeout)
    except (subprocess.SubprocessError, OSError) as e:
        print(f"[report] PyMOL render failed for {out_png}: {getattr(e, 'stderr', e)}", file=sys.stderr)
        return False
    finally:
        spec.unlink(missing_ok=True)
    return Path(out_png).exists()


def open_in_browser(path: str | Path) -> bool:
    """Open a finished report in the default browser; False if there is no browser/display (the path is still the result)."""
    try:
        return bool(webbrowser.open(Path(path).resolve().as_uri()))
    except Exception:
        return False


def receptor_for_view(receptor_pdb: str, lig_xyz: list[np.ndarray], radius: float = 14.0) -> str:
    """ATOM records of every chain with an atom within `radius` A of any ligand pose, whole chains so the ribbon is continuous.
    (Waters and hetero atoms are left out; the page embeds this once for all its viewers.)"""
    lines, chain_of, xyz = [], [], []
    for line in open(receptor_pdb):
        if line.startswith("ATOM"):
            lines.append(line)
            chain_of.append(line[21])
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not lines or not lig_xyz:
        return "".join(lines)
    xyz, lig = np.array(xyz), np.vstack(lig_xyz)
    near = set()
    for start in range(0, len(lig), 200):                 # chunked: tetramers have ~12k atoms
        d = np.linalg.norm(xyz[:, None, :] - lig[None, start:start + 200, :], axis=2).min(axis=1)
        near |= {c for c, close in zip(chain_of, d <= radius) if close}
    return "".join(l for l, c in zip(lines, chain_of) if c in near)


def pocket_residues(receptor_pdb: str, lig_xyz: np.ndarray, cutoff: float = 4.0) -> list:
    """[[chain, [resseq, ...]], ...]: every residue with an atom within `cutoff` A of the ligand pose. Computed here and handed to
    3Dmol as an explicit list, because 3Dmol's own 'within' selection did not restrict anything in the browser."""
    from scipy.spatial import cKDTree
    chain, resi, xyz = [], [], []
    for line in open(receptor_pdb):
        if line.startswith("ATOM"):
            try:
                resi.append(int(line[22:26]))
            except ValueError:
                continue
            chain.append(line[21])
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not xyz:
        return []
    hit = cKDTree(np.array(xyz)).query_ball_point(lig_xyz, r=cutoff)
    near = sorted({(chain[i], resi[i]) for h in hit for i in h})
    out: dict = {}
    for c, r in near:
        out.setdefault(c, []).append(r)
    return [[c, rs] for c, rs in out.items()]


def molecule_grid_png(smiles: list[str], legends: list[str], out_png: str, per_row: int = 4, size=(250, 200)) -> bool:
    """RDKit 2D grid image of molecules in the report's dark colours (no receptor needed)."""
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    keep = [(m, l) for m, l in zip(mols, legends) if m is not None]
    if not keep:
        return False
    opts = rdMolDraw2D.MolDrawOptions()
    try:
        rdMolDraw2D.SetDarkMode(opts)
        opts.setBackgroundColour(BG_RGB + (1.0,))
        opts.setLegendColour((0.78, 0.84, 0.95, 1.0))
        opts.legendFontSize = 15
    except Exception:
        pass
    Draw.MolsToGridImage([m for m, _ in keep], molsPerRow=per_row, subImgSize=size, legends=[l for _, l in keep],
                         drawOptions=opts, returnPNG=False).save(out_png)
    return True


def _b64(png: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()


def _fmt(v: float) -> str:
    return "–" if isinstance(v, float) and math.isnan(v) else f"{v:.3g}"


def _table_html(rows: list[dict]) -> str:
    """Sortable table inside a fixed-height scroll window; status words become coloured pills, SMILES columns stay monospace."""
    cols = list(rows[0])
    head = "".join(f'<th data-col="{i}">{html.escape(str(c).replace("_", " "))}<i></i></th>' for i, c in enumerate(cols))
    body = []
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                nan = isinstance(v, float) and math.isnan(v)
                cells.append(f'<td class="num" data-v="{"" if nan else v}">{_fmt(float(v)) if isinstance(v, float) else v}</td>')
                continue
            txt = str(v)
            low = txt.lower()
            if c == "in_site":
                cls = "ok" if low.startswith("pose") else "bad"
                cells.append(f'<td data-v="{html.escape(low)}"><span class="pill {cls}">{html.escape(txt)}</span></td>')
            elif low in PILLS:
                cells.append(f'<td data-v="{html.escape(low)}"><span class="pill {PILLS[low]}">{html.escape(txt)}</span></td>')
            elif c in ("species", "smiles"):
                cells.append(f'<td class="mono" title="{html.escape(txt)}" data-v="{html.escape(txt)}">{html.escape(txt)}</td>')
            elif len(txt) > 60:
                cells.append(f'<td class="long" title="{html.escape(txt)}" data-v="{html.escape(txt)}">{html.escape(txt)}</td>')
            else:
                cells.append(f'<td data-v="{html.escape(txt)}">{html.escape(txt)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<div class="scroll tablewrap"><table class="sortable"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div><div class="hint">{len(rows)} rows &middot; scroll inside the table, click a header to sort</div>')


_CSS = """
:root { color-scheme: dark; --bg:#0b1020; --bg2:#0f1730; --card:#121a33; --card2:#17213f; --line:#26335c; --text:#e6ebf7; --muted:#8e9bc0;
        --cyan:#4cc9f0; --magenta:#f72585; --orange:#ff9f1c; --green:#3ddc97; --amber:#ffd166; --red:#ff5c7a; --blue:#6c8cff; }
* { box-sizing: border-box; scrollbar-color: #2b3a6b transparent; scrollbar-width: thin; }
html { scroll-behavior: smooth; scroll-padding-top: 96px; }
body { margin:0; background: radial-gradient(1200px 600px at 10% -10%, #1a2550 0%, transparent 60%),
        radial-gradient(900px 500px at 100% 0%, #2a1240 0%, transparent 55%), var(--bg);
       background-attachment: fixed; color:var(--text); font: 15px/1.55 -apple-system, "Segoe UI", Inter, Helvetica, Arial, sans-serif; }
a { color: var(--cyan); text-decoration: none; }
.bar { position: sticky; top:0; z-index: 50; display:flex; align-items:center; flex-wrap:wrap; gap:6px 18px; padding: 9px 24px;
       background: rgba(11,16,32,.9); backdrop-filter: blur(10px); border-bottom: 1px solid var(--line); }
.brand { font-weight: 700; letter-spacing:.2px; white-space:nowrap; }
.brand b { color: var(--cyan); } .brand small { display:block; font-weight:400; font-size:10.5px; color:var(--muted); letter-spacing:.8px; text-transform:uppercase; margin-top:-2px; }
.runname { color: var(--muted); font-size:13px; border-left:1px solid var(--line); padding-left:18px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:30vw; }
.bar nav { margin-left:auto; display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end; }
.bar nav a { color: var(--text); font-size:13px; padding:5px 11px; border-radius:999px; border:1px solid var(--line); white-space:nowrap; background: rgba(255,255,255,.03); }
.bar nav a:hover { border-color: var(--cyan); background: var(--card2); }
.bar nav a.top { color: var(--orange); }
.bar nav a.on { background: linear-gradient(90deg, rgba(76,201,240,.28), rgba(247,37,133,.28)); border-color: var(--cyan); }
.page { max-width: 1280px; margin: 0 auto; padding: 28px 24px 60px; }
.hero { padding: 34px 32px; border:1px solid var(--line); border-radius: 18px; margin-bottom: 22px;
        background: linear-gradient(135deg, rgba(76,201,240,.10), rgba(247,37,133,.10)), var(--card); position:relative; overflow:hidden; }
.hero:before { content:""; position:absolute; right:-80px; top:-80px; width:300px; height:300px; border-radius:50%;
               background: radial-gradient(circle, rgba(255,159,28,.28), transparent 70%); }
.hero .kicker { color: var(--orange); font-weight:600; letter-spacing:1.4px; text-transform:uppercase; font-size:12px; }
.hero h1 { margin: 6px 0 4px; font-size: 34px; line-height:1.15; background: linear-gradient(90deg, var(--cyan), var(--magenta)); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero .sub { color: var(--muted); max-width: 900px; word-break: break-word; }
.chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:16px; }
.chip { font-size:12.5px; padding:5px 11px; border-radius:999px; background: rgba(255,255,255,.05); border:1px solid var(--line); color: var(--text); }
.chip b { color: var(--muted); font-weight:500; margin-right:5px; }
.tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px,1fr)); gap:12px; margin: 0 0 26px; }
.tile { background: var(--card); border:1px solid var(--line); border-radius:14px; padding:14px 16px; border-top:3px solid var(--cyan); }
.tile.magenta { border-top-color: var(--magenta); } .tile.orange { border-top-color: var(--orange); } .tile.green { border-top-color: var(--green); } .tile.blue { border-top-color: var(--blue); }
.tile .v { font-size: 30px; font-weight: 700; line-height:1.1; } .tile .l { color: var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.9px; margin-top:2px; }
.tile .s { color: var(--muted); font-size:12.5px; margin-top:6px; }
section.card { background: var(--card); border:1px solid var(--line); border-radius: 18px; padding: 22px 24px 24px; margin-bottom: 22px; }
.card-head { display:flex; align-items:center; gap:14px; margin-bottom:4px; }
.card-head .totop { margin-left:auto; font-size:12px; color: var(--muted); border:1px solid var(--line); border-radius:999px; padding:3px 11px; }
.card-head .totop:hover { color: var(--orange); border-color: var(--orange); }
.stepno { font-weight:700; font-size:13px; color: var(--bg); background: linear-gradient(135deg, var(--cyan), var(--magenta)); border-radius:9px; padding:4px 9px; }
.card h2 { margin:0; font-size:22px; }
.summary { color: var(--muted); margin: 6px 0 16px; max-width: 980px; }
h3 { font-size:13px; text-transform:uppercase; letter-spacing:1px; color: var(--orange); margin: 22px 0 8px; }
.scroll { overflow:auto; border:1px solid var(--line); border-radius:12px; background: var(--bg2); }
.tablewrap { max-height: 380px; }
table { border-collapse: collapse; width:100%; font-size: 13px; }
th { position: sticky; top:0; background: var(--card2); color: var(--muted); text-align:left; font-weight:600; text-transform:uppercase; letter-spacing:.6px;
     font-size:11px; padding: 9px 12px; cursor:pointer; white-space:nowrap; border-bottom:1px solid var(--line); z-index:2; user-select:none; }
th i { display:inline-block; width:10px; margin-left:4px; font-style:normal; color:var(--cyan); }
th.asc i:after { content:"▲"; } th.desc i:after { content:"▼"; }
td { padding: 7px 12px; border-bottom: 1px solid rgba(38,51,92,.55); white-space:nowrap; }
tr:nth-child(even) td { background: rgba(255,255,255,.018); } tr:hover td { background: rgba(76,201,240,.07); }
td.num { text-align:right; font-variant-numeric: tabular-nums; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
td.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; max-width: 360px; overflow:hidden; text-overflow:ellipsis; color:#cdd8ff; }
td.long { max-width: 380px; overflow:hidden; text-overflow:ellipsis; color: var(--muted); }
.pill { display:inline-block; padding:2px 9px; border-radius:999px; font-size:11.5px; font-weight:600; border:1px solid; }
.pill.ok { color: var(--green); border-color: rgba(61,220,151,.45); background: rgba(61,220,151,.10); }
.pill.warn { color: var(--amber); border-color: rgba(255,209,102,.45); background: rgba(255,209,102,.10); }
.pill.bad { color: var(--red); border-color: rgba(255,92,122,.45); background: rgba(255,92,122,.10); }
.pill.seed { color: var(--blue); border-color: rgba(108,140,255,.5); background: rgba(108,140,255,.12); }
.hint { color: var(--muted); font-size: 12px; margin: 6px 2px 0; }
.strip { display:flex; gap:14px; overflow-x:auto; padding: 12px; scroll-snap-type: x proximity; }
.strip figure { margin:0; flex: 0 0 auto; width: 330px; background: var(--card2); border:1px solid var(--line); border-radius:12px; overflow:hidden; scroll-snap-align:start; }
.strip img { display:block; width:100%; } .strip.grid2d { justify-content: center; } .strip.grid2d figure { width: auto; max-width: 100%; }
.strip figcaption { font-size: 12px; color: var(--muted); padding: 8px 12px 10px; } .strip figcaption b { color: var(--text); }
.viewwrap { margin-top: 6px; }
.tabs { display:flex; gap:6px; overflow-x:auto; padding: 2px 0 10px; }
.tabs button { flex: 0 0 auto; font: inherit; font-size:12.5px; color: var(--text); background: var(--card2); border:1px solid var(--line); padding:5px 12px; border-radius:999px; cursor:pointer; }
.tabs button:hover { border-color: var(--cyan); } .tabs button.active { background: linear-gradient(90deg, rgba(76,201,240,.35), rgba(247,37,133,.35)); border-color: var(--cyan); }
.viewer { position:relative; width:100%; height: 520px; border:1px solid var(--line); border-radius:14px; overflow:hidden; background: #0b1020; }
.legend { display:flex; gap:16px; flex-wrap:wrap; font-size:12px; color: var(--muted); margin-top:8px; }
.legend i { display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
.chart { margin: 2px 0 18px; }
.chart svg { width:100%; height:auto; display:block; background: var(--bg2); border:1px solid var(--line); border-radius:12px; }
.chart svg text { fill: #aab6d8; font-family: -apple-system, "Segoe UI", Inter, Helvetica, Arial, sans-serif; font-size: 11px; }
.chart .ct { color: var(--orange); font-size:13px; text-transform:uppercase; letter-spacing:1px; margin: 0 0 8px; font-weight:600; }
.vcards { display:grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap:12px; margin-bottom:14px; }
.vcard { background: var(--bg2); border:1px solid var(--line); border-radius:14px; padding:12px 14px; border-left:4px solid var(--blue); }
.vcard.ok { border-left-color: var(--green); } .vcard.warn { border-left-color: var(--amber); } .vcard.bad { border-left-color: var(--red); }
.vtop { display:flex; align-items:center; gap:10px; margin-bottom:8px; } .vname { font-family: ui-monospace, Menlo, monospace; font-size:11.5px; color:#cdd8ff; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.vm { display:grid; grid-template-columns: repeat(4, 1fr); gap:6px; }
.vm div { background: rgba(255,255,255,.03); border-radius:8px; padding:6px 8px; } .vm b { display:block; font-size:16px; } .vm small { color: var(--muted); font-size:11px; }
.vflag { color: var(--muted); font-size:12px; margin-top:8px; }
.note { font-size:12.5px; color: var(--muted); margin-top: 28px; text-align:center; }
#up { position: fixed; right: 22px; bottom: 22px; z-index: 60; width: 44px; height: 44px; border-radius: 50%; border:1px solid var(--cyan);
      background: rgba(18,26,51,.92); color: var(--cyan); font-size: 20px; cursor: pointer; display:none; box-shadow: 0 6px 20px rgba(0,0,0,.45); }
#up:hover { background: var(--card2); color: var(--orange); border-color: var(--orange); }
::-webkit-scrollbar { height:10px; width:10px; } ::-webkit-scrollbar-thumb { background:#2b3a6b; border-radius:8px; } ::-webkit-scrollbar-track { background: transparent; }
@media (max-width: 760px) { .hero h1 { font-size: 26px; } .runname { display:none; } .bar nav { margin-left:0; justify-content:flex-start; } .viewer { height: 400px; } }
"""

_JS = """
const PROT = %%PROT%%;
const STEPS = %%STEPS%%;
const state = {};
function show(step, i) {
  const S = STEPS[step]; if (!S) return;
  const m = S[i];
  let st = state[step];
  if (!st) {
    const v = $3Dmol.createViewer(document.getElementById('v_' + step), { backgroundColor: '#0b1020' });
    const prot = v.addModel(PROT, 'pdb');
    st = state[step] = { v: v, prot: prot, lig: null, ref: null };
  }
  const v = st.v, pid = st.prot.getID();
  if (st.lig) v.removeModel(st.lig);
  if (st.ref) v.removeModel(st.ref);
  v.setStyle({ model: pid }, { cartoon: { color: '#7b8cff', opacity: 0.5, arrows: true } });                // the protein as a ribbon
  st.lig = v.addModel(m.sdf, 'sdf');
  const lid = st.lig.getID();
  v.setStyle({ model: lid }, { stick: { radius: 0.3, colorscheme: m.lc || 'cyanCarbon' } });                         // the ligand: thick sticks
  (m.pocket || []).forEach(g => v.addStyle({ model: pid, chain: g[0], resi: g[1] },                           // pocket residues: thin sticks
        { stick: { radius: 0.13, colorscheme: { prop: 'elem', map: { C: 0x9fd8a8, N: 0x7aa2ff, O: 0xff7a90, S: 0xffd166, H: 0xe6ebf7 } } } }));
  st.ref = null;
  if (m.ref) { st.ref = v.addModel(m.ref, 'sdf'); v.setStyle({ model: st.ref.getID() }, { line: { color: '#cfd6ee' } }); }
  v.zoomTo({ model: lid }); v.zoom(0.75); v.render();
  document.querySelectorAll('#t_' + step + ' button').forEach((b, k) => b.classList.toggle('active', k === i));
}
Object.keys(STEPS).forEach(k => show(k, 0));
// sortable tables
document.querySelectorAll('table.sortable').forEach(t => {
  t.querySelectorAll('th').forEach((th, c) => th.addEventListener('click', () => {
    const dir = th.classList.contains('asc') ? -1 : 1;
    t.querySelectorAll('th').forEach(x => x.classList.remove('asc', 'desc'));
    th.classList.add(dir === 1 ? 'asc' : 'desc');
    const rows = Array.from(t.tBodies[0].rows);
    const key = r => { const v = r.cells[c].dataset.v; const n = parseFloat(v); return (v !== '' && !isNaN(n) && /^[-+0-9.eE]+$/.test(v)) ? n : (v || '').toLowerCase(); };
    rows.sort((a, b) => { const x = key(a), y = key(b); return (x > y ? 1 : x < y ? -1 : 0) * dir; });
    rows.forEach(r => t.tBodies[0].appendChild(r));
  }));
});
// the floating bar highlights the section in view; a round button returns to the top
const links = Array.from(document.querySelectorAll('.bar nav a[href^="#"]:not(.top)'));
const obs = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) links.forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#' + e.target.id)); }), { rootMargin: '-30% 0px -60% 0px' });
document.querySelectorAll('section.card').forEach(s => obs.observe(s));
const up = document.getElementById('up');
window.addEventListener('scroll', () => { up.style.display = window.scrollY > 500 ? 'block' : 'none'; });
up.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
"""

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>%%TITLE%%</title>
<script src="%%CDN%%"></script>
<style>%%CSS%%</style></head><body>
<a id="top"></a>
<header class="bar">
  <div class="brand">CafChem <b>MedRAg</b><small>medicinal chemistry agent</small></div>
  <div class="runname">%%TITLE%%</div>
  <nav><a class="top" href="#top">&uarr; Top</a>%%NAV%%</nav>
</header>
<div class="page">
%%HERO%%
%%TILES%%
%%SECTIONS%%
<p class="note">Generated by CafChem MedRAg &middot; %%DATE%% &middot; PyMOL stills and RDKit grids are embedded; the rotatable views load 3Dmol.js from a CDN (internet needed on first open).</p>
</div>
<button id="up" title="Back to top">&uarr;</button>
<script>%%JS%%</script>
</body></html>
"""


class Report:
    def __init__(self, run_dir: str | Path, title: str = "MedRAg report", receptor: str | None = None,
                 subtitle: str = "", reference_sdf: str | None = None):
        self.dir = Path(run_dir)
        self.assets = self.dir / "report_assets"
        self.assets.mkdir(parents=True, exist_ok=True)
        self.title, self.subtitle, self.receptor, self.reference = title, subtitle, receptor, reference_sdf
        self.chips: list[tuple[str, str]] = []
        self.stats: list[dict] = []
        self.steps: list[dict] = []

    def set_header(self, subtitle: str = "", chips: list[tuple[str, str]] | None = None, stats: list[dict] | None = None):
        """subtitle: line under the title; chips: [(label, value)] facts about the run; stats: summary tiles
        [{"label", "value", "sub", "tone": cyan|magenta|orange|green|blue}]."""
        self.subtitle = subtitle or self.subtitle
        self.chips, self.stats = chips or [], stats or []

    def add_step(self, key: str, title: str, summary: str = "", table: list[dict] | None = None,
                 poses: list[dict] | None = None, max_images: int = 6, molecules: list[dict] | None = None,
                 chart_html: str = "", ligand: str = "cyan", ref_name: str = "reference pose"):
        """chart_html: trusted HTML (SVG charts, cards) shown under the summary. ligand: 'cyan' or 'magenta', the ligand colour in
        stills and the 3D view (so docked and relaxed poses look different). ref_name: what the grey reference pose is.
        poses: [{"name", "sdf" (mol block text), "caption", optional "reference_sdf" drawn grey behind it}]. Each gets a PyMOL still
        (first `max_images`) and a tab in the 3Dmol viewer. molecules: [{"smiles", "legend", optional "group"}] drawn as one 2D grid per
        group (steps before any pose exists). Without a receptor, or without poses, the step is table/text only."""
        self.steps.append(dict(key=key, title=title, summary=summary, table=table or [], poses=poses or [],
                               max_images=max_images, molecules=molecules or [], chart_html=chart_html, ligand=ligand, ref_name=ref_name))

    def _reference_block(self) -> str | None:
        try:
            return Path(self.reference).read_text() if self.reference and Path(self.reference).is_file() else None
        except OSError:
            return None

    def _step_html(self, n: int, s: dict, steps_js: dict, ref_block: str | None) -> str:
        out = [f'<section class="card" id="{s["key"]}"><div class="card-head"><span class="stepno">{n:02d}</span>'
               f'<h2>{html.escape(s["title"])}</h2><a class="totop" href="#top">&uarr; top</a></div>'
               f'<p class="summary">{html.escape(s["summary"])}</p>']
        if s["chart_html"]:
            out.append(f'<div class="chart">{s["chart_html"]}</div>')
        if s["table"]:
            out.append(_table_html(s["table"]))
        if s["molecules"]:
            groups: dict = {}
            for m in s["molecules"]:
                groups.setdefault(m.get("group", ""), []).append(m)
            for gi, (gname, ms) in enumerate(groups.items()):
                png = self.assets / f'{s["key"]}_grid_{gi}.png'
                if molecule_grid_png([m["smiles"] for m in ms], [m.get("legend", "") for m in ms], str(png)):
                    out.append(f'<h3>{html.escape(gname or "Molecules")}</h3><div class="scroll"><div class="strip grid2d"><figure>'
                               f'<img src="{_b64(png)}" alt="molecule grid"></figure></div></div>')
        if s["poses"] and self.receptor:
            figs, models = [], []
            for k, p in enumerate(s["poses"]):
                ref_text = p.get("reference_sdf") or ref_block
                models.append(dict(sdf=p["sdf"], ref=ref_text, lc=s["ligand"] + "Carbon"))
                if k < s["max_images"]:
                    sdf = self.assets / f'{s["key"]}_{k}.sdf'
                    png = self.assets / f'{s["key"]}_{k}.png'
                    ref_arg = self.reference
                    # PyMOL stills are cached: re-render only when the ligand (or its overlay), label or style changed
                    stamp = json.dumps([STYLE_VERSION, p["sdf"], p.get("reference_sdf"), p["name"], str(self.receptor), self.reference, s["ligand"]])
                    stamp_file = png.with_suffix(".stamp")
                    fresh = png.exists() and stamp_file.exists() and stamp_file.read_text() == stamp
                    sdf.write_text(p["sdf"])
                    if p.get("reference_sdf"):
                        ref_path = self.assets / f'{s["key"]}_{k}_ref.sdf'
                        ref_path.write_text(p["reference_sdf"])
                        ref_arg = str(ref_path)
                    if fresh or render_pose_png(self.receptor, str(sdf), str(png), ref_arg, label=p["name"], ligand_color=s["ligand"]):
                        stamp_file.write_text(stamp)
                        figs.append(f'<figure><img src="{_b64(png)}" alt="{html.escape(p["name"])}"><figcaption><b>{html.escape(p["name"])}</b> '
                                    f'{html.escape(p.get("caption", ""))}</figcaption></figure>')
            if figs:
                out.append(f'<h3>PyMOL stills</h3><div class="scroll"><div class="strip">{"".join(figs)}</div></div>'
                           f'<div class="hint">scroll sideways for more</div>')
            tabs = "".join(f'<button onclick="show(\'{s["key"]}\',{i})">{html.escape(p["name"])}</button>' for i, p in enumerate(s["poses"]))
            lig_hex = {"cyan": "#4cc9f0", "magenta": "#f72585"}.get(s["ligand"], "#4cc9f0")
            ref_legend = (f'<span><i style="background:#cfd6ee"></i>{html.escape(s["ref_name"])} (grey lines)</span>'
                          if ref_block or any(p.get("reference_sdf") for p in s["poses"]) else "")
            out.append(f'<h3>Rotatable view</h3><div class="viewwrap"><div class="tabs" id="t_{s["key"]}">{tabs}</div>'
                       f'<div class="viewer" id="v_{s["key"]}"></div>'
                       '<div class="legend"><span><i style="background:#7b8cff"></i>protein ribbon</span>'
                       '<span><i style="background:#9fd8a8"></i>pocket residues within 4 &Aring; (thin sticks)</span>'
                       f'<span><i style="background:{lig_hex}"></i>{"relaxed" if s["ligand"] == "magenta" else "docked"} ligand (thick sticks)</span>{ref_legend}'
                       '<span>drag to rotate, scroll to zoom</span></div></div>')
            steps_js[s["key"]] = models
        out.append("</section>")
        return "".join(out)

    def write(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else self.dir / "report.html"
        steps_js: dict = {}
        ref_block = self._reference_block()
        sections = "".join(self._step_html(n, s, steps_js, ref_block) for n, s in enumerate(self.steps, 1))
        lig_xyz = []
        for models in steps_js.values():
            for m in models:
                mol = Chem.MolFromMolBlock(m["sdf"], removeHs=False, sanitize=False)
                m["pocket"] = []
                if mol is not None and mol.GetNumConformers():
                    xyz = mol.GetConformer().GetPositions()
                    lig_xyz.append(xyz)
                    if self.receptor:
                        m["pocket"] = pocket_residues(self.receptor, xyz)
        prot = receptor_for_view(self.receptor, lig_xyz) if (self.receptor and lig_xyz) else ""
        nav = "".join(f'<a href="#{s["key"]}">{html.escape(s["title"])}</a>' for s in self.steps)
        hero = (f'<div class="hero"><div class="kicker">CafChem MedRAg &middot; medicinal chemistry agent</div>'
                f'<h1>{html.escape(self.title)}</h1><div class="sub">{html.escape(self.subtitle)}</div>'
                f'<div class="chips">{"".join(f"<span class=chip><b>{html.escape(a)}</b>{html.escape(str(b))}</span>" for a, b in self.chips)}</div></div>')
        tiles = ("<div class=tiles>" + "".join(
            f'<div class="tile {t.get("tone", "cyan")}"><div class="v">{html.escape(str(t["value"]))}</div>'
            f'<div class="l">{html.escape(t["label"])}</div><div class="s">{html.escape(t.get("sub", ""))}</div></div>' for t in self.stats) + "</div>") if self.stats else ""

        def js_json(x) -> str:
            return json.dumps(x).replace("</", "<\\/")
        page = (_PAGE.replace("%%TITLE%%", html.escape(self.title)).replace("%%CDN%%", CDN).replace("%%CSS%%", _CSS)
                .replace("%%NAV%%", nav).replace("%%HERO%%", hero).replace("%%TILES%%", tiles).replace("%%SECTIONS%%", sections)
                .replace("%%DATE%%", time.strftime("%Y-%m-%d %H:%M")).replace("%%JS%%", _JS.replace("%%PROT%%", js_json(prot)).replace("%%STEPS%%", js_json(steps_js))))
        path.write_text(page)
        return path
