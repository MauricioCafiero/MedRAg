"""Per-step HTML report: a results table, PyMOL stills and a rotatable 3Dmol.js viewer for each pipeline step.

Usage:
    rep = Report(run_dir, title="HMGCR hit-to-lead", receptor="path/to/receptor.pdb")
    rep.add_step("dock", "Docking", "Top poses at the FBI site", table=[{...}, ...],
                 poses=[{"name": "pose 1", "sdf": block, "caption": "vina -9.1"}, ...])
    rep.write()          # -> <run_dir>/report.html  (single file; images are embedded)

The viewer page follows cafquantum/code/view.py (3Dmol.js from the same CDN, one viewer switched between models);
PyMOL runs headless in a subprocess (report_pymol.py) because it keeps global state.
"""
from __future__ import annotations

import base64
import html
import json
import subprocess
import sys
import webbrowser
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

CDN = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.0/3Dmol-min.js"
STATUS_CLASS = {"pass": "ok", "warn": "warn", "fail": "bad"}


def render_pose_png(receptor: str, ligand_file: str, out_png: str, reference: str | None = None,
                    label: str = "", width: int = 700, height: int = 520, timeout: int = 180) -> bool:
    """PyMOL still of one ligand in its pocket. Returns False (and leaves no image) if PyMOL fails."""
    spec = Path(out_png).with_suffix(".json")
    spec.write_text(json.dumps(dict(receptor=str(receptor), ligand=str(ligand_file), out=str(out_png),
                                    reference=reference, label=label, width=width, height=height)))
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


def pocket_pdb_text(receptor_pdb: str, lig_xyz: np.ndarray, radius: float = 12.0) -> str:
    """ATOM records of every residue with an atom within `radius` A of the ligand (keeps the 3Dmol page small)."""
    lines, keys, coords = [], [], []
    for line in open(receptor_pdb):
        if line.startswith(("ATOM", "HETATM")) and line[17:20].strip() not in ("HOH", "WAT"):
            lines.append(line)
            keys.append(line[21:27])
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    xyz = np.array(coords)
    d = np.linalg.norm(xyz[:, None, :] - lig_xyz[None, :, :], axis=2).min(axis=1)
    near = {k for k, dist in zip(keys, d) if dist <= radius}
    return "".join(l for l, k in zip(lines, keys) if k in near)


def molecule_grid_png(smiles: list[str], legends: list[str], out_png: str, per_row: int = 4, size=(230, 190)) -> bool:
    """RDKit 2D grid image of molecules (no receptor needed): the picture for steps that have no pose yet."""
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    keep = [(m, l) for m, l in zip(mols, legends) if m is not None]
    if not keep:
        return False
    Draw.MolsToGridImage([m for m, _ in keep], molsPerRow=per_row, subImgSize=size, legends=[l for _, l in keep],
                         returnPNG=False).save(out_png)
    return True


def _b64(png: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()


def _table_html(rows: list[dict]) -> str:
    cols = list(rows[0])
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    body = ""
    for r in rows:
        cells = ""
        for c in cols:
            v = r.get(c, "")
            cls = STATUS_CLASS.get(str(v).lower(), "")
            txt = f"{v:.3g}" if isinstance(v, float) else str(v)
            cells += f'<td class="{cls}">{html.escape(txt)}</td>'
        body += f"<tr>{cells}</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<script src="{cdn}"></script>
<style>
  body {{ font-family: -apple-system, Helvetica, sans-serif; margin: 0 auto; max-width: 1100px; padding: 16px; color: #222; }}
  h1 {{ margin-bottom: 4px; }} .sub {{ color: #777; margin-bottom: 20px; }}
  nav a {{ margin-right: 14px; font-size: 14px; }}
  section {{ border-top: 1px solid #ddd; padding: 18px 0; }}
  table {{ border-collapse: collapse; font-size: 13px; margin: 10px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 4px 10px; text-align: left; }} th {{ background: #f3f3f3; }}
  td.ok {{ background: #dff3df; }} td.warn {{ background: #fff0c8; }} td.bad {{ background: #f8d7d7; }}
  .gallery {{ display: flex; flex-wrap: wrap; gap: 12px; }}
  figure {{ margin: 0; }} figure img {{ width: 340px; border: 1px solid #ddd; }} figcaption {{ font-size: 12px; color: #555; }}
  .viewer {{ position: relative; width: 100%; height: 460px; border: 1px solid #ddd; margin-top: 8px; }}
  .tabs button {{ border: 1px solid #ccc; background: #fff; padding: 3px 10px; margin: 2px; border-radius: 4px; cursor: pointer; }}
  .tabs button.active {{ background: #3b6fa0; color: #fff; }}
  .note {{ font-size: 12px; color: #777; }}
</style></head><body>
<h1>{title}</h1><div class="sub">{subtitle}</div>
<nav>{nav}</nav>
{sections}
<p class="note">Rotatable views use 3Dmol.js from a CDN (internet needed on first open). Stills are PyMOL: protein cartoon,
pocket residues green, ligand cyan, polar contacts &lt; 3.5 &Aring; yellow, reference pose grey.</p>
<script>
const STEPS = {steps_json};
const views = {{}};
function show(step, i) {{
  const s = STEPS[step]; if (!s) return;
  if (!views[step]) views[step] = $3Dmol.createViewer(document.getElementById('v_' + step), {{ backgroundColor: 'white' }});
  const v = views[step], m = s[i];
  v.clear();
  v.addModel(m.pocket, 'pdb');
  v.setStyle({{}}, {{ cartoon: {{ color: 'lightgrey', opacity: 0.6 }} }});
  v.setStyle({{ within: {{ distance: 5, sel: {{ model: 1 }} }} }}, {{ stick: {{ radius: 0.12, colorscheme: 'greenCarbon' }} }});
  v.addModel(m.sdf, 'sdf');
  v.setStyle({{ model: 1 }}, {{ stick: {{ radius: 0.25, colorscheme: 'cyanCarbon' }} }});
  v.zoomTo({{ model: 1 }}); v.render();
  document.querySelectorAll('#t_' + step + ' button').forEach((b, k) => b.classList.toggle('active', k === i));
}}
Object.keys(STEPS).forEach(k => show(k, 0));
</script></body></html>
"""


class Report:
    def __init__(self, run_dir: str | Path, title: str = "medrag report", receptor: str | None = None,
                 subtitle: str = "", reference_sdf: str | None = None):
        self.dir = Path(run_dir)
        self.assets = self.dir / "report_assets"
        self.assets.mkdir(parents=True, exist_ok=True)
        self.title, self.subtitle, self.receptor, self.reference = title, subtitle, receptor, reference_sdf
        self.steps: list[dict] = []

    def add_step(self, key: str, title: str, summary: str = "", table: list[dict] | None = None,
                 poses: list[dict] | None = None, max_images: int = 6, molecules: list[dict] | None = None):
        """poses: [{"name", "sdf" (mol block text), "caption", optional "reference_sdf" drawn grey behind it}]. Each gets a PyMOL still (first `max_images`) and a
        tab in the 3Dmol viewer. Without a receptor, or without poses, the step is table/text only.
        molecules: [{"smiles", "legend"}] drawn as one 2D grid image (for steps before any pose exists)."""
        self.steps.append(dict(key=key, title=title, summary=summary, table=table or [], poses=poses or [],
                               max_images=max_images,
                               molecules=molecules or []))

    def _step_html(self, s: dict, steps_js: dict) -> str:
        out = [f'<section id="{s["key"]}"><h2>{html.escape(s["title"])}</h2><p>{html.escape(s["summary"])}</p>']
        if s["table"]:
            out.append(_table_html(s["table"]))
        if s["molecules"]:
            png = self.assets / f'{s["key"]}_grid.png'
            if molecule_grid_png([m["smiles"] for m in s["molecules"]], [m.get("legend", "") for m in s["molecules"]], str(png)):
                out.append(f'<div class="gallery"><figure><img src="{_b64(png)}" style="width:100%;max-width:960px"></figure></div>')
        if s["poses"] and self.receptor:
            figs, models = [], []
            for n, p in enumerate(s["poses"]):
                lig = Chem.MolFromMolBlock(p["sdf"], removeHs=False, sanitize=False)
                xyz = lig.GetConformer().GetPositions()
                models.append(dict(sdf=p["sdf"], pocket=pocket_pdb_text(self.receptor, xyz)))
                if n < s["max_images"]:
                    sdf = self.assets / f'{s["key"]}_{n}.sdf'
                    png = self.assets / f'{s["key"]}_{n}.png'
                    ref = self.reference
                    # PyMOL stills are cached: re-render only when the ligand (or its reference overlay) or label changed
                    stamp = json.dumps([p["sdf"], p.get("reference_sdf"), p["name"], str(self.receptor), self.reference])
                    stamp_file = png.with_suffix(".stamp")
                    fresh = png.exists() and stamp_file.exists() and stamp_file.read_text() == stamp
                    sdf.write_text(p["sdf"])
                    if p.get("reference_sdf"):          # e.g. the docked pose behind a relaxed one
                        ref_path = self.assets / f'{s["key"]}_{n}_ref.sdf'
                        ref_path.write_text(p["reference_sdf"])
                        ref = str(ref_path)
                    if fresh or render_pose_png(self.receptor, str(sdf), str(png), ref, label=p["name"]):
                        stamp_file.write_text(stamp)
                        figs.append(f'<figure><img src="{_b64(png)}"><figcaption><b>{html.escape(p["name"])}</b> '
                                    f'{html.escape(p.get("caption", ""))}</figcaption></figure>')
            out.append(f'<div class="gallery">{"".join(figs)}</div>')
            tabs = "".join(f'<button onclick="show(\'{s["key"]}\',{i})">{html.escape(p["name"])}</button>'
                           for i, p in enumerate(s["poses"]))
            out.append(f'<div class="tabs" id="t_{s["key"]}">{tabs}</div><div class="viewer" id="v_{s["key"]}"></div>')
            steps_js[s["key"]] = models
        out.append("</section>")
        return "".join(out)

    def write(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else self.dir / "report.html"
        steps_js: dict = {}
        sections = "".join(self._step_html(s, steps_js) for s in self.steps)
        nav = "".join(f'<a href="#{s["key"]}">{html.escape(s["title"])}</a>' for s in self.steps)
        path.write_text(_PAGE.format(title=html.escape(self.title), subtitle=html.escape(self.subtitle), cdn=CDN,
                                     nav=nav, sections=sections, steps_json=json.dumps(steps_js)))
        return path
