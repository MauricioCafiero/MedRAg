"""Headless PyMOL renderer, run in its own process (PyMOL keeps global state).
usage: report_pymol.py <spec.json>
spec: {"receptor": pdb, "ligand": sdf/mol/pdb path, "out": png, "width": 700, "height": 520, "label": str,
       "reference": optional ligand file drawn as a thin grey overlay (e.g. the crystal pose)}
Cartoon receptor, sticks for residues within 5 A of the ligand, ligand in cyan, polar contacts as yellow dashes.
"""
import json
import sys

import pymol
from pymol import cmd

spec = json.load(open(sys.argv[1]))
pymol.finish_launching(["pymol", "-cq"])
cmd.reinitialize()
cmd.load(spec["receptor"], "rec")
cmd.load(spec["ligand"], "lig")
cmd.remove("solvent")
cmd.hide("everything")
cmd.set_color("medrag_bg", [0.043, 0.063, 0.125])      # the report page's dark navy
cmd.bg_color("medrag_bg")
cmd.show("cartoon", "rec")
cmd.color("slate", "rec")
cmd.set("cartoon_transparency", 0.35)
cmd.select("pocket", "byres (rec within 5 of lig)")
cmd.show("sticks", "pocket")
cmd.util.cbag("pocket")
cmd.color("palegreen", "pocket and elem C")
cmd.show("sticks", "lig")
cmd.color("cyan", "lig and elem C")
cmd.set("stick_radius", 0.15, "pocket")
cmd.set("stick_radius", 0.25, "lig")
if spec.get("reference"):
    cmd.load(spec["reference"], "ref")
    cmd.show("lines", "ref")
    cmd.color("grey40", "ref")
cmd.distance("hb", "lig and (elem N+O)", "pocket and (elem N+O)", 3.5, mode=0)
cmd.hide("labels", "hb")
cmd.color("yellow", "hb")
cmd.set("dash_width", 3)
cmd.orient("lig")
cmd.zoom("lig", 9)
cmd.set("ray_shadow", 0)
cmd.set("antialias", 2)
cmd.png(spec["out"], int(spec.get("width", 700)), int(spec.get("height", 520)), dpi=100, ray=1)
cmd.quit()
