"""filter step: protonate at pH 7.4, drop duplicates / the seed itself, then grade each candidate on drug-likeness
(RDKit QED + Lipinski), structural alerts (RDKit FilterCatalog) and ADMET (MoDrAg admet_node, ADMET-AI models).

Status per candidate: fail (dropped from docking), warn (kept, flagged), pass. The cut-offs below are PROVISIONAL
defaults for an early screen, not validated numbers."""
from __future__ import annotations

import sys
import warnings
from collections import Counter

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from medrag import ui
from medrag.state import RunState
from medrag.tools import vendor
from medrag.tools.vendor import VENDOR

sys.path.insert(0, str(VENDOR / "cafprot"))
import cafprot                                                       # noqa: E402  (vendored, not a package)

RDLogger.DisableLog("rdApp.*")

PH = 7.4
CUTS = dict(
    qed=(0.35, 0.2),                 # (warn below, fail below)
    lipinski_violations=(1, 2),      # (warn above, fail above)
    tanimoto_to_seed=(0.25, 0.1),    # (warn below, fail below): is it still an analogue?
    hERG=(0.5, 0.8), AMES=(0.5, 0.8), DILI=(0.7, 0.9),   # ADMET-AI probabilities (warn above, fail above)
    Carcinogens_Lagunin=(0.5, 0.85),   # noisy model, so a high bar to fail
    Skin_Reaction=(0.85, 0.95),        # sensitisation: ADMET-AI scores every catechol 0.7-0.9, so a lower bar flags the whole series
    LD50_mg_kg=(300.0, 50.0),          # (warn below, fail below) predicted rat oral LD50; GHS category 3 / category 1-2 limits
    CYP_inhibitors=(0, 3),             # number of the 5 isoforms predicted to be inhibited (warn above 0, fail above 3)
    Bioavailability_Ma=(0.5, 0.2),     # probability of F >= 30% (warn below, fail below)
    BBB_avoid=0.7,                     # bbb="avoid": warn above this probability; "required": warn below 0.5, fail below 0.2
)
CYP_ISOFORMS = ("CYP1A2_Veith", "CYP2C19_Veith", "CYP2C9_Veith", "CYP2D6_Veith", "CYP3A4_Veith")
CYP_INHIBITED_AT = 0.6

_PAINS = FilterCatalogParams()
_PAINS.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_BRENK = FilterCatalogParams()
_BRENK.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
PAINS, BRENK = FilterCatalog(_PAINS), FilterCatalog(_BRENK)


# Motifs no real molecule of this kind has; generators (masked-LM infilling, GPT) do produce them. A cumulene or alkyne inside a
# small ring is hopelessly strained, and the docked pose of one cannot even be matched back to its SMILES.
_UNREALISTIC = [(Chem.MolFromSmarts(sma), name) for sma, name in (
    ("[#6;R]=[#6;R]=[#6;R]", "allene inside a ring"),
    ("[#6;r3,r4,r5,r6,r7]#[#6;r3,r4,r5,r6,r7]", "alkyne inside a small ring"),
    ("[#6;r3,r4]=[#6;r3,r4]=,#[#6,#7]", "cumulated bonds in a 3/4-membered ring"),
)]


def unrealistic_motif(mol: Chem.Mol) -> str | None:
    """Name of the first unrealistic motif in `mol`, or None."""
    return next((name for patt, name in _UNREALISTIC if mol.HasSubstructMatch(patt)), None)


def protonate(smiles: str, ph: float = PH) -> str:
    """cafprot (Dimorphite-DL) charge state at `ph`, with aromatic phenolates and [n-] neutralised again: Dimorphite flags
    catechols/phenols as ambiguous and 'class-mean' deprotonates them, although phenol pKa is ~10 (catechol ~9.3), so at pH 7.4
    they are neutral. Amines and carboxylic acids keep cafprot's assignment. Electron-poor phenols (pKa < 8) are therefore
    missed - a known limitation."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        norm = cafprot.normalize_smiles(smiles, ph, warn_if_ambiguous=False)
    if any("protonation failed" in str(w.message) for w in caught) and not getattr(protonate, "_warned", False):
        protonate._warned = True                      # said once: cafprot returns the input unchanged when its dependency is missing
        ui.warn("charge states were NOT assigned (cafprot's dimorphite-dl is missing: run setup.sh); molecules keep their input charges")
    mol = Chem.MolFromSmiles(norm)
    if mol is None:
        return smiles
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        if a.GetFormalCharge() == -1 and ((a.GetSymbol() == "O" and a.GetDegree() == 1 and a.GetNeighbors()[0].GetIsAromatic())
                                          or (a.GetSymbol() == "N" and a.GetIsAromatic())):
            a.SetFormalCharge(0)
            a.SetNumExplicitHs(0)
            a.SetNoImplicit(False)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return norm
    return Chem.MolToSmiles(out)


def _rank(value: float, warn: float, fail: float, high_is_bad: bool) -> str:
    bad = (lambda v, t: v > t) if high_is_bad else (lambda v, t: v < t)
    return "fail" if bad(value, fail) else "warn" if bad(value, warn) else "pass"


def _props(mol: Chem.Mol) -> dict:
    hbd, hba = Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol)
    mw, logp = Descriptors.MolWt(mol), Descriptors.MolLogP(mol)
    return dict(qed=QED.qed(mol), mw=mw, logp=logp, hbd=hbd, hba=hba, tpsa=Descriptors.TPSA(mol),
                rotb=Lipinski.NumRotatableBonds(mol),
                lipinski_violations=int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10))


def _admet(smiles: list[str]) -> list[dict | None]:
    """ADMET-AI predictions through MoDrAg's admet_node (isolated process; first call loads the model, ~10 s)."""
    res = vendor.call("MoDrAg_CLI", "admet_node", "admet_node", cwd="runs/_admet_cwd", smiles=smiles)
    return res[0] if res and res[0] is not None else [None] * len(smiles)


# A candidate is only judged "worse" on an ADMET endpoint when it is clearly worse than the SEED: each cut-off is no tighter
# than the seed's own value plus a margin. Absolute cut-offs alone reject whole series the lead belongs to (HMGCR: rosuvastatin
# itself has DILI 0.95, the median of its 151 analogues is 0.92).
MARGINS = dict(hERG=0.05, AMES=0.05, DILI=0.05, Carcinogens_Lagunin=0.05, Skin_Reaction=0.05, Bioavailability_Ma=0.05,
               CYP_inhibitors=0.0, LD50_factor=0.7)
_HIGH_BAD = ("hERG", "AMES", "DILI", "Carcinogens_Lagunin", "Skin_Reaction", "CYP_inhibitors")


def admet_values(adm: dict, mol: Chem.Mol) -> dict:
    """The graded ADMET quantities of one molecule as floats (ADMET-AI values arrive as strings). LD50_Zhu is
    log10(1/(mol/kg)) - higher is MORE toxic (the admet_node docstring says log mg/kg; checked against caffeine, nicotine,
    strychnine and paracetamol, whose conversions land within ~5x of their known LD50) - so it is converted to mg/kg."""
    f = lambda k: float(adm[k]) if k in adm else None
    v = {k: f(k) for k in ("hERG", "AMES", "DILI", "Carcinogens_Lagunin", "Skin_Reaction", "Bioavailability_Ma", "BBB_Martins")}
    if f("LD50_Zhu") is not None:
        v["LD50_mg_kg"] = 10 ** (-f("LD50_Zhu")) * Descriptors.MolWt(mol) * 1000
    cyp = [i for i in CYP_ISOFORMS if f(i) is not None and f(i) > CYP_INHIBITED_AT]
    if any(f(i) is not None for i in CYP_ISOFORMS):
        v["CYP_inhibitors"] = len(cyp)
        v["CYP_inhibited"] = [i.split("_")[0] for i in cyp]
    return {k: x for k, x in v.items() if x is not None}


def effective_cuts(ref: dict | None) -> dict:
    """CUTS with every ADMET cut-off relaxed to at least the seed's value `ref` (+ margin); `ref` None = the plain defaults."""
    cuts = {k: v for k, v in CUTS.items() if isinstance(v, tuple)}
    if not ref:
        return cuts
    for k in _HIGH_BAD:
        if k in ref:
            lim = ref[k] + MARGINS[k]
            cuts[k] = (max(cuts[k][0], lim), max(cuts[k][1], lim))
    if "Bioavailability_Ma" in ref:
        lim = ref["Bioavailability_Ma"] - MARGINS["Bioavailability_Ma"]
        cuts["Bioavailability_Ma"] = (min(cuts["Bioavailability_Ma"][0], lim), min(cuts["Bioavailability_Ma"][1], lim))
    if "LD50_mg_kg" in ref:
        lim = ref["LD50_mg_kg"] * MARGINS["LD50_factor"]
        cuts["LD50_mg_kg"] = (min(cuts["LD50_mg_kg"][0], lim), min(cuts["LD50_mg_kg"][1], lim))
    return cuts


def _grade_admet(vals: dict, cuts: dict, bbb: str) -> dict:
    """Check statuses from one molecule's admet_values against `cuts`."""
    checks = {}
    for k in _HIGH_BAD:
        if k in vals:
            checks[k] = _rank(vals[k], *cuts[k], True)
    for k in ("LD50_mg_kg", "Bioavailability_Ma"):
        if k in vals:
            checks[k] = _rank(vals[k], *cuts[k], False)
    if "BBB_Martins" in vals and bbb != "ignore":
        checks["BBB"] = ("warn" if vals["BBB_Martins"] > CUTS["BBB_avoid"] else "pass") if bbb == "avoid" \
            else _rank(vals["BBB_Martins"], 0.5, 0.2, False)
    return checks


def filter_candidates(state: RunState, use_admet: bool = True, bbb: str = "avoid") -> dict:
    """Grade every candidate in the run state; stores rec['filter'] and rec['species'] (the protonated SMILES to dock).
    bbb: 'avoid' (peripheral target: warn if likely brain-penetrant), 'required' (CNS target: warn/fail if unlikely), 'ignore'."""
    seed = state.data["seed"]
    t0 = ui.step_start("filter", f"{len(state.candidates)} molecules | pH {PH}, motifs, QED/Lipinski/PAINS, ADMET-AI")
    seed_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(seed), 2, 2048)
    seen, recs = {}, []
    for smi, rec in state.candidates.items():
        species = protonate(smi)
        mol = Chem.MolFromSmiles(species)
        reasons = []
        if mol is None:
            rec["filter"] = dict(status="fail", reasons=["invalid after protonation"])
            continue
        if smi == seed or species == protonate(seed):
            rec["filter"] = dict(status="seed", reasons=["the molecule being optimised"])
            continue
        if species in seen:
            rec["filter"] = dict(status="fail", reasons=[f"duplicate of {seen[species]}"])
            continue
        bad = unrealistic_motif(mol)
        if bad:
            rec["filter"] = dict(status="fail", reasons=[f"unrealistic structure: {bad}"])
            continue
        seen[species] = smi
        rec["species"] = species
        p = _props(mol)
        p["tanimoto_to_seed"] = DataStructs.TanimotoSimilarity(seed_fp, AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
        status = {"qed": _rank(p["qed"], *CUTS["qed"], False),
                  "lipinski_violations": _rank(p["lipinski_violations"], *CUTS["lipinski_violations"], True),
                  }
        if any(src["source"] in ("genmask", "bioisostere") for src in rec["sources"]):
            # similarity to the seed only grades analogue generators; GPT molecules are new compounds for the target
            status["tanimoto_to_seed"] = _rank(p["tanimoto_to_seed"], *CUTS["tanimoto_to_seed"], False)
        pains = [e.GetDescription() for e in PAINS.GetMatches(mol)]
        brenk = [e.GetDescription() for e in BRENK.GetMatches(mol)]
        p["alerts"] = dict(pains=pains, brenk=brenk)
        if pains:
            status["pains"] = "warn"                      # warn only: the catechol seed itself is a PAINS match
        rec["filter"] = dict(props=p, checks=status, reasons=reasons)
        recs.append(rec)
    eff = effective_cuts(None)
    if use_admet and recs:
        seed_species = protonate(seed)
        out = _admet([seed_species] + [r["species"] for r in recs])
        seed_vals = admet_values(out[0], Chem.MolFromSmiles(seed_species)) if out[0] else {}
        state.data["seed_admet"] = seed_vals
        eff = effective_cuts(seed_vals)
        for rec, adm in zip(recs, out[1:]):
            rec["filter"]["admet"] = adm
            if adm:
                vals = admet_values(adm, Chem.MolFromSmiles(rec["species"]))
                rec["filter"]["checks"].update(_grade_admet(vals, eff, bbb))
                rec["filter"]["derived"] = {k: vals[k] for k in ("LD50_mg_kg", "CYP_inhibitors", "CYP_inhibited") if k in vals}
    for rec in recs:
        f = rec["filter"]
        f["reasons"] = [f"{k} {s}" for k, s in f["checks"].items() if s != "pass"]
        f["status"] = "fail" if "fail" in f["checks"].values() else "warn" if "warn" in f["checks"].values() else "pass"
    tally = {s: sum(r.get("filter", {}).get("status") == s for r in state.candidates.values()) for s in ("pass", "warn", "fail")}
    top = Counter(x for r in recs for x in r["filter"]["reasons"] if x.endswith("fail")).most_common(3)
    sv = state.data.get("seed_admet") or {}
    if sv:
        ui.note("ADMET cut-offs are no tighter than the seed's own values (seed: "
                + ", ".join(f"{k} {sv[k]:.2f}" for k in ("hERG", "AMES", "DILI") if k in sv) + ")")
    ui.step_done("filter", f"{tally['pass']} pass, {tally['warn']} warn, {tally['fail']} fail"
                 + (f" | most common fail flags: {', '.join(f'{k} ({v})' for k, v in top)}" if top else ""), t0)
    state.log_step("filter", ph=PH, cuts=CUTS, effective_admet_cuts={k: v for k, v in eff.items() if k in _HIGH_BAD + ("LD50_mg_kg", "Bioavailability_Ma")},
                   seed_admet=state.data.get("seed_admet"), admet=use_admet, bbb=bbb, **tally)
    state.save()
    return tally
