"""
MoDrAg ADMET node: predicts ADMET (absorption, distribution, metabolism,
excretion, toxicity) endpoints from SMILES using the local ADMET-AI Chemprop
v2 models (pip package `admet-ai`; the ~13 MB of MPNN ensemble weights ship
inside the wheel - no separate download, no network access at runtime).

Ported from ~/python_mac/admet/code/admet_tool.py (that repo also holds the
Admetica second-opinion variant). The heavy chemprop/lightning imports are
lazy (inside the first call) so modrag startup is unchanged, and the model
is loaded once and cached per process. CPU is faster than MPS for these
tiny MPNNs, matching the admet repo's finding.
"""

import os
import sys

# Module-level print flag - set from modrag.py
print_flag = True

os.environ.setdefault("TQDM_DISABLE", "1")  # silence plain tqdm bars

# Built-in curated subset: the 22 endpoints most useful for early screening.
# The agent picks 'subset' (default) for a quick safety read, 'full' for all
# columns, or passes an explicit list of endpoint names.
SUBSET_ENDPOINTS = [
    # toxicity
    'AMES',                      # Ames mutagenicity (probability)
    'hERG',                      # hERG cardiotoxicity / QT risk (probability)
    'DILI',                      # drug-induced liver injury (probability)
    'ClinTox',                   # clinical toxicity / withdrawn drugs (probability)
    'Carcinogens_Lagunin',       # carcinogenicity (probability)
    'Skin_Reaction',             # skin sensitization (probability)
    'LD50_Zhu',                  # acute oral toxicity, log mg/kg (value)
    # metabolism (CYP inhibition = *_Veith, substrate = *_CarbonMangels)
    'CYP1A2_Veith', 'CYP2C19_Veith', 'CYP2C9_Veith', 'CYP2D6_Veith',
    'CYP3A4_Veith',
    'CYP2C9_Substrate_CarbonMangels',
    'CYP2D6_Substrate_CarbonMangels',
    'CYP3A4_Substrate_CarbonMangels',
    # absorption / distribution
    'Pgp_Broccatelli',           # P-glycoprotein efflux (probability)
    'HIA_Hou',                   # human intestinal absorption (probability)
    'Bioavailability_Ma',        # oral bioavailability >= 30% (probability)
    'BBB_Martins',               # blood-brain barrier permeability (probability)
    'PAMPA_NCATS',               # passive permeability (probability)
    'Caco2_Wang',                # Caco-2 permeability, log cm/s (value)
    'PPBR_AZ',                   # plasma protein binding, % (value)
]

ALL_ENDPOINTS = SUBSET_ENDPOINTS + [
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53',          # stress response
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
    'NR-ER-LBD', 'NR-PPAR-gamma',                                # nuclear receptors
    'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ',          # clearance
    'Half_Life_Obach', 'VDss_Lombardo',                           # distribution/duration
    'Solubility_AqSolDB', 'Lipophilicity_AstraZeneca',            # physico
    'HydrationFreeEnergy_FreeSolv',
]

_MODEL = None  # cached ADMETModel across calls


def _get_model():
    """Load ADMET-AI once; chemprop/lightning imports happen lazily here."""
    global _MODEL
    if _MODEL is None:
        if print_flag:
            print("Loading ADMET-AI model (one-time, ~10s)...")
        from admet_ai import ADMETModel
        _MODEL = ADMETModel()
        _MODEL.device = "cpu"  # CPU is faster than MPS for these tiny models
    return _MODEL


def admet_node(smiles=None, endpoints: str = 'subset',
               include_percentiles: bool = False) -> (list, str, list):
    """Predicts ADMET properties (absorption, distribution, metabolism,
    excretion, toxicity) for one or more molecules from SMILES strings using
    local ADMET-AI Chemprop v2 models - no network access. Use this to screen
    candidate molecules for safety and PK before synthesis: mutagenicity,
    cardiotoxicity (hERG), liver injury, CYP inhibition/substrate,
    absorption/permeability, brain penetration, acute toxicity (LD50), and
    more. Invalid SMILES are reported per-molecule, never fatal.

      Args:
        smiles: a list of SMILES strings (a single SMILES string also works)
        endpoints: which endpoints to report:
            "subset" (default) - a curated panel of 22 endpoints most useful
                for early toxicity/safety screening: AMES mutagenicity, hERG
                cardiotoxicity, DILI, clinical toxicity, carcinogenicity,
                skin sensitization, LD50 (acute oral toxicity log mg/kg),
                CYP1A2/2C9/2C19/2D6/3A4 inhibition, CYP2C9/2D6/3A4 substrate,
                Pgp efflux, intestinal absorption, oral bioavailability,
                BBB permeability, PAMPA and Caco-2 permeability, plasma
                protein binding.
            "full" - every ADMET-AI column (~52 per molecule: adds nuclear
                receptors, stress responses, clearance, half-life, volume of
                distribution, solubility, lipophilicity, ...)
            or a list of endpoint names, e.g. ["AMES", "hERG"].
        include_percentiles: if True, also report each endpoint's percentile
            rank vs. DrugBank approved drugs (percent 0-100, so 90 = higher
            than 90% of approved drugs).

      Returns:
        admet_dicts: list of dicts, one per input molecule, mapping endpoint
            name -> predicted value. Probabilities are in [0,1] (higher =
            higher likelihood/risk); regression endpoints are in native units
            (e.g. LD50_Zhu is log mg/kg, Caco2_Wang is log cm/s).
        admet_string: a string of the per-molecule predictions for the user
        images: None (this tool generates no images)
    """
    print("admet tool")
    print('===================================================')

    # Coerce common argument shapes into a list of SMILES (the LLM calls this
    # node with **arguments and often passes a bare string, None, or an int).
    if smiles is None:
        return [], 'admet_node failed: no smiles provided', None
    if isinstance(smiles, str):
        smiles = [smiles]
    items = [str(s).strip() for s in smiles]

    # Resolve the endpoint column set against the requested mode. Order is
    # fixed by SUBSET_ENDPOINTS / the requested list, not by the model.
    if isinstance(endpoints, (list, tuple)):
        pick = [str(e) for e in endpoints]
        if print_flag:
            print(f'custom endpoint list ({len(pick)} names)')
    elif str(endpoints).strip().lower() in ('full', 'all'):
        pick = None  # keep every column
        if print_flag:
            print('full endpoint set')
    else:
        pick = SUBSET_ENDPOINTS
        if print_flag:
            print(f'curated subset ({len(SUBSET_ENDPOINTS)} endpoints)')

    # Validate SMILES first; per-molecule errors never abort the batch.
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")  # silence "SMILES Parse Error" on bad input

    valid, idx, blocks, admet_dicts = [], [], [""] * len(items), [None] * len(items)
    for i, s in enumerate(items):
        s = (s or "").strip()
        if s and Chem.MolFromSmiles(s) is not None:
            valid.append(s)
            idx.append(i)
        else:
            blocks[i] = f'[{i + 1}] {s!r}: ERROR - invalid SMILES'

    if not valid:
        return [], '\n\n'.join(blocks), None

    model = _get_model()

    # chemprop/lightning print tqdm bars + rank-zero chatter to stdout/stderr;
    # redirect both to devnull during predict (real errors still raise).
    # Flush first: anything already buffered (the tool header above) would
    # otherwise be written to fd 1 while it still points at devnull.
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    so, se = os.dup(1), os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        preds = model.predict(smiles=valid)  # DataFrame aligned to `valid`
    finally:
        os.dup2(so, 1)
        os.dup2(se, 2)
        os.close(devnull)
        os.close(so)
        os.close(se)

    all_cols = list(preds.columns)
    note, missing = '', []
    if pick is not None:
        # Walk the requested names in order; optionally interleave each
        # endpoint's DrugBank percentile column right after it.
        pct_suffix = '_drugbank_approved_percentile'
        cols, missing = [], []
        for name in pick:
            if name in all_cols:
                cols.append(name)
                if include_percentiles and (name + pct_suffix) in all_cols:
                    cols.append(name + pct_suffix)
                elif include_percentiles:
                    missing.append(name + pct_suffix)
            else:
                missing.append(name)
    else:
        cols = [c for c in all_cols
                if include_percentiles or not c.endswith('_drugbank_approved_percentile')]

    if missing:
        note = (f'Note: endpoint names not known to ADMET-AI: '
                f'{", ".join(missing)} (valid names are the ADMET-AI '
                f'columns, e.g. {", ".join(SUBSET_ENDPOINTS[:4])}...)')
        if print_flag:
            print(note)

    for j, i in enumerate(idx):
        row = preds.iloc[j]
        values = {}
        lines = [f'[{i + 1}] {valid[j]}']
        for c in cols:
            v = row[c]
            text = f'{float(v):.4g}' if hasattr(v, '__float__') else f'{v}'
            values[c] = text
            lines.append(f'  {c}: {text}')
        blocks[i] = '\n'.join(lines)
        admet_dicts[i] = values

    if print_flag:
        print(f'predicted {len(cols)} endpoints for {len(valid)} molecule(s)')

    return admet_dicts, '\n\n'.join(blocks + ([note] if note else [])), None