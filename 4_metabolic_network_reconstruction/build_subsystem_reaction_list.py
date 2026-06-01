"""
Build a list of iMM1865 reactions for each subsystem named in
list_subsystem_of_interest.txt.

Strategy
--------
The interest list uses informal pathway names that do not all match the
"subsystem" annotation field of the iMM1865 SBML model. For each interest-
list entry we therefore use one (or both) of two selection rules:

1. Subsystem match - the reaction's `subsystem` attribute matches a known
   iMM1865 subsystem string (e.g. "Citric acid cycle" for TCA cycle).
2. Metabolite/keyword match - the reaction touches a curated set of
   metabolite IDs (e.g. glc_D, lac_L, akg, gln_L, pcreat) and/or its ID
   matches a curated regex (e.g. ^HEX1, ^CK, ^PDH, ^LDH).

The mapping below was chosen using common BiGG / Recon3D / iMM1865
naming conventions:
  * Glucose uptake/phosphorylation    -> glc_D transport + hexokinase (HEX1, GLCt*)
  * Fatty acid uptake & beta-oxidation-> "Fatty acid oxidation" subsystem
                                          plus extracellular FA transport
  * Ketone metabolism                  -> reactions on bhb, acac, acac_c (BDHm, OCOAT1m, ACACT*)
  * Glycolysis                         -> "Glycolysis/gluconeogenesis"
  * Lactate metabolism                 -> LDH_*, L-lactate transport (lac_L)
  * Pyruvate oxidation                 -> PDHm + pyr mitochondrial transport
  * TCA cycle                          -> "Citric acid cycle"
  * Oxidative phosphorylation          -> "Oxidative phosphorylation"
  * Glutamate / aKG metabolism         -> "Glutamate metabolism" + akg/glu transaminases
  * Glutamine metabolism               -> reactions on gln_L (GLNS, GLUNm, GLNt*)
  * Creatine phosphate buffering       -> creatine kinase (CK*) + creat/pcreat transport
  * Redox shuttles                     -> malate-aspartate (AKGMALtm, ASPGLUm, MDH*, GOT*)
                                          + glycerol-3-P shuttle (G3PD*)

Each rule is documented next to its definition so the selection logic can
be reproduced from scratch.

Output
------
results_subsystem_reactions/
    <safe_subsystem_name>.csv       per-subsystem reactions
    all_subsystems_reactions.csv    long-form table (subsystem, rxn_id,
                                    rxn_name, subsystem_in_model,
                                    reaction_string, selection_rule)
"""

import os
import re
import lzma
import tempfile
import cobra
import pandas as pd

MODEL_XZ = "iMM1865/iMM1865.xml.xz"
OUT_DIR  = "results_subsystem_reactions"

# ---------------------------------------------------------------------------
# Load model (decompress the xz-packaged SBML to a temp file first because
# cobra.io.read_sbml_model expects a path).
# ---------------------------------------------------------------------------
with lzma.open(MODEL_XZ) as f:
    raw = f.read()
with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as t:
    t.write(raw)
    tmp_path = t.name
model = cobra.io.read_sbml_model(tmp_path)
os.unlink(tmp_path)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def by_subsystem(names):
    """Reactions whose .subsystem is in `names` (exact, case-insensitive)."""
    names_lc = {n.lower() for n in names}
    return [r for r in model.reactions
            if r.subsystem and r.subsystem.lower() in names_lc]

def by_metabolite(met_ids, id_regex=None, exclude_subsystems=None):
    """Reactions that consume/produce any metabolite whose *base* id (before
    the compartment suffix _c/_m/_e/...) is in `met_ids`. Optionally filter
    further by reaction-id regex and/or drop reactions whose subsystem is in
    `exclude_subsystems`."""
    met_ids = set(met_ids)
    pat = re.compile(id_regex) if id_regex else None
    excl = {s.lower() for s in (exclude_subsystems or [])}
    hits = []
    for r in model.reactions:
        mets = {m.id.rsplit("_", 1)[0] for m in r.metabolites}
        if not (mets & met_ids):
            continue
        if pat and not pat.search(r.id):
            continue
        if r.subsystem and r.subsystem.lower() in excl:
            continue
        hits.append(r)
    return hits

def by_id_regex(id_regex):
    pat = re.compile(id_regex)
    return [r for r in model.reactions if pat.search(r.id)]

def dedup(rxns):
    seen, out = set(), []
    for r in rxns:
        if r.id in seen: continue
        seen.add(r.id); out.append(r)
    return out

# ---------------------------------------------------------------------------
# Selection rules per interest-list subsystem.
#
# Each entry returns a list of cobra Reaction objects together with a short
# tag describing *why* the reaction was selected (so the user can audit).
# ---------------------------------------------------------------------------
def sel_glucose_uptake():
    # extracellular/cytosolic glucose transport + hexokinase phosphorylation
    transport = by_metabolite({"glc__D", "glc_D"},
                              id_regex=r"(GLCt|GLUT|HEX1|GLCter|GLCtg)")
    hk = by_id_regex(r"^(HEX1|GLCK|GLK)")
    return dedup(transport + hk)

def sel_fa_uptake_betaox():
    # iMM1865 has a dedicated subsystem; add extracellular FA transports
    fa_ox = by_subsystem(["Fatty acid oxidation"])
    # restrict to extracellular transport of named long-chain FAs;
    # the metabolite-id base must equal one of these (no "fa*" prefix
    # matching since that grabbed FAD-containing reactions).
    fa_bases = {"hdca", "ocdca", "ocdcea", "ttdca", "lnlc", "lnlnca",
                "arach", "arachd", "crvnc", "dca", "ddca", "hdcea",
                "strdnc", "tmndnc", "clpnd", "eicostet"}
    fa_transport = [r for r in model.reactions
                    if r.subsystem == "Transport, extracellular"
                    and any(m.id.rsplit("_", 1)[0] in fa_bases
                            for m in r.metabolites)]
    return dedup(fa_ox + fa_transport)

def sel_ketone():
    # 3-hydroxybutyrate (bhb), acetoacetate (acac), acetoacetyl-CoA (aacoa)
    return dedup(by_metabolite({"bhb", "acac", "aacoa"},
                               id_regex=r"(BDH|OCOAT|ACACT|HMGC|HMGL|ACS)"))

def sel_glycolysis():
    return by_subsystem(["Glycolysis/gluconeogenesis"])

def sel_lactate():
    # any reaction touching L-lactate, plus LDH IDs
    return dedup(by_metabolite({"lac__L", "lac_L"})
                 + by_id_regex(r"^(LDH|L_LACt|L_LACtm)"))

def sel_pyruvate_ox():
    # pyruvate dehydrogenase + pyruvate mito transport
    return dedup(by_id_regex(r"^(PDH|PYRt2m|PYRt2|MPC)"))

def sel_tca():
    return by_subsystem(["Citric acid cycle"])

def sel_oxphos():
    return by_subsystem(["Oxidative phosphorylation"])

def sel_glu_akg():
    base = by_subsystem(["Glutamate metabolism"])
    # transaminases producing/consuming akg+glu
    trans = by_id_regex(r"^(AKGMAL|GLUDx|GLUDy|ASPTA|ALATA|GLNS|GLUN)")
    return dedup(base + trans)

def sel_glutamine():
    # Restrict to canonical glutamine-handling reactions and transport;
    # exclude amino-acyl tRNA charging and protein-formation reactions
    # which technically consume gln_L but are not "glutamine metabolism".
    excl = {"Protein formation", "Peptide metabolism"}
    hits = by_metabolite({"gln__L", "gln_L"},
                         id_regex=r"^(GLN|GLU|ASN|GMP|CTP|NAD|PRPP|GF6P|"
                                  r"r\d+|EX_gln|sink_gln)",
                         exclude_subsystems=excl)
    hits += by_id_regex(r"^(GLNS|GLUN|GLNtm|GLNt|r0941|GLUDC)")
    return dedup(hits)

def sel_creatine():
    # creatine kinase (CK*) and creatine/phosphocreatine transport
    return dedup(by_id_regex(r"^(CK|CREAT|PCREAT)")
                 + by_metabolite({"creat", "pcreat"}))

def sel_redox_shuttles():
    # malate-aspartate shuttle + glycerol-3-phosphate shuttle
    mas = by_id_regex(r"^(AKGMALtm|MALtm|ASPGLUm|GOT2m|GOT1|MDH|MDHm)")
    g3p = by_id_regex(r"^(G3PD1|G3PD2m|GPDm|GPDH)")
    return dedup(mas + g3p)

GROUPS = [
    ("Uptake",         "Glucose uptake/phosphorylation",  sel_glucose_uptake),
    ("Uptake",         "Fatty acid uptake & beta-oxidation", sel_fa_uptake_betaox),
    ("Uptake",         "Ketone metabolism",               sel_ketone),
    ("Central carbon", "Glycolysis",                      sel_glycolysis),
    ("Central carbon", "Lactate metabolism",              sel_lactate),
    ("Central carbon", "Pyruvate oxidation",              sel_pyruvate_ox),
    ("Central carbon", "TCA cycle",                       sel_tca),
    ("Energy",         "Oxidative phosphorylation",       sel_oxphos),
    ("Nitrogen",       "Glutamate/aKG metabolism",        sel_glu_akg),
    ("Nitrogen",       "Glutamine metabolism",            sel_glutamine),
    ("Energy/redox",   "Creatine phosphate buffering",    sel_creatine),
    ("Energy/redox",   "Redox shuttles",                  sel_redox_shuttles),
]

# ---------------------------------------------------------------------------
# Run selections, write CSVs.
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
long_rows = []
for group, name, fn in GROUPS:
    rxns = fn()
    rows = [{
        "group":              group,
        "subsystem_interest": name,
        "rxn_id":             r.id,
        "rxn_name":           r.name,
        "subsystem_in_model": r.subsystem or "",
        "reaction":           r.build_reaction_string(use_metabolite_names=False),
        "lb": r.lower_bound, "ub": r.upper_bound,
    } for r in rxns]
    df = pd.DataFrame(rows)
    safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    df.to_csv(f"{OUT_DIR}/{safe}.csv.xz", index=False)
    long_rows.extend(rows)
    print(f"{name:38s} -> {len(rxns):4d} reactions")

pd.DataFrame(long_rows).to_csv(
    f"{OUT_DIR}/all_subsystems_reactions.csv.xz", index=False)
print(f"\nWrote {len(long_rows)} rows to {OUT_DIR}/all_subsystems_reactions.csv")
