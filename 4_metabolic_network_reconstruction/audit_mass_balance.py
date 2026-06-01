"""
audit_mass_balance.py
=====================

Diagnostic audit of the FBA solution stored in
    results_fba_constrained_fit_B6/flux_distribution__hierarchical_atp.csv

Purpose
-------
A prior biological sanity check on `fba_constrained_by_MRI.ipynb` (a CORDA-built
muscle-specific iMM1865 model, MRI-constrained, ATP demand objective) flagged
three concerns about the saved fluxes:

  1. Acetyl-CoA imbalance.
       Palmitate beta-oxidation + PDHm should produce ~0.57 mmol/gDW/h of
       mitochondrial acetyl-CoA (`accoa_m`), but citrate synthase (CSm) only
       carried ~0.03. Where did the rest of the acetyl-CoA go?

  2. Succinate orphan.
       Succinate dehydrogenase (`SUCD1m`, complex II) is zero in every
       condition while AKGDm and CSm are non-zero. Stoichiometric mass-balance
       at steady state (S v = 0) must still hold inside the LP, so something
       else is consuming succinate (or producing it).

  3. PFK = 0 in some conditions.
       Phosphofructokinase (`PFK`) carries zero flux in `Gastrocnemius__dia`,
       `Soleus__norm`, and `TibialisAnt__dia`, while `PGI` runs at ~0.11.
       Fructose-6-P is being made; FBP must come from a non-PFK route. A
       candidate is `r0191` (UTP-dependent F6P -> FBP).

This script is a *read-only* offline audit. It loads the muscle model and the
saved flux CSV, then for each target metabolite computes a mass-balance table:

        contribution_i = stoich_coef_i * flux_i

summed over every reaction touching that metabolite. At a feasible FBA optimum
this sum must be ~ 0 (numerically), so the table tells us *which alternative
producers and consumers FBA picked* to close the balance. That answers the
biological question (which sink absorbed the unaccounted acetyl-CoA, which
reaction substitutes for SUCD1m, which substitutes for PFK).

Inputs
------
  - metabolic_network/muscle_specific_model.json.xz
        CORDA-extracted muscle model in compressed COBRApy JSON.
  - metabolic_network/results_fba_constrained_fit_B6/
        flux_distribution__hierarchical_atp.csv
        Long-form CSV: columns reaction_id, flux, condition, tissue, group, week.
        7655 reactions x 6 conditions = 45,930 rows.

Outputs
-------
  - results_fba_constrained_fit_B6/audit/
        mass_balance_<met>.csv  (one per metabolite in TARGET_METABOLITES)
        audit_summary.md        (human-readable per-condition narrative)

Run
---
    python metabolic_network/audit_mass_balance.py

The script is intentionally self-contained (no helper imports from the
notebook) so it can be re-run independently. It is also heavily commented so
that the analysis is reproducible from the comments alone, per project policy
for deliverable scripts.
"""

from __future__ import annotations

import json
import lzma
from pathlib import Path

import cobra
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Repo-relative paths. Resolved from the location of this script so the audit
# does not depend on the caller's CWD.
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "muscle_specific_model.json.xz"
RESULTS_DIR = HERE / "results_fba_constrained_fit_B6"
FLUX_CSV = RESULTS_DIR / "flux_distribution__hierarchical_atp.csv"
AUDIT_DIR = RESULTS_DIR / "audit"

# Metabolites to audit. Each is examined per-condition for full producer /
# consumer breakdown. Notes on why each is in the list:
#
#   accoa_m   - mitochondrial acetyl-CoA. The acetyl-CoA imbalance suspect:
#               where did palmitate-derived acetyl-CoA go if not into CSm?
#   accoa_c   - cytosolic acetyl-CoA. Lipid resynthesis / acetylation sinks
#               could siphon acetyl-CoA out of the mitochondrion via citrate
#               lyase; auditing this catches such routing.
#   succ_m    - mitochondrial succinate. SUCD1m=0 means a non-canonical
#               consumer is closing the balance.
#   succoa_m  - mitochondrial succinyl-CoA. AKGDm produces succoa_m, NOT
#               succ_m directly. The bridge AKGDm -> succoa_m -> succ_m runs
#               through SUCOASm (succinyl-CoA synthetase). If SUCOASm is not
#               carrying flux, succoa_m must be sunk by something else
#               (heme synthesis, propionyl-CoA, etc.).
#   fdp_c     - cytosolic fructose-1,6-bisphosphate. The PFK=0 puzzle: who is
#               making FBP if PFK is off but glycolysis is running?
TARGET_METABOLITES = ["accoa_m", "accoa_c", "succ_m", "succoa_m", "fdp_c"]

# Numerical thresholds. Contributions smaller than CONTRIBUTION_TOL (in
# mmol/gDW/h, same units as fluxes) are dropped from the per-metabolite tables
# as numerical noise. CLOSURE_TOL is what we accept as "mass-balance closes".
# COBRApy / glpk / CPLEX typically report ~1e-9 residuals for a feasible LP.
CONTRIBUTION_TOL = 1e-9
CLOSURE_TOL = 1e-6

# Number of top producers and top consumers to surface per condition in the
# markdown narrative. The full list is always saved to the CSV.
TOP_N = 5


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_model(path: Path) -> cobra.Model:
    """
    Load the CORDA-extracted muscle model from a compressed COBRApy JSON file.

    The notebook fba_constrained_by_MRI.ipynb stores the model as `.json.xz`
    (lzma-compressed JSON dict). cobra.io.model_from_dict reconstructs the
    in-memory cobra.Model from that dict. We load it once at the top of the
    script and reuse the metabolite/reaction stoichiometry maps for every
    condition.
    """
    with lzma.open(path, "rt") as fh:
        model_dict = json.load(fh)
    return cobra.io.model_from_dict(model_dict)


def load_flux_matrix(csv_path: Path) -> pd.DataFrame:
    """
    Read the long-form flux CSV and pivot to a reaction_id x condition matrix.

    Returned DataFrame:
        index   = reaction_id (str)
        columns = condition labels (e.g. 'Gastrocnemius__norm__w9')
        values  = signed flux in mmol/gDW/h (cobra convention: positive flux
                  runs in the reaction's "forward" direction as defined by
                  rxn.metabolites coefficients).

    Reactions present in the model but missing from the CSV would normally be
    a bug; we leave them as NaN so they are obvious if they appear.
    """
    df = pd.read_csv(csv_path)
    pivot = df.pivot(index="reaction_id", columns="condition", values="flux")
    return pivot


# ---------------------------------------------------------------------------
# Core audit: per-metabolite mass-balance table
# ---------------------------------------------------------------------------

def metabolite_balance_table(
    model: cobra.Model,
    flux_matrix: pd.DataFrame,
    metabolite_id: str,
) -> pd.DataFrame:
    """
    Build a long-form mass-balance table for a single metabolite.

    For metabolite m and reaction r with stoichiometric coefficient s_{m,r}
    (positive = produced by r in forward direction, negative = consumed),
    the contribution of r to dm/dt under flux v_r is:

            contribution = s_{m,r} * v_r

    Steady state requires sum_r contribution = 0 for every internal
    metabolite. We enumerate every reaction touching m, look up its flux per
    condition from flux_matrix, multiply, and emit one row per
    (condition, reaction).

    Returned columns:
        condition, reaction_id, reaction_name, subsystem, stoich_coef,
        flux, contribution

    Sign convention reminder: for accoa_m, CSm has stoich_coef = -1
    (consumes accoa_m); a positive CSm flux therefore yields a negative
    contribution (consumption), which is what we want.
    """
    met = model.metabolites.get_by_id(metabolite_id)

    # Snapshot: (reaction_id, stoich_coef, name, subsystem). We pull these
    # before iterating across conditions so we don't repeatedly re-walk the
    # cobra object graph.
    rxn_records = [
        {
            "reaction_id": rxn.id,
            "stoich_coef": rxn.metabolites[met],
            "reaction_name": rxn.name,
            "subsystem": rxn.subsystem or "",
        }
        for rxn in met.reactions
    ]

    # Build the full long-form table: cross product of reactions x conditions.
    rows = []
    for cond in flux_matrix.columns:
        for rec in rxn_records:
            rid = rec["reaction_id"]
            # Reactions in the model but missing from the CSV produce NaN.
            # We propagate NaN so the closure check will surface the gap.
            flux = flux_matrix.at[rid, cond] if rid in flux_matrix.index else np.nan
            rows.append({
                "condition": cond,
                "reaction_id": rid,
                "reaction_name": rec["reaction_name"],
                "subsystem": rec["subsystem"],
                "stoich_coef": rec["stoich_coef"],
                "flux": flux,
                "contribution": rec["stoich_coef"] * flux,
            })

    out = pd.DataFrame(rows)

    # Drop numerical noise but keep the row even if flux is exactly zero AND
    # the reaction is biologically interesting (e.g. SUCD1m, PFK). We keep
    # zero-flux rows for an explicit allow-list of reactions of interest so
    # the audit_summary.md can confirm "this reaction was indeed at zero".
    keep_zero = {"SUCD1m", "PFK", "CSm", "PDHm", "AKGDm", "SUCOASm",
                 "ACCOACrm", "r0191", "FBA", "HEX1", "LDH_L"}
    mask = (out["contribution"].abs() > CONTRIBUTION_TOL) | (out["reaction_id"].isin(keep_zero))
    out = out[mask].copy()

    # Sort within each condition by absolute contribution descending so
    # producers and consumers with the largest impact appear at the top.
    out["abs_contribution"] = out["contribution"].abs()
    out = out.sort_values(
        by=["condition", "abs_contribution"],
        ascending=[True, False],
        kind="mergesort",  # stable, so reaction order is reproducible
    ).drop(columns=["abs_contribution"])

    return out.reset_index(drop=True)


def closure_per_condition(table: pd.DataFrame) -> pd.Series:
    """
    Sum of contributions per condition. At a feasible FBA optimum this should
    be ~0 within solver tolerance (CLOSURE_TOL). Significantly larger values
    indicate either (a) a reaction missing from the saved CSV, or (b) a real
    LP infeasibility / numerical leak in the solver, both of which we want
    to surface immediately.
    """
    return table.groupby("condition")["contribution"].sum()


# ---------------------------------------------------------------------------
# Markdown narrative
# ---------------------------------------------------------------------------

def render_summary(
    balance_tables: dict[str, pd.DataFrame],
    closures: dict[str, pd.Series],
    flux_matrix: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Write a human-readable markdown summary alongside the per-metabolite CSVs.

    Structure:
      1. Closure overview table (rows = metabolites, cols = conditions, values
         = sum_of_contributions). Should be all near-zero.
      2. Per-metabolite section. For each condition: top-N producers
         (positive contributions) and top-N consumers (negative
         contributions), with reaction name, flux, and contribution.
      3. Targeted callouts for the three originally flagged questions:
         - Where did unaccounted acetyl-CoA go? (top non-CSm consumer of accoa_m)
         - Who consumed succinate, given SUCD1m=0? (top non-SUCD1m consumer of succ_m)
         - Who made FBP when PFK=0? (top producer of fdp_c in PFK=0 conditions)
    """
    lines: list[str] = []
    lines.append("# Mass-balance audit of FBA solution")
    lines.append("")
    lines.append(
        "Generated by `audit_mass_balance.py`. Input fluxes from "
        "`results_fba_constrained_fit_B6/flux_distribution__hierarchical_atp.csv`. "
        "Numerical tolerance: contributions <"
        f" {CONTRIBUTION_TOL:g} dropped; closure threshold = {CLOSURE_TOL:g}."
    )
    lines.append("")

    # --- 1. Closure overview ---------------------------------------------
    lines.append("## 1. Closure (sum of contributions per metabolite, per condition)")
    lines.append("")
    lines.append(
        "At a feasible FBA optimum every internal metabolite satisfies "
        "`sum(stoich * flux) = 0`. Values far from zero indicate a problem "
        "with the saved solution (missing reactions, solver leak, or post-hoc "
        "edits to the CSV). Values within `CLOSURE_TOL` are healthy."
    )
    lines.append("")
    closure_df = pd.DataFrame(closures)  # rows = condition, cols = metabolite
    lines.append(closure_df.to_markdown(floatfmt=".2e"))
    lines.append("")
    bad = (closure_df.abs() > CLOSURE_TOL)
    if bad.any().any():
        lines.append("**WARNING:** the following (condition, metabolite) pairs exceed CLOSURE_TOL:")
        for cond in closure_df.index:
            for met in closure_df.columns:
                if bad.at[cond, met]:
                    lines.append(f"  - {cond} / {met}: {closure_df.at[cond, met]:.3e}")
        lines.append("")
    else:
        lines.append("All closures within tolerance — saved solution is internally consistent.")
        lines.append("")

    # --- 2. Per-metabolite top-N producers / consumers --------------------
    for met_id, table in balance_tables.items():
        lines.append(f"## 2.{list(balance_tables).index(met_id)+1} `{met_id}`")
        lines.append("")
        for cond, group in table.groupby("condition"):
            producers = group[group["contribution"] > 0].head(TOP_N)
            consumers = group[group["contribution"] < 0].head(TOP_N)
            lines.append(f"### {cond}")
            lines.append("")
            lines.append(f"**Top {TOP_N} producers** (sign of contribution > 0):")
            lines.append("")
            if producers.empty:
                lines.append("_(none above tolerance)_")
            else:
                lines.append(producers[
                    ["reaction_id", "reaction_name", "stoich_coef", "flux", "contribution"]
                ].to_markdown(index=False, floatfmt=".4g"))
            lines.append("")
            lines.append(f"**Top {TOP_N} consumers** (sign of contribution < 0):")
            lines.append("")
            if consumers.empty:
                lines.append("_(none above tolerance)_")
            else:
                lines.append(consumers[
                    ["reaction_id", "reaction_name", "stoich_coef", "flux", "contribution"]
                ].to_markdown(index=False, floatfmt=".4g"))
            lines.append("")

    # --- 3. Targeted callouts --------------------------------------------
    lines.append("## 3. Answers to the originally flagged questions")
    lines.append("")

    # 3a. Acetyl-CoA: largest non-CSm consumer per condition
    lines.append("### 3a. Where did the unaccounted acetyl-CoA go?")
    lines.append("")
    lines.append(
        "Top non-`CSm` consumer of `accoa_m` per condition (the reaction "
        "absorbing acetyl-CoA that *would* in a closed TCA cycle have gone "
        "to citrate synthase):"
    )
    lines.append("")
    accoa_m = balance_tables["accoa_m"]
    callout_rows = []
    for cond, group in accoa_m.groupby("condition"):
        non_cs = group[(group["contribution"] < 0) & (group["reaction_id"] != "CSm")]
        if non_cs.empty:
            callout_rows.append({"condition": cond, "reaction_id": "(none)",
                                 "reaction_name": "", "flux": np.nan, "contribution": 0.0})
        else:
            top = non_cs.iloc[0]
            callout_rows.append({
                "condition": cond,
                "reaction_id": top["reaction_id"],
                "reaction_name": top["reaction_name"],
                "flux": top["flux"],
                "contribution": top["contribution"],
            })
    lines.append(pd.DataFrame(callout_rows).to_markdown(index=False, floatfmt=".4g"))
    lines.append("")

    # 3b. Succinate: largest non-SUCD1m consumer
    lines.append("### 3b. Where did succinate go, given `SUCD1m = 0`?")
    lines.append("")
    lines.append(
        "Top non-`SUCD1m` consumer of `succ_m` per condition. Note: `AKGDm` "
        "does not directly produce `succ_m` — it produces `succoa_m`, which "
        "is then converted by `SUCOASm`. See the `succoa_m` table for that "
        "step."
    )
    lines.append("")
    succ = balance_tables["succ_m"]
    callout_rows = []
    for cond, group in succ.groupby("condition"):
        non_sd = group[(group["contribution"] < 0) & (group["reaction_id"] != "SUCD1m")]
        if non_sd.empty:
            callout_rows.append({"condition": cond, "reaction_id": "(none)",
                                 "reaction_name": "", "flux": np.nan, "contribution": 0.0})
        else:
            top = non_sd.iloc[0]
            callout_rows.append({
                "condition": cond,
                "reaction_id": top["reaction_id"],
                "reaction_name": top["reaction_name"],
                "flux": top["flux"],
                "contribution": top["contribution"],
            })
    lines.append(pd.DataFrame(callout_rows).to_markdown(index=False, floatfmt=".4g"))
    lines.append("")

    # 3c. FBP / PFK: top producer per condition, with PFK and r0191 fluxes side by side.
    lines.append("### 3c. Who made FBP (`fdp_c`) when `PFK = 0`?")
    lines.append("")
    lines.append(
        "Per condition: flux of `PFK` (canonical ATP-dependent), `r0191` "
        "(UTP-dependent F6P->FBP, a CORDA-retained alternative), and the "
        "single top producer of `fdp_c`."
    )
    lines.append("")
    fdp = balance_tables["fdp_c"]
    callout_rows = []
    for cond in flux_matrix.columns:
        pfk_flux = flux_matrix.at["PFK", cond] if "PFK" in flux_matrix.index else np.nan
        r0191_flux = flux_matrix.at["r0191", cond] if "r0191" in flux_matrix.index else np.nan
        producers = fdp[(fdp["condition"] == cond) & (fdp["contribution"] > 0)]
        if producers.empty:
            top_id, top_contrib = "(none)", 0.0
        else:
            top = producers.iloc[0]
            top_id, top_contrib = top["reaction_id"], top["contribution"]
        callout_rows.append({
            "condition": cond,
            "PFK_flux": pfk_flux,
            "r0191_flux": r0191_flux,
            "top_producer": top_id,
            "top_producer_contribution": top_contrib,
        })
    lines.append(pd.DataFrame(callout_rows).to_markdown(index=False, floatfmt=".4g"))
    lines.append("")

    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[audit] loading model from {MODEL_PATH.relative_to(HERE.parent)}")
    model = load_model(MODEL_PATH)
    print(f"[audit]   model: {len(model.reactions)} reactions, "
          f"{len(model.metabolites)} metabolites")

    print(f"[audit] loading fluxes from {FLUX_CSV.relative_to(HERE.parent)}")
    flux_matrix = load_flux_matrix(FLUX_CSV)
    print(f"[audit]   flux matrix: {flux_matrix.shape[0]} reactions x "
          f"{flux_matrix.shape[1]} conditions")
    print(f"[audit]   conditions: {list(flux_matrix.columns)}")

    balance_tables: dict[str, pd.DataFrame] = {}
    closures: dict[str, pd.Series] = {}

    for met_id in TARGET_METABOLITES:
        if met_id not in [m.id for m in model.metabolites]:
            print(f"[audit] WARNING: metabolite {met_id!r} not in model; skipping")
            continue
        print(f"[audit] auditing metabolite {met_id}")
        table = metabolite_balance_table(model, flux_matrix, met_id)
        closure = closure_per_condition(table)
        out_csv = AUDIT_DIR / f"mass_balance_{met_id}.csv"
        table.to_csv(out_csv, index=False)
        print(f"[audit]   wrote {out_csv.relative_to(HERE.parent)}  "
              f"({len(table)} rows)")
        print(f"[audit]   closure (max abs): {closure.abs().max():.3e}")
        balance_tables[met_id] = table
        closures[met_id] = closure

    summary_path = AUDIT_DIR / "audit_summary.md"
    render_summary(balance_tables, closures, flux_matrix, summary_path)
    print(f"[audit] wrote {summary_path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
