#!/usr/bin/env python3
"""Generate Supplementary Table S3: week-9 DMI-derived flux summaries.

The table is descriptive: it summarizes mouse-level tissue estimates used by
the downstream DMI-constrained FBA/GEM workflow, without formal hypothesis
testing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]

# This is the per-replicate target table written by
# metabolic_network/v6b_met1_modelFitC6/run_fba_fva_flux_sampling_per_replicate.py
# from the met1/modelFitC6 C6 fitting output. It is the exact mouse-tissue table
# consumed downstream by the week-9 C6-aware FBA sampling run.
INPUT_PATH = (
    PROJECT_ROOT
    / "metabolic_network"
    / "v6b_met1_modelFitC6"
    / "results_fba_constrained_met1_modelFitC6_C6_sampling_per_replicate"
    / "condition_summary_used__met1_modelFitC6_c6_aware_fva_sampling_atp_per_replicate_n50.csv.gz"
)

OUT_CSV = HERE / "supplementary_table_week9_dmi_flux_summary.csv"
OUT_MD = HERE / "supplementary_table_week9_dmi_flux_summary.md"
OUT_XLSX = HERE / "supplementary_table_week9_dmi_flux_summary.xlsx"

CONTROL_MICE = {49, 50, 51, 52}
HFD_MICE = {45, 47, 48, 53, 54, 55, 56}
MOUSE_TO_GROUP = {mouse: "norm" for mouse in CONTROL_MICE} | {mouse: "dia" for mouse in HFD_MICE}

TISSUE_LABELS = {
    "gas": "Gastrocnemius",
    "Gastrocnemius": "Gastrocnemius",
    "sol": "Soleus",
    "Soleus": "Soleus",
    "tib": "Tibialis anterior",
    "TibialisAnt": "Tibialis anterior",
    "Tibialis anterior": "Tibialis anterior",
}
MUSCLE_TISSUES = ["Gastrocnemius", "Soleus", "Tibialis anterior"]
STATE_LABELS = {"norm": "Control", "dia": "HFD"}

BASE_FLUXES = [
    "MR_glc",
    "V_lac_loss",
    "V_ox",
    "phi_tca",
    "V_TCA_total",
    "V_dil_tca",
]
DERIVED_FLUXES = [
    "V_ox_over_MR_glc",
    "V_TCA_total_over_MR_glc",
]
ALL_FLUXES = BASE_FLUXES + DERIVED_FLUXES


def require_columns(df: pd.DataFrame, columns: list[str], source: Path) -> None:
    """Fail loudly if required columns are unavailable."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def canonical_mouse_id(series: pd.Series) -> pd.Series:
    """Convert mouse IDs such as 'mouse56' or 56.0 to integer mouse numbers."""
    extracted = series.astype(str).str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")


def add_flux_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Standardize mean-valued flux columns and derive requested ratios."""
    out = df.copy()
    mapping: dict[str, str] = {}
    missing: list[str] = []

    for flux in BASE_FLUXES:
        candidates = [f"{flux}_mean", flux]
        found = next((col for col in candidates if col in out.columns), None)
        if found is None:
            missing.append(flux)
            continue
        out[flux] = pd.to_numeric(out[found], errors="coerce")
        mapping[flux] = found

    required_for_ratios = {"MR_glc", "V_ox", "V_TCA_total"}
    if required_for_ratios.difference(out.columns):
        missing.extend([flux for flux in DERIVED_FLUXES if flux not in out.columns])
    else:
        out["V_ox_over_MR_glc"] = out["V_ox"] / out["MR_glc"].replace(0, np.nan)
        out["V_TCA_total_over_MR_glc"] = out["V_TCA_total"] / out["MR_glc"].replace(0, np.nan)

    created = [flux for flux in DERIVED_FLUXES if flux in out.columns]
    return out, missing, created


def summarize_fluxes(df: pd.DataFrame) -> pd.DataFrame:
    """Return one descriptive row per tissue, state, and flux."""
    long_df = df.melt(
        id_vars=["mouseID_int", "tissue_readable", "state"],
        value_vars=[flux for flux in ALL_FLUXES if flux in df.columns],
        var_name="Flux",
        value_name="value",
    ).dropna(subset=["value"])

    summary = (
        long_df.groupby(["tissue_readable", "state", "Flux"], as_index=False)
        .agg(
            **{
                "N mouse tissue": ("mouseID_int", "nunique"),
                "Mean": ("value", "mean"),
                "Between Mouse SD": ("value", lambda s: float(pd.Series(s).std(ddof=1))),
                "Median": ("value", "median"),
            }
        )
    )

    wide = summary.pivot_table(
        index=["tissue_readable", "Flux"],
        columns="state",
        values="Mean",
        aggfunc="first",
    )
    delta = (wide.get("HFD") - wide.get("Control")).rename("HFD - Control")
    pct = (100.0 * delta / wide.get("Control")).rename("HFD - Control [%]")

    summary = summary.merge(
        pd.concat([delta, pct], axis=1).reset_index(),
        on=["tissue_readable", "Flux"],
        how="left",
    )
    summary["Tissue"] = pd.Categorical(summary["tissue_readable"], MUSCLE_TISSUES, ordered=True)
    summary["State"] = pd.Categorical(summary["state"], ["Control", "HFD"], ordered=True)
    summary["Flux"] = pd.Categorical(summary["Flux"], ALL_FLUXES, ordered=True)
    summary = summary.sort_values(["Tissue", "Flux", "State"])

    return summary[
        [
            "Tissue",
            "State",
            "Flux",
            "N mouse tissue",
            "Mean",
            "Between Mouse SD",
            "Median",
            "HFD - Control",
            "HFD - Control [%]",
        ]
    ]


def write_markdown_table(table: pd.DataFrame, path: Path) -> None:
    caption = (
        "Table S3. Week-9 DMI-derived flux estimates used for FBA constraints. "
        "Mouse-level fitted flux estimates were summarized by muscle and diet "
        "state for the week-9 conditions used in the DMI-constrained FBA analysis. "
        "Values report the mean, between-mouse SD, and median across mouse-tissue "
        "fits. HFD minus Control differences are shown as absolute and percent "
        "changes. The table is descriptive and does not represent formal "
        "statistical testing."
    )
    path.write_text(caption + "\n\n" + table.to_markdown(index=False) + "\n")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input table not found: {INPUT_PATH}")

    raw = pd.read_csv(INPUT_PATH)
    n_before = len(raw)
    require_columns(raw, ["mouseID", "week", "tissue"], INPUT_PATH)

    df = raw.copy()
    df["mouseID_int"] = canonical_mouse_id(df["mouseID"])
    if df["mouseID_int"].isna().any():
        bad = df.loc[df["mouseID_int"].isna(), "mouseID"].unique().tolist()
        raise ValueError(f"Could not parse mouseID values: {bad}")

    # Hard override group labels according to the manuscript rules. This ensures
    # mice 54 and 56 are HFD/dia even if an upstream metadata file says otherwise.
    unmapped = sorted(set(df["mouseID_int"].astype(int)) - set(MOUSE_TO_GROUP))
    if unmapped:
        raise ValueError(f"Mouse IDs are not in the required group map: {unmapped}")
    df["group_corrected"] = df["mouseID_int"].astype(int).map(MOUSE_TO_GROUP)
    df["state"] = df["group_corrected"].map(STATE_LABELS)

    df["tissue_readable"] = df["tissue"].map(TISSUE_LABELS).fillna(df["tissue"].astype(str))
    df = df[
        (df["week"].astype(str).str.lower().isin({"w9", "9", "week9", "week_9"}))
        & (df["tissue_readable"].isin(MUSCLE_TISSUES))
    ].copy()
    n_after_filter = len(df)

    df, missing_fluxes, created_ratios = add_flux_columns(df)
    available_fluxes = [flux for flux in ALL_FLUXES if flux in df.columns]
    if not available_fluxes:
        raise ValueError("No requested flux columns were available after standardization.")
    required_missing = [flux for flux in BASE_FLUXES if flux in missing_fluxes]
    if required_missing:
        raise ValueError(f"Missing required fitted flux columns: {required_missing}")

    table = summarize_fluxes(df)
    numeric_cols = ["Mean", "Between Mouse SD", "Median", "HFD - Control", "HFD - Control [%]"]
    table[numeric_cols] = table[numeric_cols].round(3)

    table.to_csv(OUT_CSV, index=False)
    write_markdown_table(table, OUT_MD)
    try:
        table.to_excel(OUT_XLSX, index=False)
        xlsx_status = str(OUT_XLSX)
    except ImportError:
        xlsx_status = "not written; pandas Excel dependency is unavailable"

    expected_by_state = {
        "Control": sorted(CONTROL_MICE),
        "HFD": sorted(HFD_MICE),
    }
    observed_by_state = {
        state: sorted(df.loc[df["state"] == state, "mouseID_int"].astype(int).unique().tolist())
        for state in ["Control", "HFD"]
    }
    missing_expected = {
        state: sorted(set(expected_by_state[state]) - set(observed_by_state[state]))
        for state in ["Control", "HFD"]
    }
    counts = (
        df.groupby(["tissue_readable", "state"])["mouseID_int"]
        .nunique()
        .rename("n_unique_mice")
        .reset_index()
        .sort_values(["tissue_readable", "state"])
    )

    print("Supplementary Table S3 audit")
    print(f"Input file used: {INPUT_PATH}")
    print(f"Rows before filtering: {n_before}")
    print(f"Rows after filtering to week-9 muscle data: {n_after_filter}")
    print(f"Unique mice by state: {observed_by_state}")
    print(f"Expected week-9 mice absent from input by state: {missing_expected}")
    print(f"Unique tissues retained: {sorted(df['tissue_readable'].unique().tolist())}")
    print("Mouse-level tissue summaries per tissue/state:")
    print(counts.to_string(index=False))
    print(f"Missing flux columns: {sorted(set(missing_fluxes))}")
    print(f"Derived ratio columns created: {created_ratios}")
    print("Column standardization used:")
    for flux in BASE_FLUXES:
        source_col = f"{flux}_mean" if f"{flux}_mean" in raw.columns else flux if flux in raw.columns else "MISSING"
        print(f"  {flux} <- {source_col}")
    print(f"Wrote CSV: {OUT_CSV}")
    print(f"Wrote Markdown: {OUT_MD}")
    print(f"Wrote Excel: {xlsx_status}")


if __name__ == "__main__":
    main()
