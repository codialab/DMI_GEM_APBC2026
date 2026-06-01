"""
Build reaction lists for every native subsystem annotation in the
muscle-specific metabolic model.

Why this script exists
----------------------
`build_subsystem_reaction_list.py` creates a curated reaction map for a small
set of MRI-focused subsystems. For downstream flux-sampling analyses we may
also want to project sampled reaction fluxes onto *all* model-native subsystem
annotations. This script extracts those native annotations directly from the
COBRA JSON model and writes a reusable reaction-to-subsystem table.

Output
------
results_native_subsystem_reactions/
    all_native_subsystems_reactions.csv.xz
        Long-form table with one row per reaction and its native subsystem.

    native_subsystem_summary.csv
        One row per native subsystem with reaction counts and gene coverage.

    per_subsystem/
        <safe_subsystem_name>.csv.xz
            Reaction table for one native subsystem.

Notes
-----
The model-native subsystem names are not curated for this MRI/transcriptomics
question. Expect broad technical classes such as transport, exchange/demand,
and drug metabolism. Those can be filtered later before making paper figures.
"""

from __future__ import annotations

import argparse
import json
import lzma
import re
from pathlib import Path
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = HERE / "muscle_specific_model.json.xz"
DEFAULT_OUT_DIR = HERE / "results_native_subsystem_reactions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all native model subsystem-to-reaction mappings."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"COBRA JSON model path, optionally .xz compressed. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--min-reactions",
        type=int,
        default=1,
        help="Only write per-subsystem files for subsystems with at least this many reactions.",
    )
    return parser.parse_args()


def load_json_model(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    open_fn = lzma.open if path.suffix == ".xz" else open
    with open_fn(path, "rt") as handle:
        model = json.load(handle)

    if "reactions" not in model:
        raise ValueError(f"{path} does not look like a COBRA JSON model: missing 'reactions'.")
    return model


def safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return safe or "unassigned"


def normalize_subsystem(value: Any) -> str:
    if value is None:
        return "unassigned"
    text = str(value).strip()
    return text if text else "unassigned"


def reaction_string(reaction: dict[str, Any]) -> str:
    metabolites = reaction.get("metabolites", {})
    lower_bound = float(reaction.get("lower_bound", 0.0))

    left = []
    right = []
    for met_id, coefficient in sorted(metabolites.items()):
        coefficient = float(coefficient)
        if coefficient == 0:
            continue
        term = met_id if abs(coefficient) == 1 else f"{abs(coefficient):g} {met_id}"
        if coefficient < 0:
            left.append(term)
        else:
            right.append(term)

    arrow = "<=>" if lower_bound < 0 else "-->"
    left_text = " + ".join(left) if left else "0"
    right_text = " + ".join(right) if right else "0"
    return f"{left_text} {arrow} {right_text}"


def annotation_value(annotation: dict[str, Any], key: str) -> str:
    value = annotation.get(key, "")
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return str(value) if value is not None else ""


def build_reaction_table(model: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for reaction in model["reactions"]:
        annotation = reaction.get("annotation") or {}
        gene_rule = reaction.get("gene_reaction_rule") or ""
        rows.append(
            {
                "subsystem": normalize_subsystem(reaction.get("subsystem")),
                "rxn_id": reaction.get("id", ""),
                "rxn_name": reaction.get("name", ""),
                "reaction": reaction_string(reaction),
                "lb": reaction.get("lower_bound", ""),
                "ub": reaction.get("upper_bound", ""),
                "gene_reaction_rule": gene_rule,
                "has_gene_rule": bool(str(gene_rule).strip()),
                "bigg_reaction": annotation_value(annotation, "bigg.reaction"),
                "kegg_reaction": annotation_value(annotation, "kegg.reaction"),
                "reactome": annotation_value(annotation, "reactome"),
                "metanetx_reaction": annotation_value(annotation, "metanetx.reaction"),
                "ec_code": annotation_value(annotation, "ec-code"),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["subsystem", "rxn_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_summary_table(reaction_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        reaction_df.groupby("subsystem", as_index=False)
        .agg(
            reaction_count=("rxn_id", "nunique"),
            reactions_with_gene_rule=("has_gene_rule", "sum"),
        )
        .sort_values(["reaction_count", "subsystem"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary["fraction_with_gene_rule"] = (
        summary["reactions_with_gene_rule"] / summary["reaction_count"]
    )
    return summary


def write_outputs(reaction_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path, min_reactions: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_subsystem_dir = out_dir / "per_subsystem"
    per_subsystem_dir.mkdir(parents=True, exist_ok=True)

    all_path = out_dir / "all_native_subsystems_reactions.csv.xz"
    summary_path = out_dir / "native_subsystem_summary.csv"

    reaction_df.to_csv(all_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    for row in summary_df.itertuples(index=False):
        if row.reaction_count < min_reactions:
            continue
        subsystem_df = reaction_df[reaction_df["subsystem"] == row.subsystem]
        subsystem_path = per_subsystem_dir / f"{safe_name(row.subsystem)}.csv.xz"
        subsystem_df.to_csv(subsystem_path, index=False)

    print(f"Native subsystems: {len(summary_df)}")
    print(f"Reactions written: {len(reaction_df)}")
    print(f"Wrote long table: {all_path}")
    print(f"Wrote summary:    {summary_path}")
    print(f"Wrote per-subsystem files to: {per_subsystem_dir}")


def main() -> None:
    args = parse_args()
    model = load_json_model(args.model)
    reaction_df = build_reaction_table(model)
    summary_df = build_summary_table(reaction_df)
    write_outputs(reaction_df, summary_df, args.out_dir, args.min_reactions)


if __name__ == "__main__":
    main()
