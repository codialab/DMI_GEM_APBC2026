#!/usr/bin/env python3
"""
load_datasets.py

Loads normalized expression matrices and sample annotations for all datasets.

Datasets
--------
GSE17576  : Affymetrix microarray  – HFD obesity study
GSE103726 : Affymetrix microarray  – blunt muscle injury study
GSE159558 : RNA-seq                – USP21 muscle KO study
GSE182686 : RNA-seq                – FGF6 muscle delivery study
GSE161693 : RNA-seq                – Notch2 muscle KO study
GSE241830 : Affymetrix Clariom S   – TAS1R2 ribosomal pulldown study
GSE275818 : RNA-seq                – Running distance + HFD study
GSE305719 : RNA-seq                – Multi-organ HFD time course (DNB analysis)

Use in IPython / Jupyter
------------------------
    %run load_datasets.py
    # variables 'datasets', 'expr', 'sample_ann' are now in your namespace

    # Access a specific dataset:
    expr       = datasets["GSE161693"]["expr"]
    sample_ann = datasets["GSE161693"]["sample_ann"]

    # Reload after re-running prepare_transcriptome_datasets.py:
    %run load_datasets.py

Import into another script
--------------------------
    from load_datasets import load_dataset, load_all_datasets
    datasets = load_all_datasets()
"""

import os
import pandas as pd

# __file__ is not defined in interactive IPython/Jupyter sessions
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DATASETS = ["GSE17576", "GSE103726", "GSE159558", "GSE182686",
            "GSE161693", "GSE241830", "GSE275818", "GSE305719"]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_dataset(gse_id):
    """
    Load one dataset's normalized expression and sample annotations.

    Parameters
    ----------
    gse_id : str  e.g. "GSE17576"

    Returns
    -------
    expr       : DataFrame  genes/probes × samples
    sample_ann : DataFrame  samples × metadata columns
    """
    folder    = os.path.join(BASE_DIR, gse_id)
    expr_path = os.path.join(folder, "normalized_expression.csv.gz")
    ann_path  = os.path.join(folder, "sample_annotation.csv.gz")

    if not os.path.exists(expr_path):
        raise FileNotFoundError(
            f"{gse_id}: normalized_expression.csv.gz not found — "
            "run prepare_transcriptome_datasets.py first."
        )
    if not os.path.exists(ann_path):
        raise FileNotFoundError(
            f"{gse_id}: sample_annotation.csv.gz not found — "
            "run prepare_transcriptome_datasets.py first."
        )

    expr       = pd.read_csv(expr_path, index_col=0, compression="gzip")
    sample_ann = pd.read_csv(ann_path,  index_col=0, compression="gzip")
    return expr, sample_ann


def load_all_datasets(verbose=True):
    """
    Load all datasets.

    Returns
    -------
    dict  { gse_id: {"expr": DataFrame, "sample_ann": DataFrame} }
    """
    result = {}
    for gse_id in DATASETS:
        try:
            expr, sample_ann = load_dataset(gse_id)
            result[gse_id] = {"expr": expr, "sample_ann": sample_ann}
            if verbose:
                cond = (
                    sample_ann["condition"].value_counts().to_dict()
                    if "condition" in sample_ann.columns
                    else "—"
                )
                print(
                    f"  {gse_id}  {expr.shape[0]:>6,} genes × {expr.shape[1]:>2} samples"
                    f"   conditions: {cond}"
                )
        except FileNotFoundError as e:
            print(f"  SKIPPED  {e}")

    return result


# ── Run on %run / direct execution ────────────────────────────────────────────

print("Loading datasets ...\n")
datasets = load_all_datasets()

for key, hash in datasets.items():
    print(f"{key}: expr {hash['expr'].shape}, sample_ann {hash['sample_ann'].shape}")

# # Expose the most-used variables directly in the namespace for convenience
# expr       = None
# sample_ann = None
# if datasets:
#     _first = next(iter(datasets))
#     expr       = datasets[_first]["expr"]
#     sample_ann = datasets[_first]["sample_ann"]

# print(f"\ndatasets       dict with keys: {list(datasets.keys())}")
# print(f"expr           expression matrix of '{_first}' ({expr.shape})")
# print(f"sample_ann     annotations of '{_first}'  ({sample_ann.shape})")
# print("\nTo access another dataset:")
# print("  expr       = datasets['GSE161693']['expr']")
# print("  sample_ann = datasets['GSE161693']['sample_ann']")
