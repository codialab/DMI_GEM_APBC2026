#!/usr/bin/env python3
"""
prepare_transcriptome_datasets.py

Reads, normalizes, and annotates selected skeletal muscle transcriptome datasets
for downstream gene / pathway enrichment analysis.

Datasets
--------
GSE17576  : Affymetrix microarray  – HFD obesity study
            De Wilde et al. (2009) J Nutrigenet Nutrigenomics
GSE305719 : RNA-seq (Illumina)              – Multi-organ HFD time course
            Oku et al. (2025) NOLTA – Dynamical network biomarker analysis

Normalization strategy
----------------------
Microarray  : authors' GC-RMA (GSE17576) values are read
              from the series matrix file (already normalized + quantile-
              normalized by authors). Values are in LINEAR scale → we apply
              log2(x + 1) to put them on a log scale.

RNA-seq     : raw integer counts → variance-stabilizing transformation (VST)
              via PyDESeq2. VST produces log2-scale, library-size-normalised
              values with stabilised variance across the expression range.
              Low-count genes are filtered before VST.

Outputs (written to each dataset's folder)
------------------------------------------
  normalized_expression.csv.xz  – genes × samples, normalized values
  sample_annotation.csv.xz      – one row per sample with condition labels

Usage
-----
    python prepare_transcriptome_datasets.py

Requirements
------------
    pip install pydeseq2 pandas numpy mygene
"""

import os
import re
import sys
import json
import logging
import warnings
import urllib.request

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Dataset registry ──────────────────────────────────────────────────────────

DATASETS = {
    "GSE17576": {
        "type": "microarray",
        "series_matrix": "GSE17576_series_matrix.txt.gz",
        "publication": (
            "An 8-week high-fat diet induces obesity and insulin resistance "
            "with small changes in the muscle transcriptome of C57BL/6J mice"
        ),
    },
    "GSE305719": {
        "type": "rnaseq",
        "counts_file": "GSE305719_data_count.csv.gz",
        "publication": (
            "Dynamical network biomarker analysis using multi-organ RNA "
            "sequencing in metabolic syndrome model mice"
        ),
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# Microarray helpers
# ═════════════════════════════════════════════════════════════════════════════

def parse_geo_series_matrix(matrix_path, log2_transform=True):
    """
    Parse a GEO series matrix file directly without GEOparse.

    GEOparse does not populate GSM expression tables when reading series
    matrix files (it only fills metadata), so we parse the file ourselves:
      - Expression data lives between !series_matrix_table_begin / _end markers
      - Sample metadata is in the !Sample_* header lines

    Parameters
    ----------
    matrix_path   : str   path to the series matrix file (.gz or plain)
    log2_transform: bool  apply log2(x + 1) to the expression values.
                          Set to False when the matrix already contains
                          log2-scale values (e.g. SST-RMA from TAC 4.0).

    Returns
    -------
    expr        : DataFrame, probes × samples
    sample_ann  : DataFrame, samples × metadata fields
    """
    import gzip
    from io import StringIO

    log.info(f"    Parsing {os.path.basename(matrix_path)} ...")

    opener = gzip.open if matrix_path.endswith(".gz") else open
    meta_lines = []
    data_lines = []
    in_data    = False

    with opener(matrix_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if "!series_matrix_table_begin" in line:
                in_data = True
                continue
            elif "!series_matrix_table_end" in line:
                in_data = False
                continue

            if in_data:
                data_lines.append(line)
            elif line.startswith("!Sample_"):
                meta_lines.append(line)

    # ── Expression matrix ─────────────────────────────────────────────────
    expr = pd.read_csv(
        StringIO("\n".join(data_lines)),
        sep="\t",
        index_col=0,
    ).astype(float)
    expr.columns = [c.strip('"') for c in expr.columns]
    expr.index   = expr.index.astype(str).str.strip('"')
    expr.index.name = "probe_id"

    if log2_transform:
        expr = np.log2(expr + 1)

    # ── Sample metadata ────────────────────────────────────────────────────
    meta: dict[str, list] = {}
    for line in meta_lines:
        parts = line.split("\t")
        key   = parts[0][1:]                          # strip leading '!'
        vals  = [v.strip('"') for v in parts[1:]]
        meta.setdefault(key, []).append(vals)

    sample_ids = meta.get("Sample_geo_accession", [[]])[0]
    n          = len(sample_ids)

    records = []
    for i in range(n):
        row = {"sample_id": sample_ids[i]}

        for field in ("Sample_title", "Sample_source_name_ch1"):
            rows = meta.get(field, [[]])
            if rows and i < len(rows[0]):
                col = field.replace("Sample_", "").lower()
                row[col] = rows[0][i]

        for char_row in meta.get("Sample_characteristics_ch1", []):
            if i < len(char_row) and ":" in char_row[i]:
                k, v = char_row[i].split(":", 1)
                col  = k.strip().lower().replace(" ", "_").replace("/", "_")
                row[col] = v.strip()

        records.append(row)

    sample_ann = pd.DataFrame(records).set_index("sample_id")
    return expr, sample_ann


def annotate_gse17576(sample_ann):
    """
    Add a 'condition' column to GSE17576 sample annotations.

    Conditions (from !Sample_characteristics_ch1):
      protocol : 'low fat diet' | 'high fat diet'
      time     : 'day 0'        | 'day 56'

    Combined condition labels:
      LFD_day0, LFD_day56, HFD_day56
    """
    def make_condition(row):
        protocol = row.get("protocol", "").lower()
        time     = row.get("time", "").lower().replace(" ", "")
        diet = "LFD" if "low" in protocol else "HFD"
        return f"{diet}_{time}"

    sample_ann = sample_ann.copy()
    sample_ann["condition"] = sample_ann.apply(make_condition, axis=1)
    sample_ann["replicate"]  = (
        sample_ann.groupby("condition").cumcount() + 1
    ).astype(str)
    return sample_ann


def annotate_gse103726(sample_ann):
    """
    Add condition columns to GSE103726 sample annotations.

    The only GEO characteristic is 'tissue', so conditions are decoded from
    the sample title which follows the pattern:
        {timepoint}_{injury}_{diet}_{mouse_id}
    e.g. '1h_C_10_7416'

    Fields
    ------
    timepoint : 1h, 6h, 24h, 3d, 8d
    injury    : C (sham/control) | T (injured)
    diet      : 10 (normal weight) | 60 (obese)
    condition : {injury}_{diet}  e.g. 'sham_normal', 'injured_obese'
    """
    injury_map = {"C": "sham",    "T": "injured"}
    diet_map   = {"10": "normal", "60": "obese"}

    sample_ann = sample_ann.copy()
    parsed = sample_ann["title"].str.extract(
        r"^(?P<timepoint>[^_]+)_(?P<injury>[CT])_(?P<diet>10|60)_(?P<mouse_id>\d+)$"
    )
    sample_ann["timepoint"] = parsed["timepoint"]
    sample_ann["injury"]    = parsed["injury"].map(injury_map)
    sample_ann["diet"]      = parsed["diet"].map(diet_map)
    sample_ann["mouse_id"]  = parsed["mouse_id"]
    sample_ann["condition"] = sample_ann["injury"] + "_" + sample_ann["diet"]
    return sample_ann


def annotate_gse241830(sample_ann):
    """
    Add a 'condition' column to GSE241830 sample annotations.

    All 5 samples are sedentary soleus myocyte ribosomal pulldowns from
    Myogenin-Cre × RiboTag mice — a single-condition characterization study.
    """
    sample_ann = sample_ann.copy()
    sample_ann["condition"] = "sedentary_soleus"
    sample_ann["replicate"] = (
        sample_ann.groupby("condition").cumcount() + 1
    ).astype(str)
    return sample_ann


# ═════════════════════════════════════════════════════════════════════════════
# RNA-seq helpers
# ═════════════════════════════════════════════════════════════════════════════

def filter_low_counts(counts_df, min_count=10, min_samples=2):
    """
    Remove genes that have fewer than `min_count` reads in at least
    `min_samples` samples. This prevents VST instability on near-zero genes.
    """
    keep = (counts_df >= min_count).sum(axis=1) >= min_samples
    log.info(
        f"    Low-count filter: kept {keep.sum()} / {len(keep)} genes "
        f"(removed {(~keep).sum()})"
    )
    return counts_df[keep]


def vst_normalize(counts_df, sample_ann, design_factor="condition"):
    """
    Apply variance-stabilizing transformation (VST) via PyDESeq2.

    PyDESeq2 mirrors the DESeq2 R package and estimates per-gene dispersion
    from the count data, then fits a dispersion-mean trend used to compute
    the VST. The output is on a log2 scale, library-size normalised, with
    variance stabilised across the expression range.

    Parameters
    ----------
    counts_df     : DataFrame, genes × samples, raw integer counts
    sample_ann    : DataFrame, samples × metadata (must contain design_factor)
    design_factor : column in sample_ann to use as the design variable

    Returns
    -------
    DataFrame, genes × samples, VST-normalised log2 values
    """
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.default_inference import DefaultInference
    except ImportError:
        raise ImportError("pydeseq2 is required. Install with: pip install pydeseq2")

    # Round before casting: some tools (e.g. StrandNGS) emit fractional counts
    counts_T = counts_df.T.round().astype(int)
    metadata = sample_ann[[design_factor]].loc[counts_T.index]

    log.info(
        f"    PyDESeq2 VST: {counts_T.shape[0]} samples × {counts_T.shape[1]} genes ..."
    )

    dds = DeseqDataSet(
        counts=counts_T,
        metadata=metadata,
        design_factors=design_factor,
        refit_cooks=True,
        inference=DefaultInference(),
    )

    dds.fit_size_factors()
    dds.fit_genewise_dispersions()
    dds.fit_dispersion_trend()
    dds.fit_MAP_dispersions()
    dds.vst(use_design=False)

    vst = pd.DataFrame(
        dds.layers["vst_counts"],
        index=counts_T.index,
        columns=counts_T.columns,
    ).T
    vst.index.name = "gene"
    return vst


# ═════════════════════════════════════════════════════════════════════════════
# MGI gene symbol mapping
# ═════════════════════════════════════════════════════════════════════════════

def _load_json_cache(cache_path):
    """Load a JSON mapping cache from disk; return empty dict if absent."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return {}


def _save_json_cache(cache_path, data):
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)


def _mygene_querymany(ids, scopes, folder, cache_filename):
    """
    Query mygene.info for official mouse gene symbols, with local JSON caching.
    Results are stored in <folder>/<cache_filename> so subsequent runs skip
    the network call.

    Parameters
    ----------
    ids            : list of query strings (Entrez IDs or gene symbols)
    scopes         : mygene query scope, e.g. 'entrezgene' or 'symbol,alias'
    folder         : dataset folder path (for cache location)
    cache_filename : JSON filename for the cache

    Returns
    -------
    dict  { query_id (str) : mgi_symbol_or_None }
    """
    try:
        import mygene
    except ImportError:
        raise ImportError("mygene is required. Install with: pip install mygene")

    cache_path = os.path.join(folder, cache_filename)
    cache = _load_json_cache(cache_path)

    uncached = [str(i) for i in ids if str(i) not in cache]

    if uncached:
        log.info(
            f"    mygene: querying {len(uncached):,} identifiers "
            f"(scope='{scopes}') ..."
        )
        mg = mygene.MyGeneInfo()
        for start in range(0, len(uncached), 1000):
            batch = uncached[start : start + 1000]
            results = mg.querymany(
                batch,
                scopes=scopes,
                fields="symbol",
                species="mouse",
                verbose=False,
            )
            for r in results:
                qid    = str(r.get("query", ""))
                symbol = (
                    r["symbol"]
                    if not r.get("notfound", False) and "symbol" in r
                    else None
                )
                cache[qid] = symbol
        _save_json_cache(cache_path, cache)
        log.info(f"    mygene: cache saved → {cache_filename}")

    return cache


def map_entrez_to_mgi(expr, folder):
    """
    GSE17576: rename BrainArray Entrez-based probe IDs ('12345_at') to
    current MGI gene symbols via mygene.info.

    The BrainArray Mm_ENTREZG_10 CDF guarantees one probe per Entrez gene,
    so no aggregation is needed — only a rename.
    """
    probe_ids  = expr.index.tolist()
    entrez_ids = [p.split("_")[0] for p in probe_ids]

    cache = _mygene_querymany(entrez_ids, "entrezgene", folder, "mygene_cache.json")

    n_before  = len(expr)
    new_names = [cache.get(e) for e in entrez_ids]

    expr = expr.copy()
    expr.index = new_names
    expr = expr[expr.index.notna() & (expr.index != "")]
    expr = expr[~expr.index.duplicated(keep="first")]
    expr.index.name = "gene"

    log.info(
        f"    Entrez→MGI: {len(expr):,} genes mapped, "
        f"{n_before - len(expr):,} probes dropped (no mapping found)"
    )
    return expr


def _fetch_gpl6246_annotation(folder):
    """
    Download the GPL6246 (Affymetrix MoGene 1.0 ST) platform annotation from
    NCBI GEO FTP and cache it locally as a CSV.

    Returns
    -------
    Series  transcript_cluster_id (str) → gene_symbol (str or NaN)
    """
    import gzip as gz_mod
    from io import StringIO

    cache_path = os.path.join(folder, "gpl6246_gene_symbols.csv.gz")

    if os.path.exists(cache_path):
        s = pd.read_csv(cache_path, index_col=0, compression="gzip").squeeze()
        s.index = s.index.astype(str)   # CSV round-trip converts numeric IDs to int64
        log.info(f"    GPL6246 annotation loaded from cache ({len(s):,} probes)")
        return s

    url = (
        "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/"
        "GPL6246/annot/GPL6246.annot.gz"
    )
    log.info("    Downloading GPL6246 platform annotation from NCBI GEO ...")

    with urllib.request.urlopen(url) as resp:
        raw = gz_mod.decompress(resp.read()).decode("utf-8", errors="replace")

    lines, in_table = [], False
    for line in raw.splitlines():
        if "!platform_table_begin" in line:
            in_table = True
            continue
        elif "!platform_table_end" in line:
            break
        elif in_table:
            lines.append(line)

    annot = pd.read_csv(
        StringIO("\n".join(lines)), sep="\t", index_col=0, low_memory=False
    )
    annot.index = annot.index.astype(str)

    sym_col = next(
        (c for c in annot.columns if "gene symbol" in c.lower()), None
    )
    if sym_col is None:
        raise ValueError(
            f"Cannot find 'Gene Symbol' column in GPL6246 annotation. "
            f"Columns present: {list(annot.columns)}"
        )

    gene_sym = annot[sym_col].copy()
    gene_sym = gene_sym.apply(
        lambda x: x.split(" /// ")[0].strip() if isinstance(x, str) else None
    )
    gene_sym = gene_sym.replace({"---": None, "": None})

    gene_sym.to_frame("gene_symbol").to_csv(cache_path, compression="gzip")
    log.info(
        f"    GPL6246 annotation cached → gpl6246_gene_symbols.csv.gz "
        f"({gene_sym.notna().sum():,} / {len(gene_sym):,} probes annotated)"
    )
    return gene_sym


def _fetch_gpl23038_annotation(folder):
    """
    Stream the GPL23038 (Affymetrix Clariom S Mouse) platform annotation from
    the NCBI GEO SOFT file, stopping immediately after the platform table to
    avoid downloading the full 1.3 GB file.

    GPL23038 has no pre-built annot/ file. The SOFT file's platform table
    (~36 K rows) precedes all sample data, so we read only until the
    '!platform_table_end' sentinel.

    Gene symbols are extracted from the 'mrna_assignment' column, which encodes
    RefSeq entries in the form:
        "NM_XXXXX // RefSeq // Mus musculus <description> (Symbol), mRNA ..."
    We take the first symbol found in any '(Symbol),' pattern.

    Returns
    -------
    Series  transcript_cluster_id (str) → gene_symbol (str or NaN)
    """
    import gzip as gz_mod
    from io import StringIO

    cache_path = os.path.join(folder, "gpl23038_gene_symbols.csv.gz")

    if os.path.exists(cache_path):
        s = pd.read_csv(cache_path, index_col=0, compression="gzip").squeeze()
        s.index = s.index.astype(str)   # CSV round-trip converts numeric IDs to int64
        log.info(f"    GPL23038 annotation loaded from cache ({len(s):,} probes)")
        return s

    url = (
        "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL23nnn/"
        "GPL23038/soft/GPL23038_family.soft.gz"
    )
    log.info(
        "    Streaming GPL23038 platform annotation from NCBI GEO SOFT file "
        "(reading platform table only) ..."
    )

    # Stream line-by-line; stop after the platform table to avoid the 1.3 GB download
    table_lines = []
    in_table    = False
    with urllib.request.urlopen(url) as resp:
        with gz_mod.GzipFile(fileobj=resp) as gz:
            for raw_line in gz:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")
                if "!platform_table_begin" in line:
                    in_table = True
                    continue
                if "!platform_table_end" in line:
                    break
                if in_table:
                    table_lines.append(line)

    annot = pd.read_csv(
        StringIO("\n".join(table_lines)), sep="\t", index_col=0, low_memory=False
    )
    annot.index = annot.index.astype(str)

    # Extract gene symbol from the mrna_assignment column.
    # RefSeq entries follow the pattern: "Mus musculus ... (Symbol), mRNA"
    _sym_re = re.compile(r"\(([A-Za-z][A-Za-z0-9]+)\),")

    def _extract_symbol(mrna_assignment):
        if not isinstance(mrna_assignment, str):
            return None
        m = _sym_re.search(mrna_assignment)
        return m.group(1) if m else None

    mrna_col = next(
        (c for c in annot.columns if "mrna_assignment" in c.lower()), None
    )
    if mrna_col is None:
        raise ValueError(
            f"Cannot find 'mrna_assignment' column in GPL23038 annotation. "
            f"Columns present: {list(annot.columns)}"
        )

    gene_sym = annot[mrna_col].apply(_extract_symbol)
    gene_sym.name = "gene_symbol"

    gene_sym.to_frame("gene_symbol").to_csv(cache_path, compression="gzip")
    log.info(
        f"    GPL23038 annotation cached → gpl23038_gene_symbols.csv.gz "
        f"({gene_sym.notna().sum():,} / {len(gene_sym):,} probes annotated)"
    )
    return gene_sym


def map_affy_mogene_to_mgi(expr, folder):
    """
    GSE103726: map Affymetrix MoGene 1.0 ST transcript cluster IDs to MGI
    gene symbols using the official GPL6246 platform annotation from NCBI GEO.

    Multiple probes mapping to the same gene are collapsed by taking their
    mean — the standard approach for microarray re-analysis.
    """
    gene_sym = _fetch_gpl6246_annotation(folder)

    expr = expr.copy()
    expr.index = expr.index.astype(str).map(gene_sym)

    n_before = expr.shape[0]
    expr = expr[expr.index.notna() & (expr.index != "")]
    n_named  = expr.shape[0]

    expr = expr.groupby(expr.index).mean()
    expr.index.name = "gene"

    log.info(
        f"    GPL6246→MGI: {n_before - n_named:,} probes unmapped (dropped), "
        f"{n_named - expr.shape[0]:,} duplicate probes collapsed by mean → "
        f"{expr.shape[0]:,} unique genes"
    )
    return expr


def map_clariom_s_to_mgi(expr, folder):
    """
    GSE241830: map Affymetrix Clariom S transcript cluster IDs to MGI gene
    symbols using the GPL23038 platform annotation from NCBI GEO.

    Multiple transcript clusters mapping to the same gene are collapsed by
    taking their mean.
    """
    gene_sym = _fetch_gpl23038_annotation(folder)

    expr = expr.copy()
    expr.index = expr.index.astype(str).map(gene_sym)

    n_before = expr.shape[0]
    expr = expr[expr.index.notna() & (expr.index != "")]
    n_named  = expr.shape[0]

    expr = expr.groupby(expr.index).mean()
    expr.index.name = "gene"

    log.info(
        f"    GPL23038→MGI: {n_before - n_named:,} probes unmapped (dropped), "
        f"{n_named - expr.shape[0]:,} duplicate probes collapsed by mean → "
        f"{expr.shape[0]:,} unique genes"
    )
    return expr


def validate_symbols_mgi(expr, folder):
    """
    GSE159558 / GSE182686 / GSE161693: validate existing mouse gene symbols
    against MGI via mygene.info and update any outdated aliases to the current
    official symbol. Symbols not recognised by mygene are kept as-is.
    """
    symbols = expr.index.tolist()
    cache   = _mygene_querymany(
        symbols, "symbol,alias", folder, "mygene_cache.json"
    )

    new_names = [cache.get(s) or s for s in symbols]
    n_updated = sum(a != b for a, b in zip(symbols, new_names))

    expr = expr.copy()
    expr.index = new_names
    expr = expr[~expr.index.duplicated(keep="first")]
    expr.index.name = "gene"

    log.info(
        f"    MGI symbol validation: {n_updated} aliases updated, "
        f"{len(expr):,} unique genes retained"
    )
    return expr


def map_to_mgi_symbols(expr, gse_id, folder):
    """Dispatch to the correct MGI mapping strategy for each dataset."""
    log.info("    Mapping gene identifiers to MGI standard symbols ...")
    if gse_id == "GSE17576":
        return map_entrez_to_mgi(expr, folder)
    elif gse_id == "GSE103726":
        return map_affy_mogene_to_mgi(expr, folder)
    elif gse_id == "GSE241830":
        return map_clariom_s_to_mgi(expr, folder)
    elif gse_id in ("GSE159558", "GSE182686", "GSE161693", "GSE275818", "GSE305719"):
        return validate_symbols_mgi(expr, folder)
    else:
        log.warning(f"    No MGI mapping defined for {gse_id}, index unchanged")
        return expr


# ═════════════════════════════════════════════════════════════════════════════
# Per-dataset processors
# ═════════════════════════════════════════════════════════════════════════════

def process_gse17576(folder, config):
    """
    GSE17576 – HFD obesity, Affymetrix Mouse Genome 430 2.0 Array.

    Normalization: GC-RMA (slow) with custom CDF Mm_ENTREZG_10 — already
    applied by authors. Probe IDs are Entrez gene IDs (from the BrainArray
    custom CDF). Values in the series matrix are in linear scale.

    Design
    ------
    10 samples × LFD day 0  (baseline)
    10 samples × LFD day 56
    10 samples × HFD day 56
    """
    matrix_path = os.path.join(folder, config["series_matrix"])
    expr, sample_ann = parse_geo_series_matrix(matrix_path)
    sample_ann = annotate_gse17576(sample_ann)
    log.info(f"    Conditions: {sample_ann['condition'].value_counts().to_dict()}")
    return expr, sample_ann


def process_gse103726(folder, config):
    """
    GSE103726 – Blunt muscle injury, Affymetrix MoGene 1.0 ST Array.

    Normalization: RMA (via affy package) — already applied by authors.
    Values in the series matrix are in linear scale. Probe IDs are Affymetrix
    transcript cluster IDs (numeric).

    Design (5 timepoints × 2 injury states × 2 diets × 3 replicates = 60)
    ------
    timepoints : 1h, 6h, 24h, 3d, 8d
    injury     : sham (C) | injured (T)
    diet       : normal weight (10) | obese (60)
    """
    matrix_path = os.path.join(folder, config["series_matrix"])
    expr, sample_ann = parse_geo_series_matrix(matrix_path)
    sample_ann = annotate_gse103726(sample_ann)
    log.info(f"    Conditions: {sample_ann['condition'].value_counts().to_dict()}")
    return expr, sample_ann


def process_gse159558(folder, config):
    """
    GSE159558 – USP21 muscle KO, Illumina NovaSeq 6000.

    The expression file contains Read_Count, FPKM, and TPM columns for all
    samples combined in a single file. We extract only the Read_Count columns
    and normalise with VST via PyDESeq2.

    Samples (confirmed from GEO GSM4832489–GSM4832494)
    -------
    L_GC_6, L_GC_7, L_GC_8       : USP21 flox/flox littermate controls (WT)
    MKO_GC_1, MKO_GC_2, MKO_GC_4 : Ckmm-Cre; USP21 flox/flox (muscle KO)
    """
    counts_path = os.path.join(folder, config["counts_file"])
    log.info(f"    Reading {config['counts_file']} ...")
    df = pd.read_csv(counts_path, sep="\t", compression="gzip")

    count_cols   = [c for c in df.columns if c.endswith("_Read_Count")]
    sample_names = [c.replace("_Read_Count", "") for c in count_cols]

    counts = df[count_cols].copy()
    counts.columns = sample_names

    counts.index = df["Gene_Symbol"]
    counts = counts[counts.index.notna() & (counts.index != "")]
    counts = counts[~counts.index.duplicated(keep="first")]
    counts.index.name = "gene"

    sample_ann = pd.DataFrame(
        {
            "sample_id": sample_names,
            "condition": [
                "WT_control" if s.startswith("L_GC") else "USP21_MKO"
                for s in sample_names
            ],
            "genotype": [
                "USP21_flox/flox"
                if s.startswith("L_GC")
                else "Ckmm-Cre;USP21_flox/flox"
                for s in sample_names
            ],
            "replicate": [s.split("_")[-1] for s in sample_names],
        }
    ).set_index("sample_id")

    log.info(f"    Conditions: {sample_ann['condition'].value_counts().to_dict()}")

    counts = filter_low_counts(counts)
    vst    = vst_normalize(counts, sample_ann, design_factor="condition")
    return vst, sample_ann


def process_gse182686(folder, config):
    """
    GSE182686 – FGF6 delivery, Illumina HiSeq 2500.

    Raw counts file has gene symbols as the row index and fei_1–fei_6 as
    column names. Sample identity confirmed from GEO (GSM5534724–5534729):
    the series matrix lists samples in the order FGF6-1, Ctrl-1, FGF6-2,
    Ctrl-2, FGF6-3, Ctrl-3 — matching the fei_1–fei_6 column order.

    Samples
    -------
    fei_1, fei_3, fei_5 : AAV9-FGF6 (treated)
    fei_2, fei_4, fei_6 : AAV9-Control (control)
    """
    counts_path = os.path.join(folder, config["counts_file"])
    log.info(f"    Reading {config['counts_file']} ...")
    counts = pd.read_csv(counts_path, sep="\t", compression="gzip", index_col=0)
    counts.index.name = "gene"

    sample_ann = pd.DataFrame(
        {
            "sample_id": ["fei_1", "fei_2", "fei_3", "fei_4", "fei_5", "fei_6"],
            "geo_title": [
                "FGF6-1", "Ctrl-1", "FGF6-2", "Ctrl-2", "FGF6-3", "Ctrl-3"
            ],
            "condition": [
                "FGF6", "Control", "FGF6", "Control", "FGF6", "Control"
            ],
            "treatment": [
                "AAV9-FGF6",     "AAV9-Control",
                "AAV9-FGF6",     "AAV9-Control",
                "AAV9-FGF6",     "AAV9-Control",
            ],
            "replicate": ["1", "1", "2", "2", "3", "3"],
        }
    ).set_index("sample_id")

    log.info(f"    Conditions: {sample_ann['condition'].value_counts().to_dict()}")

    counts = filter_low_counts(counts)
    vst    = vst_normalize(counts, sample_ann, design_factor="condition")
    return vst, sample_ann


def process_gse161693(folder, config):
    """
    GSE161693 – Notch2 muscle KO, Illumina NovaSeq 6000.

    Data are split across 4 supplementary files — one per pairwise comparison
    (WT: control vs unloading / control vs diabetes; KO: same). Each file
    contains 4+4 samples with '(raw)' and '(normalized)' columns. The control
    samples (WT_CT and KO_CT) appear in two files each; we deduplicate them.

    We merge all unique raw-count columns into a single matrix, then apply VST.

    Groups (n=4 replicates each, 24 samples total)
    -------
    WT_CT  : wild-type control
    WT_HU  : wild-type hindlimb unloading (muscle atrophy model)
    WT_DB  : wild-type diabetes (STZ-induced)
    KO_CT  : Notch2 muscle-specific KO control
    KO_HU  : Notch2 KO + hindlimb unloading
    KO_DB  : Notch2 KO + diabetes
    """
    log.info("    Reading 4 supplementary count files ...")

    def _read_raw_counts(path):
        """Read one supplementary file; return DataFrame(gene_symbol → raw_count_cols)."""
        df = pd.read_csv(path, sep="\t", skiprows=4, index_col=0,
                         compression="gzip", low_memory=False)
        raw_cols = [c for c in df.columns if c.endswith("(raw)")]
        symbols  = df["Gene Symbol"].astype(str)
        counts   = df[raw_cols].copy()
        counts.columns = [c.replace("(raw)", "").strip() for c in raw_cols]
        counts.index   = symbols
        counts = counts[counts.index.notna() & (counts.index != "") & (counts.index != "nan")]
        counts = counts.groupby(counts.index).mean()
        return counts

    files = config["suppl_files"]
    wt_cu = _read_raw_counts(os.path.join(folder, files[0]))  # WT_CT1-4, WT_HU1-4
    wt_cd = _read_raw_counts(os.path.join(folder, files[1]))  # WT_CT1-4 (dup), WT_DB1-4
    ko_cu = _read_raw_counts(os.path.join(folder, files[2]))  # KO_CT1-4, KO_HU1-4
    ko_cd = _read_raw_counts(os.path.join(folder, files[3]))  # KO_CT1-4 (dup), KO_DB1-4

    # Drop duplicated control columns — keep them from the first occurrence only
    wt_hu_cols = [c for c in wt_cu.columns if c.startswith("WT_HU")]
    wt_db_cols = [c for c in wt_cd.columns if c.startswith("WT_DB")]
    ko_hu_cols = [c for c in ko_cu.columns if c.startswith("KO_HU")]
    ko_db_cols = [c for c in ko_cd.columns if c.startswith("KO_DB")]
    wt_ct_cols = sorted(c for c in wt_cu.columns if c.startswith("WT_CT"))
    ko_ct_cols = sorted(c for c in ko_cu.columns if c.startswith("KO_CT"))

    # Align on gene index (outer join, fill missing with 0)
    ordered_cols = wt_ct_cols + wt_hu_cols + wt_db_cols + ko_ct_cols + ko_hu_cols + ko_db_cols
    parts = [
        wt_cu[wt_ct_cols + wt_hu_cols],
        wt_cd[wt_db_cols],
        ko_cu[ko_ct_cols + ko_hu_cols],
        ko_cd[ko_db_cols],
    ]
    counts = parts[0]
    for part in parts[1:]:
        counts = counts.join(part, how="outer")
    counts = counts.fillna(0)
    counts = counts[ordered_cols]
    counts.index.name = "gene"

    log.info(f"    Merged count matrix: {counts.shape[0]:,} genes × {counts.shape[1]} samples")

    # Condition map: prefix → condition label
    def _condition(col_name):
        prefix = col_name[:5]  # 'WT_CT', 'WT_HU', 'WT_DB', 'KO_CT', 'KO_HU', 'KO_DB'
        return {
            "WT_CT": "WT_control",
            "WT_HU": "WT_unloading",
            "WT_DB": "WT_diabetes",
            "KO_CT": "KO_control",
            "KO_HU": "KO_unloading",
            "KO_DB": "KO_diabetes",
        }[prefix]

    def _genotype(col_name):
        return "Notch2_mKO" if col_name.startswith("KO") else "WT"

    def _treatment(col_name):
        suffix = col_name[3:5]  # 'CT', 'HU', 'DB'
        return {"CT": "control", "HU": "hindlimb_unloading", "DB": "diabetes"}[suffix]

    sample_ann = pd.DataFrame(
        {
            "sample_id": counts.columns.tolist(),
            "condition": [_condition(c) for c in counts.columns],
            "genotype":  [_genotype(c)  for c in counts.columns],
            "treatment": [_treatment(c) for c in counts.columns],
            "replicate": [re.search(r"\d+$", c).group() for c in counts.columns],
        }
    ).set_index("sample_id")

    log.info(f"    Conditions: {sample_ann['condition'].value_counts().to_dict()}")

    counts = filter_low_counts(counts)
    vst    = vst_normalize(counts, sample_ann, design_factor="condition")
    return vst, sample_ann


def process_gse241830(folder, config):
    """
    GSE241830 – TAS1R2 ribosomal pulldown, Affymetrix Clariom S Mouse.

    Normalization: SST-RMA via Transcriptome Analysis Console 4.0 — already
    applied by authors. Values in the series matrix are in LOG2 scale (SST-RMA
    standard output). Probe IDs are Clariom S transcript cluster IDs.

    All 5 samples are sedentary soleus myocytes from Myogenin-Cre × RiboTag
    mice — a single-condition characterization of actively translated mRNAs.
    """
    matrix_path = os.path.join(folder, config["series_matrix"])
    # log2_transform=False: SST-RMA values are already on a log2 scale
    expr, sample_ann = parse_geo_series_matrix(matrix_path, log2_transform=False)
    sample_ann = annotate_gse241830(sample_ann)
    log.info(f"    Conditions: {sample_ann['condition'].value_counts().to_dict()}")
    return expr, sample_ann


def process_gse275818(folder, config):
    """
    GSE275818 – Running distance + HFD, Illumina HiSeq 4000.

    The count file contains 36 sample columns (6 groups × 6 replicates) plus
    9 annotation columns (gene_name, gene_chr, etc.). We extract only the
    sample columns (those that do not contain '_' or match known annotation
    column names) and use gene_name for the gene index.

    Groups (n=6 replicates each, 36 samples total)
    ------
    CS    : chow diet, sedentary
    HFS   : high-fat diet, sedentary
    HFuR  : high-fat diet + unlimited voluntary running
    HF25  : high-fat diet + 25 m/day defined running
    HF50  : high-fat diet + 50 m/day defined running
    HF75  : high-fat diet + 75 m/day defined running
    """
    counts_path = os.path.join(folder, config["counts_file"])
    log.info(f"    Reading {config['counts_file']} ...")
    df = pd.read_csv(counts_path, sep="\t", compression="gzip", index_col=0)

    # Annotation columns follow the sample columns in this file
    annotation_cols = {
        "gene_name", "gene_chr", "gene_start", "gene_end",
        "gene_strand", "gene_length", "gene_biotype", "gene_description", "tf_family",
    }
    sample_cols = [c for c in df.columns if c not in annotation_cols]

    counts = df[sample_cols].copy()
    counts.index = df["gene_name"].astype(str)
    counts = counts[counts.index.notna() & (counts.index != "") & (counts.index != "nan")]
    counts = counts[~counts.index.duplicated(keep="first")]
    counts.index.name = "gene"

    # Derive group prefix from sample name (e.g. 'HF25_3' → 'HF25')
    condition_map = {
        "CS":   "chow_sedentary",
        "HFS":  "HFD_sedentary",
        "HFuR": "HFD_unlimited_run",
        "HF25": "HFD_25m_run",
        "HF50": "HFD_50m_run",
        "HF75": "HFD_75m_run",
    }

    def _group(col_name):
        # Sample names are like 'CS_1', 'HFuR_3', 'HF25_6', etc.
        prefix = col_name.rsplit("_", 1)[0]
        return condition_map.get(prefix, prefix)

    sample_ann = pd.DataFrame(
        {
            "sample_id": sample_cols,
            "condition": [_group(c) for c in sample_cols],
            "group":     [c.rsplit("_", 1)[0] for c in sample_cols],
            "replicate": [c.rsplit("_", 1)[1] for c in sample_cols],
        }
    ).set_index("sample_id")

    log.info(f"    Conditions: {sample_ann['condition'].value_counts().to_dict()}")

    counts = filter_low_counts(counts)
    vst    = vst_normalize(counts, sample_ann, design_factor="condition")
    return vst, sample_ann


def process_gse305719(folder, config):
    """
    GSE305719 – Multi-organ HFD time course (metabolic syndrome model mice).
    Oku et al. (2025), NOLTA — DNB analysis across 13 organs and 14–16 time
    points in C57BL/6J mice fed a high-fat diet from 8 weeks of age.

    The single supplementary CSV contains raw integer counts with gene symbols
    in the first column and 1,104 sample columns named `{organ}_{time}_{rep}`.

    Organ codes (13)
    ----------------
        bat   : interscapular brown adipose tissue
        ewat  : epididymal white adipose tissue (16 timepoints — incl. 16w, 20w)
        iwat  : inguinal white adipose tissue
        liver : liver
        panc  : pancreas
        gut   : small intestine
        sol   : soleus muscle
        gas   : gastrocnemius muscle
        hypo  : hypothalamus
        pit   : pituitary gland
        neo   : neocortex
        hipp  : hippocampus
        olf   : olfactory bulb

    Time points (HFD onset = 0d): 0d, 1d, 3d, 1w, 10d, 2w, 3w, 4w, 5w, 6w, 7w,
    8w, 10w, 12w. eWAT additionally has 16w and 20w. Six biological replicates
    per (organ × timepoint).

    The `condition` column is set to `{organ}_{timepoint}` (e.g. 'liver_4w'),
    which is what VST uses as the design factor. Organ and timepoint are also
    stored as separate columns to make downstream subsetting easier.
    """
    counts_path = os.path.join(folder, config["counts_file"])
    log.info(f"    Reading {config['counts_file']} ...")
    counts = pd.read_csv(counts_path, compression="gzip", index_col=0)
    counts.index = counts.index.astype(str)
    counts = counts[counts.index.notna() & (counts.index != "") & (counts.index != "nan")]
    counts = counts[~counts.index.duplicated(keep="first")]
    counts.index.name = "gene"

    # Sample names follow `{organ}_{timepoint}_{replicate}` — split on the
    # last two underscores so that timepoints like '10d', '12w' are preserved.
    def _split(col):
        parts = col.rsplit("_", 2)
        return parts[0], parts[1], parts[2]   # organ, timepoint, replicate

    organ, timepoint, replicate = zip(*[_split(c) for c in counts.columns])

    sample_ann = pd.DataFrame(
        {
            "sample_id": list(counts.columns),
            "organ":     list(organ),
            "timepoint": list(timepoint),
            "replicate": list(replicate),
            "diet":      ["HFD"] * counts.shape[1],
            "condition": [f"{o}_{t}" for o, t in zip(organ, timepoint)],
        }
    ).set_index("sample_id")

    log.info(
        f"    {counts.shape[0]:,} genes × {counts.shape[1]} samples; "
        f"{sample_ann['organ'].nunique()} organs × "
        f"{sample_ann['timepoint'].nunique()} timepoints"
    )

    counts = filter_low_counts(counts)
    # Use 'organ' as the VST design factor — it is the dominant source of
    # variance in this multi-organ dataset and keeps the design parsimonious
    # given the very large number of (organ × timepoint) combinations.
    vst = vst_normalize(counts, sample_ann, design_factor="organ")
    return vst, sample_ann


# ═════════════════════════════════════════════════════════════════════════════
# Dataset dispatcher
# ═════════════════════════════════════════════════════════════════════════════

PROCESSORS = {
    "GSE17576":  process_gse17576,
    "GSE305719": process_gse305719,
}


def process_dataset(gse_id, config):
    folder = os.path.join(BASE_DIR, gse_id)

    log.info("")
    log.info("=" * 60)
    log.info(f"  {gse_id}")
    log.info(f"  {config['publication']}")
    log.info(f"  Type : {config['type']}")
    log.info("=" * 60)

    processor = PROCESSORS[gse_id]
    expr, sample_ann = processor(folder, config)

    # ── Map to MGI standard gene symbols ──────────────────────────────────
    expr = map_to_mgi_symbols(expr, gse_id, folder)

    # ── Save outputs ──────────────────────────────────────────────────────
    expr_path = os.path.join(folder, "normalized_expression.csv.xz")
    ann_path  = os.path.join(folder, "sample_annotation.csv.xz")
    compression = {"method": "xz", "preset": 9}

    expr.to_csv(expr_path, compression=compression)
    sample_ann.to_csv(ann_path, compression=compression)

    log.info(
        f"    Saved: normalized_expression.csv.xz  "
        f"({expr.shape[0]:,} genes × {expr.shape[1]} samples)"
    )
    log.info("    Saved: sample_annotation.csv.xz")


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Transcriptome dataset preparation")
    log.info(f"Base directory : {BASE_DIR}")

    for gse_id, config in DATASETS.items():
        try:
            process_dataset(gse_id, config)
        except Exception as exc:
            log.error(f"  FAILED for {gse_id}: {exc}")
            raise

    log.info("")
    log.info("=" * 60)
    log.info("Selected datasets processed successfully.")
    log.info("")
    log.info("Output files per folder:")
    log.info("  normalized_expression.csv.xz  — log2-scale, ready for enrichment")
    log.info("  sample_annotation.csv.xz      — condition labels per sample")


if __name__ == "__main__":
    main()
