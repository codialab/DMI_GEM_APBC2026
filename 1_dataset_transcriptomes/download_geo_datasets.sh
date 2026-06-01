#!/bin/bash
# Download transcriptome datasets from GEO for your 5 papers
# Papers 3 and 5 do not have public GEO deposits (see notes below)

set -e
OUTDIR="geo_datasets"
mkdir -p "$OUTDIR"

echo "============================================"
echo "Paper 1: 8-week high-fat diet (GSE17576)"
echo "  Microarray - Affymetrix Mouse Genome 430 2.0"
echo "============================================"
mkdir -p "$OUTDIR/GSE17576"
wget -P "$OUTDIR/GSE17576" \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE17nnn/GSE17576/suppl/GSE17576_RAW.tar"
wget -P "$OUTDIR/GSE17576" \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE17nnn/GSE17576/matrix/GSE17576_series_matrix.txt.gz"
cd "$OUTDIR/GSE17576" && tar xf GSE17576_RAW.tar && cd ../..
echo "Done: GSE17576"
echo ""

echo "============================================"
echo "Paper 2: Fgf6 skeletal muscle (GSE182686)"
echo "  RNA-Seq + MeDIP-Seq"
echo "============================================"
mkdir -p "$OUTDIR/GSE182686"
wget -P "$OUTDIR/GSE182686" \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE182nnn/GSE182686/suppl/GSE182686_RAW.tar"
wget -P "$OUTDIR/GSE182686" \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE182nnn/GSE182686/matrix/GSE182686_series_matrix.txt.gz"
cd "$OUTDIR/GSE182686" && tar xf GSE182686_RAW.tar 2>/dev/null; cd ../..
echo "Done: GSE182686"
echo ""

echo "============================================"
echo "Paper 4: USP21 ablation (GSE159558)"
echo "  RNA-Seq"
echo "============================================"
mkdir -p "$OUTDIR/GSE159558"
wget -P "$OUTDIR/GSE159558" \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE159nnn/GSE159558/suppl/GSE159558_RAW.tar"
wget -P "$OUTDIR/GSE159558" \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE159nnn/GSE159558/matrix/GSE159558_series_matrix.txt.gz"
cd "$OUTDIR/GSE159558" && tar xf GSE159558_RAW.tar 2>/dev/null; cd ../..
echo "Done: GSE159558"
echo ""

echo "============================================"
echo "NOTES on missing datasets:"
echo ""
echo "Paper 3 (Werner et al. 2018 - blunt muscle injury):"
echo "  Microarray data was NOT deposited in a public repository."
echo "  Contact: Uwe Knippschild, Ulm University"
echo ""
echo "Paper 5 (Tamaki et al. 2016 - diabetic mice ectopic fat):"
echo "  Used qPCR for individual genes, not genome-wide transcriptomics."
echo "  No transcriptome dataset available."
echo "============================================"
echo ""
echo "All available datasets downloaded to: $OUTDIR/"
