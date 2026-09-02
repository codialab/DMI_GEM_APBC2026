This repository stores the code and processed data for Uncertainty-aware integration of deuterium metabolic imaging with genome-scale metabolic modeling reveals diet-associated feasible-space remodeling in mouse skeletal muscle, a conference paper of APBC 2026. This study integrates deuterium metabolic imaging (DMI), transcriptomics, kinetic model fitting, and genome-scale metabolic modelling (GEM) in mouse skeletal muscle.

## Repository structure

```text
DMI_GEM_APBC2026/
├── 1_dataset_transcriptomes/              # transcriptome preprocessing
├── 2_extract_DMI_met/                     # DMI metabolite extraction / ROI processing
├── 3_model_fit/                           # kinetic DMI model fitting and uncertainty estimation
├── 4_metabolic_network_reconstruction/    # muscle-specific GEM reconstruction and subsystem lists
├── LICENSE
└── README.md
```

Several intermediate and final outputs are included in the repository. Therefore, downstream stages can be inspected or rerun without necessarily repeating every computationally expensive upstream step.

## Software requirements

### Python

Python **3.10 or newer** is required by the syntax used in the analysis scripts. Python 3.10 or 3.11 is a conservative choice for reproducing the current environment.

Create and activate a virtual environment, for example:

```bash
git clone https://github.com/codialab/DMI_GEM_APBC2026.git
cd DMI_GEM_APBC2026

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the main Python dependencies:

```bash
pip install \
    numpy \
    pandas \
    scipy \
    joblib \
    matplotlib \
    jupyterlab \
    pydeseq2 \
    mygene \
    openpyxl
```

For genome-scale metabolic modelling and CORDA reconstruction:

```bash
pip install cobra==0.31.1 corda==0.5.1
```

The reconstruction notebook used Gurobi 13.0.1:

```bash
pip install gurobipy==13.0.1
```

A valid Gurobi license is recommended for fast reconstruction. If Gurobi is unavailable, the notebook falls back to GLPK. Installing `swiglpk` explicitly is recommended if GLPK will be used:

```bash
pip install swiglpk
```

`openpyxl` is needed only for scripts that export Excel workbooks.

> **Reproducibility note:** exact versions are recorded in the stage-4 notebook for COBRApy (0.31.1), CORDA (0.5.1), and Gurobi (13.0.1). Other package versions are not currently pinned. For archival reproducibility, a tested `requirements.txt` or Conda environment file should be added after a clean-environment run.

## Running the analysis

### 1. Transcriptome preprocessing

Directory:

```text
1_dataset_transcriptomes/
```

The stage prepares the GSE17576 microarray dataset and the GSE305719 RNA-seq dataset. Raw source files and processed expression/annotation tables are already included in their respective subdirectories.

Run:

```bash
cd 1_dataset_transcriptomes
python prepare_transcriptome_datasets.py
cd ..
```

Main outputs are written into each dataset directory:

```text
GSE17576/normalized_expression.csv.xz
GSE17576/sample_annotation.csv.xz
GSE305719/normalized_expression.csv.xz
GSE305719/sample_annotation.csv.xz
```

`load_datasets.py` is a helper for loading the prepared datasets and is not required as a separate pipeline step.

**Important:** `download_geo_datasets.sh` is an older download helper and does not currently match the two-dataset preprocessing workflow above. The required source files are already included in the repository, so this shell script is not needed for the main reproduction workflow.

### 2. DMI metabolite extraction

Directory:

```text
2_extract_DMI_met/
```

The primary workflow is notebook-based. Start Jupyter from this directory:

```bash
cd 2_extract_DMI_met
jupyter lab
```

Open and execute:

```text
extract_met_conc.ipynb
```

`plot_ROI.ipynb` can be used for ROI visualization/QC.

The repository already contains the principal processed outputs used by stage 3:

```text
data_tissue_vals_model_fitting.joblib.xz
table_voxel_met_conc.csv.xz
```

It also contains `Data_Samia.tar.xz`, ROI definitions in `lib_roi_masks.py`, and the associated ROI/visualization resources.

After execution, return to the repository root:

```bash
cd ..
```

### 3. Kinetic model fitting

Directory:

```text
3_model_fit/
```

The preferred reproducible entry point for the main C6 fit is the Python script rather than the notebook:

```bash
cd 3_model_fit
python model_fittingC6_improved.py
```

The script reads:

```text
../2_extract_DMI_met/data_tissue_vals_model_fitting.joblib.xz
```

and performs the two-phase DMI kinetic fitting procedure. It includes lightweight sanity checks before launching the full calculation and uses multiprocessing across independent mouse/week fits.

Principal outputs include:

```text
fitC6a_results_MRI_fluxes_noGly_oxDilution_KTfixed.joblib.xz
fitC6b_results_MRI_fluxes_AllTissues_noGly_oxDilution_KTfixed.joblib.xz
```

To estimate parameter/flux uncertainty from the C6 results, run:

```bash
python model_fittingC7_estimate_variations.py
```

Optional command-line filters are available, for example:

```bash
python model_fittingC7_estimate_variations.py --mouse-ids 49 50 --weeks w9
```

The C7 script is resumable and reuses existing uncertainty results unless `--force` is supplied.

> **Compute note:** the C6 fitting stage can be computationally expensive and launches multiple worker processes. Run it on a machine with sufficient CPU and memory resources.

The notebooks `model_fittingC6_improved.ipynb` and `model_fittingC7b_improved.ipynb` are retained for interactive inspection, but the Python scripts should be preferred for reproducible batch execution.

After execution:

```bash
cd ..
```

### 4. Muscle-specific genome-scale metabolic model reconstruction

Directory:

```text
4_metabolic_network_reconstruction/
```

This stage constructs a muscle-specific model from iMM1865 using CORDA and GSE17576 expression data.

**Run Jupyter from inside this directory.** The notebook uses `Path('..')` to identify the repository root, so its working directory matters.

```bash
cd 4_metabolic_network_reconstruction
jupyter lab muscle_specific_model.ipynb
```

Run all cells in order.

The notebook reads:

```text
../1_dataset_transcriptomes/GSE17576/normalized_expression.csv.xz
../1_dataset_transcriptomes/GSE17576/sample_annotation.csv.xz
../1_dataset_transcriptomes/GSE17576/mygene_cache.json
iMM1865/iMM1865.xml.xz
```

and writes:

```text
muscle_specific_model.xml
muscle_specific_model.json
muscle_gene_confidence.csv
```

The repository also contains compressed copies (`muscle_specific_model.xml.xz` and `muscle_specific_model.json.xz`) of the reconstructed model.

To generate the manually curated pathway/subsystem reaction lists from the original iMM1865 model, run from this same directory:

```bash
python build_subsystem_reaction_list.py
```

To generate the model-native subsystem list from the newly reconstructed JSON model, run:

```bash
python build_native_subsystem_reaction_list.py --model muscle_specific_model.json
```

Alternatively, when using the precomputed compressed model already included in the repository, the default command is sufficient:

```bash
python build_native_subsystem_reaction_list.py
```

This writes results under:

```text
results_native_subsystem_reactions/
```

Return to the repository root when finished:

```bash
cd ..
```

## Precomputed outputs and partial reruns

The repository intentionally includes several processed/intermediate files. In particular:

- stage 1 contains normalized transcriptome matrices and sample annotations;
- stage 2 contains the serialized DMI data object required by stage 3;
- stage 3 contains fitted C6/C7 results and manuscript-oriented summary tables;
- stage 4 contains the reconstructed muscle-specific model and subsystem reaction tables.

This allows users to inspect or reproduce downstream analyses without rerunning every upstream computation.

## Current auxiliary-script notes

Two scripts are **not part of the four-stage core reproduction path in the repository as currently distributed**:

1. `3_model_fit/make_supp_table_week9_dmi_flux_summary.py` expects an upstream `metabolic_network/v6b_met1_modelFitC6/...` table that is not included in this repository. The resulting supplementary table files are already committed.
2. `4_metabolic_network_reconstruction/audit_mass_balance.py` expects `results_fba_constrained_fit_B6/flux_distribution__hierarchical_atp.csv`, which is not included in the current repository. It should therefore be treated as an auxiliary/legacy audit unless the corresponding upstream FBA results are supplied.

## Working-directory convention

For reproducibility, execute code from the numbered directory that contains it, for example:

```bash
cd 3_model_fit
python model_fittingC6_improved.py
```

This is particularly important for `4_metabolic_network_reconstruction/muscle_specific_model.ipynb` and `build_subsystem_reaction_list.py`, which contain working-directory-relative paths.

## Data and model sources

The repository uses transcriptomic data from GEO accessions **GSE17576** and **GSE305719**, and the mouse genome-scale metabolic model **iMM1865**. See the scripts/notebooks and `1_dataset_transcriptomes/list_of_publications.txt` for source-publication details.

## License

This repository is distributed under the MIT License. See `LICENSE` for details.
