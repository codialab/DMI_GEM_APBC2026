"""DMI metabolic-flux fitting -- C6 oxidative/dilution branch.

This file is the reproducible, annotated implementation of the C6 model used for
the manuscript sensitivity branch. It is derived from the B6 KT-fixed model, but
it changes the biological backbone of the voxel fit.

Why C6 exists relative to B6
---------------------------
B6 fitted voxel-level ``MR_glc, f_lac, V_TCA, V_out`` and then derived
``V_ox = 2 * MR_glc * (1 - f_lac)``. That made every glucose label not assigned
to the lactate branch become oxidative label entry by definition. For skeletal
muscle this is too restrictive: label can be diluted by unlabeled Lac/Pyr pools,
cleared from the observed Lac/Pyr compartment, enter alanine/glycogen/glycerol
branches, or mix with non-glucose substrates before appearing in Glx.

C6 follows a Mathy-style dynamic DMI interpretation more closely: the fitted
oxidative entry ``V_ox`` is a DMI-visible glucose-label-derived flux from the
joint Lac/Pyr pool into the TCA-associated intermediate pool. Total effective
TCA/Glx turnover is not independently fitted; it is derived as
``V_TCA_total = V_ox / phi_tca``, where ``phi_tca`` is the glucose-derived
fraction of effective TCA/Glx turnover.

Fitted parameters
-----------------
Global per mouse-week: plasma input parameters
``a1, plasma_delay_min, plasma_tau_rise, plasma_w, kp_slow, kp_gap``.

Tissue-level per muscle: ``eta, V_x, fv``. The maximum transport capacity is
not fitted independently; for each voxel ``T_max = eta * MR_glc``.

Voxel-level: ``MR_glc, V_lac_loss, V_ox, phi_tca``.

Reported quantities
-------------------
Primary fitted outputs are ``MR_glc, V_lac_loss, V_ox, phi_tca``. Primary
derived outputs are ``T_max, V_lac_source, V_TCA_total, V_dil_tca``. Deprecated
aliases are retained only for compatibility: ``V_lac = V_lac_source = MR_glc``
and ``V_out = V_lac_loss``. They must not be interpreted as the old B6
``V_lac = 2 * MR_glc * f_lac`` or as pure lactate export.

FBA interpretation
------------------
``MR_glc`` and ``V_ox`` are the GEM-facing DMI constraints. ``V_lac_loss`` is a
low-confidence label-clearance/dilution term, not a hard lactate secretion flux.
``V_TCA_total, V_dil_tca, phi_tca`` are dilution/turnover diagnostics, not direct
DMI-measured GEM constraints.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import differential_evolution, minimize

try:
    OUTPUT_DIR = Path(__file__).resolve().parent
except NameError:
    OUTPUT_DIR = Path(os.getcwd())

# C6 lives in a subdirectory so outputs can be kept separate from B6. The
# extracted metabolite inputs and ROI helpers now live in ../2_extract_DMI_met.
PROJECT_DIR = OUTPUT_DIR.parent
EXTRACT_DIR = PROJECT_DIR / "2_extract_DMI_met"
DATA_DIR = OUTPUT_DIR
sys.path.insert(0, str(EXTRACT_DIR))
from lib_roi_masks import NX, NY, f_get_roi_masks

SCAN_DURATION_SEC = 280
SCAN_DURATION_MIN = SCAN_DURATION_SEC / 60.0
TISSUES = ["TibialisAnt", "Gastrocnemius", "Soleus"]
WEEKS = ["basal", "w3", "w6", "w9"]
MODEL_TAG = "branch_split_noGly_vC6_jointPlasma_oxDilution_KTfixed"
RESULTS_FILENAME_PHASE1 = (
    "fitC6a_results_MRI_fluxes_noGly_oxDilution_KTfixed.joblib.xz"
)
RESULTS_FILENAME = (
    "fitC6b_results_MRI_fluxes_AllTissues_noGly_oxDilution_KTfixed.joblib.xz"
)

# KT is now a fixed constant (literature value 5.8 mM for skeletal-muscle
# GLUT4-mediated glucose transport). It enters the ODE through ``fp["KT"]`` in
# exactly the same place as before; it just no longer participates in the fit.
KT_FIXED_MM = 5.8

FIXED_PARAMS = dict(
    f_gly=0.157,
    f_tca=0.385,
    Glc_p_total=8.0,
    Glc_cell=8.0,
    L=1.5,
    K=0.10,
    Glx=10.0,
    kie_gly=1.042,
    kie_tca=1.035,
    plasma_model="delayed_rise_biexponential",
    kp=0.020,
    KT=KT_FIXED_MM,
)

# Global plasma parameters (Phase 2, multi-tissue). Identical to v6.
N_GLOBAL_PARAMS = 6
BOUNDS_GLOBAL = [
    (10.0, 60.0),    # a1
    (0.0, 12.0),     # plasma_delay_min
    (0.2, 12.0),     # plasma_tau_rise
    (0.05, 0.95),    # plasma_w
    (0.001, 0.030),  # kp_slow
    (0.002, 0.110),  # kp_gap
]

# Tissue block has three free parameters. ``eta`` replaces independently fitted
# ``T_max``; each voxel uses ``T_max = eta * MR_glc``.
N_TISSUE_PARAMS = 3
BOUNDS_TISSUE = [
    (1.0, 15.0),    # eta, transport capacity multiplier: T_max = eta * MR_glc
    (0.01, 10.0),   # V_x
    (0.0, 0.15),    # fv (also denoted as f_v)
]

# Voxel block: fitted glucose label processing, Lac/Pyr label clearance,
# glucose-label-derived oxidative entry, and glucose-derived TCA fraction.
N_VOX_PARAMS = 4
BOUNDS_PER_VOX = [
    (0.001, 5.0),   # MR_glc
    (0.0, 5.0),     # V_lac_loss, apparent Lac* clearance/dilution
    (0.0, 2.0),     # V_ox, fitted glucose-label-derived oxidative entry
    (0.2, 1.0),     # phi_tca, glucose-derived fraction of TCA/Glx turnover
]

# Shared (Phase-1 single-tissue) layout: tissue block + 6 plasma + fv.
# Total 9, down from 10 in v6.
N_SHARED_PARAMS = 9
BOUNDS_SHARED = [
    (1.0, 15.0),    # eta
    (0.01, 10.0),   # V_x
    (10.0, 60.0),   # a1
    (0.0, 12.0),    # plasma_delay_min
    (0.2, 12.0),    # plasma_tau_rise
    (0.05, 0.95),   # plasma_w
    (0.001, 0.030), # kp_slow
    (0.002, 0.110), # kp_gap
    (0.0, 0.15),    # fv (also denoted as f_v)
]

# Initial values are intentionally explicit. ``eta`` is initialized from the old
# B6 scale of T_max/MR_glc, while voxel starts use direct C6 coordinates.
TISSUE_INIT = {
    "TibialisAnt": dict(
        eta=7.5,
        V_x=2.0,
        a1=35.0,
        plasma_delay_min=2.0,
        plasma_tau_rise=2.5,
        plasma_w=0.75,
        kp_fast=0.030,
        kp_slow=0.008,
        KT=KT_FIXED_MM,
        MR_glc=0.20,
        V_lac_loss=0.06,
        V_ox=0.18,
        phi_tca=0.50,
    ),
    "Gastrocnemius": dict(
        eta=10.0,
        V_x=2.0,
        a1=35.0,
        plasma_delay_min=2.0,
        plasma_tau_rise=2.5,
        plasma_w=0.75,
        kp_fast=0.030,
        kp_slow=0.008,
        KT=KT_FIXED_MM,
        MR_glc=0.15,
        V_lac_loss=0.05,
        V_ox=0.12,
        phi_tca=0.50,
    ),
    "Soleus": dict(
        eta=10.0,
        V_x=3.0,
        a1=35.0,
        plasma_delay_min=2.0,
        plasma_tau_rise=2.5,
        plasma_w=0.75,
        kp_fast=0.030,
        kp_slow=0.008,
        KT=KT_FIXED_MM,
        MR_glc=0.10,
        V_lac_loss=0.03,
        V_ox=0.08,
        phi_tca=0.50,
    ),
}

FIT_MET_KEYS = ("glc", "lac", "glx")
MET_MAP = {"Glucose": "glc", "Glutamate": "glx", "Lactate": "lac"}
MIN_SIGNAL_MM = 0.3
ROI_RADIUS = 1
PSEUDO_HUBER_DELTA = 1.5
NOISE_SIGMA_FLOOR = 0.05
LAC_LOSS_WEIGHT = 1.0
LAC_POSITIVE_POINT_WEIGHT = 1.5
LAC_TREND_WEIGHT = 0.20
GLC_POSITIVE_POINT_WEIGHT = 1.25
GLC_EARLY_POINT_WEIGHT = 1.5
GLC_TREND_WEIGHT = 0.75
GLX_POSITIVE_POINT_WEIGHT = 1.10
GLX_TREND_WEIGHT = 0.35
LAC_MIN_DYNAMIC_RANGE = 0.10
GLX_MIN_DYNAMIC_RANGE = 0.08
LAC_MIN_POSITIVE_POINTS = 2
GLX_MIN_POSITIVE_POINTS = 2
SECONDARY_CHANNEL_SKIP_THRESHOLD = 0.15
BOUND_TOL_FRACTION = 0.02
PLASMA_A1_PRIOR = 35.0
PLASMA_A1_REG_SCALE = 8.0
PLASMA_DELAY_PRIOR = 2.0
PLASMA_TAU_RISE_PRIOR = 2.5
PLASMA_W_PRIOR = 0.75
KP_FAST_PRIOR = 0.030
KP_SLOW_PRIOR = 0.008
KP_GAP_PRIOR = KP_FAST_PRIOR - KP_SLOW_PRIOR
PLASMA_REG_WEIGHT = 0.01
FV_PRIOR = 0.02
FV_REG_SCALE = 0.03
FV_REG_WEIGHT = 0.03
EPS_POOL = 1e-12
OX_FEASIBILITY_MODE = "disabled"
OX_FEASIBILITY_THRESHOLD = 2.0


def load_experiment_data():
    data_path = EXTRACT_DIR / "data_tissue_vals_model_fitting.joblib.xz"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing experiment data file: {data_path}")
    data = joblib.load(data_path)
    hash_mouseID_2_df_tissue_val = data[0]
    hash_mouseID2group = data[1]
    df_image_data = data[2]
    hash_mouseID2week2maskMap = data[3]
    hash_mouseID2week2H1anatomical = data[4]

    available_combos = (
        df_image_data[["mouseID", "week"]]
        .drop_duplicates()
        .sort_values(["mouseID", "week"])
        .values.tolist()
    )
    available_combos = [
        (mid, wk)
        for mid, wk in available_combos
        if mid in hash_mouseID2week2maskMap and wk in hash_mouseID2week2maskMap[mid]
    ]
    return {
        "data_path": data_path,
        "hash_mouseID_2_df_tissue_val": hash_mouseID_2_df_tissue_val,
        "hash_mouseID2group": hash_mouseID2group,
        "df_image_data": df_image_data,
        "hash_mouseID2week2maskMap": hash_mouseID2week2maskMap,
        "hash_mouseID2week2H1anatomical": hash_mouseID2week2H1anatomical,
        "available_combos": available_combos,
    }


DATA = load_experiment_data()
roi_masks_all = f_get_roi_masks()
hash_mouseID2group = DATA["hash_mouseID2group"]
df_image_data = DATA["df_image_data"]
hash_mouseID2week2maskMap = DATA["hash_mouseID2week2maskMap"]
available_combos = DATA["available_combos"]


def get_expanded_roi(tissue, week, mouseID, radius=ROI_RADIUS):
    seed_mask = roi_masks_all.get(tissue, {}).get(week, {}).get(mouseID, None)
    if seed_mask is None:
        return []

    seed_rows, seed_cols = np.where(seed_mask)
    if len(seed_rows) == 0:
        return []
    r0, c0 = int(seed_rows[0]), int(seed_cols[0])

    candidates = [
        (r0 + dr, c0 + dc)
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if 0 <= r0 + dr < NX and 0 <= c0 + dc < NY
    ]

    tissue_mask = hash_mouseID2week2maskMap.get(mouseID, {}).get(week, None)
    if tissue_mask is None:
        return candidates
    return [(r, c) for r, c in candidates if tissue_mask[r, c]]


def extract_voxel_timeseries(mouseID, week, roi_coords):
    sub = df_image_data[
        (df_image_data["mouseID"] == mouseID) & (df_image_data["week"] == week)
    ].copy()
    if sub.empty:
        return []

    sub_post = sub[sub["time"] >= 0].copy()
    voxel_data = []
    for row, col in roi_coords:
        vox = {"coords": (row, col)}
        valid = True
        time_ref = None

        for mri_name, key in MET_MAP.items():
            met_rows = sub_post[sub_post["mname"] == mri_name].sort_values("time")
            if met_rows.empty:
                valid = False
                break

            time_min = met_rows["time"].values.astype(float) / 60.0
            if time_ref is None:
                time_ref = time_min
            elif not np.allclose(time_min, time_ref):
                valid = False
                break

            imgs = np.stack(met_rows["cur_image"].values)
            vox[key] = imgs[:, row, col].astype(float)

        if not valid:
            continue

        vox["time_min"] = time_ref
        if np.max(np.abs(vox["glc"])) < MIN_SIGNAL_MM:
            continue
        voxel_data.append(vox)

    return voxel_data


def compute_tmax(eta, MR_glc):
    """Voxel transport capacity implied by the tissue-level multiplier."""

    return float(eta) * float(MR_glc)


def compute_derived_fluxes(MR_glc, V_lac_loss, V_ox, phi_tca, eta=None):
    """Return all C6 derived, deprecated, and diagnostic flux quantities.

    ``V_lac_source`` is the ODE source term from glucose label processing into
    the observed Lac/Pyr label pool. The old ``V_lac`` column is retained as a
    deprecated alias for this source term only; it is not the old branch-split
    quantity ``2 * MR_glc * f_lac``.
    """

    MR_glc = float(MR_glc)
    V_lac_loss = float(max(V_lac_loss, 0.0))
    V_ox = float(max(V_ox, 0.0))
    phi_tca = float(np.clip(phi_tca, BOUNDS_PER_VOX[3][0], BOUNDS_PER_VOX[3][1]))
    V_TCA_total = V_ox / max(phi_tca, EPS_POOL)
    V_dil_tca = max(V_TCA_total - V_ox, 0.0)
    ox_capacity = max(2.0 * MR_glc - V_lac_loss, EPS_POOL)
    ox_ratio = V_ox / ox_capacity
    derived = dict(
        V_lac_source=MR_glc,
        V_lac=MR_glc,  # deprecated compatibility alias
        V_out=V_lac_loss,  # deprecated compatibility alias
        V_TCA_total=V_TCA_total,
        V_dil_tca=V_dil_tca,
        V_ox_feasibility_capacity=ox_capacity,
        V_ox_feasibility_ratio=ox_ratio,
        V_ox_feasibility_flag=bool(ox_ratio > OX_FEASIBILITY_THRESHOLD),
    )
    if eta is not None:
        derived["T_max"] = compute_tmax(eta, MR_glc)
    return derived


def encode_plasma_decay_params(kp_fast, kp_slow):
    kp_slow = float(np.clip(kp_slow, BOUNDS_GLOBAL[4][0], BOUNDS_GLOBAL[4][1]))
    kp_fast = float(max(kp_fast, kp_slow + BOUNDS_GLOBAL[5][0]))
    kp_gap = np.clip(kp_fast - kp_slow, BOUNDS_GLOBAL[5][0], BOUNDS_GLOBAL[5][1])
    return kp_slow, float(kp_gap)


def decode_plasma_decay_params(kp_slow_param, kp_gap_param):
    kp_slow = float(np.clip(kp_slow_param, BOUNDS_GLOBAL[4][0], BOUNDS_GLOBAL[4][1]))
    kp_gap = float(np.clip(kp_gap_param, BOUNDS_GLOBAL[5][0], BOUNDS_GLOBAL[5][1]))
    kp_fast = min(kp_slow + kp_gap, 0.120)
    if kp_fast <= kp_slow:
        kp_fast = min(kp_slow + BOUNDS_GLOBAL[5][0], 0.120)
    return kp_fast, kp_slow



def unpack_voxel_params(theta, offset, eta=None):
    """Decode one C6 voxel block and attach reproducible derived quantities."""

    MR_glc = float(theta[offset])
    V_lac_loss = float(max(theta[offset + 1], 0.0))
    V_ox = float(max(theta[offset + 2], 0.0))
    phi_tca = float(np.clip(theta[offset + 3], BOUNDS_PER_VOX[3][0], BOUNDS_PER_VOX[3][1]))
    return dict(
        MR_glc=MR_glc,
        V_lac_loss=V_lac_loss,
        V_ox=V_ox,
        phi_tca=phi_tca,
        **compute_derived_fluxes(MR_glc, V_lac_loss, V_ox, phi_tca, eta=eta),
    )


def init_voxel_block(init):
    """Initial C6 voxel vector: [MR_glc, V_lac_loss, V_ox, phi_tca]."""

    return [
        init["MR_glc"],
        init["V_lac_loss"],
        init["V_ox"],
        init["phi_tca"],
    ]



def normalize_plasma_params(fp):
    plasma_w = float(np.clip(fp.get("plasma_w", 1.0), 0.0, 1.0))
    if "kp_gap" in fp:
        kp_fast, kp_slow = decode_plasma_decay_params(fp.get("kp_slow", 0.02), fp["kp_gap"])
    else:
        kp_fast = float(fp.get("kp_fast", fp.get("kp", 0.02)))
        kp_slow = float(fp.get("kp_slow", fp.get("kp", 0.02)))
        if kp_fast <= kp_slow:
            kp_fast = min(kp_slow + KP_GAP_PRIOR, 0.120)
    plasma_tau_rise = max(float(fp.get("plasma_tau_rise", 1.0)), 1e-6)
    if kp_fast < kp_slow:
        kp_fast, kp_slow = kp_slow, kp_fast
    return plasma_w, kp_fast, kp_slow, plasma_tau_rise



def plasma_input(t_min, fp):
    if t_min < 0:
        return 0.0

    a1 = float(fp.get("a1", 35.0))
    plasma_delay_min = max(float(fp.get("plasma_delay_min", 0.0)), 0.0)
    effective_t = t_min - plasma_delay_min
    if effective_t <= 0.0:
        return 0.0

    if fp.get("plasma_model", "delayed_rise_biexponential") == "single_exponential":
        kp = fp.get("kp", 0.02)
        return a1 * np.exp(-kp * effective_t)

    plasma_w, kp_fast, kp_slow, plasma_tau_rise = normalize_plasma_params(fp)
    rise_factor = 1.0 - np.exp(-effective_t / plasma_tau_rise)
    decay_factor = (
        plasma_w * np.exp(-kp_fast * effective_t)
        + (1.0 - plasma_w) * np.exp(-kp_slow * effective_t)
    )
    return a1 * rise_factor * decay_factor



def _pool_fraction(label_amount, pool_total):
    return label_amount / max(pool_total + label_amount, EPS_POOL)


def ode_rhs(y, t, eta, V_x, MR_glc, V_lac_loss, V_ox, phi_tca, fp):
    """C6 four-state labeled-pool ODE.

    The equations intentionally separate fitted glucose-label-derived oxidative
    entry (``V_ox``) from derived total effective TCA/Glx turnover
    (``V_TCA_total``). This keeps the DMI-visible constraint used for GEM work
    distinct from unlabeled substrate dilution.
    """

    Glc_star, L_star, K_star, Glx_star = y

    Glc_star = max(float(Glc_star), 0.0)
    L_star = max(float(L_star), 0.0)
    K_star = max(float(K_star), 0.0)
    Glx_star = max(float(Glx_star), 0.0)

    Glc_p_star = plasma_input(t, fp)

    Glc_c = fp["Glc_cell"]
    Glc_p = fp["Glc_p_total"]
    L = fp["L"]
    K = fp["K"]
    Glx = fp["Glx"]
    f_gly = fp["f_gly"]
    f_tca = fp["f_tca"]
    KT = fp["KT"]

    kie_gly = fp.get("kie_gly", 1.0)
    kie_tca = fp.get("kie_tca", 1.0)
    T_max = compute_tmax(eta, MR_glc)
    V_TCA_total = compute_derived_fluxes(MR_glc, V_lac_loss, V_ox, phi_tca)["V_TCA_total"]
    MR_glc_iso = MR_glc / kie_gly
    V_ox_iso = V_ox / kie_tca

    dGlc_star = (
        T_max * Glc_p_star / max(KT + Glc_p + Glc_p_star, EPS_POOL)
        - T_max * Glc_star / max(KT + Glc_c + Glc_star, EPS_POOL)
        - MR_glc_iso * _pool_fraction(Glc_star, Glc_c)
    )

    dL_star = (
        (1 - f_gly) * MR_glc_iso * _pool_fraction(Glc_star, Glc_c)
        - V_lac_loss * _pool_fraction(L_star, L)
        - V_ox_iso * _pool_fraction(L_star, L)
    )

    dK_star = (
        (1 - f_tca) * V_ox_iso * _pool_fraction(L_star, L)
        - (V_TCA_total + V_x) * _pool_fraction(K_star, K)
        + V_x * _pool_fraction(Glx_star, Glx)
    )

    dGlx_star = V_x * (_pool_fraction(K_star, K) - _pool_fraction(Glx_star, Glx))
    return [dGlc_star, dL_star, dK_star, dGlx_star]



def simulate_voxel(t_eval_min, eta, V_x, MR_glc, V_lac_loss, V_ox, phi_tca, fp):
    y0 = [0.0, 0.0, 0.0, 0.0]
    try:
        sol = odeint(
            ode_rhs,
            y0,
            t_eval_min,
            args=(eta, V_x, MR_glc, V_lac_loss, V_ox, phi_tca, fp),
            rtol=1e-5,
            atol=1e-7,
            mxstep=5000,
        )
    except Exception:
        return None

    sol[:, 0] = np.clip(sol[:, 0], 0.0, fp["Glc_cell"])
    sol[:, 1] = np.clip(sol[:, 1], 0.0, fp["L"])
    sol[:, 2] = np.clip(sol[:, 2], 0.0, fp["K"])
    sol[:, 3] = np.clip(sol[:, 3], 0.0, fp["Glx"])

    glc_cell_star = sol[:, 0]
    lac_star = sol[:, 1]
    glx_star = sol[:, 3]
    fv = fp.get("fv", 0.0)
    glc_p_star_t = np.array([plasma_input(ti, fp) for ti in t_eval_min])
    glc_pred = (1.0 - fv) * glc_cell_star + fv * glc_p_star_t
    return {"glc": glc_pred, "lac": lac_star, "glx": glx_star}



def simulate_voxel_window_averaged(
    t_eval_min,
    eta,
    V_x,
    MR_glc,
    V_lac_loss,
    V_ox,
    phi_tca,
    fp,
    scan_duration_min=SCAN_DURATION_MIN,
    n_window_samples=9,
):
    t_eval_min = np.asarray(t_eval_min, dtype=float)
    half_window = 0.5 * scan_duration_min
    window_grids = []
    for t_center in t_eval_min:
        if np.isclose(t_center, 0.0):
            window_grids.append(np.array([0.0]))
        else:
            start = max(0.0, t_center - half_window)
            stop = t_center + half_window
            window_grids.append(np.linspace(start, stop, n_window_samples))

    dense_grid = np.unique(np.concatenate(window_grids))
    pred_dense = simulate_voxel(dense_grid, eta, V_x, MR_glc, V_lac_loss, V_ox, phi_tca, fp)
    if pred_dense is None:
        return None

    pred_avg = {}
    for key in FIT_MET_KEYS:
        pred_avg[key] = np.array(
            [np.mean(np.interp(window_t, dense_grid, pred_dense[key])) for window_t in window_grids]
        )
    return pred_avg



def estimate_noise_sigma(values, sigma_floor=NOISE_SIGMA_FLOOR):
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return max(float(np.std(values)), sigma_floor)

    diffs = np.diff(values)
    diff_mad = np.median(np.abs(diffs - np.median(diffs)))
    if diff_mad > 0:
        sigma = diff_mad / (0.6745 * np.sqrt(2.0))
    else:
        sigma = np.std(values - np.mean(values))
    return max(float(sigma), sigma_floor)


def summarize_channel_quality(values, key):
    values = np.asarray(values, dtype=float)
    sigma = estimate_noise_sigma(values)
    dynamic_range = float(np.ptp(values)) if values.size else 0.0
    peak_abs = float(np.max(np.abs(values))) if values.size else 0.0
    positive_points = int(np.sum(values > 0.0))
    snr = dynamic_range / max(sigma, 1e-12)

    if key == "glc":
        reliability = 1.0 if peak_abs >= MIN_SIGNAL_MM else 0.0
    elif key == "lac":
        reliability = min(1.0, dynamic_range / LAC_MIN_DYNAMIC_RANGE)
        reliability *= min(1.0, positive_points / LAC_MIN_POSITIVE_POINTS)
        reliability *= min(1.0, snr / 3.0)
    else:
        reliability = min(1.0, dynamic_range / GLX_MIN_DYNAMIC_RANGE)
        reliability *= min(1.0, positive_points / GLX_MIN_POSITIVE_POINTS)
        reliability *= min(1.0, snr / 2.5)

    return {
        "sigma": float(sigma),
        "dynamic_range": dynamic_range,
        "peak_abs": peak_abs,
        "positive_points": positive_points,
        "snr": float(snr),
        "reliability": float(np.clip(reliability, 0.0, 1.0)),
    }



def pseudo_huber_loss(residuals, delta=PSEUDO_HUBER_DELTA):
    residuals = np.asarray(residuals, dtype=float)
    return delta ** 2 * (np.sqrt(1.0 + (residuals / delta) ** 2) - 1.0)



def weighted_channel_loss(obs, pred, key, time_min=None, reliability=1.0):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    sigma = estimate_noise_sigma(obs)
    standardized_resid = (pred - obs) / sigma
    point_weights = np.ones_like(standardized_resid)

    if key == "lac":
        point_weights[obs > 0.0] *= LAC_POSITIVE_POINT_WEIGHT
    elif key == "glc":
        point_weights[obs > 0.0] *= GLC_POSITIVE_POINT_WEIGHT
        if time_min is not None:
            time_min = np.asarray(time_min, dtype=float)
            early_mask = time_min <= 15.0
            point_weights[early_mask & (obs > 0.0)] *= GLC_EARLY_POINT_WEIGHT
    elif key == "glx":
        point_weights[obs > 0.0] *= GLX_POSITIVE_POINT_WEIGHT

    loss = np.mean(point_weights * pseudo_huber_loss(standardized_resid))
    if key == "glc" and obs.size >= 3:
        obs_diff = np.diff(obs)
        pred_diff = np.diff(pred)
        diff_sigma = max(estimate_noise_sigma(obs_diff), 0.5 * sigma)
        trend_resid = (pred_diff - obs_diff) / diff_sigma
        trend_weights = np.ones_like(trend_resid)
        trend_weights[obs_diff > 0.0] *= GLC_POSITIVE_POINT_WEIGHT
        loss += GLC_TREND_WEIGHT * np.mean(trend_weights * pseudo_huber_loss(trend_resid))
    elif key == "glx" and obs.size >= 3 and reliability > 0.35:
        obs_diff = np.diff(obs)
        pred_diff = np.diff(pred)
        diff_sigma = max(estimate_noise_sigma(obs_diff), 0.5 * sigma)
        trend_resid = (pred_diff - obs_diff) / diff_sigma
        trend_weights = np.ones_like(trend_resid)
        trend_weights[obs_diff > 0.0] *= GLX_POSITIVE_POINT_WEIGHT
        loss += reliability * GLX_TREND_WEIGHT * np.mean(
            trend_weights * pseudo_huber_loss(trend_resid)
        )
    elif key == "lac" and obs.size >= 3 and reliability > 0.50:
        obs_diff = np.diff(obs)
        pred_diff = np.diff(pred)
        diff_sigma = max(estimate_noise_sigma(obs_diff), 0.5 * sigma)
        trend_resid = (pred_diff - obs_diff) / diff_sigma
        loss += reliability * LAC_TREND_WEIGHT * np.mean(pseudo_huber_loss(trend_resid))
    return float(loss)



def unpack_shared_params(theta):
    """Decode the Phase-1 (single-tissue) parameter block.

    Layout (length ``N_SHARED_PARAMS = 9``):
    ``[eta, V_x, a1, plasma_delay_min, plasma_tau_rise, plasma_w,
       kp_slow, kp_gap, fv]``.

    KT is not present in this vector; it is supplied via ``FIXED_PARAMS["KT"]``.
    """
    (
        eta,
        V_x,
        a1,
        plasma_delay_min,
        plasma_tau_rise,
        plasma_w,
        kp_slow,
        kp_gap,
        fv,
    ) = theta[:N_SHARED_PARAMS]
    kp_fast, kp_slow = decode_plasma_decay_params(kp_slow, kp_gap)
    return dict(
        eta=float(eta),
        V_x=V_x,
        a1=a1,
        plasma_delay_min=plasma_delay_min,
        plasma_tau_rise=plasma_tau_rise,
        plasma_w=plasma_w,
        kp_fast=kp_fast,
        kp_slow=kp_slow,
        KT=KT_FIXED_MM,  # constant; reported for downstream schema compatibility
        fv=np.clip(fv, 0.0, 0.15),
    )


def weighted_mean_from_pairs(weighted_costs):
    total_weight = sum(weight for weight, _ in weighted_costs)
    if total_weight <= 0.0:
        return None
    return sum(weight * cost for weight, cost in weighted_costs) / total_weight


def is_near_bound(value, bounds, tol_fraction=BOUND_TOL_FRACTION):
    low, high = bounds
    span = high - low
    tol = tol_fraction * span
    return bool(value <= low + tol or value >= high - tol)


def collect_bound_hits(global_params, tissue_params_by_name, voxel_rows):
    hits = []
    global_bounds = {
        "a1": BOUNDS_GLOBAL[0],
        "plasma_delay_min": BOUNDS_GLOBAL[1],
        "plasma_tau_rise": BOUNDS_GLOBAL[2],
        "plasma_w": BOUNDS_GLOBAL[3],
        "kp_slow": BOUNDS_GLOBAL[4],
        "kp_fast": (BOUNDS_GLOBAL[4][0] + BOUNDS_GLOBAL[5][0], 0.120),
    }
    for name, bounds in global_bounds.items():
        if is_near_bound(global_params[name], bounds):
            hits.append(f"global:{name}")

    # KT is fixed and T_max is voxel-derived, so only eta, V_x, and fv are audited.
    tissue_bounds = {
        "eta": BOUNDS_TISSUE[0],
        "V_x": BOUNDS_TISSUE[1],
        "fv": BOUNDS_TISSUE[2],
    }
    for tissue_name, params in tissue_params_by_name.items():
        for name, bounds in tissue_bounds.items():
            if is_near_bound(params[name], bounds):
                hits.append(f"tissue:{tissue_name}:{name}")

    voxel_bounds = {
        "MR_glc": BOUNDS_PER_VOX[0],
        "V_lac_loss": BOUNDS_PER_VOX[1],
        "V_ox": BOUNDS_PER_VOX[2],
        "phi_tca": BOUNDS_PER_VOX[3],
    }
    for tissue_name, voxel in voxel_rows:
        coords = voxel["coords"]
        for name, bounds in voxel_bounds.items():
            if is_near_bound(voxel[name], bounds):
                hits.append(f"voxel:{tissue_name}:{coords}:{name}")
    return hits


def build_result_diagnostics(global_params, tissue_results, final_cost, success):
    voxel_rows = []
    metrics = {"glc": [], "lac": [], "glx": []}
    negative_counts = {"glc": 0, "lac": 0, "glx": 0}
    tissue_params = {}

    for tissue_name, tissue_data in tissue_results.items():
        tissue_params[tissue_name] = {
            "eta": tissue_data["eta"],
            "V_x": tissue_data["V_x"],
            "fv": tissue_data["fv"],
        }
        for voxel in tissue_data["voxels"]:
            voxel_rows.append((tissue_name, voxel))
            for key in FIT_MET_KEYS:
                value = float(voxel[f"r2_{key}"])
                metrics[key].append(value)
                if value < 0.0:
                    negative_counts[key] += 1

    mean_r2_by_met = {
        key: (float(np.mean(values)) if values else float("nan"))
        for key, values in metrics.items()
    }
    frac_negative_r2_by_met = {
        key: (negative_counts[key] / len(metrics[key]) if metrics[key] else float("nan"))
        for key in FIT_MET_KEYS
    }
    bound_hits = collect_bound_hits(global_params, tissue_params, voxel_rows)
    include_in_summary = bool(success)

    return {
        "include_in_summary": include_in_summary,
        "mean_r2_by_met": mean_r2_by_met,
        "frac_negative_r2_by_met": frac_negative_r2_by_met,
        "bound_hit_count": len(bound_hits),
        "bound_hits": bound_hits,
        "n_tissues": len(tissue_results),
        "n_voxels": len(voxel_rows),
        "final_cost": float(final_cost),
    }



def plasma_regularization(a1, plasma_delay_min, plasma_tau_rise, plasma_w, kp_fast, kp_slow, fv=None):
    reg = PLASMA_REG_WEIGHT * (
        ((a1 - PLASMA_A1_PRIOR) / PLASMA_A1_REG_SCALE) ** 2
        + ((plasma_delay_min - PLASMA_DELAY_PRIOR) / 3.0) ** 2
        + ((plasma_tau_rise - PLASMA_TAU_RISE_PRIOR) / 4.0) ** 2
        + ((plasma_w - PLASMA_W_PRIOR) / 0.20) ** 2
        + ((kp_fast - KP_FAST_PRIOR) / 0.015) ** 2
        + ((kp_slow - KP_SLOW_PRIOR) / 0.006) ** 2
    )
    if fv is not None:
        reg += FV_REG_WEIGHT * ((float(fv) - FV_PRIOR) / FV_REG_SCALE) ** 2
    return reg



def joint_objective(theta, voxel_data_list, fixed_base, n_shared=N_SHARED_PARAMS):
    shared = unpack_shared_params(theta)
    fp = {
        **fixed_base,
        "a1": shared["a1"],
        "plasma_delay_min": shared["plasma_delay_min"],
        "plasma_tau_rise": shared["plasma_tau_rise"],
        "plasma_w": shared["plasma_w"],
        "kp_fast": shared["kp_fast"],
        "kp_slow": shared["kp_slow"],
        # KT comes from fixed_base unchanged; not overridden here.
        "fv": shared["fv"],
    }
    metabolite_costs = {key: [] for key in FIT_MET_KEYS}
    for voxel_idx, vox in enumerate(voxel_data_list):
        offset = n_shared + N_VOX_PARAMS * voxel_idx
        vp = unpack_voxel_params(theta, offset, eta=shared["eta"])
        pred = simulate_voxel_window_averaged(
            vox["time_min"],
            shared["eta"],
            shared["V_x"],
            vp["MR_glc"],
            vp["V_lac_loss"],
            vp["V_ox"],
            vp["phi_tca"],
            fp,
        )
        if pred is None:
            return 1e12
        for key in FIT_MET_KEYS:
            quality = summarize_channel_quality(vox[key], key)
            if key != "glc" and quality["reliability"] < SECONDARY_CHANNEL_SKIP_THRESHOLD:
                continue
            metabolite_costs[key].append(
                (
                    max(quality["reliability"], 1e-6),
                    weighted_channel_loss(
                        vox[key],
                        pred[key],
                        key,
                        time_min=vox["time_min"],
                        reliability=quality["reliability"],
                    ),
                )
            )

    if len(metabolite_costs["glc"]) == 0:
        return 1e12

    channel_weights = {"glc": 1.0, "lac": LAC_LOSS_WEIGHT, "glx": 1.0}
    present_keys = [key for key in FIT_MET_KEYS if metabolite_costs[key]]
    total_cost = sum(
        channel_weights[key] * weighted_mean_from_pairs(metabolite_costs[key])
        for key in present_keys
    ) / sum(channel_weights[key] for key in present_keys)
    total_cost += plasma_regularization(
        shared["a1"],
        shared["plasma_delay_min"],
        shared["plasma_tau_rise"],
        shared["plasma_w"],
        shared["kp_fast"],
        shared["kp_slow"],
        fv=shared["fv"],
    )
    return float(total_cost)



def build_bounds(n_voxels):
    return BOUNDS_SHARED + BOUNDS_PER_VOX * n_voxels



def build_x0(tissue, n_voxels):
    """Build a Phase-1 initial-guess vector matching ``BOUNDS_SHARED`` (length 9)
    plus ``N_VOX_PARAMS`` per voxel. KT is **not** packed here.
    """
    init = TISSUE_INIT.get(tissue, TISSUE_INIT["Gastrocnemius"])
    kp_slow_init, kp_gap_init = encode_plasma_decay_params(init["kp_fast"], init["kp_slow"])
    return np.array(
        [
            init["eta"],
            init["V_x"],
            init["a1"],
            init["plasma_delay_min"],
            init["plasma_tau_rise"],
            init["plasma_w"],
            kp_slow_init,
            kp_gap_init,
            FV_PRIOR,
        ]
        + init_voxel_block(init) * n_voxels
    )



def fit_sample(mouseID, week, tissue, radius=ROI_RADIUS, verbose=True):
    roi_coords = get_expanded_roi(tissue, week, mouseID, radius=radius)
    if not roi_coords:
        return None

    voxel_data = extract_voxel_timeseries(mouseID, week, roi_coords)
    if not voxel_data:
        return None

    n_voxels = len(voxel_data)
    bounds = build_bounds(n_voxels)
    fixed_base = dict(FIXED_PARAMS)

    if verbose:
        grp = hash_mouseID2group.get(mouseID, "?")
        print(
            f"  Fitting Mouse {mouseID} ({grp}) | {week} | {tissue} | {n_voxels} voxels ...",
            end="",
            flush=True,
        )

    de_result = differential_evolution(
        joint_objective,
        bounds=bounds,
        args=(voxel_data, fixed_base),
        maxiter=150,
        popsize=8,
        tol=1e-5,
        seed=42,
        workers=1,
        polish=False,
        init="latinhypercube",
        x0=build_x0(tissue, n_voxels),
    )
    lbfgs_result = minimize(
        joint_objective,
        x0=de_result.x,
        args=(voxel_data, fixed_base),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-6}
    )
    if not lbfgs_result.success:
        rng = np.random.default_rng(seed=99)
        x0_perturbed = np.clip(
            lbfgs_result.x + rng.normal(0, 1e-3, size=lbfgs_result.x.shape),
            [b[0] for b in bounds], [b[1] for b in bounds],
        )
        retry = minimize(
            joint_objective, x0=x0_perturbed,
            args=(voxel_data, fixed_base),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-6},
        )
        if retry.fun < lbfgs_result.fun:
            lbfgs_result = retry

    theta = lbfgs_result.x
    shared = unpack_shared_params(theta)
    fp_final = {
        **FIXED_PARAMS,
        "a1": shared["a1"],
        "plasma_delay_min": shared["plasma_delay_min"],
        "plasma_tau_rise": shared["plasma_tau_rise"],
        "plasma_w": shared["plasma_w"],
        "kp_fast": shared["kp_fast"],
        "kp_slow": shared["kp_slow"],
        # KT is taken from FIXED_PARAMS unchanged.
        "fv": shared["fv"],
    }

    vox_results = []
    for voxel_idx, vox in enumerate(voxel_data):
        offset = N_SHARED_PARAMS + N_VOX_PARAMS * voxel_idx
        vp = unpack_voxel_params(theta, offset, eta=shared["eta"])
        pred = simulate_voxel_window_averaged(
            vox["time_min"],
            shared["eta"],
            shared["V_x"],
            vp["MR_glc"],
            vp["V_lac_loss"],
            vp["V_ox"],
            vp["phi_tca"],
            fp_final,
        )
        r2 = {}
        for key in FIT_MET_KEYS:
            obs = vox[key]
            model = pred[key] if pred is not None else np.zeros_like(obs)
            ss_res = np.sum((obs - model) ** 2)
            ss_tot = np.sum((obs - obs.mean()) ** 2)
            r2[key] = 1 - ss_res / (ss_tot + 1e-12)
        vox_results.append(
            dict(
                coords=vox["coords"],
                **vp,
                fv=shared["fv"],
                r2_glc=r2["glc"],
                r2_lac=r2["lac"],
                r2_glx=r2["glx"],
                r2_mean=(r2["glc"] + r2["lac"] + r2["glx"]) / 3.0,
                voxel_data=vox,
                pred=pred,
            )
        )

    mean_r2 = np.mean([vr["r2_mean"] for vr in vox_results])
    quality_ok = mean_r2 > 0.5 and lbfgs_result.fun < 3.0
    include_in_summary = bool(lbfgs_result.success) or quality_ok
    if verbose:
        print(
            f"  done. mean R²={mean_r2:.3f} cost={lbfgs_result.fun:.2f} "
            f"eta={shared['eta']:.2f} KT(fixed)={KT_FIXED_MM:.2f} fv={shared['fv']:.3f} "
            f"a1={shared['a1']:.2f} delay={shared['plasma_delay_min']:.2f} min "
            f"tau_rise={shared['plasma_tau_rise']:.2f} min w={shared['plasma_w']:.2f} "
            f"kp_fast={shared['kp_fast']:.3f} kp_slow={shared['kp_slow']:.3f}"
        )

    return dict(
        model_version=MODEL_TAG,
        mouseID=mouseID,
        week=week,
        tissue=tissue,
        group=hash_mouseID2group.get(mouseID, "unknown"),
        eta=shared["eta"],
        V_x=shared["V_x"],
        KT=KT_FIXED_MM,  # constant; reported for schema compatibility
        fv=shared["fv"],
        a1=shared["a1"],
        plasma_model=FIXED_PARAMS["plasma_model"],
        plasma_delay_min=shared["plasma_delay_min"],
        plasma_tau_rise=shared["plasma_tau_rise"],
        plasma_w=shared["plasma_w"],
        kp_fast=shared["kp_fast"],
        kp_slow=shared["kp_slow"],
        voxels=vox_results,
        final_cost=lbfgs_result.fun,
        success=lbfgs_result.success,
        optimizer_message=lbfgs_result.message,
        diagnostics=build_result_diagnostics(
            shared,
            {
                tissue: {
                    "eta": shared["eta"],
                    "V_x": shared["V_x"],
                    "fv": shared["fv"],
                    "voxels": vox_results,
                }
                for tissue in [tissue]
            },
            lbfgs_result.fun,
            lbfgs_result.success,
        ),
        include_in_summary=include_in_summary,
    )



def compute_tissue_offset(tissue_idx, n_voxels_per_tissue):
    offset = N_GLOBAL_PARAMS
    for idx in range(tissue_idx):
        offset += N_TISSUE_PARAMS + N_VOX_PARAMS * n_voxels_per_tissue[idx]
    return offset



def unpack_global_params(theta):
    a1, plasma_delay_min, plasma_tau_rise, plasma_w, kp_slow, kp_gap = theta[:N_GLOBAL_PARAMS]
    kp_fast, kp_slow = decode_plasma_decay_params(kp_slow, kp_gap)
    return dict(
        a1=a1,
        plasma_delay_min=plasma_delay_min,
        plasma_tau_rise=plasma_tau_rise,
        plasma_w=plasma_w,
        kp_fast=kp_fast,
        kp_slow=kp_slow,
    )



def unpack_tissue_params(theta, tissue_offset):
    """Decode one tissue block: ``[eta, V_x, fv]``.

    ``eta`` is shared across voxels within the muscle and converts each voxel's
    fitted ``MR_glc`` into its own ``T_max``.
    """
    eta, V_x, fv = theta[tissue_offset : tissue_offset + N_TISSUE_PARAMS]
    return dict(eta=float(eta), V_x=V_x, fv=np.clip(fv, 0.0, 0.15))



def build_bounds_multi(n_voxels_per_tissue):
    bounds = list(BOUNDS_GLOBAL)
    for n_voxels in n_voxels_per_tissue:
        bounds += BOUNDS_TISSUE + BOUNDS_PER_VOX * n_voxels
    return bounds



def build_x0_multi(tissues, n_voxels_per_tissue, warm_start=None):
    if warm_start is not None:
        x0 = list(warm_start["global"])
        for tissue_idx, (_, n_voxels) in enumerate(zip(tissues, n_voxels_per_tissue)):
            x0 += list(warm_start["tissue"][tissue_idx])
            for vox_x0 in warm_start["voxels"][tissue_idx][:n_voxels]:
                x0 += list(vox_x0)
        return np.array(x0)

    kp_slow_init, kp_gap_init = encode_plasma_decay_params(KP_FAST_PRIOR, KP_SLOW_PRIOR)
    x0 = [
        PLASMA_A1_PRIOR,
        PLASMA_DELAY_PRIOR,
        PLASMA_TAU_RISE_PRIOR,
        PLASMA_W_PRIOR,
        kp_slow_init,
        kp_gap_init,
    ]
    for tissue, n_voxels in zip(tissues, n_voxels_per_tissue):
        init = TISSUE_INIT.get(tissue, TISSUE_INIT["Gastrocnemius"])
        # Tissue block: eta, V_x, fv (no KT and no independent T_max).
        x0 += [init["eta"], init["V_x"], FV_PRIOR]
        x0 += init_voxel_block(init) * n_voxels
    return np.array(x0)



def joint_objective_multi(theta, tissues_voxel_data, fixed_base):
    n_voxels_per_tissue = [len(vd) for _, vd in tissues_voxel_data]
    glob = unpack_global_params(theta)
    fp_base = {
        **fixed_base,
        "a1": glob["a1"],
        "plasma_delay_min": glob["plasma_delay_min"],
        "plasma_tau_rise": glob["plasma_tau_rise"],
        "plasma_w": glob["plasma_w"],
        "kp_fast": glob["kp_fast"],
        "kp_slow": glob["kp_slow"],
    }

    all_costs = {key: [] for key in FIT_MET_KEYS}
    for tissue_idx, (tissue_name, voxel_data_list) in enumerate(tissues_voxel_data):
        tissue_offset = compute_tissue_offset(tissue_idx, n_voxels_per_tissue)
        tissue_params = unpack_tissue_params(theta, tissue_offset)
        # KT is inherited from fp_base (== FIXED_PARAMS["KT"]); only fv is tissue-specific.
        fp_tissue = {**fp_base, "fv": tissue_params["fv"]}

        for voxel_idx, vox in enumerate(voxel_data_list):
            voxel_offset = tissue_offset + N_TISSUE_PARAMS + N_VOX_PARAMS * voxel_idx
            vp = unpack_voxel_params(theta, voxel_offset, eta=tissue_params["eta"])
            pred = simulate_voxel_window_averaged(
                vox["time_min"],
                tissue_params["eta"],
                tissue_params["V_x"],
                vp["MR_glc"],
                vp["V_lac_loss"],
                vp["V_ox"],
                vp["phi_tca"],
                fp_tissue,
            )
            if pred is None:
                return 1e12
            for key in FIT_MET_KEYS:
                quality = summarize_channel_quality(vox[key], key)
                if key != "glc" and quality["reliability"] < SECONDARY_CHANNEL_SKIP_THRESHOLD:
                    continue
                all_costs[key].append(
                    (
                        max(quality["reliability"], 1e-6),
                        weighted_channel_loss(
                            vox[key],
                            pred[key],
                            key,
                            time_min=vox["time_min"],
                            reliability=quality["reliability"],
                        ),
                    )
                )

    if len(all_costs["glc"]) == 0:
        return 1e12

    channel_weights = {"glc": 1.0, "lac": LAC_LOSS_WEIGHT, "glx": 1.0}
    present_keys = [key for key in FIT_MET_KEYS if all_costs[key]]
    total_cost = sum(
        channel_weights[key] * weighted_mean_from_pairs(all_costs[key])
        for key in present_keys
    ) / sum(channel_weights[key] for key in present_keys)
    total_cost += plasma_regularization(
        glob["a1"],
        glob["plasma_delay_min"],
        glob["plasma_tau_rise"],
        glob["plasma_w"],
        glob["kp_fast"],
        glob["kp_slow"],
    )
    total_cost += FV_REG_WEIGHT * sum(
        (
            (
                unpack_tissue_params(theta, compute_tissue_offset(tissue_idx, n_voxels_per_tissue))["fv"]
                - FV_PRIOR
            )
            / FV_REG_SCALE
        )
        ** 2
        for tissue_idx in range(len(tissues_voxel_data))
    )
    return float(total_cost)



def build_warm_start_from_phase1(mouseID, week, phase1_results, tissues_voxel_data):
    res_map = {
        res["tissue"]: res
        for res in phase1_results
        if res["mouseID"] == mouseID and res["week"] == week
    }
    if not res_map:
        return None

    plasma_keys = [
        "a1",
        "plasma_delay_min",
        "plasma_tau_rise",
        "plasma_w",
        "kp_fast",
        "kp_slow",
    ]
    plasma_vals = np.mean([[res_map[tissue][key] for key in plasma_keys] for tissue in res_map], axis=0)
    kp_slow_init, kp_gap_init = encode_plasma_decay_params(plasma_vals[4], plasma_vals[5])
    x0_global = [
        plasma_vals[0],
        plasma_vals[1],
        plasma_vals[2],
        plasma_vals[3],
        kp_slow_init,
        kp_gap_init,
    ]

    x0_tissue = []
    x0_voxels = []
    for tissue_name, voxel_data_list in tissues_voxel_data:
        tissue_result = res_map.get(tissue_name)
        if tissue_result is not None:
            # KT and independent T_max are not packed; tissue block is [eta, V_x, fv].
            x0_tissue.append(
                [
                    tissue_result["eta"],
                    tissue_result["V_x"],
                    tissue_result.get("fv", FV_PRIOR),
                ]
            )
            vox_block = []
            for voxel_idx, _ in enumerate(voxel_data_list):
                if voxel_idx < len(tissue_result["voxels"]):
                    rv = tissue_result["voxels"][voxel_idx]
                    vox_block.append([rv["MR_glc"], rv["V_lac_loss"], rv["V_ox"], rv["phi_tca"]])
                else:
                    init = TISSUE_INIT.get(tissue_name, TISSUE_INIT["Gastrocnemius"])
                    vox_block.append(init_voxel_block(init))
            x0_voxels.append(vox_block)
        else:
            init = TISSUE_INIT.get(tissue_name, TISSUE_INIT["Gastrocnemius"])
            x0_tissue.append([init["eta"], init["V_x"], FV_PRIOR])
            x0_voxels.append(
                [init_voxel_block(init) for _ in voxel_data_list]
            )

    return {"global": x0_global, "tissue": x0_tissue, "voxels": x0_voxels}



def fit_sample_multi(mouseID, week, x0=None, de_maxiter=50, verbose=True):
    tissues_voxel_data = []
    for tissue in TISSUES:
        roi_coords = get_expanded_roi(tissue, week, mouseID, radius=ROI_RADIUS)
        if not roi_coords:
            continue
        voxel_data = extract_voxel_timeseries(mouseID, week, roi_coords)
        if voxel_data:
            tissues_voxel_data.append((tissue, voxel_data))

    if not tissues_voxel_data:
        return None

    tissue_names = [tissue for tissue, _ in tissues_voxel_data]
    n_voxels_per_tissue = [len(vd) for _, vd in tissues_voxel_data]
    bounds = build_bounds_multi(n_voxels_per_tissue)
    fixed_base = dict(FIXED_PARAMS)
    x0_vec = build_x0_multi(tissue_names, n_voxels_per_tissue, warm_start=x0)

    if verbose:
        grp = hash_mouseID2group.get(mouseID, "?")
        tissue_summary = "+".join(
            f"{tissue}({n_voxels})" for tissue, n_voxels in zip(tissue_names, n_voxels_per_tissue)
        )
        print(
            f"  [Phase 2] Mouse {mouseID} ({grp}) | {week} | {tissue_summary} ...",
            end="",
            flush=True,
        )

    de_result = differential_evolution(
        joint_objective_multi,
        bounds=bounds,
        args=(tissues_voxel_data, fixed_base),
        maxiter=de_maxiter,
        popsize=8,
        tol=1e-5,
        seed=42,
        workers=1,
        polish=False,
        init="latinhypercube",
        x0=x0_vec,
    )
    lbfgs_result = minimize(
        joint_objective_multi,
        x0=de_result.x,
        args=(tissues_voxel_data, fixed_base),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-6},
    )
    if not lbfgs_result.success:
        rng = np.random.default_rng(seed=99)
        x0_perturbed = np.clip(
            lbfgs_result.x + rng.normal(0, 1e-3, size=lbfgs_result.x.shape),
            [b[0] for b in bounds], [b[1] for b in bounds],
        )
        retry = minimize(
            joint_objective_multi, x0=x0_perturbed,
            args=(tissues_voxel_data, fixed_base),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-6},
        )
        if retry.fun < lbfgs_result.fun:
            lbfgs_result = retry

    theta = lbfgs_result.x
    glob = unpack_global_params(theta)
    fp_global = {
        **FIXED_PARAMS,
        "a1": glob["a1"],
        "plasma_delay_min": glob["plasma_delay_min"],
        "plasma_tau_rise": glob["plasma_tau_rise"],
        "plasma_w": glob["plasma_w"],
        "kp_fast": glob["kp_fast"],
        "kp_slow": glob["kp_slow"],
    }

    tissue_results = {}
    for tissue_idx, (tissue_name, voxel_data_list) in enumerate(tissues_voxel_data):
        tissue_offset = compute_tissue_offset(tissue_idx, n_voxels_per_tissue)
        tissue_params = unpack_tissue_params(theta, tissue_offset)
        fp_tissue = {**fp_global, "fv": tissue_params["fv"]}
        vox_results = []
        for voxel_idx, vox in enumerate(voxel_data_list):
            voxel_offset = tissue_offset + N_TISSUE_PARAMS + N_VOX_PARAMS * voxel_idx
            vp = unpack_voxel_params(theta, voxel_offset, eta=tissue_params["eta"])
            pred = simulate_voxel_window_averaged(
                vox["time_min"],
                tissue_params["eta"],
                tissue_params["V_x"],
                vp["MR_glc"],
                vp["V_lac_loss"],
                vp["V_ox"],
                vp["phi_tca"],
                fp_tissue,
            )
            r2 = {}
            for key in FIT_MET_KEYS:
                obs = vox[key]
                model = pred[key] if pred is not None else np.zeros_like(obs)
                ss_res = np.sum((obs - model) ** 2)
                ss_tot = np.sum((obs - obs.mean()) ** 2)
                r2[key] = 1 - ss_res / (ss_tot + 1e-12)
            vox_results.append(
                dict(
                    coords=vox["coords"],
                    **vp,
                    fv=tissue_params["fv"],
                    r2_glc=r2["glc"],
                    r2_lac=r2["lac"],
                    r2_glx=r2["glx"],
                    r2_mean=(r2["glc"] + r2["lac"] + r2["glx"]) / 3.0,
                    voxel_data=vox,
                    pred=pred,
                )
            )
        tissue_results[tissue_name] = dict(
            eta=tissue_params["eta"],
            V_x=tissue_params["V_x"],
            KT=KT_FIXED_MM,  # constant; reported for schema compatibility
            fv=tissue_params["fv"],
            voxels=vox_results,
        )

    all_r2 = [vr["r2_mean"] for td in tissue_results.values() for vr in td["voxels"]]
    mean_r2 = np.mean(all_r2)
    quality_ok = mean_r2 > 0.5 and lbfgs_result.fun < 3.0
    include_in_summary = bool(lbfgs_result.success) or quality_ok
    if verbose:
        all_r2 = [vr["r2_mean"] for tissue_data in tissue_results.values() for vr in tissue_data["voxels"]]
        print(
            f"  done. mean R²={np.mean(all_r2):.3f} cost={lbfgs_result.fun:.2f} "
            f"a1={glob['a1']:.2f} delay={glob['plasma_delay_min']:.2f} min "
            f"tau_rise={glob['plasma_tau_rise']:.2f} min w={glob['plasma_w']:.2f} "
            f"kp_fast={glob['kp_fast']:.3f} kp_slow={glob['kp_slow']:.3f} "
            f"KT(fixed)={KT_FIXED_MM:.2f}"
        )

    return dict(
        model_version=f"{MODEL_TAG}_allTissues",
        mouseID=mouseID,
        week=week,
        group=hash_mouseID2group.get(mouseID, "unknown"),
        a1=glob["a1"],
        plasma_delay_min=glob["plasma_delay_min"],
        plasma_tau_rise=glob["plasma_tau_rise"],
        plasma_w=glob["plasma_w"],
        kp_fast=glob["kp_fast"],
        kp_slow=glob["kp_slow"],
        plasma_model=FIXED_PARAMS["plasma_model"],
        tissues=tissue_results,
        final_cost=lbfgs_result.fun,
        success=lbfgs_result.success,
        optimizer_message=lbfgs_result.message,
        diagnostics=build_result_diagnostics(glob, tissue_results, lbfgs_result.fun, lbfgs_result.success),
        include_in_summary=include_in_summary,
    )



def run_phase2_task(mouseID, week, warm_start, de_maxiter=100):
    return fit_sample_multi(mouseID, week, x0=warm_start, de_maxiter=de_maxiter, verbose=False)



def run_light_sanity_checks():
    x0_check = build_x0("Gastrocnemius", 2)
    bounds_check = build_bounds(2)
    assert N_SHARED_PARAMS == 9
    assert len(BOUNDS_SHARED) == N_SHARED_PARAMS
    assert N_TISSUE_PARAMS == 3
    assert len(BOUNDS_TISSUE) == N_TISSUE_PARAMS
    assert len(x0_check) == len(bounds_check)
    shared_check = unpack_shared_params(x0_check)
    assert 0.0 <= shared_check["fv"] <= 0.15
    assert BOUNDS_TISSUE[0][0] <= shared_check["eta"] <= BOUNDS_TISSUE[0][1]
    assert shared_check["KT"] == KT_FIXED_MM
    voxel_check = unpack_voxel_params(x0_check, N_SHARED_PARAMS, eta=shared_check["eta"])
    assert np.isclose(voxel_check["T_max"], shared_check["eta"] * voxel_check["MR_glc"])
    assert BOUNDS_PER_VOX[3][0] <= voxel_check["phi_tca"] <= BOUNDS_PER_VOX[3][1]
    assert voxel_check["V_TCA_total"] >= voxel_check["V_ox"]
    assert voxel_check["V_dil_tca"] >= 0.0

    diag_roi = get_expanded_roi("Gastrocnemius", "w6", 54)
    diag_vox = extract_voxel_timeseries(54, "w6", diag_roi)
    if diag_vox:
        diag_theta = build_x0("Gastrocnemius", len(diag_vox))
        diag_cost = joint_objective(diag_theta, diag_vox, dict(FIXED_PARAMS))
        print(
            f"Sanity checks passed. Mouse 54 w6 Gastrocnemius initial objective={diag_cost:.3f} "
            f"(KT fixed at {KT_FIXED_MM} mM)"
        )
    else:
        print(
            f"Sanity checks passed (KT fixed at {KT_FIXED_MM} mM). "
            "Diagnostic voxel set for Mouse 54 w6 Gastrocnemius is empty."
        )



def build_phase2_warm_starts(tasks_p2, phase1_results):
    warm_starts = {}
    for mouseID, week in tasks_p2:
        tissues_voxel_data = []
        for tissue in TISSUES:
            roi_coords = get_expanded_roi(tissue, week, mouseID, radius=ROI_RADIUS)
            if not roi_coords:
                continue
            voxel_data = extract_voxel_timeseries(mouseID, week, roi_coords)
            if voxel_data:
                tissues_voxel_data.append((tissue, voxel_data))
        warm_starts[(mouseID, week)] = build_warm_start_from_phase1(
            mouseID, week, phase1_results, tissues_voxel_data
        )
    return warm_starts



def run_two_phase_fitting(max_workers=None, phase2_de_maxiter=50):
    results_path_p1 = DATA_DIR / RESULTS_FILENAME_PHASE1
    results_path_p2 = DATA_DIR / RESULTS_FILENAME
    mp_context = get_context("fork" if sys.platform.startswith("linux") else "spawn")

    if results_path_p1.exists():
        phase1_results = joblib.load(results_path_p1)
        print(f"Loaded {len(phase1_results)} Phase-1 results from {results_path_p1}")
    else:
        phase1_results = []

    done_keys_p1 = {(res["mouseID"], res["week"], res["tissue"]) for res in phase1_results}
    tasks_p1 = [
        (mouseID, week, tissue)
        for mouseID, week in available_combos
        for tissue in TISSUES
        if (mouseID, week, tissue) not in done_keys_p1
    ]

    worker_cap = max_workers or (os.cpu_count() or 4)
    n_workers_p1 = max(1, min(worker_cap, len(tasks_p1) or 1))
    print(f"Phase 1: {len(done_keys_p1)} done, {len(tasks_p1)} remaining -- {n_workers_p1} workers.")

    if tasks_p1:
        total_p1 = len(done_keys_p1) + len(tasks_p1)
        with ProcessPoolExecutor(max_workers=n_workers_p1, mp_context=mp_context) as executor:
            future_to_key = {
                executor.submit(fit_sample, mouseID, week, tissue, ROI_RADIUS, False): (mouseID, week, tissue)
                for mouseID, week, tissue in tasks_p1
            }
            for future in as_completed(future_to_key):
                mouseID, week, tissue = future_to_key[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"  ERROR Mouse {mouseID} {week} {tissue}: {exc}")
                    result = None
                if result is not None:
                    phase1_results.append(result)
                    done_keys_p1.add((mouseID, week, tissue))
                joblib.dump(phase1_results, results_path_p1, compress=("xz", 3))
                print(f"  [P1 {len(done_keys_p1)}/{total_p1}] Mouse {mouseID} {week} {tissue}")

    print(f"Phase 1 complete -- {len(phase1_results)} tissue fits.")

    if results_path_p2.exists():
        all_results = joblib.load(results_path_p2)
        print(f"Loaded {len(all_results)} Phase-2 results from {results_path_p2}")
    else:
        all_results = []

    done_keys_p2 = {(res["mouseID"], res["week"]) for res in all_results}
    tasks_p2 = [(mouseID, week) for mouseID, week in available_combos if (mouseID, week) not in done_keys_p2]
    n_workers_p2 = max(1, min(worker_cap, len(tasks_p2) or 1))
    print(f"Phase 2: {len(done_keys_p2)} done, {len(tasks_p2)} remaining -- {n_workers_p2} workers.")

    if tasks_p2:
        total_p2 = len(done_keys_p2) + len(tasks_p2)
        warm_starts = build_phase2_warm_starts(tasks_p2, phase1_results)
        with ProcessPoolExecutor(max_workers=n_workers_p2, mp_context=mp_context) as executor:
            future_to_key = {
                executor.submit(run_phase2_task, mouseID, week, warm_starts[(mouseID, week)], phase2_de_maxiter):
                (mouseID, week)
                for mouseID, week in tasks_p2
            }
            for future in as_completed(future_to_key):
                mouseID, week = future_to_key[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"  ERROR Mouse {mouseID} {week}: {exc}")
                    result = None
                if result is not None:
                    all_results.append(result)
                    done_keys_p2.add((mouseID, week))
                joblib.dump(all_results, results_path_p2, compress=("xz", 3))
                print(f"  [P2 {len(done_keys_p2)}/{total_p2}] Mouse {mouseID} {week}")

    print(f"All fits complete. {len(all_results)} joint results saved to {results_path_p2}")
    return phase1_results, all_results


if __name__ == "__main__":
    print(f"Configuration loaded from {DATA_DIR}")
    print(f"KT is fixed at {KT_FIXED_MM} mM (literature value).")
    print(f"Phase-1 output: {RESULTS_FILENAME_PHASE1}")
    print(f"Phase-2 output: {RESULTS_FILENAME}")
    run_light_sanity_checks()
    run_two_phase_fitting()
