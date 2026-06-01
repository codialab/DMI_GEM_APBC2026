"""Estimate local uncertainty around stored C6 oxidative/dilution DMI fits.

C7 is the uncertainty companion for ``model_fittingC6_improved.py``. It
reconstructs the exact C6 optimizer vector, estimates a local finite-difference
covariance, and propagates that covariance to C6's derived quantities. The
coordinate system is:

- global plasma parameters,
- tissue parameters ``eta, V_x, fv``,
- voxel parameters ``MR_glc, V_lac_loss, V_ox, phi_tca``.

The old B6 branch-split variables ``f_lac`` and independently fitted ``V_TCA``
are deliberately absent. Derived uncertainty is reported for ``T_max``,
``V_lac_source``, ``V_TCA_total``, ``V_dil_tca``, and the oxidative feasibility
diagnostic.

Original docstring follows.

----

This module is intentionally a *post-fit* companion to
`model_fittingC6_improved.py`.

Why a separate v7 file exists
-----------------------------
The v6 script performs the actual nonlinear optimization that infers fluxes from the
MRI time courses. The user asked for uncertainty estimates, but also explicitly asked
that the v6 files remain unchanged. This file therefore treats the saved v6 Phase-2
results as fixed optima and performs a second analysis pass around those optima.

What uncertainty is estimated here
----------------------------------
We estimate *local* uncertainty around each converged Phase-2 solution. Concretely,
we reconstruct the parameter vector that v6 optimized, rebuild an approximation of the
effective residual vector implied by the v6 loss, estimate a finite-difference Jacobian
of that residual vector, and then form a Gauss-Newton style covariance approximation.

That means the intervals produced here should be interpreted as:

1. local to the stored v6 optimum,
2. conditional on the v6 model structure,
3. conditional on the v6 regularization strength,
4. approximate rather than exact posterior intervals.

What this file deliberately does *not* claim
--------------------------------------------
This is not a full bootstrap and it is not a full Bayesian hierarchical inference
procedure. It does not quantify uncertainty arising from alternative model forms,
alternative tissue masks, or optimizer failure modes that move the solution to an
entirely different basin. It also does not change the fitted point estimates.

Why a local Gaussian approximation is a pragmatic first step
-----------------------------------------------------------
The underlying problem is expensive: every objective evaluation solves many ODEs,
there are shared plasma parameters coupled across tissues, and the v6 loss includes
reliability-aware channel skipping and trend terms. A full refit-based bootstrap would
be substantially more expensive than a single v6 run. The local approximation used here
is much cheaper, gives actionable diagnostics about identifiability, and is sufficient
to flag cases where uncertainty is obviously unstable.

Implementation outline
----------------------
The code is organized in the same order as the mathematical workflow:

1. load the stored v6 Phase-2 results,
2. reconstruct the optimized parameter vector and parameter metadata,
3. rebuild the effective residual vector implied by the v6 loss,
4. estimate a Jacobian of that residual vector by finite differences,
5. convert that Jacobian into a local covariance estimate,
6. propagate uncertainty to directly reported flux quantities,
7. stress-test that local approximation with a small number of one-parameter profile-style slices,
8. save the original fit plus an `uncertainty` subtree in new v7 artifacts.

The annotations are intentionally extensive so the logic can be reimplemented from
scratch without having to reverse engineer every intermediate quantity.
"""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

import model_fittingC6_improved as mf


MODEL_TAG = "branch_split_noGly_vC7_uncertaintyLocalApprox_oxDilution_KTfixed"
SOURCE_RESULTS_FILENAME = mf.RESULTS_FILENAME
RESULTS_FILENAME = "fitC7_results_MRI_fluxes_AllTissues_noGly_oxDilution_KTfixed_uncertainty.joblib.xz"
SUMMARY_FILENAME = "fitC7_results_MRI_fluxes_AllTissues_noGly_oxDilution_KTfixed_uncertainty_summary.csv.xz"

# These values control the numerical stability of the finite-difference covariance
# approximation rather than the biological model itself.
FD_REL_STEP = 2e-4
FD_MIN_STEP = 1e-5
FD_BOUND_SHRINK = 0.45
RIDGE_RELATIVE = 1e-8
COND_USABLE = 5e9
COND_WARN = 1e10
COND_FAIL = 1e14
CORRELATION_REPORT_LIMIT = 20
CORRELATION_HEATMAP_PARAM_LIMIT = 12
PROFILE_CHECK_COUNT = 2
PROFILE_GRID_POINTS = 7
PROFILE_SIGMA_WIDTH = 2.0
DRAW_COUNT = 400
Z_95 = 1.959963984540054

# The upstream C6 artifact was generated before the final condition curation:
# mice 54 and 56 are diet-induced/HFD animals, encoded as the `dia` group here.
MOUSE_GROUP_OVERRIDES = {
    54: "dia",
    56: "dia",
}


@dataclass(frozen=True)
class PackedResult:
    """Container holding the reconstructed v6 parameter vector for one fit.

    The stored v6 result already contains the fitted values, but not the exact optimizer
    vector in flattened form. Rebuilding that vector here gives v7 a single canonical
    coordinate system for finite differences and covariance calculations.
    """

    result: dict[str, Any]
    theta: np.ndarray
    bounds: list[tuple[float, float]]
    parameter_info: list[dict[str, Any]]
    tissue_names: list[str]
    n_voxels_per_tissue: list[int]
    tissues_voxel_data: list[tuple[str, list[dict[str, Any]]]]


def result_key(result: dict[str, Any]) -> tuple[int, str]:
    """Return the natural unique key for a stored joint fit."""

    return int(result["mouseID"]), str(result["week"])


def corrected_group(mouse_id: int, group: str | None) -> str:
    """Return the curated condition label for a mouse."""

    return MOUSE_GROUP_OVERRIDES.get(int(mouse_id), group if group is not None else "unknown")


def apply_group_overrides(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply curated mouse-to-condition labels in-place and return the same list."""

    for result in results:
        result["group"] = corrected_group(result["mouseID"], result.get("group"))
    return results


def load_phase2_results(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the stored v6 Phase-2 fits that will act as v7's input optima.

    The entire uncertainty analysis is anchored to these stored solutions. If a fit was
    never successfully written by v6, there is no v7 uncertainty estimate for it.
    """

    results_path = Path(path) if path is not None else mf.DATA_DIR / SOURCE_RESULTS_FILENAME
    if not results_path.exists():
        raise FileNotFoundError(f"Missing source Phase-2 results: {results_path}")
    results = joblib.load(results_path)
    if not isinstance(results, list):
        raise TypeError("Expected the stored v6 results to be a list of per-sample dictionaries.")
    return apply_group_overrides(results)


def ordered_tissue_names(result: dict[str, Any]) -> list[str]:
    """Return tissues in the canonical v6 order, excluding tissues absent from the fit."""

    return [tissue for tissue in mf.TISSUES if tissue in result.get("tissues", {})]


def encode_global_params(result: dict[str, Any]) -> list[float]:
    """Map stored global plasma parameters back into the optimizer parameterization.

    v6 stores `kp_fast` and `kp_slow`, but the optimizer worked with `kp_slow` and the
    strictly positive gap `kp_gap = kp_fast - kp_slow`. We therefore re-encode to the
    optimizer coordinates before doing any finite-difference work.
    """

    kp_slow, kp_gap = mf.encode_plasma_decay_params(result["kp_fast"], result["kp_slow"])
    return [
        float(result["a1"]),
        float(result["plasma_delay_min"]),
        float(result["plasma_tau_rise"]),
        float(result["plasma_w"]),
        float(kp_slow),
        float(kp_gap),
    ]


def pack_result(result: dict[str, Any]) -> PackedResult:
    """Flatten one stored joint fit into the exact parameter hierarchy used by C6.

    The resulting vector has the same structure as the Phase-2 optimizer variable:

    - 6 global plasma parameters
    - 3 tissue-level parameters per tissue present in the fit (eta, V_x, fv)
    - 4 voxel-level parameters per fitted voxel (MR_glc, V_lac_loss, V_ox, phi_tca)
    """

    tissue_names = ordered_tissue_names(result)
    if not tissue_names:
        raise ValueError(f"Result {result_key(result)} contains no tissues.")

    theta: list[float] = []
    bounds: list[tuple[float, float]] = []
    parameter_info: list[dict[str, Any]] = []
    n_voxels_per_tissue: list[int] = []

    global_names = ["a1", "plasma_delay_min", "plasma_tau_rise", "plasma_w", "kp_slow", "kp_gap"]
    for name, value, bound in zip(global_names, encode_global_params(result), mf.BOUNDS_GLOBAL):
        theta.append(float(value))
        bounds.append(tuple(bound))
        parameter_info.append({"level": "global", "name": name})

    tissue_param_names = ["eta", "V_x", "fv"]
    voxel_param_names = ["MR_glc", "V_lac_loss", "V_ox", "phi_tca"]
    for tissue_name in tissue_names:
        tissue_data = result["tissues"][tissue_name]
        n_voxels_per_tissue.append(len(tissue_data["voxels"]))

        for name, bound in zip(tissue_param_names, mf.BOUNDS_TISSUE):
            theta.append(float(tissue_data[name]))
            bounds.append(tuple(bound))
            parameter_info.append({"level": "tissue", "tissue": tissue_name, "name": name})

        for voxel_index, voxel in enumerate(tissue_data["voxels"]):
            for name, bound in zip(voxel_param_names, mf.BOUNDS_PER_VOX):
                theta.append(float(voxel[name]))
                bounds.append(tuple(bound))
                parameter_info.append(
                    {
                        "level": "voxel",
                        "tissue": tissue_name,
                        "voxel_index": voxel_index,
                        "coords": tuple(voxel["coords"]),
                        "name": name,
                    }
                )

    tissues_voxel_data = [
        (tissue_name, [voxel["voxel_data"] for voxel in result["tissues"][tissue_name]["voxels"]])
        for tissue_name in tissue_names
    ]

    return PackedResult(
        result=result,
        theta=np.asarray(theta, dtype=float),
        bounds=bounds,
        parameter_info=parameter_info,
        tissue_names=tissue_names,
        n_voxels_per_tissue=n_voxels_per_tissue,
        tissues_voxel_data=tissues_voxel_data,
    )


def build_bounds_multi(n_voxels_per_tissue: Iterable[int]) -> list[tuple[float, float]]:
    """Mirror the v6 Phase-2 bounds layout without importing v6's helper by side effect."""

    bounds = list(mf.BOUNDS_GLOBAL)
    for n_voxels in n_voxels_per_tissue:
        bounds.extend(mf.BOUNDS_TISSUE)
        bounds.extend(mf.BOUNDS_PER_VOX * int(n_voxels))
    return [tuple(bound) for bound in bounds]


def validate_packed_result(packed: PackedResult) -> None:
    """Cheap consistency checks before numerical differentiation.

    If these checks fail, the uncertainty calculation should stop immediately rather than
    producing intervals on a malformed parameter layout.
    """

    expected_bounds = build_bounds_multi(packed.n_voxels_per_tissue)
    if len(expected_bounds) != len(packed.theta):
        raise ValueError("Packed theta length does not match the expected Phase-2 layout.")
    if len(packed.bounds) != len(expected_bounds):
        raise ValueError("Packed bounds length does not match theta length.")
    for idx, (bound_a, bound_b) in enumerate(zip(packed.bounds, expected_bounds)):
        if tuple(bound_a) != tuple(bound_b):
            raise ValueError(f"Packed bound mismatch at parameter index {idx}: {bound_a} vs {bound_b}")


def compute_tissue_offset(packed: PackedResult, tissue_index: int) -> int:
    """Reproduce v6's offset convention for tissue blocks in the flattened vector."""

    offset = mf.N_GLOBAL_PARAMS
    for idx in range(tissue_index):
        offset += mf.N_TISSUE_PARAMS + mf.N_VOX_PARAMS * packed.n_voxels_per_tissue[idx]
    return offset


def unpack_global_params(theta: np.ndarray) -> dict[str, float]:
    """Decode the first six optimizer parameters into physical plasma parameters."""

    a1, plasma_delay_min, plasma_tau_rise, plasma_w, kp_slow, kp_gap = theta[: mf.N_GLOBAL_PARAMS]
    kp_fast, kp_slow = mf.decode_plasma_decay_params(kp_slow, kp_gap)
    return {
        "a1": float(a1),
        "plasma_delay_min": float(plasma_delay_min),
        "plasma_tau_rise": float(plasma_tau_rise),
        "plasma_w": float(plasma_w),
        "kp_fast": float(kp_fast),
        "kp_slow": float(kp_slow),
    }


def unpack_tissue_params(theta: np.ndarray, packed: PackedResult, tissue_index: int) -> dict[str, float]:
    """Decode one tissue block from the flattened Phase-2 parameter vector."""

    tissue_offset = compute_tissue_offset(packed, tissue_index)
    eta, V_x, fv = theta[tissue_offset : tissue_offset + mf.N_TISSUE_PARAMS]
    return {
        "eta": float(eta),
        "V_x": float(V_x),
        "KT": float(mf.FIXED_PARAMS["KT"]),
        "fv": float(np.clip(fv, 0.0, 0.15)),
    }


def unpack_voxel_params(theta: np.ndarray, voxel_offset: int) -> dict[str, float]:
    """Decode one C6 voxel block and compute C6 derived quantities."""

    return mf.unpack_voxel_params(theta, voxel_offset)


def build_result_from_theta(theta: np.ndarray, packed: PackedResult) -> dict[str, Any]:
    """Reconstruct a result-like dictionary from a candidate parameter vector.

    This helper is used for bound diagnostics and for packaging uncertainty summaries in a
    structure parallel to the original fit. The biological data for each voxel are reused
    from the stored v6 result to avoid any ambiguity about ROI order.
    """

    theta = np.asarray(theta, dtype=float)
    glob = unpack_global_params(theta)
    rebuilt = {
        "mouseID": packed.result["mouseID"],
        "week": packed.result["week"],
        "group": corrected_group(packed.result["mouseID"], packed.result.get("group")),
        **glob,
        "tissues": {},
    }

    for tissue_index, tissue_name in enumerate(packed.tissue_names):
        original_tissue = packed.result["tissues"][tissue_name]
        tissue_params = unpack_tissue_params(theta, packed, tissue_index)
        tissue_offset = compute_tissue_offset(packed, tissue_index)
        voxels = []
        for voxel_index, original_voxel in enumerate(original_tissue["voxels"]):
            voxel_offset = tissue_offset + mf.N_TISSUE_PARAMS + mf.N_VOX_PARAMS * voxel_index
            vox_params = mf.unpack_voxel_params(theta, voxel_offset, eta=tissue_params["eta"])
            voxels.append(
                {
                    "coords": tuple(original_voxel["coords"]),
                    **vox_params,
                }
            )

        rebuilt["tissues"][tissue_name] = {
            "eta": tissue_params["eta"],
            "V_x": tissue_params["V_x"],
            "KT": tissue_params["KT"],
            "fv": tissue_params["fv"],
            "voxels": voxels,
        }

    return rebuilt


def build_point_weights(obs: np.ndarray, key: str, time_min: np.ndarray | None) -> np.ndarray:
    """Mirror v6's per-time-point upweighting for positive observations."""

    obs = np.asarray(obs, dtype=float)
    point_weights = np.ones_like(obs, dtype=float)
    if key == "lac":
        point_weights[obs > 0.0] *= mf.LAC_POSITIVE_POINT_WEIGHT
    elif key == "glc":
        point_weights[obs > 0.0] *= mf.GLC_POSITIVE_POINT_WEIGHT
        if time_min is not None:
            time_min = np.asarray(time_min, dtype=float)
            early_mask = time_min <= 15.0
            point_weights[early_mask & (obs > 0.0)] *= mf.GLC_EARLY_POINT_WEIGHT
    elif key == "glx":
        point_weights[obs > 0.0] *= mf.GLX_POSITIVE_POINT_WEIGHT
    return point_weights


def build_trend_weight_block(
    obs: np.ndarray,
    pred: np.ndarray,
    key: str,
    reliability: float,
    sigma: float,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """Return the trend residual block that v6 adds on top of pointwise fitting.

    The v6 scalar loss combines a pointwise term and, for sufficiently informative data,
    a trend term based on first differences. Here we expose the trend part explicitly so it
    can contribute rows to the residual vector used for local covariance estimation.
    """

    if obs.size < 3:
        return None, None, 0.0

    obs_diff = np.diff(obs)
    pred_diff = np.diff(pred)
    diff_sigma = max(mf.estimate_noise_sigma(obs_diff), 0.5 * sigma)
    trend_resid = (pred_diff - obs_diff) / diff_sigma

    if key == "glc":
        trend_weights = np.ones_like(trend_resid)
        trend_weights[obs_diff > 0.0] *= mf.GLC_POSITIVE_POINT_WEIGHT
        trend_scale = mf.GLC_TREND_WEIGHT
    elif key == "glx" and reliability > 0.35:
        trend_weights = np.ones_like(trend_resid)
        trend_weights[obs_diff > 0.0] *= mf.GLX_POSITIVE_POINT_WEIGHT
        trend_scale = reliability * mf.GLX_TREND_WEIGHT
    elif key == "lac" and reliability > 0.50:
        trend_weights = np.ones_like(trend_resid)
        trend_scale = reliability * mf.LAC_TREND_WEIGHT
    else:
        return None, None, 0.0

    return trend_resid, trend_weights, float(trend_scale)


def gather_channel_blocks(theta: np.ndarray, packed: PackedResult) -> dict[str, list[dict[str, Any]]]:
    """Simulate every fitted voxel and collect channel-specific residual blocks.

    This is the central reconstruction step. The v6 optimizer minimized a scalar loss that
    aggregates pointwise and trend information across tissues and metabolites. For local
    covariance estimation we instead collect the individual standardized residual pieces
    *before* aggregation. The aggregation weights are then applied when flattening the
    blocks into a single residual vector.
    """

    theta = np.asarray(theta, dtype=float)
    glob = unpack_global_params(theta)
    fp_global = {**mf.FIXED_PARAMS, **glob}

    channel_blocks: dict[str, list[dict[str, Any]]] = {key: [] for key in mf.FIT_MET_KEYS}
    for tissue_index, tissue_name in enumerate(packed.tissue_names):
        original_tissue = packed.result["tissues"][tissue_name]
        tissue_params = unpack_tissue_params(theta, packed, tissue_index)
        fp_tissue = {**fp_global, "KT": tissue_params["KT"], "fv": tissue_params["fv"]}
        tissue_offset = compute_tissue_offset(packed, tissue_index)

        for voxel_index, original_voxel in enumerate(original_tissue["voxels"]):
            voxel_offset = tissue_offset + mf.N_TISSUE_PARAMS + mf.N_VOX_PARAMS * voxel_index
            vox_params = mf.unpack_voxel_params(theta, voxel_offset, eta=tissue_params["eta"])
            voxel_data = original_voxel["voxel_data"]

            pred = mf.simulate_voxel_window_averaged(
                voxel_data["time_min"],
                tissue_params["eta"],
                tissue_params["V_x"],
                vox_params["MR_glc"],
                vox_params["V_lac_loss"],
                vox_params["V_ox"],
                vox_params["phi_tca"],
                fp_tissue,
            )
            if pred is None:
                raise RuntimeError(
                    f"Simulation failed while evaluating {result_key(packed.result)} tissue={tissue_name} voxel={voxel_index}"
                )

            for key in mf.FIT_MET_KEYS:
                obs = np.asarray(voxel_data[key], dtype=float)
                model = np.asarray(pred[key], dtype=float)
                quality = mf.summarize_channel_quality(obs, key)

                # This reproduces the v6 hard-skip rule for weak secondary channels.
                if key != "glc" and quality["reliability"] < mf.SECONDARY_CHANNEL_SKIP_THRESHOLD:
                    continue

                sigma = quality["sigma"]
                point_resid = (model - obs) / sigma
                point_weights = build_point_weights(obs, key, np.asarray(voxel_data["time_min"], dtype=float))
                trend_resid, trend_weights, trend_scale = build_trend_weight_block(
                    obs,
                    model,
                    key,
                    quality["reliability"],
                    sigma,
                )

                channel_blocks[key].append(
                    {
                        "tissue": tissue_name,
                        "voxel_index": voxel_index,
                        "coords": tuple(original_voxel["coords"]),
                        "reliability": float(max(quality["reliability"], 1e-6)),
                        "point_resid": point_resid,
                        "point_weights": point_weights,
                        "trend_resid": trend_resid,
                        "trend_weights": trend_weights,
                        "trend_scale": float(trend_scale),
                    }
                )

    if not channel_blocks["glc"]:
        raise RuntimeError(f"No glucose residual blocks were available for {result_key(packed.result)}")
    return channel_blocks


def residual_vector_from_blocks(channel_blocks: dict[str, list[dict[str, Any]]]) -> np.ndarray:
    """Flatten channel blocks into one weighted residual vector.

    The v6 scalar objective averages within channel, then averages across present channels.
    Here we implement the analogous weighting in residual-vector form so that

    `J.T @ J`

    approximates the local curvature of the effective fitting target.
    """

    channel_weights = {"glc": 1.0, "lac": mf.LAC_LOSS_WEIGHT, "glx": 1.0}
    present_keys = [key for key in mf.FIT_MET_KEYS if channel_blocks[key]]
    channel_norm = math.sqrt(sum(channel_weights[key] for key in present_keys))
    residual_pieces: list[np.ndarray] = []

    for key in present_keys:
        blocks = channel_blocks[key]
        rel_sum = sum(block["reliability"] for block in blocks)
        if rel_sum <= 0.0:
            continue

        channel_factor = math.sqrt(channel_weights[key]) / max(channel_norm, 1e-12)
        for block in blocks:
            voxel_factor = math.sqrt(block["reliability"] / rel_sum)

            point_scale = channel_factor * voxel_factor / math.sqrt(len(block["point_resid"]))
            point_component = point_scale * np.sqrt(block["point_weights"]) * block["point_resid"]
            residual_pieces.append(np.asarray(point_component, dtype=float))

            if block["trend_resid"] is not None and block["trend_scale"] > 0.0:
                trend_scale = (
                    channel_factor
                    * voxel_factor
                    * math.sqrt(block["trend_scale"])
                    / math.sqrt(len(block["trend_resid"]))
                )
                trend_component = (
                    trend_scale * np.sqrt(block["trend_weights"]) * np.asarray(block["trend_resid"], dtype=float)
                )
                residual_pieces.append(np.asarray(trend_component, dtype=float))

    if not residual_pieces:
        raise RuntimeError("Residual vector construction produced no entries.")
    return np.concatenate(residual_pieces)


def build_residual_vector(theta: np.ndarray, packed: PackedResult) -> np.ndarray:
    """Convenience wrapper used throughout the finite-difference machinery."""

    return residual_vector_from_blocks(gather_channel_blocks(theta, packed))


def build_prior_precision_diagonal(packed: PackedResult) -> np.ndarray:
    """Convert the v6 quadratic penalties into diagonal precision contributions.

    The v6 regularization terms are quadratic in the parameters, so their Hessian is easy
    to write analytically. We inject those curvatures directly into the local information
    matrix so the covariance estimate is tied to the *penalized* objective actually used by
    v6 rather than to an unregularized least-squares surrogate.
    """

    precision = np.zeros(len(packed.theta), dtype=float)
    global_scales = [
        mf.PLASMA_A1_REG_SCALE,
        3.0,
        4.0,
        0.20,
        0.006,
        0.015,
    ]
    global_weights = [
        mf.PLASMA_REG_WEIGHT,
        mf.PLASMA_REG_WEIGHT,
        mf.PLASMA_REG_WEIGHT,
        mf.PLASMA_REG_WEIGHT,
        mf.PLASMA_REG_WEIGHT,
        mf.PLASMA_REG_WEIGHT,
    ]
    for idx, (weight, scale) in enumerate(zip(global_weights, global_scales)):
        precision[idx] = 2.0 * weight / (scale**2)

    # v6 additionally penalizes each tissue-level fv toward a prior mean.
    for idx, info in enumerate(packed.parameter_info):
        if info["level"] == "tissue" and info["name"] == "fv":
            precision[idx] += 2.0 * mf.FV_REG_WEIGHT / (mf.FV_REG_SCALE**2)
    return precision


def choose_fd_step(value: float, bounds: tuple[float, float]) -> float:
    """Choose a parameter-specific finite-difference step that respects bounds.

    The step is tied to both the parameter magnitude and the parameter span. The explicit
    bound handling matters because many fitted quantities can sit near biologically imposed
    box constraints.
    """

    low, high = bounds
    span = high - low
    base = max(abs(value), 1.0)
    step = max(FD_MIN_STEP, FD_REL_STEP * base, FD_REL_STEP * span)

    headroom_left = max(value - low, 0.0)
    headroom_right = max(high - value, 0.0)
    max_symmetric = FD_BOUND_SHRINK * min(headroom_left, headroom_right)
    if max_symmetric > 0.0:
        return min(step, max_symmetric)

    # If the parameter is effectively at a bound, still take a small inward step.
    max_one_sided = FD_BOUND_SHRINK * max(headroom_left, headroom_right)
    if max_one_sided > 0.0:
        return min(step, max_one_sided)
    return min(step, 0.25 * span)


def finite_difference_jacobian(theta: np.ndarray, packed: PackedResult) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Estimate the Jacobian of the residual vector by bounded finite differences.

    A central difference is preferred when both sides are available; otherwise a one-sided
    difference is used. Any failures are recorded as warnings so the downstream covariance
    summary can report that a fit is numerically fragile.
    """

    theta = np.asarray(theta, dtype=float)
    base_residual = build_residual_vector(theta, packed)
    n_obs = base_residual.size
    n_params = theta.size
    jacobian = np.full((n_obs, n_params), np.nan, dtype=float)
    warnings: list[str] = []

    for idx, (value, bounds) in enumerate(zip(theta, packed.bounds)):
        step = choose_fd_step(float(value), bounds)
        low, high = bounds

        plus_theta = theta.copy()
        minus_theta = theta.copy()
        can_plus = value + step <= high
        can_minus = value - step >= low

        try:
            if can_plus and can_minus:
                plus_theta[idx] = value + step
                minus_theta[idx] = value - step
                res_plus = build_residual_vector(plus_theta, packed)
                res_minus = build_residual_vector(minus_theta, packed)
                jacobian[:, idx] = (res_plus - res_minus) / (2.0 * step)
            elif can_plus:
                plus_theta[idx] = value + step
                res_plus = build_residual_vector(plus_theta, packed)
                jacobian[:, idx] = (res_plus - base_residual) / step
                warnings.append(f"one-sided-plus:{idx}")
            elif can_minus:
                minus_theta[idx] = value - step
                res_minus = build_residual_vector(minus_theta, packed)
                jacobian[:, idx] = (base_residual - res_minus) / step
                warnings.append(f"one-sided-minus:{idx}")
            else:
                warnings.append(f"no-headroom:{idx}")
                jacobian[:, idx] = 0.0
        except Exception as exc:  # pragma: no cover - numerical safeguard
            warnings.append(f"fd-failed:{idx}:{exc}")
            jacobian[:, idx] = 0.0

    return jacobian, base_residual, warnings


def scalar_objective(theta: np.ndarray, packed: PackedResult) -> float:
    """Evaluate the original v6 Phase-2 scalar objective at a candidate parameter vector.

    The covariance approximation is built from the residual-vector surrogate, but a scalar
    objective slice is still useful when checking whether the local quadratic approximation
    behaves sensibly in the neighborhood of the stored optimum.
    """

    return float(mf.joint_objective_multi(np.asarray(theta, dtype=float), packed.tissues_voxel_data, dict(mf.FIXED_PARAMS)))


def correlation_matrix_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix into a correlation matrix with safe zero handling."""

    std = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    denom = np.outer(std, std)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.divide(covariance, denom, out=np.zeros_like(covariance), where=denom > 0)
    return corr


def max_abs_offdiag(correlation: np.ndarray) -> float:
    """Return the strongest absolute off-diagonal correlation."""

    if correlation.size == 0 or correlation.shape[0] <= 1:
        return 0.0
    offdiag = np.abs(correlation - np.eye(correlation.shape[0]))
    return float(np.max(offdiag))


def parameter_short_label(info: dict[str, Any]) -> str:
    """Create compact labels for tables and heatmaps.

    The stored parameter metadata are verbose and structured, which is useful for machine
    parsing but awkward on plots. This helper generates short human-readable labels that
    still preserve the hierarchy level and tissue context.
    """

    if info.get("level") == "global":
        return f"global:{info['name']}"
    if info.get("level") == "tissue":
        return f"{info.get('tissue', '?')}:{info['name']}"
    coords = info.get("coords", ())
    return f"{info.get('tissue', '?')}:v{info.get('voxel_index', '?')}:{info['name']}@{coords}"


def parameter_scaled_uncertainty(theta: np.ndarray, covariance: np.ndarray, packed: PackedResult) -> np.ndarray:
    """Rank parameters by uncertainty relative to their feasible span.

    Raw standard errors are hard to compare across parameters with different units. Dividing
    by the optimizer box width yields a dimensionless score that is useful when choosing the
    most informative parameters for profile checks or compact heatmaps.
    """

    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    spans = np.asarray([high - low for low, high in packed.bounds], dtype=float)
    return np.divide(se, spans, out=np.zeros_like(se), where=spans > 0)


def select_profile_parameter_indices(theta: np.ndarray, covariance: np.ndarray, packed: PackedResult, max_checks: int = PROFILE_CHECK_COUNT) -> list[int]:
    """Pick a small set of informative parameters for profile-style slice checks.

    The goal is not to exhaustively scan every parameter. Instead we look at the most
    uncertain directions and intentionally diversify the selection so that the profile check
    covers both shared parameters and voxel-level parameters when available.
    """

    scaled = parameter_scaled_uncertainty(theta, covariance, packed)
    ranked = list(np.argsort(scaled)[::-1])
    selected: list[int] = []

    for preferred_levels in [("global", "tissue"), ("voxel",)]:
        for idx in ranked:
            if idx in selected:
                continue
            if packed.parameter_info[idx].get("level") in preferred_levels:
                selected.append(int(idx))
                break

    for idx in ranked:
        if len(selected) >= max_checks:
            break
        if int(idx) not in selected:
            selected.append(int(idx))
    return selected[:max_checks]


def profile_slice_checks(theta: np.ndarray, packed: PackedResult, covariance_info: dict[str, Any], max_checks: int = PROFILE_CHECK_COUNT) -> list[dict[str, Any]]:
    """Run lightweight profile-style slices for a few sensitive parameters.

    This is intentionally *not* a full profile likelihood, which would require re-optimizing
    all other parameters at each grid point. Instead we hold the remaining parameters fixed,
    evaluate the original v6 scalar objective and the residual-vector surrogate along a small
    one-dimensional grid, and compare that slice to the local quadratic approximation.

    The purpose is calibration, not formal inference. A strongly non-quadratic slice or a
    slice that still decreases materially away from the stored optimum is a warning sign that
    the local Gaussian approximation should be interpreted cautiously.
    """

    covariance = covariance_info["covariance"]
    information = covariance_info["information"]
    base_scalar = scalar_objective(theta, packed)
    base_residual = covariance_info["residual"]
    base_residual_energy = 0.5 * float(np.dot(base_residual, base_residual))

    checks: list[dict[str, Any]] = []
    for idx in select_profile_parameter_indices(theta, covariance, packed, max_checks=max_checks):
        variance = float(max(covariance[idx, idx], 0.0))
        se = math.sqrt(variance)
        low, high = packed.bounds[idx]
        if se <= 0.0:
            half_width = choose_fd_step(float(theta[idx]), (low, high))
        else:
            half_width = max(PROFILE_SIGMA_WIDTH * se, choose_fd_step(float(theta[idx]), (low, high)))
        grid_low = max(low, float(theta[idx]) - half_width)
        grid_high = min(high, float(theta[idx]) + half_width)
        if np.isclose(grid_low, grid_high):
            grid = np.array([float(theta[idx])], dtype=float)
        else:
            grid = np.linspace(grid_low, grid_high, PROFILE_GRID_POINTS)

        slice_rows: list[dict[str, float]] = []
        for value in grid:
            theta_slice = np.asarray(theta, dtype=float).copy()
            theta_slice[idx] = float(value)
            scalar_value = scalar_objective(theta_slice, packed)
            residual_value = build_residual_vector(theta_slice, packed)
            residual_energy = 0.5 * float(np.dot(residual_value, residual_value))
            delta = float(value - theta[idx])
            quadratic_rise = 0.5 * float(information[idx, idx]) * (delta**2)
            slice_rows.append(
                {
                    "value": float(value),
                    "delta": delta,
                    "scalar_objective": scalar_value,
                    "scalar_rise": float(scalar_value - base_scalar),
                    "residual_energy": residual_energy,
                    "residual_rise": float(residual_energy - base_residual_energy),
                    "quadratic_rise": quadratic_rise,
                }
            )

        noncenter = [row for row in slice_rows if abs(row["delta"]) > 1e-12]
        positive_quadratic = [row for row in noncenter if row["quadratic_rise"] > 1e-12]
        if positive_quadratic:
            log_ratios = [
                abs(math.log10(max(row["residual_rise"], 1e-12) / row["quadratic_rise"]))
                for row in positive_quadratic
            ]
            max_abs_log10_ratio = float(max(log_ratios))
        else:
            max_abs_log10_ratio = float("nan")

        min_scalar_rise = float(min(row["scalar_rise"] for row in noncenter)) if noncenter else 0.0
        if min_scalar_rise < -1e-3:
            profile_label = "slice_improves_away_from_optimum"
        elif np.isfinite(max_abs_log10_ratio) and max_abs_log10_ratio > 0.7:
            profile_label = "nonquadratic"
        else:
            profile_label = "consistent"

        checks.append(
            {
                "parameter_index": int(idx),
                "parameter": packed.parameter_info[idx],
                "parameter_label": parameter_short_label(packed.parameter_info[idx]),
                "estimate": float(theta[idx]),
                "se": float(se),
                "profile_label": profile_label,
                "min_scalar_rise": min_scalar_rise,
                "max_abs_log10_ratio": max_abs_log10_ratio,
                "grid": slice_rows,
            }
        )

    return checks


def assign_trust_label(
    condition_number: float,
    trust_flags: list[str],
    bound_hits: list[str],
    fd_warnings: list[str],
    max_abs_corr_value: float,
    effective_rank: int,
    n_params: int,
    profile_checks: list[dict[str, Any]],
) -> tuple[str, float, list[str]]:
    """Convert raw diagnostics into a graded trust label.

    The original v7 implementation had only three buckets. That was enough to detect clear
    failures, but it collapsed several distinct situations into `weakly_identified`. The
    refined version keeps a stronger separation between genuinely well-behaved fits, fits
    that are usable but not pristine, fits that are mostly limited by boundaries or mild
    numerical fragility, and fits where the local covariance approximation is genuinely hard
    to trust.
    """

    trust_score = 1.0
    refined_flags = list(trust_flags)

    if max_abs_corr_value >= 0.999:
        refined_flags.append("extreme_correlation")
        trust_score -= 0.35
    elif max_abs_corr_value >= 0.995:
        refined_flags.append("high_correlation")
        trust_score -= 0.25
    elif max_abs_corr_value >= 0.98:
        refined_flags.append("high_correlation")
        trust_score -= 0.20

    if effective_rank < max(n_params - 2, 1):
        refined_flags.append("reduced_effective_rank")
        trust_score -= 0.35

    if len(bound_hits) >= 3:
        refined_flags.append("many_bound_hits")
        trust_score -= 0.30
    elif len(bound_hits) > 0:
        trust_score -= 0.15

    if len(fd_warnings) >= 3:
        refined_flags.append("many_fd_warnings")
        trust_score -= 0.30
    elif len(fd_warnings) > 0:
        trust_score -= 0.15

    bad_profiles = [check for check in profile_checks if check["profile_label"] != "consistent"]
    if any(check["profile_label"] == "slice_improves_away_from_optimum" for check in bad_profiles):
        refined_flags.append("profile_slice_improves")
        trust_score -= 0.35
    elif bad_profiles:
        refined_flags.append("profile_nonquadratic")
        trust_score -= 0.20

    if "covariance_nonfinite" in refined_flags or "severely_ill_conditioned" in refined_flags:
        return "unstable", float(max(trust_score, 0.0)), refined_flags
    if condition_number >= COND_WARN or "extreme_correlation" in refined_flags or "profile_slice_improves" in refined_flags:
        return "weakly_identified", float(max(trust_score, 0.0)), refined_flags
    if len(bound_hits) > 0 or len(fd_warnings) > 0 or "profile_nonquadratic" in refined_flags:
        return "boundary_sensitive", float(max(trust_score, 0.0)), refined_flags
    if condition_number >= COND_USABLE or max_abs_corr_value >= 0.95:
        return "usable", float(max(trust_score, 0.0)), refined_flags
    return "well_identified", float(max(trust_score, 0.0)), refined_flags


def estimate_covariance(theta: np.ndarray, packed: PackedResult) -> dict[str, Any]:
    """Estimate local covariance and numerical diagnostics for one stored fit."""

    jacobian, residual, fd_warnings = finite_difference_jacobian(theta, packed)
    prior_precision = build_prior_precision_diagonal(packed)
    information = jacobian.T @ jacobian + np.diag(prior_precision)

    trace_scale = float(np.trace(information)) / max(information.shape[0], 1)
    ridge = RIDGE_RELATIVE * max(trace_scale, 1.0)
    information_ridge = information + ridge * np.eye(information.shape[0])

    dof = max(residual.size - theta.size, 1)
    rss = float(np.dot(residual, residual))
    sigma2 = rss / dof

    try:
        condition_number = float(np.linalg.cond(information_ridge))
    except np.linalg.LinAlgError:
        condition_number = float("inf")

    inversion_method = "inverse"
    try:
        covariance = sigma2 * np.linalg.inv(information_ridge)
    except np.linalg.LinAlgError:
        inversion_method = "pinv"
        covariance = sigma2 * np.linalg.pinv(information_ridge, rcond=1e-10)

    covariance = 0.5 * (covariance + covariance.T)
    variances = np.clip(np.diag(covariance), 0.0, None)
    standard_errors = np.sqrt(variances)
    singular_values = np.linalg.svd(information_ridge, compute_uv=False)
    correlation = correlation_matrix_from_covariance(covariance)
    max_abs_corr_value = max_abs_offdiag(correlation)
    effective_rank = int(np.sum(singular_values > max(singular_values[0] if singular_values.size else 0.0, 1.0) * 1e-10))

    trust_flags: list[str] = []
    if np.any(~np.isfinite(covariance)):
        trust_flags.append("covariance_nonfinite")
    if condition_number >= COND_WARN:
        trust_flags.append("ill_conditioned")
    if condition_number >= COND_FAIL:
        trust_flags.append("severely_ill_conditioned")
    if fd_warnings:
        trust_flags.append("finite_difference_warnings")

    rebuilt = build_result_from_theta(theta, packed)
    bound_hits = mf.collect_bound_hits(
        {key: rebuilt[key] for key in ["a1", "plasma_delay_min", "plasma_tau_rise", "plasma_w", "kp_fast", "kp_slow"]},
        {tissue: {k: rebuilt["tissues"][tissue][k] for k in ["eta", "V_x", "fv"]} for tissue in rebuilt["tissues"]},
        [
            (tissue, voxel)
            for tissue in rebuilt["tissues"]
            for voxel in rebuilt["tissues"][tissue]["voxels"]
        ],
    )
    if bound_hits:
        trust_flags.append("near_bounds")

    profile_checks = profile_slice_checks(theta, packed, {"covariance": covariance, "information": information_ridge, "residual": residual})
    trust_label, trust_score, refined_flags = assign_trust_label(
        condition_number=condition_number,
        trust_flags=trust_flags,
        bound_hits=bound_hits,
        fd_warnings=fd_warnings,
        max_abs_corr_value=max_abs_corr_value,
        effective_rank=effective_rank,
        n_params=theta.size,
        profile_checks=profile_checks,
    )

    return {
        "jacobian": jacobian,
        "residual": residual,
        "prior_precision": prior_precision,
        "information": information_ridge,
        "covariance": covariance,
        "standard_errors": standard_errors,
        "sigma2": float(sigma2),
        "rss": rss,
        "dof": int(dof),
        "condition_number": condition_number,
        "singular_values": singular_values,
        "correlation": correlation,
        "max_abs_corr": max_abs_corr_value,
        "effective_rank": effective_rank,
        "ridge": float(ridge),
        "fd_warnings": fd_warnings,
        "bound_hits": bound_hits,
        "trust_flags": refined_flags,
        "trust_label": trust_label,
        "trust_score": trust_score,
        "profile_checks": profile_checks,
        "inversion_method": inversion_method,
    }


def interval_from_mean_se(mean: float, se: float, z_value: float = Z_95) -> dict[str, float]:
    """Return a standard symmetric normal-approximation interval."""

    return {
        "estimate": float(mean),
        "se": float(se),
        "ci95_low": float(mean - z_value * se),
        "ci95_high": float(mean + z_value * se),
    }


def parameter_summary_table(theta: np.ndarray, covariance: np.ndarray, packed: PackedResult) -> list[dict[str, Any]]:
    """Convert the covariance matrix into per-parameter interval summaries."""

    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    rows = []
    for idx, info in enumerate(packed.parameter_info):
        row = dict(info)
        row["index"] = idx
        row.update(interval_from_mean_se(float(theta[idx]), float(se[idx])))
        rows.append(row)
    return rows


def selected_correlations(covariance: np.ndarray, packed: PackedResult, limit: int = CORRELATION_REPORT_LIMIT) -> list[dict[str, Any]]:
    """Return the strongest parameter correlations to help diagnose identifiability.

    Saving the full correlation matrix inside every result would be bulky and awkward to
    inspect. For a first-pass workflow, the most informative summary is usually the set of
    strongest absolute correlations away from the diagonal.
    """

    corr = correlation_matrix_from_covariance(covariance)

    pairs: list[dict[str, Any]] = []
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[1]):
            pairs.append(
                {
                    "i": i,
                    "j": j,
                    "corr": float(corr[i, j]),
                    "abs_corr": float(abs(corr[i, j])),
                    "left": packed.parameter_info[i],
                    "right": packed.parameter_info[j],
                }
            )
    pairs.sort(key=lambda item: item["abs_corr"], reverse=True)
    return pairs[:limit]


def build_correlation_heatmap_data(theta: np.ndarray, covariance: np.ndarray, packed: PackedResult, max_params: int = CORRELATION_HEATMAP_PARAM_LIMIT) -> dict[str, Any]:
    """Save a compact correlation submatrix for notebook heatmap rendering.

    The full correlation matrix can be large and is mostly redundant for interactive use.
    The notebook usually needs a readable panel for the most uncertain parameters instead of
    every single voxel variable, so we save a compact submatrix and a set of short labels.
    """

    scaled = parameter_scaled_uncertainty(theta, covariance, packed)
    ranked = list(np.argsort(scaled)[::-1][:max_params])
    corr = correlation_matrix_from_covariance(covariance)
    submatrix = corr[np.ix_(ranked, ranked)]
    return {
        "indices": [int(idx) for idx in ranked],
        "labels": [parameter_short_label(packed.parameter_info[idx]) for idx in ranked],
        "matrix": submatrix.tolist(),
    }


def draw_parameter_samples(theta: np.ndarray, covariance: np.ndarray, bounds: list[tuple[float, float]], seed: int) -> np.ndarray:
    """Generate clipped local Gaussian draws for uncertainty propagation.

    The clipping is a pragmatic compromise. It preserves the box-constrained support of the
    fitted parameters while still letting us cheaply propagate the full local covariance to
    nonlinear summaries.
    """

    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(np.asarray(theta, dtype=float), covariance, size=DRAW_COUNT, method="svd")
    for idx, (low, high) in enumerate(bounds):
        draws[:, idx] = np.clip(draws[:, idx], low, high)
    return draws


def propagate_draw_summaries(draws: np.ndarray, packed: PackedResult) -> dict[str, Any]:
    """Summarize local Gaussian draws at the tissue and voxel reporting levels."""

    tissue_rows = []
    voxel_rows = []
    for tissue_index, tissue_name in enumerate(packed.tissue_names):
        tissue_offset = compute_tissue_offset(packed, tissue_index)
        # C7: tissue block is [eta, V_x, fv]; KT is fixed and T_max is voxel-derived.
        idx_eta = tissue_offset + 0
        idx_vx = tissue_offset + 1
        idx_fv = tissue_offset + 2
        _kt_const = float(mf.FIXED_PARAMS["KT"])
        tissue_rows.append(
            {
                "tissue": tissue_name,
                "name": "KT",
                "estimate": _kt_const,
                "draw_q025": _kt_const,
                "draw_q975": _kt_const,
            }
        )
        for name, param_idx in [("eta", idx_eta), ("V_x", idx_vx), ("fv", idx_fv)]:
            values = draws[:, param_idx]
            tissue_rows.append(
                {
                    "tissue": tissue_name,
                    "name": name,
                    "estimate": float(np.mean(values)),
                    "draw_q025": float(np.quantile(values, 0.025)),
                    "draw_q975": float(np.quantile(values, 0.975)),
                }
            )

        original_tissue = packed.result["tissues"][tissue_name]
        for voxel_index, original_voxel in enumerate(original_tissue["voxels"]):
            voxel_offset = tissue_offset + mf.N_TISSUE_PARAMS + mf.N_VOX_PARAMS * voxel_index
            eta = draws[:, idx_eta]
            mr = draws[:, voxel_offset + 0]
            v_lac_loss = draws[:, voxel_offset + 1]
            v_ox = draws[:, voxel_offset + 2]
            phi_tca = np.clip(draws[:, voxel_offset + 3], mf.BOUNDS_PER_VOX[3][0], mf.BOUNDS_PER_VOX[3][1])
            t_max = eta * mr
            v_tca_total = v_ox / np.maximum(phi_tca, mf.EPS_POOL)
            v_dil_tca = np.maximum(v_tca_total - v_ox, 0.0)
            ox_capacity = np.maximum(2.0 * mr - v_lac_loss, mf.EPS_POOL)
            ox_ratio = v_ox / ox_capacity
            for name, values in [
                ("MR_glc", mr),
                ("V_lac_source", mr),
                ("V_lac", mr),
                ("V_lac_loss", v_lac_loss),
                ("V_out", v_lac_loss),
                ("V_ox", v_ox),
                ("phi_tca", phi_tca),
                ("T_max", t_max),
                ("V_TCA_total", v_tca_total),
                ("V_dil_tca", v_dil_tca),
                ("V_ox_feasibility_ratio", ox_ratio),
            ]:
                voxel_rows.append(
                    {
                        "tissue": tissue_name,
                        "voxel_index": voxel_index,
                        "coords": tuple(original_voxel["coords"]),
                        "name": name,
                        "estimate": float(np.mean(values)),
                        "draw_q025": float(np.quantile(values, 0.025)),
                        "draw_q975": float(np.quantile(values, 0.975)),
                    }
                )

    return {"tissue_draw_summary": tissue_rows, "voxel_draw_summary": voxel_rows}


def build_uncertainty_summary(theta: np.ndarray, packed: PackedResult, covariance_info: dict[str, Any]) -> dict[str, Any]:
    """Collect every uncertainty product saved back into the v7 result artifact."""

    covariance = covariance_info["covariance"]
    parameter_rows = parameter_summary_table(theta, covariance, packed)
    correlation_rows = selected_correlations(covariance, packed)
    correlation_heatmap = build_correlation_heatmap_data(theta, covariance, packed)
    draws = draw_parameter_samples(theta, covariance, packed.bounds, seed=1000 + packed.result["mouseID"])
    draw_summaries = propagate_draw_summaries(draws, packed)

    voxel_flux_rows = []
    for tissue_index, tissue_name in enumerate(packed.tissue_names):
        tissue_offset = compute_tissue_offset(packed, tissue_index)
        original_tissue = packed.result["tissues"][tissue_name]
        for voxel_index, original_voxel in enumerate(original_tissue["voxels"]):
            voxel_offset = tissue_offset + mf.N_TISSUE_PARAMS + mf.N_VOX_PARAMS * voxel_index
            voxel_params = mf.unpack_voxel_params(theta, voxel_offset, eta=theta[tissue_offset])
            voxel_flux_rows.append(
                {
                    "tissue": tissue_name,
                    "voxel_index": voxel_index,
                    "coords": tuple(original_voxel["coords"]),
                    **{name: interval_from_mean_se(float(value), 0.0) for name, value in voxel_params.items() if isinstance(value, (int, float, np.floating))},
                }
            )

    return {
        "model_version": MODEL_TAG,
        "parameter_info": packed.parameter_info,
        "parameter_summary": parameter_rows,
        "top_correlations": correlation_rows,
        "correlation_heatmap": correlation_heatmap,
        "voxel_flux_summary": voxel_flux_rows,
        "profile_checks": covariance_info["profile_checks"],
        "draw_count": DRAW_COUNT,
        **draw_summaries,
        "covariance_diagnostics": {
            "sigma2": covariance_info["sigma2"],
            "rss": covariance_info["rss"],
            "dof": covariance_info["dof"],
            "condition_number": covariance_info["condition_number"],
            "effective_rank": covariance_info["effective_rank"],
            "max_abs_corr": covariance_info["max_abs_corr"],
            "ridge": covariance_info["ridge"],
            "fd_warnings": covariance_info["fd_warnings"],
            "bound_hits": covariance_info["bound_hits"],
            "trust_flags": covariance_info["trust_flags"],
            "trust_label": covariance_info["trust_label"],
            "trust_score": covariance_info["trust_score"],
            "inversion_method": covariance_info["inversion_method"],
            "singular_values": covariance_info["singular_values"].tolist(),
        },
    }


def analyze_result(result: dict[str, Any]) -> dict[str, Any]:
    """Compute the v7 uncertainty payload for one stored v6 joint fit."""

    result = copy.deepcopy(result)
    apply_group_overrides([result])
    packed = pack_result(result)
    validate_packed_result(packed)
    covariance_info = estimate_covariance(packed.theta, packed)
    analyzed = copy.deepcopy(result)
    analyzed["uncertainty"] = build_uncertainty_summary(packed.theta, packed, covariance_info)
    return analyzed


def summarize_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a flat summary table for quick inspection and notebook loading."""

    apply_group_overrides(results)
    rows = []
    for result in results:
        uncertainty = result.get("uncertainty", {})
        diag = uncertainty.get("covariance_diagnostics", {})
        rows.append(
            {
                "mouseID": result["mouseID"],
                "week": result["week"],
                "group": result["group"],
                "success": result.get("success", False),
                "include_in_summary": result.get("include_in_summary", False),
                "trust_label": diag.get("trust_label", "missing"),
                "trust_score": diag.get("trust_score", float("nan")),
                "condition_number": diag.get("condition_number", float("nan")),
                "effective_rank": diag.get("effective_rank", float("nan")),
                "max_abs_corr": diag.get("max_abs_corr", float("nan")),
                "n_fd_warnings": len(diag.get("fd_warnings", [])),
                "n_bound_hits": len(diag.get("bound_hits", [])),
                "n_profile_checks": len(result.get("uncertainty", {}).get("profile_checks", [])),
                "n_profile_issues": sum(
                    check.get("profile_label") != "consistent"
                    for check in result.get("uncertainty", {}).get("profile_checks", [])
                ),
                "sigma2": diag.get("sigma2", float("nan")),
                "rss": diag.get("rss", float("nan")),
                "final_cost": result.get("final_cost", float("nan")),
            }
        )
    return pd.DataFrame(rows).sort_values(["week", "group", "mouseID"])


def select_results(
    results: list[dict[str, Any]],
    mouse_ids: set[int] | None = None,
    weeks: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter the input result list for partial server-side reruns."""

    selected = []
    for result in results:
        if mouse_ids is not None and int(result["mouseID"]) not in mouse_ids:
            continue
        if weeks is not None and str(result["week"]) not in weeks:
            continue
        selected.append(result)
    return selected


def run_uncertainty_estimation(
    source_results_path: Path | None = None,
    output_path: Path | None = None,
    summary_path: Path | None = None,
    mouse_ids: set[int] | None = None,
    weeks: set[str] | None = None,
    force: bool = False,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Main server-oriented entrypoint.

    The function is deliberately resumable. If the v7 output file already exists, previous
    uncertainty results are loaded and reused unless `force=True` is requested.
    """

    source_results = select_results(load_phase2_results(source_results_path), mouse_ids=mouse_ids, weeks=weeks)
    output_path = Path(output_path) if output_path is not None else mf.DATA_DIR / RESULTS_FILENAME
    summary_path = Path(summary_path) if summary_path is not None else mf.DATA_DIR / SUMMARY_FILENAME

    existing: dict[tuple[int, str], dict[str, Any]] = {}
    if output_path.exists() and not force:
        for result in apply_group_overrides(joblib.load(output_path)):
            existing[result_key(result)] = result

    analyzed_results: list[dict[str, Any]] = []
    total = len(source_results)
    for index, result in enumerate(source_results, start=1):
        key = result_key(result)
        if key in existing and not force:
            analyzed = existing[key]
            print(f"[{index}/{total}] Reusing existing uncertainty result for mouse {key[0]} {key[1]}")
        else:
            print(f"[{index}/{total}] Estimating uncertainty for mouse {key[0]} {key[1]} ...", flush=True)
            analyzed = analyze_result(result)
            print(
                "    trust="
                f"{analyzed['uncertainty']['covariance_diagnostics']['trust_label']} "
                f"score={analyzed['uncertainty']['covariance_diagnostics']['trust_score']:.2f} "
                f"cond={analyzed['uncertainty']['covariance_diagnostics']['condition_number']:.3e}"
            )
        analyzed_results.append(analyzed)
        joblib.dump(analyzed_results, output_path, compress=("xz", 3))

    summary_df = summarize_results(analyzed_results)
    summary_df.to_csv(summary_path, index=False)
    return analyzed_results, summary_df


def parse_args() -> argparse.Namespace:
    """Parse the small command-line surface needed for server use."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, default=None, help="Override the v6 Phase-2 source results path.")
    parser.add_argument("--output", type=Path, default=None, help="Override the v7 joblib output path.")
    parser.add_argument("--summary", type=Path, default=None, help="Override the v7 summary CSV path.")
    parser.add_argument("--mouse-ids", type=int, nargs="*", default=None, help="Optional subset of mouse IDs to analyze.")
    parser.add_argument("--weeks", nargs="*", default=None, help="Optional subset of weeks to analyze.")
    parser.add_argument("--force", action="store_true", help="Recompute uncertainty even if the v7 output already exists.")
    return parser.parse_args()


def run_light_sanity_checks() -> None:
    """Run fast checks that disconfirm the loader assumptions before full execution."""

    results = load_phase2_results()
    if not results:
        raise RuntimeError("No source Phase-2 results were found for uncertainty estimation.")

    sample = pack_result(results[0])
    validate_packed_result(sample)
    residual = build_residual_vector(sample.theta, sample)
    if residual.ndim != 1 or residual.size == 0:
        raise RuntimeError("Residual-vector construction failed the basic shape check.")


def main() -> None:
    """CLI entrypoint used on remote servers or from local terminals."""

    args = parse_args()
    run_light_sanity_checks()
    analyzed_results, summary_df = run_uncertainty_estimation(
        source_results_path=args.source_results,
        output_path=args.output,
        summary_path=args.summary,
        mouse_ids=set(args.mouse_ids) if args.mouse_ids is not None else None,
        weeks=set(args.weeks) if args.weeks is not None else None,
        force=args.force,
    )
    print(f"Saved {len(analyzed_results)} uncertainty-enriched results to {args.output or mf.DATA_DIR / RESULTS_FILENAME}")
    print(f"Saved flat summary table with {len(summary_df)} rows to {args.summary or mf.DATA_DIR / SUMMARY_FILENAME}")


if __name__ == "__main__":
    main()
