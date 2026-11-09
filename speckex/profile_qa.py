"""Optional Horne profile diagnostics with frozen aperture, nuisance model and pixel variances."""

import json
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Table

from . import __version__
from .sensitivity import summarize_alternatives


# Advisory criteria describe diagnostics only; none reject extractions or modify statistical errors.
FWHM_FACTOR = float(2 * np.sqrt(2 * np.log(2)))
ADVISORY_LIMITS = {"relative_width_error": .3, "amplitude_snr": 5., "bound_fraction": .1,
                   "reduced_chi2": 2., "structure_score": 4., "sensitivity_ratio": 1.,
                   "residual_lag_correlation": .5, "residual_median_score": 2.}
PROFILE_FLAGS = {1: "reference unavailable", 2: "profile poorly constrained", 4: "width near fit bound",
                 8: "profile sensitivity exceeds reference sigma", 16: "reference star near aperture"}


# Validate optional perturbations without changing the standard extraction parameter space.
def validate_profile_settings(args):
    if getattr(args, "method", "boxcar") not in ("horne", "both"):
        raise ValueError("--horne-qa requires --method horne or --method both.")
    for key, default in (("profile_centroid_fraction", .05), ("profile_fwhm_fraction", .1),
                         ("profile_trace_fraction", .05)):
        value = getattr(args, key, default)
        if not np.isfinite(value) or not 0 <= value < 1:
            raise ValueError("Profile QA perturbation fractions must be finite and in [0,1).")
    multiplier = getattr(args, "profile_sigma_multiplier", 1.)
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Profile QA uncertainty multiplier must be finite and positive.")
    if getattr(args, "profile_step_mode", "auto") not in ("auto", "fractional"):
        raise ValueError("Profile step mode must be auto or fractional.")


# Reevaluate the same finite-aperture Horne estimator directly on original pixels for any trial kernel.
def fixed_aperture_horne(result, centers_pixel, sigma_pixel):
    reference = result["methods"]["horne"]
    valid = ((reference["flags"] == 0) & np.isfinite(reference["flux"])
             & np.isfinite(reference["error"]) & (reference["error"] > 0))
    rows = np.arange(len(result["spatial"]))[:, None]
    sigma = np.broadcast_to(np.asarray(sigma_pixel, dtype=float), len(valid))
    center = np.broadcast_to(np.asarray(centers_pixel, dtype=float), len(valid))
    valid &= np.isfinite(sigma) & (sigma > 0) & np.isfinite(center)
    # Keep source pixel support and fractional aperture fixed even when the trial kernel center changes.
    used = (result["source_weights"] > 0) & valid[None, :]
    variance = result["input_error"] ** 2
    valid &= np.all(~used | (np.isfinite(variance) & (variance > 0)), axis=0)
    safe_sigma = np.where(valid, sigma, 1.)
    gaussian = np.exp(-.5 * ((rows - np.where(valid, center, 0.)) / safe_sigma) ** 2)
    inverse = np.zeros_like(variance)
    np.divide(1., variance, out=inverse, where=used & valid[None, :])
    enclosed = np.sum(result["source_weights"] * gaussian, axis=0)
    denominator = np.sum(gaussian ** 2 * inverse, axis=0)
    weights = np.zeros_like(variance)
    np.divide(enclosed * gaussian * inverse, denominator, out=weights,
              where=valid[None, :] & (denominator > 0))
    # The already-fitted background and optional stellar component are never recomputed during profile QA.
    clean = result["data"] - result["sky_image"] - result["star_image"]
    flux = np.sum(weights * np.where(weights != 0, clean, 0.), axis=0)
    valid &= (denominator > 0) & np.isfinite(flux)
    flux[~valid], weights[:, ~valid] = np.nan, 0.
    return flux, weights


# Construct the fixed nuisance design on selected source rows, including sky/star cross-covariances.
def nuisance_design(result, rows, column):
    delta = result["spatial"][rows] - result["target"][column]
    vectors = [delta ** order for order in range(result["background_degree"] + 1)]
    if result["substars"]:
        vectors.append((1 + ((result["spatial"][rows] - result["star_center"][column])
                            / result["star_width"][column]) ** 2) ** -result["star_beta"][column])
    return np.column_stack(vectors)


# Measure native-grid residuals with full conditional source covariance, fitting one signed amplitude per column.
def profile_goodness(result, centers, sigma):
    reference = result["methods"]["horne"]
    nx = len(centers)
    output = {key: np.full(nx, np.nan) for key in
              ("CHI2", "REDUCED_CHI2", "GLS_AMPLITUDE", "ASYMMETRY_SCORE", "WING_SCORE")}
    output["DOF"] = np.zeros(nx, dtype=int)
    clean = result["data"] - result["sky_image"] - result["star_image"]
    # Background fits use disjoint pixels, so their prediction covariance adds to source-pixel variance.
    for column in np.flatnonzero(reference["flags"] == 0):
        rows = np.flatnonzero(result["source_weights"][:, column] > 0)
        if len(rows) < 2:
            continue
        data = clean[rows, column]
        design = nuisance_design(result, rows, column)
        covariance = (np.diag(result["input_error"][rows, column] ** 2)
                      + design @ result["nuisance_covariance"][column] @ design.T)
        offset = rows - centers[column]
        # Roundoff at an integer trace row must not assign the central pixel an artificial left/right sign.
        tolerance = 16 * np.finfo(float).eps * max(1., abs(centers[column]))
        offset = np.where(abs(offset) <= tolerance, 0., offset)
        gaussian = np.exp(-.5 * (offset / sigma) ** 2)
        if not np.all(np.isfinite(data)) or not np.all(np.isfinite(covariance)):
            continue
        try:
            np.linalg.cholesky(covariance)
            solved = np.linalg.solve(covariance, np.column_stack([gaussian, data]))
        except np.linalg.LinAlgError:
            continue
        denominator = gaussian @ solved[:, 0]
        if not np.isfinite(denominator) or denominator <= 0:
            continue
        amplitude = float(gaussian @ solved[:, 1] / denominator)
        residual = data - amplitude * gaussian
        chi2 = float(residual @ (solved[:, 1] - amplitude * solved[:, 0]))
        output["CHI2"][column] = max(0., chi2)
        output["DOF"][column] = len(rows) - 1
        output["REDUCED_CHI2"][column] = max(0., chi2) / (len(rows) - 1)
        output["GLS_AMPLITUDE"][column] = amplitude
        # Residual projections remove the fitted amplitude's covariance; their signs expose coherent structure.
        for key, contrast in (("ASYMMETRY_SCORE", np.sign(offset)),
                              ("WING_SCORE", (np.abs(offset) > sigma).astype(float))):
            residual_variance = float(contrast @ covariance @ contrast - (contrast @ gaussian) ** 2 / denominator)
            if residual_variance > 0:
                output[key][column] = float(contrast @ residual / np.sqrt(residual_variance))
    return output


# Retain actual fixed-centroid fit information and assess its approximate diagonal-stack covariance.
def retained_profile(result):
    profile = result["methods"]["horne"]["profile"]
    sigma = float(profile["sigma_pixel"])
    x, observed, error = [np.asarray(profile[key]) for key in ("offset_pixel", "measured", "formal_error")]
    amplitude = float(profile["amplitude"])
    gaussian = np.exp(-.5 * (x / sigma) ** 2)
    valid = np.isfinite(observed) & np.isfinite(error) & (error > 0)
    dof = int(valid.sum()) - 2
    chi2 = float(np.sum(((observed[valid] - amplitude * gaussian[valid]) / error[valid]) ** 2))
    # The science fit estimated amplitude and shared width; it did not estimate a centroid or free trace.
    jacobian = np.column_stack([gaussian, amplitude * gaussian * x ** 2 / sigma ** 3])
    amplitude_error = None
    try:
        covariance = np.linalg.inv((jacobian[valid] / error[valid, None]).T
                                   @ (jacobian[valid] / error[valid, None]))
        if np.isfinite(covariance[0, 0]) and covariance[0, 0] > 0:
            amplitude_error = float(np.sqrt(covariance[0, 0]))
    except np.linalg.LinAlgError:
        pass
    width_error = float(profile.get("sigma_error_pixel", np.nan))
    relative_error = width_error / sigma if np.isfinite(width_error) and width_error > 0 else None
    scale = float(np.median(np.diff(result["spatial"])))
    good = result["methods"]["horne"]["flags"] == 0
    half_width = float(np.median(result["source_weights"][:, good].sum(axis=0)) / 2)
    low, high = .4, max(.5, half_width)
    near_bound = sigma <= low * (1 + ADVISORY_LIMITS["bound_fraction"]) or sigma >= high * (1 - ADVISORY_LIMITS["bound_fraction"])
    snr = amplitude / amplitude_error if amplitude_error is not None else None
    poorly_constrained = (relative_error is None or relative_error > ADVISORY_LIMITS["relative_width_error"]
                          or snr is None or snr < ADVISORY_LIMITS["amplitude_snr"])
    summary = {"status": profile["status"], "sigma_pixel": sigma, "sigma_arcsec": sigma * scale,
        "fwhm_pixel": sigma * FWHM_FACTOR, "fwhm_arcsec": sigma * scale * FWHM_FACTOR,
        "sigma_error_pixel": width_error if np.isfinite(width_error) and width_error > 0 else None,
        "fwhm_error_pixel": width_error * FWHM_FACTOR if np.isfinite(width_error) and width_error > 0 else None,
        "width_bounds_pixel": [low, high], "width_near_bound": bool(near_bound),
        "relative_width_error": relative_error, "amplitude": amplitude,
        "amplitude_error_approximate": amplitude_error, "amplitude_snr_approximate": snr,
        "poorly_constrained": bool(poorly_constrained), "stack_chi2": chi2,
        "stack_dof": dof, "stack_reduced_chi2_approximate": chi2 / dof if dof > 0 else None,
        "centroid_offset_from_trace_pixel": 0., "centroid_fitted": False, "centroid_error_pixel": None,
        "trace_uncertainty": None, "trace_model": "Confirmed reference-star trace plus target offset",
        "fit_error_limitations": "Formal diagonal-stack covariance omits interpolation, nuisance/upstream correlations and interpolation-induced shape bias."}
    return summary, x, observed, error, amplitude * gaussian


# Use a meaningful conditional width error when available; positional uncertainties are not fitted in this model.
def trial_profiles(result, args, fit, centers):
    sigma = fit["sigma_pixel"]
    width_step = sigma * getattr(args, "profile_fwhm_fraction", .1)
    use_error = (getattr(args, "profile_step_mode", "auto") == "auto" and not fit["poorly_constrained"]
                 and not fit["width_near_bound"] and fit["stack_reduced_chi2_approximate"] is not None
                 and fit["stack_reduced_chi2_approximate"] <= ADVISORY_LIMITS["reduced_chi2"])
    width_basis = "fraction of fitted FWHM (same fraction of sigma)"
    if use_error:
        width_step = fit["sigma_error_pixel"] * getattr(args, "profile_sigma_multiplier", 1.)
        width_basis = "conditional formal sigma error; not a calibrated profile uncertainty"
    centroid_step = fit["fwhm_pixel"] * getattr(args, "profile_centroid_fraction", .05)
    trace_step = fit["fwhm_pixel"] * getattr(args, "profile_trace_fraction", .05)
    wavelength = result["wavelength"]
    span = float(wavelength[-1] - wavelength[0])
    ramp = 2 * (wavelength - wavelength[0]) / span - 1 if span else np.zeros(len(wavelength))
    trials = [{"name": "reference", "sigma_pixel": sigma, "centers": centers.copy(), "basis": "confirmed"}]
    # Centroid is a rigid kernel offset; trace is a wavelength-dependent tilt, with the physical aperture fixed.
    for family, step, displacement, basis in (
            ("centroid", centroid_step, np.ones(len(centers)), "fraction of fitted FWHM; centroid error unavailable"),
            ("fwhm", width_step, None, width_basis),
            ("trace", trace_step, ramp, "linear chromatic kernel drift; trace covariance unavailable")):
        for sign, label in ((-1, "minus"), (1, "plus")):
            trials.append({"name": f"{family}_{label}", "sigma_pixel": sigma + sign * step if family == "fwhm" else sigma,
                "centers": centers.copy() if family == "fwhm" else centers + sign * step * displacement,
                "basis": basis})
    return trials


# Compare methods using their common finite aperture and covariance from shared source/nuisance pixels.
def compare_boxcar(result):
    if "boxcar" not in result["methods"]:
        return {}
    horne, boxcar = result["methods"]["horne"], result["methods"]["boxcar"]
    valid = (horne["flags"] == 0) & (boxcar["flags"] == 0)
    difference = np.where(valid, horne["flux"] - boxcar["flux"], np.nan)
    error = np.full(len(valid), np.nan)
    # Weight differences account for correlation; neither extraction is treated as an absolute reference.
    for column in np.flatnonzero(valid):
        weights = horne["weights"][:, column] - boxcar["weights"][:, column]
        rows = np.flatnonzero(weights != 0)
        design = nuisance_design(result, rows, column)
        vector = weights[rows] @ design
        variance = np.sum((weights[rows] * result["input_error"][rows, column]) ** 2)
        variance += vector @ result["nuisance_covariance"][column] @ vector
        if np.isfinite(variance) and variance >= 0:
            error[column] = np.sqrt(variance)
    return {"BOXCAR_FLUX": np.where(valid, boxcar["flux"], np.nan),
            "HORNE_MINUS_BOXCAR": difference, "METHOD_DIFFERENCE_SIGMA": error}


# Build advisory-only profile and flux diagnostics without mutating any confirmed extraction arrays.
def analyze_horne_qa(result, args):
    validate_profile_settings(args)
    reference = result["methods"].get("horne")
    if reference is None or reference["profile"].get("status") != "fitted" or not np.any(reference["flags"] == 0):
        return {"report": {"version": __version__, "status": "unavailable", "reason": "No successful Horne profile/extraction."}}
    fit, offsets, observed, error, model = retained_profile(result)
    scale = float(np.median(np.diff(result["spatial"])))
    centers = (result["target"] - result["spatial"][0]) / scale
    trials = trial_profiles(result, args, fit, centers)
    fluxes = np.full((len(trials), len(centers)), np.nan)
    fluxes[0] = np.where(reference["flags"] == 0, reference["flux"], np.nan)
    configurations = []
    seen = []
    # Evaluate only changed, positive-width profiles; invalid or duplicate trials remain explicitly unavailable.
    for index, trial in enumerate(trials):
        duplicate = any(trial["sigma_pixel"] == width and np.array_equal(trial["centers"], position)
                        for width, position in seen)
        status, reason = "reference" if index == 0 else "succeeded", ""
        if duplicate or not np.isfinite(trial["sigma_pixel"]) or not fit["width_bounds_pixel"][0] <= trial["sigma_pixel"] <= fit["width_bounds_pixel"][1]:
            status, reason = "skipped", "Duplicate profile or trial width outside the original fit bounds."
        else:
            seen.append((trial["sigma_pixel"], trial["centers"]))
            if index:
                fluxes[index], _ = fixed_aperture_horne(result, trial["centers"], trial["sigma_pixel"])
                if not np.any(np.isfinite(fluxes[index])):
                    status, reason = "failed", "No valid trial measurements on reference support."
        displacement = trial["centers"] - centers
        configurations.append({"name": trial["name"], "sigma_pixel": trial["sigma_pixel"],
            "fwhm_pixel": trial["sigma_pixel"] * FWHM_FACTOR, "offset_pixel_minmax": [float(np.min(displacement)), float(np.max(displacement))],
            "basis": trial["basis"], "status": status, "reason": reason,
            "valid_samples": int(np.count_nonzero(np.isfinite(fluxes[index])))})
    statistics = summarize_alternatives(fluxes[1:], reference["error"])
    goodness = profile_goodness(result, centers, fit["sigma_pixel"])
    # Aggregate conditional residual evidence and account nominally for the one estimated shared width.
    dof = int(np.sum(goodness["DOF"])) - 1
    chi2 = float(np.nansum(goodness["CHI2"]))
    pooled, sequences = {}, {}
    for key in ("ASYMMETRY_SCORE", "WING_SCORE"):
        values = goodness[key][np.isfinite(goodness[key])]
        pooled[key] = float(np.sum(values) / np.sqrt(len(values))) if len(values) else None
        # Adjacent native samples can expose sign-changing chromatic structure that cancels in a pooled sum.
        sequence = goodness[key]
        adjacent = np.isfinite(sequence[:-1]) & np.isfinite(sequence[1:])
        left, right = sequence[:-1][adjacent], sequence[1:][adjacent]
        correlation = None
        if len(left) >= 8 and np.std(left) > 0 and np.std(right) > 0:
            correlation = float(np.corrcoef(left, right)[0, 1])
        sequences[key] = {"adjacent_pairs": len(left), "lag_correlation": correlation,
                          "median_absolute_score": float(np.median(abs(values))) if len(values) else None}
    structured = any(value is not None and abs(value) > ADVISORY_LIMITS["structure_score"] for value in pooled.values())
    structured |= any(item["lag_correlation"] is not None and
        item["lag_correlation"] > ADVISORY_LIMITS["residual_lag_correlation"] and
        item["median_absolute_score"] > ADVISORY_LIMITS["residual_median_score"] for item in sequences.values())
    redchi2 = chi2 / dof if dof > 0 else None
    bad_fit = redchi2 is not None and redchi2 > ADVISORY_LIMITS["reduced_chi2"]
    flags = np.zeros(len(centers), dtype=np.int16)
    available = np.isfinite(fluxes[0])
    flags[~available] |= 1
    if fit["poorly_constrained"]:
        flags[available] |= 2
    if fit["width_near_bound"]:
        flags[available] |= 4
    flags[available & (statistics["BG_RATIO"] > ADVISORY_LIMITS["sensitivity_ratio"])] |= 8
    # Known-star proximity and asymmetric residuals warn of possible contamination, not a source identification.
    near_star = abs(result["star_center"] - result["target"]) < args.aperture_arcsec / 2 + 2 * fit["sigma_arcsec"]
    flags[available & near_star] |= 16
    advisories = {"profile_poorly_constrained": fit["poorly_constrained"], "width_near_bound": fit["width_near_bound"],
        "structured_residuals": bool(structured), "poor_conditional_2d_fit": bool(bad_fit),
        "profile_sensitive": bool(np.any(flags & 8)),
        "possible_neighbor_contamination": bool(np.any(near_star & available) or
            (pooled["ASYMMETRY_SCORE"] is not None and abs(pooled["ASYMMETRY_SCORE"]) > ADVISORY_LIMITS["structure_score"]))}
    report = {"version": __version__, "status": "completed", "profile": fit, "configurations": configurations,
        "advisories": advisories, "advisory_limits": ADVISORY_LIMITS, "qa_flags": PROFILE_FLAGS,
        "goodness": {"chi2": chi2, "nominal_dof": dof, "nominal_reduced_chi2": redchi2,
            "shared_shape_parameters_counted": 1, "source_amplitudes": int(np.count_nonzero(goodness["DOF"])),
            "pooled_residual_scores": pooled,
            "residual_sequences": sequences,
            "covariance": "diag(input variance) + X_source nuisance_covariance X_source.T",
            "degrees_of_freedom": "Per column N_source_pixels-1 (signed GLS amplitude); global sum minus one fitted shared width.",
            "limitations": "Global DOF approximate: shared width was fitted to an interpolated stack, not a joint GLS optimum. Trace fixed; upstream covariance, clipping/selection and shape uncertainty omitted."},
        "ensemble": "Population RMS of valid distinct perturbed profiles, excluding reference; at least two required.",
        "fixed": "Data, variances, physical aperture, sky/star predictions and nuisance covariance remain fixed.",
        "normalization": "Each trial Gaussian normalized to the same fixed physical finite aperture.",
        "interpretation": "Profile-choice sensitivity, not a calibrated systematic uncertainty. Statistical errors unchanged.",
        "contamination": "Asymmetry/proximity is suggestive; centroid mismatch, extended structure or PSF mismatch can mimic a neighbor.",
        "pooled_score_limitations": "Descriptive scores assume independent spectral columns; correlations and shared profile fitting limit calibration.",
        "method_comparison": "Same finite aperture, with shared-pixel/nuisance covariance in the difference error; neither method is absolute truth."}
    spectral = {"TRACE_PIXEL": centers, "TRACE_ARCSEC": result["target"].copy(), "REFERENCE_FLUX": fluxes[0],
        "SIGMA_STAT": np.where(available, reference["error"], np.nan), "REFERENCE_QUALITY": reference["flags"].copy(),
        "PROFILE_SENS": statistics["BG_SENS"], "PROFILE_RATIO": statistics["BG_RATIO"],
        "PROFILE_MIN": statistics["BG_MIN"], "PROFILE_MAX": statistics["BG_MAX"], "N_PROFILES": statistics["N_ACCEPTED"],
        "PROFILE_QA": flags, **goodness, **compare_boxcar(result)}
    for index, trial in enumerate(trials):
        spectral[f"FLUX_{trial['name']}"] = fluxes[index]
        spectral[f"DELTA_{trial['name']}"] = fluxes[index] - fluxes[0]
        spectral[f"CENTER_{trial['name']}_PIXEL"] = trial["centers"]
    # Store the observed stack and reference residuals without pretending its interpolated samples are independent.
    spatial = {"offset_pixel": offsets, "OBSERVED": observed, "FORMAL_ERROR": error, "MODEL": model,
               "RESIDUAL": observed - model}
    for trial in trials[1:]:
        if trial["sigma_pixel"] > 0:
            displacement = float(np.median(trial["centers"] - centers))
            if trial["name"].startswith("trace"):
                displacement = float(np.max(np.abs(trial["centers"] - centers))) * (-1 if trial["name"].endswith("minus") else 1)
            spatial[f"MODEL_{trial['name']}"] = fit["amplitude"] * np.exp(-.5 * ((offsets - displacement) / trial["sigma_pixel"]) ** 2)
    return {"report": report, "spectral": spectral, "spatial": spatial}


# Keep all profile QA artifacts in a separate family with shared overwrite protection.
def profile_qa_paths(prefix):
    return {"report": Path(f"{prefix}_horne_qa.json"), "spectral": Path(f"{prefix}_horne_qa.ecsv"),
            "spatial": Path(f"{prefix}_horne_spatial_qa.ecsv"), "spatial_png": Path(f"{prefix}_horne_spatial_qa.png"),
            "sensitivity_png": Path(f"{prefix}_horne_sensitivity_qa.png")}


# Draw the observed/assumed profile and residuals separately from native-grid spectral sensitivity.
def plot_horne_qa(spectral, spatial, report, paths):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    x = spatial["offset_pixel"]
    axes[0].errorbar(x, spatial["OBSERVED"], yerr=spatial["FORMAL_ERROR"], fmt="o", ms=3, label="Observed aligned stack")
    axes[0].plot(x, spatial["MODEL"], color="black", label="Confirmed Gaussian")
    for key in spatial.colnames:
        if key.startswith("MODEL_"):
            axes[0].plot(x, spatial[key], lw=.8, alpha=.6, label=key.removeprefix("MODEL_"))
    axes[0].set_ylabel("Flux density (input units)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].errorbar(x, spatial["RESIDUAL"], yerr=spatial["FORMAL_ERROR"], fmt="o", ms=3)
    axes[1].axhline(0, color="black", lw=.7)
    axes[1].set(xlabel="Offset from confirmed trace (pixel)", ylabel="Stack residual (input units)")
    figure.suptitle("Horne spatial QA: correlated stack errors are approximate\nTrial display amplitudes fixed; trace curves show maximum drift")
    figure.tight_layout()
    figure.savefig(paths["spatial_png"], dpi=140)
    plt.close(figure)
    # No smoothing, gap filling or automatic science rejection enters any plotted curve.
    wave = spectral["wavelength_angstrom"]
    figure, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    axes[0].fill_between(wave, spectral["PROFILE_MIN"], spectral["PROFILE_MAX"], alpha=.3, label="Perturbed profile range")
    axes[0].plot(wave, spectral["REFERENCE_FLUX"], lw=.8, label="Confirmed Horne")
    if "BOXCAR_FLUX" in spectral.colnames:
        axes[0].plot(wave, spectral["BOXCAR_FLUX"], lw=.8, label="Boxcar, same physical aperture")
    axes[0].set_ylabel("Finite-aperture flux density")
    axes[0].legend(fontsize=8)
    axes[1].plot(wave, spectral["PROFILE_SENS"], lw=.8)
    axes[1].set_ylabel("PROFILE_SENS")
    axes[2].plot(wave, spectral["PROFILE_RATIO"], lw=.8)
    axes[2].set_ylabel("PROFILE_SENS / sigma")
    axes[3].plot(wave, spectral["REDUCED_CHI2"], lw=.8)
    axes[3].set(xlabel="Wavelength (Å)", ylabel="Conditional reduced chi-square")
    figure.suptitle("Horne profile-choice QA: diagnostic sensitivity, not a calibrated uncertainty\n"
                    "Aperture, background, data and variance fixed; advisories never reject science")
    figure.tight_layout()
    figure.savefig(paths["sensitivity_png"], dpi=140)
    plt.close(figure)


# Persist reports and full-grid tables while leaving every science FITS/ASCII/ECSV writer untouched.
def save_horne_qa(diagnostics, result, input_report, args, prefix):
    paths = profile_qa_paths(prefix)
    if not args.overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError("Horne QA products already exist; choose another directory or overwrite.")
    report = {**diagnostics["report"], "input": input_report["input"], "flux_unit": input_report["flux_unit"]}
    products = {"report": str(paths["report"])}
    if report["status"] == "completed":
        scale = result["normalization"]
        # Report fitted amplitudes in physical input units, matching the saved spatial-profile table.
        fit = dict(report["profile"])
        fit["amplitude"] *= scale
        if fit["amplitude_error_approximate"] is not None:
            fit["amplitude_error_approximate"] *= scale
        fit["amplitude_unit"] = input_report["flux_unit"]
        report["profile"] = fit
        unit = u.Unit(input_report["flux_unit"], parse_strict="silent")
        spectral = Table({"wavelength_angstrom": result["wavelength"], **diagnostics["spectral"]})
        spatial = Table(diagnostics["spatial"])
        # Scale dimensional quantities once, keeping positions, flags, counts, ratios and chi-square dimensionless.
        for key in spectral.colnames:
            if key.startswith(("FLUX_", "DELTA_")) or key in (
                    "REFERENCE_FLUX", "SIGMA_STAT", "PROFILE_SENS", "PROFILE_MIN", "PROFILE_MAX",
                    "GLS_AMPLITUDE", "BOXCAR_FLUX", "HORNE_MINUS_BOXCAR", "METHOD_DIFFERENCE_SIGMA"):
                spectral[key] = spectral[key] * scale
                spectral[key].unit = unit
        for key in spatial.colnames:
            if key != "offset_pixel":
                spatial[key] = spatial[key] * scale
                spatial[key].unit = unit
        spectral["wavelength_angstrom"].unit = "Angstrom"
        spectral["TRACE_ARCSEC"].unit = "arcsec"
        spectral.meta = {"version": __version__, "profile_qa_report": paths["report"].name,
                         "interpretation": report["interpretation"], "qa_flags": PROFILE_FLAGS}
        spatial.meta = {"version": __version__, "profile_qa_report": paths["report"].name,
                        "errors": "Formal stack errors; interpolated samples and sky errors can be correlated."}
        spectral.write(paths["spectral"], format="ascii.ecsv", overwrite=args.overwrite)
        spatial.write(paths["spatial"], format="ascii.ecsv", overwrite=args.overwrite)
        plot_horne_qa(spectral, spatial, report, paths)
        products.update({key: str(path) for key, path in paths.items() if key != "report"})
    report["products"] = {key: Path(path).name for key, path in products.items()}
    paths["report"].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return products
