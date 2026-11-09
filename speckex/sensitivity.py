"""Optional local-background comparisons with the confirmed source weights held fixed."""

from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Table

from . import __version__
from .extraction import boxcar_columns, validate_windows


# These QA-only flags describe unavailable comparisons; they never alter science quality flags.
QA_FLAGS = {1: "reference unavailable", 2: "invalid configuration", 4: "background fit failed",
            8: "insufficient sky coverage or geometry", 16: "duplicate configuration"}


# Validate perturbation controls independently of the standard extraction settings.
def validate_sensitivity_settings(args):
    shift = getattr(args, "bg_shift_arcsec", .1)
    fraction = getattr(args, "bg_width_fraction", .2)
    if not np.isfinite(shift) or shift < 0 or not np.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("Background QA needs a finite shift >= 0 and width fraction in [0,1).")


# Generate named, reproducible alternatives; one-sided estimates use constants to avoid slope extrapolation.
def background_configurations(args):
    validate_sensitivity_settings(args)
    windows = [tuple(map(float, window)) for window in args.background_windows]
    shift = getattr(args, "bg_shift_arcsec", .1)
    fraction = getattr(args, "bg_width_fraction", .2)
    degree = args.background_degree
    configurations = [{"name": "reference", "windows_arcsec": windows, "degree": degree}]
    # Move each side relative to the source, or change widths about their existing midpoints.
    for name, amount in (("outward", shift), ("inward", -shift)):
        changed = [(low + (1 if low + high > 0 else -1) * amount,
                    high + (1 if low + high > 0 else -1) * amount)
                   for low, high in windows]
        configurations.append({"name": name, "windows_arcsec": changed, "degree": degree})
    for name, factor in (("wider", 1 + fraction), ("narrower", 1 - fraction)):
        changed = (windows if fraction == 0 else
                   [((low + high) / 2 - (high - low) * factor / 2,
                     (low + high) / 2 + (high - low) * factor / 2) for low, high in windows])
        configurations.append({"name": name, "windows_arcsec": changed, "degree": degree})
    # Test the other polynomial degree on the original windows, and each available side separately.
    configurations.append({"name": "constant" if degree else "linear",
                           "windows_arcsec": windows, "degree": 1 - degree})
    for name, side in (("left_only", -1), ("right_only", 1)):
        changed = [(low, high) for low, high in windows if np.sign(low + high) == side]
        configurations.append({"name": name, "windows_arcsec": changed, "degree": 0})
    return configurations


# Require retained sky samples and full requested windows; linear fits must bracket the source on both sides.
def supported_background(fitted, result, configuration):
    spatial, target = result["spatial"], result["target"]
    scale = float(np.median(np.diff(spatial)))
    supported = np.ones(len(target), dtype=bool)
    for low, high in configuration["windows_arcsec"]:
        supported &= ((target + low >= spatial[0] - scale / 2)
                      & (target + high <= spatial[-1] + scale / 2))
    used = fitted["background_used"]
    delta = spatial[:, None] - target
    if configuration["degree"] == 1:
        supported &= (np.sum(used & (delta < 0), axis=0) >= 2)
        supported &= (np.sum(used & (delta > 0), axis=0) >= 2)
    else:
        supported &= np.sum(used, axis=0) >= 3
    return supported


# Summarize available alternatives with a population RMS; fewer than two cannot measure sensitivity.
def summarize_alternatives(fluxes, sigma):
    available = np.isfinite(fluxes)
    count = np.sum(available, axis=0)
    mean = np.full(len(sigma), np.nan)
    np.divide(np.sum(np.where(available, fluxes, 0.), axis=0), count, out=mean, where=count > 0)
    # Sum only actual samples; missing wavelengths and failed configurations receive no replacement flux.
    square = np.sum(np.where(available, (fluxes - mean) ** 2, 0.), axis=0)
    scatter = np.full(len(sigma), np.nan)
    np.divide(square, count, out=scatter, where=count >= 2)
    scatter = np.sqrt(scatter)
    ratio = np.full(len(sigma), np.nan)
    np.divide(scatter, sigma, out=ratio, where=np.isfinite(sigma) & (sigma > 0))
    low = np.min(np.where(available, fluxes, np.inf), axis=0)
    high = np.max(np.where(available, fluxes, -np.inf), axis=0)
    low[count == 0], high[count == 0] = np.nan, np.nan
    return {"BG_MEAN": mean, "BG_MIN": low, "BG_MAX": high,
            "BG_SENS": scatter, "BG_RATIO": ratio, "N_ACCEPTED": count}


# Refit only nuisance backgrounds, applying the exact confirmed boxcar/Horne weights without profile fitting.
def analyze_background_sensitivity(result, args):
    configurations = background_configurations(args)
    nx = len(result["wavelength"])
    methods = {}
    # Keep independent method validity and copy arrays so QA can never mutate a confirmed extraction.
    for name, reference in result["methods"].items():
        valid = ((reference["flags"] == 0) & np.isfinite(reference["flux"])
                 & np.isfinite(reference["error"]) & (reference["error"] > 0))
        flux = np.full((len(configurations), nx), np.nan)
        flux[0, valid] = reference["flux"][valid]
        quality = np.zeros(flux.shape, dtype=np.int16)
        quality[:, ~valid] = 1
        methods[name] = {"flux": flux, "quality": quality, "valid": valid,
                         "sigma": np.where(valid, reference["error"], np.nan), "configurations": []}
    trace_valid = np.any([item["valid"] for item in methods.values()], axis=0)
    seen = set()
    # Fit each nuisance configuration once for all methods; preserve source, star shapes and clipping policy.
    for index, configuration in enumerate(configurations):
        reason, unavailable, fitted = "", 0, None
        try:
            validate_windows(args.aperture_arcsec, configuration["windows_arcsec"])
            signature = (configuration["degree"], tuple(sorted(configuration["windows_arcsec"])))
            if signature in seen:
                unavailable, reason = 16, "Identical to an earlier configuration."
            else:
                seen.add(signature)
            if index and not unavailable and np.any(trace_valid):
                fitted = boxcar_columns(result["data"], result["input_error"], result["spatial"],
                    result["target"], result["star_center"], result["star_width"], result["star_beta"],
                    args.aperture_arcsec, configuration["windows_arcsec"], configuration["degree"],
                    args.substars, args.star_exclude_arcsec, args.background_sigma,
                    trace_valid=trace_valid, mask=result.get("mask"))
                supported = supported_background(fitted, result, configuration)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            unavailable, reason = 2, str(exc)
        # Reference validity always bounds the comparisons, including wavelengths where Horne failed.
        for name, item in methods.items():
            valid = item["valid"]
            if unavailable:
                item["quality"][index, valid] |= unavailable
            elif index and fitted is not None:
                item["quality"][index, valid & (fitted["flags"] != 0)] |= 4
                item["quality"][index, valid & ~supported] |= 8
                accepted = item["quality"][index] == 0
                weights = result["methods"][name]["weights"]
                # Recompute flux on native pixels with frozen weights; no Specreduce/profile refit is needed.
                residual = result["data"] - fitted["sky_image"] - fitted["star_image"]
                flux = np.sum(weights * np.where(weights != 0, residual, 0.), axis=0)
                item["flux"][index, accepted] = flux[accepted]
                bad = accepted & ~np.isfinite(flux)
                item["quality"][index, bad] |= 4
                item["flux"][index, bad] = np.nan
            count = int(np.count_nonzero(np.isfinite(item["flux"][index])))
            total = int(np.count_nonzero(valid))
            status = ("reference" if index == 0 else "skipped" if unavailable else
                      "failed" if count == 0 else "succeeded" if count == total else "partial")
            if status in ("failed", "partial") and not reason:
                reason = "Unavailable fits or sky coverage on reference wavelengths; see per-sample QA flags."
            item["configurations"].append({**configuration, "status": status, "reason": reason,
                "valid_samples": count, "reference_samples": total,
                "qa_flag_counts": {str(flag): int(np.count_nonzero(item["quality"][index] & flag))
                                   for flag in QA_FLAGS}})
    # Exclude the reference from the ensemble and retain the changing per-wavelength sample count.
    for item in methods.values():
        item["delta_flux"] = item["flux"] - item["flux"][0]
        item.update(summarize_alternatives(item["flux"][1:], item["sigma"]))
    return methods


# Give QA its own protected product family without changing any science-export filenames or formats.
def sensitivity_paths(prefix, methods):
    return {name: {extension: Path(f"{prefix}_{name}_background_sensitivity.{extension}")
                   for extension in ("ecsv", "png")} for name in methods}


# Plot the unsmoothed native-grid reference, alternative range and sensitivity measures with visible gaps.
def plot_background_sensitivity(table, method, path):
    import matplotlib.pyplot as plt

    wave = np.asarray(table["wavelength_angstrom"])
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    # The range is a collection of background choices, not a confidence interval.
    axes[0].fill_between(wave, table["BG_MIN"], table["BG_MAX"], alpha=.3,
                         color="tab:orange", label="Accepted alternative range")
    axes[0].plot(wave, table["REFERENCE_FLUX"], lw=.8, label="Confirmed reference")
    axes[0].set_ylabel(f"Flux density ({table.meta['flux_unit']})")
    axes[0].legend(fontsize=9)
    axes[1].plot(wave, table["BG_SENS"], lw=.8)
    axes[1].set_ylabel("BG_SENS (same flux units)")
    axes[2].plot(wave, table["BG_RATIO"], lw=.8)
    axes[2].set(xlabel="Wavelength (Å)", ylabel="BG_SENS / reference sigma")
    figure.suptitle(f"{method}: background-choice sensitivity\n"
                    "Fixed source weights; diagnostic scatter, not a calibrated systematic uncertainty")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


# Save full-grid per-method tables with configuration provenance and an accompanying diagnostic figure.
def save_background_sensitivity(diagnostics, result, report, args, prefix):
    paths = sensitivity_paths(prefix, diagnostics)
    if not args.overwrite and any(path.exists() for pair in paths.values() for path in pair.values()):
        raise FileExistsError("Background-sensitivity products already exist; choose another directory or overwrite.")
    scale = result["normalization"]
    products = {}
    # Preserve Angstrom coordinates and original flux units in every diagnostic product.
    for name, item in diagnostics.items():
        table = Table({"wavelength_angstrom": result["wavelength"], "TARGET_SPATIAL_ARCSEC": result["target"],
            "REFERENCE_FLUX": item["flux"][0] * scale, "SIGMA_STAT": item["sigma"] * scale,
            "REFERENCE_QUALITY": result["methods"][name]["flags"]})
        for key in ("BG_MEAN", "BG_MIN", "BG_MAX", "BG_SENS", "BG_RATIO", "N_ACCEPTED"):
            table[key] = item[key] * scale if key not in ("BG_RATIO", "N_ACCEPTED") else item[key]
        for index, configuration in enumerate(item["configurations"]):
            label = configuration["name"]
            table[f"FLUX_{label}"] = item["flux"][index] * scale
            table[f"DELTA_{label}"] = item["delta_flux"][index] * scale
            table[f"QA_{label}"] = item["quality"][index]
        table["wavelength_angstrom"].unit = "Angstrom"
        table["TARGET_SPATIAL_ARCSEC"].unit = "arcsec"
        for key in table.colnames:
            if key.startswith(("FLUX_", "DELTA_")) or key in (
                    "REFERENCE_FLUX", "SIGMA_STAT", "BG_MEAN", "BG_MIN", "BG_MAX", "BG_SENS"):
                table[key].unit = u.Unit(report["flux_unit"], parse_strict="silent")
        # Record the ensemble definition and retained assumptions beside every diagnostic value.
        profile = result["methods"][name]["profile"]
        table.meta = {"version": __version__, "input": report["input"], "method": name,
            "review_confirmed": True, "flux_unit": report["flux_unit"],
            "configurations": item["configurations"], "qa_flags": QA_FLAGS,
            "shift_arcsec": getattr(args, "bg_shift_arcsec", .1),
            "width_fraction": getattr(args, "bg_width_fraction", .2),
            "aperture_arcsec": args.aperture_arcsec, "background_sigma": args.background_sigma,
            "star_exclude_arcsec": args.star_exclude_arcsec, "substars": args.substars,
            "fixed_horne_sigma_pixel": None if profile is None else profile.get("sigma_pixel"),
            "source_weights": "Exact confirmed weights; trace, aperture, profile and input errors fixed.",
            "star_model": "Shape fixed; amplitude refitted jointly with sky when subtraction is enabled.",
            "ensemble": "Valid unique alternatives only, excluding reference; population RMS (ddof=0).",
            "minimum_alternatives": 2, "missing": "NaN; no wavelength interpolation or imputation.",
            "interpretation": "Background-choice sensitivity, not a calibrated systematic uncertainty.",
            "statistical_errors": "Unchanged reference errors; no sensitivity added in quadrature.",
            "limitations": "Shared pixels correlate alternatives; masks/clipping do not certify uncontaminated sky."}
        table.write(paths[name]["ecsv"], format="ascii.ecsv", overwrite=args.overwrite)
        plot_background_sensitivity(table, name, paths[name]["png"])
        products[name] = {key: str(path) for key, path in paths[name].items()}
    return products
