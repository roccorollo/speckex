"""Auditable fractional-pixel boxcar extraction with explicitly fitted local sky."""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from . import __version__
from .coordinates import physical_axes


# Validate nonoverlapping background sidebands relative to the target center in arcsec.
def validate_windows(aperture_width, windows):
    if not np.isfinite(aperture_width) or aperture_width <= 0 or not windows:
        raise ValueError("Use a positive aperture width and at least one background window.")
    ordered = sorted(windows)
    for index, (low, high) in enumerate(ordered):
        if not np.isfinite(low + high) or low >= high:
            raise ValueError("Background windows require finite LOW < HIGH.")
        if high > -aperture_width / 2 and low < aperture_width / 2:
            raise ValueError("Background windows must not overlap the source aperture.")
        if index and low < ordered[index - 1][1]:
            raise ValueError("Background windows must not overlap each other.")


# Integrate a finite source aperture and propagate the covariance of a disjoint nuisance-model fit.
def boxcar_columns(data, error, spatial, target, star_center, star_width, star_beta,
                   aperture_width, windows, background_degree=1, sub_stars=False,
                   star_exclude=1.2, background_sigma=5., trace_valid=None, mask=None):
    validate_windows(aperture_width, windows)
    if background_degree not in (0, 1):
        raise ValueError("Background degree must be 0 (constant) or 1 (linear).")
    ny, nx = data.shape
    scale = float(np.median(np.diff(spatial)))
    if scale <= 0 or not np.allclose(np.diff(spatial), scale, rtol=1e-6):
        raise ValueError("Boxcar extraction requires uniformly spaced spatial pixels.")
    valid = np.isfinite(data) & np.isfinite(error) & (error > 0)
    if mask is not None:
        valid &= ~mask
    # Retain masks and nuisance predictions for an exact review of the pixels used in each wavelength column.
    flux, variance, background, stellar = [np.full(nx, np.nan) for _ in range(4)]
    sky_variance, star_variance = np.full(nx, np.nan), np.full(nx, np.nan)
    nuisance_count = background_degree + 1 + int(sub_stars)
    nuisance_covariance = np.full((nx, nuisance_count, nuisance_count), np.nan)
    source_weights = np.zeros((ny, nx))
    background_used, star_used = np.zeros((ny, nx), bool), np.zeros((ny, nx), bool)
    sky_image, star_image = np.full_like(data, np.nan), np.full_like(data, np.nan)
    flags = np.zeros(nx, dtype=np.int16)
    trace_valid = np.ones(nx, bool) if trace_valid is None else trace_valid
    # Fit only background sidebands and, when requested, one reference-star core; exclude all source pixels.
    for column in range(nx):
        if not trace_valid[column] or not np.isfinite(target[column]):
            flags[column] = 1
            continue
        delta = spatial - target[column]
        low, high = target[column] - aperture_width / 2, target[column] + aperture_width / 2
        overlap = np.maximum(0, np.minimum(spatial + scale / 2, high)
                             - np.maximum(spatial - scale / 2, low)) / scale
        source_weights[:, column] = overlap
        source_rows = overlap > 1e-10
        if (low < spatial[0] - scale / 2 or high > spatial[-1] + scale / 2
                or not np.all(valid[source_rows, column])):
            flags[column] = 2
            continue
        bg = np.zeros(ny, bool)
        for left, right in windows:
            bg |= (delta >= left) & (delta <= right)
        bg &= (np.abs(spatial - star_center[column]) >= star_exclude) & ~source_rows
        bg &= valid[:, column]
        core = (np.abs(spatial - star_center[column]) < star_exclude) & ~source_rows
        core &= valid[:, column] & sub_stars
        shape = (1 + ((spatial - star_center[column]) / star_width[column]) ** 2) ** (-star_beta[column])
        design = np.column_stack([delta ** order for order in range(background_degree + 1)]
                                 + ([shape] if sub_stars else []))
        used = bg | core
        coefficients, covariance = None, None
        # Clip outlying nuisance pixels iteratively, keeping source pixels out of all clipping decisions.
        for iteration in range(4):
            if np.sum(used & bg) < background_degree + 2 or np.sum(used) < design.shape[1] + 2:
                break
            weighted = design[used] / error[used, column, None]
            if np.linalg.matrix_rank(weighted) < design.shape[1]:
                break
            covariance = np.linalg.inv(weighted.T @ weighted)
            coefficients = np.linalg.lstsq(weighted, data[used, column] / error[used, column], rcond=None)[0]
            residual = (data[:, column] - design @ coefficients) / np.maximum(error[:, column], 1e-100)
            # Center sky residuals robustly so one extreme pixel cannot reject every otherwise good sky row.
            sky_residual = residual[used & bg]
            median = np.median(sky_residual)
            scatter = max(1., 1.4826 * np.median(np.abs(sky_residual - median)))
            accepted_sky = np.abs(residual - median) <= background_sigma * scatter
            updated = used & np.where(bg, accepted_sky, np.abs(residual) <= background_sigma)
            if np.array_equal(updated, used) or iteration == 3:
                break
            used = updated
            coefficients, covariance = None, None
        if coefficients is None:
            flags[column] = 4
            continue
        # A shared sky estimate correlates aperture pixels: propagate its covariance after summing the design.
        prediction = design @ coefficients
        summed_design = overlap @ design
        flux[column] = overlap[source_rows] @ (data[source_rows, column] - prediction[source_rows])
        variance[column] = np.sum((overlap[source_rows] * error[source_rows, column]) ** 2)
        variance[column] += summed_design @ covariance @ summed_design
        nuisance_covariance[column] = covariance
        sky_vector = summed_design.copy()
        if sub_stars:
            sky_vector[-1] = 0.
        sky_variance[column] = sky_vector @ covariance @ sky_vector
        star_variance[column] = summed_design[-1] ** 2 * covariance[-1, -1] if sub_stars else 0.
        sky_image[:, column] = design[:, :background_degree + 1] @ coefficients[:background_degree + 1]
        star_image[:, column] = coefficients[-1] * shape if sub_stars else 0.
        background[column] = overlap @ sky_image[:, column]
        stellar[column] = overlap @ star_image[:, column]
        background_used[:, column], star_used[:, column] = used & bg, used & core
    # Effective source masks must be empty wherever no usable spectrum value was produced.
    source_weights[:, flags != 0] = 0.
    return {"flux": flux, "error": np.sqrt(variance), "background": background,
            "stellar": stellar, "flags": flags, "source_weights": source_weights,
            "background_used": background_used, "star_used": star_used,
            "sky_image": sky_image, "star_image": star_image,
            "background_error": np.sqrt(sky_variance), "stellar_error": np.sqrt(star_variance),
            "nuisance_covariance": nuisance_covariance}


# Reopen unmodified science pixels; continuum-search clipping must never erase spectral lines during extraction.
def prepare_extraction(path, report, args):
    with fits.open(path, memmap=False) as hdus:
        data = np.array(hdus[args.hdu].data, dtype=float, copy=True)
        error = np.array(hdus[args.error_hdu].data, dtype=float, copy=True)
        header = hdus[args.hdu].header.copy()
    wavelength, spatial = physical_axes(header, data.shape, args.dispersion_axis)
    if args.dispersion_axis == 0:
        data, error = data.T, error.T
    xorder, yorder = np.argsort(wavelength), np.argsort(spatial)
    wavelength, spatial = wavelength[xorder], spatial[yorder]
    data, error = data[yorder][:, xorder], error[yorder][:, xorder]
    normalization = report["internal_flux_scale"]
    data, error = data / normalization, error / normalization
    # Borrow the reference star's chromatic trace; flag wavelengths outside coverage or across long gaps.
    accepted = [r for r in report["spectral_bins"] if r["accepted"]]
    centers = np.array([(r["start_angstrom"] + r["stop_angstrom"]) / 2 for r in accepted])
    parameters = np.array([r["star_parameters"] for r in accepted])
    star_center = np.interp(wavelength, centers, [r["star_spatial_arcsec"] for r in accepted])
    star_width = np.interp(wavelength, centers, parameters[:, 2]) * np.median(np.diff(spatial))
    star_beta = np.interp(wavelength, centers, parameters[:, 3])
    target = star_center + report["candidate"]["offset_arcsec"]
    # Judge search coverage and long gaps in detector pixels, including nonlinear wavelength grids.
    pixel = np.arange(len(wavelength))
    bin_centers_pixel = np.array([(r["start_pixel"] + r["stop_pixel"] - 1) / 2 for r in accepted])
    next_bin = np.clip(np.searchsorted(bin_centers_pixel, pixel), 0, len(accepted) - 1)
    previous_bin = np.clip(next_bin - 1, 0, len(accepted) - 1)
    nearest_pixel = np.minimum(abs(pixel - bin_centers_pixel[next_bin]),
                               abs(pixel - bin_centers_pixel[previous_bin]))
    trace_valid = ((pixel >= accepted[0]["start_pixel"])
                   & (pixel < accepted[-1]["stop_pixel"])
                   & (nearest_pixel <= 3 * args.bin_pixels))
    # Restrict the extracted spectrum to explicitly requested wavelength regions, if any.
    if args.wave_ranges:
        included = np.zeros(len(wavelength), bool)
        for low, high in args.wave_ranges:
            included |= (wavelength >= low) & (wavelength < high)
        trace_valid &= included
    result = boxcar_columns(data, error, spatial, target, star_center, star_width, star_beta,
        args.aperture_arcsec, args.background_windows, args.background_degree, args.substars,
        args.star_exclude_arcsec, args.background_sigma, trace_valid)
    result.update(wavelength=wavelength, spatial=spatial, target=target,
                  star_center=star_center, star_width=star_width, star_beta=star_beta,
                  data=data, input_error=error, normalization=normalization,
                  background_degree=args.background_degree, substars=args.substars)
    # Delegate source weighting to Specreduce while retaining the already-fitted sky and its covariance.
    from .methods import extract_methods
    extract_methods(result, args, report["candidate"]["template_sigma_arcsec"])
    return result


# Save a confirmed spectrum and the exact region masks needed to audit its local-background fit.
def save_extraction(result, report, args, prefix):
    from .export import existing_science_exports, plan_science_export, write_science_export

    # Validate all requested science exports and collision checks before writing a method's native products.
    plans = {}
    for method, extracted in result["methods"].items():
        base = str(prefix) + (f"_{method}_spectrum" if result["extraction_mode"] == "both" else "_spectrum")
        plans[method] = plan_science_export(result["wavelength"], extracted["flux"] * result["normalization"],
            extracted["error"] * result["normalization"], extracted["flags"], report["flux_unit"], base,
            gap_policy=getattr(args, "gap_policy", "split"))
        paths = [Path(base + ".ecsv"), *existing_science_exports(base)]
        if not args.overwrite and any(path.exists() for path in paths):
            raise FileExistsError("Existing spectrum products are protected; choose another directory or overwrite.")
    if Path(str(prefix) + "_regions.fits").exists() and not args.overwrite:
        raise FileExistsError("Existing extraction regions are protected.")
    # Save one explicitly named spectrum for each method; legacy single-mode filenames stay unchanged.
    for method, extracted in result["methods"].items():
        spectrum_path = (str(prefix) + f"_{method}_spectrum.ecsv" if result["extraction_mode"] == "both"
                         else str(prefix) + "_spectrum.ecsv")
        save_spectrum(result, extracted, method, report, args, spectrum_path)
    # Confirmed workflows always add finite IRAF-ready FITS and nine-significant-digit ASCII science files.
    result["science_exports"] = {}
    for method, plan in plans.items():
        metadata = {"ORIGFILE": Path(report["input"]).name, "METHOD": method.upper(),
            "CONFIRM": True, "SUBSTARS": args.substars, "APWIDTH": args.aperture_arcsec,
            "SKYDEG": args.background_degree, "ERRINPUT": "standard deviations assumed"}
        result["science_exports"][method] = write_science_export(plan, metadata, args.overwrite)
    # Export the common fitted sky regions together with each method's actual extraction weights.
    primary = fits.PrimaryHDU()
    primary.header["VERSION"] = __version__
    primary.header["CONFIRM"] = True
    primary.header["SUBSTARS"] = args.substars
    primary.header["EXTMODE"] = result["extraction_mode"]
    hdus = [primary, fits.ImageHDU(result["wavelength"], name="WAVELENGTH"),
            fits.ImageHDU(result["spatial"], name="SPATIAL"),
            fits.ImageHDU(result["source_weights"], name="SOURCE_WEIGHTS"),
            fits.ImageHDU(result["background_used"].astype(np.uint8), name="BACKGROUND_USED"),
            fits.ImageHDU(result["star_used"].astype(np.uint8), name="STAR_FIT_USED")]
    for method, extracted in result["methods"].items():
        hdus.append(fits.ImageHDU(extracted["weights"], name=method.upper() + "_WEIGHTS"))
    hdus[1].header["BUNIT"], hdus[2].header["BUNIT"] = "Angstrom", "arcsec"
    fits.HDUList(hdus).writeto(str(prefix) + "_regions.fits", overwrite=args.overwrite)


# Persist method-specific fluxes, errors and identically weighted sky/star components in original units.
def save_spectrum(result, extracted, method, report, args, path):
    scale = result["normalization"]
    table = Table({"wavelength_angstrom": result["wavelength"], "flux": extracted["flux"] * scale,
                   "error": extracted["error"] * scale, "background": extracted["background"] * scale,
                   "background_error": extracted["background_error"] * scale,
                   "subtracted_star": extracted["stellar"] * scale,
                   "subtracted_star_error": extracted["stellar_error"] * scale,
                   "snr": extracted["flux"] / extracted["error"],
                   "target_spatial_arcsec": result["target"], "quality": extracted["flags"]})
    table["wavelength_angstrom"].unit = "Angstrom"
    table.meta = {"version": __version__, "input": report["input"], "review_confirmed": True,
        "method": method, "specreduce_version": result["specreduce_version"],
        "flux_normalization": "finite aperture; no total-profile aperture correction",
        "flux_unit": report["flux_unit"],
        "substars": args.substars, "aperture_arcsec": args.aperture_arcsec,
        "background_windows_relative_to_target_arcsec": args.background_windows,
        "background_degree": args.background_degree,
        "trace": "reference-star trace plus confirmed constant target offset",
        "quality_flags": "0=usable; 1=no trace/selected coverage; 2=incomplete aperture; 4=background fit failed; 8=method/profile failed",
        "uncertainty": "Propagated 1-sigma: independent input pixels plus fitted nuisance covariance; excludes resampling and source-profile/star-shape/trace uncertainty"}
    if extracted["profile"] is not None:
        table.meta["source_profile"] = {key: value.tolist() if isinstance(value, np.ndarray) else value
                                        for key, value in extracted["profile"].items()}
    if "gaussian_fraction_in_aperture" in extracted:
        table["gaussian_fraction_in_aperture"] = extracted["gaussian_fraction_in_aperture"]
    table.write(path, format="ascii.ecsv", overwrite=args.overwrite)


# Compute display-only medians over consecutive native wavelength-pixel groups.
def display_pixel_medians(wavelength, values, good, bin_pixels):
    centers, medians = [], []
    for first in range(0, len(wavelength), bin_pixels):
        last = min(first + bin_pixels, len(wavelength))
        selected = np.arange(first, last)[good[first:last]]
        centers.append(float(np.mean(wavelength[first:last])))
        medians.append(float(np.median(values[selected])) if len(selected) else np.nan)
    return centers, medians


# Compare profile choices, finite-aperture fluxes and per-pixel S/N without ranking the extraction methods.
def plot_comparison(result, report, args, path):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(14, 10))
    wave = result["wavelength"]
    scale = float(np.median(np.diff(result["spatial"])))
    x = np.linspace(-args.aperture_arcsec, args.aperture_arcsec, 300)
    if "boxcar" in result["methods"]:
        axes[0].plot(x, (np.abs(x) < args.aperture_arcsec / 2).astype(float),
                     label="Boxcar aperture support", color="tab:blue")
    horne = result["methods"].get("horne")
    if horne is not None and horne["profile"]["status"] == "fitted":
        profile = horne["profile"]
        axes[0].errorbar(profile["offset_pixel"] * scale, profile["measured"] / profile["amplitude"],
                         yerr=profile["formal_error"] / profile["amplitude"], fmt="o", label="Aligned target profile")
        axes[0].plot(x, np.exp(-.5 * (x / profile["sigma_arcsec"]) ** 2), color="tab:orange",
                     label=f"Horne Gaussian σ={profile['sigma_arcsec']:.3f} arcsec")
    elif horne is not None:
        axes[0].text(.02, .85, "Horne profile failed: " + horne["profile"]["reason"], transform=axes[0].transAxes, wrap=True)
    axes[0].set(xlabel="Offset from confirmed target trace (arcsec)", ylabel="Relative profile / aperture support",
                title="Source Gaussian and boxcar support; profile/trace uncertainty is not included in errors")
    axes[0].legend(fontsize=8)
    # Preserve raw wavelength sampling; median overlays are display-only and do not redefine S/N.
    displayed_flux, snr_values = [], []
    for method, extracted in result["methods"].items():
        color = "tab:blue" if method == "boxcar" else "tab:orange"
        good = extracted["flags"] == 0
        flux = extracted["flux"]
        snr = flux / extracted["error"]
        axes[1].plot(wave, flux, color=color, lw=.3, alpha=.035)
        axes[1].fill_between(wave, flux - extracted["error"], flux + extracted["error"], color=color, alpha=.015)
        axes[2].plot(wave, snr, color=color, lw=.3, alpha=.05)
        snr_values.extend(snr[np.isfinite(snr)].tolist())
        centers, flux_medians = display_pixel_medians(wave, flux, good, args.bin_pixels)
        _, snr_medians = display_pixel_medians(wave, snr, good, args.bin_pixels)
        axes[1].plot(centers, flux_medians, color=color, label=method)
        axes[2].plot(centers, snr_medians, color=color, label=method)
        displayed_flux.extend(flux_medians)
    values = np.asarray(displayed_flux)
    values = values[np.isfinite(values)]
    if len(values):
        low, high = np.percentile(values, [2, 98])
        pad = max(.02, float(high - low) * .15)
        axes[1].set_ylim(low - pad, high + pad)
    if snr_values:
        low, high = np.percentile(snr_values, [1, 99])
        pad = max(.2, float(high - low) * .1)
        axes[2].set_ylim(low - pad, high + pad)
    axes[1].set(ylabel="Scaled finite-aperture flux", title="Flux ± propagated 1σ (faint bands); solid overlays are display medians")
    axes[2].set(xlabel="Wavelength (Å)", ylabel="S/N per spectral pixel", title="S/N = flux / propagated 1σ; robust display limits; no automatic method selection")
    for ax in axes[1:]:
        ax.set_xlim(wave[0], wave[-1])
        ax.legend(fontsize=8)
    figure.suptitle(f"{report['input']} | mode {result['extraction_mode']} | Specreduce {result['specreduce_version']}\n"
                   f"Flux scale {result['normalization']:.3g} {report['flux_unit']}; same trace, aperture and fitted sky for each method")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


# Visualize the exact sky samples, target aperture, and fitted sky before spectrum confirmation.
def plot_background(result, report, args, path):
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    wave, spatial, target = result["wavelength"], result["spatial"], result["target"]
    step = max(1, len(wave) // 1000)
    sample = np.arange(0, len(wave), step)
    image = gaussian_filter(np.nan_to_num(result["data"], nan=0., posinf=0., neginf=0.),
                            (.5, max(1., step / 2)))[:, sample]
    limit = max(.05, float(np.nanpercentile(np.abs(image), 70)))
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), height_ratios=[2, 1.1, 1.1])
    axes[0].pcolormesh(wave[sample], spatial, image, shading="nearest",
                       cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[0].fill_between(wave, target - args.aperture_arcsec / 2,
                         target + args.aperture_arcsec / 2, color="limegreen", alpha=.3, label="Source aperture")
    for index, (low, high) in enumerate(args.background_windows):
        axes[0].fill_between(wave, target + low, target + high, color="deepskyblue", alpha=.3,
                             label="Requested sky windows" if index == 0 else None)
    axes[0].plot(wave, result["star_center"], color="black", lw=1., label="Reference star")
    axes[0].set(ylabel="Slit coordinate (arcsec)", title="Source and background regions; display smoothing only")
    axes[0].legend(fontsize=8, loc="lower left")
    # Show exact effective samples, including exclusions, failed columns, and sigma-clipped sky pixels.
    used = (result["source_weights"][:, sample] > 0).astype(float)
    used += 2 * result["background_used"][:, sample]
    used += 3 * result["star_used"][:, sample]
    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors = ListedColormap(["white", "limegreen", "deepskyblue", "orange"])
    axes[1].pcolormesh(wave[sample], spatial, used, shading="nearest",
                       cmap=colors, norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], colors.N))
    axes[1].set(ylabel="Slit coordinate (arcsec)", title="Effective samples: green=aperture, blue=sky, orange=optional star fit; white=unused")
    axes[2].plot(wave, result["flux"], lw=.3, alpha=.12, color=".4", label="Unsmoothed spectrum")
    good = result["flags"] == 0
    display_centers, display_sky = display_pixel_medians(wave, result["background"], good, args.bin_pixels)
    _, display_flux = display_pixel_medians(wave, result["flux"], good, args.bin_pixels)
    primary_method = "boxcar" if "boxcar" in result.get("methods", {"boxcar": None}) else "horne"
    axes[2].plot(display_centers, display_sky, lw=1., label=f"{primary_method} sky: {args.bin_pixels} pixel display medians")
    axes[2].plot(display_centers, display_flux, lw=1., label=f"{primary_method} spectrum: {args.bin_pixels} pixel display medians")
    values = np.r_[display_sky, display_flux]
    values = values[np.isfinite(values)]
    if len(values):
        lo, hi = np.percentile(values, [2, 98])
        axes[2].set_ylim(lo - .1 * max(hi - lo, .01), hi + .1 * max(hi - lo, .01))
    axes[2].set(xlabel="Wavelength (Å)", ylabel="Scaled flux density", title="Per-column sky/extraction; median overlays affect display only")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlim(wave[0], wave[-1])
    figure.suptitle(f"{report['input']} | aperture width {args.aperture_arcsec:g} arcsec | mode {result.get('extraction_mode', 'boxcar')} | "
                   f"background degree {args.background_degree} | star subtraction {'ON' if args.substars else 'OFF'}\n"
                   f"Preview only until confirmed; flux scale {result['normalization']:.3g} {report['flux_unit']}")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
