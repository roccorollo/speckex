"""Reproducible continuum-candidate search near a bright spectral trace."""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import gaussian_filter, median_filter
from scipy.optimize import least_squares

from . import __version__
from .cli import hdu_selector
from .coordinates import physical_axes


# Model the bright star's core and wings with a Moffat plus a linear residual sky.
def moffat_model(y, parameters):
    amplitude, center, width, beta, background, slope = parameters
    return amplitude * (1 + ((y - center) / width) ** 2) ** (-beta) + background + slope * y


# Form common-column-weighted profiles with independent-pixel errors and explicit missing rows.
def weighted_profile(data, error, good, weights):
    norm = np.sum(good * weights, axis=1)
    profile = np.full(len(data), np.nan)
    uncertainty = np.full(len(data), np.nan)
    np.divide(np.sum(np.where(good, data, 0) * weights, axis=1), norm,
              out=profile, where=norm > 0)
    np.divide(np.sqrt(np.sum((np.where(good, error, 0) * weights) ** 2, axis=1)),
              norm, out=uncertainty, where=norm > 0)
    return profile, uncertainty


# Fit a continuum-bin star profile while protecting the full target search neighborhood.
def fit_star(profile, error, peak, scale, expected, radius, edge):
    y = np.arange(len(profile), dtype=float)
    separation = (y - peak) * scale
    valid = np.isfinite(profile) & np.isfinite(error) & (error > 0)
    valid &= (y >= edge) & (y < len(y) - edge)
    valid &= np.abs(separation - expected) > radius + .35
    if valid.sum() < 10:
        return None
    # Normalize optimizer inputs through data preparation; constrain the center near the known star.
    initial = [max(profile[int(peak)], .01), peak, 2., 2.5, np.nanmedian(profile[valid]), 0.]
    fit = least_squares(lambda par: ((moffat_model(y, par) - profile) / error)[valid],
                        initial, bounds=([0, peak - 3, .4, 1., -np.inf, -np.inf],
                                         [np.inf, peak + 3, 15., 10., np.inf, np.inf]),
                        loss="soft_l1", max_nfev=1500)
    if not fit.success or fit.x[0] / np.nanmedian(error[valid]) < 10:
        return None
    return fit.x


# Fit a Gaussian template together with a local background, allowing signed source amplitude.
def template_fit(spatial, profile, error, center, sigma, window, degree=1):
    valid = np.isfinite(profile) & np.isfinite(error) & (error > 0)
    valid &= (spatial >= window[0]) & (spatial <= window[1])
    if valid.sum() < degree + 5:
        return np.nan, np.nan
    # Project out a constant/linear (or quadratic sensitivity-check) background by weighted least squares.
    x = spatial[valid] - center
    template = np.exp(-.5 * (x / sigma) ** 2)
    design = np.column_stack([template] + [x ** order for order in range(degree + 1)])
    weighted = design / error[valid, None]
    if np.linalg.matrix_rank(weighted) < design.shape[1]:
        return np.nan, np.nan
    covariance = np.linalg.inv(weighted.T @ weighted)
    coefficients = np.linalg.lstsq(weighted, profile[valid] / error[valid], rcond=None)[0]
    return float(coefficients[0]), float(np.sqrt(covariance[0, 0]))


# Robustly stack independent spectral-bin profiles; errors are formal and not a detection calibration.
def stack_profiles(profiles, errors, indices=None):
    if indices is not None:
        profiles, errors = profiles[indices], errors[indices]
    good = np.isfinite(profiles) & np.isfinite(errors) & (errors > 0)
    weight = np.where(good, 1 / np.maximum(errors, 1e-100) ** 2, 0)
    norm = weight.sum(axis=0)
    mean, error = np.full(profiles.shape[1], np.nan), np.full(profiles.shape[1], np.nan)
    np.divide(np.sum(np.where(good, profiles, 0) * weight, axis=0), norm,
              out=mean, where=norm > 0)
    np.divide(1., np.sqrt(norm), out=error, where=norm > 0)
    return mean, error


# Search a grid of target locations and widths on the stacked profile without positivity bias in amplitudes.
def scan_profile(spatial, profile, error, expected, radius, degree=1):
    centers = np.arange(expected - radius, expected + radius + .001, .025)
    widths = np.array([.22, .30, .40, .55])
    window = (expected - radius - .7, expected + radius + .7)
    candidates = []
    for center in centers:
        for sigma in widths:
            amplitude, uncertainty = template_fit(spatial, profile, error, center, sigma, window, degree)
            if np.isfinite(amplitude) and np.isfinite(uncertainty) and uncertainty > 0:
                candidates.append((amplitude / uncertainty, center, sigma, amplitude, uncertainty))
    if not candidates:
        raise ValueError("No usable target templates within the available spatial rows.")
    return max(candidates), window


# Measure a candidate using wavelength blocks to expose sensitivity to spectral residuals and background choices.
def assess_candidate(spatial, profiles, errors, expected, radius, bootstrap, seed):
    mean, uncertainty = stack_profiles(profiles, errors)
    best, window = scan_profile(spatial, mean, uncertainty, expected, radius)
    formal_score, center, sigma, amplitude, amplitude_error = best
    # Bootstrap complete spectral bins, preserving the spatial structure within each wavelength block.
    rng = np.random.default_rng(seed)
    amplitudes, positions = [], []
    center_grid = np.arange(max(expected - radius, center - .5),
                            min(expected + radius, center + .5) + .001, .05)
    for _ in range(bootstrap):
        indices = rng.integers(0, len(profiles), len(profiles))
        sample, sample_error = stack_profiles(profiles, errors, indices)
        value, _ = template_fit(spatial, sample, sample_error, center, sigma, window)
        amplitudes.append(value)
        fits_at_center = []
        for c in center_grid:
            trial_amplitude, trial_error = template_fit(spatial, sample, sample_error, c, sigma, window)
            fits_at_center.append((trial_amplitude / trial_error, c))
        positions.append(max(fits_at_center)[1])
    # Repeat with a more flexible local background and without any Moffat subtraction in the caller.
    quadratic = scan_profile(spatial, mean, uncertainty, expected, radius, degree=2)[0]
    block_amplitudes = [template_fit(spatial, p, e, center, sigma, window)[0]
                        for p, e in zip(profiles, errors)]
    standard_error = float(np.std(amplitudes, ddof=1))
    return {
        "offset_arcsec": float(center), "template_sigma_arcsec": float(sigma),
        "template_fwhm_arcsec": float(2.354820045 * sigma),
        "amplitude": float(amplitude), "formal_amplitude_error": float(amplitude_error),
        "formal_template_score": float(formal_score),
        "block_bootstrap_amplitude_std": standard_error,
        "amplitude_to_block_scatter": float(amplitude / standard_error) if standard_error > 0 else None,
        "bootstrap_offset_percentiles_arcsec": np.percentile(positions, [16, 50, 84]).tolist(),
        "quadratic_background_offset_arcsec": float(quadratic[1]),
        "quadratic_background_amplitude": float(quadratic[3]),
        "positive_bin_fraction": float(np.mean(np.asarray(block_amplitudes) > 0)),
        "at_search_position_boundary": bool(abs(center - (expected - radius)) < .013
                                             or abs(center - (expected + radius)) < .013),
        "at_template_width_boundary": bool(sigma in (.22, .55)),
        "status": "candidate; scores are not calibrated detection significances",
    }, mean, uncertainty


# Prepare clipped continuum bins, track the star, and align each bin to its own fitted stellar center.
def analyze_image(path, args):
    with fits.open(path, memmap=False) as hdus:
        data = np.array(hdus[args.hdu].data, dtype=float, copy=True)
        error = np.array(hdus[args.error_hdu].data, dtype=float, copy=True)
        header = hdus[args.hdu].header.copy()
        error_header = hdus[args.error_hdu].header.copy()
    wavelength, spatial = physical_axes(header, data.shape, args.dispersion_axis)
    if error.shape != data.shape:
        raise ValueError("Science and standard-deviation arrays must have identical shapes.")
    # Require declared nonempty units; two missing values do not establish matching science/error units.
    science_unit, error_unit = header.get("BUNIT"), error_header.get("BUNIT")
    if (not isinstance(science_unit, str) or not science_unit.strip()
            or not isinstance(error_unit, str) or not error_unit.strip() or error_unit != science_unit):
        raise ValueError("Science and standard-deviation arrays must declare matching BUNIT.")
    if args.dispersion_axis == 0:
        data, error = data.T, error.T
    # Sort physical axes into increasing order for interpolation while retaining original pixel mapping.
    xorder, yorder = np.argsort(wavelength), np.argsort(spatial)
    wavelength, spatial = wavelength[xorder], spatial[yorder]
    data, error = data[yorder][:, xorder], error[yorder][:, xorder]
    good = np.isfinite(data) & np.isfinite(error) & (error > 0)
    if not np.any(good):
        raise ValueError("No finite science pixels with positive standard deviations.")
    normalization = float(np.median(error[good]))
    data, error = data / normalization, error / normalization
    ny, nx = data.shape
    scale = float(np.median(np.diff(spatial)))
    edge = max(2, int(np.ceil(ny * .05)))
    # Continuum-only clipping suppresses isolated spectral spikes; it is not an emission-line search.
    continuum = median_filter(np.where(good, data, 0), size=(1, 51), mode="nearest")
    clipped = good & (np.abs(data - continuum) > args.clip_sigma * error)
    good &= ~clipped
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        column_noise = np.nanmedian(np.where(good[edge:-edge], error[edge:-edge], np.nan), axis=0)
    # Ignore invalid columns without allowing NaNs to dominate the common spectral weights.
    invalid = ~np.isfinite(column_noise) | (column_noise <= 0)
    column_noise[invalid] = np.inf
    finite_noise = column_noise[np.isfinite(column_noise)]
    if len(finite_noise) < 20:
        raise ValueError("Too few usable dispersion columns.")
    threshold = np.percentile(finite_noise, args.noise_percentile)
    weights = np.where(column_noise <= threshold, 1 / column_noise ** 2, 0)
    weights = np.minimum(weights, np.percentile(weights[weights > 0], 95))
    # Restrict continuum localization to user-selected physical wavelength intervals.
    wave_ranges = getattr(args, "wave_ranges", None)
    if wave_ranges:
        included = np.zeros(nx, dtype=bool)
        for low, high in wave_ranges:
            included |= (wavelength >= low) & (wavelength < high)
        weights[~included] = 0
        if np.count_nonzero(weights) < 20:
            raise ValueError("Selected wavelength intervals contain too few usable columns.")
    whole, _ = weighted_profile(data, error, good, weights)
    peak = edge + int(np.nanargmax(whole[edge:-edge]))
    offsets = (np.arange(ny) - peak) * scale
    rows, raw_profiles, residual_profiles, aligned_errors = [], [], [], []
    # Use complete native-pixel blocks; record exact Angstrom edges and rejected bins.
    boundaries = np.r_[wavelength[0] - (wavelength[1] - wavelength[0]) / 2,
                       (wavelength[:-1] + wavelength[1:]) / 2,
                       wavelength[-1] + (wavelength[-1] - wavelength[-2]) / 2]
    for first in range(0, nx, args.bin_pixels):
        last = min(first + args.bin_pixels, nx)
        start, stop = float(boundaries[first]), float(boundaries[last])
        selected = np.zeros(nx, dtype=bool)
        selected[first:last] = True
        usable_columns = int(np.sum(weights[selected] > 0))
        if usable_columns < 10:
            rows.append({"start_angstrom": start, "stop_angstrom": stop,
                         "start_pixel": first, "stop_pixel": last, "pixels_in_bin": last - first,
                         "accepted": False,
                         "reason": "fewer than ten noise-selected columns"})
            continue
        profile, uncertainty = weighted_profile(data[:, selected], error[:, selected],
                                                good[:, selected], weights[selected])
        star = fit_star(profile, uncertainty, peak, scale, args.offset_arcsec,
                        args.search_radius_arcsec, edge)
        if star is None:
            rows.append({"start_angstrom": start, "stop_angstrom": stop,
                         "start_pixel": first, "stop_pixel": last, "pixels_in_bin": last - first,
                         "accepted": False,
                         "reason": "stellar fit failed or star amplitude/error below ten"})
            continue
        # Interpolate onto a common star-relative grid and propagate interpolation weights, not errors linearly.
        center_world = float(np.interp(star[1], np.arange(ny), spatial))
        relative = spatial - center_world
        # Subtract only one stellar component, preserving the sky for explicit background estimation.
        subtract_star = getattr(args, "substars", False)
        model = moffat_model(np.arange(ny), [*star[:4], 0., 0.]) if subtract_star else np.zeros(ny)
        raw = np.interp(offsets, relative, profile, left=np.nan, right=np.nan)
        residual = np.interp(offsets, relative, profile - model, left=np.nan, right=np.nan)
        right = np.clip(np.searchsorted(relative, offsets, side="right"), 1, ny - 1)
        fraction = (offsets - relative[right - 1]) / (relative[right] - relative[right - 1])
        interp_error = np.sqrt((1 - fraction) ** 2 * uncertainty[right - 1] ** 2
                               + fraction ** 2 * uncertainty[right] ** 2)
        interp_error[(offsets < relative[0]) | (offsets > relative[-1])] = np.nan
        # Save original zero-based row and the star's measured width for inspection across wavelength.
        star_fwhm = 2 * star[2] * np.sqrt(2 ** (1 / star[3]) - 1) * scale
        rows.append({"start_angstrom": start, "stop_angstrom": stop,
                     "start_pixel": first, "stop_pixel": last, "pixels_in_bin": last - first,
                     "accepted": True,
                     "star_pixel": float(np.interp(center_world, spatial, yorder)),
                     "star_spatial_arcsec": center_world, "star_fwhm_arcsec": float(star_fwhm),
                     "usable_columns": usable_columns, "star_parameters": star.tolist()})
        raw_profiles.append(raw)
        residual_profiles.append(residual)
        aligned_errors.append(interp_error)
    if len(raw_profiles) < 8:
        raise ValueError("At least eight valid wavelength bins are required for the candidate search.")
    # Estimate continuum candidates with both stellar-wing treatments to reveal model dependence.
    raw_profiles, residual_profiles, aligned_errors = map(np.asarray,
                                                         (raw_profiles, residual_profiles, aligned_errors))
    candidate, mean, mean_error = assess_candidate(offsets, residual_profiles, aligned_errors,
        args.offset_arcsec, args.search_radius_arcsec, args.bootstrap, args.seed)
    if getattr(args, "substars", False):
        raw_candidate, raw_mean, _ = assess_candidate(offsets, raw_profiles, aligned_errors,
            args.offset_arcsec, args.search_radius_arcsec, max(30, args.bootstrap // 3), args.seed)
    else:
        raw_candidate, raw_mean = dict(candidate), mean.copy()
    centers = np.array([r["star_spatial_arcsec"] for r in rows if r["accepted"]])
    star_world = float(np.median(centers))
    candidate_world = star_world + candidate["offset_arcsec"]
    candidate["representative_original_row_zero_based"] = float(np.interp(candidate_world, spatial, yorder))
    candidate["representative_spatial_coordinate_arcsec"] = candidate_world
    # Compare alternating independent spectral bins and contiguous wavelength quartiles at the fixed candidate.
    band_results = []
    accepted = [r for r in rows if r["accepted"]]
    for label, indices in [("alternating_A", np.arange(0, len(accepted), 2)),
                           ("alternating_B", np.arange(1, len(accepted), 2))] + [
            (f"quartile_{i + 1}", ix) for i, ix in enumerate(np.array_split(np.arange(len(accepted)), 4))]:
        band, band_error = stack_profiles(residual_profiles, aligned_errors, indices)
        score, window = scan_profile(offsets, band, band_error, args.offset_arcsec, args.search_radius_arcsec)
        fixed_amplitude, fixed_error = template_fit(offsets, band, band_error,
            candidate["offset_arcsec"], candidate["template_sigma_arcsec"], window)
        band_results.append({"label": label, "start_angstrom": accepted[indices[0]]["start_angstrom"],
            "stop_angstrom": accepted[indices[-1]]["stop_angstrom"], "bins": len(indices),
            "best_offset_arcsec": float(score[1]), "fixed_position_amplitude": fixed_amplitude,
            "fixed_position_formal_error": fixed_error})
    # Characterize adjacent spectral noise off both sources; this ratio is diagnostic, not an error rescaling.
    background_rows = (np.arange(ny) >= edge) & (np.arange(ny) < ny - edge)
    background_rows &= np.abs(offsets) > 1.5
    background_rows &= np.abs(offsets - args.offset_arcsec) > args.search_radius_arcsec + .35
    adjacent_good = good[:, 1:] & good[:, :-1] & background_rows[:, None]
    denominator = np.sqrt(error[:, 1:] ** 2 + error[:, :-1] ** 2)
    adjacent = np.full_like(denominator, np.nan)
    np.divide(data[:, 1:] - data[:, :-1], denominator, out=adjacent,
              where=adjacent_good & np.isfinite(denominator) & (denominator > 0))
    diagnostic_noise = adjacent[adjacent_good & np.isfinite(adjacent)]
    # An unavailable optional diagnostic is JSON null, not a reason to discard otherwise usable analysis.
    diagnostic_scatter = (float(1.4826 * np.median(np.abs(diagnostic_noise - np.median(diagnostic_noise))))
                          if diagnostic_noise.size else None)
    report = {
        "version": __version__, "input": str(path), "science_hdu": args.hdu,
        "error_hdu": args.error_hdu, "error_interpretation": "ASSUMED standard deviation; header does not specify statistical type",
        "settings": {k: v for k, v in vars(args).items() if k not in ("images", "output_dir")},
        "wavelength_first_last_angstrom": wavelength[[0, -1]].tolist(),
        "wavelength_step_angstrom": float(np.median(np.diff(wavelength))),
        "original_spatial_step_arcsec": float((spatial[1] - spatial[0]) * np.sign(yorder[1] - yorder[0])),
        "spatial_first_last_arcsec": spatial[[0, -1]].tolist(),
        "spatial_unit": "arcsec", "flux_unit": header.get("BUNIT", "unknown"),
        "internal_flux_scale": normalization, "amplitude_unit": "flux_unit times internal_flux_scale",
        "star_median_spatial_coordinate_arcsec": star_world,
        "star_median_original_row_zero_based": float(np.interp(star_world, spatial, yorder)),
        "bins_accepted": len(accepted), "bins_total": len(rows),
        "clipped_pixel_fraction": float(np.mean(clipped)),
        "noise_selected_column_fraction": float(np.mean(weights > 0)),
        "adjacent_background_difference_robust_std": diagnostic_scatter,
        "candidate": candidate, "without_stellar_subtraction": raw_candidate,
        "stellar_subtraction": bool(getattr(args, "substars", False)),
        "independent_band_checks": band_results, "spectral_bins": rows,
        "limitations": ["Continuum search: spectral clipping can reject genuine narrow lines.",
            "Formal errors omit spatial/spectral resampling covariance and star-model uncertainty.",
            "Block bootstrap captures bin-to-bin variation, not persistent spatial artifacts or all correlations.",
            "Position and width were searched; scores have no trials correction and are not detection significances.",
            "Positive spatial offsets mean increasing slit coordinate, not verified celestial north.",
            "No air/vacuum wavelength conversion or flux-density unit conversion was applied."],
    }
    return report, (wavelength, spatial, data, error, good, offsets, raw_profiles,
                    residual_profiles, aligned_errors, mean, mean_error, raw_mean)


# Plot smoothed continuum maps and independently measured profiles in physical units.
def plot_diagnostics(report, arrays, output):
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm

    wavelength, spatial, data, error, good, offsets, raw, residuals, errors, mean, mean_error, raw_mean = arrays
    accepted = [r for r in report["spectral_bins"] if r["accepted"]]
    centers = np.array([(r["start_angstrom"] + r["stop_angstrom"]) / 2 for r in accepted])
    candidate = report["candidate"]
    # Retain missing spectral bins as masked pixels; never interpolate across telluric gaps for the map.
    full_centers = np.array([(r["start_angstrom"] + r["stop_angstrom"]) / 2 for r in report["spectral_bins"]])
    image_raw = np.full((len(offsets), len(full_centers)), np.nan)
    image_residual = image_raw.copy()
    indices = np.searchsorted(full_centers, centers)
    image_raw[:, indices], image_residual[:, indices] = raw.T, residuals.T
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), gridspec_kw={"width_ratios": [2.2, 1]})
    # Apply normalized display smoothing only; candidate measurements use unsmoothed binned profiles.
    processed_title = ("One Moffat star removed; sky retained" if report["stellar_subtraction"]
                       else "Aligned continuum; stellar subtraction OFF")
    for ax, image, title in [(axes[0, 0], image_raw, "Continuum image; bright star retained"),
                              (axes[1, 0], image_residual, processed_title)]:
        valid = np.isfinite(image)
        norm = gaussian_filter(valid.astype(float), (.65, .7))
        smoothed = gaussian_filter(np.where(valid, image, 0), (.65, .7)) / np.maximum(norm, 1e-10)
        smoothed[~valid] = np.nan
        center = candidate["offset_arcsec"]
        background = smoothed[(offsets > center - 1.5) & (offsets < center + 1.5)]
        limit = max(float(np.nanpercentile(np.abs(background), 85)), .02)
        extent = [report["spectral_bins"][0]["start_angstrom"], report["spectral_bins"][-1]["stop_angstrom"],
                  offsets[0] - np.diff(offsets)[0] / 2, offsets[-1] + np.diff(offsets)[0] / 2]
        artist = ax.imshow(smoothed, origin="lower", aspect="auto", extent=extent,
                           cmap="RdBu_r", norm=AsinhNorm(linear_width=limit / 3, vmin=-limit, vmax=limit))
        ax.axhline(0, color="black", lw=.8)
        ax.axhline(candidate["offset_arcsec"], color="limegreen", ls="--", lw=1.3,
                   label=f"Candidate {candidate['offset_arcsec']:+.3f} arcsec")
        ax.set(xlabel="Wavelength (Å)", ylabel="Offset from fitted star (arcsec)", title=title)
        ax.legend(loc="lower left", fontsize=8)
        flux_label = f"Flux / ({report['internal_flux_scale']:.3g} {report['flux_unit']})"
        figure.colorbar(artist, ax=ax, label=flux_label, pad=.01)
    # Plot quartiles separately so a broad-band stack cannot hide a single bad wavelength region.
    for i, ix in enumerate(np.array_split(np.arange(len(raw)), 4)):
        band, _ = stack_profiles(residuals, errors, ix)
        axes[0, 1].plot(offsets, band, label=f"{centers[ix[0]]:.0f}–{centers[ix[-1]]:.0f} Å")
    axes[0, 1].axvline(candidate["offset_arcsec"], color="green", ls="--")
    center = candidate["offset_arcsec"]
    axes[0, 1].set(xlim=(center - 1.7, center + 1.5), title="Independent wavelength quartiles",
                   xlabel="Offset from star (arcsec)", ylabel="Scaled residual flux density")
    axes[0, 1].legend(fontsize=8)
    # Use a target-region vertical stretch so residual star-core errors do not flatten the source.
    region = (offsets > center - 1.5) & (offsets < center + 1.4)
    limits = np.nanpercentile(residuals[:, region], [10, 90])
    pad = max(.02, float(limits[1] - limits[0]) * .25)
    axes[0, 1].set_ylim(limits[0] - pad, limits[1] + pad)
    axes[1, 1].plot(offsets, raw_mean, color=".6", label="No stellar subtraction")
    axes[1, 1].plot(offsets, mean, color="tab:blue", label=("One star subtracted" if report["stellar_subtraction"] else "Unsubtracted stack"))
    axes[1, 1].fill_between(offsets, mean - mean_error, mean + mean_error, alpha=.2)
    axes[1, 1].axvline(candidate["offset_arcsec"], color="green", ls="--")
    axes[1, 1].set(xlim=(center - 1.7, center + 1.5), ylim=(limits[0] - pad, limits[1] + pad),
                   xlabel="Offset from star (arcsec)", ylabel="Scaled flux density",
                   title="Candidate profile; shaded errors are formal")
    axes[1, 1].legend(fontsize=8)
    figure.suptitle(f"{Path(report['input']).name} | candidate {candidate['offset_arcsec']:+.3f} arcsec "
                   f"| row {candidate['representative_original_row_zero_based']:.2f} (zero-based)\n"
                   f"Continuum search; ERRS assumed 1σ; {report['bins_accepted']} accepted bins; "
                   "display smoothed only; candidate is not a confirmed detection", fontsize=12)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


# Parse finite physical intervals for wavelength selections and spatial background windows.
def physical_range(value):
    try:
        low, high = map(float, value.split(":"))
        if not np.isfinite(low + high) or low >= high:
            raise ValueError
        return low, high
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use finite LOW:HIGH with LOW < HIGH.") from exc


# Share search settings with the reviewed extraction workflow.
def build_parser():
    cli = argparse.ArgumentParser(description="Search a faint continuum source beside a bright star in rectified 2D FITS spectra.")
    cli.add_argument("images", nargs="+", type=Path)
    cli.add_argument("--hdu", type=hdu_selector, default="FLUX")
    cli.add_argument("--error-hdu", type=hdu_selector, default="ERRS", help="ASSUMED 1-sigma standard deviation")
    cli.add_argument("--dispersion-axis", type=int, choices=(0, 1), default=1)
    cli.add_argument("--offset-arcsec", "-oa", type=float, default=3., help="Initial target offset from the reference star in arcsec")
    cli.add_argument("--substars", "-ss", action="store_true", help="Subtract one dominant nearby star; OFF by default")
    cli.add_argument("--wave-range", dest="wave_ranges", type=physical_range, action="append",
                     help="Continuum-search LOW:HIGH in Angstrom; repeat for disjoint intervals")
    cli.add_argument("--search-radius-arcsec", type=float, default=1.)
    cli.add_argument("--bin-pixels", "-b", type=int, default=50,
                     help="Adjacent wavelength pixels per bin; with --quick, save and view a binned 1D spectrum")
    cli.add_argument("--clip-sigma", type=float, default=6.)
    cli.add_argument("--noise-percentile", type=float, default=80.)
    cli.add_argument("--bootstrap", type=int, default=200)
    cli.add_argument("--seed", type=int, default=2301)
    cli.add_argument("--output-dir", type=Path, default=Path("search_results"))
    cli.add_argument("--overwrite", action="store_true")
    cli.add_argument("--version", action="version", version=__version__)
    return cli


# Run a standalone search without changing the existing profile CLI or input FITS files.
def main(argv=None):
    cli = build_parser()
    args = cli.parse_args(argv)
    # Validate numerical settings and all planned filenames before writing any products.
    if not (args.bin_pixels >= 2 and 0 < args.search_radius_arcsec and 0 < args.clip_sigma
            and 0 < args.noise_percentile <= 100 and args.bootstrap >= 30
            and np.all(np.isfinite([args.offset_arcsec, args.search_radius_arcsec,
                                   args.clip_sigma, args.noise_percentile]))):
        cli.error("Use finite settings, positive widths/clipping, noise percentile in (0,100], and at least 30 bootstraps.")
    stems = [p.stem for p in args.images]
    if len(stems) != len(set(stems)):
        cli.error("Input stems must be distinct to avoid ambiguous output filenames.")
    products = [args.output_dir / (stem + suffix) for stem in stems for suffix in ("_search.png", "_search.json", "_profiles.ecsv")]
    if not args.overwrite and any(p.exists() for p in products):
        cli.error("Search products already exist; choose another output directory or --overwrite.")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path in args.images:
            report, arrays = analyze_image(path, args)
            prefix = args.output_dir / path.stem
            plot_diagnostics(report, arrays, Path(str(prefix) + "_search.png"))
            Path(str(prefix) + "_search.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            # Export the measured aligned profiles, not just the selected candidate parameters.
            offsets, raw, residuals, errors = arrays[5:9]
            accepted = [r for r in report["spectral_bins"] if r["accepted"]]
            count = len(offsets)
            table = Table({"start_angstrom": np.repeat([r["start_angstrom"] for r in accepted], count),
                           "stop_angstrom": np.repeat([r["stop_angstrom"] for r in accepted], count),
                           "start_pixel": np.repeat([r["start_pixel"] for r in accepted], count),
                           "stop_pixel": np.repeat([r["stop_pixel"] for r in accepted], count),
                           "offset_arcsec": np.tile(offsets, len(raw)), "raw_mean": raw.ravel(),
                           "residual_mean": residuals.ravel(), "formal_error": errors.ravel()})
            table["start_angstrom"].unit = "Angstrom"
            table["stop_angstrom"].unit = "Angstrom"
            table.meta = {"version": __version__, "input": str(path),
                          "internal_flux_scale": report["internal_flux_scale"],
                          "flux_unit": report["flux_unit"],
                          "warning": "Interpolation introduces covariance; formal errors omit it."}
            table.write(str(prefix) + "_profiles.ecsv", format="ascii.ecsv", overwrite=args.overwrite)
            print(json.dumps({"input": str(path), "bins": report["bins_accepted"], "candidate": report["candidate"]}))
    except (ValueError, OSError, KeyError, IndexError) as exc:
        cli.error(str(exc))


# Enable direct module execution as a separate analysis workflow.
if __name__ == "__main__":
    main()
