"""Automatic, auditable boxcar extraction from a source location or row aperture."""

import json
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import gaussian_filter1d, median_filter

from . import __version__
from .coordinates import physical_axes, wavelength_axis
from .export import existing_science_exports, plan_science_export, write_science_export
from .extraction import boxcar_columns
from .flux_only import fit_background_column, wavelength_step
from .diagnostics import plot_extraction_regions
from .quick_binning import binned_paths, prepare_binned, save_binned
from .spectrum_plot import plot_spectrum
from .viewer import open_image


# Resolve the ordinary extension names while accepting a 2D primary image without ERR.
def load_image(path, args):
    with fits.open(path, memmap=False) as hdus:
        if args.hdu not in hdus and args.hdu != "FLUX":
            raise ValueError("Requested science HDU does not exist.")
        science_hdu = args.hdu if args.hdu in hdus else 0
        if science_hdu == 0 and args.hdu not in hdus and hdus[0].data is None:
            raise ValueError("No 2D science image in the requested or primary HDU.")
        image = hdus[science_hdu]
        data = np.asarray(image.data, dtype=float)
        if data.ndim != 2:
            raise ValueError("Quick extraction requires a rectified 2D science image.")
        header = image.header.copy()
        if args.error_hdu not in hdus and args.error_hdu != "ERRS":
            raise ValueError("Requested standard-deviation HDU does not exist.")
        error_hdu = args.error_hdu if args.error_hdu in hdus else None
        error = None if error_hdu is None else np.asarray(hdus[error_hdu].data, dtype=float)
        error_header = None if error_hdu is None else hdus[error_hdu].header.copy()
    if error is not None:
        if error.shape != data.shape:
            raise ValueError("Science and error arrays must have the same shape.")
        if error_header.get("ERRTYPE", "STDDEV").upper() not in ("STDDEV", "SIGMA"):
            raise ValueError("Quick extraction requires standard deviations, not variance or inverse variance.")
        if not header.get("BUNIT") or error_header.get("BUNIT") != header.get("BUNIT"):
            raise ValueError("Science and error images must declare matching BUNIT.")
    try:
        wavelength, spatial = physical_axes(header, data.shape, args.dispersion_axis)
        step_arcsec = float(spatial[1] - spatial[0])
    except ValueError:
        wavelength = wavelength_axis(header, data.shape, args.dispersion_axis)
        step_arcsec = None
    if args.dispersion_axis == 0:
        data = data.T
        error = None if error is None else error.T
    if not header.get("BUNIT"):
        raise ValueError("Science image needs a declared BUNIT for a science export.")
    return data, error, wavelength, step_arcsec, header, science_hdu, error_hdu


# Suppress wavelength-dependent sky levels before locating spatial features.
def spatial_profile(data):
    finite = np.isfinite(data)
    columns = np.count_nonzero(finite, axis=0) >= max(5, data.shape[0] // 3)
    if not np.any(columns):
        raise ValueError("Too few finite science columns for automatic region selection.")
    selected = data[:, columns]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        column_sky = np.nanmedian(selected, axis=0)
        profile = np.nanmedian(selected - column_sky[None, :], axis=1)
    if np.count_nonzero(np.isfinite(profile)) < 12:
        raise ValueError("Too few finite spatial rows for automatic region selection.")
    fill = float(np.nanmedian(profile))
    return gaussian_filter1d(np.where(np.isfinite(profile), profile, fill), 1.)


# Estimate the positive feature's half-maximum width without treating noise as a fitted profile.
def feature_width(profile, center):
    ny = len(profile)
    middle = int(round(center))
    radius = max(8, min(40, ny // 10))
    outer = np.r_[profile[max(0, middle - 3 * radius):max(0, middle - radius)],
                  profile[min(ny, middle + radius):min(ny, middle + 3 * radius)]]
    baseline = float(np.median(outer)) if len(outer) else float(np.median(profile))
    amplitude = profile[middle] - baseline
    if amplitude <= 0:
        return None
    threshold = baseline + amplitude / 2
    left, right = middle, middle
    while left > 0 and profile[left - 1] >= threshold:
        left -= 1
    while right < ny - 1 and profile[right + 1] >= threshold:
        right += 1
    width = right - left + 1
    return float(width) if 2 <= width <= ny // 4 else None


# Follow a bright reference trace in coarse wavelength blocks without fitting a faint target.
def reference_trace(data, reference_row):
    ny, nx = data.shape
    block = max(80, nx // 50)
    locations, centers = [], []
    for start in range(0, nx, block):
        stop = min(nx, start + block)
        with np.errstate(all="ignore"):
            profile = np.nanmedian(data[:, start:stop], axis=1)
        low, high = max(1, reference_row - 4), min(ny - 1, reference_row + 5)
        if not np.any(np.isfinite(profile[low:high])):
            continue
        center = low + int(np.nanargmax(profile[low:high]))
        if abs(center - reference_row) > 4:
            continue
        locations.append((start + stop - 1) / 2)
        centers.append(float(center))
    if len(centers) < 2:
        return np.full(nx, float(reference_row))
    return np.interp(np.arange(nx), locations, centers)


# Choose low-feature sky intervals independently on each side of the source.
def choose_sky(profile, source_center, aperture_width):
    ny = len(profile)
    width = max(5, int(np.ceil(.7 * aperture_width)))
    gap = max(2, int(np.ceil(.35 * aperture_width)))
    broad = median_filter(profile, size=max(11, 4 * width + 1), mode="nearest")
    # Penalize persistent positive and negative traces even when a broad filter follows them.
    center_level = float(np.median(profile))
    residual = np.abs(profile - broad) + np.abs(profile - center_level)
    floor = max(float(1.4826 * np.median(np.abs(profile - np.median(profile)))),
                32 * np.finfo(float).eps * max(1., float(np.max(np.abs(profile)))))
    windows = []
    for side in (-1, 1):
        options = []
        max_distance = min(ny // 3, max(5 * width, int(6 * aperture_width)))
        for distance in range(int(np.ceil(aperture_width / 2)) + gap,
                              max_distance - width + 1):
            start = (int(np.floor(source_center - distance - width + 1)) if side < 0
                     else int(np.ceil(source_center + distance)))
            stop = start + width
            if start < 1 or stop > ny - 1:
                continue
            sample = residual[start:stop]
            score = float(np.median(sample) + .6 * np.percentile(sample, 85)
                          + .08 * floor * distance / max(aperture_width, 1.))
            options.append((score, start, stop))
        if options:
            _, start, stop = min(options)
            windows.append((start, stop))
    if not windows:
        raise ValueError("No source-free sky window fits inside the image.")
    return windows


# Apply the selected row aperture and background with no synthetic pixel variances.
def unweighted_boxcar(data, target, aperture_width, sky_windows, degree, sigma):
    ny, nx = data.shape
    rows = np.arange(ny, dtype=float)
    flux, background = np.full(nx, np.nan), np.full(nx, np.nan)
    quality = np.zeros(nx, dtype=np.int16)
    source_weights = np.zeros((ny, nx), dtype=float)
    background_used = np.zeros((ny, nx), dtype=bool)
    for column in range(nx):
        delta = rows - target[column]
        overlap = np.maximum(0., np.minimum(rows + .5, target[column] + aperture_width / 2)
                             - np.maximum(rows - .5, target[column] - aperture_width / 2))
        source = overlap > 1e-10
        source_weights[:, column] = overlap
        if (target[column] - aperture_width / 2 < -.5
                or target[column] + aperture_width / 2 > ny - .5
                or not np.all(np.isfinite(data[source, column]))):
            quality[column] = 2
            continue
        sky = np.zeros(ny, bool)
        for low, high in sky_windows:
            sky |= (delta >= low) & (delta < high)
        sky &= ~source
        fitted = fit_background_column(data[sky, column], rows[sky], target[column], degree, sigma)
        if fitted is None:
            quality[column] = 4
            continue
        coefficients, xscale, used = fitted
        prediction = coefficients[0] + ((rows - target[column]) / xscale * coefficients[1]
                                         if degree else 0.)
        flux[column] = overlap[source] @ (data[source, column] - prediction[source])
        background[column] = overlap[source] @ prediction[source]
        background_used[np.flatnonzero(sky)[used], column] = True
    source_weights[:, quality != 0] = 0.
    return {"flux": flux, "background": background, "quality": quality,
            "source_weights": source_weights, "background_used": background_used}


# Determine exact or measured aperture geometry from one of the four location choices.
def select_geometry(data, profile, args, step_arcsec):
    ny = data.shape[0]
    if args.source_rows is not None:
        start, stop = args.source_rows
        if stop > ny:
            raise ValueError("Source rows exceed the image height.")
        center = (start + stop - 1) / 2
        width = float(stop - start)
        trace = np.full(data.shape[1], center)
        kind = "exact supplied source rows"
    elif args.guess_row is not None:
        if not np.isfinite(args.guess_row) or not 0 <= args.guess_row < ny:
            raise ValueError("Guess row must lie inside the image.")
        near = np.arange(max(0, int(args.guess_row) - 3), min(ny, int(args.guess_row) + 4))
        center = float(near[np.argmax(profile[near])])
        measured = feature_width(profile, center)
        width = float(max(3, int(np.ceil(1.8 * measured)))) if measured else 6.
        trace = np.full(data.shape[1], center)
        kind = "profile width" if measured else "six-pixel fallback width"
    else:
        if step_arcsec is None and args.offset_arcsec is not None:
            raise ValueError("--offset-arcsec needs angular spatial WCS or documented PIXSCALE.")
        edge = max(2, int(.03 * ny))
        star = edge + int(np.argmax(profile[edge:-edge]))
        shift = (args.offset_pixel if args.offset_pixel is not None
                 else args.offset_arcsec / step_arcsec)
        center = float(star + shift)
        if not 0 <= center < ny:
            raise ValueError("Offset places the target outside the image.")
        measured = feature_width(profile, star)
        width = float(max(3, int(np.ceil(1.5 * measured)))) if measured else 6.
        if step_arcsec is not None:
            width = min(width, max(3., args.aperture_arcsec / abs(step_arcsec)))
        trace = reference_trace(data, star) + shift
        kind = "reference-star profile width" if measured else "six-pixel fallback width"
    if center - width / 2 < -.5 or center + width / 2 > ny - .5:
        raise ValueError("Automatic source aperture extends beyond the image.")
    if args.source_rows is None:
        width = min(width, float(max(3, ny // 5)))
    return trace, center, width, kind


# Write the same science family for error-bearing inputs or an explicit flux-only family.
def save_products(prefix, data, wavelength, result, report, error, args):
    spectrum = Path(str(prefix) + "_spectrum")
    ecsv = Path(str(spectrum) + ".ecsv")
    spectrum_png = Path(str(spectrum) + ".png")
    review = Path(str(prefix) + "_review.json")
    figure = Path(str(prefix) + "_background.png")
    regions = Path(str(prefix) + "_regions.fits")
    protected = [ecsv, spectrum_png, review, figure, regions, *existing_science_exports(spectrum)]
    # Protect the separate binned family and validate its science export before writing native products.
    binned = None
    if args.quick_bin_pixels is not None:
        binned_paths_for_run = binned_paths(prefix, args.quick_bin_pixels)
        protected += [binned_paths_for_run["ecsv"], binned_paths_for_run["png"],
                      *existing_science_exports(binned_paths_for_run["base"])]
    if any(path.exists() for path in protected) and not args.overwrite:
        raise FileExistsError("Quick products already exist; use another output directory or --overwrite.")
    unit = report["flux_unit"]
    # Validate spectral WCS and science export feasibility before writing any companion file.
    if args.quick_bin_pixels is not None:
        binned = prepare_binned(wavelength, result, args.quick_bin_pixels, unit, prefix)
    plan = (plan_science_export(wavelength, result["flux"], result["error"],
                                result["quality"], unit, spectrum, gap_policy=args.gap_policy)
            if error is not None else None)
    wave = wavelength if error is None else None
    linear_step = wavelength_step(wave) if wave is not None else None
    table = {"wavelength_angstrom": wavelength, "flux": result["flux"],
             "background": result["background"], "quality": result["quality"]}
    if error is not None:
        table["error"] = result["error"]
    product = Table(table)
    product["wavelength_angstrom"].unit = "Angstrom"
    product.meta = {"version": __version__, "mode": "quick boxcar", "input": report["input"],
                    "flux_unit": unit, "uncertainty": report["uncertainty"]}
    product.write(ecsv, format="ascii.ecsv", overwrite=args.overwrite)
    if error is not None:
        science = write_science_export(plan, {"METHOD": "BOXCAR", "SPECKEX": __version__,
                                              "CONFIRM": True, "ERRINPUT": "STDDEV assumed"}, args.overwrite)
    else:
        # Reuse the validated wavelength step and converted flux in both no-error exports.
        from .export import flux_units
        factor, output_unit = flux_units(unit)
        converted_flux = result["flux"] * factor
        science_fits = Path(str(spectrum) + ".fits")
        science_txt = Path(str(spectrum) + ".txt")
        header = fits.Header()
        header["SPECKEX"], header["ERRTYPE"] = __version__, "NONE"
        header["BUNIT"], header["CUNIT1"], header["CTYPE1"] = output_unit, "Angstrom", "WAVE"
        header["CRPIX1"], header["CRVAL1"], header["CDELT1"] = 1., float(wave[0]), linear_step
        header["QUALHDU"] = "QUALITY"
        fits.HDUList([fits.PrimaryHDU(converted_flux, header=header),
                      fits.ImageHDU(result["quality"], name="QUALITY")]).writeto(
                          science_fits, overwrite=args.overwrite, checksum=True)
        np.savetxt(science_txt, np.column_stack([wave, converted_flux]),
                   fmt="%.8e", header=f"WAVELENGTH_ANGSTROM FLUX\nFlux unit: {output_unit}\nNo statistical error is available.")
        science = [{"fits": str(science_fits), "ascii": str(science_txt)}]
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(result["source_weights"], name="SOURCE_WEIGHTS"),
                  fits.ImageHDU(result["background_used"].astype(np.uint8), name="BACKGROUND_USED")]).writeto(
                      regions, overwrite=args.overwrite, checksum=True)
    report["science_exports"] = science
    report["products"] = {"ecsv": str(ecsv), "review": str(review),
                           "background_png": str(figure), "spectrum_png": str(spectrum_png),
                           "regions_fits": str(regions)}
    review.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot_extraction_regions(data, wavelength, result, report, figure)
    plot_spectrum(wavelength, result["flux"], result.get("error"), result["quality"],
                  unit, spectrum_png, Path(report["input"]).name)
    # Make the saved binned measurements the selected plot while retaining native science files.
    if binned is not None:
        report["products"]["binned"] = save_binned(binned, report, overwrite=args.overwrite)
        review.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report["products"], science


# Run a single quick extraction without touching the review-driven extraction products.
def run_one(path, args):
    data, error, wavelength, step_arcsec, header, science_hdu, error_hdu = load_image(path, args)
    profile = spatial_profile(data)
    trace, center, width, aperture_reason = select_geometry(data, profile, args, step_arcsec)
    sky = choose_sky(profile, center, width)
    windows = [(low - center, high - center) for low, high in sky]
    degree = min(args.background_degree, 1 if len(sky) == 2 else 0)
    if error is None:
        result = unweighted_boxcar(data, trace, width, windows, degree, args.background_sigma)
        uncertainty = "unavailable: no input standard-deviation image"
    else:
        finite = np.isfinite(error) & (error > 0)
        if not np.any(finite):
            raise ValueError("Error image has no finite positive standard deviations.")
        scale = float(np.median(error[finite]))
        dummy = np.full(data.shape[1], -1e6)
        measured = boxcar_columns(data / scale, error / scale, np.arange(data.shape[0], dtype=float),
                                  trace, dummy, np.ones_like(dummy), np.full_like(dummy, 2.5),
                                  width, windows, degree, star_exclude=0.,
                                  background_sigma=args.background_sigma)
        result = {"flux": measured["flux"] * scale, "error": measured["error"] * scale,
                  "background": measured["background"] * scale, "quality": measured["flags"],
                  "source_weights": measured["source_weights"],
                  "background_used": measured["background_used"]}
        uncertainty = "propagated input 1-sigma and fitted-sky covariance; excludes upstream covariance and trace/sky-selection systematics"
    if args.wave_ranges:
        included = np.zeros(len(wavelength), dtype=bool)
        for low, high in args.wave_ranges:
            included |= (wavelength >= low) & (wavelength < high)
        result["quality"][~included] = 1
        result["flux"][~included] = np.nan
        result["background"][~included] = np.nan
        if error is not None:
            result["error"][~included] = np.nan
        result["source_weights"][:, ~included] = 0.
        result["background_used"][:, ~included] = False
    if not np.any(result["quality"] == 0):
        raise ValueError("Automatic source and sky regions produced no valid columns.")
    counts = {str(int(flag)): int(count) for flag, count in zip(*np.unique(result["quality"], return_counts=True))}
    report = {"version": __version__, "input": str(path), "mode": "quick boxcar",
              "science_hdu": science_hdu, "error_hdu": error_hdu,
              "error_interpretation": "assumed 1-sigma standard deviation" if error is not None else "none",
              "flux_unit": header["BUNIT"], "source_location_kind": aperture_reason,
              "source_rows_half_open": args.source_rows,
              "source_center_original_row": center, "aperture_width_pixels": width,
              "spatial_step_arcsec_per_original_row": step_arcsec,
              "background_rows_half_open": sky,
              "background_windows_relative_pixels": windows,
              "background_degree": degree, "quality_counts": counts,
              "wave_ranges_angstrom": args.wave_ranges,
              "uncertainty": uncertainty,
              "limitations": "Automatic sky selection uses a wavelength-collapsed spatial profile and can miss wavelength-dependent contamination; inspect the 2D mask and spectrum."}
    prefix = args.output_dir / (path.stem + "_quick")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return save_products(prefix, data, wavelength, result, report, error, args)


# Route the quick option after the shared extract parser has validated location syntax.
def main(args, cli):
    if sum(value is not None for value in (args.guess_row, args.offset_arcsec,
                                           args.offset_pixel, args.source_rows)) != 1:
        cli.error("Quick extraction needs exactly one location: -gr/-s, -oa, -op, or -sr.")
    if args.method != "boxcar" or args.substars or args.horne_qa or args.background_sensitivity:
        cli.error("Quick extraction supports boxcar only, without star subtraction or optional QA trials.")
    if args.preview_only or args.confirm or args.background_windows != [(-1.5, -.9), (.9, 1.5)]:
        cli.error("Quick extraction chooses sky and confirms automatically; omit review and background-window options.")
    if args.gap_policy == "zero":
        cli.error("Quick extraction currently uses split FITS/TXT for measured errors and full-grid FITS/TXT without errors.")
    # Reject invalid pixel-bin sizes before inspecting any output names.
    if args.quick_bin_pixels is not None and args.quick_bin_pixels < 2:
        cli.error("--bin-pixels/-b requires at least two wavelength pixels.")
    if len({path.stem for path in args.images}) != len(args.images):
        cli.error("Input stems must differ to protect output names.")
    # Reject every pre-existing product before the first member of a multi-image run is written.
    if not args.overwrite:
        for path in args.images:
            prefix = args.output_dir / (path.stem + "_quick")
            spectrum = Path(str(prefix) + "_spectrum")
            fixed = [Path(str(prefix) + suffix) for suffix in
                     ("_review.json", "_background.png", "_regions.fits",
                      "_spectrum.ecsv", "_spectrum.png")]
            if args.quick_bin_pixels is not None:
                requested = binned_paths(prefix, args.quick_bin_pixels)
                fixed += [requested["ecsv"], requested["png"],
                          *existing_science_exports(requested["base"])]
            if any(item.exists() for item in [*fixed, *existing_science_exports(spectrum)]):
                cli.error("Quick products already exist; use another output directory or --overwrite.")
    try:
        for path in args.images:
            products, science = run_one(path, args)
            print(f"Saved quick boxcar: {products['ecsv']}; {len(science)} FITS/TXT science pair(s).")
            print(f"Saved 1D spectrum plot: {products['spectrum_png']}")
            # An explicit bin request selects the saved binned plot for the default viewer.
            selected_plot = products["spectrum_png"]
            if "binned" in products:
                selection = products["binned"]
                selected_plot = selection["spectrum_png"]
                print(f"Saved binned spectrum: {selection['ecsv']}; "
                      f"{len(selection['science_exports'])} FITS/TXT science pair(s); "
                      f"{selection['bin_pixels']} pixels per bin "
                      f"({selection['bin_width_angstrom']:.6g} Å for this file).")
                print(f"Saved binned 1D spectrum plot: {selected_plot}")
            if args.graphic is not False:
                opened, message = open_image(selected_plot)
                print(("Viewer: " if opened else "Viewer skipped: ") + message)
    except (ValueError, OSError, KeyError, IndexError, TypeError) as exc:
        cli.error(str(exc))
