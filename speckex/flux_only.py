"""Explicit row-aperture extraction when a 2D spectrum has no pixel-error image."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from . import __version__
from .cli import hdu_selector, pixel_range
from .coordinates import physical_axes, wavelength_axis
from .spectrum_plot import plot_spectrum
from .viewer import open_image


# Validate zero-based, half-open source and background rows before any extraction work.
def validate_regions(shape, source, background):
    ny, _ = shape
    if (len(source) != 2 or source[0] < 0 or source[1] <= source[0]
            or source[1] > ny or not background):
        raise ValueError("Source and at least one background interval must lie inside the image.")
    occupied = np.zeros(ny, dtype=bool)
    occupied[source[0]:source[1]] = True
    for low, high in background:
        if low < 0 or high <= low or high > ny or np.any(occupied[low:high]):
            raise ValueError("Background intervals must be inside the image and disjoint from each other and the source.")
        occupied[low:high] = True


# Fit only finite background pixels with ordinary least squares and robust residual clipping.
def fit_background_column(values, rows, source_center, degree, clip_sigma):
    finite = np.isfinite(values)
    if finite.sum() < degree + 3:
        return None
    xscale = max(1., float(np.max(np.abs(rows - source_center))))
    design = np.column_stack([np.ones(len(rows))] +
                             ([(rows - source_center) / xscale] if degree == 1 else []))
    flux_scale = float(np.max(np.abs(values[finite])))
    flux_scale = flux_scale if flux_scale > 0 else 1.
    scaled = values / flux_scale
    used = finite.copy()
    # Refit after each rejection; the scatter controls clipping only and is not an uncertainty estimate.
    for _ in range(5):
        if used.sum() < degree + 3:
            return None
        coefficients, _, rank, _ = np.linalg.lstsq(design[used], scaled[used], rcond=None)
        if rank != degree + 1:
            return None
        residual = scaled - design @ coefficients
        center = float(np.median(residual[used]))
        scatter = 1.4826 * float(np.median(np.abs(residual[used] - center)))
        floor = 32 * np.finfo(float).eps * max(1., float(np.max(np.abs(scaled[used]))))
        accepted = finite & (np.abs(residual - center) <= clip_sigma * max(scatter, floor))
        if np.array_equal(accepted, used):
            return coefficients * flux_scale, xscale, used
        used = accepted
    # The last accepted sample set must define the coefficients used for the saved flux.
    if used.sum() < degree + 3:
        return None
    coefficients, _, rank, _ = np.linalg.lstsq(design[used], scaled[used], rcond=None)
    return (coefficients * flux_scale, xscale, used) if rank == degree + 1 else None


# Extract a complete native wavelength grid without creating a statistical error column.
def extract_flux_only(data, source, background, degree=1, clip_sigma=5., excluded=None):
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("Flux-only extraction requires a two-dimensional science image.")
    validate_regions(data.shape, source, background)
    if degree not in (0, 1) or not np.isfinite(clip_sigma) or clip_sigma <= 0:
        raise ValueError("Use background degree 0 or 1 and a positive finite clipping threshold.")
    ny, nx = data.shape
    excluded = np.zeros(nx, dtype=bool) if excluded is None else np.asarray(excluded, dtype=bool)
    if excluded.shape != (nx,):
        raise ValueError("Excluded-column mask must match the dispersion length.")
    sky_rows = np.concatenate([np.arange(low, high) for low, high in background])
    source_rows = np.arange(*source)
    center = (source[0] + source[1] - 1) / 2
    flux = np.full(nx, np.nan)
    summed_sky = np.full(nx, np.nan)
    flags = np.zeros(nx, dtype=np.int16)
    used_count = np.zeros(nx, dtype=np.int32)
    background_used = np.zeros((ny, nx), dtype=bool)
    # Reject invalid source samples rather than scaling a partial spatial aperture.
    for column in range(nx):
        if excluded[column]:
            flags[column] = 1
            continue
        if not np.all(np.isfinite(data[source_rows, column])):
            flags[column] = 2
            continue
        fitted = fit_background_column(data[sky_rows, column], sky_rows, center, degree, clip_sigma)
        if fitted is None:
            flags[column] = 4
            continue
        coefficients, xscale, used = fitted
        source_design = np.column_stack([np.ones(len(source_rows))] +
                                        ([(source_rows - center) / xscale] if degree == 1 else []))
        sky = source_design @ coefficients
        summed_sky[column] = float(np.sum(sky))
        flux[column] = float(np.sum(data[source_rows, column] - sky))
        used_count[column] = int(used.sum())
        background_used[sky_rows[used], column] = True
    return {"flux": flux, "background": summed_sky, "quality": flags,
            "background_pixels": used_count, "background_used": background_used}


# Require a genuinely linear wavelength grid before writing a linear-WCS FITS spectrum.
def wavelength_step(wavelength_angstrom):
    wavelength = np.asarray(wavelength_angstrom, dtype=float)
    if wavelength.ndim != 1 or len(wavelength) < 2 or not np.all(np.isfinite(wavelength)):
        raise ValueError("Science FITS export needs at least two finite wavelengths.")
    step = float((wavelength[-1] - wavelength[0]) / (len(wavelength) - 1))
    expected = wavelength[0] + np.arange(len(wavelength)) * step
    tolerance = max(abs(step) * 1e-7, float(np.max(np.abs(wavelength))) * 1e-12)
    if step == 0 or not np.allclose(wavelength, expected, atol=tolerance, rtol=0):
        raise ValueError("Flux-only FITS export requires a linear wavelength grid; no resampling is performed.")
    return step


# Use the same 2D region, effective-pixel, and spectrum layout as quick extraction.
def plot_diagnostic(data, wavelength, result, source, background, path, input_name="input", degree=1):
    from .diagnostics import plot_extraction_regions

    source_weights = np.zeros(data.shape, dtype=np.uint8)
    source_weights[source[0]:source[1], result["quality"] == 0] = 1
    display_result = {**result, "source_weights": source_weights}
    report = {"input": str(input_name), "background_degree": degree, "version": __version__}
    plot_extraction_regions(data, wavelength, display_result, report, path)


# Read the input header, run extraction, and save flux with explicit missing-uncertainty metadata.
def run(args):
    prefix = Path(args.output_prefix)
    paths = {kind: Path(str(prefix) + "." + kind)
             for kind in ("fits", "ecsv", "txt", "json", "png")}
    if args.plot_spectra or args.graphic:
        paths["spectrum_png"] = Path(str(prefix) + "_spectrum.png")
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("A flux-only product already exists; choose a fresh output prefix.")
    if any(path.resolve() == args.image.resolve() for path in paths.values()):
        raise ValueError("An output path must not replace the input FITS image.")
    # Respect the selected image extension and original NumPy spatial-row coordinates.
    with fits.open(args.image, memmap=False) as hdus:
        image = hdus[args.hdu]
        data = np.asarray(image.data, dtype=float)
        header = image.header.copy()
        extension_names = [hdu.name for hdu in hdus]
    # Pixel-row extraction can continue without an angular spatial scale, but still needs wavelength WCS.
    try:
        wavelength, spatial_arcsec = physical_axes(header, data.shape, args.dispersion_axis)
        spatial_note = ("Angular WCS or documented PIXSCALE read from the FITS header; "
                        "PIXSCALE fallback coordinates are relative to the first spatial row.")
    except ValueError as spatial_error:
        wavelength = wavelength_axis(header, data.shape, args.dispersion_axis)
        spatial_arcsec = None
        spatial_note = "No usable angular spatial scale; extraction used explicit image rows. " + str(spatial_error)
    if args.dispersion_axis == 0:
        data = data.T
    step = wavelength_step(wavelength)
    excluded = np.zeros(data.shape[1], dtype=bool)
    for low, high in args.exclude_columns or []:
        if high > data.shape[1]:
            raise ValueError("Excluded column interval exceeds the dispersion length.")
        excluded[low:high] = True
    result = extract_flux_only(data, args.source_rows, args.background_rows,
                               args.background_degree, args.clip_sigma, excluded)
    if not np.any(result["quality"] == 0):
        raise ValueError("No usable flux columns; inspect the source and background regions.")
    # Preserve the input flux-density unit while recording that the output has no formal error.
    report = {
        "version": __version__, "input": str(args.image),
        "input_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "science_hdu": args.hdu, "input_extensions": extension_names,
        "input_BUNIT": header.get("BUNIT"), "dispersion_axis": args.dispersion_axis,
        "spatial_first_last_arcsec": (None if spatial_arcsec is None else
                                      [float(spatial_arcsec[0]), float(spatial_arcsec[-1])]),
        "spatial_step_arcsec": (None if spatial_arcsec is None else
                                float(spatial_arcsec[1] - spatial_arcsec[0])),
        "spatial_coordinate_note": spatial_note,
        "source_rows_half_open": list(args.source_rows),
        "background_rows_half_open": [list(rows) for rows in args.background_rows],
        "excluded_columns_half_open": [list(rows) for rows in args.exclude_columns or []],
        "background_degree": args.background_degree, "clip_sigma": args.clip_sigma,
        "quality_flags": {"0": "valid", "1": "excluded column", "2": "nonfinite source pixel",
                          "4": "insufficient or unusable background"},
        "quality_counts": {str(int(value)): int(np.sum(result["quality"] == value))
                           for value in np.unique(result["quality"])},
        "uncertainty": "Unavailable: no input error image; unweighted sky fit, no formal variance or S/N",
        "limitations": "Fixed row aperture; source/sky regions require review per image. No aperture correction, source-trace uncertainty, upstream covariance, or background-model systematic error.",
    }
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # Save the full native grid and quality flags; rejected flux samples remain NaN.
    table = Table({"wavelength_angstrom": wavelength, "flux": result["flux"],
                   "background": result["background"], "quality": result["quality"],
                   "background_pixels": result["background_pixels"]})
    table["wavelength_angstrom"].unit = "Angstrom"
    table.meta = {"version": __version__, "flux_unit": header.get("BUNIT", "unknown"),
                  "uncertainty": "unavailable", "input": str(args.image)}
    table.write(paths["ecsv"], format="ascii.ecsv")
    # The FITS primary image is flux only; a separate quality image identifies unusable pixels.
    fits_header = fits.Header()
    fits_header["SPECKEX"] = __version__
    fits_header["ERRTYPE"] = ("NONE", "No statistical error image or error extension")
    if isinstance(header.get("BUNIT"), str) and header["BUNIT"].strip():
        fits_header["BUNIT"] = header["BUNIT"]
    fits_header["CTYPE1"] = "WAVE"
    fits_header["CUNIT1"] = "Angstrom"
    fits_header["CRPIX1"] = 1.
    fits_header["CRVAL1"] = float(wavelength[0])
    fits_header["CDELT1"] = step
    fits_header["QUALHDU"] = ("QUALITY", "Nonzero value means unavailable flux")
    fits_header["HISTORY"] = "Flux-only row extraction; no input variance or formal uncertainty."
    fits_header["HISTORY"] = "Background regions and clipping are recorded in the companion JSON."
    fits.HDUList([fits.PrimaryHDU(result["flux"], header=fits_header),
                  fits.ImageHDU(result["quality"], name="QUALITY")]).writeto(paths["fits"], checksum=True)
    # Keep ASCII at two columns so a quality flag cannot be mistaken for an error estimate.
    flux_unit = header.get("BUNIT", "unknown input unit")
    np.savetxt(paths["txt"], np.column_stack([wavelength, result["flux"]]), fmt="%.8e",
               header=("WAVELENGTH_ANGSTROM FLUX\nFlux unit: " + str(flux_unit)
                       + "\nNo statistical uncertainty is available; inspect companion ECSV quality flags."))
    if "spectrum_png" in paths:
        report["spectrum_plot"] = str(paths["spectrum_png"])
    paths["json"].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot_diagnostic(data, wavelength, result, args.source_rows, args.background_rows,
                    paths["png"], args.image, args.background_degree)
    if "spectrum_png" in paths:
        plot_spectrum(wavelength, result["flux"], None, result["quality"],
                      flux_unit, paths["spectrum_png"], args.image.name)
    return paths, report


# Parse the dedicated flux-only command without altering the error-bearing reviewed workflow.
def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("image", type=Path, help="Rectified 2D science FITS")
    cli.add_argument("--hdu", type=hdu_selector, default=0, help="Science image extension (default: primary)")
    cli.add_argument("--dispersion-axis", type=int, choices=(0, 1), default=1,
                     help="NumPy dispersion axis (default: 1, horizontal)")
    cli.add_argument("--source-rows", type=pixel_range, required=True,
                     help="Zero-based half-open spatial START:STOP; inspect this image first")
    cli.add_argument("--background-rows", type=pixel_range, action="append", required=True,
                     help="Zero-based half-open source-free START:STOP; repeat for multiple bands")
    cli.add_argument("--exclude-columns", type=pixel_range, action="append",
                     help="Known invalid dispersion START:STOP; repeat as needed")
    cli.add_argument("--background-degree", type=int, choices=(0, 1), default=1)
    cli.add_argument("--clip-sigma", type=float, default=5.,
                     help="Robust background-residual clipping threshold (default: 5)")
    cli.add_argument("--output-prefix", type=Path, default=Path("flux_only_spectrum"))
    cli.add_argument("--plot-spectra", "-p", action="store_true",
                     help="Save a dedicated 1D spectrum PNG")
    graphics = cli.add_mutually_exclusive_group()
    graphics.add_argument("--graphic", "-g", dest="graphic", action="store_true",
                          help="Open the 1D PNG with the default system image viewer")
    graphics.add_argument("--no-graphic", dest="graphic", action="store_false")
    cli.set_defaults(graphic=None)
    cli.add_argument("--version", action="version", version=__version__)
    args = cli.parse_args(argv)
    try:
        paths, report = run(args)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        cli.error(str(exc))
    print("Saved flux-only spectrum: " + ", ".join(map(str, paths.values())))
    print("Statistical uncertainty unavailable; " + str(report["quality_counts"]) + " quality counts.")
    if args.graphic:
        opened, message = open_image(paths["spectrum_png"])
        print(("Viewer: " if opened else "Viewer skipped: ") + message)


# Permit direct module execution for test fixtures and environments without an installed script.
if __name__ == "__main__":
    main()
