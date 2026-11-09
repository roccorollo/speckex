"""Bin quick spectra on complete groups of adjacent wavelength pixels."""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from . import __version__
from .binning import bin_spectrum, linear_coordinates
from .export import flux_units, plan_science_export, write_science_export
from .flux_only import wavelength_step
from .spectrum_plot import plot_spectrum


# Name the separate binned product family by its exact integer pixel count.
def binned_paths(prefix, bin_pixels):
    base = Path(str(prefix) + f"_spectrum_bin{bin_pixels}")
    return {"base": base, "ecsv": Path(str(base) + ".ecsv"),
            "png": Path(str(base) + ".png"),
            "fits": Path(str(base) + ".fits"), "txt": Path(str(base) + ".txt")}


# Average complete native-pixel groups on a linear Angstrom wavelength grid.
def prepare_binned(wavelength_angstrom, result, bin_pixels, flux_unit, prefix):
    if isinstance(bin_pixels, (bool, np.bool_)) or not isinstance(bin_pixels, (int, np.integer)) or bin_pixels < 2:
        raise ValueError("--bin-pixels/-b requires an integer of at least two pixels.")
    wave, step = linear_coordinates(wavelength_angstrom)
    count = len(wave) // bin_pixels
    if count < 2:
        raise ValueError("Pixel bin size leaves fewer than two complete output bins.")
    paths = binned_paths(prefix, bin_pixels)
    used = count * bin_pixels
    shape = (count, bin_pixels)
    flux = np.asarray(result["flux"], dtype=float)
    background = np.asarray(result["background"], dtype=float)
    quality = np.asarray(result["quality"], dtype=np.int16)
    error = None if "error" not in result else np.asarray(result["error"], dtype=float)
    if any(value.shape != wave.shape for value in (flux, background, quality)) or (
            error is not None and error.shape != wave.shape):
        raise ValueError("Quick binning needs matching one-dimensional spectrum arrays.")

    # A flagged or nonfinite native sample invalidates its whole output bin.
    valid = (quality == 0) & np.isfinite(flux) & np.isfinite(background)
    if error is not None:
        valid &= np.isfinite(error) & (error > 0)
    good = valid[:used].reshape(shape).all(axis=1)
    original_flags = np.bitwise_or.reduce(quality[:used].reshape(shape), axis=1)
    binned_quality = np.where(good, 0, np.where(original_flags != 0, original_flags, 1)).astype(np.int16)
    binned_background = np.where(good, background[:used].reshape(shape).mean(axis=1), np.nan)
    if not np.any(good):
        raise ValueError("No complete valid wavelength bins remain for quick binning.")

    # Propagate only actual input standard deviations; errorless bins stay errorless.
    if error is not None:
        measured = bin_spectrum(wave, np.where(valid, flux, 0.),
                                np.where(valid, error, 0.), bin_pixels, gap_policy="zero")
        binned_wave = measured["wavelength_angstrom"]
        binned_flux = np.where(good, measured["flux"], np.nan)
        binned_error = np.where(good, measured["error"], np.nan)
        science_plan = plan_science_export(binned_wave, binned_flux, binned_error,
                                           binned_quality, flux_unit, paths["base"])
        output_step = None
    else:
        binned_wave = wave[:used].reshape(shape).mean(axis=1)
        binned_flux = np.where(good, flux[:used].reshape(shape).mean(axis=1), np.nan)
        binned_error = None
        science_plan = None
        flux_units(flux_unit)
        output_step = wavelength_step(binned_wave)
    return {"paths": paths, "wavelength_angstrom": binned_wave, "flux": binned_flux,
            "error": binned_error, "background": binned_background,
            "quality": binned_quality, "science_plan": science_plan,
            "output_step": output_step, "bin_pixels": bin_pixels,
            "bin_width_angstrom": abs(step) * bin_pixels,
            "input_pixels": len(wave), "used_pixels": used,
            "dropped_pixels": len(wave) - used}


# Save every binned sample, the IRAF science family, and its matching figure.
def save_binned(binned, report, overwrite=False):
    paths = binned["paths"]
    has_error = binned["error"] is not None
    columns = {"wavelength_angstrom": binned["wavelength_angstrom"], "flux": binned["flux"],
               "background": binned["background"], "quality": binned["quality"]}
    if has_error:
        columns["error"] = binned["error"]
    table = Table(columns)
    table["wavelength_angstrom"].unit = "Angstrom"
    table.meta = {"version": __version__, "mode": "quick boxcar binned",
                  "input": report["input"], "flux_unit": report["flux_unit"],
                  "bin_pixels": binned["bin_pixels"],
                  "bin_width_angstrom": binned["bin_width_angstrom"],
                  "dropped_pixels": binned["dropped_pixels"],
                  "uncertainty": ("Independent wavelength samples assumed; upstream covariance omitted"
                                  if has_error else "Unavailable: no input standard-deviation image")}
    table.write(paths["ecsv"], format="ascii.ecsv", overwrite=overwrite)

    # Preserve standard-deviation FITS extensions only for error-bearing inputs.
    if has_error:
        metadata = {"METHOD": "BOXCAR", "SPECKEX": __version__, "CONFIRM": True,
                    "BINFACT": binned["bin_pixels"], "BINWANG": binned["bin_width_angstrom"],
                    "NDROP": binned["dropped_pixels"], "ERRMODEL": "INDEPENDENT"}
        science = write_science_export(binned["science_plan"], metadata, overwrite,
            history=["Mean of complete adjacent native wavelength-pixel groups.",
                     "Errors use sqrt(sum(input sigma squared))/number of pixels.",
                     "Assumes independent wavelength samples; upstream covariance omitted."],
            ascii_notes=["Complete native-pixel means; missing bins remain missing.",
                         "Independent wavelength-sample error propagation; covariance omitted."])
    else:
        factor, unit = flux_units(report["flux_unit"])
        header = fits.Header()
        header["SPECKEX"], header["ERRTYPE"] = __version__, "NONE"
        header["BUNIT"], header["CUNIT1"], header["CTYPE1"] = unit, "Angstrom", "WAVE"
        header["CRPIX1"], header["CRVAL1"] = 1., float(binned["wavelength_angstrom"][0])
        header["CDELT1"] = binned["output_step"]
        header["QUALHDU"] = "QUALITY"
        header["BINFACT"], header["BINWANG"] = binned["bin_pixels"], binned["bin_width_angstrom"]
        header["NDROP"] = binned["dropped_pixels"]
        header["HISTORY"] = "Mean of complete native wavelength-pixel groups; no formal error."
        fits.HDUList([fits.PrimaryHDU(binned["flux"] * factor, header=header),
                      fits.ImageHDU(binned["quality"], name="QUALITY")]).writeto(
                          paths["fits"], overwrite=overwrite, checksum=True)
        np.savetxt(paths["txt"], np.column_stack([binned["wavelength_angstrom"],
                                                   binned["flux"] * factor]), fmt="%.8e",
                   header=f"WAVELENGTH_ANGSTROM FLUX\nFlux unit: {unit}\nNo statistical error is available.\n"
                          f"Bin size {binned['bin_pixels']} native pixels; width {binned['bin_width_angstrom']:g} Angstrom.")
        science = [{"fits": str(paths["fits"]), "ascii": str(paths["txt"])}]

    # The plotted values are exactly the saved binned measurements.
    plot_spectrum(binned["wavelength_angstrom"], binned["flux"], binned["error"],
                  binned["quality"], report["flux_unit"], paths["png"],
                  f"{Path(report['input']).name} | {binned['bin_pixels']} pixel bins",
                  show_medians=False)
    return {"ecsv": str(paths["ecsv"]), "spectrum_png": str(paths["png"]),
            "science_exports": science, "bin_pixels": binned["bin_pixels"],
            "bin_width_angstrom": binned["bin_width_angstrom"],
            "input_pixels": binned["input_pixels"], "used_pixels": binned["used_pixels"],
            "dropped_pixels": binned["dropped_pixels"],
            "valid_bins": int(np.sum(binned["quality"] == 0))}
