"""Average complete groups of adjacent spectral pixels without inventing missing errors."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits

from . import __version__
from .export import existing_science_exports, plan_science_export, write_science_export


# Preserve relevant upstream observing and extraction information without copying an obsolete WCS.
PROVENANCE_KEYS = (
    "OBJECT", "ORIGFILE", "METHOD", "APWIDTH", "SKYLOW", "SKYHIGH", "SKYDEG",
    "SUBSTARS", "ERRASSUM", "DATASHA", "EXTRACTV", "ERRINPUT", "APPROVAL",
    "EXPTIME", "TELESCOP", "INSTRUME", "DATE-OBS", "SPECSYS", "WAVETYPE",
)


# Decode a one-axis linear wavelength WCS without assuming nonlinear IRAF coordinates are linear.
def image_wavelength(header, count):
    band_image = header.get("SPECFMT") == "BANDS" and header.get("WCSDIM") == 3
    if (header.get("DC-FLAG", 0) != 0 or (header.get("WCSDIM", 1) != 1 and not band_image)
            or header.get("CTYPE1", "").strip().upper() not in ("LINEAR", "WAVE", "AWAV")
            or header.get("LTM1_1", 1.) != 1. or header.get("LTV1", 0.) != 0.):
        raise ValueError("FITS input requires an untransformed linear wavelength WCS.")
    step = header["CD1_1"] if "CD1_1" in header else header["CDELT1"]
    wavelength = header["CRVAL1"] + (np.arange(count) + 1. - header["CRPIX1"]) * step
    return (wavelength * u.Unit(header["CUNIT1"])).to_value(u.AA)


# Read a native linear spectral image, requiring explicitly identified standard deviations.
def read_fits_spectrum(path):
    with fits.open(path) as hdus:
        header = hdus[0].header
        bands = (header.get("SPECFMT") == "BANDS" and hdus[0].data is not None
                 and hdus[0].data.shape[:2] == (3, 1) and hdus[0].data.ndim == 3)
        if bands:
            if header.get("ERRTYPE") != "STDDEV" or header.get("ERRBAND") != 2 or header.get("QUALBAND") != 3:
                raise ValueError("Primary spectral bands must identify standard deviations and quality.")
            flux, error, quality = [np.array(values[0], dtype=float) for values in hdus[0].data]
            if (not np.all(np.isfinite(quality)) or np.any(quality < 0)
                    or not np.all(quality == np.floor(quality))
                    or not np.array_equal(quality != 0, error == 0)
                    or np.any((quality != 0) & (flux != 0))):
                raise ValueError("Missing spectral-band samples require nonzero quality and flux=error=0.")
            metadata = {key: header[key] for key in PROVENANCE_KEYS if key in header}
            return image_wavelength(header, len(flux)), flux, error, str(u.Unit(header["BUNIT"])), metadata
        if hdus[0].data is None or hdus[0].data.ndim != 1:
            raise ValueError("FITS input must have a 1D primary flux image; tables and 2D images are unsupported.")
        if "ERROR" not in hdus or hdus["ERROR"].data is None:
            raise ValueError("FITS input requires an ERROR image containing standard deviations.")
        error_header = hdus["ERROR"].header
        if (header.get("ERRTYPE", error_header.get("ERRTYPE", "")).strip().upper() != "STDDEV"
                or error_header.get("ERRTYPE", "STDDEV").strip().upper() != "STDDEV"):
            raise ValueError("FITS ERRTYPE must declare STDDEV; variance is not standard deviation.")
        if not header.get("BUNIT") or not error_header.get("BUNIT"):
            raise ValueError("FITS science and ERROR images must declare BUNIT.")
        flux_unit, error_unit = u.Unit(header["BUNIT"]), u.Unit(error_header["BUNIT"])
        if not error_unit.is_equivalent(flux_unit):
            raise ValueError("FITS science and ERROR units are incompatible.")
        flux = np.array(hdus[0].data, dtype=float)
        error = np.array(hdus["ERROR"].data, dtype=float) * error_unit.to(flux_unit)

        # Check any declared error coordinates before treating its pixels as co-registered uncertainties.
        wavelength = image_wavelength(header, len(flux))
        if "CRVAL1" in error_header:
            error_wave = image_wavelength(error_header, len(flux))
            if not np.allclose(wavelength, error_wave, rtol=1e-12, atol=0.):
                raise ValueError("FITS science and ERROR wavelength coordinates must match.")
        metadata = {key: header[key] for key in PROVENANCE_KEYS if key in header}
        if header["CTYPE1"].strip().upper() in ("WAVE", "AWAV"):
            metadata["WAVETYPE"] = header["CTYPE1"].strip().upper()
    return wavelength, flux, error, str(flux_unit), metadata


# Read the documented three-column ASCII layout and recognize units declared by Speckex exports.
def read_ascii_spectrum(path, wavelength_unit=None, flux_unit=None):
    lines = path.read_text().splitlines()
    comments = [line.lstrip()[1:].strip() for line in lines if line.lstrip().startswith("#")]
    wave_declaration = next((line for line in comments if line.startswith("WAVELENGTH_")), "")
    declared_wave = {"WAVELENGTH_ANGSTROM": "Angstrom", "WAVELENGTH_NM": "nm"}.get(
        wave_declaration.split()[0] if wave_declaration else "")
    if wave_declaration and declared_wave is None and wavelength_unit is None:
        raise ValueError("Unrecognized ASCII wavelength label; supply --wavelength-unit explicitly.")
    unit_prefix = "Flux and 1-sigma error unit: "
    declared_flux = next((line[len(unit_prefix):] for line in comments if line.startswith(unit_prefix)), None)

    # Explicit units can resolve unlabelled files, but cannot silently contradict a recognized declaration.
    for requested, declared in ((wavelength_unit, declared_wave), (flux_unit, declared_flux)):
        if requested is not None and declared is not None and u.Unit(requested) != u.Unit(declared):
            raise ValueError("ASCII unit override conflicts with the file's declared units.")
    wave_unit = wavelength_unit or declared_wave or "Angstrom"
    unit = flux_unit or declared_flux or "erg / (s cm2 Angstrom)"
    values = np.loadtxt(path, ndmin=2)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("ASCII input requires exactly three columns: wavelength, flux, standard deviation.")
    wavelength = (values[:, 0] * u.Unit(wave_unit)).to_value(u.AA)
    return wavelength, values[:, 1], values[:, 2], unit, {}


# Verify a contiguous uniform grid; ASCII coordinate rounding is fitted within nine-digit precision only.
def linear_coordinates(wavelength, rounded_ascii=False):
    wavelength = np.asarray(wavelength, dtype=float)
    if wavelength.ndim != 1 or len(wavelength) < 2 or not np.all(np.isfinite(wavelength)):
        raise ValueError("At least two finite wavelength coordinates are required.")
    pixel = np.arange(len(wavelength), dtype=float)
    centered = pixel - pixel.mean()
    step = float(np.dot(centered, wavelength - wavelength.mean()) / np.dot(centered, centered))
    expected = wavelength.mean() + centered * step
    precision = 6e-9 if rounded_ascii else 1e-12
    tolerance = max(abs(step) * 1e-7, np.max(np.abs(wavelength)) * precision)
    if (not np.isfinite(step) or step == 0 or not np.all(np.diff(wavelength) * step > 0)
            or np.max(np.abs(wavelength - expected)) > tolerance):
        raise ValueError("Binning requires a contiguous linear wavelength grid; process separate parts individually.")
    return expected, step


# Average flux densities and propagate independent standard deviations over complete, disjoint pixel groups.
def bin_spectrum(wavelength, flux, error, bin_size, *, rounded_ascii=False, gap_policy="split"):
    if gap_policy not in ("split", "zero"):
        raise ValueError("Gap policy must be split or zero.")
    if isinstance(bin_size, (bool, np.bool_)) or not isinstance(bin_size, (int, np.integer)) or bin_size < 1:
        raise ValueError("Bin size must be a positive integer number of input pixels.")
    wavelength, step = linear_coordinates(wavelength, rounded_ascii)
    flux, error = np.asarray(flux, dtype=float), np.asarray(error, dtype=float)
    if flux.shape != wavelength.shape or error.shape != wavelength.shape:
        raise ValueError("Wavelength, flux and standard deviations must be matching 1D arrays.")
    valid = np.isfinite(flux) & np.isfinite(error) & (error > 0)
    missing = (flux == 0) & (error == 0)
    allowed = valid | missing if gap_policy == "zero" else valid
    if not np.all(allowed):
        raise ValueError("Flux must be finite and standard deviations finite and positive; split bad regions first.")

    # Exclude the incomplete trailing group so all output pixels have the same physical width and valid linear WCS.
    count = len(wavelength) // bin_size
    if count < 2:
        raise ValueError("At least two complete output bins are required for a linear spectral image.")
    used = count * bin_size
    wave_groups = wavelength[:used].reshape(count, bin_size)
    flux_groups = flux[:used].reshape(count, bin_size)
    error_groups = error[:used].reshape(count, bin_size)
    binned_wave = wave_groups.mean(axis=1)
    binned_flux = flux_groups.mean(axis=1)
    binned_error = np.hypot.reduce(error_groups, axis=1) / bin_size
    # A bin touching missing data stays missing; partial coverage cannot silently change its spectral response.
    good_bin = valid[:used].reshape(count, bin_size).all(axis=1)
    binned_flux = np.where(good_bin, binned_flux, 0.)
    binned_error = np.where(good_bin, binned_error, 0.)
    return {"wavelength_angstrom": binned_wave, "flux": binned_flux, "error": binned_error,
            "quality": (~good_bin).astype(np.int16), "missing_bins": int(np.sum(~good_bin)),
            "input_pixels": len(wavelength), "used_pixels": used, "dropped_pixels": len(wavelength) - used,
            "bin_size": int(bin_size), "step_angstrom": step * bin_size}


# Load either supported format and write a protected FITS/ASCII pair whose default name includes the bin size.
def bin_file(path, bin_size, output_dir=None, *, wavelength_unit=None, flux_unit=None, gap_policy="split"):
    path = Path(path)
    directory = Path(output_dir) if output_dir is not None else path.parent
    stem = path.name[:-8] if path.name.lower().endswith(".fits.gz") else path.stem
    prefix = directory / (stem + f"_bin{bin_size}")
    if existing_science_exports(prefix):
        raise FileExistsError("Binned products already exist; select a new output directory.")
    is_fits = path.suffix.lower() in (".fits", ".fit", ".fts") or path.name.lower().endswith(".fits.gz")
    if is_fits:
        if wavelength_unit is not None or flux_unit is not None:
            raise ValueError("Unit overrides are for ASCII only; FITS units must be correct in its headers.")
        wavelength, flux, error, unit, metadata = read_fits_spectrum(path)
    else:
        wavelength, flux, error, unit, metadata = read_ascii_spectrum(path, wavelength_unit, flux_unit)
    binned = bin_spectrum(wavelength, flux, error, bin_size, rounded_ascii=not is_fits, gap_policy=gap_policy)

    # Record the exact input, complete-bin policy and independent-noise assumption alongside retained provenance.
    metadata.update({"BINFILE": path.name, "BINSHA": hashlib.sha256(path.read_bytes()).hexdigest(),
        "BINFACT": binned["bin_size"], "BINMODE": "MEAN", "NINPUT": binned["input_pixels"],
        "NUSED": binned["used_pixels"], "NDROP": binned["dropped_pixels"], "ERRMODEL": "INDEPENDENT",
        "COORDSRC": "ASCII linear fit" if not is_fits else "FITS linear WCS",
        "NMISSBIN": binned["missing_bins"]})
    notes = [f"Mean of {bin_size} adjacent input flux densities per complete output bin.",
        "Standard deviation = sqrt(sum(input sigma squared)) / bin size.",
        "Assumes independent input pixels; covariance and upstream systematics are omitted.",
        f"Incomplete trailing group omitted: {binned['dropped_pixels']} input pixels.",
        "FITS retains full precision; ASCII uses nine significant digits."]
    if not is_fits:
        notes.append("ASCII wavelengths fitted to a linear grid within nine-significant-digit rounding.")
    if gap_policy == "zero":
        notes.append("Bins containing any missing input pixel remain missing; no partial-bin averaging.")
    plan = plan_science_export(binned["wavelength_angstrom"], binned["flux"], binned["error"],
                               binned["quality"], unit, prefix, gap_policy=gap_policy)
    directory.mkdir(parents=True, exist_ok=True)
    products = write_science_export(plan, metadata, history=notes, ascii_notes=notes)
    return {"version": __version__, "input": str(path), "bin_size_pixels": int(bin_size),
        "input_pixels": binned["input_pixels"], "used_pixels": binned["used_pixels"],
        "dropped_pixels": binned["dropped_pixels"], "output_pixels": len(binned["flux"]),
        "missing_bins": binned["missing_bins"], "gap_policy": gap_policy,
        "bin_width_angstrom": binned["step_angstrom"], "error_model": "independent input pixels",
        "products": products}


# Expose binning independently of source finding, with explicit units for unlabelled ASCII spectra.
def main(argv=None):
    cli = argparse.ArgumentParser(description=("Bin a linear 1D spectrum by averaging complete groups of input "
        "pixels. Write IRAF-compatible FITS and three-column ASCII, with Angstrom wavelengths."))
    cli.add_argument("spectrum", type=Path)
    cli.add_argument("--bin", "--bin-pixels", "-b", dest="bin_size", type=int, required=True,
                     help="Number of adjacent input wavelength pixels per bin")
    cli.add_argument("--output-dir", type=Path, help="Default: input directory; names end in _binN.fits and _binN.txt")
    cli.add_argument("--gap-policy", choices=("split", "zero"), default="split",
                     help="zero: accept paired-zero gaps and retain bins touching gaps as missing")
    cli.add_argument("--wavelength-unit", help="ASCII only; use the header, or Angstrom for unlabelled files")
    cli.add_argument("--flux-unit", help="ASCII only; use the header, or erg / (s cm2 Angstrom) for unlabelled files")
    cli.add_argument("--version", action="version", version=__version__)
    args = cli.parse_args(argv)
    try:
        result = bin_file(args.spectrum, args.bin_size, args.output_dir,
                          wavelength_unit=args.wavelength_unit, flux_unit=args.flux_unit, gap_policy=args.gap_policy)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        cli.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


# Support the direct module invocation in addition to the shared Speckex entry point.
if __name__ == "__main__":
    main()
