"""Linear IRAF science exports with split or explicitly zero-marked missing samples."""

from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits

from . import __version__


# Convert calibrated wavelength densities to cgs per Angstrom; retain genuinely uncalibrated units.
def flux_units(flux_unit):
    if not isinstance(flux_unit, str) or not flux_unit.strip():
        raise ValueError("Science exports require a declared flux unit.")
    target = u.erg / (u.s * u.cm ** 2 * u.AA)
    try:
        original = u.Unit(flux_unit)
        if original.is_equivalent(target):
            return float(original.to(target)), target.to_string("fits")
    except ValueError:
        pass
    # Counts or instrument-specific units are not a physical flux calibration and must not be relabeled.
    return 1., flux_unit


# Locate the existing products for one science spectrum, including any earlier contiguous parts.
def existing_science_exports(prefix):
    prefix = Path(prefix)
    products = []
    for extension in ("fits", "txt"):
        direct = Path(str(prefix) + "." + extension)
        if direct.exists():
            products.append(direct)
        products.extend(prefix.parent.glob(prefix.name + "_part[0-9]*." + extension))
    return sorted(set(products))


# Preserve the wavelength grid by splitting valid runs or marking missing samples with paired zeros.
def plan_science_export(wavelength_angstrom, flux, error, quality, flux_unit, prefix, *, gap_policy="split"):
    if gap_policy not in ("split", "zero"):
        raise ValueError("Gap policy must be split or zero.")
    wavelength_angstrom, flux, error, quality = [np.asarray(values) for values in
                                                (wavelength_angstrom, flux, error, quality)]
    if (wavelength_angstrom.ndim != 1 or len(wavelength_angstrom) < 2
            or any(values.shape != wavelength_angstrom.shape for values in (flux, error, quality))):
        raise ValueError("Science exports require matching one-dimensional arrays with at least two pixels.")
    # IRAF linear WCS must represent the actual detector sampling, including gaps that will become separate files.
    wavelength = wavelength_angstrom.astype(float)
    step = float((wavelength[-1] - wavelength[0]) / (len(wavelength) - 1))
    expected = wavelength[0] + np.arange(len(wavelength)) * step
    tolerance = max(abs(step) * 1e-7, float(np.max(np.abs(wavelength))) * 1e-12)
    if (not np.all(np.isfinite(wavelength)) or not np.isfinite(step) or step == 0
            or not np.allclose(wavelength, expected, rtol=0, atol=tolerance)):
        raise ValueError("IRAF image export requires a finite, linear wavelength grid; no resampling is performed.")
    factor, unit = flux_units(flux_unit)
    flux, error = flux.astype(float) * factor, error.astype(float) * abs(factor)
    valid = (quality == 0) & np.isfinite(flux) & np.isfinite(error) & (error > 0)
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise ValueError("No finite, positive-error, unflagged science pixels to export.")
    # A full-grid product retains invalid coordinates and uses nonzero quality plus flux=error=0 as a marker.
    if gap_policy == "zero":
        flags = np.where(valid, 0, np.where(quality != 0, quality, 1))
        if not np.all(np.isfinite(flags)) or not np.all(flags == np.floor(flags)) or np.any(flags < 0):
            raise ValueError("Quality flags must be finite nonnegative integers.")
        chunks = [{"fits": Path(str(prefix) + ".fits"), "ascii": Path(str(prefix) + ".txt"),
            "wavelength": wavelength, "flux": np.where(valid, flux, 0.),
            "error": np.where(valid, error, 0.), "quality": flags,
            "start_index": 0, "stop_index": len(valid)}]
        return {"prefix": Path(prefix), "chunks": chunks, "unit": unit, "step": step,
                "total_pixels": len(valid), "valid_pixels": len(indices), "gap_policy": gap_policy}
    runs = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    chunks = []
    # A masked internal pixel starts a new part; removing it must never shift subsequent wavelengths in an image.
    for number, run in enumerate(runs, start=1):
        base = str(prefix) + (f"_part{number:03d}" if len(runs) > 1 else "")
        chunks.append({"fits": Path(base + ".fits"), "ascii": Path(base + ".txt"),
            "wavelength": wavelength[run], "flux": flux[run], "error": error[run],
            "start_index": int(run[0]), "stop_index": int(run[-1] + 1)})
    return {"prefix": Path(prefix), "chunks": chunks, "unit": unit, "step": step,
            "total_pixels": len(valid), "valid_pixels": len(indices)}


# Write either the legacy primary/error images or one primary image with associated spectral bands.
def write_science_export(plan, metadata=None, overwrite=False, *, history=None, ascii_notes=None):
    existing = existing_science_exports(plan["prefix"])
    if existing and not overwrite:
        raise FileExistsError("Science products already exist; choose another directory or explicit overwrite.")
    products = []
    zero = plan.get("gap_policy", "split") == "zero"
    for number, chunk in enumerate(plan["chunks"], start=1):
        header = fits.Header()
        for key, value in (metadata or {}).items():
            header[key] = value
        # These descriptors follow IRAF's equispec linear spectral-image format, with one-based reference pixels.
        header["SPECKEX"] = __version__
        header["BUNIT"] = plan["unit"]
        header["WCSDIM"], header["DISPAXIS"], header["DC-FLAG"] = 1, 1, 0
        header["CTYPE1"], header["CUNIT1"] = "LINEAR", "Angstrom"
        header["CRPIX1"], header["CRVAL1"] = 1., float(chunk["wavelength"][0])
        header["CDELT1"], header["CD1_1"] = plan["step"], plan["step"]
        header["LTM1_1"], header["LTV1"] = 1., 0.
        header["WAT0_001"] = "system=equispec"
        header["WAT1_001"] = "wtype=linear label=Wavelength units=Angstroms"
        header["APNUM1"] = "1 1"
        header["ERRTYPE"] = ("STDDEV", "Sigma band contains 1-sigma" if zero else "ERROR extension contains propagated 1-sigma")
        header["APCORR"] = (False, "No total-profile aperture correction")
        header["PART"] = number
        header["NPARTS"] = len(plan["chunks"])
        header["IDXSTART"] = (chunk["start_index"], "First index in full extraction, zero based")
        header["NREJECT"] = plan["total_pixels"] - plan["valid_pixels"]
        # Derived spectra can supply processing notes without inheriting inaccurate extraction-only descriptions.
        processing_history = history if history is not None else [
            ("Full original grid retained; missing flux and error set to zero." if zero else
             "Only finite unflagged pixels exported; internal gaps split into parts."),
            "Wavelengths in Angstrom. No spectral resampling or air/vacuum conversion.",
            "Errors omit resampling covariance and trace/profile/model systematics."]
        for entry in processing_history:
            header["HISTORY"] = entry
        # IRAF associates quantities along axis 3; axis 2 contains the single extracted aperture.
        if zero:
            header["WCSDIM"], header["SPECFMT"], header["GAPPOL"] = 3, "BANDS", "ZERO"
            for axis in (2, 3):
                header[f"CTYPE{axis}"] = "LINEAR"
                header[f"CRPIX{axis}"], header[f"CRVAL{axis}"] = 1., 1.
                header[f"CDELT{axis}"], header[f"CD{axis}_{axis}"] = 1., 1.
                header[f"LTM{axis}_{axis}"], header[f"LTV{axis}"] = 1., 0.
                header[f"WAT{axis}_001"] = "wtype=linear"
            header["BANDID1"] = "spectrum - flux density"
            header["BANDID2"] = "sigma - standard deviation"
            header["BANDID3"] = "quality - nonzero means missing"
            header["ERRBAND"], header["QUALBAND"] = 2, 3
            header["BUNIT1"], header["BUNIT2"], header["BUNIT3"] = plan["unit"], plan["unit"], "1"
            header["GAPVAL"] = (0., "Missing flux=error=0; not a measurement")
            header["HISTORY"] = "Missing samples have sigma=0 and nonzero quality; exclude from analysis."
            arrays = np.stack([chunk["flux"], chunk["error"], chunk["quality"]])[:, None, :]
            hdus = fits.HDUList([fits.PrimaryHDU(arrays, header=header)])
        else:
            hdus = fits.HDUList([fits.PrimaryHDU(chunk["flux"], header=header),
                                fits.ImageHDU(chunk["error"], header=header.copy(), name="ERROR")])
        hdus.writeto(chunk["fits"], overwrite=overwrite, checksum=True)
        # Eight digits after the scientific mantissa's decimal point give nine significant digits in ASCII.
        notes = ascii_notes if ascii_notes is not None else [
            "Full flags and original sampling are retained in the companion diagnostic ECSV.",
            "FITS retains full precision. ASCII uses nine significant digits; no resampling."]
        notes = list(notes)
        if zero:
            notes.append("Missing data: FLUX=0 and ERROR_STDDEV=0; valid measurements require ERROR_STDDEV>0.")
            notes.append("Zero error is a missing-data marker, not zero uncertainty. Exclude these rows from analysis.")
        heading = ("WAVELENGTH_ANGSTROM FLUX ERROR_STDDEV\n"
            f"Flux and 1-sigma error unit: {plan['unit']}\n"
            f"Speckex {__version__}; part {number}/{len(plan['chunks'])}; "
            + ("full wavelength grid with zero-marked gaps.\n" if zero else "finite unflagged pixels only.\n")
            + "\n".join(notes))
        np.savetxt(chunk["ascii"], np.column_stack([chunk["wavelength"], chunk["flux"], chunk["error"]]),
                   fmt="%.8e", header=heading)
        products.append({"fits": str(chunk["fits"]), "ascii": str(chunk["ascii"]),
            "iraf_image": str(chunk["fits"]) + ("[*,1,1]" if zero else "[0]"),
            "error_image": str(chunk["fits"]) + ("[*,1,2]" if zero else "[ERROR]"),
            "samples": len(chunk["flux"]), "wavelength_first_last_angstrom": chunk["wavelength"][[0, -1]].tolist()})
    # Clean only obsolete members of this exact science product family after the new files were successfully written.
    retained = {Path(item[key]) for item in products for key in ("fits", "ascii")}
    for path in existing:
        if path not in retained:
            path.unlink()
    return products
