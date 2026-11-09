"""Unit-checked coordinates for separable, rectified two-dimensional spectra."""

import warnings

import numpy as np
from astropy import units as u
from astropy.wcs import FITSFixedWarning, WCS


# Evaluate FITS WCS using zero-based pixel centers and convert its output units explicitly.
def physical_axes(header, shape, dispersion_axis=1):
    # Reject nonseparable or non-image inputs rather than silently dropping WCS coupling.
    if len(shape) != 2 or dispersion_axis not in (0, 1):
        raise ValueError("Physical axes require a 2D image and dispersion axis 0 or 1.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs = WCS(header, naxis=2)
    matrix = wcs.pixel_scale_matrix
    if not np.allclose(matrix - np.diag(np.diag(matrix)), 0, rtol=0, atol=1e-14):
        raise ValueError("Coupled WCS axes are unsupported; rectify the image first.")
    dispersion_fits_axis = 1 - dispersion_axis
    spatial_fits_axis = dispersion_axis
    # Evaluate axes separately; WCS handles the FITS one-based CRPIX convention internally.
    pixels = [np.zeros(shape[dispersion_axis]), np.zeros(shape[dispersion_axis])]
    pixels[dispersion_fits_axis] = np.arange(shape[dispersion_axis], dtype=float)
    wave_world = wcs.pixel_to_world_values(*pixels)[dispersion_fits_axis]
    pixels = [np.zeros(shape[1 - dispersion_axis]), np.zeros(shape[1 - dispersion_axis])]
    pixels[spatial_fits_axis] = np.arange(shape[1 - dispersion_axis], dtype=float)
    spatial_world = wcs.pixel_to_world_values(*pixels)[spatial_fits_axis]
    # Require a physical wavelength unit; a pixel scale cannot supply a missing dispersion solution.
    try:
        wavelength = (wave_world * u.Unit(wcs.world_axis_units[dispersion_fits_axis])).to_value(u.AA)
    except (ValueError, TypeError, u.UnitConversionError) as exc:
        raise ValueError("WCS must provide a wavelength unit.") from exc
    # Prefer an angular WCS; otherwise use a clearly labeled arcsec-per-pixel header scale.
    try:
        spatial = (spatial_world * u.Unit(wcs.world_axis_units[spatial_fits_axis])).to_value(u.arcsec)
    except (ValueError, TypeError, u.UnitConversionError) as exc:
        comment = str(header.comments["PIXSCALE"]).lower() if "PIXSCALE" in header else ""
        if "arcsec" not in comment or ("pixel" not in comment and "pix" not in comment):
            raise ValueError("WCS needs an angular spatial unit or documented arcsec-per-pixel PIXSCALE.") from exc
        try:
            pixel_scale = float(header["PIXSCALE"])
        except (TypeError, ValueError) as scale_exc:
            raise ValueError("PIXSCALE must be a finite positive arcsec-per-pixel value.") from scale_exc
        if not np.isfinite(pixel_scale) or pixel_scale <= 0:
            raise ValueError("PIXSCALE must be a finite positive arcsec-per-pixel value.")
        spatial = (spatial_world - spatial_world[0]) * pixel_scale
    # Require finite, strictly monotonic axes for interpolation and truthful image extents.
    for values in (wavelength, spatial):
        if len(values) < 2 or not np.all(np.isfinite(values)) or not (
            np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0)
        ):
            raise ValueError("Physical axes must be finite and strictly monotonic.")
    return wavelength, spatial


# Read a separable spectral WCS when a pixel-row extraction has no angular spatial calibration.
def wavelength_axis(header, shape, dispersion_axis=1):
    if len(shape) != 2 or dispersion_axis not in (0, 1):
        raise ValueError("Wavelength axis requires a 2D image and dispersion axis 0 or 1.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs = WCS(header, naxis=2)
    matrix = wcs.pixel_scale_matrix
    if not np.allclose(matrix - np.diag(np.diag(matrix)), 0, rtol=0, atol=1e-14):
        raise ValueError("Coupled WCS axes are unsupported; rectify the image first.")
    dispersion_fits_axis = 1 - dispersion_axis
    pixels = [np.zeros(shape[dispersion_axis]), np.zeros(shape[dispersion_axis])]
    pixels[dispersion_fits_axis] = np.arange(shape[dispersion_axis], dtype=float)
    world = wcs.pixel_to_world_values(*pixels)[dispersion_fits_axis]
    # Wavelengths still require an explicit convertible unit and monotonic detector mapping.
    try:
        wavelength = (world * u.Unit(wcs.world_axis_units[dispersion_fits_axis])).to_value(u.AA)
    except (ValueError, TypeError, u.UnitConversionError) as exc:
        raise ValueError("WCS must provide a wavelength unit.") from exc
    if (len(wavelength) < 2 or not np.all(np.isfinite(wavelength)) or not
            (np.all(np.diff(wavelength) > 0) or np.all(np.diff(wavelength) < 0))):
        raise ValueError("Wavelength axis must be finite and strictly monotonic.")
    return wavelength
