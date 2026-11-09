"""FITS loading, mean spatial profiles, and diagnostic Gaussian fits."""

import warnings

import numpy as np
from astropy.io import fits
from scipy.optimize import OptimizeWarning, curve_fit


# Read a specific image extension and orient its dispersion direction horizontally.
def load_image(path, hdu=0, dispersion_axis=1, variance_hdu=None, mask_hdu=None):
    # Copy arrays before closing FITS; mask non-finite science and invalid variances.
    with fits.open(path, memmap=False) as hdus:
        if hdus[hdu].data is None:
            raise ValueError("Selected science HDU has no image data; choose --hdu.")
        data = np.array(hdus[hdu].data, dtype=float, copy=True)
        variance = None if variance_hdu is None else np.array(
            hdus[variance_hdu].data, dtype=float, copy=True
        )
        mask = np.zeros(data.shape, dtype=bool) if mask_hdu is None else (
            np.asarray(hdus[mask_hdu].data) != 0
        )
    # Require matching two-dimensional arrays and preserve original pixel coordinates.
    if data.ndim != 2 or dispersion_axis not in (0, 1):
        raise ValueError("Science data must be 2D and dispersion_axis must be 0 or 1.")
    if mask.shape != data.shape or (variance is not None and variance.shape != data.shape):
        raise ValueError("Mask and variance shapes must match the science image.")
    mask = mask | ~np.isfinite(data)
    if variance is not None:
        mask |= ~np.isfinite(variance) | (variance <= 0)
    if dispersion_axis == 0:
        data, mask = data.T, mask.T
        variance = None if variance is None else variance.T
    return data, variance, mask


# Collapse an explicitly bounded dispersion interval without treating missing data as zero.
def spatial_profile(data, bounds, variance=None, mask=None):
    # Validate half-open pixel bounds, then count usable columns separately for every row.
    start, stop = bounds
    if not 0 <= start < stop <= data.shape[1]:
        raise ValueError(f"Dispersion interval {start}:{stop} is outside 0:{data.shape[1]}.")
    region = data[:, start:stop]
    valid = np.isfinite(region)
    if mask is not None:
        valid &= ~mask[:, start:stop]
    if variance is not None:
        var = variance[:, start:stop]
        valid &= np.isfinite(var) & (var > 0)
    count = valid.sum(axis=1)
    # Propagate independent-pixel variances for an arithmetic mean, when supplied.
    mean = np.full(data.shape[0], np.nan)
    np.divide(np.where(valid, region, 0).sum(axis=1), count, out=mean, where=count > 0)
    error = None
    if variance is not None:
        error = np.full(data.shape[0], np.nan)
        np.divide(np.sqrt(np.where(valid, var, 0).sum(axis=1)), count,
                  out=error, where=count > 0)
    return mean, error, count


# Model a positive compact source over a locally linear residual background.
def gaussian_background(y, amplitude, center, sigma, offset, slope):
    return amplitude * np.exp(-0.5 * ((y - center) / sigma) ** 2) + offset + slope * y


# Estimate a candidate centroid and width; this diagnostic fit is not a detection test.
def fit_profile(profile, error=None, fit_bounds=None, guess=None):
    # Restrict the fit to finite rows in the user-selected spatial search window.
    y = np.arange(len(profile), dtype=float)
    lower, upper = (0, len(profile)) if fit_bounds is None else fit_bounds
    if not 0 <= lower < upper <= len(profile):
        raise ValueError("Spatial fit interval is outside the image.")
    if guess is not None and not lower <= guess <= upper - 1:
        raise ValueError("Source guess must lie within the spatial fit interval.")
    valid = np.isfinite(profile) & (y >= lower) & (y < upper)
    if error is not None:
        valid &= np.isfinite(error) & (error > 0)
    if valid.sum() < 7:
        return {"status": "failed", "reason": "Need at least seven valid spatial pixels."}
    x, values = y[valid], profile[valid]
    # Fit dimensionless fluxes so numerical tolerances and parameter bounds do not depend on BUNIT.
    flux_scale = float(np.max(np.abs(values)))
    if flux_scale == 0:
        return {"status": "failed", "reason": "No nonzero spatial profile to fit."}
    values = values / flux_scale
    fit_error = None if error is None else error[valid] / flux_scale
    # Initialize with an edge-based background and constrain the source to the fit window.
    edge_count = max(1, len(x) // 5)
    edges = np.r_[0:edge_count, len(x) - edge_count:len(x)]
    slope, offset = np.polyfit(x[edges], values[edges], 1)
    residual = values - offset - slope * x
    center = float(x[np.argmax(residual)]) if guess is None else float(guess)
    amplitude = max(float(np.max(residual)), np.finfo(float).eps)
    span = x[-1] - x[0]
    initial = [amplitude, center, min(2., span / 4), offset, slope]
    bounds = ([0, lower, 0.3, -np.inf, -np.inf],
              [np.inf, upper - 1, span, np.inf, np.inf])
    # Treat an unconstrained covariance as a failed fit rather than reporting false precision.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            parameters, covariance = curve_fit(
                gaussian_background, x, values, p0=initial, bounds=bounds,
                sigma=fit_error,
                absolute_sigma=error is not None, maxfev=20000,
            )
        if not np.all(np.isfinite(covariance)):
            raise ValueError("Fit covariance is not finite.")
    except (ValueError, RuntimeError, OptimizeWarning) as exc:
        return {"status": "failed", "reason": str(exc)}
    # Restore dimensional parameters and covariance; centroid and width remain in detector pixels.
    transform = np.array([flux_scale, 1., 1., flux_scale, flux_scale])
    parameters = parameters * transform
    covariance = covariance * transform[:, None] * transform[None, :]
    # Flag boundary solutions; uncertainties omit background-model and correlation systematics.
    amplitude, center, sigma, offset, slope = map(float, parameters)
    at_boundary = (center - lower < 0.5 or upper - 1 - center < 0.5
                   or sigma < 0.31 or sigma > 0.95 * span)
    return {
        "status": "boundary" if at_boundary else "candidate",
        "amplitude": amplitude, "center_pixel": center, "sigma_pixel": sigma,
        "fwhm_pixel": float(2 * np.sqrt(2 * np.log(2)) * sigma),
        "center_error_pixel": float(np.sqrt(covariance[1, 1])),
        "offset": offset, "slope": slope,
        "uncertainty_basis": "input variance" if error is not None else "fit residuals",
    }
