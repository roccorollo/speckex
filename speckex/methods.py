"""Specreduce 1.5.1 adapters with fixed source kernels and explicit error propagation."""

import warnings
from importlib.metadata import version

import numpy as np
from astropy.modeling import models
from astropy.utils.exceptions import AstropyDeprecationWarning
from scipy.optimize import OptimizeWarning, curve_fit
from specreduce.extract import BoxcarExtract, HorneExtract
from specreduce.tracing import ArrayTrace, FlatTrace


# Supply native-grid column centers to the 1.5.1 Horne kernel without its science-only image alignment.
class NativeGridTrace(FlatTrace):
    # The inherited type is the upstream dispatch tag that leaves science, variance, and mask co-registered.
    def __init__(self, image, centers):
        super().__init__(image, float(np.median(centers)))
        self.trace = np.asarray(centers, dtype=float).copy()


# Reuse the reviewed Gaussian in Specreduce; fitting a sky-only kernel would change extraction weights.
class FixedGaussianHorne(HorneExtract):
    # Override the 1.5.1 profile-fit hook to exclude an additional polynomial sky component from the kernel.
    def _fit_gaussian_spatial_profile(self, img, disp_axis, crossdisp_axis, or_mask, bkgrd_prof):
        return models.Gaussian1D(1., 0., self.fixed_sigma_pixel) + models.Const1D(
            0., fixed={"amplitude": True})


# Reject absent or unusable variances before any library call can substitute uniform weights.
def validate_variance(variance, shape, usable):
    if variance is None:
        raise ValueError("Horne extraction requires an explicit variance array.")
    variance = np.asarray(variance, dtype=float)
    if variance.shape != shape:
        raise ValueError("Variance must match the science image shape.")
    if not np.any(usable) or np.any(~np.isfinite(variance[usable])) or np.any(variance[usable] <= 0):
        raise ValueError("Horne requires finite, strictly positive variances in every used source pixel.")
    return variance


# Fit the faint target's width from aligned, sky-subtracted aperture data; never fit the neighboring star.
def gaussian_profile(result, sigma_guess_arcsec):
    data = result["data"] - result["sky_image"] - result["star_image"]
    variance = result["input_error"] ** 2
    aperture = result["source_weights"]
    scale = float(np.median(np.diff(result["spatial"])))
    columns = np.flatnonzero(result["flags"] == 0)
    half_width = np.median(aperture[:, columns].sum(axis=0)) / 2 if len(columns) else 0.
    offsets = np.arange(-np.floor(half_width), np.floor(half_width) + 1)
    if len(columns) < 8 or len(offsets) < 5:
        raise ValueError("Gaussian profile needs at least eight usable columns and five spatial samples; widen the aperture.")
    # Interpolation is only for estimating one shared profile width; extraction remains on original pixels.
    row_center = (result["target"] - result["spatial"][0]) / scale
    location = offsets[:, None] + row_center[columns]
    # Snap only floating-point roundoff at integer rows; unused neighbors must not contribute NaNs.
    nearest = np.rint(location)
    tolerance = 16 * np.finfo(float).eps * np.maximum(1., np.abs(location))
    location = np.where(np.abs(location - nearest) <= tolerance, nearest, location)
    left = np.floor(location).astype(int)
    fraction = location - left
    left_used, right_used = (1 - fraction) > 0, fraction > 0
    if np.any(left_used & ((left < 0) | (left >= len(result["spatial"])))) or np.any(
            right_used & ((left + 1 < 0) | (left + 1 >= len(result["spatial"])))):
        raise ValueError("Gaussian fitting samples reach the image boundary.")
    # Omit a profile-estimation column only when an actual contributor has invalid science or variance.
    left_index, right_index = np.clip(left, 0, len(data) - 1), np.clip(left + 1, 0, len(data) - 1)
    left_data, right_data = data[left_index, columns], data[right_index, columns]
    left_variance, right_variance = variance[left_index, columns], variance[right_index, columns]
    valid_left = np.isfinite(left_data) & np.isfinite(left_variance) & (left_variance > 0)
    valid_right = np.isfinite(right_data) & np.isfinite(right_variance) & (right_variance > 0)
    retained = np.all((~left_used | valid_left) & (~right_used | valid_right), axis=0)
    if np.count_nonzero(retained) < 8:
        raise ValueError("Gaussian profile needs at least eight columns with valid interpolation contributors.")
    sample = ((1 - fraction) * np.where(left_used, left_data, 0.)
              + fraction * np.where(right_used, right_data, 0.))[:, retained]
    sample_variance = ((1 - fraction) ** 2 * np.where(left_used, left_variance, 0.)
                       + fraction ** 2 * np.where(right_used, right_variance, 0.))[:, retained]
    # Use common spectral weights to avoid changing the sampled source spectrum with spatial row.
    weight = 1 / np.median(sample_variance, axis=0)
    weight = np.minimum(weight, np.percentile(weight, 95))
    mean = np.sum(sample * weight, axis=1) / np.sum(weight)
    error = np.sqrt(np.sum(sample_variance * weight ** 2, axis=1)) / np.sum(weight)
    guess = float(sigma_guess_arcsec / scale)
    low, high = .4, max(.5, half_width)
    if not np.all(np.isfinite(mean)) or np.max(mean) <= 0:
        raise ValueError("No positive finite source profile for Gaussian Horne extraction.")

    # Keep the center at the confirmed trace; estimating a new centroid would change reviewed behavior.
    def model(x, amplitude, sigma):
        return amplitude * np.exp(-.5 * (x / sigma) ** 2)

    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        parameters, covariance = curve_fit(model, offsets, mean,
            p0=[float(np.max(mean)), float(np.clip(guess, low + .01, high - .01))],
            bounds=([0., low], [np.inf, high]), sigma=error, absolute_sigma=True, maxfev=10000)
    if (not np.all(np.isfinite(covariance)) or parameters[1] <= low * 1.01
            or parameters[1] >= high * .99 or parameters[0] <= 3 * np.sqrt(covariance[0, 0])):
        raise ValueError("Gaussian source profile is weak or reaches its width bounds; inspect the aperture/profile.")
    return {"sigma_pixel": float(parameters[1]), "sigma_arcsec": float(parameters[1] * scale),
            "sigma_error_pixel": float(np.sqrt(covariance[1, 1])),
            "amplitude": float(parameters[0]), "offset_pixel": offsets,
            "measured": mean, "formal_error": error, "status": "fitted"}


# Apply the same source weights to sky/star images and project their shared fitted covariance once.
def propagate_weights(result, weights, flux, profile=None):
    used = weights != 0
    variance = np.sum(np.where(used, result["input_error"] ** 2, 0.) * weights ** 2, axis=0)
    delta = result["spatial"][:, None] - result["target"]
    vectors = [np.sum(weights * delta ** order, axis=0) for order in range(result["background_degree"] + 1)]
    if result["substars"]:
        stellar_shape = (1 + ((result["spatial"][:, None] - result["star_center"])
                              / result["star_width"]) ** 2) ** (-result["star_beta"])
        vectors.append(np.sum(weights * stellar_shape, axis=0))
    vector = np.array(vectors).T
    covariance = result["nuisance_covariance"]
    variance += np.einsum("ij,ijk,ik->i", vector, covariance, vector)
    sky_vector = vector.copy()
    if result["substars"]:
        sky_vector[:, -1] = 0.
    sky_variance = np.einsum("ij,ijk,ik->i", sky_vector, covariance, sky_vector)
    star_variance = vector[:, -1] ** 2 * covariance[:, -1, -1] if result["substars"] else np.zeros(len(flux))
    sky = np.sum(weights * np.where(used, result["sky_image"], 0.), axis=0)
    star = np.sum(weights * np.where(used, result["star_image"], 0.), axis=0)
    flags = result["flags"].copy()
    bad = ~np.isfinite(flux) | ~np.isfinite(variance) | (variance <= 0)
    flags[(flags == 0) & bad] = 8
    for array in (flux, variance, sky, star, sky_variance, star_variance):
        array[flags != 0] = np.nan
    weights[:, flags != 0] = 0.
    return {"flux": flux, "error": np.sqrt(variance), "background": sky, "stellar": star,
            "background_error": np.sqrt(sky_variance), "stellar_error": np.sqrt(star_variance),
            "flags": flags, "weights": weights, "profile": profile}


# Extract requested methods with Specreduce and retain boxcar's established finite-aperture convention.
def extract_methods(result, args, sigma_guess_arcsec):
    if version("specreduce") != "1.5.1":
        raise ValueError("This tested adapter requires specreduce==1.5.1; no unverified backend fallback is used.")
    mode = getattr(args, "method", "boxcar")
    requested = ("boxcar", "horne") if mode == "both" else (mode,)
    if mode not in ("boxcar", "horne", "both"):
        raise ValueError("Extraction method must be boxcar, horne, or both.")
    data = result["data"] - result["sky_image"] - result["star_image"]
    clean = np.nan_to_num(data, nan=0., posinf=0., neginf=0.)
    scale = float(np.median(np.diff(result["spatial"])))
    centers = (result["target"] - result["spatial"][0]) / scale
    methods = {}
    # Specreduce's boxcar supplies the flux; existing source/sky covariance supplies identical legacy errors.
    if "boxcar" in requested:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", AstropyDeprecationWarning)
            extraction = BoxcarExtract(clean, ArrayTrace(clean, centers),
                width=args.aperture_arcsec / scale, mask_treatment="ignore").spectrum
        flux = np.asarray(extraction.flux.value, dtype=float)
        methods["boxcar"] = propagate_weights(result, result["source_weights"].copy(), flux)
    # A frozen Gaussian makes source, sky, and star weighting consistent and avoids fitting a kernel to sky alone.
    if "horne" in requested:
        usable = (result["source_weights"] > 0) & (result["flags"][None, :] == 0)
        variance = validate_variance(result["input_error"] ** 2, data.shape, usable)
        try:
            profile = gaussian_profile(result, sigma_guess_arcsec)
        except (ValueError, RuntimeError, OptimizeWarning) as exc:
            flags = result["flags"].copy()
            flags[flags == 0] = 8
            missing = np.full(len(centers), np.nan)
            methods["horne"] = {key: missing.copy() for key in
                ("flux", "error", "background", "stellar", "background_error", "stellar_error")}
            methods["horne"].update(flags=flags, weights=np.zeros_like(data),
                profile={"status": "failed", "reason": str(exc)})
        else:
            sigma = profile["sigma_pixel"]
            mask = ~usable
            # Only masked elements get computational placeholders; every used variance is validated unchanged.
            safe_variance = np.where(usable, variance, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                warnings.simplefilter("ignore", AstropyDeprecationWarning)
                extractor = FixedGaussianHorne(clean, NativeGridTrace(clean, centers),
                    variance=safe_variance, mask=mask, bkgrd_prof=models.Const1D(0., fixed={"amplitude": True}))
                extractor.fixed_sigma_pixel = sigma
                spectrum = extractor.spectrum
            gaussian = np.exp(-.5 * ((np.arange(len(result["spatial"]))[:, None] - centers) / sigma) ** 2)
            norm = sigma * np.sqrt(2 * np.pi)
            enclosed = np.sum(result["source_weights"] * gaussian, axis=0)
            fraction = enclosed / norm
            flux = np.asarray(spectrum.flux.value) * fraction
            inverse = np.zeros_like(variance)
            np.divide(1., variance, out=inverse, where=usable)
            denominator = np.sum(gaussian ** 2 * inverse, axis=0)
            weights = np.zeros_like(data)
            np.divide(enclosed * gaussian * inverse, denominator, out=weights, where=denominator > 0)
            methods["horne"] = propagate_weights(result, weights, flux, profile)
            methods["horne"]["gaussian_fraction_in_aperture"] = fraction
    result["methods"], result["extraction_mode"] = methods, mode
    result["specreduce_version"] = version("specreduce")
    # Maintain legacy top-level preview keys; both mode keeps boxcar there without ranking by S/N.
    primary = methods["boxcar"] if "boxcar" in methods else methods["horne"]
    for key in ("flux", "error", "background", "stellar", "background_error", "stellar_error", "flags"):
        result[key] = primary[key]
