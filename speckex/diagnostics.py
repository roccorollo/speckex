"""Common three-panel science extraction diagnostics."""

import numpy as np
from scipy.ndimage import gaussian_filter


# Show source/sky locations, exact used pixels, and the resulting boxcar spectrum.
def plot_extraction_regions(data, wavelength_angstrom, result, report, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm, BoundaryNorm, ListedColormap

    ny, nx = data.shape
    sample = np.arange(0, nx, max(1, nx // 1000))
    image = data[:, sample]
    finite = np.isfinite(image)
    scale = float(np.nanpercentile(np.abs(image[finite]), 75)) if np.any(finite) else 1.
    scale = max(scale, np.finfo(float).tiny)
    display = gaussian_filter(np.where(finite, image, 0.), (.5, 1.))
    norm = gaussian_filter(finite.astype(float), (.5, 1.))
    display = np.where(norm > .25, display / np.maximum(norm, 1e-12), np.nan)
    stretch = max(float(np.nanpercentile(np.abs(display[np.isfinite(display)]), 80)), scale)
    wave = np.asarray(wavelength_angstrom)
    rows = np.arange(ny)
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), height_ratios=[2, 1.1, 1.1])
    axes[0].pcolormesh(wave[sample], rows, display, shading="nearest", cmap="RdBu_r",
                       norm=AsinhNorm(linear_width=stretch / 3, vmin=-stretch, vmax=stretch))
    used_source = np.asarray(result["source_weights"])[:, sample] > 0
    used_sky = np.asarray(result["background_used"])[:, sample]
    # Draw the actual selected aperture and sky pixel boundaries on the 2D science image.
    extent = np.zeros_like(used_source, dtype=np.int8)
    extent[used_source] = 1
    extent[used_sky] = 2
    overlay = np.ma.masked_where(extent == 0, extent)
    colors = ListedColormap(["limegreen", "deepskyblue"])
    axes[0].pcolormesh(wave[sample], rows, overlay, shading="nearest", cmap=colors,
                       norm=BoundaryNorm([.5, 1.5, 2.5], colors.N), alpha=.28)
    axes[0].set(ylabel="Original spatial row (zero-based)",
                title="Source and background regions; image smoothing is for display only")
    # Match the reviewed extraction's green/blue/white effective-sample panel.
    mask_colors = ListedColormap(["white", "limegreen", "deepskyblue"])
    axes[1].pcolormesh(wave[sample], rows, extent, shading="nearest", cmap=mask_colors,
                       norm=BoundaryNorm([-.5, .5, 1.5, 2.5], mask_colors.N))
    axes[1].set(ylabel="Original spatial row (zero-based)",
                title="Effective samples: green=source, blue=used sky, white=unused")
    valid = np.asarray(result["quality"]) == 0
    flux = np.where(valid, result["flux"], np.nan)
    sky = np.where(valid, result["background"], np.nan)
    axes[2].plot(wave, flux, lw=.35, alpha=.18, color=".4", label="Unsmoothed spectrum")
    # Median overlays improve readability without altering any exported measurements.
    edges = np.linspace(0, nx, min(151, nx + 1), dtype=int)
    centers, smoothed_flux, smoothed_sky = [], [], []
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        centers.append(float(np.median(wave[left:right])))
        smoothed_flux.append(float(np.nanmedian(flux[left:right])) if np.any(valid[left:right]) else np.nan)
        smoothed_sky.append(float(np.nanmedian(sky[left:right])) if np.any(valid[left:right]) else np.nan)
    axes[2].plot(centers, smoothed_sky, lw=1., color="tab:blue", label="Background display medians")
    axes[2].plot(centers, smoothed_flux, lw=1., color="tab:orange", label="Boxcar display medians")
    values = np.r_[smoothed_flux, smoothed_sky]
    values = values[np.isfinite(values)]
    if len(values) > 2:
        low, high = np.percentile(values, [2, 98])
        padding = max(.1 * (high - low), .01 * max(abs(low), abs(high), 1e-30))
        axes[2].set_ylim(low - padding, high + padding)
    error_note = "input 1σ propagated" if "error" in result else "statistical uncertainty unavailable"
    axes[2].set(xlabel="Wavelength (Å)", ylabel="Flux density (input units)",
                title=f"Boxcar spectrum and fitted sky; {error_note}")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set_xlim(wave[0], wave[-1])
    figure.suptitle(f"{report['input']} | boxcar aperture | sky degree {report['background_degree']} | Speckex {report['version']}")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
