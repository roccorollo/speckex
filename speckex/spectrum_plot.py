"""Display-only one-dimensional spectrum figures with explicit quality gaps."""

from pathlib import Path

import numpy as np


# Collapse valid runs for a readable overlay without changing exported samples.
def display_medians(wavelength, flux, valid, max_groups=180):
    indices = np.flatnonzero(valid)
    if not len(indices):
        return np.array([]), np.array([])
    runs = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    centers, medians = [], []
    for run in runs:
        group_width = max(1, int(np.ceil(len(run) / max_groups)))
        for start in range(0, len(run), group_width):
            group = run[start:start + group_width]
            centers.append(float(np.median(wavelength[group])))
            medians.append(float(np.median(flux[group])))
        # A NaN separator stops the display line from crossing flagged wavelengths.
        centers.append(np.nan)
        medians.append(np.nan)
    return np.asarray(centers), np.asarray(medians)


# Draw measured flux, an optional actual 1-sigma band, and optional display medians.
def plot_spectrum(wavelength_angstrom, flux, error, quality, flux_unit, output, title, *, show_medians=True):
    import matplotlib.pyplot as plt

    wavelength, flux, quality = [np.asarray(value) for value in
                                 (wavelength_angstrom, flux, quality)]
    error = None if error is None else np.asarray(error)
    if (wavelength.ndim != 1 or flux.shape != wavelength.shape or quality.shape != wavelength.shape
            or (error is not None and error.shape != wavelength.shape)):
        raise ValueError("Spectrum plot requires matching one-dimensional arrays.")
    valid = (quality == 0) & np.isfinite(wavelength) & np.isfinite(flux)
    if error is not None:
        valid &= np.isfinite(error) & (error > 0)
    if not np.any(valid):
        raise ValueError("No finite unflagged spectrum samples to plot.")
    native = np.where(valid, flux, np.nan)
    centers, medians = display_medians(wavelength, flux, valid) if show_medians else (None, None)
    figure, axis = plt.subplots(figsize=(14, 5.5))
    # Keep the uncertainty band tied to actual saved samples, not to any display overlay.
    if error is not None:
        axis.fill_between(wavelength, native - error, native + error, where=valid,
                          color="tab:blue", alpha=.12, linewidth=0,
                          label="1σ statistical error of saved samples")
    axis.plot(wavelength, native, color=(".3" if show_medians else "tab:orange"),
              alpha=(.25 if show_medians else 1.), lw=(.4 if show_medians else 1.2),
              label=("Native flux" if show_medians else "Saved binned flux"))
    if show_medians:
        axis.plot(centers, medians, color="tab:orange", lw=1.2,
                  label="Display medians; no spectral rebinning in saved data")
    axis.axhline(0., color=".3", lw=.7, alpha=.6)
    axis.set(xlabel="Wavelength (Å)", ylabel=f"Flux density ({flux_unit})",
             title=f"{Path(title).name} | valid {int(valid.sum())}/{len(valid)} samples")
    axis.grid(alpha=.2)
    axis.legend(fontsize=8, loc="best")
    # Use robust display limits so isolated spikes do not flatten the continuum.
    shown = (medians[np.isfinite(medians)] if show_medians else native[np.isfinite(native)])
    if len(shown) >= 5:
        low, high = np.percentile(shown, [5, 95])
        # A perfectly flat synthetic or real spectrum still needs finite visible limits.
        padding = (.2 * (high - low) if high > low else
                   (.1 * abs(low) if low != 0 else 1.))
        axis.set_ylim(low - padding, high + padding)
        axis.text(.01, .02, "Robust display limits; science data unchanged",
                  transform=axis.transAxes, fontsize=8, color=".4")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
