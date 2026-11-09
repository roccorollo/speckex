"""Command-line and graphical spatial-profile inspection."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

from . import __version__
from .profile import fit_profile, gaussian_background, load_image, spatial_profile


# Parse zero-based, half-open pixel intervals consistently for both image axes.
def pixel_range(value):
    try:
        start, stop = map(int, value.split(":"))
        if start < 0 or stop <= start:
            raise ValueError
        return start, stop
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use START:STOP with 0 <= START < STOP.") from exc


# Allow either a numeric FITS extension index or a named extension.
def hdu_selector(value):
    return int(value) if value.isdecimal() else value


# Define the public CLI without requiring installation into the active environment.
def parser():
    cli = argparse.ArgumentParser(epilog=("For reviewed or quick extraction: python -m speckex extract --help; "
                                         "for errorless 2D flux: python -m speckex flux-only --help; "
                                         "for 1D binning: python -m speckex bin --help"), description=(
        "Inspect mean spatial profiles in selected dispersion ranges of a 2D FITS image. "
        "Coordinates are zero-based pixels; interval stops are excluded."
    ))
    # Group science-array selection and explicit orientation settings.
    cli.add_argument("image", type=Path)
    cli.add_argument("--version", action="version", version=__version__)
    cli.add_argument("--hdu", type=hdu_selector, default=0)
    cli.add_argument("--variance-hdu", type=hdu_selector, help="Variance, not error or inverse variance")
    cli.add_argument("--mask-hdu", type=hdu_selector, help="Nonzero pixels are excluded")
    cli.add_argument("--dispersion-axis", type=int, choices=(0, 1), default=1,
                     help="NumPy axis: 1 = FITS horizontal axis (default), 0 = vertical")
    # Expose wavelength-region selection separately from spatial-profile fitting.
    cli.add_argument("--range", dest="ranges", type=pixel_range, action="append",
                     help="Dispersion START:STOP; repeat to compare profiles")
    cli.add_argument("--fit-range", type=pixel_range, help="Spatial START:STOP for fitting")
    cli.add_argument("--guess", type=float, help="Initial source spatial pixel; use with --fit-range")
    cli.add_argument("--no-fit", action="store_true", help="Inspect measured profiles without a model")
    cli.add_argument("--interactive", action="store_true",
                     help="Drag horizontal image intervals to add profiles; R resets; close to save")
    cli.add_argument("--output", type=Path, default=Path("profile"), help="Output prefix (default: profile)")
    cli.add_argument("--overwrite", action="store_true")
    return cli


# Render image regions and profiles, optionally letting the user append intervals by dragging.
def inspect_profiles(data, variance, mask, ranges, args):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import SpanSelector

    # Show the oriented image with a robust intensity stretch for profile exploration.
    figure, (image_axes, profile_axes) = plt.subplots(2, 1, figsize=(11, 8))
    finite = data[~mask]
    if finite.size == 0:
        raise ValueError("No valid science pixels remain after masking.")
    vmin, vmax = np.percentile(finite, [5, 99])
    image_axes.imshow(np.ma.array(data, mask=mask), origin="lower", aspect="auto",
                      vmin=vmin, vmax=vmax, cmap="gray")
    image_axes.set(xlabel="Dispersion pixel", ylabel="Spatial pixel")
    records, tables, patches = [], [], []

    # Recompute each interval independently so wavelength-dependent positions remain visible.
    def redraw():
        profile_axes.clear()
        for patch in patches:
            patch.remove()
        records.clear()
        tables.clear()
        patches.clear()
        for index, interval in enumerate(ranges):
            mean, error, count = spatial_profile(data, interval, variance, mask)
            fit = {"status": "disabled"} if args.no_fit else fit_profile(
                mean, error, args.fit_range, args.guess
            )
            record = {"dispersion_start": interval[0], "dispersion_stop": interval[1], **fit}
            records.append(record)
            spatial = np.arange(len(mean))
            tables.append(Table({
                "region": np.full(len(mean), index), "spatial_pixel": spatial,
                "dispersion_start": np.full(len(mean), interval[0]),
                "dispersion_stop": np.full(len(mean), interval[1]),
                "mean": mean,
                "mean_error": np.full(len(mean), np.nan) if error is None else error,
                "valid_columns": count,
            }))
            # Keep the image highlight and its measured profile in the same color.
            line, = profile_axes.plot(spatial, mean, label=f"{interval[0]}:{interval[1]}")
            color = line.get_color()
            patches.append(image_axes.axvspan(interval[0] - .5, interval[1] - .5,
                                              color=color, alpha=.22))
            if error is not None:
                profile_axes.fill_between(spatial, mean - error, mean + error, color=color, alpha=.15)
            if fit["status"] in ("candidate", "boundary"):
                low, high = args.fit_range or (0, len(mean))
                xfit = np.linspace(low, high - 1, 400)
                model = gaussian_background(xfit, fit["amplitude"], fit["center_pixel"],
                                            fit["sigma_pixel"], fit["offset"], fit["slope"])
                profile_axes.plot(xfit, model, "--", color=color,
                                  label=f"candidate y={fit['center_pixel']:.2f}, FWHM={fit['fwhm_pixel']:.2f}")
        profile_axes.set(xlabel="Spatial pixel", ylabel="Mean per valid dispersion pixel")
        if ranges:
            profile_axes.legend(fontsize="small")
        figure.canvas.draw_idle()

    # Round mouse-selected spans to pixel edges and avoid duplicate selections.
    def select(start, stop):
        interval = (max(0, int(np.ceil(start))), min(data.shape[1], int(np.ceil(stop))))
        if interval[0] < interval[1] and interval not in ranges:
            ranges.append(interval)
            redraw()

    # Reset allows replacement of initial selections without restarting the program.
    def key_press(event):
        if event.key in ("r", "R"):
            ranges.clear()
            redraw()

    # Retain the selector until the GUI closes; headless use only renders the final figure.
    redraw()
    if args.interactive:
        import matplotlib
        from matplotlib import rcsetup
        if matplotlib.get_backend().lower() not in [b.lower() for b in rcsetup.interactive_bk]:
            raise ValueError("Interactive mode needs a GUI Matplotlib backend; use --range on a headless system.")
        figure.suptitle("Drag dispersion intervals to add profiles • R resets • Close window to save")
        selector = SpanSelector(image_axes, select, "horizontal", useblit=True,
                                props={"alpha": .3, "facecolor": "tab:blue"}, button=1)
        figure.canvas.mpl_connect("key_press_event", key_press)
        plt.show()
        selector.disconnect_events()
    figure.tight_layout()
    return figure, records, tables


# Validate inputs, run inspection, and export numerical results and the diagnostic figure.
def main(argv=None):
    # Route named workflows and let a top-level quick flag imply extraction.
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "extract":
        from .workflow import main as workflow_main
        return workflow_main(arguments[1:])
    if arguments and arguments[0] == "bin":
        from .binning import main as binning_main
        return binning_main(arguments[1:])
    if arguments and arguments[0] == "flux-only":
        from .flux_only import main as flux_only_main
        return flux_only_main(arguments[1:])
    if any(argument in ("-q", "--quick") for argument in arguments):
        from .workflow import main as workflow_main
        return workflow_main(arguments)
    cli = parser()
    args = cli.parse_args(arguments)
    # Protect existing products and validate spatial/dispersion bounds before fitting.
    paths = [Path(str(args.output) + suffix) for suffix in (".png", ".json", ".ecsv")]
    try:
        if not args.overwrite and any(path.exists() for path in paths):
            raise ValueError("An output product already exists; choose --output or use --overwrite.")
        if not args.output.parent.is_dir():
            raise ValueError("Output directory does not exist.")
        if any(path.resolve() == args.image.resolve() for path in paths):
            raise ValueError("Output must not replace the input image.")
        data, variance, mask = load_image(args.image, args.hdu, args.dispersion_axis,
                                          args.variance_hdu, args.mask_hdu)
        ranges = list(args.ranges or ([] if args.interactive else [(0, data.shape[1])]))
        for interval in ranges:
            spatial_profile(data, interval, variance, mask)
        if args.fit_range is not None and args.fit_range[1] > data.shape[0]:
            raise ValueError("Spatial fit interval is outside the image.")
        if args.guess is not None:
            low, high = args.fit_range or (0, data.shape[0])
            if not np.isfinite(args.guess) or not low <= args.guess <= high - 1:
                raise ValueError("Source guess must lie within the spatial fit interval.")
        # Export reproducible per-region fits and the individual profile samples.
        figure, records, tables = inspect_profiles(data, variance, mask, ranges, args)
        if not records:
            raise ValueError("No intervals selected; no output written.")
        from astropy.table import vstack
        import matplotlib.pyplot as plt
        table = vstack(tables)
        table.meta = {
            "version": __version__, "input": str(args.image), "science_hdu": args.hdu,
            "variance_hdu": args.variance_hdu, "mask_hdu": args.mask_hdu,
            "dispersion_axis": args.dispersion_axis, "fit_range": args.fit_range,
            "guess": args.guess, "coordinates": "zero-based; stop excluded",
            "collapse": "arithmetic mean of unmasked finite pixels",
        }
        report = {**table.meta, "profiles": records,
                  "caution": "Fits are source candidates, not detection significances. "
                             "Variance propagation assumes independent pixels; "
                             "fit errors omit background-model uncertainty."}
        figure.savefig(paths[0], dpi=150)
        paths[1].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        table.write(paths[2], format="ascii.ecsv", overwrite=args.overwrite)
        plt.close(figure)
        for record in records:
            print(json.dumps(record))
        print("Saved " + ", ".join(map(str, paths)))
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        cli.error(str(exc))
