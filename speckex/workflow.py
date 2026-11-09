"""Human-reviewed continuum localization, local background, and boxcar extraction."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

from .cli import pixel_range
from .coordinates import physical_axes
from .extraction import plot_background, plot_comparison, prepare_extraction, save_extraction, validate_windows
from .search import analyze_image, build_parser, physical_range, plot_diagnostics
from .export import existing_science_exports
from .sensitivity import (analyze_background_sensitivity, save_background_sensitivity,
                          sensitivity_paths, validate_sensitivity_settings)
from .profile_qa import analyze_horne_qa, profile_qa_paths, save_horne_qa, validate_profile_settings
from .spectrum_plot import plot_spectrum
from .viewer import open_image


# Parse editable background or wavelength interval lists, retaining signed angular offsets.
def interval_list(text):
    return [physical_range(value.strip()) for value in text.split(",") if value.strip()]


# Name each requested 1D figure once for both collision checks and saved products.
def spectrum_plot_paths(prefix, method):
    # Joint extraction labels both plots; a single method keeps its existing generic name.
    names = ("boxcar", "horne") if method == "both" else (method,)
    return {name: Path(str(prefix) + (f"_{name}_spectrum.png" if method == "both"
                                    else "_spectrum.png")) for name in names}


# Validate all editable values before running an expensive preview or accepting a confirmation.
def validate_settings(args):
    values = [args.offset_arcsec, args.search_radius_arcsec, args.clip_sigma,
              args.noise_percentile, args.aperture_arcsec, args.star_exclude_arcsec, args.background_sigma]
    if not np.all(np.isfinite(values)):
        raise ValueError("All numerical settings must be finite.")
    if (args.bin_pixels < 2 or min(args.search_radius_arcsec, args.clip_sigma, args.aperture_arcsec,
            args.star_exclude_arcsec, args.background_sigma) <= 0
            or not 0 < args.noise_percentile <= 100 or args.bootstrap < 30):
        raise ValueError("Use positive widths/clipping, noise percentile in (0,100], and at least 30 bootstraps.")
    validate_windows(args.aperture_arcsec, args.background_windows)
    if args.background_degree not in (0, 1):
        raise ValueError("Background degree must be 0 or 1.")
    if getattr(args, "method", "boxcar") not in ("boxcar", "horne", "both"):
        raise ValueError("Extraction method must be boxcar, horne, or both.")
    # Optional QA controls must be valid before preview or confirmation; defaults do not invoke the analysis.
    if getattr(args, "background_sensitivity", False):
        validate_sensitivity_settings(args)
    # Horne reliability diagnostics require an explicitly selected Horne extraction and finite perturbations.
    if getattr(args, "horne_qa", False):
        validate_profile_settings(args)


# Convert an approximate original image row to a star-relative angular search guess.
def row_to_offset(path, args, row):
    with fits.open(path, memmap=False) as hdus:
        data = np.array(hdus[args.hdu].data, dtype=float)
        header = hdus[args.hdu].header
    _, spatial = physical_axes(header, data.shape, args.dispersion_axis)
    if not np.isfinite(row) or not 0 <= row <= len(spatial) - 1:
        raise ValueError("Guess row must be a zero-based spatial pixel inside the image.")
    if args.dispersion_axis == 0:
        data = data.T
    edge = max(2, int(np.ceil(len(spatial) * .05)))
    profile = np.nanmedian(data, axis=1)
    star_row = edge + int(np.nanargmax(profile[edge:-edge]))
    return float(np.interp(row, np.arange(len(spatial)), spatial) - spatial[star_row])


# Request an approximate location when none was provided, without ever confirming extraction implicitly.
def initial_guess(path, args):
    if args.guess_row is not None:
        args.offset_arcsec = row_to_offset(path, args, args.guess_row)
    elif args.offset_pixel is not None:
        with fits.open(path, memmap=True) as hdus:
            image = hdus[args.hdu]
            _, spatial = physical_axes(image.header, image.data.shape, args.dispersion_axis)
        args.offset_arcsec = args.offset_pixel * float(spatial[1] - spatial[0])
    elif args.offset_arcsec is None:
        if args.preview_only or args.confirm or not sys.stdin.isatty():
            raise ValueError("Provide --offset-arcsec or --guess-row for noninteractive use.")
        value = input(f"{path.name}: source guess (zero-based spatial row, or offset:ARCSEC from star): ").strip()
        if value.lower().startswith("offset:"):
            args.offset_arcsec = float(value.split(":", 1)[1])
        else:
            args.offset_arcsec = row_to_offset(path, args, float(value))


# Render both saved PNGs inside an editable review window; closing it always cancels extraction.
def gui_review(search_path, background_path, args, usable, comparison_path=None):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, CheckButtons, RadioButtons, TextBox

    figure = plt.figure(figsize=(15, 11))
    display = figure.add_axes([.02, .335, .96, .645])
    artist = display.imshow(plt.imread(search_path))
    display.axis("off")
    status = figure.text(.02, .065, "Review both views. Edits require Apply & rerun. Closing cancels.", fontsize=10)
    outcome = {"action": "cancel"}
    fields = {}
    # Expose parameters that materially change localization, aperture throughput, or sky contamination.
    definitions = [
        ("offset_arcsec", "Guess offset (arcsec)", str(args.offset_arcsec)),
        ("search_radius_arcsec", "Search radius (arcsec)", str(args.search_radius_arcsec)),
        ("bin_pixels", "Bin width (pixels)", str(args.bin_pixels)),
        ("aperture_arcsec", "Aperture width (arcsec)", str(args.aperture_arcsec)),
        ("background_windows", "Sky offsets (arcsec)", ",".join(f"{a:g}:{b:g}" for a, b in args.background_windows)),
        ("background_degree", "Sky degree (0 or 1)", str(args.background_degree)),
        ("wave_ranges", "Wavelengths (Å; blank=all)", ",".join(f"{a:g}:{b:g}" for a, b in args.wave_ranges or [])),
        ("noise_percentile", "Noise percentile", str(args.noise_percentile)),
    ]
    for index, (key, label, value) in enumerate(definitions):
        column, row = index % 2, index // 2
        box_axes = figure.add_axes([.19 + column * .49, .24 - row * .049, .29, .03])
        fields[key] = TextBox(box_axes, label + "  ", initial=value)
    switches = CheckButtons(figure.add_axes([.02, .005, .20, .05]), ["Subtract one star"], [args.substars])
    views = {"Diagnostics": search_path, "Background": background_path}
    if comparison_path is not None:
        views["Extraction"] = comparison_path
    tabs = RadioButtons(figure.add_axes([.02, .275, .18, .06]), list(views), active=0)
    modes = ("boxcar", "horne", "both")
    original_method = getattr(args, "method", "boxcar")
    method_selector = RadioButtons(figure.add_axes([.24, .275, .16, .06]), modes,
                                    active=modes.index(original_method))
    initial_values = {key: box.text for key, box in fields.items()}
    original_substars = args.substars

    # Switch between full-resolution diagnostic and effective-background figures without accepting either.
    def switch_view(label):
        artist.set_data(plt.imread(views[label]))
        display.relim()
        display.autoscale_view()
        figure.canvas.draw_idle()

    # Require a recomputation for every edited field so confirmation always matches the displayed products.
    def confirm(event):
        dirty = any(fields[key].text != value for key, value in initial_values.items())
        dirty |= switches.get_status()[0] != original_substars
        dirty |= method_selector.value_selected != original_method
        if dirty or not usable:
            status.set_text("Apply & rerun before confirming." if dirty else
                            "A requested method has no usable columns; inspect Extraction or change mode/settings.")
            figure.canvas.draw_idle()
            return
        outcome["action"] = "confirm"
        plt.close(figure)

    # Parse edits as structured values; never execute user-entered expressions or shell commands.
    def rerun(event):
        proposed = copy.deepcopy(args)
        try:
            for key in ("offset_arcsec", "search_radius_arcsec", "aperture_arcsec", "noise_percentile"):
                setattr(proposed, key, float(fields[key].text))
            proposed.bin_pixels = int(fields["bin_pixels"].text)
            proposed.background_windows = interval_list(fields["background_windows"].text)
            proposed.background_degree = int(fields["background_degree"].text)
            proposed.wave_ranges = interval_list(fields["wave_ranges"].text) or None
            proposed.substars = bool(switches.get_status()[0])
            proposed.method = method_selector.value_selected
            validate_settings(proposed)
        except (ValueError, argparse.ArgumentTypeError) as exc:
            status.set_text(str(exc))
            figure.canvas.draw_idle()
            return
        vars(args).update(vars(proposed))
        outcome["action"] = "rerun"
        plt.close(figure)

    # Retain widget references for the entire event loop and bind cancellation explicitly.
    def cancel(event):
        outcome["action"] = "cancel"
        plt.close(figure)

    tabs.on_clicked(switch_view)
    buttons = [Button(figure.add_axes([left, .005, .14, .035]), label)
               for left, label in [(.50, "Apply & rerun"), (.66, "Confirm & extract"), (.82, "Cancel")]]
    for button, callback in zip(buttons, (rerun, confirm, cancel)):
        button.on_clicked(callback)
    plt.show(block=True)
    return outcome["action"]


# Offer an explicit text review for environments where PNGs are inspected in an external viewer.
def terminal_review(search_path, background_path, args, usable, comparison_path=None):
    print(f"Inspect {search_path}\nInspect {background_path}")
    if comparison_path is not None:
        print(f"Inspect {comparison_path}")
    print("Recommended checks: matching independent bands; sky free of source wings/other objects; aperture encloses the source.")
    while True:
        answer = input("Confirm and extract [c], edit and rerun [e], or cancel [q]? ").strip().lower()
        if answer == "c" and usable:
            return "confirm"
        if answer in ("q", "", "cancel"):
            return "cancel"
        if answer == "e":
            # Reuse CLI-like values in simple prompts while preserving previous values on blank answers.
            proposed = copy.deepcopy(args)
            try:
                for key, label in [("offset_arcsec", "Guess offset arcsec"), ("search_radius_arcsec", "Search radius arcsec"),
                                   ("bin_pixels", "Wavelength bin pixels"), ("aperture_arcsec", "Aperture width arcsec")]:
                    value = input(f"{label} [{getattr(args, key)}]: ").strip()
                    if value:
                        setattr(proposed, key, int(value) if key == "bin_pixels" else float(value))
                value = input("Sky windows relative to source, e.g. -1.5:-0.9,0.9:1.5 [keep]: ").strip()
                if value:
                    proposed.background_windows = interval_list(value)
                value = input("Wavelength ranges Angstrom, comma-separated; 'all' clears selection [keep]: ").strip()
                if value:
                    proposed.wave_ranges = None if value.lower() == "all" else interval_list(value)
                value = input(f"Subtract one star? y/n [{'y' if args.substars else 'n'}]: ").strip().lower()
                if value:
                    if value not in ("y", "n"):
                        raise ValueError("Choose y or n.")
                    proposed.substars = value == "y"
                value = input(f"Extraction mode boxcar/horne/both [{getattr(args, 'method', 'boxcar')}]: ").strip().lower()
                if value:
                    proposed.method = value
                validate_settings(proposed)
            except (ValueError, argparse.ArgumentTypeError) as exc:
                print(exc)
                continue
            vars(args).update(vars(proposed))
            return "rerun"
        print("A requested method has no usable extraction columns." if answer == "c" else "Choose c, e, or q.")


# Orchestrate preview revisions; write a spectrum only after an explicit review confirmation.
def main(argv=None):
    cli = build_parser()
    cli.description = ("Locate and review a source for boxcar/Horne extraction, or use --quick "
                       "for automatic boxcar aperture and sky selection.")
    # Track an explicit quick bin request while retaining pixel-bin defaults for reviewed search.
    cli.set_defaults(offset_arcsec=None, output_dir=Path("extraction_results"), bin_pixels=None)
    cli.add_argument("--guess-row", "-gr", "-s", type=float,
                     help="Approximate original zero-based spatial row; alternative to an offset")
    cli.add_argument("--offset-pixel", "-op", type=float,
                     help="Target offset in original spatial pixels from the brightest trace")
    cli.add_argument("--source-rows", "-sr", type=pixel_range,
                     help="Exact zero-based half-open boxcar rows; requires --quick")
    cli.add_argument("--quick", "-q", action="store_true",
                     help="Automatically choose a boxcar aperture and clean sky windows, then extract")
    cli.add_argument("--plot-spectra", "-p", action="store_true",
                     help="Save a dedicated 1D spectrum PNG after extraction; automatic with --quick")
    graphics = cli.add_mutually_exclusive_group()
    graphics.add_argument("--graphic", "-g", dest="graphic", action="store_true",
                          help="Open the saved 1D PNG with the default system image viewer")
    graphics.add_argument("--no-graphic", dest="graphic", action="store_false",
                          help="Do not open a viewer, including with --quick")
    cli.set_defaults(graphic=None)
    cli.add_argument("--aperture-arcsec", type=float, default=1.2, help="Full boxcar width (default 1.2 arcsec)")
    cli.add_argument("--method", choices=("boxcar", "horne", "both"), default="boxcar",
                     help="Extraction method; boxcar remains the default; both saves separate spectra")
    cli.add_argument("--gap-policy", choices=("split", "zero"), default="split",
                     help="split: separate valid regions; zero: full grid, one primary FITS with bands and one ASCII")
    cli.add_argument("--background-window", dest="background_windows", action="append", type=physical_range,
                     help="LOW:HIGH arcsec relative to target; repeat; use = before negative values")
    cli.add_argument("--background-degree", type=int, choices=(0, 1), default=1)
    cli.add_argument("--star-exclude-arcsec", type=float, default=1.2, help="Exclude reference-star neighborhood from sky windows")
    cli.add_argument("--background-sigma", type=float, default=5.)
    # Keep background-choice QA opt-in and configure angular shifts separately from fractional width changes.
    cli.add_argument("--background-sensitivity", action="store_true",
                     help="After confirmation, save separate background-choice QA tables and plots")
    cli.add_argument("--bg-shift-arcsec", type=float, default=.1,
                     help="Background QA inward/outward shift in arcsec (default 0.1; zero disables shifts)")
    cli.add_argument("--bg-width-fraction", type=float, default=.2,
                     help="Background QA fractional widening/narrowing in [0,1) (default 0.2; zero disables)")
    # Profile QA changes only trial kernels after confirmation, keeping the physical aperture and sky fixed.
    cli.add_argument("--horne-qa", action="store_true", help="Save optional Horne spatial-profile reliability diagnostics")
    cli.add_argument("--profile-step-mode", choices=("auto", "fractional"), default="auto",
                     help="auto: use a supported conditional width error; fractional: use configured fractions")
    cli.add_argument("--profile-centroid-fraction", type=float, default=.05,
                     help="Rigid kernel offset as a fraction of fitted FWHM (default 0.05)")
    cli.add_argument("--profile-fwhm-fraction", type=float, default=.1,
                     help="Fallback or fractional-mode FWHM change (default 0.1)")
    cli.add_argument("--profile-trace-fraction", type=float, default=.05,
                     help="Maximum chromatic kernel drift as a fraction of fitted FWHM (default 0.05)")
    cli.add_argument("--profile-sigma-multiplier", type=float, default=1.,
                     help="Multiplier for supported formal width errors in auto mode (default 1)")
    cli.add_argument("--review", choices=("gui", "terminal"), default="gui")
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--preview-only", action="store_true", help="Save pending review PNGs; never extract")
    mode.add_argument("--confirm", action="store_true", help="Explicit batch confirmation of supplied settings; bypass interactive review")
    args = cli.parse_args(argv)
    args.quick_bin_pixels = args.bin_pixels if args.quick else None
    if args.bin_pixels is None:
        args.bin_pixels = 50
    args.background_windows = args.background_windows or [(-1.5, -.9), (.9, 1.5)]
    choices = [args.guess_row is not None, args.offset_arcsec is not None,
               args.offset_pixel is not None, args.source_rows is not None]
    if sum(choices) > 1:
        cli.error("Choose exactly one of --guess-row, --offset-arcsec, --offset-pixel, or --source-rows.")
    if args.quick:
        from .quick import main as quick_main
        return quick_main(args, cli)
    if args.source_rows is not None:
        cli.error("--source-rows requires --quick.")
    stems = [path.stem for path in args.images]
    if len(stems) != len(set(stems)):
        cli.error("Input filenames need distinct stems.")
    try:
        # Check the GUI before expensive analysis and prevent replacement of prior extraction products.
        if args.review == "gui" and not (args.preview_only or args.confirm):
            import matplotlib
            from matplotlib import rcsetup
            if matplotlib.get_backend().lower() not in [backend.lower() for backend in rcsetup.interactive_bk]:
                raise ValueError("GUI backend unavailable. Use --preview-only or --review terminal.")
        for path in args.images:
            # Protect the full optional Horne QA family, including plots and structured reports.
            if args.horne_qa and not args.overwrite:
                if any(item.exists() for item in profile_qa_paths(args.output_dir / path.stem).values()):
                    raise ValueError("Existing Horne QA products are protected; choose another directory or --overwrite.")
            # Protect the separate QA product family before creating any previews or science files.
            if args.background_sensitivity and not args.overwrite:
                qa_paths = sensitivity_paths(args.output_dir / path.stem, ("boxcar", "horne"))
                if any(item.exists() for pair in qa_paths.values() for item in pair.values()):
                    raise ValueError("Existing background-sensitivity products are protected; choose another directory or --overwrite.")
            for suffix in ("_review.json", "_spectrum.ecsv", "_boxcar_spectrum.ecsv", "_horne_spectrum.ecsv", "_regions.fits"):
                if (args.output_dir / (path.stem + suffix)).exists() and not args.overwrite:
                    raise ValueError("Existing workflow products are protected; choose another directory or --overwrite.")
            # Protect science-only products as well, including parts left by an earlier masked-spectrum extraction.
            for suffix in ("_spectrum", "_boxcar_spectrum", "_horne_spectrum"):
                if existing_science_exports(args.output_dir / (path.stem + suffix)) and not args.overwrite:
                    raise ValueError("Existing science exports are protected; choose another directory or --overwrite.")
            if (args.plot_spectra or args.graphic) and not args.preview_only and not args.overwrite:
                plots = spectrum_plot_paths(args.output_dir / path.stem, args.method)
                if any(plot.exists() for plot in plots.values()):
                    raise ValueError("Existing 1D spectrum plot is protected; choose another directory or --overwrite.")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path in args.images:
            current = copy.deepcopy(args)
            initial_guess(path, current)
            history = []
            revision = 0
            while True:
                validate_settings(current)
                revision += 1
                print(f"Analyzing {path.name}; preview revision {revision} ...", flush=True)
                report, arrays = analyze_image(path, current)
                result = prepare_extraction(path, report, current)
                prefix = args.output_dir / path.stem
                search_path = Path(str(prefix) + f"_review{revision:02d}_search.png")
                background_path = Path(str(prefix) + f"_review{revision:02d}_background.png")
                comparison_path = (Path(str(prefix) + f"_review{revision:02d}_comparison.png")
                                   if current.method in ("horne", "both") else None)
                if not current.overwrite and (search_path.exists() or background_path.exists()
                                               or (comparison_path is not None and comparison_path.exists())):
                    raise ValueError("Preview revision already exists; use another directory or --overwrite.")
                plot_diagnostics(report, arrays, search_path)
                plot_background(result, report, current, background_path)
                if comparison_path is not None:
                    plot_comparison(result, report, current, comparison_path)
                history.append({"revision": revision, "settings": report["settings"],
                                "candidate": report["candidate"], "search_png": str(search_path),
                                "background_png": str(background_path),
                                "comparison_png": None if comparison_path is None else str(comparison_path)})
                method_summaries = {name: {"usable_columns": int(np.sum(item["flags"] == 0)),
                    "profile": None if item["profile"] is None else {
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in item["profile"].items()}}
                    for name, item in result["methods"].items()}
                review = {**report, "review_confirmed": False, "status": "pending", "history": history,
                          "spectrum_written_this_review": False,
                          "previous_spectrum_present": any(Path(str(prefix) + suffix).exists() for suffix in
                              ("_spectrum.ecsv", "_boxcar_spectrum.ecsv", "_horne_spectrum.ecsv")),
                          "extraction": {"method": current.method, "methods": method_summaries,
                                         "specreduce_version": result["specreduce_version"],
                                         "usable_columns": int(np.sum(result["flags"] == 0)),
                                         "aperture_arcsec": current.aperture_arcsec,
                                         "background_windows_relative_to_target_arcsec": current.background_windows,
                                         "background_degree": current.background_degree,
                                         "substars": current.substars}}
                review_path = Path(str(prefix) + "_review.json")
                review_path.write_text(json.dumps(review, indent=2, allow_nan=False) + "\n")
                print(f"Candidate offset {report['candidate']['offset_arcsec']:+.3f} arcsec; "
                      f"usable spectrum columns {review['extraction']['usable_columns']}; stars {'ON' if current.substars else 'OFF'}", flush=True)
                for name, item in method_summaries.items():
                    print(f"  {name}: {item['usable_columns']} usable columns; "
                          f"profile {item['profile']['status'] if item['profile'] else 'boxcar aperture'}", flush=True)
                if current.preview_only:
                    print(f"Pending review: {search_path} and {background_path}; no new spectrum written.")
                    if comparison_path is not None:
                        print(f"Extraction comparison: {comparison_path}")
                    break
                usable = all(np.any(item["flags"] == 0) for item in result["methods"].values())
                action = "confirm" if current.confirm else (
                    gui_review(search_path, background_path, current, usable, comparison_path) if current.review == "gui" else
                    terminal_review(search_path, background_path, current, usable, comparison_path))
                if action == "rerun":
                    continue
                if action == "confirm":
                    if not usable:
                        raise ValueError("A requested method has no valid extraction columns; inspect its profile or revise settings/mode.")
                    # Analyze only the final confirmed settings; no source trace, aperture or Horne refit is performed.
                    sensitivity = (analyze_background_sensitivity(result, current)
                                   if current.background_sensitivity else None)
                    # Keep profile trials independent of background trials; neither diagnostic mutates the reference.
                    horne_qa = analyze_horne_qa(result, current) if current.horne_qa else None
                    save_extraction(result, report, current, prefix)
                    review.update(review_confirmed=True, status="extracted", spectrum_written_this_review=True)
                    review["science_exports"] = result["science_exports"]
                    review_path.write_text(json.dumps(review, indent=2, allow_nan=False) + "\n")
                    # Plot saved numerical spectra only after confirmation, without altering their samples.
                    if current.plot_spectra or current.graphic:
                        review["spectrum_plots"] = {}
                        for name, spectrum_png in spectrum_plot_paths(prefix, current.method).items():
                            extracted = result["methods"][name]
                            scale = result["normalization"]
                            plot_spectrum(result["wavelength"], extracted["flux"] * scale,
                                          extracted["error"] * scale, extracted["flags"],
                                          report["flux_unit"], spectrum_png, path.name)
                            review["spectrum_plots"][name] = str(spectrum_png)
                            print(f"Saved 1D spectrum plot: {spectrum_png}")
                            if current.graphic:
                                opened, message = open_image(spectrum_png)
                                print(("Viewer: " if opened else "Viewer skipped: ") + message)
                    # QA stays in companion products and never changes statistical errors or science-export formats.
                    if sensitivity is not None:
                        review["background_sensitivity"] = save_background_sensitivity(
                            sensitivity, result, report, current, prefix)
                        for products in review["background_sensitivity"].values():
                            print(f"Saved background-choice QA: {products['ecsv']} and {products['png']}")
                    # Store Horne QA beside science products, preserving science uncertainties and formats.
                    if horne_qa is not None:
                        review["horne_qa"] = save_horne_qa(horne_qa, result, report, current, prefix)
                        print(f"Saved Horne profile QA: {review['horne_qa']['report']}")
                    for name in result["methods"]:
                        suffix = f"_{name}_spectrum.ecsv" if current.method == "both" else "_spectrum.ecsv"
                        print(f"Saved confirmed {name} spectrum: {prefix}{suffix}")
                        for products in result["science_exports"][name]:
                            print(f"Saved IRAF spectrum: {products['fits']} and {products['ascii']}")
                            print(f"IRAF: splot \"{products['iraf_image']}\"")
                else:
                    review["status"] = "cancelled"
                    print("Review cancelled; no new spectrum written.")
                review_path.write_text(json.dumps(review, indent=2, allow_nan=False) + "\n")
                break
    except (ValueError, OSError, KeyError, IndexError, EOFError, KeyboardInterrupt) as exc:
        cli.error(str(exc) or "Review interrupted; extraction was not confirmed.")


# Provide direct module execution in addition to the package's extract subcommand.
if __name__ == "__main__":
    main()
