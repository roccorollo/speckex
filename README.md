# speckex

Version **1.0.0** — inspect, locate, review, and extract a source in a
two-dimensional FITS spectrum.

Released under the [MIT License](LICENSE).

The profile command measures profiles over user-selected dispersion intervals,
plots them alongside the image, and optionally fits a positive Gaussian plus a
linear residual background to each profile. It reports candidate spatial
positions and widths. A separate continuum search models a bright star in
wavelength bins to locate a fainter nearby source. A reviewed extraction workflow
then fits local backgrounds and produces boxcar and/or Gaussian Horne spectra.
The tool does not perform a full raw-data or two-dimensional sky reduction.

## Requirements and installation

Python 3.10 or later, with NumPy, SciPy, Astropy, Matplotlib, Specreduce, and
Specutils (requirements are in `pyproject.toml`). The extraction adapter is
tested and pinned to **Specreduce 1.5.1**, with **Specutils 2.1.0** used for
verification. Other Specreduce versions are rejected rather than using an
unverified API path. From the project directory, with dependencies available:

```bash
python -m speckex --help
python -m speckex --version
```

Alternatively, install the package with `python -m pip install .` to provide the
`speckex` command. Interactive use needs a working Matplotlib GUI backend.
Installing with `python -m pip install .` resolves the declared dependencies.
The adapter supplies explicit uncertainties because Specreduce 1.5.1 does not
return them with the extracted spectra.

## Quick boxcar extraction

Use top-level `-q` or `--quick` to supply one approximate source location and
run a boxcar extraction immediately. The existing `extract -q` and
`extract --quick` forms remain equivalent. Choose one location form:
`--guess-row` (`-gr` or `-s`), `--offset-arcsec` (`-oa`),
`--offset-pixel` (`-op`), or exact zero-based half-open `--source-rows` (`-sr`).
Offsets are relative to the brightest positive trace; a positive pixel offset
moves toward increasing original spatial row. An angular offset needs an
angular spatial WCS or a documented arcsec-per-pixel `PIXSCALE` in the FITS
header. Examples, using values selected for the user's own image:

```bash
python -m speckex -q spectrum_2d.fits -gr "$SOURCE_ROW" \
    --output-dir output/quick
python -m speckex -q spectrum_2d.fits -sr "$SOURCE_START:$SOURCE_STOP" \
    --output-dir output/quick_rows
python -m speckex -q spectrum_2d.fits -sr "$SOURCE_START:$SOURCE_STOP" \
    -b 20 --output-dir output/quick_binned
```

The mode estimates an aperture width from the positive spatial profile unless
exact source rows are given. An offset follows the bright reference trace; a
row guess or row interval uses a fixed center. It ranks candidate sky intervals
on both sides using the wavelength-collapsed profile, avoiding strong positive
and negative features; one available side uses a constant sky. The selected
regions and effective pixels are saved in JSON, a region-mask FITS, and a
three-panel PNG showing the 2D image, used pixels, and spectrum. Quick mode
also saves a dedicated 1D spectrum PNG and opens it with the system's default
image viewer on Linux or macOS. Use `--no-graphic` when no viewer should open;
the PNG is still saved. The command automatically accepts its selected regions
and writes a boxcar spectrum; no review window or `--confirm` is needed.
Automatic choices are provisional:
inspect the 2D region figure, especially around nod residuals, nearby sources,
and wavelength-dependent contamination. Use the reviewed workflow or explicit
`flux-only` regions when these choices are unsuitable.

If a matching `ERRS` extension exists, quick extraction assumes it contains
1-sigma standard deviations and propagates input and fitted-background
uncertainty. An explicit `--error-hdu` can select another extension; declared
variance or inverse variance is rejected. With no error extension, the same
boxcar and sky selection runs unweighted and writes flux plus quality, without
a formal error or S/N. Wavelengths need a linear solution for FITS export.
The ECSV retains the full grid and flags; FITS and TXT are the science spectra.
Existing outputs are protected unless `--overwrite` is supplied.

With quick extraction, an explicit `--bin-pixels N` (`-b N`) also saves a
separate binned ECSV, FITS, TXT, and 1D PNG. `N` is the integer number of
adjacent native wavelength pixels per output bin. Speckex records the pixel
count and resulting width in Å in the ECSV metadata, FITS header, and JSON
report. It averages complete groups only; a trailing incomplete
group is dropped, and any group containing a flagged native sample stays invalid.
The original native-grid science files remain available. The binned PNG shows
the saved binned flux and is the figure quick mode opens by default; `--no-graphic`
still saves both plots. `-b` requires at least two pixels per bin and two complete
output bins. When input standard deviations are available, errors are propagated
as `sqrt(sum(sigma_i²))/N`, assuming independent wavelength samples; upstream
resampling covariance is not included. Without an input error image, binned
outputs contain flux and quality only, with no formal error. The same `-b` alias
sets pixel groups for search and reviewed extraction; only quick mode creates
the additional binned science spectrum. All saved wavelength columns, FITS
coordinates, JSON wavelength values, and plot wavelength axes use Å.

For reviewed `extract` and explicit `flux-only`, add `--plot-spectra` (`-p`)
to save the same separate 1D plot after extraction. Add `--graphic` (`-g`) to
save and open that plot. Only unflagged samples are drawn; gaps stay visible.
The plotted median overlay and robust display limits never alter FITS, TXT,
or ECSV values. A 1-sigma band appears only when an input error image supplied
a propagated error. If no desktop viewer is available, extraction and PNG
saving still complete.

## Flux-only extraction without a 2D error image

Use `flux-only` when a rectified 2D science FITS has no matching pixel-error
image. Inspect that image first, then supply zero-based, half-open source and
source-free background row intervals. For example, set the row limits for your
own observation before running:

```bash
python -m speckex flux-only spectrum_2d.fits \
    --source-rows "$SOURCE_START:$SOURCE_STOP" \
    --background-rows "$SKY1_START:$SKY1_STOP" \
    --background-rows "$SKY2_START:$SKY2_STOP" \
    --output-prefix output/flux_only_spectrum
```

The command reads the wavelength WCS and input flux unit from the FITS header.
An angular spatial WCS is used when present; a `PIXSCALE` header value is used
only when its comment explicitly identifies arcsec per pixel. If neither is
available, the supplied pixel-row regions still permit extraction, and the
report records that no angular spatial scale was available. The command sums
complete source rows after fitting a constant or linear background in the
specified rows, with robust clipping of background outliers. It does not
locate a source or choose background regions automatically. Negative nod traces,
other sources, and spatially varying residuals must be checked for each image;
their positions are never assumed fixed. Use `--exclude-columns START:STOP`
for known bad spectral columns.

Outputs are a full-grid ECSV table, a flux-only FITS primary image with a
`QUALITY` extension, a two-column wavelength/flux ASCII file, a three-panel
2D region/mask/spectrum plot, and a JSON provenance report. FITS declares
`ERRTYPE=NONE`. Rejected
flux samples are NaN and have nonzero quality; no error or S/N column is
invented. Flux retains the input `BUNIT` and is a spatial sum with no aperture
correction. The output FITS requires a linear wavelength grid and performs no
resampling. Existing output files are protected.

The reviewed `extract` workflow requires a valid 2D error image for its
weighted search, Horne method, and propagated uncertainties. The flux-only
background fit is unweighted, and its clipping scatter is not reported as a
measurement uncertainty. It does not account for source-dependent Poisson noise,
trace or background-model uncertainty, or resampling covariance. A flux-only
spectrum must not be interpreted as having a calibrated detection significance.

## Reviewed extraction

```bash
python -m speckex extract spectrum_vis.fits spectrum_nir.fits \
    --offset-arcsec 3 --output-dir extraction_results
```

Use your own filenames and approximate offset. Dispersion is horizontal by
default; science/error extensions default to `FLUX` and `ERRS`, with `ERRS`
**assumed to contain one-sigma standard deviations**. Confirm that interpretation
with the data producer. If the guess is omitted, the tool asks for an original
zero-based spatial pixel row, or `offset:VALUE` in arcsec from the reference star.
Alternatively supply `--guess-row ROW` (`-gr` or `-s`) or an original-row
offset with `--offset-pixel PIXELS` (`-op`). `--offset-arcsec` also accepts
`-oa`. Row guesses are file-specific; use
separate invocations when different arms need different pixel rows. A physical
offset can be shared between arms.

The tool saves a diagnostic PNG and a background PNG, then opens a Matplotlib
review window with **Diagnostics** and **Background** views. It requires a
working GUI backend. Inspect the fitted position across independent wavelength
groups and check the aperture and sky samples in the background view.

- Select the extraction mode and edit the offset, search radius, wavelength-bin size, aperture width, sky windows,
  background degree, wavelength intervals, or noise percentile in the fields.
- **Apply & rerun** recomputes both views and saves a numbered preview revision.
- **Confirm & extract** writes the spectrum only when the displayed settings
  have not changed and usable extraction columns exist.
- **Cancel**, or closing the window, leaves the current review unconfirmed.

### Extraction modes

`--method boxcar` is the default and retains the previous fractional-aperture
flux, local sky subtraction, and propagated-error behavior. `--method horne`
uses a fitted Gaussian source profile and variance weighting.
`--method both` computes and saves both methods without selecting a winner:

```bash
python -m speckex extract spectrum.fits --offset-arcsec 3 \
    --method both --output-dir extraction_results
```

Horne and both modes add an **Extraction** review view comparing the target's
spatial profile, Gaussian, boxcar support, extracted fluxes with propagated 1σ
errors, and S/N per original spectral pixel. Solid wavelength-bin medians affect
display only; they are not rebinned measurements or detection significances.
Plot limits are robustly chosen, so some raw outliers lie outside the view;
all samples remain in the saved spectra. S/N never changes the selected method.
Changing the mode requires **Apply & rerun** before confirmation.

Horne requires explicit valid variances. The workflow squares supplied standard
deviations (`ERRS` by default); **verify their statistical meaning**. Missing or
unusable error arrays stop the run. Invalid source pixels retain the existing
whole-column flag; every unmasked used variance must be finite and positive.
All-zero variances are rejected before Specreduce can replace them with uniform weights.

The Gaussian width is fitted to aligned, sky/star-subtracted source-aperture
data, with its center fixed to the reviewed trace. It needs eight usable
wavelength columns and five spatial samples; wider apertures may be needed for
coarsely sampled images. Widths that hit bounds or weak/unconstrained fits are
flagged with quality code **8**, not replaced with the search guess or a different
method. Confirmation is blocked if a requested method has no usable columns;
inspect its profile or explicitly change the mode/settings and rerun.

Profile interpolation ignores neighbors with zero weight and excludes columns
with invalid contributing samples from the profile estimate. At least eight
complete valid profile columns must remain; no substitute variance or width is
introduced.

Science extraction evaluates Gaussian kernels on original pixels. A
version-specific adapter avoids Specreduce 1.5.1's non-flat science-only image
shift, leaving variance and masks co-registered. The Gaussian kernel contains
no second polynomial background: both methods use the same existing fitted sky
and optional star model. The source Gaussian is frozen when extracting sky/star
spectra; each method applies its own source weights to those same models.

Both methods report the same **finite-aperture flux definition**. Specreduce's
Horne normalization is converted to the fractional-aperture sum expected for
that fitted profile; it is not reported as total-profile flux. This permits
comparisons without silently aperture-correcting boxcar. Differences can still
reveal non-Gaussian profiles, sky/star residuals, variance problems, or spatially
extended source emission.

Propagated 1σ includes source-pixel variance and the full fitted sky/star
coefficient covariance projected through each method's weights, including
sky/star cross terms. It avoids treating a shared sky estimate as independent
pixel noise. It excludes fitted source-profile uncertainty, resampling covariance,
and star-shape/trace systematics. No variance or profile is refitted to a sky-only
image to obtain the extracted sky spectrum.

Nearby-star subtraction is **off by default**. Add `--substars` (or `-ss`) to
subtract one dominant reference star, or enable the review checkbox and rerun.
The star is fitted for tracing even when subtraction is off; its flux then
remains in the data. Search subtraction removes only its stellar component,
retaining the background. During extraction, the star amplitude is fitted anew
at each wavelength with a Moffat shape interpolated from the continuum bins;
local sky is fitted jointly. This removes one modeled star, not multiple objects.

For a first pass, start with the default full aperture width of **1.2 arcsec**
and two background windows **−1.5:−0.9** and **+0.9:+1.5 arcsec relative to the
target**. These are starting values, not instrument-specific prescriptions.
The source center follows the reference-star trace plus the confirmed target
offset. The background view overlays the requested regions and separately
shows effective source, sky, and optional stellar-fit pixels after exclusions.
Its bottom panel shows the prospective sky-subtracted spectrum and fitted sky;
median overlays and image smoothing affect display only.

Useful adjustments:

- Compare **25, 50, and 100 pixel bins**: larger bins stabilize weak continuum
  profiles but reduce wavelength resolution in the position measurement.
- Restrict `--wave-range LOW:HIGH` in Å (repeat for separate regions) to
  avoid known contamination. These intervals also restrict extracted coverage.
- Compare aperture widths such as **1.0, 1.2, and 1.4 arcsec**, inspecting
  source wings and increased noise. Fluxes are aperture sums with no aperture correction.
- Move sky windows away from source wings, residual structures, other objects,
  or slit edges. Keep sky samples on both sides for a linear background;
  prefer a constant model (`--background-degree 0`) for a single clean side.
- Enable star subtraction only if its wings contaminate the target/sky region,
  and compare results with subtraction off. A poor stellar shape can imprint residuals.

For explicit background settings, use `=` before a negative interval:

```bash
python -m speckex extract spectrum.fits --offset-arcsec 3 \
    --aperture-arcsec 1.2 \
    --background-window=-1.5:-0.9 --background-window=0.9:1.5
```

The sky model is fitted independently in every wavelength column, excluding
the source aperture and the star neighborhood (`--star-exclude-arcsec`, default
1.2). Sky outlier clipping is controlled by `--background-sigma` (default 5).
The extraction reads original un-clipped science pixels, so continuum-search
clipping cannot erase a source emission line. Invalid aperture pixels flag the
column instead of scaling missing source flux. No trace extrapolation is made
beyond accepted coverage; gaps more than three bin widths from a valid reference
bin are also flagged. Shorter gaps use interpolated reference-star parameters.

The error includes independent source-pixel noise and the covariance of the
fitted nuisance model summed over the aperture. It omits resampling covariance,
trace uncertainty, and uncertainty in the star shape. Inspect region coverage
and residual structure before scientific interpretation.

Confirmed science spectra are written as **FITS and plain ASCII**, with
**wavelength in Angstrom** and calibrated **flux density and 1σ error in
erg s^-1 cm^-2 Angstrom^-1**. Uncalibrated input units such as electrons remain
uncalibrated. A single method produces `_spectrum.fits` and `_spectrum.txt`;
both mode produces `_boxcar_spectrum` and `_horne_spectrum` pairs without
selecting a preferred method. ASCII has three columns: wavelength, flux, and
standard deviation, each written with **nine significant digits**.

By default, FITS stores full-precision flux in its 1D **primary image**, with linear
FITS/IRAF spectral WCS and propagated standard deviations in the **ERROR**
image extension. Open the primary image with `splot "spectrum.fits[0]"`.
The `[0]` selector is required by IRAF for these multi-extension files. The
error image can be viewed with `splot "spectrum.fits[ERROR]"`; its presence
does not automatically supply an error model for IRAF line measurements.
See the [IRAF spectral-image conventions](https://iraf.readthedocs.io/en/latest/tasks/noao/onedspec/specwcs.html).

Default science exports contain finite, unflagged, positive-error samples only. Invalid
edges are trimmed. Internal gaps produce separate `_part001`, `_part002`, etc.
FITS/ASCII pairs, each retaining the correct wavelength zero point and pixel
spacing. No interpolation or replacement values are introduced. Nonlinear
wavelength grids cannot be represented by this linear IRAF exporter and are
rejected before science files are written; no resampling is performed.

### One file retaining gaps

Select `--gap-policy zero` to retain the entire original wavelength grid in
**one FITS and one ASCII file**:

```bash
python -m speckex extract spectrum.fits --offset-arcsec OFFSET \
    --gap-policy zero --output-dir extraction_results
```

Replace `OFFSET` with the approximate separation in arcsec. Missing samples
have **flux = 0 and standard deviation = 0**. Zero standard deviation is a
missing-data marker, not a measurement with zero uncertainty. Valid ASCII
measurements require `error > 0`; a measured zero flux with positive error
remains valid. ASCII keeps exactly three columns and nine significant digits.
No continuum interpolation is performed across gaps.

This option writes a **single primary FITS HDU, without extensions**, as an
IRAF spectral image with shape `(3, 1, N)`: flux in band 1, standard deviation
in band 2, and dimensionless quality flags in band 3. Nonzero quality marks
missing data; flux/error have the same physical units. Use:

```text
splot "spectrum.fits" line=1 band=1
```

The individual images are also accessible as `spectrum.fits[*,1,1]` for flux,
`[*,1,2]` for standard deviation, and `[*,1,3]` for quality. `splot` does not
automatically mask zero placeholders or use the sigma band for line errors;
exclude missing ranges from continuum fits and measurements. The default
`--gap-policy split` retains the earlier separate-region output layout.

For compatibility, the full-grid diagnostic ECSV remains `_spectrum.ecsv`, or
`_boxcar_spectrum.ecsv` and `_horne_spectrum.ecsv` in both mode. These diagnostics
retain wavelength in **Å**, finite-aperture flux and propagated 1σ error,
method-weighted sky and subtracted star with their own errors, S/N, target trace,
and quality flags, including rejected samples. `_regions.fits` contains aperture
weights, sky/star-fit samples, each method's actual extraction weights, and
physical coordinate arrays. Search controls, wavelength intervals and diagnostic
plots also use **Å**.
Horne metadata records the fitted Gaussian and
the per-column aperture fraction. ECSV flux-density units retain the input units;
science exports convert calibrated wavelength densities to cgs per Angstrom.
The flux is a spatial sum, not an integral over wavelength. The `_review.json`
records confirmation status and every preview revision's settings and position.
Old confirmed products remain untouched if an `--overwrite` rerun is cancelled;
the review JSON identifies whether a spectrum was written in the current review.

For headless work, `--preview-only` saves PNGs and a pending review without
extracting. `--review terminal` supplies explicit confirmation/edit prompts after
you inspect those PNGs in your viewer. `--confirm` is an explicit batch
confirmation of supplied settings and bypasses the review window. Existing
workflow products are protected unless `--overwrite` is given.

### Background-sensitivity diagnostics

Add `--background-sensitivity` to compare local-background choices after the
final extraction settings are confirmed:

```bash
python -m speckex extract spectrum.fits --offset-arcsec OFFSET \
    --method both --background-sensitivity \
    --bg-shift-arcsec 0.1 --bg-width-fraction 0.2 \
    --output-dir extraction_results
```

Replace `OFFSET` with your approximate source separation in arcsec. QA is
optional and runs only after interactive confirmation or explicit `--confirm`;
preview-only and cancelled reviews produce no new sensitivity products.
It never changes the reference science spectrum, errors, quality flags, or
FITS/ASCII formatting, and introduces no automatic extraction rejection.

The configurations are the original windows as reference; windows shifted
outward/inward by `--bg-shift-arcsec`; widths multiplied by
`1 +/- --bg-width-fraction` about their original midpoints; constant versus
linear sky on the original windows; and left-only/right-only constant sky.
Shifts must be nonnegative, and the width fraction must be in `[0,1)`.
Zero disables that perturbation family; duplicate configurations are skipped.
The defaults are starting values, not universally appropriate sky regions.

Source trace, aperture, input errors, reference-star shape and exact confirmed
source weights stay fixed. In particular, **Horne's original Gaussian profile
is not refitted**. When stellar subtraction is enabled, its amplitude is refitted
jointly with each alternative sky, retaining the same stellar shape and
subtraction setting. Sky clipping and reference-star exclusion retain their
confirmed settings. Existing masks are honored by the analysis API when carried
in the reference result; the extraction CLI does not infer a quality mask.

Overlapping or empty windows are skipped. An alternative is unavailable at
wavelengths where its fit fails, its requested windows extend beyond the spatial
image, or too few sky samples survive validity checks, masks, star exclusion
and clipping. Constants require at least three retained sky pixels; linear
models require at least two on each side, plus the existing full-rank fit and
residual-sample requirements. These checks establish fit support, not absence
of contamination by an unknown source or structured residuals.

For each method, the companion `_boxcar_background_sensitivity.ecsv` or
`_horne_background_sensitivity.ecsv` retains every original wavelength in Å,
with flux quantities in the original input flux units. It contains:

- `REFERENCE_FLUX`, `SIGMA_STAT`, reference quality and the fixed target trace.
- Each configuration's flux, `DELTA_name = FLUX_name - REFERENCE_FLUX`, and
  separate QA flags. Metadata records windows, degrees, succeeded/partial/
  failed/skipped outcomes and flag counts.
- `N_ACCEPTED`, the number of valid unique alternatives at each wavelength,
  **excluding the reference**; their mean, minimum and maximum.
- `BG_SENS = sqrt(sum((FLUX_alternative - mean_alternative)^2) / N_ACCEPTED)`
  and `BG_RATIO = BG_SENS / SIGMA_STAT`. Both are unavailable when fewer than
  two alternatives contribute. `SIGMA_STAT` is the unchanged reference
  propagated error, including its fitted nuisance-coefficient covariance.

Unavailable samples remain NaN; no wavelength samples are interpolated or
filled. Per-wavelength alternative membership can change, so inspect
`N_ACCEPTED` and QA flags when interpreting a change in sensitivity.
The companion PNG shows the reference and alternative flux range, `BG_SENS`,
and `BG_RATIO` on the native grid, retaining gaps. Existing QA files are
protected unless `--overwrite` is supplied.

**BG_SENS measures background-choice sensitivity, not a statistically
calibrated systematic uncertainty or confidence interval.** Alternatives share
pixels and are correlated. A nonzero value can reflect noise, a real gradient,
contamination, or model mismatch; it does not by itself identify a defect.
A small value does not rule out a bias common to every tested configuration.
No sensitivity term is added in quadrature to statistical errors. The test
does not assess trace/profile uncertainty, total-aperture corrections,
upstream covariance, or all plausible background models.

### Horne spatial-profile reliability QA

Add `--horne-qa` with `--method horne` or `--method both` to save optional
profile diagnostics after confirmation. It can run together with background QA:

```bash
python -m speckex extract spectrum.fits --offset-arcsec OFFSET \
    --method both --horne-qa --background-sensitivity \
    --output-dir extraction_results
```

Replace `OFFSET` with your approximate separation in arcsec. Preview-only and
cancelled reviews create no new Horne QA products. The analyses vary different
assumptions independently; their sensitivities are not combined into an error.
Science FITS, ASCII, ECSV and binning behavior remain unchanged.

The current source model fits one Gaussian amplitude and shared sigma to an
aligned spatial stack. **Its centroid is fixed to the confirmed trace**; neither
a free centroid error nor a trace covariance is fitted. QA retains that fact,
the full confirmed trace, sigma/FWHM, formal width error, imposed width bounds,
fit status and approximate stack-fit statistics. It does not invent positional
uncertainties or replace the source model.

Three one-at-a-time perturbation pairs are tested:

- Rigid source-kernel centroid offsets, default `+/- 0.05` fitted FWHM.
- Gaussian FWHM changes. In `--profile-step-mode auto` (default), use
  `--profile-sigma-multiplier` times the formal width error (default 1) when
  the fit is sufficiently constrained, away from its bounds, and its approximate
  stack residual statistic is acceptable. Otherwise use the fractional change.
- Chromatic trace-kernel shifts: a linear drift from minus to plus the chosen
  displacement across the wavelength range, and its opposite. Default maximum
  drift is `0.05` fitted FWHM. This tests a tilt rather than duplicating the rigid
  centroid offset; it does not represent every possible trace error.

Set `--profile-step-mode fractional` to force fractional width steps.
`--profile-centroid-fraction`, `--profile-fwhm-fraction` and
`--profile-trace-fraction` default to `0.05`, `0.1` and `0.05`, respectively;
each accepts `[0,1)`, with zero disabling that family in fractional mode.
The sigma multiplier must be finite and positive. The report records the actual
perturbations and their basis. Formal width errors are conditional approximations:
the original stack fit omits interpolation, background and upstream covariance.
Their use does not establish a calibrated confidence region for the profile.
Interpolation of a curved, subpixel trace can also broaden the stacked profile;
the formal width error does not include that shape bias. Conditional error
calibration on a known profile does not establish calibration after noisy
profile fitting or selection.

**The physical aperture and its fractional-pixel weights stay fixed**, even
when kernel positions move. Data, input variances, sky/star predictions and
nuisance covariance also stay fixed. Each trial recomputes the native-grid
Horne weighted estimator and its normalization to that same physical aperture;
it does not move the aperture or refit the sky. Duplicate profiles and widths
outside the original fit bounds are skipped and recorded. Reference gaps remain
missing, with no spectral interpolation.

`PROFILE_SENS` is the population RMS of the valid distinct perturbed fluxes
around their mean, excluding the reference. `PROFILE_RATIO` divides this by
the unchanged reference statistical error. At least two alternatives are
required. These quantities describe the selected perturbation ensemble; they
are not systematic-error estimates and are never added to science errors.

The 2D goodness-of-fit check uses source pixels on the original grid. With
independent input-pixel variances `D`, fixed source-row nuisance design `X` and
fitted sky/star coefficient covariance `C`, it uses residual covariance
`D + X C X^T`, including sky/star cross terms. It fits one **signed** source
amplitude per wavelength by generalized least squares. Per-column chi-square
has nominal degrees of freedom `N_source_pixels - 1`, conditional on the fixed
shape and trace. The aggregate subtracts one further degree for the estimated
shared width. That global adjustment is approximate because the width was fitted
to an interpolated stack, not jointly optimized with all native-grid amplitudes.
The trace is held fixed, so its fitted parameters are not counted again here.
Upstream covariance, clipping/selection and model uncertainty remain unmodeled.

The QA report provides advisory flags for weak/uncertain widths, proximity to
width bounds, coherent asymmetric/wing residuals, high conditional chi-square,
flux sensitivity exceeding the reference sigma, and possible contamination.
Its heuristic thresholds are recorded in the JSON report. Residual structure
can also arise from miscentering, extended emission or a non-Gaussian PSF;
these diagnostics cannot uniquely identify a neighboring source. Pooled residual
scores assume independent spectral columns and are descriptive when that
assumption fails. **No advisory automatically rejects Horne extraction.**

When boxcar exists, QA compares both methods' finite-aperture spectra. The
reported difference error propagates their **weight difference**, including
shared source pixels and nuisance covariance; it does not add their errors as
though they were independent. Neither method is an absolute reference.

Products use a separate protected family:

- `_horne_qa.json`: retained fit information, perturbations, goodness-of-fit
  assumptions, advisories and limitations.
- `_horne_qa.ecsv`: native wavelengths in Å, trace, reference flux/error,
  alternative fluxes/differences, sensitivity, valid-profile counts, conditional
  chi-square/residual scores and optional boxcar comparison. Fluxes retain input units.
- `_horne_spatial_qa.ecsv` and `.png`: observed aligned profile, Gaussian,
  residuals and plausible alternatives. Display amplitudes are fixed; trace
  curves illustrate maximum drift. Stack errors can be correlated.
- `_horne_sensitivity_qa.png`: spectra and alternative range, sensitivity,
  sensitivity/error ratio and conditional reduced chi-square.

Existing QA products require explicit `--overwrite`. A failed or unavailable
reference profile can produce an unavailable report through the analysis API;
the confirmed CLI retains its existing requirement for usable requested methods.

## Bin a 1D spectrum

Average groups of **55 adjacent input pixels**, using either input format:

```bash
python -m speckex bin spectrum.fits -b 55
# Alternative input; produces the same filenames, so use a separate directory when comparing.
python -m speckex bin spectrum.txt --bin 55 --output-dir binned_from_ascii
```

`--bin` and `--bin-pixels` are also accepted in place of `-b`. Both commands
write `spectrum_bin55.fits` and `spectrum_bin55.txt`; the default
output directory is the input directory. Existing products are protected.
The bin size is a **pixel count**, not a wavelength width. On a 0.2 Angstrom
input grid, 55 pixels span 11 Angstrom. This reduces spectral resolution.

Each bin contains the arithmetic mean of its input flux densities and the mean
wavelength. For independent input errors, the propagated standard deviation is
`sqrt(sum(sigma_i**2)) / N`. The flux density stays in its original convention;
it is not multiplied by the bin width. Calibrated wavelength densities are
converted to cgs per Angstrom in the output. Negative fluxes remain valid.
Correlated noise and upstream systematic errors are **not** included: binning
resampled spectra does not by itself calibrate their errors. No clipping,
inverse-variance weighting, interpolation of missing data, or detection claim
is introduced.

Only complete bins are saved. Any incomplete trailing group is omitted, with
its pixel count reported in the CLI output and FITS `NDROP` keyword. At least
two complete output bins are required. Process each contiguous spectrum part
separately; internal wavelength gaps, non-finite flux, and non-positive or
missing errors are rejected rather than silently bridging or filling them.

For a spectrum with paired-zero missing samples, choose the full-grid option:

```bash
python -m speckex bin spectrum.fits --bin 55 --gap-policy zero
```

The same option accepts three-column ASCII. Bins are anchored to the first
input pixel and preserve regular spacing. **Any bin containing a missing input
sample remains missing** (`flux=error=0`); placeholders never enter a reported
mean. Fully observed bins use the same mean and error formula above. The final
incomplete group is still omitted. FITS output has the single-primary-HDU band
layout described above; no separate files are created at internal gaps.

FITS input uses the Speckex science layout: a linear 1D primary flux image with
`BUNIT`, wavelength WCS, and an `ERROR` image declared as `ERRTYPE=STDDEV`.
The explicitly identified Speckex primary-band layout is also supported.
Legacy binary tables, logarithmic/nonlinear grids and other multidimensional
FITS images are not accepted by this recipe. ASCII input has exactly three columns:
wavelength, flux density, standard deviation. Speckex ASCII unit comments are
recognized; unlabelled files default to Angstrom and erg s^-1 cm^-2 Angstrom^-1.
Supply `--wavelength-unit nm` and/or `--flux-unit 'erg / (s cm2 nm)'` when those
unlabelled ASCII defaults are inappropriate. Explicit units cannot contradict
recognized file declarations. ASCII coordinate rounding is fitted to a linear
grid only within the tolerance of nine-significant-digit precision.

Output wavelengths are in Angstrom. ASCII has nine significant digits; FITS
retains full precision. Default output has a primary flux image and `ERROR`
standard deviations; use `splot "spectrum_bin55.fits[0]"`. Full-grid output uses
primary bands; use `splot "spectrum_bin55.fits" line=1 band=1`. Provenance includes the
input filename/hash, bin size, mean estimator, omitted pixels and independent
noise assumption. Available FITS extraction metadata, including the spatial
aperture, is retained.

## Measure profiles

```bash
python -m speckex spectrum.fits --hdu SCI \
    --range 100:400 --range 700:1000 \
    --fit-range 20:60 --guess 40 --output source_profile
```

The numbers above are illustrative pixel selections; choose intervals within
your image. Repeat `--range` to compare regions independently. Each spatial row
contains the arithmetic mean of its valid selected dispersion pixels. Without
`--range`, the tool uses the entire dispersion axis; inspect whether curvature or
contamination makes this unsuitable. `--no-fit` plots only measured profiles.

All coordinates are **zero-based pixels** and intervals are **START:STOP with
STOP excluded**, as in Python slicing. They are not wavelengths or FITS one-based
coordinates. Science HDU defaults to `0`; use an index or extension name when the
primary HDU is empty. Dispersion defaults to NumPy axis `1`, the horizontal FITS
image axis. Use `--dispersion-axis 0` for vertical dispersion. The image is then
transposed for plotting, so dispersion is always horizontal and spatial position
is always vertical. Coordinates along each axis keep their original pixel values.

To choose intervals graphically:

```bash
python -m speckex spectrum.fits --hdu SCI --interactive \
    --fit-range 20:60 --guess 40 --output source_profile
```

Drag horizontally on the image to add a dispersion interval. Each interval gets
a separate colored profile and fit. Press `R` to clear selections. Close the
window to save the final selection; closing without any selections writes no
products. The Matplotlib toolbar provides zoom and pan. For a headless session,
use explicit `--range` values and `MPLBACKEND=Agg`.

## Masks, variances, and output

Optional `--mask-hdu DQ` excludes **every nonzero value** in that extension.
Optional `--variance-hdu VAR` must contain variances in science units squared,
not standard deviations or inverse variances. Arrays must match the science
shape. Non-finite science pixels and non-positive or non-finite variances are
excluded. Rows with no valid columns remain NaN rather than becoming zero.
No mask or variance extensions are inferred automatically.

With variance supplied, the profile mean error is `sqrt(sum(variance))/N_valid`
and the Gaussian fit is weighted by that error. This assumes independent pixels;
resampled or otherwise correlated pixels need additional uncertainty treatment.
Without variance, fits are unweighted and formal parameter errors are scaled
from fit residuals. Neither mode includes systematic background-model errors.

The profile and supplied errors are scaled together internally for numerical
stability. Amplitudes and background terms are restored to the input flux scale,
with the covariance transformed consistently. Positions, widths, and centroid
errors remain in pixels.

Three products are written using the output prefix:

- `.png`: image, selected regions, profiles, and any successful fits.
- `.ecsv`: profile samples, valid-column counts, and mean errors (NaN when unavailable).
- `.json`: settings, version, per-region fit status, centroid, Gaussian sigma,
  FWHM, and formal centroid error in pixels.

Existing products are protected unless `--overwrite` is supplied. Fit failures
are recorded without discarding the measured profiles. Fits close to parameter
bounds are labeled `boundary` and need inspection.

## Faint-source interpretation and limitations

Selecting dispersion regions and fitting a spatial profile are complementary.
Region selection determines which pixels contribute signal and contamination;
fitting estimates spatial position and width from the resulting profile. Use
separate regions to check whether the position changes along the spectrum.
Collapsing a curved trace without aligning it can broaden the measured profile.

Start with suitably calibrated data and preferentially a sky-subtracted image.
Select relatively clean continuum regions or an emission-line region appropriate
to the source. Wider intervals can improve a weak continuum measurement when
the trace remains aligned, but can dilute an isolated emission line. Compare
independent regions and avoid strong sky residuals, detector defects, and other
sources. The profile command does not identify cosmic rays automatically or
sigma-clip the data; supply a bad-pixel mask where available.

For an expected faint source near a brighter object, constrain `--fit-range` to
the relevant spatial neighborhood; `--guess` is only an initial position, not a
prior or a constraint fixing the fitted centroid. This model assumes one positive,
approximately Gaussian source over a locally linear background. It cannot model
multiple blended sources or extended morphology. A fitted Gaussian can also fit
noise: `candidate` means optimization succeeded, **not a confirmed detection**.
The tool deliberately does not report calibrated detection significance.

[Specreduce FitTrace](https://specreduce.readthedocs.io/en/stable/api/specreduce.tracing.FitTrace.html)
provides dispersion binning and fits positions as a function of dispersion;
its `disp_bounds` controls one contiguous interval. Its Gaussian peak method,
initial position, and spatial window are useful controls for faint-source tracing.
[Specreduce extraction](https://specreduce.readthedocs.io/en/stable/extraction.html)
then uses the trace for boxcar or optimal extraction. Horne optimal extraction
requires a variance array. Both boxcar and Gaussian Horne extraction are available
in the reviewed workflow above. The linked documentation can describe newer
Specreduce releases; this tool's adapter is verified against installed version 1.5.1.

## Search beside a bright star

```bash
python -m speckex.search spectrum_vis.fits spectrum_nir.fits \
    --hdu FLUX --error-hdu ERRS --offset-arcsec 3 \
    --search-radius-arcsec 1 --bin-pixels 50 --output-dir search_results
```

The filenames and separation above are illustrative. Horizontal dispersion is
the default. The error extension must contain **standard deviations**, not
variances or inverse variances. Verify its statistical interpretation with the
data producer; the extension name and matching flux units alone do not prove it.
This workflow requires a dominant reference star and approximately sky-subtracted,
rectified 2D images. It checks science/error shapes and matching `BUNIT` and does
not infer quality masks.

Both science and error HDUs must declare a nonempty, identical `BUNIT`;
missing declarations are rejected. Confirm units with the data producer rather
than inferring them from pixel values. An optional adjacent-background noise
diagnostic is saved as JSON `null` when no valid sample is available; this does
not prevent analysis of an otherwise usable source.

The science WCS is evaluated with Astropy and its output explicitly converted to
Å and arcsec. For diagonal linear WCS, zero-based pixel `p` corresponds to
`CRVAL + (p + 1 - CRPIX) * CDELT`, in `CUNIT`. Missing/incompatible units and
coupled axes are rejected. Wavelength units and flux-density units are separate:
a flux density per Å remains per Å when input wavelength coordinates are
converted from another declared unit. No air/vacuum or barycentric correction
is inferred or applied.

The continuum search performs these operations:

1. Excludes non-finite data and non-positive errors. Clips spectral deviations
   beyond `--clip-sigma` (default 6) times the supplied error relative to a
   51-column running median. **Clipping can reject real narrow lines**, so this
   workflow is unsuitable as a general emission-line search.
2. Keeps columns below `--noise-percentile` (default 80) of median spatial noise,
   with common inverse-variance column weights capped at their 95th percentile.
   Noise selection uses no target flux.
3. Forms profiles in consecutive `--bin-pixels` wavelength-pixel groups,
   including a final shorter group when present. Fits a Moffat star plus linear
   background outside the protected search interval (plus a 0.35-arcsec margin).
   Discards bins with fewer than ten usable columns or unsuccessful/weak stellar
   fits, then aligns retained profiles to their fitted star centers. At least
   eight accepted bins are required.
4. Scans signed Gaussian amplitudes with local linear backgrounds on a
   0.025-arcsec position grid and sigma values 0.22, 0.30, 0.40, and 0.55 arcsec.
   Uses unsubtracted profiles by default; `--substars` enables removal of one
   stellar component and comparison against unsubtracted profiles. It repeats
   with quadratic backgrounds and
   compares alternating bins and wavelength quartiles.
5. Bootstraps whole wavelength bins (`--bootstrap`, default 200; `--seed` controls
   reproducibility). Compare larger wavelength bins to assess the assumed
   correlation scale. The position bootstrap fixes the selected template width
   and searches within 0.5 arcsec of the optimum on a 0.05-arcsec grid, bounded
   by the original search interval. Its percentiles are conditional diagnostics,
   **not complete astrometric confidence intervals**.

Each input produces `_search.png`, `_search.json`, and `_profiles.ecsv` products.
JSON records settings, accepted/rejected bins, stellar traces, candidate offsets,
background-model comparisons, noise diagnostics, and limitations. ECSV contains
all aligned per-bin raw/subtracted profiles and formal errors. Amplitudes and
errors use an internal flux scale recorded in both files; multiply by that scale
to recover the input `BUNIT`. PNG smoothing affects display only, and rejected
wavelength bins remain blank. The representative candidate row uses the median
stellar position; use per-bin traces to follow wavelength-dependent positions.

Formal scores omit interpolation/resampling covariance, systematic residuals,
and stellar-fit uncertainty. The bootstrap captures bin-to-bin variation but
cannot rule out persistent spatial artifacts; it is neither a null simulation
nor a trials correction. A search returns its best candidate even for noise-only
data. Inspect independent bands and model sensitivity before interpreting it as
a source. Positive offsets mean increasing slit coordinate; sky north requires
additional orientation information. Template FWHM is selected from a grid and
is not a deconvolved source-size measurement.

## Verification

Run synthetic automated tests with:

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -v
```

Tests cover known source positions, a faint synthetic source, separate dispersion
regions, masked means, variance propagation, FITS orientation and extensions,
CLI output products, overwrite protection, physical unit conversions, and faint
neighbor recovery beside a drifting bright trace, boxcar/background covariance,
optional stellar removal, emission-feature preservation, and review confirmation.
Method tests also check native-grid Horne weights on curved traces, identical
sky weighting, nuisance covariance, finite-aperture flux normalization, strict
variance validation, and uncertainties against independent noisy realizations.
Background-sensitivity tests cover constant, linear and curved sky, nearby-source
contamination, spectral sky peaks, missing/masked pixels, fixed Horne weights,
alternative membership, and preservation of confirmed science outputs.
Horne QA tests compare the native estimator with independent finite-aperture sums,
validate source/nuisance covariance and degrees of freedom, inject correct and
mismatched profiles, and count failed as well as successful faint-source fits.
Synthetic tests do not establish
performance on a particular instrument or detection significance on real data.
