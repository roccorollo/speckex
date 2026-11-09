# Changelog

## 1.0.0 — 2026-10-21

- Make `--bin-pixels N`/`-b N` count adjacent native wavelength pixels in
  quick extraction, continuum search, and reviewed diagnostic displays.
  Remove the wavelength-width `--bin-nm` interface; quick binned filenames
  now use `_binN`.
- Use Å for wavelength ranges, diagnostic ECSV columns, JSON wavelength values,
  FITS coordinate arrays, and every plot axis. FITS and TXT science spectra
  continue to use Å. Input FITS and optional external ASCII input may declare
  other convertible units, which are converted when read.
- Rename affected ECSV columns and JSON keys to identify their Å units.
  This output-schema and CLI change is intentionally incompatible with 0.x.
- Record each search bin's exact pixel interval and apply the reviewed trace-gap
  limit in pixels, including for nonlinear wavelength grids.

## 0.13.0 — 2026-10-21

- Let quick extraction use an explicit `--bin-nm`/`-b` width to save an
  additional binned 1D ECSV, FITS, TXT, and PNG while retaining native products.
  Plot and open the binned version by default when requested.
- Average complete native wavelength-pixel groups, record requested and actual
  physical widths, invalidate groups containing flagged samples, and propagate
  real standard deviations under the independent-sample assumption. Keep
  errorless binned outputs free of fabricated uncertainties.
- Add `-b` as an alias for the existing `--bin-nm` search and reviewed-extraction
  setting; the default quick behavior is unchanged without an explicit width.

## 0.12.0 — 2026-10-21

- Save a dedicated, gap-aware 1D spectrum PNG with every quick extraction.
  Include native flux, display-only median overlay, and a 1-sigma band only
  when a propagated input error exists.
- Add `--plot-spectra`/`-p` for reviewed and explicit flux-only extraction,
  plus `--graphic`/`-g` to open the 1D figure with the default Linux or macOS
  image viewer. Quick extraction opens it by default; `--no-graphic` keeps
  saving the figure in headless or batch use.
- Keep viewer failure separate from extraction success and protect existing
  plot files along with science products.
- Consolidate reviewed spectrum-plot filenames and reuse already computed
  wavelength and flux values in quick exports. Output formats and extraction
  calculations are unchanged.

## 0.11.1 — 2026-10-21

- Let a top-level `-q` or `--quick` invoke `extract --quick`, with the flag
  before or after the FITS filename. Preserve the existing `extract -q` form
  and profile-command behavior.

## 0.11.0 — 2026-10-21

- Add `extract --quick`/`-q` for immediate boxcar extraction from an approximate
  row, angular or pixel offset, or exact source-row interval. Add `-gr`/`-s`,
  `-oa`, `-op`, and `-sr` location aliases; keep reviewed extraction behavior.
- Select a profile-based source aperture and nearby low-feature sky intervals
  automatically. Preserve selected rows, effective source/sky masks, quality
  flags, and selection limitations in the quick products.
- Propagate standard-deviation input and fitted-sky covariance when an error
  image exists. With no error image, use an unweighted sky fit and export flux
  without a fabricated uncertainty.
- Give quick and explicit flux-only extraction a consistent three-panel 2D
  image, effective-region, and spectrum figure styled after reviewed extraction.

## 0.10.0 — 2026-10-21

- Add `flux-only` extraction for rectified 2D FITS data without a pixel-error
  image. Use explicitly selected source and background rows, an unweighted
  clipped background fit, and quality flags; never assign a fabricated variance.
- Export native-grid ECSV, flux-only FITS with `ERRTYPE=NONE` and quality image,
  two-column ASCII, diagnostic plot, and JSON provenance. Preserve the existing
  error-bearing reviewed extraction and its science exports.
- Read a documented arcsec-per-pixel `PIXSCALE` when the spatial WCS is in
  pixels. Permit flux-only row extraction without any angular spatial scale
  when wavelength WCS remains valid.
- Add synthetic and real-file regressions for missing-error data, WCS fallback,
  sky-region selection, output metadata, and existing error-bearing behavior.

## 0.9.0 — 2026-10-21

- Add optional `--horne-qa` reports, ECSV tables and spatial/spectral plots after
  confirmation, with rigid centroid, FWHM and chromatic trace-kernel perturbations.
- Freeze data, variances, background/star predictions, nuisance covariance and
  physical aperture. Reevaluate native-grid Horne weights with consistent
  finite-aperture normalization; retain standard spectra and statistical errors.
- Retain the actual fixed-centroid fit, width/FWHM, formal width error, trace and
  fit limits. Prefer supported conditional width errors or configurable fractions;
  do not invent centroid/trace uncertainties or interpolate missing wavelengths.
- Add covariance-aware conditional 2D chi-square, explicit fitted-parameter
  accounting, residual structure/proximity advisories, and shared-covariance
  boxcar/Horne comparisons. No diagnostic automatically rejects extraction.
- Add independent analytical, injection/recovery and workflow regressions for
  correct/mismatched profiles, variable seeing, faint-source fit acceptance,
  contamination, missing data, output protection and unchanged science exports.
  Preserve existing background QA and binning; add no dependencies.

## 0.8.0 — 2026-10-21

- Add opt-in `--background-sensitivity` QA after extraction confirmation, with
  configurable inward/outward sky-window shifts and fractional width changes,
  alternative constant/linear models, and one-sided constant estimates.
- Freeze the confirmed trace, aperture and exact boxcar/Horne source weights.
  Refit only the local nuisance model, preserving stellar shapes and existing
  clipping/exclusion settings; never refit the Horne source profile during QA.
- Write separate per-method ECSV tables and PNGs with configuration outcomes,
  native-grid flux differences, alternative counts/ranges, population RMS
  `BG_SENS` and its ratio to the unchanged reference statistical error.
- Skip unsupported configurations, preserve missing wavelengths without
  interpolation, require two alternatives for sensitivity, and distinguish
  diagnostic scatter from a calibrated systematic uncertainty. Preserve
  default extraction behavior and existing science FITS/ASCII formatting.
- Add analytical and workflow regressions for background models, contamination,
  spectral sky peaks, missing/masked pixels, frozen profiles, confirmation,
  output protection and unchanged science products. No new dependencies.

## 0.7.0 — 2026-10-20

- Add optional `--gap-policy zero` to reviewed extraction and 1D binning.
  Retain the complete regular wavelength grid in one FITS/ASCII pair, marking
  missing samples with flux=0 and standard deviation=0. Positive error remains
  required for a valid measurement; no interpolation is introduced.
- Store full-grid FITS flux, sigma and quality as associated bands in one
  primary HDU without extensions, with IRAF spectral WCS and band identifiers.
  Keep ASCII at three columns and nine significant digits.
- Accept primary-band FITS and paired-zero ASCII gaps during explicitly
  selected full-grid binning. Any complete bin touching a missing pixel remains
  missing; fully observed means and independent-noise errors are unchanged.
- Preserve existing split-file defaults, scientific estimators, native-grid
  extraction, diagnostic ECSV, units and overwrite protection.

## 0.6.0 — 2026-10-20

- Add `speckex bin` for linear 1D science FITS and three-column ASCII spectra.
  Write both IRAF-compatible FITS and nine-significant-digit ASCII, using
  Angstrom wavelengths and `_binN` in default output filenames.
- Average complete adjacent-pixel groups without changing the flux-density
  convention; propagate independent standard deviations in quadrature divided
  by the pixel count. Explicitly record the omitted incomplete trailing group
  and the exclusion of correlated noise and upstream systematics.
- Require finite flux, valid standard deviations and contiguous linear sampling;
  protect existing outputs and retain input hashes and extraction provenance.
- Allow derived exports to supply accurate processing notes without changing
  the existing extraction export defaults.

## 0.5.0 — 2026-10-20

- Add native confirmed FITS and plain ASCII science exports for boxcar, Horne,
  and both modes. Use Angstrom wavelengths and calibrated cgs flux density per
  Angstrom; preserve uncalibrated count units without inventing a calibration.
- Write full-precision flux in a 1D primary FITS image with IRAF equispec linear
  WCS, and propagated standard deviations in an ERROR image extension.
- Use nine significant digits in the three-column ASCII science files.
- Trim invalid edges and split internal gaps into independently calibrated
  contiguous parts. Never fill bad samples or collapse gaps onto a false
  wavelength grid. Nonlinear grids are rejected without resampling.
- Retain existing ECSV spectra, masks, diagnostic coordinates and search options
  for compatibility. Protect the new science products against unintended writes;
  record exported paths in confirmed review JSON.

## 0.4.1 — 2026-10-20

- Normalize standalone profile fits and supplied errors before optimization;
  restore dimensional parameters and covariance so physical flux units do not
  alter fitted positions, widths, or formal centroid uncertainties.
- Ignore zero-weight interpolation neighbors when estimating the Horne profile,
  including floating-point roundoff at integer rows. Exclude invalid contributing
  columns from profile estimation and still require eight valid columns.
- Save an unavailable adjacent-background noise diagnostic as JSON null and
  avoid division by invalid errors, preserving otherwise usable analysis.
- Require nonempty matching BUNIT declarations in science and error HDUs for
  continuum search and reviewed extraction. Previously, two missing declarations
  passed validation; such inputs now receive an explicit error.
- Add focused regressions for all four independent QA findings. The extraction
  estimators, variance model, and documented scientific limitations are unchanged.

## 0.4.0 — 2026-10-20

- Add `--method boxcar|horne|both`; retain boxcar as default and use
  Specreduce 1.5.1 as the tested extraction backend.
- Preserve the trace, fractional aperture, missing-pixel flags, local sky/star
  fits, and covariance propagation. Verify boxcar against previous results.
- Fit a target Gaussian for Horne extraction on the original detector grid;
  avoid upstream science-only trace alignment and additional background kernels.
- Reject missing/unusable used variances without uniform-weight fallback;
  flag weak/unconstrained Gaussian profile fits.
- Normalize Horne to the same finite-aperture convention as boxcar and propagate
  input-pixel noise plus shared fitted sky/star covariance.
- Save separately labeled spectra, 1-sigma errors, method-weighted sky/star
  components, S/N, and extraction weights in both mode.
- Add profile/flux/S/N comparison previews and editable review-mode selection;
  require confirmation without automatically choosing a method by S/N.

## 0.3.0 — 2026-10-20

- Distribute source code, tests, README, and MIT license in a versioned release archive.
- Add `speckex extract` with an initial source guess, diagnostic/background
  review views, editable settings, revision history, and explicit confirmation
  before writing a boxcar spectrum. Closing or cancelling a review writes no new spectrum.
- Add physical source apertures, configurable local sky windows, constant or
  linear backgrounds, sky outlier rejection, and per-wavelength effective-region plots.
- Propagate source-pixel variance and the covariance of shared fitted background
  and optional stellar-amplitude coefficients. Export spectra and effective masks.
- Make nearby-star subtraction opt-in through `--substars` / `-ss` in both
  search and extraction. Subtract one stellar component while retaining the sky;
  fit its amplitude at each wavelength during extraction.
- Add repeatable wavelength intervals, pixel-row guesses, preview-only runs,
  terminal review, and explicit batch confirmation.
- Keep continuum-search clipping separate from extraction, preserving original
  narrow spectral features. Retain previous recorded analysis products unchanged.

## 0.2.0 — 2026-10-20

- Add a continuum-candidate search beside a bright spectral trace, with
  unit-checked FITS coordinates in nm and arcsec.
- Support a standard-deviation extension, continuum outlier clipping, and
  target-independent dispersion-column noise selection.
- Fit Moffat stellar profiles in wavelength bins and align profiles to the
  wavelength-dependent stellar center while protecting the target region.
- Compare Gaussian-template searches with and without stellar subtraction,
  linear/quadratic backgrounds, wavelength splits, and spectral-bin bootstrap.
- Export physical-coordinate PNG diagnostics, aligned profiles, and detailed
  provenance. Scores remain uncalibrated candidate diagnostics.

## 0.1.0 — 2026-10-20

- Initial spatial-profile inspection tool for two-dimensional FITS spectra.
- Select multiple dispersion intervals through the CLI or graphical dragging.
- Compare per-region spatial means and optional Gaussian plus linear-background fits.
- Support explicit science, variance, and mask extensions and either dispersion axis.
- Export diagnostic figures, profile samples, fit results, and processing settings.
- Report fits as candidates; spectrum tracing and extraction remain future work.
