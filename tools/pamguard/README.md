# PAMGuard patch: fractional-octave whistle bins for delphinID

`delphinid_octave_bins.patch` lets PAMGuard's delphinID whistle transform use fractional-octave bands
(e.g. third-octave, 10 Hz–4 kHz) instead of its fixed 100 Hz bins, matching
`AcousticParams(binning="octave", bands_per_octave=3)` in speciesclassifiers.

- Opt-in: models are unchanged unless their `delphinID.pdtf` `whistles2spectrum` step has
  `"binning": "octave"` (and optionally `"bandsperoctave"`). Existing models keep 100 Hz bins.
- Band edges: centres `1000 * 2^(n/b)` Hz, bands kept when their centre lies within the frequency
  limits widened by half a band (same rule as `transforms.octave_band_edges`).
- Counting rules (segment start, PAMGuard's seconds-vs-milliseconds segment length quirk) are unchanged.

Apply from the PAMGuard repo root: `git apply path/to/delphinid_octave_bins.patch`
(or `patch -p1 < ...`). Touches `Whsitle2Spectrum.java` and `DelphinIDTransform.java`.
Not yet compiled or tested inside PAMGuard.
