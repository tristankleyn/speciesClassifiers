"""delphinID feature transforms, ported from PAMGuard.

These reproduce PAMGuard's delphinID transforms so that a model trained on features made here
gives the same predictions inside PAMGuard. Sources:

- ``clicks2spectrum``: ``rawDeepLearningClassifier/dlClassification/delphinID/Clicks2Spectrum.java``
  and ``PamguardMVC/RawDataTransforms.getPowerSpectrum``
- ``whistles2spectrum``: ``Whsitle2Spectrum.java`` and ``Whistles2Image.whistContours2Points``
- ``spectrum_*``: ``org.jamdev.jpamutils.spectrum.Spectrum`` (jpamutils 1.0.0)
- segmentation: ``DelphinIDUtils.segmentDetectionData`` / ``SegmenterProcess``

Each transform is named as in PAMGuard's ``delphinID.pdtf`` file, so a transform list can be
written straight into the PAMGuard model file (see ``apply_transforms``).
"""

import numpy as np

WHISTLE_BIN_HZ = 100.0  # fixed in PAMGuard's whistle2AverageArray


# ---------------------------------------------------------------- clicks ---------------
def click_power_spectrum(wave, fft_len, hann=False):
    """Power spectrum of one click: |FFT|^2 of the waveform, zero-padded or truncated (from the
    start) to ``fft_len``, first ``fft_len / 2`` bins (DC up to but excluding Nyquist)."""
    wave = np.asarray(wave, dtype=float)
    if hann:
        wave = wave * np.hanning(len(wave))
    padded = np.zeros(fft_len)
    n = min(fft_len, len(wave))
    padded[:n] = wave[:n]
    spec = np.fft.rfft(padded)[: fft_len // 2]
    return spec.real ** 2 + spec.imag ** 2


def clicks2spectrum(waves, fft_len=512, spectrum_db=True, hann=False):
    """Average spectrum of a group of clicks (PAMGuard ``clicks2spectrum``).

    Each click's power spectrum is converted to 20*log10 (as PAMGuard does), averaged over
    clicks, and the minimum is subtracted.
    """
    if len(waves) == 0:
        raise ValueError("No clicks in group")
    total = np.zeros(fft_len // 2)
    for w in waves:
        ps = click_power_spectrum(w, fft_len, hann)
        if spectrum_db:
            with np.errstate(divide="ignore"):
                ps = 20 * np.log10(ps)
        total += ps
    mean = total / len(waves)
    return mean - mean.min()


# ---------------------------------------------------------------- whistles -------------
def whistle_points(contours, seg_start, min_frag_ms=0.0):
    """Time-frequency points of the whistle contours in a segment.

    ``contours``: list of dicts with ``start`` (s, absolute), ``times`` (s, per slice),
    ``freqs`` (Hz, per slice) and ``duration`` (s). Contours shorter than ``min_frag_ms`` are
    dropped. Returns an (n, 2) array of [seconds from segment start, Hz].
    """
    pts = []
    for c in contours:
        if c["duration"] * 1000.0 < min_frag_ms:
            continue
        t = np.asarray(c["times"], float)
        rel = (c["start"] - seg_start) + (t - t[0])
        pts.append(np.column_stack([rel, np.asarray(c["freqs"], float)]))
    return np.vstack(pts) if pts else np.empty((0, 2))


def whistles2spectrum(contours, seg_start, seg_len, freq_range=(2000.0, 20000.0), min_frag_ms=200.0):
    """Normalised histogram of whistle contour frequencies in 100 Hz bins
    (PAMGuard ``whistles2spectrum``).

    Uses every contour in the segment (starting or ending in it), keeping points from the
    segment start onwards.

    Note: PAMGuard compares point times (s) against the segment length in ms, so contour points
    after the segment end are kept. This is reproduced so features match PAMGuard.
    """
    pts = whistle_points(contours, seg_start, min_frag_ms)
    fmin, fmax = freq_range
    nbins = int((fmax - fmin) / WHISTLE_BIN_HZ)
    seg_len_ms = seg_len * 1000.0  # PAMGuard quirk, see docstring
    keep = (pts[:, 0] >= 0) & (pts[:, 0] < seg_len_ms)
    f = pts[keep, 1]
    edges = fmin + WHISTLE_BIN_HZ * np.arange(nbins + 1)
    counts = np.array([np.sum((f >= edges[i]) & (f < edges[i + 1])) for i in range(nbins)], float)
    total = counts.sum()
    return counts / total if total else counts


def whistle_density(contours, seg_start, seg_len):
    """PAMGuard's whistle detection density for a segment: number of contour points divided by
    the number of time slices in the segment (slice length from the first contour)."""
    if not contours:
        return 0.0
    pts = [whistle_points([c], seg_start) for c in contours]
    t0 = pts[0][:, 0]
    if len(t0) < 2:
        return np.nan
    step_ms = 1000.0 * np.mean(np.diff(t0))
    n_slices = (seg_len * 1000.0) / step_ms
    return sum(len(p) for p in pts) / n_slices


# ---------------------------------------------------------------- spectrum ops ---------
def spectrum_smooth(x, window):
    """Trailing moving average over the last ``window`` values (fewer at the start)."""
    x = np.asarray(x, float)
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(1, len(x) + 1)
    lo = np.maximum(0, i - window)
    return (c[i] - c[lo]) / (i - lo)


def spectrum_trim_freq(x, fmin, fmax, sample_rate):
    """Keep bins between ``fmin`` and ``fmax`` (Hz), for a spectrum spanning 0 to sample_rate/2."""
    x = np.asarray(x, float)
    span = sample_rate / 2.0
    i0 = int(fmin / span * len(x))
    i1 = int(fmax / span * len(x))
    if i0 < 0 or i1 >= len(x) or i1 == i0:
        raise ValueError(f"Cannot trim spectrum of {len(x)} bins to {fmin}-{fmax} Hz at {sample_rate} Hz")
    return x[i0:i1]


def spectrum_downsample_mean(x, factor):
    """Average consecutive blocks of ``factor`` bins (last block may be shorter)."""
    x = np.asarray(x, float)
    return np.array([x[i:i + factor].mean() for i in range(0, len(x), factor)])


def spectrum_normalise_sum(x):
    """Divide by the sum so the spectrum sums to 1."""
    x = np.asarray(x, float)
    return x / x.sum()


def apply_transforms(spectrum, transforms, sample_rate=None):
    """Apply the ``spectrum_*`` steps of a PAMGuard transform list to a spectrum.

    ``transforms`` is the list in a ``delphinID.pdtf`` file (dicts with ``name`` and
    ``params``). The first step (``clicks2spectrum`` / ``whistles2spectrum``) is skipped here;
    it makes the spectrum. ``sample_rate`` is needed for ``spectrum_trim_freq``.
    """
    x = np.asarray(spectrum, float)
    for t in transforms:
        name, p = t["name"], t.get("params", {})
        if name in ("clicks2spectrum", "whistles2spectrum"):
            continue
        if name == "spectrum_smooth":
            x = spectrum_smooth(x, int(p["window"]))
        elif name == "spectrum_trim_freq":
            x = spectrum_trim_freq(x, float(p["fmin"]), float(p["fmax"]), sample_rate)
        elif name == "spectrum_downsample_mean":
            x = spectrum_downsample_mean(x, int(p["factor"]))
        elif name == "spectrum_normalise_sum":
            x = spectrum_normalise_sum(x)
        else:
            raise ValueError(f"Unsupported transform: {name}")
    return x


# ---------------------------------------------------------------- segmentation ---------
def segment_starts(det_start, det_end, frame_len, hop, data_start=None):
    """Frame (segment) start times and the detections in each, as PAMGuard groups them.

    A detection belongs to a frame if it starts or ends inside it. Times in seconds.
    Returns a list of (frame_start, indices) for every frame from ``data_start`` (default: the
    first detection) until the last detection ends, including empty frames.
    """
    s = np.asarray(det_start, float)
    e = np.asarray(det_end, float)
    t = s.min() if data_start is None else data_start
    out = []
    while t < e.max():
        inside = ((s >= t) & (s < t + frame_len)) | ((e >= t) & (e < t + frame_len))
        out.append((t, np.flatnonzero(inside)))
        t += hop
    return out
