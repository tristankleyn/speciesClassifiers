"""Click windows: inter-click intervals (ICIs) and spectra of the clicks in each window.

Clicks are detections from the PAMGuard Click Detector (``io.read_clicks``) or any table with a ``start``
column (Unix s, UTC; ``duration`` in s optional). A window's ICIs are the gaps between consecutive clicks in
it; gaps longer than ``max_ici_s`` are pauses between click trains and are left out. All clicks in a window
are pooled (no separation of click trains or animals).

Spectra (``click_spectra``): each click's waveform is cut to ``n_samples`` around its largest absolute
sample, tapered (Hann), zero-padded to ``nfft`` and Fourier transformed; within ``fmin_hz``-``fmax_hz`` the
peak frequency, spectral centroid, -3 dB and -10 dB bandwidths (contiguous band around the peak above half /
a tenth of the peak power) and the share of power in ``profile_bands`` equal bands are measured. Windows take
the median (and spread) over their clicks, and the mean band profile. Measured while reading the binaries,
so the waveforms aren't kept.
"""

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .contours import day_of

# key: (group, label, on by default)
CLICK_FEATURES = {
    "ici_mean_s": ("Inter-click interval", "mean", True),
    "ici_median_s": ("Inter-click interval", "median", True),
    "ici_std_s": ("Inter-click interval", "standard deviation", True),
    "ici_p05_s": ("Inter-click interval", "5th percentile", True),
    "ici_p95_s": ("Inter-click interval", "95th percentile", True),
    "click_peak_hz_median": ("Click spectrum (median over clicks)", "peak frequency", True),
    "click_centroid_hz_median": ("Click spectrum (median over clicks)", "centroid frequency", True),
    "click_bw3_hz_median": ("Click spectrum (median over clicks)", "-3 dB bandwidth", True),
    "click_bw10_hz_median": ("Click spectrum (median over clicks)", "-10 dB bandwidth", True),
    "click_peak_hz_iqr": ("Click spectrum (spread over clicks)", "peak frequency, interquartile range", True),
    "click_centroid_hz_iqr": ("Click spectrum (spread over clicks)", "centroid frequency, interquartile range",
                              True),
    "click_profile": ("Click spectrum profile", "share of power per frequency band (mean over clicks)", True),
    "n_clicks": ("Window", "number of clicks (depends on distance)", False),
}
CLICK_MEASURES = ("peak_hz", "centroid_hz", "bw3_hz", "bw10_hz")     # per click


@dataclass
class ClickSpectrumParams:
    """Click spectrum settings.

    n_samples      samples of each click used, centred on its largest absolute sample
    nfft           FFT length (zero-padded); frequency resolution = sample rate / nfft
    fmin_hz        measurements use this band only (fmax_hz None = up to half the sample rate)
    fmax_hz
    profile_bands  equal-width bands of the spectrum profile (0 = no profile)
    """

    n_samples: int = 128
    nfft: int = 256
    fmin_hz: float = 5000.0
    fmax_hz: float = None
    profile_bands: int = 16

    def to_dict(self):
        return asdict(self)


def profile_edges(sr, sp: ClickSpectrumParams):
    fmax = min(sp.fmax_hz or sr / 2, sr / 2)
    return np.linspace(sp.fmin_hz, fmax, sp.profile_bands + 1)


def click_spectra(waves, sr, sp: ClickSpectrumParams = None):
    """Per-click spectral measurements (DataFrame, one row per waveform): ``peak_hz``, ``centroid_hz``,
    ``bw3_hz``, ``bw10_hz`` and ``csp_<lo>_<hi>`` band shares. Vectorised over clicks."""
    sp = sp or ClickSpectrumParams()
    n = len(waves)
    edges = profile_edges(sr, sp) if sp.profile_bands else None
    pcols = [f"csp_{int(edges[k])}_{int(edges[k + 1])}" for k in range(sp.profile_bands)] if sp.profile_bands else []
    cols = list(CLICK_MEASURES) + pcols
    if n == 0:
        return pd.DataFrame(columns=cols)
    L = sp.n_samples
    seg = np.zeros((n, L))
    for i, w in enumerate(waves):
        w = np.asarray(w if w is not None else [], float).ravel()
        if not len(w):
            continue
        c = int(np.argmax(np.abs(w)))
        a = max(0, min(c - L // 2, len(w) - L))
        x = w[a:a + L]
        x = x - x.mean()
        seg[i, :len(x)] = x * np.hanning(len(x)) if len(x) > 1 else x
    P = np.abs(np.fft.rfft(seg, n=max(sp.nfft, L), axis=1)) ** 2
    f = np.fft.rfftfreq(max(sp.nfft, L), 1.0 / sr)
    band = (f >= sp.fmin_hz) & (f <= min(sp.fmax_hz or sr / 2, sr / 2))
    P, f = P[:, band], f[band]
    tot = P.sum(axis=1)
    ok = tot > 0
    pk = np.argmax(P, axis=1)
    pmax = P[np.arange(n), pk]
    idx = np.arange(P.shape[1])
    df = f[1] - f[0] if len(f) > 1 else np.nan
    out = {"peak_hz": f[pk], "centroid_hz": (P * f).sum(axis=1) / np.where(ok, tot, 1)}
    for name, frac in (("bw3_hz", 0.5), ("bw10_hz", 0.1)):
        below = P < (pmax * frac)[:, None]
        left = np.where(below & (idx < pk[:, None]), idx, -1).max(axis=1)
        right = np.where(below & (idx > pk[:, None]), idx, P.shape[1]).min(axis=1)
        out[name] = (right - left - 1) * df
    res = pd.DataFrame(out)
    if pcols:
        b = np.clip(np.digitize(f, edges) - 1, 0, sp.profile_bands - 1)
        B = np.zeros((n, sp.profile_bands))
        for k in range(sp.profile_bands):
            B[:, k] = P[:, b == k].sum(axis=1)
        B = B / np.where(ok, tot, 1)[:, None]
        res = pd.concat([res, pd.DataFrame(B, columns=pcols)], axis=1)
    res.loc[~ok, :] = np.nan
    return res[cols]


@dataclass
class ClickWindowParams:
    """Click window settings.

    window_s, overlap  window length (s) and overlap (0.5 = 50 %)
    min_clicks         windows with fewer clicks are dropped
    max_ici_s          gaps between clicks longer than this (s) are not intervals (pauses between trains)
    min_ici_s          gaps shorter than this (s) are left out (e.g. echoes or multipath of one click)
    """

    window_s: float = 10.0
    overlap: float = 0.5
    min_clicks: int = 5
    max_ici_s: float = 1.0
    min_ici_s: float = 0.0

    def to_dict(self):
        return asdict(self)


def prepare_clicks(clicks):
    """Click table -> sorted ``start``, ``end``, ``duration_s``, ``day`` (other columns dropped)."""
    t = pd.to_numeric(pd.Series(np.asarray(clicks["start"], float)), errors="coerce")
    dur = (pd.to_numeric(pd.Series(np.asarray(clicks["duration"], float)), errors="coerce").fillna(0)
           if "duration" in clicks else pd.Series(np.zeros(len(t))))
    c = pd.DataFrame({"start": t.to_numpy(), "duration_s": dur.to_numpy()})
    meas = [m for m in clicks.columns if m in CLICK_MEASURES or str(m).startswith("csp_")]
    for m in meas:
        c[m] = pd.to_numeric(pd.Series(np.asarray(clicks[m])), errors="coerce").to_numpy()
    c = c.dropna(subset=["start"])
    c["end"] = c["start"] + c["duration_s"]
    c = c.sort_values("start", kind="mergesort").reset_index(drop=True)
    c["day"] = day_of(c["start"]) if len(c) else []
    return c[["start", "end", "duration_s", "day", *meas]]


def click_windows(clicks, params: ClickWindowParams = None):
    """Sliding windows over prepared clicks (``prepare_clicks``) with their ICI features. Windows are aligned
    to the minute of each day's first click; only windows with at least ``min_clicks`` clicks and one
    interval are kept."""
    params = params or ClickWindowParams()
    ici = ["ici_mean_s", "ici_median_s", "ici_std_s", "ici_p05_s", "ici_p95_s"]
    meas = [m for m in CLICK_MEASURES if m in clicks]
    prof = [c for c in clicks.columns if str(c).startswith("csp_")]
    spec = ([f"click_{m}_median" for m in meas] +
            [f"click_{m}_iqr" for m in ("peak_hz", "centroid_hz") if m in meas] + prof)
    cols = ["w_start", "w_end", "day", "n_clicks", "n_intervals", *ici, *spec]
    if not len(clicks):
        return pd.DataFrame(columns=cols)
    M = clicks[meas].to_numpy(float) if meas else None
    Pm = clicks[prof].to_numpy(float) if prof else None
    L = params.window_s
    step = L * (1 - params.overlap)
    rows = []
    for day, g in clicks.groupby("day", sort=True):
        gi = g.index.to_numpy()
        t = g["start"].to_numpy(float)
        t0 = np.floor(t.min() / 60) * 60
        k = np.unique(np.concatenate([np.floor((t - t0) / step).astype(int) - j
                                      for j in range(int(np.ceil(L / step)))]))
        ws = t0 + k[k >= 0] * step
        lo = np.searchsorted(t, ws, "left")
        hi = np.searchsorted(t, ws + L, "left")
        for s, a, b in zip(ws, lo, hi):
            n = b - a
            if n < max(params.min_clicks, 2):
                continue
            d = np.diff(t[a:b])
            d = d[(d <= params.max_ici_s) & (d > params.min_ici_s)]
            if not len(d):
                continue
            row = [s, s + L, day, n, len(d), d.mean(), np.median(d), d.std(ddof=1) if len(d) > 1 else 0.0,
                   np.quantile(d, 0.05), np.quantile(d, 0.95)]
            if spec:
                sel = gi[a:b]
                if meas:
                    m = M[sel]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        q = np.nanquantile(m, [0.25, 0.5, 0.75], axis=0) if np.isfinite(m).any() else \
                            np.full((3, len(meas)), np.nan)
                    row += list(q[1])
                    row += [q[2][meas.index(x)] - q[0][meas.index(x)] for x in ("peak_hz", "centroid_hz") if x in meas]
                if prof:
                    pm = Pm[sel]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        row += list(np.nanmean(pm, axis=0))
            rows.append(tuple(row))
    return pd.DataFrame(rows, columns=cols).sort_values("w_start", kind="mergesort").reset_index(drop=True)


def event_periods(calls, columns):
    """Call annotations -> one labelled period per event (first to last call), for labelling click windows
    by the event they fall in."""
    g = calls.groupby("event_id", sort=False)
    p = g.agg(start=("start", "min"), end=("end", "max"))
    for c in columns:
        if c != "event_id":
            p[c] = g[c].agg(lambda s: s.mode().iloc[0] if len(s.dropna()) else "none")
    return p.reset_index()
