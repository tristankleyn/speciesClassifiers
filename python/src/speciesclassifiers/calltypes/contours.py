"""Whistle contours: filters and shape measurements.

Input is the whistle table from ``io.read_whistles`` (one row per contour: ``start`` in Unix s,
``times`` in s and ``freqs`` in Hz per time slice, ``slice_s``, ``snr_db``, ...).
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class ContourParams:
    """Contour filters.

    fmin_hz, fmax_hz  contour points outside this band are clipped off
    min_contour_s     contours shorter than this (after clipping) are dropped
    min_snr_db        contours with a lower detector SNR (20 log10 signal/noise) are dropped; None = off
    snr_top_frac      alternatively keep only this top share (0-1) of contours by SNR; None = off
    """

    fmin_hz: float = 800.0
    fmax_hz: float = 16000.0
    min_contour_s: float = 0.4
    min_snr_db: float = None
    snr_top_frac: float = None

    def to_dict(self):
        return asdict(self)


def contour_params(t, f, slice_s):
    """Shape measurements of one contour. ``t`` starts at 0 (s), ``f`` in Hz."""
    n = len(f)
    dur = (t[-1] - t[0]) + slice_s
    d = np.diff(f) / np.diff(t) if n > 1 else np.array([0.0])
    sgn = np.sign(np.diff(f)) if n > 1 else np.array([0.0])
    nz = sgn[sgn != 0]
    tt = t - t[0]
    p = {
        "duration_s": dur, "start_freq_hz": f[0], "end_freq_hz": f[-1],
        "min_freq_hz": f.min(), "max_freq_hz": f.max(), "freq_range_hz": f.max() - f.min(),
        "mean_freq_hz": f.mean(), "std_freq_hz": f.std(),
        "freq_at_quarter_hz": f[int(round((n - 1) * 0.25))],
        "freq_at_half_hz": f[int(round((n - 1) * 0.5))],
        "freq_at_threequarter_hz": f[int(round((n - 1) * 0.75))],
        "time_of_min_freq_rel": tt[np.argmin(f)] / tt[-1] if tt[-1] > 0 else 0.0,
        "time_of_max_freq_rel": tt[np.argmax(f)] / tt[-1] if tt[-1] > 0 else 0.0,
        "overall_slope_hz_s": (f[-1] - f[0]) / dur,
        "linreg_slope_hz_s": np.polyfit(tt, f, 1)[0] if n > 2 else 0.0,
        "quad_curvature": np.polyfit(tt, f, 2)[0] if n > 3 else 0.0,
        "mean_abs_slope_hz_s": np.abs(d).mean(),
        "std_slope_hz_s": d.std(),
        "frac_up": (sgn > 0).mean(), "frac_down": (sgn < 0).mean(),
        "n_inflections": int((np.diff(nz) != 0).sum()) if len(nz) > 1 else 0,
    }
    p["inflections_per_s"] = p["n_inflections"] / dur
    return p


def day_of(seconds):
    """UTC day (YYYYMMDD) of Unix times."""
    return pd.to_datetime(np.asarray(seconds, float), unit="s", utc=True).strftime("%Y%m%d").to_numpy()


def prepare_contours(whistles: pd.DataFrame, params: ContourParams = None):
    """Clip contours to the band, apply duration/SNR filters, measure each contour.

    Returns ``(contours, points)``: one row per kept contour (measurements, ``start``/``end`` in
    Unix s, ``day``, ``uid``, ``snr_db``, ``amplitude_db``) sorted by start, and per contour a tuple
    ``(t from contour start, f, slice_s)`` of its points.
    """
    params = params or ContourParams()
    rows, pts = [], []
    for r in whistles.itertuples(index=False):
        slice_s = float(r.slice_s)
        t_all = np.asarray(r.times, float)
        t_all = np.round((t_all - t_all[0]) / slice_s) * slice_s   # whole slices, as PAMGuard counts them
        f_all = np.asarray(r.freqs, float)
        m = (f_all >= params.fmin_hz) & (f_all <= params.fmax_hz)
        if m.sum() < 3:
            continue
        t, f = t_all[m], f_all[m]
        if (t[-1] - t[0]) + slice_s < params.min_contour_s:
            continue
        snr = getattr(r, "snr_db", np.nan)
        if params.min_snr_db is not None and not (snr >= params.min_snr_db):
            continue
        p = contour_params(t, f, slice_s)
        start = float(r.start) + float(t[0])
        p.update({"uid": getattr(r, "uid", None), "start": start, "end": start + p["duration_s"],
                  "snr_db": snr, "amplitude_db": getattr(r, "amplitude_db", np.nan)})
        rows.append(p)
        pts.append((t - t[0], f, slice_s))
    c = pd.DataFrame(rows)
    if not len(c):
        return c, []
    c["day"] = day_of(c["start"])
    if params.snr_top_frac:
        thr = c["snr_db"].quantile(1 - params.snr_top_frac)
        keep = (c["snr_db"] >= thr).to_numpy()
        c, pts = c[keep], [p for p, k in zip(pts, keep) if k]
    order = np.argsort(c["start"].to_numpy(), kind="stable")
    return c.iloc[order].reset_index(drop=True), [pts[i] for i in order]
