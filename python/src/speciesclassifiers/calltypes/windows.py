"""Windows of contours, their features, and labels from annotations.

A window is a short stretch of time (sliding, or cut around each annotated call). Its features
summarise the contours in it: mean and spread of contour measurements, the longest contour's
shape, how much of the window is tonal, and a frequency profile of the contour points.
All times are Unix seconds (UTC).
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .contours import day_of

AGG_PARAMS = ["duration_s", "start_freq_hz", "end_freq_hz", "min_freq_hz", "max_freq_hz",
              "freq_range_hz", "mean_freq_hz", "std_freq_hz", "linreg_slope_hz_s",
              "quad_curvature", "mean_abs_slope_hz_s", "frac_up", "frac_down",
              "n_inflections", "time_of_min_freq_rel", "time_of_max_freq_rel"]

# Window columns that are labels/metadata, never features
WINDOW_META = {"w_start", "w_end", "ann_idx", "day", "label", "uncertain", "cluster", "label_source",
               "block"}


@dataclass
class WindowParams:
    """Window settings.

    mode                  "sliding" or "annotation" (one window around each annotated call)
    window_s, overlap     sliding window length (s) and overlap (0.5 = 50 %)
    annotation_pad_s      annotation mode: pad each call by this much (s)
    assign_by             a contour belongs to a window if it "overlap"s it or its "midpoint" is inside
    min_contours          windows with fewer contours are dropped
    min_occupancy         ... or with less of their time covered by contour points (0-1)
    min_longest_contour_s ... or whose longest contour is shorter than this (s)
    freq_profile_bins     >0 adds a normalised histogram of contour-point frequencies (this many bins
                          between fmin and fmax) as features
    fmin_hz, fmax_hz      band of the frequency profile (normally the contour filter band)
    """

    mode: str = "sliding"
    window_s: float = 4.0
    overlap: float = 0.5
    annotation_pad_s: float = 0.1
    assign_by: str = "overlap"
    min_contours: int = 2
    min_occupancy: float = 0.0
    min_longest_contour_s: float = 0.0
    freq_profile_bins: int = 32
    fmin_hz: float = 800.0
    fmax_hz: float = 16000.0

    def to_dict(self):
        return asdict(self)


def pair_intervals(a_s, a_e, b_s, b_e):
    """All (i, j, overlap) where interval a_i overlaps b_j; b sorted by start."""
    a_s, a_e, b_s, b_e = (np.asarray(x, float) for x in (a_s, a_e, b_s, b_e))
    max_len = (b_e - b_s).max() if len(b_s) else 0
    lo = np.searchsorted(b_s, a_s - max_len, "left")
    hi = np.searchsorted(b_s, a_e, "left")
    I, J = [], []
    for i in range(len(a_s)):
        if hi[i] > lo[i]:
            js = np.arange(lo[i], hi[i])
            js = js[b_e[js] > a_s[i]]
            I.append(np.full(len(js), i))
            J.append(js)
    if not I:
        return np.array([], int), np.array([], int), np.array([])
    I, J = np.concatenate(I), np.concatenate(J)
    return I, J, np.minimum(a_e[I], b_e[J]) - np.maximum(a_s[I], b_s[J])


def make_windows(contours, params: WindowParams, annotations=None):
    """Window start/end times. Sliding windows are aligned to the minute of each day's first
    contour and kept only if a contour could fall in them."""
    if params.mode == "annotation":
        a = annotations[(annotations["start"] >= contours["start"].min() - 60) &
                        (annotations["end"] <= contours["end"].max() + 60)]
        return pd.DataFrame({"w_start": a["start"].to_numpy() - params.annotation_pad_s,
                             "w_end": a["end"].to_numpy() + params.annotation_pad_s,
                             "ann_idx": a.index.to_numpy()})
    L = params.window_s
    step = L * (1 - params.overlap)
    wins = []
    for _, g in contours.groupby("day"):
        t0 = np.floor(g["start"].min() / 60) * 60
        cs, ce = g["start"].to_numpy() - t0, g["end"].to_numpy() - t0
        first = np.maximum(np.ceil((cs - L) / step + 1e-9), 0).astype(int)
        last = np.floor((ce - 1e-9) / step).astype(int)
        idx = np.unique(np.concatenate([np.arange(a, b + 1) for a, b in zip(first, last) if b >= a]))
        ws = t0 + idx * step
        wins.append(pd.DataFrame({"w_start": ws, "w_end": ws + L}))
    return pd.concat(wins, ignore_index=True).sort_values("w_start").reset_index(drop=True)


def window_features(windows, contours, points, params: WindowParams):
    """Features per window from its contours; windows failing the window filters are dropped."""
    ws, we = windows["w_start"].to_numpy(float), windows["w_end"].to_numpy(float)
    cs, ce = contours["start"].to_numpy(float), contours["end"].to_numpy(float)
    if params.assign_by == "midpoint":
        mid = (cs + ce) / 2
        I, J, _ = pair_intervals(ws, we, mid, mid + 1e-9)
    else:
        I, J, _ = pair_intervals(ws, we, cs, ce)
    ex = contours.loc[J, AGG_PARAMS].reset_index(drop=True)
    ex["w"], ex["cidx"] = I, J
    g = ex.groupby("w")
    feat = g[AGG_PARAMS].agg(["mean", "std"])
    feat.columns = [f"{a}_{b}" for a, b in feat.columns]
    feat.insert(0, "n_contours", g.size())
    feat["snr_db_mean"] = pd.Series(contours["snr_db"].to_numpy()[J]).groupby(I).mean()
    lc_rows = ex.loc[g["duration_s"].idxmax().to_numpy()]          # longest contour per window
    lc = contours.loc[lc_rows["cidx"].to_numpy(), AGG_PARAMS].add_prefix("lc_")
    lc.index = lc_rows["w"].to_numpy()
    feat = feat.join(lc)
    # contour points in each window; edges compared in integer nanoseconds so points that sit
    # exactly on a window edge are assigned the same way every time
    ws_ns, we_ns, cs_ns = (np.round(x * 1e9).astype(np.int64) for x in (ws, we, cs))
    pw, pf, pdur = [], [], []
    for i, j in zip(I, J):
        t = cs_ns[j] + (points[j][0] * 1e9).astype(np.int64)
        m = (t >= ws_ns[i]) & (t < we_ns[i])
        pw.append(np.full(m.sum(), i))
        pf.append(points[j][1][m])
        pdur.append(np.full(m.sum(), points[j][2]))
    pw, pf, pdur = (np.concatenate(x) if x else np.array([]) for x in (pw, pf, pdur))
    gp = pd.Series(pf).groupby(pw)
    feat["tonal_s"] = pd.Series(pdur).groupby(pw).sum()            # seconds of contour in the window
    feat["occupancy"] = feat["tonal_s"] / pd.Series((we_ns - ws_ns) / 1e9)[feat.index]
    feat["pts_mean_freq_hz"] = gp.mean()
    feat["pts_std_freq_hz"] = gp.std()
    feat["pts_q10_hz"] = gp.quantile(0.1)
    feat["pts_q90_hz"] = gp.quantile(0.9)
    nb = params.freq_profile_bins
    if nb:
        edges = np.linspace(params.fmin_hz, params.fmax_hz, nb + 1)
        b = np.clip(np.digitize(pf, edges) - 1, 0, nb - 1)
        prof = pd.crosstab(pw, b).reindex(columns=range(nb), fill_value=0)
        prof = prof.div(prof.sum(axis=1), axis=0)
        prof.columns = [f"prof_{int(edges[k])}_{int(edges[k + 1])}" for k in range(nb)]
        feat = feat.join(prof)
    feat[["tonal_s", "occupancy"]] = feat[["tonal_s", "occupancy"]].fillna(0)
    out = windows.join(feat, how="inner")
    out = out[(out["n_contours"] >= params.min_contours) & (out["occupancy"] >= params.min_occupancy) &
              (out["lc_duration_s"] >= params.min_longest_contour_s)]
    out = out.reset_index(drop=True)
    out.insert(2, "day", day_of(out["w_start"]))
    return out


def label_windows(windows, calls, params: WindowParams, min_call_coverage=0.5, single_label_only=True,
                  carry=(), ignore_for_mixed=("untyped",)):
    """Label windows from call-level annotations (``start``, ``end``, ``label`` in Unix s).

    Annotation mode: each window takes its call's label. Sliding mode: a window takes the label of the
    call it overlaps most among calls it covers at least ``min_call_coverage`` of; windows touching a
    call they don't cover enough of are "partial"; windows with more than one label are "mixed"
    (if ``single_label_only``; labels in ``ignore_for_mixed``, e.g. calls of no particular type, don't
    count); others are "none". ``carry``: other annotation columns to copy (e.g. "ecotype").
    ``uncertain`` is copied if present.
    """
    w = windows.reset_index(drop=True).copy()
    cols = ["label", *carry, *(["uncertain"] if "uncertain" in calls else [])]
    if params.mode == "annotation":
        a = calls.loc[w["ann_idx"].to_numpy()]
        for c in cols:
            w[c] = a[c].to_numpy()
        return w
    calls = calls.sort_values("start")
    I, J, ov = pair_intervals(w["w_start"], w["w_end"], calls["start"], calls["end"])
    dur = np.maximum((calls["end"] - calls["start"]).to_numpy()[J], 1e-9)
    p = pd.DataFrame({"w": I, "ov": ov, "cov": ov / dur, "a": J})
    full = p[p["cov"] >= min_call_coverage]
    best = full.sort_values("ov", ascending=False).drop_duplicates("w").set_index("w")
    for c in cols:
        w[c] = False if c == "uncertain" else "none"
        w.loc[best.index, c] = calls[c].to_numpy()[best["a"].to_numpy()]
    touched = sorted(set(p["w"]) - set(best.index))
    w.loc[touched, "label"] = "partial"
    if single_label_only:
        lab = pd.Series(calls["label"].to_numpy()[p["a"].to_numpy()])
        typed = ~lab.isin(ignore_for_mixed).to_numpy()
        multi = lab[typed].groupby(p["w"].to_numpy()[typed]).nunique()
        w.loc[multi[multi > 1].index, "label"] = "mixed"
    return w


def label_windows_by_period(windows, periods, columns=("label",), source="periods", overwrite=False):
    """Add labels from longer labelled periods (annotations format: ``start``, ``end`` + columns),
    by the window's midpoint. Windows already labelled from an earlier source are kept unless
    ``overwrite``. Adds ``label_source``. Use several calls in priority order to combine sources.
    """
    w = windows.copy()
    mid = ((w["w_start"] + w["w_end"]) / 2).to_numpy()
    if "label_source" not in w:
        w["label_source"] = "unannotated"
    for c in columns:
        if c not in w:
            w[c] = "none"
    p = periods.sort_values("start")
    I, J, _ = pair_intervals(mid, mid + 1e-9, p["start"], p["end"])
    for i, j in zip(I, J):
        if overwrite or w.at[i, "label_source"] == "unannotated":
            for c in columns:
                w.at[i, c] = p[c].iloc[j]
            w.at[i, "label_source"] = source
    return w
