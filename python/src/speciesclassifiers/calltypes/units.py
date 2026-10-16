"""Single detections as units, instead of windows: one row per contour (its shape measurements) or per
click (its spectrum and the interval to the previous click), in the same table format as windows
(``w_start``, ``w_end`` = the detection's start and end, ``day``), so the rest of the call-type method
(labels, discovery, library, blocks) works unchanged.

``WindowParams.unit`` / ``ClickWindowParams.unit`` choose: "window" (sliding windows) or "detection".
"""

import numpy as np
import pandas as pd

from .windows import AGG_PARAMS, make_windows, pair_intervals, window_features

DETECTION_META = {"cidx"}      # row of the detection in the prepared contours / clicks


def contour_units(contours):
    """Prepared contours (``prepare_contours``) -> one row per contour: ``w_start``, ``w_end``, ``day``,
    ``cidx`` and its measurements (``AGG_PARAMS``, ``snr_db``)."""
    cols = [c for c in [*AGG_PARAMS, "snr_db"] if c in contours]
    d = contours[cols].copy().reset_index(drop=True)
    d.insert(0, "cidx", np.arange(len(contours)))
    d.insert(0, "day", contours["day"].to_numpy() if "day" in contours else None)
    d.insert(0, "w_end", contours["end"].to_numpy(float))
    d.insert(0, "w_start", contours["start"].to_numpy(float))
    return d


def click_units(clicks, max_ici_s=1.0, min_ici_s=0.0):
    """Prepared clicks (``clicks.prepare_clicks``, sorted by start) -> one row per click: ``w_start``,
    ``w_end``, ``day``, ``cidx``, its spectrum measurements and ``ici_prev_s`` (the interval to the previous
    click; missing if longer than ``max_ici_s`` (a pause) or not longer than ``min_ici_s``)."""
    t = clicks["start"].to_numpy(float)
    gap = np.diff(t, prepend=np.nan)
    gap[(gap > max_ici_s) | (gap <= min_ici_s)] = np.nan
    d = clicks.drop(columns=[c for c in ("start", "end", "duration_s", "day") if c in clicks]).reset_index(drop=True)
    d.insert(0, "cidx", np.arange(len(clicks)))
    d.insert(0, "day", clicks["day"].to_numpy() if "day" in clicks else None)
    d.insert(0, "w_end", clicks["end"].to_numpy(float))
    d.insert(0, "w_start", t)
    d["ici_prev_s"] = gap
    return d


def contour_points(contours, points, wp):
    """The units of whistles: windows of contours (``wp.unit`` "window") or single contours."""
    if getattr(wp, "unit", "window") == "detection":
        return contour_units(contours)
    return window_features(make_windows(contours, wp), contours, points, wp)


def click_points(clicks, cwp):
    """The units of clicks: windows of clicks or single clicks."""
    from .clicks import click_windows
    if getattr(cwp, "unit", "window") == "detection":
        return click_units(clicks, cwp.max_ici_s, cwp.min_ici_s)
    return click_windows(clicks, cwp)


def label_detections(units, calls, carry=(), min_cover=0.5, ignore_for_mixed=("untyped",)):
    """Label single detections from call annotations (``start``, ``end``, ``label``): a detection takes
    the label of the call that covers most of it, if that is at least ``min_cover`` of the detection;
    detections touching calls less than that are "partial"; covered enough by calls with different labels,
    "mixed"; others "none". ``carry``: other annotation columns to copy."""
    w = units.reset_index(drop=True).copy()
    cols = ["label", *carry, *(["uncertain"] if "uncertain" in calls else [])]
    for c in cols:
        w[c] = False if c == "uncertain" else "none"
    if not len(w) or not len(calls):
        return w
    calls = calls.sort_values("start")
    ns = lambda x: np.round(np.asarray(x, float) * 1e6).astype(np.int64)     # noqa: E731
    ws, we = ns(w["w_start"]), ns(w["w_end"])
    I, J, ov = pair_intervals(ws, we, ns(calls["start"]), ns(calls["end"]))
    p = pd.DataFrame({"w": I, "a": J, "ov": ov, "cov": ov / np.maximum(we[I] - ws[I], 1)})
    full = p[p["cov"] >= min_cover]
    best = full.sort_values("ov", ascending=False).drop_duplicates("w").set_index("w")
    for c in cols:
        w.loc[best.index, c] = calls[c].to_numpy()[best["a"].to_numpy()]
    w.loc[sorted(set(p["w"]) - set(best.index)), "label"] = "partial"
    lab = pd.Series(calls["label"].to_numpy()[full["a"].to_numpy()])
    typed = ~lab.isin(ignore_for_mixed).to_numpy()
    multi = lab[typed].groupby(full["w"].to_numpy()[typed]).nunique()
    w.loc[multi[multi > 1].index, "label"] = "mixed"
    return w
