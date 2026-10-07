"""Per-detection features straight from detections (PAMGuard binaries, or the JSON / CSV formats of the
examples), for random forest base classifiers of several detection types.

- Whistles: each contour's shape measurements (as the call-type method: duration, start/end/min/max/mean
  frequency, frequency range and SD, slopes, curvature, rising/falling shares, inflections, where the
  extremes fall).
- Clicks: each click's spectrum (peak and centroid frequency, -3 / -10 dB bandwidths, band profile) and
  the interval to the previous click (missing if longer than ``max_ici_s``, i.e. a pause between trains).

Detections inside labelled periods (annotations: ``event_id, start, end`` + label columns) take the
period's event and labels; others are dropped (training) or kept unlabelled (new data).
"""

import numpy as np
import pandas as pd


def whistle_features():
    from speciesclassifiers.calltypes.windows import AGG_PARAMS
    return list(AGG_PARAMS)


def whistle_detections(whistles, contour_params=None):
    """Whistle table (``io.read_whistles`` or contours JSON) -> one row per kept contour: ``start``,
    ``end`` and the shape measurements."""
    from speciesclassifiers.calltypes import ContourParams, prepare_contours
    c, _ = prepare_contours(whistles, ContourParams(**(contour_params or {})))
    return c[["start", "end", *whistle_features()]].reset_index(drop=True)


def click_feature_names(clicks):
    from speciesclassifiers.calltypes.clicks import CLICK_MEASURES
    return [m for m in CLICK_MEASURES if m in clicks] + [c for c in clicks.columns if str(c).startswith("csp_")] + \
        ["ici_prev_s"]


def click_detections(clicks, max_ici_s=1.0):
    """Click table (with per-click spectrum measurements) -> one row per click: ``start``, ``end``, the
    spectrum measurements and ``ici_prev_s``."""
    from speciesclassifiers.calltypes.clicks import prepare_clicks
    c = prepare_clicks(clicks)
    gap = np.diff(c["start"].to_numpy(float), prepend=np.nan)
    gap[(gap > max_ici_s) | (gap <= 0)] = np.nan
    c["ici_prev_s"] = gap
    return c.drop(columns=["duration_s", "day"]).reset_index(drop=True)


def label_by_periods(det, periods, keep_unlabelled=False):
    """Give detections the event and label columns of the period their start falls in."""
    from speciesclassifiers.calltypes.windows import pair_intervals
    p = periods.sort_values("start").reset_index(drop=True)
    t = det["start"].to_numpy(float)
    I, J, _ = pair_intervals(t, t + 1e-9, p["start"].to_numpy(float), p["end"].to_numpy(float))
    first = pd.Series(J, index=I).groupby(level=0).first()
    out = det.copy()
    cols = [c for c in p.columns if c not in ("start", "end")]
    for c in cols:
        out[c] = None
        out.loc[first.index, c] = p[c].to_numpy()[first.to_numpy()]
    if not keep_unlabelled:
        out = out[out["event_id"].notna()]
    return out.reset_index(drop=True)
