"""Click windows: inter-click intervals (ICIs) of the clicks in each window.

Clicks are detections from the PAMGuard Click Detector (``io.read_clicks``) or any table with a ``start``
column (Unix s, UTC; ``duration`` in s optional). A window's ICIs are the gaps between consecutive clicks in
it; gaps longer than ``max_ici_s`` are pauses between click trains and are left out. All clicks in a window
are pooled (no separation of click trains or animals).
"""

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
    "n_clicks": ("Window", "number of clicks (depends on distance)", False),
}


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
    c = pd.DataFrame({"start": t.to_numpy(), "duration_s": dur.to_numpy()}).dropna(subset=["start"])
    c["end"] = c["start"] + c["duration_s"]
    c = c.sort_values("start", kind="mergesort").reset_index(drop=True)
    c["day"] = day_of(c["start"]) if len(c) else []
    return c[["start", "end", "duration_s", "day"]]


def click_windows(clicks, params: ClickWindowParams = None):
    """Sliding windows over prepared clicks (``prepare_clicks``) with their ICI features. Windows are aligned
    to the minute of each day's first click; only windows with at least ``min_clicks`` clicks and one
    interval are kept."""
    params = params or ClickWindowParams()
    cols = ["w_start", "w_end", "day", "n_clicks", "n_intervals", *[k for k in CLICK_FEATURES if k != "n_clicks"]]
    if not len(clicks):
        return pd.DataFrame(columns=cols)
    L = params.window_s
    step = L * (1 - params.overlap)
    rows = []
    for day, g in clicks.groupby("day", sort=True):
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
            rows.append((s, s + L, day, n, len(d), d.mean(), np.median(d), d.std(ddof=1) if len(d) > 1 else 0.0,
                         np.quantile(d, 0.05), np.quantile(d, 0.95)))
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
