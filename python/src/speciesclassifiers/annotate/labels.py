"""Labels given to windows (by labelling clusters), kept apart from the clusters so they survive
reclustering, and their export as annotations.

Labels live in ``labels/labels.json`` in the project: {detection type: {window id: label}}, a window id
being its start time (Unix s, 3 decimals). Labelling a cluster labels its windows at that moment;
reclustering keeps every window's label.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def window_ids(windows):
    return [f"{t:.3f}" for t in windows["w_start"].to_numpy(float)]


class LabelStore:
    def __init__(self, path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=0), encoding="utf-8")
        tmp.replace(self.path)

    def of(self, det):
        return self.data.setdefault(det, {})

    def column(self, det, windows, unlabelled=""):
        """Each window's label (``unlabelled`` if none)."""
        lab = self.data.get(det, {})
        return np.array([lab.get(i, unlabelled) for i in window_ids(windows)], dtype=object)

    def set(self, det, ids, label):
        """Label windows (label None or "" = remove their labels)."""
        lab = self.of(det)
        for i in ids:
            if label:
                lab[i] = str(label)
            else:
                lab.pop(i, None)
        self.save()

    def clear(self, label=None, det=None):
        """Remove all labels, or one label's, of one detection type or all."""
        for d in [det] if det else list(self.data):
            if label is None:
                self.data[d] = {}
            else:
                self.data[d] = {k: v for k, v in self.data.get(d, {}).items() if v != label}
        self.save()

    def counts(self, det=None):
        vals = [v for d, lab in self.data.items() if det in (None, d) for v in lab.values()]
        return pd.Series(vals, dtype=object).value_counts().to_dict()


def labelled_periods(windows_by_type, store: LabelStore, max_gap_s=60.0, min_windows=1):
    """Labelled windows -> annotations: per detection type, consecutive windows with the same label whose
    gap (start of one to end of the previous) is at most ``max_gap_s`` merge into one period (event).
    Columns ``event_id, label, start, end`` (UTC date-times), ``detector``, ``n_windows``."""
    rows = []
    for det, w in windows_by_type.items():
        lab = store.column(det, w)
        m = lab != ""
        if not m.any():
            continue
        t = pd.DataFrame({"s": w["w_start"].to_numpy(float)[m], "e": w["w_end"].to_numpy(float)[m],
                          "label": lab[m]}).sort_values(["label", "s"], kind="mergesort")
        for label, g in t.groupby("label", sort=True):
            s, e = g["s"].to_numpy(), g["e"].to_numpy()
            run_end = np.maximum.accumulate(e)
            new = np.r_[True, s[1:] - run_end[:-1] > max_gap_s]
            grp = np.cumsum(new)
            for k in np.unique(grp):
                sel = grp == k
                if sel.sum() >= min_windows:
                    rows.append((det, label, s[sel].min(), e[sel].max(), int(sel.sum())))
    if not rows:
        return pd.DataFrame(columns=["event_id", "label", "start", "end", "detector", "n_windows"])
    a = pd.DataFrame(rows, columns=["detector", "label", "s", "e", "n_windows"]).sort_values("s", kind="mergesort")
    a["event_id"] = [f"{lab}_{d}_{i + 1:04d}" for i, (lab, d) in enumerate(zip(a["label"], a["detector"]))]
    fmt = lambda x: pd.to_datetime(x, unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]  # noqa
    a["start"], a["end"] = fmt(a["s"]), fmt(a["e"])
    return a[["event_id", "label", "start", "end", "detector", "n_windows"]].reset_index(drop=True)
