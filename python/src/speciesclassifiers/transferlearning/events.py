"""Turn base-classifier predictions into event-level features.

Input: detection predictions from one or more base classifiers in the standard classifier output
format (e.g. delphinID whistle and click models, a random forest on ROCCA data, or anything else).

Each event gets, for every base classifier, the mean predicted probability of each of its classes
(renormalised to sum 1). These are the event features, named ``<classifier>|<class>``. If an event
has no predictions from a base classifier, that classifier's features are 0. Detection counts per
classifier (``n_<classifier>``) are kept for filtering but are not features.
"""

import numpy as np
import pandas as pd

from ..io.annotations import read_annotations
from ..outputs import validate_output

SEP = "|"


def _start_seconds(predictions):
    if "start" in predictions:
        return predictions["start"].to_numpy(float)
    if "time" in predictions:
        t = pd.to_datetime(predictions["time"], utc=True, format="mixed")
        return (t - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()
    raise ValueError("Predictions need a 'time' (ISO) or 'start' (Unix s) column to assign events")


def assign_events(predictions, events):
    """Give each prediction the event whose period contains its time.

    ``events``: CSV path or table with ``event_id``, ``start``, ``end`` and optionally ``label`` and
    other columns (e.g. ``encounter``), in the annotations format; or the output of
    ``io.read_recordings``. Predictions outside every event are dropped. Event columns are copied.
    """
    ev = read_annotations(events, require_label=False)
    t = _start_seconds(predictions)
    base = predictions.drop(columns=[c for c in ev.columns if c in predictions and c != "event_id"])
    out = []
    for _, row in ev.iterrows():
        m = (t >= row["start"]) & (t < row["end"])
        if m.any():
            d = base[m].copy()
            for c in ev.columns:
                if c not in ("start", "end"):
                    d[c] = row[c]
            out.append(d)
    if not out:
        raise ValueError("No predictions fall inside any event period")
    return pd.concat(out, ignore_index=True)


def time_bin_events(predictions, minutes=10):
    """Events as fixed, clock-aligned time bins (e.g. 10 min), named by their start time (UTC)."""
    t = _start_seconds(predictions)
    bins = np.floor(t / (60 * minutes)) * 60 * minutes
    d = predictions.copy()
    d["event_id"] = pd.to_datetime(bins, unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
    return d


def event_features(predictions, keep_cols=None):
    """One row per event: features ``<classifier>|<class>`` (mean probabilities, 0 if the
    classifier has no predictions in the event) and counts ``n_<classifier>``.

    ``keep_cols``: event-level columns to carry over (default: any column constant within every
    event, e.g. ``label``, ``encounter``). Returns ``(events, feature_columns)``.
    """
    validate_output(predictions[[c for c in predictions.columns if c != "score"]])
    p = predictions.copy()
    mean = p.groupby(["event_id", "classifier", "class"], sort=False)["probability"].mean()
    mean = mean / mean.groupby(level=["event_id", "classifier"]).transform("sum")
    wide = mean.unstack(["classifier", "class"])
    features = [f"{c}{SEP}{k}" for c, k in wide.columns]
    wide.columns = features
    wide = wide.fillna(0.0)
    counts = p.groupby(["event_id", "classifier"])["detection_id"].nunique().unstack(fill_value=0)
    counts.columns = [f"n_{c}" for c in counts.columns]
    if keep_cols is None:
        skip = {"detection_id", "event_id", "time", "start", "classifier", "voc_type", "class", "probability",
                "score", "true_class", "test_group", "fold"}
        cand = [c for c in p.columns if c not in skip]
        keep_cols = [c for c in cand if (p.groupby("event_id")[c].nunique(dropna=False) <= 1).all()]
    meta = p.groupby("event_id")[list(keep_cols)].first() if keep_cols else pd.DataFrame(index=wide.index)
    times = p.assign(_t=_start_seconds(p) if ("start" in p or "time" in p) else np.nan).groupby("event_id")["_t"]
    meta["first_detection"], meta["last_detection"] = times.min(), times.max()
    out = meta.join(counts).join(wide).reset_index()
    return out, features


def filter_events(events, min_detections, mode="or"):
    """Keep events with enough detections.

    ``min_detections``: {classifier: minimum}, e.g. {"delphinID_clicks": 5, "delphinID_whistles": 3}.
    ``mode``: 'or' keeps events meeting any minimum, 'and' only those meeting all.
    """
    if not min_detections:
        return events
    ok = [events.get(f"n_{c}", pd.Series(0, index=events.index)) >= n for c, n in min_detections.items()]
    keep = np.logical_or.reduce(ok) if mode == "or" else np.logical_and.reduce(ok)
    return events[keep].reset_index(drop=True)
