"""Annotations: labelled time periods in a simple CSV.

Required columns (one row per labelled period):

    event_id   ID of the encounter/event the period belongs to (several rows may share one)
    label      class, e.g. species code
    start      period start, ISO date-time, e.g. 2018-09-09 21:55:00 (UTC)
    end        period end, same format

Any other columns (e.g. location, recorder) are kept and copied to detections and frames.
See examples/annotations_example.csv.
"""

import pandas as pd

REQUIRED = ["event_id", "label", "start", "end"]


def read_annotations(path_or_df):
    """Read and check an annotations CSV (or DataFrame). Times become Unix seconds (UTC)."""
    a = pd.read_csv(path_or_df) if not isinstance(path_or_df, pd.DataFrame) else path_or_df.copy()
    a.columns = [c.strip() for c in a.columns]
    missing = [c for c in REQUIRED if c not in a.columns]
    if missing:
        raise ValueError(f"Annotations need columns {REQUIRED}; missing {missing}")
    for c in ["start", "end"]:
        t = pd.to_datetime(a[c], utc=True, errors="coerce")
        bad = a.loc[t.isna(), c]
        if len(bad):
            raise ValueError(f"Can't read '{c}' times, e.g. {bad.iloc[0]!r}. Use e.g. 2018-09-09 21:55:00")
        a[c] = (t - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    if (a["end"] <= a["start"]).any():
        raise ValueError(f"Rows where end is not after start: {list(a.index[a['end'] <= a['start']])}")
    a["event_id"] = a["event_id"].astype(str)
    a["label"] = a["label"].astype(str)
    return a


def label_detections(detections, annotations):
    """Keep detections that start inside an annotated period and add its columns.

    A detection inside two overlapping periods is kept once per period.
    """
    out = []
    extra = [c for c in annotations.columns if c not in ("start", "end")]
    for _, row in annotations.iterrows():
        d = detections[(detections["start"] >= row["start"]) & (detections["start"] < row["end"])].copy()
        for c in extra:
            d[c] = row[c]
        out.append(d)
    if not out:
        return detections.iloc[0:0]
    return pd.concat(out, ignore_index=True)


def periods(annotations):
    """(start, end) pairs, for limiting which binary files are read."""
    return list(zip(annotations["start"], annotations["end"]))
