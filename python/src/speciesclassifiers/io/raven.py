"""Convert Raven selection tables to the annotations format (event_id/label/start/end in UTC).

Raven tables give times either as clock times (``Begin Date`` + ``Begin Clock Time``, added with the
"Begin Date"/"Begin Clock Time" measurements) or as seconds from the start of the recording
(``Begin Time (s)``). Tab-separated .txt, .csv and Excel (all sheets) are read.
"""

import datetime as _dt
from pathlib import Path

import pandas as pd


def _read(path):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.concat(pd.read_excel(path, sheet_name=None).values(), ignore_index=True)
    sep = "\t" if path.suffix.lower() == ".txt" else ","
    return pd.read_csv(path, sep=sep)


def _clock_to_td(x):
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, _dt.time):
        return pd.Timedelta(hours=x.hour, minutes=x.minute, seconds=x.second, microseconds=x.microsecond)
    return pd.to_timedelta(str(x))


def read_raven(path, label_col="Annotation", date_col=None, clock_col="Begin Clock Time",
               recording_start=None, event_id=None, label_parser=None, keep_cols=()):
    """Read a Raven selection table (or several, as a list) into annotations.

    label_col        column with the label (e.g. "Annotation"), or a list of alternative names
                     (tables/sheets that name it differently)
    date_col         column with the date; default: "Begin Date" or "Date", whichever exists
    clock_col        column with the begin clock time
    recording_start  if the table has no clock times: UTC start of the recording (e.g.
                     "2018-09-09 21:00:00"); times are then Begin Time (s) after it
    event_id         value or column for ``event_id`` (default: the source file name)
    label_parser     optional function label -> label (e.g. to tidy call-type names); returning
                     None drops the selection
    keep_cols        extra columns to keep (e.g. "Low Freq (Hz)", "Ecotype")

    End = start + Delta Time (s) when available (clock end times can be truncated), else End Time.
    Returns a DataFrame with ``event_id, label, start, end`` (UTC text, microseconds) and ``keep_cols``.
    """
    if isinstance(path, (list, tuple)):
        return pd.concat([read_raven(p, label_col, date_col, clock_col, recording_start, event_id,
                                     label_parser, keep_cols) for p in path], ignore_index=True)
    d = _read(path)
    if recording_start is not None:
        t0 = pd.Timestamp(recording_start, tz="UTC")
        start = t0 + pd.to_timedelta(d["Begin Time (s)"], unit="s")
        end = t0 + pd.to_timedelta(d["End Time (s)"], unit="s")
    else:
        date_col = date_col or next((c for c in ("Begin Date", "Date") if c in d), None)
        if date_col is None or clock_col not in d:
            raise ValueError("No clock times found: give date_col/clock_col, or recording_start")
        date = pd.to_datetime(d[date_col]).dt.normalize()
        start = (date + d[clock_col].map(_clock_to_td)).dt.tz_localize("UTC")
        end = start + pd.to_timedelta(d["End Time (s)"] - d["Begin Time (s)"], unit="s")
    if "Delta Time (s)" in d:
        dt = pd.to_numeric(d["Delta Time (s)"], errors="coerce")
        end = (start + pd.to_timedelta(dt, unit="s")).where(dt.notna(), end)
    cols = [label_col] if isinstance(label_col, str) else list(label_col)
    label = pd.Series([None] * len(d), dtype=object)
    for c in cols:                                   # first non-empty of the given columns
        if c in d:
            label = label.where(label.notna(), d[c])
    if label_parser is not None:
        label = label.map(label_parser)
    eid = (d[event_id] if isinstance(event_id, str) and event_id in d else
           event_id if event_id is not None else Path(path).stem)
    out = pd.DataFrame({"event_id": eid, "label": label,
                        "start": start.dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        "end": end.dt.strftime("%Y-%m-%d %H:%M:%S.%f")})
    for c in keep_cols:
        out[c] = d[c].to_numpy()
    out = out.dropna(subset=["start", "end", "label"])
    bad = out["end"] <= out["start"]
    if bad.any():
        print(f"Dropped {bad.sum()} selection(s) with zero or negative duration")
        out = out[~bad]
    return out.reset_index(drop=True)
