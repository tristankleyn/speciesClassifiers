"""Quick checks of input data, with plain-language findings for the app.

Each check returns a dict: ``ok`` (bool), ``messages`` (list of (level, text), level "ok" / "warn" /
"error"), ``table`` (optional DataFrame preview) and ``info`` (values the app can use, e.g. columns).
"""

from pathlib import Path

import pandas as pd


def _result(ok=True):
    return {"ok": ok, "messages": [], "table": None, "info": {}}


def _err(r, text):
    r["ok"] = False
    r["messages"].append(("error", text))
    return r


def _class_warnings(r, counts, label, min_groups=2):
    small = counts[counts["groups"] < min_groups]
    for _, row in small.iterrows():
        r["messages"].append(("warn", f"Class '{row[label]}' has only {row['groups']} group(s): leave-one-group-out "
                                      f"testing needs at least {min_groups}."))
    if len(counts) < 2:
        _err(r, "At least 2 classes are needed.")


def check_rf_data(d):
    """Random forest data: ROCCA contour stats (file/folder) or a feature table CSV."""
    from .workflows import rf_read
    r = _result()
    path = d.get("path", "")
    if not path or not Path(path).exists():
        return _err(r, f"Not found: {path or '(no path)'}")
    if d["type"] == "table" and Path(path).is_file():
        cols = list(pd.read_csv(path, nrows=0).columns)
        r["info"]["columns"] = cols
        for c in (d["label_col"], d["group_col"]):
            if c not in cols:
                return _err(r, f"Column '{c}' not found. Columns: {', '.join(cols[:30])}")
    try:
        data, feats, label, group = rf_read(d, path)
    except Exception as e:
        return _err(r, f"Couldn't read the data: {e}")
    if d.get("classes"):
        data = data[data[label].astype(str).isin(d["classes"])]
    counts = data.groupby(label).agg(groups=(group, "nunique"), detections=(group, "size")).reset_index()
    r["messages"].append(("ok", f"{len(data)} detections in {data[group].nunique()} groups, {len(counts)} classes, "
                                f"{len(feats)} features."))
    r["info"].update(classes=sorted(data[label].astype(str).unique()), features=feats,
                     columns=list(data.columns))
    _class_warnings(r, counts, label)
    r["table"] = counts
    return r


def check_contours(source, path):
    """Whistle contours: a folder of PAMGuard binaries or a contours JSON."""
    r = _result()
    if not path or not Path(path).exists():
        return _err(r, f"Not found: {path or '(no path)'}")
    if source == "binaries":
        from speciesclassifiers.io.pamguard_binary import WHISTLE_PATTERN, file_start_time, list_binaries
        if not Path(path).is_dir():
            return _err(r, "Choose the folder that holds the binary files (.pgdf).")
        files = list_binaries(path, WHISTLE_PATTERN)
        if not files:
            n_any = sum(1 for _ in Path(path).rglob("*.pgdf"))
            return _err(r, f"No Whistle and Moan Detector binaries (file names containing '{WHISTLE_PATTERN}') "
                           f"found; {n_any} other .pgdf files.")
        times = [t for t in (file_start_time(f) for f in files) if t]
        msg = f"{len(files)} whistle binary files"
        if times:
            t0, t1 = (pd.Timestamp(x, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M") for x in (min(times), max(times)))
            msg += f", {t0} to {t1} UTC"
        r["messages"].append(("ok", msg + "."))
        r["info"]["n_files"] = len(files)
        try:
            import speciesclassifiers.io.pamguard_binary as pb
            pb._pypamguard()
        except ImportError:
            _err(r, "Reading binaries needs pypamguard: pip install pypamguard")
        return r
    from .workflows import load_whistles
    try:
        wh = load_whistles("json", path)
    except Exception as e:
        return _err(r, f"Couldn't read the contours file: {e}")
    missing = [c for c in ("start", "times", "freqs", "slice_s") if c not in wh]
    if missing:
        return _err(r, f"Contours need fields start, times, freqs, slice_s; missing {missing}.")
    days = pd.to_datetime(wh["start"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    r["messages"].append(("ok", f"{len(wh)} contours on {days.nunique()} days ({days.min()} to {days.max()})."))
    r["table"] = days.value_counts().sort_index().rename_axis("day").reset_index(name="contours")
    return r


def check_annotations(lab):
    """Call annotations (CSV or annotation tables): counts per call type and day, other label columns."""
    from .workflows import load_calls, other_label_columns
    r = _result()
    try:
        calls = load_calls(lab)
    except Exception as e:
        return _err(r, f"Couldn't read the annotations: {e}")
    if not len(calls):
        return _err(r, "No annotated calls found.")
    days = pd.to_datetime(calls["start"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    others = other_label_columns(calls, {**lab, "other_labels": None})
    r["messages"].append(("ok", f"{len(calls)} calls, {calls['label'].nunique()} labels, on {days.nunique()} days."))
    if others:
        r["messages"].append(("ok", "Other label columns: " + ", ".join(
            f"{c} ({', '.join(map(str, calls[c].dropna().unique()[:6]))})" for c in others)))
    if days.nunique() < 2:
        r["messages"].append(("warn", "Annotations from only one day: verdicts need labels from at least "
                                      "'min_label_days' days, and testing is by day."))
    r["table"] = pd.crosstab(calls["label"], days).reset_index()
    r["info"] = {"other_labels": others, "values": {c: sorted(map(str, calls[c].dropna().unique())) for c in others},
                 "labels": sorted(calls["label"].astype(str).unique())}
    return r
