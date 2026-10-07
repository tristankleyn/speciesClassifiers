"""Quick checks of input data, with plain-language findings for the app.

Each check returns a dict: ``ok`` (bool), ``messages`` (list of (level, text), level "ok" / "warn" /
"error"), ``table`` (optional DataFrame preview) and ``info`` (values the app can use, e.g. columns).
"""

import json
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


def check_binaries(path, voc_type="whistle", r=None):
    """A folder of PAMGuard binaries from the Whistle and Moan Detector or the Click Detector."""
    from speciesclassifiers.io.pamguard_binary import CLICK_PATTERN, WHISTLE_PATTERN, file_start_time, list_binaries
    r = r or _result()
    if not path or not Path(path).is_dir():
        return _err(r, f"Choose the folder that holds the binary files (.pgdf): {path or '(no path)'}")
    pattern, what = (CLICK_PATTERN, "Click Detector") if voc_type == "click" else (WHISTLE_PATTERN,
                                                                                    "Whistle and Moan Detector")
    files = list_binaries(path, pattern)
    if not files:
        n_any = sum(1 for _ in Path(path).rglob("*.pgdf"))
        return _err(r, f"No {what} binaries (file names containing '{pattern}') found; {n_any} other .pgdf files.")
    times = [t for t in (file_start_time(f) for f in files) if t]
    msg = f"{len(files)} {voc_type} binary files"
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


def check_dl_data(d):
    """delphinID: binaries + an annotations CSV, or a frames CSV (+ its params JSON) from notebook 01."""
    from speciesclassifiers.delphinid import AcousticParams
    r = _result()
    if d["source"] == "frames":
        for k, what in (("frames_csv", "Frames CSV"), ("frames_params", "Parameters JSON")):
            if not d.get(k) or not Path(d[k]).is_file():
                return _err(r, f"{what} not found: {d.get(k) or '(no path)'}")
        try:
            frames = pd.read_csv(d["frames_csv"])
            meta = json.loads(Path(d["frames_params"]).read_text())
            ap = AcousticParams(**{k: v for k, v in meta["acoustic_params"].items()
                                   if k in AcousticParams.__dataclass_fields__})
        except Exception as e:
            return _err(r, f"Couldn't read the frames: {e}")
        n_in = sum(c[0] == "f" and c[1:].isdigit() for c in frames.columns)
        expected = ap.n_bins(meta.get("sample_rate"))
        if n_in != expected:
            _err(r, f"The frames have {n_in} inputs but their parameters give {expected}.")
        g = "event_id"
        counts = frames.groupby(frames["label"].astype(str)).agg(groups=(g, "nunique"), frames=(g, "size")).reset_index()
        r["messages"].append(("ok", f"{len(frames)} {ap.voc_type} frames, {frames[g].nunique()} events, "
                                    f"{len(counts)} labels, {n_in} inputs per frame."))
        _class_warnings(r, counts, "label")
        r["table"] = counts
        r["info"]["classes"] = sorted(counts["label"])
        return r
    check_binaries(d.get("binary_folder"), d.get("voc_type", "whistle"), r)
    from speciesclassifiers.io import read_annotations
    try:
        ann = read_annotations(d.get("annotations"))
    except Exception as e:
        return _err(r, f"Couldn't read the annotations: {e}")
    ann["hours"] = (ann["end"] - ann["start"]) / 3600
    counts = ann.groupby("label").agg(events=("event_id", "nunique"), periods=("event_id", "size"),
                                      hours=("hours", "sum")).reset_index().round(2)
    r["messages"].append(("ok", f"{len(ann)} annotated periods, {ann['event_id'].nunique()} events, "
                                f"{len(counts)} labels."))
    _class_warnings(r, counts.rename(columns={"events": "groups"}), "label")
    r["table"] = counts
    r["info"]["classes"] = sorted(counts["label"].astype(str))
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


def check_clicks(source, path):
    """Clicks: a folder of PAMGuard Click Detector binaries or a clicks CSV (``start``, optional ``duration``)."""
    r = _result()
    if not path or not Path(path).exists():
        return _err(r, f"Not found: {path or '(no path)'}")
    if source == "binaries":
        return check_binaries(path, "click", r)
    from .workflows import load_clicks
    try:
        cl = load_clicks("csv", path)
    except Exception as e:
        return _err(r, f"Couldn't read the clicks: {e}")
    days = pd.to_datetime(cl["start"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    r["messages"].append(("ok", f"{len(cl)} clicks on {days.nunique()} days ({days.min()} to {days.max()})."))
    r["table"] = days.value_counts().sort_index().rename_axis("day").reset_index(name="clicks")
    return r


def check_annotations(lab):
    """Call annotations (CSV or annotation tables) or labelled periods: counts per label and day/event, and
    the label columns with their values (for building the classification)."""
    from .workflows import load_calls, other_label_columns
    r = _result()
    try:
        calls = load_calls(lab)
    except Exception as e:
        return _err(r, f"Couldn't read the annotations: {e}")
    if not len(calls):
        return _err(r, "No annotations found.")
    days = pd.to_datetime(calls["start"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    others = other_label_columns(calls, {**lab, "other_labels": None})
    periods = lab.get("source") == "periods"
    if periods:
        hours = ((calls["end"] - calls["start"]) / 3600).sum()
        r["messages"].append(("ok", f"{len(calls)} labelled periods ({hours:.1f} h), {calls['event_id'].nunique()} "
                                    f"events, on {days.nunique()} days."))
        if not others:
            return _err(r, "No label columns found (besides event_id, start, end).")
        main = others[0]
        t = calls.groupby(calls[main].astype(str)).agg(events=("event_id", "nunique"), periods=("event_id", "size"))
        t["hours"] = (calls.assign(h=(calls["end"] - calls["start"]) / 3600).groupby(calls[main].astype(str))["h"]
                      .sum().round(2))
        r["table"] = t.reset_index()
        few = t[t["events"] < 2].index.tolist()
        if few:
            r["messages"].append(("warn", f"Only one event for: {', '.join(few)}. A cluster needs a label from at "
                                          "least 2 events (setting) before it can indicate it."))
    else:
        r["messages"].append(("ok", f"{len(calls)} calls, {calls['label'].nunique()} labels, on {days.nunique()} days."))
        if days.nunique() < 2:
            r["messages"].append(("warn", "Annotations from only one day: verdicts need labels from at least "
                                          "'min_label_days' days or events."))
        r["table"] = pd.crosstab(calls["label"], days).reset_index()
    if others:
        r["messages"].append(("ok", "Label columns: " + ", ".join(
            f"{c} ({', '.join(map(str, calls[c].dropna().unique()[:8]))})" for c in others)))
    cols = others + ([] if periods else ["label"])
    r["info"] = {"other_labels": others, "columns": cols,
                 "values": {c: sorted(map(str, calls[c].dropna().unique())) for c in cols},
                 "labels": sorted(calls["label"].astype(str).unique()) if "label" in calls else []}
    return r


TF_MIN = (2, 16)        # Keras 3 (model.export) arrived with TensorFlow 2.16
TF_PACKAGES = ("tensorflow", "tensorflow-cpu", "tensorflow-macos", "tensorflow-intel")


def _version_tuple(v):
    out = []
    for part in v.split(".")[:2]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits or 0))
    return tuple(out)


def tensorflow_status():
    """Is a suitable TensorFlow installed in this Python? Reads package versions only (fast, no import).

    Returns {"ok", "installed", "version", "keras", "message"}.
    """
    from importlib import invalidate_caches
    from importlib.metadata import PackageNotFoundError, version
    invalidate_caches()
    tf = None
    for name in TF_PACKAGES:
        try:
            tf = version(name)
            break
        except PackageNotFoundError:
            continue
    try:
        keras = version("keras")
    except PackageNotFoundError:
        keras = None
    need = ".".join(map(str, TF_MIN))
    if tf is None:
        return {"ok": False, "installed": False, "version": None, "keras": keras,
                "message": "TensorFlow isn't installed."}
    if _version_tuple(tf) < TF_MIN or (keras and _version_tuple(keras) < (3, 0)):
        return {"ok": False, "installed": True, "version": tf, "keras": keras,
                "message": f"TensorFlow {tf} is installed, but delphinID needs {need} or newer (with Keras 3)."}
    return {"ok": True, "installed": True, "version": tf, "keras": keras,
            "message": f"TensorFlow {tf} is installed."}
