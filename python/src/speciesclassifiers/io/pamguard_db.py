"""Read PAMGuard databases (.sqlite3): deep learning predictions and recording times.

Deep learning classifier modules write one table per module, ``<Module_Name>_Predictions``, with one
row per classified segment: ``UTC`` and a JSON ``Predicition`` column (PAMGuard's spelling) holding
``{"class_id": [...], "predictions": [[p1, p2, ...]]}``. Class names are not stored, so they must be
given (e.g. from the model's ``delphinID.pdtf``, see ``classes_from_model``).
"""

import json
import sqlite3
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _connect(db):
    if not Path(db).exists():
        raise FileNotFoundError(db)
    return sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)


def list_prediction_tables(db):
    """Names of deep learning prediction tables in the database, with row counts."""
    with _connect(db) as c:
        names = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")
                 if r[0].endswith("_Predictions")]
        return {n: c.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0] for n in names}


def classes_from_model(path):
    """Class names from a PAMGuard model zip or ``.pdtf`` file."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(".pdtf") and not n.startswith("__MACOSX"))
            pdtf = json.loads(z.read(name))
    else:
        pdtf = json.loads(path.read_text())
    return list(pdtf["class_info"]["name_class"])


def _utc_seconds(utc_text, millis=None):
    t = pd.to_datetime(pd.Series(utc_text).str.strip(), utc=True, format="mixed")
    return (t - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()


def read_dl_predictions(db, table, classes, classifier=None, voc_type="other"):
    """Predictions from one deep learning table, in the standard classifier output format.

    table       table name, with or without the ``_Predictions`` suffix
    classes     class names in model output order
    classifier  name to give the classifier in the output (default: the table name)
    voc_type    'whistle', 'click' or 'other'

    Rows with missing or negative predictions are dropped. ``event_id`` is left empty; assign events
    with ``assign_events``. Adds ``start`` (Unix s) for that.
    """
    table = table if table.endswith("_Predictions") else f"{table}_Predictions"
    with _connect(db) as c:
        cols = [r[1] for r in c.execute(f'PRAGMA table_info("{table}")')]
        pcol = "Predicition" if "Predicition" in cols else "Prediction"
        d = pd.read_sql(f'SELECT UID, UTC, "{pcol}" AS pred FROM "{table}"', c)
    probs, keep = [], []
    for p in d["pred"]:
        try:
            a = np.asarray(json.loads(p)["predictions"][0], float)
            ok = len(a) == len(classes) and np.all(a >= 0)
        except (TypeError, ValueError, KeyError, IndexError):
            a, ok = None, False
        keep.append(ok)
        probs.append(a)
    bad = len(d) - sum(keep)
    if bad:
        print(f"{table}: dropped {bad} rows with missing/negative predictions or not {len(classes)} classes")
    d = d[keep].reset_index(drop=True)
    p = np.vstack([probs[i] for i, k in enumerate(keep) if k]) if len(d) else np.empty((0, len(classes)))
    p = p / p.sum(axis=1, keepdims=True)
    k = len(classes)
    name = classifier or table.removesuffix("_Predictions")
    utc = d["UTC"].str.strip()
    return pd.DataFrame({
        "detection_id": np.repeat((name + ":" + d["UID"].astype(str)).to_numpy(), k),
        "event_id": "",
        "time": np.repeat(pd.to_datetime(utc, utc=True, format="mixed").dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ").to_numpy(), k),
        "classifier": name, "voc_type": voc_type,
        "class": np.tile(list(classes), len(d)),
        "probability": p.ravel(),
        "start": np.repeat(_utc_seconds(utc), k),
    })


def read_recordings(db, max_hours=24):
    """One event per recording file, from the ``Sound_Acquisition`` table.

    Each file runs from its last ``Start`` row to its last row (Continue/Stop) within ``max_hours``
    after that. Files without a Start row are skipped. Returns ``event_id`` (file name),
    ``start`` and ``end`` (Unix s), as for annotations.
    """
    with _connect(db) as c:
        d = pd.read_sql("SELECT UTC, Status, SystemName FROM Sound_Acquisition", c)
    d["Status"] = d["Status"].astype(str).str.strip()
    d["SystemName"] = d["SystemName"].astype(str).str.strip()
    d["t"] = _utc_seconds(d["UTC"])
    rows = []
    for name, g in d[d["SystemName"].ne("None")].groupby("SystemName"):
        starts = g.loc[g["Status"] == "Start", "t"]
        if starts.empty:
            continue
        t0 = starts.max()
        after = g.loc[(g["t"] >= t0) & (g["t"] <= t0 + max_hours * 3600), "t"]
        if after.max() > t0:
            rows.append({"event_id": name, "start": t0, "end": after.max()})
    return pd.DataFrame(rows, columns=["event_id", "start", "end"]).sort_values("start", ignore_index=True)
