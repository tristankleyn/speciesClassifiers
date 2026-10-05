"""Read PAMGuard ROCCA contour statistics (RoccaContourStats*.csv).

Two layouts are supported:

- **single file / flat folder**: labels come from the CSV's ``KnownSpecies`` (-> ``label``) and
  ``EncounterID`` (-> ``event_id``) columns;
- **folder hierarchy** (``levels``): each folder level names a column, e.g.
  ``levels=["label", "location", "event_id"]`` for ``root/Dde/SiteA/enc001/RoccaContourStats.csv``.

Whistle rows have FREQMAX > 0 and FREQPEAK == 0; click rows have FREQMAX == 0 and FREQPEAK != 0.
"""

from pathlib import Path

import pandas as pd

FILE_PATTERN = "RoccaContourStats"

WHISTLE_FIRST, WHISTLE_LAST = "FREQMAX", "STEPDUR"
WHISTLE_OMIT_DEFAULT = ["DCQUARTER1MEAN", "DCQUARTER2MEAN", "DCQUARTER3MEAN", "DCQUARTER4MEAN",
                        "DCMEAN", "DCSTDDEV", "RMSSIGNAL", "OVERLAP"]
CLICK_FEATURES = ["DURATION", "FREQCENTER", "FREQPEAK", "BW3DB", "BW3DBLOW", "BW3DBHIGH", "BW10DB",
                  "BW10DBLOW", "BW10DBHIGH", "NCROSSINGS", "SWEEPRATE", "MEANTIMEZC", "MEDIANTIMEZC",
                  "VARIANCETIMEZC"]


def rocca_features(columns, voc_type="whistle", omit=None):
    """ROCCA feature columns for whistles (FREQMAX..STEPDUR) or clicks, minus ``omit``."""
    columns = list(columns)
    omit = set(WHISTLE_OMIT_DEFAULT if omit is None and voc_type == "whistle" else omit or [])
    if voc_type == "whistle":
        feats = columns[columns.index(WHISTLE_FIRST): columns.index(WHISTLE_LAST) + 1]
    elif voc_type == "click":
        feats = [c for c in CLICK_FEATURES if c in columns]
    else:
        raise ValueError("voc_type must be 'whistle' or 'click'")
    return [c for c in feats if c not in omit]


def _read_one(path, voc_type):
    d = pd.read_csv(path)
    if voc_type == "whistle":
        d = d[(d["FREQMAX"] != 0) & (d["FREQPEAK"] == 0)]
    else:
        d = d[(d["FREQMAX"] == 0) & (d["FREQPEAK"] != 0)]
    return d.assign(source_file=Path(path).name)


def read_rocca(path, voc_type="whistle", levels=None, omit=None, filter_min=None, filter_max=None):
    """Read ROCCA contour stats into a table with ``label``, ``event_id`` and feature columns.

    path        a RoccaContourStats CSV, a folder of them, or the root of a folder hierarchy
    levels      column names for each folder level (hierarchy layout); None = flat
    omit        feature columns to leave out (default for whistles: DC* means, RMSSIGNAL, OVERLAP)
    filter_min  {column: minimum} rows below are dropped, e.g. {"DURATION": 0.1}
    filter_max  {column: maximum}

    Returns ``(table, feature_columns)``. Rows with missing feature values are dropped.
    """
    path = Path(path)
    frames = []
    if path.is_file():
        frames.append(_read_one(path, voc_type))
    elif levels:
        for f in sorted(path.rglob(f"*{FILE_PATTERN}*.csv")):
            parts = f.relative_to(path).parts[:-1]
            if len(parts) < len(levels):
                continue
            frames.append(_read_one(f, voc_type).assign(**dict(zip(levels, parts))))
    else:
        frames += [_read_one(f, voc_type) for f in sorted(path.glob(f"*{FILE_PATTERN}*.csv"))]
    if not frames:
        raise FileNotFoundError(f"No {FILE_PATTERN}*.csv files found in {path}")
    d = pd.concat(frames, ignore_index=True)
    if not levels or "label" not in levels:
        d["label"] = d["KnownSpecies"].astype(str)
    if not levels or "event_id" not in levels:
        d["event_id"] = d["EncounterID"].astype(str)
    feats = rocca_features(d.columns, voc_type, omit)
    for col, v in (filter_min or {}).items():
        d = d[d[col] >= v]
    for col, v in (filter_max or {}).items():
        d = d[d[col] <= v]
    n = len(d)
    d = d.dropna(subset=feats).reset_index(drop=True)
    if len(d) < n:
        print(f"Dropped {n - len(d)} rows with missing feature values")
    return d, feats


def read_feature_table(path_or_df, label_col="label", group_col="event_id", features=None, exclude=()):
    """Read any detection table: one row per detection, a label column, a group column and numeric
    feature columns. ``features`` = None uses every numeric column except label, group and ``exclude``.

    Returns ``(table, feature_columns)``.
    """
    d = pd.read_csv(path_or_df) if not isinstance(path_or_df, pd.DataFrame) else path_or_df.copy()
    for c in (label_col, group_col):
        if c not in d:
            raise ValueError(f"Column '{c}' not found; columns are {list(d.columns)[:15]}...")
    if features is None:
        skip = {label_col, group_col, *exclude}
        features = [c for c in d.columns if c not in skip and pd.api.types.is_numeric_dtype(d[c])]
    n = len(d)
    d = d.dropna(subset=features).reset_index(drop=True)
    if len(d) < n:
        print(f"Dropped {n - len(d)} rows with missing feature values")
    return d, list(features)
