"""Validation and reshaping for the standard classifier output format."""

import pandas as pd

REQUIRED_COLUMNS = ["detection_id", "event_id", "classifier", "voc_type", "class", "probability"]
VOC_TYPES = {"whistle", "click", "other"}
SUM_TOLERANCE = 0.01


def validate_output(df: pd.DataFrame) -> pd.DataFrame:
    """Check that ``df`` follows the standard classifier output format.

    Returns ``df`` unchanged if valid; raises ``ValueError`` listing every problem otherwise.
    """
    problems = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in ["detection_id", "event_id", "classifier", "class"]:
        if df[col].isna().any():
            problems.append(f"'{col}' has missing values")

    bad_voc = set(df["voc_type"].dropna().unique()) - VOC_TYPES
    if bad_voc or df["voc_type"].isna().any():
        problems.append(f"'voc_type' must be one of {sorted(VOC_TYPES)}; found {sorted(map(str, bad_voc))}")

    prob = pd.to_numeric(df["probability"], errors="coerce")
    if prob.isna().any() or (prob < 0).any() or (prob > 1).any():
        problems.append("'probability' must be numeric between 0 and 1")
    else:
        keys = ["classifier", "detection_id"]
        if df.duplicated(keys + ["class"]).any():
            problems.append("duplicate (classifier, detection_id, class) rows")
        sums = prob.groupby([df[k] for k in keys]).sum()
        off = sums[(sums - 1).abs() > SUM_TOLERANCE]
        if len(off):
            problems.append(f"{len(off)} detection(s) whose probabilities don't sum to 1, e.g. {off.index[0]}")

    if "time" in df.columns:
        t = pd.to_datetime(df["time"], errors="coerce", utc=True)
        if (t.isna() & df["time"].notna()).any():
            problems.append("'time' has values that aren't ISO 8601 datetimes")

    if problems:
        raise ValueError("Invalid classifier output:\n- " + "\n- ".join(problems))
    return df


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Long -> wide: one row per (classifier, detection_id), one column per class."""
    validate_output(df)
    id_cols = [c for c in ["classifier", "voc_type", "detection_id", "event_id", "time"] if c in df.columns]
    wide = df.pivot_table(index=id_cols, columns="class", values="probability", aggfunc="first")
    wide.columns.name = None
    return wide.reset_index()
