"""How well do labelled call types separate? Feature transform + grouped random-forest evaluation.

Windows are split into training and test folds by day (or by time block when classes don't span
enough days), so a call type must be recognised on days the model hasn't seen.
"""

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .windows import WINDOW_META

EXCLUDE_LABELS_DEFAULT = ("none", "untyped", "partial", "mixed", "AMBIGUOUS")


@dataclass
class EvalParams:
    """Settings for selecting labelled windows and evaluating them.

    label_col             column with the labels to evaluate (e.g. "label" for call types, or "ecotype")
    exclude_labels        labels that aren't classes (no call, untyped, partial/mixed windows...)
    drop_uncertain        drop windows whose annotation was marked uncertain
    classes               fix the classes compared (None = all with enough windows)
    max_per_class_per_day cap per class and day, so one long encounter can't dominate (None = off)
    min_class_n           drop classes with fewer windows
    exclude_features      feature names containing any of these are not used (e.g. level-dependent ones)
    cv_grouping           "auto" (day if every class spans >= 2 days and there are >= 3 days, else time
                          block), "day" or "block"
    block_min             time-block length (min) for block grouping
    rf_trees              trees in the random forest
    random_state          random seed
    """

    label_col: str = "label"
    exclude_labels: tuple = EXCLUDE_LABELS_DEFAULT
    drop_uncertain: bool = False
    classes: list = None
    max_per_class_per_day: int = 150
    min_class_n: int = 8
    exclude_features: list = field(default_factory=lambda: ["amplitude", "snr", "signal", "noise", "mean_width"])
    cv_grouping: str = "auto"
    block_min: int = 10
    rf_trees: int = 300
    random_state: int = 42

    def to_dict(self):
        return asdict(self)


def select_rows(windows, params: EvalParams):
    """Labelled windows to evaluate: exclusions, class choice, per-class-per-day cap, minimum size."""
    w = windows.copy()
    w["label"] = w[params.label_col].astype(str)
    w = w[~w["label"].isin(params.exclude_labels)]
    if params.drop_uncertain and "uncertain" in w:
        w = w[~w["uncertain"].astype(bool)]
    if params.classes:
        w = w[w["label"].isin(params.classes)]
    if params.max_per_class_per_day and len(w):
        w = pd.concat([g.sample(min(len(g), params.max_per_class_per_day), random_state=params.random_state)
                       for _, g in w.groupby(["label", "day"])])
    counts = w["label"].value_counts()
    w = w[w["label"].isin(counts[counts >= params.min_class_n].index)]
    return w.reset_index(drop=True)


def fit_feature_spec(windows, exclude_features=()):
    """Learn the feature transform: which columns, fill values for missing values, which columns get
    log1p. Stored in the library so new windows are transformed identically."""
    cols = [c for c in windows.columns if c not in WINDOW_META and pd.api.types.is_numeric_dtype(windows[c])
            and not pd.api.types.is_bool_dtype(windows[c])
            and not any(x.lower() in c.lower() for x in exclude_features)]
    X = windows[cols].astype(float)
    std_cols = [c for c in cols if c.endswith("_std")]
    X[std_cols] = X[std_cols].fillna(0)          # one contour in the window -> spread 0
    med = X.median()
    X = X.fillna(med)
    keep = [c for c in X.columns if X[c].std() > 0]
    X = X[keep]
    log_cols = [c for c in keep if X[c].min() >= 0 and X[c].skew() > 2]
    return {"cols": keep, "std_fill_cols": [c for c in std_cols if c in keep],
            "medians": {c: float(med[c]) for c in keep}, "log_cols": log_cols}


def apply_feature_spec(windows, spec):
    """Window features -> transformed feature matrix (before standardising)."""
    X = pd.DataFrame({c: (windows[c].astype(float) if c in windows.columns else np.nan) for c in spec["cols"]},
                     index=windows.index)
    X[spec["std_fill_cols"]] = X[spec["std_fill_cols"]].fillna(0)
    X = X.fillna(pd.Series(spec["medians"]))
    for c in spec["log_cols"]:
        X[c] = np.log1p(X[c])
    return X


def feature_matrix(windows, exclude_features=()):
    return apply_feature_spec(windows, fit_feature_spec(windows, exclude_features))


def cv_groups(windows, params: EvalParams):
    """Grouping for cross-validation and the grouping actually used ("day" or "block")."""
    grouping = params.cv_grouping
    if grouping == "auto":
        per_class_days = windows.groupby("label")["day"].nunique()
        grouping = "day" if per_class_days.min() >= 2 and windows["day"].nunique() >= 3 else "block"
    if grouping == "day":
        return windows["day"].astype(str).to_numpy(), grouping
    block = (np.floor(windows["w_start"] / (60 * params.block_min)) * 60 * params.block_min).astype(np.int64)
    return (windows["day"].astype(str) + "_" + block.astype(str)).to_numpy(), grouping


def evaluate(windows, params: EvalParams = None):
    """Grouped random-forest evaluation of selected, labelled windows (see ``select_rows``).

    Returns a dict: window/class/feature counts, cross-validated balanced accuracy, macro-F1, chance
    and chance-corrected skill (0 = chance, 1 = perfect), per-class recall, confusion matrix, feature
    importance, and clustering agreement with labels (k-means AMI, label silhouette).
    """
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (adjusted_mutual_info_score, balanced_accuracy_score, confusion_matrix,
                                 f1_score, recall_score, silhouette_score)
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    params = params or EvalParams()
    w = windows
    y = w["label"].to_numpy()
    X = feature_matrix(w, params.exclude_features)
    labels = sorted(np.unique(y))
    res = {"n_windows": len(y), "n_classes": len(labels), "n_features": X.shape[1],
           "class_counts": w["label"].value_counts().to_dict()}
    if len(labels) < 2:
        return res
    groups, grouping = cv_groups(w, params)
    n_splits = int(min(5, pd.Series(y).value_counts().min(), len(np.unique(groups))))
    res["cv_grouping"] = grouping
    if n_splits >= 2:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=params.random_state)
        pred = np.empty(len(y), dtype=object)
        imps = []
        for tr, te in cv.split(X, y, groups):
            rf = RandomForestClassifier(params.rf_trees, class_weight="balanced", min_samples_leaf=2,
                                        n_jobs=-1, random_state=params.random_state)
            rf.fit(X.iloc[tr], y[tr])
            pred[te] = rf.predict(X.iloc[te])
            imps.append(rf.feature_importances_)
        res["n_folds"] = n_splits
        res["rf_bal_acc"] = balanced_accuracy_score(y, pred)
        res["rf_macro_f1"] = f1_score(y, pred, average="macro", zero_division=0)
        res["chance"] = 1 / len(labels)
        res["skill"] = (res["rf_bal_acc"] - res["chance"]) / (1 - res["chance"])
        rec = recall_score(y, pred, labels=labels, average=None, zero_division=0)
        res["recall_per_class"] = {l: float(r) for l, r in zip(labels, rec)}
        res["confusion"] = pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=labels, columns=labels)
        res["feature_importance"] = pd.Series(np.mean(imps, axis=0), index=X.columns).sort_values(ascending=False)
        res["top_features"] = list(res["feature_importance"].index[:5])
        res["predictions"] = pred
    Xs = StandardScaler().fit_transform(X)
    cl = KMeans(len(labels), n_init=10, random_state=params.random_state).fit_predict(Xs)
    res["kmeans_ami"] = adjusted_mutual_info_score(y, cl)
    res["label_silhouette"] = silhouette_score(Xs, y) if len(y) > len(labels) else np.nan
    return res


def results_row(res, settings=None):
    """One flat row (for an experiments log) from ``evaluate`` results plus any settings."""
    skip = {"confusion", "feature_importance", "predictions"}
    row = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in res.items() if k not in skip}
    return {**(settings or {}), **row}
