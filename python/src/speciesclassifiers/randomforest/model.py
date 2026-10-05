"""Random forest classifiers built from scratch on detection features.

Works on any table with one row per detection: a label column, a group column (encounter,
location, ...) and numeric feature columns (e.g. ROCCA contour stats, or delphinID frames).

Each tree is grown on a balanced sample: the same number of detections from every class (the size
of the smallest class), drawn with replacement, as R's ``randomForest(strata=, sampsize=)``.
Detection predictions are averaged within a group to give the group prediction.

The decision score of a prediction is p1 * (p1 - p2): the top probability times its margin over
the second. Detections scoring below ``min_score`` are left out of group predictions.
"""

from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd

from ..outputs import validate_output


@dataclass
class RFParams:
    """Random forest settings.

    n_trees        trees in the forest
    mtry           features tried at each split (None = square root of the number of features)
    node_size      minimum detections in a leaf
    max_per_group  most detections taken (at random) from one group for training
    prune          share (0-1) of each class's training detections to drop: those furthest from
                   their class centre on the first two principal components
    seed           random seed
    """

    n_trees: int = 500
    mtry: int = None
    node_size: int = 25
    max_per_group: int = 25
    prune: float = 0.0
    seed: int = 42

    def to_dict(self):
        return asdict(self)


def decision_score(probs):
    """p1 * (p1 - p2) per row of a probability array."""
    p = np.sort(np.asarray(probs, float), axis=1)[:, ::-1]
    second = p[:, 1] if p.shape[1] > 1 else 0
    return p[:, 0] * (p[:, 0] - second)


def cap_per_group(df, group_col, n_max, seed=0):
    """At most ``n_max`` random rows per group."""
    if not n_max:
        return df
    return df.sample(frac=1, random_state=seed).groupby(group_col, sort=False).head(n_max)


def prune_outliers(df, features, label_col, prune):
    """Drop the ``prune`` share of each class furthest from its class centre in PC1-PC2 space
    (PCA on unscaled features, as Classify-rocca)."""
    if not prune:
        return df
    x = df[features].to_numpy(float)
    x = x - x.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    pcs = x @ vt[:2].T
    keep = np.ones(len(df), bool)
    labels = df[label_col].to_numpy()
    for c in np.unique(labels):
        m = labels == c
        d = np.linalg.norm(pcs[m] - pcs[m].mean(0), axis=1)
        keep[np.flatnonzero(m)[d >= np.quantile(d, 1 - prune)]] = False
    return df[keep]


class RandomForestModel:
    """A fitted random forest plus what is needed to apply it to new data."""

    def __init__(self, features, classes, params: RFParams, label_col="label"):
        self.features, self.classes, self.params, self.label_col = list(features), list(classes), params, label_col
        self.medians = None
        self.forest = None

    def fit(self, df):
        from imblearn.ensemble import BalancedRandomForestClassifier

        p = self.params
        x = df[self.features].astype(float)
        self.medians = x.median()
        mtry = p.mtry or max(1, int(np.sqrt(len(self.features))))
        self.forest = BalancedRandomForestClassifier(
            n_estimators=p.n_trees, max_features=mtry, min_samples_leaf=p.node_size,
            sampling_strategy="all", replacement=True, bootstrap=False,
            random_state=p.seed, n_jobs=-1 if len(df) > 5000 else 1)  # threads only pay off on big data
        self.forest.fit(x.fillna(self.medians).to_numpy(), df[self.label_col].astype(str).to_numpy())
        order = [list(self.forest.classes_).index(c) for c in self.classes]
        self._order = order
        return self

    def predict_proba(self, df):
        x = df[self.features].astype(float).fillna(self.medians).to_numpy()
        return self.forest.predict_proba(x)[:, self._order]

    def importance(self):
        """Mean decrease in impurity per feature, largest first."""
        return pd.Series(self.forest.feature_importances_, self.features).sort_values(ascending=False)

    def save(self, path):
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path):
        return joblib.load(path)


def fit(df, features, label_col="label", group_col="event_id", params: RFParams = None):
    """Fit a random forest on all of ``df`` (pruned and capped per group as set in ``params``)."""
    params = params or RFParams()
    train = prune_outliers(df, features, label_col, params.prune)
    train = cap_per_group(train, group_col, params.max_per_group, params.seed)
    classes = sorted(train[label_col].astype(str).unique())
    return RandomForestModel(features, classes, params, label_col).fit(train)


def detection_output(df, probs, classes, classifier, voc_type, label_col=None, id_col=None,
                     event_col="event_id", time_col=None):
    """Detection predictions -> standard classifier output (long format), plus ``score``."""
    n, k = len(df), len(classes)
    ids = (df[id_col].astype(str) if id_col else pd.Series(df.index.astype(str), index=df.index)).to_numpy()
    out = pd.DataFrame({
        "detection_id": np.repeat(ids, k),
        "event_id": np.repeat(df[event_col].astype(str).to_numpy(), k),
        "classifier": classifier, "voc_type": voc_type,
        "class": np.tile(classes, n),
        "probability": np.asarray(probs, float).ravel(),
        "score": np.repeat(decision_score(probs), k),
    })
    if time_col and time_col in df:
        out.insert(2, "time", np.repeat(df[time_col].to_numpy(), k))
    if label_col and label_col in df:
        out["true_class"] = np.repeat(df[label_col].astype(str).to_numpy(), k)
    return out


def group_predictions(detections, min_score=0.0, group_col="event_id"):
    """Mean detection probabilities per group, using detections with score >= ``min_score``.

    One row per group: class probabilities, ``predicted``, ``score``, ``n`` (detections used),
    ``n_total`` and ``true_class`` if known.
    """
    wide = detections.pivot_table(index=["detection_id", group_col], columns="class", values="probability").reset_index()
    info = detections.drop_duplicates("detection_id").set_index("detection_id")
    wide["score"] = wide["detection_id"].map(info["score"])
    classes = sorted(detections["class"].unique())
    used = wide[wide["score"] >= min_score]
    g = used.groupby(group_col)[classes].mean()
    g["predicted"] = g.idxmax(axis=1)
    g["score"] = decision_score(g[classes].to_numpy())
    g["n"] = used.groupby(group_col).size()
    g = g.reindex(wide[group_col].unique())
    g["n_total"] = wide.groupby(group_col).size()
    if "true_class" in info:
        g["true_class"] = wide.groupby(group_col)["detection_id"].first().map(info["true_class"])
    return g.reset_index()


def cross_validate(df, features, label_col="label", group_col="event_id", params: RFParams = None,
                   classifier="randomforest", voc_type="whistle", id_col=None, time_col=None, progress=True):
    """Leave-one-group-out cross-validation.

    For each group, a forest is trained on all other groups (pruned and capped per group) and
    predicts every detection of the left-out group. Returns detection predictions in the standard
    output format, with ``score``, ``true_class`` and ``test_group``.
    """
    params = params or RFParams()
    df = df.copy()
    df[label_col] = df[label_col].astype(str)
    groups = list(pd.unique(df[group_col]))
    out = []
    for k, g in enumerate(groups):
        test, rest = df[df[group_col] == g], df[df[group_col] != g]
        if test[label_col].iloc[0] not in set(rest[label_col]):
            print(f"[{k + 1}/{len(groups)}] {g}: skipped, its class has no other groups to train on")
            continue
        m = fit(rest, features, label_col, group_col, params)
        p = m.predict_proba(test)
        d = detection_output(test, p, m.classes, classifier, voc_type, label_col, id_col,
                             event_col=group_col, time_col=time_col)
        d["test_group"] = str(g)
        out.append(d)
        if progress:
            mean = p.mean(0)
            print(f"[{k + 1}/{len(groups)}] {g} ({test[label_col].iloc[0]}): predicted {m.classes[mean.argmax()]} "
                  f"(score {decision_score(mean[None])[0]:.2f}, {len(test)} detections)")
    preds = pd.concat(out, ignore_index=True)
    validate_output(preds)
    return preds


def summarise(detections, min_score=0.0, min_group_score=0.0, group_col="event_id"):
    """Accuracy of cross-validation predictions.

    Detections below ``min_score`` are left out of group predictions; groups below
    ``min_group_score`` are counted as discarded.
    """
    g = group_predictions(detections, min_score, group_col).dropna(subset=["predicted"])
    kept = g[g["score"] >= min_group_score]
    correct = kept["predicted"] == kept["true_class"]
    by_class = correct.groupby(kept["true_class"]).mean()
    return {
        "group_accuracy": correct.mean(),
        "mean_class_accuracy": by_class.mean(),
        "accuracy_by_class": by_class,
        "groups_discarded": 1 - len(kept) / max(len(g), 1),
        "confusion": pd.crosstab(kept["true_class"], kept["predicted"]),
        "groups": g,
    }


def threshold_curve(detections, thresholds=None, group_col="event_id"):
    """Group accuracy and share of groups kept, for a range of group score thresholds."""
    thresholds = np.linspace(0, 0.5, 26) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        s = summarise(detections, min_group_score=t, group_col=group_col)
        rows.append({"threshold": t, "group_accuracy": s["group_accuracy"],
                     "mean_class_accuracy": s["mean_class_accuracy"], "groups_kept": 1 - s["groups_discarded"]})
    return pd.DataFrame(rows)


def predict(model: RandomForestModel, df, group_col="event_id", classifier="randomforest", voc_type="whistle",
            id_col=None, time_col=None, min_score=0.0):
    """Apply a saved model to new detections. Returns (detection predictions, group predictions)."""
    missing = [f for f in model.features if f not in df]
    if missing:
        raise ValueError(f"New data is missing features the model uses: {missing}")
    p = model.predict_proba(df)
    d = detection_output(df, p, model.classes, classifier, voc_type, None, id_col, group_col, time_col)
    return d, group_predictions(d, min_score)  # groups are in the output's event_id column
