"""Event classifiers trained on base-classifier outputs (transfer learning).

A random forest (``speciesclassifiers.randomforest``) learns to map event features (mean
base-classifier probabilities) to new labels. Testing leaves one group (e.g. encounter) out at a
time; every event of the left-out group is predicted. The unit predicted is the event.
"""

import numpy as np
import pandas as pd

from ..randomforest import model as rf
from ..randomforest.model import RFParams, RandomForestModel, decision_score


def event_params(n_trees=1000, mtry=None, node_size=5, max_per_group=None, prune=0.0, seed=42):
    """Random forest settings for event classifiers (defaults as Classify-delphinID)."""
    return RFParams(n_trees, mtry, node_size, max_per_group, prune, seed)


def cross_validate(events, features, label_col="label", group_col="event_id", params: RFParams = None,
                   classifier="eventclassifier", progress=True):
    """Leave-one-group-out cross-validation of an event classifier.

    Returns event predictions in the standard classifier output format: ``detection_id`` is the
    event, ``event_id`` the group (encounter), plus ``score`` and ``true_class``.
    """
    params = params or event_params()
    return rf.cross_validate(events, features, label_col, group_col, params, classifier=classifier,
                             voc_type="other", id_col="event_id", progress=progress, unit="events")


def event_table(predictions):
    """Standard-format event predictions -> one row per event: class probabilities, predicted, score."""
    w = predictions.pivot_table(index="detection_id", columns="class", values="probability")
    classes = list(w.columns)
    w["predicted"] = w[classes].idxmax(axis=1)
    w["score"] = decision_score(w[classes].to_numpy())
    first = predictions.drop_duplicates("detection_id").set_index("detection_id")
    w["group"] = first["event_id"]
    if "true_class" in first:
        w["true_class"] = first["true_class"]
    return w.reset_index().rename(columns={"detection_id": "event"})


def summarise(predictions, min_score=0.0):
    """Event-level accuracy; events scoring below ``min_score`` count as discarded."""
    ev = event_table(predictions)
    kept = ev[ev["score"] >= min_score]
    correct = kept["predicted"] == kept["true_class"]
    by_class = correct.groupby(kept["true_class"]).mean()
    return {
        "event_accuracy": correct.mean(),
        "mean_class_accuracy": by_class.mean(),
        "accuracy_by_class": by_class,
        "events_discarded": 1 - len(kept) / max(len(ev), 1),
        "confusion": pd.crosstab(kept["true_class"], kept["predicted"]),
        "events": ev,
    }


def threshold_curve(predictions, thresholds=None):
    """Event accuracy and share of events kept across score thresholds."""
    thresholds = np.linspace(0, 0.5, 26) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        s = summarise(predictions, t)
        rows.append({"threshold": t, "event_accuracy": s["event_accuracy"],
                     "mean_class_accuracy": s["mean_class_accuracy"], "events_kept": 1 - s["events_discarded"]})
    return pd.DataFrame(rows)


def fit(events, features, label_col="label", group_col="event_id", params: RFParams = None):
    """Train the final event classifier on all events."""
    return rf.fit(events, features, label_col, group_col, params or event_params())


def predict(model: RandomForestModel, events, classifier="eventclassifier"):
    """Classify new events (from ``event_features``). Features the new data lack (e.g. a base
    classifier with no predictions at all) are 0. Returns one row per event."""
    ev = events.copy()
    for f in model.features:
        if f not in ev:
            ev[f] = 0.0
    p = model.predict_proba(ev)
    out = ev.drop(columns=model.features).copy()
    for i, c in enumerate(model.classes):
        out[c] = p[:, i]
    out["predicted"] = np.array(model.classes)[p.argmax(1)]
    out["score"] = decision_score(p)
    return out
