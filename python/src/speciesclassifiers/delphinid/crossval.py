"""Grouped training and cross-validation for delphinID.

Groups (encounters, locations, trials, ...) are the unit of independence:

- **Cross-validation** leaves one group out at a time as the test set (jackknife).
- **Bootstrapping**: within each fold, the model is trained on ``n_bootstraps`` successive random
  subsamples of the remaining data, continuing from the previous bootstrap's weights. Each
  subsample takes at most ``max_per_group`` examples per group, so no single group dominates,
  while across bootstraps the model still sees most of the data.

Each bootstrap subsample is:
1. capped at ``max_per_group`` random examples per group;
2. (optionally) class-balanced down to the smallest class, removing examples from the class's
   largest group first, so group diversity is kept;
3. split per class into training (1 - ``val_fraction``) and validation (``val_fraction``) sets.

Event predictions sum frame probabilities over the event and renormalise.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .model import CNNParams, TrainingParams, build_model, one_hot, predict, train


@dataclass
class ResamplingParams:
    """Grouped resampling.

    max_per_group   most examples taken from any one group in each bootstrap
    n_bootstraps    random subsamples the model is trained on in turn
    val_fraction    share of each bootstrap used for validation
    balance_classes balance classes in each bootstrap (down to the smallest class)
    """

    max_per_group: int = 30
    n_bootstraps: int = 5
    val_fraction: float = 0.33
    balance_classes: bool = True

    def to_dict(self):
        return asdict(self)


def feature_columns(frames):
    """Feature columns f1..fN, in order."""
    cols = [c for c in frames.columns if c.startswith("f") and c[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


def bootstrap_sample(labels, groups, resampling: ResamplingParams, rng):
    """Indices (train, val) for one bootstrap subsample. ``labels``/``groups`` are arrays."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    idx = []
    for g in np.unique(groups):
        gi = np.flatnonzero(groups == g)
        if len(gi) > resampling.max_per_group:
            gi = rng.choice(gi, resampling.max_per_group, replace=False)
        idx.extend(gi)
    idx = np.array(idx)

    by_class = {c: idx[labels[idx] == c] for c in np.unique(labels[idx])}
    if resampling.balance_classes:
        n_min = min(len(v) for v in by_class.values())
        for c, ci in by_class.items():
            ci = list(rng.permutation(ci))
            while len(ci) > n_min:
                g_counts = pd.Series(groups[ci]).value_counts()
                biggest = g_counts.index[0]
                ci.pop(next(k for k, i in enumerate(ci) if groups[i] == biggest))
            by_class[c] = np.array(ci)

    tr, va = [], []
    for ci in by_class.values():
        ci = rng.permutation(ci)
        n_val = int(round(len(ci) * resampling.val_fraction))
        va.extend(ci[:n_val])
        tr.extend(ci[n_val:])
    return rng.permutation(tr), rng.permutation(va)


def fit_bootstrapped(frames, classes, label_col="label", group_col="event_id",
                     cnn: CNNParams = None, training: TrainingParams = None,
                     resampling: ResamplingParams = None, seed=None, verbose=0):
    """Build a model and train it on ``n_bootstraps`` successive subsamples of ``frames``.

    Returns (model, history) where history has one row per bootstrap and epoch.
    """
    training = training or TrainingParams()
    resampling = resampling or ResamplingParams()
    seed = training.seed if seed is None else seed
    feats = feature_columns(frames)
    x = frames[feats].to_numpy("float32")
    y = one_hot(frames[label_col].to_list(), classes)
    model = build_model(len(feats), len(classes), cnn, training)
    rng = np.random.default_rng(seed)
    hist = []
    for b in range(resampling.n_bootstraps):
        tr, va = bootstrap_sample(frames[label_col].to_numpy(), frames[group_col].to_numpy(), resampling, rng)
        h = train(model, x[tr], y[tr], x[va], y[va], training, verbose=verbose)
        for e in range(len(h.history["loss"])):
            hist.append({"bootstrap": b + 1, "epoch": e + 1, "n_train": len(tr), "n_val": len(va),
                         **{k: v[e] for k, v in h.history.items()}})
    return model, pd.DataFrame(hist)


def _clear_session():
    """Free memory between folds."""
    import keras
    keras.backend.clear_session()


def to_standard_output(frames, probs, classes, classifier, voc_type, label_col=None):
    """Frame predictions -> standard classifier output (long format)."""
    n = len(frames)
    det_id = (frames["event_id"].astype(str) + "_" + frames["frame_start"].round(3).astype(str)).to_numpy()
    out = pd.DataFrame({
        "detection_id": np.repeat(det_id, len(classes)),
        "event_id": np.repeat(frames["event_id"].astype(str).to_numpy(), len(classes)),
        "classifier": classifier,
        "voc_type": voc_type,
        "class": np.tile(classes, n),
        "probability": np.asarray(probs, float).ravel(),
    })
    if label_col:
        out["true_class"] = np.repeat(frames[label_col].to_numpy(), len(classes))
    return out


def cross_validate(frames, classes=None, label_col="label", group_col="event_id",
                   cnn: CNNParams = None, training: TrainingParams = None,
                   resampling: ResamplingParams = None, test_groups=None,
                   classifier="delphinID", voc_type="whistle", progress=True):
    """Leave-one-group-out cross-validation with bootstrapped training.

    For each test group a fresh model is trained (``fit_bootstrapped``) on all other groups and
    used to predict every frame of the test group.

    Returns ``(predictions, history)``: frame predictions in the standard classifier output
    format (plus ``true_class`` and ``test_group``), and the training history of every fold.
    Use ``summarise`` for accuracies.
    """
    training = training or TrainingParams()
    classes = list(classes) if classes is not None else sorted(frames[label_col].unique())
    frames = frames[frames[label_col].isin(classes)].reset_index(drop=True)
    feats = feature_columns(frames)
    groups = list(test_groups) if test_groups is not None else list(pd.unique(frames[group_col]))
    preds, hists = [], []
    for k, g in enumerate(groups):
        test = frames[frames[group_col] == g]
        rest = frames[frames[group_col] != g]
        missing = set(classes) - set(rest[label_col])
        if missing:
            print(f"[{k + 1}/{len(groups)}] {g}: skipped, no training data left for {sorted(missing)}")
            continue
        model, h = fit_bootstrapped(rest, classes, label_col, group_col, cnn, training, resampling,
                                    seed=training.seed + k)
        p = predict(model, test[feats].to_numpy("float32"))
        out = to_standard_output(test, p, classes, classifier, voc_type, label_col)
        out["test_group"] = g
        preds.append(out)
        hists.append(h.assign(test_group=g))
        if progress:
            acc = (np.array(classes)[p.argmax(1)] == test[label_col].to_numpy()).mean()
            ev = np.array(classes)[p.sum(0).argmax()]
            print(f"[{k + 1}/{len(groups)}] {g}: {len(test)} frames, frame accuracy {acc:.2f}, "
                  f"group prediction {ev} (true {test[label_col].iloc[0]})")
        del model
        _clear_session()
    return pd.concat(preds, ignore_index=True), pd.concat(hists, ignore_index=True)


def event_predictions(predictions):
    """Sum frame probabilities per event and renormalise. One row per event, one column per class,
    plus ``predicted`` (and ``true_class`` if present)."""
    keys = ["classifier", "event_id"]
    ev = predictions.groupby(keys + ["class"])["probability"].sum().unstack("class")
    ev = ev.div(ev.sum(axis=1), axis=0)
    ev["predicted"] = ev.idxmax(axis=1)
    ev["n_frames"] = predictions.groupby(keys)["detection_id"].nunique()
    if "true_class" in predictions:
        ev["true_class"] = predictions.groupby(keys)["true_class"].first()
    return ev.reset_index()


def summarise(predictions):
    """Accuracy summary of cross-validation predictions.

    Returns a dict with overall frame and event accuracy, per-class event accuracy, per-group
    frame accuracy, and frame/event confusion matrices (rows true, columns predicted).
    """
    wide = predictions.pivot_table(index=["event_id", "detection_id", "true_class"], columns="class",
                                   values="probability").reset_index()
    classes = sorted(predictions["class"].unique())
    wide["predicted"] = wide[classes].idxmax(axis=1)
    ev = event_predictions(predictions)
    wide["correct"] = wide["predicted"] == wide["true_class"]
    ev["correct"] = ev["predicted"] == ev["true_class"]
    return {
        "frame_accuracy": wide["correct"].mean(),
        "event_accuracy": ev["correct"].mean(),
        "event_accuracy_by_class": ev.groupby("true_class")["correct"].mean(),
        "frame_accuracy_by_event": wide.groupby("event_id")["correct"].mean(),
        "frame_confusion": pd.crosstab(wide["true_class"], wide["predicted"]),
        "event_confusion": pd.crosstab(ev["true_class"], ev["predicted"]),
        "events": ev,
    }
