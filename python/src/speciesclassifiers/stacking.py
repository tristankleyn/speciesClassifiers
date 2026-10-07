"""Final layer over base classifiers of several detection types (stacking), per clock-aligned block.

Base classifiers (one per detection type, e.g. random forests or delphinID CNNs on whistles and clicks) are
tested by leaving one event out: each event's detections are predicted by a model trained without that event.
Those held-out probabilities are averaged per block (``block_minutes``, clock-aligned, within an event) and per
type, and the types are put side by side in one vector per block:

    <type>|<class>   mean probability of the class over the type's detections in the block (0 if none)
    has_<type>       1 if the type has detections in the block, else 0

A balanced random forest is trained on these block vectors and tested by leaving one event out (all blocks of
the event). For new data, the base classifiers trained on all events predict the detections and the final
forest trained on all blocks decides each block.
"""

import numpy as np
import pandas as pd


SEP = "|"


def _seconds(pred):
    if "start" in pred:
        return pred["start"].to_numpy(float)
    t = pd.to_datetime(pred["time"], utc=True, format="mixed")
    return (t - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()


def block_vectors(preds, block_minutes=10, types=None, by_event=True):
    """Detection predictions of several types (standard output, long; with ``voc_type``, ``event_id`` and
    ``time`` or ``start``; optional ``true_class``) -> one row per block: ``block_id``, ``event_id``,
    ``block_start`` (Unix s), ``true_class`` (majority, if known), ``n_<type>``, and the features.
    Returns (table, feature names)."""
    p = preds.copy()
    p["voc_type"] = p["voc_type"].astype(str)
    types = list(types or sorted(p["voc_type"].unique()))
    t = _seconds(p)
    blk = np.floor(t / (60 * block_minutes)) * 60 * block_minutes
    p["block_start"] = blk
    p["event_id"] = p["event_id"].astype(str) if by_event else pd.Series(blk).map(lambda x: f"{x:.0f}").to_numpy()
    p["block_id"] = p["event_id"] + "@" + pd.Series(blk, index=p.index).map(lambda x: f"{x:.0f}")
    classes = sorted(p["class"].astype(str).unique())
    mean = p.groupby(["block_id", "voc_type", "class"])["probability"].mean()
    mean = mean / mean.groupby(level=["block_id", "voc_type"]).transform("sum")
    wide = mean.unstack(["voc_type", "class"])
    names = [f"{ty}{SEP}{c}" for ty in types for c in classes]
    wide.columns = [f"{a}{SEP}{b}" for a, b in wide.columns]
    wide = wide.reindex(columns=names).fillna(0.0)
    n = p.drop_duplicates(["block_id", "voc_type", "detection_id"]).groupby(["block_id", "voc_type"]).size() \
        .unstack(fill_value=0).reindex(columns=types, fill_value=0)
    for ty in types:
        wide[f"has_{ty}"] = (n[ty].reindex(wide.index).fillna(0) > 0).astype(float)
    feats = names + [f"has_{ty}" for ty in types]
    meta = p.groupby("block_id").agg(event_id=("event_id", "first"), block_start=("block_start", "first"))
    if "true_class" in p:
        meta["true_class"] = p.drop_duplicates(["block_id", "detection_id", "voc_type"]).groupby("block_id")[
            "true_class"].agg(lambda s: s.astype(str).value_counts().index[0])
    for ty in types:
        meta[f"n_{ty}"] = n[ty].reindex(meta.index).fillna(0).astype(int)
    out = meta.join(wide[feats]).reset_index()
    return out, feats


def heldout_rows(pred_blocks, classes):
    """Block predictions (standard output from ``transferlearning``-style cross-validation: ``detection_id`` =
    block, ``event_id``) joined with block starts -> rows for ``calltypes.multiclass`` (``true``, ``unit``,
    ``w_start``, ``p_<class>``)."""
    w = pred_blocks.pivot_table(index=["detection_id", "event_id", "true_class"], columns="class",
                                values="probability").reset_index()
    out = pd.DataFrame({"true": w["true_class"].astype(str), "unit": w["event_id"].astype(str),
                        "block": w["detection_id"].astype(str)})
    out["w_start"] = out["block"].str.split("@").str[-1].astype(float)
    for c in classes:
        out[f"p_{c}"] = w[c].to_numpy() if c in w else 0.0
    return out.sort_values(["unit", "w_start"]).reset_index(drop=True)


def type_rows(blocks, voc_type, classes):
    """What one detection type alone says per block (its mean probabilities; NaN without detections), as
    rows for ``calltypes.multiclass``."""
    out = pd.DataFrame({"true": blocks["true_class"].astype(str), "unit": blocks["event_id"].astype(str),
                        "block": blocks["block_id"], "w_start": blocks["block_start"].astype(float)})
    has = blocks[f"has_{voc_type}"].to_numpy() > 0
    for c in classes:
        out[f"p_{c}"] = np.where(has, blocks[f"{voc_type}{SEP}{c}"], np.nan)
    return out


def cross_validate_final(blocks, features, params, progress=True):
    """Leave one event out over block vectors. Returns standard-output block predictions."""
    from .randomforest import model as rf
    return rf.cross_validate(blocks, features, "true_class", "event_id", params, classifier="final-forest",
                             voc_type="other", id_col="block_id", progress=progress, unit="blocks")


def fit_final(blocks, features, params):
    from .randomforest import model as rf
    return rf.fit(blocks, features, "true_class", "event_id", params)


def decide_blocks(model, blocks, block_minutes=10, min_score=0.0):
    """New blocks (``block_vectors(..., by_event=False)``) -> class probabilities, ``decision``, ``score``."""
    from .randomforest.model import decision_score
    b = blocks.copy()
    for f in model.features:
        if f not in b:
            b[f] = 0.0
    P = model.predict_proba(b)
    score = decision_score(P)
    out = b[["block_start"] + [c for c in b.columns if c.startswith("n_")]].copy()
    out.insert(1, "block_end", out["block_start"] + 60 * block_minutes)
    for i, c in enumerate(model.classes):
        out[f"p_{c}"] = P[:, i]
    out["predicted"] = np.array(model.classes)[P.argmax(1)]
    out["score"] = score
    out["decision"] = np.where(score >= min_score, out["predicted"], "uncertain")
    return out.sort_values("block_start").reset_index(drop=True)
