"""Random forest final layer: decide each block from the mix of clusters its windows fall in.

For a decision block, each detection type's windows are assigned to that type's clusters (as for new data);
the block's features are the share of its windows in each cluster (and unassigned), per type, side by
side. A balanced random forest (every tree sees the same number of blocks of each class) learns the
classes from labelled blocks. With a pathway, every layer has its own forest on its own clusters.

Strict test (``strict_layer_loo``): for each held-out event (unit), every detection type is clustered
again without that event (feature weights, clusters and cluster thresholds), all windows are assigned to
those clusters, the forest is trained on the other events' blocks and predicts the held-out event's blocks.
Nothing about the held-out event is used, so the accuracy is what new data can expect (given these
settings).

Forests are stored in the library as plain arrays (``forest_to_dict``) and applied with numpy only
(``forest_proba``), so ``standalone.py`` gives the same result.
"""

import base64
import contextlib
import io
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .multiclass import unit_col


@dataclass
class ForestParams:
    """Final-layer forest settings.

    n_trees           trees
    min_samples_leaf  minimum blocks in a leaf
    max_features      features tried per split ("sqrt", or a number)
    n_jobs            held-out events tested in parallel (processes; -1 = one per CPU core)
    seed              random seed
    """

    n_trees: int = 500
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    n_jobs: int = -1
    seed: int = 42

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------------ block features ----------
def feature_names(cluster_ids):
    """{detector: [cluster ids]} -> feature names, in a fixed order."""
    out = []
    for d, ids in cluster_ids.items():
        out += [f"{d}_c{int(c)}" for c in sorted(ids)] + [f"{d}_unassigned"]
    return out


def block_key(w_start, block_minutes):
    blk = int(round(block_minutes * 60 * 1e9))
    return (np.round(np.asarray(w_start, float) * 1e9).astype(np.int64) // blk) * blk


def block_shares(assigned, block_minutes, by_unit=True):
    """{detector: windows with ``cluster`` (and ``unit``)} -> DataFrame of cluster shares per block
    (index: (unit, block) or block; columns ``<det>_c<id>``, ``<det>_unassigned``; types missing from a
    block are 0) and ``n_<det>`` (windows)."""
    parts = []
    for d, w in assigned.items():
        if not len(w):
            continue
        key = block_key(w["w_start"], block_minutes)
        idx = [w["unit"].astype(str).to_numpy(), key] if by_unit else [key]
        cl = np.where(w["cluster"].to_numpy() >= 0, [f"{d}_c{int(c)}" for c in w["cluster"]], f"{d}_unassigned")
        ct = pd.crosstab(idx, cl)
        n = ct.sum(axis=1)
        sh = ct.div(n, axis=0)
        sh[f"n_{d}"] = n
        parts.append(sh)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1).fillna(0.0)
    out.index.names = ["unit", "block"] if by_unit else ["block"]
    return out


def feature_matrix(shares, names):
    return shares.reindex(columns=names, fill_value=0.0).to_numpy(float) if len(shares) else np.empty((0, len(names)))


def block_truth(labelled, block_minutes):
    """{detector: labelled windows with ``unit`` and ``true``} -> majority true class per (unit, block)."""
    rows = [pd.DataFrame({"unit": w["unit"].astype(str).to_numpy(), "block": block_key(w["w_start"], block_minutes),
                          "true": w["true"].astype(str).to_numpy()}) for w in labelled.values() if len(w)]
    t = pd.concat(rows)
    return t.groupby(["unit", "block"])["true"].agg(lambda s: s.value_counts().index[0])


# ------------------------------------------------------------------------ forest ------------------
def fit_forest(X, y, fp: ForestParams, n_jobs=1):
    from speciesclassifiers.randomforest.model import BalancedForest
    mf = fp.max_features
    mf = int(mf) if isinstance(mf, (int, float)) or (isinstance(mf, str) and mf.isdigit()) else mf
    return BalancedForest(fp.n_trees, mf, fp.min_samples_leaf, fp.seed, n_jobs).fit(X, y)


def _b64(a, dtype):
    a = np.ascontiguousarray(a, dtype=dtype)
    return {"shape": list(a.shape), "dtype": np.dtype(dtype).str, "b64": base64.b64encode(a.tobytes()).decode()}


def _unb64(d):
    return np.frombuffer(base64.b64decode(d["b64"]), dtype=d["dtype"]).reshape(d["shape"])


def forest_to_dict(model, names, min_score=0.0):
    """A fitted BalancedForest -> JSON-able dict: one set of node arrays per tree (children, feature,
    threshold, class probabilities in the forest's class order)."""
    trees = []
    k = len(model.classes_)
    for t in model.trees_:
        tr = t.tree_
        val = tr.value[:, 0, :]
        full = np.zeros((tr.node_count, k))
        full[:, t.classes_] = val / np.maximum(val.sum(axis=1, keepdims=True), 1e-12)
        trees.append({"left": _b64(tr.children_left, "<i4"), "right": _b64(tr.children_right, "<i4"),
                      "feature": _b64(tr.feature, "<i4"), "threshold": _b64(tr.threshold, "<f8"),
                      "value": _b64(full, "<f4")})
    return {"classes": [str(c) for c in model.classes_], "features": list(names), "min_score": float(min_score),
            "importance": [float(x) for x in model.feature_importances_], "trees": trees}


_ARRAYS = {}


def forest_proba(fd, X):
    """Class probabilities (rows of X) from a stored forest; X is cast to float32 as in training."""
    X = np.asarray(X, dtype=np.float32).astype(np.float64)
    out = np.zeros((len(X), len(fd["classes"])))
    hit = _ARRAYS.get(id(fd["trees"]))
    if hit is None or hit[0] is not fd["trees"]:      # decoded trees, cached without changing the dict
        hit = _ARRAYS[id(fd["trees"])] = (fd["trees"], [tuple(_unb64(t[k]) for k in
                                                              ("left", "right", "feature", "threshold", "value"))
                                                        for t in fd["trees"]])
    arrays = hit[1]
    for left, right, feat, thr, val in arrays:
        node = np.zeros(len(X), dtype=np.int64)
        active = left[node] != -1
        while active.any():
            i = np.where(active)[0]
            n = node[i]
            go_left = X[i, feat[n]] <= thr[n]
            node[i] = np.where(go_left, left[n], right[n])
            active = left[node] != -1
        out += val[node]
    return out / max(len(arrays), 1)


# ------------------------------------------------------------------------ strict test -------------
def _assign(pool, result, task, block_minutes):
    """Assign windows to a discover result's clusters as new data would be (frozen library)."""
    from .library import assign_windows, build_library, standardise
    from .windows import WindowParams
    lib = build_library(result, None, WindowParams(), [task], block_minutes=block_minutes)
    ids, _, _ = assign_windows(standardise(pool, lib), lib)
    return ids, [c["id"] for c in lib["clusters"]]


def _labelled(pool, task, classes, unit):
    lab = pool[task.column].astype(str)
    m = lab.isin(classes).to_numpy()
    return m


def _fold(u, pools, task, classes, dps, rp, block_minutes, fp):
    """One held-out unit: recluster each type without it, assign, train on other units, predict u."""
    from .discover import discover
    unit = unit_col(rp)
    assigned, ids = {}, {}
    with contextlib.redirect_stdout(io.StringIO()):
        for d, pool in pools.items():
            train = pool[pool[unit].astype(str) != u].reset_index(drop=True)
            if not _labelled(train, task, classes, unit).any():
                continue
            res = discover(train, dps[d], [task], rp, loo=False)
            m = _labelled(pool, task, classes, unit)
            lab = pool[m].reset_index(drop=True)
            cl, ids[d] = _assign(lab, res, task, block_minutes)
            assigned[d] = pd.DataFrame({"w_start": lab["w_start"].to_numpy(), "unit": lab[unit].astype(str).to_numpy(),
                                        "true": lab[task.column].astype(str).to_numpy(), "cluster": cl})
    if not assigned:
        return None
    names = feature_names(ids)
    sh = block_shares(assigned, block_minutes)
    truth = block_truth(assigned, block_minutes).reindex(sh.index)
    test = sh.index.get_level_values("unit") == u
    if not test.any() or (~test).sum() == 0 or truth[~test].nunique() < 2:
        return None
    X = feature_matrix(sh, names)
    model = fit_forest(X[~test], truth[~test].to_numpy(), fp)
    P = model.predict_proba(X[test])
    out = pd.DataFrame(0.0, index=range(int(test.sum())), columns=[f"p_{c}" for c in classes])
    for j, c in enumerate(model.classes_):
        out[f"p_{c}"] = P[:, j]
    out.insert(0, "true", truth[test].to_numpy())
    out.insert(1, "unit", u)
    out.insert(2, "block", sh.index.get_level_values("block")[test].to_numpy())
    out["w_start"] = out["block"] / 1e9
    for d in pools:
        out[f"n_{d}"] = sh.loc[test, f"n_{d}"].to_numpy().astype(int) if f"n_{d}" in sh else 0
    return out


def strict_layer_loo(pools, task, classes, dps, rp, block_minutes=10, fp: ForestParams = None):
    """Held-out block predictions of one layer's forest, reclustering without each held-out unit.

    pools    {detector: discovery pool of this layer (with the layer's label column ``task.column``)}
    dps      {detector: DiscoverParams}
    Returns rows (``true``, ``unit``, ``block``, ``w_start``, ``p_<class>``, ``n_<det>``), ready for
    ``multiclass.aggregate``.
    """
    from joblib import Parallel, delayed
    fp = fp or ForestParams()
    unit = unit_col(rp)
    units = sorted(set().union(*[set(p.loc[_labelled(p, task, classes, unit), unit].astype(str)) for p in pools.values()]))
    n_jobs = (os.cpu_count() or 1) if fp.n_jobs in (-1, None) else max(1, int(fp.n_jobs))
    n_jobs = max(1, min(n_jobs, len(units)))
    print(f"  strict test: {len(units)} held-out {unit}s, reclustering each time ({n_jobs} in parallel)")
    rows, done = [], 0
    jobs = (delayed(_fold)(u, pools, task, classes, dps, rp, block_minutes, fp) for u in units)
    for r in Parallel(n_jobs=n_jobs, return_as="generator_unordered")(jobs):
        done += 1
        print(f"  held-out {done}/{len(units)}")
        if r is not None:
            rows.append(r)
    out = pd.concat(rows, ignore_index=True).sort_values(["unit", "block"], ignore_index=True) if rows else pd.DataFrame()
    out.attrs["classes"] = list(classes)
    return out


def final_forest(pools, results, task, classes, unit, block_minutes, fp: ForestParams, min_score=0.0):
    """The forest for new data: all labelled blocks, windows assigned to the full clustering of each type
    (``results``: {detector: discover result}). Returns (forest dict, importance Series)."""
    assigned, ids = {}, {}
    for d, pool in pools.items():
        if d not in results:
            continue
        lab = pool[_labelled(pool, task, classes, unit)].reset_index(drop=True)
        cl, ids[d] = _assign(lab, results[d], task, block_minutes)
        assigned[d] = pd.DataFrame({"w_start": lab["w_start"].to_numpy(), "unit": lab[unit].astype(str).to_numpy(),
                                    "true": lab[task.column].astype(str).to_numpy(), "cluster": cl})
    names = feature_names(ids)
    sh = block_shares(assigned, block_minutes)
    truth = block_truth(assigned, block_minutes).reindex(sh.index)
    model = fit_forest(feature_matrix(sh, names), truth.to_numpy(), fp, n_jobs=-1)
    fd = forest_to_dict(model, names, min_score)
    return fd, pd.Series(fd["importance"], index=names).sort_values(ascending=False)
