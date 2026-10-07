"""Multi-class label tasks: for each cluster, a probability for every value of a label (e.g. every species),
tested by leaving out one unit (event or day) at a time, and combined over clock-aligned blocks.

Cluster probabilities: each cluster's labelled windows are counted per class (capped per unit, so one
long event can't dominate), optionally rebalanced so every class counts equally overall, and shrunk
towards the overall class rate by ``prior_strength`` pseudo-windows (Dirichlet). A cluster's verdict is
the most probable class whose credible interval lies above its overall rate, coming from at least
``min_label_days`` units; otherwise "uncertain". With two classes this is the same rule as the binary
tasks.

Block decisions: the class probabilities of a block's windows (from their clusters) are averaged; the
block's score is p1 x (p1 - p2); the top class is the decision if the score reaches ``min_score``, else
"uncertain".
"""

import numpy as np
import pandas as pd

NON_LABELS = {"none", "partial", "mixed", "untyped", "unknown", "unlabelled", "n/a", "nan", "", "AMBIGUOUS"}


def task_classes(w, task):
    """The classes of a multi-class task: as listed, or every value in the column except non-labels and
    ``task.exclude``."""
    if task.classes:
        return [str(c) for c in task.classes]
    vals = w[task.column].dropna().astype(str)
    skip = NON_LABELS | {str(x) for x in (task.exclude or ())}
    return sorted(v for v in vals.unique() if v not in skip)


def unit_col(rp):
    return "day" if rp.unit in (None, "day") else rp.unit


def _capped(sub, col, classes, unit, cap):
    ct = sub.groupby([unit, col]).size()
    if cap:
        ct = ct.clip(upper=cap)
    ct = ct.groupby(level=1).sum()
    return np.array([float(ct.get(c, 0)) for c in classes])


def class_stats(w, task, rp, classes=None):
    """Per-cluster class probabilities and verdicts for one multi-class task (index: cluster)."""
    from scipy.stats import beta as _beta

    classes = classes or task_classes(w, task)
    name, col, unit = task.name, task.column, unit_col(rp)
    lab = w[(w["cluster"] >= 0) & w[col].astype(str).isin(classes)].assign(**{col: lambda d: d[col].astype(str)})
    total = _capped(lab, col, classes, unit, rp.cap_per_day)
    k = len(classes)
    if rp.balanced and (total > 0).all():
        weight = total.mean() / total                     # every class counts equally overall
        base = np.full(k, 1.0 / k)
    else:
        weight = np.ones(k)
        base = total / total.sum() if total.sum() else np.full(k, 1.0 / k)
    q = (1 - rp.ci) / 2
    rows = []
    for c in sorted(w.loc[w["cluster"] >= 0, "cluster"].unique()):
        g = lab[lab["cluster"] == c]
        x = (_capped(g, col, classes, unit, rp.cap_per_day) if len(g) else np.zeros(k)) * weight
        a = x + rp.prior_strength * base
        A = a.sum()
        p = a / A
        lo = _beta.ppf(q, a, A - a)
        units = np.array([g.loc[g[col] == cl, unit].nunique() for cl in classes])
        ok = (lo > base) & (units >= rp.min_label_days)
        verdict = classes[int(np.argmax(np.where(ok, p, -1)))] if ok.any() else "uncertain"
        r = {"cluster": c, f"{name}_verdict": verdict, f"{name}_n_labelled": int(len(g))}
        for i, cl in enumerate(classes):
            r[f"{name}_p_{cl}"] = p[i]
            r[f"{name}_lo_{cl}"] = lo[i]
            r[f"{name}_n_{cl}"] = int((g[col] == cl).sum())
            r[f"{name}_units_{cl}"] = int(units[i])
        rows.append(r)
    out = pd.DataFrame(rows).set_index("cluster") if rows else pd.DataFrame()
    out.attrs.update(classes=classes, base=dict(zip(classes, base)))
    return out


def decide(P, classes, min_score=0.0):
    """Rows of class probabilities -> (decision, score). Score = p1 x (p1 - p2); NaN rows -> "no_data"."""
    P = np.asarray(P, float)
    srt = np.sort(np.nan_to_num(P, nan=0.0), axis=1)
    p1 = srt[:, -1]
    p2 = srt[:, -2] if P.shape[1] > 1 else np.zeros(len(P))
    score = p1 * (p1 - p2)
    top = np.array(classes, dtype=object)[np.nanargmax(np.nan_to_num(P, nan=-1.0), axis=1)] if len(P) else []
    empty = np.isnan(P).all(axis=1) if len(P) else np.array([], bool)
    dec = np.where(empty, "no_data", np.where(score >= min_score, top, "uncertain"))
    return dec, np.where(empty, np.nan, score)


def loo_multiclass(w, task, rp):
    """Leave one unit (event or day) out: clusters stay fixed; class probabilities are recomputed without
    the unit and given to its labelled windows. Returns held-out window predictions (true class, unit,
    time, cluster and ``p_<class>``; probabilities NaN for unassigned windows)."""
    classes = task_classes(w, task)
    unit = unit_col(rp)
    col = task.column
    labelled = w[w[col].astype(str).isin(classes)]
    preds = []
    for u in sorted(labelled[unit].astype(str).unique()):
        train = w[w[unit].astype(str) != u]
        if not train[col].astype(str).isin(classes).any():
            continue
        st = class_stats(train, task, rp, classes)
        test = labelled[labelled[unit].astype(str) == u]
        P = np.full((len(test), len(classes)), np.nan)
        if len(st):
            known = test["cluster"].isin(st.index).to_numpy() & (test["cluster"].to_numpy() >= 0)
            cols = [f"{task.name}_p_{c}" for c in classes]
            P[known] = st.loc[test["cluster"].to_numpy()[known], cols].to_numpy()
        df = pd.DataFrame(P, columns=[f"p_{c}" for c in classes])
        df.insert(0, "true", test[col].astype(str).to_numpy())
        df.insert(1, "unit", u)
        df.insert(2, "day", test["day"].astype(str).to_numpy())
        df.insert(3, "w_start", test["w_start"].to_numpy())
        df.insert(4, "cluster", test["cluster"].to_numpy())
        preds.append(df)
    out = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    out.attrs["classes"] = classes
    return out


def aggregate(window_preds, classes, level="block", block_minutes=10):
    """Held-out window predictions -> rows per window, block (clock-aligned, within a unit) or unit:
    mean class probabilities of the windows with probabilities, and the true (majority) class."""
    P = window_preds
    pc = [f"p_{c}" for c in classes]
    if level == "window":
        return P[["true", "unit"] + pc].reset_index(drop=True)
    keys = ["unit"] if level == "unit" else ["unit", (np.floor(P["w_start"] / (60 * block_minutes))).astype(np.int64)]
    g = P.groupby(keys, sort=False)
    out = g[pc].mean()
    out["true"] = g["true"].agg(lambda s: s.value_counts().index[0])
    out["n_windows"] = g.size()
    out["n_with_probs"] = g[pc[0]].count()
    return out.reset_index(drop=True)


def performance(rows, classes, min_score=0.0):
    """Accuracy at one threshold: coverage (share decided), accuracy when decided, balanced accuracy,
    per-class recall and the confusion matrix (rows true; columns include "uncertain")."""
    dec, score = decide(rows[[f"p_{c}" for c in classes]].to_numpy(), classes, min_score)
    true = rows["true"].to_numpy()
    decided = ~np.isin(dec, ["uncertain", "no_data"])
    rec = {c: float(np.mean(dec[true == c] == c)) if (true == c).any() else np.nan for c in classes}
    acc_c = {c: float(np.mean(dec[(true == c) & decided] == c)) if ((true == c) & decided).any() else np.nan
             for c in classes}
    conf = pd.crosstab(pd.Series(true, name="true"), pd.Series(dec, name="decision"))
    return {"n": len(rows), "coverage": float(decided.mean()) if len(rows) else np.nan,
            "accuracy": float(np.mean(dec[decided] == true[decided])) if decided.any() else np.nan,
            "balanced_accuracy": float(np.nanmean(list(acc_c.values()))) if decided.any() else np.nan,
            "recall": rec, "confusion": conf, "decisions": dec, "scores": score}


def threshold_curve(rows, classes, thresholds=None):
    thresholds = np.linspace(0, 0.5, 26) if thresholds is None else thresholds
    return pd.DataFrame([{"threshold": t, **{k: v for k, v in performance(rows, classes, t).items()
                                              if k in ("coverage", "accuracy", "balanced_accuracy")}}
                         for t in thresholds])
