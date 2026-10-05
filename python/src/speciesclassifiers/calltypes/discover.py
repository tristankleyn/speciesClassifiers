"""Discover call types: cluster labelled and unlabelled windows, describe the clusters, and test how
reliably clusters predict binary labels (e.g. species A vs B) on days they weren't built from.

Clustering ("hybrid"): features are standardised and weighted by their random-forest importance for
telling the known call types apart; windows are grouped by cosine distance (agglomerative, average
linkage, or HDBSCAN); then a Euclidean check moves windows that are far from every other member of
their cluster (from a different time block) to unassigned (-1).
"""

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .evaluate import EXCLUDE_LABELS_DEFAULT, apply_feature_spec, fit_feature_spec


@dataclass
class DiscoverParams:
    """Clustering and cluster-summary settings.

    clusterer          "agglomerative" (deterministic; every window grouped) or "hdbscan" (density based)
    cosine_threshold   agglomerative: max average cosine distance to merge groups (lower = more clusters)
    min_cluster_size   smaller groups are unassigned (-1)
    min_samples        HDBSCAN neighbourhood density
    hdbscan_selection  HDBSCAN "leaf" (finer clusters) or "eom" (broader)
    euclid_outlier_q   a window is unassigned if its Euclidean distance to the nearest member of its cluster
                       (from another time block) exceeds this quantile of the cluster's own such distances
    block_min          time-block length (min) for that check
    max_windows        subsample unlabelled windows above this (the distance matrix is n x n)
    new_p_value, new_min_days, new_max_day_share, new_min_n
                       a cluster is a candidate new call type if it holds significantly fewer known call
                       types than expected (one-sided binomial test), spans >= new_min_days days, no day
                       holds more than new_max_day_share of it, and it has >= new_min_n windows
    examples_per_cluster most typical windows per cluster listed for listening
    exclude_features   features containing any of these are not used
    random_state       random seed
    """

    clusterer: str = "agglomerative"
    cosine_threshold: float = 0.7
    min_cluster_size: int = 15
    min_samples: int = 5
    hdbscan_selection: str = "leaf"
    euclid_outlier_q: float = 0.95
    block_min: int = 10
    max_windows: int = 8000
    new_p_value: float = 0.01
    new_min_days: int = 2
    new_max_day_share: float = 0.8
    new_min_n: int = 25
    examples_per_cluster: int = 10
    exclude_features: list = field(default_factory=lambda: ["amplitude", "snr", "signal", "noise", "mean_width"])
    random_state: int = 42

    def to_dict(self):
        return asdict(self)


@dataclass
class LabelTask:
    """A binary label to judge per cluster, e.g. LabelTask("species", "species", "KW", "Humpback")."""

    name: str
    column: str
    positive: str
    negative: str


@dataclass
class ReliabilityParams:
    """How cluster label verdicts are made.

    cap_per_day       each day contributes at most this many windows per label (None = no cap)
    prior_strength    shrink cluster proportions towards the overall rate by this many pseudo-windows
    ci                credible interval width
    min_label_days    a verdict needs its label from at least this many days
    decision_windows  periods over which held-out window predictions are combined (majority vote)
    """

    cap_per_day: int = 50
    prior_strength: float = 10
    ci: float = 0.90
    min_label_days: int = 2
    decision_windows: tuple = ("1min", "2min", "5min", "10min", "20min")

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------------ pool -------------
def discovery_pool(windows, annotated_days, label_col="label", exclude_labels=EXCLUDE_LABELS_DEFAULT,
                   unlabelled_values=("untyped",), include_none=False, include_unannotated_days=True):
    """Windows to cluster: ``label`` = known call type, or "unlabelled".

    On ``annotated_days`` (fully reviewed): typed calls keep their label; ``unlabelled_values`` (e.g.
    calls of no particular type) become "unlabelled"; windows with no call ("none") are known not to
    contain calls and are left out unless ``include_none``. Windows from other days are "unlabelled"
    if ``include_unannotated_days``.
    """
    w = windows.copy()
    lab = w[label_col].astype(str)
    on = w["day"].isin(list(annotated_days)).to_numpy()
    typed = on & ~lab.isin(list(exclude_labels)).to_numpy()
    unl = on & lab.isin(list(unlabelled_values)).to_numpy()
    if include_none:
        unl |= on & (lab == "none").to_numpy()
    if include_unannotated_days:
        unl |= ~on
    w["label"] = np.where(typed, lab, "unlabelled")
    w["annotated_day"] = on
    return w[typed | unl].reset_index(drop=True)


def _blocks(w, block_min):
    b = (np.floor(w["w_start"].to_numpy(float) / (60 * block_min)) * 60 * block_min).astype(np.int64)
    return (w["day"].astype(str) + "_" + pd.Series(b).astype(str)).to_numpy()


# ------------------------------------------------------------------------ clustering -------
def hybrid_cluster(X, w, params: DiscoverParams):
    """Importance-weighted cosine clustering + Euclidean outlier check.

    Returns (clusters, state): state holds the scaler (mean, scale), feature weights, per-cluster
    Euclidean thresholds, standardised features, weighted unit vectors and the cosine distances.
    """
    from sklearn.metrics import pairwise_distances

    Xa = X.to_numpy(float)
    mean, scale = Xa.mean(axis=0), Xa.std(axis=0)
    scale[scale == 0] = 1.0
    Xs = (Xa - mean) / scale
    lab = (w["label"] != "unlabelled").to_numpy()
    if len(np.unique(w.loc[lab, "label"])) >= 2:            # weights learned from known call types
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(300, class_weight="balanced", min_samples_leaf=2, n_jobs=-1,
                                    random_state=params.random_state).fit(Xs[lab], w.loc[lab, "label"])
        wts = np.sqrt(rf.feature_importances_ / rf.feature_importances_.mean())
        top = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(f"  feature weights from {lab.sum()} labelled windows; top: {', '.join(top.index[:5])}")
    else:
        wts = np.ones(Xs.shape[1])
        print("  fewer than 2 known call types -> unweighted cosine")
    Z = Xs * wts
    Dc = pairwise_distances(Z, metric="cosine").astype(np.float64)
    np.fill_diagonal(Dc, 0)
    if params.clusterer == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        cl = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                     distance_threshold=params.cosine_threshold).fit_predict(Dc)
        sizes = pd.Series(cl).value_counts()
        cl = np.where(np.isin(cl, sizes[sizes < params.min_cluster_size].index), -1, cl)
        order = pd.Series(cl[cl >= 0]).value_counts().index          # renumber 0..k-1, largest first
        remap = {old: new for new, old in enumerate(order)}
        cl = np.array([remap.get(v, -1) for v in cl])
        print(f"  agglomerative (cosine threshold {params.cosine_threshold}): "
              f"{len(order)} clusters >= {params.min_cluster_size} windows")
    else:
        from sklearn.cluster import HDBSCAN
        cl = HDBSCAN(min_cluster_size=params.min_cluster_size, min_samples=params.min_samples,
                     metric="precomputed", cluster_selection_method=params.hdbscan_selection).fit_predict(Dc)
    blocks = _blocks(w, params.block_min)
    n_out, euclid_thr = 0, {}
    for c in np.unique(cl[cl >= 0]):
        idx = np.where(cl == c)[0]
        De = pairwise_distances(Xs[idx])
        De[blocks[idx][:, None] == blocks[idx][None, :]] = np.inf
        d = De.min(axis=1)
        fin = np.isfinite(d)
        if fin.sum() < 5:
            euclid_thr[int(c)] = float("inf")
            continue
        thr = np.quantile(d[fin], params.euclid_outlier_q)
        euclid_thr[int(c)] = float(thr)
        out = idx[fin & (d > thr)]
        cl[out] = -1
        n_out += len(out)
    print(f"  Euclidean check moved {n_out} windows to unassigned (-1)")
    Zu = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    return cl, {"mean": mean, "scale": scale, "weights": wts, "euclid_thr": euclid_thr, "Xs": Xs, "Zu": Zu,
                "Dc": Dc}


# ------------------------------------------------------------------------ summaries --------
def summarise_clusters(w, cl, space, params: DiscoverParams, tasks=()):
    """One row per cluster (-1 = unassigned) plus the most typical windows of each cluster."""
    from scipy.stats import binom

    w = w.assign(cluster=cl)
    overall = (w["label"] != "unlabelled").mean()
    rows, examples = [], []
    for c, g in w.groupby("cluster"):
        lab = g.loc[g["label"] != "unlabelled", "label"].value_counts()
        r = {"cluster": c, "n": len(g), "n_labelled": int(lab.sum()), "frac_labelled": lab.sum() / len(g),
             "top_call_type": lab.index[0] if len(lab) else "",
             "top_share_of_labelled": lab.iloc[0] / lab.sum() if len(lab) else np.nan,
             "call_types": "; ".join(f"{k}:{v}" for k, v in lab.head(5).items()),
             "n_days": g["day"].nunique(),
             "days": "; ".join(f"{k}:{v}" for k, v in g["day"].value_counts().items()),
             "median_mean_freq_hz": g["mean_freq_hz_mean"].median(),
             "median_lc_duration_s": g["lc_duration_s"].median(),
             "median_n_contours": g["n_contours"].median()}
        for t in tasks:                                  # make-up of each label column
            if t.column in g:
                vc = g[t.column].value_counts(normalize=True)
                for v in sorted(w[t.column].dropna().unique()):
                    r[f"frac_{t.column}_{v}"] = float(vc.get(v, 0.0))
                r[f"{t.column}_mix"] = "; ".join(f"{k}:{v:.0%}" for k, v in vc.items())
        if "label_source" in g:
            r["frac_from_annotated_source"] = (g["label_source"] != "unannotated").mean()
        r["expected_labelled"] = overall * len(g)
        r["p_underlabelled"] = binom.cdf(int(lab.sum()), len(g), overall)
        r["largest_day_share"] = g["day"].value_counts(normalize=True).iloc[0]
        r["candidate_new"] = bool(c != -1 and len(g) >= params.new_min_n and r["p_underlabelled"] < params.new_p_value
                                  and r["n_days"] >= params.new_min_days
                                  and r["largest_day_share"] <= params.new_max_day_share)
        rows.append(r)
        if c != -1:                                      # most typical = closest to the centre, no overlaps
            idx = g.index.to_numpy()
            d = np.linalg.norm(space[idx] - space[idx].mean(axis=0), axis=1)
            picked = []
            for i in idx[np.argsort(d)]:
                if all(not (g.at[i, "w_start"] < g.at[p, "w_end"] and g.at[p, "w_start"] < g.at[i, "w_end"])
                       for p in picked):
                    picked.append(i)
                if len(picked) >= params.examples_per_cluster:
                    break
            cols = ["cluster", "day", "w_start", "w_end", "label", "label_source", "n_contours",
                    *[t.column for t in tasks]]
            ex = g.loc[picked]
            examples.append(ex[[c for c in dict.fromkeys(cols) if c in ex.columns]].assign(rank=range(1, len(ex) + 1)))
    summ = pd.DataFrame(rows).sort_values(["candidate_new", "n"], ascending=[False, False])
    return summ, (pd.concat(examples) if examples else pd.DataFrame()), w


# ------------------------------------------------------------------------ label verdicts ---
def _capped_counts(sub, col, pos, neg, cap):
    ct = sub.groupby(["day", col]).size()
    if cap:
        ct = ct.clip(upper=cap)
    ct = ct.groupby(level=1).sum()
    return float(ct.get(pos, 0)), float(ct.get(neg, 0))


def task_stats(w, task: LabelTask, rp: ReliabilityParams):
    """Per-cluster verdict for one binary label task.

    p = share of ``positive`` among the cluster's labelled windows, shrunk towards the overall (base)
    rate by ``prior_strength`` pseudo-windows, with a ``ci`` credible interval. Verdict = positive if
    the interval lies above the base rate, negative if below, else "uncertain"; the winning label must
    come from at least ``min_label_days`` days. (Comparing with the base rate corrects for imbalance.)
    """
    from scipy.stats import beta as _beta

    name, col, pos, neg = task.name, task.column, task.positive, task.negative
    lab = w[(w["cluster"] >= 0) & w[col].isin([pos, neg])]
    k0, q = rp.prior_strength, (1 - rp.ci) / 2
    X, M = _capped_counts(lab, col, pos, neg, rp.cap_per_day)
    base = X / (X + M) if X + M else np.nan
    rows = []
    for c in sorted(w.loc[w["cluster"] >= 0, "cluster"].unique()):
        g = lab[lab["cluster"] == c]
        x, m = _capped_counts(g, col, pos, neg, rp.cap_per_day) if len(g) else (0.0, 0.0)
        a, b = x + k0 * base, m + k0 * (1 - base)
        dp = g.loc[g[col] == pos, "day"].nunique()
        dn = g.loc[g[col] == neg, "day"].nunique()
        p = a / (a + b)
        lo, hi = _beta.ppf(q, a, b), _beta.ppf(1 - q, a, b)
        verdict = (pos if lo > base and dp >= rp.min_label_days else
                   neg if hi < base and dn >= rp.min_label_days else "uncertain")
        rows.append({"cluster": c, f"{name}_n_{pos}": int((g[col] == pos).sum()),
                     f"{name}_n_{neg}": int((g[col] == neg).sum()),
                     f"{name}_p_{pos}_raw": x / (x + m) if x + m else np.nan,
                     f"{name}_p_{pos}": p, f"{name}_p_{pos}_lo": lo, f"{name}_p_{pos}_hi": hi,
                     f"{name}_base_{pos}": base, f"{name}_enrichment": p / base if base else np.nan,
                     f"{name}_days_{pos}": dp, f"{name}_days_{neg}": dn,
                     f"{name}_years_{pos}": g.loc[g[col] == pos, "day"].str[:4].nunique(),
                     f"{name}_years_{neg}": g.loc[g[col] == neg, "day"].str[:4].nunique(),
                     f"{name}_verdict": verdict})
    return pd.DataFrame(rows).set_index("cluster")


def cluster_label_reliability(w, tasks, rp: ReliabilityParams):
    """All tasks + each cluster's share of windows per year (to expose year/recorder confounding)."""
    w = w.assign(day=w["day"].astype(str))
    out = [task_stats(w, t, rp) for t in tasks]
    yr = pd.crosstab(w["cluster"], w["day"].str[:4], normalize="index").add_prefix("frac_year_")
    return pd.concat(out + [yr], axis=1)


def loo_label_prediction(w, tasks, rp: ReliabilityParams):
    """Leave-one-day-out: clusters stay fixed; each cluster's verdict is recomputed without the held-out
    day and used to predict that day's labelled windows (as a frozen library would predict new data).
    Unassigned windows and "uncertain" clusters count as no prediction.

    Returns (per day, overall, held-out window predictions per task).
    """
    w = w.assign(day=w["day"].astype(str))
    per_day, overall, window_preds = [], [], {}
    for task in tasks:
        name, col, pos, neg = task.name, task.column, task.positive, task.negative
        labelled = w[w[col].isin([pos, neg])]
        preds = []
        for d in sorted(labelled["day"].unique()):
            train = w[w["day"] != d]
            if train[col].isin([pos]).sum() == 0 or train[col].isin([neg]).sum() == 0:
                continue
            st = task_stats(train, task, rp)[f"{name}_verdict"]
            test = labelled[labelled["day"] == d]
            pred = test["cluster"].map(st).fillna("uncertain").to_numpy()
            pred[test["cluster"].to_numpy() < 0] = "uncertain"
            covered = pred != "uncertain"
            per_day.append({"task": name, "day": d, "true_labels": "; ".join(
                f"{k}:{v}" for k, v in test[col].value_counts().items()),
                "n": len(test), "coverage": covered.mean(),
                "accuracy_when_predicted": (pred[covered] == test[col].to_numpy()[covered]).mean()
                if covered.any() else np.nan})
            preds.append(pd.DataFrame({"true": test[col].to_numpy(), "pred": pred, "day": d,
                                       "w_start": test["w_start"].to_numpy()}))
        if not preds:
            continue
        P = pd.concat(preds)
        window_preds[name] = P
        C = P[P["pred"] != "uncertain"]
        rec = {lab: (C.loc[C["true"] == lab, "pred"] == lab).mean() for lab in (pos, neg)}
        cov = {lab: (P.loc[P["true"] == lab, "pred"] != "uncertain").mean() for lab in (pos, neg)}
        overall.append({"task": name, "n_windows": len(P), "coverage": (P["pred"] != "uncertain").mean(),
                        f"recall_{pos}": rec[pos], f"recall_{neg}": rec[neg],
                        f"coverage_{pos}": cov[pos], f"coverage_{neg}": cov[neg],
                        "balanced_accuracy_when_predicted": (np.mean([v for v in rec.values() if np.isfinite(v)])
                                                             if any(np.isfinite(v) for v in rec.values()) else np.nan),
                        "chance": 0.5})
    return pd.DataFrame(per_day), pd.DataFrame(overall), window_preds


def _floor(seconds, period):
    return pd.to_datetime(np.asarray(seconds, float), unit="s", utc=True).floor(period)


def decision_window_performance(window_preds, tasks, rp: ReliabilityParams):
    """Combine held-out window predictions over longer periods (e.g. 5 min): each block's prediction is
    the strict majority of its non-uncertain window predictions; its truth the majority label of its
    windows. One row per task x period x class, plus an "overall" row (with chance-corrected skill)."""
    rows = []
    for task in tasks:
        name, pos, neg = task.name, task.positive, task.negative
        P = window_preds.get(name)
        if P is None or not len(P):
            continue
        for blk in ["window", *rp.decision_windows]:
            if blk == "window":
                B = P[["true", "pred"]]
            else:
                recs = []
                for _, x in P.groupby([P["day"], _floor(P["w_start"], blk)]):
                    v = x["pred"][x["pred"] != "uncertain"].value_counts()
                    pred = v.index[0] if len(v) and v.iloc[0] > v.sum() / 2 else "uncertain"
                    recs.append((x["true"].value_counts().index[0], pred))
                B = pd.DataFrame(recs, columns=["true", "pred"])
            effs, recalls, f1s = [], [], []
            for lab in (pos, neg):
                t = B[B["true"] == lab]
                n, cov = len(t), (t["pred"] != "uncertain").mean() if len(t) else np.nan
                tp = (t["pred"] == lab).sum()
                called = (B["pred"] == lab).sum()
                acc = tp / (t["pred"] != "uncertain").sum() if n and (t["pred"] != "uncertain").any() else np.nan
                rec = tp / n if n else np.nan
                prec = tp / called if called else np.nan
                f1 = 2 * prec * rec / (prec + rec) if prec and rec and np.isfinite(prec) and np.isfinite(rec) else np.nan
                rows.append({"task": name, "decision_window": blk, "class": lab, "n_blocks": n, "coverage": cov,
                             "recall": rec, "precision": prec, "accuracy_when_predicted": acc, "f1": f1})
                f1s.append(f1)
                if n:
                    effs.append(cov * (acc if np.isfinite(acc) else 0.5) + (1 - cov) * 0.5)
                    recalls.append(rec)
            covered = B[B["pred"] != "uncertain"]
            fin = [f for f in f1s if np.isfinite(f)]
            rows.append({"task": name, "decision_window": blk, "class": "overall", "n_blocks": len(B),
                         "coverage": (B["pred"] != "uncertain").mean(),
                         "recall": np.mean(recalls) if recalls else np.nan, "precision": np.nan,
                         "accuracy_when_predicted": (covered["pred"] == covered["true"]).mean() if len(covered) else np.nan,
                         "f1": np.mean(fin) if fin else np.nan,
                         "skill": (np.mean(effs) - 0.5) / 0.5 if effs else np.nan})
    return pd.DataFrame(rows)


def scorecard(w, cl, Dc, loo_all, reliability, tasks):
    """Headline numbers for comparing discover runs."""
    from sklearn.metrics import adjusted_mutual_info_score, silhouette_score

    sc = {"n_windows": len(w), "n_days": w["day"].nunique(), "n_clusters": len(set(cl) - {-1}),
          "frac_unassigned": float((cl == -1).mean())}
    a = np.where(cl >= 0)[0]
    if len(set(cl[a])) >= 2:
        idx = np.random.default_rng(0).choice(a, min(len(a), 3000), replace=False)
        sc["silhouette"] = silhouette_score(Dc[np.ix_(idx, idx)], cl[idx], metric="precomputed")
    A = w.iloc[a]
    for t in tasks:
        L = A[A[t.column].isin([t.positive, t.negative])]
        sc[f"{t.name}_ami"] = adjusted_mutual_info_score(L[t.column], L["cluster"]) if len(L) > 1 else np.nan
        v = reliability.loc[reliability.index >= 0, f"{t.name}_verdict"]
        sc[f"{t.name}_clusters_with_verdict"] = int((v.notna() & (v != "uncertain")).sum())
        r = loo_all[loo_all["task"] == t.name] if len(loo_all) else pd.DataFrame()
        if len(r):
            sc[f"{t.name}_loo_coverage"] = float(r["coverage"].iloc[0])
            sc[f"{t.name}_loo_bal_acc"] = float(r["balanced_accuracy_when_predicted"].iloc[0])
            effs = []
            for lab in (t.positive, t.negative):
                cv, rc = float(r[f"coverage_{lab}"].iloc[0]), float(r[f"recall_{lab}"].iloc[0])
                if np.isfinite(cv):
                    effs.append(cv * (rc if np.isfinite(rc) else 0.5) + (1 - cv) * 0.5)
            sc[f"{t.name}_skill"] = (np.mean(effs) - 0.5) / 0.5 if effs else np.nan
    L = A[A["label"] != "unlabelled"]
    sc["calltype_ami"] = adjusted_mutual_info_score(L["label"], L["cluster"]) if len(L) > 1 else np.nan
    return sc


# ------------------------------------------------------------------------ main -------------
def discover(pool, params: DiscoverParams = None, tasks=(), rp: ReliabilityParams = None, loo=True):
    """Cluster a discovery pool (see ``discovery_pool``) and describe the clusters.

    Returns a dict: ``windows`` (with ``cluster``), ``summary`` (per cluster, incl. verdicts),
    ``reliability``, ``examples``, ``loo_by_day``, ``loo_summary``, ``decision_windows``, ``scorecard``,
    ``calltype_by_cluster``, and ``state`` (feature spec, scaler, weights, thresholds: what the library
    needs).
    """
    params = params or DiscoverParams()
    rp = rp or ReliabilityParams()
    w = pool
    if len(w) > params.max_windows:
        known = np.where(w["label"] != "unlabelled")[0]
        unl = np.where(w["label"] == "unlabelled")[0]
        pick = np.random.default_rng(params.random_state).choice(
            unl, max(params.max_windows - len(known), 0), replace=False)
        w = w.iloc[np.sort(np.r_[known, pick])].reset_index(drop=True)
        print(f"  subsampled to {len(w)} windows (max_windows)")
    spec = fit_feature_spec(w, params.exclude_features)
    X = apply_feature_spec(w, spec)
    cl, state = hybrid_cluster(X, w, params)
    summ, examples, w = summarise_clusters(w, cl, state["Zu"], params, tasks)
    rel = cluster_label_reliability(w, tasks, rp) if tasks else pd.DataFrame(index=sorted(set(cl) - {-1}))
    summ = summ.merge(rel, left_on="cluster", right_index=True, how="left")
    loo_day, loo_all, wp = loo_label_prediction(w, tasks, rp) if (loo and tasks) else (pd.DataFrame(), pd.DataFrame(), {})
    dwp = decision_window_performance(wp, tasks, rp) if wp else pd.DataFrame()
    sc = scorecard(w, cl, state["Dc"], loo_all, rel, tasks)
    if len(dwp):
        for _, r in dwp[(dwp["class"] == "overall") & (dwp["decision_window"] != "window")].iterrows():
            sc[f"{r['task']}_skill_{r['decision_window']}"] = r["skill"]
    lab = w[w["label"] != "unlabelled"]
    state = {**state, "spec": spec, "X": X}
    return {"windows": w, "summary": summ, "reliability": rel, "examples": examples, "loo_by_day": loo_day,
            "loo_summary": loo_all, "decision_windows": dwp, "scorecard": sc,
            "calltype_by_cluster": pd.crosstab(lab["label"], lab["cluster"]), "state": state}
