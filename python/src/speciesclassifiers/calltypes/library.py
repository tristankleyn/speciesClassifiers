"""Frozen call-type library: everything needed to classify new contours the same way as a
``discover`` run, stored as one JSON file.

New windows are assigned to the nearest cluster (importance-weighted cosine distance to the
cluster centroid) unless that distance exceeds the clustering threshold or the window is further
(Euclidean, standardised) from the cluster's nearest member than the cluster's outlier threshold;
then it is unassigned (-1). Each window takes its cluster's verdict per label task; windows are
combined into clock-aligned decision blocks by strict majority vote; optional user-defined rules then
override block decisions (e.g. mark blocks with almost no frequency variation as noise).

``standalone.py`` reproduces ``classify_contours`` with numpy only, for use outside this package.
"""

import base64
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .contours import ContourParams, prepare_contours
from .evaluate import apply_feature_spec
from .windows import AGG_PARAMS, WindowParams, make_windows, pair_intervals, window_features

LIBRARY_FORMAT = "speciesclassifiers.calltypes.library/1"
RULE_OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
            "==": np.equal, "!=": np.not_equal}
BLOCK_VALUES = ("n_windows", "n_assigned", "n_contours", "block_meanfreq_range_hz",
                "block_median_abs_slope_hz_s", "block_median_freq_hz", "block_median_contour_s")
NS = 1_000_000_000


def _b64_f32(a):
    a = np.ascontiguousarray(a, dtype="<f4")
    return {"shape": list(a.shape), "dtype": "<f4", "b64": base64.b64encode(a.tobytes()).decode()}


def _from_b64(d):
    return np.frombuffer(base64.b64decode(d["b64"]), dtype=d["dtype"]).reshape(d["shape"]).astype(float)


def _plain(x):
    """numpy scalars -> Python, for JSON."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def check_rules(rules):
    """Validate decision rules: a list of {"if": <block value>, "op": <, <=, >, >=, ==, !=,
    "value": number, "then": <decision>}. Returns them as a list of dicts."""
    out = []
    for r in rules or []:
        r = dict(r)
        if r.get("if") not in BLOCK_VALUES:
            raise ValueError(f"rule 'if' must be one of {BLOCK_VALUES}, got {r.get('if')!r}")
        if r.get("op") not in RULE_OPS:
            raise ValueError(f"rule 'op' must be one of {list(RULE_OPS)}, got {r.get('op')!r}")
        if not isinstance(r.get("value"), (int, float)) or not isinstance(r.get("then"), str):
            raise ValueError(f"rule needs a numeric 'value' and a text 'then': {r}")
        out.append({"if": r["if"], "op": r["op"], "value": float(r["value"]), "then": r["then"]})
    return out


# ------------------------------------------------------------------------ build ------------
def build_library(result, contour_params: ContourParams, window_params: WindowParams, tasks,
                  block_minutes=10, rules=None, requires=None, detector=None, name=None):
    """Freeze a ``discover`` result into a library (dict; save with ``save_library``).

    result          output of ``discover`` (clusters, verdicts, feature transform, scaler, weights)
    contour_params, window_params   the settings the discover windows were made with
    tasks           the LabelTasks whose verdicts the library reports
    block_minutes   length of the clock-aligned decision blocks
    rules           optional decision rules applied to each block after voting (see ``check_rules``),
                    in order; the first that matches sets every task's decision to its "then" value
    requires        optional {task: (other task, value)}: report the task only in blocks where the
                    other task's decision is that value (e.g. {"ecotype": ("species", "KW")}), else "n/a"
    detector        optional {"sample_rate", "fft", "hop"} of the whistle detector, for converting peak
                    bins / slice numbers to Hz / s with ``standalone.contour_from_bins``
    """
    st, w, rel = result["state"], result["windows"], result["reliability"]
    spec, X = st["spec"], st["X"]
    Xa = X.to_numpy(float)
    mean, scale = np.asarray(st["mean"], float), np.asarray(st["scale"], float)
    Xs = (Xa - mean) / scale
    wts = np.asarray(st["weights"], float)
    Z = Xs * wts
    Zu = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    if contour_params.snr_top_frac:
        print("Note: snr_top_frac is relative to each batch of contours classified; "
              "consider a fixed min_snr_db for the library.")
    cl = w["cluster"].to_numpy()
    clusters = []
    for c in sorted(set(cl[cl >= 0])):
        idx = np.where(cl == c)[0]
        thr = st["euclid_thr"].get(int(c), float("inf"))
        entry = {"id": int(c), "n": int(len(idx)),
                 # mean of member unit vectors: 1 - u . centroid = average cosine distance to the members
                 "centroid": Zu[idx].mean(axis=0).tolist(),
                 "euclid_thr": None if not np.isfinite(thr) else float(thr),
                 "members": _b64_f32(Xs[idx]), "verdicts": {}}
        for t in tasks:
            r = rel.loc[c]
            entry["verdicts"][t.name] = {
                "verdict": str(r[f"{t.name}_verdict"]), "p_positive": float(r[f"{t.name}_p_{t.positive}"]),
                "n_positive": int(r[f"{t.name}_n_{t.positive}"]), "n_negative": int(r[f"{t.name}_n_{t.negative}"])}
        clusters.append(entry)
    requires = {k: list(v) for k, v in (requires or {}).items()}
    names = [t.name for t in tasks]
    for k, (other, _) in requires.items():
        if k not in names or other not in names:
            raise ValueError(f"requires refers to unknown task: {k} -> {other}")
    created = datetime.now(timezone.utc)
    lib = {
        "format": LIBRARY_FORMAT,
        "library_version": name or "calltypes-" + created.strftime("%Y%m%d-%H%M%S"),
        "created_utc": created.isoformat(),
        "n_training_windows": int(len(w)), "n_training_days": int(w["day"].nunique()),
        "contour_params": contour_params.to_dict(),
        "window_params": {**window_params.to_dict(), "mode": "sliding"},
        "cosine_threshold": float(result["params"].cosine_threshold),
        "block_minutes": block_minutes,
        "vote_rule": "strict majority of non-uncertain window predictions",
        "tasks": [{"name": t.name, "positive": t.positive, "negative": t.negative} for t in tasks],
        "requires": requires,
        "rules": check_rules(rules),
        "detector": detector,
        "agg_params": list(AGG_PARAMS),
        "feature_spec": {**spec, "mean": mean.tolist(), "scale": scale.tolist()},
        "weights": wts.tolist(),
        "clusters": clusters,
    }
    return _plain(lib)


def save_library(lib, path):
    with open(path, "w") as f:
        json.dump(_plain(lib), f)
    return path


def load_library(path):
    with open(path) as f:
        lib = json.load(f)
    if lib.get("format") != LIBRARY_FORMAT:
        raise ValueError(f"{path} is not a call-type library ({LIBRARY_FORMAT})")
    return lib


def with_rules(lib, rules):
    """A copy of the library with different decision rules (None/[] = no rules)."""
    return {**lib, "rules": check_rules(rules)}


# ------------------------------------------------------------------------ classify ---------
def standardise(windows, lib):
    """Window features -> standardised matrix, with the library's frozen transform."""
    fs = lib["feature_spec"]
    X = apply_feature_spec(windows, fs).to_numpy(float)
    return (X - np.asarray(fs["mean"])) / np.asarray(fs["scale"])


def _members(lib):
    return [_from_b64(c["members"]) for c in lib["clusters"]]


def assign_windows(Xs, lib):
    """Cluster id per window (-1 = unassigned), cosine distance to the nearest centroid, Euclidean
    distance to that cluster's nearest member."""
    if len(Xs) == 0:
        return np.array([], int), np.array([]), np.array([])
    C = np.asarray([c["centroid"] for c in lib["clusters"]], float)
    Z = Xs * np.asarray(lib["weights"], float)
    U = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    cosd = 1.0 - U @ C.T
    best = np.argmin(cosd, axis=1)
    mem = _members(lib)
    ids, dcos, deuc = [], [], []
    for i, b in enumerate(best):
        c = lib["clusters"][b]
        dc = float(cosd[i, b])
        de = float(np.sqrt(((mem[b] - Xs[i]) ** 2).sum(axis=1)).min())
        ok = dc <= lib["cosine_threshold"] and (c["euclid_thr"] is None or de <= c["euclid_thr"])
        ids.append(c["id"] if ok else -1)
        dcos.append(dc)
        deuc.append(de)
    return np.array(ids, int), np.array(dcos), np.array(deuc)


def classify_windows(windows, lib):
    """Add ``cluster``, ``cos_dist``, ``euc_dist`` and ``<task>_pred`` columns to a window table."""
    ids, dcos, deuc = assign_windows(standardise(windows, lib), lib)
    w = windows.assign(cluster=ids, cos_dist=dcos, euc_dist=deuc)
    byid = {c["id"]: c for c in lib["clusters"]}
    for t in lib["tasks"]:
        w[f"{t['name']}_pred"] = [byid[i]["verdicts"][t["name"]]["verdict"] if i >= 0 else "uncertain"
                                  for i in ids]
    return w


def _vote(preds, labels):
    n = {lab: int(sum(p == lab for p in preds)) for lab in labels}
    tot = sum(n.values())
    if tot == 0:
        return "uncertain", None, n
    top = max(labels, key=lambda lab: n[lab])
    return (top if n[top] > tot / 2 else "uncertain"), n[top] / tot, n


def _median(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def apply_rules(block, rules):
    """The first rule matching the block's summary values, or None."""
    for r in rules:
        v = block.get(r["if"])
        if v is not None and np.isfinite(v) and bool(RULE_OPS[r["op"]](v, r["value"])):
            return r
    return None


def block_votes(windows, contours, lib, period=None):
    """Decision blocks (clock-aligned, ``block_minutes`` long) from classified windows.

    Per block: window/contour counts, summary values the rules can use, per task the vote counts,
    the raw majority decision (``raw_<task>``) and the final decision (``<task>``) after rules and
    ``requires``. ``period`` = (start, end) in Unix s also lists blocks with no windows
    ("no_detections"); otherwise only blocks with windows are returned.
    """
    blk = int(round(lib["block_minutes"] * 60 * NS))
    tasks, rules, req = lib["tasks"], lib["rules"], lib.get("requires") or {}
    ws_ns = np.round(windows["w_start"].to_numpy(float) * 1e9).astype(np.int64)
    bstart = (ws_ns // blk) * blk
    # contours in each window, as in window_features
    cs, ce = contours["start"].to_numpy(float), contours["end"].to_numpy(float)
    wsf, wef = windows["w_start"].to_numpy(float), windows["w_end"].to_numpy(float)
    if lib["window_params"]["assign_by"] == "midpoint":
        mid = (cs + ce) / 2
        I, J, _ = pair_intervals(wsf, wef, mid, mid + 1e-9)
    else:
        I, J, _ = pair_intervals(wsf, wef, cs, ce)
    keys = sorted(set(bstart.tolist()))
    if period is not None:
        p0 = (int(round(period[0] * 1e9)) // blk) * blk
        keys = sorted(set(keys) | set(range(p0, int(round(period[1] * 1e9)), blk)))
        keys = [k for k in keys if k < period[1] * 1e9 and k + blk > period[0] * 1e9]
    mf = contours["mean_freq_hz"].to_numpy(float)
    dur = contours["duration_s"].to_numpy(float)
    rows = []
    for b in keys:
        wi = np.where(bstart == b)[0]
        g = windows.iloc[wi]
        cidx = np.unique(J[np.isin(I, wi)])
        r = {"block_start": b / NS, "block_end": (b + blk) / NS, "n_windows": len(wi),
             "n_assigned": int((g["cluster"] >= 0).sum()) if len(wi) else 0, "n_contours": len(cidx),
             "block_meanfreq_range_hz": float(mf[cidx].max() - mf[cidx].min()) if len(cidx) else None,
             "block_median_abs_slope_hz_s": _median(g["mean_abs_slope_hz_s_mean"]) if len(wi) else None,
             "block_median_freq_hz": _median(g["pts_mean_freq_hz"]) if len(wi) else None,
             "block_median_contour_s": float(np.median(dur[cidx])) if len(cidx) else None}
        for t in tasks:
            dec, share, n = _vote(g[f"{t['name']}_pred"].tolist() if len(wi) else [], (t["positive"], t["negative"]))
            r[f"{t['name']}_n_{t['positive']}"] = n[t["positive"]]
            r[f"{t['name']}_n_{t['negative']}"] = n[t["negative"]]
            r[f"{t['name']}_vote_share"] = share
            r[f"raw_{t['name']}"] = dec if len(wi) else "no_detections"
        rule = apply_rules(r, rules) if len(wi) else None
        r["rule"] = None if rule is None else f"{rule['if']} {rule['op']} {rule['value']:g}"
        for t in tasks:
            name = t["name"]
            r[name] = rule["then"] if rule else r[f"raw_{name}"]
        for name, (other, value) in req.items():
            if len(wi) and not rule and r[other] != value:
                r[name] = "n/a"
        cl = sorted(g["cluster"].value_counts().items(), key=lambda kv: (-kv[1], kv[0])) if len(wi) else []
        r["cluster_votes"] = ";".join(f"{int(k)}:{int(v)}" for k, v in cl)
        rows.append(r)
    cols = ["block_start", "block_end", *BLOCK_VALUES[:3], *[t["name"] for t in tasks], "rule",
            *[f"raw_{t['name']}" for t in tasks]]
    df = pd.DataFrame(rows)
    if not len(df):
        return pd.DataFrame(columns=cols)
    return df[cols + [c for c in df.columns if c not in cols]]


def classify_contours(whistles, lib, period=None):
    """Reference path: whistle table (as from ``io.read_whistles``) -> (windows, blocks, contours).

    Applies the library's contour filters, window settings, feature transform, cluster assignment,
    block votes and rules. ``standalone.classify`` gives the same result with numpy only.
    """
    cp = ContourParams(**lib["contour_params"])
    wp = WindowParams(**lib["window_params"])
    contours, points = prepare_contours(whistles, cp)
    if not len(contours):
        empty = pd.DataFrame(columns=["w_start", "w_end", "cluster"])
        return empty, block_votes(empty, pd.DataFrame(columns=["start", "end", "mean_freq_hz", "duration_s"]),
                                  lib, period), contours
    w = window_features(make_windows(contours, wp), contours, points, wp)
    if not len(w):
        return w.assign(cluster=[]), block_votes(w.assign(cluster=[]), contours, lib, period), contours
    w = classify_windows(w, lib)
    return w, block_votes(w, contours, lib, period), contours


def self_check(result, lib):
    """Re-assign the discover windows with the frozen library: share of clustered windows that land in
    their own cluster, that become unassigned, and of unclustered windows that stay unassigned."""
    st = result["state"]
    ids, _, _ = assign_windows(standardise(result["windows"], lib), lib)
    cl = result["windows"]["cluster"].to_numpy()
    m = cl >= 0
    return {"same_cluster": float(np.mean(ids[m] == cl[m])), "became_unassigned": float(np.mean(ids[m] == -1)),
            "unclustered_stay_unassigned": float(np.mean(ids[~m] == -1)) if (~m).any() else np.nan,
            "scaler_matches": bool(np.allclose(standardise(result["windows"], lib), st["Xs"], atol=1e-8))}


def standard_output(windows, lib, event_col=None, classifier_prefix="calltypes"):
    """Classified windows -> the standard classifier output (one classifier per label task), so call-type
    clusters can feed ``transferlearning``. A window's probability of the positive label is its cluster's
    (shrunk) share of that label; unassigned windows are left out. ``event_col``: window column to use as
    ``event_id`` (default: the UTC day)."""
    byid = {c["id"]: c for c in lib["clusters"]}
    w = windows[windows["cluster"] >= 0]
    det = [f"w{t:.3f}" for t in w["w_start"]]
    ev = w[event_col].astype(str).to_numpy() if event_col else \
        pd.to_datetime(w["w_start"], unit="s", utc=True).dt.strftime("%Y%m%d").to_numpy()
    time = pd.to_datetime(w["w_start"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ").to_numpy()
    rows = []
    for t in lib["tasks"]:
        p = np.array([byid[c]["verdicts"][t["name"]]["p_positive"] for c in w["cluster"]], float)
        for cls, prob in ((t["positive"], p), (t["negative"], 1 - p)):
            rows.append(pd.DataFrame({"detection_id": det, "event_id": ev, "time": time,
                                      "classifier": f"{classifier_prefix}-{t['name']}", "voc_type": "whistle",
                                      "class": cls, "probability": prob, "cluster": w["cluster"].to_numpy()}))
    cols = ["detection_id", "event_id", "time", "classifier", "voc_type", "class", "probability", "cluster"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.concat(rows, ignore_index=True).sort_values(["classifier", "detection_id", "class"],
                                                          ignore_index=True)
