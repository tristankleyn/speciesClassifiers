"""Standalone call-type classifier: numpy only, one file, no other dependencies.

Classifies whistle contours with a library exported by ``speciesclassifiers.calltypes`` (see
``build_library`` / ``save_library``), giving the same result as ``library.classify_contours``.
Copy this file to wherever the classifier needs to run (a server, a scheduled job, another tool).

Input: a list of contours, each a dict with
    start    contour start, Unix seconds (UTC)
    times    time of each slice (s; any origin, only differences are used)
    freqs    peak frequency of each slice (Hz)
    slice_s  slice length (s) = FFT hop / sample rate
    snr_db   optional detector SNR (dB)
If your detections store peak bins and slice numbers, convert them with ``contour_from_bins``.

Usage in Python:
    lib = load_library("library.json")
    windows, blocks = classify(contours, lib)                    # blocks with windows only
    windows, blocks = classify(contours, lib, period=(t0, t1))   # every block in [t0, t1)
From the command line (contours as a JSON list, optionally gzipped):
    python standalone.py library.json contours.json blocks.csv [windows.csv]
"""

import base64
import csv
import gzip
import json
import sys
from datetime import datetime, timezone

import numpy as np

NS = 1_000_000_000
LIBRARY_FORMAT = "speciesclassifiers.calltypes.library/1"
OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
       "==": np.equal, "!=": np.not_equal}


def load_library(path):
    with open(path) as f:
        lib = json.load(f)
    if lib.get("format") != LIBRARY_FORMAT:
        raise ValueError(f"{path} is not a call-type library ({LIBRARY_FORMAT})")
    return lib


def contour_from_bins(start, slice_numbers, peak_bins, detector, snr_db=None):
    """Contour dict from slice numbers and peak FFT bins, using the library's ``detector`` settings
    ({"sample_rate", "fft", "hop"})."""
    sr, fft, hop = detector["sample_rate"], detector["fft"], detector["hop"]
    sl = np.asarray(slice_numbers, float)
    return {"start": float(start), "times": (sl - sl[0]) * hop / sr,
            "freqs": np.asarray(peak_bins, float) * sr / fft, "slice_s": hop / sr,
            "snr_db": np.nan if snr_db is None else float(snr_db)}


def _day(t):
    return datetime.fromtimestamp(float(t), timezone.utc).strftime("%Y%m%d")


def _nanmean(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return float(v.mean()) if len(v) else np.nan


def _std1(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else np.nan


# ------------------------------------------------------------------------ contours ---------
def contour_params(t, f, slice_s):
    """Shape measurements of one contour (``t`` from 0 s, ``f`` in Hz)."""
    n = len(f)
    dur = (t[-1] - t[0]) + slice_s
    d = np.diff(f) / np.diff(t) if n > 1 else np.array([0.0])
    sgn = np.sign(np.diff(f)) if n > 1 else np.array([0.0])
    nz = sgn[sgn != 0]
    tt = t - t[0]
    p = {
        "duration_s": dur, "start_freq_hz": f[0], "end_freq_hz": f[-1],
        "min_freq_hz": f.min(), "max_freq_hz": f.max(), "freq_range_hz": f.max() - f.min(),
        "mean_freq_hz": f.mean(), "std_freq_hz": f.std(),
        "freq_at_quarter_hz": f[int(round((n - 1) * 0.25))],
        "freq_at_half_hz": f[int(round((n - 1) * 0.5))],
        "freq_at_threequarter_hz": f[int(round((n - 1) * 0.75))],
        "time_of_min_freq_rel": tt[np.argmin(f)] / tt[-1] if tt[-1] > 0 else 0.0,
        "time_of_max_freq_rel": tt[np.argmax(f)] / tt[-1] if tt[-1] > 0 else 0.0,
        "overall_slope_hz_s": (f[-1] - f[0]) / dur,
        "linreg_slope_hz_s": np.polyfit(tt, f, 1)[0] if n > 2 else 0.0,
        "quad_curvature": np.polyfit(tt, f, 2)[0] if n > 3 else 0.0,
        "mean_abs_slope_hz_s": np.abs(d).mean(),
        "std_slope_hz_s": d.std(),
        "frac_up": (sgn > 0).mean(), "frac_down": (sgn < 0).mean(),
        "n_inflections": int((np.diff(nz) != 0).sum()) if len(nz) > 1 else 0,
    }
    p["inflections_per_s"] = p["n_inflections"] / dur
    return p


def prepare_contours(raw, cp):
    """Clip to the band, apply the duration / SNR filters, measure; sorted by start."""
    kept = []
    for r in raw:
        slice_s = float(r["slice_s"])
        t_all = np.asarray(r["times"], float)
        t_all = np.round((t_all - t_all[0]) / slice_s) * slice_s
        f_all = np.asarray(r["freqs"], float)
        m = (f_all >= cp["fmin_hz"]) & (f_all <= cp["fmax_hz"])
        if m.sum() < 3:
            continue
        t, f = t_all[m], f_all[m]
        if (t[-1] - t[0]) + slice_s < cp["min_contour_s"]:
            continue
        snr = r.get("snr_db")
        snr = np.nan if snr is None else float(snr)
        if cp.get("min_snr_db") is not None and not (snr >= cp["min_snr_db"]):
            continue
        p = contour_params(t, f, slice_s)
        start = float(r["start"]) + float(t[0])
        kept.append({"p": p, "start": start, "end": start + p["duration_s"], "snr_db": snr,
                     "t": t - t[0], "f": f, "slice_s": slice_s})
    if kept and cp.get("snr_top_frac"):
        snr = np.array([c["snr_db"] for c in kept])
        thr = np.nanquantile(snr, 1 - cp["snr_top_frac"])
        kept = [c for c, s in zip(kept, snr) if s >= thr]
    order = np.argsort(np.array([c["start"] for c in kept]), kind="stable")
    return [kept[i] for i in order]


# ------------------------------------------------------------------------ windows ----------
def make_windows(contours, wp):
    """Sliding window starts (s), on a grid aligned to the minute of each UTC day's first contour."""
    L = wp["window_s"]
    step = L * (1 - wp["overlap"])
    days = {}
    for c in contours:
        days.setdefault(_day(c["start"]), []).append(c)
    ws = []
    for day in sorted(days):
        g = days[day]
        t0 = np.floor(min(c["start"] for c in g) / 60) * 60
        cs = np.array([c["start"] for c in g]) - t0
        ce = np.array([c["end"] for c in g]) - t0
        first = np.maximum(np.ceil((cs - L) / step + 1e-9), 0).astype(int)
        last = np.floor((ce - 1e-9) / step).astype(int)
        idx = np.unique(np.concatenate([np.arange(a, b + 1) for a, b in zip(first, last) if b >= a]))
        ws.append(t0 + idx * step)
    return np.sort(np.concatenate(ws)) if ws else np.array([])


def _members_of(ws, we, contours, assign_by):
    cs = np.array([c["start"] for c in contours])
    ce = np.array([c["end"] for c in contours])
    if assign_by == "midpoint":
        mid = (cs + ce) / 2
        return [np.where((mid < we[i]) & (mid + 1e-9 > ws[i]))[0] for i in range(len(ws))]
    return [np.where((cs < we[i]) & (ce > ws[i]))[0] for i in range(len(ws))]


def window_features(ws, contours, wp, agg_params):
    """Feature dict per window (windows failing the window filters are dropped)."""
    L = wp["window_s"]
    we = ws + L
    nb = int(wp["freq_profile_bins"])
    edges = np.linspace(wp["fmin_hz"], wp["fmax_hz"], nb + 1) if nb else None
    prof_names = [f"prof_{int(edges[k])}_{int(edges[k + 1])}" for k in range(nb)] if nb else []
    members = _members_of(ws, we, contours, wp["assign_by"])
    cs_ns = np.round(np.array([c["start"] for c in contours]) * 1e9).astype(np.int64)
    out = []
    for i, J in enumerate(members):
        if len(J) == 0:
            continue
        ws_ns, we_ns = int(np.round(ws[i] * 1e9)), int(np.round(we[i] * 1e9))
        feat = {"w_start": float(ws[i]), "w_end": float(we[i]), "day": _day(ws[i]),
                "n_contours": len(J), "contour_idx": J.tolist()}
        for a in agg_params:
            v = np.array([contours[j]["p"][a] for j in J], float)
            feat[f"{a}_mean"] = float(v.mean())
            feat[f"{a}_std"] = _std1(v)
        feat["snr_db_mean"] = _nanmean([contours[j]["snr_db"] for j in J])
        durs = np.array([contours[j]["p"]["duration_s"] for j in J])
        lc = contours[J[int(np.argmax(durs))]]["p"]                  # longest contour (first if tied)
        for a in agg_params:
            feat[f"lc_{a}"] = lc[a]
        pf, pdur = [], []
        for j in J:
            c = contours[j]
            tp = cs_ns[j] + (c["t"] * 1e9).astype(np.int64)
            m = (tp >= ws_ns) & (tp < we_ns)
            pf.append(c["f"][m])
            pdur.append(np.full(int(m.sum()), c["slice_s"]))
        pf, pdur = np.concatenate(pf), np.concatenate(pdur)
        feat["tonal_s"] = float(pdur.sum()) if len(pdur) else 0.0
        feat["occupancy"] = feat["tonal_s"] / ((we_ns - ws_ns) / 1e9)
        if len(pf):
            feat["pts_mean_freq_hz"] = float(pf.mean())
            feat["pts_std_freq_hz"] = _std1(pf)
            feat["pts_q10_hz"] = float(np.quantile(pf, 0.1))
            feat["pts_q90_hz"] = float(np.quantile(pf, 0.9))
            if nb:
                b = np.clip(np.digitize(pf, edges) - 1, 0, nb - 1)
                cnt = np.bincount(b, minlength=nb).astype(float)
                for k in range(nb):
                    feat[prof_names[k]] = cnt[k] / cnt.sum()
        if (feat["n_contours"] >= wp["min_contours"] and feat["occupancy"] >= wp["min_occupancy"]
                and feat["lc_duration_s"] >= wp["min_longest_contour_s"]):
            out.append(feat)
    return out


# ------------------------------------------------------------------------ classification ---
def transform(feats, lib):
    """Frozen feature transform -> standardised matrix (windows x features)."""
    fs = lib["feature_spec"]
    std_fill, logc, med = set(fs["std_fill_cols"]), set(fs["log_cols"]), fs["medians"]
    X = np.empty((len(feats), len(fs["cols"])))
    for i, f in enumerate(feats):
        for k, c in enumerate(fs["cols"]):
            v = f.get(c)
            v = np.nan if v is None else float(v)
            if np.isnan(v):
                v = 0.0 if c in std_fill else med[c]
            X[i, k] = np.log1p(v) if c in logc else v
    return (X - np.asarray(fs["mean"])) / np.asarray(fs["scale"])


def _members(lib):
    if "_members" not in lib:
        lib["_members"] = [np.frombuffer(base64.b64decode(c["members"]["b64"]), dtype=c["members"]["dtype"])
                           .reshape(c["members"]["shape"]).astype(float) for c in lib["clusters"]]
    return lib["_members"]


def assign(Xs, lib):
    """Nearest cluster by weighted cosine distance; unassigned (-1) if that distance exceeds the
    library's threshold or the window fails the cluster's Euclidean outlier check."""
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


def _vote(preds, labels):
    n = {lab: sum(1 for p in preds if p == lab) for lab in labels}
    tot = sum(n.values())
    if tot == 0:
        return "uncertain", None, n
    top = max(labels, key=lambda lab: n[lab])
    return (top if n[top] > tot / 2 else "uncertain"), n[top] / tot, n


def _median(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def blocks_from_windows(windows, contours, lib, period=None):
    """Clock-aligned decision blocks: majority votes, then the library's rules and ``requires``."""
    blk = int(round(lib["block_minutes"] * 60 * NS))
    by_block = {}
    for w in windows:
        by_block.setdefault((int(np.round(w["w_start"] * 1e9)) // blk) * blk, []).append(w)
    keys = set(by_block)
    if period is not None:
        p0 = (int(round(period[0] * 1e9)) // blk) * blk
        keys |= set(range(p0, int(round(period[1] * 1e9)), blk))
        keys = {k for k in keys if k < period[1] * 1e9 and k + blk > period[0] * 1e9}
    req = lib.get("requires") or {}
    blocks = []
    for b in sorted(keys):
        ws = by_block.get(b, [])
        cidx = sorted({j for w in ws for j in w["contour_idx"]})
        mf = [contours[j]["p"]["mean_freq_hz"] for j in cidx]
        r = {"block_start": b / NS, "block_end": (b + blk) / NS, "n_windows": len(ws),
             "n_assigned": sum(1 for w in ws if w["cluster"] >= 0), "n_contours": len(cidx),
             "block_meanfreq_range_hz": float(max(mf) - min(mf)) if mf else None,
             "block_median_abs_slope_hz_s": _median([w["mean_abs_slope_hz_s_mean"] for w in ws]) if ws else None,
             "block_median_freq_hz": _median([w.get("pts_mean_freq_hz", np.nan) for w in ws]) if ws else None,
             "block_median_contour_s": float(np.median([contours[j]["p"]["duration_s"] for j in cidx]))
             if cidx else None}
        for t in lib["tasks"]:
            dec, share, n = _vote([w[f"{t['name']}_pred"] for w in ws], (t["positive"], t["negative"]))
            r[f"{t['name']}_n_{t['positive']}"] = n[t["positive"]]
            r[f"{t['name']}_n_{t['negative']}"] = n[t["negative"]]
            r[f"{t['name']}_vote_share"] = share
            r[f"raw_{t['name']}"] = dec if ws else "no_detections"
        rule = None
        if ws:
            for x in lib["rules"]:
                v = r.get(x["if"])
                if v is not None and np.isfinite(v) and bool(OPS[x["op"]](v, x["value"])):
                    rule = x
                    break
        r["rule"] = None if rule is None else f"{rule['if']} {rule['op']} {rule['value']:g}"
        for t in lib["tasks"]:
            r[t["name"]] = rule["then"] if rule else r[f"raw_{t['name']}"]
        for name, (other, value) in req.items():
            if ws and not rule and r[other] != value:
                r[name] = "n/a"
        cl = {}
        for w in ws:
            cl[w["cluster"]] = cl.get(w["cluster"], 0) + 1
        r["cluster_votes"] = ";".join(f"{k}:{v}" for k, v in sorted(cl.items(), key=lambda kv: (-kv[1], kv[0])))
        blocks.append(r)
    first = ["block_start", "block_end", "n_windows", "n_assigned", "n_contours",
             *[t["name"] for t in lib["tasks"]], "rule", *[f"raw_{t['name']}" for t in lib["tasks"]]]
    return [{**{k: r[k] for k in first}, **{k: v for k, v in r.items() if k not in first}} for r in blocks]


def classify(raw_contours, lib, period=None):
    """Contours -> (windows, blocks), as lists of dicts. See the module docstring."""
    contours = prepare_contours(raw_contours, lib["contour_params"])
    feats = window_features(make_windows(contours, lib["window_params"]), contours, lib["window_params"],
                            lib["agg_params"])
    ids, dcos, deuc = assign(transform(feats, lib), lib)
    byid = {c["id"]: c for c in lib["clusters"]}
    for f, i, dc, de in zip(feats, ids, dcos, deuc):
        f["cluster"], f["cos_dist"], f["euc_dist"] = int(i), dc, de
        for t in lib["tasks"]:
            f[f"{t['name']}_pred"] = byid[int(i)]["verdicts"][t["name"]]["verdict"] if i >= 0 else "uncertain"
    return feats, blocks_from_windows(feats, contours, lib, period)


def write_csv(rows, path, drop=("contour_idx",)):
    if not rows:
        open(path, "w").close()
        return
    cols = [c for c in rows[0] if c not in drop]
    for r in rows[1:]:
        cols += [c for c in r if c not in cols and c not in drop]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    library = load_library(sys.argv[1])
    opener = gzip.open if sys.argv[2].endswith(".gz") else open
    with opener(sys.argv[2], "rt") as fh:
        data = json.load(fh)
    win, blk_rows = classify(data, library)
    write_csv(blk_rows, sys.argv[3])
    if len(sys.argv) > 4:
        write_csv(win, sys.argv[4])
    print(f"{len(data)} contours -> {len(win)} windows -> {len(blk_rows)} blocks")
