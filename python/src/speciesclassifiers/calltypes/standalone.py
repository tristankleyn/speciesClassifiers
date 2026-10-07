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

Libraries with clicks (several detection types) take a dict of detections per type:
    windows, blocks = classify({"whistle": contours, "click": clicks}, lib)
where clicks are dicts with ``start`` (Unix s), optional ``duration`` (s) and, for the spectrum features,
either the measurements (``peak_hz``, ``centroid_hz``, ``bw3_hz``, ``bw10_hz``, ``csp_*``) or ``wave`` (the
click's samples) and ``sample_rate``. From the command line, add the
clicks as a CSV (``start``, optional ``duration``; optionally gzipped) and use "-" for no contours:
    python standalone.py library.json contours.json blocks.csv [windows.csv] --clicks clicks.csv
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


def _decide(mean, classes, min_score):
    """Mean class probabilities -> (decision, score); score = p1 x (p1 - p2)."""
    if np.isnan(mean).all():
        return "uncertain", None
    srt = np.sort(mean)
    score = float(srt[-1] * (srt[-1] - (srt[-2] if len(srt) > 1 else 0.0)))
    return (classes[int(np.argmax(mean))] if score >= min_score else "uncertain"), score


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
             "block_median_abs_slope_hz_s": _median([w.get("mean_abs_slope_hz_s_mean", np.nan) for w in ws])
             if ws else None,
             "block_median_freq_hz": _median([w.get("pts_mean_freq_hz", np.nan) for w in ws]) if ws else None,
             "block_median_contour_s": float(np.median([contours[j]["p"]["duration_s"] for j in cidx]))
             if cidx else None}
        for t in lib["tasks"]:
            if t.get("classes"):      # multi-class: mean class probabilities of the assigned windows
                classes = t["classes"]
                P = np.array([[w[f"{t['name']}_p_{c}"] for c in classes] for w in ws if w["cluster"] >= 0], float)
                mean = P.mean(axis=0) if len(P) else np.full(len(classes), np.nan)
                for c, v in zip(classes, mean):
                    r[f"{t['name']}_p_{c}"] = None if np.isnan(v) else float(v)
                dec, score = _decide(mean, classes, t.get("min_score", 0.0))
                r[f"{t['name']}_score"] = score
                r[f"raw_{t['name']}"] = dec if ws else "no_detections"
                continue
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


def classify_windows(feats, lib):
    """Copies of the window feature dicts with this library's cluster, verdicts and class probabilities."""
    out = [dict(f) for f in feats]
    ids, dcos, deuc = assign(transform(out, lib), lib)
    byid = {c["id"]: c for c in lib["clusters"]}
    for f, i, dc, de in zip(out, ids, dcos, deuc):
        f["cluster"], f["cos_dist"], f["euc_dist"] = int(i), dc, de
        for t in lib["tasks"]:
            f[f"{t['name']}_pred"] = byid[int(i)]["verdicts"][t["name"]]["verdict"] if i >= 0 else "uncertain"
            for c in t.get("classes") or []:
                f[f"{t['name']}_p_{c}"] = byid[int(i)]["verdicts"][t["name"]]["probs"][c] if i >= 0 else np.nan
    return out


def pathway_blocks(feats, contours, lib, period=None):
    """Blocks through a pathway library (``children``): each layer's decision and score, ``path`` and
    ``decision`` (the deepest confident decision). A plain library gives its usual blocks."""
    wins = classify_windows(feats, lib)
    blocks = blocks_from_windows(wins, contours, lib, period)
    layer = (lib.get("layer") or {}).get("name")
    if not layer:
        return wins, blocks
    blk = int(round(lib["block_minutes"] * 60 * NS))
    for b in blocks:
        b["decision"] = b[layer]
        b["path"] = b[layer] if b[layer] not in ("uncertain", "no_detections") else ""
    for g, child in (lib.get("children") or {}).items():
        keys = {int(round(b["block_start"] * NS)) for b in blocks if b[layer] == g}
        if not keys:
            continue
        sub = [f for f in feats if (int(np.round(f["w_start"] * 1e9)) // blk) * blk in keys]
        _, cblocks = pathway_blocks(sub, contours, child)
        by = {int(round(c["block_start"] * NS)): c for c in cblocks}
        for b in blocks:
            c = by.get(int(round(b["block_start"] * NS)))
            if b[layer] != g or c is None:
                continue
            for k, v in c.items():
                if k in ("decision", "path") or k not in b:
                    if k == "decision":
                        if v not in ("uncertain", "no_detections", None):
                            b["decision"] = v
                    elif k == "path":
                        if v:
                            b["path"] = f"{g} > {v}"
                    else:
                        b[k] = v
    return wins, blocks


# ------------------------------------------------------------------------ clicks -----------
CLICK_MEASURES = ("peak_hz", "centroid_hz", "bw3_hz", "bw10_hz")


def click_spectra(waves, sr, sp):
    """Per-click spectral measurements, as ``clicks.click_spectra`` in the package: list of dicts."""
    n, L = len(waves), sp["n_samples"]
    if n == 0:
        return []
    fmax = min(sp.get("fmax_hz") or sr / 2, sr / 2)
    nb = sp.get("profile_bands") or 0
    edges = np.linspace(sp["fmin_hz"], fmax, nb + 1) if nb else None
    pcols = [f"csp_{int(edges[k])}_{int(edges[k + 1])}" for k in range(nb)] if nb else []
    seg = np.zeros((n, L))
    for i, w in enumerate(waves):
        w = np.asarray(w if w is not None else [], float).ravel()
        if not len(w):
            continue
        c = int(np.argmax(np.abs(w)))
        a = max(0, min(c - L // 2, len(w) - L))
        x = w[a:a + L]
        x = x - x.mean()
        seg[i, :len(x)] = x * np.hanning(len(x)) if len(x) > 1 else x
    nfft = max(sp["nfft"], L)
    P = np.abs(np.fft.rfft(seg, n=nfft, axis=1)) ** 2
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    band = (f >= sp["fmin_hz"]) & (f <= fmax)
    P, f = P[:, band], f[band]
    tot = P.sum(axis=1)
    ok = tot > 0
    pk = np.argmax(P, axis=1)
    pmax = P[np.arange(n), pk]
    idx = np.arange(P.shape[1])
    df = f[1] - f[0] if len(f) > 1 else np.nan
    out = {"peak_hz": f[pk], "centroid_hz": (P * f).sum(axis=1) / np.where(ok, tot, 1)}
    for name, frac in (("bw3_hz", 0.5), ("bw10_hz", 0.1)):
        below = P < (pmax * frac)[:, None]
        left = np.where(below & (idx < pk[:, None]), idx, -1).max(axis=1)
        right = np.where(below & (idx > pk[:, None]), idx, P.shape[1]).min(axis=1)
        out[name] = (right - left - 1) * df
    if nb:
        b = np.clip(np.digitize(f, edges) - 1, 0, nb - 1)
        B = np.stack([P[:, b == k].sum(axis=1) for k in range(nb)], axis=1) / np.where(ok, tot, 1)[:, None]
        for k, c in enumerate(pcols):
            out[c] = B[:, k]
    rows = []
    for i in range(n):
        rows.append({k: (float(v[i]) if ok[i] else np.nan) for k, v in out.items()})
    return rows


def prepare_clicks(raw, spectrum=None):
    """Click dicts (``start``, optional ``duration``, and either per-click measurements such as ``peak_hz``
    or ``wave`` + ``sample_rate``) -> sorted dicts with ``start``, ``end``, ``p`` and measurements."""
    raw = list(raw)
    if spectrum is not None and raw and "wave" in raw[0]:
        by_sr = {}
        for i, c in enumerate(raw):
            by_sr.setdefault(float(c["sample_rate"]), []).append(i)
        meas = [None] * len(raw)
        for sr, ids in by_sr.items():
            for i, m in zip(ids, click_spectra([raw[i]["wave"] for i in ids], sr, spectrum)):
                meas[i] = m
        raw = [{**c, **m} for c, m in zip(raw, meas)]
    out = []
    for c in raw:
        t = float(c["start"])
        if not np.isfinite(t):
            continue
        d = c.get("duration")
        d = 0.0 if d is None or d == "" or not np.isfinite(float(d)) else float(d)
        r = {"start": t, "end": t + d, "p": {"mean_freq_hz": np.nan, "duration_s": d}}
        for k, v in c.items():
            if k in CLICK_MEASURES or k.startswith("csp_"):
                r[k] = np.nan if v is None or v == "" else float(v)
        out.append(r)
    order = np.argsort(np.array([c["start"] for c in out]), kind="stable")
    return [out[i] for i in order]


def _nanq(m, q):
    m = m[np.isfinite(m)]
    return float(np.quantile(m, q)) if len(m) else np.nan


def click_windows(clicks, cwp):
    """Sliding windows over prepared clicks with their inter-click interval and spectrum features."""
    L, step = cwp["window_s"], cwp["window_s"] * (1 - cwp["overlap"])
    meas = [m for m in CLICK_MEASURES if clicks and m in clicks[0]]
    prof = [k for k in (clicks[0] if clicks else {}) if k.startswith("csp_")]
    days = {}
    for i, c in enumerate(clicks):
        days.setdefault(_day(c["start"]), []).append(i)
    out = []
    for day in sorted(days):
        idx = np.array(days[day])
        t = np.array([clicks[i]["start"] for i in idx])
        t0 = np.floor(t.min() / 60) * 60
        k = np.unique(np.concatenate([np.floor((t - t0) / step).astype(int) - j for j in range(int(np.ceil(L / step)))]))
        for s in t0 + k[k >= 0] * step:
            a, b = np.searchsorted(t, s, "left"), np.searchsorted(t, s + L, "left")
            if b - a < max(cwp["min_clicks"], 2):
                continue
            d = np.diff(t[a:b])
            d = d[(d <= cwp["max_ici_s"]) & (d > cwp.get("min_ici_s", 0.0))]
            if not len(d):
                continue
            w = {"w_start": float(s), "w_end": float(s + L), "day": day, "n_clicks": int(b - a),
                 "n_intervals": len(d), "ici_mean_s": float(d.mean()), "ici_median_s": float(np.median(d)),
                 "ici_std_s": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                 "ici_p05_s": float(np.quantile(d, 0.05)), "ici_p95_s": float(np.quantile(d, 0.95)),
                 "contour_idx": idx[a:b].tolist()}
            sel = idx[a:b]
            for m in meas:
                w[f"click_{m}_median"] = _nanq(np.array([clicks[i][m] for i in sel]), 0.5)
            for m in ("peak_hz", "centroid_hz"):
                if m in meas:
                    v = np.array([clicks[i][m] for i in sel])
                    w[f"click_{m}_iqr"] = _nanq(v, 0.75) - _nanq(v, 0.25)
            for c in prof:
                v = np.array([clicks[i][c] for i in sel])
                v = v[np.isfinite(v)]
                w[c] = float(v.mean()) if len(v) else np.nan
            out.append(w)
    return sorted(out, key=lambda w: w["w_start"])


# ------------------------------------------------------------------------ several detection types ---
def merge_evidence(rows, bases, weights=None):
    """{detector: [class probabilities] or None}, {detector: {class: rate}} -> merged {class: p} or None.
    Evidence p / rate multiplied over detectors (to the power of each weight; a class a detector doesn't know
    is neutral); one detector with data keeps its probabilities."""
    have = {d: p for d, p in rows.items() if p is not None and not all(np.isnan(v) for v in p.values())}
    classes = list(dict.fromkeys(c for d in rows for c in (bases.get(d) or {})))
    if not have:
        return None
    if len(have) == 1:
        p = next(iter(have.values()))
        return {c: float(np.nan_to_num(p.get(c, 0.0), nan=0.0)) for c in classes}
    logev = np.zeros(len(classes))
    for d, p in have.items():
        for i, c in enumerate(classes):
            v, base = p.get(c, np.nan), (bases.get(d) or {}).get(c)
            if base is not None and v is not None and not np.isnan(v):
                logev[i] += float((weights or {}).get(d, 1.0)) * np.log(max(v, 1e-9) / base)
    ev = np.exp(logev - logev.max())
    return dict(zip(classes, ev / ev.sum()))


def detector_windows(det, raw, lib):
    if det == "click":
        ev = prepare_clicks(raw, lib.get("spectrum_params"))
        return click_windows(ev, lib["window_params"]), ev
    contours = prepare_contours(raw, lib["contour_params"])
    return window_features(make_windows(contours, lib["window_params"]), contours, lib["window_params"],
                           lib["agg_params"]), contours


def _unb64(d):
    return np.frombuffer(base64.b64decode(d["b64"]), dtype=d["dtype"]).reshape(d["shape"])


def forest_proba(fd, X):
    """Class probabilities from a stored final-layer forest (as ``forest.forest_proba`` in the package)."""
    X = np.asarray(X, dtype=np.float32).astype(np.float64)
    out = np.zeros((len(X), len(fd["classes"])))
    if "_arrays" not in fd:
        fd["_arrays"] = [tuple(_unb64(t[k]) for k in ("left", "right", "feature", "threshold", "value"))
                         for t in fd["trees"]]
    for left, right, feat, thr, val in fd["_arrays"]:
        node = np.zeros(len(X), dtype=np.int64)
        active = left[node] != -1
        while active.any():
            i = np.where(active)[0]
            n = node[i]
            node[i] = np.where(X[i, feat[n]] <= thr[n], left[n], right[n])
            active = left[node] != -1
        out += val[node]
    return out / max(len(fd["_arrays"]), 1)


def _forest_row(fd, wins_by_det, use):
    """Cluster shares of one block's windows (types in ``use``) -> forest probabilities, or None."""
    x = dict.fromkeys(fd["features"], 0.0)
    any_w = False
    for d in use:
        ws = wins_by_det.get(d) or []
        if not ws:
            continue
        any_w = True
        n = {}
        for w in ws:
            k = f"{d}_c{w['cluster']}" if w["cluster"] >= 0 else f"{d}_unassigned"
            n[k] = n.get(k, 0) + 1
        for k, c in n.items():
            if k in x:
                x[k] = c / len(ws)
    if not any_w:
        return None
    p = forest_proba(fd, np.array([[x[f] for f in fd["features"]]]))[0]
    return dict(zip(fd["classes"], p))


def merged_blocks(feats, events, libs, weights=None, period=None, forest=None):
    """Blocks for one pathway layer (and those under it), merging the detection types (or deciding with the
    final-layer ``forest``). Same columns as ``detectors.merged_blocks`` in the package."""
    libs = {d: lb for d, lb in libs.items() if lb is not None and d in feats}
    first = next(iter(libs.values()))
    layer = first["layer"]["name"]
    blk = int(round(first["block_minutes"] * 60 * NS))
    task = first["tasks"][0]
    per, bases, classes, wb = {}, {}, [], {}
    for d, lb in libs.items():
        wins, bl = pathway_blocks(feats[d], events[d], {**lb, "children": {}}, period)
        per[d] = {int(round(b["block_start"] * NS)): b for b in bl}
        for w in wins:
            wb.setdefault((int(np.round(w["w_start"] * 1e9)) // blk) * blk, {}).setdefault(d, []).append(w)
        cl = lb["tasks"][0]["classes"]
        classes += [c for c in cl if c not in classes]
        bases[d] = lb.get("class_base") or {c: 1.0 / len(cl) for c in cl}
    keys = sorted(set().union(*[set(v) for v in per.values()]))
    if forest is not None:
        classes = list(forest["classes"])
    out = []
    for k in keys:
        r = {"block_start": k / NS, "block_end": (k + blk) / NS}
        rows, rule_then, rule_txt = {}, None, None
        for d, lb in libs.items():
            b = per[d].get(k)
            r[f"n_windows_{d}"] = b["n_windows"] if b else 0
            r[f"{layer}_{d}"] = b[layer] if b else "no_detections"
            if b is None:
                rows[d] = None
            elif b.get("rule"):
                rows[d] = None
                if rule_then is None:
                    rule_then, rule_txt = b[layer], f"{d}: {b['rule']}"
            else:
                rows[d] = {c: np.nan if b.get(f"{layer}_p_{c}") is None else b[f"{layer}_p_{c}"]
                           for c in lb["tasks"][0]["classes"]}
        if forest is not None:
            use = [d for d in libs if per[d].get(k) is not None and not per[d][k].get("rule")]
            m = _forest_row(forest, wb.get(k, {}), use)
            ms = forest.get("min_score", 0.0)
        else:
            m = merge_evidence(rows, bases, weights)
            ms = task.get("min_score", 0.0)
        if m is None:
            dec, score = (rule_then, None) if rule_then is not None else ("no_detections", None)
        else:
            dec, score = _decide(np.array([m[c] for c in classes]), classes, ms)
        for c in classes:
            r[f"{layer}_p_{c}"] = None if m is None else float(m[c])
        r[f"{layer}_score"] = score
        r[layer] = dec
        r["rule"] = rule_txt if (m is None and rule_then is not None) else None
        r["decision"] = dec
        r["path"] = "" if (dec in ("uncertain", "no_detections") or r["rule"]) else dec
        out.append(r)
    groups = list(dict.fromkeys(g for lb in libs.values() for g in (lb.get("children") or {})))
    for g in groups:
        sel = {int(round(r["block_start"] * NS)) for r in out if r[layer] == g}
        child = {d: (lb.get("children") or {}).get(g) for d, lb in libs.items()}
        if not sel or all(v is None for v in child.values()):
            continue
        sub = {d: [f for f in fs if (int(np.round(f["w_start"] * 1e9)) // blk) * blk in sel] for d, fs in feats.items()}
        fc = (forest.get("children") or {}).get(g) if forest is not None else None
        by = {int(round(c["block_start"] * NS)): c for c in merged_blocks(sub, events, child, weights, forest=fc)}
        for r in out:
            c = by.get(int(round(r["block_start"] * NS)))
            if r[layer] != g or c is None:
                continue
            for key, v in c.items():
                if key in ("block_start", "block_end", "rule") or key.startswith("n_windows_"):
                    continue
                if key == "decision":
                    if v not in ("uncertain", "no_detections", None):
                        r["decision"] = v
                elif key == "path":
                    if v:
                        r["path"] = f"{g} > {v}"
                else:
                    r[key] = v
    return out


def classify_multi(raw, lib, period=None):
    """{detector: detections} -> ({detector: windows}, blocks) for a library with several detection types."""
    feats, events, libs = {}, {}, {}
    for d, lb in lib["detectors"].items():
        if raw.get(d) is None:
            continue
        feats[d], events[d] = detector_windows(d, raw[d], lb)
        libs[d] = lb
    if not libs:
        raise ValueError("No detections of the library's detection types: " + ", ".join(lib["detectors"]))
    wins = {d: classify_windows(feats[d], libs[d]) for d in libs}
    return wins, merged_blocks(feats, events, libs, lib.get("weights"), period, forest=lib.get("forest"))


def read_clicks_csv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        try:
            t = float(r["start"])
        except ValueError:
            dt = datetime.fromisoformat(r["start"].strip().replace("Z", "+00:00"))
            t = (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()
        c = {"start": t, "duration": float(r["duration"]) if r.get("duration") not in (None, "") else 0.0}
        for k, v in r.items():
            if k in CLICK_MEASURES or k.startswith("csp_"):
                c[k] = float(v) if v not in (None, "") else np.nan
        out.append(c)
    return out


def classify(raw_contours, lib, period=None):
    """Contours -> (windows, blocks), as lists of dicts. See the module docstring. For a library with several
    detection types, pass {"whistle": contours, "click": clicks}; windows are then {detector: windows}."""
    if lib.get("multi_detector"):
        raw = raw_contours if isinstance(raw_contours, dict) else {"whistle": raw_contours}
        return classify_multi(raw, lib, period)
    contours = prepare_contours(raw_contours, lib["contour_params"])
    feats = window_features(make_windows(contours, lib["window_params"]), contours, lib["window_params"],
                            lib["agg_params"])
    return pathway_blocks(feats, contours, lib, period)


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
    args = sys.argv[1:]
    clicks_path = None
    if "--clicks" in args:
        i = args.index("--clicks")
        clicks_path = args[i + 1]
        args = args[:i] + args[i + 2:]
    if len(args) < 3:
        sys.exit(__doc__)
    library = load_library(args[0])
    data = None
    if args[1] != "-":
        opener = gzip.open if args[1].endswith(".gz") else open
        with opener(args[1], "rt") as fh:
            data = json.load(fh)
    if library.get("multi_detector"):
        raw_in = {"whistle": data, "click": read_clicks_csv(clicks_path) if clicks_path else None}
        win, blk_rows = classify(raw_in, library)
        if len(args) > 3:
            for det, rows_ in win.items():
                write_csv(rows_, args[3] if det == "whistle" else args[3].replace(".csv", f"_{det}.csv"))
        n_win = ", ".join(f"{len(v)} {k} windows" for k, v in win.items())
    else:
        win, blk_rows = classify(data, library)
        if len(args) > 3:
            write_csv(win, args[3])
        n_win = f"{len(win)} windows"
    write_csv(blk_rows, args[2])
    print(f"{n_win} -> {len(blk_rows)} blocks")
