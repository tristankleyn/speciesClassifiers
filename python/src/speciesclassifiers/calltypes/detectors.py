"""Several detection types (whistles and clicks) in one classification.

Each detection type has its own windows, features, clusters and held-out test (``discover_pathway`` run on
its own windows). For a decision block, each layer of the pathway combines the detection types' class
probabilities with ``multiclass.merge_evidence`` (product of each type's evidence relative to its class
rates; a class one type doesn't know is neutral for it; a block with only one type keeps that type's
probabilities), then decides as usual (score = p1 x (p1 - p2) >= the layer's ``min_score``). The block then
goes down to the next layer of the decided group, again with every detection type that has that layer.

A decision rule (e.g. "noise") that matches a detection type's block removes that type's evidence from
the block; if no other type has evidence, the block takes the rule's decision.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .library import LIBRARY_FORMAT, _plain, block_votes, classify_windows
from .multiclass import decide, merge_evidence

DETECTORS = ("whistle", "click")
NO_DECISION = ("uncertain", "no_detections", "no_data")


def build_multi_library(libs, weights=None, block_minutes=10, name=None):
    """One library for several detection types: {detector: pathway library}, merged per layer."""
    created = datetime.now(timezone.utc)
    return _plain({"format": LIBRARY_FORMAT, "multi_detector": True,
                   "library_version": name or "calltypes-" + created.strftime("%Y%m%d-%H%M%S"),
                   "created_utc": created.isoformat(), "block_minutes": block_minutes,
                   "weights": {d: float((weights or {}).get(d, 1.0)) for d in libs},
                   "detectors": libs})


def click_events(clicks):
    """Prepared clicks as the 'contours' ``block_votes`` counts per block."""
    return pd.DataFrame({"start": clicks["start"], "end": clicks["end"], "mean_freq_hz": np.nan,
                         "duration_s": clicks["duration_s"]})


def detector_windows(detector, raw, lib):
    """Raw detections of one type -> (windows, events for block counts), with the library's settings."""
    if detector == "click":
        from .clicks import ClickWindowParams, click_windows, prepare_clicks
        wp = {k: v for k, v in lib["window_params"].items() if k in ClickWindowParams.__dataclass_fields__}
        c = prepare_clicks(raw)
        return click_windows(c, ClickWindowParams(**wp)), click_events(c)
    from .contours import ContourParams, prepare_contours
    from .windows import WindowParams, make_windows, window_features
    cp, wp = ContourParams(**lib["contour_params"]), WindowParams(**lib["window_params"])
    contours, points = prepare_contours(raw, cp)
    if not len(contours):
        return pd.DataFrame(columns=["w_start", "w_end"]), pd.DataFrame(columns=["start", "end", "mean_freq_hz",
                                                                                 "duration_s"])
    return window_features(make_windows(contours, wp), contours, points, wp), contours


def _key(t, blk):
    return (np.round(np.asarray(t, float) * 1e9).astype(np.int64) // blk) * blk


def merged_blocks(windows, events, libs, weights=None, period=None):
    """Decision blocks for one pathway layer (and the layers under it), merging detection types.

    windows, events  {detector: windows / events (contours or clicks)}
    libs             {detector: pathway library node of this layer} (a detector without the layer: None)
    Columns: block_start, block_end, ``n_windows_<det>``, per layer ``<layer>`` (merged decision),
    ``<layer>_score``, ``<layer>_p_<class>``, ``<layer>_<det>`` (each type's own decision), then ``rule``,
    ``path`` and ``decision`` (deepest confident decision).
    """
    libs = {d: lib for d, lib in libs.items() if lib is not None and d in windows}
    first = next(iter(libs.values()))
    layer, blk_min = first["layer"]["name"], first["block_minutes"]
    blk = int(round(blk_min * 60 * 1e9))
    task = first["tasks"][0]
    min_score = task.get("min_score", 0.0)
    per, bases, classes = {}, {}, []
    for d, lib in libs.items():
        w = windows[d]
        w = classify_windows(w, lib) if len(w) else w.assign(cluster=pd.Series(dtype=int))
        b = block_votes(w, events[d], lib, period)
        b.index = _key(b["block_start"], blk)
        per[d] = b
        t = lib["tasks"][0]
        classes += [c for c in t["classes"] if c not in classes]
        bases[d] = (lib.get("class_base") or {}) or {c: 1.0 / len(t["classes"]) for c in t["classes"]}
    keys = sorted(set().union(*[set(b.index) for b in per.values()]))
    out = pd.DataFrame({"block_start": np.array(keys, float) / 1e9, "block_end": (np.array(keys, float) + blk) / 1e9},
                       index=keys)
    probs, rule_then, rule_txt = {}, pd.Series(None, index=keys, dtype=object), pd.Series(None, index=keys, dtype=object)
    for d, b in per.items():
        b = b.reindex(keys)
        out[f"n_windows_{d}"] = b["n_windows"].fillna(0).astype(int).to_numpy()
        P = pd.DataFrame({c: pd.to_numeric(b[f"{layer}_p_{c}"], errors="coerce") if f"{layer}_p_{c}" in b
                          else np.nan for c in libs[d]["tasks"][0]["classes"]}, index=keys)
        ruled = b["rule"].notna().to_numpy() if "rule" in b else np.zeros(len(keys), bool)
        P[ruled] = np.nan
        first_rule = ruled & rule_then.isna().to_numpy()
        rule_then[first_rule] = b.loc[first_rule, layer].to_numpy()
        rule_txt[first_rule] = (f"{d}: " + b.loc[first_rule, "rule"].astype(str)).to_numpy()
        probs[d] = P
        out[f"{layer}_{d}"] = b[layer].fillna("no_detections").to_numpy()
    M = merge_evidence(probs, bases, weights).reindex(columns=classes)
    dec, score = decide(M.to_numpy(float), classes, min_score)
    dec = np.asarray(dec, dtype=object)
    none = dec == "no_data"
    has_rule = rule_then.notna().to_numpy()
    dec[none & has_rule] = rule_then[none & has_rule].to_numpy()
    dec[none & ~has_rule] = "no_detections"
    for c in classes:
        out[f"{layer}_p_{c}"] = M[c].to_numpy()
    out[f"{layer}_score"] = score
    out[layer] = dec
    out["rule"] = np.where(none & has_rule, rule_txt.to_numpy(), None)
    out["decision"] = dec
    out["path"] = np.where(np.isin(dec, NO_DECISION) | (none & has_rule), "", dec)
    groups = list(dict.fromkeys(g for lib in libs.values() for g in (lib.get("children") or {})))
    for g in groups:
        sel = np.array(keys)[dec == g]
        if not len(sel):
            continue
        sub_w = {d: w[np.isin(_key(w["w_start"], blk), sel)].reset_index(drop=True) if len(w) else w
                 for d, w in windows.items()}
        child = {d: (lib.get("children") or {}).get(g) for d, lib in libs.items()}
        if not any(v is not None for v in child.values()):
            continue
        cb = merged_blocks(sub_w, events, child, weights)
        cb.index = _key(cb["block_start"], blk)
        idx = np.where(dec == g)[0]
        k = np.array(keys)[idx]
        for col in cb.columns:
            if col in ("block_start", "block_end") or col.startswith("n_windows_") or col == "rule":
                continue
            vals = cb[col].reindex(k).to_numpy()
            if col == "decision":
                ok = ~pd.Series(vals).isin([*NO_DECISION, np.nan, None]).to_numpy()
                out.iloc[idx[ok], out.columns.get_loc("decision")] = vals[ok]
            elif col == "path":
                ok = pd.Series(vals).fillna("").astype(str).str.len().to_numpy() > 0
                out.iloc[idx[ok], out.columns.get_loc("path")] = [f"{g} > {v}" for v in vals[ok]]
            else:
                if col not in out.columns:
                    out[col] = None
                out.iloc[idx, out.columns.get_loc(col)] = vals
    return out.reset_index(drop=True)


def classify_multi(raw, mlib, period=None):
    """{detector: raw detections (whistle contours / clicks)} -> ({detector: windows}, blocks)."""
    wins, evs, libs = {}, {}, {}
    for d, lib in mlib["detectors"].items():
        if raw.get(d) is None:
            continue
        wins[d], evs[d] = detector_windows(d, raw[d], lib)
        libs[d] = lib
    if not libs:
        raise ValueError("No detections of the library's detection types: " + ", ".join(mlib["detectors"]))
    return wins, merged_blocks(wins, evs, libs, mlib.get("weights"), period)
