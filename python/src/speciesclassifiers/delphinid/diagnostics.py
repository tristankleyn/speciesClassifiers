"""Diagnostics of delphinID cross-validation: held-out frame predictions joined with frame measurements, and
how filtering frames and events changes frame and event accuracy.

Frame measurements: the frame's own columns (``n_detections``, ``density`` for whistles) and values taken from
its input spectrum (the model's inputs): peak frequency, lowest / highest frequency with energy, and the width
between them, plus the prediction's confidence (top probability) and score (top x its lead over the second).

Events are decided from their kept frames (summed probabilities, renormalised, as in ``summarise``); event
filters (minimum kept frames, score, confidence) leave events undecided. Accuracies count kept frames and
decided events only; coverage says how many were kept.
"""

import numpy as np
import pandas as pd

from .frames import AcousticParams
from .transforms import WHISTLE_BIN_HZ

FRAME_MEASURES = {   # column: (label, unit)
    "confidence": ("Confidence (top probability)", ""),
    "score": ("Score (top x lead over second)", ""),
    "n_detections": ("Detections in the frame", ""),
    "density": ("Whistle density", ""),
    "peak_freq_hz": ("Peak frequency", "Hz"),
    "min_freq_hz": ("Lowest frequency with energy", "Hz"),
    "max_freq_hz": ("Highest frequency with energy", "Hz"),
    "bandwidth_hz": ("Frequency span", "Hz"),
}


def bin_centres(acoustic: AcousticParams, sample_rate=None, n=None):
    """Centre frequency (Hz) of each model input."""
    fmin, fmax = acoustic.freq_range
    d = max(int(acoustic.downsample or 1), 1)
    if acoustic.voc_type == "whistle":
        n_raw = int((fmax - fmin) / WHISTLE_BIN_HZ)
        c = np.array([fmin + WHISTLE_BIN_HZ * (k * d + min(d, n_raw - k * d) / 2) for k in range(int(np.ceil(n_raw / d)))])
    else:
        nfft = acoustic.fft_len // 2
        w = (sample_rate / 2) / nfft
        i0, i1 = int(fmin / (sample_rate / 2) * nfft), int(fmax / (sample_rate / 2) * nfft)
        n_raw = i1 - i0
        c = np.array([(i0 + k * d + min(d, n_raw - k * d) / 2) * w for k in range(int(np.ceil(n_raw / d)))])
    return c[:n] if n is not None else c


def spectrum_measures(X, freqs, voc_type):
    """Peak, lowest / highest frequency with energy, and span, per frame (rows of X)."""
    X = np.asarray(X, float)
    pk = freqs[np.argmax(X, axis=1)]
    if voc_type == "whistle":                  # histogram of contour points: any share > 0
        on = X > 0
    else:                                      # dB spectra: bins in the upper half of the frame's range
        lo, hi = X.min(axis=1, keepdims=True), X.max(axis=1, keepdims=True)
        on = X >= lo + 0.5 * (hi - lo)
    any_on = on.any(axis=1)
    first = np.where(any_on, freqs[np.argmax(on, axis=1)], np.nan)
    last = np.where(any_on, freqs[len(freqs) - 1 - np.argmax(on[:, ::-1], axis=1)], np.nan)
    return pd.DataFrame({"peak_freq_hz": pk, "min_freq_hz": first, "max_freq_hz": last, "bandwidth_hz": last - first})


def frame_table(frames, preds, acoustic: AcousticParams, sample_rate=None):
    """Held-out predictions (standard output) joined with their frames' measurements. One row per frame:
    ``detection_id``, ``event_id``, ``true_class``, ``p_<class>``, ``predicted``, ``correct``, measurements."""
    w = preds.pivot_table(index=["detection_id", "event_id", "true_class"], columns="class",
                          values="probability").reset_index()
    classes = [c for c in w.columns if c not in ("detection_id", "event_id", "true_class")]
    P = w[classes].to_numpy(float)
    srt = np.sort(P, axis=1)
    out = pd.DataFrame({"detection_id": w["detection_id"].astype(str), "event_id": w["event_id"].astype(str),
                        "true_class": w["true_class"].astype(str)})
    for i, c in enumerate(classes):
        out[f"p_{c}"] = P[:, i]
    out["predicted"] = np.array(classes)[P.argmax(1)]
    out["correct"] = out["predicted"] == out["true_class"]
    out["confidence"] = srt[:, -1]
    out["score"] = srt[:, -1] * (srt[:, -1] - (srt[:, -2] if P.shape[1] > 1 else 0))
    f = frames.copy()
    f["detection_id"] = (f["event_id"].astype(str) + "_" + f["frame_start"].round(3).astype(str)).to_numpy()
    feats = [c for c in f.columns if c[0] == "f" and c[1:].isdigit()]
    meas = spectrum_measures(f[feats].to_numpy(float), bin_centres(acoustic, sample_rate, len(feats)),
                             acoustic.voc_type)
    keep = ["detection_id", "frame_start"] + [c for c in ("n_detections", "density") if c in f]
    f = pd.concat([f[keep].reset_index(drop=True), meas], axis=1).drop_duplicates("detection_id")
    out = out.merge(f, on="detection_id", how="left")
    out.attrs["classes"] = classes
    return out


def apply_filters(t, frame_filters=None, event_filters=None):
    """Frame filters {column: (min, max)} (None = no bound); event filters ``min_frames``, ``min_score``,
    ``min_confidence``. Returns (kept frames, events table with ``decided``)."""
    classes = t.attrs["classes"]
    keep = np.ones(len(t), bool)
    for col, (lo, hi) in (frame_filters or {}).items():
        if col not in t:
            continue
        v = t[col].to_numpy(float)
        if lo is not None:
            keep &= ~(v < lo)
        if hi is not None:
            keep &= ~(v > hi)
    kept = t[keep]
    ef = event_filters or {}
    allev = t.groupby("event_id")["true_class"].first()
    pc = [f"p_{c}" for c in classes]
    g = kept.groupby("event_id")
    ev = g[pc].sum()
    ev = ev.div(ev.sum(axis=1), axis=0)
    ev.columns = classes
    ev = ev.reindex(allev.index)
    ev["true_class"] = allev
    ev["n_frames"] = g.size().reindex(allev.index).fillna(0).astype(int)
    ev["n_frames_all"] = t.groupby("event_id").size()
    P = ev[classes].to_numpy(float)
    has = ~np.isnan(P).any(axis=1)
    srt = np.sort(np.nan_to_num(P), axis=1)
    ev["predicted"] = np.where(has, np.array(classes)[np.nan_to_num(P, nan=-1).argmax(1)], "")
    ev["confidence"] = np.where(has, srt[:, -1], np.nan)
    ev["score"] = np.where(has, srt[:, -1] * (srt[:, -1] - (srt[:, -2] if P.shape[1] > 1 else 0)), np.nan)
    dec = has & (ev["n_frames"].to_numpy() >= (ef.get("min_frames") or 1))
    if ef.get("min_score") is not None:
        dec &= ~(ev["score"].to_numpy() < ef["min_score"])
    if ef.get("min_confidence") is not None:
        dec &= ~(ev["confidence"].to_numpy() < ef["min_confidence"])
    ev["decided"] = dec
    ev["correct"] = dec & (ev["predicted"] == ev["true_class"])
    return kept, ev.reset_index()


def summary(t, kept, ev):
    """Coverage, accuracy, balanced accuracy and confusion matrices of kept frames and decided events."""
    classes = t.attrs["classes"]

    def acc(true, pred):
        ok = true == pred
        per = {c: float(ok[true == c].mean()) if (true == c).any() else np.nan for c in classes}
        return (float(ok.mean()) if len(ok) else np.nan,
                float(np.nanmean(list(per.values()))) if len(ok) else np.nan, per)
    fa, fb, fper = acc(kept["true_class"].to_numpy(), kept["predicted"].to_numpy())
    d = ev[ev["decided"]]
    ea, eb, eper = acc(d["true_class"].to_numpy(), d["predicted"].to_numpy())
    return {"frames_total": len(t), "frames_kept": len(kept), "frame_accuracy": fa, "frame_balanced": fb,
            "frame_by_class": fper, "events_total": len(ev), "events_decided": int(ev["decided"].sum()),
            "event_accuracy": ea, "event_balanced": eb, "event_by_class": eper,
            "frame_confusion": pd.crosstab(kept["true_class"], kept["predicted"]),
            "event_confusion": pd.crosstab(d["true_class"], d["predicted"])}
