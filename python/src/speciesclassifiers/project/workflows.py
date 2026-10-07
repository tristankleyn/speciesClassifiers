"""The stages of each classifier workflow, as run by ``runner.run``.

Each stage reads the project settings and the outputs of earlier stages, and writes its outputs to its
own folder. ``keys`` are the settings it depends on; ``after`` is the stage whose outputs it uses.
"""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

EXAMPLES = Path(__file__).resolve().parents[4] / "examples"


@dataclass
class Stage:
    name: str
    title: str
    run: Callable
    keys: list = field(default_factory=list)
    after: str = None                 # stage whose outputs this one uses (None = the one before)
    optional: bool = False            # optional stages run only when ``enabled(config)``
    enabled: Callable = lambda cfg: True
    version: int = 1                  # bump when the stage's code changes its outputs


@dataclass
class Workflow:
    name: str
    title: str
    description: str
    stages: list
    defaults: Callable
    example: Callable = None          # settings that point at the bundled example data

    def stage(self, name):
        return next(s for s in self.stages if s.name == name)


def _save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _write_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else default


# ============================================================================ random forest ==========
def rf_defaults():
    from speciesclassifiers.calltypes.clicks import ClickSpectrumParams
    return {
        "data": {"type": "rocca", "path": "", "voc_type": "whistle", "label_col": "label",
                 "group_col": "event_id", "omit": [], "classes": [], "filter_min": {}, "filter_max": {},
                 # type "binaries": base classifiers per detection type, from detections + labelled periods
                 "use_whistles": True, "use_clicks": False, "whistle_source": "binaries", "whistle_path": "",
                 "click_source": "binaries", "click_path": "", "annotations": "",
                 "contours": {"fmin_hz": 2000.0, "fmax_hz": 20000.0, "min_contour_s": 0.3, "min_snr_db": None},
                 "click_spectrum": ClickSpectrumParams().to_dict(), "max_ici_s": 1.0},
        "settings": {"n_trees": 500, "mtry": None, "node_size": 25, "max_per_group": 25, "prune": 0.0, "seed": 42},
        "final": {"block_minutes": 10, "n_trees": 500, "node_size": 1},
        "decision": {"min_score": 0.02, "min_group_score": 0.0, "min_scores": {}},
        "new_data": {"path": "", "whistle_source": "binaries", "whistle_path": "", "click_source": "binaries",
                     "click_path": "", "start": "", "end": ""},
    }


def rf_binaries_example():
    """Random forest base classifiers on whistles and clicks of the synthetic call-type example, labelled by
    species from labelled periods, with the final forest per block."""
    c, k = str(EXAMPLES / "contours_example_synthetic.json.gz"), str(EXAMPLES / "clicks_example_synthetic.csv.gz")
    return {"data": {"type": "binaries", "use_whistles": True, "use_clicks": True, "whistle_source": "json",
                     "whistle_path": c, "click_source": "csv", "click_path": k,
                     "annotations": str(EXAMPLES / "periods_example_synthetic.csv"), "label_col": "species",
                     "contours": {"fmin_hz": 800.0, "fmax_hz": 16000.0, "min_contour_s": 0.4}},
            "settings": {"n_trees": 200, "node_size": 5},
            "final": {"block_minutes": 5, "n_trees": 200},
            "new_data": {"whistle_source": "json", "whistle_path": c, "click_source": "csv", "click_path": k,
                         "start": "2024-06-04"}}


def rf_types(cfg):
    d = cfg["data"]
    return [t for t, on in (("whistle", d.get("use_whistles", True)), ("click", d.get("use_clicks", False))) if on]


def rf_binaries(cfg):
    return cfg["data"].get("type") == "binaries"


def rf_load_type(d, voc_type, src, path, periods=None, start="", end=""):
    """Detections of one type with their features: binaries (only files overlapping ``periods``) or the
    contours JSON / clicks CSV formats. Returns (detections, feature names)."""
    from speciesclassifiers.randomforest.detections import (click_detections, click_feature_names,
                                                           whistle_detections, whistle_features)
    if voc_type == "whistle":
        if src == "binaries":
            from speciesclassifiers.io import read_whistles
            raw, _ = read_whistles(path, periods=periods)
        else:
            raw = load_whistles(src, path)
        det = whistle_detections(raw, {**{"min_snr_db": None}, **(d.get("contours") or {})})
        feats = whistle_features()
    else:
        if src == "binaries":
            from speciesclassifiers.calltypes.clicks import ClickSpectrumParams, click_spectra
            from speciesclassifiers.io import read_clicks
            sp = ClickSpectrumParams(**{**ClickSpectrumParams().to_dict(), **(d.get("click_spectrum") or {})})
            raw, _ = read_clicks(path, periods=periods, keep_wave=False,
                                 measure=lambda waves, sr: click_spectra(waves, sr, sp))
            raw = raw.drop(columns=["wave", "uid", "file"])
        else:
            raw = load_clicks(src, path)
        det = click_detections(raw, d.get("max_ici_s", 1.0))
        feats = click_feature_names(det)
    if start:
        det = det[det["start"] >= pd.Timestamp(start, tz="UTC").timestamp()]
    if end:
        det = det[det["start"] < pd.Timestamp(end, tz="UTC").timestamp() + 86400]
    det = det.reset_index(drop=True)
    det.insert(0, "detection_id", [f"{voc_type[0]}{i}" for i in range(len(det))])
    det.insert(1, "time", pd.to_datetime(det["start"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
    return det, feats


def rf_example():
    return {"data": {"type": "rocca", "path": str(EXAMPLES / "RoccaContourStats_example_synthetic.csv")},
            "new_data": {"path": str(EXAMPLES / "RoccaContourStats_example_synthetic.csv")}}


def rf_read(d, path, new=False):
    """Detections and feature names from the project's data settings."""
    from speciesclassifiers.randomforest import read_feature_table, read_rocca
    if d["type"] == "rocca":
        data, feats = read_rocca(path, d["voc_type"], None, d.get("omit") or None,
                                 d.get("filter_min") or None, d.get("filter_max") or None)
        return data, feats, "label", "event_id"
    label, group = d["label_col"], d["group_col"]
    if new:
        raw = pd.read_csv(path)
        if label not in raw:
            raw[label] = "unknown"
        path = raw
    data, feats = read_feature_table(path, label, group, exclude=d.get("omit") or ())
    for col, v in (d.get("filter_min") or {}).items():
        data = data[data[col] >= v]
    for col, v in (d.get("filter_max") or {}).items():
        data = data[data[col] <= v]
    return data.reset_index(drop=True), feats, label, group


def rf_data_binaries(p, cfg):
    from speciesclassifiers.io import read_annotations
    from speciesclassifiers.randomforest.detections import label_by_periods
    d = cfg["data"]
    lab = d.get("label_col") or "label"
    periods = read_annotations(d["annotations"], require_label=False)
    if lab not in periods:
        raise ValueError(f"The annotations have no '{lab}' column; columns: {', '.join(periods.columns)}")
    periods = periods[["event_id", "start", "end", lab]].rename(columns={lab: "label"})
    periods["label"] = periods["label"].astype(str)
    if d.get("classes"):
        periods = periods[periods["label"].isin(d["classes"])]
    spans = list(zip(periods["start"], periods["end"]))
    out = p.stage_dir("data")
    for f in out.glob("detections_*.pkl"):
        f.unlink()
    meta = {"types": [], "features": {}, "label_col": "label", "group_col": "event_id", "n_detections": {}}
    counts = []
    for ty in rf_types(cfg):
        src, path = d.get(f"{ty}_source", "binaries"), d.get(f"{ty}_path", "")
        det, feats = rf_load_type(d, ty, src, path, spans if src == "binaries" else None)
        det = label_by_periods(det, periods)
        det["event_id"] = det["event_id"].astype(str)
        _save_pickle(det, out / f"detections_{ty}.pkl")
        meta["types"].append(ty)
        meta["features"][ty] = feats
        meta["n_detections"][ty] = len(det)
        c = det.groupby("label").agg(events=("event_id", "nunique"), detections=("event_id", "size"))
        counts.append(c.add_prefix(f"{ty}_"))
        print(f"{ty}s: {len(det)} detections in {det['event_id'].nunique()} events, {len(feats)} features")
    tbl = pd.concat(counts, axis=1).fillna(0).astype(int).rename_axis("label").reset_index()
    tbl.to_csv(out / "class_counts.csv", index=False)
    meta["n_groups"] = int(periods["event_id"].nunique())
    _write_json(meta, out / "meta.json")


def rf_data(p, cfg):
    d = cfg["data"]
    if rf_binaries(cfg):
        return rf_data_binaries(p, cfg)
    data, feats, label, group = rf_read(d, d["path"])
    if d.get("classes"):
        data = data[data[label].astype(str).isin(d["classes"])]
    data[label] = data[label].astype(str)
    data[group] = data[group].astype(str)
    out = p.stage_dir("data")
    _save_pickle(data, out / "detections.pkl")
    counts = data.groupby(label).agg(groups=(group, "nunique"), detections=(group, "size")).reset_index()
    counts.to_csv(out / "class_counts.csv", index=False)
    _write_json({"features": feats, "label_col": label, "group_col": group, "n_detections": len(data),
                 "n_groups": data[group].nunique()}, out / "meta.json")
    print(f"{len(data)} detections, {data[group].nunique()} groups, {len(feats)} features")


def rf_crossval(p, cfg):
    from speciesclassifiers.randomforest import RFParams, cross_validate
    meta = read_json(p.folder / "data" / "meta.json")
    if rf_binaries(cfg):
        out = p.stage_dir("crossval")
        for f in out.glob("cv_*.csv"):
            f.unlink()
        for ty in meta["types"]:
            data = load_pickle(p.folder / "data" / f"detections_{ty}.pkl")
            print(f"{ty.capitalize()}s: leave one event out ({data['event_id'].nunique()} events)")
            preds = cross_validate(data, meta["features"][ty], "label", "event_id", RFParams(**cfg["settings"]),
                                   classifier=f"randomforest-{ty}", voc_type=ty, id_col="detection_id",
                                   time_col="time")
            preds.to_csv(out / f"cv_{ty}.csv", index=False)
        return
    data = load_pickle(p.folder / "data" / "detections.pkl")
    preds = cross_validate(data, meta["features"], meta["label_col"], meta["group_col"], RFParams(**cfg["settings"]),
                           classifier="randomforest", voc_type=cfg["data"]["voc_type"])
    preds.to_csv(p.stage_dir("crossval") / "cv_detection_predictions.csv", index=False)


def rf_final(p, cfg):
    """Final layer: the base classifiers' held-out probabilities averaged per block and type, then a balanced
    random forest tested by leaving one event out; and the final forest on all blocks."""
    meta = read_json(p.folder / "data" / "meta.json")
    preds = [pd.read_csv(p.folder / "crossval" / f"cv_{ty}.csv") for ty in meta["types"]]
    out = p.stage_dir("final")
    final_stage(out, pd.concat(preds, ignore_index=True), meta["types"], cfg["final"])


def final_stage(out, preds, types, fcfg):
    """Shared by the random forest and delphinID methods: block vectors from held-out base predictions,
    leave-one-event-out final forest, per-type comparison, and the final forest on all blocks."""
    from speciesclassifiers.randomforest import RFParams
    from speciesclassifiers.stacking import (block_vectors, cross_validate_final, fit_final, heldout_rows,
                                             type_rows)
    bm = fcfg.get("block_minutes", 10)
    for f in [*out.glob("heldout_*"), out / "final_model.joblib", out / "importance.csv"]:
        f.unlink(missing_ok=True)
    blocks, feats = block_vectors(preds, bm, types)
    blocks.to_csv(out / "blocks.csv", index=False)
    classes = sorted(blocks["true_class"].astype(str).unique())
    params = RFParams(n_trees=fcfg.get("n_trees", 500), node_size=fcfg.get("node_size", 1), max_per_group=None)
    print(f"Final layer: {len(blocks)} blocks of {bm} min in {blocks['event_id'].nunique()} events; "
          f"leave one event out")
    cv = cross_validate_final(blocks, feats, params)
    rows = heldout_rows(cv, classes)
    rows.to_csv(out / "heldout_final.csv", index=False)
    info = {"classes": classes, "unit": "event_id", "base": {c: 1 / len(classes) for c in classes}, "path": [],
            "column": "label", "types": types, "final": True, "block_minutes": bm}
    _write_json(info, out / "heldout_final.json")
    for ty in types:
        tr = type_rows(blocks, ty, classes)
        tr.to_csv(out / f"heldout_final__{ty}.csv", index=False)
        _write_json({**info, "type": ty, "final": False}, out / f"heldout_final__{ty}.json")
    from speciesclassifiers.calltypes import multiclass as mc
    for lvl in ("block", "unit"):
        perf = mc.performance(mc.aggregate(rows, classes, lvl, bm), classes, 0.0)
        print(f"  {lvl}s: accuracy {perf['accuracy']:.2f}, balanced {perf['balanced_accuracy']:.2f}")
    m = fit_final(blocks, feats, params)
    m.save(out / "final_model.joblib")
    m.importance().rename("importance").to_csv(out / "importance.csv", index_label="feature")


def rf_model(p, cfg):
    from speciesclassifiers.randomforest import RFParams, fit
    meta = read_json(p.folder / "data" / "meta.json")
    if rf_binaries(cfg):
        out = p.stage_dir("model")
        for ty in meta["types"]:
            data = load_pickle(p.folder / "data" / f"detections_{ty}.pkl")
            print(f"Training the final {ty} model on all events...")
            m = fit(data, meta["features"][ty], "label", "event_id", RFParams(**cfg["settings"]))
            m.save(out / f"randomforest_{ty}.joblib")
            m.importance().rename("importance").to_csv(out / f"feature_importance_{ty}.csv")
        return
    data = load_pickle(p.folder / "data" / "detections.pkl")
    print("Training the final model on all data...")
    m = fit(data, meta["features"], meta["label_col"], meta["group_col"], RFParams(**cfg["settings"]))
    out = p.stage_dir("model")
    m.save(out / "randomforest_model.joblib")
    m.importance().rename("importance").to_csv(out / "feature_importance.csv")
    print(f"Saved {out / 'randomforest_model.joblib'}")


def rf_apply_binaries(p, cfg):
    from speciesclassifiers.randomforest import RandomForestModel
    from speciesclassifiers.randomforest.model import detection_output
    from speciesclassifiers.stacking import block_vectors, decide_blocks
    meta = read_json(p.folder / "data" / "meta.json")
    n, d = cfg["new_data"], cfg["data"]
    preds = []
    out = p.stage_dir("apply")
    for ty in meta["types"]:
        path = n.get(f"{ty}_path")
        if not path:
            print(f"No new {ty}s chosen: skipped")
            continue
        det, _ = rf_load_type(d, ty, n.get(f"{ty}_source", "binaries"), path, None, n.get("start", ""),
                              n.get("end", ""))
        m = RandomForestModel.load(p.folder / "model" / f"randomforest_{ty}.joblib")
        det["event_id"] = "new"
        pr = detection_output(det, m.predict_proba(det), m.classes, f"randomforest-{ty}", ty, id_col="detection_id",
                              time_col="time")
        pr.to_csv(out / f"detection_predictions_{ty}.csv", index=False)
        preds.append(pr)
        print(f"{len(det)} new {ty}s")
    if not preds:
        raise ValueError("Choose the new data's detections first.")
    fm = RandomForestModel.load(p.folder / "final" / "final_model.joblib")
    bm = cfg["final"].get("block_minutes", 10)
    blocks, _ = block_vectors(pd.concat(preds, ignore_index=True), bm, meta["types"], by_event=False)
    res = decide_blocks(fm, blocks, bm, float((cfg["decision"].get("min_scores") or {}).get("final", 0.0)))
    res.to_csv(out / "blocks.csv", index=False)
    print(f"{len(res)} blocks: " + ", ".join(f"{k} {v}" for k, v in res["decision"].value_counts().items()))


def rf_apply(p, cfg):
    from speciesclassifiers.randomforest import RandomForestModel, predict
    if rf_binaries(cfg):
        return rf_apply_binaries(p, cfg)
    m = RandomForestModel.load(p.folder / "model" / "randomforest_model.joblib")
    meta = read_json(p.folder / "data" / "meta.json")
    new, _, _, group = rf_read(cfg["data"], cfg["new_data"]["path"], new=True)
    new[group] = new[group].astype(str)
    dec = cfg["decision"]
    det, grp = predict(m, new, group_col=group, voc_type=cfg["data"]["voc_type"], min_score=dec["min_score"])
    grp["decision"] = np.where(grp["score"] >= dec["min_group_score"], grp["predicted"], "uncertain")
    out = p.stage_dir("apply")
    det.to_csv(out / "new_detection_predictions.csv", index=False)
    grp.to_csv(out / "new_group_predictions.csv", index=False)
    print(f"{len(new)} new detections in {grp.shape[0]} groups: " +
          ", ".join(f"{k} {v}" for k, v in grp["decision"].value_counts().items()))
    del meta


RANDOMFOREST = Workflow(
    "randomforest", "Random forest",
    "Classify species (or any label) from measurements of individual detections, e.g. PAMGuard's ROCCA "
    "contour stats or any table of detection features. Detections are combined per encounter.",
    [Stage("data", "Read detections", rf_data, ["data"]),
     Stage("crossval", "Cross-validation (leave one group out)", rf_crossval, ["settings"], version=2),
     Stage("model", "Final model", rf_model, version=2),
     Stage("final", "Final layer: random forest per block (leave one event out)", rf_final, ["final"],
           after="crossval", optional=True, enabled=rf_binaries),
     Stage("apply", "Classify new data", rf_apply, ["new_data", "decision", "final"], after="model", optional=True,
           enabled=lambda cfg: bool(cfg.get("new_data", {}).get("path") or (rf_binaries(cfg) and (
               cfg["new_data"].get("whistle_path") or cfg["new_data"].get("click_path")))))],
    rf_defaults, rf_example)


# ============================================================================ call types ============
def ct_defaults():
    from speciesclassifiers.calltypes import ContourParams
    cp = ContourParams().to_dict()
    return {
        "data": {"source": "binaries", "path": "", "use_whistles": True, "use_clicks": False,
                 "click_source": "binaries", "click_path": ""},
        "labels": {"source": "periods", "path": "", "table_paths": [], "table_label_col": "Annotation",
                   "table_keep_cols": [], "other_labels": None, "tasks": [], "requires": {},
                   "include_unannotated_days": True, "pathway": None},
        "settings": {
            "contours": cp,
            "windows": {"window_s": 4.0, "overlap": 0.5, "assign_by": "overlap", "min_contours": 2,
                        "min_occupancy": 0.0, "min_longest_contour_s": 0.0, "freq_profile_bins": 32},
            "clicks": {"window_s": 10.0, "overlap": 0.5, "min_clicks": 5, "max_ici_s": 1.0, "min_ici_s": 0.0},
            "click_spectrum": {"n_samples": 128, "nfft": 256, "fmin_hz": 5000.0, "fmax_hz": None,
                               "profile_bands": 16},
            "features": {"whistle": None, "click": None},
            "evaluate": {"max_per_class_per_day": 150, "min_class_n": 8, "rf_trees": 300},
            "discover": {"clusterer": "agglomerative", "cosine_threshold": 0.7, "min_cluster_size": 15,
                         "euclid_outlier_q": 0.95},
            "reliability": {"prior_strength": 10, "ci": 0.9, "min_label_days": 2, "unit": "event_id",
                            "balanced": True},
        },
        "decision": {"block_minutes": 10, "rules": [], "min_scores": {}, "weights": {"whistle": 1.0, "click": 1.0},
                     "method": "multiply", "forest": {"n_trees": 500, "min_samples_leaf": 1, "n_jobs": -1}},
        "new_data": {"source": "binaries", "path": "", "start": "", "end": "", "click_source": "binaries",
                     "click_path": ""},
    }


def ct_example():
    c = str(EXAMPLES / "contours_example_synthetic.json.gz")
    k = str(EXAMPLES / "clicks_example_synthetic.csv.gz")
    return {"data": {"source": "json", "path": c, "use_clicks": True, "click_source": "csv", "click_path": k},
            "labels": {"source": "csv", "path": str(EXAMPLES / "call_annotations_example_synthetic.csv"),
                       "other_labels": ["species", "ecotype"], "tasks": [], "requires": {},
                       "pathway": {"name": "species", "column": "species",
                                   "children": {"X": {"name": "ecotype", "column": "ecotype", "exclude": ["n/a"]}}}},
            "settings": {"discover": {"min_cluster_size": 10}},
            "decision": {"block_minutes": 5,
                         "rules": [{"if": "block_meanfreq_range_hz", "op": "<", "value": 200, "then": "noise"}]},
            "new_data": {"source": "json", "path": c, "start": "2024-06-04", "end": "", "click_source": "csv",
                         "click_path": k}}


def load_whistles(source, path, start="", end=""):
    """Whistle contours from PAMGuard binaries or a contours JSON (optionally .gz)."""
    import gzip
    if source == "binaries":
        from speciesclassifiers.io import read_whistles
        periods = None
        if start or end:
            t0 = pd.Timestamp(start or "1970-01-01", tz="UTC").timestamp()
            t1 = pd.Timestamp(end or "2100-01-01", tz="UTC").timestamp() + (86400 if end else 0)
            periods = [(t0, t1)]
        wh, settings = read_whistles(path, periods=periods)
        print(f"Detector settings: {settings}")
    else:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as f:
            wh = pd.DataFrame(json.load(f))
    if start:
        wh = wh[wh["start"] >= pd.Timestamp(start, tz="UTC").timestamp()]
    if end:
        wh = wh[wh["start"] < pd.Timestamp(end, tz="UTC").timestamp() + 86400]
    return wh.reset_index(drop=True)


def load_clicks(source, path, start="", end="", spectrum=None):
    """Click detections (``start`` Unix s, ``duration`` s) from PAMGuard Click Detector binaries or a CSV
    (optionally .gz; ``start`` as Unix seconds or a UTC date-time; optional per-click measurements such as
    ``peak_hz``). ``spectrum``: click spectrum settings (dict) to measure each click's spectrum from the
    binaries' waveforms (None = times only)."""
    t0 = pd.Timestamp(start, tz="UTC").timestamp() if start else None
    t1 = pd.Timestamp(end, tz="UTC").timestamp() + 86400 if end else None
    if source == "binaries":
        from speciesclassifiers.io import read_clicks
        periods = [(t0 or 0, t1 or 4e9)] if (t0 or t1) else None
        measure = None
        if spectrum is not None:
            from speciesclassifiers.calltypes.clicks import ClickSpectrumParams, click_spectra
            sp = ClickSpectrumParams(**spectrum)
            measure = lambda waves, sr: click_spectra(waves, sr, sp)     # noqa: E731
        cl, _ = read_clicks(path, periods=periods, keep_wave=False, measure=measure)
        cl = cl.drop(columns=["wave", "uid", "file"])
    else:
        cl = pd.read_csv(path)
        if "start" not in cl:
            raise ValueError(f"The clicks CSV needs a 'start' column; it has {', '.join(cl.columns[:20])}")
        if not pd.api.types.is_numeric_dtype(cl["start"]):
            cl["start"] = pd.to_datetime(cl["start"], utc=True).astype("int64") / 1e9
    if t0 is not None:
        cl = cl[cl["start"] >= t0]
    if t1 is not None:
        cl = cl[cl["start"] < t1]
    return cl.reset_index(drop=True)


def ct_detectors(cfg):
    """Detection types used: "whistle" and/or "click"."""
    d = cfg["data"]
    return [x for x, on in (("whistle", d.get("use_whistles", True)), ("click", d.get("use_clicks", False))) if on]


def ct_multi(cfg):
    """Clicks in use, or the forest final layer: each detection type is clustered on its own and combined
    per block (multiplied evidence, or the forest)."""
    return "click" in ct_detectors(cfg) or ct_uses_forest(cfg)


def ct_uses_forest(cfg):
    return (cfg.get("decision") or {}).get("method") == "forest"


def ct_features(cfg, det):
    """The chosen feature keys of a detection type (None = the defaults)."""
    from speciesclassifiers.calltypes.windows import default_features
    f = (cfg["settings"].get("features") or {}).get(det)
    if f is None and det == "click":
        return default_features("click")
    return f


def ct_click_params(cfg):
    from speciesclassifiers.calltypes.clicks import ClickWindowParams
    return ClickWindowParams(**cfg["settings"].get("clicks", {}))


def ct_click_spectrum(cfg):
    from speciesclassifiers.calltypes.clicks import ClickSpectrumParams
    return {**ClickSpectrumParams().to_dict(), **(cfg["settings"].get("click_spectrum") or {})}


def _windows_file(p, det):
    return p.folder / "windows" / ("windows.pkl" if det == "whistle" else f"windows_{det}.pkl")


def load_calls(lab):
    """Call annotations (``source`` "csv" or "tables") or labelled periods (``source`` "periods": a CSV with
    ``event_id, start, end`` and label columns such as ``species``; no call types needed)."""
    from speciesclassifiers.io import read_annotations
    if lab["source"] == "periods":
        return read_annotations(lab["path"], require_label=False)
    if lab["source"] == "csv":
        return read_annotations(lab["path"])
    from speciesclassifiers.io.annotation_tables import read_annotation_table
    t = read_annotation_table(list(lab["table_paths"]), label_col=lab["table_label_col"],
                              keep_cols=tuple(lab.get("table_keep_cols") or ()))
    return read_annotations(t)


def other_label_columns(calls, lab):
    if lab.get("other_labels") is not None:
        return list(lab["other_labels"])
    skip = ("event_id", "start", "end", "uncertain") + (() if lab.get("source") == "periods" else ("label",))
    return [c for c in calls.columns if c not in skip]


def ct_window_params(cfg):
    from speciesclassifiers.calltypes import WindowParams
    c = cfg["settings"]["contours"]
    return WindowParams(**cfg["settings"]["windows"], fmin_hz=c["fmin_hz"], fmax_hz=c["fmax_hz"])


def ct_contours(p, cfg):
    from speciesclassifiers.calltypes import ContourParams, prepare_contours
    d = cfg["data"]
    out = p.stage_dir("contours")
    if "whistle" not in ct_detectors(cfg):
        (out / "contours.pkl").unlink(missing_ok=True)
        _write_json({"n_read": 0, "n_kept": 0, "days": [], "note": "Whistles not used"}, out / "summary.json")
        print("Whistles not used")
        return
    wh = load_whistles(d["source"], d["path"])
    print(f"{len(wh)} contours read; measuring...")
    c, pts = prepare_contours(wh, ContourParams(**cfg["settings"]["contours"]))
    _save_pickle((c, pts), out / "contours.pkl")
    _write_json({"n_read": len(wh), "n_kept": len(c), "days": sorted(c["day"].unique().tolist()) if len(c) else []},
                out / "summary.json")
    print(f"{len(c)} of {len(wh)} contours kept")


def ct_clicks(p, cfg):
    from speciesclassifiers.calltypes.clicks import prepare_clicks
    d = cfg["data"]
    raw = load_clicks(d.get("click_source", "binaries"), d.get("click_path", ""), spectrum=ct_click_spectrum(cfg))
    c = prepare_clicks(raw)
    out = p.stage_dir("clicks")
    _save_pickle(c, out / "clicks.pkl")
    _write_json({"n_clicks": len(c), "days": sorted(c["day"].unique().tolist()) if len(c) else []},
                out / "summary.json")
    spec = "with spectra" if "peak_hz" in c else "times only (no waveforms: no spectral features)"
    print(f"{len(c)} clicks read, {spec}")


def ct_windows(p, cfg):
    from speciesclassifiers.calltypes import label_windows, make_windows, window_features
    from speciesclassifiers.calltypes.windows import label_windows_by_period
    dets = ct_detectors(cfg)
    if not dets:
        raise ValueError("Choose at least one detection type (whistles or clicks).")
    calls = load_calls(cfg["labels"])
    carry = other_label_columns(calls, cfg["labels"])
    periods = cfg["labels"]["source"] == "periods"
    cols = ["event_id"] + [c for c in carry if c != "event_id"]
    wins = {}
    if "whistle" in dets:
        c, pts = load_pickle(p.folder / "contours" / "contours.pkl")
        wp = ct_window_params(cfg)
        w = window_features(make_windows(c, wp), c, pts, wp)
        if periods:     # each window takes the labels of the period it falls in
            w = label_windows_by_period(w.assign(label="none"), calls, columns=cols)
            w.loc[w["label_source"] != "unannotated", "label"] = "untyped"     # no call types
        else:
            w = label_windows(w, calls, wp, carry=list(dict.fromkeys([*carry, "event_id"])))
        wins["whistle"] = w
    if "click" in dets:
        from speciesclassifiers.calltypes.clicks import click_windows, event_periods
        clicks = load_pickle(p.folder / "clicks" / "clicks.pkl")
        w = click_windows(clicks, ct_click_params(cfg))
        per = calls if periods else event_periods(calls, cols)    # call annotations: the event's span
        w = label_windows_by_period(w.assign(label="none"), per, columns=cols)
        w.loc[w["label_source"] != "unannotated", "label"] = "untyped"
        wins["click"] = w
    days = sorted(pd.to_datetime(calls["start"], unit="s", utc=True).dt.strftime("%Y%m%d").unique())
    out = p.stage_dir("windows")
    for f in list(out.glob("windows*")):
        f.unlink()
    for det, w in wins.items():
        sfx = "" if det == "whistle" else f"_{det}"
        _save_pickle(w, out / f"windows{sfx}.pkl")
        w.drop(columns=[x for x in w.columns if x.startswith("prof_")]).to_csv(out / f"windows{sfx}.csv", index=False)
    w0 = wins.get("whistle", next(iter(wins.values())))
    pd.crosstab(w0["label"], w0["day"]).to_csv(out / "labels_by_day.csv")
    _write_json({"n_windows": len(w0), "annotated_days": days, "other_labels": carry, "n_calls": len(calls),
                 "label_counts": w0["label"].value_counts().to_dict(), "detectors": dets,
                 "n_windows_by_type": {d: len(w) for d, w in wins.items()}}, out / "summary.json")
    what = "labelled periods" if periods else "annotated calls"
    print(", ".join(f"{len(w)} {d} windows" for d, w in wins.items()) + f"; {len(calls)} {what} on {len(days)} days")


def ct_evaluate(p, cfg):
    from speciesclassifiers.calltypes import EvalParams, evaluate, select_rows
    out = p.stage_dir("evaluate")
    if "whistle" not in ct_detectors(cfg):
        _write_json({"note": "Call types are evaluated for whistles only."}, out / "evaluate.json")
        print("Whistles not used: skipped")
        return
    w = load_pickle(p.folder / "windows" / "windows.pkl")
    ep = EvalParams(**cfg["settings"]["evaluate"], features=ct_features(cfg, "whistle"))
    sel = select_rows(w, ep)
    if sel["label"].nunique() < 2:
        _write_json({"note": "Fewer than 2 call types with enough labelled windows: nothing to evaluate."},
                    out / "evaluate.json")
        print("Fewer than 2 call types with enough windows: skipped")
        return
    print(f"Evaluating {len(sel)} labelled windows, {sel['label'].nunique()} call types...")
    res = evaluate(sel, ep)
    keep = {k: v for k, v in res.items() if k not in ("confusion", "feature_importance", "predictions")}
    _write_json(keep, out / "evaluate.json")
    if "confusion" in res:
        res["confusion"].to_csv(out / "confusion.csv")
        res["feature_importance"].rename("importance").to_csv(out / "feature_importance.csv")
    print(f"Balanced accuracy {res.get('rf_bal_acc', float('nan')):.2f}, skill {res.get('skill', float('nan')):.2f}")


def ct_tasks(cfg):
    from speciesclassifiers.calltypes import LabelTask
    return [LabelTask(**t) for t in cfg["labels"].get("tasks") or []]


def ct_discover(p, cfg):
    from speciesclassifiers.calltypes import DiscoverParams, ReliabilityParams, discover, discovery_pool
    days = read_json(p.folder / "windows" / "summary.json")["annotated_days"]
    tasks = ct_tasks(cfg)
    rp = ReliabilityParams(**{**cfg["settings"]["reliability"], "block_minutes": cfg["decision"]["block_minutes"]})
    tree = cfg["labels"].get("pathway")
    if ct_multi(cfg):
        return ct_discover_multi(p, cfg, days, tree, rp)
    w = load_pickle(p.folder / "windows" / "windows.pkl")
    pool = discovery_pool(w, days, include_unannotated_days=cfg["labels"].get("include_unannotated_days", True))
    print(f"Clustering {len(pool)} windows...")
    dp = DiscoverParams(**cfg["settings"]["discover"], features=ct_features(cfg, "whistle"))
    if tree:
        return ct_discover_pathway(p, pool, tree, dp, rp)
    r = discover(pool, dp, tasks, rp)
    out = p.stage_dir("discover")
    _save_pickle(r, out / "discover.pkl")
    for name, m in r.get("multiclass", {}).items():      # held-out predictions, for the app's curves
        if len(m.get("window_preds", [])):
            m["window_preds"].to_csv(out / f"heldout_{name}.csv", index=False)
            _write_json({"classes": m["classes"], "unit": m["unit"], "base": m["base"]}, out / f"heldout_{name}.json")
    r["summary"].to_csv(out / "clusters.csv", index=False)
    _write_json(r["scorecard"], out / "scorecard.json")
    if len(r["decision_windows"]):
        r["decision_windows"].to_csv(out / "decision_windows.csv", index=False)
    print(f"{r['scorecard']['n_clusters']} clusters, {r['scorecard']['frac_unassigned']:.0%} of windows unassigned")


def ct_discover_multi(p, cfg, days, tree, rp):
    """Each detection type clustered through the pathway on its own; held-out results merged per block."""
    from speciesclassifiers.calltypes import DiscoverParams, discovery_pool
    from speciesclassifiers.calltypes import multiclass as mc
    from speciesclassifiers.calltypes.pathway import check_pathway, discover_pathway, iter_results, layer_summaries
    if not tree:
        raise ValueError("Using clicks needs a classification: set up its layers on the Labels step.")
    problems = check_pathway(tree)
    if problems:
        raise ValueError("Pathway: " + " ".join(problems))
    out = p.stage_dir("discover")
    for old in [*out.glob("heldout_*"), *out.glob("clusters*.csv"), *out.glob("*.pkl")]:
        old.unlink()
    weights = cfg["decision"].get("weights") or {}
    prs, layers, preds, bases, nodes = {}, [], {}, {}, {}
    for det in ct_detectors(cfg):
        w = load_pickle(_windows_file(p, det))
        pool = discovery_pool(w, days, include_unannotated_days=cfg["labels"].get("include_unannotated_days", True))
        print(f"{det.capitalize()}s: clustering {len(pool)} windows...")
        dp = DiscoverParams(**cfg["settings"]["discover"], features=ct_features(cfg, det))
        pr = prs[det] = discover_pathway(pool, tree, dp, rp)
        _save_pickle(pr, out / f"pathway_{det}.pkl")
        layers.append(layer_summaries(pr).assign(type=det))
        for r in iter_results(pr):
            name = r["node"]["name"]
            r["result"]["summary"].to_csv(out / f"clusters_{name}__{det}.csv", index=False)
            m = r["result"]["multiclass"][name]
            nodes.setdefault(name, (r["path"], r["node"]["column"]))
            if len(m.get("window_preds", [])):
                m["window_preds"].to_csv(out / f"heldout_{name}__{det}.csv", index=False)
                _write_json({"classes": m["classes"], "unit": m["unit"], "base": m["base"], "path": r["path"],
                             "column": r["node"]["column"], "type": det}, out / f"heldout_{name}__{det}.json")
                preds.setdefault(name, {})[det] = m["window_preds"]
                bases.setdefault(name, {})[det] = m["base"]
    merged_rows = []
    for name, P in preds.items():
        hm = mc.merge_heldout(P, bases[name], weights, cfg["decision"]["block_minutes"])
        if not len(hm):
            continue
        classes = hm.attrs["classes"]
        hm.to_csv(out / f"heldout_{name}.csv", index=False)
        path, column = nodes[name]
        _write_json({"classes": classes, "unit": mc.unit_col(rp), "base": {c: 1 / len(classes)
                     for c in classes}, "path": path, "column": column, "merged": True, "types": list(P)},
                    out / f"heldout_{name}.json")
        row = {"layer": name, "within": " > ".join(path) or "(all)", "classes": ", ".join(classes),
               "windows": sum(len(x) for x in P.values()), "clusters": None, "type": "merged"}
        for lvl in ("block", "unit"):
            perf = mc.performance(mc.aggregate(hm, classes, lvl, cfg["decision"]["block_minutes"]), classes, 0.0)
            row[f"{lvl}_accuracy"], row[f"{lvl}_balanced_accuracy"] = perf["accuracy"], perf["balanced_accuracy"]
        merged_rows.append(row)
    lt = pd.concat([*layers, pd.DataFrame(merged_rows)], ignore_index=True)
    lt = lt[["layer", "type", *[c for c in lt.columns if c not in ("layer", "type")]]]
    lt.to_csv(out / "layers.csv", index=False)
    _write_json({"pathway": True, "types": list(prs), "layers": list(dict.fromkeys(lt["layer"])),
                 "n_clusters": {f"{r['node']['name']}__{d}": r["result"]["scorecard"]["n_clusters"]
                                for d, pr in prs.items() for r in iter_results(pr)}}, out / "scorecard.json")
    print(lt.to_string(index=False))


def ct_forest(p, cfg):
    """Final layer: a balanced random forest per layer on the blocks' cluster shares, tested strictly
    (each held-out event reclustered without it)."""
    from speciesclassifiers.calltypes import DiscoverParams, LabelTask, ReliabilityParams, discovery_pool
    from speciesclassifiers.calltypes import multiclass as mc
    from speciesclassifiers.calltypes.forest import ForestParams, final_forest, strict_layer_loo
    from speciesclassifiers.calltypes.pathway import node_label_col, node_labels
    out = p.stage_dir("forest")
    for old in [*out.glob("heldout_*"), *out.glob("importance_*"), out / "forest.pkl", out / "layers.csv"]:
        old.unlink(missing_ok=True)
    tree = cfg["labels"].get("pathway")
    if not tree:
        raise ValueError("The forest final layer needs a classification: set up its layers on the Labels step.")
    bm = cfg["decision"]["block_minutes"]
    rp = ReliabilityParams(**{**cfg["settings"]["reliability"], "block_minutes": bm})
    unit = mc.unit_col(rp)
    fp = ForestParams(**{**ForestParams().to_dict(), **(cfg["decision"].get("forest") or {})})
    days = read_json(p.folder / "windows" / "summary.json")["annotated_days"]
    dets = ct_detectors(cfg)
    pools = {d: discovery_pool(load_pickle(_windows_file(p, d)), days,
                               include_unannotated_days=cfg["labels"].get("include_unannotated_days", True))
             for d in dets}
    prs = {d: load_pickle(p.folder / "discover" / f"pathway_{d}.pkl") for d in dets}
    dps = {d: DiscoverParams(**cfg["settings"]["discover"], features=ct_features(cfg, d)) for d in dets}
    rows = []

    def layer(node, pools, prs, path):
        name, col = node["name"], node_label_col(node)
        pools = {d: x.assign(**{col: node_labels(x, node)}) for d, x in pools.items()}
        classes = list(dict.fromkeys(c for d in prs for c in prs[d]["result"]["multiclass"][name]["classes"]))
        task = LabelTask(name, col, classes=classes)
        print(f"Layer '{name}'" + (f" (within {' > '.join(path)})" if path else "") + f": classes {', '.join(classes)}")
        held = strict_layer_loo({d: pools[d] for d in prs}, task, classes, {d: dps[d] for d in prs}, rp, bm, fp)
        if len(held):
            held.to_csv(out / f"heldout_{name}.csv", index=False)
            _write_json({"classes": classes, "unit": unit, "base": {c: 1 / len(classes) for c in classes},
                         "path": list(path), "column": node["column"], "forest": True, "types": list(prs)},
                        out / f"heldout_{name}.json")
            row = {"layer": name, "within": " > ".join(path) or "(all)", "classes": ", ".join(classes),
                   "blocks": len(held)}
            for lvl in ("block", "unit"):
                perf = mc.performance(mc.aggregate(held, classes, lvl, bm), classes, 0.0)
                row[f"{lvl}_accuracy"], row[f"{lvl}_balanced_accuracy"] = perf["accuracy"], perf["balanced_accuracy"]
            rows.append(row)
        fd, imp = final_forest({d: pools[d] for d in prs}, {d: prs[d]["result"] for d in prs}, task, classes, unit,
                               bm, fp)
        imp.rename("importance").to_csv(out / f"importance_{name}.csv", index_label="feature")
        fd["layer"] = name
        fd["children"] = {}
        for g, child in (node.get("children") or {}).items():
            sub_prs = {d: pr["children"][g] for d, pr in prs.items() if g in pr["children"]}
            if not sub_prs:
                continue
            sub = {d: pools[d][pools[d][col].to_numpy() == g].reset_index(drop=True) for d in sub_prs}
            fd["children"][g] = layer(child, sub, sub_prs, path + (g,))
        return fd

    forest = layer(tree, pools, prs, ())
    _save_pickle(forest, out / "forest.pkl")
    lt = pd.DataFrame(rows)
    lt.to_csv(out / "layers.csv", index=False)
    print(lt.to_string(index=False))


def ct_discover_pathway(p, pool, tree, dp, rp):
    from speciesclassifiers.calltypes.pathway import check_pathway, discover_pathway, iter_results, layer_summaries
    problems = check_pathway(tree)
    if problems:
        raise ValueError("Pathway: " + " ".join(problems))
    pr = discover_pathway(pool, tree, dp, rp)
    out = p.stage_dir("discover")
    for old in list(out.glob("heldout_*")) + list(out.glob("clusters_*.csv")):
        old.unlink()
    _save_pickle(pr, out / "pathway.pkl")
    layers = layer_summaries(pr)
    layers.to_csv(out / "layers.csv", index=False)
    for r in iter_results(pr):
        name = r["node"]["name"]
        res = r["result"]
        res["summary"].to_csv(out / f"clusters_{name}.csv", index=False)
        m = res["multiclass"][name]
        if len(m.get("window_preds", [])):
            m["window_preds"].to_csv(out / f"heldout_{name}.csv", index=False)
            _write_json({"classes": m["classes"], "unit": m["unit"], "base": m["base"], "path": r["path"],
                         "column": r["node"]["column"]}, out / f"heldout_{name}.json")
    _write_json({"pathway": True, "layers": layers["layer"].tolist(),
                 "n_clusters": {r["node"]["name"]: r["result"]["scorecard"]["n_clusters"] for r in iter_results(pr)}},
                out / "scorecard.json")
    print(layers.to_string(index=False))


def ct_library(p, cfg):
    from speciesclassifiers.calltypes import ContourParams, build_library, save_library, self_check
    if ct_multi(cfg):
        return ct_library_multi(p, cfg)
    if cfg["labels"].get("pathway"):
        return ct_library_pathway(p, cfg)
    r = load_pickle(p.folder / "discover" / "discover.pkl")
    lib = build_library(r, ContourParams(**cfg["settings"]["contours"]), ct_window_params(cfg), ct_tasks(cfg),
                        block_minutes=cfg["decision"]["block_minutes"], rules=cfg["decision"].get("rules") or None,
                        requires={k: tuple(v) for k, v in (cfg["labels"].get("requires") or {}).items()},
                        name=f"{cfg.get('name', 'calltypes')}",
                        min_scores=cfg["decision"].get("min_scores") or {})
    out = p.stage_dir("library")
    save_library(lib, out / "library.json")
    import shutil
    from speciesclassifiers.calltypes import standalone
    shutil.copy(standalone.__file__, out / "standalone.py")
    sc = self_check(r, lib)
    _write_json(sc, out / "self_check.json")
    print(f"Library: {len(lib['clusters'])} clusters; {sc['same_cluster']:.0%} of clustered training windows "
          "re-assigned to their own cluster")


def ct_library_pathway(p, cfg):
    import shutil

    from speciesclassifiers.calltypes import ContourParams, save_library, standalone
    from speciesclassifiers.calltypes.pathway import build_pathway_library
    pr = load_pickle(p.folder / "discover" / "pathway.pkl")
    lib = build_pathway_library(pr, ContourParams(**cfg["settings"]["contours"]), ct_window_params(cfg),
                                block_minutes=cfg["decision"]["block_minutes"],
                                rules=cfg["decision"].get("rules") or None, name=cfg.get("name", "calltypes"),
                                min_scores=cfg["decision"].get("min_scores") or {})
    out = p.stage_dir("library")
    save_library(lib, out / "library.json")
    shutil.copy(standalone.__file__, out / "standalone.py")
    for f in ("self_check.json",):
        (out / f).unlink(missing_ok=True)
    n = sum(1 for _ in _lib_nodes(lib))
    print(f"Pathway library: {n} layer(s)")


def ct_library_multi(p, cfg):
    import shutil

    from speciesclassifiers.calltypes import ContourParams, save_library, standalone
    from speciesclassifiers.calltypes.detectors import build_multi_library
    from speciesclassifiers.calltypes.pathway import build_pathway_library
    libs = {}
    for det in ct_detectors(cfg):
        pr = load_pickle(p.folder / "discover" / f"pathway_{det}.pkl")
        whistle = det == "whistle"
        libs[det] = build_pathway_library(
            pr, ContourParams(**cfg["settings"]["contours"]) if whistle else None,
            ct_window_params(cfg) if whistle else ct_click_params(cfg), block_minutes=cfg["decision"]["block_minutes"],
            rules=(cfg["decision"].get("rules") or None) if whistle else None, name=cfg.get("name", "calltypes"),
            min_scores=cfg["decision"].get("min_scores") or {})
    if "click" in libs:
        libs["click"]["spectrum_params"] = ct_click_spectrum(cfg)
    lib = build_multi_library(libs, cfg["decision"].get("weights"), cfg["decision"]["block_minutes"],
                              cfg.get("name", "calltypes"))
    if ct_uses_forest(cfg):
        forest = load_pickle(p.folder / "forest" / "forest.pkl")
        ms = cfg["decision"].get("min_scores") or {}

        def set_ms(fd):
            fd["min_score"] = float(ms.get(fd["layer"], 0.0))
            for c in fd["children"].values():
                set_ms(c)
        set_ms(forest)
        lib["forest"] = forest
    out = p.stage_dir("library")
    save_library(lib, out / "library.json")
    shutil.copy(standalone.__file__, out / "standalone.py")
    (out / "self_check.json").unlink(missing_ok=True)
    print("Library: " + ", ".join(f"{d}s {sum(1 for _ in _lib_nodes(lb))} layer(s)" for d, lb in libs.items()))


def _lib_nodes(lib):
    yield lib
    for c in (lib.get("children") or {}).values():
        yield from _lib_nodes(c)


def ct_apply(p, cfg):
    from speciesclassifiers.calltypes import classify_contours, load_library, standard_output
    lib = load_library(p.folder / "library" / "library.json")
    n = cfg["new_data"]
    if lib.get("multi_detector"):
        return ct_apply_multi(p, lib, n)
    wh = load_whistles(n["source"], n["path"], n.get("start", ""), n.get("end", ""))
    print(f"Classifying {len(wh)} contours...")
    if lib.get("layer"):
        from speciesclassifiers.calltypes.pathway import classify_contours_pathway
        w, blocks, _ = classify_contours_pathway(wh, lib)
        out = p.stage_dir("apply")
        w.drop(columns=[x for x in w.columns if x.startswith("prof_")]).to_csv(out / "windows.csv", index=False)
        blocks.to_csv(out / "blocks.csv", index=False)
        (out / "window_predictions_standard.csv").unlink(missing_ok=True)
        print(f"{len(w)} windows -> {len(blocks)} decision blocks")
        return
    w, blocks, _ = classify_contours(wh, lib)
    out = p.stage_dir("apply")
    w.drop(columns=[x for x in w.columns if x.startswith("prof_")]).to_csv(out / "windows.csv", index=False)
    blocks.to_csv(out / "blocks.csv", index=False)
    if len(w):
        standard_output(w, lib).to_csv(out / "window_predictions_standard.csv", index=False)
    print(f"{len(w)} windows -> {len(blocks)} decision blocks")


def ct_apply_multi(p, lib, n):
    from speciesclassifiers.calltypes.detectors import classify_multi
    raw = {}
    if "whistle" in lib["detectors"]:
        raw["whistle"] = load_whistles(n["source"], n["path"], n.get("start", ""), n.get("end", ""))
    if "click" in lib["detectors"]:
        if not n.get("click_path"):
            raise ValueError("The library uses clicks: choose the new data's clicks.")
        raw["click"] = load_clicks(n.get("click_source", "binaries"), n["click_path"], n.get("start", ""),
                                   n.get("end", ""), spectrum=lib["detectors"]["click"].get("spectrum_params"))
    print("Classifying " + ", ".join(f"{len(v)} {d}s" for d, v in raw.items()) + "...")
    wins, blocks = classify_multi(raw, lib)
    out = p.stage_dir("apply")
    for f in [*out.glob("windows*.csv"), out / "window_predictions_standard.csv"]:
        f.unlink(missing_ok=True)
    for d, w in wins.items():
        w.drop(columns=[x for x in w.columns if x.startswith("prof_")]).to_csv(
            out / ("windows.csv" if d == "whistle" else f"windows_{d}.csv"), index=False)
    blocks.to_csv(out / "blocks.csv", index=False)
    print(", ".join(f"{len(w)} {d} windows" for d, w in wins.items()) + f" -> {len(blocks)} decision blocks")


CALLTYPES = Workflow(
    "calltypes", "Call types",
    "Find call types in whistle contours and judge them for labels you choose (e.g. species, population), "
    "from call annotations on some days. Classifies new data in clock-aligned blocks.",
    [Stage("contours", "Read and measure contours", ct_contours,
           ["data.source", "data.path", "data.use_whistles", "settings.contours"]),
     Stage("clicks", "Read clicks", ct_clicks, ["data.click_source", "data.click_path", "settings.click_spectrum"],
           optional=True,
           enabled=lambda cfg: bool(cfg.get("data", {}).get("use_clicks"))),
     Stage("windows", "Windows, features and labels", ct_windows,
           ["labels.source", "labels.path", "labels.table_paths", "labels.table_label_col",
            "labels.table_keep_cols", "labels.other_labels", "settings.windows", "settings.clicks",
            "data.use_whistles", "data.use_clicks"]),
     Stage("evaluate", "Evaluate labelled call types", ct_evaluate, ["settings.evaluate", "settings.features"]),
     Stage("discover", "Discover (cluster + verdicts)", ct_discover,
           ["labels.tasks", "labels.pathway", "labels.include_unannotated_days", "settings.discover",
            "settings.reliability", "settings.features", "decision.block_minutes", "decision.weights"],
           after="windows"),
     Stage("forest", "Final layer: random forest (strict test)", ct_forest,
           ["decision.method", "decision.forest", "decision.block_minutes"], optional=True,
           enabled=ct_uses_forest),
     Stage("library", "Build library", ct_library, ["decision", "labels.requires"]),
     Stage("apply", "Classify new data", ct_apply, ["new_data"], optional=True,
           enabled=lambda cfg: bool(cfg.get("new_data", {}).get("path") or cfg.get("new_data", {}).get("click_path")))],
    ct_defaults, ct_example)


# ============================================================================ delphinID =============
def dl_defaults():
    from speciesclassifiers.delphinid import CNNParams, TrainingParams
    return {
        "data": {"source": "binaries", "binary_folder": "", "annotations": "", "voc_type": "whistle",
                 "sample_rate": None, "frames_csv": "", "frames_params": "",
                 # source "multi": several detection types, each from binaries or frames, + a final forest
                 "use_whistles": True, "use_clicks": False, "whistle_source": "binaries", "click_source": "binaries",
                 "whistle_frames_csv": "", "whistle_frames_params": "", "click_frames_csv": "",
                 "click_frames_params": "",
                 "acoustic_by_type": {
                     "whistle": {"fmin_hz": 2000.0, "fmax_hz": 20000.0, "frame_len": 4.0, "hop": 4.0, "downsample": 2,
                                 "min_density": 0.05, "min_frag_ms": 200.0},
                     "click": {"fmin_hz": 10000.0, "fmax_hz": 40000.0, "frame_len": 4.0, "hop": 4.0, "downsample": 2,
                               "min_clicks": 3, "fft_len": 512, "smooth_window": 3}}},
        "labels": {"classes": [], "group_col": "event_id"},
        "settings": {
            "acoustic": {"fmin_hz": 2000.0, "fmax_hz": 20000.0, "frame_len": 4.0, "hop": 4.0, "downsample": 2,
                         "min_density": 0.05, "min_frag_ms": 200.0, "min_clicks": 3, "fft_len": 512,
                         "smooth_window": 3},
            "cnn": CNNParams().to_dict(),
            "training": TrainingParams().to_dict(),
            "resampling": {"max_per_group": 30, "n_bootstraps": 5},
            "crossval": {"run": True, "n_folds": None},
        },
        "final": {"block_minutes": 10, "n_trees": 500, "node_size": 1},
        "decision": {"min_scores": {}},
        "new_data": {"source": "binaries", "binary_folder": "", "start": "", "end": "",
                     "whistle_frames_csv": "", "click_frames_csv": ""},
        # how new-data frames are grouped into blocks and filtered (chosen after classifying; no rerun)
        "view": {"block_minutes": 1, "min_frames": 1, "min_score": 0.0, "min_density": 0.0, "min_detections": 1},
        "model_name": "delphinID_whistles",
    }


def dl_multi_example():
    """delphinID on synthetic whistle and click frames of the same events, with the final forest per block."""
    ex = {t: (str(EXAMPLES / f"frames{'' if t == 'whistle' else '_click'}_example_synthetic.csv"),
              str(EXAMPLES / f"frames{'' if t == 'whistle' else '_click'}_example_synthetic_params.json"))
          for t in ("whistle", "click")}
    return {"data": {"source": "multi", "use_whistles": True, "use_clicks": True,
                     **{f"{t}_source": "frames" for t in ex},
                     **{f"{t}_frames_csv": v[0] for t, v in ex.items()},
                     **{f"{t}_frames_params": v[1] for t, v in ex.items()}},
            "settings": {"training": {"epochs": 8}, "resampling": {"n_bootstraps": 2},
                         "crossval": {"run": True, "n_folds": 3}},
            "final": {"block_minutes": 10, "n_trees": 200},
            "new_data": {"source": "frames", **{f"{t}_frames_csv": v[0] for t, v in ex.items()}},
            "model_name": "delphinID_example"}


def dl_multi(cfg):
    return cfg["data"].get("source") == "multi"


def dl_types(cfg):
    d = cfg["data"]
    return [t for t, on in (("whistle", d.get("use_whistles", True)), ("click", d.get("use_clicks", False))) if on]


def dl_acoustic_type(cfg, voc_type):
    from speciesclassifiers.delphinid import AcousticParams
    a = dict(dl_defaults()["data"]["acoustic_by_type"][voc_type])
    a.update((cfg["data"].get("acoustic_by_type") or {}).get(voc_type) or {})
    fr = (float(a.pop("fmin_hz")), float(a.pop("fmax_hz")))
    return AcousticParams(voc_type=voc_type, freq_range=fr,
                          **{k: v for k, v in a.items() if k in AcousticParams.__dataclass_fields__})


def dl_example():
    return {"data": {"source": "frames", "frames_csv": str(EXAMPLES / "frames_example_synthetic.csv"),
                     "frames_params": str(EXAMPLES / "frames_example_synthetic_params.json")},
            "settings": {"training": {"epochs": 10}, "resampling": {"n_bootstraps": 2},
                         "crossval": {"run": True, "n_folds": 4}},
            "model_name": "delphinID_example"}


def require_tensorflow():
    from .checks import tensorflow_status
    st = tensorflow_status()
    if not st["ok"]:
        raise RuntimeError(st["message"] + " In the app: Run > Install TensorFlow, or train in Google Colab "
                           "(Run > Train in Google Colab). Otherwise: pip install \"speciesclassifiers[delphinid]\"")


def dl_acoustic(cfg):
    from speciesclassifiers.delphinid import AcousticParams
    a = dict(cfg["settings"]["acoustic"])
    fr = (float(a.pop("fmin_hz")), float(a.pop("fmax_hz")))
    return AcousticParams(voc_type=cfg["data"]["voc_type"], freq_range=fr, **a)


def dl_frames_multi(p, cfg):
    import shutil
    d = cfg["data"]
    out = p.stage_dir("frames")
    for f in [*out.glob("frames_*.csv"), *out.glob("params_*.json")]:
        f.unlink()
    g = cfg["labels"].get("group_col") or "event_id"
    counts = []
    for ty in dl_types(cfg):
        if d.get(f"{ty}_source", "binaries") == "frames":
            shutil.copy(d[f"{ty}_frames_csv"], out / f"frames_{ty}.csv")
            shutil.copy(d[f"{ty}_frames_params"], out / f"params_{ty}.json")
            frames = pd.read_csv(out / f"frames_{ty}.csv")
        else:
            from speciesclassifiers.delphinid import make_frames
            acoustic = dl_acoustic_type(cfg, ty)
            print(f"Making {ty} frames from the annotated periods...")
            frames, info = make_frames(d["binary_folder"], d["annotations"], acoustic,
                                       sample_rate=d.get("sample_rate"))
            frames.to_csv(out / f"frames_{ty}.csv", index=False)
            _write_json({"acoustic_params": acoustic.to_dict(), "sample_rate": info["sample_rate"],
                         "recording_settings": info, "binary_folder": d["binary_folder"],
                         "annotations_csv": d["annotations"]}, out / f"params_{ty}.json")
        frames["label"] = frames["label"].astype(str)
        c = frames.groupby("label").agg(groups=(g, "nunique"), frames=(g, "size"))
        counts.append(c.add_prefix(f"{ty}_"))
        print(f"{ty}s: {len(frames)} frames, {frames[g].nunique()} groups, {frames['label'].nunique()} labels")
    pd.concat(counts, axis=1).fillna(0).astype(int).rename_axis("label").reset_index().to_csv(
        out / "class_counts.csv", index=False)


def dl_frames(p, cfg):
    import shutil
    if dl_multi(cfg):
        return dl_frames_multi(p, cfg)
    d = cfg["data"]
    out = p.stage_dir("frames")
    if d["source"] == "frames":
        shutil.copy(d["frames_csv"], out / "frames.csv")
        shutil.copy(d["frames_params"], out / "params.json")
        frames = pd.read_csv(out / "frames.csv")
    else:
        from speciesclassifiers.delphinid import make_frames
        acoustic = dl_acoustic(cfg)
        print(f"Making {acoustic.voc_type} frames from the annotated periods...")
        frames, info = make_frames(d["binary_folder"], d["annotations"], acoustic, sample_rate=d.get("sample_rate"))
        frames.to_csv(out / "frames.csv", index=False)
        _write_json({"acoustic_params": acoustic.to_dict(), "sample_rate": info["sample_rate"],
                     "recording_settings": info, "binary_folder": d["binary_folder"],
                     "annotations_csv": d["annotations"]}, out / "params.json")
        if info.get("events_without_frames"):
            print(f"Events without any frames: {', '.join(map(str, info['events_without_frames']))}")
    g = cfg["labels"].get("group_col") or "event_id"
    frames["label"] = frames["label"].astype(str)
    frames.groupby("label").agg(groups=(g, "nunique"), frames=(g, "size")).reset_index().to_csv(
        out / "class_counts.csv", index=False)
    print(f"{len(frames)} frames, {frames[g].nunique()} groups, {frames['label'].nunique()} labels")


def dl_training_data(p, cfg, voc_type=None):
    """Frames, classes, acoustic params and sample rate for training (labels with < 2 groups dropped).
    ``voc_type``: one detection type's frames (several detection types)."""
    from speciesclassifiers.delphinid import AcousticParams
    sfx = f"_{voc_type}" if voc_type else ""
    frames = pd.read_csv(p.folder / "frames" / f"frames{sfx}.csv")
    frames["label"] = frames["label"].astype(str)
    g = cfg["labels"].get("group_col") or "event_id"
    if cfg["labels"].get("classes"):
        frames = frames[frames["label"].isin(cfg["labels"]["classes"])]
    per = frames.groupby("label")[g].nunique()
    few = per[per < 2].index.tolist()
    if few:
        print(f"Leaving out labels with fewer than 2 groups: {few}")
        frames = frames[~frames["label"].isin(few)]
    meta = read_json(p.folder / "frames" / f"params{sfx}.json")
    acoustic = AcousticParams(**{k: v for k, v in meta["acoustic_params"].items()
                                 if k in AcousticParams.__dataclass_fields__})
    return frames.reset_index(drop=True), sorted(frames["label"].unique()), acoustic, meta.get("sample_rate"), g


def dl_params(cfg):
    from speciesclassifiers.delphinid import CNNParams, ResamplingParams, TrainingParams
    s = cfg["settings"]
    return CNNParams(**s["cnn"]), TrainingParams(**s["training"]), ResamplingParams(**s["resampling"])


def dl_frame_ids(frames):
    return (frames["event_id"].astype(str) + "_" + frames["frame_start"].round(3).astype(str)).to_numpy()


def dl_crossval_multi(p, cfg):
    from speciesclassifiers.delphinid import cross_validate, summarise
    out = p.stage_dir("crossval")
    for f in out.glob("cv_*"):
        f.unlink()
    cnn, training, resampling = dl_params(cfg)
    n_folds = cfg["settings"]["crossval"].get("n_folds")
    summ = {}
    for ty in dl_types(cfg):
        frames, classes, acoustic, _, g = dl_training_data(p, cfg, ty)
        print(f"{ty.capitalize()}s: {len(classes)} classes, {frames[g].nunique()} groups"
              + (f", {n_folds} folds" if n_folds else ", leaving out one group at a time"))
        preds, hist = cross_validate(frames, classes, "label", g, cnn, training, resampling, n_folds=n_folds,
                                     classifier=f"{cfg.get('model_name', 'delphinID')}_{ty}", voc_type=ty)
        start = pd.Series(frames["frame_start"].to_numpy(float), index=dl_frame_ids(frames))
        preds["start"] = preds["detection_id"].map(start).to_numpy()
        preds.to_csv(out / f"cv_{ty}.csv", index=False)
        hist.to_csv(out / f"cv_history_{ty}.csv", index=False)
        s = summarise(preds)
        summ[ty] = {"frame_accuracy": s["frame_accuracy"], "event_accuracy": s["event_accuracy"],
                    "event_accuracy_by_class": s["event_accuracy_by_class"].to_dict()}
        print(f"  {ty}s: frame accuracy {s['frame_accuracy']:.0%}, event accuracy {s['event_accuracy']:.0%}")
    _write_json(summ, out / "cv_summary.json")


def dl_crossval(p, cfg):
    require_tensorflow()
    from speciesclassifiers.delphinid import cross_validate, summarise
    if dl_multi(cfg):
        return dl_crossval_multi(p, cfg)
    frames, classes, acoustic, _, g = dl_training_data(p, cfg)
    cnn, training, resampling = dl_params(cfg)
    n_folds = cfg["settings"]["crossval"].get("n_folds")
    print(f"Cross-validation: {len(classes)} classes, {frames[g].nunique()} groups"
          + (f", {n_folds} folds" if n_folds else ", leaving out one group at a time"))
    preds, hist = cross_validate(frames, classes, "label", g, cnn, training, resampling, n_folds=n_folds,
                                 classifier=cfg.get("model_name", "delphinID"), voc_type=acoustic.voc_type)
    out = p.stage_dir("crossval")
    preds.to_csv(out / "cv_predictions.csv", index=False)
    hist.to_csv(out / "cv_history.csv", index=False)
    s = summarise(preds)
    _write_json({"frame_accuracy": s["frame_accuracy"], "event_accuracy": s["event_accuracy"],
                 "event_accuracy_by_class": s["event_accuracy_by_class"].to_dict()}, out / "cv_summary.json")
    print(f"Frame accuracy {s['frame_accuracy']:.0%}, event accuracy {s['event_accuracy']:.0%}")


def dl_model_multi(p, cfg):
    from speciesclassifiers.delphinid import export_pamguard, fit_bootstrapped
    cnn, training, resampling = dl_params(cfg)
    out = p.stage_dir("model")
    for old in [*out.glob("*.zip"), *out.glob("history*.csv")]:
        old.unlink()
    base = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cfg.get("model_name") or "delphinID")
    cv = read_json(p.folder / "crossval" / "cv_summary.json") or {}
    models = {}
    for ty in dl_types(cfg):
        frames, classes, acoustic, sr, g = dl_training_data(p, cfg, ty)
        print(f"Training the final {ty} model on all {len(frames)} frames ({resampling.n_bootstraps} bootstraps)...")
        model, hist = fit_bootstrapped(frames, classes, "label", g, cnn, training, resampling, progress=True)
        hist.to_csv(out / f"history_{ty}.csv", index=False)
        info = {"cnn_params": cnn.to_dict(), "training_params": training.to_dict(),
                "resampling_params": resampling.to_dict(),
                "training_data": {"n_frames": len(frames), "group_col": g,
                                  "groups_per_class": frames.groupby("label")[g].nunique().to_dict()}}
        if ty in cv:
            info["cross_validation"] = cv[ty]
        z = export_pamguard(model, acoustic, classes, out / f"{base}_{ty}", sample_rate=sr, info=info)
        models[ty] = {"zip": Path(z).name, "classes": classes, "acoustic_params": acoustic.to_dict(),
                      "sample_rate": sr}
        print(f"Saved the PAMGuard model {z}")
        del model
    _write_json(models, out / "models.json")


def dl_model(p, cfg):
    require_tensorflow()
    from speciesclassifiers.delphinid import export_pamguard, fit_bootstrapped
    if dl_multi(cfg):
        return dl_model_multi(p, cfg)
    frames, classes, acoustic, sr, g = dl_training_data(p, cfg)
    cnn, training, resampling = dl_params(cfg)
    print(f"Training the final model on all {len(frames)} frames ({resampling.n_bootstraps} bootstraps)...")
    model, hist = fit_bootstrapped(frames, classes, "label", g, cnn, training, resampling, progress=True)
    out = p.stage_dir("model")
    for old in out.glob("*.zip"):
        old.unlink()
    hist.to_csv(out / "history.csv", index=False)
    info = {"cnn_params": cnn.to_dict(), "training_params": training.to_dict(),
            "resampling_params": resampling.to_dict(),
            "training_data": {"n_frames": len(frames), "group_col": g,
                              "groups_per_class": frames.groupby("label")[g].nunique().to_dict()}}
    cv = read_json(p.folder / "crossval" / "cv_summary.json")
    if cv and p.stage_states(cfg).get("crossval") == "done":
        info["cross_validation"] = cv
    name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cfg.get("model_name") or "delphinID")
    z = export_pamguard(model, acoustic, classes, out / name, sample_rate=sr, info=info)
    print(f"Saved the PAMGuard model {z}")


def dl_final(p, cfg):
    types = [t for t in dl_types(cfg) if (p.folder / "crossval" / f"cv_{t}.csv").is_file()]
    preds = pd.concat([pd.read_csv(p.folder / "crossval" / f"cv_{t}.csv") for t in types], ignore_index=True)
    final_stage(p.stage_dir("final"), preds, types, cfg["final"])


def dl_new_frames(n, cfg, voc_type, acoustic, sample_rate):
    """Frames of new, unlabelled data: from binaries (event = UTC day) or a frames CSV."""
    if n.get("source") == "frames":
        return pd.read_csv(n[f"{voc_type}_frames_csv"])
    from speciesclassifiers.delphinid import click_frames, whistle_frames
    from speciesclassifiers.io import read_clicks, read_whistles
    t0 = pd.Timestamp(n["start"], tz="UTC").timestamp() if n.get("start") else 0
    t1 = pd.Timestamp(n["end"], tz="UTC").timestamp() + 86400 if n.get("end") else 4e9
    per = [(t0, t1)] if (n.get("start") or n.get("end")) else None
    if voc_type == "click":
        det, sr = read_clicks(n["binary_folder"], per, sample_rate)
    else:
        det, _ = read_whistles(n["binary_folder"], per, sample_rate)
    det = det[(det["start"] >= t0) & (det["start"] < t1)].reset_index(drop=True)
    det["event_id"] = pd.to_datetime(det["start"], unit="s", utc=True).dt.strftime("%Y%m%d")
    return click_frames(det, acoustic, sr) if voc_type == "click" else whistle_frames(det, acoustic)


def dl_frame_predictions(frames, probs, classes, acoustic, sample_rate=None):
    """New-data frame predictions, one row per frame with everything about it: start / end (Unix s and UTC),
    detections (contours or clicks) in the frame, whistle density, spectrum values (peak, lowest and highest
    frequency with energy, span), class probabilities, prediction, confidence and score."""
    from speciesclassifiers.delphinid.crossval import feature_columns
    from speciesclassifiers.delphinid.diagnostics import bin_centres, spectrum_measures
    feats = feature_columns(frames)
    P = np.asarray(probs, float)
    srt = np.sort(P, axis=1)
    t0 = frames["frame_start"].to_numpy(float)
    t1 = t0 + float(acoustic.frame_len)
    iso = lambda t: pd.to_datetime(t, unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"  # noqa: E731
    out = pd.DataFrame({"frame_start": t0, "frame_end": t1, "start_utc": iso(t0), "end_utc": iso(t1),
                        "day": frames["event_id"].astype(str).to_numpy()})
    for c in ("n_detections", "density", "file"):
        if c in frames:
            out[c] = frames[c].to_numpy()
    meas = spectrum_measures(frames[feats].to_numpy(float), bin_centres(acoustic, sample_rate, len(feats)),
                             acoustic.voc_type)
    out = pd.concat([out, meas.reset_index(drop=True)], axis=1)
    for k, c in enumerate(classes):
        out[f"p_{c}"] = P[:, k]
    out["predicted"] = np.array(classes)[P.argmax(1)]
    out["confidence"] = srt[:, -1]
    out["score"] = srt[:, -1] * (srt[:, -1] - (srt[:, -2] if P.shape[1] > 1 else 0))
    return out.sort_values("frame_start", kind="mergesort").reset_index(drop=True)


def dl_blocks(frames, probs=None, classes=None, block_minutes=1, min_frames=1, min_score=0.0, min_density=None,
              min_detections=None):
    """Frames -> clock-aligned blocks: mean probability per class over the block's kept frames, the top class,
    its score (top x lead over the second) and the decision ("uncertain" below ``min_score`` or with fewer
    than ``min_frames`` kept frames). ``frames`` is a table from ``dl_frame_predictions`` (then ``probs`` and
    ``classes`` come from its ``p_`` columns), or frames with ``probs``. Frames below ``min_density`` (whistle
    density) or ``min_detections`` are left out first."""
    from speciesclassifiers.randomforest.model import decision_score
    if probs is None:
        classes = [c[2:] for c in frames.columns if c.startswith("p_")]
        probs = frames[[f"p_{c}" for c in classes]].to_numpy(float)
    keep = np.ones(len(frames), bool)
    if min_density is not None and "density" in frames:
        keep &= frames["density"].to_numpy(float) >= min_density
    if min_detections is not None and "n_detections" in frames:
        keep &= frames["n_detections"].to_numpy(float) >= min_detections
    blk = 60 * float(block_minutes)
    b = np.floor(frames["frame_start"].to_numpy(float)[keep] / blk) * blk
    P = pd.DataFrame(np.asarray(probs, float)[keep], columns=classes)
    P["block_start"] = b
    g = P.groupby("block_start")
    out = g[classes].mean()
    out.insert(0, "n_frames", g.size())
    out.insert(0, "block_end", out.index + blk)
    out = out.reset_index()
    M = out[classes].to_numpy()
    out["predicted"] = np.array(classes)[M.argmax(1)] if len(out) else []
    out["score"] = decision_score(M) if len(out) else []
    ok = (out["score"] >= min_score) & (out["n_frames"] >= min_frames)
    out["decision"] = np.where(ok, out["predicted"], "uncertain")
    out.insert(2, "start_utc", pd.to_datetime(out["block_start"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S"))
    return out.rename(columns={c: f"p_{c}" for c in classes})


def dl_apply_single(p, cfg):
    """New data with one detection type: the PAMGuard model predicts each frame (the training resolution);
    blocks and frame filters are chosen afterwards on the Classify screen."""
    from speciesclassifiers.delphinid import load_pamguard_zip
    from speciesclassifiers.delphinid.crossval import feature_columns, to_standard_output
    from speciesclassifiers.delphinid.export import cleanup
    zips = sorted((p.folder / "model").glob("*.zip"))
    if not zips:
        raise RuntimeError("No trained model: run the model stage first.")
    _, classes, acoustic, sr, _ = dl_training_data(p, cfg)
    n = {**dl_defaults()["new_data"], **(cfg.get("new_data") or {})}
    print(f"Making {acoustic.voc_type} frames of the new data...")
    frames = dl_new_frames(n, cfg, acoustic.voc_type, acoustic, sr)
    out = p.stage_dir("apply")
    for f in out.glob("*.csv"):
        f.unlink()
    if not len(frames):
        raise ValueError(f"No {acoustic.voc_type} frames in the new data (check the folder, dates and frame settings).")
    fn, _ = load_pamguard_zip(zips[0])
    try:
        probs = fn(frames[feature_columns(frames)].to_numpy("float32"))
    finally:
        cleanup(fn)
    dl_frame_predictions(frames, probs, classes, acoustic, sr).to_csv(out / "frame_predictions.csv", index=False)
    pr = to_standard_output(frames, probs, classes, cfg.get("model_name", "delphinID"), acoustic.voc_type)
    pr["start"] = pr["detection_id"].map(pd.Series(frames["frame_start"].to_numpy(float),
                                                     index=dl_frame_ids(frames))).to_numpy()
    pr.to_csv(out / "frame_predictions_standard.csv", index=False)
    print(f"{len(frames)} frames of {acoustic.frame_len:g} s classified: " +
          ", ".join(f"{k} {v}" for k, v in pd.Series(np.array(classes)[np.asarray(probs).argmax(1)])
                    .value_counts().items()))


def dl_apply(p, cfg):
    """New data: each type's PAMGuard model predicts its frames; the final forest decides each block."""
    require_tensorflow()
    if not dl_multi(cfg):
        return dl_apply_single(p, cfg)
    from speciesclassifiers.delphinid import AcousticParams, load_pamguard_zip
    from speciesclassifiers.delphinid.crossval import feature_columns, to_standard_output
    from speciesclassifiers.delphinid.export import cleanup
    from speciesclassifiers.randomforest import RandomForestModel
    from speciesclassifiers.stacking import block_vectors, decide_blocks
    models = read_json(p.folder / "model" / "models.json")
    n = cfg["new_data"]
    out = p.stage_dir("apply")
    preds = []
    for ty, m in models.items():
        acoustic = AcousticParams(**{k: v for k, v in m["acoustic_params"].items()
                                     if k in AcousticParams.__dataclass_fields__})
        frames = dl_new_frames(n, cfg, ty, acoustic, m.get("sample_rate"))
        if not len(frames):
            print(f"No {ty} frames in the new data")
            continue
        fn, _ = load_pamguard_zip(p.folder / "model" / m["zip"])
        try:
            probs = fn(frames[feature_columns(frames)].to_numpy("float32"))
        finally:
            cleanup(fn)
        pr = to_standard_output(frames, probs, m["classes"], f"{cfg.get('model_name', 'delphinID')}_{ty}", ty)
        pr["start"] = pr["detection_id"].map(pd.Series(frames["frame_start"].to_numpy(float),
                                                         index=dl_frame_ids(frames))).to_numpy()
        pr.to_csv(out / f"frame_predictions_{ty}.csv", index=False)
        preds.append(pr)
        print(f"{len(frames)} new {ty} frames")
    if not preds:
        raise ValueError("No frames in the new data.")
    fm = RandomForestModel.load(p.folder / "final" / "final_model.joblib")
    bm = cfg["final"].get("block_minutes", 10)
    blocks, _ = block_vectors(pd.concat(preds, ignore_index=True), bm, list(models), by_event=False)
    res = decide_blocks(fm, blocks, bm, float((cfg.get("decision", {}).get("min_scores") or {}).get("final", 0.0)))
    res.to_csv(out / "blocks.csv", index=False)
    print(f"{len(res)} blocks: " + ", ".join(f"{k} {v}" for k, v in res["decision"].value_counts().items()))


DELPHINID = Workflow(
    "delphinid", "delphinID (deep learning)",
    "Train a delphinID neural network on detection frames (averaged spectra of whistle contours or clicks) "
    "and export it for PAMGuard's deep learning classifier.",
    [Stage("frames", "Detection frames", dl_frames, ["data", "settings.acoustic", "labels.group_col"]),
     Stage("crossval", "Cross-validation", dl_crossval,
           ["labels", "settings.cnn", "settings.training", "settings.resampling", "settings.crossval", "model_name"],
           after="frames", optional=True,
           enabled=lambda cfg: dl_multi(cfg) or bool(cfg.get("settings", {}).get("crossval", {}).get("run", True))),
     Stage("model", "Final model + PAMGuard zip", dl_model,
           ["labels", "settings.cnn", "settings.training", "settings.resampling", "model_name"], after="frames"),
     Stage("final", "Final layer: random forest per block (leave one event out)", dl_final, ["final"],
           after="crossval", optional=True, enabled=dl_multi),
     Stage("apply", "Classify new data", dl_apply, ["new_data", "decision", "final"], after="model", optional=True,
           enabled=lambda cfg: bool((cfg.get("new_data") or {}).get("binary_folder") or (
               (cfg.get("new_data") or {}).get("source") == "frames")))],
    dl_defaults, dl_example)


WORKFLOWS = {"randomforest": RANDOMFOREST, "calltypes": CALLTYPES, "delphinid": DELPHINID}
