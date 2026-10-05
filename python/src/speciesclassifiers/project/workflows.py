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
    return {
        "data": {"type": "rocca", "path": "", "voc_type": "whistle", "label_col": "label",
                 "group_col": "event_id", "omit": [], "classes": [], "filter_min": {}, "filter_max": {}},
        "settings": {"n_trees": 500, "mtry": None, "node_size": 25, "max_per_group": 25, "prune": 0.0, "seed": 42},
        "decision": {"min_score": 0.02, "min_group_score": 0.0},
        "new_data": {"path": ""},
    }


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


def rf_data(p, cfg):
    d = cfg["data"]
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
    data = load_pickle(p.folder / "data" / "detections.pkl")
    preds = cross_validate(data, meta["features"], meta["label_col"], meta["group_col"], RFParams(**cfg["settings"]),
                           classifier="randomforest", voc_type=cfg["data"]["voc_type"])
    preds.to_csv(p.stage_dir("crossval") / "cv_detection_predictions.csv", index=False)


def rf_model(p, cfg):
    from speciesclassifiers.randomforest import RFParams, fit
    meta = read_json(p.folder / "data" / "meta.json")
    data = load_pickle(p.folder / "data" / "detections.pkl")
    print("Training the final model on all data...")
    m = fit(data, meta["features"], meta["label_col"], meta["group_col"], RFParams(**cfg["settings"]))
    out = p.stage_dir("model")
    m.save(out / "randomforest_model.joblib")
    m.importance().rename("importance").to_csv(out / "feature_importance.csv")
    print(f"Saved {out / 'randomforest_model.joblib'}")


def rf_apply(p, cfg):
    from speciesclassifiers.randomforest import RandomForestModel, predict
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
     Stage("crossval", "Cross-validation (leave one group out)", rf_crossval, ["settings"]),
     Stage("model", "Final model", rf_model),
     Stage("apply", "Classify new data", rf_apply, ["new_data", "decision"], optional=True,
           enabled=lambda cfg: bool(cfg.get("new_data", {}).get("path")))],
    rf_defaults, rf_example)


# ============================================================================ call types ============
def ct_defaults():
    from speciesclassifiers.calltypes import ContourParams
    cp = ContourParams().to_dict()
    return {
        "data": {"source": "binaries", "path": ""},
        "labels": {"source": "csv", "path": "", "table_paths": [], "table_label_col": "Annotation",
                   "table_keep_cols": [], "other_labels": None, "tasks": [], "requires": {},
                   "include_unannotated_days": True},
        "settings": {
            "contours": cp,
            "windows": {"window_s": 4.0, "overlap": 0.5, "assign_by": "overlap", "min_contours": 2,
                        "min_occupancy": 0.0, "min_longest_contour_s": 0.0, "freq_profile_bins": 32},
            "evaluate": {"max_per_class_per_day": 150, "min_class_n": 8, "rf_trees": 300},
            "discover": {"clusterer": "agglomerative", "cosine_threshold": 0.7, "min_cluster_size": 15,
                         "euclid_outlier_q": 0.95},
            "reliability": {"prior_strength": 10, "ci": 0.9, "min_label_days": 2},
        },
        "decision": {"block_minutes": 10, "rules": []},
        "new_data": {"source": "binaries", "path": "", "start": "", "end": ""},
    }


def ct_example():
    c = str(EXAMPLES / "contours_example_synthetic.json.gz")
    return {"data": {"source": "json", "path": c},
            "labels": {"source": "csv", "path": str(EXAMPLES / "call_annotations_example_synthetic.csv"),
                       "other_labels": ["species", "ecotype"],
                       "tasks": [{"name": "species", "column": "species", "positive": "X", "negative": "Y"},
                                 {"name": "ecotype", "column": "ecotype", "positive": "E1", "negative": "E2"}],
                       "requires": {"ecotype": ["species", "X"]}},
            "settings": {"discover": {"min_cluster_size": 10}},
            "decision": {"block_minutes": 5,
                         "rules": [{"if": "block_meanfreq_range_hz", "op": "<", "value": 200, "then": "noise"}]},
            "new_data": {"source": "json", "path": c, "start": "2024-06-04", "end": ""}}


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


def load_calls(lab):
    from speciesclassifiers.io import read_annotations
    if lab["source"] == "csv":
        return read_annotations(lab["path"])
    from speciesclassifiers.io.annotation_tables import read_annotation_table
    t = read_annotation_table(list(lab["table_paths"]), label_col=lab["table_label_col"],
                              keep_cols=tuple(lab.get("table_keep_cols") or ()))
    return read_annotations(t)


def other_label_columns(calls, lab):
    if lab.get("other_labels") is not None:
        return list(lab["other_labels"])
    return [c for c in calls.columns if c not in ("event_id", "label", "start", "end", "uncertain")]


def ct_window_params(cfg):
    from speciesclassifiers.calltypes import WindowParams
    c = cfg["settings"]["contours"]
    return WindowParams(**cfg["settings"]["windows"], fmin_hz=c["fmin_hz"], fmax_hz=c["fmax_hz"])


def ct_contours(p, cfg):
    from speciesclassifiers.calltypes import ContourParams, prepare_contours
    d = cfg["data"]
    wh = load_whistles(d["source"], d["path"])
    print(f"{len(wh)} contours read; measuring...")
    c, pts = prepare_contours(wh, ContourParams(**cfg["settings"]["contours"]))
    out = p.stage_dir("contours")
    _save_pickle((c, pts), out / "contours.pkl")
    _write_json({"n_read": len(wh), "n_kept": len(c), "days": sorted(c["day"].unique().tolist()) if len(c) else []},
                out / "summary.json")
    print(f"{len(c)} of {len(wh)} contours kept")


def ct_windows(p, cfg):
    from speciesclassifiers.calltypes import label_windows, make_windows, window_features
    c, pts = load_pickle(p.folder / "contours" / "contours.pkl")
    wp = ct_window_params(cfg)
    w = window_features(make_windows(c, wp), c, pts, wp)
    calls = load_calls(cfg["labels"])
    carry = other_label_columns(calls, cfg["labels"])
    w = label_windows(w, calls, wp, carry=carry)
    days = sorted(pd.to_datetime(calls["start"], unit="s", utc=True).dt.strftime("%Y%m%d").unique())
    out = p.stage_dir("windows")
    _save_pickle(w, out / "windows.pkl")
    w.drop(columns=[x for x in w.columns if x.startswith("prof_")]).to_csv(out / "windows.csv", index=False)
    pd.crosstab(w["label"], w["day"]).to_csv(out / "labels_by_day.csv")
    _write_json({"n_windows": len(w), "annotated_days": days, "other_labels": carry, "n_calls": len(calls),
                 "label_counts": w["label"].value_counts().to_dict()}, out / "summary.json")
    print(f"{len(w)} windows; {len(calls)} annotated calls on {len(days)} days")


def ct_evaluate(p, cfg):
    from speciesclassifiers.calltypes import EvalParams, evaluate, select_rows
    w = load_pickle(p.folder / "windows" / "windows.pkl")
    ep = EvalParams(**cfg["settings"]["evaluate"])
    sel = select_rows(w, ep)
    out = p.stage_dir("evaluate")
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
    w = load_pickle(p.folder / "windows" / "windows.pkl")
    days = read_json(p.folder / "windows" / "summary.json")["annotated_days"]
    pool = discovery_pool(w, days, include_unannotated_days=cfg["labels"].get("include_unannotated_days", True))
    print(f"Clustering {len(pool)} windows...")
    tasks = ct_tasks(cfg)
    r = discover(pool, DiscoverParams(**cfg["settings"]["discover"]), tasks,
                 ReliabilityParams(**cfg["settings"]["reliability"]))
    out = p.stage_dir("discover")
    _save_pickle(r, out / "discover.pkl")
    r["summary"].to_csv(out / "clusters.csv", index=False)
    _write_json(r["scorecard"], out / "scorecard.json")
    if len(r["decision_windows"]):
        r["decision_windows"].to_csv(out / "decision_windows.csv", index=False)
    print(f"{r['scorecard']['n_clusters']} clusters, {r['scorecard']['frac_unassigned']:.0%} of windows unassigned")


def ct_library(p, cfg):
    from speciesclassifiers.calltypes import ContourParams, build_library, save_library, self_check
    r = load_pickle(p.folder / "discover" / "discover.pkl")
    lib = build_library(r, ContourParams(**cfg["settings"]["contours"]), ct_window_params(cfg), ct_tasks(cfg),
                        block_minutes=cfg["decision"]["block_minutes"], rules=cfg["decision"].get("rules") or None,
                        requires={k: tuple(v) for k, v in (cfg["labels"].get("requires") or {}).items()},
                        name=f"{cfg.get('name', 'calltypes')}")
    out = p.stage_dir("library")
    save_library(lib, out / "library.json")
    import shutil
    from speciesclassifiers.calltypes import standalone
    shutil.copy(standalone.__file__, out / "standalone.py")
    sc = self_check(r, lib)
    _write_json(sc, out / "self_check.json")
    print(f"Library: {len(lib['clusters'])} clusters; {sc['same_cluster']:.0%} of clustered training windows "
          "re-assigned to their own cluster")


def ct_apply(p, cfg):
    from speciesclassifiers.calltypes import classify_contours, load_library, standard_output
    lib = load_library(p.folder / "library" / "library.json")
    n = cfg["new_data"]
    wh = load_whistles(n["source"], n["path"], n.get("start", ""), n.get("end", ""))
    print(f"Classifying {len(wh)} contours...")
    w, blocks, _ = classify_contours(wh, lib)
    out = p.stage_dir("apply")
    w.drop(columns=[x for x in w.columns if x.startswith("prof_")]).to_csv(out / "windows.csv", index=False)
    blocks.to_csv(out / "blocks.csv", index=False)
    if len(w):
        standard_output(w, lib).to_csv(out / "window_predictions_standard.csv", index=False)
    print(f"{len(w)} windows -> {len(blocks)} decision blocks")


CALLTYPES = Workflow(
    "calltypes", "Call types",
    "Find call types in whistle contours and judge them for labels you choose (e.g. species, population), "
    "from call annotations on some days. Classifies new data in clock-aligned blocks.",
    [Stage("contours", "Read and measure contours", ct_contours, ["data", "settings.contours"]),
     Stage("windows", "Windows, features and labels", ct_windows,
           ["labels.source", "labels.path", "labels.table_paths", "labels.table_label_col",
            "labels.table_keep_cols", "labels.other_labels", "settings.windows"]),
     Stage("evaluate", "Evaluate labelled call types", ct_evaluate, ["settings.evaluate"]),
     Stage("discover", "Discover (cluster + verdicts)", ct_discover,
           ["labels.tasks", "labels.include_unannotated_days", "settings.discover", "settings.reliability"],
           after="windows"),
     Stage("library", "Build library", ct_library, ["decision", "labels.requires"]),
     Stage("apply", "Classify new data", ct_apply, ["new_data"], optional=True,
           enabled=lambda cfg: bool(cfg.get("new_data", {}).get("path")))],
    ct_defaults, ct_example)


WORKFLOWS = {"randomforest": RANDOMFOREST, "calltypes": CALLTYPES}
