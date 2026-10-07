import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from speciesclassifiers.calltypes import (ContourParams, DiscoverParams, LabelTask, ReliabilityParams, WindowParams,
                                          build_library, classify_contours, discover, discovery_pool,
                                          make_windows, prepare_contours, save_library, window_features,
                                          with_min_scores)
from speciesclassifiers.calltypes import multiclass as mc
from speciesclassifiers.calltypes import standalone
from speciesclassifiers.calltypes.windows import label_windows_by_period
from speciesclassifiers.io import read_annotations

EX = Path(__file__).parents[2] / "examples"


def _w(seed=0):
    """3 species over 9 events; clusters 0/1/2 mostly one species each, cluster 3 mixed."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(9):
        sp = "ABC"[e % 3]
        for k in range(40):
            c = {"A": 0, "B": 1, "C": 2}[sp] if rng.random() < 0.8 else 3
            rows.append({"cluster": c, "species": sp, "event_id": f"e{e}", "day": f"d{e % 4}",
                         "w_start": 1e9 + e * 1e4 + k * 2})
    return pd.DataFrame(rows)


def test_class_stats_verdicts():
    w = _w()
    rp = ReliabilityParams(unit="event_id", min_label_days=2)
    st = mc.class_stats(w, LabelTask("species", "species"), rp)
    assert list(st["species_verdict"]) == ["A", "B", "C", "uncertain"]
    p = st[[f"species_p_{c}" for c in "ABC"]].to_numpy()
    assert np.allclose(p.sum(axis=1), 1)
    st1 = mc.class_stats(w[w["event_id"].isin(["e0", "e1", "e2"])], LabelTask("species", "species"), rp)
    assert set(st1["species_verdict"]) == {"uncertain"}        # each species from one event only


def test_loo_and_performance():
    w = _w()
    t, rp = LabelTask("species", "species", exclude=["C"]), ReliabilityParams(unit="event_id", min_label_days=2)
    wp = mc.loo_multiclass(w, t, rp)
    assert set(wp["true"]) == {"A", "B"} and set(wp["unit"]) == {f"e{i}" for i in (0, 1, 3, 4, 6, 7)}
    units = mc.aggregate(wp, ["A", "B"], "unit")
    perf = mc.performance(units, ["A", "B"])
    assert perf["accuracy"] == 1.0 and perf["coverage"] == 1.0
    curve = mc.threshold_curve(mc.aggregate(wp, ["A", "B"], "block", 10), ["A", "B"])
    assert curve["coverage"].is_monotonic_decreasing
    dec, score = mc.decide(np.array([[0.6, 0.4], [0.5, 0.5], [np.nan, np.nan]]), ["A", "B"], 0.05)
    assert list(dec) == ["A", "uncertain", "no_data"] and np.isclose(score[0], 0.12)


def test_periods_multiclass_library_and_standalone(tmp_path):
    with gzip.open(EX / "contours_example_synthetic.json.gz", "rt") as f:
        raw = json.load(f)
    calls = read_annotations(EX / "call_annotations_example_synthetic.csv")
    per = calls.groupby("event_id").agg(start=("start", "min"), end=("end", "max"),
                                        species=("species", "first")).reset_index()
    cp, wp = ContourParams(), WindowParams()
    c, pts = prepare_contours(pd.DataFrame(raw), cp)
    w = window_features(make_windows(c, wp), c, pts, wp)
    w = label_windows_by_period(w.assign(label="none"), per, columns=("event_id", "species"))
    w.loc[w["label_source"] != "unannotated", "label"] = "untyped"
    days = sorted(pd.to_datetime(per["start"], unit="s").dt.strftime("%Y%m%d").unique())
    t = LabelTask("species", "species")
    r = discover(discovery_pool(w, days), DiscoverParams(min_cluster_size=10), [t],
                 ReliabilityParams(unit="event_id", block_minutes=5))
    assert r["scorecard"]["species_unit_accuracy"] == 1.0
    lib = build_library(r, cp, wp, [t], block_minutes=5, min_scores={"species": 0.1})
    path = save_library(lib, tmp_path / "lib.json")
    day4 = [x for x in raw if x["start"] >= pd.Timestamp("2024-06-04", tz="UTC").timestamp()]
    wins, b, _ = classify_contours(pd.DataFrame(day4), lib)
    assert {"X", "Y"} <= set(b["species"]) and "species_score" in b
    w2, b2 = standalone.classify(day4, standalone.load_library(path))
    assert list(pd.DataFrame(b2)["species"]) == list(b["species"])
    assert np.allclose(pd.DataFrame(b2)["species_score"].astype(float), b["species_score"].astype(float))
    strict = classify_contours(pd.DataFrame(day4), with_min_scores(lib, {"species": 0.99}))[1]
    assert set(strict["species"]) <= {"uncertain", "no_detections"}
