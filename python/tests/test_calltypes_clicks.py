"""Clicks as a detection type, feature choice, and merging detection types."""
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from speciesclassifiers.calltypes import multiclass as mc
from speciesclassifiers.calltypes.clicks import ClickWindowParams, click_windows, prepare_clicks
from speciesclassifiers.calltypes.evaluate import fit_feature_spec
from speciesclassifiers.calltypes.windows import default_features, feature_catalog, feature_columns

EX = Path(__file__).resolve().parents[2] / "examples"


def test_click_windows_ici():
    t0 = 1_717_200_000.0
    starts = np.r_[t0 + np.arange(20) * 0.1, t0 + 5 + np.arange(10) * 0.2]   # two trains, gap 3.1 s
    w = click_windows(prepare_clicks(pd.DataFrame({"start": starts})),
                      ClickWindowParams(window_s=10, overlap=0.5, min_clicks=5, max_ici_s=1.0))
    full = w[w["n_clicks"] == 30].iloc[0]
    assert full["n_intervals"] == 28                      # the 3.1 s gap is not an interval
    assert full["ici_median_s"] == pytest.approx(0.1, abs=1e-5) and full["ici_p95_s"] == pytest.approx(0.2, abs=1e-5)
    assert full["ici_mean_s"] == pytest.approx((19 * 0.1 + 9 * 0.2) / 28, abs=1e-5)


def test_feature_choice():
    keys = {f["key"] for f in feature_catalog("whistle")}
    assert "snr_db_mean" in keys and "snr_db_mean" not in default_features("whistle")
    assert default_features("click") == ["ici_mean_s", "ici_median_s", "ici_std_s", "ici_p05_s", "ici_p95_s"]
    cols = ["n_contours", "duration_s_mean", "prof_800_1275", "prof_1275_1750", "snr_db_mean"]
    assert feature_columns(cols, ["duration_s_mean", "freq_profile"]) == ["duration_s_mean", "prof_800_1275",
                                                                          "prof_1275_1750"]
    rng = np.random.default_rng(0)
    w = pd.DataFrame(rng.normal(size=(30, 3)), columns=["n_contours", "duration_s_mean", "snr_db_mean"])
    assert fit_feature_spec(w, features=["duration_s_mean"])["cols"] == ["duration_s_mean"]
    assert "snr_db_mean" not in fit_feature_spec(w, exclude_features=["snr"])["cols"]


def test_merge_evidence():
    a = pd.DataFrame({"X": [0.8, 0.5, np.nan], "Y": [0.2, 0.5, np.nan]})
    b = pd.DataFrame({"X": [0.6, np.nan, 0.3], "Z": [0.4, np.nan, 0.7]})
    m = mc.merge_evidence({"w": a, "c": b}, {"w": {"X": .5, "Y": .5}, "c": {"X": .5, "Z": .5}})
    assert list(m.columns) == ["X", "Y", "Z"]
    # block 0: X 1.6*1.2, Y 0.4*1 (unknown to c), Z 1*0.8 (unknown to w)
    e = np.array([1.6 * 1.2, 0.4, 0.8])
    assert np.allclose(m.iloc[0], e / e.sum())
    assert np.allclose(m.iloc[1], [0.5, 0.5, 0.0])       # only whistles: their probabilities
    assert np.allclose(m.iloc[2], [0.3, 0.0, 0.7])       # only clicks
    w2 = mc.merge_evidence({"w": a, "c": b}, {"w": {"X": .5, "Y": .5}, "c": {"X": .5, "Z": .5}}, {"c": 0.0})
    assert np.allclose(w2.iloc[0], np.array([1.6, 0.4, 1.0]) / 3.0)   # weight 0: clicks ignored


def test_standalone_matches_multi(tmp_path):
    from speciesclassifiers.calltypes import load_library
    from speciesclassifiers.calltypes.detectors import classify_multi
    from speciesclassifiers.project import WORKFLOWS, Project, run
    from speciesclassifiers.project.workflows import load_clicks, load_whistles
    p = Project.create(tmp_path / "ct", "calltypes", config=WORKFLOWS["calltypes"].example())
    assert run(p.folder, ["contours", "clicks", "windows", "discover", "library"]) == "finished"
    lib = load_library(p.folder / "library" / "library.json")
    c, k = EX / "contours_example_synthetic.json.gz", EX / "clicks_example_synthetic.csv.gz"
    wh, cl = load_whistles("json", c, "2024-06-03"), load_clicks("csv", k, "2024-06-03")
    period = (wh["start"].min() - 3600, wh["start"].max() + 3600)
    _, b = classify_multi({"whistle": wh, "click": cl}, lib, period)
    spec = importlib.util.spec_from_file_location("sa", p.folder / "library" / "standalone.py")
    sa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sa)
    t0 = pd.Timestamp("2024-06-03", tz="UTC").timestamp()
    with gzip.open(c, "rt") as f:
        raw = {"whistle": [x for x in json.load(f) if x["start"] >= t0],
               "click": [x for x in sa.read_clicks_csv(k) if x["start"] >= t0]}
    _, b2 = sa.classify(raw, json.loads((p.folder / "library" / "library.json").read_text()), period)
    b2 = pd.DataFrame(b2)
    assert len(b) == len(b2) and set(b.columns) == set(b2.columns)
    for col in b.columns:
        x, y = b[col], b2[col]
        if col.endswith("_score") or "_p_" in col or col.startswith(("n_windows", "block_")):
            assert np.allclose(pd.to_numeric(x).astype(float), pd.to_numeric(y).astype(float), equal_nan=True), col
        else:
            assert (x.fillna("").astype(str) == y.fillna("").astype(str).replace("None", "")).all(), col
    assert {"Y", "E1", "E2", "noise"} <= set(b["decision"])
