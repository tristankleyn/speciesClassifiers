"""Random forest final layer: stored forests, strict test, and the same decisions in standalone.py."""
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from speciesclassifiers.calltypes.forest import (ForestParams, block_shares, feature_names, fit_forest,
                                                 forest_proba, forest_to_dict)

EX = Path(__file__).resolve().parents[2] / "examples"
SA = Path(__file__).resolve().parents[1] / "src/speciesclassifiers/calltypes/standalone.py"


def _sa(path=SA):
    spec = importlib.util.spec_from_file_location("sa", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_stored_forest_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.random((90, 10)).astype(np.float32)
    y = np.array(list("abc"))[rng.integers(0, 3, 90)]
    m = fit_forest(X, y, ForestParams(n_trees=40))
    fd = json.loads(json.dumps(forest_to_dict(m, [str(i) for i in range(10)])))
    Xt = np.r_[X[:10], rng.random((20, 10))]
    assert np.allclose(m.predict_proba(Xt), forest_proba(fd, Xt))
    assert np.allclose(_sa().forest_proba(fd, Xt), forest_proba(fd, Xt))


def test_block_shares():
    w = pd.DataFrame({"w_start": [0, 60, 120, 700], "unit": ["e1"] * 4, "cluster": [0, 0, -1, 2]})
    sh = block_shares({"whistle": w}, 10)
    assert feature_names({"whistle": [0, 2]}) == ["whistle_c0", "whistle_c2", "whistle_unassigned"]
    first = sh.iloc[0]
    assert np.isclose(first["whistle_c0"], 2 / 3) and np.isclose(first["whistle_unassigned"], 1 / 3)
    assert first["n_whistle"] == 3 and sh.iloc[1]["whistle_c2"] == 1


def test_forest_project(tmp_path):
    from speciesclassifiers.calltypes import load_library
    from speciesclassifiers.calltypes.detectors import classify_multi
    from speciesclassifiers.project import WORKFLOWS, Project, run
    from speciesclassifiers.project.workflows import load_clicks, load_whistles
    p = Project.create(tmp_path / "ct", "calltypes", config=WORKFLOWS["calltypes"].example())
    p.update_config({"decision": {"method": "forest", "forest": {"n_trees": 100, "n_jobs": 1}}})
    assert run(p.folder, ["contours", "clicks", "windows", "discover", "forest", "library"]) == "finished"
    held = pd.read_csv(p.folder / "forest" / "heldout_species.csv")
    assert {"true", "unit", "p_X", "p_Y"} <= set(held.columns) and held["unit"].nunique() >= 5
    lib = load_library(p.folder / "library" / "library.json")
    assert lib["forest"]["layer"] == "species" and "X" in lib["forest"]["children"]
    c, k = EX / "contours_example_synthetic.json.gz", EX / "clicks_example_synthetic.csv.gz"
    wh, cl = load_whistles("json", c, "2024-06-03"), load_clicks("csv", k, "2024-06-03")
    period = (wh["start"].min() - 3600, wh["start"].max() + 3600)
    _, b = classify_multi({"whistle": wh, "click": cl}, lib, period)
    assert {"Y", "E1", "E2", "noise"} <= set(b["decision"])
    sa = _sa(p.folder / "library" / "standalone.py")
    t0 = pd.Timestamp("2024-06-03", tz="UTC").timestamp()
    with gzip.open(c, "rt") as f:
        raw = {"whistle": [x for x in json.load(f) if x["start"] >= t0],
               "click": [x for x in sa.read_clicks_csv(k) if x["start"] >= t0]}
    _, b2 = sa.classify(raw, json.loads((p.folder / "library" / "library.json").read_text()), period)
    b2 = pd.DataFrame(b2)
    assert set(b.columns) == set(b2.columns)
    for col in b.columns:
        x, y = b[col], b2[col]
        if col.endswith("_score") or "_p_" in col or col.startswith(("n_windows", "block_")):
            assert np.allclose(pd.to_numeric(x).astype(float), pd.to_numeric(y).astype(float), equal_nan=True), col
        else:
            assert (x.fillna("").astype(str) == y.fillna("").astype(str).replace("None", "")).all(), col
