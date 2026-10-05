import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from speciesclassifiers.calltypes import (ContourParams, DiscoverParams, LabelTask, ReliabilityParams, WindowParams,
                                          build_library, check_rules, classify_contours, discover, discovery_pool,
                                          label_windows, load_library, make_windows, prepare_contours, save_library,
                                          self_check, standard_output, window_features, with_rules)
from speciesclassifiers.calltypes import standalone
from speciesclassifiers.io import read_annotations

EX = Path(__file__).parents[2] / "examples"
DAY4 = pd.Timestamp("2024-06-04", tz="UTC").timestamp()
TASKS = [LabelTask("species", "species", "X", "Y"), LabelTask("ecotype", "ecotype", "E1", "E2")]
RULES = [{"if": "block_meanfreq_range_hz", "op": "<", "value": 200, "then": "noise"}]


@pytest.fixture(scope="module")
def built():
    with gzip.open(EX / "contours_example_synthetic.json.gz", "rt") as f:
        raw = json.load(f)
    calls = read_annotations(EX / "call_annotations_example_synthetic.csv")
    cp, wp = ContourParams(), WindowParams()
    c, pts = prepare_contours(pd.DataFrame(raw), cp)
    w = label_windows(window_features(make_windows(c, wp), c, pts, wp), calls, wp, carry=("species", "ecotype"))
    pool = discovery_pool(w, annotated_days=["20240601", "20240602", "20240603"])
    r = discover(pool, DiscoverParams(min_cluster_size=10, new_min_n=10), TASKS, ReliabilityParams(), loo=False)
    lib = build_library(r, cp, wp, TASKS, block_minutes=5, rules=RULES, requires={"ecotype": ("species", "X")})
    return raw, r, lib


def test_library_self_check(built):
    _, r, lib = built
    sc = self_check(r, lib)
    assert sc["scaler_matches"] and sc["same_cluster"] == 1.0
    v = {c["verdicts"]["species"]["verdict"] for c in lib["clusters"]}
    assert {"X", "Y"} <= v


def test_reference_blocks_rules_requires(built, tmp_path):
    raw, _, lib = built
    lib = load_library(save_library(lib, tmp_path / "lib.json"))
    new = pd.DataFrame([x for x in raw if x["start"] >= DAY4])
    w, b, _ = classify_contours(new, lib)
    assert (w["cluster"] >= 0).mean() > 0.5
    noise = b[b["rule"].notna()]
    assert len(noise) and (noise["species"] == "noise").all() and (noise["block_meanfreq_range_hz"] < 200).all()
    calls = b[b["rule"].isna()]
    assert set(calls["species"]) <= {"X", "Y", "uncertain"} and {"X", "Y"} <= set(calls["species"])
    assert (calls.loc[calls["species"] != "X", "ecotype"] == "n/a").all()       # requires
    b0 = classify_contours(new, with_rules(lib, None))[1]
    assert b0["rule"].isna().all() and (b0["species"] == b0["raw_species"]).all()
    bp = classify_contours(new, lib, period=(DAY4, DAY4 + 86400))[1]
    assert len(bp) == 288 and (bp.loc[bp["n_windows"] == 0, "species"] == "no_detections").all()


def test_standalone_matches_reference(built, tmp_path):
    raw, _, lib = built
    path = save_library(lib, tmp_path / "lib.json")
    for subset in (raw, [x for x in raw if x["start"] >= DAY4]):
        w, b, _ = classify_contours(pd.DataFrame(subset), lib)
        w2, b2 = standalone.classify(subset, standalone.load_library(path))
        w2, b2 = pd.DataFrame(w2), pd.DataFrame(b2)
        assert np.array_equal(w["w_start"], w2["w_start"]) and np.array_equal(w["cluster"], w2["cluster"])
        cols = lib["feature_spec"]["cols"]
        assert np.allclose(w[cols].astype(float), w2[cols].astype(float), rtol=1e-9, equal_nan=True)
        for c in b.columns:
            x, y = pd.to_numeric(b[c], errors="coerce"), pd.to_numeric(b2[c], errors="coerce")
            if x.notna().any():
                assert np.allclose(x.astype(float), y.astype(float), equal_nan=True), c
            else:
                assert b[c].fillna("").astype(str).tolist() == b2[c].fillna("").astype(str).tolist(), c


def test_standalone_cli(built, tmp_path):
    raw, _, lib = built
    save_library(lib, tmp_path / "lib.json")
    with gzip.open(tmp_path / "c.json.gz", "wt") as f:
        json.dump([x for x in raw if x["start"] >= DAY4], f)
    out = subprocess.run([sys.executable, standalone.__file__, str(tmp_path / "lib.json"), str(tmp_path / "c.json.gz"),
                          str(tmp_path / "blocks.csv")], capture_output=True, text=True, check=True)
    assert "blocks" in out.stdout and "noise" in set(pd.read_csv(tmp_path / "blocks.csv")["species"])


def test_contour_from_bins():
    c = standalone.contour_from_bins(100.0, [10, 11, 12], [100, 101, 102], {"sample_rate": 96000, "fft": 2048, "hop": 1024})
    assert np.allclose(c["times"], [0, 1024 / 96000, 2048 / 96000]) and np.isclose(c["freqs"][0], 100 * 96000 / 2048)


def test_check_rules():
    assert check_rules(None) == []
    with pytest.raises(ValueError):
        check_rules([{"if": "unknown", "op": "<", "value": 1, "then": "x"}])
    with pytest.raises(ValueError):
        check_rules([{"if": "n_windows", "op": "~", "value": 1, "then": "x"}])


def test_standard_output(built):
    from speciesclassifiers.outputs import validate_output
    raw, _, lib = built
    w, _, _ = classify_contours(pd.DataFrame([x for x in raw if x["start"] >= DAY4]), lib)
    out = standard_output(w, lib)
    validate_output(out)
    assert set(out["classifier"]) == {"calltypes-species", "calltypes-ecotype"}
    assert out["detection_id"].nunique() == (w["cluster"] >= 0).sum()
