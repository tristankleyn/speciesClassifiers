import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from speciesclassifiers.calltypes import (ContourParams, DiscoverParams, ReliabilityParams, WindowParams, label_windows,
                                          make_windows, prepare_contours, save_library, window_features)
from speciesclassifiers.calltypes import discovery_pool, standalone
from speciesclassifiers.calltypes.pathway import (build_pathway_library, check_pathway, classify_contours_pathway,
                                                  discover_pathway, layer_summaries, node_labels)
from speciesclassifiers.io import read_annotations

EX = Path(__file__).parents[2] / "examples"
DAY4 = pd.Timestamp("2024-06-04", tz="UTC").timestamp()


@pytest.fixture(scope="module")
def data():
    with gzip.open(EX / "contours_example_synthetic.json.gz", "rt") as f:
        raw = json.load(f)
    calls = read_annotations(EX / "call_annotations_example_synthetic.csv")
    cp, wp = ContourParams(), WindowParams()
    c, pts = prepare_contours(pd.DataFrame(raw), cp)
    w = label_windows(window_features(make_windows(c, wp), c, pts, wp), calls, wp,
                      carry=("species", "ecotype", "event_id"))
    pool = discovery_pool(w, ["20240601", "20240602", "20240603"])
    return raw, pool, cp, wp


def test_node_labels_and_checks():
    w = pd.DataFrame({"species": ["Dd", "Sco", "pilot", "Gg/Sco", "none"]})
    node = {"name": "g", "column": "species", "groups": {"A": ["Dd", "Sco"], "B": ["pilot"]}, "exclude": ["Gg/Sco"]}
    assert list(node_labels(w, node)) == ["A", "A", "B", "none", "none"]
    assert check_pathway(node) == []
    bad = {"name": "g", "column": "species", "groups": {"A": ["Dd"], "B": ["Dd"]}, "children": {"C": {}}}
    assert len(check_pathway(bad)) == 3


def test_two_layer_pathway(data, tmp_path):
    raw, pool, cp, wp = data
    tree = {"name": "species", "column": "species", "min_score": 0.05,
            "children": {"X": {"name": "ecotype", "column": "ecotype", "exclude": ["n/a"]}}}
    pr = discover_pathway(pool, tree, DiscoverParams(min_cluster_size=10),
                          ReliabilityParams(unit="event_id", block_minutes=5))
    s = layer_summaries(pr)
    assert list(s["layer"]) == ["species", "ecotype"] and list(s["within"]) == ["(all)", "X"]
    assert s.loc[1, "classes"] == "E1, E2"
    lib = build_pathway_library(pr, cp, wp, block_minutes=5)
    assert set(lib["children"]) == {"X"} and lib["children"]["X"]["layer"]["path"] == ["X"]
    day4 = pd.DataFrame([x for x in raw if x["start"] >= DAY4])
    w, b, _ = classify_contours_pathway(day4, lib)
    x = b[b["species"] == "X"]
    assert len(x) and x["path"].str.startswith("X > ").any() and set(x["decision"]) <= {"X", "E1", "E2"}
    assert (b.loc[b["species"] == "Y", "decision"] == "Y").all()
    path = save_library(lib, tmp_path / "lib.json")
    w2, b2 = standalone.classify(day4.to_dict("records"), standalone.load_library(path))
    b2 = pd.DataFrame(b2)
    assert list(b2["decision"]) == list(b["decision"]) and list(b2["path"]) == list(b["path"])
    assert np.allclose(pd.to_numeric(b2["ecotype_score"]).fillna(-1), pd.to_numeric(b["ecotype_score"]).fillna(-1))


def test_grouped_layer(data):
    _, pool, cp, wp = data
    tree = {"name": "group", "column": "label", "groups": {"G1": ["A", "D"], "G2": ["B", "C"]},
            "children": {"G1": {"name": "calltype", "column": "label", "classes": ["A", "D"]}}}
    pr = discover_pathway(pool, tree, DiscoverParams(min_cluster_size=10), ReliabilityParams(unit="event_id"))
    m = pr["result"]["multiclass"]["group"]
    assert m["classes"] == ["G1", "G2"] and m["unit_performance"]["accuracy"] > 0.8
    assert pr["children"]["G1"]["result"]["multiclass"]["calltype"]["classes"] == ["A", "D"]
