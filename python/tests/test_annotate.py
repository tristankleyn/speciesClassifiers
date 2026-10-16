import numpy as np
import pandas as pd
import pytest

from speciesclassifiers.annotate import ClusterParams, LabelStore, labelled_periods
from speciesclassifiers.annotate.labels import window_ids

umap = pytest.importorskip("umap")


def blobs(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(c, 0.3, (n, 4)) for c in (0, 5, 10)])
    w = pd.DataFrame(X, columns=list("abcd"))
    w["w_start"] = 1.7e9 + np.arange(len(w)) * 2.0
    w["w_end"] = w["w_start"] + 4
    w.loc[3, "a"] = np.nan                      # missing values take the median
    return w


def test_embed_and_cluster_finds_blobs():
    from speciesclassifiers.annotate import embed_and_cluster
    w = blobs()
    m, info = embed_and_cluster(w, list("abcd"), ClusterParams(min_cluster_size=20))
    assert len(m) == len(w) and list(m.columns) == ["x", "y", "cluster"]
    assert info["n_clusters"] >= 3
    truth = np.repeat([0, 1, 2], 60)
    c = m["cluster"].to_numpy()
    assert (c >= 0).mean() > 0.9
    for k in set(c[c >= 0]):                    # no cluster mixes blobs
        assert len(set(truth[c == k])) == 1


def test_labels_store_and_periods(tmp_path):
    w = pd.DataFrame({"w_start": [0.0, 2, 4, 100, 102, 300], "w_end": [4.0, 6, 8, 104, 106, 304]}) + 1.7e9
    s = LabelStore(tmp_path / "labels" / "labels.json")
    ids = window_ids(w)
    s.set("whistle", ids[:5], "KW")
    s.set("whistle", ids[5:], "HW")
    a = labelled_periods({"whistle": w}, LabelStore(s.path), max_gap_s=60)   # reread from disk
    assert list(a["label"]) == ["KW", "KW", "HW"] and list(a["n_windows"]) == [3, 2, 1]
    assert list(labelled_periods({"whistle": w}, s, max_gap_s=120)["n_windows"]) == [5, 1]
    assert a.loc[0, "start"].startswith("2023-11-14 22:13:20")
    s.set("whistle", ids[:1], None)
    assert s.counts() == {"KW": 4, "HW": 1}
    s.clear("KW")
    assert s.counts() == {"HW": 1}
    s.clear()
    assert s.counts() == {}


def test_annotate_workflow_example(tmp_path):
    from speciesclassifiers.project import WORKFLOWS, Project
    from speciesclassifiers.project.runner import run
    from speciesclassifiers.project.workflows import an_map_file, load_pickle
    p = Project.create(tmp_path / "an", "annotate", config=WORKFLOWS["annotate"].example())
    run(p.folder)
    assert set(p.stage_states().values()) == {"done"}
    for det, unit in (("whistle", "detection"), ("click", "window")):      # the defaults
        m = load_pickle(an_map_file(p, det))
        assert {"w_start", "w_end", "x", "y", "cluster"} <= set(m.columns) and m["cluster"].max() >= 1
        assert (m["unit"] == unit).all()
    from speciesclassifiers.project.workflows import _windows_file
    w = load_pickle(_windows_file(p, "whistle"))
    assert w["unit_id"].is_unique and "cidx" in w and "duration_s" in w
    cfg = p.config()                                  # single clicks
    cfg["settings"]["units"]["click"] = "detection"
    p.save_config(cfg)
    run(p.folder)
    assert p.stage_states()["windows"] == "done"
    w = load_pickle(_windows_file(p, "click"))
    assert {"peak_hz", "ici_prev_s", "unit_id"} <= set(w.columns) and w["unit_id"].is_unique
    assert (load_pickle(an_map_file(p, "click"))["unit"] == "detection").all()


def test_unit_ids():
    from speciesclassifiers.annotate.labels import unit_ids
    assert unit_ids([1.0, 1.0, 2.0, 1.0004]) == ["1.000", "1.000_1", "2.000", "1.000_2"]


def test_map_figure_and_viewer_helpers():
    pytest.importorskip("nicegui")
    from speciesclassifiers.app.annotate_ui import distinctive, map_figure, nice
    w = blobs()
    m = pd.DataFrame({"w_start": w["w_start"], "x": w["a"].fillna(0), "y": w["b"],
                      "cluster": np.repeat([0, 1, -1], 60)})
    labels = np.array(["KW"] * 30 + [""] * 150, dtype=object)
    f = map_figure({"map": m}, labels, "cluster")
    names = [t.get("name", "") for t in f["data"]]
    assert names[:3] == ["cluster 0 (60)", "cluster 1 (60)", "unclustered (60)"]
    assert sum(len(t["customdata"]) for t in f["data"] if "customdata" in t) == len(m)
    f = map_figure({"map": m}, labels, "label")
    assert [t["name"] for t in f["data"][:2]] == ["KW (30)", "(unlabelled) (150)"]
    assert distinctive(w, list("abcd"), (m["cluster"] == 1).to_numpy(), k=2)[0] in list("abcd")
    assert nice("prof_2700_3175", {}) == "share of contour points at 2.7-3.175 kHz"
