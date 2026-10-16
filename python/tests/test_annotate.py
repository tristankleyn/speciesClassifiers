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
    for det in ("whistle", "click"):
        m = load_pickle(an_map_file(p, det))
        assert {"w_start", "w_end", "x", "y", "cluster"} <= set(m.columns) and m["cluster"].max() >= 1
