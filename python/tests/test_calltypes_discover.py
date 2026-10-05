import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from speciesclassifiers.calltypes import (DiscoverParams, LabelTask, ReliabilityParams, discover, discovery_pool,
                                          task_stats)
from speciesclassifiers.calltypes.windows import label_windows_by_overlap

T0 = pd.Timestamp("2024-06-01", tz="UTC").timestamp()


def _windows(seed=0):
    """3 call types (A, B on species X; C on species Y) over 4 days, plus untyped and unannotated windows."""
    rng = np.random.default_rng(seed)
    proto = {"A": (6000, 0.2, 1.0), "B": (11000, -0.6, 0.5), "C": (3000, 0.9, 2.0)}
    rows = []
    for d in range(4):
        day = f"2024060{d + 1}"
        for k in range(60):
            lab = "ABC"[k % 3] if k % 5 else "untyped"
            p = proto["ABC"[k % 3]]
            t = T0 + d * 86400 + k * 30
            rows.append({"w_start": t, "w_end": t + 4, "day": day, "label": lab if d < 3 else "untyped",
                         "species": "Y" if "ABC"[k % 3] == "C" else "X",
                         "mean_freq_hz_mean": p[0] + rng.normal(0, 400), "slope": p[1] + rng.normal(0, 0.1),
                         "lc_duration_s": p[2] + rng.normal(0, 0.1), "n_contours": int(rng.integers(2, 5)),
                         "noise_feat": rng.normal()})
    return pd.DataFrame(rows)


def test_pool():
    w = _windows()
    pool = discovery_pool(w, annotated_days=["20240601", "20240602", "20240603"])
    assert set(pool["label"]) == {"A", "B", "C", "unlabelled"}
    assert (pool.loc[pool["day"] == "20240604", "label"] == "unlabelled").all()
    pool2 = discovery_pool(w, annotated_days=["20240601"], include_unannotated_days=False)
    assert set(pool2["day"]) == {"20240601"}


def test_discover_end_to_end():
    pool = discovery_pool(_windows(), annotated_days=["20240601", "20240602", "20240603"])
    task = LabelTask("species", "species", "X", "Y")
    r = discover(pool, DiscoverParams(cosine_threshold=0.5, min_cluster_size=10, new_min_n=5),
                 [task], ReliabilityParams(prior_strength=2))
    assert r["scorecard"]["n_clusters"] >= 2 and r["scorecard"]["calltype_ami"] > 0.5
    rel = r["reliability"]
    assert set(rel["species_verdict"].dropna()) <= {"X", "Y", "uncertain"}
    assert (rel["species_verdict"] != "uncertain").sum() >= 2          # clusters separate X from Y
    assert r["scorecard"]["species_loo_coverage"] > 0.5 and r["scorecard"]["species_skill"] > 0.5
    assert {"window", "5min"} <= set(r["decision_windows"]["decision_window"])
    assert len(r["examples"]) and "candidate_new" in r["summary"]
    assert {"mean", "scale", "weights", "euclid_thr", "spec"} <= set(r["state"])


def test_verdict_needs_days():
    w = pd.DataFrame({"cluster": [0] * 40 + [1] * 40, "day": ["d1"] * 80,
                      "sp": ["X"] * 38 + ["Y"] * 2 + ["Y"] * 38 + ["X"] * 2})
    st = task_stats(w, LabelTask("sp", "sp", "X", "Y"), ReliabilityParams(min_label_days=2))
    assert set(st["sp_verdict"]) == {"uncertain"}                       # one day only
    st1 = task_stats(w, LabelTask("sp", "sp", "X", "Y"), ReliabilityParams(min_label_days=1))
    assert list(st1["sp_verdict"]) == ["X", "Y"]


def test_label_by_overlap():
    w = pd.DataFrame({"w_start": [0.0, 10.0, 20.0, 30.0], "w_end": [4.0, 14.0, 24.0, 34.0], "day": "d"})
    calls = pd.DataFrame({"start": [1, 2, 11, 21, 22], "end": [2, 3, 12, 22, 23],
                          "eco": ["S", "U", "U", "S", "T"]})
    out = label_windows_by_overlap(w, calls, "eco", value_map={"S": "SRKW", "T": "Biggs"}, other="unknown",
                                   weak_values=("unknown",))
    assert list(out["eco"]) == ["SRKW", "unknown", "mixed", "none"]
