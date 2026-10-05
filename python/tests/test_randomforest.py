import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("imblearn")

from speciesclassifiers.outputs import validate_output
from speciesclassifiers.randomforest import (RandomForestModel, RFParams, cross_validate, decision_score, fit,
                                             predict, read_feature_table, read_rocca, summarise, threshold_curve)
from speciesclassifiers.randomforest.model import cap_per_group, prune_outliers

ROCCA = os.environ.get("SC_ROCCA_CSV")


def _data(n_groups=4, n=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c, mu in [("A", 0.0), ("B", 2.0)]:
        for g in range(n_groups):
            for _ in range(n):
                rows.append({"label": c, "event_id": f"{c}{g}", "x1": rng.normal(mu, 1), "x2": rng.normal(-mu, 1),
                             "noise": rng.normal()})
    return pd.DataFrame(rows)


def test_decision_score():
    assert np.allclose(decision_score([[0.7, 0.2, 0.1], [0.5, 0.5, 0]]), [0.7 * 0.5, 0])


def test_cap_and_prune():
    d = _data()
    assert cap_per_group(d, "event_id", 5).groupby("event_id").size().max() == 5
    p = prune_outliers(d, ["x1", "x2"], "label", 0.1)
    assert len(p) == pytest.approx(0.9 * len(d), abs=4)


def test_cross_validate_and_predict(tmp_path):
    d, feats = read_feature_table(_data())
    assert feats == ["x1", "x2", "noise"]
    params = RFParams(n_trees=50, node_size=5, max_per_group=10)
    preds = cross_validate(d, feats, params=params, progress=False)
    validate_output(preds.drop(columns=["score"]))
    s = summarise(preds)
    assert s["group_accuracy"] == 1.0
    assert threshold_curve(preds)["groups_kept"].iloc[0] == 1.0
    m = fit(d, feats, params=params)
    path = m.save(tmp_path / "rf.joblib")
    m2 = RandomForestModel.load(path)
    det, grp = predict(m2, d.drop(columns="label"))
    assert len(grp) == d["event_id"].nunique() and set(grp["predicted"]) == {"A", "B"}
    with pytest.raises(ValueError, match="missing features"):
        predict(m2, d.drop(columns="x1"))


def test_read_rocca_hierarchy(tmp_path):
    cols = ["Source", "EncounterID", "KnownSpecies", "FREQMAX", "FREQMIN", "DCMEAN", "DURATION", "STEPDUR", "FREQPEAK"]
    for sp in ["Dde", "Ttr"]:
        for enc in ["e1", "e2"]:
            f = tmp_path / sp / enc
            f.mkdir(parents=True)
            pd.DataFrame([["s", "x", "x", 8000, 5000, 1, 0.5, 0.01, 0]] * 3, columns=cols).to_csv(
                f / "RoccaContourStats_1.csv", index=False)
    d, feats = read_rocca(tmp_path, levels=["label", "event_id"])
    assert feats == ["FREQMAX", "FREQMIN", "DURATION", "STEPDUR"]  # DCMEAN omitted by default
    assert sorted(d["label"].unique()) == ["Dde", "Ttr"] and d["event_id"].nunique() == 2


@pytest.mark.skipif(not ROCCA, reason="SC_ROCCA_CSV not set")
def test_real_rocca_file():
    d, feats = read_rocca(ROCCA)
    assert len(feats) > 30 and d["event_id"].nunique() > 5
    preds = cross_validate(d, feats, params=RFParams(n_trees=100), progress=False)
    assert summarise(preds)["group_accuracy"] > 0.6
