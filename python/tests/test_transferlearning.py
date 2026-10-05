import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("imblearn")

from speciesclassifiers.transferlearning import (assign_events, cross_validate, event_features, event_params,
                                                 filter_events, fit, predict, summarise, threshold_curve,
                                                 time_bin_events)

EX = Path(__file__).parents[2] / "examples"
DB = os.environ.get("SC_EXAMPLE_DB")


def _preds():
    rows = []
    for ev, t0, items in [("e1", "2024-01-01T00:00:10Z", [("w", ["A", "B"], [0.8, 0.2]), ("w", ["A", "B"], [0.6, 0.4]),
                                                         ("c", ["A", "B", "C"], [0.1, 0.1, 0.8])]),
                          ("e2", "2024-01-01T01:00:10Z", [("c", ["A", "B", "C"], [0.3, 0.3, 0.4])])]:
        for k, (clf, classes, p) in enumerate(items):
            for c, v in zip(classes, p):
                rows.append({"detection_id": f"{ev}{clf}{k}", "event_id": "", "time": t0, "classifier": clf,
                             "voc_type": "other", "class": c, "probability": v})
    return pd.DataFrame(rows)


def test_assign_and_features():
    events = pd.DataFrame({"event_id": ["E1", "E2"], "label": ["X", "Y"], "encounter": ["k1", "k2"],
                           "start": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
                           "end": ["2024-01-01 00:10:00", "2024-01-01 01:10:00"]})
    a = assign_events(_preds(), events)
    ev, feats = event_features(a)
    assert feats == ["w|A", "w|B", "c|A", "c|B", "c|C"]
    e1 = ev.set_index("event_id").loc["E1"]
    assert np.isclose(e1["w|A"], 0.7) and e1["n_w"] == 2 and e1["label"] == "X" and e1["encounter"] == "k1"
    e2 = ev.set_index("event_id").loc["E2"]
    assert e2["w|A"] == 0 and e2["w|B"] == 0 and e2["n_w"] == 0  # no whistle predictions -> zeros
    assert not any(c.startswith("n_") for c in feats)
    assert len(filter_events(ev, {"w": 1, "c": 1}, "and")) == 1
    assert len(filter_events(ev, {"w": 1, "c": 1}, "or")) == 2


def test_time_bins():
    b = time_bin_events(_preds(), 30)
    assert set(b["event_id"]) == {"2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"}


def test_cross_validate_example():
    a = assign_events(pd.read_csv(EX / "base_predictions_example_synthetic.csv"), EX / "events_example_synthetic.csv")
    ev, feats = event_features(a)
    params = event_params(n_trees=200)
    preds = cross_validate(ev, feats, group_col="encounter", params=params, progress=False)
    assert preds["detection_id"].nunique() == len(ev)
    s = summarise(preds)
    assert s["event_accuracy"] > 0.6
    assert threshold_curve(preds)["events_kept"].iloc[0] == 1
    m = fit(ev, feats, group_col="encounter", params=params)
    out = predict(m, ev.drop(columns=[feats[0]]))  # a missing feature is filled with 0
    assert len(out) == len(ev) and set(out["predicted"]) <= set(m.classes)


@pytest.mark.skipif(not DB, reason="SC_EXAMPLE_DB not set")
def test_pamguard_database():
    from speciesclassifiers.io import list_prediction_tables, read_dl_predictions, read_recordings
    tables = list_prediction_tables(DB)
    assert tables["Deep_Learning_Classifier___Clicks_Predictions"] == 138
    c = read_dl_predictions(DB, "Deep_Learning_Classifier___Clicks", ["Dde", "Ggr", "Gme", "Lal", "Ttr"], "clicks", "click")
    assert c["detection_id"].nunique() == 138
    with pytest.raises(Exception):
        read_dl_predictions(DB, "Deep_Learning_Classifier___Clicks", ["a", "b"]).pipe(lambda d: d.iloc[0])
    rec = read_recordings(DB)
    assert len(rec) == 11 and (rec["end"] > rec["start"]).all()
    ev, feats = event_features(assign_events(c, rec))
    assert ev["n_clicks"].sum() == 138
