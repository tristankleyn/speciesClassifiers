"""Final layer over base classifiers of several detection types (random forest method from detections)."""
import numpy as np
import pandas as pd

from speciesclassifiers.stacking import block_vectors, heldout_rows, type_rows


def _preds(ty, rows):
    out = []
    for i, (ev, t, true, px) in enumerate(rows):
        for c, p in (("X", px), ("Y", 1 - px)):
            out.append({"detection_id": f"{ty}{i}", "event_id": ev, "start": t, "classifier": f"rf-{ty}",
                        "voc_type": ty, "class": c, "probability": p, "true_class": true})
    return pd.DataFrame(out)


def test_block_vectors():
    w = _preds("whistle", [("e1", 0, "X", 0.8), ("e1", 60, "X", 0.6), ("e2", 3000, "Y", 0.1)])
    k = _preds("click", [("e1", 30, "X", 0.9), ("e1", 700, "X", 0.7)])
    b, feats = block_vectors(pd.concat([w, k]), 10, ["whistle", "click"])
    assert feats == ["whistle|X", "whistle|Y", "click|X", "click|Y", "has_whistle", "has_click"]
    b = b.set_index("block_id")
    first = b.loc["e1@0"]
    assert np.isclose(first["whistle|X"], 0.7) and np.isclose(first["click|X"], 0.9)
    assert first["has_whistle"] == 1 and first["n_whistle"] == 2 and first["n_click"] == 1
    later = b.loc["e1@600"]                         # clicks only: whistle probabilities 0, flag 0
    assert later["has_whistle"] == 0 and later["whistle|X"] == 0 and later["click|X"] == 0.7
    assert b.loc["e2@3000", "true_class"] == "Y"
    tr = type_rows(b.reset_index(), "whistle", ["X", "Y"]).set_index("block")
    assert np.isnan(tr.loc["e1@600", "p_X"]) and np.isclose(tr.loc["e1@0", "p_X"], 0.7)


def test_rf_binaries_project(tmp_path):
    from speciesclassifiers.project import Project, run
    from speciesclassifiers.project.workflows import rf_binaries_example
    p = Project.create(tmp_path / "rf", "randomforest", config=rf_binaries_example())
    assert run(p.folder) == "finished"
    h = pd.read_csv(p.folder / "final" / "heldout_final.csv")
    assert {"true", "unit", "w_start", "p_X", "p_Y"} <= set(h.columns) and h["unit"].nunique() == 7
    assert (p.folder / "final" / "heldout_final__click.csv").is_file()
    b = pd.read_csv(p.folder / "apply" / "blocks.csv")
    assert {"block_start", "decision", "p_X", "n_whistle", "n_click"} <= set(b.columns) and len(b)
    assert p.stage_states()["final"] == "done"
    p.update_config({"final": {"n_trees": 100}})
    s = p.stage_states()
    assert s["final"] == "outdated" and s["model"] == "done" and s["crossval"] == "done"
