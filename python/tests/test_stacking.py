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


def test_delphinid_multi_project(tmp_path):
    import pytest
    from speciesclassifiers.project import Project, checks, run
    from speciesclassifiers.project.workflows import dl_multi_example
    cfg = dl_multi_example()
    r = checks.check_dl_data({**cfg["data"]})
    assert r["ok"] and any("Clicks:" in m for _, m in r["messages"])
    if not checks.tensorflow_status()["ok"]:
        pytest.skip("TensorFlow not installed")
    cfg["settings"]["training"]["epochs"] = 3
    p = Project.create(tmp_path / "dl", "delphinid", config=cfg)
    assert run(p.folder) == "finished"
    assert sorted(z.name for z in (p.folder / "model").glob("*.zip")) == ["delphinID_example_click.zip",
                                                                         "delphinID_example_whistle.zip"]
    h = pd.read_csv(p.folder / "final" / "heldout_final.csv")
    assert h["unit"].nunique() == 18 and {"p_SpA", "p_SpB", "p_SpC"} <= set(h.columns)
    b = pd.read_csv(p.folder / "apply" / "blocks.csv")
    assert len(b) == 18 and {"n_whistle", "n_click", "decision"} <= set(b.columns)


def test_delphinid_multi_colab_roundtrip(tmp_path, monkeypatch):
    """Bundle -> run_bundle (as notebook 02b) -> import: crossval, model and final count as done."""
    import pytest
    from speciesclassifiers.project import Project, checks, colab, run
    from speciesclassifiers.project.workflows import dl_multi_example
    if not checks.tensorflow_status()["ok"]:
        pytest.skip("TensorFlow not installed")
    cfg = dl_multi_example()
    cfg["settings"]["training"]["epochs"] = 2
    cfg["settings"]["resampling"]["n_bootstraps"] = 1
    p = Project.create(tmp_path / "dl", "delphinid", config=cfg)
    assert run(p.folder, ["frames"]) == "finished"
    b = colab.make_bundle(p.folder)
    assert colab.notebook_url(p.config()) == colab.COLAB_URL_TYPES
    monkeypatch.chdir(tmp_path)
    res = colab.run_bundle(b, tmp_path / "work")
    msgs = colab.import_results(p.folder, res)
    assert len(msgs) == 3
    s = p.stage_states()
    assert s["crossval"] == s["model"] == s["final"] == "done"
    assert len(list((p.folder / "model").glob("*.zip"))) == 2
    assert run(p.folder, ["apply"]) == "finished"            # nothing retrained; new data classified
    assert "Final model + PAMGuard zip: up to date" in p.log_path.read_text()


def test_delphinid_diagnostics():
    """Frames joined to held-out predictions; filters change coverage and accuracy as expected."""
    import json
    from pathlib import Path

    from speciesclassifiers.delphinid import AcousticParams
    from speciesclassifiers.delphinid.diagnostics import apply_filters, bin_centres, frame_table, summary
    ex = Path(__file__).resolve().parents[2] / "examples"
    frames = pd.read_csv(ex / "frames_example_synthetic.csv")
    meta = json.loads((ex / "frames_example_synthetic_params.json").read_text())
    ap = AcousticParams(**{k: v for k, v in meta["acoustic_params"].items() if k in AcousticParams.__dataclass_fields__})
    c = bin_centres(ap, meta["sample_rate"])
    assert len(c) == ap.n_bins() and c[0] == 2100.0
    rng = np.random.default_rng(0)
    classes = ["SpA", "SpB", "SpC"]
    rows = []
    for _, f in frames.iterrows():
        p = rng.dirichlet([1, 1, 1]) * 0.5 + 0.5 * (np.array(classes) == f["label"])
        for cl, v in zip(classes, p):
            rows.append({"detection_id": f"{f['event_id']}_{round(f['frame_start'], 3)}", "event_id": f["event_id"],
                         "classifier": "d", "voc_type": "whistle", "class": cl, "probability": v,
                         "true_class": f["label"]})
    t = frame_table(frames, pd.DataFrame(rows), ap, meta["sample_rate"])
    assert len(t) == len(frames) and t["peak_freq_hz"].notna().all() and set(t.attrs["classes"]) == set(classes)
    s_all = summary(t, *apply_filters(t))
    assert s_all["frames_kept"] == len(t) and s_all["events_decided"] == 18
    k, ev = apply_filters(t, {"confidence": (0.7, None)}, {"min_frames": 10})
    s = summary(t, k, ev)
    assert s["frames_kept"] < len(t) and (k["confidence"] >= 0.7).all()
    assert s["frame_accuracy"] >= s_all["frame_accuracy"] and s["events_decided"] < 18
    assert s["frame_confusion"].to_numpy().sum() == s["frames_kept"]


def test_delphinid_classify_new_single(tmp_path):
    """One detection type: the trained zip classifies new frames in clock-aligned blocks."""
    import pytest
    from speciesclassifiers.project import WORKFLOWS, Project, checks, run
    from speciesclassifiers.project.workflows import EXAMPLES, dl_blocks
    P = np.array([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8]])
    f = pd.DataFrame({"frame_start": [0.0, 30.0, 70.0]})
    b = dl_blocks(f, P, ["A", "B"], block_minutes=1, min_frames=2)
    assert list(b["n_frames"]) == [2, 1] and np.isclose(b["p_A"][0], 0.8)
    assert list(b["decision"]) == ["A", "uncertain"]
    if not checks.tensorflow_status()["ok"]:
        pytest.skip("TensorFlow not installed")
    cfg = WORKFLOWS["delphinid"].example()
    cfg["settings"].update({"training": {"epochs": 2}, "resampling": {"n_bootstraps": 1},
                            "crossval": {"run": False, "n_folds": 2}})
    cfg["new_data"] = {"source": "frames", "whistle_frames_csv": str(EXAMPLES / "frames_example_synthetic.csv"),
                       "binary_folder": "", "block_minutes": 5}
    p = Project.create(tmp_path / "dl1", "delphinid", config=cfg)
    assert p.stage_states()["apply"] == "pending"
    assert run(p.folder) == "finished"
    b = pd.read_csv(p.folder / "apply" / "blocks.csv")
    assert len(b) >= 18 and {"decision", "n_frames", "p_SpA"} <= set(b.columns)      # events may span blocks
