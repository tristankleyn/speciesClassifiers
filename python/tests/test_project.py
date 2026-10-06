import json

import pandas as pd
import pytest

pytest.importorskip("sklearn")

from speciesclassifiers.project import WORKFLOWS, Project, run
from speciesclassifiers.project import checks
from speciesclassifiers.project.workflows import EXAMPLES

pytestmark = pytest.mark.skipif(not EXAMPLES.is_dir(), reason="examples/ not found")


def _rf(tmp_path):
    cfg = WORKFLOWS["randomforest"].example()
    cfg["settings"] = {"n_trees": 50}
    return Project.create(tmp_path / "rf", "randomforest", config=cfg)


def test_create_and_config(tmp_path):
    p = _rf(tmp_path)
    cfg = p.config()
    assert cfg["classifier"] == "randomforest" and cfg["settings"]["n_trees"] == 50
    assert cfg["settings"]["node_size"] == 25                     # defaults kept
    with pytest.raises(FileExistsError):
        Project.create(p.folder, "randomforest")
    assert set(p.stage_states().values()) == {"pending"}


def test_rf_run_resume_and_outdated(tmp_path):
    p = _rf(tmp_path)
    assert run(p.folder) == "finished"
    assert set(p.stage_states().values()) == {"done"}
    g = pd.read_csv(p.folder / "apply" / "new_group_predictions.csv")
    assert {"decision", "predicted", "score"} <= set(g.columns)
    p.update_config({"decision": {"min_group_score": 0.3}})      # threshold: only "apply" is affected
    assert p.stage_states() == {"data": "done", "crossval": "done", "model": "done", "apply": "outdated"}
    run(p.folder)
    log = p.log_path.read_text()
    assert "Cross-validation (leave one group out): up to date" in log
    p.update_config({"settings": {"n_trees": 60}})
    assert p.stage_states()["data"] == "done" and p.stage_states()["crossval"] == "outdated"


def test_failure_is_recorded(tmp_path):
    p = Project.create(tmp_path / "bad", "randomforest", config={"data": {"path": str(tmp_path / "missing.csv")}})
    assert run(p.folder) == "failed"
    st = p.status()
    assert st["stages"]["data"]["state"] == "failed" and "missing" in st["stages"]["data"]["error"]
    assert p.stage_states()["data"] == "failed" and not p.is_running()


def test_calltypes_run(tmp_path):
    p = Project.create(tmp_path / "ct", "calltypes", config=WORKFLOWS["calltypes"].example())
    assert run(p.folder) == "finished"
    lib = json.loads((p.folder / "library" / "library.json").read_text())
    assert lib["block_minutes"] == 5 and lib["rules"]
    assert (p.folder / "library" / "standalone.py").is_file()
    b = pd.read_csv(p.folder / "apply" / "blocks.csv")
    assert "noise" in set(b["species"]) and {"X", "Y"} <= set(b["species"])
    p.update_config({"settings": {"evaluate": {"rf_trees": 50}}})   # evaluate doesn't feed discover
    s = p.stage_states()
    assert s["evaluate"] == "outdated" and s["discover"] == "done" and s["library"] == "done"


def test_checks():
    r = checks.check_rf_data({**WORKFLOWS["randomforest"].defaults()["data"],
                              "path": str(EXAMPLES / "RoccaContourStats_example_synthetic.csv")})
    assert r["ok"] and len(r["info"]["classes"]) == 2
    assert not checks.check_rf_data({"type": "rocca", "path": "nowhere.csv"})["ok"]
    r = checks.check_contours("json", str(EXAMPLES / "contours_example_synthetic.json.gz"))
    assert r["ok"] and len(r["table"]) == 4
    r = checks.check_annotations({"source": "csv", "path": str(EXAMPLES / "call_annotations_example_synthetic.csv")})
    assert r["ok"] and set(r["info"]["other_labels"]) == {"species", "ecotype"}


def test_cli(tmp_path):
    from speciesclassifiers.__main__ import main
    main(["new", str(tmp_path / "c"), "--type", "calltypes", "--example"])
    assert Project(tmp_path / "c").config()["data"]["source"] == "json"


def test_app_imports():
    pytest.importorskip("nicegui")
    from speciesclassifiers.app import main as app_main
    from speciesclassifiers.app.settings import SETTINGS
    assert callable(app_main) and set(SETTINGS) == set(WORKFLOWS)


def test_delphinid_frames_and_colab_roundtrip(tmp_path):
    """Frames stage, Colab bundle, and importing a results zip (no TensorFlow needed)."""
    import zipfile

    from speciesclassifiers.project import colab
    p = Project.create(tmp_path / "dl", "delphinid", config=WORKFLOWS["delphinid"].example())
    assert run(p.folder, ["frames"]) == "finished"
    assert p.stage_states()["frames"] == "done"
    r = checks.check_dl_data(p.config()["data"])
    assert r["ok"] and len(r["info"]["classes"]) == 3
    b = colab.make_bundle(p.folder)
    with zipfile.ZipFile(b) as z:
        meta = json.loads(z.read("bundle.json"))
        assert {"frames.csv", "params.json"} <= set(z.namelist())
    assert meta["settings"]["N_FOLDS"] == 4 and meta["settings"]["EPOCHS"] == 10
    res = tmp_path / "results.zip"              # what notebook 02 writes
    with zipfile.ZipFile(res, "w") as z:
        z.writestr("results.json", json.dumps({"project": meta["project"], "fingerprints": meta["fingerprints"]}))
        z.writestr("delphinID_example.zip", b"model")
        z.writestr("cv_predictions.csv", "detection_id,event_id,classifier,voc_type,class,probability\n")
    msgs = colab.import_results(p.folder, res)
    assert len(msgs) == 2 and p.stage_states() == {"frames": "done", "crossval": "done", "model": "done"}
    p.update_config({"settings": {"training": {"epochs": 11}}})       # settings changed after training
    assert p.stage_states()["model"] == "outdated"


def test_tensorflow_status():
    st = checks.tensorflow_status()
    assert set(st) == {"ok", "installed", "version", "keras", "message"}
    assert checks._version_tuple("2.15.1") < checks.TF_MIN <= checks._version_tuple("2.16.0rc0")


def test_delphinid_train(tmp_path):
    pytest.importorskip("tensorflow")
    cfg = WORKFLOWS["delphinid"].example()
    cfg["settings"] = {"training": {"epochs": 2}, "resampling": {"n_bootstraps": 1}, "crossval": {"n_folds": 2}}
    p = Project.create(tmp_path / "dlt", "delphinid", config=cfg)
    assert run(p.folder) == "finished"
    assert list((p.folder / "model").glob("*.zip")) and (p.folder / "crossval" / "cv_predictions.csv").is_file()
