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
