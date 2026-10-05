import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from speciesclassifiers.delphinid import AcousticParams, build_model, predict
from speciesclassifiers.delphinid.export import cleanup, export_pamguard, load_pamguard_zip, make_pdtf

REF = os.environ.get("SC_PAMGUARD_TEST_DATA")


def test_export_round_trip(tmp_path):
    ap = AcousticParams.whistles()
    classes = ["Dde", "Ggr", "Ttr"]
    m = build_model(ap.n_bins(), len(classes))
    z = export_pamguard(m, ap, classes, tmp_path / "mywhistles", info={"note": "test"})
    names = zipfile.ZipFile(z).namelist()
    assert "mywhistles/saved_model.pb" in names and "mywhistles/delphinID.pdtf" in names
    fn, pdtf = load_pamguard_zip(z)
    x = np.random.default_rng(0).random((5, 90)).astype("float32")
    assert np.allclose(fn(x), predict(m, x), atol=1e-6)
    assert pdtf["class_info"]["name_class"] == classes
    assert pdtf["model_info"]["input_shape"] == [-1, 90, 1]
    info = json.loads(zipfile.ZipFile(z).read("mywhistles/speciesclassifiers_info.json"))
    assert info["note"] == "test" and info["pamguard_settings"]["segment_length_ms"] == 4000
    cleanup(fn)


def test_export_checks_inputs(tmp_path):
    m = build_model(80, 2)
    with pytest.raises(ValueError, match="sample_rate"):
        export_pamguard(m, AcousticParams.clicks(), ["a", "b"], tmp_path / "c")
    with pytest.raises(ValueError, match="inputs"):
        export_pamguard(m, AcousticParams.clicks(), ["a", "b"], tmp_path / "c", sample_rate=192000)


@pytest.mark.skipif(not REF, reason="SC_PAMGUARD_TEST_DATA not set")
@pytest.mark.parametrize("zipname,folder,ap,sr", [
    ("whistleclassifier.zip", "whistleclassifier", AcousticParams.whistles(), None),
    ("clickClassifier_Jan25.zip", "clickClassifier_Jan25", AcousticParams.clicks(), 96000),
])
def test_pdtf_matches_published(zipname, folder, ap, sr):
    pub = json.loads(zipfile.ZipFile(Path(REF) / zipname).read(f"{folder}/delphinID.pdtf"))
    ours = make_pdtf(ap, pub["class_info"]["name_class"], pub["model_info"]["input_shape"][1])
    assert set(ours) == set(pub)
    assert ours["seg_size"] == pub["seg_size"] and ours["model_info"]["input_shape"] == pub["model_info"]["input_shape"]
    strip = lambda ts: [(t["name"], {k: v for k, v in t["params"].items() if k != "minclks"}) for t in ts]
    assert strip(ours["transforms"]) == strip(pub["transforms"])
