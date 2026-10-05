import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from speciesclassifiers.io import label_detections, periods, read_annotations
from speciesclassifiers.io.pamguard_binary import file_start_time, infer_fft, infer_sample_rate, list_binaries

EXAMPLE = Path(__file__).parents[2] / "examples" / "annotations_example.csv"


def test_example_annotations_read():
    a = read_annotations(EXAMPLE)
    assert list(a["event_id"]) == ["enc001", "enc001", "enc002", "enc003"]
    assert a.loc[0, "start"] == pd.Timestamp("2018-09-09 21:55:00", tz="UTC").timestamp()
    assert "location" in a


def test_annotations_errors():
    with pytest.raises(ValueError, match="missing"):
        read_annotations(pd.DataFrame({"event_id": [1], "start": ["2020-01-01"], "end": ["2020-01-02"]}))
    with pytest.raises(ValueError, match="end is not after start"):
        read_annotations(pd.DataFrame({"event_id": [1], "label": ["a"], "start": ["2020-01-02"], "end": ["2020-01-01"]}))
    with pytest.raises(ValueError, match="Can't read"):
        read_annotations(pd.DataFrame({"event_id": [1], "label": ["a"], "start": ["soon"], "end": ["2020-01-01"]}))


def test_label_detections():
    a = read_annotations(EXAMPLE)
    t0 = a.loc[0, "start"]
    det = pd.DataFrame({"start": [t0 - 1, t0 + 1, t0 + 700]})
    lab = label_detections(det, a)
    assert len(lab) == 1 and lab.loc[0, "label"] == "Dde" and lab.loc[0, "location"] == "Site A"


def test_file_selection(tmp_path):
    for hhmm in ["2150", "2155", "2200", "2205", "2300"]:
        (tmp_path / f"WhistlesMoans_Contours_20180909_{hhmm}00.pgdf").touch()
    (tmp_path / "Click_Detector_Clicks_20180909_215500.pgdf").touch()
    a = read_annotations(pd.DataFrame({"event_id": ["e"], "label": ["x"],
                                       "start": ["2018-09-09 21:56:00"], "end": ["2018-09-09 22:01:00"]}))
    names = [f.name[-11:-5] for f in list_binaries(tmp_path, "Contours", periods(a))]
    assert names == ["215500", "220000"]
    assert file_start_time("x_20180909_215500.pgdf") == pd.Timestamp("2018-09-09 21:55", tz="UTC").timestamp()


def test_inference():
    ms = np.array([0, 1000, 2500]) + 1.6e12
    assert infer_sample_rate((ms - ms[0]) / 1000 * 96001, ms) == 96000
    hop, fft = infer_fft([25, 281], [25599, 287743], [26624, 21504], np.array([25, 20], dtype="int16"))
    assert (hop, fft) == (1024, 2048)


BIN = os.environ.get("SC_PAMGUARD_BINARIES")


@pytest.mark.skipif(not BIN, reason="SC_PAMGUARD_BINARIES not set")
def test_read_real_binaries():
    pytest.importorskip("pypamguard")
    from speciesclassifiers.io import read_clicks, read_whistles
    c, sr = read_clicks(BIN, progress=False)
    w, st = read_whistles(BIN, progress=False)
    assert len(c) and sr > 0 and len(w)
    assert st["fft_len"] in (512, 1024, 2048, 4096)
    assert all(np.all(f > 0) for f in w["freqs"])
