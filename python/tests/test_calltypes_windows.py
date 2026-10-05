import os

import numpy as np
import pandas as pd
import pytest

from speciesclassifiers.calltypes import (ContourParams, WindowParams, contour_params, label_windows,
                                          label_windows_by_period, make_windows, prepare_contours,
                                          window_features)
from speciesclassifiers.io import read_annotations
from speciesclassifiers.io.annotation_tables import read_annotation_table

T0 = pd.Timestamp("2024-06-01 10:00:00", tz="UTC").timestamp()
SL = 1024 / 96000


def _contour(start, f0, f1, n):
    sl = np.arange(n)
    return {"start": T0 + start, "duration": n * SL, "times": sl * SL, "freqs": np.linspace(f0, f1, n),
            "slice_s": SL, "snr_db": 3.0, "amplitude_db": 100.0, "uid": int(start * 10)}


def _whistles():
    return pd.DataFrame([_contour(0.5, 5000, 9000, 60), _contour(1.0, 6000, 6100, 50),
                         _contour(9.0, 12000, 8000, 80), _contour(30.0, 300, 500, 80),   # outside band
                         _contour(40.0, 5000, 5100, 5)])                                  # too short


def test_contour_params_line():
    t = np.arange(10) * 0.01
    p = contour_params(t, 1000 + 1000 * t, 0.01)
    assert np.isclose(p["linreg_slope_hz_s"], 1000) and p["frac_up"] == 1 and p["n_inflections"] == 0
    assert np.isclose(p["duration_s"], 0.1)


def test_prepare_filters():
    c, pts = prepare_contours(_whistles(), ContourParams())
    assert len(c) == 3 and len(pts) == 3
    assert c["start"].is_monotonic_increasing and set(c["day"]) == {"20240601"}
    c2, _ = prepare_contours(_whistles(), ContourParams(min_snr_db=10))
    assert len(c2) == 0


def test_sliding_windows_and_features():
    c, pts = prepare_contours(_whistles(), ContourParams())
    wp = WindowParams(window_s=4, overlap=0.5, min_contours=1, freq_profile_bins=8)
    w = window_features(make_windows(c, wp), c, pts, wp)
    assert (w["w_end"] - w["w_start"] == 4).all()
    assert w["n_contours"].max() == 2
    prof = w.filter(like="prof_")
    assert prof.shape[1] == 8 and np.allclose(prof.sum(axis=1), 1)
    assert ((w["occupancy"] > 0) & (w["occupancy"] <= 1)).all()


def test_labels_partial_mixed_periods():
    c, pts = prepare_contours(_whistles(), ContourParams())
    wp = WindowParams(window_s=4, overlap=0.5, min_contours=1)
    w = window_features(make_windows(c, wp), c, pts, wp)
    calls = read_annotations(pd.DataFrame({
        "event_id": "e", "label": ["A", "B", "untyped"],
        "start": pd.to_datetime([T0 + 0.5, T0 + 1.0, T0 + 9.0], unit="s", utc=True).strftime("%Y-%m-%d %H:%M:%S.%f"),
        "end": pd.to_datetime([T0 + 1.2, T0 + 1.5, T0 + 9.8], unit="s", utc=True).strftime("%Y-%m-%d %H:%M:%S.%f")}))
    lw = label_windows(w, calls, wp)
    assert "mixed" in set(lw["label"])            # A and B in one window
    assert "untyped" in set(lw["label"])          # untyped alone is a label, not mixed
    periods = read_annotations(pd.DataFrame({"event_id": "p", "label": "x", "species": "KW",
                                             "start": ["2024-06-01 10:00:00"], "end": ["2024-06-01 10:00:05"]}))
    pw = label_windows_by_period(lw, periods, columns=["species"], source="events")
    assert set(pw.loc[pw["w_start"] < T0 + 4, "species"]) == {"KW"}
    assert set(pw["label_source"]) <= {"events", "unannotated"}


def test_annotation_table_clock_and_relative(tmp_path):
    f = tmp_path / "sel.txt"
    pd.DataFrame({"Selection": [1, 2], "Begin Time (s)": [1.0, 5.0], "End Time (s)": [2.0, 5.0],
                  "Begin Date": ["2024/06/01", "2024/06/01"], "Begin Clock Time": ["10:00:01.000", "10:00:05.000"],
                  "Call": ["S01", "S02"]}).to_csv(f, sep="\t", index=False)
    a = read_annotation_table(f, label_col="Call")
    assert list(a["label"]) == ["S01"] and a.loc[0, "start"].startswith("2024-06-01 10:00:01")  # zero-length dropped
    b = read_annotation_table(f, label_col="Call", recording_start="2024-06-01 09:00:00")
    assert b.loc[0, "end"].startswith("2024-06-01 09:00:02") and b.loc[0, "event_id"] == "sel"


BIN = os.environ.get("SC_KW_BINARIES")


@pytest.mark.skipif(not BIN, reason="SC_KW_BINARIES not set")
def test_real_binaries():
    pytest.importorskip("pypamguard")
    from speciesclassifiers.io import read_whistles
    raw, _ = read_whistles(BIN, progress=False)
    c, pts = prepare_contours(raw, ContourParams())
    wp = WindowParams()
    w = window_features(make_windows(c, wp), c, pts, wp)
    assert len(c) > 0 and len(w) > 0 and w.shape[1] > 80
