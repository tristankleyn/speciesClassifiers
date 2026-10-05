import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from speciesclassifiers.delphinid import AcousticParams, click_frames, whistle_frames
from speciesclassifiers.delphinid import transforms as T


# ---------------------------------------------------------------- spectrum ops ---------
def test_smooth_is_trailing_mean():
    assert np.allclose(T.spectrum_smooth([1, 2, 3, 4], 3), [1, 1.5, 2, 3])


def test_downsample_keeps_partial_block():
    assert np.allclose(T.spectrum_downsample_mean([1, 2, 3, 4, 5], 2), [1.5, 3.5, 5])


def test_trim_matches_pamguard_indexing():
    x = np.arange(256.0)  # 0-48 kHz at 96 kHz
    y = T.spectrum_trim_freq(x, 10000, 40000, 96000)
    assert len(y) == 160 and y[0] == 53


def test_click_spectrum_long_clicks():
    rng = np.random.default_rng(0)
    w = rng.normal(size=800)
    w[700] = 50
    assert not np.allclose(T.click_power_spectrum(w, 512, mode="pamguard"),
                           T.click_power_spectrum(w, 512, mode="legacy"))
    assert np.allclose(T.click_power_spectrum(w[:300], 512, mode="pamguard"),
                       T.click_power_spectrum(w[:300], 512, mode="legacy"))


def test_segments_include_detections_ending_inside():
    segs = T.segment_starts([0.0, 3.9], [0.1, 4.2], frame_len=4, hop=4)
    assert [list(i) for _, i in segs] == [[0, 1], [1]]


# ---------------------------------------------------------------- params / frames ------
def test_n_bins_match_published_models():
    assert AcousticParams.whistles().n_bins() == 90
    assert AcousticParams.clicks().n_bins(96000) == 80


def test_click_frames_shape_and_threshold():
    rng = np.random.default_rng(1)
    clicks = pd.DataFrame({
        "start": [0.1, 0.5, 1.0, 1.5, 9.0],
        "duration": [0.001] * 5,
        "wave": [rng.normal(size=200) for _ in range(5)],
        "event_id": "e1",
        "label": "Dde",
    })
    fr = click_frames(clicks, AcousticParams.clicks(min_clicks=3), 96000)
    assert len(fr) == 1  # the lone click at 9 s doesn't make a frame
    assert fr.loc[0, "n_detections"] == 4 and fr.loc[0, "label"] == "Dde"
    feats = fr.filter(regex=r"^f\d+$")
    assert feats.shape[1] == 80 and np.isclose(feats.iloc[0].sum(), 1)


def test_whistle_frames():
    t = np.arange(0, 0.5, 0.01)
    contours = pd.DataFrame({
        "start": [0.2, 1.0],
        "duration": [0.5, 0.5],
        "times": [t, t],
        "freqs": [np.linspace(5000, 9000, len(t)), np.linspace(12000, 8000, len(t))],
        "event_id": "e1",
    })
    fr = whistle_frames(contours, AcousticParams.whistles(min_density=0))
    assert len(fr) == 1 and fr.filter(regex=r"^f\d+$").shape[1] == 90


# ---------------------------------------------------------------- PAMGuard reference ---
# Set SC_PAMGUARD_TEST_DATA to PAMGuard's src/test/resources/rawDeepLearningClassifier/DelphinID
REF = os.environ.get("SC_PAMGUARD_TEST_DATA")
needs_ref = pytest.mark.skipif(not REF, reason="SC_PAMGUARD_TEST_DATA not set")


def _pdtf(zipname, folder):
    return json.loads(zipfile.ZipFile(Path(REF) / zipname).read(f"{folder}/delphinID.pdtf"))


def _datenum_to_s(d):
    return (d - 719529) * 86400.0


@needs_ref
def test_click_features_match_reference():
    sio = pytest.importorskip("scipy.io")
    m = sio.loadmat(Path(REF) / "clicks_20200918_123234_classified.mat", squeeze_me=True, struct_as_record=False)
    ref = sio.loadmat(Path(REF) / "click_preds_PAM_20200918_123234_366.mat", squeeze_me=True)["spectrumpython"]
    sr = float(m["samplerate"])
    st = np.array([c.millis / 1000 for c in m["clicks"]])
    en = st + np.array([c.duration / sr for c in m["clicks"]])
    waves = [np.atleast_1d(c.wave) for c in m["clicks"]]
    transforms = _pdtf("clickClassifier_Jan25.zip", "clickClassifier_Jan25")["transforms"]
    segs = T.segment_starts(st, en, 4, 1, data_start=_datenum_to_s(m["filedate"]))
    n = 0
    for i, (t, idx) in enumerate(segs[: len(ref)]):
        if len(idx) == 0 or ref[i].sum() == 0:
            continue
        x = T.apply_transforms(T.clicks2spectrum([waves[j] for j in idx], 512, mode="legacy"), transforms, sr)
        np.testing.assert_allclose(x, ref[i], rtol=1e-5, atol=1e-9)
        n += 1
    assert n > 40


@needs_ref
def test_whistle_features_match_reference():
    sio = pytest.importorskip("scipy.io")
    w = sio.loadmat(Path(REF) / "whistle_contours_20200918_123234.mat", squeeze_me=True, struct_as_record=False)
    r = sio.loadmat(Path(REF) / "whistle_preds_PAM_20200918_123234_366.mat", squeeze_me=True)
    sr, fl, hop = float(w["samplerate"]), float(w["fftlen"]), float(w["ffthop"])
    contours = []
    for x in w["whistles"]:
        sl = np.array([s.sliceNumber for s in np.atleast_1d(x.sliceData)], float)
        contours.append({"start": x.millis / 1000, "times": sl * hop / sr,
                         "freqs": np.atleast_1d(x.contour).astype(float) * sr / fl,
                         "duration": x.sampleDuration / sr})
    st = np.array([c["start"] for c in contours])
    en = st + np.array([c["duration"] for c in contours])
    transforms = _pdtf("whistleclassifier.zip", "whistleclassifier")["transforms"]
    segs = T.segment_starts(st, en, 4, 1)  # delphinID segments from the first whistle
    for k, s in enumerate(r["starttimes"]):
        t, idx = segs[s]
        spec = T.whistles2spectrum([contours[j] for j in idx], t, 4, (2000, 20000), 200, mode="legacy")
        np.testing.assert_allclose(T.apply_transforms(spec, transforms), r["spectrumpython"][k], rtol=1e-5, atol=1e-9)
