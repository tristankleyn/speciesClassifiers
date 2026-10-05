import numpy as np
import pandas as pd
import pytest

from speciesclassifiers.delphinid.crossval import ResamplingParams, bootstrap_sample


def test_bootstrap_caps_groups_and_balances():
    # class A: one big group (100) + one small (5); class B: three groups of 10
    labels = np.array(["A"] * 105 + ["B"] * 30)
    groups = np.array(["a1"] * 100 + ["a2"] * 5 + ["b1"] * 10 + ["b2"] * 10 + ["b3"] * 10)
    rp = ResamplingParams(max_per_group=20, val_fraction=0.25)
    tr, va = bootstrap_sample(labels, groups, rp, np.random.default_rng(0))
    idx = np.concatenate([tr, va])
    assert len(set(idx)) == len(idx)
    counts = pd.Series(labels[idx]).value_counts()
    assert counts["A"] == counts["B"] == 25  # A capped to 20+5, B 30 -> balanced to 25
    g = pd.Series(groups[idx]).value_counts()
    assert g["a1"] <= 20 and g["a2"] == 5  # small group kept whole
    assert abs(len(va) / len(idx) - 0.25) < 0.05


def test_bootstrap_without_balancing():
    labels = np.array(["A"] * 50 + ["B"] * 10)
    groups = np.array(["a"] * 50 + ["b"] * 10)
    tr, va = bootstrap_sample(labels, groups, ResamplingParams(max_per_group=100, balance_classes=False),
                              np.random.default_rng(0))
    assert len(tr) + len(va) == 60


def _synthetic_frames(n_events=4, n_frames=25, k=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c, centre in [("A", 5), ("B", 15), ("C", 25)]:
        for e in range(n_events):
            for f in range(n_frames):
                x = rng.random(k) * 0.2
                x[centre - 3:centre + 3] += 1 + rng.normal(0, 0.1)
                rows.append({"event_id": f"{c}{e}", "frame_start": f * 4.0, "label": c,
                             **{f"f{i + 1}": v for i, v in enumerate(x / x.sum())}})
    return pd.DataFrame(rows)


def test_cross_validate_end_to_end():
    pytest.importorskip("keras")
    from speciesclassifiers.delphinid.crossval import cross_validate, summarise
    from speciesclassifiers.delphinid.model import TrainingParams
    from speciesclassifiers.outputs import validate_output

    frames = _synthetic_frames()
    preds, hist = cross_validate(
        frames, training=TrainingParams(epochs=4, batch_size=8, learning_rate=0.01),
        resampling=ResamplingParams(max_per_group=10, n_bootstraps=2),
        test_groups=["A0", "B1", "C2"], progress=False)
    validate_output(preds)
    assert set(preds["test_group"]) == {"A0", "B1", "C2"}
    assert hist["bootstrap"].max() == 2
    s = summarise(preds)
    assert s["event_accuracy"] == 1.0 and s["frame_accuracy"] > 0.9
