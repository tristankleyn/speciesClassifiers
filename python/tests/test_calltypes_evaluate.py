import numpy as np
import pandas as pd
import pytest

from speciesclassifiers.calltypes import (EvalParams, apply_feature_spec, evaluate, fit_feature_spec, results_row,
                                          select_rows)

T0 = pd.Timestamp("2024-06-01", tz="UTC").timestamp()


def _windows(n_days=4, per=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for lab, mu in [("A", 0), ("B", 3), ("C", -3)]:
            for k in range(per):
                t = T0 + d * 86400 + k * 60
                rows.append({"w_start": t, "w_end": t + 4, "day": f"2024060{d + 1}", "label": lab,
                             "n_contours": rng.integers(2, 6), "mean_freq_hz_mean": 8000 + 1000 * mu + rng.normal(0, 800),
                             "duration_s_mean": abs(rng.normal(1, 0.2)), "duration_s_std": np.nan if k % 5 == 0 else 0.1,
                             "skewed": rng.exponential(1) ** 4, "snr_db_mean": rng.normal(),
                             "constant": 1.0})
    rows.append({**rows[0], "label": "none"})
    rows.append({**rows[0], "label": "rare"})
    return pd.DataFrame(rows)


def test_select_rows():
    w = select_rows(_windows(), EvalParams(max_per_class_per_day=10))
    assert set(w["label"]) == {"A", "B", "C"}            # 'none' excluded, 'rare' below min_class_n
    assert w.groupby(["label", "day"]).size().max() == 10


def test_feature_spec():
    w = _windows()
    spec = fit_feature_spec(w, exclude_features=["snr"])
    assert "constant" not in spec["cols"] and "snr_db_mean" not in spec["cols"]
    assert "duration_s_std" in spec["std_fill_cols"] and "skewed" in spec["log_cols"]
    X = apply_feature_spec(w.drop(columns=["skewed"]), spec)      # missing column -> median
    assert not X.isna().any().any() and np.allclose(X["skewed"], np.log1p(spec["medians"]["skewed"]))


def test_evaluate():
    pytest.importorskip("sklearn")
    p = EvalParams(rf_trees=50)
    res = evaluate(select_rows(_windows(), p), p)
    assert res["cv_grouping"] == "day" and res["n_classes"] == 3
    assert res["skill"] > 0.8 and res["confusion"].to_numpy().sum() == res["n_windows"]
    assert res["top_features"][0] == "mean_freq_hz_mean"
    row = results_row(res, {"window_s": 4})
    assert row["window_s"] == 4 and "confusion" not in row
    res_b = evaluate(select_rows(_windows(n_days=2), p), p)          # too few days -> time blocks
    assert res_b["cv_grouping"] == "block"
