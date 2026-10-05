import pandas as pd
import pytest

from speciesclassifiers.outputs import to_wide, validate_output


def example():
    return pd.DataFrame({
        "detection_id": ["w1", "w1", "c1", "c1"],
        "event_id": ["e1"] * 4,
        "time": ["2024-09-18T10:00:00Z"] * 2 + ["2024-09-18T10:00:04Z"] * 2,
        "classifier": ["dID-w"] * 2 + ["dID-c"] * 2,
        "voc_type": ["whistle"] * 2 + ["click"] * 2,
        "class": ["Dde", "Ttr", "Dde", "Ttr"],
        "probability": [0.7, 0.3, 0.55, 0.45],
    })


def test_valid():
    validate_output(example())


def test_probabilities_must_sum_to_one():
    df = example()
    df.loc[0, "probability"] = 0.9
    with pytest.raises(ValueError, match="sum to 1"):
        validate_output(df)


def test_missing_column():
    with pytest.raises(ValueError, match="Missing"):
        validate_output(example().drop(columns="event_id"))


def test_bad_voc_type():
    df = example()
    df["voc_type"] = "song"
    with pytest.raises(ValueError, match="voc_type"):
        validate_output(df)


def test_to_wide():
    wide = to_wide(example())
    assert len(wide) == 2
    assert {"Dde", "Ttr"} <= set(wide.columns)
