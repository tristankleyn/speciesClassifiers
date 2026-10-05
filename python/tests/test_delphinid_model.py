import numpy as np
import pytest

pytest.importorskip("keras")

from speciesclassifiers.delphinid.model import CNNParams, TrainingParams, build_model, one_hot, predict, train


def test_shape_matches_published_models():
    # Published whistle model: 90 inputs, 7 classes, 368 units into the hidden dense layer
    m = build_model(90, 7)
    assert m.input_shape == (None, 90, 1) and m.output_shape == (None, 7)
    dense = [l for l in m.layers if l.__class__.__name__ == "Dense"][0]
    assert dense.kernel.shape == (368, 10)
    # Published click model: 80 inputs -> 320
    dense = [l for l in build_model(80, 5).layers if l.__class__.__name__ == "Dense"][0]
    assert dense.kernel.shape == (320, 10)


def test_params_change_model():
    m = build_model(90, 3, CNNParams(n_filters=8, dense_size=6, kernel_size=5))
    convs = [l for l in m.layers if l.__class__.__name__ == "Conv1D"]
    assert [c.kernel.shape for c in convs] == [(5, 1, 8), (7, 8, 8)]


def test_learns_separable_classes():
    rng = np.random.default_rng(0)
    n, k = 120, 30
    labels = np.array(["A", "B"] * (n // 2))
    x = rng.random((n, k)) * 0.1
    x[labels == "A", :10] += 1
    x[labels == "B", -10:] += 1
    x = x / x.sum(1, keepdims=True)
    y = one_hot(labels, ["A", "B"])
    tp = TrainingParams(epochs=15, batch_size=8, learning_rate=0.01)
    m = build_model(k, 2, training=tp)
    train(m, x[:80], y[:80], x[80:], y[80:], tp)
    p = predict(m, x[80:])
    assert np.allclose(p.sum(1), 1, atol=1e-5)
    assert (p.argmax(1) == y[80:].argmax(1)).mean() > 0.9
