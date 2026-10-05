"""delphinID CNN: build, train and predict.

The model shape is fixed (as in the published delphinID models):

    Conv1D(k) -> MaxPool -> Conv1D(k + 2) -> MaxPool -> LeakyReLU -> Flatten
    -> Dense (L2) -> Dropout -> Dense softmax

Its sizes and training settings are adjustable through ``CNNParams`` and ``TrainingParams``.
TensorFlow is imported only when a model is built, so the rest of the package works without it.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class CNNParams:
    """CNN hyperparameters (defaults: published delphinID models).

    n_filters     filters in each convolutional layer
    kernel_size   kernel size of the first convolutional layer (the second uses kernel_size + 2)
    max_pool      pooling window after each convolutional layer
    leaky_relu    slope of the LeakyReLU for negative inputs
    dense_size    units in the hidden dense layer
    dropout       fraction of hidden units dropped during training
    l2            L2 regularisation on the hidden dense layer
    """

    n_filters: int = 16
    kernel_size: int = 3
    max_pool: int = 2
    leaky_relu: float = 0.1
    dense_size: int = 10
    dropout: float = 0.3
    l2: float = 1e-4

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainingParams:
    """Training settings.

    learning_rate   Adam learning rate
    epochs          training epochs (per bootstrap when cross-validating)
    batch_size      examples per weight update
    patience        stop early if validation loss hasn't improved for this many epochs
    seed            random seed for weight initialisation and shuffling
    """

    learning_rate: float = 0.0005
    epochs: int = 20
    batch_size: int = 2
    patience: int = 20
    seed: int = 42

    def to_dict(self):
        return asdict(self)


def _keras():
    try:
        import keras
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "delphinID models need TensorFlow: pip install \"speciesclassifiers[delphinid]\" "
            "(or run the notebook in Google Colab)") from e
    return keras


def build_model(n_inputs, n_classes, cnn: CNNParams = None, training: TrainingParams = None):
    """Build and compile a delphinID CNN for ``n_inputs`` features and ``n_classes`` classes."""
    keras = _keras()
    cnn = cnn or CNNParams()
    training = training or TrainingParams()
    keras.utils.set_random_seed(training.seed)
    L = keras.layers
    model = keras.Sequential([
        keras.Input(shape=(n_inputs, 1), name="spectrum"),
        L.Conv1D(cnn.n_filters, cnn.kernel_size, padding="same"),
        L.MaxPooling1D(cnn.max_pool, padding="same"),
        L.Conv1D(cnn.n_filters, cnn.kernel_size + 2, padding="same"),
        L.MaxPooling1D(cnn.max_pool, padding="same"),
        L.LeakyReLU(negative_slope=cnn.leaky_relu),
        L.Flatten(),
        L.Dense(cnn.dense_size, kernel_regularizer=keras.regularizers.l2(cnn.l2)),
        L.Dropout(cnn.dropout),
        L.Dense(n_classes, activation="softmax", name="probabilities"),
    ], name="delphinID")
    model.compile(optimizer=keras.optimizers.Adam(training.learning_rate),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _as_input(x):
    x = np.asarray(x, dtype="float32")
    return x[..., None] if x.ndim == 2 else x


def one_hot(labels, classes):
    """One-hot encode ``labels`` against the ordered list ``classes``."""
    idx = {c: i for i, c in enumerate(classes)}
    y = np.zeros((len(labels), len(classes)), dtype="float32")
    y[np.arange(len(labels)), [idx[l] for l in labels]] = 1
    return y


def train(model, x_train, y_train, x_val=None, y_val=None, training: TrainingParams = None, verbose=0):
    """Fit ``model`` (continuing from its current weights). ``y`` is one-hot.

    Returns the Keras History. Early stopping uses validation loss when validation data are given
    and restores the best weights.
    """
    keras = _keras()
    training = training or TrainingParams()
    callbacks, val = [], None
    if x_val is not None and len(x_val):
        val = (_as_input(x_val), np.asarray(y_val, "float32"))
        callbacks.append(keras.callbacks.EarlyStopping(
            "val_loss", patience=training.patience, restore_best_weights=True))
    return model.fit(_as_input(x_train), np.asarray(y_train, "float32"), validation_data=val,
                     epochs=training.epochs, batch_size=training.batch_size, shuffle=True,
                     callbacks=callbacks, verbose=verbose)


def predict(model, x, batch_size=256):
    """Class probabilities, shape (n, n_classes)."""
    return model.predict(_as_input(x), batch_size=batch_size, verbose=0)
