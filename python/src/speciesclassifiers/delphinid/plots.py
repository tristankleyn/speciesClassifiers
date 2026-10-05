"""Plots for delphinID results (needs matplotlib)."""

import numpy as np

from ..plots import plot_confusion  # noqa: F401  (re-exported)


def _plt():
    import matplotlib.pyplot as plt
    return plt


def plot_history(history, ax=None):
    """Mean training and validation loss per epoch across folds, one panel per bootstrap."""
    plt = _plt()
    boots = sorted(history["bootstrap"].unique())
    if ax is None:
        fig, ax = plt.subplots(1, len(boots), figsize=(2.6 * len(boots), 2.8), sharey=True, squeeze=False)
        ax = ax[0]
    for a, b in zip(np.atleast_1d(ax), boots):
        h = history[history["bootstrap"] == b].groupby("epoch")[["loss", "val_loss"]].mean()
        a.plot(h.index, h["loss"], label="training")
        a.plot(h.index, h["val_loss"], label="validation")
        a.set(title=f"Bootstrap {b}", xlabel="Epoch")
    np.atleast_1d(ax)[0].set_ylabel("Loss (mean over folds)")
    np.atleast_1d(ax)[0].legend()
    return ax
