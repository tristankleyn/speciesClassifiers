"""Plots for delphinID results (needs matplotlib)."""

import numpy as np


def _plt():
    import matplotlib.pyplot as plt
    return plt


def plot_confusion(confusion, title="", ax=None, normalise=True):
    """Confusion matrix (rows true, columns predicted) as a heatmap. Cells show row % and counts."""
    plt = _plt()
    labels = sorted(set(confusion.index) | set(confusion.columns))
    cm = confusion.reindex(index=labels, columns=labels, fill_value=0)
    counts = cm.to_numpy()
    rows = counts.sum(1, keepdims=True)
    share = np.divide(counts, rows, out=np.zeros_like(counts, float), where=rows > 0)
    ax = ax or plt.subplots(figsize=(1 + 0.8 * len(labels), 0.8 + 0.7 * len(labels)))[1]
    ax.imshow(share if normalise else counts, cmap="Blues", vmin=0, vmax=1 if normalise else None)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if counts[i, j]:
                ax.text(j, i, f"{100 * share[i, j]:.0f}%\n({counts[i, j]})", ha="center", va="center",
                        fontsize=8, color="white" if share[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    return ax


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
