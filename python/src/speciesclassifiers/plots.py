"""Shared plots (needs matplotlib)."""

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


def plot_importance(importance, n=15, ax=None):
    """Horizontal bar chart of the ``n`` most important features (a Series, largest first)."""
    plt = _plt()
    top = importance.head(n)[::-1]
    ax = ax or plt.subplots(figsize=(5, 0.3 * len(top) + 1))[1]
    ax.barh(top.index, top.to_numpy())
    ax.set(xlabel="Importance (mean decrease in impurity)", title=f"Top {len(top)} features")
    return ax
