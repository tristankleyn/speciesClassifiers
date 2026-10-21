"""Draw a few random renderings of every call type per class -> plots/shapes_<class>.png."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.catalog import CLASSES, call_types   # noqa: E402
from sim.shapes import render                  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "plots"
COLOURS = ["#1f77b4", "#d62728", "#2ca02c"]


def check(cls, n=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for name, ct in call_types(cls).items():
        empty, fmin, fmax, dmax = 0, np.inf, 0, 0
        for _ in range(n):
            cs = render(ct, rng, dt=rng.uniform(0.016, 0.022))
            if not cs:
                empty += 1
                continue
            fmin = min(fmin, min(c.f.min() for c in cs)); fmax = max(fmax, max(c.f.max() for c in cs))
            dmax = max(dmax, max(c.t[-1] for c in cs))
        rows.append((name, empty, fmin, fmax, dmax))
    return rows


def plot(cls, draws=3, seed=1):
    rng = np.random.default_rng(seed)
    cts = call_types(cls)
    n = len(cts)
    ncol = min(6, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.1 * nrow), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, (name, ct) in zip(axes.flat, cts.items()):
        for d in range(draws):
            for c in render(ct, rng, dt=0.02):
                a = float(np.clip(1 + c.level_db / 20, 0.15, 1))
                ax.plot(c.t, c.f, color=COLOURS[d], lw=1.2, alpha=a)
        ax.set_title(f"{name}\n[{ct['model']}]", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_ylim(bottom=0)
    fig.supxlabel("time (s)", fontsize=8); fig.supylabel("frequency (Hz)", fontsize=8)
    fig.suptitle(f"{cls}: 3 random draws per call type (fainter = quieter component)", fontsize=10)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    p = OUT / f"shapes_{cls}.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p, n


if __name__ == "__main__":
    for cls in (sys.argv[1:] or CLASSES):
        bad = [r for r in check(cls) if r[1] > 0]
        p, n = plot(cls)
        print(f"{cls}: {n} call types -> {p.name}; empty renders: {bad if bad else 'none'}")
