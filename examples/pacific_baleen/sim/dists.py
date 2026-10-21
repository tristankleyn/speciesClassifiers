"""Sampling of the distribution notation used in configs/*.yaml (see configs/README.md)."""
from __future__ import annotations

import math
import numpy as np

DEFAULT_CV = 0.1


def _trunc_normal(rng, mean, sd, lo, hi):
    if sd <= 0:
        return float(np.clip(mean, lo, hi))
    for _ in range(100):
        x = rng.normal(mean, sd)
        if lo <= x <= hi:
            return float(x)
    return float(np.clip(mean, lo, hi))


def spread(spec) -> float | None:
    """Population SD implied by a {mean, ...} spec (sd > cv > range/4 > 10% CV)."""
    if not isinstance(spec, dict) or "mean" not in spec or not isinstance(spec["mean"], (int, float)):
        return None
    m = float(spec["mean"])
    if isinstance(spec.get("sd"), (int, float)):
        return float(spec["sd"])
    if isinstance(spec.get("cv"), (int, float)):
        return float(spec["cv"]) * abs(m)
    r = spec.get("range")
    if isinstance(r, list) and len(r) == 2:
        return (float(r[1]) - float(r[0])) / 4.0
    return DEFAULT_CV * abs(m)


def sample(spec, rng, default=None):
    """Draw one value. Numbers/strings pass through; unknown dicts return `default`."""
    if spec is None:
        return default
    if isinstance(spec, bool):
        return spec
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, str):
        return spec
    if isinstance(spec, list):
        return spec
    if not isinstance(spec, dict):
        return default
    if "uniform" in spec:
        a, b = spec["uniform"]
        return float(rng.uniform(a, b))
    if "uniform_int" in spec:
        a, b = spec["uniform_int"]
        return int(rng.integers(int(a), int(b) + 1))
    if "loguniform" in spec:
        a, b = spec["loguniform"]
        return float(math.exp(rng.uniform(math.log(a), math.log(b))))
    if "lognormal_median" in spec:
        m = float(spec["lognormal_median"])
        lo, hi = spec.get("range", [m / 4, m * 4])
        sig = max(math.log(hi / lo) / 4.0, 1e-6)
        return math.exp(_trunc_normal(rng, math.log(m), sig, math.log(lo), math.log(hi)))
    if "choice" in spec:
        opts = spec["choice"]
        w = spec.get("weights")
        if w is not None:
            w = np.asarray(w, float)
            w = w / w.sum()
        pick = opts[int(rng.choice(len(opts), p=w))]
        return sample(pick, rng, default=pick)
    if "mean" in spec and isinstance(spec["mean"], (int, float)):
        m = float(spec["mean"])
        sd = spread(spec)
        r = spec.get("range")
        if isinstance(r, list) and len(r) == 2:
            lo, hi = float(r[0]), float(r[1])
        else:
            lo, hi = m - 3 * sd, m + 3 * sd
        if isinstance(spec.get("min"), (int, float)):
            lo = max(lo, float(spec["min"]))
        return _trunc_normal(rng, m, sd, lo, hi)
    return default


def individualise(spec, rng, ratio):
    """Per-caller version of a {mean, ...} spec: mean drawn from the population, SD x ratio."""
    sd = spread(spec)
    if sd is None:
        return spec
    out = dict(spec)
    out["mean"] = sample(spec, rng)
    out["sd"] = sd * ratio
    out.pop("cv", None)
    return out
