"""Synthetic click detections for the encounters of the synthetic whistle example (run make_calltype_example.py
first). Each group clicks in trains with its own typical inter-click interval; no clicks during noise."""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

EX = Path(__file__).resolve().parents[1]   # examples/
rng = np.random.default_rng(11)
ENC = {0: ["E1", "Y"], 1: ["E2", "E1"], 2: ["Y", "E2", "E1"], 3: ["E1", "Y", "noise"]}
ICI = {"E1": 0.09, "E2": 0.16, "Y": 0.05}       # typical inter-click interval (s)

with gzip.open(EX / "contours_example_synthetic.json.gz", "rt") as f:
    starts = np.sort([c["start"] for c in json.load(f)])
days = pd.to_datetime(starts, unit="s", utc=True).strftime("%Y-%m-%d")
clicks = []
for d, day in enumerate(sorted(set(days))):
    t = starts[days == day]
    cuts = np.where(np.diff(t) > 1800)[0]
    for e, (a, b) in enumerate(zip(np.r_[0, cuts + 1], np.r_[cuts, len(t) - 1])):
        grp = ENC[d][e]
        if grp == "noise":
            continue
        s = t[a] + rng.uniform(-30, 30)
        while s < t[b]:
            ici = ICI[grp] * rng.lognormal(0, 0.15)          # this train's interval
            n = int(rng.uniform(1.0, 4.0) / ici)
            x = s + np.cumsum(ici * rng.lognormal(0, 0.08, n))
            clicks += [(round(v, 5), round(rng.uniform(5e-5, 2e-4), 6)) for v in x]
            s = x[-1] + rng.uniform(1.5, 6.0)

out = pd.DataFrame(clicks, columns=["start", "duration"]).sort_values("start")
out.to_csv(EX / "clicks_example_synthetic.csv.gz", index=False, compression="gzip")
print(len(out), "clicks")
