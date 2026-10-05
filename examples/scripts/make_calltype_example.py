"""Synthetic whistle contours + call annotations for the calltypes notebook and tests."""
from pathlib import Path
EX = Path(__file__).resolve().parents[1]   # examples/
import gzip
import json
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
SL = 1024 / 96000          # slice length (s)
# call type -> (start f, end f, duration s, wiggle Hz, species, ecotype)
TYPES = {"A": (4500, 8000, 1.0, 150, "X", "E1"), "B": (10000, 7000, 0.6, 100, "X", "E2"),
         "C": (3000, 3600, 1.5, 400, "Y", ""), "D": (12000, 14500, 0.5, 80, "X", "E1")}
REPERTOIRE = {"E1": "AD", "E2": "B", "Y": "C"}
DAYS = ["2024-06-01", "2024-06-02", "2024-06-03", "2024-06-04"]
ENC = {0: ["E1", "Y"], 1: ["E2", "E1"], 2: ["Y", "E2", "E1"], 3: ["E1", "Y", "noise"]}


def contour(t0, typ):
    f0, f1, dur, wig, *_ = TYPES[typ]
    dur *= rng.uniform(0.85, 1.15)
    n = int(dur / SL)
    x = np.linspace(0, 1, n)
    shift = rng.normal(0, 200)
    f = f0 + (f1 - f0) * x + wig * np.sin(2 * np.pi * x * (2 if typ == "C" else 1)) + shift + rng.normal(0, 40, n)
    return {"start": round(t0, 4), "times": [round(k * SL, 4) for k in range(n)],
            "freqs": [int(v) for v in np.round(f)], "slice_s": SL, "snr_db": round(float(rng.uniform(4, 20)), 1)}


contours, calls = [], []
for d, day in enumerate(DAYS):
    for e, grp in enumerate(ENC[d]):
        t = pd.Timestamp(f"{day} {8 + 3 * e:02d}:00:00", tz="UTC").timestamp() + rng.uniform(0, 600)
        end = t + 300
        while t < end:
            if grp == "noise":            # flat tonal noise, e.g. a pump
                c = {"start": round(t, 4), "times": [round(k * SL, 4) for k in range(60)],
                     "freqs": [int(v) for v in 2000 + rng.normal(0, 15, 60)], "slice_s": SL, "snr_db": 6.0}
                contours.append(c)
                t += rng.uniform(0.6, 1.5)
                continue
            typ = rng.choice(list(REPERTOIRE[grp]))
            c = contour(t, typ)
            contours.append(c)
            if rng.random() < 0.6:        # a second, overlapping contour of the same call (e.g. a harmonic)
                c2 = contour(t + rng.uniform(0, 0.1), typ)
                c2["freqs"] = [min(v * 2, 15900) for v in c2["freqs"]] if TYPES[typ][1] < 7900 else c2["freqs"]
                contours.append(c2)
            sp, eco = TYPES[typ][4], TYPES[typ][5]
            cend = c["start"] + len(c["times"]) * SL
            if d < 3:
                label = typ if rng.random() < 0.7 else "untyped"
                calls.append({"event_id": f"enc{d + 1}{e + 1}", "label": label, "species": sp,
                              "ecotype": eco or "n/a", "start": round(c["start"] - 0.05, 3), "end": round(cend + 0.05, 3)})
            t += rng.uniform(3, 12)

contours.sort(key=lambda c: c["start"])
fmt = lambda s: pd.to_datetime(s, unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
a = pd.DataFrame(calls)
a["start"], a["end"] = fmt(a["start"]), fmt(a["end"])
a.to_csv(EX / "call_annotations_example_synthetic.csv", index=False)
with gzip.open(EX / "contours_example_synthetic.json.gz", "wt") as f:
    json.dump(contours, f, separators=(",", ":"))
print(len(contours), "contours,", len(a), "calls")
