"""Synthetic whistle frames for trying the training notebook (not real data)."""
from pathlib import Path
EX = Path(__file__).resolve().parents[1]   # examples/
import json
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
n_bins = 90  # 2-20 kHz, 100 Hz bins, downsample 2 -> 200 Hz per input
freqs = 2000 + 200 * np.arange(n_bins) + 100
species = {"SpA": (8000, 2500), "SpB": (11000, 3000), "SpC": (14000, 2000)}
rows = []
t0 = 1.7e9
for sp, (centre, width) in species.items():
    for e in range(6):
        ev_centre = centre + rng.normal(0, 700)  # events differ a bit (like real encounters)
        for f in range(int(rng.integers(15, 40))):
            n_whistles = rng.integers(1, 5)
            hist = np.zeros(n_bins)
            for _ in range(n_whistles):
                c = rng.normal(ev_centre, width)
                lo, hi = sorted(np.clip([c - rng.uniform(500, 3000), c + rng.uniform(500, 3000)], 2000, 19999))
                hist += ((freqs >= lo) & (freqs <= hi)) * rng.uniform(0.5, 1.5)
            hist += rng.random(n_bins) * 0.05
            hist /= hist.sum()
            rows.append({"event_id": f"{sp}_ev{e + 1}", "frame_start": t0 + 4 * f, "n_detections": int(n_whistles),
                         "density": round(float(rng.uniform(0.05, 1.5)), 3), "label": sp,
                         **{f"f{i + 1}": round(float(v), 6) for i, v in enumerate(hist)}})
        t0 += 86400
df = pd.DataFrame(rows)
df.to_csv(EX / "frames_example_synthetic.csv", index=False)
params = {"note": "SYNTHETIC example data for trying 02_train_delphinID; not real recordings",
          "acoustic_params": {"voc_type": "whistle", "freq_range": [2000.0, 20000.0], "frame_len": 4.0, "hop": 4.0,
                              "min_clicks": 3, "min_density": 0.05, "min_frag_ms": 200, "fft_len": 512,
                              "smooth_window": 3, "downsample": 2},
          "sample_rate": 96000.0}
open(EX / "frames_example_synthetic_params.json", "w").write(json.dumps(params, indent=1))
print(df.shape, df.groupby("label").event_id.nunique().to_dict())
