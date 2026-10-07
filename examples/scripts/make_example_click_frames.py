"""Synthetic click frames for the events of the synthetic whistle frames (run make_example_frames.py first), so
delphinID can be tried with two detection types (not real data). Each species has its own typical click peak
frequency; some whistle frames have no click frame and some click frames fall between whistle frames."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

EX = Path(__file__).resolve().parents[1]   # examples/
rng = np.random.default_rng(2027)
sr, fft_len, fmin, fmax, down = 96000.0, 512, 10000.0, 40000.0, 2
nfft = fft_len // 2
lo, hi = int(fmin / (sr / 2) * nfft), int(fmax / (sr / 2) * nfft)
n_bins = int(np.ceil((hi - lo) / down))
freqs = (np.arange(lo, hi, down) + down / 2) * (sr / 2) / nfft
peak = {"SpA": 24000, "SpB": 28000, "SpC": 33000}
w = pd.read_csv(EX / "frames_example_synthetic.csv", usecols=["event_id", "frame_start", "label"])
rows = []
for ev, g in w.groupby("event_id", sort=False):
    sp = g["label"].iloc[0]
    ev_peak = peak[sp] + rng.normal(0, 1500)
    starts = np.sort(np.r_[g["frame_start"].to_numpy()[rng.random(len(g)) < 0.7],
                           g["frame_start"].to_numpy()[: int(rng.integers(1, 4))] + 2.0])
    for t in starts:
        n_clicks = int(rng.integers(3, 30))
        spec = np.exp(-0.5 * ((freqs - rng.normal(ev_peak, 2500)) / rng.uniform(3000, 7000)) ** 2)
        spec = 10 * np.log10(spec + 0.05 + rng.random(n_bins) * 0.05)
        spec = (spec - spec.min()) / (spec.max() - spec.min())
        rows.append({"event_id": ev, "frame_start": round(float(t), 3), "n_detections": n_clicks, "label": sp,
                     **{f"f{i + 1}": round(float(v), 6) for i, v in enumerate(spec)}})
df = pd.DataFrame(rows)
df.to_csv(EX / "frames_click_example_synthetic.csv", index=False)
params = {"note": "SYNTHETIC example click frames (not real recordings)",
          "acoustic_params": {"voc_type": "click", "freq_range": [fmin, fmax], "frame_len": 4.0, "hop": 4.0,
                              "min_clicks": 3, "min_density": 0.05, "min_frag_ms": 200, "fft_len": fft_len,
                              "smooth_window": 3, "downsample": down},
          "sample_rate": sr}
(EX / "frames_click_example_synthetic_params.json").write_text(json.dumps(params, indent=1))
print(df.shape, n_bins)
