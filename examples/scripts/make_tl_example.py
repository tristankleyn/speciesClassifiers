"""Synthetic base-classifier predictions + labelled events for trying transfer learning (not real data)."""
from pathlib import Path
EX = Path(__file__).resolve().parents[1]   # examples/
import numpy as np, pandas as pd
rng = np.random.default_rng(11)
wcls = ["Dde", "Ggr", "Gme", "Lac", "Lal", "Oor", "Ttr"]; ccls = ["Dde", "Ggr", "Gme", "Lal", "Ttr"]
# new labels the base models were never trained on; each has its own typical base-model "fingerprint"
labels = {"SpeciesX": ("Dde", "Ggr"), "SpeciesY": ("Ttr", "Ttr"), "SpeciesZ": ("Gme", "Lal")}
preds, events = [], []
t = pd.Timestamp("2024-05-01 08:00", tz="UTC")
for lab, (wfav, cfav) in labels.items():
    for enc in range(4):
        for e in range(int(rng.integers(3, 6))):
            ev = f"{lab[-1]}{enc + 1}_{e + 1}"
            start = t; end = t + pd.Timedelta(minutes=10)
            events.append({"event_id": ev, "label": lab, "encounter": f"{lab[-1]}{enc + 1}", "start": start.strftime("%Y-%m-%d %H:%M:%S"), "end": end.strftime("%Y-%m-%d %H:%M:%S")})
            for name, classes, fav, n in [("delphinID_whistles", wcls, wfav, rng.integers(0, 25)), ("delphinID_clicks", ccls, cfav, rng.integers(0, 40))]:
                if rng.random() < 0.15: n = 0     # some events lack one classifier
                for k in range(n):
                    a = np.ones(len(classes)); a[classes.index(fav)] += 0.7
                    p = rng.dirichlet(a * 0.5)
                    tt = start + pd.Timedelta(seconds=float(rng.uniform(0, 600)))
                    for c, v in zip(classes, p):
                        preds.append({"detection_id": f"{name}:{ev}:{k}", "event_id": "", "time": tt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                                      "classifier": name, "voc_type": "whistle" if "whistle" in name else "click", "class": c, "probability": round(float(v), 6)})
            t = end + pd.Timedelta(minutes=30)
        t += pd.Timedelta(days=1)
pd.DataFrame(preds).to_csv(EX / "base_predictions_example_synthetic.csv", index=False)
pd.DataFrame(events).to_csv(EX / "events_example_synthetic.csv", index=False)
print(len(preds), len(events))
