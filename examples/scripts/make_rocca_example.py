"""Synthetic ROCCA-format whistle contour stats (column layout of PAMGuard's RoccaContourStats; values made up)."""
from pathlib import Path
EX = Path(__file__).resolve().parents[1]   # examples/
import numpy as np, pandas as pd, sys
from speciesclassifiers.randomforest.rocca import rocca_features
src = sys.argv[1] if len(sys.argv) > 1 else EX / "RoccaContourStats_example_synthetic.csv"
cols = list(pd.read_csv(src, nrows=0).columns)  # column layout of a RoccaContourStats file
feats = rocca_features(cols, "whistle", omit=[])
rng = np.random.default_rng(7)
base = {f: rng.uniform(1, 10) for f in feats}
hz = {f: 1000 for f in feats if f.startswith("FREQ") and not any(k in f for k in ["RATIO", "SLOPE", "NUM", "PERCENT", "SWEEP", "STEP", "BW", "SPREAD"])}
species = {"SpeciesA": 0.0, "SpeciesB": 0.25}
rows = []
for sp, shift in species.items():
    for e in range(8):
        ev = {f: rng.normal(0, 1.5) for f in feats}
        for _ in range(int(rng.integers(5, 25))):
            r = {c: np.nan for c in cols}
            r.update({"Source": "synthetic", "EncounterID": f"{sp[-1]}{e + 1}", "KnownSpecies": sp, "SamplingRate": 96000,
                      "FREQPEAK": 0, "ClassifiedSpecies": "Ambig"})
            for f in feats:
                v = base[f] * (1 + shift * (hash(f) % 3 - 1) * 0.3) + ev[f] + rng.normal(0, 2.5)
                r[f] = round(abs(v) * hz.get(f, 1), 4)
            r["FREQMAX"] = max(r["FREQMAX"], 1.0)
            rows.append(r)
pd.DataFrame(rows, columns=cols).to_csv(EX / "RoccaContourStats_example_synthetic.csv", index=False)
print(len(rows))
