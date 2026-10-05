# Roadmap

Built in small chunks. Each chunk ends with working, tested code.

| # | Chunk | Language | Status |
|---|---|---|---|
| 1 | Repo skeleton, standard classifier output format + validators | Py + R | ✅ |
| 2 | delphinID feature transforms: whistle and click detection frames, ported from PAMGuard's Java transforms and checked against PAMGuard's test files | Py | ✅ |
| 3 | delphinID model builder: fixed CNN shape, grouped parameters (acoustic / CNN / training) | Py | ⏳ |
| 4 | delphinID grouped cross-validation: leave-one-group-out, bootstrapped train/val subsamples capped per group | Py | |
| 5 | delphinID export to PAMGuard (SavedModel + `delphinID.pdtf`, zipped) | Py | |
| 6 | delphinID notebook (runs locally or in Google Colab) | Py | |
| 7 | `io`: read PAMGuard databases / binaries into standard tables | Py + R | |
| 8 | `randomforest`: from scratch on detection features (from ClassifyStuff `Classify-rocca`) | Py + R | |
| 9 | `transferlearning`: event classifiers on standard outputs (from ClassifyStuff `Classify-delphinID`) | Py + R | |
| 10 | `calltypes`: split the contour pipeline into modules, generalise, library export, notebook | Py | |
| 11 | Docs, examples, CI | – | |

## delphinID parameters (planned)

- **Acoustic:** frequency range, number of input bins, frame length (s), minimum detections per frame
- **CNN** (shape fixed: Conv1D → MaxPool → Conv1D → MaxPool → LeakyReLU → Dense → Dropout → softmax): filters, kernel size, max pool, LeakyReLU slope, dense size, dropout, L2
- **Training:** learning rate, epochs per bootstrap, batch size, patience, validation split, seed
- **Grouped resampling** (two controls): `max_per_group` (cap on examples per group in each bootstrap, default 30) and `n_bootstraps` (resamples per fold, default 5)

## Running delphinID without installing TensorFlow

- Google Colab: the notebook opens in Colab with TensorFlow preinstalled; users upload data and download the PAMGuard zip.
- Local: `pip install "speciesclassifiers[delphinid]"` in a fresh environment. The models are small, so CPU is enough.
