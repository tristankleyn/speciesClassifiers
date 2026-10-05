# Roadmap

Built in small chunks. Each chunk ends with working, tested code.

| # | Chunk | Language | Status |
|---|---|---|---|
| 1 | Repo skeleton, standard classifier output format + validators | Py + R | ✅ |
| 2 | delphinID feature transforms: whistle and click detection frames, ported from PAMGuard's Java transforms and checked against PAMGuard's test files | Py | ✅ |
| 3 | delphinID model builder: fixed CNN shape, grouped parameters (acoustic / CNN / training) | Py | ✅ |
| 4 | delphinID grouped cross-validation: leave-one-group-out, bootstrapped train/val subsamples capped per group | Py | ✅ |
| 5 | delphinID export to PAMGuard (SavedModel + `delphinID.pdtf`, zipped) — awaiting PAMGuard load test | Py | ✅ |
| 6a | Notebook `01_make_frames`: PAMGuard binaries + annotations CSV → frames CSV | Py | ✅ |
| 6b | Notebook `02_train_delphinID`: frames CSV → cross-validation → final model → PAMGuard zip (local or Colab) | Py | ✅ |
| 7 | `io`: read PAMGuard databases / binaries into standard tables (Python binary reader + annotations done in 6a) | Py + R | |
| 8a | `randomforest` (Python) + notebook `03_randomforest`: ROCCA or any feature table, from ClassifyStuff `Classify-rocca` | Py | ✅ |
| 8b | `randomforest` in the R package + Quarto notebook | R | ⏳ |
| 9 | `transferlearning`: event classifiers on standard outputs (from ClassifyStuff `Classify-delphinID`) | Py + R | |
| 10 | `calltypes`: split the contour pipeline into modules, generalise, library export, notebook | Py | |
| 11 | Docs, examples, CI | – | |

## delphinID parameters

- **Acoustic** (`AcousticParams`): frequency range, frame length and hop, `min_clicks` / `min_density`, minimum whistle fragment, FFT length, smoothing, downsampling. The number of model inputs follows from these.
- **CNN** (`CNNParams`; shape fixed: Conv1D → MaxPool → Conv1D → MaxPool → LeakyReLU → Dense → Dropout → softmax): filters, kernel size, max pool, LeakyReLU slope, dense size, dropout, L2
- **Training** (`TrainingParams`): learning rate, epochs per bootstrap, batch size, patience, seed
- **Grouped resampling** (`ResamplingParams`): `max_per_group` (cap on examples per group in each bootstrap, default 30) and `n_bootstraps` (resamples per fold, default 5)
- **Cross-validation**: leave-one-group-out by default; `n_folds` groups the test groups into folds for large datasets

Defaults reproduce the published models; loading their weights into `build_model` gives identical predictions.

## Running delphinID without installing TensorFlow

- Google Colab: the notebook opens in Colab with TensorFlow preinstalled; users upload data and download the PAMGuard zip.
- Local: `pip install "speciesclassifiers[delphinid]"` in a fresh environment. The models are small, so CPU is enough.
