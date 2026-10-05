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
| 7 | `io`: PAMGuard binaries + annotations (6a), databases (9a/9b) | Py + R | ✅ |
| 8a | `randomforest` (Python) + notebook `03_randomforest`: ROCCA or any feature table, from ClassifyStuff `Classify-rocca` | Py | ✅ |
| 8b | `randomforest` in the R package + Quarto notebook `03_randomforest.qmd` | R | ✅ |
| 9a | `transferlearning` (Python) + PAMGuard database reader + notebook `04_transferlearning` (from ClassifyStuff `Classify-delphinID`) | Py | ✅ |
| 9b | `transferlearning` + database reader in the R package + Quarto notebook `04_transferlearning.qmd` | R | ✅ |
| 10a | `calltypes`: contour filters + measurements, windows + features, labels from call annotations / periods; annotation table converter (matches the original pipeline exactly) | Py | ✅ |
| 10b | `calltypes`: evaluate how well labelled call types separate (grouped RF CV; matches the original on 3 annotated days) | Py | ✅ |
| 10c | `calltypes`: discover (hybrid clustering, cluster summaries, user-defined binary label verdicts, leave-one-day-out, decision blocks; matches the original exactly on 3 annotated days) | Py | ✅ |
| 10d | `calltypes`: frozen library (JSON), reference classifier + numpy-only `standalone.py` (identical results), clock-aligned decision blocks with user-defined rules and task dependencies, notebook `05_calltypes` with a synthetic example | Py | ✅ |
| 11 | Docs (`docs/`), example data notes + generator scripts, GitHub Actions (Python 3.10/3.12, delphinID + notebooks, R package + Quarto notebooks), call-type windows in the standard output format | – | ✅ |

Module documentation: [docs/README.md](README.md).
