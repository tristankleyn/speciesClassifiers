# speciesClassifiers

Tools for building acoustic species classifiers from [PAMGuard](https://www.pamguard.org/) detections.

[![tests](https://github.com/tristankleyn/speciesClassifiers/actions/workflows/tests.yml/badge.svg)](https://github.com/tristankleyn/speciesClassifiers/actions/workflows/tests.yml)

Documentation: [docs/](docs/README.md) · Data formats: [docs/data_formats.md](docs/data_formats.md)

## What's here

| Module | What it does | Python | R |
|---|---|---|---|
| `delphinid` | Build your own delphinID classifier: a lightweight CNN trained on detection frames (averaged spectra of whistle or click detections), exportable to PAMGuard | ✅ | – (TensorFlow) |
| `randomforest` | Build a random forest classifier from scratch on detection features | ✅ | ✅ |
| `transferlearning` | Retrain on the outputs of any base classifier (delphinID, random forest, …) to make event-level classifiers for new labels | ✅ | ✅ |
| `calltypes` | Build a call-type library from whistle contours: windowed contour features, clustering, and a frozen library for classifying new data | ✅ | – |

Shared pieces:
- `io` — reading PAMGuard binaries and databases, annotations and annotation tables
- `outputs` — the [standard classifier output](schemas/classifier_output.md) that links the modules together

User-facing workflows live in `notebooks/` (Jupyter for Python, Quarto for R). Each notebook has its adjustable parameters at the top.

| Notebook | Does |
|---|---|
| `notebooks/python/01_make_frames.ipynb` | PAMGuard binaries + annotations CSV → delphinID detection frames |
| `notebooks/python/02_train_delphinID.ipynb` | Frames → cross-validation → final model → PAMGuard zip (runs in Google Colab too) |
| `notebooks/python/03_randomforest.ipynb`, `notebooks/R/03_randomforest.qmd` | Random forest from scratch on ROCCA contour stats or any feature table |
| `notebooks/python/04_transferlearning.ipynb`, `notebooks/R/04_transferlearning.qmd` | Event classifiers for new labels, retrained on any base classifiers' outputs (e.g. delphinID predictions in a PAMGuard database) |
| `notebooks/python/05_calltypes.ipynb` | Whistle contours + call annotations → windows → call-type evaluation → discovery (clusters + label verdicts) → frozen library → decision blocks for new data; `calltypes/standalone.py` classifies with numpy only |

Notebooks 02–05 run straight away on the small examples in `examples/` (see [examples/README.md](examples/README.md)); 01 needs PAMGuard binaries.

## Layout

```
python/                 Python package (speciesclassifiers)
R/speciesClassifiers/   R package
notebooks/python/       Jupyter notebooks
notebooks/R/            Quarto notebooks
schemas/                Shared data formats
examples/               Small example data
docs/                   Documentation and roadmap
tools/                  run_notebooks.py (notebook smoke test, used by CI)
.github/workflows/      Tests on every push: Python 3.10 and 3.12, delphinID + notebooks, R package + Quarto notebooks
```

## Install

Python:
```
pip install -e python                          # core
pip install -e "python[pamguard]"              # + reading PAMGuard binaries (pypamguard)
pip install -e "python[delphinid]"             # + TensorFlow for delphinID training
pip install -e "python[notebooks,dev]"         # + Jupyter, matplotlib, pytest
```
Or straight from GitHub: `pip install "speciesclassifiers @ git+https://github.com/tristankleyn/speciesClassifiers#subdirectory=python"`.

R:
```r
remotes::install_local("R/speciesClassifiers")
# or: remotes::install_github("tristankleyn/speciesClassifiers", subdir = "R/speciesClassifiers")
```

## Tests
```
pytest python/tests                      # Python
python tools/run_notebooks.py            # run notebooks 02-05 on the examples
Rscript -e 'testthat::test_local("R/speciesClassifiers")'
```
Some tests use real data when pointed at it with environment variables (`SC_PAMGUARD_TEST_DATA`, `SC_PAMGUARD_BINARIES`, `SC_KW_BINARIES`, `SC_EXAMPLE_DB`, `SC_ROCCA_CSV`); they are skipped otherwise.

## History

This repository supersedes [which.dolphin](https://github.com/tristankleyn/which.dolphin) and [ClassifyStuff](https://github.com/tristankleyn/ClassifyStuff).
