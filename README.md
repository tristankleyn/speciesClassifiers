# speciesClassifiers

Tools for building acoustic species classifiers from [PAMGuard](https://www.pamguard.org/) detections.

> 🚧 Under construction. See [docs/ROADMAP.md](docs/ROADMAP.md) for progress.

## What's here

| Module | What it does | Python | R |
|---|---|---|---|
| `delphinid` | Build your own delphinID classifier: a lightweight CNN trained on detection frames (averaged spectra of whistle or click detections), exportable to PAMGuard | ✅ | – (TensorFlow) |
| `randomforest` | Build a random forest classifier from scratch on detection features | ✅ | ✅ |
| `transferlearning` | Retrain on the outputs of any base classifier (delphinID, random forest, …) to make event-level classifiers for new labels | ✅ | ✅ |
| `calltypes` | Build a call-type library from whistle contours: windowed contour features, clustering, and a frozen library for classifying new data | ✅ | – |

Shared pieces:
- `io` — reading PAMGuard binaries and databases
- `outputs` — the [standard classifier output](schemas/classifier_output.md) that links the modules together

User-facing workflows live in `notebooks/` (Jupyter for Python, Quarto for R). Each notebook has its adjustable parameters at the top.

| Notebook | Does |
|---|---|
| `notebooks/python/01_make_frames.ipynb` | PAMGuard binaries + annotations CSV → delphinID detection frames |
| `notebooks/python/02_train_delphinID.ipynb` | Frames → cross-validation → final model → PAMGuard zip (runs in Google Colab too) |
| `notebooks/python/03_randomforest.ipynb`, `notebooks/R/03_randomforest.qmd` | Random forest from scratch on ROCCA contour stats or any feature table |

Notebooks 02 and 03 run straight away on synthetic examples in `examples/`.

## Layout

```
python/                 Python package (speciesclassifiers)
R/speciesClassifiers/   R package
notebooks/python/       Jupyter notebooks
notebooks/R/            Quarto notebooks
schemas/                Shared data formats
examples/               Small example data
docs/                   Documentation and roadmap
```

## Install

Python:
```
pip install -e python            # core
pip install -e "python[delphinid]"   # adds TensorFlow for delphinID training
```

R:
```r
remotes::install_local("R/speciesClassifiers")
```

## History

This repository supersedes [which.dolphin](https://github.com/tristankleyn/which.dolphin) and [ClassifyStuff](https://github.com/tristankleyn/ClassifyStuff).
