# The app

A step-by-step interface for building classifiers, running on your own computer in your browser.

## Starting it
| You have | Do |
|---|---|
| No Python (or don't want to touch it) | Double-click `start_app.bat` (Windows) or `start_app.command` (macOS), or run `./start_app.sh` (Linux) |
| Python | `pip install -e "python[app]"` from the repository folder, then `speciesclassifiers app` |

The launchers use [uv](https://docs.astral.sh/uv/), a small tool that downloads Python and the app's packages into the repository folder (`python/.venv`) on first start (a few minutes, needs internet). Later starts take seconds. Nothing else is installed system-wide; deleting the repository folder removes it all. Keep the launcher window open while you use the app.

## Projects
Each classifier you build is a **project**: a folder (by default in `speciesClassifiers projects` in your home folder) holding
- `project.json`: your answers and settings, saved as you change them
- `status.json`: which stages are done
- `run.log`: everything the runs printed
- one folder per stage with its outputs (e.g. `crossval/`, `model/`, `library/`, `apply/`)

Projects can be reopened from the home page (recent projects) or with **Open project**. They can also be run from the command line: `speciesclassifiers run <folder>`, `speciesclassifiers status <folder>`.

## Stages, resuming and changing settings
A project runs as a series of stages (e.g. read detections → cross-validation → final model → classify new data). Each stage remembers the settings and input files it was made from:
- **Run** does only what isn't done yet, so a stopped run resumes where it left off.
- Changing a setting marks the stages it affects as *settings changed* (amber); the next Run redoes only those. Choosing a decision threshold, for example, only reruns classifying new data.
- Runs continue in the background if you close the browser; reopen the project to see progress. **Stop** ends a run between steps.

## Screens
1. **Data**: where your detections are (ROCCA contour stats / a feature table; PAMGuard binaries with an annotations CSV, or frames from notebook 01, for delphinID; for call types: whistles (PAMGuard binaries / a contours JSON) and/or clicks (PAMGuard binaries / a clicks CSV)). **Check** reads them and reports counts, date ranges and problems before you run anything.
2. **Labels** (call types): labelled periods (events) or call annotations, and the **classification**: one or more layers, each deciding a label (every value a class, or values grouped into classes), with optional next layers within a class or group (e.g. group, then species within the group). Testing leaves out one event (or day) at a time.
3. **Settings**: the main settings with short explanations; less common ones under *Advanced*. Call types: the features each detection type uses (view their values and tick them), click window settings, the weight of whistles and clicks when combined, and how each block is decided (multiplied evidence, or a random forest on cluster shares tested strictly).
4. **Run**: stages, progress bar, log.
5. **Results**: cross-validated accuracy, confusion matrices, clusters and their verdicts. Random forest: sliders for the decision thresholds with the threshold curve. Call types: per layer (the final forest, combined / multiplied, or whistles / clicks alone; with the forest, which clusters matter), block and event accuracy with a threshold slider and curve, and the layer's clusters; block length and decision rules.
6. **Classify new data · export**: apply the classifier to new data (group decisions, or a presence timeline of decision blocks) and find the outputs: the model, the call-type library and `standalone.py`, predictions in the standard format.

## delphinID and TensorFlow
delphinID models are trained with TensorFlow (2.16 or newer). The Run screen checks whether a suitable version is installed. If not, it offers two routes:
- **Train in Google Colab** (nothing to install): the app makes the detection frames, then packs them with your settings into a *bundle* (in the project's `colab` folder). Open notebook `02_train_delphinID` in Colab, upload the bundle, set `BUNDLE` to its name and run all cells; the notebook downloads a results zip. **Import results from Colab** puts the cross-validation results and the PAMGuard model into the project. If you change training settings afterwards, those stages show as *settings changed*.
- **Install TensorFlow** on this computer (about 600 MB, a few minutes), from the app. Training then runs locally like the other stages.

Without TensorFlow, **Run** makes the detection frames only. The **Export to PAMGuard** screen shows the model zip and the settings to use in PAMGuard's Deep Learning Classifier.

## Not in the app yet
Transfer learning: use notebook `04_transferlearning` (Python or R).
