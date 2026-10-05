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
1. **Data**: where your detections are (ROCCA contour stats / a feature table, or PAMGuard whistle binaries / a contours JSON). **Check** reads them and reports counts, date ranges and problems before you run anything.
2. **Labels** (call types): call annotations (CSV or annotation tables), and the yes/no label tasks clusters are judged on (e.g. species X or Y; ecotype only when species is X).
3. **Settings**: the main settings with short explanations; less common ones under *Advanced*.
4. **Run**: stages, progress bar, log.
5. **Results**: cross-validated accuracy, confusion matrices, clusters and their verdicts. Random forest: sliders for the decision thresholds with the threshold curve. Call types: block length and decision rules.
6. **Classify new data · export**: apply the classifier to new data (group decisions, or a presence timeline of decision blocks) and find the outputs: the model, the call-type library and `standalone.py`, predictions in the standard format.

## Not in the app yet
delphinID and transfer learning: use notebooks `01_make_frames`, `02_train_delphinID` (runs in Google Colab, no TensorFlow install needed) and `04_transferlearning`.
