# delphinID

Lightweight CNNs that classify **detection frames**: the averaged spectrum of the whistle contours (or clicks) in a few seconds of detections. Trained models export to a zip that PAMGuard's deep learning classifier loads. Python only (TensorFlow).

Notebooks: `01_make_frames` (PAMGuard binaries + annotations → frames CSV), `02_train_delphinID` (frames → cross-validation → final model → PAMGuard zip; runs locally or in Google Colab).

## Parameters
- **Acoustic** (`AcousticParams`): `voc_type`, `freq_range`, `frame_len` and `hop`, `min_clicks` / `min_density` (default 0.05), `min_frag_ms`, `fft_len`, `smooth_window`, `downsample`. The number of model inputs follows from these. Frames are computed exactly as PAMGuard computes them ([details](delphinid_features.md)).
- **CNN** (`CNNParams`; shape fixed: Conv1D → MaxPool → Conv1D → MaxPool → LeakyReLU → Dense → Dropout → softmax): filters, kernel size, max pool, LeakyReLU slope, dense size, dropout, L2.
- **Training** (`TrainingParams`): learning rate, epochs per bootstrap, batch size, patience, seed.
- **Grouped resampling** (`ResamplingParams`): `max_per_group` (cap on frames per group in each bootstrap, default 30) and `n_bootstraps` (resamples per fold, default 5).
- **Cross-validation**: leave-one-group-out by default; `n_folds` groups the test groups into folds for large datasets.

Defaults reproduce the published models; loading their weights into `build_model` gives identical predictions.

## Export to PAMGuard
`export_pamguard(model, acoustic, classes, "model.zip")` writes a SavedModel plus `delphinID.pdtf` (the transforms and class names PAMGuard needs), zipped. The number of classes comes from `classes`. `load_pamguard_zip` loads one back for checking.

## In the app
Create a **Deep Learning Classifier** project: binaries + an annotations CSV (or frames from notebook 01) → frames → cross-validation → final model → PAMGuard zip. See [app.md](app.md#delphinid-and-tensorflow) for the TensorFlow check and the Colab route.

## Without installing TensorFlow
- Google Colab: notebook 02 opens in Colab with TensorFlow preinstalled; upload frames (or a bundle from the app), download the zip.
- Locally: `pip install -e "python[delphinid]"` (TensorFlow 2.16 or newer) in a fresh environment. The models are small, so a CPU is enough.

## Several detection types, with a final layer per block
In the app, data source **Whistles and/or clicks**: each ticked type gets its own delphinID model, from PAMGuard binaries (one folder and annotations CSV for all types; frames made with each type's acoustic settings) or from a frames CSV made with notebook 01.

1. **Cross-validation** (always on here) gives held-out frame predictions: every event's frames are predicted by a model trained without that event (one model per event, or per fold of events with `n_folds`).
2. **Final layer** (`speciesclassifiers.stacking`, as for the [random forest](randomforest.md#several-detection-types-from-binaries-with-a-final-layer-per-block)): the held-out probabilities are averaged per clock-aligned block and type, put side by side with a `has_<type>` input per type, and a balanced random forest is tested by leaving one event out.
3. **Final models**: one PAMGuard zip per type (each also usable on its own in PAMGuard) and the final forest.
4. **New data**: frames from binaries (or frames CSVs) are predicted with each type's zip and each block is decided by the final forest (`apply/blocks.csv`).

**Google Colab** (free GPU, nothing to install): on the Run screen, *Make Colab bundle* packs the project's frames and settings; notebook `02b_train_delphinID_types` runs the project's own cross-validation, final models and final layer from the bundle (`speciesclassifiers.project.colab.run_bundle`) and downloads a results zip; *Import results from Colab* puts them back into the project, where they count as done while the settings are unchanged. The GPU helps less than for large networks (the CNN is small): setting **Folds** (e.g. 10) cuts the number of models far more than leaving one event out at a time.

## Advanced diagnostics
Results → **Advanced diagnostics** opens a page over the held-out (cross-validation) frame predictions (`delphinid.diagnostics`). Each frame has its prediction's **confidence** (top probability) and **score** (top × lead over the second), its **detections** and whistle **density**, and values from its input spectrum: **peak frequency**, **lowest / highest frequency with energy** and the **span** between them (for clicks: bins in the upper half of the frame's dB range).

- **Frame filters** (a range per measurement) leave frames out of frame accuracy and of their event's decision.
- **Event filters** (minimum kept frames, minimum event score or confidence) leave events undecided.
- The page shows frames kept and events decided, frame and event accuracy (kept / decided only), confusion matrices, per-class results, the events, and the distribution of any measurement for correct and wrong frames, with the filtered-out range greyed.

Filters are saved with the project; they don't change the model. With several detection types, switch between them at the top.
