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

## Without installing TensorFlow
- Google Colab: notebook 02 opens in Colab with TensorFlow preinstalled; upload frames, download the zip.
- Locally: `pip install -e "python[delphinid]"` in a fresh environment. The models are small, so a CPU is enough.
