"""Settings shown in the app, per classifier type: (group title, [field specs]). See ``widgets.field``."""

RANDOMFOREST = [
    ("Random forest", [
        {"key": "settings.n_trees", "label": "Trees", "kind": "int", "min": 10, "step": 50,
         "help": "more = steadier results, slower"},
        {"key": "settings.max_per_group", "label": "Max detections per group", "kind": "int", "min": 1,
         "help": "picked at random, so big encounters don't dominate training"},
        {"key": "settings.node_size", "label": "Min detections per leaf", "kind": "int", "min": 1,
         "help": "larger = simpler trees"},
        {"key": "settings.mtry", "label": "Features tried per split", "kind": "int", "min": 1, "optional": True,
         "advanced": True, "help": "blank = square root of the number of features"},
        {"key": "settings.prune", "label": "Outlier pruning (0-1)", "kind": "number", "min": 0, "max": 0.9,
         "step": 0.05, "advanced": True, "help": "share of each class's training detections to drop as outliers"},
        {"key": "settings.seed", "label": "Random seed", "kind": "int", "advanced": True},
    ]),
]

CALLTYPES = [
    ("Contours", [
        {"key": "settings.contours.fmin_hz", "label": "Lowest frequency (Hz)", "kind": "number", "step": 100,
         "help": "contour points outside the band are clipped off"},
        {"key": "settings.contours.fmax_hz", "label": "Highest frequency (Hz)", "kind": "number", "step": 100},
        {"key": "settings.contours.min_contour_s", "label": "Shortest contour (s)", "kind": "number", "step": 0.1,
         "min": 0},
        {"key": "settings.contours.min_snr_db", "label": "Minimum SNR (dB)", "kind": "number", "optional": True,
         "advanced": True, "help": "blank = no SNR filter"},
    ]),
    ("Windows", [
        {"key": "settings.windows.window_s", "label": "Window length (s)", "kind": "number", "step": 0.5, "min": 0.5},
        {"key": "settings.windows.overlap", "label": "Overlap (0-0.9)", "kind": "number", "step": 0.25, "min": 0,
         "max": 0.9},
        {"key": "settings.windows.min_contours", "label": "Min contours per window", "kind": "int", "min": 1},
        {"key": "settings.windows.assign_by", "label": "Contour belongs to a window by", "kind": "select",
         "options": {"overlap": "overlapping it", "midpoint": "its midpoint"}, "advanced": True},
        {"key": "settings.windows.min_occupancy", "label": "Min occupancy (0-1)", "kind": "number", "step": 0.05,
         "advanced": True, "help": "share of the window covered by contour points"},
        {"key": "settings.windows.min_longest_contour_s", "label": "Min longest contour (s)", "kind": "number",
         "step": 0.1, "advanced": True},
        {"key": "settings.windows.freq_profile_bins", "label": "Frequency profile bins", "kind": "int", "min": 0,
         "advanced": True},
    ]),
    ("Discovery (clustering)", [
        {"key": "settings.discover.cosine_threshold", "label": "Cluster breadth (cosine threshold)", "kind": "number",
         "step": 0.05, "min": 0.05, "max": 1.5, "help": "larger = fewer, broader clusters"},
        {"key": "settings.discover.min_cluster_size", "label": "Min windows per cluster", "kind": "int", "min": 2},
        {"key": "settings.discover.clusterer", "label": "Method", "kind": "select",
         "options": {"agglomerative": "agglomerative (threshold)", "hdbscan": "HDBSCAN (density)"}, "advanced": True},
        {"key": "settings.discover.euclid_outlier_q", "label": "Outlier check quantile", "kind": "number",
         "step": 0.01, "min": 0.5, "max": 1, "advanced": True},
    ]),
    ("Label verdicts", [
        {"key": "settings.reliability.min_label_days", "label": "Days a verdict's label must come from",
         "kind": "int", "min": 1},
        {"key": "settings.reliability.prior_strength", "label": "Shrinkage (pseudo-windows)", "kind": "number",
         "min": 0, "advanced": True, "help": "pulls small clusters towards the overall rate"},
        {"key": "settings.reliability.ci", "label": "Credible interval", "kind": "number", "step": 0.05,
         "min": 0.5, "max": 0.99, "advanced": True},
    ]),
    ("Call-type evaluation", [
        {"key": "settings.evaluate.max_per_class_per_day", "label": "Max windows per call type per day",
         "kind": "int", "advanced": True},
        {"key": "settings.evaluate.min_class_n", "label": "Min windows per call type", "kind": "int",
         "advanced": True},
        {"key": "settings.evaluate.rf_trees", "label": "Trees", "kind": "int", "advanced": True},
    ]),
]

WHISTLE = ("data.voc_type", "whistle")
CLICK = ("data.voc_type", "click")
FROM_BINARIES = ("data.source", "binaries")

DELPHINID = [
    ("Model", [
        {"key": "model_name", "label": "Model name", "kind": "text", "help": "used for the PAMGuard zip"},
        {"key": "labels.classes", "label": "Classes to use (blank = all)", "kind": "list", "help": "comma-separated"},
        {"key": "labels.group_col", "label": "Group column", "kind": "text", "advanced": True,
         "help": "independent unit for testing (default event_id)"},
    ]),
    ("Detection frames", [
        {"key": "settings.acoustic.frame_len", "label": "Frame length (s)", "kind": "number", "step": 0.5, "min": 0.5,
         "when": FROM_BINARIES},
        {"key": "settings.acoustic.hop", "label": "Step between frames (s)", "kind": "number", "step": 0.5, "min": 0.1,
         "when": FROM_BINARIES, "help": "smaller than the frame length = overlapping frames"},
        {"key": "settings.acoustic.fmin_hz", "label": "Lowest frequency (Hz)", "kind": "number", "step": 1000,
         "when": FROM_BINARIES, "help": "published models: whistles 2000-20000, clicks 10000-40000"},
        {"key": "settings.acoustic.fmax_hz", "label": "Highest frequency (Hz)", "kind": "number", "step": 1000,
         "when": FROM_BINARIES},
        {"key": "settings.acoustic.min_density", "label": "Min whistle density (0-1)", "kind": "number",
         "step": 0.01, "when": WHISTLE, "also": FROM_BINARIES, "help": "contour points per time slice in a frame"},
        {"key": "settings.acoustic.min_frag_ms", "label": "Shortest contour (ms)", "kind": "number", "step": 50,
         "when": WHISTLE, "also": FROM_BINARIES, "advanced": True},
        {"key": "settings.acoustic.min_clicks", "label": "Min clicks per frame", "kind": "int", "min": 1,
         "when": CLICK, "also": FROM_BINARIES},
        {"key": "settings.acoustic.fft_len", "label": "Click FFT length", "kind": "int", "when": CLICK,
         "also": FROM_BINARIES, "advanced": True},
        {"key": "settings.acoustic.smooth_window", "label": "Spectrum smoothing (bins)", "kind": "int",
         "when": CLICK, "also": FROM_BINARIES, "advanced": True},
        {"key": "settings.acoustic.downsample", "label": "Downsampling (bins averaged)", "kind": "int", "min": 1,
         "when": FROM_BINARIES, "advanced": True},
    ]),
    ("Training", [
        {"key": "settings.training.epochs", "label": "Epochs per bootstrap", "kind": "int", "min": 1},
        {"key": "settings.resampling.n_bootstraps", "label": "Bootstraps", "kind": "int", "min": 1,
         "help": "successive subsamples the model is trained on"},
        {"key": "settings.resampling.max_per_group", "label": "Max frames per group per bootstrap", "kind": "int",
         "min": 1},
        {"key": "settings.training.learning_rate", "label": "Learning rate", "kind": "number", "step": 0.0001,
         "advanced": True},
        {"key": "settings.training.batch_size", "label": "Batch size", "kind": "int", "min": 1, "advanced": True},
        {"key": "settings.training.patience", "label": "Early stopping patience (epochs)", "kind": "int",
         "advanced": True},
        {"key": "settings.training.seed", "label": "Random seed", "kind": "int", "advanced": True},
    ]),
    ("Cross-validation", [
        {"key": "settings.crossval.run", "label": "Test the model by cross-validation", "kind": "switch"},
        {"key": "settings.crossval.n_folds", "label": "Folds", "kind": "int", "min": 2, "optional": True,
         "help": "blank = leave one group out at a time (thorough, slow); e.g. 5 = five models"},
    ]),
    ("Network (shape is fixed)", [
        {"key": "settings.cnn.n_filters", "label": "Filters per layer", "kind": "int", "advanced": True},
        {"key": "settings.cnn.kernel_size", "label": "Kernel size", "kind": "int", "advanced": True},
        {"key": "settings.cnn.max_pool", "label": "Max pooling", "kind": "int", "advanced": True},
        {"key": "settings.cnn.dense_size", "label": "Dense units", "kind": "int", "advanced": True},
        {"key": "settings.cnn.dropout", "label": "Dropout (0-1)", "kind": "number", "step": 0.05, "advanced": True},
        {"key": "settings.cnn.leaky_relu", "label": "LeakyReLU slope", "kind": "number", "step": 0.05,
         "advanced": True},
        {"key": "settings.cnn.l2", "label": "L2 regularisation", "kind": "number", "step": 0.0001, "advanced": True},
    ]),
]

SETTINGS = {"randomforest": RANDOMFOREST, "calltypes": CALLTYPES, "delphinid": DELPHINID}


def applies(spec, cfg):
    """Whether a field applies to the current settings (``when`` / ``also``: (dotted key, value))."""
    from speciesclassifiers.project.project import get_path
    return all(get_path(cfg, k) == v for k, v in [c for c in (spec.get("when"), spec.get("also")) if c])
