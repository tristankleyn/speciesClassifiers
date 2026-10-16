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
    ("Whistle measurements", [
        {"key": "data.contours.fmin_hz", "label": "Lowest frequency (Hz)", "kind": "number", "step": 100,
         "help": "contour points outside the band are clipped off"},
        {"key": "data.contours.fmax_hz", "label": "Highest frequency (Hz)", "kind": "number", "step": 100},
        {"key": "data.contours.min_contour_s", "label": "Shortest contour (s)", "kind": "number", "step": 0.1,
         "min": 0},
    ]),
    ("Click measurements", [
        {"key": "data.click_spectrum.fmin_hz", "label": "Lowest frequency (Hz)", "kind": "number", "step": 1000,
         "help": "spectrum measurements use this band only"},
        {"key": "data.click_spectrum.fmax_hz", "label": "Highest frequency (Hz)", "kind": "number", "step": 1000,
         "optional": True, "help": "blank = half the sample rate"},
        {"key": "data.max_ici_s", "label": "Longest inter-click interval (s)", "kind": "number", "step": 0.1,
         "help": "a longer gap to the previous click counts as a pause (no interval)"},
        {"key": "data.click_spectrum.n_samples", "label": "Samples per click", "kind": "int", "advanced": True},
        {"key": "data.click_spectrum.nfft", "label": "FFT length", "kind": "int", "advanced": True},
        {"key": "data.click_spectrum.profile_bands", "label": "Spectrum profile bands", "kind": "int",
         "advanced": True},
    ]),
    ("Final layer (blocks)", [
        {"key": "final.block_minutes", "label": "Block length", "kind": "select",
         "options": {1: "1 min", 2: "2 min", 5: "5 min", 10: "10 min", 15: "15 min", 20: "20 min", 30: "30 min",
                     60: "1 hour"}},
        {"key": "final.n_trees", "label": "Trees", "kind": "int", "min": 10, "step": 100},
        {"key": "final.node_size", "label": "Min blocks per leaf", "kind": "int", "min": 1, "advanced": True},
    ]),
]
for _title, _specs in RANDOMFOREST[1:]:        # binaries only
    for _s in _specs:
        _s.setdefault("when", ("data.type", "binaries"))
        if _title.startswith("Whistle"):
            _s.setdefault("also", ("data.use_whistles", True))
        elif _title.startswith("Click"):
            _s.setdefault("also", ("data.use_clicks", True))
for _s in RANDOMFOREST[0][1]:
    if _s["key"] == "settings.max_per_group":
        _s["help"] = "picked at random, so big encounters/events don't dominate training"

USE_WHISTLES = ("data.use_whistles", True)
USE_CLICKS = ("data.use_clicks", True)

CALLTYPES = [
    ("Whistle contours", [
        {"key": "settings.contours.fmin_hz", "label": "Lowest frequency (Hz)", "kind": "number", "step": 100,
         "help": "contour points outside the band are clipped off"},
        {"key": "settings.contours.fmax_hz", "label": "Highest frequency (Hz)", "kind": "number", "step": 100},
        {"key": "settings.contours.min_contour_s", "label": "Shortest contour (s)", "kind": "number", "step": 0.1,
         "min": 0},
        {"key": "settings.contours.min_snr_db", "label": "Minimum SNR (dB)", "kind": "number", "optional": True,
         "advanced": True, "help": "blank = no SNR filter"},
    ]),
    ("Whistle windows", [
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
        {"key": "settings.features.whistle", "label": "Whistle features used", "kind": "features",
         "detector": "whistle", "advanced": True},
    ]),
    ("Click windows", [
        {"key": "settings.clicks.window_s", "label": "Window length (s)", "kind": "number", "step": 1, "min": 1,
         "help": "clicks come in trains: longer windows than for whistles"},
        {"key": "settings.clicks.overlap", "label": "Overlap (0-0.9)", "kind": "number", "step": 0.25, "min": 0,
         "max": 0.9},
        {"key": "settings.clicks.min_clicks", "label": "Min clicks per window", "kind": "int", "min": 2},
        {"key": "settings.clicks.max_ici_s", "label": "Longest inter-click interval (s)", "kind": "number",
         "step": 0.1, "min": 0.01, "advanced": True,
         "help": "longer gaps are pauses between click trains, not intervals"},
        {"key": "settings.clicks.min_ici_s", "label": "Shortest inter-click interval (s)", "kind": "number",
         "step": 0.001, "min": 0, "advanced": True, "help": "shorter gaps (e.g. echoes) are left out"},
        {"key": "settings.features.click", "label": "Click features used", "kind": "features",
         "detector": "click", "advanced": True},
    ]),
    ("Click spectrum", [
        {"key": "settings.click_spectrum.fmin_hz", "label": "Lowest frequency (Hz)", "kind": "number", "step": 1000,
         "min": 0, "help": "spectrum measurements use this band only"},
        {"key": "settings.click_spectrum.fmax_hz", "label": "Highest frequency (Hz)", "kind": "number", "step": 1000,
         "optional": True, "help": "blank = half the sample rate"},
        {"key": "settings.click_spectrum.n_samples", "label": "Samples per click", "kind": "int", "min": 16,
         "advanced": True, "help": "taken around the click's peak; clicks are often ~100 samples at 96 kHz"},
        {"key": "settings.click_spectrum.nfft", "label": "FFT length", "kind": "int", "min": 16, "advanced": True,
         "help": "zero-padded; frequency resolution = sample rate / FFT length"},
        {"key": "settings.click_spectrum.profile_bands", "label": "Spectrum profile bands", "kind": "int", "min": 0,
         "advanced": True},
    ]),
    ("Final decision", [
        {"key": "decision.method", "label": "How each block is decided", "kind": "select",
         "options": {"multiply": "Multiply evidence", "forest": "Random forest"},
         "help": "multiply each type's cluster probabilities, or a balanced random forest on the share of a "
                 "block's windows in each cluster (tested strictly: clusters redone without each held-out event; "
                 "slower)"},
        {"key": "decision.forest.n_trees", "label": "Trees", "kind": "int", "min": 10, "step": 100,
         "when": ("decision.method", "forest")},
        {"key": "decision.forest.min_samples_leaf", "label": "Min blocks per leaf", "kind": "int", "min": 1,
         "when": ("decision.method", "forest"), "advanced": True},
        {"key": "decision.forest.n_jobs", "label": "Held-out events tested at once", "kind": "int",
         "when": ("decision.method", "forest"), "advanced": True, "help": "-1 = one per CPU core"},
    ]),
    ("Combining whistles and clicks", [
        {"key": "decision.weights.whistle", "label": "Weight of whistles", "kind": "number", "step": 0.25, "min": 0,
         "also": USE_WHISTLES, "help": "1 = full say; 0 = ignored when clicks are there too (multiplying only)"},
        {"key": "decision.weights.click", "label": "Weight of clicks", "kind": "number", "step": 0.25, "min": 0,
         "also": USE_WHISTLES},
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

def _ct_defaults():
    from speciesclassifiers.project.workflows import ct_defaults
    return ct_defaults()


_CT_DEFAULTS = _ct_defaults()
for _title, _specs in CALLTYPES:      # whistle settings only with whistles, click settings only with clicks
    for _s in _specs:
        from speciesclassifiers.project.project import get_path as _gp
        if _s.get("kind") != "features" and _gp(_CT_DEFAULTS, _s["key"]) is not None:
            _s.setdefault("default", _gp(_CT_DEFAULTS, _s["key"]))     # shown for projects made before the setting
        if _title.startswith("Whistle"):
            _s.setdefault("when", USE_WHISTLES)
        elif _title.startswith(("Click", "Combining")):
            _s.setdefault("when", USE_CLICKS)

def _fill_defaults(groups, defaults):
    from speciesclassifiers.project.project import get_path
    for _t, specs in groups:
        for sp in specs:
            v = get_path(defaults, sp["key"])
            if sp.get("kind") != "features" and v is not None:
                sp.setdefault("default", v)


def _rf_defaults():
    from speciesclassifiers.project.workflows import rf_defaults
    return rf_defaults()


_fill_defaults(RANDOMFOREST, _rf_defaults())

MULTI = ("data.source", "multi")
DELPHINID += [
    ("Whistle frames", [
        {"key": "data.acoustic_by_type.whistle.fmin_hz", "label": "Lowest frequency (Hz)", "kind": "number",
         "step": 1000},
        {"key": "data.acoustic_by_type.whistle.fmax_hz", "label": "Highest frequency (Hz)", "kind": "number",
         "step": 1000},
        {"key": "data.acoustic_by_type.whistle.frame_len", "label": "Frame length (s)", "kind": "number", "step": 0.5},
        {"key": "data.acoustic_by_type.whistle.hop", "label": "Step between frames (s)", "kind": "number",
         "step": 0.5},
        {"key": "data.acoustic_by_type.whistle.min_density", "label": "Min whistle density (0-1)", "kind": "number",
         "step": 0.01},
        {"key": "data.acoustic_by_type.whistle.min_frag_ms", "label": "Shortest contour (ms)", "kind": "number",
         "step": 50, "advanced": True},
        {"key": "data.acoustic_by_type.whistle.downsample", "label": "Downsampling (bins averaged)", "kind": "int",
         "advanced": True},
    ]),
    ("Click frames", [
        {"key": "data.acoustic_by_type.click.fmin_hz", "label": "Lowest frequency (Hz)", "kind": "number",
         "step": 1000},
        {"key": "data.acoustic_by_type.click.fmax_hz", "label": "Highest frequency (Hz)", "kind": "number",
         "step": 1000},
        {"key": "data.acoustic_by_type.click.frame_len", "label": "Frame length (s)", "kind": "number", "step": 0.5},
        {"key": "data.acoustic_by_type.click.hop", "label": "Step between frames (s)", "kind": "number", "step": 0.5},
        {"key": "data.acoustic_by_type.click.min_clicks", "label": "Min clicks per frame", "kind": "int", "min": 1},
        {"key": "data.acoustic_by_type.click.fft_len", "label": "Click FFT length", "kind": "int", "advanced": True},
        {"key": "data.acoustic_by_type.click.smooth_window", "label": "Spectrum smoothing (bins)", "kind": "int",
         "advanced": True},
        {"key": "data.acoustic_by_type.click.downsample", "label": "Downsampling (bins averaged)", "kind": "int",
         "advanced": True},
    ]),
    ("Final layer (blocks)", [
        {"key": "final.block_minutes", "label": "Block length", "kind": "select",
         "options": {1: "1 min", 2: "2 min", 5: "5 min", 10: "10 min", 15: "15 min", 20: "20 min", 30: "30 min",
                     60: "1 hour"}},
        {"key": "final.n_trees", "label": "Trees", "kind": "int", "min": 10, "step": 100},
        {"key": "final.node_size", "label": "Min blocks per leaf", "kind": "int", "min": 1, "advanced": True},
    ]),
]
for _title, _specs in DELPHINID[-3:]:
    for _s in _specs:
        _s.setdefault("when", MULTI)
        if _title.startswith("Whistle"):
            _s.setdefault("also", ("data.use_whistles", True))
        elif _title.startswith("Click"):
            _s.setdefault("also", ("data.use_clicks", True))
for _title, _specs in DELPHINID:
    for _s in _specs:
        if _s["key"] in ("settings.crossval.run",):
            _s["help"] = "always on with several detection types (the final layer needs held-out predictions)"


def _dl_defaults():
    from speciesclassifiers.project.workflows import dl_defaults
    return dl_defaults()


_fill_defaults(DELPHINID, _dl_defaults())

def cluster_specs(det):
    """Clustering settings of one detection type (also shown on the map page)."""
    k = f"settings.cluster.{det}"
    return [
        {"key": f"{k}.min_cluster_size", "label": "Smallest cluster (points)", "kind": "int", "min": 2,
         "help": "larger = fewer, bigger clusters"},
        {"key": f"{k}.min_samples", "label": "Neighbourhood density", "kind": "int", "min": 0,
         "help": "0 = same as the smallest cluster; larger = more points left unclustered"},
        {"key": f"{k}.selection", "label": "Cluster selection", "kind": "select",
         "options": {"eom": "fewer, larger clusters", "leaf": "more, finer clusters"}},
        {"key": f"{k}.n_neighbors", "label": "Map neighbours (UMAP)", "kind": "int", "min": 2,
         "help": "larger = more global structure; smaller = more local detail"},
        {"key": f"{k}.min_dist", "label": "Map spacing (UMAP min distance)", "kind": "number", "step": 0.05,
         "min": 0, "max": 1, "help": "0 = tight clumps"},
        {"key": f"{k}.pca_variance", "label": "PCA variance kept (0-1)", "kind": "number", "step": 0.05,
         "min": 0.5, "max": 1, "advanced": True},
        {"key": f"{k}.cluster_dims", "label": "Cluster in dimensions", "kind": "int", "min": 2, "max": 20,
         "advanced": True, "help": "2 = on the map itself; more = a separate UMAP for clustering"},
        {"key": f"{k}.metric", "label": "Distance", "kind": "select",
         "options": {"euclidean": "Euclidean", "cosine": "cosine", "manhattan": "Manhattan"}, "advanced": True},
        {"key": f"{k}.seed", "label": "Random seed", "kind": "int", "advanced": True},
    ]


import copy as _copy
ANNOTATE = [_copy.deepcopy(g) for g in CALLTYPES if g[0] in ("Whistle contours", "Whistle windows", "Click windows",
                                                               "Click spectrum")]
UNIT_W, UNIT_C = "settings.units.whistle", "settings.units.click"
for _g in ANNOTATE:
    for _s in _g[1]:
        if _s.get("kind") == "features":
            _s.pop("advanced", None)
        if _g[0] == "Whistle windows":
            _s["also"] = (UNIT_W, "window")
        elif _g[0] == "Click windows" and "ici" not in _s["key"]:
            _s["also"] = (UNIT_C, "window")
    if _g[0] == "Whistle contours":
        _g[1].append({"key": "settings.features.whistle_detection", "label": "Contour features used",
                      "kind": "features", "detector": "whistle_detection", "when": USE_WHISTLES,
                      "also": (UNIT_W, "detection")})
    if _g[0] == "Click spectrum":
        _g[1].append({"key": "settings.features.click_detection", "label": "Click features used",
                      "kind": "features", "detector": "click_detection", "when": USE_CLICKS,
                      "also": (UNIT_C, "detection")})
ANNOTATE.insert(0, ("What each point on the map is", [
    {"key": UNIT_W, "label": "Whistles", "kind": "select", "when": USE_WHISTLES,
     "options": {"detection": "each contour", "window": "windows of contours"},
     "help": "single contours show call shapes best; windows summarise what is heard together"},
    {"key": UNIT_C, "label": "Clicks", "kind": "select", "when": USE_CLICKS,
     "options": {"detection": "each click", "window": "windows of clicks (trains)"},
     "help": "windows carry inter-click interval patterns; single clicks only their spectrum and the interval "
             "to the previous click (many points: slower)"}]))
ANNOTATE += [("Whistle clustering", [dict(s, when=USE_WHISTLES) for s in cluster_specs("whistle")]),
             ("Click clustering", [dict(s, when=USE_CLICKS) for s in cluster_specs("click")])]


# call types: single detections or windows (added after the copy above, so cluster-and-label keeps its own)
CT_UNIT_W, CT_UNIT_C = "settings.windows.unit", "settings.clicks.unit"
for _title, _specs in CALLTYPES:
    for _s in _specs:
        if _title == "Whistle windows":
            _s["also"] = (CT_UNIT_W, "window")
        elif _title == "Click windows" and "ici" not in _s["key"]:
            _s["also"] = (CT_UNIT_C, "window")
    if _title == "Whistle contours":
        _specs.append({"key": "settings.features.whistle_detection", "label": "Contour features used",
                       "kind": "features", "detector": "whistle_detection", "when": USE_WHISTLES,
                       "also": (CT_UNIT_W, "detection")})
    if _title == "Click spectrum":
        _specs.append({"key": "settings.features.click_detection", "label": "Click features used",
                       "kind": "features", "detector": "click_detection", "when": USE_CLICKS,
                       "also": (CT_UNIT_C, "detection")})
CALLTYPES.insert(0, ("What is clustered", [
    {"key": CT_UNIT_W, "label": "Whistles", "kind": "select", "when": USE_WHISTLES, "default": "window",
     "options": {"window": "windows of contours", "detection": "each contour"},
     "help": "single contours: their own shape; windows: what is heard together"},
    {"key": CT_UNIT_C, "label": "Clicks", "kind": "select", "when": USE_CLICKS, "default": "window",
     "options": {"window": "windows of clicks (trains)", "detection": "each click"},
     "help": "windows carry inter-click interval patterns; single clicks only their spectrum and the interval "
             "to the previous click"}]))


def _an_defaults():
    from speciesclassifiers.project.workflows import an_defaults
    return an_defaults()


_fill_defaults(ANNOTATE, _an_defaults())

SETTINGS = {"randomforest": RANDOMFOREST, "calltypes": CALLTYPES, "delphinid": DELPHINID, "annotate": ANNOTATE}
# values of keys that older projects may not have yet
CONDITION_DEFAULTS = {"data.use_whistles": True, "data.use_clicks": False, "decision.method": "multiply",
                      "settings.units.whistle": "detection", "settings.units.click": "window",
                      "settings.windows.unit": "window", "settings.clicks.unit": "window"}


def applies(spec, cfg):
    """Whether a field applies to the current settings (``when`` / ``also``: (dotted key, value))."""
    from speciesclassifiers.project.project import get_path

    def val(k):
        v = get_path(cfg, k)
        return CONDITION_DEFAULTS.get(k) if v is None else v
    return all(val(k) == v for k, v in [c for c in (spec.get("when"), spec.get("also")) if c])
