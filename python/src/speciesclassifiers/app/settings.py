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

SETTINGS = {"randomforest": RANDOMFOREST, "calltypes": CALLTYPES}
