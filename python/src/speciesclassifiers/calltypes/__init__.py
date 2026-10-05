"""Call-type libraries from whistle contours: windowed contour features, evaluation of how well
labelled call types separate, clustering to discover call types, and a frozen library for
classifying new data."""

from .contours import ContourParams, contour_params, prepare_contours
from .discover import (DiscoverParams, LabelTask, ReliabilityParams, cluster_label_reliability,
                       decision_window_performance, discover, discovery_pool, hybrid_cluster,
                       loo_label_prediction, summarise_clusters, task_stats)
from .evaluate import (EvalParams, apply_feature_spec, evaluate, feature_matrix, fit_feature_spec,
                       results_row, select_rows)
from .library import (apply_rules, assign_windows, block_votes, build_library, check_rules, classify_contours,
                      classify_windows, load_library, save_library, self_check, with_rules)
from .windows import (AGG_PARAMS, WindowParams, label_windows, label_windows_by_overlap, label_windows_by_period,
                      make_windows, window_features)

__all__ = ["ContourParams", "contour_params", "prepare_contours", "WindowParams", "make_windows",
           "window_features", "label_windows", "label_windows_by_period", "label_windows_by_overlap", "AGG_PARAMS", "EvalParams",
           "select_rows", "fit_feature_spec", "apply_feature_spec", "feature_matrix", "evaluate", "results_row",
           "DiscoverParams", "LabelTask", "ReliabilityParams", "discovery_pool", "hybrid_cluster", "summarise_clusters",
           "task_stats", "cluster_label_reliability", "loo_label_prediction", "decision_window_performance", "discover",
           "build_library", "save_library", "load_library", "with_rules", "check_rules", "assign_windows",
           "classify_windows", "block_votes", "apply_rules", "classify_contours", "self_check"]
