# Read PAMGuard ROCCA contour statistics (RoccaContourStats*.csv) and generic detection tables.

ROCCA_PATTERN <- "RoccaContourStats"
WHISTLE_OMIT_DEFAULT <- c("DCQUARTER1MEAN", "DCQUARTER2MEAN", "DCQUARTER3MEAN", "DCQUARTER4MEAN",
                          "DCMEAN", "DCSTDDEV", "RMSSIGNAL", "OVERLAP")
CLICK_FEATURES <- c("DURATION", "FREQCENTER", "FREQPEAK", "BW3DB", "BW3DBLOW", "BW3DBHIGH", "BW10DB",
                    "BW10DBLOW", "BW10DBHIGH", "NCROSSINGS", "SWEEPRATE", "MEANTIMEZC", "MEDIANTIMEZC",
                    "VARIANCETIMEZC")

#' ROCCA feature columns
#'
#' Whistles: FREQMAX to STEPDUR; clicks: a fixed set of click measurements. Minus `omit`.
#' @param columns column names of a ROCCA table
#' @param voc_type "whistle" or "click"
#' @param omit features to leave out (default for whistles: DC means, RMSSIGNAL, OVERLAP)
rocca_features <- function(columns, voc_type = "whistle", omit = NULL) {
  if (is.null(omit) && voc_type == "whistle") omit <- WHISTLE_OMIT_DEFAULT
  if (voc_type == "whistle") {
    feats <- columns[match("FREQMAX", columns):match("STEPDUR", columns)]
  } else if (voc_type == "click") {
    feats <- intersect(CLICK_FEATURES, columns)
  } else {
    stop("voc_type must be 'whistle' or 'click'")
  }
  setdiff(feats, omit)
}

.read_rocca_file <- function(path, voc_type) {
  d <- utils::read.csv(path, check.names = FALSE)
  keep <- if (voc_type == "whistle") d$FREQMAX != 0 & d$FREQPEAK == 0 else d$FREQMAX == 0 & d$FREQPEAK != 0
  d <- d[which(keep), , drop = FALSE]
  if (nrow(d)) d$source_file <- basename(path)
  d
}

#' Read ROCCA contour stats
#'
#' @param path a RoccaContourStats CSV, a folder of them, or the root of a folder hierarchy
#' @param voc_type "whistle" or "click"
#' @param levels column names for each folder level (hierarchy layout), e.g.
#'   c("label", "location", "event_id"); NULL = labels from the CSV's KnownSpecies / EncounterID
#' @param omit features to leave out
#' @param filter_min,filter_max named lists of limits, e.g. list(DURATION = 0.1)
#' @return list(data = table with `label`, `event_id` and features, features = feature names)
read_rocca <- function(path, voc_type = "whistle", levels = NULL, omit = NULL,
                       filter_min = list(), filter_max = list()) {
  if (file.exists(path) && !dir.exists(path)) {
    parts <- list(.read_rocca_file(path, voc_type))
  } else if (!is.null(levels)) {
    files <- list.files(path, pattern = paste0(ROCCA_PATTERN, ".*\\.csv$"), recursive = TRUE)
    parts <- list()
    for (f in sort(files)) {
      dirs <- utils::head(strsplit(f, "/", fixed = TRUE)[[1]], -1)
      if (length(dirs) < length(levels)) next
      d <- .read_rocca_file(file.path(path, f), voc_type)
      if (!nrow(d)) next
      for (i in seq_along(levels)) d[[levels[i]]] <- dirs[i]
      parts[[length(parts) + 1]] <- d
    }
  } else {
    files <- list.files(path, pattern = paste0(ROCCA_PATTERN, ".*\\.csv$"), full.names = TRUE)
    parts <- lapply(sort(files), .read_rocca_file, voc_type = voc_type)
  }
  parts <- Filter(function(x) !is.null(x) && nrow(x) > 0, parts)
  if (!length(parts)) stop(sprintf("No %s*.csv data found in %s", ROCCA_PATTERN, path))
  cols <- Reduce(union, lapply(parts, names))
  d <- do.call(rbind, lapply(parts, function(x) { x[setdiff(cols, names(x))] <- NA; x[cols] }))
  if (is.null(levels) || !"label" %in% levels) d$label <- as.character(d$KnownSpecies)
  if (is.null(levels) || !"event_id" %in% levels) d$event_id <- as.character(d$EncounterID)
  feats <- rocca_features(names(d), voc_type, omit)
  for (v in names(filter_min)) d <- d[d[[v]] >= filter_min[[v]], , drop = FALSE]
  for (v in names(filter_max)) d <- d[d[[v]] <= filter_max[[v]], , drop = FALSE]
  n <- nrow(d)
  d <- d[stats::complete.cases(d[feats]), , drop = FALSE]
  if (nrow(d) < n) message(sprintf("Dropped %d rows with missing feature values", n - nrow(d)))
  rownames(d) <- NULL
  list(data = d, features = feats)
}

#' Read any detection table
#'
#' One row per detection, a label column, a group column and numeric feature columns.
#' @param path_or_df CSV path or data frame
#' @param features feature columns; NULL = every numeric column except label, group and `exclude`
#' @return list(data, features)
read_feature_table <- function(path_or_df, label_col = "label", group_col = "event_id",
                               features = NULL, exclude = character()) {
  d <- if (is.data.frame(path_or_df)) path_or_df else utils::read.csv(path_or_df, check.names = FALSE)
  for (c in c(label_col, group_col)) if (!c %in% names(d)) stop(sprintf("Column '%s' not found", c))
  if (is.null(features)) {
    num <- vapply(d, is.numeric, logical(1))
    features <- setdiff(names(d)[num], c(label_col, group_col, exclude))
  }
  n <- nrow(d)
  d <- d[stats::complete.cases(d[features]), , drop = FALSE]
  if (nrow(d) < n) message(sprintf("Dropped %d rows with missing feature values", n - nrow(d)))
  rownames(d) <- NULL
  list(data = d, features = features)
}
