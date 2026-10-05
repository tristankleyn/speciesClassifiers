# Transfer learning: event classifiers retrained on the standard outputs of any base classifiers.
# Mirrors speciesclassifiers.transferlearning (Python).
#
# Each event gets, for every base classifier, the mean predicted probability of each of its classes
# (renormalised to sum 1), named "<classifier>|<class>". A base classifier with no predictions in an
# event contributes zeros. Detection counts (n_<classifier>) are kept for filtering, not as features.

FEATURE_SEP <- "|"

.pred_seconds <- function(p) {
  if ("start" %in% names(p)) return(p$start)
  if ("time" %in% names(p)) return(.to_unix(p$time))
  stop("Predictions need a 'time' (ISO) or 'start' (Unix s) column to assign events")
}

#' Give each prediction the event whose period contains its time
#'
#' @param events CSV path or table (event_id, start, end, optional label and other columns), or the
#'   output of read_recordings(). Predictions outside every event are dropped; event columns are copied.
assign_events <- function(predictions, events) {
  ev <- read_annotations(events, require_label = FALSE)
  t <- .pred_seconds(predictions)
  base <- predictions[setdiff(names(predictions), setdiff(names(ev), "event_id"))]
  out <- list()
  for (i in seq_len(nrow(ev))) {
    m <- t >= ev$start[i] & t < ev$end[i]
    if (any(m)) {
      d <- base[m, , drop = FALSE]
      for (c in setdiff(names(ev), c("start", "end"))) d[[c]] <- ev[[c]][i]
      out[[length(out) + 1]] <- d
    }
  }
  if (!length(out)) stop("No predictions fall inside any event period")
  res <- do.call(rbind, out); rownames(res) <- NULL
  res
}

#' Events as fixed, clock-aligned time bins, named by their start time (UTC)
time_bin_events <- function(predictions, minutes = 10) {
  b <- floor(.pred_seconds(predictions) / (60 * minutes)) * 60 * minutes
  predictions$event_id <- paste0(sub(" ", "T", .iso(b, 0)), "Z")
  predictions
}

#' Event features
#'
#' @param keep_cols event-level columns to carry over (default: columns constant within every event)
#' @return list(events = one row per event, features = feature column names)
event_features <- function(predictions, keep_cols = NULL) {
  validate_output(predictions[setdiff(names(predictions), "score")])
  p <- predictions
  key <- paste(p$classifier, p$class, sep = FEATURE_SEP)
  pairs <- unique(data.frame(classifier = p$classifier, class = p$class, key = key, stringsAsFactors = FALSE))
  features <- pairs$key
  evs <- unique(p$event_id)
  mean_tab <- tapply(p$probability, list(p$event_id, key), mean)
  mean_tab <- mean_tab[evs, features, drop = FALSE]
  mean_tab[is.na(mean_tab)] <- 0
  for (cl in unique(pairs$classifier)) {           # renormalise within each classifier
    cols <- pairs$key[pairs$classifier == cl]
    s <- rowSums(mean_tab[, cols, drop = FALSE])
    mean_tab[s > 0, cols] <- mean_tab[s > 0, cols, drop = FALSE] / s[s > 0]
  }
  counts <- tapply(p$detection_id, list(p$event_id, p$classifier), function(x) length(unique(x)))
  counts <- counts[evs, , drop = FALSE]; counts[is.na(counts)] <- 0
  colnames(counts) <- paste0("n_", colnames(counts))
  if (is.null(keep_cols)) {
    skip <- c("detection_id", "event_id", "time", "start", "classifier", "voc_type", "class", "probability",
              "score", "true_class", "test_group", "fold")
    cand <- setdiff(names(p), skip)
    keep_cols <- cand[vapply(cand, function(c) all(tapply(p[[c]], p$event_id, function(x) length(unique(x)) <= 1)),
                             logical(1))]
  }
  first <- p[match(evs, p$event_id), keep_cols, drop = FALSE]
  t <- .pred_seconds(p)
  out <- data.frame(event_id = evs, first, first_detection = as.vector(tapply(t, p$event_id, min)[evs]),
                    last_detection = as.vector(tapply(t, p$event_id, max)[evs]),
                    as.data.frame(unclass(counts)), as.data.frame(unclass(mean_tab)),
                    check.names = FALSE, stringsAsFactors = FALSE)
  rownames(out) <- NULL
  list(events = out, features = features)
}

#' Keep events with enough detections
#'
#' @param min_detections named list, e.g. list(delphinID_clicks = 5, delphinID_whistles = 3)
#' @param mode "or" keeps events meeting any minimum, "and" only those meeting all
filter_events <- function(events, min_detections, mode = "or") {
  if (!length(min_detections)) return(events)
  ok <- sapply(names(min_detections), function(c) {
    n <- events[[paste0("n_", c)]]
    if (is.null(n)) n <- rep(0, nrow(events))
    n >= min_detections[[c]]
  })
  ok <- matrix(ok, nrow = nrow(events))
  keep <- if (mode == "or") apply(ok, 1, any) else apply(ok, 1, all)
  out <- events[keep, , drop = FALSE]; rownames(out) <- NULL
  out
}

#' Random forest settings for event classifiers (defaults as Classify-delphinID)
tl_params <- function(n_trees = 1000, mtry = NULL, node_size = 5, prune = 0, seed = 42) {
  rf_params(n_trees, mtry, node_size, max_per_group = NULL, prune = prune, seed = seed)
}

#' Leave-one-group-out cross-validation of an event classifier
#'
#' @return event predictions in the standard output format: detection_id = event, event_id = group
tl_cross_validate <- function(events, features, label_col = "label", group_col = "event_id", params = tl_params(),
                              classifier = "eventclassifier", progress = TRUE) {
  rf_cross_validate(events, features, label_col, group_col, params, classifier = classifier, voc_type = "other",
                    id_col = "event_id", progress = progress, unit = "events")
}

#' One row per event: class probabilities, predicted, score, group, true_class
tl_event_table <- function(predictions) {
  w <- stats::reshape(predictions[c("detection_id", "class", "probability")], idvar = "detection_id",
                      timevar = "class", direction = "wide")
  names(w) <- sub("^probability\\.", "", names(w))
  classes <- setdiff(names(w), "detection_id")
  pm <- as.matrix(w[classes])
  w$predicted <- classes[max.col(pm, ties.method = "first")]
  w$score <- decision_score(pm)
  first <- predictions[match(w$detection_id, predictions$detection_id), ]
  w$group <- first$event_id
  if ("true_class" %in% names(first)) w$true_class <- first$true_class
  names(w)[names(w) == "detection_id"] <- "event"
  rownames(w) <- NULL
  w
}

#' Event-level accuracy; events scoring below `min_score` count as discarded
tl_summarise <- function(predictions, min_score = 0) {
  ev <- tl_event_table(predictions)
  kept <- ev[ev$score >= min_score, ]
  correct <- kept$predicted == kept$true_class
  by_class <- tapply(correct, kept$true_class, mean)
  list(event_accuracy = mean(correct), mean_class_accuracy = mean(by_class), accuracy_by_class = by_class,
       events_discarded = 1 - nrow(kept) / max(nrow(ev), 1),
       confusion = table(true = kept$true_class, predicted = kept$predicted), events = ev)
}

#' Event accuracy and share of events kept across score thresholds
tl_threshold_curve <- function(predictions, thresholds = seq(0, 0.5, by = 0.02)) {
  do.call(rbind, lapply(thresholds, function(t) {
    s <- tl_summarise(predictions, t)
    data.frame(threshold = t, event_accuracy = s$event_accuracy, mean_class_accuracy = s$mean_class_accuracy,
               events_kept = 1 - s$events_discarded)
  }))
}

#' Train the final event classifier on all events
tl_fit <- function(events, features, label_col = "label", group_col = "event_id", params = tl_params()) {
  rf_fit(events, features, label_col, group_col, params)
}

#' Classify new events; features the new data lack are 0. One row per event.
tl_predict <- function(model, events) {
  for (f in setdiff(model$features, names(events))) events[[f]] <- 0
  p <- rf_predict_proba(model, events)
  out <- events[setdiff(names(events), model$features)]
  for (c in model$classes) out[[c]] <- p[, c]
  out$predicted <- model$classes[max.col(p, ties.method = "first")]
  out$score <- decision_score(p)
  out
}
