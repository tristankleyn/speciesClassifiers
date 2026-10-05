# Random forest classifiers built from scratch on detection features.
# Mirrors speciesclassifiers.randomforest (Python): same parameters, same outputs.

#' Random forest settings
#'
#' @param n_trees trees in the forest
#' @param mtry features tried at each split (NULL = square root of the number of features)
#' @param node_size minimum detections in a leaf
#' @param max_per_group most detections taken (at random) from one group for training
#' @param prune share (0-1) of each class's training detections to drop: those furthest from their
#'   class centre on the first two principal components
#' @param seed random seed
rf_params <- function(n_trees = 500, mtry = NULL, node_size = 25, max_per_group = 25, prune = 0, seed = 42) {
  list(n_trees = n_trees, mtry = mtry, node_size = node_size, max_per_group = max_per_group,
       prune = prune, seed = seed)
}

#' Decision score per row of a probability matrix: p1 * (p1 - p2)
decision_score <- function(probs) {
  probs <- as.matrix(probs)
  apply(probs, 1, function(p) {
    p <- sort(p, decreasing = TRUE)
    p[1] * (p[1] - if (length(p) > 1) p[2] else 0)
  })
}

#' At most `n_max` random rows per group
cap_per_group <- function(df, group_col, n_max, seed = 0) {
  if (is.null(n_max) || !n_max) return(df)
  set.seed(seed)
  idx <- unlist(lapply(split(seq_len(nrow(df)), df[[group_col]]), function(i) {
    if (length(i) > n_max) i[sample.int(length(i), n_max)] else i
  }), use.names = FALSE)
  df[sort(idx), , drop = FALSE]
}

#' Drop the `prune` share of each class furthest from its class centre in PC1-PC2 space
prune_outliers <- function(df, features, label_col, prune) {
  if (!prune) return(df)
  pcs <- stats::prcomp(df[features], scale. = FALSE)$x[, 1:2, drop = FALSE]
  keep <- rep(TRUE, nrow(df))
  for (c in unique(df[[label_col]])) {
    m <- which(df[[label_col]] == c)
    d <- sqrt(rowSums(sweep(pcs[m, , drop = FALSE], 2, colMeans(pcs[m, , drop = FALSE]))^2))
    keep[m[d >= stats::quantile(d, 1 - prune)]] <- FALSE
  }
  df[keep, , drop = FALSE]
}

#' Fit a random forest
#'
#' Each tree is grown on the same number of detections from every class (the size of the smallest
#' class), as `randomForest(strata =, sampsize =)`. Training data are pruned and capped per group
#' as set in `params`. Missing feature values are filled with training medians.
#' @return an `sc_randomforest` object
rf_fit <- function(df, features, label_col = "label", group_col = "event_id", params = rf_params()) {
  train <- prune_outliers(df, features, label_col, params$prune)
  train <- cap_per_group(train, group_col, params$max_per_group, params$seed)
  y <- factor(as.character(train[[label_col]]))
  x <- train[features]
  medians <- vapply(x, stats::median, numeric(1), na.rm = TRUE)
  for (f in features) x[[f]][is.na(x[[f]])] <- medians[[f]]
  set.seed(params$seed)
  mtry <- if (is.null(params$mtry)) max(1, floor(sqrt(length(features)))) else params$mtry
  forest <- randomForest::randomForest(x = x, y = y, ntree = params$n_trees, mtry = mtry,
                                       nodesize = params$node_size, strata = y,
                                       sampsize = rep(min(table(y)), nlevels(y)), importance = TRUE)
  structure(list(forest = forest, features = features, classes = levels(y), medians = medians,
                 params = params, label_col = label_col), class = "sc_randomforest")
}

#' Class probabilities for new detections (columns in `model$classes` order)
rf_predict_proba <- function(model, df) {
  x <- df[model$features]
  for (f in model$features) x[[f]][is.na(x[[f]])] <- model$medians[[f]]
  p <- stats::predict(model$forest, x, type = "prob")
  p[, model$classes, drop = FALSE]
}

#' Feature importance (mean decrease in Gini impurity), largest first
rf_importance <- function(model) {
  imp <- randomForest::importance(model$forest, type = 2)[, 1]
  sort(imp, decreasing = TRUE)
}

#' Detection predictions in the standard classifier output format, plus `score`
detection_output <- function(df, probs, classes, classifier, voc_type, label_col = NULL, id_col = NULL,
                             event_col = "event_id") {
  n <- nrow(df); k <- length(classes)
  ids <- if (is.null(id_col)) rownames(df) else as.character(df[[id_col]])
  out <- data.frame(
    detection_id = rep(ids, each = k),
    event_id = rep(as.character(df[[event_col]]), each = k),
    classifier = classifier, voc_type = voc_type,
    class = rep(classes, times = n),
    probability = as.vector(t(as.matrix(probs))),
    score = rep(decision_score(probs), each = k),
    stringsAsFactors = FALSE
  )
  if (!is.null(label_col) && label_col %in% names(df)) out$true_class <- rep(as.character(df[[label_col]]), each = k)
  out
}

#' Group predictions: mean detection probabilities per group, from detections with score >= min_score
group_predictions <- function(detections, min_score = 0, group_col = "event_id") {
  wide <- stats::reshape(detections[c("detection_id", group_col, "class", "probability")],
                         idvar = c("detection_id", group_col), timevar = "class", direction = "wide")
  names(wide) <- sub("^probability\\.", "", names(wide))
  classes <- sort(unique(detections$class))
  first <- detections[!duplicated(detections$detection_id), ]
  wide$score <- first$score[match(wide$detection_id, first$detection_id)]
  groups <- unique(wide[[group_col]])
  rows <- lapply(groups, function(g) {
    w <- wide[wide[[group_col]] == g, ]
    used <- w[w$score >= min_score, classes, drop = FALSE]
    p <- if (nrow(used)) colMeans(used) else stats::setNames(rep(NA_real_, length(classes)), classes)
    r <- data.frame(g, t(p), check.names = FALSE, stringsAsFactors = FALSE)
    names(r)[1] <- group_col
    r$predicted <- if (nrow(used)) classes[which.max(p)] else NA
    r$score <- if (nrow(used)) decision_score(t(p)) else NA
    r$n <- nrow(used)
    r$n_total <- nrow(w)
    if ("true_class" %in% names(first)) r$true_class <- first$true_class[match(w$detection_id[1], first$detection_id)]
    r
  })
  do.call(rbind, rows)
}

#' Leave-one-group-out cross-validation
#'
#' For each group, a forest trained on all other groups predicts every detection of that group.
#' @return detection predictions (standard classifier output, plus `score`, `true_class`, `test_group`)
rf_cross_validate <- function(df, features, label_col = "label", group_col = "event_id", params = rf_params(),
                              classifier = "randomforest", voc_type = "whistle", id_col = NULL, progress = TRUE) {
  df[[label_col]] <- as.character(df[[label_col]])
  groups <- unique(df[[group_col]])
  out <- list()
  for (k in seq_along(groups)) {
    g <- groups[k]
    test <- df[df[[group_col]] == g, , drop = FALSE]
    rest <- df[df[[group_col]] != g, , drop = FALSE]
    if (!test[[label_col]][1] %in% rest[[label_col]]) {
      message(sprintf("[%d/%d] %s: skipped, its class has no other groups to train on", k, length(groups), g))
      next
    }
    m <- rf_fit(rest, features, label_col, group_col, params)
    p <- rf_predict_proba(m, test)
    d <- detection_output(test, p, m$classes, classifier, voc_type, label_col, id_col, group_col)
    d$test_group <- as.character(g)
    out[[length(out) + 1]] <- d
    if (progress) {
      mp <- colMeans(p)
      cat(sprintf("[%d/%d] %s (%s): predicted %s (score %.2f, %d detections)\n", k, length(groups), g,
                  test[[label_col]][1], m$classes[which.max(mp)], decision_score(t(mp)), nrow(test)))
    }
  }
  preds <- do.call(rbind, out)
  rownames(preds) <- NULL
  validate_output(preds)
  preds
}

#' Accuracy of cross-validation predictions
#'
#' @param min_score detections below this are left out of group predictions
#' @param min_group_score groups below this are counted as discarded
rf_summarise <- function(detections, min_score = 0, min_group_score = 0, group_col = "event_id") {
  g <- group_predictions(detections, min_score, group_col)
  g <- g[!is.na(g$predicted), ]
  kept <- g[g$score >= min_group_score, ]
  correct <- kept$predicted == kept$true_class
  by_class <- tapply(correct, kept$true_class, mean)
  list(group_accuracy = mean(correct), mean_class_accuracy = mean(by_class), accuracy_by_class = by_class,
       groups_discarded = 1 - nrow(kept) / max(nrow(g), 1),
       confusion = table(true = kept$true_class, predicted = kept$predicted), groups = g)
}

#' Group accuracy and share of groups kept across group score thresholds
rf_threshold_curve <- function(detections, thresholds = seq(0, 0.5, by = 0.02), group_col = "event_id") {
  do.call(rbind, lapply(thresholds, function(t) {
    s <- rf_summarise(detections, min_group_score = t, group_col = group_col)
    data.frame(threshold = t, group_accuracy = s$group_accuracy, mean_class_accuracy = s$mean_class_accuracy,
               groups_kept = 1 - s$groups_discarded)
  }))
}

#' Apply a model to new detections
#' @return list(detections = detection predictions, groups = group predictions)
rf_predict <- function(model, df, group_col = "event_id", classifier = "randomforest", voc_type = "whistle",
                       id_col = NULL, min_score = 0) {
  missing <- setdiff(model$features, names(df))
  if (length(missing)) stop("New data is missing features the model uses: ", paste(missing, collapse = ", "))
  p <- rf_predict_proba(model, df)
  d <- detection_output(df, p, model$classes, classifier, voc_type, NULL, id_col, group_col)
  list(detections = d, groups = group_predictions(d, min_score))  # groups are in the event_id column
}

#' Save / load a model (.rds)
rf_save <- function(model, path) { saveRDS(model, path); invisible(path) }
rf_load <- function(path) readRDS(path)
