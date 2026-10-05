# Standard classifier output: see schemas/classifier_output.md

REQUIRED_COLUMNS <- c("detection_id", "event_id", "classifier", "voc_type", "class", "probability")
VOC_TYPES <- c("whistle", "click", "other")

#' Check that a data frame follows the standard classifier output format
#'
#' @param df data frame in long format (one row per detection x class)
#' @return `df`, invisibly, if valid; otherwise an error listing every problem
validate_output <- function(df) {
  missing <- setdiff(REQUIRED_COLUMNS, names(df))
  if (length(missing)) stop("Missing required columns: ", paste(missing, collapse = ", "))
  problems <- character()

  for (col in c("detection_id", "event_id", "classifier", "class")) {
    if (anyNA(df[[col]])) problems <- c(problems, sprintf("'%s' has missing values", col))
  }
  if (anyNA(df$voc_type) || !all(df$voc_type %in% VOC_TYPES)) {
    problems <- c(problems, sprintf("'voc_type' must be one of %s", paste(VOC_TYPES, collapse = ", ")))
  }

  prob <- suppressWarnings(as.numeric(df$probability))
  if (anyNA(prob) || any(prob < 0 | prob > 1)) {
    problems <- c(problems, "'probability' must be numeric between 0 and 1")
  } else {
    if (anyDuplicated(df[c("classifier", "detection_id", "class")])) {
      problems <- c(problems, "duplicate (classifier, detection_id, class) rows")
    }
    sums <- tapply(prob, paste(df$classifier, df$detection_id, sep = " / "), sum)
    off <- sums[abs(sums - 1) > 0.01]
    if (length(off)) {
      problems <- c(problems, sprintf("%d detection(s) whose probabilities don't sum to 1, e.g. %s",
                                      length(off), names(off)[1]))
    }
  }

  if ("time" %in% names(df)) {
    t <- as.POSIXct(sub("Z$", "", df$time), format = "%Y-%m-%dT%H:%M:%OS", tz = "UTC")
    if (any(is.na(t) & !is.na(df$time))) problems <- c(problems, "'time' has values that aren't ISO 8601 datetimes")
  }

  if (length(problems)) stop("Invalid classifier output:\n- ", paste(problems, collapse = "\n- "))
  invisible(df)
}

#' Long to wide: one row per (classifier, detection_id), one column per class
#'
#' @param df data frame in the standard classifier output format
#' @return wide data frame
to_wide <- function(df) {
  validate_output(df)
  id_cols <- intersect(c("classifier", "voc_type", "detection_id", "event_id", "time"), names(df))
  wide <- stats::reshape(df[c(id_cols, "class", "probability")], idvar = id_cols,
                         timevar = "class", direction = "wide")
  names(wide) <- sub("^probability\\.", "", names(wide))
  rownames(wide) <- NULL
  wide
}
