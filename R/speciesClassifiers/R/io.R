# Annotations / events tables and PAMGuard databases. Mirrors speciesclassifiers.io (Python).

# Times: ISO text ("2018-09-09 21:55:00", "2018-09-09T21:55:00.123Z") or Unix seconds -> Unix seconds (UTC)
.to_unix <- function(x) {
  if (is.numeric(x)) return(as.numeric(x))
  x <- sub("Z$", "", sub("T", " ", trimws(as.character(x))))
  t <- as.POSIXct(x, format = "%Y-%m-%d %H:%M:%OS", tz = "UTC")
  day <- is.na(t) & !is.na(x)
  t[day] <- as.POSIXct(x[day], format = "%Y-%m-%d", tz = "UTC")
  as.numeric(t)
}

.iso <- function(seconds, digits = 3) {
  format(as.POSIXct(seconds, origin = "1970-01-01", tz = "UTC"), paste0("%Y-%m-%d %H:%M:%OS", digits))
}

#' Read and check an annotations / events table
#'
#' Columns: event_id, label, start, end (UTC, e.g. 2018-09-09 21:55:00), plus any others.
#' Times become Unix seconds.
#' @param require_label FALSE allows unlabelled events (event_id, start, end)
read_annotations <- function(path_or_df, require_label = TRUE) {
  a <- if (is.data.frame(path_or_df)) path_or_df else utils::read.csv(path_or_df, check.names = FALSE)
  names(a) <- trimws(names(a))
  required <- c("event_id", if (require_label) "label", "start", "end")
  missing <- setdiff(required, names(a))
  if (length(missing)) stop("Annotations need columns ", paste(required, collapse = ", "),
                            "; missing ", paste(missing, collapse = ", "))
  for (c in c("start", "end")) {
    t <- .to_unix(a[[c]])
    if (anyNA(t)) stop(sprintf("Can't read '%s' times, e.g. '%s'. Use e.g. 2018-09-09 21:55:00", c, a[[c]][is.na(t)][1]))
    a[[c]] <- t
  }
  if (any(a$end <= a$start)) stop("Rows where end is not after start: ", paste(which(a$end <= a$start), collapse = ", "))
  a$event_id <- as.character(a$event_id)
  if ("label" %in% names(a)) a$label <- as.character(a$label)
  a
}

.db <- function(db) {
  if (!file.exists(db)) stop("Database not found: ", db)
  DBI::dbConnect(RSQLite::SQLite(), db, flags = RSQLite::SQLITE_RO)
}

#' Deep learning prediction tables in a PAMGuard database, with row counts
list_prediction_tables <- function(db) {
  con <- .db(db); on.exit(DBI::dbDisconnect(con))
  tabs <- grep("_Predictions$", DBI::dbListTables(con), value = TRUE)
  stats::setNames(vapply(tabs, function(t) DBI::dbGetQuery(con, sprintf('SELECT COUNT(*) n FROM "%s"', t))$n,
                         numeric(1)), tabs)
}

#' Class names from a PAMGuard model zip or .pdtf file
classes_from_model <- function(path) {
  if (grepl("\\.zip$", path)) {
    files <- utils::unzip(path, list = TRUE)$Name
    f <- grep("\\.pdtf$", files[!grepl("^__MACOSX", files)], value = TRUE)[1]
    tmp <- tempfile(); utils::unzip(path, f, exdir = tmp)
    path <- file.path(tmp, f)
  }
  unlist(jsonlite::fromJSON(path)$class_info$name_class)
}

#' Predictions from one deep learning table, in the standard classifier output format
#'
#' @param table table name, with or without the "_Predictions" suffix
#' @param classes class names in model output order
#' @param classifier name for the classifier in the output (default: table name)
#' @param voc_type "whistle", "click" or "other"
#' @return standard output with empty event_id and a `start` column (Unix s) for assigning events
read_dl_predictions <- function(db, table, classes, classifier = NULL, voc_type = "other") {
  if (!grepl("_Predictions$", table)) table <- paste0(table, "_Predictions")
  con <- .db(db); on.exit(DBI::dbDisconnect(con))
  cols <- DBI::dbListFields(con, table)
  pcol <- if ("Predicition" %in% cols) "Predicition" else "Prediction"
  d <- DBI::dbGetQuery(con, sprintf('SELECT UID, UTC, "%s" AS pred FROM "%s"', pcol, table))
  probs <- lapply(d$pred, function(p) {
    a <- tryCatch(as.numeric(jsonlite::fromJSON(p)$predictions[1, ]), error = function(e) NULL)
    if (is.null(a) || length(a) != length(classes) || any(is.na(a)) || any(a < 0)) NULL else a
  })
  keep <- !vapply(probs, is.null, logical(1))
  if (any(!keep)) message(sprintf("%s: dropped %d rows with missing/negative predictions or not %d classes",
                                  table, sum(!keep), length(classes)))
  d <- d[keep, , drop = FALSE]
  p <- if (nrow(d)) do.call(rbind, probs[keep]) else matrix(numeric(), 0, length(classes))
  p <- p / rowSums(p)
  name <- if (is.null(classifier)) sub("_Predictions$", "", table) else classifier
  k <- length(classes)
  start <- .to_unix(d$UTC)
  data.frame(
    detection_id = rep(paste0(name, ":", d$UID), each = k),
    event_id = "",
    time = rep(sub(" ", "T", paste0(.iso(start), "Z")), each = k),
    classifier = name, voc_type = voc_type,
    class = rep(classes, times = nrow(d)),
    probability = as.vector(t(p)),
    start = rep(start, each = k),
    stringsAsFactors = FALSE
  )
}

#' One event per recording file, from the Sound_Acquisition table
#'
#' Each file runs from its last Start row to its last row (Continue/Stop) within `max_hours`.
#' Files without a Start row are skipped. Returns event_id (file name), start, end (Unix s).
read_recordings <- function(db, max_hours = 24) {
  con <- .db(db); on.exit(DBI::dbDisconnect(con))
  d <- DBI::dbGetQuery(con, "SELECT UTC, Status, SystemName FROM Sound_Acquisition")
  d$Status <- trimws(d$Status); d$SystemName <- trimws(d$SystemName)
  d$t <- .to_unix(d$UTC)
  d <- d[!is.na(d$SystemName) & d$SystemName != "None", ]
  rows <- lapply(split(d, d$SystemName), function(g) {
    starts <- g$t[g$Status == "Start"]
    if (!length(starts)) return(NULL)
    t0 <- max(starts)
    after <- g$t[g$t >= t0 & g$t <= t0 + max_hours * 3600]
    if (max(after) > t0) data.frame(event_id = g$SystemName[1], start = t0, end = max(after)) else NULL
  })
  out <- do.call(rbind, rows)
  out <- out[order(out$start), ]
  rownames(out) <- NULL
  out
}
