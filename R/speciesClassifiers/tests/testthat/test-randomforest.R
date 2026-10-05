make_data <- function(n_groups = 4, n = 20, seed = 0) {
  set.seed(seed)
  rows <- list()
  for (c in c("A", "B")) {
    mu <- if (c == "A") 0 else 2
    for (g in seq_len(n_groups)) {
      rows[[length(rows) + 1]] <- data.frame(label = c, event_id = paste0(c, g), x1 = rnorm(n, mu),
                                             x2 = rnorm(n, -mu), noise = rnorm(n))
    }
  }
  do.call(rbind, rows)
}

test_that("decision score", {
  expect_equal(decision_score(rbind(c(0.7, 0.2, 0.1), c(0.5, 0.5, 0))), c(0.7 * 0.5, 0))
})

test_that("cap and prune", {
  d <- make_data()
  expect_equal(max(table(cap_per_group(d, "event_id", 5)$event_id)), 5)
  expect_equal(nrow(prune_outliers(d, c("x1", "x2"), "label", 0.1)), 0.9 * nrow(d), tolerance = 4 / nrow(d))
})

test_that("cross-validation, save and predict", {
  r <- read_feature_table(make_data())
  expect_equal(r$features, c("x1", "x2", "noise"))
  p <- rf_params(n_trees = 50, node_size = 5, max_per_group = 10)
  preds <- rf_cross_validate(r$data, r$features, params = p, progress = FALSE)
  expect_silent(validate_output(preds[setdiff(names(preds), "score")]))
  s <- rf_summarise(preds)
  expect_equal(s$group_accuracy, 1)
  expect_equal(rf_threshold_curve(preds)$groups_kept[1], 1)
  m <- rf_fit(r$data, r$features, params = p)
  f <- tempfile(fileext = ".rds"); rf_save(m, f)
  out <- rf_predict(rf_load(f), transform(r$data, recording = paste0(event_id, "_rec")), group_col = "recording")
  expect_setequal(out$groups$event_id, paste0(unique(r$data$event_id), "_rec"))
  expect_error(rf_predict(m, r$data[setdiff(names(r$data), "x1")]), "missing features")
})

test_that("ROCCA folder hierarchy", {
  root <- tempfile(); cols <- c("Source", "EncounterID", "KnownSpecies", "FREQMAX", "FREQMIN", "DCMEAN",
                                "DURATION", "STEPDUR", "FREQPEAK")
  for (sp in c("Dde", "Ttr")) for (enc in c("e1", "e2")) {
    dir.create(file.path(root, sp, enc), recursive = TRUE)
    d <- as.data.frame(matrix(c("s", "x", "x", 8000, 5000, 1, 0.5, 0.01, 0), nrow = 3, ncol = 9, byrow = TRUE))
    names(d) <- cols
    for (c in cols[4:9]) d[[c]] <- as.numeric(d[[c]])
    utils::write.csv(d, file.path(root, sp, enc, "RoccaContourStats_1.csv"), row.names = FALSE)
  }
  r <- read_rocca(root, levels = c("label", "event_id"))
  expect_equal(r$features, c("FREQMAX", "FREQMIN", "DURATION", "STEPDUR"))
  expect_setequal(unique(r$data$label), c("Dde", "Ttr"))
})

test_that("real ROCCA file", {
  f <- Sys.getenv("SC_ROCCA_CSV")
  skip_if(f == "", "SC_ROCCA_CSV not set")
  r <- read_rocca(f)
  preds <- rf_cross_validate(r$data, r$features, params = rf_params(n_trees = 200), progress = FALSE)
  expect_gt(rf_summarise(preds)$group_accuracy, 0.6)
})
