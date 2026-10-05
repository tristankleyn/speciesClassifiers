ex <- function(f) {
  p <- file.path("..", "..", "..", "..", "examples", f)
  if (!file.exists(p)) p <- file.path(Sys.getenv("SC_EXAMPLES"), f)
  p
}

preds <- function() {
  mk <- function(id, t, clf, classes, p) data.frame(detection_id = id, event_id = "", time = t, classifier = clf,
                                                   voc_type = "other", class = classes, probability = p)
  rbind(mk("e1w0", "2024-01-01T00:00:10Z", "w", c("A", "B"), c(0.8, 0.2)),
        mk("e1w1", "2024-01-01T00:00:10Z", "w", c("A", "B"), c(0.6, 0.4)),
        mk("e1c2", "2024-01-01T00:00:10Z", "c", c("A", "B", "C"), c(0.1, 0.1, 0.8)),
        mk("e2c0", "2024-01-01T01:00:10Z", "c", c("A", "B", "C"), c(0.3, 0.3, 0.4)))
}

test_that("assign events and features", {
  events <- data.frame(event_id = c("E1", "E2"), label = c("X", "Y"), encounter = c("k1", "k2"),
                       start = c("2024-01-01 00:00:00", "2024-01-01 01:00:00"),
                       end = c("2024-01-01 00:10:00", "2024-01-01 01:10:00"))
  r <- event_features(assign_events(preds(), events))
  expect_equal(r$features, c("w|A", "w|B", "c|A", "c|B", "c|C"))
  e <- r$events
  expect_equal(e[e$event_id == "E1", "w|A"], 0.7)
  expect_equal(e[e$event_id == "E1", "n_w"], 2)
  expect_equal(e[e$event_id == "E1", "encounter"], "k1")
  expect_equal(unname(unlist(e[e$event_id == "E2", c("w|A", "w|B", "n_w")])), c(0, 0, 0))
  expect_equal(nrow(filter_events(e, list(w = 1, c = 1), "and")), 1)
  expect_equal(nrow(filter_events(e, list(w = 1, c = 1), "or")), 2)
})

test_that("time bins", {
  expect_setequal(unique(time_bin_events(preds(), 30)$event_id), c("2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"))
})

test_that("cross-validation on the synthetic example", {
  f <- ex("base_predictions_example_synthetic.csv")
  skip_if(!file.exists(f), "examples not found")
  a <- assign_events(utils::read.csv(f), ex("events_example_synthetic.csv"))
  r <- event_features(a)
  p <- tl_params(n_trees = 200)
  cv <- tl_cross_validate(r$events, r$features, group_col = "encounter", params = p, progress = FALSE)
  expect_equal(length(unique(cv$detection_id)), nrow(r$events))
  expect_gt(tl_summarise(cv)$event_accuracy, 0.6)
  m <- tl_fit(r$events, r$features, group_col = "encounter", params = p)
  out <- tl_predict(m, r$events[setdiff(names(r$events), r$features[1])])
  expect_equal(nrow(out), nrow(r$events))
})

test_that("PAMGuard database", {
  db <- Sys.getenv("SC_EXAMPLE_DB")
  skip_if(db == "", "SC_EXAMPLE_DB not set")
  expect_equal(list_prediction_tables(db)[["Deep_Learning_Classifier___Clicks_Predictions"]], 138)
  c <- read_dl_predictions(db, "Deep_Learning_Classifier___Clicks", c("Dde", "Ggr", "Gme", "Lal", "Ttr"), "clicks", "click")
  expect_equal(length(unique(c$detection_id)), 138)
  rec <- read_recordings(db)
  expect_equal(nrow(rec), 11)
  r <- event_features(assign_events(c, rec))
  expect_equal(sum(r$events$n_clicks), 138)
})
