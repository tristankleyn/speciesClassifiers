example <- function() {
  data.frame(
    detection_id = c("w1", "w1", "c1", "c1"),
    event_id = "e1",
    time = rep(c("2024-09-18T10:00:00Z", "2024-09-18T10:00:04Z"), each = 2),
    classifier = rep(c("dID-w", "dID-c"), each = 2),
    voc_type = rep(c("whistle", "click"), each = 2),
    class = c("Dde", "Ttr", "Dde", "Ttr"),
    probability = c(0.7, 0.3, 0.55, 0.45)
  )
}

test_that("valid output passes", {
  expect_silent(validate_output(example()))
})

test_that("probabilities must sum to 1", {
  df <- example(); df$probability[1] <- 0.9
  expect_error(validate_output(df), "sum to 1")
})

test_that("missing column is caught", {
  expect_error(validate_output(example()[-2]), "Missing")
})

test_that("to_wide gives one row per detection", {
  w <- to_wide(example())
  expect_equal(nrow(w), 2)
  expect_true(all(c("Dde", "Ttr") %in% names(w)))
})
