# Transfer learning

Event classifiers for new labels, retrained on the outputs of classifiers you already have (delphinID models run in PAMGuard, random forests, call-type libraries, ...) without retraining those. Python and R.

Notebooks: `04_transferlearning` (`.ipynb` and `.qmd`).

## How it works
1. Base-classifier predictions (standard output format: CSVs, or deep learning tables in a PAMGuard database) are grouped into **events**: labelled periods (`assign_events`), recordings, or fixed time bins (`time_bin_events`).
2. Each event is described by the **mean probability of each class from each base classifier** (`classifier|class`). A base classifier with no predictions in an event contributes zeros. Detection counts are reported but not used as features.
3. A random forest (`event_params()`: 1000 trees, node size 5) maps event features to labels, tested leaving one group out at a time.

`filter_events` drops events with too few detections. Predictions for new events come out in the standard output format.
