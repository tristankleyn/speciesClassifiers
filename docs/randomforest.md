# Random forests from scratch

Detection-level random forests on ROCCA contour stats or any feature table, tested by leaving out one group (e.g. encounter) at a time. Python and R give the same workflow (`randomforest` / `rf_*`).

Notebooks: `03_randomforest` (`.ipynb` and `.qmd`).

## Parameters (`RFParams` / `rf_params()`)
- `n_trees` (500), `mtry` (default √features), `node_size` (25)
- `max_per_group` (25): at most this many detections per group, picked at random, so large encounters don't dominate
- `prune` (0): share of each class's training detections to drop as outliers (furthest from their class centre on the first two principal components)
- `seed`

Classes are balanced in every tree (balanced random forest: each tree samples the same number of detections per class).

## Outputs
- Detection predictions in the standard output format, plus a **decision score** p1 × (p1 − p2) (top probability × its margin over the second).
- Group (event) predictions: mean probabilities of the detections above a minimum score.
- Accuracy summaries and a threshold curve (group accuracy vs share of groups kept as the minimum group score rises).
- The fitted model (`.joblib` / `.rds`) and `predict` for new data.
