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

## Several detection types from binaries, with a final layer per block
In the app, data type **PAMGuard binaries** (or the contours JSON / clicks CSV formats of the examples) with an annotations CSV of labelled periods (`event_id, start, end` + a label column such as `species`):

1. **Base classifiers**, one per detection type (`randomforest.detections`):
   - whistles: each contour's shape measurements (as in [call types](calltypes.md): duration, start/end/min/max/mean frequency, range, SD, slopes, curvature, rising/falling shares, inflections, where the extremes fall);
   - clicks: each click's spectrum (peak and centroid frequency, −3/−10 dB bandwidths, band profile) and the interval to the previous click (missing if longer than `max_ici_s`).
   Detections inside a labelled period take its event and label. Each type is tested by leaving one event out: every event's detections are predicted by a forest trained without that event.
2. **Final layer** (`speciesclassifiers.stacking`): those held-out probabilities are averaged per clock-aligned block (`final.block_minutes`) and per type, and put side by side: `<type>|<class>` (0 if the type has no detections in the block) and `has_<type>` (1/0). A balanced random forest on the block vectors is tested by leaving one event out (`final/heldout_final.csv`); `heldout_final__<type>.csv` shows what each type alone decides.
3. **New data**: the base forests trained on all events predict each detection; the final forest (trained on all blocks' held-out vectors) decides each block (`apply/blocks.csv`), with the minimum block score set on the Results screen.

The same final layer is used for delphinID with several detection types (next).
