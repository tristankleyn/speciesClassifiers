# Call types

Build a call-type library from whistle contours and classify new recordings with it. Python only. Notebook: `05_calltypes` (runs on a synthetic example by default).

## Steps
1. **Contours** (`ContourParams`): points outside `fmin_hz`–`fmax_hz` are clipped off; contours shorter than `min_contour_s` are dropped; optional `min_snr_db` or `snr_top_frac`. Each contour is measured (duration, start/end/min/max/mean frequency, slopes, curvature, inflections, ...).
2. **Windows** (`WindowParams`): sliding windows (`window_s`, `overlap`) on a grid aligned to each day's first contour (or one window per annotated call, `mode="annotation"`). A contour belongs to a window if it overlaps it (or its midpoint is inside, `assign_by="midpoint"`). Features: mean and spread of the contour measurements, the longest contour's measurements, tonal time and occupancy, contour-point frequency quantiles and a frequency profile (`freq_profile_bins`). Windows with too few contours, too little occupancy or too short a longest contour are dropped.
3. **Labels**: `label_windows` gives each window the call it overlaps most among calls it covers at least `min_call_coverage` of; windows that only clip a call are `partial`, windows with more than one call type `mixed`, others `none`. `label_windows_by_period` / `label_windows_by_overlap` add labels from longer periods or other annotation sources.
4. **Evaluate** (`EvalParams`): labelled windows, capped per call type and day, in a random forest cross-validated by day (or time block). Reports balanced accuracy, skill (0 = chance, 1 = perfect), confusion matrix, feature importance.
5. **Discover** (`DiscoverParams`, `ReliabilityParams`): all windows (labelled, untyped, and optionally unannotated days) are standardised and weighted by random-forest importance for the known call types, then clustered on cosine distance (agglomerative with `cosine_threshold`, or HDBSCAN). An outlier check removes windows far (Euclidean) from every member of their cluster recorded in other time blocks. Clusters are summarised by their call types and days, with candidate new call types flagged.
   - **Label tasks** (`LabelTask(name, column, positive, negative)`): each cluster's share of `positive` is shrunk towards the overall rate (`prior_strength`) with a credible interval (`ci`). The verdict is `positive` if the interval is above the overall rate, `negative` if below, else `uncertain`; the winning label must come from at least `min_label_days` days.
   - **Leave-one-day-out**: verdicts recomputed without each day, then used to predict it; reported per window and for longer decision blocks (`decision_windows`).
6. **Library**: `build_library` freezes everything into JSON ([format](data_formats.md#call-type-library-json)); `self_check` re-assigns the training windows to confirm it.
7. **Classify**: `classify_contours(whistles, library)` → windows and decision blocks; `standard_output` turns windows into the standard classifier output (probability = the cluster's shrunk share of each label), e.g. for transfer learning.

## Assigning new windows
A window goes to the cluster with the nearest centroid (weighted cosine distance), unless that distance exceeds `cosine_threshold` or its Euclidean distance to the cluster's nearest member exceeds the cluster's outlier threshold: then it is unassigned (`-1`, prediction `uncertain`). It takes its cluster's verdict for each task.

## Decision blocks
Windows are combined per clock-aligned block of `block_minutes`. Per task, the decision is the label with a **strict majority** of the windows that have a verdict (`uncertain` otherwise; `no_detections` for empty blocks when a `period` is given).

- **`requires`**: report a task only when another task's decision is a given value, e.g. `{"ecotype": ("species", "X")}`; otherwise `n/a`.
- **Rules** (none by default): checked in order after voting; the first that matches sets every task's decision to its `then` value. `raw_<task>` keeps the vote, `rule` names the rule that fired.

```python
rules = [
    {"if": "block_meanfreq_range_hz", "op": "<", "value": 200, "then": "noise"},
    {"if": "block_median_abs_slope_hz_s", "op": ">", "value": 6000, "then": "noise"},
]
```

| Block value | |
|---|---|
| `n_windows`, `n_assigned`, `n_contours` | counts in the block |
| `block_meanfreq_range_hz` | range of the contours' mean frequencies |
| `block_median_abs_slope_hz_s` | median over windows of the mean absolute contour slope ("wobble") |
| `block_median_freq_hz` | median over windows of the mean contour-point frequency |
| `block_median_contour_s` | median contour duration |

Operators: `<`, `<=`, `>`, `>=`, `==`, `!=`. Change a saved library's rules with `with_rules(library, rules)`.

## Standalone classifier
`python/src/speciesclassifiers/calltypes/standalone.py` is one file that needs only numpy. Copy it with `library.json` to wherever classification runs (a server, a scheduled job). It reads contours as a list of dicts (see [data formats](data_formats.md#whistle-contours); `contour_from_bins` converts peak FFT bins and slice numbers using the library's `detector` settings) and gives the same windows and blocks as `classify_contours`; the tests check this.

```
python standalone.py library.json contours.json blocks.csv [windows.csv]
```
```python
import standalone
lib = standalone.load_library("library.json")
windows, blocks = standalone.classify(contours, lib, period=(t0, t1))
```
