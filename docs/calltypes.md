# Call types

Build a call-type library from whistle contours (and, optionally, click detections) and classify new recordings with it. Python only. Notebook: `05_calltypes` (runs on a synthetic example by default).

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

## Labels from periods, multi-class layers and pathways
- **Labelled periods** (`event_id, start, end` + label columns such as `species`): each window takes the labels of the period it falls in; no call types needed. Without call types, the cluster feature weights are learned from the label being classified.
- **Multi-class layers** (`multiclass.py`): every cluster gets a probability for each class (counts capped per event, classes balanced, shrunk towards the overall rate) and a verdict when one class is reliably above its overall rate on at least `min_label_days` events (or days). Tested by leaving out one event (or day) at a time; a block's class probabilities are the average of its windows'; the block is decided if its score p1 × (p1 − p2) reaches the layer's minimum score.
- **Pathways** (`pathway.py`): a tree of layers, e.g. layer 1 decides a group (values grouped into classes), and a next layer within a group decides among that group's values, clustered afresh on that group's windows. New data goes down the tree; a block's `decision` is the deepest confident decision and `path` shows the route (e.g. `A > Dd`). Libraries store the tree, and `standalone.py` follows it.

```python
tree = {"name": "group", "column": "species", "groups": {"A": ["Dd", "Sco"], "B": ["pilot", "Gg"]},
        "children": {"A": {"name": "species_A", "column": "species"}}}
pr = discover_pathway(pool, tree, DiscoverParams(), ReliabilityParams(unit="event_id"))
lib = build_pathway_library(pr, contour_params, window_params, min_scores={"group": 0.05})
```

## Feature choice
Each detection type's features can be chosen (`DiscoverParams.features` / `EvalParams.features`: a list of keys from `windows.feature_catalog(detector)`; `freq_profile` stands for all frequency-profile bands). By default every whistle feature is used except the level-dependent SNR; for clicks the inter-click interval and spectrum features (the number of clicks is available but off). The library stores the columns used, so new data gets the same set. In the app: Settings → Advanced → *Features used*, with each feature's median and 5th–95th percentile in your windows, and per class.

## Clicks, and combining detection types
- **Click windows** (`clicks.py`, `ClickWindowParams`): sliding windows (default 10 s, 50 % overlap) over click times. Features: mean, median, SD, 5th and 95th percentile of the inter-click intervals, and the click count. Gaps longer than `max_ici_s` (default 1 s) are pauses between trains, not intervals; gaps shorter than `min_ici_s` are left out (e.g. echoes). Windows need `min_clicks` clicks. All clicks in a window are pooled.
- **Click spectra** (`ClickSpectrumParams`, `click_spectra`): measured while the binaries are read, so the waveforms aren't kept (about 15 % slower to read). Each click: `n_samples` (default 128) around its largest sample, Hann taper, `nfft`-point FFT (default 256: 375 Hz resolution at 96 kHz); within `fmin_hz` (5 kHz) to `fmax_hz` (half the sample rate): peak frequency, centroid, −3 dB and −10 dB bandwidths (the contiguous band around the peak above half / a tenth of the peak power) and the share of power in `profile_bands` (16) bands. Windows: median over clicks of each measurement, interquartile range of peak and centroid, and the mean band profile. A clicks CSV can carry the per-click measurements as columns (`peak_hz`, `centroid_hz`, `bw3_hz`, `bw10_hz`, `csp_<lo>_<hi>`).
- **Input**: PAMGuard Click Detector binaries (`io.read_clicks`), or a CSV with `start` (Unix s or UTC date-time) and optional `duration` (s).
- **Labels**: click windows take the labels of the labelled period (or, with call annotations, the event's first-to-last call) they fall in.
- **Each type on its own**: whistles and clicks get their own windows, features, clusters and held-out tests, through every layer of the pathway. A class needn't exist in both.
- **Merging** (`multiclass.merge_evidence`, `detectors.py`): per block and layer, each type's evidence for a class is its probability over the class's overall rate; the merged probabilities are the product of the evidence (each to the power of the type's weight, default 1), normalised. A class a type hasn't learned is neutral (×1) for that type; a block with only one type keeps that type's probabilities. The block is then decided as usual and goes down to the next layer. A rule that matches a type's block (e.g. "noise") removes that type's evidence; if no other type has evidence, the block takes the rule's decision.
- **Held-out test**: each type's held-out windows are averaged per block and merged the same way (`merge_heldout`), so the app shows combined accuracy as well as each type's own.

```python
from speciesclassifiers.calltypes.detectors import build_multi_library, classify_multi
lib = build_multi_library({"whistle": whistle_lib, "click": click_lib}, weights={"whistle": 1, "click": 1})
windows, blocks = classify_multi({"whistle": whistles, "click": clicks}, lib)
```

### Random forest final layer (`forest.py`; `decision.method = "forest"`)
Instead of multiplying evidence, each layer can decide its blocks with a balanced random forest (every tree sees the same number of blocks of each class). A block's inputs are the share of its windows in each cluster (and unassigned), per detection type, side by side; types missing from a block are 0. Works with whistles alone too.

- **Strict test**: for every held-out event, each detection type is clustered again without it (feature weights, clusters, outlier thresholds), all windows are assigned to those clusters as new data would be, the forest is trained on the other events' blocks and predicts the held-out event's blocks. Nothing about the held-out event is used. This means one clustering run per event and type, run in parallel (`decision.forest.n_jobs`); expect minutes to tens of minutes.
- **Final forest**: trained on all labelled blocks with the full clustering; stored in the library (`forest`, one per layer with `children`) as plain arrays, so `standalone.py` applies it with numpy only and gives the same decisions.
- Outputs (`forest/`): `heldout_<layer>.csv` (held-out block predictions), `importance_<layer>.csv` (which cluster shares matter), `layers.csv`. Rules still apply: a type whose block matches a rule is left out of that block.
- Settings: `n_trees` (500), `min_samples_leaf` (1), `n_jobs` (-1 = one per core). The block score and minimum-score thresholds work as before.

Blocks have, per layer, the merged decision, score and probabilities, plus each type's own decision (`<layer>_whistle`, `<layer>_click`) and window counts.

The yes/no tasks with majority-vote blocks (as in the killer whale example) remain available.

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
python standalone.py library.json contours.json blocks.csv --clicks clicks.csv   # library with clicks ("-" = no contours)
```
```python
import standalone
lib = standalone.load_library("library.json")
windows, blocks = standalone.classify(contours, lib, period=(t0, t1))
```
