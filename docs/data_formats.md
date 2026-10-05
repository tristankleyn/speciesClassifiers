# Data formats

All times are UTC. In tables, times are text (`2024-06-01 08:06:15.006`) or Unix seconds; the packages use Unix seconds internally.

## Annotations (`examples/annotations_example.csv`)
Labelled periods, e.g. encounters, used to label detections and detection frames.

| Column | Required | |
|---|---|---|
| `event_id` | yes | Encounter / event the period belongs to (used for grouped cross-validation) |
| `label` | yes | Class (e.g. species code); not needed for unlabelled new data |
| `start`, `end` | yes | Period start and end |
| others | no | Kept (e.g. `location`) |

Read with `io.read_annotations` (Python) / `read_annotations` (R).

## Call annotations (`examples/call_annotations_example_synthetic.csv`)
One row per annotated call, same columns as annotations, with the call type as `label`. Extra label columns (e.g. `species`, `ecotype`) are copied to windows by `calltypes.label_windows(..., carry=[...])`. Calls of no particular type can be labelled `untyped`.

Annotation tables (one row per selection, with begin/end times relative to a recording or as clock times) convert to this format with `io.read_annotation_table`.

## Whistle contours
- From PAMGuard binaries: `io.read_whistles(folder)` gives one row per contour: `start` (Unix s), `duration`, `times` (s, per slice), `freqs` (Hz, per slice), `slice_s`, `snr_db`, `amplitude_db`, `uid`, `file`. The detector's sample rate and FFT settings are worked out from the files.
- As JSON (`examples/contours_example_synthetic.json.gz`, the input of `calltypes/standalone.py`): a list of `{"start", "times", "freqs", "slice_s", "snr_db" (optional)}`.

## Detection frames (`examples/frames_example_synthetic.csv`)
delphinID model inputs, one row per frame: `event_id`, `frame_start`, `n_detections`, `density` (whistles), `label`, and the input bins `f1`, `f2`, .... The acoustic parameters that made them are saved next to them (`*_params.json`).

## Feature tables (`examples/RoccaContourStats_example_synthetic.csv`)
One row per detection with numeric features, a label column and a group column (e.g. `event_id`). ROCCA contour stats files are read with `randomforest.read_rocca`; any other table with `read_feature_table`.

## Standard classifier output
Every classifier writes predictions in one long format (one row per detection × class): see [schemas/classifier_output.md](../schemas/classifier_output.md).

## Call-type library (JSON)
Written by `calltypes.save_library`; everything needed to classify new contours:
- `contour_params`, `window_params`: filters and window settings
- `feature_spec`: feature columns, fill values, log-transformed columns, standardisation (`mean`, `scale`)
- `weights`: feature weights of the cosine distance
- `cosine_threshold`, `clusters` (each: `id`, `centroid`, `euclid_thr`, `members` as base64 float32, `verdicts` per task)
- `tasks`, `requires`, `block_minutes`, `rules`, optional `detector` (sample rate, FFT length, hop)
