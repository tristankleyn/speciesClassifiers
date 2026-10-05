# Standard classifier output

Every classifier in this repo (delphinID, random forest, call types) writes its predictions in this format. `transferlearning` reads only this format, so it works with any base classifier.

One row per **detection × class** ("long" format). A detection can be a single detection, a detection frame/window, or any other unit the classifier scores.

| Column | Type | Required | Description |
|---|---|---|---|
| `detection_id` | string | yes | Unique ID of the scored unit within a classifier |
| `event_id` | string | yes | Group the detection belongs to (encounter, event, location, trial, …). Used for grouped training and aggregation |
| `time` | ISO 8601 datetime (UTC) | no | Start time of the detection |
| `classifier` | string | yes | Name of the classifier that made the prediction (include a version if useful, e.g. `delphinID-whistle-v1`) |
| `voc_type` | string | yes | `whistle`, `click` or `other` |
| `class` | string | yes | Class label |
| `probability` | float, 0–1 | yes | Predicted probability for `class` |

Rules:
- Each (`classifier`, `detection_id`) has one row per class, and its probabilities sum to 1 (± 0.01).
- Extra columns are allowed and kept (e.g. `n_detections`, `channel`, `true_class`).
- Files are CSV, UTF-8, with a header.

Example:
```
detection_id,event_id,time,classifier,voc_type,class,probability
w001,enc12,2024-09-18T10:00:00Z,delphinID-whistle-v1,whistle,Dde,0.71
w001,enc12,2024-09-18T10:00:00Z,delphinID-whistle-v1,whistle,Ttr,0.29
c001,enc12,2024-09-18T10:00:04Z,delphinID-click-v1,click,Dde,0.55
c001,enc12,2024-09-18T10:00:04Z,delphinID-click-v1,click,Ttr,0.45
```

Helpers to validate the format and convert it to wide format (one column per class):
- Python: `speciesclassifiers.outputs.validate_output()`, `to_wide()`
- R: `validate_output()`, `to_wide()`
