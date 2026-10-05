# Example data

Small files so every notebook runs straight away. Files named `*_synthetic*` are **made up**, not real recordings; they only show the formats and let the workflows run. Formats: [docs/data_formats.md](../docs/data_formats.md).

| File | Used by | What it is |
|---|---|---|
| `annotations_example.csv` | 01 | Annotations format (labelled encounter periods) |
| `frames_example_synthetic.csv`, `frames_example_synthetic_params.json` | 02 | delphinID whistle frames and the acoustic parameters that "made" them |
| `RoccaContourStats_example_synthetic.csv` | 03 | ROCCA contour stats (real column layout, made-up values) |
| `base_predictions_example_synthetic.csv`, `events_example_synthetic.csv` | 04 | Base-classifier predictions (standard output format) and labelled events |
| `contours_example_synthetic.json.gz`, `call_annotations_example_synthetic.csv` | 05 | 4 days of whistle contours (4 call types, 2 "species", 2 "ecotypes", plus tonal noise on day 4) and call annotations for days 1–3 |

`scripts/` has the scripts that generate the synthetic files (`python examples/scripts/make_calltype_example.py`, ...).
