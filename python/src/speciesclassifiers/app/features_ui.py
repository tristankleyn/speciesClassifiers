"""Settings: view and choose the features (measurements) of each detection type."""

import numpy as np
import pandas as pd
from nicegui import run, ui

from speciesclassifiers.calltypes.windows import PROFILE_PREFIX, default_features, feature_catalog

NON_LABELS = {"none", "partial", "mixed", "untyped", "unknown", "unlabelled", "n/a", "nan", "", "AMBIGUOUS"}


def _fmt(v):
    if v is None or not np.isfinite(v):
        return ""
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3g}"


def feature_stats(project, detector):
    """Per feature: median and 5th-95th percentile over all windows, and the median per class of the first
    label column (from the windows stage, if it has run). Returns (rows by key, class column, classes)."""
    from speciesclassifiers.project.workflows import _windows_file, load_pickle
    f = _windows_file(project, detector.split("_")[0])
    if not f.is_file():
        return {}, None, []
    w = load_pickle(f)
    cfg = project.config()
    tree = (cfg.get("labels") or {}).get("pathway") or {}
    col = tree.get("column")
    classes = []
    if col and col in w:
        v = w[col].astype(str)
        classes = [c for c in v.value_counts().index if c not in NON_LABELS][:6]
    out = {}
    for item in feature_catalog(detector):
        k = item["key"]
        if k in PROFILE_PREFIX and detector.endswith("_detection") and not any(
                c.startswith(PROFILE_PREFIX[k]) for c in w.columns):
            continue
        if k in PROFILE_PREFIX:
            n = sum(c.startswith(PROFILE_PREFIX[k]) for c in w.columns)
            out[k] = {"median": f"{n} bands" if n else "", "range": ""}
            continue
        if k not in w:
            continue
        x = pd.to_numeric(w[k], errors="coerce")
        q = x.quantile([0.05, 0.5, 0.95]).to_numpy() if x.notna().any() else [np.nan] * 3
        r = {"median": _fmt(q[1]), "range": f"{_fmt(q[0])} to {_fmt(q[2])}" if np.isfinite(q[0]) else ""}
        for c in classes:
            r[f"c_{c}"] = _fmt(x[w[col].astype(str) == c].median())
        out[k] = r
    return out, col, classes


def features_field(saver, spec):
    """A settings row: how many features are used, and a dialog to view their values and choose them."""
    det, key = spec["detector"], spec["key"]
    cat = feature_catalog(det)
    defaults = default_features(det)

    def chosen():
        v = saver.get(key)
        return list(defaults if v is None else v)

    def summary():
        c = chosen()
        txt = f"{len(c)} of {len(cat)} used"
        return txt + (" (the defaults)" if set(c) == set(defaults) else "")

    def save(keys):
        if not keys:
            ui.notify("Choose at least one feature.", type="warning")
            return False
        saver.set(key, None if set(keys) == set(defaults) else [i["key"] for i in cat if i["key"] in set(keys)])
        lbl.text = summary()
        return True

    async def open_dialog():
        with ui.dialog() as dlg, ui.card().classes("w-[72rem] max-w-full"):
            ui.label(spec["label"]).classes("text-lg font-medium")
            ui.label("Ticked features describe each point (window or detection) for clustering. Values are from your "
                     "data once they have been made (Run). Level-dependent features (SNR, number of clicks) "
                     "depend on distance to the animals, so they're off by default.").classes(
                "text-sm text-gray-600")
            holder = ui.column().classes("w-full")
            with holder:
                ui.spinner()
        dlg.open()
        stats, col, classes = await run.io_bound(feature_stats, saver.project, det)
        holder.clear()
        cols = [{"name": "group", "label": "Group", "field": "group", "align": "left"},
                {"name": "label", "label": "Feature", "field": "label", "align": "left"},
                {"name": "median", "label": "Median", "field": "median"},
                {"name": "range", "label": "5th to 95th percentile", "field": "range"}]
        cols += [{"name": f"c_{c}", "label": f"{col} {c}", "field": f"c_{c}"} for c in classes]
        rows = [{**i, **stats.get(i["key"], {})} for i in cat]
        with holder:
            if not stats:
                ui.label("No windows yet: run the project to see the values.").classes("text-xs text-gray-500")
            tbl = ui.table(columns=cols, rows=rows, row_key="key", selection="multiple",
                           pagination=0).classes("w-full overflow-x-auto").props("dense flat")
            tbl.selected = [r for r in rows if r["key"] in set(chosen())]

            def set_sel(keys):
                tbl.selected = [r for r in rows if r["key"] in set(keys)]
                tbl.update()

            with ui.row().classes("gap-2 items-center"):
                ui.button("Defaults", on_click=lambda: set_sel(defaults)).props("flat dense")
                ui.button("All", on_click=lambda: set_sel([r["key"] for r in rows])).props("flat dense")
                ui.space()
                ui.button("Cancel", on_click=dlg.close).props("flat")
                ui.button("Use these", icon="check",
                          on_click=lambda: save([r["key"] for r in tbl.selected]) and dlg.close())

    with ui.row().classes("items-center gap-2 w-full no-wrap"):
        ui.label(spec["label"] + ":").classes("text-sm")
        lbl = ui.label(summary()).classes("text-sm")
        ui.button("View and choose", icon="checklist", on_click=open_dialog).props("outline dense no-caps")
