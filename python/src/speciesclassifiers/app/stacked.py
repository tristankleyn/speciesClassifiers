"""App screens shared by methods with base classifiers per detection type and a final forest per block
(random forest and delphinID from binaries)."""

from pathlib import Path

import pandas as pd
from nicegui import run, ui

from speciesclassifiers.project import checks

from .widgets import ImageSlot, path_row, show_check, table

TYPE_NAMES = {"whistle": "Whistles", "click": "Clicks"}
SOURCES = {"whistle": {"binaries": "PAMGuard binaries (folder)", "json": "Contours JSON (.json or .json.gz)"},
           "click": {"binaries": "PAMGuard binaries (folder)", "csv": "Clicks CSV (.csv or .csv.gz)"}}


def detection_types_panel(saver, prefix="data", annotations=True):
    """Tick whistles and/or clicks; source and path of each; the labelled periods (training)."""
    with ui.row().classes("gap-6"):
        for ty, default in (("whistle", True), ("click", False)):
            ui.checkbox(TYPE_NAMES[ty], value=saver.get(f"data.use_{ty}s", default),
                        on_change=lambda e, ty=ty: (saver.set(f"data.use_{ty}s", e.value), body.refresh()))

    async def check(ty, out):
        out.clear()
        with out:
            ui.spinner()
        d = saver.project.config()[prefix]
        src, path = d.get(f"{ty}_source", "binaries"), d.get(f"{ty}_path", "")
        r = await run.io_bound(checks.check_contours if ty == "whistle" else checks.check_clicks, src, path)
        show_check(out, r)

    async def check_periods(out):
        out.clear()
        with out:
            ui.spinner()
        d = saver.project.config()["data"]
        r = await run.io_bound(checks.check_annotations, {"source": "periods", "path": d.get("annotations", "")})
        show_check(out, r)

    @ui.refreshable
    def body():
        cfg = saver.project.config()
        d = cfg["data"]
        types = [t for t, dflt in (("whistle", True), ("click", False)) if d.get(f"use_{t}s", dflt)]
        if not types:
            ui.label("Choose at least one detection type.").classes("text-amber-700")
        for ty in types:
            ui.label(TYPE_NAMES[ty]).classes("text-base font-medium mt-2")
            key = f"{prefix}.{ty}_source"
            ui.select(SOURCES[ty], value=saver.get(key, "binaries"), label="Source",
                      on_change=lambda e, key=key: saver.set(key, e.value)).classes("w-96")
            path_row(saver, f"{prefix}.{ty}_path", "Folder or file", "any", (".json", ".csv", ".gz"))
            if prefix == "data":
                ui.button(f"Check {ty}s", icon="fact_check", on_click=lambda ty=ty: check(ty, outs[ty]))
                outs[ty] = ui.column().classes("w-full")
        if annotations:
            ui.label("Labelled periods").classes("text-base font-medium mt-2")
            ui.markdown("A CSV with `event_id, start, end` (UTC) and label columns (e.g. `species`). Detections "
                        "inside a period are labelled with it; testing leaves one event out at a time.")
            path_row(saver, "data.annotations", "Annotations CSV", "file", (".csv",))
            from .widgets import field
            field(saver, {"key": "data.label_col", "label": "Label column", "kind": "text",
                          "help": "the column with the classes, e.g. species"})
            field(saver, {"key": "data.classes", "label": "Classes to use (blank = all)", "kind": "list",
                          "help": "comma-separated"})
            ui.button("Check annotations", icon="fact_check", on_click=lambda: check_periods(pout))
            pout = ui.column().classes("w-full")
    outs = {}
    body()


async def stacked_results(p, saver, base_note):
    """Per-type base classifier results (held-out events) and the final forest per block."""
    import json
    from .results import layer_section, metric, not_ready
    from speciesclassifiers.randomforest import summarise
    ui.markdown("#### Base classifiers\n" + base_note)
    if not_ready(p, "crossval", "cross-validation results"):
        return
    with ui.row().classes("gap-3"):
        for f in sorted((p.folder / "crossval").glob("cv_*.csv")):
            ty = f.stem[3:]
            if ty == "predictions" or ty == "detection_predictions":
                continue
            preds = await run.io_bound(pd.read_csv, f, dtype={"event_id": str})
            s = await run.io_bound(summarise, preds, 0.0, 0.0)
            metric(f"{TYPE_NAMES.get(ty, ty)}: event accuracy", s["group_accuracy"],
                   hint=f"{len(s['groups'])} events, mean probabilities")
    ui.markdown("#### Final layer\nEach type's held-out probabilities are averaged per block (each event's "
                "detections were predicted by a model trained without that event) and the types are put side "
                "by side, with a 'has detections' input per type. A balanced random forest on these block "
                "vectors is tested by leaving one event out. Switch to a type to see what it alone decides "
                "(its averaged probabilities).")
    if not_ready(p, "final", "final layer results"):
        return
    meta = json.loads((p.folder / "final" / "heldout_final.json").read_text())
    variants = {"final/heldout_final": "Final forest"}
    for ty in meta.get("types") or []:
        if (p.folder / "final" / f"heldout_final__{ty}.json").is_file():
            variants[f"final/heldout_final__{ty}"] = TYPE_NAMES.get(ty, ty) + " alone"
    await layer_section(p, saver, "final", meta.get("block_minutes", 10), variants=variants,
                        title="Block decisions", note="Thresholds apply to the final forest's decisions for "
                        "new data (after classifying again).")
    imp = p.folder / "final" / "importance.csv"
    if imp.is_file():
        with ui.expansion("Final forest: which inputs matter", icon="park").classes("w-full"):
            table(pd.read_csv(imp).round(4), rows=15)


async def stacked_apply(p, saver, start_run, outputs):
    """New data per detection type, block decisions over time, outputs."""
    from .results import ImageSlot as _IS, _csv, metric, output_row, timeline_fig  # noqa: F401
    cfg = saver.project.config()
    ui.markdown("#### Classify new data\nThe base classifiers predict each detection; the final forest decides "
                "each clock-aligned block. Optionally limit to a date range.")
    d = cfg["data"]
    for ty, dflt in (("whistle", True), ("click", False)):
        if not d.get(f"use_{ty}s", dflt):
            continue
        ui.label(TYPE_NAMES[ty]).classes("text-sm font-medium mt-2")
        key = f"new_data.{ty}_source"
        ui.select(SOURCES[ty], value=saver.get(key, "binaries"), label="Source",
                  on_change=lambda e, key=key: saver.set(key, e.value)).classes("w-80")
        path_row(saver, f"new_data.{ty}_path", "Folder or file", "any", (".json", ".csv", ".gz"))
    from .widgets import field
    with ui.row().classes("gap-4"):
        field(saver, {"key": "new_data.start", "label": "From day (YYYY-MM-DD, optional)", "kind": "text"})
        field(saver, {"key": "new_data.end", "label": "To day (inclusive, optional)", "kind": "text"})
    ui.button("Classify new data", icon="play_arrow", on_click=lambda: start_run(["apply"]))
    st = p.stage_states()
    bl = p.folder / "apply" / "blocks.csv"
    if st.get("apply") in ("done", "outdated") and bl.is_file():
        if st["apply"] == "outdated":
            ui.label("Settings changed since: classify again to update.").classes("text-amber-700")
        blocks = _csv(bl)
        with ui.row().classes("gap-3"):
            for k, v in blocks["decision"].value_counts().items():
                metric(str(k), int(v), "{:d}", "blocks")
        if len(blocks):
            ui.markdown("#### Presence over time")
            slot = ImageSlot("w-full max-w-4xl")
            bm = cfg.get("final", {}).get("block_minutes", 10)
            await slot.fill(lambda: timeline_fig(blocks, "decision", bm))
        show = blocks.copy()
        show.insert(0, "block", pd.to_datetime(show["block_start"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M"))
        with ui.expansion("Blocks", icon="table_rows").classes("w-full"):
            table(show.drop(columns=["block_start", "block_end"]).round(3), rows=15)
    ui.markdown(f"#### Outputs\nIn the project folder `{p.folder}`:")
    for label, path in outputs:
        if Path(path).is_file():
            output_row(label, path, p.folder)
