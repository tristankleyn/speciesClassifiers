"""Advanced diagnostics of delphinID cross-validation: filter held-out frames and events and see how frame and
event accuracy change."""

import json
from urllib.parse import quote

import numpy as np
import pandas as pd
from nicegui import background_tasks, run, ui

from speciesclassifiers.project import Project

from .widgets import ImageSlot, Saver, confusion_table, new_fig, table

_CACHE = {}


def _sources(p, ty):
    f = p.folder
    if ty:
        return f / "frames" / f"frames_{ty}.csv", f / "frames" / f"params_{ty}.json", f / "crossval" / f"cv_{ty}.csv"
    return f / "frames" / "frames.csv", f / "frames" / "params.json", f / "crossval" / "cv_predictions.csv"


def load_table(p, ty):
    from speciesclassifiers.delphinid import AcousticParams
    from speciesclassifiers.delphinid.diagnostics import frame_table
    fr, pa, cv = _sources(p, ty)
    key = (str(cv), cv.stat().st_mtime, fr.stat().st_mtime)
    if key not in _CACHE:
        meta = json.loads(pa.read_text())
        ap = AcousticParams(**{k: v for k, v in meta["acoustic_params"].items()
                               if k in AcousticParams.__dataclass_fields__})
        t = frame_table(pd.read_csv(fr), pd.read_csv(cv, dtype={"event_id": str}), ap, meta.get("sample_rate"))
        _CACHE.clear()
        _CACHE[key] = t
    return _CACHE[key]


def diagnostics_link(p, ty=""):
    q = f"/diagnostics?path={quote(str(p.folder), safe='/:')}" + (f"&type={ty}" if ty else "")
    ui.button("Advanced diagnostics", icon="query_stats", on_click=lambda: ui.navigate.to(q)).props("outline")


@ui.page("/diagnostics")
async def diagnostics_page(path: str, type: str = ""):  # noqa: A002 (query parameter name)
    from speciesclassifiers.delphinid.diagnostics import FRAME_MEASURES, apply_filters, summary

    from .main import header
    from .results import metric
    p = Project(path)
    cfg = p.config()
    header(f"{cfg.get('name', p.folder.name)} · Diagnostics")
    multi = (cfg.get("data") or {}).get("source") == "multi"
    types = [t for t in ("whistle", "click") if (p.folder / "crossval" / f"cv_{t}.csv").is_file()] if multi else []
    ty = type or (types[0] if types else "")
    with ui.row().classes("w-full items-center gap-3 px-4 pt-3"):
        ui.button("Back to the project", icon="arrow_back",
                  on_click=lambda: ui.navigate.to(f"/project?path={quote(str(p.folder), safe='/:')}")).props("flat")
        if len(types) > 1:
            ui.toggle({t: t.capitalize() + "s" for t in types}, value=ty,
                      on_change=lambda e: ui.navigate.to(
                          f"/diagnostics?path={quote(str(p.folder), safe='/:')}&type={e.value}")).props("dense no-caps")
    if not _sources(p, ty)[2].is_file():
        ui.label("No cross-validation results yet: run the project first.").classes("p-4 text-gray-600")
        return
    with ui.row().classes("px-4"):
        sp = ui.spinner(size="lg")
    t = await run.io_bound(load_table, p, ty)
    sp.delete()
    classes = t.attrs["classes"]
    saver = Saver(p)
    skey = f"diagnostics.{ty or 'frames'}"
    saved = saver.get(skey) or {}
    measures = {c: v for c, v in FRAME_MEASURES.items() if c in t and t[c].nunique() > 1}
    state = {"frame": dict(saved.get("frame") or {}), "event": dict(saved.get("event") or {})}

    def kick():
        background_tasks.create(refresh())

    with ui.row().classes("w-full no-wrap items-start gap-6 p-4"):
        with ui.column().classes("w-96 shrink-0 gap-1"):
            ui.label("Frame filters").classes("text-lg font-medium")
            ui.label("Frames outside a range are left out of frame accuracy and of their event's decision.").classes(
                "text-xs text-gray-500")
            controls = {}
            for col, (label, unit) in measures.items():
                v = t[col].astype(float)
                lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
                step = 0.01 if hi - lo <= 2 else (1.0 if hi - lo <= 200 else 50.0)
                lo, hi = float(np.floor(lo / step) * step), float(np.ceil(hi / step) * step)
                lo, hi = round(lo, 6), round(hi, 6)
                if col == "density":          # decimal 0-2; the top end means "2 or more"
                    lo, hi, step = 0.0, 2.0, 0.01
                cur = state["frame"].get(col) or [lo, hi]
                cur = [lo if cur[0] is None else cur[0], hi if cur[1] is None else cur[1]]
                lab = ui.label().classes("text-sm mt-2")

                def text(c=col, la=label, u=unit, lb=lab):
                    a, b = state["frame"].get(c) or [None, None]
                    rng = "all" if a is None and b is None else \
                        f"{'min' if a is None else f'{a:g}'} to {'max' if b is None else f'{b:g}'}"
                    lb.text = f"{la}{f' ({u})' if u else ''}: {rng}"
                r = ui.range(min=lo, max=hi, step=step, value={"min": cur[0], "max": cur[1]}).props(
                    "label").classes("w-full mt-6")
                controls[col] = (r, lo, hi, text)
                text()

                def changed(e, c=col, lo=lo, hi=hi, text=text, r=r):
                    v = e.args if isinstance(e.args, dict) else r.value
                    a, b = float(v["min"]), float(v["max"])
                    state["frame"][c] = [None if a <= lo else a, None if b >= hi else b]
                    if state["frame"][c] == [None, None]:
                        state["frame"].pop(c)
                    text()
                    kick()
                r.on("change", changed)
            ui.label("Event filters").classes("text-lg font-medium mt-4")
            ui.label("Events failing these are undecided: left out of event accuracy.").classes(
                "text-xs text-gray-500")
            ev_state = state["event"]
            ui.number("Min kept frames per event", value=ev_state.get("min_frames") or 1, min=1, step=1,
                      on_change=lambda e: (ev_state.update(min_frames=int(e.value or 1)), kick())).classes("w-64")
            ui.label("Min event score (top x lead over second)").classes("text-sm mt-2")
            ui.slider(min=0, max=1, step=0.01, value=ev_state.get("min_score") or 0).props("label").on(
                "change", lambda e: (ev_state.update(min_score=float(e.args) or None), kick())).classes("w-full")
            ui.label("Min event confidence (top probability)").classes("text-sm mt-2")
            ui.slider(min=round(1 / len(classes), 2), max=1, step=0.01,
                      value=ev_state.get("min_confidence") or round(1 / len(classes), 2)).props("label").on(
                "change", lambda e: (ev_state.update(min_confidence=float(e.args) if float(e.args) > 1 / len(classes)
                                                     else None), kick())).classes("w-full")

            def reset():
                saver.set(skey, {})
                ui.navigate.reload()
            ui.button("Reset filters", icon="restart_alt", on_click=reset).props("flat").classes("mt-4")
        out = ui.column().classes("grow min-w-0 gap-3")

    busy = {"n": 0}

    def draw_hist(col, rng):
        fig, ax = new_fig(5.6, 3.2)
        v = t[col].astype(float)
        bins = np.histogram_bin_edges(v.dropna(), bins=40)
        ax.hist(v[t["correct"]], bins=bins, alpha=0.7, label="correct", color="#2a6fdb")
        ax.hist(v[~t["correct"]], bins=bins, alpha=0.7, label="wrong", color="#e8743b")
        a, b = rng or (None, None)
        if a is not None:
            ax.axvspan(bins[0], a, color="k", alpha=0.08)
        if b is not None:
            ax.axvspan(b, bins[-1], color="k", alpha=0.08)
        lab, unit = FRAME_MEASURES[col]
        ax.set(xlabel=lab + (f" ({unit})" if unit else ""), ylabel="Frames", title="All held-out frames (grey: filtered out)")
        ax.legend(fontsize=8)
        return fig

    async def refresh():
        busy["n"] += 1
        me = busy["n"]
        saver.set(skey, {"frame": state["frame"], "event": state["event"]})
        ff = {c: tuple(v) for c, v in state["frame"].items()}
        kept, ev = await run.io_bound(apply_filters, t, ff, state["event"])
        s = await run.io_bound(summary, t, kept, ev)
        if me != busy["n"]:
            return
        out.clear()
        with out:
            with ui.row().classes("gap-3"):
                metric("Frames kept", s["frames_kept"] / max(s["frames_total"], 1),
                       hint=f"{s['frames_kept']} of {s['frames_total']}")
                metric("Frame accuracy", s["frame_accuracy"], hint="of kept frames")
                metric("Frame balanced", s["frame_balanced"])
                metric("Events decided", s["events_decided"] / max(s["events_total"], 1),
                       hint=f"{s['events_decided']} of {s['events_total']}")
                metric("Event accuracy", s["event_accuracy"], hint="of decided events")
                metric("Event balanced", s["event_balanced"])
            with ui.row().classes("gap-6 items-start w-full"):
                confusion_table(s["frame_confusion"], "Kept frames (rows: true)")
                confusion_table(s["event_confusion"], "Decided events (rows: true)")
            with ui.row().classes("items-center gap-3"):
                ui.label("Distribution of").classes("text-sm")
                sel = ui.select({c: v[0] for c, v in measures.items()}, value=hist_col["c"]).classes("w-72")
            s3 = ImageSlot("w-full max-w-3xl")

            async def redraw_hist(e=None):
                hist_col["c"] = sel.value
                await s3.fill(lambda: draw_hist(sel.value, state["frame"].get(sel.value)))
            sel.on_value_change(redraw_hist)
            evs = ev.assign(kept_share=(ev["n_frames"] / ev["n_frames_all"]).round(2))
            cols = ["event_id", "true_class", "predicted", "decided", "correct", "score", "confidence", "n_frames",
                    "n_frames_all", "kept_share"]
            with ui.expansion("Events", icon="table_rows").classes("w-full"):
                table(evs[cols].sort_values(["decided", "correct", "score"]).round(3), rows=15)
            with ui.row().classes("items-center gap-3 mt-2"):
                ui.switch("Show the frame confusion matrix", value=show_fcm["on"],
                          on_change=lambda e: (show_fcm.update(on=e.value), kick()))
                ui.label("each cell overlays the input spectra of the kept frames of that true / predicted "
                         "class (diagonal = correct, blue)").classes("text-xs text-gray-500")
            s4 = ImageSlot("w-full max-w-5xl") if show_fcm["on"] else None
        await redraw_hist()
        if s4:
            from speciesclassifiers.delphinid.diagnostics import plot_frame_confusion
            await s4.fill(lambda: plot_frame_confusion(kept, classes, t.attrs["features"], t.attrs["freqs"]))

    hist_col = {"c": "confidence" if "confidence" in measures else next(iter(measures))}
    show_fcm = {"on": False}
    await refresh()
