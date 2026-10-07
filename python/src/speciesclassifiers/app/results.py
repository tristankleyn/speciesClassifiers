"""Results and "classify new data" panels of the app, per classifier type."""

from pathlib import Path

import numpy as np
import pandas as pd
from nicegui import ui

from speciesclassifiers.project.workflows import read_json

from nicegui import run

from .widgets import ImageSlot, field, new_fig, open_folder, path_row, table

_CACHE = {}


def _csv(path, **kw):
    """Read a CSV once per file version."""
    path = Path(path)
    if not path.is_file():
        return None
    key = (str(path), path.stat().st_mtime)
    if key not in _CACHE:
        _CACHE[key] = pd.read_csv(path, **kw)
    return _CACHE[key]


def _curve(path):
    """Threshold curve of the cross-validation predictions (cached per file version)."""
    from speciesclassifiers.randomforest import threshold_curve
    key = ("curve", str(path), Path(path).stat().st_mtime)
    if key not in _CACHE:
        _CACHE[key] = threshold_curve(_csv(path, dtype={"event_id": str}))
    return _CACHE[key]


def metric(label, value, fmt="{:.0%}", hint=""):
    with ui.card().classes("p-3 min-w-36"):
        ui.label(label).classes("text-xs text-gray-500")
        txt = "–" if value is None or (isinstance(value, float) and not np.isfinite(value)) else fmt.format(value)
        ui.label(txt).classes("text-2xl font-medium")
        if hint:
            ui.label(hint).classes("text-xs text-gray-500")


def not_ready(p, stage, what):
    state = p.stage_states()[stage]
    if state in ("done", "outdated"):
        if state == "outdated":
            ui.label("Settings changed since this was made: Run again to update.").classes("text-amber-700")
        return False
    ui.label(f"No {what} yet: go to Run.").classes("text-gray-600")
    return True


def output_row(label, path, root=None):
    shown = Path(path).relative_to(root) if root else path
    with ui.row().classes("items-center gap-2 no-wrap"):
        ui.icon("description").classes("text-gray-500")
        ui.label(label).classes("text-sm w-64")
        ui.label(str(shown)).classes("text-xs text-gray-500 font-mono")
        ui.button(icon="folder_open", on_click=lambda: open_folder(Path(path).parent)).props("flat dense")


# ============================================================================ random forest ==========
async def rf_results(p, saver):
    from speciesclassifiers.plots import plot_confusion, plot_importance
    from speciesclassifiers.randomforest import summarise
    if saver.get("data.type") == "binaries":
        from .stacked import stacked_results
        return await stacked_results(p, saver, "A random forest per detection type on each detection's "
                                     "measurements, tested by leaving one event out (event accuracy: the mean "
                                     "probabilities of the event's detections).")
    ui.markdown("#### Cross-validation\nEach group (encounter) was classified by a model trained without it. "
                "Group predictions average the probabilities of its detections; **scores** measure confidence "
                "(top probability × its lead over the second). Raise the thresholds to leave unsure detections "
                "and groups undecided: fewer decisions, more of them right.")
    if not_ready(p, "crossval", "cross-validation results"):
        return
    path = p.folder / "crossval" / "cv_detection_predictions.csv"
    dec = saver.get("decision", {})
    with ui.row().classes("items-center gap-8 w-full"):
        with ui.column().classes("w-80 gap-8"):
            ui.label("Minimum detection score").classes("text-sm")
            s1 = ui.slider(min=0, max=0.5, step=0.01, value=dec.get("min_score", 0.02)).props("label-always")
            ui.label("detections below are left out of their group's average").classes("text-xs text-gray-500")
        with ui.column().classes("w-80 gap-8"):
            ui.label("Minimum group score").classes("text-sm")
            s2 = ui.slider(min=0, max=0.5, step=0.01, value=dec.get("min_group_score", 0.0)).props("label-always")
            ui.label("groups below are reported as 'uncertain'").classes("text-xs text-gray-500")
    box = ui.column().classes("w-full")
    with box:
        ui.spinner(size="lg")
    latest = {"n": 0}

    def draw_conf(conf):
        fig, ax = new_fig(4.6, 4)
        plot_confusion(conf, "Groups (rows: true, columns: predicted)", ax=ax)
        return fig

    def draw_curve(c, mg):
        fig, ax = new_fig(5, 3.4)
        ax.plot(c["threshold"], c["group_accuracy"], label="group accuracy")
        ax.plot(c["threshold"], c["mean_class_accuracy"], label="mean class accuracy")
        ax.plot(c["threshold"], c["groups_kept"], "--", label="groups decided")
        ax.axvline(mg, color="k", lw=0.8)
        ax.set(xlabel="Minimum group score", ylim=(0, 1.05), title="Threshold curve")
        ax.legend(fontsize=8)
        return fig

    async def show():
        latest["n"] += 1
        me = latest["n"]
        ms, mg = float(s1.value), float(s2.value)
        saver.set("decision", {"min_score": ms, "min_group_score": mg})
        preds = await run.io_bound(_csv, path, dtype={"event_id": str})
        s = await run.io_bound(summarise, preds, ms, mg)
        c = await run.io_bound(_curve, path)
        if me != latest["n"]:           # a newer slider move is on its way
            return
        box.clear()
        with box:
            with ui.row().classes("gap-3"):
                metric("Group accuracy", s["group_accuracy"], hint="of groups decided")
                metric("Mean class accuracy", s["mean_class_accuracy"])
                metric("Groups decided", 1 - s["groups_discarded"])
                metric("Groups", len(s["groups"]), "{:d}")
            with ui.row().classes("gap-4 items-start w-full"):
                conf_slot = ImageSlot("w-[26rem]") if len(s["confusion"]) else None
                curve_slot = ImageSlot("w-[28rem]")
            g = s["groups"].assign(correct=lambda x: x["predicted"] == x["true_class"])
            g["decision"] = np.where(g["score"] >= mg, g["predicted"], "uncertain")
            with ui.expansion("Groups", icon="table_rows").classes("w-full"):
                table(g[["event_id", "true_class", "decision", "predicted", "score", "n", "n_total", "correct"]]
                      .sort_values(["correct", "score"]), rows=15)
        if conf_slot:
            await conf_slot.fill(lambda: draw_conf(s["confusion"]))
        await curve_slot.fill(lambda: draw_curve(c, mg))
    s1.on("change", show)
    s2.on("change", show)
    imp = p.folder / "model" / "feature_importance.csv"
    imp_slot = None
    if imp.is_file() and p.stage_states()["model"] == "done":
        with ui.expansion("Feature importance (final model)", icon="bar_chart").classes("w-full"):
            imp_slot = ImageSlot("w-[32rem]")
    ui.label("The thresholds are saved with the project and used when classifying new data.").classes(
        "text-sm text-gray-600")
    await show()
    if imp_slot:
        def draw_imp():
            fig, ax = new_fig(6, 5)
            plot_importance(pd.read_csv(imp, index_col=0)["importance"].sort_values(ascending=False), ax=ax)
            return fig
        await imp_slot.fill(draw_imp)


async def rf_apply(p, saver, start_run):
    if saver.get("data.type") == "binaries":
        from .stacked import stacked_apply
        f = p.folder
        return await stacked_apply(p, saver, start_run, [
            ("Base model: whistles", f / "model" / "randomforest_whistle.joblib"),
            ("Base model: clicks", f / "model" / "randomforest_click.joblib"),
            ("Final forest", f / "final" / "final_model.joblib"),
            ("Held-out base predictions: whistles", f / "crossval" / "cv_whistle.csv"),
            ("Held-out base predictions: clicks", f / "crossval" / "cv_click.csv"),
            ("Block vectors (training)", f / "final" / "blocks.csv"),
            ("New data: block decisions", f / "apply" / "blocks.csv"),
            ("New data: whistle predictions (standard format)", f / "apply" / "detection_predictions_whistle.csv"),
            ("New data: click predictions (standard format)", f / "apply" / "detection_predictions_click.csv")])
    ui.markdown("#### Classify new data\nSame format as the training data (ROCCA contour stats, or a table with "
                "the same feature columns and a group column). Uses the thresholds chosen under Results.")
    path_row(saver, "new_data.path", "New data (file or folder)", "any", (".csv",))
    ui.button("Classify new data", icon="play_arrow", on_click=lambda: start_run(["apply"]))
    st = p.stage_states()
    grp = p.folder / "apply" / "new_group_predictions.csv"
    if st["apply"] in ("done", "outdated") and grp.is_file():
        if st["apply"] == "outdated":
            ui.label("Settings or thresholds changed since: classify again to update.").classes("text-amber-700")
        g = await run.io_bound(_csv, grp, dtype={"event_id": str})
        with ui.row().classes("gap-3"):
            for k, v in g["decision"].value_counts().items():
                metric(str(k), int(v), "{:d}", "groups")
        cols = ["event_id", "decision", "predicted", "score", "n", "n_total"]
        table(g[cols + [c for c in g.columns if c not in cols + ["true_class"]]], rows=15)
    ui.markdown(f"#### Outputs\nIn the project folder `{p.folder}`:")
    for label, path in [("Model", p.folder / "model" / "randomforest_model.joblib"),
                        ("Cross-validation predictions", p.folder / "crossval" / "cv_detection_predictions.csv"),
                        ("New data: group decisions", grp),
                        ("New data: detection predictions (standard format)",
                         p.folder / "apply" / "new_detection_predictions.csv")]:
        if Path(path).is_file():
            output_row(label, path, p.folder)


# ============================================================================ call types ============
async def ct_results(p, saver):
    from speciesclassifiers.plots import plot_confusion
    conf_slot = None
    if not_ready(p, "windows", "windows"):
        return
    ws = read_json(p.folder / "windows" / "summary.json", {})
    by = ws.get("n_windows_by_type") or {"whistle": ws.get("n_windows", 0)}
    ui.markdown(f"#### Windows\n{', '.join(f'{n} {d} windows' for d, n in by.items())}; {ws.get('n_calls', 0)} "
                f"annotations on {len(ws.get('annotated_days', []))} days.")
    lbd = _csv(p.folder / "windows" / "labels_by_day.csv")
    if lbd is not None:
        with ui.expansion("Window labels by day", icon="table_rows").classes("w-full"):
            table(lbd.rename(columns={lbd.columns[0]: "label"}), rows=15)

    ui.markdown("#### How well do the labelled call types separate?\nA random forest on labelled windows, "
                "tested on days it didn't see. Skill: 0 = chance, 1 = perfect.")
    ev = read_json(p.folder / "evaluate" / "evaluate.json")
    if ev is None:
        ui.label("Not run yet.").classes("text-gray-600")
    elif "note" in ev:
        ui.label(ev["note"] + " (Not needed when classifying from labelled periods.)").classes("text-gray-600")
    else:
        with ui.row().classes("gap-3 items-start"):
            metric("Balanced accuracy", ev.get("rf_bal_acc"))
            metric("Skill", ev.get("skill"), "{:.2f}")
            metric("Call types", ev.get("n_classes"), "{:d}")
            metric("Windows", ev.get("n_windows"), "{:d}")
        conf = _csv(p.folder / "evaluate" / "confusion.csv", index_col=0)
        if conf is not None:
            conf_slot = ImageSlot("w-[26rem]")

            def draw_conf():
                fig, ax = new_fig(4.8, 4.2)
                plot_confusion(conf, "Call types (rows: true)", ax=ax)
                return fig

    layered = list((p.folder / "discover").glob("heldout_*.json"))
    if not layered:
        ui.markdown("#### Discovered clusters\nWindows grouped by shape. A cluster's verdict for a label task is "
                    "given only when it is reliably one label, on enough days.")
    if not_ready(p, "discover", "clusters"):
        if conf_slot:
            await conf_slot.fill(draw_conf)
        return
    if layered:
        if conf_slot:
            await conf_slot.fill(draw_conf)
        await layers_results(p, saver)
        ui.markdown("#### Decisions\nNew data is decided per clock-aligned block. Optional **rules** then override "
                    "a block's decision, e.g. mark blocks whose contours barely vary in frequency as noise. Changes "
                    "apply after the next Run (library + new data).")
        decision_editor(saver)
        return
    sc = read_json(p.folder / "discover" / "scorecard.json", {})
    tasks = saver.get("labels.tasks", []) or []
    with ui.row().classes("gap-3"):
        metric("Clusters", sc.get("n_clusters"), "{:d}")
        metric("Windows unassigned", sc.get("frac_unassigned"))
        metric("Call-type agreement (AMI)", sc.get("calltype_ami"), "{:.2f}", "0 = none, 1 = perfect")
        for t in tasks:
            n = t["name"]
            metric(f"{n}: windows decided", sc.get(f"{n}_loo_coverage"), hint="left-out days")
            metric(f"{n}: accuracy when decided", sc.get(f"{n}_loo_bal_acc"), hint="balanced, left-out days")
    cl = _csv(p.folder / "discover" / "clusters.csv")
    if cl is not None:
        cols = ["cluster", "n", "n_days", "top_call_type", "call_types", "candidate_new"] + \
               [f"{t['name']}_verdict" for t in tasks if f"{t['name']}_verdict" in cl]
        table(cl[[c for c in cols if c in cl]], rows=12)
    dw = _csv(p.folder / "discover" / "decision_windows.csv")
    if dw is not None and len(dw):
        with ui.expansion("Combining windows over longer blocks (left-out days)", icon="timeline").classes("w-full"):
            ui.label("Coverage = share of blocks decided; accuracy when decided; skill (0 = chance).").classes(
                "text-xs text-gray-500")
            o = dw[dw["class"] == "overall"]
            t = o.pivot(index="decision_window", columns="task", values=["coverage", "accuracy_when_predicted", "skill"])
            order = ["window"] + [x for x in o["decision_window"].unique() if x != "window"]
            t = t.reindex([x for x in order if x in t.index])
            t.columns = [f"{b} · {a}" for a, b in t.columns]
            table(t.reset_index(), rows=10)
    ui.markdown("#### Decisions\nNew data is decided per clock-aligned block: the majority of windows with a "
                "verdict. Optional **rules** then override a block's decision, e.g. mark blocks whose contours "
                "barely vary in frequency as noise. Changes apply after the next Run (library + new data).")
    decision_editor(saver)
    if conf_slot:
        await conf_slot.fill(draw_conf)


BLOCK_VALUES = {"block_meanfreq_range_hz": "range of contour mean frequencies (Hz)",
                "block_median_abs_slope_hz_s": "median contour wobble (Hz/s)",
                "block_median_freq_hz": "median frequency (Hz)", "block_median_contour_s": "median contour length (s)",
                "n_windows": "windows", "n_assigned": "windows assigned to a cluster", "n_contours": "contours"}


def decision_editor(saver):
    field(saver, {"key": "decision.block_minutes", "label": "Block length", "kind": "select",
                  "options": {1: "1 min", 2: "2 min", 5: "5 min", 10: "10 min", 15: "15 min", 20: "20 min",
                              30: "30 min", 60: "1 hour"}})

    @ui.refreshable
    def rules_ui():
        rules = saver.get("decision.rules", []) or []
        for i, r in enumerate(rules):
            def upd(i=i, **kw):
                rs = saver.get("decision.rules", [])
                rs[i].update(kw)
                saver.set("decision.rules", rs)
            with ui.row().classes("items-center gap-2"):
                ui.label("If")
                ui.select(BLOCK_VALUES, value=r["if"], on_change=lambda e, upd=upd: upd(**{"if": e.value})
                          ).classes("w-72")
                ui.select(["<", "<=", ">", ">=", "==", "!="], value=r["op"],
                          on_change=lambda e, upd=upd: upd(op=e.value)).classes("w-16")
                ui.number(value=r["value"], on_change=lambda e, upd=upd: e.value is not None and upd(
                    value=float(e.value))).classes("w-28")
                ui.label("then")
                ui.input(value=r["then"], on_change=lambda e, upd=upd: upd(then=e.value)).classes("w-32")
                ui.button(icon="delete", on_click=lambda i=i: remove(i)).props("flat dense")

    def remove(i):
        rs = saver.get("decision.rules", [])
        rs.pop(i)
        saver.set("decision.rules", rs)
        rules_ui.refresh()

    def add():
        rs = saver.get("decision.rules", []) or []
        rs.append({"if": "block_meanfreq_range_hz", "op": "<", "value": 200.0, "then": "noise"})
        saver.set("decision.rules", rs)
        rules_ui.refresh()
    ui.label("Rules (checked in order; the first that matches decides)").classes("text-sm mt-2")
    rules_ui()
    ui.button("Add rule", icon="add", on_click=add).props("outline dense")


def _layer_names(p):
    f = p.folder / "discover" / "layers.csv"
    return list(dict.fromkeys(pd.read_csv(f)["layer"])) if f.is_file() else []


def timeline_fig(blocks, task, block_minutes):
    """Days (rows) × time of day: one bar per block, coloured by the task's decision."""
    from matplotlib.patches import Rectangle
    b = blocks.copy()
    t = pd.to_datetime(b["block_start"], unit="s", utc=True)
    b["day"] = t.dt.strftime("%Y-%m-%d")
    b["hour"] = t.dt.hour + t.dt.minute / 60
    days = sorted(b["day"].unique())
    cats = [c for c in b[task].astype(str).unique() if c != "no_detections"]
    fixed = {"uncertain": "#bdbdbd", "noise": "#616161", "n/a": "#e0e0e0"}
    palette = ["#2a6fdb", "#e8743b", "#19a979", "#945ecf", "#ed4a7b", "#13a4b4", "#bf8b00"]
    colours, k = {}, 0
    for c in sorted(cats, key=lambda c: (c in fixed, c)):
        colours[c] = fixed.get(c) or palette[k % len(palette)]
        k += c not in fixed
    fig, ax = new_fig(9, 0.4 * len(days) + 1.4)
    w = block_minutes / 60
    for i, d in enumerate(days):
        g = b[b["day"] == d]
        for c in cats:
            x = g.loc[g[task].astype(str) == c, "hour"]
            if len(x):
                ax.broken_barh([(h, w) for h in x], (i - 0.4, 0.8), color=colours[c])
    ax.set(yticks=range(len(days)), yticklabels=days, xlim=(0, 24), xticks=range(0, 25, 3),
           xlabel="Time of day (UTC)", ylim=(len(days) - 0.5, -0.5))
    ax.grid(axis="x", lw=0.3)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=colours[c]) for c in colours], labels=list(colours),
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False, title=task)
    return fig


async def ct_apply(p, saver, start_run):
    d = saver.project.config()["data"]
    whistles, clicks = d.get("use_whistles", True), d.get("use_clicks", False)
    ui.markdown("#### Classify new data\nDetections to classify with the library, optionally limited to a date "
                "range.")
    if whistles:
        ui.label("Whistle contours: PAMGuard binaries or a contours JSON").classes("text-sm font-medium")
        ui.select({"binaries": "PAMGuard binaries (folder)", "json": "Contours JSON"},
                  value=saver.get("new_data.source"), label="Source",
                  on_change=lambda e: saver.set("new_data.source", e.value)).classes("w-80")
        path_row(saver, "new_data.path", "Folder or file", "any", (".json", ".gz"))
    if clicks:
        ui.label("Clicks: PAMGuard binaries or a clicks CSV").classes("text-sm font-medium mt-2")
        ui.select({"binaries": "PAMGuard binaries (folder)", "csv": "Clicks CSV"},
                  value=saver.get("new_data.click_source", "binaries"), label="Source",
                  on_change=lambda e: saver.set("new_data.click_source", e.value)).classes("w-80")
        path_row(saver, "new_data.click_path", "Folder or file", "any", (".csv", ".gz"))
    with ui.row().classes("gap-4"):
        field(saver, {"key": "new_data.start", "label": "From day (YYYY-MM-DD, optional)", "kind": "text"})
        field(saver, {"key": "new_data.end", "label": "To day (inclusive, optional)", "kind": "text"})
    ui.button("Classify new data", icon="play_arrow", on_click=lambda: start_run(["apply"]))
    st = p.stage_states()
    bl = p.folder / "apply" / "blocks.csv"
    if st["apply"] in ("done", "outdated") and bl.is_file():
        if st["apply"] == "outdated":
            ui.label("Settings changed since: classify again to update.").classes("text-amber-700")
        blocks = _csv(bl)
        if "decision" in blocks:        # pathway: the final decision, then each layer (and each type's own)
            layers = [c for c in _layer_names(p) if c in blocks]
            tasks = ["decision"] + layers + [f"{c}_{t}" for c in layers for t in ("whistle", "click")
                                              if f"{c}_{t}" in blocks]
        else:
            tasks = [t["name"] for t in saver.get("labels.tasks", []) or [] if t["name"] in blocks]
        if tasks and len(blocks):
            ui.markdown("#### Presence over time")
            sel = ui.select(tasks, value=tasks[0], label="Show").classes("w-48")
            holder = ui.column().classes("w-full max-w-4xl")
            bm = read_json(p.folder / "library" / "library.json", {}).get("block_minutes", 10)

            async def draw():
                holder.clear()
                with holder:
                    slot = ImageSlot("w-full")
                task = sel.value
                await slot.fill(lambda: timeline_fig(blocks, task, bm))
            sel.on_value_change(draw)
            with ui.row().classes("gap-3"):
                for t in tasks[:1 + len(_layer_names(p))]:
                    for k, v in blocks[t].value_counts().items():
                        metric(f"{t}: {k}", int(v), "{:d}", "blocks")
        show = blocks.copy()
        show.insert(0, "block", pd.to_datetime(show["block_start"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M"))
        with ui.expansion("Blocks", icon="table_rows").classes("w-full"):
            table(show.drop(columns=["block_start", "block_end"]), rows=15)
        if tasks and len(blocks):
            outputs(p, bl)
            await draw()
            return
    outputs(p, bl)


def outputs(p, bl):
    clicks = (p.folder / "apply" / "windows_click.csv").is_file()
    ui.markdown("#### Outputs\nThe library holds everything needed to classify new detections the same way. "
                "`standalone.py` (next to it) runs it anywhere with numpy only: "
                "`python standalone.py library.json contours.json blocks.csv`" +
                (" (with clicks: add `--clicks clicks.csv`; use `-` for no contours)" if clicks else "") +
                f". In the project folder `{p.folder}`:")
    for label, path in [("Library", p.folder / "library" / "library.json"),
                        ("Standalone classifier", p.folder / "library" / "standalone.py"),
                        ("Decision blocks", bl),
                        ("Windows" + (" (whistles)" if clicks else ""), p.folder / "apply" / "windows.csv"),
                        ("Windows (clicks)", p.folder / "apply" / "windows_click.csv"),
                        ("Window predictions (standard format)", p.folder / "apply" / "window_predictions_standard.csv"),
                        ("Clusters", p.folder / "discover" / "clusters.csv")]:
        if Path(path).is_file():
            output_row(label, path, p.folder)


# ============================================================================ delphinID =============
def _dl_summary(path):
    from speciesclassifiers.delphinid import summarise
    key = ("dl", str(path), Path(path).stat().st_mtime)
    if key not in _CACHE:
        _CACHE[key] = summarise(_csv(path, dtype={"event_id": str, "test_group": str}))
    return _CACHE[key]


def _event_curve(ev, classes):
    """Event accuracy and share of events decided as the minimum event score rises."""
    from speciesclassifiers.randomforest.model import decision_score
    score = decision_score(ev[classes].to_numpy())
    rows = []
    for t in np.linspace(0, 0.5, 26):
        keep = score >= t
        rows.append({"threshold": t, "kept": keep.mean(),
                     "accuracy": ev.loc[keep, "correct"].mean() if keep.any() else np.nan})
    return pd.DataFrame(rows), score


async def dl_results(p, saver):
    from speciesclassifiers.plots import plot_confusion
    ui.markdown("#### Cross-validation\nEach fold's frames were classified by a model trained without their "
                "groups. **Frame accuracy**: single frames (a few seconds each). **Event accuracy**: whole events, "
                "from the summed probabilities of their frames.")
    st = p.status().get("stages", {})
    cfg = p.config()
    if not cfg["settings"]["crossval"].get("run", True) and p.stage_states()["crossval"] != "done":
        ui.label("Cross-validation is switched off (Settings), so there are no test results.").classes(
            "text-gray-600")
    elif not not_ready(p, "crossval", "cross-validation results"):
        if st.get("crossval", {}).get("source"):
            ui.label(f"Trained in {st['crossval']['source']}.").classes("text-xs text-gray-500")
        path = p.folder / "crossval" / "cv_predictions.csv"
        s = await run.io_bound(_dl_summary, path)
        ev = s["events"]
        classes = sorted(s["event_confusion"].columns.union(s["event_confusion"].index))
        classes = [c for c in classes if c in ev.columns]
        curve, score = await run.io_bound(_event_curve, ev, classes)
        with ui.row().classes("gap-3"):
            metric("Frame accuracy", s["frame_accuracy"])
            metric("Event accuracy", s["event_accuracy"])
            metric("Events", len(ev), "{:d}")
            metric("Classes", len(classes), "{:d}")
        with ui.row().classes("gap-4 items-start w-full"):
            slots = [ImageSlot("w-[24rem]"), ImageSlot("w-[24rem]"), ImageSlot("w-[26rem]")]
        acc = s["event_accuracy_by_class"].rename("event accuracy").reset_index()
        with ui.expansion("Events", icon="table_rows").classes("w-full"):
            e = ev.assign(score=score)[["event_id", "true_class", "predicted", "correct", "score", "n_frames"]]
            table(acc, rows=10)
            table(e.sort_values(["correct", "score"]), rows=15)
        hist = _csv(p.folder / "crossval" / "cv_history.csv")
        hist_slot = None
        if hist is not None and len(hist):
            with ui.expansion("Training curves", icon="show_chart").classes("w-full"):
                ui.label("Validation loss well above training loss, or rising, suggests overfitting: try more "
                         "dropout, fewer epochs or fewer dense units.").classes("text-xs text-gray-500")
                hist_slot = ImageSlot("w-full max-w-4xl")

        def conf(c, title):
            fig, ax = new_fig(4.2, 3.8)
            plot_confusion(c, title, ax=ax)
            return fig

        def draw_curve():
            fig, ax = new_fig(4.8, 3.4)
            ax.plot(curve["threshold"], curve["accuracy"], label="event accuracy")
            ax.plot(curve["threshold"], curve["kept"], "--", label="events decided")
            ax.set(xlabel="Minimum event score", ylim=(0, 1.05), title="Event threshold curve")
            ax.legend(fontsize=8)
            return fig

        def draw_hist():
            boots = sorted(hist["bootstrap"].unique())
            from matplotlib.figure import Figure
            fig = Figure(figsize=(2.6 * len(boots), 2.8))
            axes = np.atleast_1d(fig.subplots(1, len(boots), sharey=True))
            for a, b in zip(axes, boots):
                h = hist[hist["bootstrap"] == b].groupby("epoch")[["loss", "val_loss"]].mean()
                a.plot(h.index, h["loss"], label="training")
                a.plot(h.index, h["val_loss"], label="validation")
                a.set(title=f"Bootstrap {b}", xlabel="Epoch")
            axes[0].set_ylabel("Loss (mean over folds)")
            axes[0].legend(fontsize=8)
            return fig
        ui.label("Event score: top probability × its lead over the second. In PAMGuard, events can be judged "
                 "the same way from the model's frame predictions.").classes("text-xs text-gray-500")
        await slots[0].fill(lambda: conf(s["frame_confusion"], "Frames"))
        await slots[1].fill(lambda: conf(s["event_confusion"], "Events"))
        await slots[2].fill(draw_curve)
        if hist_slot:
            await hist_slot.fill(draw_hist)
    fc = _csv(p.folder / "frames" / "class_counts.csv")
    if fc is not None:
        with ui.expansion("Training frames per class", icon="table_rows").classes("w-full"):
            table(fc, rows=15)


async def dl_export(p, saver, start_run):
    ui.markdown("#### Export to PAMGuard\nThe final model is trained on all frames and saved as a zip that "
                "PAMGuard's **Deep Learning Classifier** loads directly.")
    if not_ready(p, "model", "model"):
        return
    zips = sorted((p.folder / "model").glob("*.zip"))
    if not zips:
        ui.label("No model zip found: run again.").classes("text-amber-700")
        return
    z = zips[0]
    output_row("PAMGuard model", z, p.folder)
    info = await run.io_bound(_zip_info, z)
    ui.markdown("""#### Using the model in PAMGuard
1. Add a **Deep Learning Classifier** module. Set its source to the **Whistle and Moan Detector** (whistle model) or the **Click Detector** (click model).
2. Choose **delphinID** as the model type and select the zip.
3. Set the segment length, hop and minimum detection value to those below.

A click model only suits recordings at the sample rate it was trained on, because that sets which FFT bins become its inputs.""")
    if info:
        rows = pd.DataFrame([{"setting": k, "value": str(v)} for k, v in info.get("pamguard_settings", {}).items()])
        if len(rows):
            table(rows, rows=12)
        cv = info.get("cross_validation")
        if cv:
            ui.label(f"Cross-validated accuracy stored in the zip: frames {cv.get('frame_accuracy', 0):.0%}, "
                     f"events {cv.get('event_accuracy', 0):.0%}.").classes("text-sm text-gray-600")
    for label, path in [("Training frames", p.folder / "frames" / "frames.csv"),
                        ("Cross-validation predictions (standard format)", p.folder / "crossval" / "cv_predictions.csv"),
                        ("Final model training history", p.folder / "model" / "history.csv")]:
        if Path(path).is_file():
            output_row(label, path, p.folder)


def _zip_info(path):
    import json
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith("speciesclassifiers_info.json"))
            return json.loads(z.read(name))
    except Exception:
        return None


# ============================================================================ call-type layers ======
def _layer_data(p, name, block_minutes, min_score, variant=None):
    """Held-out rows and performance of one layer (cached per file version and threshold). ``variant``: the
    held-out file stem relative to the project (default ``discover/heldout_<name>``)."""
    import json

    from speciesclassifiers.calltypes import multiclass as mc
    stem = variant or f"discover/heldout_{name}"
    f = p.folder / f"{stem}.csv"
    meta = json.loads((p.folder / f"{stem}.json").read_text())
    key = ("layer", str(f), f.stat().st_mtime, block_minutes)
    if key not in _CACHE:
        wp = pd.read_csv(f, dtype={"unit": str, "true": str, "day": str})
        cl = meta["classes"]
        blocks = mc.aggregate(wp, cl, "block", block_minutes)
        units = mc.aggregate(wp, cl, "unit")
        _CACHE[key] = (meta, blocks, units, mc.threshold_curve(blocks, cl, np.linspace(0, 0.5, 51)),
                       mc.threshold_curve(units, cl, np.linspace(0, 0.5, 51)))
    meta, blocks, units, bcurve, ucurve = _CACHE[key]
    cl = meta["classes"]
    return meta, mc.performance(blocks, cl, min_score), mc.performance(units, cl, min_score), bcurve, ucurve


async def layer_section(p, saver, name, block_minutes, variants=None, title=None, note=None):
    """One layer's held-out results with a threshold slider (saved as ``decision.min_scores[name]``).
    ``variants``: {held-out file stem: label} to switch between (default: the call-type layer's files)."""
    import json
    given = variants is not None
    stem0 = next(iter(variants)) if given else f"discover/heldout_{name}"
    meta = json.loads((p.folder / f"{stem0}.json").read_text())
    within = " > ".join(meta.get("path") or [])
    unit = "event" if meta.get("unit") == "event_id" else meta.get("unit", "day")
    types = [] if given else (meta.get("types") or [])
    forest = not given and saver.get("decision.method") == "forest" and \
        (p.folder / "forest" / f"heldout_{name}.json").is_file()
    if not given:
        variants = {}
        if forest:
            variants[f"forest/heldout_{name}"] = "Forest (final)"
        variants[f"discover/heldout_{name}"] = "Multiplied" if forest else ("Combined" if meta.get("merged") else "")
        for t in types if meta.get("merged") else []:
            if (p.folder / "discover" / f"heldout_{name}__{t}.json").is_file():
                variants[f"discover/heldout_{name}__{t}"] = {"whistle": "Whistles", "click": "Clicks"}.get(t, t)
    first = next(iter(variants))
    view = {"v": first}
    with ui.card().classes("w-full p-4").props("flat bordered"):
        ui.label(title or (f"Layer: {name}" + (f"  ·  within {within}" if within else ""))).classes(
            "text-lg font-medium")
        ui.label(f"Classes: {', '.join(meta['classes'])}. Tested by leaving out one {unit} at a time; blocks are "
                 f"{block_minutes} min.").classes("text-sm text-gray-600")
        if len(variants) > 1 and note:
            ui.label(note).classes("text-xs text-gray-500")
        elif len(variants) > 1:
            ui.label("Each detection type is clustered on its own. " + (
                "'Forest (final)': a random forest decides each block from the share of its windows in each "
                "cluster, tested strictly (clusters redone without each held-out event); this is what new data "
                "gets. 'Multiplied' multiplies each type's evidence instead." if forest else
                "'Combined' multiplies their evidence per block (the decision new data gets). A class one type "
                "hasn't learned counts as neutral for it.")).classes("text-xs text-gray-500")
        if len(variants) > 1:
            async def pick(e):
                view["v"] = e.value
                await show()
            ui.toggle(variants, value=first, on_change=pick).props("dense no-caps")
        cur = float((saver.get("decision.min_scores") or {}).get(name, 0.0))
        with ui.row().classes("items-center gap-4 w-full"):
            ui.label("Minimum block score").classes("text-sm")
            sl = ui.slider(min=0, max=0.5, step=0.01, value=cur).props("label-always").classes("w-80 mt-6")
            ui.label("blocks below are 'uncertain' (score = top probability × its lead over the second)").classes(
                "text-xs text-gray-500")
        box = ui.column().classes("w-full")

        def draw_curve(curve, t, title):
            fig, ax = new_fig(4.6, 3.2)
            ax.plot(curve["threshold"], curve["accuracy"], label="accuracy when decided")
            ax.plot(curve["threshold"], curve["balanced_accuracy"], label="balanced accuracy")
            ax.plot(curve["threshold"], curve["coverage"], "--", label="decided")
            ax.axvline(t, color="k", lw=0.8)
            ax.set(xlabel="Minimum score", ylim=(0, 1.05), title=title)
            ax.legend(fontsize=7)
            return fig

        def draw_conf(conf, title):
            from speciesclassifiers.plots import plot_confusion
            n = max(len(conf.index), len(conf.columns))
            fig, ax = new_fig(1.2 + 0.75 * n, 1.0 + 0.65 * n)
            plot_confusion(conf, title, ax=ax)
            return fig

        async def show():
            t = float(sl.value)
            ms = dict(saver.get("decision.min_scores") or {})
            if float(ms.get(name, 0.0)) != t:
                ms[name] = t
                saver.set("decision.min_scores", ms)
            v = view["v"]
            m, bp, up, bc, uc = await run.io_bound(_layer_data, p, name, block_minutes, t, v)
            box.clear()
            with box:
                with ui.row().classes("gap-3"):
                    metric("Blocks decided", bp["coverage"])
                    metric("Block accuracy", bp["accuracy"], hint="of blocks decided")
                    metric("Balanced (blocks)", bp["balanced_accuracy"])
                    metric(f"{unit.capitalize()}s decided", up["coverage"])
                    metric(f"{unit.capitalize()} accuracy", up["accuracy"], hint=f"of {unit}s decided")
                rec = pd.DataFrame({"class": list(bp["recall"]),
                                    "blocks: share correct": [bp["recall"][c] for c in bp["recall"]],
                                    f"{unit}s: share correct": [up["recall"].get(c) for c in bp["recall"]]})
                ui.label("Accuracy, confusion matrix and share correct count decided blocks only; blocks below "
                         "the minimum score are left out (see 'Blocks decided').").classes("text-xs text-gray-500")
                with ui.row().classes("gap-4 items-start w-full"):
                    s1, s2 = ImageSlot("w-[24rem]"), ImageSlot("w-[22rem]")
                    with ui.column().classes("min-w-64"):
                        table(rec, rows=12)
            what = variants.get(v, "") + ": " if len(variants) > 1 else ""
            await s1.fill(lambda: draw_curve(bc, t, f"{what}blocks, threshold curve"))
            await s2.fill(lambda: draw_conf(bp["confusion"], f"{what}blocks (rows: true)"))
        sl.on("change", show)
        await show()
        if given:
            return box
        files = [({"whistle": "Whistles", "click": "Clicks"}.get(t, t), p.folder / "discover" / f"clusters_{name}__{t}.csv") for t in types] or \
            [("", p.folder / "discover" / f"clusters_{name}.csv")]
        if not files[0][1].is_file():
            files = [("", p.folder / "discover" / "clusters.csv")]
        imp = _csv(p.folder / "forest" / f"importance_{name}.csv") if forest else None
        if imp is not None:
            with ui.expansion("Forest: which clusters matter", icon="park").classes("w-full"):
                ui.label("Importance of each input (share of a block's windows in a cluster of a detection type) "
                         "to the final forest; see the cluster tables below for what each cluster is.").classes(
                    "text-xs text-gray-500")
                table(imp.round(4), rows=15)
        for what, cf in files:
            cl = _csv(cf)
            if cl is None:
                continue
            with ui.expansion(f"Clusters of this layer{' · ' + what if what else ''}", icon="hub").classes("w-full"):
                pc = [f"{name}_p_{c}" for c in m_classes(cl, name)]
                cols = ["cluster", "n", "n_days", f"{name}_verdict"] + pc + \
                       [c for c in cl.columns if c.startswith("median_")]
                t = cl[[c for c in cols if c in cl]].rename(columns={f"{name}_verdict": "verdict",
                                                                      **{c: c.replace(f"{name}_p_", "p ") for c in pc}})
                table(t, rows=15)


def m_classes(cl, name):
    pre = f"{name}_p_"
    return [c[len(pre):] for c in cl.columns if c.startswith(pre)]


async def layers_results(p, saver):
    import json
    lib_bm = saver.get("decision.block_minutes", 10)
    layers_csv = p.folder / "discover" / "layers.csv"
    names = list(dict.fromkeys(pd.read_csv(layers_csv)["layer"])) if layers_csv.is_file() else \
        [f.stem[len("heldout_"):] for f in sorted((p.folder / "discover").glob("heldout_*.json")) if "__" not in f.stem]
    ui.markdown("#### Classification layers\nEach layer's clusters are judged on held-out data: windows from the "
                "left-out event get the class probabilities of their clusters (computed without that event); a "
                "block's probabilities are the average of its windows'. Raise a layer's threshold to leave unsure "
                "blocks undecided. Thresholds apply to new data after the next Run (library + new data).")
    for name in names:
        if (p.folder / "discover" / f"heldout_{name}.json").is_file():
            await layer_section(p, saver, name, json.loads(
                (p.folder / "discover" / f"heldout_{name}.json").read_text()).get("block_minutes", lib_bm))
