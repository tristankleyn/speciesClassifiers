"""Results and "classify new data" panels of the app, per classifier type."""

from pathlib import Path

import numpy as np
import pandas as pd
from nicegui import ui

from speciesclassifiers.project.workflows import read_json

from .widgets import field, fig_src, open_folder, path_row, table

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


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


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
def rf_results(p, saver):
    from speciesclassifiers.plots import plot_confusion, plot_importance
    from speciesclassifiers.randomforest import summarise, threshold_curve
    ui.markdown("#### Cross-validation\nEach group (encounter) was classified by a model trained without it. "
                "Group predictions average the probabilities of its detections; **scores** measure confidence "
                "(top probability × its lead over the second). Raise the thresholds to leave unsure detections "
                "and groups undecided: fewer decisions, more of them right.")
    if not_ready(p, "crossval", "cross-validation results"):
        return
    preds = _csv(p.folder / "crossval" / "cv_detection_predictions.csv", dtype={"event_id": str})
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
    curve = {}

    def show():
        ms, mg = float(s1.value), float(s2.value)
        saver.set("decision", {"min_score": ms, "min_group_score": mg})
        s = summarise(preds, ms, mg)
        if "c" not in curve:
            curve["c"] = threshold_curve(preds)
        box.clear()
        with box:
            with ui.row().classes("gap-3"):
                metric("Group accuracy", s["group_accuracy"], hint="of groups decided")
                metric("Mean class accuracy", s["mean_class_accuracy"])
                metric("Groups decided", 1 - s["groups_discarded"])
                metric("Groups", len(s["groups"]), "{:d}")
            plt = _plt()
            with ui.row().classes("gap-4 items-start w-full"):
                if len(s["confusion"]):
                    fig, ax = plt.subplots(figsize=(4.6, 4))
                    plot_confusion(s["confusion"], "Groups (rows: true, columns: predicted)", ax=ax)
                    ui.image(fig_src(fig)).classes("w-[26rem]")
                c = curve["c"]
                fig, ax = plt.subplots(figsize=(5, 3.4))
                ax.plot(c["threshold"], c["group_accuracy"], label="group accuracy")
                ax.plot(c["threshold"], c["mean_class_accuracy"], label="mean class accuracy")
                ax.plot(c["threshold"], c["groups_kept"], "--", label="groups decided")
                ax.axvline(mg, color="k", lw=0.8)
                ax.set(xlabel="Minimum group score", ylim=(0, 1.05), title="Threshold curve")
                ax.legend(fontsize=8)
                ui.image(fig_src(fig)).classes("w-[28rem]")
            g = s["groups"].assign(correct=lambda x: x["predicted"] == x["true_class"])
            g["decision"] = np.where(g["score"] >= mg, g["predicted"], "uncertain")
            with ui.expansion("Groups", icon="table_rows").classes("w-full"):
                table(g[["event_id", "true_class", "decision", "predicted", "score", "n", "n_total", "correct"]]
                      .sort_values(["correct", "score"]), rows=15)
    s1.on("change", lambda: show())
    s2.on("change", lambda: show())
    show()
    imp = p.folder / "model" / "feature_importance.csv"
    if imp.is_file() and p.stage_states()["model"] == "done":
        with ui.expansion("Feature importance (final model)", icon="bar_chart").classes("w-full"):
            fig, ax = _plt().subplots(figsize=(6, 5))
            plot_importance(pd.read_csv(imp, index_col=0)["importance"].sort_values(ascending=False), ax=ax)
            ui.image(fig_src(fig)).classes("w-[32rem]")
    ui.label("The thresholds are saved with the project and used when classifying new data.").classes(
        "text-sm text-gray-600")


def rf_apply(p, saver, start_run):
    ui.markdown("#### Classify new data\nSame format as the training data (ROCCA contour stats, or a table with "
                "the same feature columns and a group column). Uses the thresholds chosen under Results.")
    path_row(saver, "new_data.path", "New data (file or folder)", "any", (".csv",))
    ui.button("Classify new data", icon="play_arrow", on_click=lambda: start_run(["apply"]))
    st = p.stage_states()
    grp = p.folder / "apply" / "new_group_predictions.csv"
    if st["apply"] in ("done", "outdated") and grp.is_file():
        if st["apply"] == "outdated":
            ui.label("Settings or thresholds changed since: classify again to update.").classes("text-amber-700")
        g = _csv(grp, dtype={"event_id": str})
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
def ct_results(p, saver):
    from speciesclassifiers.plots import plot_confusion
    plt = _plt()
    if not_ready(p, "windows", "windows"):
        return
    ws = read_json(p.folder / "windows" / "summary.json", {})
    ui.markdown(f"#### Windows\n{ws.get('n_windows', 0)} windows; {ws.get('n_calls', 0)} annotated calls on "
                f"{len(ws.get('annotated_days', []))} days.")
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
        ui.label(ev["note"]).classes("text-amber-700")
    else:
        with ui.row().classes("gap-3 items-start"):
            metric("Balanced accuracy", ev.get("rf_bal_acc"))
            metric("Skill", ev.get("skill"), "{:.2f}")
            metric("Call types", ev.get("n_classes"), "{:d}")
            metric("Windows", ev.get("n_windows"), "{:d}")
        conf = _csv(p.folder / "evaluate" / "confusion.csv", index_col=0)
        if conf is not None:
            fig, ax = plt.subplots(figsize=(4.8, 4.2))
            plot_confusion(conf, "Call types (rows: true)", ax=ax)
            ui.image(fig_src(fig)).classes("w-[26rem]")

    ui.markdown("#### Discovered clusters\nWindows grouped by shape. A cluster's verdict for a label task is "
                "given only when it is reliably one label, on enough days.")
    if not_ready(p, "discover", "clusters"):
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


def timeline_fig(blocks, task, block_minutes):
    """Days (rows) × time of day: one bar per block, coloured by the task's decision."""
    plt = _plt()
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
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(days) + 1.4))
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
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colours[c]) for c in colours], labels=list(colours),
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False, title=task)
    return fig


def ct_apply(p, saver, start_run):
    ui.markdown("#### Classify new data\nWhistle contours to classify with the library: PAMGuard binaries or "
                "a contours JSON, optionally limited to a date range.")
    ui.select({"binaries": "PAMGuard binaries (folder)", "json": "Contours JSON"}, value=saver.get("new_data.source"),
              label="Source", on_change=lambda e: saver.set("new_data.source", e.value)).classes("w-80")
    path_row(saver, "new_data.path", "Folder or file", "any", (".json", ".gz"))
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
        tasks = [t["name"] for t in saver.get("labels.tasks", []) or [] if t["name"] in blocks]
        if tasks and len(blocks):
            ui.markdown("#### Presence over time")
            sel = ui.select(tasks, value=tasks[0], label="Show").classes("w-48")
            img = ui.image().classes("w-full max-w-4xl")
            bm = read_json(p.folder / "library" / "library.json", {}).get("block_minutes", 10)

            def draw():
                img.set_source(fig_src(timeline_fig(blocks, sel.value, bm)))
            sel.on_value_change(draw)
            draw()
            with ui.row().classes("gap-3"):
                for t in tasks:
                    for k, v in blocks[t].value_counts().items():
                        metric(f"{t}: {k}", int(v), "{:d}", "blocks")
        show = blocks.copy()
        show.insert(0, "block", pd.to_datetime(show["block_start"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M"))
        with ui.expansion("Blocks", icon="table_rows").classes("w-full"):
            table(show.drop(columns=["block_start", "block_end"]), rows=15)
    ui.markdown("#### Outputs\nThe library holds everything needed to classify new contours the same way. "
                "`standalone.py` (next to it) runs it anywhere with numpy only: "
                "`python standalone.py library.json contours.json blocks.csv`. In the project folder "
                f"`{p.folder}`:")
    for label, path in [("Library", p.folder / "library" / "library.json"),
                        ("Standalone classifier", p.folder / "library" / "standalone.py"),
                        ("Decision blocks", bl), ("Windows", p.folder / "apply" / "windows.csv"),
                        ("Window predictions (standard format)", p.folder / "apply" / "window_predictions_standard.csv"),
                        ("Clusters", p.folder / "discover" / "clusters.csv")]:
        if Path(path).is_file():
            output_row(label, path, p.folder)
