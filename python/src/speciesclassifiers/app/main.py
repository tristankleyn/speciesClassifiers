"""The speciesClassifiers app: set up, run and review classifier projects in your browser.

Start it with ``speciesclassifiers app`` (or ``python -m speciesclassifiers app``, or the start_app
launcher). Everything runs on this computer; projects are ordinary folders (see ``project.Project``).
"""

import json
import socket
import time
from pathlib import Path

from nicegui import app, run, ui

from speciesclassifiers.project import WORKFLOWS, Project, start_background
from speciesclassifiers.project import checks
from speciesclassifiers.project.workflows import EXAMPLES

from . import results
from .settings import SETTINGS
from .widgets import Saver, field, open_folder, path_row, pick_path, show_check

HOME = Path.home() / ".speciesclassifiers"
RECENT = HOME / "recent.json"
DEFAULT_PARENT = Path.home() / "speciesClassifiers projects"

STATE_ICON = {"done": ("check_circle", "text-green-600"), "outdated": ("update", "text-amber-600"),
              "failed": ("error", "text-red-600"), "pending": ("radio_button_unchecked", "text-gray-400"),
              "running": ("autorenew", "text-blue-600"), "skipped": ("remove_circle_outline", "text-gray-300")}
STATE_TEXT = {"done": "done", "outdated": "settings changed: will rerun", "failed": "failed",
              "pending": "not run yet", "running": "running", "skipped": "skipped (not set up)"}

GUIDE = [
    ("Measurements of individual detections (e.g. PAMGuard ROCCA contour stats, or any table of detection "
     "features) from encounters of known species", "randomforest"),
    ("Whistle contours (PAMGuard Whistle and Moan Detector) with some calls annotated by type, and labels such as "
     "species or population to tell apart", "calltypes"),
    ("Labelled encounters, and I want a delphinID deep-learning model to run in PAMGuard", "delphinid"),
    ("Outputs of classifiers I already have (e.g. delphinID in a PAMGuard database), to retrain for new labels",
     "transferlearning"),
]
COMING = {"delphinid": "delphinID isn't in the app yet: use notebooks 01_make_frames and 02_train_delphinID "
                       "(02 runs in Google Colab, so no TensorFlow install is needed).",
          "transferlearning": "Transfer learning isn't in the app yet: use notebook 04_transferlearning "
                              "(Python or R)."}


# ------------------------------------------------------------------------ recent projects ---------
def recent():
    try:
        paths = json.loads(RECENT.read_text())
    except (OSError, ValueError):
        return []
    return [p for p in paths if Project(p).exists()]


def remember(folder):
    HOME.mkdir(parents=True, exist_ok=True)
    paths = [str(folder)] + [p for p in recent() if p != str(folder)]
    RECENT.write_text(json.dumps(paths[:15]))


def open_project(folder):
    remember(Project(folder).folder)
    ui.navigate.to(f"/project?path={Project(folder).folder}")


# ------------------------------------------------------------------------ home ---------------------
def header(title="speciesClassifiers"):
    with ui.header().classes("bg-slate-800 items-center"):
        ui.link("🐬 speciesClassifiers", "/").classes("text-white text-lg no-underline font-medium")
        ui.label(title if title != "speciesClassifiers" else "").classes("text-slate-300")


@ui.page("/")
def home():
    header()
    with ui.column().classes("w-full max-w-4xl mx-auto p-4 gap-6"):
        ui.markdown("## Build a classifier from your acoustic detections\n"
                    "Answer a few questions, point to your data, and run. Everything stays on this computer; "
                    "each project is a folder you can reopen at any time.")
        with ui.row().classes("gap-4"):
            ui.button("New project", icon="add", on_click=new_project_dialog).props("size=lg")
            ui.button("Open project", icon="folder_open", on_click=open_project_dialog).props("outline size=lg")
            ui.button("Try an example", icon="science", on_click=example_dialog).props("outline size=lg")
        rec = recent()
        if rec:
            ui.label("Recent projects").classes("text-lg font-medium")
            with ui.column().classes("gap-1 w-full"):
                for path in rec:
                    p = Project(path)
                    cfg = p.config()
                    with ui.card().classes("w-full cursor-pointer hover:bg-slate-50").on(
                            "click", lambda path=path: open_project(path)):
                        with ui.row().classes("items-center justify-between w-full"):
                            with ui.column().classes("gap-0"):
                                ui.label(cfg.get("name", Path(path).name)).classes("font-medium")
                                ui.label(path).classes("text-xs text-gray-500")
                            ui.badge(WORKFLOWS[cfg["classifier"]].title).props("outline")
        with ui.expansion("How do the classifiers work?", icon="help_outline").classes("w-full"):
            ui.markdown("All of them turn small units (detections, frames, windows) into features, classify them, "
                        "combine the results over encounters or time blocks, and only decide when confident enough. "
                        "See `docs/how_it_works.md` in the repository for a one-page overview.")


async def new_project_dialog(classifier=None):
    with ui.dialog() as dialog, ui.card().classes("w-[48rem] max-w-full"):
        ui.label("New project").classes("text-xl font-medium")
        ui.label("What do you have?").classes("font-medium")
        choice = ui.radio({c: text for text, c in GUIDE}, value=classifier).props("dense")
        note = ui.label().classes("text-sm")
        name = ui.input("Project name", value="my classifier").classes("w-96")
        with ui.row().classes("items-center w-full no-wrap"):
            parent = ui.input("Create it in folder", value=str(DEFAULT_PARENT)).classes("grow")

            async def browse():
                pth = await pick_path(parent.value, "dir", title="Folder to create the project in")
                if pth:
                    parent.value = pth
            ui.button(icon="folder_open", on_click=browse).props("flat")
        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            create = ui.button("Create", icon="add")

        def update():
            c = choice.value
            if c in COMING:
                note.text = COMING[c]
                note.classes(replace="text-sm text-amber-700")
                create.disable()
            elif c:
                note.text = f"→ {WORKFLOWS[c].title}: {WORKFLOWS[c].description}"
                note.classes(replace="text-sm text-slate-700")
                create.enable()
            else:
                create.disable()
        choice.on_value_change(update)
        update()

        def do_create():
            folder = Path(parent.value).expanduser() / name.value.strip()
            try:
                p = Project.create(folder, choice.value, name=name.value.strip())
            except Exception as e:
                ui.notify(str(e), type="negative")
                return
            dialog.close()
            open_project(p.folder)
        create.on_click(do_create)
    dialog.open()


async def open_project_dialog():
    pth = await pick_path(str(DEFAULT_PARENT if DEFAULT_PARENT.exists() else Path.home()), "dir",
                          title="Open the project folder (the one with project.json)")
    if pth:
        if Project(pth).exists():
            open_project(pth)
        else:
            ui.notify("No project.json in that folder", type="warning")


async def example_dialog():
    if not EXAMPLES.is_dir():
        ui.notify("Example data not found (it comes with the repository's examples/ folder)", type="warning")
        return
    with ui.dialog() as dialog, ui.card().classes("w-[36rem]"):
        ui.label("Try an example (synthetic data)").classes("text-lg font-medium")
        for c in ("randomforest", "calltypes"):
            wf = WORKFLOWS[c]
            with ui.card().classes("w-full cursor-pointer hover:bg-slate-50").on(
                    "click", lambda c=c: dialog.submit(c)):
                ui.label(wf.title).classes("font-medium")
                ui.label(wf.description).classes("text-sm text-gray-600")
    c = await dialog
    if not c:
        return
    k = 1
    while (DEFAULT_PARENT / f"example {WORKFLOWS[c].title.lower()} {k}").exists():
        k += 1
    folder = DEFAULT_PARENT / f"example {WORKFLOWS[c].title.lower()} {k}"
    Project.create(folder, c, name=f"Example: {WORKFLOWS[c].title}", config=WORKFLOWS[c].example())
    open_project(folder)


# ------------------------------------------------------------------------ project page ------------
@ui.page("/project")
def project_page(path: str):
    p = Project(path)
    if not p.exists():
        header()
        ui.label(f"No project in {path}").classes("p-4 text-red-700")
        return
    cfg = p.config()
    wf = p.workflow
    kind = cfg["classifier"]
    header(f"{cfg.get('name', p.folder.name)} · {wf.title}")
    saved = {"t": 0}

    def on_saved(_key):
        saved["t"] = time.time()
        refresh_status()

    saver = Saver(p, on_saved)
    steps = (["data", "labels"] if kind == "calltypes" else ["data"]) + ["settings", "run", "results", "apply"]
    names = {"data": "Data", "labels": "Labels", "settings": "Settings", "run": "Run", "results": "Results",
             "apply": "Classify new data · export"}
    titles = {s: f"{k} · {names[s]}" for k, s in enumerate(steps, 1)}

    with ui.row().classes("w-full no-wrap items-start p-2 gap-4"):
        with ui.column().classes("w-64 shrink-0 gap-2"):
            tabs = ui.tabs().props("vertical align=left").classes("w-full")
            with tabs:
                for s in steps:
                    ui.tab(s, titles[s]).classes("justify-start")
            ui.separator()
            stage_box = ui.column().classes("gap-0 w-full")
            savedl = ui.label().classes("text-xs text-gray-500")
            with ui.row().classes("gap-1"):
                ui.button("Open folder", icon="folder", on_click=lambda: open_folder(p.folder)).props("flat dense")
                ui.button("Home", icon="home", on_click=lambda: ui.navigate.to("/")).props("flat dense")
        with ui.tab_panels(tabs, value="data").classes("grow min-w-0") as panels:
            with ui.tab_panel("data"):
                data_panel(p, saver, kind)
            if kind == "calltypes":
                with ui.tab_panel("labels"):
                    labels_panel(p, saver)
            with ui.tab_panel("settings"):
                settings_panel(saver, kind)
            with ui.tab_panel("run"):
                run_ui = run_panel(p)
            with ui.tab_panel("results"):
                results_box = ui.column().classes("w-full")
            with ui.tab_panel("apply"):
                apply_box = ui.column().classes("w-full")

    last = {"mtime": None, "panel": None}

    def refresh_results(force=False):
        mt = p.status_path.stat().st_mtime if p.status_path.exists() else 0
        if not force and mt == last["mtime"] and panels.value == last["panel"]:
            return
        last["mtime"], last["panel"] = mt, panels.value
        if panels.value == "results":
            results_box.clear()
            with results_box:
                (results.rf_results if kind == "randomforest" else results.ct_results)(p, saver)
        elif panels.value == "apply":
            apply_box.clear()
            with apply_box:
                (results.rf_apply if kind == "randomforest" else results.ct_apply)(p, saver, start_run)

    def refresh_status():
        states = p.stage_states()
        stage_box.clear()
        with stage_box:
            ui.label("Stages").classes("text-xs text-gray-500")
            for stage in wf.stages:
                icon, col = STATE_ICON[states[stage.name]]
                with ui.row().classes("items-center gap-1 no-wrap"):
                    ui.icon(icon).classes(col + (" animate-spin" if states[stage.name] == "running" else ""))
                    ui.label(stage.title).classes("text-sm").tooltip(STATE_TEXT[states[stage.name]])
        savedl.text = "All changes saved" if saved["t"] else ""

    def start_run(stages=None, force=False):
        if p.is_running():
            ui.notify("Already running", type="warning")
            return
        start_background(p.folder, stages, force)
        ui.notify("Started: progress is on the Run tab. You can close the browser; it keeps running.")
        panels.value = "run"

    run_ui["start"] = start_run
    panels.on_value_change(lambda: refresh_results(True))

    def tick():
        run_ui["refresh"]()
        refresh_status()
        if panels.value in ("results", "apply") and not p.is_running():
            refresh_results()
    refresh_status()
    ui.timer(1.0, tick)


# ------------------------------------------------------------------------ data / labels -----------
def data_panel(p, saver, kind):
    if kind == "randomforest":
        ui.markdown("#### Detections\nOne row per detection with its measurements, a **label** (class) and a "
                    "**group** (encounter or other independent unit, used for testing). PAMGuard ROCCA contour "
                    "stats files already have these (`KnownSpecies`, `EncounterID`).")
        typ = ui.select({"rocca": "ROCCA contour stats (file or folder of RoccaContourStats*.csv)",
                         "table": "Any table of detections (CSV)"}, value=saver.get("data.type"),
                        label="Data type", on_change=lambda e: (saver.set("data.type", e.value), opts.refresh()))
        typ.classes("w-[32rem]")
        path_row(saver, "data.path", "File or folder", "any", (".csv",))

        @ui.refreshable
        def opts():
            if saver.get("data.type") == "rocca":
                field(saver, {"key": "data.voc_type", "label": "Detections are", "kind": "select",
                              "options": {"whistle": "whistles", "click": "clicks"}})
            else:
                field(saver, {"key": "data.label_col", "label": "Label column", "kind": "text"})
                field(saver, {"key": "data.group_col", "label": "Group column (e.g. encounter)", "kind": "text"})
            field(saver, {"key": "data.classes", "label": "Classes to use (blank = all)", "kind": "list",
                          "help": "comma-separated"})
            field(saver, {"key": "data.omit", "label": "Features to leave out", "kind": "list",
                          "help": "comma-separated column names"})
        opts()
        async def check():
            out.clear()
            with out:
                ui.spinner()
            r = await run.io_bound(checks.check_rf_data, saver.project.config()["data"])
            show_check(out, r)
        ui.button("Check data", icon="fact_check", on_click=check)
        out = ui.column().classes("w-full")
    else:
        ui.markdown("#### Whistle contours\nA folder of PAMGuard **Whistle and Moan Detector** binary files "
                    "(searched in subfolders too), or a contours JSON file (the format of the example data).")
        ui.select({"binaries": "PAMGuard binaries (folder)", "json": "Contours JSON (.json or .json.gz)"},
                  value=saver.get("data.source"), label="Source",
                  on_change=lambda e: saver.set("data.source", e.value)).classes("w-96")
        path_row(saver, "data.path", "Folder or file", "any", (".json", ".gz"))
        async def check():
            out.clear()
            with out:
                ui.spinner()
            d = saver.project.config()["data"]
            r = await run.io_bound(checks.check_contours, d["source"], d["path"])
            show_check(out, r)
        ui.button("Check contours", icon="fact_check", on_click=check)
        out = ui.column().classes("w-full")


def labels_panel(p, saver):
    ui.markdown("#### Call annotations\nOne row per annotated call: `event_id, label, start, end` (UTC) plus "
                "any other label columns (e.g. `species`, `ecotype`). Calls of no particular type can be labelled "
                "`untyped`. Or use annotation tables (one per recording/day; text or Excel), converted for you.")
    ui.select({"csv": "Annotations CSV", "tables": "Annotation tables"}, value=saver.get("labels.source"),
              label="Source", on_change=lambda e: (saver.set("labels.source", e.value), src.refresh())).classes("w-96")

    @ui.refreshable
    def src():
        if saver.get("labels.source") == "csv":
            path_row(saver, "labels.path", "Annotations CSV", "file", (".csv",))
        else:
            field(saver, {"key": "labels.table_paths", "label": "Annotation table files", "kind": "list",
                          "help": "comma-separated paths"})
            field(saver, {"key": "labels.table_label_col", "label": "Label column", "kind": "text"})
            field(saver, {"key": "labels.table_keep_cols", "label": "Other label columns to keep", "kind": "list"})
    src()
    info = {"other_labels": [], "values": {}}

    async def check():
        out.clear()
        with out:
            ui.spinner()
        r = await run.io_bound(checks.check_annotations, saver.project.config()["labels"])
        show_check(out, r)
        info.update(r.get("info") or {})
        tasks_ui.refresh()
    ui.button("Check annotations", icon="fact_check", on_click=check)
    out = ui.column().classes("w-full")
    tasks_box = ui.column().classes("w-full")

    with tasks_box:
        ui.markdown("#### Label tasks\nEach task is a yes/no question every cluster of windows is judged on, e.g. "
                    "*species: X or Y?* A cluster gets a verdict only if it's reliably one or the other, on enough "
                    "days. Check the annotations first to fill the choices.")

        @ui.refreshable
        def tasks_ui():
            tasks = saver.get("labels.tasks", []) or []
            req = saver.get("labels.requires", {}) or {}
            cols = info.get("other_labels") or sorted({t["column"] for t in tasks})
            for i, t in enumerate(tasks):
                vals = info.get("values", {}).get(t["column"]) or [t["positive"], t["negative"]]
                vals = [v for v in vals if v]

                def upd(i=i, **kw):
                    ts = saver.get("labels.tasks", [])
                    ts[i].update(kw)
                    if "column" in kw:
                        ts[i]["name"] = kw["column"]
                    saver.set("labels.tasks", ts)
                    tasks_ui.refresh()
                with ui.card().classes("w-full"):
                    with ui.row().classes("items-center gap-2"):
                        ui.select(cols or [t["column"]], value=t["column"], label="Label column",
                                  on_change=lambda e, upd=upd: upd(column=e.value)).classes("w-48")
                        ui.select(vals, value=t["positive"] or None, label="This...", new_value_mode="add-unique",
                                  on_change=lambda e, upd=upd: upd(positive=e.value)).classes("w-40")
                        ui.label("or")
                        ui.select(vals, value=t["negative"] or None, label="...that?", new_value_mode="add-unique",
                                  on_change=lambda e, upd=upd: upd(negative=e.value)).classes("w-40")
                        ui.button(icon="delete", on_click=lambda i=i: remove(i)).props("flat dense")
                    others = [o["name"] for o in tasks if o["name"] != t["name"]]
                    if others:
                        cond = req.get(t["name"])
                        by_name = {q["name"]: q for q in tasks}
                        opts_ = {"": "always"}
                        for o in others:
                            q = by_name[o]
                            for v in info.get("values", {}).get(q["column"]) or [q["positive"], q["negative"]]:
                                if v:
                                    opts_[f"{o}={v}"] = f"only when {o} is {v}"
                        if cond and f"{cond[0]}={cond[1]}" not in opts_:
                            opts_[f"{cond[0]}={cond[1]}"] = f"only when {cond[0]} is {cond[1]}"
                        ui.select(opts_, value=f"{cond[0]}={cond[1]}" if cond else "", label="Report this",
                                  on_change=lambda e, n=t["name"]: set_req(n, e.value)).classes("w-72")

        def remove(i):
            ts = saver.get("labels.tasks", [])
            ts.pop(i)
            saver.set("labels.tasks", ts)
            tasks_ui.refresh()

        def set_req(name, v):
            req = saver.get("labels.requires", {}) or {}
            req.pop(name, None)
            if v:
                o, val = v.split("=", 1)
                req[name] = [o, val]
            saver.set("labels.requires", req)

        def add():
            cols = info.get("other_labels") or ["species"]
            ts = saver.get("labels.tasks", []) or []
            c = cols[min(len(ts), len(cols) - 1)]
            vals = info.get("values", {}).get(c, ["", ""]) + ["", ""]
            ts.append({"name": c, "column": c, "positive": vals[0], "negative": vals[1]})
            saver.set("labels.tasks", ts)
            tasks_ui.refresh()
        tasks_ui()
        ui.button("Add label task", icon="add", on_click=add).props("outline")
        field(saver, {"key": "labels.include_unannotated_days", "label": "Also cluster windows from days without "
                      "annotations", "kind": "switch"})


def settings_panel(saver, kind):
    ui.markdown("#### Settings\nThe defaults are a good start. Changing a setting marks the stages it affects "
                "to be rerun; finished stages it doesn't affect are kept.")
    groups = SETTINGS[kind]
    for title, specs in groups:
        basic = [s for s in specs if not s.get("advanced")]
        adv = [s for s in specs if s.get("advanced")]
        ui.label(title).classes("text-base font-medium mt-2")
        for s in basic:
            field(saver, s)
        if adv:
            with ui.expansion("Advanced", icon="tune").classes("w-full"):
                for s in adv:
                    field(saver, s)


# ------------------------------------------------------------------------ run ---------------------
def run_panel(p):
    ui.markdown("#### Run\nStages already done with the same settings are skipped, so you can stop and resume. "
                "Runs continue in the background if you close the browser.")
    h = {}
    with ui.row().classes("gap-2"):
        run_btn = ui.button("Run", icon="play_arrow", on_click=lambda: h["start"]())
        ui.button("Rerun everything", icon="replay", on_click=lambda: confirm_force()).props("outline")
        stop_btn = ui.button("Stop", icon="stop", on_click=lambda: (p.request_stop(), ui.notify("Stopping..."))
                             ).props("outline color=negative")
    stage_label = ui.label().classes("font-medium mt-2")
    bar = ui.linear_progress(value=0, show_value=False).classes("w-full")
    msg = ui.label().classes("text-sm text-gray-600 font-mono")
    errors = ui.column().classes("w-full")
    with ui.expansion("Log", icon="article", value=False).classes("w-full"):
        log = ui.log(max_lines=400).classes("w-full h-80 text-xs")
    seen = {"n": None}

    async def confirm_force():
        with ui.dialog() as d, ui.card():
            ui.label("Rerun every stage from scratch?")
            with ui.row():
                ui.button("Cancel", on_click=lambda: d.submit(False)).props("flat")
                ui.button("Rerun", on_click=lambda: d.submit(True))
        if await d:
            h["start"](None, True)

    def refresh():
        st = p.status()
        r = st.get("run", {})
        running = p.is_running(st)
        run_btn.set_enabled(not running)
        stop_btn.set_enabled(running)
        if running:
            stage = r.get("stage")
            title = next((s.title for s in p.workflow.stages if s.name == stage), "Starting")
            stage_label.text = f"Running: {title}"
            prog = r.get("progress")
            if prog is None:
                bar.props("indeterminate")
            else:
                bar.props(remove="indeterminate")
                bar.value = prog
            msg.text = r.get("message", "")
        else:
            bar.props(remove="indeterminate")
            state = r.get("state")
            if state == "running":
                state = "stopped unexpectedly"
            stage_label.text = {None: "Not run yet", "finished": "Last run finished",
                                "failed": "Last run failed", "stopped": "Stopped"}.get(state, f"Last run: {state}")
            bar.value = 1 if state == "finished" else 0
            msg.text = (r.get("ended") or "")
        errors.clear()
        with errors:
            for s in p.workflow.stages:
                e = st.get("stages", {}).get(s.name, {})
                if e.get("state") == "failed":
                    ui.label(f"{s.title} failed: {e.get('error')}").classes("text-red-700")
        size = p.log_path.stat().st_size if p.log_path.exists() else 0
        if size != seen["n"]:
            seen["n"] = size
            log.clear()
            for line in p.log_tail(300):
                log.push(line)
    refresh()
    h["refresh"] = refresh
    return h


# ------------------------------------------------------------------------ start -------------------
def _free_port(start=8765):
    for port in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)):
                return port
    return start


def main(port=None, show=True):
    import matplotlib
    matplotlib.use("Agg")
    app.native.window_args = {}
    ui.run(title="speciesClassifiers", port=port or _free_port(), host="127.0.0.1", reload=False, show=show,
           favicon="🐬", show_welcome_message=True)


if __name__ in {"__main__", "__mp_main__"}:
    main()
