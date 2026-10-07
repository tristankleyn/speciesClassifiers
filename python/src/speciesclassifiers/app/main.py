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

from . import diagnostics_page as _diagnostics  # noqa: F401  (registers /diagnostics)
from . import results
from .settings import SETTINGS, applies
from .widgets import Saver, field, open_folder, path_row, pick_path, show_check

HOME = Path.home() / ".speciesclassifiers"
RECENT = HOME / "recent.json"
DEFAULT_PARENT = Path.home() / "speciesClassifiers projects"

STATE_ICON = {"done": ("check_circle", "text-green-600"), "outdated": ("update", "text-amber-600"),
              "failed": ("error", "text-red-600"), "pending": ("radio_button_unchecked", "text-gray-400"),
              "running": ("autorenew", "text-blue-600"), "skipped": ("remove_circle_outline", "text-gray-300")}
STATE_TEXT = {"done": "done", "outdated": "settings changed: will rerun", "failed": "failed",
              "pending": "not run yet", "running": "running", "skipped": "skipped (not set up)"}

METHODS = [   # (key, title, icon, one-sentence caption, requirements)
    ("randomforest", "Create Random Forest Classifier", "forest",
     "Classifies species from measurements of individual detections, combined per encounter.",
     ["**Detections**: PAMGuard ROCCA contour stats (`RoccaContourStats*.csv`, a file or folder), or any CSV "
      "with one row per detection and numeric feature columns",
      "**Labels**: the class of each detection (ROCCA: `KnownSpecies`; other CSVs: a label column)",
      "**Groups**: the encounter each detection belongs to (ROCCA: `EncounterID`; other CSVs: a group column), "
      "with at least 2 encounters per class"]),
    ("delphinid", "Create Deep Learning Classifier", "psychology",
     "Trains a delphinID neural network on averaged spectra of whistles or clicks, to run in PAMGuard.",
     ["**Detections**: PAMGuard binary files from the Whistle and Moan Detector (whistles) or Click Detector "
      "(clicks)",
      "**Annotations**: a CSV of labelled encounter periods: `event_id, label, start, end` (UTC)",
      "**Encounters**: several per class, for testing on encounters the model hasn't seen",
      "**TensorFlow**: installed locally, or train in Google Colab (no install)"]),
    ("calltypes", "Create Clustering Classifier (call types)", "hub",
     "Groups whistle contours into call types and learns which labels (e.g. species) each call type indicates.",
     ["**Detections**: a folder of PAMGuard Whistle and Moan Detector binary files (or a contours JSON)",
      "**Call annotations**: a CSV with one row per call: `event_id, label` (call type, or `untyped`), "
      "`start, end` (UTC), plus optional label columns such as `species`; or annotation tables",
      "**Days**: annotated calls from at least 2 days; days without annotations can be included too"]),
]
COMING = {"transferlearning": "Transfer learning isn't in the app yet: use notebook 04_transferlearning "
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
    from urllib.parse import quote
    ui.navigate.to(f"/project?path={quote(str(Project(folder).folder), safe='/:')}")


# ------------------------------------------------------------------------ home ---------------------
def header(title="speciesClassifiers"):
    with ui.header().classes("bg-slate-800 items-center"):
        with ui.link(target="/").classes("text-white no-underline flex items-center gap-2"):
            ui.icon("graphic_eq", size="sm")
            ui.label("speciesClassifiers").classes("text-lg font-medium")
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
    state = {"choice": classifier}
    with ui.dialog() as dialog, ui.card().classes("w-[48rem] max-w-full"):
        ui.label("New project").classes("text-xl font-medium")
        ui.label("Choose a method").classes("text-sm text-gray-600")

        @ui.refreshable
        def methods():
            for key, title, icon, caption, reqs in METHODS:
                chosen = state["choice"] == key
                card = ui.card().classes("w-full cursor-pointer p-3 " + (
                    "border-2 border-blue-500 bg-blue-50" if chosen else "border border-gray-200 hover:bg-slate-50"))
                card.props("flat")
                card.on("click", lambda key=key: choose(key))
                with card:
                    with ui.row().classes("items-center gap-3 no-wrap"):
                        ui.icon(icon, size="md").classes("text-blue-600" if chosen else "text-gray-500")
                        with ui.column().classes("gap-0"):
                            ui.label(title).classes("font-medium")
                            ui.label(caption).classes("text-sm text-gray-600")
                    if chosen:
                        ui.label("What you need").classes("text-sm font-medium mt-2")
                        ui.markdown("\n".join(f"- {r}" for r in reqs)).classes("text-sm")
                        if key in COMING:
                            ui.label(COMING[key]).classes("text-sm text-amber-700")

        def choose(key):
            state["choice"] = key
            methods.refresh()
            update()
        methods()
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
            c = state["choice"]
            create.set_enabled(bool(c) and c not in COMING)
        update()

        def do_create():
            folder = Path(parent.value).expanduser() / name.value.strip()
            try:
                p = Project.create(folder, state["choice"], name=name.value.strip())
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
        from speciesclassifiers.project.workflows import rf_binaries_example
        choices = [(c, WORKFLOWS[c].title, WORKFLOWS[c].description, WORKFLOWS[c].example)
                   for c in ("randomforest", "delphinid", "calltypes")]
        choices.insert(1, ("randomforest", "Random forest: whistles + clicks",
                           "A base random forest per detection type (whistle contours, click spectra) and a final "
                           "forest that decides each time block from both.", rf_binaries_example))
        from speciesclassifiers.project.workflows import dl_multi_example
        choices.insert(3, ("delphinid", "delphinID: whistles + clicks",
                           "A delphinID model per detection type (whistle and click frames) and a final forest that "
                           "decides each time block from both. Needs TensorFlow.", dl_multi_example))
        for i, (c, title, desc, _) in enumerate(choices):
            with ui.card().classes("w-full cursor-pointer hover:bg-slate-50").on(
                    "click", lambda i=i: dialog.submit(i)):
                ui.label(title).classes("font-medium")
                ui.label(desc).classes("text-sm text-gray-600")
    i = await dialog
    if i is None:
        return
    c, title, _, example = choices[i]
    k = 1
    while (DEFAULT_PARENT / f"example {title.lower().replace(':', '')} {k}").exists():
        k += 1
    folder = DEFAULT_PARENT / f"example {title.lower().replace(':', '')} {k}"
    Project.create(folder, c, name=f"Example: {title}", config=example())
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
             "apply": "Export to PAMGuard" if kind == "delphinid" and cfg["data"].get("source") != "multi"
             else "Classify new data · export"}
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
                next_button(panels, steps, "data", names)
            if kind == "calltypes":
                with ui.tab_panel("labels"):
                    labels_panel(p, saver)
                    next_button(panels, steps, "labels", names)
            with ui.tab_panel("settings"):
                settings_ui = ui.refreshable(settings_panel)
                settings_ui(saver, kind)
                next_button(panels, steps, "settings", names)
            with ui.tab_panel("run"):
                run_ui = run_panel(p, kind)
                run_next = ui.row()
            with ui.tab_panel("results"):
                results_box = ui.column().classes("w-full")
                next_button(panels, steps, "results", names)
            with ui.tab_panel("apply"):
                apply_box = ui.column().classes("w-full")

    last = {"mtime": None, "panel": None, "busy": False, "states": None, "run_state": None}

    async def refresh_results(force=False):
        """(Re)build the results / apply panel when it's shown or the outputs changed. Heavy work runs in
        background threads, so the page stays responsive."""
        if last["busy"]:
            return
        mt = p.status_path.stat().st_mtime if p.status_path.exists() else 0
        if not force and mt == last["mtime"] and panels.value == last["panel"]:
            return
        last["mtime"], last["panel"] = mt, panels.value
        if panels.value not in ("results", "apply"):
            return
        box = results_box if panels.value == "results" else apply_box
        last["busy"] = True
        try:
            box.clear()
            with box:
                if panels.value == "results":
                    await {"randomforest": results.rf_results, "calltypes": results.ct_results,
                           "delphinid": results.dl_results}[kind](p, saver)
                else:
                    await {"randomforest": results.rf_apply, "calltypes": results.ct_apply,
                           "delphinid": results.dl_export}[kind](p, saver, start_run)
        finally:
            last["busy"] = False

    def refresh_status():
        states = p.stage_states()
        savedl.text = "All changes saved" if saved["t"] else ""
        if states == last["states"]:
            return
        last["states"] = states
        stage_box.clear()
        with stage_box:
            ui.label("Stages").classes("text-xs text-gray-500")
            for stage in wf.stages:
                icon, col = STATE_ICON[states[stage.name]]
                with ui.row().classes("items-center gap-1 no-wrap"):
                    ui.icon(icon).classes(col + (" animate-spin" if states[stage.name] == "running" else ""))
                    ui.label(stage.title).classes("text-sm").tooltip(STATE_TEXT[states[stage.name]])

    def start_run(stages=None, force=False):
        if p.is_running():
            ui.notify("Already running", type="warning")
            return
        if kind == "delphinid" and stages is None:
            from speciesclassifiers.project.checks import tensorflow_status
            if not tensorflow_status()["ok"]:
                stages = ["frames"]
                ui.notify("No suitable TensorFlow here, so only the detection frames are made. Train in Google "
                          "Colab or install TensorFlow (options above).", type="warning", multi_line=True)
        start_background(p.folder, stages, force)
        last["run_state"] = "running"
        ui.notify("Started: progress is on the Run tab. You can close the browser; it keeps running.")
        panels.value = "run"

    def update_run_next():
        st = p.status().get("run", {})
        state = "running" if p.is_running() else st.get("state")
        if state == last["run_state"]:
            return
        if last["run_state"] == "running" and state == "finished":
            ui.notify("Run finished: see the results", type="positive")
        last["run_state"] = state
        run_next.clear()
        if state == "finished" or (state is None and all(v in ("done", "skipped") for v in p.stage_states().values())):
            with run_next:
                ui.button(f"Next: {names['results']}", icon="arrow_forward",
                          on_click=lambda: setattr(panels, "value", "results")).props("size=lg").classes("mt-4")

    run_ui["start"] = start_run
    def on_panel():
        if panels.value == "settings":
            settings_ui.refresh()          # some settings depend on the data choices (e.g. whistles or clicks)
        return refresh_results(True)
    panels.on_value_change(on_panel)

    async def tick():
        run_ui["refresh"]()
        if "tf" in run_ui and panels.value == "run":
            run_ui["tf"]()
        refresh_status()
        update_run_next()
        if panels.value in ("results", "apply") and not p.is_running():
            await refresh_results()
    refresh_status()
    update_run_next()
    ui.timer(1.0, tick)


def next_button(panels, steps, step, names):
    i = steps.index(step)
    if i + 1 < len(steps):
        nxt = steps[i + 1]
        ui.button(f"Next: {names[nxt]}", icon="arrow_forward",
                  on_click=lambda: setattr(panels, "value", nxt)).props("size=lg").classes("mt-6")


# ------------------------------------------------------------------------ data / labels -----------
def dl_data_panel(p, saver):
    ui.markdown("#### Detections and labels\nPAMGuard binary files plus an **annotations CSV** of labelled "
                "periods: `event_id, label, start, end` (UTC), one row per period (several periods can share an "
                "event). Only detections inside annotated periods are used. Or use frames already made with "
                "notebook 01.")
    ui.select({"binaries": "PAMGuard binaries + annotations CSV", "frames": "Frames CSV made with notebook 01",
               "multi": "Whistles and/or clicks: a model per type + a final layer per block"},
              value=saver.get("data.source"), label="Source",
              on_change=lambda e: (saver.set("data.source", e.value), opts.refresh())).classes("w-[32rem]")

    def set_voc(v):
        saver.set("data.voc_type", v)
        fmin, fmax = (10000.0, 40000.0) if v == "click" else (2000.0, 20000.0)
        saver.set("settings.acoustic.fmin_hz", fmin)
        saver.set("settings.acoustic.fmax_hz", fmax)
        name = saver.get("model_name", "")
        if name.startswith("delphinID_"):
            saver.set("model_name", f"delphinID_{v}s")

    @ui.refreshable
    def opts():
        if saver.get("data.source") == "multi":
            from .stacked import dl_types_panel
            dl_types_panel(saver)
            return
        if saver.get("data.source") == "frames":
            path_row(saver, "data.frames_csv", "Frames CSV", "file", (".csv",))
            path_row(saver, "data.frames_params", "Frame parameters JSON (made with the frames)", "file", (".json",))
        else:
            ui.select({"whistle": "Whistles (Whistle and Moan Detector)", "click": "Clicks (Click Detector)"},
                      value=saver.get("data.voc_type"), label="Detections",
                      on_change=lambda e: set_voc(e.value)).classes("w-96")
            path_row(saver, "data.binary_folder", "Folder of binary files", "dir")
            path_row(saver, "data.annotations", "Annotations CSV", "file", (".csv",))
            field(saver, {"key": "data.sample_rate", "label": "Sample rate (Hz)", "kind": "number", "optional": True,
                          "help": "blank = work it out from the binaries"})
    opts()

    async def check():
        out.clear()
        with out:
            ui.spinner()
        r = await run.io_bound(checks.check_dl_data, saver.project.config()["data"])
        show_check(out, r)
    ui.button("Check data", icon="fact_check", on_click=check)
    out = ui.column().classes("w-full")


def data_panel(p, saver, kind):
    if kind == "delphinid":
        return dl_data_panel(p, saver)
    if kind == "randomforest":
        ui.markdown("#### Detections\nEither **PAMGuard binaries** (whistles and/or clicks) with labelled periods: "
                    "a base random forest per detection type, and a final forest that decides each time block. Or "
                    "a table with one row per detection, its measurements, a **label** (class) and a **group** "
                    "(encounter): PAMGuard ROCCA contour stats files already have these (`KnownSpecies`, "
                    "`EncounterID`).")
        typ = ui.select({"binaries": "PAMGuard binaries: whistles and/or clicks, with labelled periods",
                         "rocca": "ROCCA contour stats (file or folder of RoccaContourStats*.csv)",
                         "table": "Any table of detections (CSV)"}, value=saver.get("data.type"),
                        label="Data type", on_change=lambda e: (saver.set("data.type", e.value), opts.refresh()))
        typ.classes("w-[32rem]")

        @ui.refreshable
        def opts():
            if saver.get("data.type") == "binaries":
                from .stacked import detection_types_panel
                detection_types_panel(saver)
                return
            path_row(saver, "data.path", "File or folder", "any", (".csv",))
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

            async def check():
                out.clear()
                with out:
                    ui.spinner()
                r = await run.io_bound(checks.check_rf_data, saver.project.config()["data"])
                show_check(out, r)
            ui.button("Check data", icon="fact_check", on_click=check)
            out = ui.column().classes("w-full")
        opts()
    else:
        ct_data_panel(p, saver)


def ct_data_panel(p, saver):
    ui.markdown("#### Detections\nWhistles, clicks or both. Each detection type gets its own windows, "
                "measurements and clusters; their class probabilities are combined for each decision block.")
    with ui.row().classes("gap-6"):
        ui.checkbox("Whistles", value=saver.get("data.use_whistles", True),
                    on_change=lambda e: (saver.set("data.use_whistles", e.value), body.refresh()))
        ui.checkbox("Clicks", value=saver.get("data.use_clicks", False), on_change=lambda e: set_clicks(e.value))

    def set_clicks(on):
        saver.set("data.use_clicks", on)
        d = saver.project.config()["data"]
        if on and not d.get("click_path") and d.get("source") == "binaries" and d.get("path"):
            saver.set("data.click_path", d["path"])       # usually the same binaries folder
            saver.set("data.click_source", "binaries")
        body.refresh()

    async def check(kind_, out):
        out.clear()
        with out:
            ui.spinner()
        d = saver.project.config()["data"]
        if kind_ == "whistle":
            r = await run.io_bound(checks.check_contours, d["source"], d["path"])
        else:
            r = await run.io_bound(checks.check_clicks, d.get("click_source", "binaries"), d.get("click_path", ""))
        show_check(out, r)

    @ui.refreshable
    def body():
        d = saver.project.config()["data"]
        if not d.get("use_whistles", True) and not d.get("use_clicks"):
            ui.label("Choose at least one detection type.").classes("text-amber-700")
        if d.get("use_whistles", True):
            ui.markdown("##### Whistle contours\nA folder of PAMGuard **Whistle and Moan Detector** binary files "
                        "(searched in subfolders too), or a contours JSON file (the format of the example data).")
            ui.select({"binaries": "PAMGuard binaries (folder)", "json": "Contours JSON (.json or .json.gz)"},
                      value=d.get("source"), label="Source",
                      on_change=lambda e: saver.set("data.source", e.value)).classes("w-96")
            path_row(saver, "data.path", "Folder or file", "any", (".json", ".gz"))
            ui.button("Check contours", icon="fact_check", on_click=lambda: check("whistle", wout))
            wout = ui.column().classes("w-full")
        if d.get("use_clicks"):
            ui.markdown("##### Clicks\nA folder of PAMGuard **Click Detector** binary files (often the same folder "
                        "as the whistles), or a CSV with a `start` column (UTC; Unix seconds or date-time) and "
                        "optionally `duration` (s). From binaries, each click's spectrum is measured too (peak "
                        "frequency, bandwidths, band profile); a CSV can carry these as columns.")
            ui.select({"binaries": "PAMGuard binaries (folder)", "csv": "Clicks CSV (.csv or .csv.gz)"},
                      value=d.get("click_source", "binaries"), label="Source",
                      on_change=lambda e: saver.set("data.click_source", e.value)).classes("w-96")
            path_row(saver, "data.click_path", "Folder or file", "any", (".csv", ".gz"))
            ui.button("Check clicks", icon="fact_check", on_click=lambda: check("click", cout))
            cout = ui.column().classes("w-full")
    body()


def labels_panel(p, saver):
    ui.markdown("#### Annotations\nEither **labelled periods** (events): one row per period with `event_id, start, end` "
                "(UTC) and label columns such as `species`; no call types needed. Or **call annotations**: one row "
                "per call with `event_id, label` (call type, or `untyped`), `start, end`, plus optional label columns; "
                "or annotation tables (one per recording/day; text or Excel), converted for you.")
    ui.select({"periods": "Labelled periods (events) CSV", "csv": "Call annotations CSV",
               "tables": "Annotation tables"}, value=saver.get("labels.source"),
              label="Source", on_change=lambda e: (saver.set("labels.source", e.value), src.refresh())).classes("w-96")

    @ui.refreshable
    def src():
        if saver.get("labels.source") in ("csv", "periods"):
            path_row(saver, "labels.path", "Annotations CSV", "file", (".csv",))
        else:
            field(saver, {"key": "labels.table_paths", "label": "Annotation table files", "kind": "list",
                          "help": "comma-separated paths"})
            field(saver, {"key": "labels.table_label_col", "label": "Label column", "kind": "text"})
            field(saver, {"key": "labels.table_keep_cols", "label": "Other label columns to keep", "kind": "list"})
    src()
    info = {"other_labels": [], "values": {}, "columns": []}

    async def check(quiet=False):
        out.clear()
        with out:
            ui.spinner()
        r = await run.io_bound(checks.check_annotations, saver.project.config()["labels"])
        show_check(out, r)
        info.update(r.get("info") or {})
        editor.refresh()
        tasks_ui.refresh()
    ui.button("Check annotations", icon="fact_check", on_click=check)
    out = ui.column().classes("w-full")

    ui.markdown("#### Classification\nWhat should the clusters tell apart? Add a layer for the label you care about "
                "(e.g. `species`): each value becomes a class, or group values into classes. Add a next layer within "
                "a class or group to decide further (e.g. the species within a group).")
    from .pathway_ui import pathway_editor
    editor = pathway_editor(saver, info)
    with ui.row().classes("items-end gap-6 mt-2"):
        field(saver, {"key": "settings.reliability.unit", "label": "Test by leaving out one", "kind": "select",
                      "options": {"event_id": "event at a time", "day": "day at a time"}})
        field(saver, {"key": "settings.reliability.min_label_days", "label": "Events (or days) a cluster's label "
                      "must come from", "kind": "int", "min": 1})
    field(saver, {"key": "labels.include_unannotated_days", "label": "Also cluster windows from days without "
                  "annotations", "kind": "switch"})
    if saver.get("labels.path") or saver.get("labels.table_paths"):
        ui.timer(0.2, check, once=True)
    tasks_box = ui.expansion("Yes/no label tasks (as in the killer whale example)", icon="rule",
                             value=bool(saver.get("labels.tasks")) and not saver.get("labels.pathway")
                             ).classes("w-full")

    with tasks_box:
        ui.markdown("Each task is a yes/no question every cluster is judged on, e.g. *species: X or Y?*, decided "
                    "per block by majority vote and tested by leaving out days. Used only when no classification "
                    "layers are set up above.")

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


def settings_panel(saver, kind):
    ui.markdown("#### Settings\nThe defaults are a good start. Changing a setting marks the stages it affects "
                "to be rerun; finished stages it doesn't affect are kept.")
    groups = SETTINGS[kind]
    cfg = saver.project.config()
    for title, specs in groups:
        specs = [s for s in specs if applies(s, cfg)]
        if not specs:
            continue
        basic = [s for s in specs if not s.get("advanced")]
        adv = [s for s in specs if s.get("advanced")]
        ui.label(title).classes("text-base font-medium mt-2")
        for s in basic:
            _setting(saver, s)
        if adv:
            with ui.expansion("Advanced", icon="tune").classes("w-full"):
                for s in adv:
                    _setting(saver, s)


def _setting(saver, spec):
    if spec.get("kind") == "features":
        from .features_ui import features_field
        return features_field(saver, spec)
    return field(saver, spec)


# ------------------------------------------------------------------------ run ---------------------
def run_panel(p, kind):
    ui.markdown("#### Run\nStages already done with the same settings are skipped, so you can stop and resume. "
                "Runs continue in the background if you close the browser.")
    h = {}
    if kind == "delphinid":
        from .tensorflow_panel import tensorflow_box
        h["tf"] = tensorflow_box(p, lambda *a, **k: h["start"](*a, **k))
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
        key = tuple(f.stat().st_mtime if f.exists() else 0 for f in (p.status_path, p.log_path))
        if key == seen.get("key") and not p.is_running():
            return
        seen["key"] = key
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
FAVICON = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><rect width="24" height="24" rx="5" '
           'fill="#1e293b"/><path d="M5 10v4M8.5 7v10M12 4v16M15.5 8v8M19 10.5v3" stroke="#fff" '
           'stroke-width="2" stroke-linecap="round"/></svg>')


def _free_port(start=8765):
    for port in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)):
                return port
    return start


def main(port=None, show=True):
    import threading

    import matplotlib
    matplotlib.use("Agg")
    threading.Thread(target=lambda: __import__("matplotlib.figure"), daemon=True).start()   # warm up
    app.native.window_args = {}
    ui.run(title="speciesClassifiers", port=port or _free_port(), host="127.0.0.1", reload=False, show=show,
           favicon=FAVICON, show_welcome_message=True)


if __name__ in {"__main__", "__mp_main__"}:
    main()
