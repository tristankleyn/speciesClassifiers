"""delphinID needs TensorFlow: check it, offer to install it here, or train in Google Colab instead."""

import shutil
import subprocess
import sys
from pathlib import Path

from nicegui import run, ui

from speciesclassifiers.project import colab
from speciesclassifiers.project.checks import TF_MIN, tensorflow_status

from .widgets import open_folder, pick_path

LOG = Path.home() / ".speciesclassifiers" / "tensorflow_install.log"
TF_SPEC = f"tensorflow>={TF_MIN[0]}.{TF_MIN[1]}"
_install = {"proc": None}


def _install_cmd():
    uv = shutil.which("uv")
    if uv:   # the launcher's environment (made by uv) has no pip; uv installs into it directly
        return [uv, "pip", "install", "--python", sys.executable, "--upgrade", TF_SPEC]
    return [sys.executable, "-m", "pip", "install", "--upgrade", TF_SPEC]


def start_install():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG, "w", encoding="utf-8")
    log.write(f"$ {' '.join(_install_cmd())}\n")
    log.flush()
    _install["proc"] = subprocess.Popen(_install_cmd(), stdout=log, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL)


def installing():
    p = _install["proc"]
    return p is not None and p.poll() is None


def tensorflow_box(p, start_run):
    """Card at the top of the Run screen of a delphinID project."""
    box = ui.column().classes("w-full")
    state = {"key": None}

    def render():
        st = tensorflow_status()
        key = (st["ok"], st["version"], installing(), _install["proc"] and _install["proc"].returncode,
               p.stage_states().get("frames"))
        if key == state["key"]:
            return
        state["key"] = key
        box.clear()
        with box:
            if st["ok"]:
                with ui.row().classes("items-center gap-1"):
                    ui.icon("check_circle").classes("text-green-600")
                    ui.label(f"{st['message']} Training runs on this computer.").classes("text-green-800")
                with ui.expansion("Or train in Google Colab", icon="cloud").classes("w-full"):
                    colab_steps(p, start_run)
                return
            with ui.card().classes("w-full bg-amber-50 border border-amber-300").props("flat"):
                with ui.row().classes("items-center gap-1"):
                    ui.icon("warning").classes("text-amber-700")
                    ui.label(st["message"]).classes("font-medium text-amber-900")
                ui.label("delphinID models are trained with TensorFlow. Two options:").classes("text-sm")
                with ui.row().classes("w-full gap-4 items-start no-wrap"):
                    with ui.column().classes("w-1/2 gap-1"):
                        ui.label("1 · Train in Google Colab (recommended, nothing to install)").classes(
                            "font-medium")
                        ui.label("Free, in your browser, with TensorFlow ready. The app makes the detection frames "
                                 "here; Colab trains the model; you import the results.").classes("text-sm")
                        colab_steps(p, start_run)
                    with ui.column().classes("w-1/2 gap-1"):
                        what = "Upgrade" if st["installed"] else "Install"
                        ui.label(f"2 · {what} TensorFlow on this computer").classes("font-medium")
                        ui.label(f"Downloads about 600 MB and takes a few minutes ({TF_SPEC}). Then everything "
                                 "runs here.").classes("text-sm")
                        if installing():
                            with ui.row().classes("items-center gap-2"):
                                ui.spinner()
                                ui.label("Installing... you can keep using the app.").classes("text-sm")
                        else:
                            rc = _install["proc"].returncode if _install["proc"] else None
                            if rc not in (None, 0):
                                ui.label(f"The install failed (see {LOG}). Google Colab still works.").classes(
                                    "text-sm text-red-700")
                            ui.button(f"{what} TensorFlow", icon="download",
                                      on_click=lambda: (start_install(), render())).props("outline")
                        ui.label(f"Or in a terminal: pip install \"{TF_SPEC}\"").classes(
                            "text-xs text-gray-500 font-mono")
                if LOG.is_file() and (installing() or (_install["proc"] and _install["proc"].returncode)):
                    with ui.expansion("Install log", icon="article").classes("w-full"):
                        lines = LOG.read_text(errors="replace").splitlines()[-30:]
                        ui.label("\n".join(lines)).classes("text-xs font-mono whitespace-pre-wrap")

    render()
    return render


def colab_steps(p, start_run):
    multi = (p.config().get("data") or {}).get("source") == "multi"
    states = p.stage_states()
    frames_done = states.get("frames") == "done"

    async def bundle():
        try:
            path = await run.io_bound(colab.make_bundle, p.folder)
        except Exception as e:
            ui.notify(str(e), type="warning")
            return
        ui.notify(f"Bundle saved: {path.name}", type="positive")
        open_folder(path.parent)

    async def import_zip():
        z = await pick_path(str(p.folder), "file", (".zip",), title="Results zip from Colab")
        if not z:
            return
        try:
            msgs = await run.io_bound(colab.import_results, p.folder, z)
        except Exception as e:
            ui.notify(f"Couldn't import: {e}", type="negative")
            return
        ui.notify("Imported " + "; ".join(msgs), type="positive", multi_line=True)

    with ui.column().classes("gap-2 text-sm"):
        with ui.row().classes("items-center gap-2"):
            ui.label("a.")
            ui.button("Make frames", icon="play_arrow", on_click=lambda: start_run(["frames"])).props(
                "dense outline" + (" disable" if frames_done else ""))
            if frames_done:
                ui.label("done").classes("text-green-700")
        with ui.row().classes("items-center gap-2"):
            ui.label("b.")
            ui.button("Make Colab bundle", icon="archive", on_click=bundle).props("dense outline")
            ui.label("saved in the project's colab folder").classes("text-gray-500")
        with ui.row().classes("items-center gap-2"):
            ui.label("c.")
            ui.link("Open the training notebook in Colab", colab.notebook_url(p.config()), new_tab=True)
        ui.label("In Colab: upload the bundle (Files panel), set BUNDLE = \"<bundle name>.zip\", then "
                 "Runtime > Run all. The results zip downloads at the end." +
                 (" For a GPU: Runtime > Change runtime type > T4 GPU. Every detection type is cross-validated, "
                  "then the final models and the final layer are trained." if multi else "")).classes(
            "text-gray-600 ml-5")
        with ui.row().classes("items-center gap-2"):
            ui.label("d.")
            ui.button("Import results from Colab", icon="upload", on_click=import_zip).props("dense outline")
