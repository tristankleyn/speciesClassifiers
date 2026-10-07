"""Small building blocks for the app: settings fields bound to project.json, a file/folder picker,
figures as images, and check results."""

import base64
import io
import os
import string
import subprocess
import sys
from pathlib import Path

from nicegui import ui

from speciesclassifiers.project.project import get_path


def nested(dotted, value):
    """"a.b.c", v -> {"a": {"b": {"c": v}}}"""
    out = value
    for k in reversed(dotted.split(".")):
        out = {k: out}
    return out


class Saver:
    """Writes single settings to project.json as they change, and tells the page."""

    def __init__(self, project, on_saved=None):
        self.project, self.on_saved = project, on_saved

    def set(self, dotted, value):
        cfg = self.project.config()
        d = cfg
        *head, last = dotted.split(".")
        for k in head:
            d = d.setdefault(k, {})
        d[last] = value
        self.project.save_config(cfg)
        if self.on_saved:
            self.on_saved(dotted)

    def get(self, dotted, default=None):
        v = get_path(self.project.config(), dotted)
        return default if v is None else v


def field(saver, spec):
    """One settings field from a spec dict: key, label, kind (number/int/text/select/switch/list), help,
    options, optional (blank = None)."""
    key, label, kind = spec["key"], spec["label"], spec.get("kind", "number")
    val = get_path(saver.project.config(), key)
    if val is None and spec.get("default") is not None:
        val = spec["default"]
    tip = spec.get("help", "")

    def save(e, conv=lambda v: v):
        v = e.value if hasattr(e, "value") else e
        try:
            v = conv(v)
        except (TypeError, ValueError):
            return
        saver.set(key, v)

    with ui.row().classes("items-center gap-2 w-full no-wrap"):
        if kind in ("number", "int"):
            conv = (lambda v: None if v in (None, "") else (int(v) if kind == "int" else float(v)))
            el = ui.number(label, value=val, step=spec.get("step"), min=spec.get("min"), max=spec.get("max"),
                           on_change=lambda e: save(e, conv)).classes("w-64")
            if spec.get("optional"):
                el.props('clearable placeholder="automatic"')
        elif kind == "select":
            el = ui.select(spec["options"], value=val, label=label, on_change=save).classes("w-64")
        elif kind == "switch":
            el = ui.switch(label, value=bool(val), on_change=save)
        elif kind == "list":
            el = ui.input(label, value=", ".join(map(str, val or [])),
                          on_change=lambda e: save(e, lambda v: [x.strip() for x in v.split(",") if x.strip()])
                          ).classes("w-96")
        else:
            el = ui.input(label, value=val or "", on_change=save).classes("w-96")
        if tip:
            ui.label(tip).classes("text-xs text-gray-500")
    return el


def new_fig(w, h):
    """A matplotlib figure and axes that can be drawn in a background thread (no pyplot)."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=(w, h))
    return fig, fig.subplots()


def fig_src(fig):
    """Matplotlib figure -> data URL for ui.image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class ImageSlot:
    """A place for a figure that is drawn in a background thread, so the page stays responsive."""

    def __init__(self, classes=""):
        with ui.element("div").classes(classes) as self.box:
            self.spinner = ui.spinner(size="lg").classes("m-8")
            self.img = ui.image().classes("w-full")
            self.img.set_visibility(False)

    async def fill(self, draw):
        from nicegui import run
        try:
            src = await run.io_bound(lambda: fig_src(draw()))
            self.img.set_source(src)
            self.img.set_visibility(True)
            self.spinner.delete()
        except (RuntimeError, KeyError, ValueError):   # page or panel was closed meanwhile
            pass


def show_check(container, result):
    """Render a check result (see project.checks) into a container."""
    container.clear()
    colours = {"ok": "text-green-700", "warn": "text-amber-700", "error": "text-red-700"}
    icons = {"ok": "check_circle", "warn": "warning", "error": "error"}
    with container:
        for level, text in result["messages"]:
            with ui.row().classes("items-center gap-1 no-wrap"):
                ui.icon(icons[level]).classes(colours[level])
                ui.label(text).classes(colours[level])
        t = result.get("table")
        if t is not None and len(t):
            table(t, rows=12)


def table(df, rows=10, title=None):
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(3)
    df = df.astype(object).where(df.notna(), "")
    return ui.table.from_pandas(df, title=title, pagination=rows).props("dense flat bordered").classes("w-full")


def confusion_frame(conf):
    """Confusion counts (rows true, columns predicted) -> table: true class, one column of predictions per
    class, total predicted (kept), recall (correct / that class's total) and precision (correct / predictions
    of that class)."""
    import pandas as pd
    conf = conf if conf is not None else pd.DataFrame()
    labels = sorted(set(map(str, conf.index)) | set(map(str, conf.columns)))
    cm = conf.copy()
    cm.index, cm.columns = cm.index.map(str), cm.columns.map(str)
    cm = cm.reindex(index=labels, columns=labels, fill_value=0).fillna(0).astype(int)
    diag = pd.Series({c: cm.at[c, c] for c in labels})
    rows, cols = cm.sum(axis=1), cm.sum(axis=0)
    out = cm.copy()
    out.columns = [f"→ {c}" for c in labels]
    out.insert(0, "true", labels)
    out["total"] = rows.to_numpy()
    out["recall"] = [f"{d / r:.0%}" if r else "–" for d, r in zip(diag, rows)]
    out["precision"] = [f"{d / c:.0%}" if c else "–" for d, c in zip(diag, cols)]
    total = {"true": "all", **{f"→ {c}": int(cols[c]) for c in labels}, "total": int(rows.sum()),
             "recall": f"{diag.sum() / rows.sum():.0%}" if rows.sum() else "–", "precision": ""}
    return pd.concat([out.reset_index(drop=True), pd.DataFrame([total])], ignore_index=True)


def confusion_table(conf, title=None):
    """A confusion matrix as a table (see ``confusion_frame``): rows are the true classes, '→ X' columns the
    predictions; the last row totals the predictions (its recall is the overall accuracy)."""
    df = confusion_frame(conf)
    with ui.column().classes("gap-1"):
        if title:
            ui.label(title).classes("text-sm font-medium")
        t = ui.table.from_pandas(df.astype(object), pagination=0).props("dense flat bordered hide-bottom")
        t.add_slot("body-cell", r"""
            <q-td :props="props" :class="(props.col.name === '→ ' + props.row.true) ? 'bg-blue-100 font-bold' :
                  (props.row.true === 'all' ? 'text-gray-600' : '')">{{ props.value }}</q-td>""")
    return t


def open_folder(path):
    path = str(path)
    try:
        if os.name == "nt":
            os.startfile(path)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:  # e.g. no file manager
        ui.notify(f"Couldn't open {path}: {e}", type="warning")


async def pick_path(start=None, mode="file", suffixes=None, title="Choose"):
    """Browse this computer's folders. mode: "file", "dir" or "any". Returns the chosen path or None."""
    cur = Path(start).expanduser() if start else Path.home()
    if cur.is_file():
        cur = cur.parent
    while not cur.exists() and cur != cur.parent:
        cur = cur.parent
    state = {"dir": cur if cur.exists() else Path.home()}

    with ui.dialog() as dialog, ui.card().classes("w-[44rem] max-w-full"):
        ui.label(title).classes("text-lg font-medium")
        with ui.row().classes("items-center gap-1 w-full no-wrap"):
            ui.button(icon="arrow_upward", on_click=lambda: go(state["dir"].parent)).props("flat dense")
            ui.button(icon="home", on_click=lambda: go(Path.home())).props("flat dense")
            if os.name == "nt":
                for d in string.ascii_uppercase:
                    if Path(f"{d}:/").exists():
                        ui.button(f"{d}:", on_click=lambda d=d: go(Path(f"{d}:/"))).props("flat dense")
            loc = ui.input(value=str(state["dir"])).classes("grow").props("dense")
            loc.on("keydown.enter", lambda: go(Path(loc.value)))
        listing = ui.column().classes("w-full h-96 overflow-auto gap-0")
        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=lambda: dialog.submit(None)).props("flat")
            if mode in ("dir", "any"):
                ui.button("Use this folder", on_click=lambda: dialog.submit(str(state["dir"])))

    def go(d):
        d = Path(d)
        if not d.is_dir():
            ui.notify(f"Not a folder: {d}", type="warning")
            return
        state["dir"] = d
        loc.value = str(d)
        listing.clear()
        try:
            entries = sorted(d.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except OSError as e:
            with listing:
                ui.label(str(e)).classes("text-red-700")
            return
        with listing:
            shown = 0
            for x in entries:
                if x.name.startswith("."):
                    continue
                try:
                    is_dir = x.is_dir()
                except OSError:
                    continue
                if not is_dir and (mode == "dir" or (suffixes and not any(x.name.lower().endswith(s) for s in suffixes))):
                    continue
                shown += 1
                if shown > 500:
                    ui.label("(more entries not shown)").classes("text-gray-500")
                    break
                with ui.row().classes("items-center gap-2 cursor-pointer hover:bg-gray-100 w-full px-1 py-0.5") as r:
                    ui.icon("folder" if is_dir else "description").classes(
                        "text-amber-600" if is_dir else "text-gray-500")
                    ui.label(x.name)
                if is_dir:
                    r.on("click", lambda x=x: go(x))
                else:
                    r.on("click", lambda x=x: dialog.submit(str(x)))

    go(state["dir"])
    return await dialog


def path_row(saver, key, label, mode, suffixes=None, on_change=None):
    with ui.row().classes("items-center w-full no-wrap gap-1"):
        inp = ui.input(label, value=saver.get(key, "")).classes("grow")

        def changed(e):
            saver.set(key, e.value.strip())
            if on_change:
                on_change()
        inp.on_value_change(changed)

        async def browse():
            pth = await pick_path(inp.value or None, mode, suffixes, title=label)
            if pth:
                inp.value = pth
        ui.button(icon="folder_open", on_click=browse).props("flat").tooltip("Browse")
    return inp
