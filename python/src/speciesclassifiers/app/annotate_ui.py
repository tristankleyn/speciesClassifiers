"""Cluster and label: the interactive map of windows, the cluster viewer and labelling whole clusters."""

import numpy as np
import pandas as pd
from nicegui import background_tasks, run, ui

from speciesclassifiers.annotate import LabelStore
from speciesclassifiers.annotate.labels import window_ids

from .widgets import ImageSlot, table

# categorical hues in fixed order (validated palette); past 8, the marker shape changes too
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "star", "triangle-down", "x"]
GREY = "#b8b8b8"
UNLABELLED = "(unlabelled)"
_CACHE = {}


def style(i):
    return PALETTE[i % 8], SYMBOLS[(i // 8) % 8]


def store(p):
    return LabelStore(p.folder / "labels" / "labels.json")


# ------------------------------------------------------------------------ data ---------------------
def load_view(p, det):
    """Map (x, y, cluster), windows (features) and, for whistles, contours of one detection type;
    cached until the files change."""
    from speciesclassifiers.project.workflows import _windows_file, an_map_file, load_pickle, read_json
    files = [an_map_file(p, det), _windows_file(p, det)]
    if det == "whistle":
        files.append(p.folder / "contours" / "contours.pkl")
    key = (str(p.folder), det, tuple(f.stat().st_mtime if f.is_file() else 0 for f in files))
    if key not in _CACHE:
        m = load_pickle(files[0])
        w = load_pickle(files[1])
        if len(w) != len(m) or not np.allclose(w["w_start"].to_numpy(float), m["w_start"].to_numpy(float)):
            raise ValueError("The map is older than the windows: run again")
        info = (read_json(p.folder / "cluster" / "info.json", {}) or {}).get(det, {})
        unit = str(m["unit"].iloc[0]) if "unit" in m and len(m) else "window"
        v = {"map": m, "windows": w, "features": info.get("features") or [], "info": info,
             "ids": np.array(window_ids(w)), "unit": unit,
             "word": ({"whistle": "contours", "click": "clicks"}[det] if unit == "detection" else f"{det} windows")}
        if det == "whistle" and files[2].is_file():
            v["contours"] = load_pickle(files[2])
        for k in [k for k in _CACHE if k[:2] == key[:2]]:
            _CACHE.pop(k)
        _CACHE[key] = v
    return _CACHE[key]


def feature_names(det, unit="window"):
    from speciesclassifiers.calltypes.windows import feature_catalog
    names = {}
    if unit == "detection":
        return {i["key"]: i["label"] for i in feature_catalog(f"{det}_detection")}
    for i in feature_catalog(det):
        g = i["group"]
        if det == "click":
            names[i["key"]] = f"{g.lower()}: {i['label']}" if g.startswith("Inter") else \
                i["label"] + (" (spread over clicks)" if "spread" in g else "")
        else:
            names[i["key"]] = i["label"] if g in ("Window", "Contour points") else f"{i['label']} ({g.lower()})"
    return names


def nice(col, names):
    if col in names:
        return names[col]
    for pre, what in (("prof_", "share of contour points at"), ("csp_", "click spectrum band")):
        if col.startswith(pre):
            rest = col[len(pre):]
            try:
                a, b = (float(x) / 1000 for x in rest.split("_"))
                rest = f"{a:g}-{b:g} kHz"
            except ValueError:
                pass
            return f"{what} {rest}"
    return col


def distinctive(w, cols, sel, k=6):
    """The k features whose cluster median is furthest from the overall median (in IQRs)."""
    X = w[cols].apply(pd.to_numeric, errors="coerce")
    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    d = ((X[sel].median() - X.median()) / iqr).abs().dropna()
    return list(d.sort_values(ascending=False).index[:k])


# ------------------------------------------------------------------------ figures ------------------
def map_figure(v, labels, color_by, height=640):
    """Plotly figure (as a dict) of the map: one trace per cluster or label; a cluster's number is written
    at its centre."""
    m = v["map"]
    n = len(m)
    size = 6 if n < 5000 else (4 if n < 30000 else 3)
    t = pd.to_datetime(m["w_start"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
    traces = []
    if color_by == "cluster":
        groups = [(c, f"cluster {c}" if c >= 0 else "unclustered") for c in sorted(m["cluster"].unique())]
        key = m["cluster"].to_numpy()
    else:
        order = sorted(set(labels[labels != ""]))           # colour follows the label, not its count
        groups = [(x, x) for x in order] + [("", UNLABELLED)]
        key = labels
    for i, (g, name) in enumerate(sorted(groups, key=lambda g: (g[0] in (-1, ""), 0))):
        idx = np.where(key == g)[0]
        if not len(idx):
            continue
        unl = g in (-1, "")
        col, sym = (GREY, "circle") if unl else style(i)
        traces.append({
            "type": "scattergl", "mode": "markers", "name": f"{name} ({len(idx)})",
            "x": m["x"].to_numpy()[idx].round(4).tolist(), "y": m["y"].to_numpy()[idx].round(4).tolist(),
            "customdata": idx.tolist(),
            "text": [f"cluster {c}<br>{lab or 'unlabelled'}<br>{tt} UTC" for c, lab, tt in
                     zip(m["cluster"].to_numpy()[idx], labels[idx], t[idx])],
            "hovertemplate": "%{text}<extra></extra>",
            "marker": {"color": col, "symbol": sym, "size": size, "opacity": 0.45 if unl else 0.8,
                       "line": {"width": 0}}})
    cen = m[m["cluster"] >= 0].groupby("cluster")[["x", "y"]].median()
    if len(cen):
        traces.append({"type": "scatter", "mode": "text", "x": cen["x"].round(4).tolist(),
                       "y": cen["y"].round(4).tolist(), "text": [str(c) for c in cen.index],
                       "textfont": {"size": 13, "color": "#222"}, "hoverinfo": "skip", "showlegend": False})
    ax = {"showgrid": False, "zeroline": False, "showticklabels": False, "title": {"text": ""}}
    return {"data": traces,
            "layout": {"height": height, "margin": {"l": 10, "r": 10, "t": 10, "b": 10}, "xaxis": ax,
                       "yaxis": {**ax, "scaleanchor": "x"}, "plot_bgcolor": "white", "dragmode": "zoom",
                       "legend": {"itemsizing": "constant", "font": {"size": 11}}, "hovermode": "closest"},
            "config": {"displaylogo": False, "scrollZoom": True}}


def hist_figure(w, cols, sel, names, colour, word="windows"):
    from matplotlib.figure import Figure
    k = len(cols)
    nc = 3
    nr = max(1, int(np.ceil(k / nc)))
    fig = Figure(figsize=(10, 2.4 * nr))
    axes = fig.subplots(nr, nc, squeeze=False).ravel()
    for ax, c in zip(axes, cols):
        x = pd.to_numeric(w[c], errors="coerce").to_numpy(float)
        ok = np.isfinite(x)
        lo, hi = np.nanquantile(x[ok], [0.005, 0.995]) if ok.any() else (0, 1)
        if hi <= lo:
            lo, hi = lo - 0.5, hi + 0.5
        bins = np.linspace(lo, hi, 40)
        ax.hist(np.clip(x[ok], lo, hi), bins=bins, density=True, color=GREY, alpha=0.7, label=f"all {word}")
        xs = x[sel & ok]
        if len(xs):
            ax.hist(np.clip(xs, lo, hi), bins=bins, density=True, histtype="stepfilled", alpha=0.55, color=colour,
                    label="this cluster")
        unit = "Hz/s" if "hz_s" in c else ("Hz" if "_hz" in c else ("s" if c.endswith("_s") or "_s_" in c else ""))
        ax.set_title(nice(c, names) + (f" ({unit})" if unit and not c.startswith("prof_") else ""), fontsize=9)
        ax.set_yticks([])
        ax.tick_params(labelsize=8)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    for ax in axes[k:]:
        ax.set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def contour_figure(v, idx, colour, seed=0):
    """Whistles: up to 40 of the cluster's points overlaid (single contours from their start; windows: their
    contours at their time in the window), and 4 of them one by one."""
    from matplotlib.figure import Figure
    w, (c, pts) = v["windows"], v["contours"]
    single = v.get("unit") == "detection"
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, min(40, len(idx)), replace=False) if len(idx) else idx
    cs, ce = c["start"].to_numpy(float), c["end"].to_numpy(float)
    fig = Figure(figsize=(10, 6.2))
    gs = fig.add_gridspec(3, 4)
    top = fig.add_subplot(gs[0:2, :])
    small = [fig.add_subplot(gs[2, j]) for j in range(4)]
    L = float((w["w_end"] - w["w_start"]).median()) if not single else 0.0
    fmax = 0
    for n, i in enumerate(pick):
        ws, we = float(w["w_start"].iloc[i]), float(w["w_end"].iloc[i])
        rows = [int(w["cidx"].iloc[i])] if single else np.where((cs < we) & (ce > ws))[0]
        for j in rows:
            t, f, _ = pts[j]
            x = (0.0 if single else cs[j] - ws) + np.asarray(t)
            y = np.asarray(f) / 1000
            fmax = max(fmax, y.max())
            if single:
                L = max(L, x.max())
            top.plot(x, y, color=colour, alpha=0.35, lw=1)
            if n < 4:
                small[n].plot(x, y, color=colour, lw=1.2)
        if n < 4:
            small[n].set_title(pd.Timestamp(ws, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M:%S"), fontsize=8)
    what = "contours" if single else "windows"
    top.set(xlabel="Time from contour start (s)" if single else "Time in window (s)", ylabel="Frequency (kHz)",
            title=f"{len(pick)} of the cluster's {len(idx)} {what}")
    top.title.set_fontsize(10)
    for ax in [top] + small:
        ax.set_xlim(0, L or 1)
        ax.set_ylim(0, fmax * 1.05 if fmax else 1)
        ax.tick_params(labelsize=8)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    for ax in small[len(pick):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def click_figure(v, idx, colour):
    """Clicks: spectrum profiles (cluster vs all; windows: mean over their clicks) and inter-click intervals
    (windows: their median; single clicks: the interval to the previous click)."""
    from matplotlib.figure import Figure
    w = v["windows"]
    word = v.get("word", "windows").split()[-1]
    prof = [c for c in w.columns if str(c).startswith("csp_")]
    fig = Figure(figsize=(10, 3.4))
    a1, a2 = fig.subplots(1, 2)
    if prof:
        P = w[prof].to_numpy(float)
        k = np.arange(1, len(prof) + 1)
        for i in idx[:200]:
            a1.plot(k, P[i], color=colour, alpha=0.08, lw=1)
        a1.plot(k, np.nanmedian(P, axis=0), color="#555", lw=2, label=f"all {word} (median)")
        a1.plot(k, np.nanmedian(P[idx], axis=0), color=colour, lw=2.5, label="this cluster (median)")
        a1.set(xlabel="Spectrum band (low to high frequency)", ylabel="Share of energy")
        a1.legend(fontsize=8, frameon=False)
    else:
        a1.text(0.5, 0.5, "No click spectrum profiles in these data", ha="center", va="center", transform=a1.transAxes)
        a1.set_axis_off()
    ici = "ici_median_s" if "ici_median_s" in w else ("ici_prev_s" if "ici_prev_s" in w else None)
    if ici:
        x = w[ici].to_numpy(float) * 1000
        bins = np.linspace(0, (np.nanquantile(x, 0.99) if np.isfinite(x).any() else 1) or 1, 40)
        a2.hist(x[np.isfinite(x)], bins=bins, density=True, color=GREY, alpha=0.7, label=f"all {word}")
        xi = x[idx]
        if np.isfinite(xi).any():
            a2.hist(xi[np.isfinite(xi)], bins=bins, density=True, color=colour, alpha=0.55, label="this cluster")
        a2.set(xlabel="Median inter-click interval (ms)" if ici == "ici_median_s" else
               "Interval to the previous click (ms)")
        a2.set_yticks([])
        a2.legend(fontsize=8, frameon=False)
    for ax in (a1, a2):
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------ page ---------------------
async def an_results(p, saver):
    from speciesclassifiers.project.workflows import ct_detectors

    from .results import not_ready
    if not_ready(p, "cluster", "map"):
        return
    dets = [d for d in ct_detectors(p.config()) if (p.folder / "cluster" / f"map_{d}.pkl").is_file()]
    if not dets:
        ui.label("No map yet: go to Run.").classes("text-gray-600")
        return
    st = {"det": dets[0], "color": "cluster"}

    with ui.row().classes("w-full items-center gap-4"):
        if len(dets) > 1:
            ui.toggle({d: d.capitalize() + "s" for d in dets}, value=st["det"],
                      on_change=lambda e: (st.update(det=e.value), background_tasks.create(draw()))).props("dense no-caps")
        ui.toggle({"cluster": "Colour by cluster", "label": "Colour by label"}, value="cluster",
                  on_change=lambda e: (st.update(color=e.value), background_tasks.create(draw()))).props("dense no-caps")
        ui.label("Click a point to view and label its cluster. Scroll to zoom; double-click to reset.").classes(
            "text-xs text-gray-500")
    summary = ui.row().classes("w-full items-center gap-2")
    with ui.row().classes("w-full no-wrap items-start gap-4"):
        holder = ui.column().classes("grow min-w-0")
        side = ui.column().classes("w-72 shrink-0 gap-1")
    view = {}

    async def draw():
        view["gen"] = me = view.get("gen", 0) + 1         # a newer draw replaces this one
        holder.clear()
        with holder:
            ui.spinner(size="lg")
        try:
            v = await run.io_bound(load_view, p, st["det"])
        except Exception as e:      # noqa: BLE001
            holder.clear()
            with holder:
                ui.label(str(e)).classes("text-red-700")
            return
        labels = store(p).column(st["det"], v["windows"])
        fig = await run.io_bound(map_figure, v, labels, st["color"])
        if me != view["gen"]:
            return
        view["v"] = v
        holder.clear()
        with holder:
            plot = ui.plotly(fig).classes("w-full")
            plot.on("plotly_click", on_click)
        draw_side(v, labels)

    def draw_side(v, labels):
        summary.clear()
        cnt = pd.Series(labels[labels != ""]).value_counts()
        with summary:
            ui.label(f"{len(labels)} {v['word']} · {v['info'].get('n_clusters', '?')} clusters · "
                     f"{(labels != '').sum()} labelled").classes("text-sm")
            for lab, n in cnt.items():
                ui.chip(f"{lab}: {n}").props("dense outline")
        side.clear()
        m = v["map"]
        with side:
            ui.label("Clusters").classes("text-sm font-medium")
            with ui.scroll_area().classes("h-[560px] w-full border rounded"):
                for i, c in enumerate(sorted(m["cluster"].unique(), key=lambda c: (c < 0, c))):
                    sel = (m["cluster"] == c).to_numpy()
                    lab = pd.Series(labels[sel]).replace("", UNLABELLED).value_counts()
                    col = GREY if c < 0 else style(c)[0]
                    txt = ", ".join(f"{k} {n / sel.sum():.0%}" for k, n in lab.items() if k != UNLABELLED)
                    with ui.row().classes("items-center gap-2 no-wrap w-full px-2 py-1 cursor-pointer "
                                          "hover:bg-slate-50").on("click", lambda c=c: open_cluster(int(c))):
                        ui.element("div").style(f"width:12px;height:12px;border-radius:6px;background:{col}")
                        with ui.column().classes("gap-0"):
                            ui.label(("unclustered" if c < 0 else f"cluster {c}") + f" · {sel.sum()}").classes(
                                "text-sm")
                            ui.label(txt or "unlabelled").classes("text-xs text-gray-500")

    async def on_click(e):
        pts = (e.args or {}).get("points") or []
        if not pts:
            return
        cd = pts[0].get("customdata")
        if cd is None:
            return
        i = int(cd[0] if isinstance(cd, list) else cd)
        await open_cluster(int(view["v"]["map"]["cluster"].iloc[i]))

    async def open_cluster(c):
        v = view["v"]
        det = st["det"]
        m, w = v["map"], v["windows"]
        names, word = feature_names(det, v["unit"]), v["word"]
        sel = (m["cluster"] == c).to_numpy()
        idx = np.where(sel)[0]
        colour = GREY if c < 0 else style(c)[0]
        s = store(p)
        labels = s.column(det, w)
        known = sorted(set(s.counts()))
        cols = [f for f in v["features"] if f in w]
        top = distinctive(w, cols, sel) if cols else []
        name = f"Unclustered {word}" if c < 0 else f"Cluster {c}"
        t = pd.to_datetime(w["w_start"].to_numpy()[idx], unit="s", utc=True)
        with ui.dialog().props("maximized=false") as dlg, ui.card().classes("w-[72rem] max-w-full"):
            with ui.row().classes("w-full items-center"):
                ui.element("div").style(f"width:14px;height:14px;border-radius:7px;background:{colour}")
                ui.label(name).classes("text-xl font-medium")
                ui.space()
                ui.button(icon="close", on_click=dlg.close).props("flat round dense")
            lab = pd.Series(labels[idx]).replace("", UNLABELLED).value_counts()
            ui.label(f"{len(idx)} {word} ({len(idx) / len(w):.1%} of all) · {t.min():%Y-%m-%d %H:%M} to "
                     f"{t.max():%Y-%m-%d %H:%M} UTC on {t.strftime('%Y%m%d').nunique()} days · labels: " +
                     ", ".join(f"{k} {n}" for k, n in lab.items())).classes("text-sm text-gray-600")
            with ui.row().classes("items-center gap-3 w-full"):
                lab_in = ui.select(known, label="Label", with_input=True, new_value_mode="add-unique",
                                   value=None).classes("w-64")
                only_new = ui.checkbox(f"Only {word.split()[-1]} without a label", value=False)

                def apply(label):
                    ids = v["ids"][idx]
                    if only_new.value and label:
                        ids = ids[labels[idx] == ""]
                    st_ = store(p)
                    st_.set(det, ids, label)
                    ui.notify((f"Labelled {len(ids)} {word} '{label}'" if label else
                               f"Removed the labels of {len(ids)} {word}"), type="positive")
                    dlg.close()
                    background_tasks.create(draw())

                ui.button("Label cluster", icon="sell",
                          on_click=lambda: apply(str(lab_in.value).strip()) if lab_in.value else
                          ui.notify("Choose or type a label", type="warning"))
                ui.button("Remove its labels", icon="label_off", on_click=lambda: apply(None)).props("flat")
            with ui.tabs().classes("w-full") as tabs:
                ui.tab("feat", "Features")
                ui.tab("ex", "Contours" if det == "whistle" else "Click spectra")
                ui.tab("win", "Table")
            with ui.tab_panels(tabs, value="feat").classes("w-full"):
                with ui.tab_panel("feat"):
                    ui.label(f"Its {word} (colour) against all {word} (grey). First: the features where it "
                             "differs most.").classes("text-xs text-gray-500")
                    pick = ui.select({c_: nice(c_, names) for c_ in cols}, multiple=True, value=top,
                                     label="Features shown").classes("w-full").props("use-chips")
                    hs = ImageSlot("w-full")
                    pick.on_value_change(lambda e: fill_hist(e.value))
                with ui.tab_panel("ex"):
                    es = ImageSlot("w-full")
                with ui.tab_panel("win"):
                    show = ["w_start"] + top
                    tb = w.iloc[idx[:2000]][show].copy()
                    if len(idx) > 2000:
                        ui.label(f"The first 2000 of {len(idx)} {word}").classes("text-xs text-gray-500")
                    tb.insert(0, "label", labels[idx[:2000]])
                    tb["w_start"] = pd.to_datetime(tb["w_start"], unit="s", utc=True).dt.strftime(
                        "%Y-%m-%d %H:%M:%S")
                    tb = tb.rename(columns={"w_start": "start (UTC)", **{c_: nice(c_, names) for c_ in top}})
                    table(tb.round(3).reset_index(drop=True), rows=10)

        async def fill_hist(chosen):
            hs.img.set_visibility(False)
            if chosen:
                await hs.fill(lambda: hist_figure(w, list(chosen)[:12], sel, names, colour, word.split()[-1]))
        dlg.open()
        await fill_hist(top)
        if det == "whistle" and "contours" in v:
            await es.fill(lambda: contour_figure(v, idx, colour))
        elif det == "click":
            await es.fill(lambda: click_figure(v, idx, colour))

    await draw()
