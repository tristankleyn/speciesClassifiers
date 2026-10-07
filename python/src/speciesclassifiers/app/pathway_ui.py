"""Labels screen of call-type projects: the classification pathway builder (layers of decisions)."""

import copy

from nicegui import ui

NON_LABELS = {"none", "partial", "mixed", "untyped", "unknown", "unlabelled", "n/a", "nan", "", "AMBIGUOUS"}


def _get(tree, path):
    node = tree
    for g in path:
        node = node["children"][g]
    return node


def _classes(node, values):
    if node.get("groups"):
        return list(node["groups"])
    excl = set(node.get("exclude") or [])
    return [v for v in values if v not in excl and v not in NON_LABELS]


def pathway_editor(saver, info):
    """Refreshable editor for ``labels.pathway``. ``info`` holds the label columns and their values."""

    def tree():
        return copy.deepcopy(saver.get("labels.pathway") or None)

    def save(t):
        saver.set("labels.pathway", t)
        editor.refresh()

    def edit(path, fn):
        t = tree()
        fn(_get(t, path))
        save(t)

    def values_of(col):
        return [v for v in (info.get("values", {}).get(col) or []) if v not in NON_LABELS]

    def new_node(column, within=None):
        names = set()
        t = tree()
        if t:
            stack = [t]
            while stack:
                n = stack.pop()
                names.add(n["name"])
                stack += list((n.get("children") or {}).values())
        base = f"{column}_{within}" if within else column
        name, k = base, 2
        while name in names:
            name, k = f"{base}_{k}", k + 1
        return {"name": name, "column": column}

    def add_root():
        cols = info.get("columns") or info.get("other_labels") or []
        if not cols:
            ui.notify("Check the annotations first, so the label columns are known.", type="warning")
            return
        save(new_node(cols[0]))

    def add_child(path, cls):
        def fn(node):
            node.setdefault("children", {})[cls] = new_node(node["column"], cls)
        edit(path, fn)

    def remove(path):
        if not path:
            save(None)
            return
        t = tree()
        parent = _get(t, path[:-1])
        parent["children"].pop(path[-1], None)
        save(t)

    def set_column(path, col):
        def fn(node):
            node["column"] = col
            node.pop("groups", None)
            node.pop("exclude", None)
            node.pop("children", None)
        edit(path, fn)

    def set_mode(path, grouped):
        def fn(node):
            node.pop("children", None)
            if grouped:
                vals = _classes({"exclude": node.get("exclude")}, values_of(node["column"]))
                node["groups"] = {"Group A": vals[: len(vals) // 2 or 1], "Group B": vals[len(vals) // 2 or 1:]}
            else:
                node.pop("groups", None)
        edit(path, fn)

    def rename_group(path, old, new):
        new = new.strip()
        if not new or new == old:
            return

        def fn(node):
            if new in node["groups"]:
                ui.notify(f"There's already a group called {new}", type="warning")
                return
            node["groups"] = {(new if k == old else k): v for k, v in node["groups"].items()}
            if old in (node.get("children") or {}):
                node["children"][new] = node["children"].pop(old)
        edit(path, fn)

    def set_group_values(path, g, vals):
        def fn(node):
            for k in node["groups"]:            # a value belongs to one group only
                if k != g:
                    node["groups"][k] = [v for v in node["groups"][k] if v not in vals]
            node["groups"][g] = list(vals)
        edit(path, fn)

    def add_group(path):
        def fn(node):
            k = len(node["groups"]) + 1
            name = f"Group {chr(64 + k)}" if k <= 26 else f"Group {k}"
            node["groups"][name] = []
        edit(path, fn)

    def remove_group(path, g):
        def fn(node):
            node["groups"].pop(g, None)
            (node.get("children") or {}).pop(g, None)
        edit(path, fn)

    def card(node, path, allowed=None):
        depth = len(path)
        cols = info.get("columns") or info.get("other_labels") or [node["column"]]
        if node["column"] not in cols:
            cols = [node["column"], *cols]
        vals = values_of(node["column"]) or sorted({v for vs in (node.get("groups") or {}).values() for v in vs})
        if allowed is not None and allowed[0] == node["column"]:     # within a group: only its values
            vals = [v for v in vals if v in allowed[1]]
        with ui.card().classes("w-full p-3").props("flat bordered").style(f"margin-left: {depth * 28}px"):
            with ui.row().classes("items-center gap-3 w-full"):
                ui.icon("account_tree" if depth == 0 else "subdirectory_arrow_right").classes("text-gray-500")
                ui.label(f"Layer {depth + 1}" + (f" · within {' > '.join(path)}" if path else "")).classes(
                    "font-medium")
                ui.space()
                ui.button(icon="delete", on_click=lambda: remove(path)).props("flat dense").tooltip(
                    "Remove this layer" + (" and the layers under it" if node.get("children") else ""))
            with ui.row().classes("items-end gap-3 w-full"):
                name = ui.input("Layer name", value=node["name"]).classes("w-48")
                name.on("blur", lambda e, n=name: edit(path, lambda nd: nd.update(name=n.value.strip() or nd["name"])))
                ui.select(cols, value=node["column"], label="Label column",
                          on_change=lambda e: set_column(path, e.value)).classes("w-48")
                ui.toggle({False: "Each value is a class", True: "Groups of values"}, value=bool(node.get("groups")),
                          on_change=lambda e: set_mode(path, e.value)).props("dense no-caps")
            if not node.get("groups"):
                ui.select(vals, value=list(node.get("exclude") or []), multiple=True, label="Leave out values",
                          on_change=lambda e: edit(path, lambda nd: nd.update(exclude=list(e.value)))).props(
                    "use-chips").classes("w-[28rem]")
            else:
                assigned = {v for vs in node["groups"].values() for v in vs}
                for g, gv in node["groups"].items():
                    with ui.row().classes("items-end gap-2 w-full no-wrap"):
                        gi = ui.input("Group", value=g).classes("w-40")
                        gi.on("blur", lambda e, gi=gi, g=g: rename_group(path, g, gi.value))
                        ui.select(vals, value=list(gv), multiple=True, label="Values in this group",
                                  on_change=lambda e, g=g: set_group_values(path, g, e.value)).props(
                            "use-chips").classes("grow")
                        ui.button(icon="close", on_click=lambda g=g: remove_group(path, g)).props("flat dense")
                left = [v for v in vals if v not in assigned]
                with ui.row().classes("items-center gap-2"):
                    ui.button("Add group", icon="add", on_click=lambda: add_group(path)).props("flat dense")
                    if left:
                        ui.label(f"Not in any group (left out): {', '.join(left)}").classes("text-xs text-gray-500")
            classes = _classes(node, vals)
            free = [c for c in classes if c not in (node.get("children") or {})]
            if free:
                with ui.row().classes("items-center gap-1 mt-1"):
                    ui.label("Next layer within:").classes("text-xs text-gray-500")
                    for c in free:
                        ui.button(c, icon="add", on_click=lambda c=c: add_child(path, c)).props(
                            "flat dense no-caps size=sm")
        for c, child in (node.get("children") or {}).items():
            inside = node["groups"].get(c, []) if node.get("groups") else [c]
            if allowed is not None and allowed[0] == node["column"]:
                inside = [v for v in inside if v in allowed[1]]
            card(child, [*path, c], (node["column"], inside))

    @ui.refreshable
    def editor():
        t = tree()
        if not t:
            ui.label("No classification set up yet.").classes("text-gray-600")
            ui.button("Add the first layer", icon="add", on_click=add_root)
            return
        card(t, [])
        ui.label("Each layer is clustered on its own windows and tested separately. New data goes down the "
                 "layers: a block decided as a group (or class) with a next layer is then decided by that layer."
                 ).classes("text-xs text-gray-500")

    editor()
    return editor
