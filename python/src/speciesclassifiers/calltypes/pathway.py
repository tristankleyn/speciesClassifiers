"""Classification pathways: layers of multi-class decisions, e.g. first a group (A / B / C), then the species
within the group the block was assigned to.

A pathway is a tree of nodes (dicts):

    {"name": "group", "column": "species",
     "groups": {"A": ["Dd", "Sco"], "B": ["pilot", "Gg"]},      # optional: classes are groups of values
     "exclude": ["Gg/Sco"],                                     # optional: values left out
     "min_score": 0.0,                                          # block decision threshold for this layer
     "children": {"A": {"name": "species_A", "column": "species"}}}

Without ``groups`` a node's classes are the column's values (or ``classes``, if listed). A child node only
sees the windows whose label falls in its parent group, and is clustered afresh on them (its own
feature weights and clusters), so each layer is tuned to the distinctions it has to make.

New data: each block is decided by the first layer; a block decided as a group with a child layer is then
decided by that layer from the same windows, and so on. A block's ``decision`` is the deepest confident
decision (e.g. the group when the species layer is uncertain).
"""

import numpy as np
import pandas as pd

from .discover import DiscoverParams, LabelTask, ReliabilityParams, discover
from .multiclass import NON_LABELS


def node_label_col(node):
    return f"__layer_{node['name']}"


def node_classes(node, values=None):
    if node.get("groups"):
        return list(node["groups"])
    if node.get("classes"):
        return [str(c) for c in node["classes"]]
    skip = NON_LABELS | {str(x) for x in node.get("exclude") or ()}
    return sorted(v for v in pd.Series(values).dropna().astype(str).unique() if v not in skip)


def node_labels(w, node, classes=None):
    """The node's label for each window: its group (or value), or "none" if not one of the node's classes."""
    v = w[node["column"]].astype(str)
    excl = {str(x) for x in node.get("exclude") or ()}
    if node.get("groups"):
        m = {str(val): g for g, vals in node["groups"].items() for val in vals}
        out = v.map(m).fillna("none")
    else:
        classes = classes or node_classes(node, v)
        out = v.where(v.isin(classes), "none")
    return out.where(~v.isin(excl), "none")


def walk(tree, path=()):
    """(path, node) for every node, parents first; path is the sequence of group decisions leading to it."""
    yield path, tree
    for g, child in (tree.get("children") or {}).items():
        yield from walk(child, path + (g,))


def check_pathway(tree, values=None):
    """Problems with a pathway definition (list of messages; empty = fine)."""
    msgs, names = [], set()
    for path, node in walk(tree):
        if not node.get("name") or not node.get("column"):
            msgs.append(f"A layer at {' > '.join(path) or 'the top'} needs a name and a label column.")
            continue
        if node["name"] in names:
            msgs.append(f"Layer names must be unique: '{node['name']}' is used twice.")
        names.add(node["name"])
        groups = node.get("groups") or {}
        seen = {}
        for g, vals in groups.items():
            if not vals:
                msgs.append(f"Group '{g}' in layer '{node['name']}' has no values.")
            for v in vals:
                if v in seen:
                    msgs.append(f"'{v}' is in two groups of layer '{node['name']}' ({seen[v]}, {g}).")
                seen[v] = g
        for g in node.get("children") or {}:
            if groups and g not in groups:
                msgs.append(f"Layer '{node['name']}' has a next layer for '{g}', which isn't one of its groups.")
        if groups and len(groups) < 2:
            msgs.append(f"Layer '{node['name']}' needs at least 2 groups.")
    return msgs


def discover_pathway(pool, tree, params: DiscoverParams = None, rp: ReliabilityParams = None, loo=True,
                     _path=()):
    """Run ``discover`` for every layer of a pathway: the first on the whole pool, each child on the windows
    labelled with its parent group. Returns {"node", "path", "result", "children": {group: ...}}."""
    params = params or DiscoverParams()
    rp = rp or ReliabilityParams()
    col = node_label_col(tree)
    w = pool.copy()
    w[col] = node_labels(w, tree)
    task = LabelTask(tree["name"], col, classes=node_classes(tree, w[col][w[col] != "none"]))
    print(f"Layer '{tree['name']}'" + (f" (within {' > '.join(_path)})" if _path else "") +
          f": {len(w)} windows, classes {', '.join(task.classes)}")
    result = discover(w, params, [task], rp, loo=loo)
    out = {"node": {k: v for k, v in tree.items() if k != "children"}, "path": list(_path), "result": result,
           "children": {}}
    for g, child in (tree.get("children") or {}).items():
        sub = pool[w[col].to_numpy() == g].reset_index(drop=True)
        if len(sub) < max(2 * params.min_cluster_size, 10):
            print(f"Layer '{child['name']}': only {len(sub)} windows in group '{g}', skipped")
            continue
        out["children"][g] = discover_pathway(sub, child, params, rp, loo, _path + (g,))
    return out


def iter_results(pr):
    yield pr
    for c in pr["children"].values():
        yield from iter_results(c)


def build_pathway_library(pr, contour_params, window_params, block_minutes=10, rules=None, name=None,
                          min_scores=None):
    """Library for a pathway: the first layer's library, with ``children`` {group: library} for next layers.
    ``min_scores`` {layer name: threshold} overrides each layer's ``min_score``."""
    from .library import build_library
    node = pr["node"]
    task = LabelTask(node["name"], node_label_col(node), classes=pr["result"]["multiclass"][node["name"]]["classes"])
    ms = {node["name"]: (min_scores or {}).get(node["name"], node.get("min_score", 0.0))}
    lib = build_library(pr["result"], contour_params, window_params, [task], block_minutes=block_minutes,
                        rules=rules if not pr["path"] else None, name=name, min_scores=ms)
    lib["layer"] = {"name": node["name"], "column": node["column"], "groups": node.get("groups"),
                    "path": pr["path"]}
    lib["children"] = {g: build_pathway_library(c, contour_params, window_params, block_minutes, None, name,
                                                min_scores) for g, c in pr["children"].items()}
    return lib


def pathway_blocks(windows, contours, lib, period=None):
    """Decision blocks through a pathway library: columns per layer (its decision and score), ``path`` and
    ``decision`` (the deepest confident decision)."""
    from .library import block_votes, classify_windows
    w = classify_windows(windows, lib)
    b = block_votes(w, contours, lib, period)
    layer = lib["layer"]["name"]
    b["decision"] = b[layer]
    b["path"] = b[layer].where(~b[layer].isin(["uncertain", "no_detections"]), "")
    blk = int(round(lib["block_minutes"] * 60 * 1e9))
    wkey = (np.round(windows["w_start"].to_numpy(float) * 1e9).astype(np.int64) // blk) * blk
    bkey = np.round(b["block_start"].to_numpy(float) * 1e9).astype(np.int64)
    for g, child in (lib.get("children") or {}).items():
        sel = bkey[(b[layer] == g).to_numpy()]
        if not len(sel):
            continue
        cb = pathway_blocks(windows[np.isin(wkey, sel)].reset_index(drop=True), contours, child)
        cb = cb.set_index(np.round(cb["block_start"].to_numpy(float) * 1e9).astype(np.int64))
        idx = np.where((b[layer] == g).to_numpy())[0]
        keys = bkey[idx]
        for col in [c for c in cb.columns if c not in b.columns or c in ("decision", "path")]:
            if col in ("block_start", "block_end"):
                continue
            vals = cb[col].reindex(keys).to_numpy()
            if col == "decision":
                ok = ~pd.Series(vals).isin(["uncertain", "no_detections", np.nan, None]).to_numpy()
                b.loc[b.index[idx[ok]], "decision"] = vals[ok]
            elif col == "path":
                ok = pd.Series(vals).fillna("").astype(str).str.len().to_numpy() > 0
                b.loc[b.index[idx[ok]], "path"] = [f"{g} > {v}" for v in vals[ok]]
            else:
                if col not in b.columns:
                    b[col] = None
                b.loc[b.index[idx], col] = vals
    return b


def classify_contours_pathway(whistles, lib, period=None):
    """Whistle table -> (windows, blocks, contours) through a pathway library."""
    from .contours import ContourParams, prepare_contours
    from .library import block_votes
    from .windows import WindowParams, make_windows, window_features
    cp, wp = ContourParams(**lib["contour_params"]), WindowParams(**lib["window_params"])
    contours, points = prepare_contours(whistles, cp)
    if not len(contours):
        empty = pd.DataFrame(columns=["w_start", "w_end", "cluster"])
        return empty, block_votes(empty, pd.DataFrame(columns=["start", "end", "mean_freq_hz", "duration_s"]),
                                  lib, period), contours
    w = window_features(make_windows(contours, wp), contours, points, wp)
    return w, pathway_blocks(w, contours, lib, period), contours


def layer_summaries(pr):
    """One row per layer: where it sits, its classes, windows, clusters and held-out performance."""
    rows = []
    for r in iter_results(pr):
        name = r["node"]["name"]
        res = r["result"]
        m = res["multiclass"][name]
        row = {"layer": name, "within": " > ".join(r["path"]) or "(all)", "classes": ", ".join(m["classes"]),
               "windows": len(res["windows"]), "clusters": res["scorecard"]["n_clusters"]}
        for lvl in ("block", "unit"):
            p = m.get(f"{lvl}_performance")
            if p:
                row[f"{lvl}_accuracy"] = p["accuracy"]
                row[f"{lvl}_balanced_accuracy"] = p["balanced_accuracy"]
        rows.append(row)
    return pd.DataFrame(rows)
