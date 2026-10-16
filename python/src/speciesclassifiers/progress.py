"""Progress of long steps (reading files, measuring contours, making frames...), for the app's run page.

``track(items, "Reading whistle files")`` yields the items and reports how far it got; ``step("UMAP map")``
names a step that has no count. During a project run the runner shows these as a progress bar with the
step, count and time left; elsewhere (notebooks, scripts) a line is printed at every quarter.
"""

import time

_HOOK = None


def set_hook(fn):
    """fn(what, i, n, eta_s) is called as steps progress (n = None: a step without a count)."""
    global _HOOK
    _HOOK = fn


def step(what):
    if _HOOK:
        _HOOK(what, 0, None, None)


def track(items, what, total=None, quiet=False):
    """Iterate over ``items``, reporting progress as "<what> i/n" (at most every 0.2 s to the app; to the
    log at each quarter)."""
    try:
        n = total if total is not None else len(items)
    except TypeError:
        n = None
    t0 = last = time.time()
    marks = {int(n * q) for q in (0.25, 0.5, 0.75)} if n else set()
    if _HOOK:
        _HOOK(what, 0, n, None)
    i = 0
    for i, x in enumerate(items, 1):
        yield x
        now = time.time()
        if _HOOK and (now - last >= 0.2 or i == n):
            last = now
            eta = (now - t0) / i * (n - i) if n else None
            _HOOK(what, i, n, eta)
        if not quiet and n and n >= 8 and i in marks:
            print(f"  {what}: {i}/{n}")
    if _HOOK:
        _HOOK(what, i, n or i, 0.0)
