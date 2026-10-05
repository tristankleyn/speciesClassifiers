"""Execute notebooks on the example data (a smoke test; used by CI).

    python tools/run_notebooks.py                      # all notebooks that run on examples/
    python tools/run_notebooks.py 03_randomforest      # just these
    python tools/run_notebooks.py --full               # without the QUICK settings below

Notebooks run in place (notebooks/python/), so their output folders appear there (git-ignored).
The executed notebooks themselves are not saved.
"""
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parents[1] / "notebooks" / "python"
# 01 needs PAMGuard binaries, so it isn't run here
DEFAULT = ["02_train_delphinID", "03_randomforest", "04_transferlearning", "05_calltypes"]
# Smaller settings so the smoke test is quick (parameter assignments are replaced before running)
QUICK = {"02_train_delphinID": {"EPOCHS": 2, "N_BOOTSTRAPS": 1, "N_FOLDS": 2}}


def override(nb, values):
    for name, value in values.items():
        pat = re.compile(rf"^({name}\s*=\s*)[^#\n]*?(\s*(#.*)?)$", re.M)
        hits = 0
        for c in nb.cells:
            if c.cell_type == "code":
                c.source, n = pat.subn(lambda m: f"{m.group(1)}{value!r}{m.group(2)}", c.source)
                hits += n
        if not hits:
            raise KeyError(f"parameter {name} not found")


def run(name, quick=True):
    path = HERE / f"{name}.ipynb"
    nb = nbformat.read(path, as_version=4)
    if quick:
        override(nb, QUICK.get(name, {}))
    t = time.time()
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    print(f"{name}: ok ({time.time() - t:.0f} s)", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    quick = "--full" not in args
    names = [a for a in args if not a.startswith("--")] or DEFAULT
    failed = []
    for n in names:
        try:
            run(n, quick)
        except Exception as e:  # report every failing notebook, then fail
            print(f"{n}: FAILED\n{e}", flush=True)
            failed.append(n)
    sys.exit(1 if failed else 0)
