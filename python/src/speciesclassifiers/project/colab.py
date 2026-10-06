"""Train a delphinID project in Google Colab: pack its frames and settings into a bundle for notebook
02_train_delphinID, and bring the notebook's results back into the project.

The bundle records the fingerprints of the training stages, so imported results count as done only while
the project's settings are unchanged (otherwise they show as "settings changed").
"""

import json
import shutil
import zipfile
from pathlib import Path

from .project import Project, now_iso

COLAB_URL = ("https://colab.research.google.com/github/tristankleyn/speciesClassifiers/blob/main/"
             "notebooks/python/02_train_delphinID.ipynb")


def notebook_settings(cfg):
    """Project settings as notebook 02's parameter names."""
    s = cfg["settings"]
    c, t, r = s["cnn"], s["training"], s["resampling"]
    return {
        "CLASSES": cfg["labels"].get("classes") or None, "GROUP_COL": cfg["labels"].get("group_col") or "event_id",
        "MODEL_NAME": cfg.get("model_name") or "delphinID",
        "N_FILTERS": c["n_filters"], "KERNEL_SIZE": c["kernel_size"], "MAX_POOL": c["max_pool"],
        "LEAKY_RELU": c["leaky_relu"], "DENSE_SIZE": c["dense_size"], "DROPOUT": c["dropout"], "L2": c["l2"],
        "LEARNING_RATE": t["learning_rate"], "EPOCHS": t["epochs"], "BATCH_SIZE": t["batch_size"],
        "PATIENCE": t["patience"], "SEED": t["seed"],
        "MAX_PER_GROUP": r["max_per_group"], "N_BOOTSTRAPS": r["n_bootstraps"],
        "RUN_CROSS_VALIDATION": bool(s["crossval"].get("run", True)), "N_FOLDS": s["crossval"].get("n_folds"),
    }


def make_bundle(folder):
    """Zip the project's frames and training settings for Colab. Needs the frames stage done."""
    p = Project(folder)
    if p.stage_states().get("frames") != "done":
        raise RuntimeError("Make the detection frames first (Run > Make frames)")
    cfg = p.config()
    fps = p.fingerprints(cfg)
    out = p.stage_dir("colab")
    name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cfg.get("name", "project"))
    path = out / f"{name}_colab_bundle.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(p.folder / "frames" / "frames.csv", "frames.csv")
        z.write(p.folder / "frames" / "params.json", "params.json")
        z.writestr("bundle.json", json.dumps({
            "project": cfg.get("name"), "created": now_iso(), "settings": notebook_settings(cfg),
            "fingerprints": {k: fps[k] for k in ("crossval", "model")}}, indent=1))
    return path


def import_results(folder, results_zip):
    """Put the results zip from notebook 02 into the project. Returns a list of messages."""
    p = Project(folder)
    msgs = []
    with zipfile.ZipFile(results_zip) as z:
        names = z.namelist()
        if "results.json" not in names:
            raise ValueError("Not a results zip from notebook 02 (no results.json)")
        res = json.loads(z.read("results.json"))
        st = p.status()
        st.setdefault("stages", {})
        fps = p.fingerprints()
        if "cv_predictions.csv" in names:
            d = p.stage_dir("crossval")
            for n in ("cv_predictions.csv", "cv_history.csv", "cv_summary.json"):
                if n in names:
                    (d / n).write_bytes(z.read(n))
            st["stages"]["crossval"] = {"state": "done", "fingerprint": res["fingerprints"]["crossval"],
                                        "finished": now_iso(), "source": "Google Colab"}
            msgs.append("cross-validation results")
        zips = [n for n in names if n.endswith(".zip")]
        if zips:
            d = p.stage_dir("model")
            for old in d.glob("*.zip"):
                old.unlink()
            for n in zips:
                (d / Path(n).name).write_bytes(z.read(n))
            st["stages"]["model"] = {"state": "done", "fingerprint": res["fingerprints"]["model"],
                                     "finished": now_iso(), "source": "Google Colab"}
            msgs.append(f"the PAMGuard model ({', '.join(Path(n).name for n in zips)})")
        p.save_status(st)
    if any(res["fingerprints"].get(k) != fps[k] for k in ("crossval", "model")):
        msgs.append("note: settings changed since the bundle was made, so these show as 'settings changed'")
    return msgs


def copy_notebook(dest):
    """Copy notebook 02 next to the bundle (to upload to Colab by hand if the GitHub link isn't available)."""
    src = Path(__file__).resolve().parents[4] / "notebooks" / "python" / "02_train_delphinID.ipynb"
    if src.is_file():
        shutil.copy(src, dest)
        return Path(dest)
    return None
