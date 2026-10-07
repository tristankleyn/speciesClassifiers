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
# several detection types: the whole project's training stages run from the bundle
COLAB_URL_TYPES = ("https://colab.research.google.com/github/tristankleyn/speciesClassifiers/blob/main/"
                   "notebooks/python/02b_train_delphinID_types.ipynb")
TYPE_STAGES = ("crossval", "model", "final")


def _is_multi(cfg):
    return (cfg.get("data") or {}).get("source") == "multi"


def notebook_url(cfg):
    return COLAB_URL_TYPES if _is_multi(cfg) else COLAB_URL


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
    if _is_multi(cfg):
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("project.json", json.dumps(cfg, indent=1))
            for f in sorted((p.folder / "frames").glob("*")):
                z.write(f, f"frames/{f.name}")
            z.writestr("bundle.json", json.dumps({
                "project": cfg.get("name"), "created": now_iso(), "multi": True,
                "fingerprints": {k: fps[k] for k in TYPE_STAGES}}, indent=1))
        return path
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(p.folder / "frames" / "frames.csv", "frames.csv")
        z.write(p.folder / "frames" / "params.json", "params.json")
        z.writestr("bundle.json", json.dumps({
            "project": cfg.get("name"), "created": now_iso(), "settings": notebook_settings(cfg),
            "fingerprints": {k: fps[k] for k in ("crossval", "model")}}, indent=1))
    return path


def run_bundle(bundle_zip, workdir="bundle_project"):
    """Several detection types (notebook 02b): run the project's training stages (cross-validation of each
    type, final models, final layer) from a bundle, and zip the results for the app. Returns the zip path."""
    import time
    work = Path(workdir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(bundle_zip) as z:
        z.extractall(work)
    info = json.loads((work / "bundle.json").read_text())
    if not info.get("multi"):
        raise ValueError("This bundle is for one detection type: use notebook 02_train_delphinID.")
    cfg = json.loads((work / "project.json").read_text())
    (work / "status.json").write_text(json.dumps({"stages": {}}))
    p = Project(work)
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        print(f"TensorFlow {tf.__version__}; GPU: {', '.join(g.name for g in gpus) if gpus else 'none (CPU only)'}")
    except ImportError:
        raise RuntimeError("TensorFlow isn't installed here.")
    from .workflows import dl_crossval, dl_final, dl_model
    for name, fn in (("Cross-validation", dl_crossval), ("Final models + PAMGuard zips", dl_model),
                     ("Final layer", dl_final)):
        print(f"--- {name}", flush=True)
        t0 = time.time()
        fn(p, cfg)
        print(f"    ({(time.time() - t0) / 60:.1f} min)", flush=True)
    name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cfg.get("name", "project"))
    out = Path(f"{name}_results.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("results.json", json.dumps({"project": info["project"], "multi": True,
                                               "fingerprints": info["fingerprints"]}))
        for stage in TYPE_STAGES:
            for f in sorted((work / stage).glob("*")):
                if f.is_file():
                    z.write(f, f"{stage}/{f.name}")
    print("Saved", out.resolve())
    return out


def _import_multi(p, z, res):
    names = z.namelist()
    st = p.status()
    st.setdefault("stages", {})
    msgs = []
    for stage in TYPE_STAGES:
        files = [n for n in names if n.startswith(f"{stage}/") and not n.endswith("/")]
        if not files:
            continue
        d = p.stage_dir(stage)
        for old in d.glob("*"):
            if old.is_file():
                old.unlink()
        for n in files:
            (d / Path(n).name).write_bytes(z.read(n))
        st["stages"][stage] = {"state": "done", "fingerprint": res["fingerprints"][stage], "finished": now_iso(),
                               "source": "Google Colab"}
        msgs.append({"crossval": "cross-validation results", "model": "the PAMGuard models",
                     "final": "the final layer"}[stage])
    p.save_status(st)
    fps = p.fingerprints()
    if any(res["fingerprints"].get(k) != fps[k] for k in TYPE_STAGES):
        msgs.append("note: settings changed since the bundle was made, so these show as 'settings changed'")
    return msgs


def import_results(folder, results_zip):
    """Put the results zip from notebook 02 (or 02b) into the project. Returns a list of messages."""
    p = Project(folder)
    msgs = []
    with zipfile.ZipFile(results_zip) as z:
        names = z.namelist()
        if "results.json" not in names:
            raise ValueError("Not a results zip from notebook 02 (no results.json)")
        res = json.loads(z.read("results.json"))
        if res.get("multi"):
            return _import_multi(p, z, res)
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


def copy_notebook(dest, multi=False):
    """Copy notebook 02 (02b for several detection types) next to the bundle (to upload to Colab by hand if
    the GitHub link isn't available)."""
    nb = "02b_train_delphinID_types.ipynb" if multi else "02_train_delphinID.ipynb"
    src = Path(__file__).resolve().parents[4] / "notebooks" / "python" / nb
    if src.is_file():
        shutil.copy(src, dest)
        return Path(dest)
    return None
