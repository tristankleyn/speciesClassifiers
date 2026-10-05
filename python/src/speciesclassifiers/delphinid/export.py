"""Export delphinID models for PAMGuard's deep learning module.

A PAMGuard delphinID model is a zip holding one folder with:
- a TensorFlow SavedModel (``saved_model.pb`` + ``variables/``),
- ``delphinID.pdtf``: JSON with the class names, input/output shapes, frame length and the
  feature transforms PAMGuard applies before the model,
- ``speciesclassifiers_info.json``: how the model was made (all parameters) and the PAMGuard
  settings it expects. PAMGuard ignores this file.

In PAMGuard: Deep Learning module -> select the zip as the model. Set the segment hop and the
minimum detections (clicks) / density (whistles) to the values listed in the info file.
"""

import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import __version__
from .frames import AcousticParams

PDTF_NAME = "delphinID.pdtf"
INFO_NAME = "speciesclassifiers_info.json"


def _clean(v):
    """JSON-friendly numbers: whole floats as ints (as in the published files)."""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return int(v) if float(v).is_integer() else float(v)
    return v


def make_pdtf(acoustic: AcousticParams, classes, n_inputs, description="delphinID for acoustic deep learning"):
    """The ``delphinID.pdtf`` contents as a dict."""
    return _clean({
        "framework_info": {"framework": "delphinid"},
        "model_info": {"output_shape": [-1, len(classes)], "input_shape": [-1, n_inputs, 1]},
        "class_info": {"name_class": list(map(str, classes)), "num_class": len(classes)},
        "transforms": acoustic.transforms(),
        "description": description,
        "version_info": {"version": 1},
        "seg_size": {"size_ms": acoustic.frame_len * 1000},
    })


def export_pamguard(model, acoustic: AcousticParams, classes, path, sample_rate=None,
                    description="delphinID for acoustic deep learning", info=None):
    """Write ``model`` as a PAMGuard delphinID zip at ``path``.

    ``sample_rate`` (Hz) is required for click models (it sets which FFT bins are kept) and is
    recorded in the info file. ``info`` is any extra dict to store (e.g. CNN/training params,
    cross-validation accuracy). Returns the zip path.
    """
    path = Path(path)
    if path.suffix != ".zip":
        path = path.with_suffix(".zip")
    n_inputs = int(model.input_shape[1])
    expected = acoustic.n_bins(sample_rate) if acoustic.voc_type == "whistle" or sample_rate else None
    if acoustic.voc_type == "click" and sample_rate is None:
        raise ValueError("Click models need sample_rate")
    if expected != n_inputs:
        raise ValueError(f"Model has {n_inputs} inputs but the acoustic parameters give {expected}")
    if int(model.output_shape[-1]) != len(classes):
        raise ValueError(f"Model has {model.output_shape[-1]} outputs but {len(classes)} classes were given")

    pdtf = make_pdtf(acoustic, classes, n_inputs, description)
    meta = _clean({
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "speciesclassifiers_version": __version__,
        "classes": list(map(str, classes)),
        "acoustic_params": acoustic.to_dict(),
        "sample_rate": sample_rate,
        "pamguard_settings": {
            "segment_length_ms": acoustic.frame_len * 1000,
            "segment_hop_ms": acoustic.hop * 1000,
            "min_detection_value": acoustic.min_clicks if acoustic.voc_type == "click" else acoustic.min_density,
            "min_detection_value_means": "minimum clicks per segment" if acoustic.voc_type == "click"
            else "minimum whistle density per segment",
        },
        **(info or {}),
    })

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / path.stem
        model.export(str(folder), verbose=False)
        (folder / PDTF_NAME).write_text(json.dumps(pdtf, indent=1))
        (folder / INFO_NAME).write_text(json.dumps(meta, indent=1, default=str))
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(folder.rglob("*")):
                z.write(f, f.relative_to(tmp))
    return path


def load_pamguard_zip(path):
    """Load a PAMGuard delphinID zip (made here or published) for checking.

    Returns ``(predict_fn, pdtf)``: ``predict_fn(x)`` maps an (n, n_inputs) array to class
    probabilities, using the SavedModel's ``serving_default`` signature as PAMGuard does.
    """
    import tensorflow as tf

    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(path) as z:
        z.extractall(tmp, [n for n in z.namelist() if not n.startswith("__MACOSX")])
    pb = next(p for p in tmp.rglob("saved_model.pb"))
    pdtf = json.loads(next(tmp.rglob("*.pdtf")).read_text())
    sig = tf.saved_model.load(str(pb.parent)).signatures["serving_default"]

    def predict_fn(x):
        x = np.asarray(x, "float32")
        x = x[..., None] if x.ndim == 2 else x
        return list(sig(tf.constant(x)).values())[0].numpy()

    predict_fn.tmpdir = tmp
    return predict_fn, pdtf


def cleanup(predict_fn):
    """Remove the temporary folder made by ``load_pamguard_zip``."""
    shutil.rmtree(getattr(predict_fn, "tmpdir", ""), ignore_errors=True)
