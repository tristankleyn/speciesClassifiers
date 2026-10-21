"""Load class configs and list every call type with refs resolved."""
from __future__ import annotations

from pathlib import Path
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
CLASSES = ["blue", "fin", "right", "humpback", "sei", "gray", "minke", "brydes"]


def load(cls: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / f"{cls}.yaml").read_text())


def lookup(doc: dict, path: str):
    node = doc
    for key in path.split("."):
        node = node[key]
    return node


def resolve(doc: dict, ct: dict, depth=0) -> dict:
    if isinstance(ct, dict) and "ref" in ct and depth < 5:
        base = resolve(doc, lookup(doc, ct["ref"]), depth + 1)
        return {**base, **{k: v for k, v in ct.items() if k != "ref"}}
    return ct


def call_types(cls: str) -> dict[str, dict]:
    """{name: definition} for every shape-bearing call type in a class config."""
    doc = load(cls)
    out = {}
    for name, ct in (doc.get("shared_call_types") or {}).items():
        out[name] = resolve(doc, ct)
    for name, ct in (doc.get("call_types") or {}).items():
        out[name] = resolve(doc, ct)
    for pop, p in (doc.get("populations") or {}).items():
        for name, ct in (p.get("call_types") or {}).items():
            ct = resolve(doc, ct)
            strip = lambda d: {k: v for k, v in d.items() if k not in ("note", "src")}
            if any(strip(ct) == strip(v) for v in out.values()):
                continue                       # pure ref to a shared type
            out[f"{pop}.{name}"] = ct
        if "hf_hz" in p and "hf_component" in doc:   # fin: Southern Hemisphere HF tone, per population
            hf = dict(doc["hf_component"])
            hf.pop("duration", None)
            hf["f"] = p["hf_hz"]
            hf["duration"] = {"mean": 0.7, "sd": 0.3, "range": [0.3, 1.2]}
            out[f"{pop}.hf_component"] = hf
    return {k: v for k, v in out.items() if isinstance(v, dict) and "model" in v}
