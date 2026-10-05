"""A project: one folder holding a classifier's settings, progress and outputs.

    my_project/
        project.json     answers and settings (what to build, from which data, with which parameters)
        status.json      which stages are done (with a fingerprint of what they were made from) and the
                         current run's progress
        run.log          everything the runs printed
        <stage>/         each stage's outputs (e.g. data/, crossval/, model/)

A stage is redone only when its settings, its input files or an earlier stage changed, so a run can be
stopped and resumed, and changing a late setting (e.g. a decision threshold) reruns only what it affects.
"""

import copy
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

CONFIG = "project.json"
STATUS = "status.json"
LOG = "run.log"
STOP = "stop.request"
HEARTBEAT_S = 1.0          # a running job updates status.json this often
STALE_S = 10.0             # a run whose heartbeat is older than this has died


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _read_json(path, default):
    for _ in range(20):                       # the runner may be replacing it just now (Windows)
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return copy.deepcopy(default)
        except (PermissionError, json.JSONDecodeError):
            time.sleep(0.05)
    return copy.deepcopy(default)


def _write_json(path, data):
    tmp = Path(str(path) + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    for _ in range(50):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:               # reader has it open (Windows): retry
            time.sleep(0.05)
    os.replace(tmp, path)


def deep_merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def get_path(cfg, dotted):
    for k in dotted.split("."):
        if not isinstance(cfg, dict) or k not in cfg:
            return None
        cfg = cfg[k]
    return cfg


def _file_stamp(value):
    """(size, mtime) of files named in a setting, so changed input files make a stage outdated."""
    stamps = []
    for v in value if isinstance(value, list) else [value]:
        if isinstance(v, str) and v and len(v) < 1000:
            p = Path(v)
            if p.is_file():
                st = p.stat()
                stamps.append([v, st.st_size, int(st.st_mtime)])
    return stamps


class Project:
    def __init__(self, folder):
        self.folder = Path(folder).expanduser().resolve()

    # ------------------------------------------------------------------ files
    @property
    def config_path(self):
        return self.folder / CONFIG

    @property
    def status_path(self):
        return self.folder / STATUS

    @property
    def log_path(self):
        return self.folder / LOG

    @property
    def stop_path(self):
        return self.folder / STOP

    def exists(self):
        return self.config_path.is_file()

    def stage_dir(self, stage):
        d = self.folder / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------ config
    @classmethod
    def create(cls, folder, classifier, name=None, config=None):
        from .workflows import WORKFLOWS
        if classifier not in WORKFLOWS:
            raise ValueError(f"Unknown classifier type {classifier!r}; choose from {list(WORKFLOWS)}")
        p = cls(folder)
        if p.exists():
            raise FileExistsError(f"{p.folder} already holds a project")
        p.folder.mkdir(parents=True, exist_ok=True)
        cfg = deep_merge(WORKFLOWS[classifier].defaults(), config or {})
        cfg.update({"name": name or p.folder.name, "classifier": classifier, "created": now_iso()})
        p.save_config(cfg)
        return p

    def config(self):
        return _read_json(self.config_path, {})

    def save_config(self, cfg):
        cfg = dict(cfg)
        cfg["modified"] = now_iso()
        _write_json(self.config_path, cfg)

    def update_config(self, changes):
        cfg = deep_merge(self.config(), changes)
        self.save_config(cfg)
        return cfg

    @property
    def workflow(self):
        from .workflows import WORKFLOWS
        return WORKFLOWS[self.config()["classifier"]]

    # ------------------------------------------------------------------ status
    def status(self):
        return _read_json(self.status_path, {"stages": {}, "run": {}})

    def save_status(self, st):
        _write_json(self.status_path, st)

    def fingerprints(self, cfg=None):
        """Fingerprint per stage: its settings, its input files' sizes/times, and the stage before it."""
        cfg = cfg or self.config()
        fps, prev = {}, ""
        for stage in self.workflow.stages:
            up = fps[stage.after] if stage.after else prev
            parts = {k: get_path(cfg, k) for k in stage.keys}
            files = [_file_stamp(v) for v in parts.values()]
            blob = json.dumps([stage.name, stage.version, parts, files, up], sort_keys=True, default=str)
            fps[stage.name] = prev = hashlib.sha1(blob.encode()).hexdigest()[:16]
        return fps

    def stage_states(self, cfg=None):
        """{stage: "done" | "outdated" | "failed" | "pending" | "running" | "skipped"}"""
        cfg = cfg or self.config()
        st = self.status()
        fps = self.fingerprints(cfg)
        run = st.get("run", {})
        running = self.is_running(st)
        out = {}
        for stage in self.workflow.stages:
            s = st["stages"].get(stage.name, {})
            if running and run.get("stage") == stage.name:
                out[stage.name] = "running"
            elif stage.optional and not stage.enabled(cfg):
                out[stage.name] = "skipped"
            elif s.get("state") == "done":
                out[stage.name] = "done" if s.get("fingerprint") == fps[stage.name] else "outdated"
            elif s.get("state") == "failed":
                out[stage.name] = "failed"
            else:
                out[stage.name] = "pending"
        return out

    def is_running(self, st=None):
        run = (st or self.status()).get("run", {})
        return run.get("state") == "running" and time.time() - run.get("heartbeat", 0) < STALE_S

    def request_stop(self):
        self.stop_path.write_text("stop", encoding="utf-8")

    def log_tail(self, n=200):
        if not self.log_path.is_file():
            return []
        with open(self.log_path, encoding="utf-8", errors="replace") as f:
            return f.read()[-50_000:].splitlines()[-n:]
