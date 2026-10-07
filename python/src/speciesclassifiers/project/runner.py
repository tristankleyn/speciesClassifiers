"""Run a project's stages, skipping those already done with the same settings and inputs.

Progress goes to ``status.json`` (current stage, fraction done when the stage reports "k/n", last
message, a heartbeat) and everything printed goes to ``run.log``. A run stops cleanly between heartbeats
when ``stop.request`` appears in the project folder (see ``Project.request_stop``).
"""

import os
import re
import subprocess
import sys
import threading
import time
import traceback

from .project import HEARTBEAT_S, Project, now_iso

PROGRESS = re.compile(r"(\d+)\s*/\s*(\d+)")


class _Tee:
    """File-like: writes to the log and updates the run's message / progress from each line."""

    def __init__(self, log, state, lock, echo=None):
        self.log, self.state, self.lock, self.echo = log, state, lock, echo
        self._buf = ""

    def write(self, s):
        if self.log.closed:
            return len(s)
        self.log.write(s)
        self.log.flush()
        if self.echo:
            self.echo.write(s)
        self._buf += s
        *lines, self._buf = self._buf.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            with self.lock:
                self.state["run"]["message"] = line[:300]
                m = None
                for m in PROGRESS.finditer(line):
                    pass
                if m and 0 < int(m.group(2)) and int(m.group(1)) <= int(m.group(2)):
                    self.state["run"]["progress"] = int(m.group(1)) / int(m.group(2))
        return len(s)

    def flush(self):
        if not self.log.closed:
            self.log.flush()

    def close(self):          # logging handlers (e.g. TensorFlow's) close their stream at exit
        pass

    def isatty(self):
        return False


def _needed(workflow, targets):
    """Stages to consider: the targets and everything they depend on."""
    names = [s.name for s in workflow.stages]
    dep = {}
    for i, s in enumerate(workflow.stages):
        dep[s.name] = s.after or (names[i - 1] if i else None)
    need = set()
    for t in targets:
        while t and t not in need:
            need.add(t)
            t = dep[t]
    return need


def run(folder, stages=None, force=False, echo=False):
    """Run the project in ``folder``: all stages (``stages=None``) or the named ones plus whatever they need.

    ``force`` reruns stages even if they're up to date. Returns the final run state ("finished",
    "failed" or "stopped").
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    p = Project(folder)
    if not p.exists():
        raise FileNotFoundError(f"No project in {p.folder}")
    if p.is_running():
        raise RuntimeError("This project is already running")
    p.stop_path.unlink(missing_ok=True)
    cfg = p.config()
    wf = p.workflow
    fps = p.fingerprints(cfg)
    state = p.status()
    state.setdefault("stages", {})
    state["run"] = {"state": "running", "pid": os.getpid(), "started": now_iso(), "heartbeat": time.time(),
                    "stage": None, "progress": None, "message": "Starting"}
    lock = threading.Lock()
    p.save_status(state)
    done = threading.Event()

    def heartbeat():
        while not done.wait(HEARTBEAT_S):
            with lock:
                state["run"]["heartbeat"] = time.time()
                if p.stop_path.exists():
                    state["run"].update(state="stopped", message="Stopped by user", ended=now_iso())
                    p.save_status(state)
                    p.stop_path.unlink(missing_ok=True)
                    os._exit(1)
                p.save_status(state)

    threading.Thread(target=heartbeat, daemon=True).start()
    targets = stages or [s.name for s in wf.stages]
    need = _needed(wf, targets)
    old_out, old_err = sys.stdout, sys.stderr
    log = open(p.log_path, "a", encoding="utf-8")
    sys.stdout = sys.stderr = _Tee(log, state, lock, old_out if echo else None)
    result = "finished"
    try:
        print(f"\n===== Run started {now_iso()} =====")
        for stage in wf.stages:
            if stage.name not in need:
                continue
            if stage.optional and not stage.enabled(cfg):
                continue
            prev = state["stages"].get(stage.name, {})
            if not force and prev.get("state") == "done" and prev.get("fingerprint") == fps[stage.name]:
                print(f"--- {stage.title}: up to date")
                continue
            with lock:
                state["run"].update(stage=stage.name, progress=None, message=stage.title)
                state["stages"][stage.name] = {"state": "running", "started": now_iso()}
                p.save_status(state)
            print(f"--- {stage.title}")
            t0 = time.time()
            try:
                stage.run(p, cfg)
            except Exception as e:  # record, log, and stop the run
                traceback.print_exc()
                with lock:
                    state["stages"][stage.name] = {"state": "failed", "error": f"{type(e).__name__}: {e}",
                                                   "finished": now_iso()}
                result = "failed"
                break
            with lock:
                state["stages"][stage.name] = {"state": "done", "fingerprint": fps[stage.name],
                                               "finished": now_iso(), "seconds": round(time.time() - t0, 1)}
                p.save_status(state)
        print(f"===== Run {result} {now_iso()} =====")
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        log.close()
        done.set()
        with lock:
            state["run"].update(state=result, ended=now_iso(), progress=None, stage=None,
                                message="Finished" if result == "finished" else state["run"].get("message"))
            p.save_status(state)
    return result


def start_background(folder, stages=None, force=False):
    """Start ``run`` in a separate process that keeps going if the app is closed. Returns the process."""
    cmd = [sys.executable, "-m", "speciesclassifiers", "run", str(folder)]
    if stages:
        cmd += ["--stages", *stages]
    if force:
        cmd.append("--force")
    kw = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        kw["creationflags"] = 0x08000000 | 0x00000200        # no console window, own process group
    else:
        kw["start_new_session"] = True
    return subprocess.Popen(cmd, **kw)
