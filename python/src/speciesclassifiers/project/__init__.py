"""Projects: a folder with a classifier's settings, progress and outputs, run stage by stage (resumable).
Used by the app and the command line (``python -m speciesclassifiers``)."""

from .project import Project
from .runner import run, start_background
from .workflows import WORKFLOWS

__all__ = ["Project", "run", "start_background", "WORKFLOWS"]
