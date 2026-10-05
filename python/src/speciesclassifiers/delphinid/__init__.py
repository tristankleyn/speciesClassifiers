"""delphinID: build lightweight CNN classifiers on detection frames and export them to PAMGuard."""

from .frames import AcousticParams, click_frames, whistle_frames

__all__ = ["AcousticParams", "click_frames", "whistle_frames"]
