"""delphinID: build lightweight CNN classifiers on detection frames and export them to PAMGuard."""

from .crossval import (ResamplingParams, cross_validate, event_predictions, fit_bootstrapped,
                       summarise)
from .frames import AcousticParams, click_frames, whistle_frames
from .model import CNNParams, TrainingParams, build_model, one_hot, predict, train

__all__ = ["AcousticParams", "click_frames", "whistle_frames",
           "CNNParams", "TrainingParams", "build_model", "one_hot", "predict", "train",
           "ResamplingParams", "fit_bootstrapped", "cross_validate", "event_predictions", "summarise"]
