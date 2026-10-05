"""Detection frames: delphinID's model inputs.

A detection frame summarises the detections in a short time window (default 4 s) as one 1D
spectrum. Frames are made per group (``event_id``) so a frame never mixes encounters.

Inputs are plain tables, so detections can come from PAMGuard binaries, databases or elsewhere:

- clicks: one row per click with ``start`` (s), ``duration`` (s), ``wave`` (1D array), ``event_id``
- whistles: one row per contour with ``start`` (s), ``duration`` (s), ``times`` (s, per slice),
  ``freqs`` (Hz, per slice), ``event_id``

Any other columns that are constant within an event (e.g. ``label``, ``location``) are copied to
the frames.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import transforms as T


@dataclass
class AcousticParams:
    """Acoustic parameters for making detection frames.

    voc_type        'click' or 'whistle'
    freq_range      (min, max) frequency in Hz of the model input
    frame_len       frame length in seconds
    hop             step between frame starts in seconds (default: frame_len, i.e. no overlap)
    min_clicks      clicks: minimum clicks in a frame for it to be used
    min_density     whistles: minimum whistle density (contour points / time slices) for a frame
    min_frag_ms     whistles: contours shorter than this (ms) are ignored
    fft_len         clicks: FFT length for click spectra
    smooth_window   clicks: moving-average window (bins) applied to the averaged spectrum
    downsample      factor to reduce the number of bins by averaging neighbours

    The number of model inputs follows from these: see ``n_bins()``.
    """

    voc_type: str = "whistle"
    freq_range: tuple = (2000.0, 20000.0)
    frame_len: float = 4.0
    hop: float = None
    min_clicks: int = 3
    min_density: float = 0.05
    min_frag_ms: float = 200.0
    fft_len: int = 512
    smooth_window: int = 3
    downsample: int = 2

    def __post_init__(self):
        if self.voc_type not in ("click", "whistle"):
            raise ValueError("voc_type must be 'click' or 'whistle'")
        if self.hop is None:
            self.hop = self.frame_len
        self.freq_range = tuple(float(f) for f in self.freq_range)

    @classmethod
    def clicks(cls, **kw):
        """Defaults of the published delphinID click classifier."""
        return cls(**{"voc_type": "click", "freq_range": (10000.0, 40000.0), **kw})

    @classmethod
    def whistles(cls, **kw):
        """Defaults of the published delphinID whistle classifier."""
        return cls(**{"voc_type": "whistle", "freq_range": (2000.0, 20000.0), **kw})

    def transforms(self):
        """Transform list in PAMGuard ``delphinID.pdtf`` format."""
        fmin, fmax = self.freq_range
        if self.voc_type == "whistle":
            steps = [{"name": "whistles2spectrum",
                      "params": {"minfreq": fmin, "maxfreq": fmax, "minfragmillis": self.min_frag_ms}}]
        else:
            steps = [{"name": "clicks2spectrum",
                      "params": {"minclks": self.min_clicks, "spectrum_dB": 1, "fft_len": self.fft_len, "hann": 0}},
                     {"name": "spectrum_smooth", "params": {"window": self.smooth_window}},
                     {"name": "spectrum_trim_freq", "params": {"fmin": fmin, "fmax": fmax}}]
        if self.downsample > 1:
            steps.append({"name": "spectrum_downsample_mean", "params": {"factor": self.downsample}})
        steps.append({"name": "spectrum_normalise_sum", "params": {}})
        return steps

    def n_bins(self, sample_rate=None):
        """Number of model inputs. Clicks need the sample rate."""
        fmin, fmax = self.freq_range
        if self.voc_type == "whistle":
            n = int((fmax - fmin) / T.WHISTLE_BIN_HZ)
        else:
            if sample_rate is None:
                raise ValueError("Clicks need sample_rate to work out the number of bins")
            nfft = self.fft_len // 2
            n = int(fmax / (sample_rate / 2) * nfft) - int(fmin / (sample_rate / 2) * nfft)
        return int(np.ceil(n / self.downsample))

    def to_dict(self):
        return asdict(self)


def _event_info(df, skip):
    """Columns that are constant within the event (e.g. label), to copy onto frames."""
    info = {}
    for c in df.columns:
        if c in skip:
            continue
        try:
            vals = df[c].dropna().unique()
        except TypeError:
            continue
        if len(vals) == 1:
            info[c] = vals[0]
    return info


def _feature_frame(rows, n):
    cols = [f"f{i + 1}" for i in range(n)]
    meta = pd.DataFrame([r[0] for r in rows])
    feats = pd.DataFrame(np.vstack([r[1] for r in rows]) if rows else np.empty((0, n)), columns=cols)
    return pd.concat([meta, feats], axis=1)


def click_frames(clicks: pd.DataFrame, params: AcousticParams, sample_rate: float) -> pd.DataFrame:
    """Make click detection frames.

    Returns one row per frame: ``event_id``, ``frame_start`` (s), ``n_detections``, copied event
    columns, and features ``f1..fN``. Frames with fewer than ``params.min_clicks`` clicks are dropped.
    """
    if params.voc_type != "click":
        raise ValueError("params.voc_type must be 'click'")
    _check_cols(clicks, ["start", "duration", "wave", "event_id"])
    transforms = params.transforms()
    rows = []
    for ev, d in clicks.groupby("event_id", sort=False):
        d = d.sort_values("start")
        info = _event_info(d, {"start", "duration", "wave", "event_id"})
        s = d["start"].to_numpy(float)
        waves = d["wave"].to_list()
        for t, idx in T.segment_starts(s, s + d["duration"].to_numpy(float), params.frame_len, params.hop):
            if len(idx) < max(params.min_clicks, 1):
                continue
            spec = T.clicks2spectrum([waves[i] for i in idx], params.fft_len)
            x = T.apply_transforms(spec, transforms, sample_rate)
            rows.append(({"event_id": ev, "frame_start": t, "n_detections": len(idx), **info}, x))
    return _feature_frame(rows, params.n_bins(sample_rate))


def whistle_frames(contours: pd.DataFrame, params: AcousticParams) -> pd.DataFrame:
    """Make whistle detection frames.

    Returns one row per frame: ``event_id``, ``frame_start`` (s), ``n_detections``, ``density``,
    copied event columns, and features ``f1..fN``. Frames with density below
    ``params.min_density`` are dropped.
    """
    if params.voc_type != "whistle":
        raise ValueError("params.voc_type must be 'whistle'")
    _check_cols(contours, ["start", "duration", "times", "freqs", "event_id"])
    transforms = params.transforms()
    rows = []
    skip = {"start", "duration", "times", "freqs", "event_id"}
    for ev, d in contours.groupby("event_id", sort=False):
        d = d.sort_values("start")
        info = _event_info(d, skip)
        recs = d[["start", "duration", "times", "freqs"]].to_dict("records")
        s = d["start"].to_numpy(float)
        for t, idx in T.segment_starts(s, s + d["duration"].to_numpy(float), params.frame_len, params.hop):
            if len(idx) == 0:
                continue
            seg = [recs[i] for i in idx]
            density = T.whistle_density(seg, t, params.frame_len)
            if not density >= params.min_density:
                continue
            spec = T.whistles2spectrum(seg, t, params.frame_len, params.freq_range, params.min_frag_ms)
            if spec.sum() == 0:
                continue
            x = T.apply_transforms(spec, transforms)
            rows.append(({"event_id": ev, "frame_start": t, "n_detections": len(idx),
                          "density": density, **info}, x))
    return _feature_frame(rows, params.n_bins())


def _check_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
