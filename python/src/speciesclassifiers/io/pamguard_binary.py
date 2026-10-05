"""Read PAMGuard binary files (.pgdf) into detection tables.

Click and whistle tables have the columns ``delphinid.click_frames`` / ``whistle_frames`` expect:

- clicks:   ``start`` (s, Unix time), ``duration`` (s), ``wave`` (channel 0), ``uid``, ``file``
- whistles: ``start``, ``duration``, ``times`` (s, per slice), ``freqs`` (Hz, per slice), ``uid``, ``file``

Binaries don't store the sample rate or the whistle detector's FFT settings, so these are
worked out from the detections in each file (see ``infer_sample_rate``, ``infer_fft``) unless
given. Requires ``pypamguard`` (``pip install pypamguard``).
"""

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FILE_TIME = re.compile(r"_(\d{8})_(\d{6})\.pgdf$")
CLICK_PATTERN = "Click_Detector_Clicks"
WHISTLE_PATTERN = "Contours"


def _pypamguard():
    try:
        import pypamguard
    except ImportError as e:  # pragma: no cover
        raise ImportError("Reading PAMGuard binaries needs pypamguard: pip install pypamguard") from e
    try:  # pypamguard prints INFO/WARNING lines for every file
        from pypamguard.logger import Verbosity, logger
        logger.set_verbosity(Verbosity.ERROR)
    except ImportError:  # pragma: no cover
        pass
    return pypamguard


def file_start_time(path):
    """Start time (Unix s) from a binary file name ``..._YYYYMMDD_HHMMSS.pgdf``, or None."""
    m = FILE_TIME.search(Path(path).name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).timestamp()


def list_binaries(folder, pattern, periods=None):
    """Binary files under ``folder`` whose name contains ``pattern``, sorted by time.

    ``periods``: optional (start, end) pairs in Unix s; only files that may hold detections in a
    period are returned (a file is assumed to run until the next file starts).
    """
    files = sorted((p for p in Path(folder).rglob("*.pgdf") if pattern in p.name),
                   key=lambda p: (file_start_time(p) or 0, p.name))
    if periods is None or not files:
        return files
    starts = np.array([file_start_time(f) or np.nan for f in files])
    ends = np.append(starts[1:], np.inf)
    keep = []
    for f, s, e in zip(files, starts, ends):
        if np.isnan(s) or any(s < pe and e > ps for ps, pe in periods):
            keep.append(f)
    return keep


STANDARD_RATES = [8000, 16000, 22050, 24000, 32000, 44100, 48000, 62500, 64000, 88200, 96000, 128000,
                  144000, 192000, 200000, 250000, 256000, 288000, 384000, 500000, 512000, 576000, 768000]


def infer_sample_rate(start_samples, millis):
    """Sample rate (Hz) from detection start samples against their times.

    Snapped to a standard rate if within 0.5%, otherwise rounded to 1 Hz.
    """
    s = np.asarray(start_samples, float)
    t = np.asarray(millis, float) / 1000.0
    if len(s) < 2 or np.ptp(t) == 0:
        return None
    sr = np.polyfit(t, s, 1)[0]
    nearest = min(STANDARD_RATES, key=lambda r: abs(r - sr))
    return float(nearest if abs(nearest - sr) / nearest < 0.005 else round(sr))


def infer_fft(slice_numbers_first, start_samples, sample_durations, n_slices):
    """Whistle detector FFT hop and length (samples) from contour slice numbers and durations."""
    sn = np.asarray(slice_numbers_first, float)
    ss = np.asarray(start_samples, float)
    ok = sn > 0
    hop = int(round(np.median((ss[ok] + 1) / sn[ok])))
    fft = np.median(np.asarray(sample_durations, float) - (np.asarray(n_slices, float) - 1) * hop)
    fft_len = int(2 ** round(np.log2(fft)))  # FFT lengths are powers of two
    return hop, fft_len


def _load(path):
    return _pypamguard().load_pamguard_binary_file(str(path))


def read_clicks(folder, periods=None, sample_rate=None, pattern=CLICK_PATTERN, progress=True):
    """Read click detections from all click binaries under ``folder``.

    ``periods``: optional (start, end) Unix-s pairs to limit which files are read.
    Returns (clicks table, sample rate).
    """
    rows, rates = [], []
    files = list_binaries(folder, pattern, periods)
    for k, f in enumerate(files):
        pf = _load(f)
        if not pf.data:
            continue
        if sample_rate is None and len(pf.data) > 1:
            sr = infer_sample_rate([d.start_sample for d in pf.data], [d.millis for d in pf.data])
            if sr:
                rates.append(sr)
        for d in pf.data:
            w = np.asarray(d.wave, float)
            rows.append({"start": d.millis / 1000.0, "sample_duration": d.sample_duration,
                         "wave": w[0] if w.ndim > 1 else w, "uid": d.uid, "file": f.name})
        if progress and (k + 1) % 50 == 0:
            print(f"  read {k + 1}/{len(files)} files")
    sr = sample_rate or _consensus(rates, "sample rate")
    df = pd.DataFrame(rows, columns=["start", "sample_duration", "wave", "uid", "file"])
    df.insert(1, "duration", df.pop("sample_duration") / sr if sr else np.nan)
    return df, sr


def read_whistles(folder, periods=None, sample_rate=None, fft_len=None, fft_hop=None,
                  pattern=WHISTLE_PATTERN, progress=True):
    """Read whistle contours from all Whistle and Moan Detector binaries under ``folder``.

    Frequencies are the contour's peak bin times ``sample_rate / fft_len``; slice times are
    slice numbers times ``fft_hop / sample_rate``. Returns (whistles table, settings dict).
    """
    rows, rates, hops, ffts = [], [], [], []
    files = list_binaries(folder, pattern, periods)
    for k, f in enumerate(files):
        pf = _load(f)
        if not pf.data:
            continue
        sr, hop, fl = sample_rate, fft_hop, fft_len
        if sr is None and len(pf.data) > 1:
            sr = infer_sample_rate([d.start_sample for d in pf.data], [d.millis for d in pf.data])
        if hop is None or fl is None:
            h, l = infer_fft([np.asarray(d.slice_numbers)[0] for d in pf.data], [d.start_sample for d in pf.data],
                             [d.sample_duration for d in pf.data], [d.n_slices for d in pf.data])
            hop, fl = hop or h, fl or l
        rates.append(sr), hops.append(hop), ffts.append(fl)
        for d in pf.data:
            sl = np.asarray(d.slice_numbers, float)
            rows.append({"start": d.millis / 1000.0, "duration": d.sample_duration / sr,
                         "times": sl * hop / sr, "freqs": np.asarray(d.contour, float) * sr / fl,
                         "uid": d.uid, "file": f.name})
        if progress and (k + 1) % 50 == 0:
            print(f"  read {k + 1}/{len(files)} files")
    settings = {"sample_rate": _consensus([r for r in rates if r], "sample rate"),
                "fft_hop": _consensus(hops, "FFT hop"), "fft_len": _consensus(ffts, "FFT length")}
    return pd.DataFrame(rows, columns=["start", "duration", "times", "freqs", "uid", "file"]), settings


def _consensus(values, what):
    """Most common value; warn if files disagree (e.g. recordings at different sample rates)."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    vals, counts = np.unique(np.round(values), return_counts=True)
    if len(vals) > 1:
        print(f"Warning: files have different {what}s: {dict(zip(vals.tolist(), counts.tolist()))}. "
              f"Using {vals[counts.argmax()]:g}; set it explicitly or process them separately.")
    return float(vals[counts.argmax()])
