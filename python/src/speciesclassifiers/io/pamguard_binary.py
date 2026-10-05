"""Read PAMGuard binary files (.pgdf) into detection tables.

Click and whistle tables have the columns ``delphinid.click_frames`` / ``whistle_frames`` expect:

- clicks:   ``start`` (s, Unix time), ``duration`` (s), ``wave`` (channel 0), ``uid``, ``file``
- whistles: ``start``, ``duration``, ``times`` (s, per slice), ``freqs`` (Hz, per slice), ``slice_s``,
  ``snr_db`` (20 log10 signal/noise), ``amplitude_db``, ``uid``, ``file``

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
    import datetime as _dt
    if not hasattr(_dt, "UTC"):  # pypamguard uses datetime.UTC (Python >= 3.11)
        _dt.UTC = _dt.timezone.utc
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
    """Whistle detector FFT hop and length (samples) from contour slice numbers and durations
    (rough; ``detect_whistle_settings`` is exact when peak data are available)."""
    sn = np.asarray(slice_numbers_first, float)
    ss = np.asarray(start_samples, float)
    ok = sn > 0
    hop = int(round(np.median((ss[ok] + 1) / sn[ok])))
    fft = np.median(np.asarray(sample_durations, float) - (np.asarray(n_slices, float) - 1) * hop)
    fft_len = int(2 ** round(np.log2(fft)))  # FFT lengths are powers of two
    return hop, fft_len


def detect_whistle_settings(contours):
    """(sample rate, hop, FFT length) of the whistle detector, from one file's contours.

    Two quantities are exact: the hop (start sample / first slice number) and the frequency bin width
    (lower frequency limit / lowest peak bin), which fixes sample rate / FFT length. The FFT length is
    the power of two whose sample rate is closest to the rough rate from detection times; the rate
    is accepted if it is within 0.5 % of a standard rate. Returns (None, hop, None, bin width) when
    the rate can't be pinned down (e.g. one contour); the other files' settings are then used.
    """
    hops, fsteps = [], []
    for x in contours:
        s0 = np.asarray(x.slice_numbers)[0]
        if s0 > 0:
            hops.append(x.start_sample / s0)
        if getattr(x, "peak_data", None) is not None and x.freq_limits is not None:
            low = np.concatenate([np.asarray(p)[:, 0] for p in x.peak_data])
            if low.min() > 0:
                fsteps.append(np.asarray(x.freq_limits)[0] / low.min())
    hop = int(round(np.median(hops))) if hops else None
    fstep = float(np.median(fsteps)) if fsteps else None
    rough = infer_sample_rate([x.start_sample for x in contours], [x.millis for x in contours]) \
        if len(contours) > 1 else None
    if fstep is None:
        if hop is None or rough is None:
            return None, hop, None, None
        _, fft = infer_fft([np.asarray(x.slice_numbers)[0] for x in contours], [x.start_sample for x in contours],
                           [x.sample_duration for x in contours], [x.n_slices for x in contours])
        return rough, hop, fft, None
    if rough is None:
        return None, hop, None, fstep
    fft = min((2 ** k for k in range(6, 17)), key=lambda n: abs(np.log(fstep * n / rough)))
    sr = fstep * fft
    nearest = min(STANDARD_RATES, key=lambda r: abs(r - sr))
    if abs(nearest - sr) / nearest < 0.005:
        return float(nearest), hop, fft, fstep
    return None, hop, None, fstep   # not a standard rate: too few contours to tell -> use the other files'



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
    slice numbers times ``fft_hop / sample_rate``. Settings are worked out per file; files with too
    few contours to tell (e.g. a single contour) use the most common settings of the other files.
    Returns (whistles table, settings dict).
    """
    files = list_binaries(folder, pattern, periods)
    loaded = []                                  # (file, contours, (sr, hop, fft) or Nones)
    for k, f in enumerate(files):
        pf = _load(f)
        if not pf.data:
            continue
        d = pf.data
        sr, hop, fl, fstep = detect_whistle_settings(d)
        sr, hop, fl = sample_rate or sr, fft_hop or hop, fft_len or fl
        loaded.append((f, d, sr, hop, fl, fstep))
        if progress and (k + 1) % 50 == 0:
            print(f"  read {k + 1}/{len(files)} files")
    settings = {"sample_rate": sample_rate or _consensus([x[2] for x in loaded], "sample rate"),
                "fft_hop": _consensus([x[3] for x in loaded], "FFT hop"),
                "fft_len": _consensus([x[4] for x in loaded], "FFT length")}
    rows = []
    for f, d, sr, hop, fl, fstep in loaded:
        sr = sr or settings["sample_rate"]
        hop = hop or settings["fft_hop"]
        if not fl:   # rate unknown for this file: FFT length from the consensus rate and its bin width
            fl = int(2 ** round(np.log2(sr / fstep))) if (sr and fstep) else settings["fft_len"]
        if not (sr and hop and fl):
            print(f"Skipping {f.name}: can't work out its detector settings")
            continue
        for x in d:
            sl = np.asarray(x.slice_numbers, float)
            sig = float(x.signal) if getattr(x, "signal", None) is not None else np.nan
            noi = float(x.noise) if getattr(x, "noise", None) is not None else np.nan
            rows.append({"start": x.millis / 1000.0, "duration": x.sample_duration / sr,
                         "times": sl * hop / sr, "freqs": np.asarray(x.contour, float) * sr / fl,
                         "slice_s": hop / sr,
                         "snr_db": 20 * np.log10(sig / noi) if sig > 0 and noi > 0 else np.nan,
                         "amplitude_db": float(x.amplitude) if getattr(x, "amplitude", None) is not None else np.nan,
                         "uid": x.uid, "file": f.name})
    cols = ["start", "duration", "times", "freqs", "slice_s", "snr_db", "amplitude_db", "uid", "file"]
    return pd.DataFrame(rows, columns=cols), settings


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
