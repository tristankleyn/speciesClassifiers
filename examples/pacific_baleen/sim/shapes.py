"""Call shape models: call-type definition -> clean contours (before detector emulation).

A rendered call is a list of Contour objects, each a single frequency track sampled every
`dt` seconds (as a whistle-and-moan detector reports), with a relative level in dB used
later by the detector emulation (quieter parts are lost first at range).

Models: tone, sweep, modulated, harmonic, pulsed_band, pulse_series, multitone, multipart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .dists import sample

F_MIN = 5.0          # Hz floor for any drawn frequency
F_MAX = 8000.0       # nothing above the highest simulated Nyquist
MIN_SIDEBAND_HZ = 3.0


@dataclass
class Contour:
    t: np.ndarray            # s, relative to call start
    f: np.ndarray            # Hz
    level_db: float = 0.0    # relative to the loudest component of the call
    part: str = ""

    def shifted(self, dt_s=0.0, df_hz=0.0, dlevel=0.0, part=None):
        return Contour(self.t + dt_s, self.f + df_hz, self.level_db + dlevel, part or self.part)


# --------------------------------------------------------------------------- helpers

def _s(spec, rng, default=None):
    v = sample(spec, rng, default)
    return default if v is None else v


def _freq(spec, rng, default=None):
    v = _s(spec, rng, default)
    return None if v is None else float(np.clip(v, F_MIN, F_MAX))


def _dur(ct, rng, default=1.0):
    d = float(_s(ct.get("duration"), rng, default))
    scale = _s(ct.get("duration_scale"), rng, 1.0)
    return max(d * float(scale), 0.05)


def _times(dur, dt):
    n = max(int(round(dur / dt)), 2)
    return np.arange(n) * dt


def _smooth_path(u, knots_u, knots_f):
    """Piecewise smoothstep through knots (no overshoot)."""
    f = np.empty_like(u)
    for i in range(len(knots_u) - 1):
        a, b = knots_u[i], knots_u[i + 1]
        m = (u >= a) & (u <= b) if i == len(knots_u) - 2 else (u >= a) & (u < b)
        x = (u[m] - a) / max(b - a, 1e-9)
        x = x * x * (3 - 2 * x)
        f[m] = knots_f[i] + (knots_f[i + 1] - knots_f[i]) * x
    return f


def _start_end(ct, rng):
    """f_start / f_end with the fallbacks used across the configs."""
    fs = _freq(ct.get("f_start"), rng)
    fe_spec = ct.get("f_end")
    if isinstance(fe_spec, dict) and "ratio_to_start" in fe_spec:
        fe = fs * float(_s(fe_spec["ratio_to_start"], rng, 1.0))
    else:
        fe = _freq(fe_spec, rng)
    if fs is None and "f" in ct:
        fs = _freq(ct["f"], rng)
    if fe is None and "end_ratio" in ct and fs is not None:
        fe = fs * float(_s(ct["end_ratio"], rng, 1.0))
    if fs is None and fe is not None:
        fs = fe
    if fe is None:
        fe = fs
    return fs, fe


def _harmonic_set(ct, rng, f_fund_max):
    """Return [(k, level_db)] for harmonics k >= 2 of a contour."""
    out = []
    h = ct.get("harmonics")
    levels = ct.get("levels_db") or {}
    step = lambda: float(rng.uniform(-12, -4))  # per-harmonic fall-off (assumed)
    ks = []
    if isinstance(h, list):
        ks = [int(k) for k in h if int(k) >= 2]
    elif isinstance(h, dict):
        if "max_hz" in h:
            top = float(_s(h["max_hz"], rng, 0))
            ks = list(range(2, int(top // max(f_fund_max, 1)) + 1))
        elif "n" in h:
            ks = list(range(2, int(_s(h["n"], rng, 1)) + 1))
    if "harmonics_to_hz" in ct:
        top = float(_s(ct["harmonics_to_hz"], rng, 0))
        ks = list(range(2, int(top // max(f_fund_max, 1)) + 1))
    if "harmonic_p" in ct:
        ks = [int(k) for k, p in ct["harmonic_p"].items() if rng.random() < float(p)]
    lvl = 0.0
    for k in sorted(set(ks))[:40]:
        spec = levels.get(k, levels.get(str(k)))
        if spec is not None:
            out.append((k, float(_s(spec, rng, -10.0))))
        else:
            lvl += step()
            out.append((k, lvl))
    return out


def _add_harmonics(contours, ct, rng):
    if not contours:
        return contours
    base = contours[0]
    extra = []
    for k, lv in _harmonic_set(ct, rng, float(base.f.max())):
        f = base.f * k
        m = f <= F_MAX
        if m.sum() >= 2:
            extra.append(Contour(base.t[m], f[m], base.level_db + lv, f"{base.part}h{k}"))
    return contours + extra


# --------------------------------------------------------------------------- models

def m_tone(ct, rng, dt):
    dur = _dur(ct, rng)
    t = _times(dur, dt)
    fs, fe = _start_end(ct, rng)
    f = np.linspace(fs, fe, t.size)
    if "bandwidth_hz" in ct:                       # narrowband tone drawn as a slight downsweep across its width
        bw = float(_s(ct["bandwidth_hz"], rng, 0))
        f = f + np.linspace(bw / 2, -bw / 2, t.size)
    if isinstance(ct.get("f_range"), dict):        # wandering narrow moan
        lo = _freq(ct["f_range"].get("low"), rng, fs * 0.8)
        hi = _freq(ct["f_range"].get("high"), rng, fs * 1.2)
        amp = abs(hi - lo) / 2 * rng.uniform(0.3, 1.0)
        per = rng.uniform(dur / 2, 2 * dur)
        f = f + amp * np.sin(2 * np.pi * t / per + rng.uniform(0, 2 * np.pi))
    c = [Contour(t, np.clip(f, F_MIN, F_MAX), float(_s(ct.get("level_db"), rng, 0.0)), "tone")]
    return _add_harmonics(c, ct, rng)


def m_sweep(ct, rng, dt):
    alt = ct.get("alt")
    if isinstance(alt, dict) and rng.random() < float(alt.get("p", 0)):
        ct = {**ct, **{k: v for k, v in alt.items() if k in ("f_start", "f_end", "duration")}}
    dur = _dur(ct, rng)
    t = _times(dur, dt)
    fs, fe = _start_end(ct, rng)
    c = float(_s(ct.get("curvature"), rng, rng.uniform(-0.3, 0.3)))
    u = t / t[-1]
    f = fs + (fe - fs) * (u + np.clip(c, -1, 1) * u * (1 - u))
    one = _add_harmonics([Contour(t, np.clip(f, F_MIN, F_MAX), 0.0, "sweep")], ct, rng)
    burst = ct.get("burst")
    if isinstance(burst, dict):                     # e.g. Bryde's Be6: 1-4 sweeps ~0.5 s apart
        n = int(_s(burst.get("sweeps"), rng, 1))
        gap = float(_s(burst.get("gap_s"), rng, 0.5))
        out = []
        for i in range(n):
            out += [c_.shifted(dt_s=i * (dur + gap), part=f"{c_.part}{i}") for c_ in one]
        return out
    return one


def m_modulated(ct, rng, dt):
    dur = _dur(ct, rng)
    t = _times(dur, dt)
    if "f_low" in ct and "f_start" not in ct:       # only a band + peak reported
        lo, hi = _freq(ct["f_low"], rng), _freq(ct["f_high"], rng)
        pk = _freq(ct.get("f_peak"), rng, (lo + hi) / 2)
        shape = _s(ct.get("shape"), rng, "flat")
        if shape == "up":
            fs, fe = rng.uniform(lo, pk), rng.uniform(pk, hi)
        elif shape == "down":
            fs, fe = rng.uniform(pk, hi), rng.uniform(lo, pk)
        else:
            fs, fe = pk * rng.uniform(0.9, 1.1), pk * rng.uniform(0.9, 1.1)
    else:
        fs, fe = _start_end(ct, rng)
        pk = _freq(ct.get("f_peak"), rng, (fs + fe) / 2)
    if rng.random() < float(ct.get("reverse_p", 0)):
        fs, fe = fe, fs
    p = float(_s(ct.get("peak_position"), rng, rng.uniform(0.2, 0.8)))
    u = t / t[-1]
    f = _smooth_path(u, [0.0, p, 1.0], [fs, pk, fe])
    fm = ct.get("fm")
    if isinstance(fm, dict):
        rate = float(_s(fm.get("rate_hz"), rng, 2.0))
        depth = float(_s(fm.get("depth_hz"), rng, 0.0))
        f = f + depth * np.sin(2 * np.pi * rate * t + rng.uniform(0, 2 * np.pi))
    keep = np.ones(t.size, bool)
    if "tonal_fraction" in ct:                      # hybrid calls: only the tonal section
        frac = float(_s(ct["tonal_fraction"], rng, 1.0))
        n = max(int(t.size * frac), 2)
        i0 = int(rng.integers(0, t.size - n + 1))
        keep[:] = False
        keep[i0:i0 + n] = True
    if "pulsed_onset_s" in ct:                      # pulsive onset not emitted
        keep &= t >= float(_s(ct["pulsed_onset_s"], rng, 0.0))
    t, f = t[keep], f[keep]
    if t.size < 2:
        return []
    c = [Contour(t - t[0], np.clip(f, F_MIN, F_MAX), 0.0, "mod")]
    return _add_harmonics(c, ct, rng)


def m_harmonic(ct, rng, dt):
    dur = _dur(ct, rng)
    t = _times(dur, dt)
    f0s = _freq(ct.get("f0_start", ct.get("f0")), rng)
    f0e = _freq(ct.get("f0_end"), rng, f0s)
    f = np.linspace(f0s, f0e, t.size)
    ds = ct.get("downstep")
    if isinstance(ds, dict):                        # e.g. NE Pacific blue B2 downstep
        opts = [v for k, v in ds.items() if k not in ("src", "note")]
        pick = opts[int(rng.integers(len(opts)))] if opts else None
        if isinstance(pick, dict):
            at = float(_s(pick.get("at_s"), rng, dur / 2))
            f = f - np.where(t >= at, float(_s(pick.get("size_hz"), rng, 0.5)), 0.0)
    hct = dict(ct)
    if not any(k in ct for k in ("harmonics", "harmonic_p", "harmonics_to_hz")):
        hct["harmonics"] = [2, 3]
    return _add_harmonics([Contour(t, f, float(_s((ct.get("levels_db") or {}).get(1), rng, 0.0)), "f0")], hct, rng)


def m_pulsed_band(ct, rng, dt):
    """Pulsed/AM call: carrier with parallel sidebands at the pulse repetition rate."""
    dur = _dur(ct, rng)
    t = _times(dur, dt)
    if "components" in ct:                          # explicit components (blue A unit)
        out = []
        for i, comp in enumerate(ct["components"]):
            fs, fe = _start_end(comp, rng)
            out.append(Contour(t, np.linspace(fs, fe, t.size), float(_s(comp.get("level_db"), rng, 0.0)), f"c{i}"))
        return out
    car = _freq(ct.get("carrier", ct.get("f_peak")), rng)
    if car is None:
        car = _freq(ct.get("f_start"), rng, 100.0)
    c_end = car
    if "end_ratio" in ct:
        c_end = car * float(_s(ct["end_ratio"], rng, 1.0))
    if "sweep_hz" in ct:
        c_end = car + float(_s(ct["sweep_hz"], rng, 0.0))
    path = np.linspace(car, c_end, t.size)
    cfm = ct.get("carrier_fm")
    if isinstance(cfm, dict):
        depth = float(_s(cfm.get("depth_hz"), rng, 0.0))
        path = path + depth * np.sin(2 * np.pi * t / rng.uniform(dur / 2, 2 * dur) + rng.uniform(0, 2 * np.pi))
    sp = _s(ct.get("sideband_spacing", ct.get("pulse_rate")), rng, 0.0)
    sp = float(sp) if isinstance(sp, (int, float)) else 0.0
    out = [Contour(t, np.clip(path, F_MIN, F_MAX), 0.0, "carrier")]
    if sp >= MIN_SIDEBAND_HZ:
        if "f_low" in ct and "f_high" in ct:        # band filled from f_low to f_high
            lo, hi = _freq(ct["f_low"], rng), _freq(ct["f_high"], rng)
            nmax = int(_s(ct.get("n_sidebands"), rng, 8))   # cap: far sidebands are too weak to matter
            ks = [k for k in range(-nmax, nmax + 1) if k != 0 and lo <= car + k * sp <= hi]
        else:
            n = int(_s(ct.get("n_sidebands"), rng, 2))
            ks = [k for k in range(-n, n + 1) if k != 0]
        for k in ks:
            f = path + k * sp
            if f.min() <= F_MIN:
                continue
            out.append(Contour(t, np.clip(f, F_MIN, F_MAX), -abs(k) * float(rng.uniform(1.5, 4.0)), f"sb{k:+d}"))
    return out


def m_pulse_series(ct, rng, dt, variant=None):
    """Series of short downswept pulses (Antarctic minke bio-duck; humpback creak)."""
    if "variants" in ct:
        vs = ct["variants"]
        key = variant or list(vs)[int(rng.integers(len(vs)))]
        v = vs[key]
        n = int(round(float(_s(v.get("pulses"), rng, 4))))
        reg = ct.get("regional_pulses")
        if reg is not None and rng.random() < 0.5:
            n = int(round(float(_s(reg, rng, n))))
        n = max(n, 1)
        ps = max(float(_s(v.get("pulse_s"), rng, 0.1)), 2 * dt)
        pint = max(float(_s(v.get("pulse_interval_s"), rng, ps + 0.1)), ps + dt)
        step = float(v.get("peak_step_hz", 0.0))
        fs, fe = _freq(v.get("f_start"), rng), _freq(v.get("f_end"), rng)
        pk = _freq(v.get("f_peak"), rng, (fs + fe) / 2)
        hct = {"harmonics_to_hz": ct.get("harmonics_to_hz")} if "harmonics_to_hz" in ct else {}
        out = []
        for i in range(n):
            t = _times(ps, dt)
            f = _smooth_path(t / t[-1], [0, 0.5, 1], [fs + i * step, pk + i * step, fe + i * step])
            out += _add_harmonics([Contour(t + i * pint, np.clip(f, F_MIN, F_MAX), 0.0, f"p{i}")], hct, rng)
        return out
    dur = _dur(ct, rng)                              # creak: pulses along a modulated track (assumed timing)
    track = m_modulated({k: v for k, v in ct.items() if k != "harmonics"}, rng, dt)[0]
    out, t0, i = [], 0.0, 0
    while t0 < dur:
        pl = rng.uniform(0.05, 0.15)
        m = (track.t >= t0) & (track.t < t0 + pl)
        if m.sum() >= 2:
            out.append(Contour(track.t[m], track.f[m], 0.0, f"p{i}"))
        t0 += pl + rng.uniform(0.05, 0.2)
        i += 1
    return out


def _render_sub(spec, rng, dt, parent_dur=None):
    spec = dict(spec)
    if "duration" not in spec and parent_dur is not None:
        spec["duration"] = parent_dur
    model = spec.get("model") or spec.get("kind")
    if model is None:
        model = "sweep" if ("f_start" in spec and "f_end" in spec) else "tone"
    return render({**spec, "model": model}, rng, dt)


def m_multitone(ct, rng, dt):
    out = []
    comps = ct.get("components")
    if isinstance(comps, dict):                      # one component spec (minke star-wars unit A)
        return _render_sub({**comps, "duration": ct.get("duration")}, rng, dt)
    if isinstance(comps, list):
        t_next = 0.0
        pdur = _dur(ct, rng) if "duration" in ct else None
        for i, comp in enumerate(comps):
            if rng.random() > float(comp.get("p", 1.0)):
                continue
            sub = _render_sub({k: v for k, v in comp.items() if k not in ("p", "lag_s")}, rng, dt, pdur)
            if not sub:
                continue
            if "model" in comp:                       # simultaneous components
                start = 0.0
            elif "lag_s" in comp:
                start = float(_s(comp["lag_s"], rng, 0.0))
            else:                                     # alternating tones: sequential
                start = t_next + (rng.uniform(0.0, 0.3) if out else 0.0)
            sub = [c.shifted(dt_s=start, part=f"m{i}{c.part}") for c in sub]
            t_next = max(c.t[-1] for c in sub) + dt
            out += sub
    elif isinstance(ct.get("units"), dict):          # repeated units (humpback feeding call)
        u = ct["units"]
        total = float(_s(ct.get("total_s"), rng, 10.0))
        t0, i = 0.0, 0
        while t0 < total:
            ud = float(_s(u.get("unit_s"), rng, 1.0))
            sub = m_tone({"f_start": u.get("f"), "f_end": u.get("f_end"), "bandwidth_hz": u.get("bandwidth_hz"), "duration": ud}, rng, dt)
            out += [c.shifted(dt_s=t0, part=f"u{i}") for c in sub]
            t0 += ud + float(_s(u.get("gap_s"), rng, 0.2))
            i += 1
    if out and rng.random() < float(ct.get("repeat_p", 0)):   # quieter repetition 3.3-4.3 s later (Bryde's)
        lag = rng.uniform(3.3, 4.3)
        dl = rng.uniform(-10, -3)
        out += [c.shifted(dt_s=lag, dlevel=dl, part=c.part + "r") for c in list(out)]
    return out


def m_multipart(ct, rng, dt):
    segs = ct.get("segments") or ct.get("parts") or []
    var = ct.get("variants")
    if isinstance(var, dict) and "choice" in var and ct.get("segments"):
        v = _s(var, rng, "full")
        segs = segs[: {"A_only": 1, "A_plus_B": 2}.get(v, len(segs))]
    out, t0, prev_end_f = [], 0.0, None
    for i, seg in enumerate(segs):
        seg = dict(seg)
        reps = int(_s(seg.pop("repeat", 1), rng, 1))
        gap = float(_s(seg.pop("gap_s", 0.0), rng, 0.0))
        kind = seg.get("kind") or seg.get("model")
        if kind == "sweep" and "f_start" not in seg and prev_end_f is not None:
            seg["f_start"] = prev_end_f + float(seg.get("start_offset_hz", 0.0))
        for r in range(reps):
            sub = _render_sub(seg, rng, dt)
            if not sub:
                continue
            sub = [c.shifted(dt_s=t0, dlevel=float(_s(seg.get("level_db"), rng, 0.0)), part=f"s{i}{r}{c.part}") for c in sub]
            out += sub
            prev_end_f = float(sub[0].f[-1])
            t0 = max(c.t[-1] for c in sub) + dt + gap
    # join consecutive contiguous segments into one track (a detector sees one continuous contour)
    joined: list[Contour] = []
    for c in out:
        is_h = "h" in c.part[-3:]
        li = next((j for j in range(len(joined) - 1, -1, -1) if joined[j].part.endswith("main")), None)
        last = joined[li] if li is not None else None
        if (last is not None and not is_h and c.level_db >= last.level_db - 6
                and abs(c.t[0] - last.t[-1]) <= 1.5 * dt
                and abs(c.f[0] - last.f[-1]) <= max(0.05 * last.f[-1], 2.0)):
            tt, ff = (c.t[1:], c.f[1:]) if c.t[0] <= last.t[-1] else (c.t, c.f)
            joined[li] = Contour(np.r_[last.t, tt], np.r_[last.f, ff], last.level_db, last.part)
        else:
            joined.append(Contour(c.t, c.f, c.level_db, c.part + ("" if is_h else "main")))
    return joined


MODELS = {
    "tone": m_tone, "sweep": m_sweep, "modulated": m_modulated, "harmonic": m_harmonic,
    "pulsed_band": m_pulsed_band, "pulse_series": m_pulse_series, "multitone": m_multitone,
    "multipart": m_multipart,
}


def render(ct: dict, rng: np.random.Generator, dt: float = 0.02) -> list[Contour]:
    model = ct.get("model")
    if model not in MODELS:
        raise ValueError(f"unknown shape model {model!r}")
    out = MODELS[model](ct, rng, dt)
    return [c for c in out if c.t.size >= 2 and np.all(np.isfinite(c.f))]
