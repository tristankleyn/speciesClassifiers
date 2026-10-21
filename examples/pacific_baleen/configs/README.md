# Class configs (simulator input)

One YAML per class, built from `../literature_call_parameters_<species>.csv` (call shapes) and
`../encounter_context_<species>.csv` (encounter behaviour). `src:` gives the CSV row(s) behind each value;
`assumed` marks values no study reported.

## Parameter notation
- `{mean: m, sd: s}` — truncated normal (truncated at `min`/`max` if given, else ±3 SD).
- `{mean: m, range: [a, b]}` — reported range only: SD = (b - a) / 4, truncated at the range.
- `{lognormal_median: m, range: [a, b]}` — lognormal with that median; range ≈ 5th–95th percentile.
- `{uniform: [a, b]}`, `{choice: [..], weights: [..]}` (weights default equal), plain numbers are fixed.
- Times in seconds unless the key ends in `_min` / `_h`; frequencies in Hz.

## Structure
```
class, drift (per call-type parameter: Hz/yr and reference year), sd_inflation (between-encounter, default 1.5)
populations:
  <pop>:
    call_types:            # shape models: tone | sweep | multipart | harmonic | pulsed_band | pulse_series | multitone | modulated
      <name>: {model, params..., levels_db (relative detectability), src}
    contexts:              # behaviours; every context gets the same number of encounters
      <name>:
        vocal_p, group_size, callers        # animals present, chance each is vocal, callers
        sequence: {...}                     # how call types are strung together (song phrase, irregular, series)
        timeline: {...}                     # dive-cycle calling periods / surfacing gaps, spells / silences
        range_km, chorus_p, encounter_len_min
```
Values marked `assumed` are starting guesses to revisit after the pilot run.
