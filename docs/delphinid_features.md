# delphinID features

delphinID models classify **detection frames**: one 1D spectrum summarising the detections in a short window (default 4 s).

| | Whistles | Clicks |
|---|---|---|
| Per frame | Histogram of contour frequencies, 100 Hz bins | Mean click power spectrum (dB, minimum subtracted) |
| Then | downsample → normalise to sum 1 | smooth → trim to frequency range → downsample → normalise to sum 1 |
| Frame filter | whistle density ≥ `min_density` | clicks ≥ `min_clicks` |
| Published model input | 2–20 kHz → 90 bins | 10–40 kHz at 96 kHz, FFT 512 → 80 bins |

A detection belongs to a frame if it starts **or** ends inside it, as in PAMGuard.

## `mode`: `pamguard` vs `legacy`

The code reproduces PAMGuard's delphinID module (`mode="pamguard"`, default) and, separately, the features the original published models were trained on (`mode="legacy"`). Both are checked against PAMGuard's own test files (`src/test/resources/rawDeepLearningClassifier/DelphinID`): `legacy` matches every reference frame to ~1e-8.

They differ in two cases:

| Case | `pamguard` (PAMGuard's Java) | `legacy` (original training, via PAMpal) |
|---|---|---|
| Click longer than the FFT length | first `fft_len` samples | `fft_len` samples centred on the peak |
| Whistle that starts before the frame but ends inside it | its points from the frame start onwards are used | contour is left out |

In the test data this affects 4 of 52 click frames and 13 of 45 whistle frames.

Shared PAMGuard behaviour (reproduced in both modes): whistle contour points after the frame end are kept, because PAMGuard compares point times in seconds against the frame length in milliseconds.
