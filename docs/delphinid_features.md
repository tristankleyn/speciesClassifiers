# delphinID features

delphinID models classify **detection frames**: one 1D spectrum summarising the detections in a short window (default 4 s).

| | Whistles | Clicks |
|---|---|---|
| Per frame | Histogram of contour frequencies, 100 Hz bins | Mean click power spectrum (dB, minimum subtracted) |
| Then | downsample → normalise to sum 1 | smooth → trim to frequency range → downsample → normalise to sum 1 |
| Frame filter | whistle density ≥ `min_density` (default 0.05) | clicks ≥ `min_clicks` (default 3) |
| Published model input | 2–20 kHz → 90 bins | 10–40 kHz at 96 kHz, FFT 512 → 80 bins |

Features reproduce PAMGuard's delphinID module, so a model trained on them sees the same inputs when it runs in PAMGuard. They're checked against PAMGuard's test files (`src/test/resources/rawDeepLearningClassifier/DelphinID`).

PAMGuard behaviour reproduced here:
- A detection belongs to a frame if it starts **or** ends inside it.
- Clicks longer than the FFT length are truncated from the start.
- A whistle that starts before a frame but ends inside it contributes its points from the frame start onwards.
- Whistle contour points after the frame end are kept (PAMGuard compares point times in seconds against the frame length in milliseconds).

The original published models were trained on features made with PAMpal, which differ in two of these cases: long clicks were clipped around their peak, and whistles starting before a frame were left out. In PAMGuard's test data this affects 4 of 52 click frames and 13 of 45 whistle frames.
