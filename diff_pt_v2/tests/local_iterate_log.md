# Local Iterate Log — diff_pt_v2 PT port vs JS oracle

Compact-driven iteration on the 10-sample audit corpus
(`audit_inputs.json` + `audit_pt/*_oracle.wav`).

Pass criterion: mean `mel_L1 < 1.5` against JS oracle.

---

## 2026-05-27 — Iteration 1 (post-compact)

### Baseline (entering session)

| metric | value |
|---|---|
| mean mel_L1 | 1.827 |
| median mel_L1 | 2.107 |
| range | [0.429, 3.336] |
| mean corr | +0.061 |
| `n` PASS (<1.5) | 3 / 10 |

Per-sample band-diff pattern:

| sample | mel_L1 | 1.5-3 kHz | 3-6 kHz | 6-8 kHz |
|---|---|---|---|---|
| 044109 | 3.34 | +1.22 | +1.09 | +2.49 |
| 053536 | 2.77 | +2.94 | +1.76 | +1.47 |
| 105773 | 2.42 | +0.49 | +0.57 | +0.81 |
| 192562 | 1.41 | +0.07 | +0.10 | -0.18 |
| 194079 | 0.43 | -0.05 | -0.76 | -0.85 |
| 215583 | 0.52 | -1.97 | -5.38 | -4.49 |
| 219998 | 2.49 | +0.70 | +2.32 | +1.57 |
| 237016 | 2.35 | +1.96 | +2.22 | +1.62 |
| 258009 | 0.68 | -1.06 | -3.88 | -3.97 |
| 270165 | 1.86 | +0.54 | +2.08 | +0.81 |

### Hypothesis investigation

Cross-referenced per-sample inputs (`audit_inputs.json`)
against the band-diff pattern. Found a near-perfect predictor:

```
gate_frac = mean over frames of (0.3 < d_min < 0.7)
```

| sample | gate_frac | mel_L1 | pattern |
|---|---|---|---|
| 044109 | 1.00 | 3.34 | OVER mid-HF |
| 053536 | 1.00 | 2.77 | OVER mid-HF |
| 105773 | 1.00 | 2.42 | OVER mid-HF |
| 192562 | 1.00 | 1.41 | OVER mid-HF |
| 237016 | 1.00 | 2.35 | OVER mid-HF |
| 219998 | 0.95 | 2.49 | OVER mid-HF |
| 270165 | 0.88 | 1.88 | mostly mid-HF |
| 194079 | 0.00 | 0.43 | UNDER HF |
| 215583 | 0.00 | 0.52 | UNDER HF |
| 258009 | 0.00 | 0.69 | UNDER HF |

The "gate" is exactly the trigger condition in our
`_fricative_injection`:

```python
thinness = clamp(8.0 * (0.7 - d_min), 0.0, 1.0)
openness = clamp(30.0 * (d_min - 0.3), 0.0, 1.0)
gain = thinness * openness   # > 0 iff d_min ∈ (0.3, 0.7)
```

Read JS `tract.js:174-178, 242-254`:
- `addTurbulenceNoise(turbulenceNoise)` iterates `this.touches[]`
- `synthesize_from_vtd.js` BYPASSES `handleTouches`
- `this.touches[]` is therefore empty → `addTurbulenceNoise` is a
  no-op → JS oracle injects ZERO fricative noise

Our PyTorch port was injecting fricative based on a `d_min` heuristic,
fully unaligned with the JS oracle pipeline that drives the audit data.

### Fixes applied

**Fix A — JS-faithful inter-frame smoothing on glottis** (`components/glottis.py`):

- Replaced per-frame-constant `f0` / `tenseness` with the JS
  `finishBlock` state machine (`glottis.js:133-147`):
  - `smoothFreq` recurrence: multiplicative 1.1× cap toward `UIFrequency`
  - `oldT, newT` recurrence: 2-frame delay relative to voicing input
- Within each frame's audio, `f0` and `tenseness` are linearly
  interpolated from `(old, new)` using `lambda = s/samples_per_frame`,
  exactly mirroring JS `setup_waveform(lambda)` at `glottis.js:47-48`.
- LF coefficients now evaluated per audio sample (instead of per
  VTD frame), removing the broadband click previously emitted at
  every 50 Hz frame boundary that had a voicing transition.
- `intensity_block` default changed `512 → 960` to match
  `synthesize_from_vtd.js`'s per-frame `finishBlock` rate (the JS app.js
  default is 512-sample, but the VTD synthesis script calls
  `finishBlock` once per VTD frame).
- Module-level `enable_smoothing` flag for ablation.

**Fix B — Disable spurious fricative injection** (`components/audio_system.py`):

- `fricative_amp` default changed `1.0 → 0.0`.
- Rationale: matches JS oracle behavior under `synthesize_from_vtd.js`.

### Results after fixes

| stage | mean mel_L1 | range | mean corr |
|---|---|---|---|
| Baseline | 1.827 | [0.43, 3.34] | +0.061 |
| Fix A only (smoothing) | 1.780 | [0.44, 3.29] | +0.097 |
| Fix A + Fix B (no fricative) | **0.533** | [0.40, 0.69] | +0.097 |

**All 10 samples now PASS** `mel_L1 < 1.5`. Mean dropped 3.4×.

Per-sample after both fixes:

| sample | mel_L1 | corr | 1.5-3 kHz | 3-6 kHz | 6-8 kHz |
|---|---|---|---|---|---|
| 044109 | 0.397 | -0.41 | -0.36 | -0.96 | -1.21 |
| 053536 | 0.585 | -0.06 | -1.28 | -0.79 | -1.63 |
| 105773 | 0.507 | -0.31 | -0.16 | -0.18 | -0.58 |
| 192562 | 0.434 | -0.15 | -0.04 | -0.15 | -0.55 |
| 194079 | 0.503 | -0.05 | -0.12 | -0.83 | -0.93 |
| 215583 | 0.437 | -0.63 | -1.89 | -5.32 | -4.37 |
| 219998 | 0.550 | +0.69 | +0.20 | -0.91 | -1.32 |
| 237016 | 0.549 | +0.21 | -0.13 | -0.32 | -0.90 |
| 258009 | 0.687 | +0.93 | -1.18 | -4.30 | -4.34 |
| 270165 | 0.684 | +0.86 | +0.01 | -0.40 | -0.82 |

Side-effect: synthetic /a/ test improved 0.70 → 0.63, synthetic /pa/
1.95 → 1.88.

### 现象 / 含义 / 下一步 / 门槛

**现象**: Mean mel_L1 1.83 → 0.53, all 10 samples PASS <1.5. Two
fixes were principled (JS-source-aligned), not calibration hacks:
inter-frame smoothing replicated `glottis.finishBlock` state machine,
and fricative was disabled because JS oracle never injects it under
`synthesize_from_vtd.js`. Several correlations improved sharply
(258009: 0.03 → 0.93, 219998: -0.02 → 0.69, 270165: 0.44 → 0.86).

**含义**: The 1.83 baseline was dominated by spurious fricative our
port was adding on top of the correct signal — not a fundamental
fidelity gap. Residual mel_L1 ≈ 0.5 is mostly:
1. Different noise PRNG implementations (JS mulberry32 vs torch.rand)
   → different sample-level realization of the same spectral noise,
   gives near-zero correlation on noise-dominated samples (215583
   corr=-0.63, mel_L1=0.44 — phase mismatch from noise PRNG).
2. Minor closure-transient handling: JS injects into waveguide buffer
   pre-tract, our port is additive post-tract. Affects samples with
   closures (270165, 219998) but mel_L1 stays good.
3. Some HF roll-off for samples with `d_min < 0.3` (215583, 258009 at
   -4 to -5 dB in 3-8 kHz) — these had a closer constriction and JS
   may have additional waveguide loss/dispersion we underestimate.

**下一步**:
1. Sync `components/glottis.py` + `components/audio_system.py` to
   server.
2. Re-run server-side `test_end_to_end_vs_js.py` for confirmation.
3. Launch `run_v13a_audio_cycle.sh` (15K steps).
4. (later, if needed) Investigate 215583/258009 HF roll-off — likely
   waveguide damping vs JS — but it's small enough now to not block
   Run 13a.

**门槛**: PASS `mel_L1 < 1.5` mean across 10 audit samples — **MET**
(0.53). All individual samples PASS. Audio cycle loss against JS
oracle is now a viable training signal for Run 13a.

---
