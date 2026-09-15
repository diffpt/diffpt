# V34 — K-anchor anti-aliasing ablation 

**Type:** local, no-training experiment. **Date started:** 2026-06-16.
**Script:** `diff_pt_v2/tests/test_v34_kanchor_antialiasing.py`
**Claim under test:** the K-anchor sub-frame IR interpolation is specifically *anti-aliasing*
(it suppresses the 50 Hz frame-rate artefact and its harmonics), not generic smoothing — and its
error scales as O((Δr/K)²).

The artefact is the **amplitude-modulation line at the 50 Hz frame rate**. It is measured exactly as
the rest of the programme measures it (`res_local/analyze_tremor.py`): Hilbert envelope → down-sample
to 1 kHz → periodogram → `env_50Hz_above_floor` = mean[48–52 Hz] − mean[200–400 Hz] floor, in dB.

**Design note (corrected from the chapter wording).** §the original plan says the PSD test "holds the diameter
constant". A *truly* constant diameter makes every per-frame anchor IR identical, so there is no
frame-boundary discontinuity and hence **no** 50 Hz line for any K — a null test. To exercise the
artefact (and the O((Δr/K)²) law) the diameter must change between frames. The PSD test therefore
uses a **constant glottal excitation (120 Hz impulse train) + a controlled linear diameter ramp**
(constant Δ per frame). The chapter's the original plan wording will be updated to match.

---

## The three tests and what each reflects

| Test | Drives | Measures | What a PASS reflects |
|---|---|---|---|
| **1 — 50 Hz line vs K** | constant excitation + linear diameter ramp; sweep K | `env_50Hz_above_floor` vs K; log-log slope of the linear artefact power vs K | K-anchor suppresses the frame-rate artefact monotonically, asymptoting near the natural baseline; slope consistent with the (Δr/K)² error theory |
| **2 — step transient** | abrupt one-frame diameter step; K=1 vs K=5 | audio-rate envelope around the step | K>1 spreads the filter switch across the sub-frame cross-fade window (≈ `samples_per_subframe`) instead of a hard frame-boundary switch |
| **3 — sinusoidal modulation transfer** | diameter modulated sinusoidally at a sweep of rates; K=5 | output AM depth vs modulation rate | K-anchor passes diameter modulation below the sub-frame Nyquist (50·K/2 Hz) and attenuates above it — i.e. it band-limits, the signature of anti-aliasing rather than generic low-pass smoothing |

Setup constants: internal SR 48 kHz, frame rate 50 Hz (`samples_per_frame=960`), F0 120 Hz impulse
train, uniform base tube, velum closed (0.01). Waveguide driven directly (`WaveguideTract`,
`n_subframe`=K) to isolate the K-anchor from glottis/noise/transient.

---

## RESULTS (2026-06-16, `V34_results.json`)

### Headline architectural finding (emerged during the run)
The first attempt — a bare K sweep on a smooth ramp — showed the 50 Hz line **identical (−24.5 dB,
below floor) for every K from 1 to 24**. Reason: **the current `WaveguideTract` cross-fades even at
`n_subframe=1`** (it builds K+1 = 2 anchor IRs at λ=0=prev and λ=1=cur and blends them across the
frame). So `K=1` is *not* the hard-switch that produced the historical +15 dB artefact; that
hard-switch (one cur-r IR per frame, no cross-fade) is a removed code path. A bare K sweep is
therefore close to a null test. The meaningful comparison is **HARD-SWITCH vs CROSS-FADE**, which I
reconstructed via `_compute_reflections → _run_ir_loop → _frame_aligned_convolve`.

### Test 1 — hard-switch vs cross-fade (5 Hz wobble, ±0.7 mm)
| config | 50 Hz above floor |
|---|---|
| hard-switch (no cross-fade) | −10.77 dB |
| K-anchor K=1 | −13.60 dB |
| K-anchor K=2 | −16.81 dB |
| K-anchor K=5 | −16.71 dB |
| K-anchor K=10 | −16.49 dB |

**Observed:** the cross-fade lowers the 50 Hz line by **≈6 dB** vs the hard-switch (−10.8 → −16.7 at K=5).
K=1 gets only half of that (−13.6); K≥2 reaches the full reduction and then plateaus.
**Means:** the K-anchor *does* act as an anti-frame-rate mechanism (the cross-fade is the lever), and
the benefit saturates by K≈2. NB all values are *below* the broadband envelope floor — in this
synthetic impulse-train/uniform-tube setup the artefact is weaker in absolute terms than the
real-speech +15 dB; the **relative** 6 dB is the valid result.

### Test 2 — does K (fineness) matter? fast 15 Hz wobble
| K | 50 Hz above floor |
|---|---|
| 1 | **−6.28 dB**  (worst — even worse than hard-switch) |
| 2 | −23.80 dB |
| 3 | −19.48 dB |
| 5 | −22.09 dB |
| 8 | −22.61 dB |
| 16 | −23.19 dB |
| (hard-switch ref) | −13.97 dB |

**Observed:** on fast dynamics K **does** matter, but not the way the original plan predicts. **K=1 is the worst**
(−6.3 dB, ~8 dB worse than K=5, and even worse than the hard-switch): a single full-frame linear
segment lags and itself modulates at the frame rate. **K≥2 captures the benefit and plateaus**
(−19 to −23 dB, ±~4 dB measurement wiggle), so the default **K=5 sits safely on the plateau**.
**Means:** the operative justification for K=5 is "K=1 is insufficient, K≥2 suffices, K=5 is safe and
cheap" — **not** the monotonic decrease / slope −2 / asymptote-at-K≈20 story in the original plan.

### Test 3 — step transient (hard-switch vs K=5)
hard-switch peak |d(env)/dt| = 0.104; K=5 = 0.108; ratio 0.97 (≈ 1).
**Observed:** inconclusive — this envelope-slope metric does not distinguish the two (it is dominated by
the impulse-train excitation near the step, not the filter switch). **Means:** the step test as
designed is not diagnostic; needs a different observable (e.g. the per-sample effective-IR
trajectory) or should be dropped.

---

## Next steps
1. **the original plan needs rewriting** — its three predictions (monotonic 50 Hz decrease with K, log-log
   slope −2, asymptote near K≈20) are **not** what the code does. The defensible, evidence-backed
   claims are: (a) the cross-fade suppresses the frame-rate line by ~6 dB vs a hard-switch;
   (b) K=1 is insufficient, K≥2 suffices, K=5 is the safe cheap default. Flag to user before editing
   (it changes a thesis claim).
2. Optionally re-run with a real glottal source + a vowel tract (not uniform tube) so the absolute
   50 Hz level is comparable to the real-speech +15 dB, not below-floor.
3. Redesign or drop Test 3 (step) — current metric non-diagnostic.

## Decision gate
The K-anchor's value as a frame-rate-artefact fix is **confirmed in relative terms** (cross-fade −6 dB
vs hard-switch; K≥2 needed). The chapter's specific K-scaling claims are **falsified for the current
code** and must be replaced. No code change to the waveguide is warranted — K=5 is justified.

---

## Realistic re-run (real LF glottal source + real ORAL vowel tract, 2026-06-16)

Script `test_v34_realistic.py`, raw `V34_realistic_results.json`. Excitation = the project's
`GlottalSourceGenerator` (LF pulse + aspiration, via `DifferentiableAudioSystem`); base tract = a real
steady-vowel VTD from `v1/py/data_phase4_s2/worker0_batch0000.npz` (velum forced to 0.01 = oral; the
loaded sample was nasalised). Replaces the synthetic impulse-train + uniform tube.

### Test 1 — hard-switch vs cross-fade, two dynamics regimes
| trajectory | hard-switch | K=1 | K=2 | K=5 | K=10 | hard − K=5 |
|---|---|---|---|---|---|---|
| smooth 5 Hz wobble | −21.22 | −21.98 | −21.96 | −21.98 | −21.98 | **+0.76 dB** |
| frame-rate jitter (model-like) | −11.73 | −11.48 | −17.09 | −17.17 | −17.13 | **+5.44 dB** |

**Observed:** the artefact is governed by **how jittery the VTD is frame-to-frame**. Smooth motion → the
cross-fade barely matters (+0.8 dB). **Per-frame jitter** (i.i.d. perturbation on mid-oral dims,
mimicking the model's noisy predictions) → the cross-fade suppresses the 50 Hz line by **+5.4 dB**,
and **K=1 is insufficient** (−11.48, ≈ hard-switch) while **K≥2 captures the benefit** (−17), K=5 safe.
**Means:** matches the programme's original diagnosis exactly — the +15 dB tremor came from the model
predicting *frame-rate-noisy* VTDs (which `L_temp` later smoothed); the K-anchor's role is to stop that
jitter aliasing into a 50 Hz line. Smooth articulation never needed it.

### Test 2 — fast 15 Hz *smooth* wobble: K spread 0.45 dB (hard-switch −20.05, K-anchor ≈ −20.7).
A smooth fast modulation is still well-interpolated → K barely matters. Confirms the lever is
**jitter**, not modulation rate.

### Test 3 — step: ratio 0.84, again non-diagnostic (metric dominated by excitation).

### Reconciliation
Synthetic and realistic runs agree: **cross-fade vs hard-switch ≈ +5–6 dB under high frame-to-frame
variation; K=1 insufficient; K≥2 suffices; K=5 safe.** Absolute levels sit below the broadband floor in
these controlled tests — the operational +15 dB arose from the full model-predicted trajectory on a
specific real slow-speech sample through the whole pipeline; here we reproduce the **mechanism and the
relative ~5 dB**, not the exact magnitude. No waveguide code change warranted (K=5 justified).

---

## DSP audit for the thesis review (2026-06-16)

Triggered by a chapter-review question (are the differentiability approximations sound?).
Code read of `waveguide.py` + `glottis.py` settled three items; two needed measurement
(`verify_dsp_audit.py` → `verify_dsp_audit_results.json`, 12 s CPU).

**Settled by code read (no measurement):**
- **Convolution is LINEAR (overlap-save), not circular.** `forward()` front-pads `n_ir−1`
  zeros, `unfold`s overlapping windows (`spf_sub+n_ir−1`, stride `spf_sub`), valid `conv1d`
  → keep `spf_sub`. No circular wrap. (The OLA `_frame_aligned_convolve` is the old K=1 path.)
- **Cross-fade = coefficient-interp at K+1 anchors** (each `r_k=r_prev(1−λ)+r_cur·λ` → a
  physical intermediate IR) **+ output-domain blend within a sub-frame** (geometry step Δr/K).
- **LF source is phase-continuous** (`phase=inc.cumsum(); t=phase.fmod(1)` — no period-reset
  branch); f0/tenseness lerped to audio rate. Bonus: a *second* 50 Hz source — per-frame-constant
  glottal clicks — was already fixed here by that interpolation (glottis.py docstring).
- **Frame-0 prev=cur** (degenerate, no spurious transition); IR carries the true tract
  propagation delay → causal source–filter alignment.

**Test A — 250 Hz (=K·50) line** (envelope-spectrum dB above non-50-multiple floor):
| traj | scheme | 50 | 100 | 150 | 200 | 250 | 300 | 350 |
|---|---|---|---|---|---|---|---|---|
| wobble10 | hard | 15.7 | 21.1 | 14.2 | 6.7 | 17.9 | −0.1 | 22.9 |
| wobble10 | K=5  | 9.4 | 19.5 | 10.6 | 4.7 | 15.3 | 0.8 | 22.6 |
| jitter | hard | 4.9 | 13.3 | 9.7 | 2.8 | 13.8 | 4.2 | 15.1 |
| jitter | K=5  | 5.1 | 10.1 | 6.9 | 4.2 | 16.1 | 2.0 | 15.6 |

*Observed:* no dominant new 250 Hz line — K=5 *lowers* the low-freq frame-rate lines (50/100/150)
on smooth wobble; at 250 Hz it is at most +2.3 dB vs hard under heavy jitter and *lower* on smooth.
(Caveat: per-harmonic envelope numbers are confounded by the F0=120 excitation; robust read =
"no prominent K·50 line.") *Means:* the C0-but-not-C1 sub-frame discontinuity is benign — its
energy ∝ Δr/K and sits at 250 Hz, above the slow-tremor band. The cross-fade trades the dominant
50 Hz hard-switch line for, at worst, a marginal high-freq ripple. K=5 validated.

**Test B — 512-tap IR truncation vs a 4096-tap reference (the real finding):**
| geometry | tail energy >512 | 99.9% energy @ | LSD(512 vs 4096) |
|---|---|---|---|
| open_vowel | 0.003% | 323 taps | 0.033 dB |
| alveolar_constriction 0.05 | 0.024% | 409 taps | 0.155 dB |
| near_closure 0.02 (interior) | 0.052% | 465 taps | 0.312 dB |
| **lip_near_closed 0.05** | **25.8%** | **1988 taps (41 ms)** | **1.454 dB** |

*Observed:* 512 taps (10.7 ms) captures ≥99.9% of IR energy for open vowels and **interior**
constrictions (LSD < 0.32 dB, negligible). A sustained **LIP** near-closure (bilabial config,
oral segment 44 = 0.05) rings to ~2000 taps: **26% of IR energy is beyond tap 512, LSD = 1.45 dB.**
*Means:* sealing the radiating END makes a high-Q resonator; an interior constriction still loses
energy out the open lip and decays fast. So 512-tap truncation is negligible everywhere **except a
held bilabial closure** — which is itself a near-silent / low-radiation regime, and closures are
brief, so practical impact is bounded. Still a real, documentable approximation error at bilabial
configs. *Next:* optional — `n_ir_samples ≥ 2048` if bilabial-closure fidelity matters (≈4× cost),
else accept as a documented limitation. The chapter's "512 ≈ longer than a frame's ringing" wording
should be softened to "adequate except for a sustained lip near-closure." *Gate:* no code change
forced; 512 stays default. Flagged for the chapter as an honest bound + a possible config knob.
