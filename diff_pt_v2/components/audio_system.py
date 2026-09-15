"""End-to-end Differentiable Pink Trombone audio system.

Composes the previously-built sub-components into a single
`nn.Module` whose `forward` takes Pink Trombone control inputs
(VTD, voicing, F0) and emits a fully-differentiable 16 kHz waveform:

    vtd      (B, T, 1 + n_oral)         per-frame velum + oral diameters
    voicing  (B, T) ∈ [0, 1]            per-frame voicing scalar
    f0       (B, T) Hz                  per-frame fundamental frequency
            │
            ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  1. NoiseFilterBank   → (aspirate_48k, fricative_48k) (B, T_a) │
   │  2. GlottalSource     → glottal_48k (B, T_a)  [LF + aspiration] │
   │  3. WaveguideTract    → audio_48k (B, T_a)                      │
   │     (sample-level continuous waveguide: glottal injected at     │
   │      R[0], lambda-interpolated reflections across frame         │
   │      boundaries, JS-PT faithful per tract.js:133-203)           │
   │  4. Fricative inject  additive: thinness*openness*fricative     │
   │  5. Closure transients additive bursts at release sample idx    │
   │  6. Downsample 48k → 16k via torchaudio Resample                │
   │  7. Optional valid_frames masking                               │
   └─────────────────────────────────────────────────────────────────┘
            │
            ▼
        audio_16k (B, T_audio_out)  where T_audio_out = T * (sr_out/frame_rate)

Architecture is a sample-level continuous waveguide, mirroring JS PT's
per-sample tract.runStep(glottal, fric, lambda) loop. The JS reference
is `v1/js/src/models/tract.js` (calc_junctions/runStep at lines
133-203) and `v1/js/src/synthesize_from_vtd.js:181-194` (sample loop
with 2× oversample). The glottal source is injected directly at R[0]
of the waveguide each audio sample; reflection coefficients are
linearly interpolated between prev-frame and cur-frame values across
each frame (matching JS PT's lambda semantics).

History (this block was stale until 2026-08-05 -- it described a
sample-level continuous waveguide that is NOT what runs today):
  * v1  per-frame IR, K=1, hard switch at the frame boundary. Produced a
        +10 dB 50 Hz line from inter-frame IR discontinuity.
  * v2  sample-level continuous waveguide. Correct but sequential:
        96,000 dependent steps per second of audio.
  * NOW K-anchor frozen-reflection IRs (K = n_subframe = 5, so 6 anchors
        per frame at lambda = 0, 1/5, ..., 1) + overlap-add convolution,
        with an audio-rate linear cross-fade between consecutive anchors.
        The anchors are shared across the frame boundary by construction
        (IR_{f,K} == IR_{f+1,0}), so the effective IR is continuous in t
        and the 50 Hz artifact does not return. Cost is 512 sequential
        steps per anchor, independent of utterance length -- the reason
        the cycle loss is trainable at all. See waveguide.py.

Fricative injection: JS injects turbulence into the waveguide buffers at
the constriction (tract.js:242-270). Since v1 (2026-08) this port does
the same rather than adding post-tract: the injection is carried through
the SAME IR pass by superposition, sum_i g_i (s * h_i) = s * IR(inj=g),
so the per-segment spatial location is preserved, not lost. Verified by
tools/test_superposition.py. Set fricative_amp=0.0 for the old silent
behaviour.

Closure-release transients: detected by `detect_release_events` on
per-frame `d_min` (min oral diameter). With burst_mode="tract" (the
training default) the pulse is injected inside the waveguide at the
obstructed segment, as JS does; "post" is the legacy additive path and
"off" disables it. Measured against tools/oracle_nolag.js the in-tract
path is within -0.00 dB of JS on unwindowed burst rms.

Refs:
    design/01_js_audit.md §B,§C,§D,§E
    design/02_port_strategy.md
    v1/js/src/models/audioSystem.js
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as TT

from diff_pt_v2.components.glottis import GlottalSourceGenerator
from diff_pt_v2.components.noise import NoiseFilterBank
from diff_pt_v2.components.transient import (
    closure_transient,
    detect_release_events,
)
from diff_pt_v2.components.waveguide import WaveguideTract


class DifferentiableAudioSystem(nn.Module):
    """Composable, end-to-end differentiable Pink Trombone audio system.

    Args:
        sr_internal: internal synthesis sample rate. JS uses 48000
            (audioSystem.js:38). Must be divisible by `frame_rate`.
        sr_out: output (downsampled) sample rate. Our training data
            and audio-cycle loss target operate at 16000 Hz.
        frame_rate: VTD/voicing/F0 input frame rate (frames per second).
            Default 50 → samples_per_frame = 960 at 48 kHz.
        n_oral: oral tract segments (JS default 44).
        n_nose: nose segments. JS Pink Trombone uses
            noseLength = floor(28 * n / 44); with n_oral = 44 this is 28.
            (Earlier ports used 18 from a porting-time miscount; 28 is the
            value the JS corpus was generated with.)
            backward compatibility with older callers; the sample-level
            continuous waveguide no longer produces an IR.
        fricative_freq: bandpass centre frequency for the turbulence
            noise source (Hz). JS uses 1000 (audioSystem.js:63).
        aspirate_freq: bandpass centre frequency for the aspiration
            noise source (Hz). JS uses 500 (audioSystem.js:62).
        closure_threshold: oral diameter (mm) below which we treat
            the tract as closed for release-event detection. JS uses
            0.05 for velopharyngeal closure (audit §C); we re-use it
            here for oral closure.
        closure_eps: ε added to the reflection-coefficient denominator
            so that A → 0 stays differentiable (replaces JS's hard
            `if A==0: r=0.999` at tract.js:309).
        seed: deterministic PRNG seed for the noise / Simplex tables.
        resample_lowpass_width: torchaudio Resample `lowpass_filter_width`.
            Larger = sharper anti-alias, more compute. The default (6)
            is torchaudio's library default and is plenty for 48 → 16k.
        intensity_init: starting value of the glottal intensity ramp. Use 1.0
            for corpora whose warm-up was cut before writing (see glottis.py).
        fricative_amp: post-tract amplitude of the additive fricative
            injection. JS's `addTurbulenceNoise` (tract.js:242-254) iterates
            over `tract.touches[]` — under `synthesize_from_vtd.js` no
            touches are ever registered, so JS injects ZERO fricative.
            Our prior heuristic (d_min ∈ (0.3, 0.7) → inject) was wrong
            against the JS oracle: it fired for 7/10 audit samples,
            producing the +1 to +3 dB OVER-energy in 1.5-6 kHz band that
            confused early diagnostics. Default 0.0 disables it. Re-enable
            (e.g. 1.0) only when explicit constriction touches are wired
            in (full PT control mode).

    Forward:
        vtd:          (B, T, 1 + n_oral)  velum + oral diameters (mm)
        voicing:      (B, T) ∈ [0, 1]
        f0:           (B, T) Hz
        valid_frames: (B,) int or None — if given, audio after the
                      `valid_frames[b]`-th frame is zeroed for batch
                      element `b`.

    Returns:
        audio: (B, T_audio_out), where
                   T_audio_out = T * (sr_out // frame_rate)
                   (= T * 320 for sr_out=16000, frame_rate=50).
    """

    def __init__(
        self,
        sr_internal: int = 48000,
        sr_out: int = 16000,
        frame_rate: int = 50,
        n_oral: int = 44,
        n_nose: int = 28,
        n_ir_samples: int = 512,
        fricative_freq: float = 1000.0,
        aspirate_freq: float = 500.0,
        closure_threshold: float = 0.05,
        closure_velum_gate: bool = True,
        closure_velum_area_threshold: float = 0.05,
        closure_eps: float = 1e-6,
        seed: int = 31337,
        resample_lowpass_width: int = 6,
        fricative_amp: float = 0.0,
        burst_mode: str = "post",
        burst_frame_offset: int = 1,
        intensity_init: float = 0.0,
        output_scale: float = 0.125,
    ) -> None:
        super().__init__()

        # ---- Sanity checks ------------------------------------------ #
        assert sr_internal % frame_rate == 0, (
            f"sr_internal ({sr_internal}) must be divisible by frame_rate "
            f"({frame_rate})."
        )
        assert sr_internal % sr_out == 0, (
            f"sr_internal ({sr_internal}) must be a clean integer "
            f"multiple of sr_out ({sr_out}) for the integer-ratio "
            f"resampler. Got {sr_internal}/{sr_out}={sr_internal/sr_out}."
        )
        assert sr_out % frame_rate == 0, (
            f"sr_out ({sr_out}) must be divisible by frame_rate "
            f"({frame_rate}) so each VTD frame maps to an integer "
            f"number of output samples."
        )

        self.sr_internal = int(sr_internal)
        self.sr_out = int(sr_out)
        self.frame_rate = int(frame_rate)
        self.n_oral = int(n_oral)
        self.n_nose = int(n_nose)
        self.n_ir_samples = int(n_ir_samples)
        self.closure_threshold = float(closure_threshold)
        # Velum gate on the closure-transient detector. Default True =
        # physically-correct behaviour matching JS addTransient (fire only
        # when the velopharyngeal port is closed, noseA[0] = velum^2 < thr).
        # Set False to reproduce the legacy ungated behaviour.
        self.closure_velum_gate = bool(closure_velum_gate)
        self.closure_velum_area_threshold = float(closure_velum_area_threshold)
        self.closure_eps = float(closure_eps)
        self.seed = int(seed)
        self.fricative_amp = float(fricative_amp)
        # Plosive burst routing:
        #   "post"  — legacy: a raw exponential added to the finished waveform
        #             (default; preserves upstream behaviour bit-for-bit)
        #   "tract" — JS-faithful: injected inside the waveguide at the
        #             obstructed segment (tract.js:228-229)
        #   "off"   — no burst at all; needed to measure the burst's own
        #             contribution by differencing against it
        assert burst_mode in ("post", "tract", "off"), burst_mode
        self.burst_mode = str(burst_mode)
        # Frames to delay the injected burst relative to the detected release.
        # MEASURED against tools/oracle_nolag.js (the de-lagged, frication-capable
        # instrument): offset 1 gives -0.00 dB (ratio 0.9994); offset 0 gives
        # -1.69 dB. So 1 is correct for that instrument, because oracle_nolag
        # still calls reshapeTract at end-of-frame to keep the lastObstruction /
        # burst bookkeeping (oracle_nolag.js:137) — only the DIAMETER
        # double-smoothing was removed, not the burst schedule.
        #
        # OPEN: the scope analysis argues 0 is correct for the CORPUS GENERATOR
        # (audioSystem.js), whose control blocks are 512 samples rather than 960
        # — a third timing schedule that neither oracle reproduces. That claim is
        # NOT verifiable with the instruments in this tree. Default follows the
        # measurement; revisit if a 512-block-accurate oracle is built.
        self.burst_frame_offset = int(burst_frame_offset)
        self.intensity_init = float(intensity_init)
        # Final output amplitude calibration (JS audioSystem.js:122 uses *0.125
        # after summing lipOut+noseOut over its 2× oversampling pair).
        self.output_scale = float(output_scale)

        # JS `addTurbulenceNoise` skips touches with index < 2
        # (tract.js:248). Held as a buffer so it follows .to(device).
        _gate = torch.ones(self.n_oral)
        _gate[:2] = 0.0
        self.register_buffer("_seg_gate", _gate, persistent=False)

        self.samples_per_frame_int = self.sr_internal // self.frame_rate
        self.samples_per_frame_out = self.sr_out // self.frame_rate

        # ---- Sub-modules -------------------------------------------- #
        self.glottis = GlottalSourceGenerator(
            sr=self.sr_internal,
            frame_rate=self.frame_rate,
            seed=self.seed,
            intensity_init=self.intensity_init,
        )
        self.tract = WaveguideTract(
            n_oral=self.n_oral,
            n_nose=self.n_nose,
            samples_per_frame=self.samples_per_frame_int,
            closure_eps=self.closure_eps,
            # 2026-08-08: this was NOT forwarded, so n_ir_samples was accepted
            # here, stored, and then silently dropped -- WaveguideTract always
            # used its own default of 512 and --n-ir-steps had zero effect.
            # It is not a cosmetic knob: waveguide.py:102-108 records that at
            # 512 a sustained lip near-closure is under-resolved (~26% tail
            # energy, ~1.5 dB), i.e. this is exactly the parameter you would
            # turn to investigate closure fidelity. Defaults match (512) so
            # forwarding it changes nothing for existing configs.
            n_ir_samples=self.n_ir_samples,
        )
        self.noise = NoiseFilterBank(
            sr=self.sr_internal,
            seed=self.seed,
            aspirate_freq=float(aspirate_freq),
            fricative_freq=float(fricative_freq),
        )

        # Resampler: 48 k → 16 k via torchaudio (integer-ratio kaiser
        # FIR). The instance pre-computes its anti-alias kernel at
        # construction so per-call cost is just one matmul / conv.
        self.resampler = TT.Resample(
            orig_freq=self.sr_internal,
            new_freq=self.sr_out,
            lowpass_filter_width=int(resample_lowpass_width),
        )

    # ------------------------------------------------------------------ #
    # Fricative injection — per-frame gated additive turbulence.
    #
    # JS reference (tract.js:242-270):
    #   for each touch (index, diameter):
    #       if diameter > 0:
    #           thinness = clamp(8*(0.7 - diameter), 0, 1)
    #           openness = clamp(30*(diameter - 0.3), 0, 1)
    #           noise = turbulenceNoise * thinness * openness
    #                 * getNoiseModulator()
    #           inject at R[index] and L[index]
    #
    # Our Option-2 port collapses this to a single per-frame gain
    # applied to the audio-rate fricative noise (additive, not waveguide-
    # injected). The thinness×openness factor is computed from the
    # per-frame `d_min` (min oral diameter) so that the gate "fires"
    # whenever any segment in the frame is in the (0.3, 0.7) mm range.
    # ------------------------------------------------------------------ #
    def _burst_sources(self, vtd: torch.Tensor, T_audio_int: int):
        """Plosive-burst interior source: (signal (B, T_audio), weights (B, T, n_oral)).

        JS injects the burst INSIDE the waveguide, at the segment that was
        obstructed, exactly like turbulence (tract.js:222-231)::

            amplitude = strength * 2**(-exponent * timeAlive)     # 0.3, 200
            this.R[trans.position] += amplitude/2
            this.L[trans.position] += amplitude/2

        and fires it when an obstruction clears with the velum closed
        (tract.js:121-124)::

            if (lastObstruction > -1 && newLastObstruction === -1 && noseA[0] < 0.05)
                addTransient(lastObstruction)

        where ``lastObstruction`` is the HIGHEST-indexed segment with
        ``diameter <= 0`` in the previous frame.

        The previous implementation instead added a raw one-pole exponential to
        the FINISHED waveform. That has two consequences: the burst receives no
        tract colouring at all (it is a ~22 Hz thump, about 33 dB down at 1 kHz,
        rather than a formant-shaped release), and it sits outside the waveguide
        so it cannot be shaped by the very constriction that produced it.

        Routing it through the same interior-injection path as frication fixes
        both, and costs one more anchor bank.

        APPROXIMATION, stated plainly: the spatial weight is per frame, so if two
        releases at DIFFERENT positions overlap in one frame, only the most
        recent position is used. The envelope decays 2**(-200t), i.e. ~24 dB per
        20 ms frame, so overlap is rare and its residual is small — but it is not
        exact, unlike the frication path.
        """
        B, T, _ = vtd.shape
        device, dtype = vtd.device, vtd.dtype
        spf = self.samples_per_frame_int

        d = vtd[..., 1:].detach()                                          # (B, T, n_oral)
        d_min = d.min(dim=-1).values                                       # (B, T)
        velum = vtd[..., 0].detach() if self.closure_velum_gate else None

        events = detect_release_events(
            d_min, samples_per_frame=spf, threshold=self.closure_threshold,
            velum_per_frame=velum,
            velum_area_threshold=self.closure_velum_area_threshold,
        )

        sig = torch.zeros(B, T_audio_int, device=device, dtype=dtype)
        w = torch.zeros(B, T, self.n_oral, device=device, dtype=dtype)
        n = torch.arange(T_audio_int, device=device, dtype=dtype)

        for b in range(B):
            for s_idx in events[b].tolist():
                # Two DIFFERENT frames are involved and conflating them zeroes
                # the burst entirely (measured):
                #   f_rel  — the detected release frame; the obstruction position
                #            must be read from f_rel-1, the last CLOSED frame.
                #   f_inj  — where the burst is injected: one frame later. JS reads
                #            `newLastObstruction` from the PRE-moveTowards diameter
                #            (tract.js:112-114) and fires addTransient at the END of
                #            reshapeTract, so relative to a detector run on the
                #            recorded (post-smoothing) VTD the release lands one
                #            frame late. Worth -1.7 dB -> -0.06 dB on total energy.
                f_rel = int(s_idx) // spf
                if f_rel < 1:
                    continue
                f = min(f_rel + self.burst_frame_offset, T - 1)
                s_idx = f * spf
                # JS `lastObstruction`: highest-index segment with d <= 0 in the
                # frame BEFORE the release. Our closure floor is not exactly 0,
                # so the closure_threshold is used as the obstruction test.
                obstructed = (d[b, f_rel - 1] <= self.closure_threshold)
                if not bool(obstructed.any()):
                    continue
                pos = int(torch.nonzero(obstructed)[-1].item())
                # 0.3 * 2**(-200 * t), t in seconds from the release sample.
                t_sec = torch.clamp(n - float(s_idx), min=0.0) / float(self.sr_internal)
                env = 0.3 * torch.pow(torch.tensor(2.0, device=device, dtype=dtype),
                                      -200.0 * t_sec)
                env = torch.where(n >= float(s_idx), env, torch.zeros_like(env))
                # JS lifeTime = 0.2 s
                env = torch.where(t_sec <= 0.2, env, torch.zeros_like(env))
                sig[b] = sig[b] + env
                # /2 R/L split (tract.js:228-229); `inj` is added to both.
                lo_f = f
                hi_f = min(T, f + 11)          # 0.2 s = 10 frames
                w[b, lo_f:hi_f, pos] = 0.5
        return sig, w

    def _fricative_weights(self, vtd: torch.Tensor) -> torch.Tensor:
        """Per-segment in-tract turbulence injection weights. (B, T, n_oral)

        REPLACES the former `_fricative_injection`, which had three defects:
        a coordinate-system error, a per-frame `min` that discarded position,
        and post-tract addition that bypassed the resonances entirely.

        COORDINATE SYSTEM — the subtle part
        -----------------------------------
        `addTurbulenceNoise` reads `touch[1]`, a **UI touch diameter**, and
        applies (tract.js:262-263)::

            thinness = clamp(8*(0.7 - d_touch), 0, 1)
            openness = clamp(30*(d_touch - 0.3), 0, 1)

        but `handleTouches` subtracts 0.3 before a touch reaches the tract::

            tract.js:400      diameter -= 0.3
            tract.js:401      if (diameter<0) diameter = 0
            tract.js:418-420  targetDiameter[..] = diameter + (..)*shrink   # shrink=0 at the centre

        so at the constriction centre `targetDiameter = d_touch - 0.3`, i.e.
        `d_touch = d_vtd + 0.3`. Substituting gives the law IN VTD COORDINATES::

            thinness = clamp(8*(0.4 - d_vtd), 0, 1)
            openness = clamp(30 * d_vtd,      0, 1)      -> support (0, 0.4)

        The previous code applied the *touch-coordinate* formula to
        *VTD-coordinate* data, i.e. the gate sat 0.3 off. Consequences, both
        measured: on real predictions the old law gives mean gain 0.012 (gate
        effectively shut) versus 0.99 for the corrected one; and the stock PT
        rest profile (0.6 in the pharynx) falls INSIDE the mis-shifted window
        but correctly OUTSIDE the true one, which is why a naive
        weight-every-segment scheme produced spurious pharyngeal noise.

        Because the true window is (0, 0.4) and PT's neutral geometry lies
        outside it, weighting every segment self-gates down to the actual
        constriction — no argmax, hence nothing non-differentiable.

        POSITION — JS injects at `floor(index)+1` (tract.js:266), one segment
        lipward of the touch, and skips `index < 2` (tract.js:248). Both are
        reproduced here.

        The constant `0.66` (tract.js:252) and the `/2` R/L split
        (tract.js:266-267) are folded in, since they are constants and the
        injection is linear. `getNoiseModulator()` is time-varying at audio
        rate and is applied to the noise signal instead, in `forward`.
        """
        d = vtd[..., 1:]                                                  # (B, T, n_oral)
        thinness = torch.clamp(8.0 * (0.4 - d), 0.0, 1.0)
        openness = torch.clamp(30.0 * d, 0.0, 1.0)
        g = thinness * openness                                           # (B, T, n_oral)

        # JS guard `if (index < 2) continue` (tract.js:248).
        g = g * self._seg_gate

        # JS injects at floor(index)+1 → shift one segment toward the lips,
        # leaving segment 0 unexcited.
        g = F.pad(g[..., :-1], (1, 0))

        # 0.66 (tract.js:252) and the /2 R/L split (tract.js:266-267).
        return (0.66 * 0.5) * g

    # ------------------------------------------------------------------ #
    # Closure-release transients.
    #
    # Pipeline:
    #   1. d_min_per_frame = vtd[..., 1:].min(-1)         — (B, T)
    #   2. release_events_list = detect_release_events(...)
    #         (list of B variable-length sample-index tensors)
    #   3. Pad to common K with -1 sentinels → (B, K).
    #   4. closure_transient(...) → (B, T_audio_int).
    # ------------------------------------------------------------------ #
    def _closure_transient_signal(
        self,
        d_min_per_frame: torch.Tensor,   # (B, T)
        T_audio_int: int,
        velum_per_frame: Optional[torch.Tensor] = None,   # (B, T)
    ) -> torch.Tensor:
        B = d_min_per_frame.shape[0]
        device = d_min_per_frame.device

        # `detect_release_events` is non-differentiable (hard <, >=).
        # We detach the input defensively so that no spurious gradients
        # are produced through the comparison branch.
        events_list = detect_release_events(
            d_min_per_frame.detach(),
            samples_per_frame=self.samples_per_frame_int,
            threshold=self.closure_threshold,
            velum_per_frame=(velum_per_frame.detach()
                             if velum_per_frame is not None else None),
            velum_area_threshold=self.closure_velum_area_threshold,
        )

        max_k = max((int(t.numel()) for t in events_list), default=0)
        if max_k == 0:
            # Nothing to add. Return a zero buffer rather than calling
            # closure_transient with a degenerate (B, 1) sentinel-only
            # tensor (which also works, but this avoids the no-op
            # scatter-add).
            return torch.zeros(
                (B, T_audio_int),
                dtype=d_min_per_frame.dtype,
                device=device,
            )

        # Pad each row to max_k with the -1 "no event" sentinel that
        # `closure_transient` recognises.
        padded = torch.full(
            (B, max_k), -1, dtype=torch.long, device=device,
        )
        for b, ev in enumerate(events_list):
            if ev.numel() > 0:
                padded[b, : ev.numel()] = ev.to(torch.long)

        transient_signal = closure_transient(
            padded,
            n_samples_total=T_audio_int,
            sr=self.sr_internal,
        )                                                                # (B, T_audio_int)
        # Match the output dtype to the surrounding pipeline.
        return transient_signal.to(d_min_per_frame.dtype)

    # ------------------------------------------------------------------ #
    # valid_frames masking.
    #
    # `valid_frames[b]` is the number of *valid* leading VTD frames for
    # batch element b. Audio samples corresponding to frames past
    # valid_frames[b] are zeroed (post-resample, at sr_out resolution).
    # ------------------------------------------------------------------ #
    def _apply_valid_mask(
        self,
        audio_out: torch.Tensor,         # (B, T_out)
        valid_frames: torch.Tensor,      # (B,) int
    ) -> torch.Tensor:
        B, T_out = audio_out.shape
        spf_out = self.samples_per_frame_out
        # Per-batch number of valid output samples.
        valid_samples = (valid_frames.to(torch.long) * spf_out).clamp(0, T_out)
        # Index grid (1, T_out); compare against (B, 1) threshold.
        idx = torch.arange(T_out, device=audio_out.device).view(1, T_out)
        mask = (idx < valid_samples.view(B, 1)).to(audio_out.dtype)      # (B, T_out)
        return audio_out * mask

    # ================================================================== #
    # Forward
    # ================================================================== #
    def forward(
        self,
        vtd: torch.Tensor,
        voicing: torch.Tensor,
        f0: torch.Tensor,
        valid_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ---- Shape checks ------------------------------------------- #
        assert vtd.dim() == 3, f"vtd must be (B, T, 1+n_oral); got {vtd.shape}"
        assert vtd.shape[-1] == 1 + self.n_oral, (
            f"vtd last dim expected {1 + self.n_oral}; got {vtd.shape[-1]}"
        )
        B, T, _ = vtd.shape
        assert voicing.shape == (B, T), (
            f"voicing shape expected ({B},{T}); got {tuple(voicing.shape)}"
        )
        assert f0.shape == (B, T), (
            f"f0 shape expected ({B},{T}); got {tuple(f0.shape)}"
        )

        device = vtd.device
        spf_int = self.samples_per_frame_int
        T_audio_int = T * spf_int

        # ---- 1. Noise sources (48 kHz) ------------------------------ #
        aspirate_noise, fricative_noise = self.noise(
            n_samples=T_audio_int,
            batch_size=B,
            device=device,
        )                                                                # both (B, T_audio_int)
        # Match dtype to the surrounding pipeline (vtd may be e.g. fp64
        # in tests; noise comes back float32).
        if aspirate_noise.dtype != vtd.dtype:
            aspirate_noise = aspirate_noise.to(vtd.dtype)
            fricative_noise = fricative_noise.to(vtd.dtype)

        # ---- 2. Glottal source -------------------------------------- #
        # `noise_mod` = JS getNoiseModulator() (glottis.js:111-114). PT applies
        # it to the in-tract turbulence as well (tract.js:259), not only to
        # aspiration; it was previously computed but never exposed.
        glottal, noise_mod = self.glottis(
            f0, voicing, aspirate_noise, return_noise_mod=True,
        )                                                                # (B, T_audio_int) x2

        # ---- 3. Sample-level continuous waveguide ------------------- #
        # Glottal source is injected directly at R[0] each audio sample;
        # reflection coefficients lambda-interpolate between prev-frame
        # and cur-frame across each frame boundary. See waveguide.py.
        # ---- 3b. In-tract turbulence (frication) -------------------- #
        # PT injects turbulence INSIDE the waveguide at the constriction
        # (tract.js:266-267), so it is shaped by the front and back cavities.
        # Superposition lets the whole spatial distribution ride one extra IR
        # pass; see WaveguideTract._run_ir_loop. fricative_amp = 0.0 keeps the
        # legacy (frication-free) behaviour bit-for-bit.
        interior = []
        if self.fricative_amp > 0.0:
            interior.append((
                self.fricative_amp * fricative_noise * noise_mod,          # signal
                self._fricative_weights(vtd),                              # weights
            ))
        if self.burst_mode == "tract":
            bsig, bw = self._burst_sources(vtd, T_audio_int)
            # lerp=False. The burst's energy envelope has a 3.6 ms time
            # constant while a frame is 20 ms, so lerping the injection weight
            # from the previous frame's zero multiplies away almost all of it.
            # Analytic loss: -11.87 dB; measured -12.21 dB (isolated bench) and
            # -12.43 dB (end-to-end). Turning it off recovers that.
            #
            # An earlier note here claimed the opposite ("turning off the lerp
            # made it worse"). That measurement was invalid: it differenced a
            # closed stimulus against an open one, in which the burst is only
            # 2.674% of the energy and the JS renderer's own per-frame
            # reshapeTract re-smoothing dominates. Difference against
            # burst_mode="off" instead.
            interior.append((bsig, bw, False))
        audio_48k = self.tract(
            vtd, glottal, interior=interior if interior else None,
        )                                                                  # (B, T_audio_int)

        # ---- 4. Fricative injection (additive, post-tract) ---------- #
        # `d_min` is the per-frame minimum oral diameter (skip index 0
        # which is the velum). `.values` keeps the same shape (B, T).
        # (the former post-tract fricative addition lived here; it bypassed
        # the tract entirely and has been replaced by the in-tract injection
        # above. `d_min_per_frame` is still needed by the release detector.)
        d_min_per_frame = vtd[..., 1:].min(dim=-1).values                 # (B, T)

        # ---- 5. Closure-release transients (additive, post-tract) --- #
        # Velum gate (default on): fire only on oral (closed-port) releases,
        # matching JS addTransient; pass the per-frame velum = vtd[..., 0].
        # Only in "post" mode; "tract" injects inside the waveguide and
        # "off" emits none (used to isolate the burst by differencing).
        if self.burst_mode == "post":
            velum_for_gate = vtd[..., 0] if self.closure_velum_gate else None
            transient_signal = self._closure_transient_signal(
                d_min_per_frame, T_audio_int, velum_per_frame=velum_for_gate
            )                                                             # (B, T_audio_int)
            audio_48k = audio_48k + transient_signal

        # ---- 5b. Final output scaling (JS audioSystem.js:122) -------- #
        # JS does `output[i] = vocalOutput * 0.125` after summing
        # lipOutput + noseOutput across the 2× oversampled sub-step pair.
        # Our sample-level waveguide ALSO sums oversampled sub-steps but
        # leaves the *0.125 to be applied here, matching JS exactly.
        audio_48k = audio_48k * self.output_scale

        # ---- 6. Downsample 48 kHz → 16 kHz -------------------------- #
        # torchaudio Resample works on the last dimension. The integer
        # ratio (3:1) + kaiser-window FIR is fully differentiable. The
        # Resample module pre-computes its FIR kernel in float32, so we
        # cast the audio to the kernel dtype if necessary (e.g. when the
        # caller is operating in float64) and restore the original dtype
        # afterwards.
        resampler_dtype = self.resampler.kernel.dtype
        input_dtype = audio_48k.dtype
        if input_dtype != resampler_dtype:
            audio_out = self.resampler(audio_48k.to(resampler_dtype)).to(
                input_dtype
            )
        else:
            audio_out = self.resampler(audio_48k)                        # (B, ~T*spf_out)
        # Crop / right-pad to the exact expected output length so that
        # downstream consumers can rely on `T * spf_out`. Resample may
        # produce a sample or two more/less depending on the filter
        # transient — we standardize here.
        T_out_target = T * self.samples_per_frame_out
        cur_len = audio_out.shape[-1]
        if cur_len > T_out_target:
            audio_out = audio_out[..., :T_out_target]
        elif cur_len < T_out_target:
            pad = T_out_target - cur_len
            audio_out = F.pad(audio_out, (0, pad))

        # ---- 7. Optional valid_frames masking ----------------------- #
        if valid_frames is not None:
            assert valid_frames.shape == (B,), (
                f"valid_frames must be (B,); got {tuple(valid_frames.shape)}"
            )
            audio_out = self._apply_valid_mask(audio_out, valid_frames)

        return audio_out


# ---- Smoke test ---------------------------------------------------------- #
if __name__ == "__main__":
    import time
    torch.manual_seed(0)

    B, T = 2, 10
    n_oral = 44
    sr_int, sr_out, frame_rate = 48000, 16000, 50
    spf_out = sr_out // frame_rate          # 320
    T_audio_out_expected = T * spf_out      # 3200

    print("=== DifferentiableAudioSystem smoke test ===")
    sys_mod = DifferentiableAudioSystem(
        sr_internal=sr_int,
        sr_out=sr_out,
        frame_rate=frame_rate,
        n_oral=n_oral,
        n_nose=28,
        n_ir_samples=512,
    )
    n_params = sum(p.numel() for p in sys_mod.parameters())
    n_buffers = sum(b.numel() for b in sys_mod.buffers())
    print(f"params={n_params}, buffers={n_buffers}, B={B}, T={T}, "
          f"sr_int={sr_int}, sr_out={sr_out}, frame_rate={frame_rate}")
    print(f"samples_per_frame_int={sys_mod.samples_per_frame_int}, "
          f"samples_per_frame_out={sys_mod.samples_per_frame_out}")

    # ---- Realistic-ish synthetic inputs ----------------------------- #
    # VTD: velum ~0.05 (closed nose), oral baseline ~1.0 mm with jitter.
    vtd = torch.full((B, T, 1 + n_oral), 1.0)
    vtd[..., 0] = 0.05
    vtd[..., 1:] = vtd[..., 1:] + 0.2 * torch.randn(B, T, n_oral)
    # Force one closure-release event in batch 0 around frame 5.
    vtd[0, 3:5, 1:] = 0.01     # closed at frames 3-4
    vtd[0, 5:, 1:] = 1.0       # open at frames 5+
    vtd = vtd.clamp(min=0.001)

    voicing = 0.3 + 0.6 * torch.rand(B, T)             # ∈ [0.3, 0.9]
    f0 = 100.0 + 100.0 * torch.rand(B, T)              # ∈ [100, 200]

    # ---- Forward pass ----------------------------------------------- #
    t0 = time.time()
    audio = sys_mod(vtd, voicing, f0)
    elapsed_fwd = time.time() - t0
    print(f"\n[fwd]   audio.shape   = {tuple(audio.shape)}   "
          f"(expected ({B}, {T_audio_out_expected}))")
    print(f"[fwd]   audio.dtype   = {audio.dtype}")
    print(f"[fwd]   finite        = {torch.isfinite(audio).all().item()}")
    print(f"[fwd]   min/max       = {audio.min().item():+.4f} / "
          f"{audio.max().item():+.4f}")
    rms = audio.pow(2).mean(dim=-1).sqrt()
    print(f"[fwd]   per-batch RMS = {[f'{v:.4f}' for v in rms.tolist()]}")
    print(f"[fwd]   abs().mean    = {audio.abs().mean().item():.4f}")
    print(f"[fwd]   elapsed       = {elapsed_fwd*1000:.1f} ms")
    assert audio.shape == (B, T_audio_out_expected), "wrong audio shape"
    assert torch.isfinite(audio).all(), "audio has non-finite values"
    assert audio.abs().max() > 1e-4, "audio has trivial / zero energy"

    # ---- Backward pass ---------------------------------------------- #
    vtd_g = vtd.detach().clone().requires_grad_(True)
    voicing_g = voicing.detach().clone().requires_grad_(True)
    f0_g = f0.detach().clone().requires_grad_(True)
    t0 = time.time()
    audio_g = sys_mod(vtd_g, voicing_g, f0_g)
    loss = audio_g.pow(2).mean()
    loss.backward()
    elapsed_bwd = time.time() - t0
    print(f"\n[bwd]   loss          = {loss.item():.6f}")
    print(f"[bwd]   elapsed (fwd+bwd) = {elapsed_bwd*1000:.1f} ms")

    for name, t in [("vtd", vtd_g), ("voicing", voicing_g), ("f0", f0_g)]:
        g = t.grad
        finite = bool(torch.isfinite(g).all().item())
        amean = g.abs().mean().item()
        amax = g.abs().max().item()
        print(f"[bwd]   grad {name:<8s} finite={finite} "
              f"abs.mean={amean:.4e}  abs.max={amax:.4e}")
        assert finite, f"non-finite gradient in {name}"
        assert amean > 0.0, f"zero gradient in {name}"

    # ---- valid_frames mask test ------------------------------------- #
    vf = torch.tensor([T, T // 2])    # batch 0: full; batch 1: half
    audio_masked = sys_mod(vtd, voicing, f0, valid_frames=vf)
    half_samples = (T // 2) * spf_out
    tail_energy_b1 = audio_masked[1, half_samples:].abs().sum().item()
    tail_energy_b0 = audio_masked[0, half_samples:].abs().sum().item()
    print(f"\n[mask]  batch1 tail energy (should be 0) = {tail_energy_b1:.4e}")
    print(f"[mask]  batch0 tail energy (should be >0) = {tail_energy_b0:.4e}")
    assert tail_energy_b1 < 1e-8, "valid_frames mask did not zero batch1 tail"
    assert tail_energy_b0 > 1e-4, "batch0 should retain energy past midpoint"

    # ---- Determinism: same inputs → same output --------------------- #
    audio2 = sys_mod(vtd, voicing, f0)
    eq = torch.equal(audio, audio2)
    print(f"\n[det]   identical-inputs determinism: {eq}")
    assert eq, "DifferentiableAudioSystem must be deterministic"

    # ---- Memory snapshot -------------------------------------------- #
    # Rough byte count for the largest intermediate (B, T_audio_int).
    T_audio_int = T * sys_mod.samples_per_frame_int
    mb_48k = (B * T_audio_int * 4) / 1e6
    print(f"\n[mem]   audio_48k bytes = {B*T_audio_int*4} "
          f"(~{mb_48k:.2f} MB at fp32, B={B}, T_audio_int={T_audio_int})")

    print("\nOK")
