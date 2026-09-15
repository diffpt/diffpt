"""Glottal source generator — PyTorch port of `v1/js/src/models/glottis.js`.

LF (Liljencrants-Fant) glottal pulse + aspiration noise. Per-AUDIO-SAMPLE
LF coefficients (one Rd per sample) with **phase-continuous across frame
boundaries** via cumsum-based phase tracking. JS-equivalent inter-frame
smoothing of `frequency` and `tenseness` is replicated as follows:

* `frequency` is passed through a multiplicative 1.1× cap (matching JS
  finishBlock lines 133-136 `smoothFrequency` recurrence). The resulting
  per-frame "smoothFreq" sequence is then linearly interpolated across
  every frame from (prev frame's smoothFreq) → (current smoothFreq) at
  audio rate, mirroring JS setup_waveform's `lambda` interpolation
  (glottis.js:47-48).
* `tenseness` is linearly interpolated across each frame from (prev frame
  UITenseness) → (current UITenseness) at audio rate. Mirrors JS
  oldTenseness/newTenseness lerp in setup_waveform.

This avoids the "broadband click" artifact our prior per-frame-constant
implementation produced at every frame boundary that had a voicing or f0
transition (≈ 50 Hz tonal energy injection visible in 1.5-6 kHz bands).

Per-frame contract:
- (f0, voicing) at VTD rate (50 Hz). Smoothing + interpolation produces
  per-audio-sample f0_audio / tenseness_audio. LF coefficients then
  evaluated per audio sample. No global state object — fully
  differentiable.
- Aspiration noise uses the same sample-level t_frac as modulator.
- Intensity ramp follows JS spec **per audio frame of 960 samples**
  (+0.13 / frame = 50 Hz finishBlock rate matching synthesize_from_vtd.js,
  not the 512-sample block of audioSystem.js's pt_processor).
  That block-size difference is real and unresolved: the CORPUS generator
  runs its whole control loop at 512 samples (93.75 Hz -- see
  utils.js:39 `numberOfSamples = floor(sr*len/512 + 1)`), while the VTD
  labels are snapshots taken every 960 samples (50 Hz, audioSystem.js:132).
  The 960 is not arbitrary: 960 @ 48k == 320 @ 16k == the SSL stride
  (dataset.py:72), chosen so labels align with the encoder frame-for-frame
  with no resampler. The incommensurability with 512 was a known,
  documented cost at the time (AUDIO_GENERATION_PROPOSAL.md:284-299).
  It does NOT affect the ramp, because the corpus discards 9600 samples of
  warm-up and the ramp saturates after 4096 either way, so
  intensity_init=1.0 is exact. It DOES affect burst timing (see
  audio_system.burst_frame_offset) and it puts up to ~0.9 frames of drift
  between the VTD labels and control_arr, from which f0 and voiceness_gt
  are read. Full arithmetic in docs/corpus_timing_and_intensity.md.

Refs: design/01_js_audit.md §B,§E, design/02_port_strategy.md, glottis.js,
synthesize_from_vtd.js (lambda interpolation, isTouched=true, per-frame
finishBlock).
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn


def _build_simplex1_lookup(size: int, seed: int) -> np.ndarray:
    """Pre-computed 1-D Simplex-style noise in [-1, 1].

    Replaces JS SimplexNoiseNode.simplex1 (utils/SimplexNoiseNode.js:122-124).
    The JS source uses `simplex1` only for small-amplitude (0.02) low-frequency
    modulation, so a deterministic seeded low-pass-smoothed white noise of the
    same statistics suffices. Deterministic across runs given `seed`.
    """
    rng = np.random.RandomState(seed)
    raw = rng.randn(size).astype(np.float64)
    kernel_len = 17  # ~0.7 inter-sample correlation, mimics Simplex band-limit
    n = np.arange(kernel_len)
    kernel = 0.5 - 0.5 * np.cos(2 * math.pi * n / (kernel_len - 1))
    kernel = kernel / kernel.sum()
    smoothed = np.convolve(raw, kernel, mode="same")
    smoothed = smoothed / max(np.abs(smoothed).max(), 1e-12)
    return smoothed.astype(np.float32)


class GlottalSourceGenerator(nn.Module):
    """Phase-continuous LF pulse + aspiration noise generator.

    Args:
        sr: audio sample rate (48000, JS line 5).
        frame_rate: VTD frame rate (50 → samples_per_frame = 960).
        tenseness_clamp: bounds applied to tenseness to keep sqrt() safe.
        simplex_lookup_size: size of 1-D Simplex lookup buffer.
        seed: deterministic seed for the lookup.
        intensity_block: block size (samples) for JS-spec intensity ramp.
        intensity_init: value of the ramp at block 0. The ramp itself is JS
            behaviour and is kept; this only says where it starts.
            0.0 (default) = JS `intensity` fresh from construction.
            0.13          = JS after its first finishBlock, i.e. exactly the
                            JS sequence 0.13*(f+1) — see note at the ramp.
            1.0           = already saturated, i.e. a corpus whose warm-up
                            was cut off before it was written to disk.

    Forward:
        f0:             (B, T)        Hz, per-frame F0
        voicing:        (B, T) ∈ [0,1] per-frame voicing
        aspirate_noise: (B, T_audio)   pre-filtered 500 Hz bandpass noise,
                                       where T_audio = T * samples_per_frame
    Returns:
        (B, T_audio) glottal source waveform = LF pulse + aspiration.
    """

    def __init__(
        self,
        sr: int = 48000,
        frame_rate: int = 50,
        tenseness_clamp: tuple = (0.001, 0.999),
        simplex_lookup_size: int = 65536,
        seed: int = 31337,
        intensity_block: int = 960,
        intensity_init: float = 0.0,
        smooth_freq_init: float = 140.0,
        smooth_freq_cap: float = 1.1,
        enable_smoothing: bool = True,
    ) -> None:
        super().__init__()
        self.sr = int(sr)
        self.frame_rate = int(frame_rate)
        assert self.sr % self.frame_rate == 0, "sr must be divisible by frame_rate"
        self.samples_per_frame = self.sr // self.frame_rate  # 960
        self.tenseness_clamp = tuple(tenseness_clamp)
        self.simplex_lookup_size = int(simplex_lookup_size)
        self.intensity_block = int(intensity_block)
        self.intensity_init = float(intensity_init)
        self.seed = int(seed)
        self.smooth_freq_init = float(smooth_freq_init)
        self.smooth_freq_cap = float(smooth_freq_cap)
        self.enable_smoothing = bool(enable_smoothing)

        lut = _build_simplex1_lookup(self.simplex_lookup_size, self.seed)
        self.register_buffer("simplex_lookup", torch.from_numpy(lut))

    # --------------------------------------------------------------------- #
    # LF coefficients per (B, T) tenseness — mirrors glottis.js:46-87.
    # --------------------------------------------------------------------- #
    def _compute_lf_coeffs(self, tenseness: torch.Tensor) -> dict:
        # JS line 49 (Rd = 3*(1-tenseness)) + line 53-54 (clamp [0.5, 2.7]).
        rd = torch.clamp(3.0 * (1.0 - tenseness), 0.5, 2.7)
        ra = -0.01 + 0.048 * rd                                            # JS 56
        rk = 0.224 + 0.118 * rd                                            # JS 57
        rg = (rk / 4.0) * (0.5 + 1.2 * rk) / (0.11 * rd - ra * (0.5 + 1.2 * rk))  # JS 58
        ta = ra
        tp = 1.0 / (2.0 * rg)                                              # JS 61
        te = tp + tp * rk                                                  # JS 62
        epsilon = 1.0 / ta                                                 # JS 64
        shift = torch.exp(-epsilon * (1.0 - te))                           # JS 65
        delta = 1.0 - shift                                                # JS 66
        rhs = ((1.0 / epsilon) * (shift - 1.0) + (1.0 - te) * shift) / delta  # JS 68-69
        total_lower = -(te - tp) / 2.0 + rhs                               # JS 71
        total_upper = -total_lower                                         # JS 72
        omega = math.pi / tp                                               # JS 74
        s = torch.sin(omega * te)                                          # JS 75
        # y must stay positive for log; with Rd∈[0.5,2.7] it is, but clamp
        # defensively against fp edge cases.
        y = torch.clamp(-math.pi * s * total_upper / (tp * 2.0), min=1e-30)  # JS 76
        alpha = torch.log(y) / (tp / 2.0 - te)                             # JS 77-78
        e0 = -1.0 / (s * torch.exp(alpha * te))                            # JS 79
        return {
            "alpha": alpha, "E0": e0, "epsilon": epsilon, "shift": shift,
            "Delta": delta, "Te": te, "Tp": tp, "omega": omega,
        }

    # --------------------------------------------------------------------- #
    # Normalized LF waveform (JS lines 116-124). Vectorized; both branches
    # computed, then masked. Mask itself is a comparison so its gradient is
    # detached (Heaviside), but gradients flow through both arm expressions.
    # --------------------------------------------------------------------- #
    def _compute_lf_waveform(self, coeffs: dict, in_period_t: torch.Tensor) -> torch.Tensor:
        te    = coeffs["Te"].unsqueeze(-1)
        alpha = coeffs["alpha"].unsqueeze(-1)
        e0    = coeffs["E0"].unsqueeze(-1)
        omega = coeffs["omega"].unsqueeze(-1)
        eps   = coeffs["epsilon"].unsqueeze(-1)
        shift = coeffs["shift"].unsqueeze(-1)
        delta = coeffs["Delta"].unsqueeze(-1)
        t = in_period_t  # (B, T, S)

        # Opening phase (t <= Te). Clamp arg to avoid fp overflow.
        alpha_t = torch.clamp(alpha * t, min=-50.0, max=50.0)
        open_phase = e0 * torch.exp(alpha_t) * torch.sin(omega * t)
        # Return phase (t > Te).
        ret_arg = torch.clamp(-eps * (t - te), min=-50.0, max=50.0)
        return_phase = (-torch.exp(ret_arg) + shift) / delta

        mask = (t > te).to(open_phase.dtype)
        return mask * return_phase + (1.0 - mask) * open_phase

    # --------------------------------------------------------------------- #
    # JS smoothFrequency / oldT,newT inter-frame smoothing.
    # --------------------------------------------------------------------- #
    @staticmethod
    def _smooth_step(prev: torch.Tensor, target: torch.Tensor, cap: float) -> torch.Tensor:
        """JS glottis.js:133-136 — multiplicative cap toward `target` per call."""
        up = prev * cap
        down = prev / cap
        return torch.where(
            target > prev, torch.minimum(up, target),
            torch.where(target < prev, torch.maximum(down, target), target),
        )

    def _build_smoothed_pairs(
        self, f0: torch.Tensor, voicing: torch.Tensor,
    ) -> tuple:
        """Replicate JS finishBlock state for f0 and tenseness across frames.

        Returns four (B, T) tensors `(f0_old, f0_new, tens_old, tens_new)` such
        that during frame `f`'s audio, the running value equals
        `lerp(old[f], new[f], lambda)` with `lambda = s/samples_per_frame`.

        JS state machine (synthesize_from_vtd.js):
            seed.handleTouches([f0[0], voicing[0]])
            isTouched = true            # disables (3-UIT)*(1-intensity) boost
            seed.finishBlock()          # one smoothing step from init=(140, 0.6)

            for f in range(T):
                UIFrequency  = f0[f]
                UITenseness  = voicing_to_tens(voicing[f])
                # ... audio loop using current (oldFreq, newFreq, oldT, newT) ...
                finishBlock()           # oldFreq<-newFreq; newFreq<-smoothFreq;
                                        # oldT<-newT;        newT<-UITenseness

        Trace (cap = 1.1, init f = 140, init t = 0.6):
            sf[0]=140                                     ← init
            sf[1]=smooth(sf[0], f0[0])                    ← seed finishBlock
            sf[f+1]=smooth(sf[f], f0[f-1])  for f>=1      ← end-of-frame-{f-1}
            ut[0]=0.6                                     ← init
            ut[1]=UIT[0]                                  ← seed
            ut[2]=UIT[0]                                  ← end-of-frame 0 (same UIT)
            ut[f+1]=UIT[f-1]  for f>=2                    ← end-of-frame-{f-1}

        Frame f's audio uses (sf[f], sf[f+1]) and (ut[f], ut[f+1]) as
        `(old, new)`. Note the 1-frame delay on f0 and 2-frame delay on
        voicing — both are exact JS-faithful consequences of the state
        machine and the seed initialization.
        """
        B, T = f0.shape
        device = f0.device
        dtype = f0.dtype
        cap = self.smooth_freq_cap
        init_f = self.smooth_freq_init

        # --- f0 smoothed sequence sf, length T+1 -------------------------- #
        sf = torch.empty((B, T + 1), dtype=dtype, device=device)
        sf[:, 0] = init_f
        sf[:, 1] = self._smooth_step(sf[:, 0], f0[:, 0], cap)
        for f in range(1, T):
            sf[:, f + 1] = self._smooth_step(sf[:, f], f0[:, f - 1], cap)

        # --- tenseness sequence ut, length T+1 ---------------------------- #
        uit = 1.0 - torch.cos(voicing * (math.pi * 0.5))                 # (B, T)
        ut = torch.empty((B, T + 1), dtype=dtype, device=device)
        ut[:, 0] = 0.6  # JS init oldTenseness
        ut[:, 1] = uit[:, 0]  # seed (handleTouches → newT = UIT[0])
        if T >= 2:
            ut[:, 2:] = uit[:, : T - 1]

        f0_old = sf[:, :-1]
        f0_new = sf[:, 1:]
        tens_old = ut[:, :-1]
        tens_new = ut[:, 1:]
        return f0_old, f0_new, tens_old, tens_new

    def _interp_to_audio(self, old_seq: torch.Tensor, new_seq: torch.Tensor) -> torch.Tensor:
        """Linearly interpolate (B, T) frame values to (B, T_audio) audio rate.

        Frame f spans audio samples `[f*S, (f+1)*S)` and runs
        `lerp(old_seq[f], new_seq[f], s/S)` where `s` is the within-frame
        sample index — matches JS `setup_waveform(lambda)` lambda = s/S.
        """
        B, T = old_seq.shape
        S = self.samples_per_frame
        lam = torch.arange(S, device=old_seq.device, dtype=old_seq.dtype) / float(S)
        out = old_seq.unsqueeze(-1) * (1.0 - lam) + new_seq.unsqueeze(-1) * lam
        return out.reshape(B, T * S)

    # --------------------------------------------------------------------- #
    # Forward.
    # --------------------------------------------------------------------- #
    def forward(
        self,
        f0: torch.Tensor,
        voicing: torch.Tensor,
        aspirate_noise: torch.Tensor,
        return_noise_mod: bool = False,
    ):
        assert f0.shape == voicing.shape, "f0/voicing must have same (B, T) shape"
        B, T = f0.shape
        S = self.samples_per_frame
        T_audio = T * S
        assert aspirate_noise.shape == (B, T_audio), (
            f"aspirate_noise shape: expected ({B}, {T_audio}), got {tuple(aspirate_noise.shape)}"
        )
        device, dtype = f0.device, f0.dtype
        lo, hi = self.tenseness_clamp

        # 1. JS-faithful (oldFreq, newFreq), (oldT, newT) per frame.
        if self.enable_smoothing:
            f0_old, f0_new, tens_old, tens_new = self._build_smoothed_pairs(
                torch.clamp(f0, min=20.0), voicing
            )
        else:
            # Fallback: per-frame constant (legacy pre-smoothing behavior).
            f0_old = torch.clamp(f0, min=20.0)
            f0_new = f0_old
            uit = 1.0 - torch.cos(voicing * (math.pi * 0.5))
            tens_old = uit
            tens_new = uit

        # 2. Linearly interpolate to audio rate (mirrors JS lambda in
        #    setup_waveform; lambda = s/samples_per_frame across the frame).
        f0_audio = self._interp_to_audio(f0_old, f0_new)                  # (B, T_audio)
        f0_audio = torch.clamp(f0_audio, min=20.0)
        tens_audio_raw = self._interp_to_audio(tens_old, tens_new)        # (B, T_audio)
        # The [lo, hi] clamp exists to keep the LF-coefficient algebra safe.
        # It must NOT reach `loudness` — see the note at the `loudness` line.
        tens_audio = torch.clamp(tens_audio_raw, lo, hi)

        # 3. Per-AUDIO-SAMPLE LF coefficients (one Rd per sample). Cheap
        #    arithmetic; ~50 FLOPs per sample × T_audio samples.
        coeffs = self._compute_lf_coeffs(tens_audio)                      # all (B, T_audio)

        # 4. Phase-continuous t_frac via cumsum at audio rate.
        phase_inc = f0_audio / float(self.sr)                              # (B, T_audio)
        phase = phase_inc.cumsum(dim=-1)                                   # (B, T_audio)
        t_frac = phase.fmod(1.0)                                           # (B, T_audio)

        # 5. LF waveform per audio sample. Reshape to (B, T, S) for the
        #    rest of the pipeline that's still frame-block organized.
        wave = self._compute_lf_waveform_per_sample(coeffs, t_frac)        # (B, T_audio)
        wave_frames = wave.view(B, T, S)

        # 6. Intensity ramp — JS spec: +0.13 per finishBlock. In
        #    synthesize_from_vtd.js finishBlock is called once per FRAME
        #    (S=960 samples @ 48k → 50 Hz), so default intensity_block=960.
        #
        #    The ramp is a fade-in that JS applies when phonation STARTS. The
        #    training corpus does not contain that moment, and the arithmetic
        #    is exact rather than approximate (docs/corpus_timing_and_intensity.md):
        #      - the corpus generator's control block is 512 samples, so
        #        finishBlock fires every 512 and the ramp saturates after
        #        8 blocks = 4096 samples = 85.3 ms;
        #      - generate.py:2378 discards WARMUP_SAMPLES_48K = 9600 samples.
        #    4096 << 9600, so every stored sample is at intensity == 1.0
        #    throughout. Rendering it with a ramp puts un-learnable error on
        #    the first 8 frames: measured +0.912 L of mel-L1 against the real
        #    corpus audio (acceptance bar is 0.35 L), with the remaining frames
        #    unchanged to 7e-03. `intensity_init=1.0` aligns to that WITHOUT
        #    deleting the mechanism, which is genuine PT behaviour and is still
        #    needed to render an onset from a cold tract.
        #
        #    Note the ramp is a function of the sample index alone — a constant
        #    w.r.t. every learnable quantity — so neither the ramp nor this
        #    parameter sits on any gradient path, and the cost is one add.
        #
        #    Known offset: JS increments BEFORE the block is rendered, giving
        #    0.13*(f+1); we give 0.13*f. intensity_init=0.13 makes the two
        #    sequences identical, should an exact-JS render ever be wanted.
        sample_idx_flat = torch.arange(T_audio, device=device, dtype=dtype)
        block_idx = torch.div(sample_idx_flat,
                              float(self.intensity_block),
                              rounding_mode="floor")
        intensity_audio = torch.clamp(self.intensity_init + 0.13 * block_idx,
                                      0.0, 1.0)                            # (T_audio,)
        intensity_bts = intensity_audio.view(1, T, S)
        # `loudness` must be computed from the UNCLAMPED tenseness.
        #
        # JS: loudness = UITenseness ** 0.25 (glottis.js:123 via the renderer's
        # `glottis.loudness = Math.pow(glottis.UITenseness, 0.25)`), so at
        # voiceness = 0 -> UITenseness = 1 - cos(0) = 0 -> loudness = 0 and the
        # LF pulse is EXACTLY silent, leaving aspiration only.
        #
        # Reading the CLAMPED value here instead put a floor of
        # 0.001 ** 0.25 = 0.17783 under loudness, leaking a voiced pulse where
        # JS is silent. Measured on a steady tract at voiceness = 0:
        # +31.7 dB (38.6x) at the tract output, +28.9 dB at the glottal source.
        # After this fix the steady-state ratio is 1.029. Outputs are unchanged
        # for voiceness >= 0.0285, where the clamp never bound.
        #
        # It also removed the gradient: d(loudness)/d(voicing) was exactly 0 for
        # voicing < 0.0285 — a dead zone across the whole unvoiced region, which
        # is precisely where fricatives and aspirated stops live.
        #
        # `min=1e-12` is retained only to keep pow() differentiable at 0; it is
        # 12 orders of magnitude below the value that caused the defect.
        loudness = torch.pow(torch.clamp(tens_audio_raw, min=1e-12), 0.25)  # (B, T_audio)
        wave_frames = wave_frames * intensity_bts * loudness.view(B, T, S)

        # 7. Aspiration noise (JS 92-108). All factors per audio sample.
        sin_part = torch.clamp(torch.sin(2.0 * math.pi * t_frac), min=0.0)  # (B, T_audio)
        voiced_mod = 0.1 + 0.2 * sin_part
        tens_bts = tens_audio.view(B, T, S)
        sin_bts = voiced_mod.view(B, T, S)
        noise_mod = sin_bts + (1.0 - tens_bts * intensity_bts) * 0.3
        asp_env = intensity_bts * (1.0 - torch.sqrt(tens_bts)) * noise_mod

        # 8. Simplex modulation (JS 105). Lookup keyed on totalTime*1.99.
        lut_idx_f = (sample_idx_flat * (1.99 / float(self.sr))) * (
            self.simplex_lookup_size / 60.0
        )
        lut_idx = (lut_idx_f.long() % self.simplex_lookup_size)
        simplex_factor = (0.02 * self.simplex_lookup[lut_idx] + 0.2).view(1, T, S)

        asp_noise_bts = aspirate_noise.view(B, T, S)
        aspiration_frames = asp_env * asp_noise_bts * simplex_factor

        out = (wave_frames + aspiration_frames).reshape(B, T_audio)
        if return_noise_mod:
            # JS getNoiseModulator() (glottis.js:111-114) is applied to BOTH
            # the aspiration path (above) AND the in-tract turbulence
            # (tract.js:259 `turbulenceNoise *= this.glottis.getNoiseModulator()`).
            # It was already computed here but never exposed, so the frication
            # path silently omitted it — worth ~15.7 dB on fully-voiced frames
            # and ~6.8 dB on unvoiced ones.
            return out, noise_mod.reshape(B, T_audio)
        return out

    # --------------------------------------------------------------------- #
    # LF waveform evaluated per-audio-sample (coeffs and t both (B, T_audio)).
    # Same as `_compute_lf_waveform` but no `.unsqueeze(-1)` since the
    # coefficient tensors are already at audio rate.
    # --------------------------------------------------------------------- #
    def _compute_lf_waveform_per_sample(
        self, coeffs: dict, t: torch.Tensor,
    ) -> torch.Tensor:
        te = coeffs["Te"]
        alpha = coeffs["alpha"]
        e0 = coeffs["E0"]
        omega = coeffs["omega"]
        eps = coeffs["epsilon"]
        shift = coeffs["shift"]
        delta = coeffs["Delta"]
        alpha_t = torch.clamp(alpha * t, min=-50.0, max=50.0)
        open_phase = e0 * torch.exp(alpha_t) * torch.sin(omega * t)
        ret_arg = torch.clamp(-eps * (t - te), min=-50.0, max=50.0)
        return_phase = (-torch.exp(ret_arg) + shift) / delta
        mask = (t > te).to(open_phase.dtype)
        return mask * return_phase + (1.0 - mask) * open_phase


# ---- Smoke test ---------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 2, 50  # 1 second @ 50 fps
    sr, fr = 48000, 50
    spf = sr // fr  # 960
    T_audio = T * spf

    gen = GlottalSourceGenerator(sr=sr, frame_rate=fr)
    print(f"GlottalSourceGenerator: samples_per_frame={gen.samples_per_frame}, "
          f"intensity_block={gen.intensity_block}")

    f0 = torch.full((B, T), 140.0)
    voicing = torch.full((B, T), 0.6)
    aspirate_noise = torch.randn(B, T_audio) * 0.3

    out = gen(f0, voicing, aspirate_noise)
    print(f"output.shape       = {tuple(out.shape)}   (expected ({B}, {T_audio}))")
    print(f"output.dtype       = {out.dtype}")
    print(f"is finite          = {torch.isfinite(out).all().item()}")
    print(f"output.min/max     = {out.min().item():.4f} / {out.max().item():.4f}")
    print(f"output.mean        = {out.mean().item():.4e}")
    print(f"output.std         = {out.std().item():.4f}")
    print(f"output.abs().mean  = {out.abs().mean().item():.4f}")
    mid = (T // 2) * spf
    print(f"mid-frame samples  = {[f'{v:.4f}' for v in out[0, mid:mid+8].tolist()]}")

    # ---- A1 verification: phase continuity across frame boundaries ----- #
    # Use first 10 frames (samples 0..9599) with F0=140 Hz, voicing=0.6.
    boundary_samples = [960, 1920, 2880, 3840, 4800, 5760, 6720, 7680, 8640]
    print("A1 boundary jumps (F0=140, voicing=0.6, batch idx 0):")
    for b_idx in boundary_samples:
        jump = (out[0, b_idx] - out[0, b_idx - 1]).item()
        print(f"  sample {b_idx:>4d}: out[{b_idx}] - out[{b_idx - 1}] = {jump:+.4e}")

    # ---- A2 verification: intensity ramp at sample-level (per 512 block) ---- #
    sample_idx_check = [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 3938, 5000]
    block_idx_check = torch.tensor(sample_idx_check, dtype=torch.float32) / 512.0
    intensity_check = torch.clamp(0.13 * block_idx_check.floor(), 0.0, 1.0)
    print("A2 intensity values (block=512, +0.13/block, clamp 1.0):")
    for s, v in zip(sample_idx_check, intensity_check.tolist()):
        print(f"  sample {s:>4d}: intensity = {v:.4f}")

    # Gradient sanity.
    f0_g = f0.clone().requires_grad_(True)
    voicing_g = voicing.clone().requires_grad_(True)
    out_g = gen(f0_g, voicing_g, aspirate_noise)
    out_g.pow(2).mean().backward()
    print(f"grad f0 finite     = {torch.isfinite(f0_g.grad).all().item()}, "
          f"abs.mean = {f0_g.grad.abs().mean().item():.4e}")
    print(f"grad voicing finite= {torch.isfinite(voicing_g.grad).all().item()}, "
          f"abs.mean = {voicing_g.grad.abs().mean().item():.4e}")
    print("OK")
