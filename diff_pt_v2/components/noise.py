"""Noise filter bank — PyTorch port of `v1/js/src/models/audioSystem.js` lines
46-67 + `v1/js/src/controllers/biquadFilter.js`.

Provides the two coloured-noise sources used by the Pink Trombone audio system:

- **Aspiration source**: white noise → 500 Hz biquad bandpass @ Q=0.5
  (audioSystem.js line 62). Fed into the glottis (post-modulated by the
  noise modulator + tenseness; see GlottalSourceGenerator).
- **Fricative source**: white noise → 1000 Hz biquad bandpass @ Q=0.5
  (audioSystem.js line 65). Injected at constriction sites by the tract
  (see tract.js:242-270).

Implementation notes
--------------------
- JS uses a seeded mulberry32 PRNG (seed = 31337, utils/seedrandom.js) to
  emit uniform values in [0, 1), then converts to `Math.random() - 0.5`
  (uniform in [-0.5, 0.5], std ≈ 1/sqrt(12) ≈ 0.289). The PyTorch port
  uses a `torch.Generator` reset to the same integer seed on every forward
  call so output is bit-deterministic given identical
  (n_samples, batch_size, device).
- The PyTorch port draws uniform `[0, 1)` via `torch.rand` and subtracts
  0.5 to match the JS distribution exactly (uniform in [-0.5, 0.5]).
  Earlier versions used `torch.randn` (unit-variance Gaussian) which made
  the post-biquad output ~3.46× louder than the JS oracle (since
  Gaussian std = 1 vs JS uniform std ≈ 0.289 ≈ 1/sqrt(12)). That silently
  miscalibrated downstream audio cycle losses against JS-rendered targets.
- The biquad coefficients exactly mirror biquadFilter.js lines 10-17,
  normalized by a0 so the recurrence form is the standard
  `y = b0·x + b1·x[-1] + b2·x[-2] - a1·y[-1] - a2·y[-2]`. We use
  `torchaudio.functional.biquad` for the recurrence (differentiable).
- The output of the biquad is non-differentiable w.r.t. the (random)
  white-noise excitation; gradients flow through any downstream
  multiplicative envelope (e.g. `getNoiseModulator`) and the filter
  coefficients (which are buffers here, not parameters), as expected.

Refs: design/01_js_audit.md §D, design/02_port_strategy.md
(`components/noise.py` section), audioSystem.js (46-67), biquadFilter.js.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchaudio.functional as F


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _bandpass_coeffs(freq: float, q: float, sr: int) -> Tuple[float, float, float, float, float, float]:
    """Compute normalized 2nd-order bandpass biquad coefficients.

    Mirrors biquadFilter.js lines 10-17 verbatim, then normalizes by a0
    so the difference equation becomes
    `y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2] - a1·y[n-1] - a2·y[n-2]`.

    Returns:
        (b0, b1, b2, a0, a1, a2) — all already divided by the original a0,
        so the returned a0 is exactly 1.0.
    """
    w0 = 2.0 * math.pi * freq / sr
    alpha = math.sin(w0) / (2.0 * q)
    b0 = math.sin(w0) / 2.0          # JS line 12
    b1 = 0.0                         # JS line 13
    b2 = -math.sin(w0) / 2.0         # JS line 14
    a0 = 1.0 + alpha                 # JS line 15
    a1 = -2.0 * math.cos(w0)         # JS line 16
    a2 = 1.0 - alpha                 # JS line 17

    # Normalize: divide all by a0 so the recurrence is a0-free.
    return (b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0)


# ----------------------------------------------------------------------------
# NoiseFilterBank
# ----------------------------------------------------------------------------
class NoiseFilterBank(nn.Module):
    """Deterministic seeded white-noise + dual biquad bandpass filter bank.

    Produces the (aspirate, fricative) noise pair consumed by the
    glottis (500 Hz bandpass aspiration) and the tract (1000 Hz bandpass
    turbulence). Deterministic: identical arguments yield identical
    outputs across calls (per-call generator reset).

    Args:
        sr: audio sample rate (Hz). JS uses 48000 (audioSystem.js line 38,
            biquadFilter.js line 3).
        seed: PRNG seed for the white-noise stream. JS uses 31337
            (audioSystem.js line 48).
        aspirate_freq: bandpass centre frequency for the aspiration noise
            (Hz). JS uses 500 (audioSystem.js line 62).
        fricative_freq: bandpass centre frequency for the fricative noise
            (Hz). JS uses 1000 (audioSystem.js line 63).
        q: bandpass quality factor. JS uses 0.5 for both filters
            (audioSystem.js lines 62-63).

    Forward:
        n_samples:  desired stream length (samples @ sr).
        batch_size: how many independent batch elements to emit.
        device:     output device (defaults to module's buffer device).

    Returns:
        aspirate:  (batch_size, n_samples) — 500 Hz bandpass white noise.
        fricative: (batch_size, n_samples) — 1000 Hz bandpass white noise.
    """

    def __init__(
        self,
        sr: int = 48000,
        seed: int = 31337,
        aspirate_freq: float = 500.0,
        fricative_freq: float = 1000.0,
        q: float = 0.5,
    ) -> None:
        super().__init__()
        self.sr = int(sr)
        self.seed = int(seed)
        self.aspirate_freq = float(aspirate_freq)
        self.fricative_freq = float(fricative_freq)
        self.q = float(q)

        # Pre-compute coefficients once at construction. Stored as
        # non-trainable buffers so they move with .to(device) / .cuda().
        a_b0, a_b1, a_b2, a_a0, a_a1, a_a2 = _bandpass_coeffs(
            self.aspirate_freq, self.q, self.sr
        )
        f_b0, f_b1, f_b2, f_a0, f_a1, f_a2 = _bandpass_coeffs(
            self.fricative_freq, self.q, self.sr
        )

        # Stack as (3,) tensors for b, a so torchaudio's biquad can be called
        # with scalar args (it expects per-call floats; we'll unpack below).
        self.register_buffer(
            "aspirate_coeffs",
            torch.tensor([a_b0, a_b1, a_b2, a_a0, a_a1, a_a2], dtype=torch.float32),
        )
        self.register_buffer(
            "fricative_coeffs",
            torch.tensor([f_b0, f_b1, f_b2, f_a0, f_a1, f_a2], dtype=torch.float32),
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        n_samples: int,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = self.aspirate_coeffs.device

        # 1. Deterministic uniform white noise in [-0.5, 0.5] — matches JS
        #    `Math.random() - 0.5` exactly (std ≈ 1/sqrt(12) ≈ 0.289). We
        #    reset the generator every call so identical args → identical
        #    noise (JS behaviour: `mulberry32(31337)` is re-seeded at the
        #    start of each startSound call; see audioSystem.js:46-48).
        gen = torch.Generator(device=device if device.type == "cpu" else "cpu")
        gen.manual_seed(self.seed)
        # Generate on CPU then move to target device. (CUDA generators
        # accept manual_seed but the bit-stream differs between CPU/CUDA;
        # generating on CPU keeps determinism device-independent.)
        white = (
            torch.rand(
                (batch_size, n_samples),
                generator=gen,
                dtype=torch.float32,
                device="cpu",
            )
            - 0.5
        ).to(device)

        # 2. Bandpass via torchaudio.functional.biquad (differentiable IIR).
        # `lfilter` (the kernel that biquad delegates to) hard-asserts
        # `dtype == fp32 || dtype == fp64`. Under bf16 AMP the surrounding
        # autocast context will silently downcast our fp32 white noise to
        # bf16, blowing up the assertion. Disable autocast for this scope
        # and force the input to fp32. The fp32 filtered output is cast
        # back to the surrounding pipeline dtype by `audio_system.forward`
        # (line 438-440).
        a_b0, a_b1, a_b2, a_a0, a_a1, a_a2 = self.aspirate_coeffs.tolist()
        f_b0, f_b1, f_b2, f_a0, f_a1, f_a2 = self.fricative_coeffs.tolist()

        with torch.amp.autocast(device_type=white.device.type, enabled=False):
            white_fp32 = white.float()
            aspirate = F.biquad(
                white_fp32, a_b0, a_b1, a_b2, a_a0, a_a1, a_a2,
            )
            fricative = F.biquad(
                white_fp32, f_b0, f_b1, f_b2, f_a0, f_a1, f_a2,
            )

        return aspirate, fricative


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    bank = NoiseFilterBank()
    print(f"NoiseFilterBank: sr={bank.sr}, seed={bank.seed}, "
          f"aspirate={bank.aspirate_freq} Hz, fricative={bank.fricative_freq} Hz, "
          f"q={bank.q}")
    print(f"aspirate coeffs (b0,b1,b2,a0,a1,a2) = "
          f"{bank.aspirate_coeffs.tolist()}")
    print(f"fricative coeffs (b0,b1,b2,a0,a1,a2) = "
          f"{bank.fricative_coeffs.tolist()}")

    # -- Test 1: shape ------------------------------------------------------
    N, B = 4800, 2
    asp, fri = bank(n_samples=N, batch_size=B)
    print(f"\n[shape] aspirate  = {tuple(asp.shape)} (expect ({B}, {N}))")
    print(f"[shape] fricative = {tuple(fri.shape)} (expect ({B}, {N}))")
    assert asp.shape == (B, N) and fri.shape == (B, N)

    # -- Test 2: finite -----------------------------------------------------
    print(f"[finite] aspirate finite={torch.isfinite(asp).all().item()}, "
          f"fricative finite={torch.isfinite(fri).all().item()}")

    # -- Test 3: determinism ------------------------------------------------
    asp2, fri2 = bank(n_samples=N, batch_size=B)
    asp_eq = torch.equal(asp, asp2)
    fri_eq = torch.equal(fri, fri2)
    print(f"[determ] same args → same output: aspirate={asp_eq}, "
          f"fricative={fri_eq}")
    assert asp_eq and fri_eq

    # -- Test 4: spectral peak ---------------------------------------------
    # We verify two things:
    #   (a) Analytic filter magnitude response |H(e^jω)| peaks at the
    #       design frequency. This is the tight (sub-Hz) check.
    #   (b) Empirical noise-output spectrum, smoothed by a running mean,
    #       peaks near the design frequency. With Q=0.5 the bandpass is
    #       broad (BW ≈ f₀/Q = 1000 Hz around f₀), so individual FFT
    #       bins fluctuate; smoothing is required for a clean peak.
    N_spec = 480_000   # 10 s
    asp_s, fri_s = bank(n_samples=N_spec, batch_size=1)
    spec_a = torch.fft.rfft(asp_s[0]).abs()
    spec_f = torch.fft.rfft(fri_s[0]).abs()
    freqs = torch.fft.rfftfreq(N_spec, d=1.0 / bank.sr)

    # Running-mean smoothing (~10 Hz averaging window).
    import torch.nn.functional as F_nn
    k = 51
    spec_a_s = F_nn.avg_pool1d(
        spec_a.view(1, 1, -1), kernel_size=k, stride=1, padding=k // 2
    ).view(-1)
    spec_f_s = F_nn.avg_pool1d(
        spec_f.view(1, 1, -1), kernel_size=k, stride=1, padding=k // 2
    ).view(-1)
    peak_a_hz = freqs[spec_a_s.argmax()].item()
    peak_f_hz = freqs[spec_f_s.argmax()].item()
    print(f"\n[spec] aspirate empirical peak ≈ {peak_a_hz:.1f} Hz "
          f"(design 500 Hz, Q=0.5 → broad)")
    print(f"[spec] fricative empirical peak ≈ {peak_f_hz:.1f} Hz "
          f"(design 1000 Hz, Q=0.5 → broad)")

    # Analytic filter magnitude check (precise).
    def _analytic_peak(coeffs):
        b0, b1, b2, a0, a1, a2 = coeffs
        w = 2.0 * math.pi * freqs / bank.sr
        ejw = torch.exp(-1j * w)
        ejw2 = torch.exp(-2j * w)
        H = (b0 + b1 * ejw + b2 * ejw2) / (a0 + a1 * ejw + a2 * ejw2)
        return freqs[H.abs().argmax()].item()

    apk = _analytic_peak(bank.aspirate_coeffs.tolist())
    fpk = _analytic_peak(bank.fricative_coeffs.tolist())
    print(f"[spec] aspirate |H| peak     = {apk:.2f} Hz (expect 500.00)")
    print(f"[spec] fricative |H| peak    = {fpk:.2f} Hz (expect 1000.00)")
    assert abs(apk - 500.0) < 2.0
    assert abs(fpk - 1000.0) < 2.0
    # Empirical peak is broad; ±300 Hz tolerance accounts for the Q=0.5 width.
    assert abs(peak_a_hz - 500.0) < 300.0
    assert abs(peak_f_hz - 1000.0) < 400.0

    # -- Test 5: stats ------------------------------------------------------
    print(f"\n[stats] aspirate  mean={asp.mean().item():+.4e}  "
          f"std={asp.std().item():.4f}  abs.max={asp.abs().max().item():.4f}")
    print(f"[stats] fricative mean={fri.mean().item():+.4e}  "
          f"std={fri.std().item():.4f}  abs.max={fri.abs().max().item():.4f}")

    print("\nOK")
