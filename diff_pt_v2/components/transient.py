"""Closure-release transient pulses — PyTorch port of `v1/js/src/models/tract.js`
lines 211-240 (`addTransient` + `processTransients`).

In the JS Pink Trombone, a transient pulse is emitted when an oral
constriction (`diameter == 0`) clears while the velopharyngeal port is
closed (`noseA[0] < 0.05`). The pulse has the form

    amplitude(t) = strength * 2^(-exponent * t),   t ∈ [0, lifetime)

with the JS defaults `strength = 0.3`, `exponent = 200`, `lifetime = 0.2 s`.
JS injects half the amplitude into the rightward (`R`) and half into the
leftward (`L`) waveguide buffers at the constriction position (lines
228-229). For the Option-2 frame-IR port we treat the transient as an
additive audio-rate signal, summed at the release sample index.

Two public symbols:

- `closure_transient(release_times, n_samples_total, …)`: build the
  batched audio-rate transient signal from a set of release sample
  indices.
- `detect_release_events(d_min_per_frame, samples_per_frame, threshold)`:
  detect frame indices where the minimum oral diameter crosses **upward**
  through `threshold` (closure → open) and convert to sample indices for
  feeding `closure_transient`.

Differentiability
-----------------
- `closure_transient` is differentiable w.r.t. `strength` (a Python
  scalar by default — promote to a tensor if you need grad). It is *not*
  differentiable w.r.t. the integer `release_times` indices. If we ever
  need grad through release timing we can switch to a soft Gaussian
  placement; the current hard scatter is the additive feature the
  tract model expects.
- `detect_release_events` performs a hard `<`/`>=` comparison, so its
  output (sample indices) is non-differentiable. This is consistent with
  the JS source which performs identical hard comparisons (tract.js:248,
  349-356).

Refs: design/01_js_audit.md §C (transient pulses, lines 211-240),
design/02_port_strategy.md (`components/transient.py` section),
tract.js (211-240).
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch


# ----------------------------------------------------------------------------
# closure_transient
# ----------------------------------------------------------------------------
def closure_transient(
    release_times: torch.Tensor,
    n_samples_total: int,
    sr: int = 48000,
    strength: float = 0.3,
    exponent: float = 200.0,
    lifetime: float = 0.2,
) -> torch.Tensor:
    """Emit exponentially-decaying transient bursts at each release sample.

    Each (batch, release) event contributes

        amp(t - t_i) = strength * 2^(-exponent * (t - t_i) / sr)

    for `t_i <= t < t_i + lifetime * sr`, zero elsewhere. Multiple events
    in the same batch element are summed (matches JS `R[pos] += amp/2`
    accumulation semantics, but here we treat the transient as a
    waveform-domain additive feature, not split between R/L buffers).

    Args:
        release_times: (B, n_releases_max) int / long tensor of release
            sample indices. Use a sentinel value < 0 to mark "no event"
            in unused slots; those rows are ignored.
        n_samples_total: total audio length (samples @ sr) of the output
            signal.
        sr: audio sample rate (Hz). JS uses 48000 (tract.js:7).
        strength: peak amplitude at the release sample. JS default 0.3
            (tract.js:217).
        exponent: decay exponent (base 2). JS default 200 (tract.js:218).
        lifetime: burst lifetime in seconds. JS default 0.2 (tract.js:216).

    Returns:
        transient_signal: (B, n_samples_total) — sum of all transient
            bursts per batch element.
    """
    if release_times.dim() != 2:
        raise ValueError(
            f"release_times must be 2-D (B, n_releases_max); got "
            f"{tuple(release_times.shape)}"
        )

    B, K = release_times.shape
    device = release_times.device
    # We materialize the output in float32 — strength and exponent are
    # python scalars by default; promote if the caller passed tensors.
    if isinstance(strength, torch.Tensor):
        out_dtype = strength.dtype
    else:
        out_dtype = torch.float32

    # ----- Pre-compute the decay kernel (single 1-D buffer reused) ----- #
    # kernel[k] = strength * 2^(-exponent * k / sr),  k = 0..L-1.
    L = int(math.ceil(lifetime * sr))
    if L <= 0 or n_samples_total <= 0 or K == 0:
        return torch.zeros((B, n_samples_total), dtype=out_dtype, device=device)

    k_arange = torch.arange(L, dtype=torch.float32, device=device)
    # 2^x = exp(x * ln 2). Using exp keeps a single device kernel.
    kernel = float(strength) * torch.exp(
        -float(exponent) * k_arange / float(sr) * math.log(2.0)
    )                                                             # (L,)
    kernel = kernel.to(out_dtype)

    # ----- Vectorized placement via scatter-add into a padded buffer --- #
    # For each event, write `kernel` starting at sample `t_i`. We avoid a
    # Python loop over events by:
    #   1. Building per-event (batch, sample) index grids of shape
    #      (B, K, L) such that sample = t_i + k for k in 0..L-1.
    #   2. Building per-event amplitudes of the same shape: kernel[k]
    #      broadcast across (B, K).
    #   3. Masking out-of-range entries (t < 0 sentinel, or
    #      t + k >= n_samples_total).
    #   4. Flattening to 1-D index/value pairs and scatter-adding into a
    #      per-batch flat buffer.

    # release_times: (B, K) → (B, K, 1); kernel: (L,) → (1, 1, L).
    rt = release_times.to(torch.long).unsqueeze(-1)               # (B, K, 1)
    k_idx = torch.arange(L, dtype=torch.long, device=device).view(1, 1, L)
    sample_idx = rt + k_idx                                       # (B, K, L)

    # Validity mask: t_i must be >= 0 (sentinel for "no event"); each
    # absolute sample idx must be in [0, n_samples_total).
    valid_event = (rt >= 0)                                       # (B, K, 1)
    in_range = (sample_idx >= 0) & (sample_idx < n_samples_total) # (B, K, L)
    mask = valid_event & in_range                                 # (B, K, L)

    # Clamp indices to a safe value for the scatter-add. Masked entries
    # are written with 0 amplitude so the clamp doesn't pollute the
    # output even though they share indices with legitimate writes.
    safe_idx = sample_idx.clamp(min=0, max=n_samples_total - 1)   # (B, K, L)

    # Amplitudes per (B, K, L). Broadcast kernel.
    amp = kernel.view(1, 1, L).expand(B, K, L)                    # (B, K, L)
    amp = amp * mask.to(amp.dtype)                                # zero out invalid

    # Flatten the (K, L) per-batch event dimension.
    safe_idx_flat = safe_idx.reshape(B, K * L)                    # (B, K*L)
    amp_flat = amp.reshape(B, K * L)                              # (B, K*L)

    out = torch.zeros((B, n_samples_total), dtype=out_dtype, device=device)
    out.scatter_add_(dim=1, index=safe_idx_flat, src=amp_flat)
    return out


# ----------------------------------------------------------------------------
# detect_release_events
# ----------------------------------------------------------------------------
def detect_release_events(
    d_min_per_frame: torch.Tensor,
    samples_per_frame: int = 960,
    threshold: float = 0.05,
    velum_per_frame: Optional[torch.Tensor] = None,
    velum_area_threshold: float = 0.05,
) -> List[torch.Tensor]:
    """Find frame indices where d_min crosses upward through `threshold`.

    A "release" event at frame `t` is defined by

        d_min[t-1] <  threshold   AND   d_min[t] >= threshold

    matching the JS closure-clears-then-emit-transient pattern (tract.js
    addTransient is invoked when an obstruction `d <= 0` clears, line
    349-356 in handleTouches). The resulting sample index is
    `t * samples_per_frame` — i.e. the start of the frame at which the
    upward crossing was observed.

    Velum gate (optional): the original synthesiser fires `addTransient`
    only when the velopharyngeal port is closed (`noseA[0] < 0.05`, where
    `noseA[0] = velum^2`), i.e. the *oral* configuration. When
    `velum_per_frame` is supplied, a release is emitted only if the velum
    is closed at the release frame, suppressing the physically-incorrect
    firing on nasal (open-velum) releases. When it is `None`, the gate is
    off (legacy behaviour — fires on every oral near-closure release).

    Args:
        d_min_per_frame: (B, T) — per-frame minimum oral diameter (mm).
        samples_per_frame: audio samples per VTD frame. JS uses 960
            (= 48000 / 50 fps, audioSystem.js:42).
        threshold: closure threshold (mm). JS uses 0.05 for the
            velopharyngeal port (tract.js, audit §C), and we re-use the
            same threshold here for oral closure detection.
        velum_per_frame: (B, T) per-frame velum diameter, or None. When
            given, gates firing on `velum^2 < velum_area_threshold`.
        velum_area_threshold: closed-port area threshold. JS uses 0.05 on
            `noseA[0] = velum^2`.

    Returns:
        List of length B. Each element is a 1-D `torch.long` tensor of
        release sample indices for that batch row. Tensors may be empty
        if no releases are detected.
    """
    if d_min_per_frame.dim() != 2:
        raise ValueError(
            f"d_min_per_frame must be 2-D (B, T); got "
            f"{tuple(d_min_per_frame.shape)}"
        )

    B, T = d_min_per_frame.shape
    if T < 2:
        # Cannot detect a crossing without at least 2 frames.
        return [torch.empty((0,), dtype=torch.long,
                            device=d_min_per_frame.device)
                for _ in range(B)]

    # Hard threshold mask. Cast to bool for logical ops.
    above = (d_min_per_frame >= threshold)                        # (B, T)
    # Upward crossing at frame t requires below at t-1 AND above at t.
    below_prev = ~above[:, :-1]                                   # (B, T-1)
    above_curr = above[:, 1:]                                     # (B, T-1)
    cross = below_prev & above_curr                               # (B, T-1)
    # Velum gate: match JS addTransient, which fires only when the
    # velopharyngeal port is closed (noseA[0] = velum^2 < 0.05 — the oral
    # configuration). Evaluated at the *release* (current) frame, which is
    # the [:, 1:] slice aligned with `above_curr`. Suppresses the
    # physically-incorrect firing on nasal (open-velum) releases.
    if velum_per_frame is not None:
        velum_closed = (velum_per_frame * velum_per_frame) < velum_area_threshold  # (B, T)
        cross = cross & velum_closed[:, 1:]                       # (B, T-1)
    # Map back to frame indices in [1, T-1].

    out: List[torch.Tensor] = []
    for b in range(B):
        # nonzero returns indices in [0, T-2]; add 1 to get the "current"
        # frame index where the crossing was completed.
        frame_idx = cross[b].nonzero(as_tuple=False).squeeze(-1) + 1
        sample_idx = (frame_idx.to(torch.long) * int(samples_per_frame))
        out.append(sample_idx)
    return out


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    print("=== closure_transient ===")
    # 1 batch, 1 release at sample 1000, n_samples_total = 10000.
    rt = torch.tensor([[1000]], dtype=torch.long)
    sig = closure_transient(rt, n_samples_total=10000, sr=48000,
                            strength=0.3, exponent=200.0, lifetime=0.2)
    print(f"shape = {tuple(sig.shape)} (expect (1, 10000))")
    assert sig.shape == (1, 10000)
    print(f"finite = {torch.isfinite(sig).all().item()}")

    # At sample 1000 the burst starts at t=0 → amp = strength * 2^0 = 0.3.
    peak_val = sig[0, 1000].item()
    print(f"sig[0, 1000] = {peak_val:.6f} (expect ~0.3)")
    assert abs(peak_val - 0.3) < 1e-5

    # The argmax should fall at the release sample.
    argmax_idx = sig[0].argmax().item()
    print(f"argmax     = {argmax_idx} (expect 1000)")
    assert argmax_idx == 1000

    # Decay sanity: amp at t = lifetime/2 should be strength * 2^(-100*0.1*sr/sr)
    # → strength * 2^(-10) ≈ 0.3 * 9.77e-4 ≈ 2.93e-4.
    mid = 1000 + int(0.1 * 48000)   # t = 0.1 s, k=4800
    expected_mid = 0.3 * 2.0 ** (-200.0 * (mid - 1000) / 48000.0)
    print(f"sig[0, {mid}] = {sig[0, mid].item():.4e}  (expect {expected_mid:.4e})")
    assert abs(sig[0, mid].item() - expected_mid) < 1e-6

    # Beyond release + lifetime*sr should be zero (kernel has finite support).
    far = 1000 + int(0.2 * 48000) + 100
    if far < 10000:
        print(f"sig[0, {far}] = {sig[0, far].item():.4e}  (expect 0.0)")
        assert abs(sig[0, far].item()) < 1e-12

    # Pre-release samples should be zero.
    print(f"sig[0, 999]  = {sig[0, 999].item():.4e}  (expect 0.0)")
    assert abs(sig[0, 999].item()) < 1e-12

    # ---- Multi-release / multi-batch sanity --------------------------- #
    rt2 = torch.tensor([[100, 5000], [2000, -1]], dtype=torch.long)
    sig2 = closure_transient(rt2, n_samples_total=10000)
    print(f"\nmulti shape  = {tuple(sig2.shape)} (expect (2, 10000))")
    print(f"row0 peak @ 100  = {sig2[0, 100].item():.4f} (expect 0.3)")
    print(f"row0 peak @ 5000 = {sig2[0, 5000].item():.4f} (expect 0.3)")
    print(f"row1 peak @ 2000 = {sig2[1, 2000].item():.4f} (expect 0.3)")
    # The sentinel slot (-1) should not have placed anything. Pick a
    # sample that's beyond the release-at-2000 burst lifetime
    # (2000 + 9600 = 11600, so any index < 2000 is safe).
    print(f"row1 sample 1999 = {sig2[1, 1999].item():.4e} (expect 0.0)")
    assert abs(sig2[0, 100].item() - 0.3) < 1e-5
    assert abs(sig2[0, 5000].item() - 0.3) < 1e-5
    assert abs(sig2[1, 2000].item() - 0.3) < 1e-5
    assert abs(sig2[1, 1999].item()) < 1e-12

    print("\n=== detect_release_events ===")
    # Spec test: [0.05, 0.02, 0.0, 0.0, 0.08, 0.5]
    #   t=0: no prior → no event
    #   t=1: d[0]=0.05 not < 0.05 → no event
    #   t=2: d[1]=0.02 < 0.05, d[2]=0.00 not >= 0.05 → no event
    #   t=3: d[2]=0.00 < 0.05, d[3]=0.00 not >= 0.05 → no event
    #   t=4: d[3]=0.00 < 0.05, d[4]=0.08 >= 0.05 → release at sample 4*960=3840
    #   t=5: d[4]=0.08 not < 0.05 → no event
    d_min = torch.tensor([[0.05, 0.02, 0.0, 0.0, 0.08, 0.5]])
    releases = detect_release_events(d_min, samples_per_frame=960, threshold=0.05)
    print(f"len(releases) = {len(releases)} (expect 1)")
    print(f"releases[0]   = {releases[0].tolist()} (expect [3840])")
    assert releases[0].tolist() == [3840]

    # Multi-batch test.
    d_min_b = torch.tensor([
        [0.05, 0.02, 0.0,  0.0,  0.08, 0.5],   # release at frame 4
        [0.5,  0.5,  0.5,  0.5,  0.5,  0.5],   # no transitions
        [0.0,  0.1,  0.0,  0.2,  0.0,  0.3],   # releases at frames 1, 3, 5
    ])
    rel_b = detect_release_events(d_min_b, samples_per_frame=960, threshold=0.05)
    print(f"\nbatch=3 results:")
    for i, r in enumerate(rel_b):
        print(f"  row {i}: {r.tolist()}")
    assert rel_b[0].tolist() == [3840]
    assert rel_b[1].tolist() == []
    assert rel_b[2].tolist() == [960, 2880, 4800]

    # ---- End-to-end: detection feeds closure_transient ---------------- #
    print("\n=== integration ===")
    # Use d_min_b from above. Pad each row's release list to the max
    # length with -1 sentinels and stack.
    max_k = max(len(r) for r in rel_b)
    rt_pad = torch.full((len(rel_b), max(max_k, 1)), -1, dtype=torch.long)
    for i, r in enumerate(rel_b):
        if r.numel() > 0:
            rt_pad[i, :r.numel()] = r
    print(f"padded release indices:\n{rt_pad.tolist()}")
    # 6 frames * 960 samples/frame = 5760 total.
    total = 6 * 960
    sig_full = closure_transient(rt_pad, n_samples_total=total)
    print(f"output shape = {tuple(sig_full.shape)} (expect ({len(rel_b)}, {total}))")
    print(f"finite       = {torch.isfinite(sig_full).all().item()}")
    print(f"row 0 peak   = {sig_full[0].max().item():.4f} @ {sig_full[0].argmax().item()}")
    print(f"row 1 peak   = {sig_full[1].max().item():.4e} (expect 0 — no events)")
    print(f"row 2 peak   = {sig_full[2].max().item():.4f}")

    print("\nOK")
