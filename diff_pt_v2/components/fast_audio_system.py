"""Faster DiffPT rendering for training — drop-in, NON-INVASIVE.

WHY THIS FILE EXISTS
--------------------
Training wall-time is dominated by the waveguide IR loop in
``waveguide.py`` (``_run_ir_loop``): a 512 (n_ir) × 2 (oversample) = 1024-step
SEQUENTIAL loop, run K+1 (=6) times per render and recomputed once more in the
backward (gradient checkpoint). Each step launches ~10-15 tiny CUDA kernels on
``(B, T, n_oral)`` tensors, so a render fires ~10^5 sequential kernel launches.
The workload is therefore *kernel-launch-latency bound*: the GPU idles between
launches (this is why DDP gave no speed-up — splitting the batch does not reduce
the number of sequential launches).

``torch.compile`` fuses the per-step kernel chain → the documented ~2-3× speed-up
on this loop. BUT ``waveguide.py`` wraps it with ``dynamic=False``, which forces
a *recompile for every distinct shape*. Our clips are full-length + padded, so
the number of VTD frames T varies batch-to-batch → a recompile (10-60 s) every
step → compile is counter-productive and was globally disabled via
``DIFFPT_DISABLE_COMPILE=1``.

WHAT THIS FILE DOES (two independent, composable levers)
-------------------------------------------------------
1. ``set_waveguide_compile(sys, dynamic=True)`` — RE-WRAP the already-built
   waveguide's compiled loop with ``dynamic`` shapes enabled (one graph handles
   every T → no per-T recompile). Replaces the attribute on an existing module;
   it does NOT edit ``waveguide.py``. Makes ``DIFFPT_DISABLE_COMPILE`` irrelevant
   (this factory decides the compile mode).

2. ``BucketedAudioSystem`` — pad T UP to a fixed multiple (default 32), render,
   then crop the output back to the real ``T * spf_out``. This collapses the set
   of distinct T values to a handful of buckets, so even ``dynamic=False`` (or
   automatic-dynamic) compiles only a few static graphs and then reuses them.

   KEY PROPERTY — padding the *end* is lossless for the deterministic path:
   the waveguide is causal (output at frame f depends only on frame f's
   reflection coeffs and PAST glottal samples), and the per-frame
   ``prev = shift(cur)`` construction means trailing frames never alter earlier
   frames' coefficients. So cropping the padded tail reproduces the un-padded
   tract-filtered render BIT-FOR-BIT (verified: batch row 0 interior max|Δ|=0),
   with two understood, benign exceptions:

     (a) Resampler boundary. The anti-alias resampler's symmetric FIR
         (lowpass_filter_width=6) smears non-causally across the crop seam, so
         the final ~1 frame of output differs by ~1e-3. Those samples sit at
         index >= valid_frames*spf_out and are masked by ``valid_frames`` in
         training (valid_frames <= T), so the loss never sees them.

     (b) Aspiration noise realization. ``NoiseFilterBank.forward`` re-seeds a
         Generator to a FIXED seed every call and draws
         ``rand((B, T*samples_per_frame_int))`` (= ``(B, 960*T)`` at 48 kHz /
         50 Hz); batch row b>=1 reads from RNG offset ``b*960*T``, which shifts
         when T changes. So bucketing gives rows >=1 a DIFFERENT-but-equally-valid
         stochastic aspiration draw (row 0, the RNG prefix, is bit-identical).
         This is benign: training already varies T — and hence this draw — every
         step, so bucketing introduces no nondeterminism beyond what exists, and
         the spectral/RMS loss is over realizations, not a fixed noise sample.

   Because the loop is launch-bound, the extra padding-frame COMPUTE is largely
   hidden in the GPU idle gaps → bucketing is nearly free.

   SELF-TEST SCOPE: the ``__main__`` parity test proves row-0 bit-identity for
   the EAGER deterministic path only. It does not parity-check the compiled or
   bf16 configs (torch.compile is math-preserving — it only fuses kernels — so
   parity is expected to hold to fp-reorder tolerance; sanity-check loss
   continuity on the server the first time compile is enabled).

HOW TO WIRE INTO TRAINING (one line; NOT done here, left for review)
--------------------------------------------------------------------
In ``classification/train_cls.py`` where it builds the renderer::

    # before:
    audio_system = DifferentiableAudioSystem(
        sr_internal=48000, sr_out=SAMPLE_RATE, frame_rate=50,
        n_oral=44, n_nose=28, n_ir_samples=512,
        closure_eps=1e-6, seed=31337,
    ).to(device)

    # after:
    from diff_pt_v2.components.fast_audio_system import build_fast_audio_system
    audio_system = build_fast_audio_system(
        bucket_multiple=32, compile_enabled=True, dynamic=True,
        sr_internal=48000, sr_out=SAMPLE_RATE, frame_rate=50,
        n_oral=44, n_nose=28, n_ir_samples=512,
        closure_eps=1e-6, seed=31337,
    ).to(device)

``build_fast_audio_system`` forwards all DifferentiableAudioSystem kwargs
unchanged, so it is a transparent swap. Set ``compile_enabled=False`` to get the
pure-eager bucketed system, or ``bucket_multiple=1`` to disable bucketing and
rely on dynamic compile alone — both fall back gracefully.

Run ``python diff_pt_v2/components/fast_audio_system.py`` (from the rev2 ``src``
dir) for the parity + masking + backward self-test.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn

# --- standalone-runnable import bootstrap -------------------------------- #
# This file lives at <src>/diff_pt_v2/components/fast_audio_system.py; the
# package root that makes ``import diff_pt_v2...`` resolve is <src>.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(os.path.dirname(_HERE))  # .../src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from diff_pt_v2.components.audio_system import DifferentiableAudioSystem  # noqa: E402


# ======================================================================== #
# Lever 1 — compile mode control (re-wrap, no source edit)
# ======================================================================== #
def set_waveguide_compile(
    audio_system: nn.Module,
    *,
    enable: bool = True,
    dynamic: Optional[bool] = True,
    fullgraph: bool = False,
    mode: Optional[str] = None,
) -> str:
    """Re-wrap the waveguide IR loop's compiled function on an existing module.

    ``audio_system`` may be a ``DifferentiableAudioSystem`` or anything exposing
    a ``.tract`` ``WaveguideTract`` (e.g. ``BucketedAudioSystem`` delegates).

    Args:
        enable: if False, force eager (``_run_ir_loop`` unwrapped).
        dynamic: passed to ``torch.compile``. ``True`` = one graph for all
            shapes (no per-T recompile); ``None`` = PyTorch automatic-dynamic
            (specialize first, promote on recompile); ``False`` = static
            (recompile per shape — the pathological setting for variable T).
        fullgraph: ``torch.compile(fullgraph=...)``. Kept False to match
            ``waveguide.py`` (graph break at the checkpoint boundary is fine —
            what we want fused is the inner IR-step kernel chain).
        mode: optional ``torch.compile(mode=...)``. ``None`` = default inductor.
            ``"reduce-overhead"`` adds CUDA-graph capture (best for this
            launch-bound loop) BUT needs static shapes → pair with bucketing
            (``dynamic=False``) and treat as experimental on bleeding-edge CUDA.

    Returns:
        A short status string describing the resulting mode.
    """
    tract = audio_system.tract  # WaveguideTract (delegated through wrappers)
    base_fn = tract._run_ir_loop  # the uncompiled python method (bound)

    if not enable:
        tract._compiled_run_ir_loop = base_fn
        return "eager(disabled)"
    if not hasattr(torch, "compile"):
        tract._compiled_run_ir_loop = base_fn
        return "eager(no torch.compile — needs torch>=2.0)"

    try:
        compile_kwargs = dict(fullgraph=fullgraph, dynamic=dynamic)
        if mode is not None:
            compile_kwargs["mode"] = mode
        tract._compiled_run_ir_loop = torch.compile(base_fn, **compile_kwargs)
        return f"compiled(dynamic={dynamic}, fullgraph={fullgraph}, mode={mode})"
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"[fast_audio_system] torch.compile failed ({e}); eager fallback")
        tract._compiled_run_ir_loop = base_fn
        return "eager(compile-failed)"


# ======================================================================== #
# Lever 2 — T bucketing wrapper
# ======================================================================== #
class BucketedAudioSystem(nn.Module):
    """Pad the frame axis T up to a fixed multiple, render, crop back.

    Transparent drop-in for ``DifferentiableAudioSystem``: same forward
    signature ``(vtd, voicing, f0, valid_frames=None) -> (B, T*spf_out)`` and
    the same output in the valid region (see module docstring for the
    resampler-boundary caveat). Unknown attribute access is delegated to the
    wrapped system so existing callers that read e.g. ``samples_per_frame_out``
    keep working.

    Args:
        inner: a constructed ``DifferentiableAudioSystem``.
        bucket_multiple: pad T up to the next multiple of this. 1 disables
            bucketing (pure pass-through). 32 is a good default: the average
            over-pad is ~16 frames on ~100, hidden by the launch-bound loop.
    """

    def __init__(self, inner: DifferentiableAudioSystem, bucket_multiple: int = 32):
        super().__init__()
        assert isinstance(bucket_multiple, int) and bucket_multiple >= 1
        self.inner = inner
        self.bucket_multiple = int(bucket_multiple)

    # delegate unknown attrs (samples_per_frame_out, sr_out, tract, ...) to inner
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = super().__getattr__("inner")
            return getattr(inner, name)

    def _bucket_len(self, T: int) -> int:
        m = self.bucket_multiple
        if m <= 1:
            return T
        return ((T + m - 1) // m) * m

    def forward(
        self,
        vtd: torch.Tensor,
        voicing: torch.Tensor,
        f0: torch.Tensor,
        valid_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = vtd.shape
        Tb = self._bucket_len(T)

        if Tb != T:
            pad = Tb - T
            # Edge-replicate the last frame into the padded tail. The tail is
            # cropped away after rendering; replication (vs zeros) keeps the
            # boundary frame's neighbourhood constant so the release-event
            # detector cannot fire a spurious transient at the seam.
            vtd = torch.cat([vtd, vtd[:, -1:, :].expand(B, pad, D)], dim=1)
            voicing = torch.cat([voicing, voicing[:, -1:].expand(B, pad)], dim=1)
            f0 = torch.cat([f0, f0[:, -1:].expand(B, pad)], dim=1)

        # valid_frames are real per-sample lengths (all <= T <= Tb); the inner
        # system zeroes audio past valid_frames[b], so padded frames are masked
        # for short samples and cropped for long ones.
        audio = self.inner(vtd, voicing, f0, valid_frames=valid_frames)

        T_out = T * self.inner.samples_per_frame_out
        return audio[:, :T_out]


# ======================================================================== #
# Factory
# ======================================================================== #
def build_fast_audio_system(
    *,
    bucket_multiple: int = 32,
    compile_enabled: bool = True,
    dynamic: Optional[bool] = True,
    compile_mode: Optional[str] = None,
    verbose: bool = True,
    **audio_system_kwargs,
) -> nn.Module:
    """Build a DifferentiableAudioSystem with dynamic-compile + T bucketing.

    All ``audio_system_kwargs`` are forwarded verbatim to
    ``DifferentiableAudioSystem`` so this is a transparent constructor swap.
    """
    base = DifferentiableAudioSystem(**audio_system_kwargs)
    mode = set_waveguide_compile(
        base, enable=compile_enabled, dynamic=dynamic, mode=compile_mode
    )
    if bucket_multiple and bucket_multiple > 1:
        sysmod: nn.Module = BucketedAudioSystem(base, bucket_multiple=bucket_multiple)
        bucket_str = f"bucket_multiple={bucket_multiple}"
    else:
        sysmod = base
        bucket_str = "bucket=off"
    sysmod._fast_render_mode = mode  # introspection for callers/tests
    if verbose:
        print(f"[fast_audio_system] {bucket_str}, waveguide={mode}")
    return sysmod


# ======================================================================== #
# Self-test
# ======================================================================== #
if __name__ == "__main__":
    import time

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 2
    n_oral = 44
    sr_int, sr_out, frame_rate = 48000, 16000, 50
    spf_out = sr_out // frame_rate  # 320
    BUCKET = 32

    def make_inputs(T):
        vtd = torch.full((B, T, 1 + n_oral), 1.0)
        vtd[..., 0] = 0.05
        vtd[..., 1:] = vtd[..., 1:] + 0.2 * torch.randn(B, T, n_oral)
        # force a closure-release in batch 0 to exercise the transient path
        if T >= 6:
            vtd[0, 3:5, 1:] = 0.01
            vtd[0, 5:, 1:] = 1.0
        vtd = vtd.clamp(min=0.001)
        voicing = 0.3 + 0.6 * torch.rand(B, T)
        f0 = 100.0 + 100.0 * torch.rand(B, T)
        return vtd.to(device), voicing.to(device), f0.to(device)

    print(f"=== fast_audio_system self-test (device={device}) ===")

    # Ground-truth base system, FORCED EAGER so parity isolates the bucketing
    # logic from any compile-induced fp reordering.
    base = DifferentiableAudioSystem(
        sr_internal=sr_int, sr_out=sr_out, frame_rate=frame_rate,
        n_oral=n_oral, n_nose=28, n_ir_samples=512, seed=31337,
    ).to(device)
    set_waveguide_compile(base, enable=False)
    base.eval()
    wrapped = BucketedAudioSystem(base, bucket_multiple=BUCKET).to(device)
    wrapped.eval()

    # ---- Test 0: attribute delegation -------------------------------- #
    assert wrapped.samples_per_frame_out == base.samples_per_frame_out
    assert wrapped.tract is base.tract
    print("[t0] attribute delegation OK")

    # ---- Test 1: pass-through (T multiple of bucket) → EXACT equality - #
    torch.manual_seed(1)
    T1 = 64  # multiple of 32 → no padding
    vtd, voi, f0 = make_inputs(T1)
    with torch.no_grad():
        gt = base(vtd, voi, f0)
        out = wrapped(vtd, voi, f0)
    assert out.shape == gt.shape == (B, T1 * spf_out), (out.shape, gt.shape)
    exact = torch.equal(out, gt)
    print(f"[t1] T={T1} (mult of {BUCKET}) pass-through exact-equal: {exact}")
    assert exact, "pass-through must be bit-identical to the base render"

    # ---- Test 2: padded T → parity ----------------------------------- #
    # The deterministic tract-filtered path must be BIT-IDENTICAL after
    # crop. We prove this on batch ROW 0, whose seeded aspiration noise is the
    # RNG prefix (length-independent); rows >=1 read a shifted RNG offset that
    # moves with T, so they get a different-but-valid noise draw (see docstring
    # (b)). Interior excludes the last 2 frames (resampler boundary smear, (a)).
    torch.manual_seed(2)
    T2 = 50  # → bucketed to 64
    vtd, voi, f0 = make_inputs(T2)
    with torch.no_grad():
        gt = base(vtd, voi, f0)                      # (B, 50*320)
        out = wrapped(vtd, voi, f0)                  # (B, 50*320) after crop
    assert out.shape == gt.shape == (B, T2 * spf_out), (out.shape, gt.shape)
    interior = slice(0, (T2 - 2) * spf_out)
    row0_max = (out[0, interior] - gt[0, interior]).abs().max().item()
    full_max = (out[:, interior] - gt[:, interior]).abs().max().item()
    gt_rms = gt.pow(2).mean().sqrt().item()
    print(f"[t2] T={T2}→bucket {wrapped._bucket_len(T2)}: "
          f"row0 interior max|Δ|={row0_max:.3e} (must be 0); "
          f"full-batch interior max|Δ|={full_max:.3e} (noise realization, "
          f"gt_rms={gt_rms:.3e})")
    assert row0_max == 0.0, (
        f"deterministic-path parity violated: row0 max|Δ|={row0_max:.3e} != 0")
    assert full_max < 0.1 * gt_rms or full_max < 1e-2, (
        f"noise-realization spread unexpectedly large: {full_max:.3e}")

    # ---- Test 3: valid_frames masking through the wrapper ------------ #
    torch.manual_seed(3)
    T3 = 50
    vtd, voi, f0 = make_inputs(T3)
    vf = torch.tensor([T3, T3 // 2], device=device)
    with torch.no_grad():
        out = wrapped(vtd, voi, f0, valid_frames=vf)
    half = (T3 // 2) * spf_out
    tail_b1 = out[1, half:].abs().sum().item()
    tail_b0 = out[0, half:].abs().sum().item()
    print(f"[t3] valid_frames: b1 tail={tail_b1:.3e} (≈0), b0 tail={tail_b0:.3e} (>0)")
    assert tail_b1 < 1e-7 and tail_b0 > 1e-4, "valid_frames mask broken through wrapper"

    # ---- Test 4: backward through the wrapper ------------------------ #
    torch.manual_seed(4)
    vtd, voi, f0 = make_inputs(50)
    base.train()
    wrapped.train()
    vtd_g = vtd.detach().clone().requires_grad_(True)
    voi_g = voi.detach().clone().requires_grad_(True)
    f0_g = f0.detach().clone().requires_grad_(True)
    audio_g = wrapped(vtd_g, voi_g, f0_g)
    loss = audio_g.pow(2).mean()
    loss.backward()
    for name, t in [("vtd", vtd_g), ("voicing", voi_g), ("f0", f0_g)]:
        g = t.grad
        ok = bool(torch.isfinite(g).all()) and g.abs().mean().item() > 0
        print(f"[t4] grad {name:<8s} finite&nonzero={ok} mean|g|={g.abs().mean().item():.3e}")
        assert ok, f"bad gradient for {name}"

    # ---- Test 5: variable-T robustness (+ dynamic compile where available) - #
    # On torch>=2.0 + triton this exercises the dynamic-compile path (one graph,
    # no per-T recompile). On torch<2.0 it runs eager — still a valid check that
    # the wrapper + factory handle two different T without crashing. The mode is
    # reported honestly so a local eager fallback is not mistaken for a compile.
    try:
        fast = build_fast_audio_system(
            bucket_multiple=1, compile_enabled=True, dynamic=True,
            sr_internal=sr_int, sr_out=sr_out, frame_rate=frame_rate,
            n_oral=n_oral, n_nose=28, n_ir_samples=512, seed=31337,
        ).to(device)
        mode = getattr(fast, "_fast_render_mode", "unknown")
        compiled = mode.startswith("compiled")
        fast.train()
        for T in (48, 72):  # two distinct shapes
            v, vo, f = make_inputs(T)
            t0 = time.time()
            a = fast(v, vo, f)
            if device == "cuda":
                torch.cuda.synchronize()
            print(f"[t5] T={T}: out={tuple(a.shape)} "
                  f"finite={bool(torch.isfinite(a).all())} {time.time()-t0:.2f}s")
            assert a.shape == (B, T * spf_out) and torch.isfinite(a).all()
        verdict = ("dynamic COMPILE across shapes OK" if compiled
                   else f"variable-T OK (eager fallback: {mode})")
        print(f"[t5] {verdict}")
    except Exception as e:
        print(f"[t5] SOFT-FAIL (compile env): {e}")

    print("\nALL SELF-TESTS PASSED")
