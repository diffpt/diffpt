"""Waveguide vocal tract — PyTorch port of `v1/js/src/models/tract.js`.

**Run 16 (2026-06-05) speed-up**: the IR loop method ``_run_ir_loop`` is
wrapped with ``torch.compile(fullgraph=False, dynamic=False)`` for ~2-3×
speedup on the inner sequential loop. The first forward call after
construction triggers a compilation trace (typically 10-60 s on modern
GPUs); subsequent calls reuse the compiled graph.

WARNING (measured 2026-08-11): the two figures above were never verified at
the current 512-step configuration and are wrong there.  Compilation took
362 s on a local CPU and had not finished after more than ten minutes on a
server GPU, because Dynamo fully unrolls range(512) into a graph of 5000+
nodes.  See the torch.compile comment in __init__ for the detail.
**Keep DIFFPT_DISABLE_COMPILE=1.** Set the environment
variable ``DIFFPT_DISABLE_COMPILE=1`` to disable (e.g., for debugging or
on machines without Triton).

**Sub-frame K-anchor IR interpolation** (current scheme):

Each VTD frame is split into K sub-frames. We compute (K+1) anchor IRs
per frame at lambdas [0, 1/K, 2/K, …, 1]. Within sub-frame s of frame f,
the audio output is a linear cross-fade between the IR_s and IR_{s+1}
convolution outputs at audio rate:

    y(t) = (1-α(t)) · conv(g, IR_{f,s})(t) + α(t) · conv(g, IR_{f,s+1})(t)
    where α(t) = (t mod (spf/K)) / (spf/K)

**Continuity property (no 50 Hz artifact)**:
- Sub-frame boundary s→s+1 within frame f: at α=1 we use IR_{f,s+1};
  at α=0 of the next sub-frame we use IR_{f,s+1} as the "lower" anchor.
  Effective IR is continuous.
- Frame boundary f→f+1: last sub-frame's upper anchor is IR_{f,K} which
  uses lambda=1 i.e. r=cur_f. First sub-frame's lower anchor of frame
  f+1 is IR_{f+1,0} which uses lambda=0 i.e. r=prev_{f+1}=cur_f (by the
  prev=shift(cur) construction). So IR_{f,K} = IR_{f+1,0} → continuous.

Hence the effective time-varying impulse response is continuous in t.
No frame-rate (50 Hz) discontinuity → no 50 Hz line in output spectrum.

**Approximation vs JS PT**:
JS PT runs a single stateful waveguide whose reflection coefficients are
sample-level lambda-interpolated; state evolves under continuously-
varying dynamics. Our K-anchor scheme runs (K+1) parallel frozen-r
waveguides per frame and linearly blends their outputs. The two are
identical to first order in Δr per sub-frame, with O((Δr/K)²) higher-
order error. For typical speech articulation rates (slow, |Δr| << 1)
and K=5 (default), the approximation is well below audible / loss-
gradient thresholds.

44 oral segments + 28 nose segments (defaults), 3-way nose-velar
junction coupling, glottis/lip terminations, 2× oversampled within
each IR sample (matching JS audioSystem.js).

History:
- v1 "per-frame IR" (K=1, no cross-fade): hard-switch at frame boundary
  → +10 dB 50 Hz artifact. Used Run 1-14.
- v2 "sample-level continuous" (exact JS-faithful, stateful waveguide):
  50 Hz fixed but 50× slower → infeasible for training iteration. Used
  briefly 2026-06-03 then replaced.
- v3 "sub-frame K-anchor IR interp" (this file): 50 Hz fixed AND fast.
  Default K=5 balances fidelity and speed.

Key JS-faithfulness:
- Internal junction (calc_junctions, tract.js:137):
  `r = prev*(1-λ) + cur*λ` — normal direction
- 3-way nose junction (runStep, tract.js:188/190/192):
  `r = cur*(1-λ) + prev*λ` — REVERSED relative to internal (JS quirk)
- Inner nose junction (calc_nose_junctions, tract.js:154-162):
  no lambda — uses r_nose directly

Key change vs JS source:
- JS line 309 hard-clamps `r = 0.999` when `A == 0` (closure). We
  smoothly regularize via `r = (A1-A2) / (A1+A2+closure_eps)`.

Refs:
    tract.js:133-142  calc_junctions(lambda)
    tract.js:154-162  calc_nose_junctions() (no lambda)
    tract.js:174-203  runStep(glottal, fric, lambda)
    tract.js:300-322  calculateReflections() (save cur→prev)
"""
from __future__ import annotations

import os

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


# Run 16 (2026-06-05): allow disabling torch.compile via env var (debugging /
# old PyTorch / no-Triton systems). Default: enabled.
_DISABLE_COMPILE = bool(int(os.environ.get("DIFFPT_DISABLE_COMPILE", "0")))


class WaveguideTract(nn.Module):
    """Digital waveguide vocal tract with sub-frame K-anchor IR interp.

    Args:
        n_oral: oral tract segments (JS default 44).
        n_nose: nose segments. JS Pink Trombone uses
            noseLength = floor(28 * n / 44); with n_oral = 44 this is 28.
            (Earlier ports used 18 from a porting-time miscount of this
            formula; 28 is the value the JS corpus was generated with.)
        samples_per_frame: internal samples per VTD frame. With
            sr_int=48 kHz and frame_rate=50 Hz this is 960.
        n_ir_samples: per-anchor IR length in samples. 512 ≈ 10.7 ms at
            48 kHz — captures ≥99.9% of the response energy for open and
            interior-constriction tracts (log-spectral error <0.3 dB vs a
            far longer IR), but a sustained lip near-closure rings ~40 ms
            and is under-resolved (~26% tail energy, ~1.5 dB; see
            tests/verify_dsp_audit.py). Kept at 512 as a speed trade-off:
            the per-anchor convolution cost scales with this length.
        n_subframe: K = sub-frames per frame. Default 5. Each frame uses
            K+1 anchor IRs at lambdas [0, 1/K, …, 1]. `samples_per_frame`
            must be divisible by K.
        damping: mouth-side propagation fade per sample (JS lines 148-149).
        oversample: waveguide sub-steps per IR sample. JS uses 2.
        r_glottal: left (glottis) boundary reflection (JS:19, 181).
        r_lip: right (lip) boundary reflection (JS:20, 182).
        closure_eps: smooth-closure ε added to reflection denominators.

    Forward:
        vtd: (B, T, 1 + n_oral)
            vtd[..., 0]   = velum diameter (mm).
            vtd[..., 1:]  = oral tract diameters (mm), length n_oral.
        glottal_audio: (B, T * samples_per_frame)
            Audio-rate glottal source.

    Returns:
        audio_internal: (B, T * samples_per_frame)
            Tract-filtered audio at internal sample rate. The
            `output_scale` (default 0.125) is applied downstream by
            DifferentiableAudioSystem.
    """

    def __init__(
        self,
        n_oral: int = 44,
        n_nose: int = 28,
        samples_per_frame: int = 960,
        n_ir_samples: int = 512,
        n_subframe: int = 5,
        damping: float = 0.999,
        oversample: int = 2,
        r_glottal: float = 0.75,
        r_lip: float = -0.85,
        closure_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_oral = int(n_oral)
        self.n_nose = int(n_nose)
        self.samples_per_frame = int(samples_per_frame)
        self.n_ir_samples = int(n_ir_samples)
        self.n_subframe = int(n_subframe)
        assert self.n_subframe >= 1
        assert self.samples_per_frame % self.n_subframe == 0, (
            f"samples_per_frame ({self.samples_per_frame}) must be divisible "
            f"by n_subframe ({self.n_subframe})"
        )
        self.samples_per_subframe = self.samples_per_frame // self.n_subframe

        self.damping = float(damping)
        self.oversample = int(oversample)
        assert self.oversample >= 1, "oversample must be ≥ 1"
        # JS uses fade=1.0 for the nose (constructor line 22). Mouth uses
        # `damping=0.999` (lines 148-149).
        self.nose_damping = 1.0
        self.r_glottal = float(r_glottal)
        self.r_lip = float(r_lip)
        self.closure_eps = float(closure_eps)

        # noseStart index (JS line 68). For (44, 28) this is 17.
        self.nose_start = self.n_oral - self.n_nose + 1

        # Pre-build the fixed-shape nose diameter profile (JS lines 77-84).
        # nose_diameter[0] is later overridden by the per-frame velum.
        nose_diam = torch.empty(self.n_nose, dtype=torch.float32)
        for i in range(self.n_nose):
            d = 2.0 * (i / self.n_nose)
            if d < 1.0:
                diameter = 0.4 + 1.6 * d
            else:
                diameter = 0.5 + 1.5 * (2.0 - d)
            nose_diam[i] = min(diameter, 1.9)
        self.register_buffer("nose_diameter", nose_diam)

        # Pre-build per-sub-frame cross-fade alphas.
        alphas_sub = torch.arange(
            self.samples_per_subframe, dtype=torch.float32
        ) / self.samples_per_subframe
        self.register_buffer("alphas_sub", alphas_sub)

        # ── Run 16 (2026-06-05): torch.compile on the IR loop ──
        # The IR loop has 512 sequential steps × ~10 tensor ops each,
        # dominated by kernel launch overhead. torch.compile fuses many
        # ops → ~2-3× speedup. fullgraph=False to allow checkpoint
        # interop (graph breaks across the checkpoint boundary); we use
        # the compiled function INSIDE the checkpoint, so each anchor's
        # IR loop compiles once and is reused.
        # Falls back gracefully if torch.compile is unavailable.
        #
        # MEASURED 2026-08-11: the two figures above ("~2-3x speedup" and the
        #   module docstring's "typically 10-60 s on modern GPUs") were never
        #   verified at the current 512-step configuration, and are wrong there.
        #
        #   range(512) in `for t in range(self.n_ir_samples)` is static, so
        #   Dynamo unrolls the loop completely into a graph of 512 x ~10 ~=
        #   5000+ nodes.  Measured:
        #     - local CPU (Inductor C++ backend, gcc): 362 s to compile
        #     - server GPU (Inductor Triton backend, ptxas): still running after
        #       more than ten minutes, aborted.  Triton's compile time is
        #       superlinear in kernel size.
        #
        #   Worse, training uses variable-length sequences while this is
        #   dynamic=False, so every new T pays that cost again and the default
        #   compile cache overflows.
        #
        #   => DIFFPT_DISABLE_COMPILE=1 is not a local-development convenience.
        #      Under this usage it is required.  Every run script sets it.
        #
        #   If the launch-bound regime is revisited (measured GPU utilisation is
        #   only ~20%), the right tool is CUDA graphs (mode="reduce-overhead"),
        #   which records the 3072 launches as one graph replay.  That needs
        #   fixed shapes, so the dataloader would need length bucketing first.
        #   Compiling a single step rather than the loop also works, but fusion
        #   then happens only within a step and the gain is small.
        self._compiled_run_ir_loop = self._run_ir_loop
        if not _DISABLE_COMPILE and hasattr(torch, "compile"):
            try:
                self._compiled_run_ir_loop = torch.compile(
                    self._run_ir_loop,
                    fullgraph=False,
                    dynamic=False,
                )
            except Exception as e:
                # Fail safe: compile errors should NOT bring down training.
                print(f"[waveguide] torch.compile failed ({e}); "
                      f"falling back to eager mode")
                self._compiled_run_ir_loop = self._run_ir_loop

    # ------------------------------------------------------------------ #
    # Reflection coefficients (smooth closure).
    # ------------------------------------------------------------------ #
    def _compute_reflections(self, vtd: torch.Tensor):
        """Compute reflection coefficients & 3-way junction terms.

        Returns:
            r_oral:  (B, T, n_oral - 1) internal oral reflections.
            r_nose:  (B, T, n_nose - 1) internal nose reflections (no lambda).
            r_left:  (B, T) 3-way nose junction, left arm (JS:319).
            r_right: (B, T) 3-way nose junction, right arm (JS:320).
            r_nose_branch: (B, T) 3-way nose junction, nose arm (JS:321).
        """
        velum = vtd[..., 0]
        d_oral = vtd[..., 1:]
        a_oral = d_oral * d_oral

        # nose diameters: fixed profile + per-frame velum override at index 0
        nose_diam_base = self.nose_diameter[1:].view(
            1, 1, self.n_nose - 1
        ).expand(velum.shape[0], velum.shape[1], self.n_nose - 1)
        d_nose = torch.cat([velum.unsqueeze(-1), nose_diam_base], dim=-1)
        a_nose = d_nose * d_nose

        eps = self.closure_eps
        r_oral = (a_oral[..., :-1] - a_oral[..., 1:]) / (
            a_oral[..., :-1] + a_oral[..., 1:] + eps
        )
        r_nose = (a_nose[..., :-1] - a_nose[..., 1:]) / (
            a_nose[..., :-1] + a_nose[..., 1:] + eps
        )

        a_left = a_oral[..., self.nose_start]
        a_right = a_oral[..., self.nose_start + 1]
        a_nose_0 = a_nose[..., 0]
        sum_3 = a_left + a_right + a_nose_0 + eps
        r_left = (2.0 * a_left - sum_3) / sum_3
        r_right = (2.0 * a_right - sum_3) / sum_3
        r_nose_branch = (2.0 * a_nose_0 - sum_3) / sum_3

        return r_oral, r_nose, r_left, r_right, r_nose_branch

    # ------------------------------------------------------------------ #
    # Frozen-r per-frame IR loop. Sequential over IR sample step t.
    # ------------------------------------------------------------------ #
    def _run_ir_loop(
        self,
        r_oral: torch.Tensor,         # (B, T, n_oral - 1)
        r_nose: torch.Tensor,         # (B, T, n_nose - 1)
        r_left: torch.Tensor,         # (B, T)
        r_right: torch.Tensor,        # (B, T)
        r_nose_branch: torch.Tensor,  # (B, T)
        inj: Optional[torch.Tensor] = None,   # (B, T, n_oral) or None
    ) -> torch.Tensor:
        """Time-domain waveguide IR. Returns (B, T, n_ir_samples).

        Each frame's IR is computed with FROZEN reflection coefficients
        (no per-sample lambda interp inside this loop). The K-anchor
        cross-fade in forward() is what produces the time-varying
        effective IR.

        ``inj`` — INTERIOR SOURCE INJECTION (added 2026-08-05).
        ------------------------------------------------------
        When None (default) this function is bit-identical to its previous
        behaviour: a unit impulse enters at the glottal end via ``jR_0``,
        giving the glottis->output impulse response.

        When given, ``inj[b, t, i]`` is additionally added to BOTH R[i] and
        L[i] at t == 0, giving the impulse response from an interior source
        distribution to the output. This mirrors JS PT, where
        ``addTurbulenceNoise`` (tract.js:178) writes into R[] and L[] at the
        very TOP of ``runStep`` — i.e. BEFORE the boundary conditions
        (tract.js:181-182) and BEFORE ``calc_junctions`` (tract.js:184) read
        them — and does so symmetrically::

            R[i+1] += noise0/2;  L[i+1] += noise0/2      (tract.js:266-267)

        Note the injection happens on EVERY sub-step (JS calls runStep twice
        per output sample with the same turbulence sample), matching the ZOH
        convention already used for ``src`` below.

        Because PT drives every injection site from the SAME turbulence
        signal (tract.js:252 passes one scalar to every touch), superposition
        lets one pass carry the whole spatial distribution::

            sum_i g_i (s * h_i) = s * (sum_i g_i h_i) = s * IR(inj=g)

        so no per-segment IR bank and no argmax site selection is needed.
        """
        device = r_oral.device
        dtype = r_oral.dtype
        B, T, _ = r_oral.shape
        n_o, n_n = self.n_oral, self.n_nose
        ns = self.nose_start
        fade_m, fade_n = self.damping, self.nose_damping

        R = torch.zeros(B, T, n_o, device=device, dtype=dtype)
        L = torch.zeros(B, T, n_o, device=device, dtype=dtype)
        nR = torch.zeros(B, T, n_n, device=device, dtype=dtype)
        nL = torch.zeros(B, T, n_n, device=device, dtype=dtype)

        rL3 = r_left.unsqueeze(-1)
        rR3 = r_right.unsqueeze(-1)
        rN3 = r_nose_branch.unsqueeze(-1)

        outputs = []
        for t in range(self.n_ir_samples):
            out_accum_lip = None
            out_accum_nose = None
            for sub in range(self.oversample):
                # Interior source injection, BEFORE the boundary conditions
                # and the junction pass read R/L — matching tract.js:178
                # (addTurbulenceNoise) preceding tract.js:181-184.
                if inj is not None and t == 0:
                    R = R + inj
                    L = L + inj

                # JS-faithful unit impulse injection at t=0, ZOH across
                # the 2 sub-steps within audio sample t.
                #
                # When computing an INTERIOR-source IR we must NOT also fire
                # the glottal impulse, or the two paths would double-count.
                # The decomposition is exact by linearity:
                #     total = glottal * IR(src=1, inj=0)
                #           + noise   * IR(src=0, inj=g)
                src = (1.0 if (t == 0 and inj is None) else 0.0)
                jR_0 = L[..., 0:1] * self.r_glottal + src
                jL_n = R[..., -1:] * self.r_lip

                R_left = R[..., :-1]
                L_right = L[..., 1:]
                w_oral = r_oral * (R_left + L_right)
                jR_inner = R_left - w_oral
                jL_inner = L_right + w_oral

                R_ns_m1 = R[..., ns - 1:ns]
                L_ns = L[..., ns:ns + 1]
                nL_0 = nL[..., 0:1]
                jL_ns_3way = rL3 * R_ns_m1 + (1.0 + rL3) * (nL_0 + L_ns)
                jR_ns_3way = rR3 * L_ns + (1.0 + rR3) * (R_ns_m1 + nL_0)
                njR_0 = rN3 * nL_0 + (1.0 + rN3) * (L_ns + R_ns_m1)

                j = ns - 1
                jR_inner = torch.cat(
                    [jR_inner[..., :j], jR_ns_3way, jR_inner[..., j + 1:]],
                    dim=-1,
                )
                jL_inner = torch.cat(
                    [jL_inner[..., :j], jL_ns_3way, jL_inner[..., j + 1:]],
                    dim=-1,
                )

                nR_left = nR[..., :-1]
                nL_right = nL[..., 1:]
                w_nose = r_nose * (nR_left + nL_right)
                njR_inner = nR_left - w_nose
                njL_inner = nL_right + w_nose

                njL_end = nR[..., -1:] * self.r_lip

                jR_full = torch.cat([jR_0, jR_inner], dim=-1)
                jL_full_shift = torch.cat([jL_inner, jL_n], dim=-1)

                R = jR_full * fade_m
                L = jL_full_shift * fade_m

                njR_full = torch.cat([njR_0, njR_inner], dim=-1)
                njL_full_shift = torch.cat([njL_inner, njL_end], dim=-1)

                nR = njR_full * fade_n
                nL = njL_full_shift * fade_n

                lip_out = R[..., -1]
                nose_out = nR[..., -1]
                if out_accum_lip is None:
                    out_accum_lip = lip_out
                    out_accum_nose = nose_out
                else:
                    out_accum_lip = out_accum_lip + lip_out
                    out_accum_nose = out_accum_nose + nose_out

            outputs.append(out_accum_lip + out_accum_nose)

        return torch.stack(outputs, dim=-1)

    # ------------------------------------------------------------------ #
    # Frame-aligned overlap-add convolution. Vectorized across (B, T).
    # ------------------------------------------------------------------ #
    def _frame_aligned_convolve(
        self,
        glottal: torch.Tensor,    # (B, T*spf)
        tract_ir: torch.Tensor,   # (B, T, n_ir)
    ) -> torch.Tensor:
        """Time-varying convolution. Returns (B, T*spf)."""
        B, T_audio_int = glottal.shape
        _, T, n_ir = tract_ir.shape
        spf = self.samples_per_frame
        assert T_audio_int == T * spf, (
            f"glottal length {T_audio_int} != T * spf {T * spf}"
        )

        glot_frames = glottal.view(B, T, spf)
        glot_flat = glot_frames.reshape(1, B * T, spf)
        ir_flat = tract_ir.reshape(B * T, 1, n_ir)
        ir_flip = ir_flat.flip(-1)
        out_len = spf + n_ir - 1
        conv_out = F.conv1d(
            glot_flat, ir_flip, groups=B * T, padding=n_ir - 1
        )
        conv_out = conv_out.view(B, T, out_len)

        # Overlap-add via F.fold.
        T_audio_full = (T - 1) * spf + out_len
        patches = conv_out.transpose(1, 2).contiguous()
        folded = F.fold(
            patches,
            output_size=(1, T_audio_full),
            kernel_size=(1, out_len),
            stride=(1, spf),
        )
        audio = folded.view(B, T_audio_full)
        return audio[:, : T * spf]

    # ------------------------------------------------------------------ #
    # Forward — sub-frame K-anchor IR interpolation.
    # ------------------------------------------------------------------ #
    def forward(
        self,
        vtd: torch.Tensor,
        glottal_audio: torch.Tensor,
        noise_audio: Optional[torch.Tensor] = None,
        inj_weights: Optional[torch.Tensor] = None,
        interior: Optional[list] = None,
    ) -> torch.Tensor:
        assert vtd.dim() == 3, f"vtd must be (B, T, 1 + n_oral); got {vtd.shape}"
        assert vtd.shape[-1] == 1 + self.n_oral, (
            f"vtd last dim expected {1 + self.n_oral}; got {vtd.shape[-1]}"
        )
        B, T, _ = vtd.shape
        K = self.n_subframe
        spf = self.samples_per_frame
        spf_sub = self.samples_per_subframe
        assert glottal_audio.shape == (B, T * spf), (
            f"glottal_audio shape expected ({B}, {T * spf}); "
            f"got {tuple(glottal_audio.shape)}"
        )

        # Per-frame reflections (current).
        r_oral_cur, r_nose, r_left_cur, r_right_cur, r_n3_cur = (
            self._compute_reflections(vtd)
        )

        # Shift to build prev (frame f's prev = frame f-1's cur).
        # Frame 0's prev = frame 0's cur (no inter-frame transition for
        # the very first frame — degenerate λ-interp on frame 0).
        r_oral_prev = torch.cat([r_oral_cur[:, :1], r_oral_cur[:, :-1]], dim=1)
        r_left_prev = torch.cat([r_left_cur[:, :1], r_left_cur[:, :-1]], dim=1)
        r_right_prev = torch.cat([r_right_cur[:, :1], r_right_cur[:, :-1]], dim=1)
        r_n3_prev = torch.cat([r_n3_cur[:, :1], r_n3_cur[:, :-1]], dim=1)

        # ---- Compute K+1 anchor IRs per frame ------------------------ #
        # Anchor k uses lambda = k/K.
        #
        # Memory strategy: wrap each anchor's _run_ir_loop in
        # torch.utils.checkpoint so the IR-loop internal state is NOT
        # retained for backward. During backward, each anchor is
        # recomputed individually → memory peaks at one anchor's worth
        # of state, not K+1 anchors'. Time cost: ~2× per-anchor IR loop.
        use_ckpt = self.training and torch.is_grad_enabled()

        # ---- Interior-source anchors (frication / burst), optional ---- #
        # Built in the same lambda sweep so the noise path sees exactly the
        # same time-varying geometry as the glottal path. The injection
        # weights themselves are lerped prev->cur alongside the reflection
        # coefficients, so a constriction forming across a frame boundary
        # ramps its turbulence rather than stepping it.
        # Normalise the two spellings into one list of (signal, weights).
        # Each entry gets its own anchor bank; by linearity the contributions
        # simply add. Frication and the plosive burst are both interior
        # sources in JS (tract.js:266-267 and :228-229 respectively) and differ
        # only in their spatial weight and temporal signal, so they share
        # this machinery.
        sources = list(interior) if interior else []
        if inj_weights is not None:
            assert noise_audio is not None, (
                "inj_weights given but noise_audio is None"
            )
            sources.append((noise_audio, inj_weights))
        # Whether to lerp a source's weights across the frame boundary.
        # Frication: YES — a constriction forms gradually, so ramping the
        # turbulence matches JS, where the gain follows the smoothed diameter.
        # Burst: NO — JS's addTransient fires at full strength on the release
        # sample (tract.js:227-229 with timeAlive = 0). Lerping from the previous
        # frame's zero weight would multiply the burst ONSET, where essentially
        # all of its energy sits, by ~0 — measured at -22.7 dB total energy
        # versus JS before this was fixed.
        # A source may be given as (signal, weights) or (signal, weights, lerp).
        sources = [(t[0], t[1], (t[2] if len(t) > 2 else True)) for t in sources]
        for sig_k, w_k, _ in sources:
            assert w_k.shape == (B, T, self.n_oral), (
                f"interior weights expected {(B, T, self.n_oral)}; "
                f"got {tuple(w_k.shape)}"
            )
            assert sig_k.shape == (B, T * spf), (
                f"interior signal expected {(B, T * spf)}; got {tuple(sig_k.shape)}"
            )
        inj_prevs = [
            (torch.cat([w[:, :1], w[:, :-1]], dim=1) if do_lerp else w)
            for _, w, do_lerp in sources
        ]
        ir_banks = [[] for _ in sources]

        ir_anchors = []
        for k in range(K + 1):
            lam = float(k) / float(K)
            # Internal junction (JS:137): prev*(1-λ) + cur*λ
            r_oral_k = r_oral_prev * (1.0 - lam) + r_oral_cur * lam
            # 3-way junction (JS:188/190/192): cur*(1-λ) + prev*λ (REVERSED)
            r_left_k  = r_left_cur  * (1.0 - lam) + r_left_prev  * lam
            r_right_k = r_right_cur * (1.0 - lam) + r_right_prev * lam
            r_n3_k    = r_n3_cur    * (1.0 - lam) + r_n3_prev    * lam

            for si, (_, w_cur, _) in enumerate(sources):
                inj_k = inj_prevs[si] * (1.0 - lam) + w_cur * lam
                if use_ckpt:
                    ir_inj_k = cp.checkpoint(
                        self._compiled_run_ir_loop,
                        r_oral_k, r_nose, r_left_k, r_right_k, r_n3_k, inj_k,
                        use_reentrant=False,
                    )
                else:
                    ir_inj_k = self._compiled_run_ir_loop(
                        r_oral_k, r_nose, r_left_k, r_right_k, r_n3_k, inj_k,
                    )
                ir_banks[si].append(ir_inj_k)

            if use_ckpt:
                # Run 16: use the compiled IR loop inside checkpoint. The
                # graph break at the checkpoint boundary is acceptable —
                # what we want compiled is the inner IR-step kernel chain.
                ir_k = cp.checkpoint(
                    self._compiled_run_ir_loop,
                    r_oral_k, r_nose, r_left_k, r_right_k, r_n3_k,
                    use_reentrant=False,
                )
            else:
                ir_k = self._compiled_run_ir_loop(
                    r_oral_k, r_nose, r_left_k, r_right_k, r_n3_k,
                )
            ir_anchors.append(ir_k)

        # ---- Per-output-sub-frame fixed-IR + audio-rate blend -------- #
        # SEMANTICS: at audio output time t (in sub-frame s of frame f),
        # apply IR = blend(anchor s of frame f, anchor s+1 of frame f) to
        # the WINDOW of past n_ir glottal samples. This means every past
        # glottal sample (from any past frame) is convolved with the IR
        # ACTIVE AT THE OUTPUT TIME, not the IR active at injection time.
        #
        # This matches JS PT sample-level semantics: at output time t,
        # the waveguide state is determined by the CURRENT r(t), so its
        # effective filter applied to any past glottal energy is IR(r(t)).
        #
        # Continuity properties (no 50 Hz artifact):
        #   - Sub-frame boundary s → s+1 within frame f: α=1 at end of s
        #     gives effective IR = anchor s+1 of frame f. α=0 at start of
        #     s+1 also gives IR = anchor s+1 of frame f. Continuous.
        #   - Frame boundary f → f+1: α=1 at end of frame f's sub-frame
        #     K-1 gives anchor K of frame f = IR(r_cur_f). α=0 at start
        #     of frame f+1's sub-frame 0 gives anchor 0 of frame f+1 =
        #     IR(r_prev_{f+1}) = IR(r_cur_f). Continuous (SAME IR).
        #
        # Implementation: pad glottal with n_ir-1 zeros at front, unfold
        # into (B, T*K, spf_sub + n_ir - 1) windows at stride spf_sub.
        # Each window provides the n_ir-1 past samples + spf_sub samples
        # in the current sub-frame. Per-window batched conv1d (valid,
        # no padding) with each output's anchor IR. No overlap-add: each
        # window directly produces its spf_sub output samples.

        audio = self._blend_convolve(glottal_audio, ir_anchors, B, T, K, spf, spf_sub)
        for si, (sig_k, _, _) in enumerate(sources):
            audio = audio + self._blend_convolve(
                sig_k, ir_banks[si], B, T, K, spf, spf_sub,
            )
        return audio

    def _blend_convolve(
        self,
        sig: torch.Tensor,          # (B, T*spf)
        ir_anchors: list,           # K+1 tensors of (B, T, n_ir)
        B: int, T: int, K: int, spf: int, spf_sub: int,
    ) -> torch.Tensor:
        """Cross-faded time-varying convolution of `sig` with an anchor bank.

        Signal-agnostic: identical machinery serves the glottal path and the
        interior-source (frication / burst) path, which is what makes the
        second source nearly free architecturally.
        """
        n_ir = self.n_ir_samples

        # Stack anchor IRs → (B, T, K+1, n_ir). Per output sub-frame s of
        # frame f: lower = anchor s of frame f, upper = anchor s+1.
        ir_stack = torch.stack(ir_anchors, dim=2)                  # (B, T, K+1, n_ir)
        ir_lowers_flat = (
            ir_stack[:, :, :K, :].reshape(B * T * K, 1, n_ir).flip(-1)
        )
        ir_uppers_flat = (
            ir_stack[:, :, 1:, :].reshape(B * T * K, 1, n_ir).flip(-1)
        )

        # Pad with n_ir-1 zeros at front so the first sub-frame's window has
        # access to "past silence" (no artifact from edge).
        g_padded = F.pad(sig, (n_ir - 1, 0))                       # (B, T*spf + n_ir - 1)

        # Unfold into per-sub-frame windows. Each window covers the
        # n_ir-1 past samples PLUS the spf_sub samples of its sub-frame.
        # Window i corresponds to sub-frame i (0-indexed across all
        # frames; sub-frame s of frame f has i = f*K + s).
        g_windows = g_padded.unfold(
            dimension=1, size=spf_sub + n_ir - 1, step=spf_sub,
        )                                                          # (B, T*K, spf_sub + n_ir - 1)

        g_flat = g_windows.reshape(1, B * T * K, spf_sub + n_ir - 1)

        # Batched conv1d, valid (no padding): output length =
        # (spf_sub + n_ir - 1) - n_ir + 1 = spf_sub.
        conv_low = F.conv1d(
            g_flat, ir_lowers_flat, groups=B * T * K, padding=0,
        ).view(B, T * K, spf_sub)
        conv_up = F.conv1d(
            g_flat, ir_uppers_flat, groups=B * T * K, padding=0,
        ).view(B, T * K, spf_sub)

        # Audio-rate blend within each sub-frame.
        alphas = self.alphas_sub.to(
            device=sig.device, dtype=sig.dtype,
        ).view(1, 1, spf_sub)
        blended = (1.0 - alphas) * conv_low + alphas * conv_up     # (B, T*K, spf_sub)

        # Concatenate sub-frames → (B, T*K*spf_sub) = (B, T*spf).
        return blended.reshape(B, T * spf)


# ---- Smoke test ---------------------------------------------------------- #
if __name__ == "__main__":
    import time

    torch.manual_seed(0)
    B, T = 2, 5
    n_oral, n_nose = 44, 28
    spf = 960

    tract = WaveguideTract(
        n_oral=n_oral, n_nose=n_nose, samples_per_frame=spf, n_subframe=5,
    )
    print(
        f"WaveguideTract: n_oral={tract.n_oral}, n_nose={tract.n_nose}, "
        f"nose_start={tract.nose_start}, spf={tract.samples_per_frame}, "
        f"K={tract.n_subframe}, spf_sub={tract.samples_per_subframe}, "
        f"n_ir={tract.n_ir_samples}"
    )
    assert tract.nose_start == 17
    assert tract.samples_per_subframe == 192

    # Realistic VTD: velum ~0.05 (closed), oral ~1.0 mm with jitter.
    vtd = torch.full((B, T, 1 + n_oral), 1.0)
    vtd[..., 0] = 0.05
    vtd[..., 1:] = vtd[..., 1:] + 0.1 * torch.randn(B, T, n_oral)

    # Synthetic glottal: 120 Hz sine at 48 kHz internal rate.
    sr_int = spf * 50
    t_arr = torch.arange(T * spf, dtype=torch.float32) / sr_int
    glottal = (0.5 * torch.sin(2 * torch.pi * 120.0 * t_arr)).unsqueeze(0)
    glottal = glottal.expand(B, -1).contiguous()

    # ── Forward pass (eval mode, no grad) ───────────────────────────── #
    tract.eval()
    t0 = time.time()
    with torch.no_grad():
        audio = tract(vtd, glottal)
    fwd_ms = (time.time() - t0) * 1000.0
    rms = audio.pow(2).mean().sqrt().item()
    print(f"output.shape       = {tuple(audio.shape)}   "
          f"(expected ({B}, {T * spf}))")
    print(f"output.dtype       = {audio.dtype}")
    print(f"finite             = {torch.isfinite(audio).all().item()}")
    print(f"output.min/max     = {audio.min().item():+.4f} / "
          f"{audio.max().item():+.4f}")
    print(f"output RMS         = {rms:.4f}")
    print(f"fwd time (eval)    = {fwd_ms:.1f} ms  "
          f"(B={B}, T={T}, K={tract.n_subframe})")
    assert audio.shape == (B, T * spf)
    assert torch.isfinite(audio).all()
    assert rms > 1e-4

    # ── Closure gradient test ───────────────────────────────────────── #
    vtd_c = vtd.clone()
    vtd_c[..., 1:][..., 30] = 0.001
    vtd_c = vtd_c.detach().requires_grad_(True)
    tract.train()
    audio_c = tract(vtd_c, glottal)
    loss = audio_c.pow(2).mean()
    loss.backward()
    g = vtd_c.grad
    print(f"closure audio finite= {torch.isfinite(audio_c).all().item()}")
    print(f"closure grad finite = {torch.isfinite(g).all().item()}")
    print(f"closure grad |·|.mean= {g.abs().mean().item():.4e}, "
          f"max = {g.abs().max().item():.4e}")
    assert torch.isfinite(g).all()
    assert g.abs().max().item() > 0.0

    # ── Plain gradient test ─────────────────────────────────────────── #
    vtd_g = vtd.clone().detach().requires_grad_(True)
    audio_g = tract(vtd_g, glottal)
    audio_g.pow(2).mean().backward()
    print(f"grad finite        = {torch.isfinite(vtd_g.grad).all().item()}, "
          f"abs.mean = {vtd_g.grad.abs().mean().item():.4e}, "
          f"abs.max = {vtd_g.grad.abs().max().item():.4e}")
    assert torch.isfinite(vtd_g.grad).all()

    print("OK")
