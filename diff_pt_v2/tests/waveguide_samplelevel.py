#!/usr/bin/env python
"""PROTOTYPE v2 — does NOT touch components/waveguide.py.

Sample-level CONTINUOUS waveguide: one stateful pass over the whole glottal signal,
carrying the scattering state (R,L,nR,nL) across frame boundaries, with reflection
coefficients lambda-interpolated PER SAMPLE between prev/cur frames. This is the exact
JS-PT-faithful scheme; unlike the K-anchor it does not 'apply the current-frame IR to
past glottal', so it carries the tract's ringing correctly when the tract changes ->
no frame-rate (50 Hz) residual. Differentiable (same scattering ops as _run_ir_loop,
just stateful). Slow (sequential per sample) — for final/eval rendering, not training.
Scattering matches WaveguideTract._run_ir_loop exactly; only state-carry + per-sample
lambda interp + glottal injection differ.
"""
from __future__ import annotations
import torch
from diff_pt_v2.components.waveguide import WaveguideTract


class WaveguideTractSampleLevel(WaveguideTract):
    def forward(self, vtd: torch.Tensor, glottal_audio: torch.Tensor) -> torch.Tensor:
        B, T, _ = vtd.shape
        spf = self.samples_per_frame
        T_audio = T * spf
        device, dtype = vtd.device, vtd.dtype
        n_o, n_n, ns = self.n_oral, self.n_nose, self.nose_start
        fade_m, fade_n = self.damping, self.nose_damping
        r_glottal, r_lip = self.r_glottal, self.r_lip

        r_oral_cur, r_nose, r_left_cur, r_right_cur, r_n3_cur = self._compute_reflections(vtd)
        sh = lambda x: torch.cat([x[:, :1], x[:, :-1]], dim=1)
        r_oral_prev = sh(r_oral_cur)
        r_left_prev, r_right_prev, r_n3_prev = sh(r_left_cur), sh(r_right_cur), sh(r_n3_cur)

        t_arr = torch.arange(T_audio, device=device)
        lam = (t_arr % spf).to(dtype) / float(spf)               # (T_audio,)
        fidx = (t_arr // spf).long()                              # (T_audio,)
        lc = lam.view(1, T_audio, 1)
        # internal/oral: prev -> cur ; 3-way: cur -> prev (reversed, as in the K-anchor)
        r_oral_ps = r_oral_prev[:, fidx, :] * (1 - lc) + r_oral_cur[:, fidx, :] * lc   # (B,T_a,n_o-1)
        l1 = lam.view(1, T_audio)
        r_left_ps = r_left_cur[:, fidx] * (1 - l1) + r_left_prev[:, fidx] * l1          # (B,T_a)
        r_right_ps = r_right_cur[:, fidx] * (1 - l1) + r_right_prev[:, fidx] * l1
        r_n3_ps = r_n3_cur[:, fidx] * (1 - l1) + r_n3_prev[:, fidx] * l1
        r_nose_ps = r_nose[:, fidx, :]                                                  # (B,T_a,n_n-1) per-frame

        R = torch.zeros(B, n_o, device=device, dtype=dtype)
        L = torch.zeros(B, n_o, device=device, dtype=dtype)
        nR = torch.zeros(B, n_n, device=device, dtype=dtype)
        nL = torch.zeros(B, n_n, device=device, dtype=dtype)
        j = ns - 1
        outputs = []
        for t in range(T_audio):
            r_oral = r_oral_ps[:, t, :]
            r_nose_t = r_nose_ps[:, t, :]
            rL3 = r_left_ps[:, t:t + 1]; rR3 = r_right_ps[:, t:t + 1]; rN3 = r_n3_ps[:, t:t + 1]
            src = glottal_audio[:, t:t + 1]
            acc = None
            for _ in range(self.oversample):
                jR_0 = L[..., 0:1] * r_glottal + src
                jL_n = R[..., -1:] * r_lip
                R_left = R[..., :-1]; L_right = L[..., 1:]
                w_oral = r_oral * (R_left + L_right)
                jR_inner = R_left - w_oral
                jL_inner = L_right + w_oral
                R_ns_m1 = R[..., ns - 1:ns]; L_ns = L[..., ns:ns + 1]; nL_0 = nL[..., 0:1]
                jL_ns_3way = rL3 * R_ns_m1 + (1.0 + rL3) * (nL_0 + L_ns)
                jR_ns_3way = rR3 * L_ns + (1.0 + rR3) * (R_ns_m1 + nL_0)
                njR_0 = rN3 * nL_0 + (1.0 + rN3) * (L_ns + R_ns_m1)
                jR_inner = torch.cat([jR_inner[..., :j], jR_ns_3way, jR_inner[..., j + 1:]], dim=-1)
                jL_inner = torch.cat([jL_inner[..., :j], jL_ns_3way, jL_inner[..., j + 1:]], dim=-1)
                nR_left = nR[..., :-1]; nL_right = nL[..., 1:]
                w_nose = r_nose_t * (nR_left + nL_right)
                njR_inner = nR_left - w_nose
                njL_inner = nL_right + w_nose
                njL_end = nR[..., -1:] * r_lip
                R = torch.cat([jR_0, jR_inner], dim=-1) * fade_m
                L = torch.cat([jL_inner, jL_n], dim=-1) * fade_m
                nR = torch.cat([njR_0, njR_inner], dim=-1) * fade_n
                nL = torch.cat([njL_inner, njL_end], dim=-1) * fade_n
                o = R[..., -1] + nR[..., -1]
                acc = o if acc is None else acc + o
            outputs.append(acc)
        return torch.stack(outputs, dim=-1)
