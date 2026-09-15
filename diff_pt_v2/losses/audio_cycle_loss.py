"""AudioCycleLoss — waveform-level cycle-consistency loss for diff-PT v2.

Replaces the cepstrally-smoothed ``compute_spectral_loss`` from
``src/classification/losses_cls.py``. Operates directly on raw 16 kHz audio.

References:
- Parallel WaveGAN (Yamamoto et al., 2020; arXiv:1910.11480) — multi-res STFT
  loss with spectral convergence + log-magnitude L1. We use the same triplet
  structure rescaled / re-ordered for 16 kHz.
- DDSP (Engel et al., 2020; arXiv:2001.04643) — multi-scale spectral loss
  averages out time-frequency uncertainty bias of any single STFT window.
- VocalTrax (Wu et al., 2023; arXiv:2309.14761) — MFCC / cepstral losses
  empirically fail to drive PT closure; MR-STFT is recommended instead.

Components: MultiResSTFTLoss (3 resolutions, sc+lm each), LogRMSEnvLoss
(closure detector, log floor 1e-4), MelL1Loss (80-mel, no liftering).
"""

from __future__ import annotations

from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# -----------------------------------------------------------------------------
# 1. Multi-Resolution STFT Loss (Parallel WaveGAN style)
# -----------------------------------------------------------------------------
class _STFTLoss(nn.Module):
    """Single-resolution STFT loss: spectral convergence + log-magnitude L1.

    Per Parallel WaveGAN (arXiv:1910.11480):
        L_sc = ||M - M_hat||_F / ||M||_F           (spectral convergence)
        L_lm = mean |log M_hat - log M|            (log-magnitude L1)
    Per-resolution total = L_sc + L_lm.
    """

    def __init__(self, fft_size: int, hop_size: int, win_length: int):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        # Hann window registered as buffer → auto device migration with .to().
        self.register_buffer("window", torch.hann_window(win_length))

    def _stft_mag(self, x: torch.Tensor) -> torch.Tensor:
        """Compute magnitude STFT of shape (B, n_freqs, n_frames)."""
        # torch.stft is differentiable. return_complex=True for torch>=1.8.
        spec = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
        # |X|: clamp to avoid log(0) and divide-by-zero in spectral convergence.
        return spec.abs().clamp(min=1e-7)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (spectral_convergence, log_magnitude_l1) — both scalars."""
        M_pred = self._stft_mag(pred)
        M_gt = self._stft_mag(gt)

        # Spectral convergence: Frobenius norm of difference / Frobenius norm of GT.
        # Use a per-batch reduction (mean over batch dim is implicit via flatten).
        num = torch.norm(M_gt - M_pred, p="fro")
        den = torch.norm(M_gt, p="fro").clamp(min=1e-7)
        sc = num / den

        # Log magnitude L1.
        lm = F.l1_loss(torch.log(M_pred), torch.log(M_gt))

        return sc, lm


class MultiResSTFTLoss(nn.Module):
    """3-resolution STFT loss for 16 kHz waveforms.

    Default: (512, 1024, 2048) / (64, 120, 240) / (240, 600, 1200). Adapted
    from Parallel WaveGAN's 24 kHz config: 240-sample window ≈15 ms (covers
    ~2 pitch periods of a 130 Hz voice → transients), 1200-sample window
    ≈75 ms (long-context spectral structure).
    """

    def __init__(
        self,
        fft_sizes: Tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: Tuple[int, ...] = (64, 120, 240),
        win_lengths: Tuple[int, ...] = (240, 600, 1200),
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths), (
            "fft_sizes, hop_sizes, win_lengths must all have the same length"
        )
        self.stft_losses = nn.ModuleList([
            _STFTLoss(n_fft, hop, win)
            for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths)
        ])

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Mean of (sc + lm) across resolutions. Scalar tensor."""
        total = pred.new_zeros(())  # scalar on same device/dtype as pred
        for stft in self.stft_losses:
            sc, lm = stft(pred, gt)
            total = total + sc + lm
        return total / len(self.stft_losses)


# -----------------------------------------------------------------------------
# 2. Log RMS Envelope L1 (closure-specific)
# -----------------------------------------------------------------------------
class LogRMSEnvLoss(nn.Module):
    """Log-RMS envelope L1 loss — closure-specific.

    Penalizes time-localized energy mismatch (plosive closures: GT near-silent
    while mis-trained model still produces energy). Log floor 1e-4 (≈-80 dB)
    → a 40 dB gap gives ~9.2 nats L1, strongly steering closure timing.
    Frame: 320 samples = 20 ms @ 16 kHz; hop: 80 samples = 5 ms.
    """

    def __init__(self, frame: int = 320, hop: int = 80, log_floor: float = 1e-4):
        super().__init__()
        self.frame = frame
        self.hop = hop
        self.log_floor = log_floor

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        """RMS envelope via torch.Tensor.unfold. Returns (B, n_frames)."""
        # unfold(dim, size, step): non-overlapping/overlapping sliding window
        # → (B, n_frames, frame). All operations downstream are differentiable.
        frames = x.unfold(dimension=-1, size=self.frame, step=self.hop)
        # mean of squares per frame, then sqrt → RMS amplitude.
        ms = frames.pow(2).mean(dim=-1)
        return torch.sqrt(ms.clamp(min=self.log_floor ** 2))

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        rms_pred = self._rms(pred)
        rms_gt = self._rms(gt)
        # log of RMS (clamp before log for safety; clamp value matches the
        # one used inside _rms so log_floor is the effective minimum).
        log_pred = torch.log(rms_pred.clamp(min=self.log_floor))
        log_gt = torch.log(rms_gt.clamp(min=self.log_floor))
        return F.l1_loss(log_pred, log_gt)


# -----------------------------------------------------------------------------
# 3. Mel L1 Loss (no cepstral liftering)
# -----------------------------------------------------------------------------
class MelL1Loss(nn.Module):
    """L1 loss on full-resolution log-mel spectrograms (NO cepstral liftering).

    The cepstral-truncated variant (``compute_spectral_loss`` in losses_cls.py)
    empirically fails at PT closure dynamics (VocalTrax arXiv:2309.14761).
    Here we keep full mel resolution so transient / closure structure is
    preserved. Same MelSpectrogram transform applied to pred and gt for
    aligned comparison.
    """

    def __init__(
        self,
        sr: int = 16000,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop: int = 320,
        log_floor: float = 1e-6,
    ):
        super().__init__()
        self.log_floor = log_floor
        # MelSpectrogram is a nn.Module — buffers (mel_fb, window) will follow
        # device migration automatically.
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            power=2.0,
            center=True,
            pad_mode="reflect",
        )

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # MelSpectrogram returns (B, n_mels, n_frames). Clamp before log.
        mel_pred = self.mel(pred).clamp(min=self.log_floor)
        mel_gt = self.mel(gt).clamp(min=self.log_floor)
        return F.l1_loss(torch.log(mel_pred), torch.log(mel_gt))


# -----------------------------------------------------------------------------
# 3b. TremorLoss — modulation-domain anti-tremor (symptom side, audio-domain)
# -----------------------------------------------------------------------------
# Ported VERBATIM (numerics identical) from the validated prototype
# ``res_local/tremor_losses.py``. This is the trainable twin of the offline
# md50 AM-depth metric. It is NOT folded into AudioCycleLoss.forward's weighted
# sum — it carries its own independent λ (passed in train_cls via
# --lambda-tremor) because its scale was calibrated separately by gradient
# parity against the cycle loss (2026-06-17).
#
# Self-test reference (from the prototype): am_depth=0.2113 for a 0.3-depth
# 50 Hz AM carrier.


def analytic_envelope(x: torch.Tensor) -> torch.Tensor:
    """|analytic(x)| along the last axis. x: (B, T) -> (B, T). Differentiable.

    Hilbert transform via the scipy.signal.hilbert FFT algorithm (one-sided
    spectrum doubling). Implemented with torch.fft so gradients flow.
    """
    N = x.shape[-1]
    Xf = torch.fft.fft(x, dim=-1)
    h = torch.zeros(N, device=x.device, dtype=x.dtype)
    if N % 2 == 0:
        h[0] = 1.0
        h[N // 2] = 1.0
        h[1:N // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(N + 1) // 2] = 2.0
    analytic = torch.fft.ifft(Xf * h.unsqueeze(0), dim=-1)
    return analytic.abs()


def _env_1k(audio: torch.Tensor, sr: int = 16000, target: int = 1000) -> torch.Tensor:
    """Envelope decimated to ~`target` Hz via boxcar avg-pool (anti-alias+decimate)."""
    env = analytic_envelope(audio)                    # (B, T) @ sr
    k = max(1, sr // target)
    env = F.avg_pool1d(env.unsqueeze(1), kernel_size=k, stride=k).squeeze(1)
    return env                                         # (B, T/k) @ ~target Hz


def am_depth(audio: torch.Tensor, sr: int = 16000, env_fs: int = 1000,
             band=(48.0, 52.0)) -> torch.Tensor:
    """md50-proportional AM depth in `band` Hz. audio (B,T) -> (B,) depth.

    depth = sqrt(sum |E_band|^2 * 2/(env_fs*L)) / mean(env). The one-sided
    periodogram density scaling (2/(fs*L), boxcar) makes this read in the SAME
    units as the validated md50, so the EXCESS over the real target's natural
    micro-modulation is the meaningful penalized quantity.
    """
    env = _env_1k(audio, sr=sr, target=env_fs)         # (B, L)
    L = env.shape[-1]
    mean_env = env.mean(dim=-1)                         # (B,)
    e = env - mean_env.unsqueeze(-1)
    E = torch.fft.rfft(e, dim=-1)                       # (B, L//2+1)
    freqs = torch.fft.rfftfreq(L, d=1.0 / env_fs).to(env.device)
    m = (freqs >= band[0]) & (freqs <= band[1])
    power = (E[:, m].abs() ** 2).sum(dim=-1) * (2.0 / (env_fs * L))   # (B,)
    # S2 (2026-06-17 review): floor the denominator (production deviates from the
    # prototype's +1e-9). On a genuinely near-silent utterance mean_env→0 turns a
    # tiny numerator into a huge depth/gradient right when training is fragile;
    # 1e-3 (~-60 dB on [-1,1] audio) caps that without touching normal speech
    # (mean_env ~0.05-0.2 >> 1e-3, so am_depth parity 0.2113 is unchanged).
    return torch.sqrt(power + 1e-12) / mean_env.clamp(min=1e-3)


class TremorLoss(nn.Module):
    """Excess 48-52 Hz (+ optional harmonic) AM depth over the real target.

    Anti-tremor "symptom-side" penalty on the RENDERED audio (2026-06-17). Take
    the amplitude envelope → its modulation spectrum → the 48-52 Hz band depth
    (== md50), and penalize ONLY the excess over the real target's own 48-52 Hz
    depth (real speech carries ~0.010 natural micro-modulation we must NOT
    erase). Optional 96-104 Hz harmonic via ``w_harm``.

    Default ``w_harm=0.0`` → a clean md50-equivalent (the prototype defaults to
    0.5; production keeps the harmonic OFF by default but available). Numerics
    are identical to the validated ``res_local/tremor_losses.py``.
    """

    def __init__(self, sr: int = 16000, env_fs: int = 1000,
                 band=(48.0, 52.0), harm=(96.0, 104.0), w_harm: float = 0.0,
                 min_samples: int = 1600):
        super().__init__()
        self.sr, self.env_fs = sr, env_fs
        self.band, self.harm, self.w_harm = band, harm, w_harm
        # S1: below this many valid samples (0.1 s @16k) the 48-52 Hz modulation
        # estimate is unreliable → skip that utterance.
        self.min_samples = int(min_samples)

    def _excess(self, ap: torch.Tensor, ag: torch.Tensor) -> torch.Tensor:
        """Scalar relu-excess of band (+harmonic) AM depth, pred over gt."""
        loss = F.relu(am_depth(ap, self.sr, self.env_fs, self.band)
                      - am_depth(ag, self.sr, self.env_fs, self.band)).mean()
        if self.w_harm > 0:
            loss = loss + self.w_harm * F.relu(
                am_depth(ap, self.sr, self.env_fs, self.harm)
                - am_depth(ag, self.sr, self.env_fs, self.harm)).mean()
        return loss

    def forward(self, audio_pred: torch.Tensor, audio_gt: torch.Tensor,
                valid_lengths=None) -> torch.Tensor:
        if audio_pred.dim() == 1:
            audio_pred = audio_pred.unsqueeze(0)
        if audio_gt.dim() == 1:
            audio_gt = audio_gt.unsqueeze(0)
        T = min(audio_pred.shape[-1], audio_gt.shape[-1])
        audio_pred, audio_gt = audio_pred[..., :T], audio_gt[..., :T]
        # S1 (2026-06-17 review): with per-sample valid SAMPLE counts, integrate
        # each utterance over ONLY its valid region. The renderer zero-pads audio
        # past the valid length; a real→0 step at the boundary leaks broadband
        # energy into the 48-52 Hz band AND deflates mean_env, biasing the md50
        # excess by the per-row padding fraction. valid_lengths=None keeps the
        # legacy full-row behavior (back-compat for unit tests).
        if valid_lengths is None:
            return self._excess(audio_pred, audio_gt)
        terms = []
        for b in range(audio_pred.shape[0]):
            Lb = max(0, min(int(valid_lengths[b]), T))
            if Lb < self.min_samples:
                continue
            terms.append(self._excess(audio_pred[b:b + 1, :Lb],
                                      audio_gt[b:b + 1, :Lb]))
        if not terms:
            return audio_pred.sum() * 0.0   # graph-preserving zero
        return torch.stack(terms).mean()


# -----------------------------------------------------------------------------
# 4. AudioCycleLoss — the top-level loss
# -----------------------------------------------------------------------------
class AudioCycleLoss(nn.Module):
    """Composite cycle-consistency loss for audio→VTD→PT→audio'.

    L_total = w_mr_stft * L_mr_stft + w_rms * L_rms

    2026-06-17: the Mel-L1 term has been REMOVED from the cycle loss
    unconditionally. The λ-calibration (res_local/loss_lambda_calibration_results.md)
    showed cycle-mel is (a) ~79% redundant with MR-STFT (cos(grad)=0.79 on vtd),
    (b) tremor-blind (20 ms hop aliases the 50 Hz AM to DC), and (c) the single
    largest raw-gradient term (2.217) — i.e. boosting/keeping it injects the
    biggest tremor-blind fitting gradient, which makes the artifact worse. The
    cycle loss is now MR-STFT + 0.5·RMS only.

    The ``MelL1Loss`` class is retained (it is imported/used by the SEPARATE
    DiffTract ``compute_spectral_loss`` lineage in spirit and may be reused), but
    AudioCycleLoss no longer instantiates or applies it. The legacy
    ``use_mel_l1`` / ``w_mel`` / ``n_mels`` / ``mel_n_fft`` / ``mel_hop`` kwargs
    are still ACCEPTED (so existing callers passing ``w_mel=...`` don't break) but
    are now NO-OPS. ``self.w_mel`` is pinned to 0.0 and ``self.mel_l1`` to None.

    All sub-losses operate on the raw 16 kHz waveform; gradients flow
    through ``audio_pred`` (the PT-resynthesized output). ``audio_gt``
    should be the detached HPRC reference.
    """

    def __init__(
        self,
        sr: int = 16000,
        # MR-STFT
        fft_sizes: Tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: Tuple[int, ...] = (64, 120, 240),
        win_lengths: Tuple[int, ...] = (240, 600, 1200),
        # RMS env
        rms_frame: int = 320,
        rms_hop: int = 80,
        rms_log_floor: float = 1e-4,
        # Mel L1 — DEPRECATED / NO-OP (kept for caller back-compat, see docstring).
        use_mel_l1: bool = True,
        n_mels: int = 80,
        mel_n_fft: int = 1024,
        mel_hop: int = 320,
        # Weights
        w_mr_stft: float = 1.0,
        w_rms: float = 0.5,
        w_mel: float = 0.1,  # DEPRECATED / NO-OP — see docstring.
        # 2026-08-12. Default False reproduces the older behaviour bit for bit.
        # See the note in forward().
        gain_invariant: bool = False,
        # 2026-08-17. Reproduces the B1 bug (detaching the gain). ABLATION ONLY,
        # so the 'predicted to diverge -> measured to diverge' evidence stays
        # reproducible. The normal path is always False.
        gain_invariant_legacy_detach: bool = False,
    ):
        super().__init__()
        self.gain_invariant = bool(gain_invariant)
        self.gain_invariant_legacy_detach = bool(gain_invariant_legacy_detach)
        self.w_mr_stft = float(w_mr_stft)
        self.w_rms = float(w_rms)
        # Mel removed unconditionally (2026-06-17). Pin to 0 / None regardless of
        # the deprecated use_mel_l1 / w_mel kwargs so the term cannot re-enter.
        self.w_mel = 0.0
        self.use_mel_l1 = False
        self.mel_l1 = None

        self.mr_stft = MultiResSTFTLoss(fft_sizes, hop_sizes, win_lengths)
        self.rms_env = LogRMSEnvLoss(frame=rms_frame, hop=rms_hop, log_floor=rms_log_floor)

    @staticmethod
    def _align(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Crop both tensors to the shorter length along the last axis.

        Common with resampling (e.g., 48 kHz → 16 kHz can leave a 1-sample
        rounding difference). We crop rather than pad to avoid injecting
        zeros that would bias the RMS / spectral losses.
        """
        T_pred = pred.shape[-1]
        T_gt = gt.shape[-1]
        if T_pred == T_gt:
            return pred, gt
        T_min = min(T_pred, T_gt)
        return pred[..., :T_min], gt[..., :T_min]

    def forward(
        self, audio_pred: torch.Tensor, audio_gt: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute cycle-consistency loss.

        Args:
            audio_pred: (B, T_audio) at sr Hz — differentiable PT output.
            audio_gt:   (B, T_audio) at sr Hz — HPRC reference (detached).

        Returns:
            total_loss: scalar tensor (differentiable wrt audio_pred).
            breakdown: dict with keys ``mr_stft``, ``rms_env``, ``mel_l1``,
                       ``total``. Values are Python floats (for logging).
                       ``mel_l1`` is always 0.0 (term removed 2026-06-17) but
                       kept in the dict so the downstream log schema is stable.
        """
        if audio_pred.dim() == 1:
            audio_pred = audio_pred.unsqueeze(0)
        if audio_gt.dim() == 1:
            audio_gt = audio_gt.unsqueeze(0)

        audio_pred, audio_gt = self._align(audio_pred, audio_gt)

        # ---- 2026-08-12  gain-invariant scoring (default False = old behaviour) ----
        # Why it exists: the RMS of the active part of the real corpus is 10.6 dB
        # BELOW what the renderer emits at the initial voicing value (measured),
        # and the system carries no gain parameter at all -- the model outputs
        # only {vtd, voicing}, the renderer has no learnable output gain, and
        # neither the dataset nor this loss normalises. Voicing therefore became
        # the only knob able to make up those 10 dB, and its full range happens
        # to be 9.2-10.7 dB.
        # Controlled experiment (tools/diag_level.py section 5): attenuating the
        # target by -6 / -11 dB slides the loss-optimal voicing to the floor of
        # its range; with this switch on the curves for all three attenuations
        # coincide point for point, and attenuation becomes irrelevant.
        #
        # How: match the RMS of pred to gt PER UTTERANCE, then score.
        # It must be per-utterance, not per-frame: normalising per frame would
        # erase the RMS envelope entirely, which is exactly what the rms_env
        # term exists to preserve.
        if self.gain_invariant:
            _eps = 1e-8
            _rp = audio_pred.pow(2).mean(dim=-1, keepdim=True).sqrt()
            _rg = audio_gt.pow(2).mean(dim=-1, keepdim=True).sqrt()
            # CORRECTION 2026-08-17: the gain must NOT be detached here.
            # An earlier comment justified detaching it as stopping the model
            # from manipulating the ratio by changing its output level. That
            # reasoning is backwards. Measured on the real module, scaling pred
            # by x3.4 ... x0.03:
            #   detached:  loss is constant at 1.3335 (scale-invariant, as
            #              intended) but <grad, pred> is constant at +0.317 -- a
            #              'turn the volume down' force whose magnitude does not
            #              decay as the volume falls and which the loss cannot
            #              see. Obeying it never lowers the loss, so the force
            #              never decays and there is no fixed point. Meanwhile
            #              ||grad|| is strictly proportional to 1/rms(pred)
            #              (0.085 / 0.288 / 0.959 / 2.877 / 9.590), so the
            #              quieter it gets the larger the gradient -- this is the
            #              56 -> 686 curve in the 130-step probe.
            #   attached:  <grad, pred> = -1.8e-8, i.e. zero; loss and ||grad||
            #              barely move.
            # The mathematics: with the gain attached, c*x = rms_gt * x / rms(x)
            # is homogeneous of degree zero in x, so the gradient is
            # automatically orthogonal to x. Detaching takes the gradient at one
            # scaling point instead, leaving a non-zero radial component. That
            # radial force has the closed form (1/2) * sc (the
            # spectral-convergence term), always >= 0, vanishing only at pred == gt.
            _gain = _rg / _rp.clamp_min(_eps)
            if self.gain_invariant_legacy_detach:
                _gain = _gain.detach()          # ablation path only, reproduces B1
            audio_pred = audio_pred * _gain

        l_mr = self.mr_stft(audio_pred, audio_gt)
        l_rms = self.rms_env(audio_pred, audio_gt)

        # Mel-L1 removed unconditionally (2026-06-17). Cycle = MR-STFT + 0.5·RMS.
        total = (
            self.w_mr_stft * l_mr
            + self.w_rms * l_rms
        )

        breakdown = {
            "gain_invariant": float(bool(self.gain_invariant)),
            "mr_stft": float(l_mr.detach().item()),
            "rms_env": float(l_rms.detach().item()),
            "mel_l1": 0.0,
            "total": float(total.detach().item()),
        }
        return total, breakdown


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import math

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device = {device}")

    loss_fn = AudioCycleLoss().to(device)
    print(f"[smoke] AudioCycleLoss instantiated. "
          f"w_mr_stft={loss_fn.w_mr_stft}, w_rms={loss_fn.w_rms}, w_mel={loss_fn.w_mel}")
    # 2026-06-17: mel must be REMOVED from the cycle loss (even though w_mel=0.1
    # is still passed by legacy callers / the default kwarg).
    assert loss_fn.w_mel == 0.0 and loss_fn.mel_l1 is None and not loss_fn.use_mel_l1, \
        "AudioCycleLoss must no longer carry the Mel-L1 term"
    # ...and even if a caller explicitly asks for mel, it stays off.
    _lf2 = AudioCycleLoss(use_mel_l1=True, w_mel=0.5).to(device)
    assert _lf2.w_mel == 0.0 and _lf2.mel_l1 is None, "mel kwargs must be no-ops"
    print("[smoke] mel-removal confirmed (w_mel pinned 0, mel_l1=None, kwargs no-op)")

    # ---- (a) basic random shapes ----
    pred = torch.randn(2, 32000, device=device)
    gt = torch.randn(2, 32000, device=device)
    total, bd = loss_fn(pred, gt)
    print(f"\n[a] random pred vs random gt, shape (2,32000):")
    print(f"    pred.shape={pred.shape}, gt.shape={gt.shape}")
    print(f"    total={float(total):.4f}, breakdown={bd}")
    assert total.dim() == 0, "total must be scalar"
    assert set(bd.keys()) == {"mr_stft", "rms_env", "mel_l1", "total"}, bd.keys()
    for k, v in bd.items():
        assert math.isfinite(v), f"{k} not finite: {v}"

    # ---- (b) pred == gt → should be ~0 ----
    pred_eq = gt.clone()
    total_eq, bd_eq = loss_fn(pred_eq, gt)
    print(f"\n[b] pred == gt (identical):")
    print(f"    total={float(total_eq):.6f}, breakdown={bd_eq}")
    assert float(total_eq) < 1e-3, f"identical inputs should give ~0 loss, got {float(total_eq)}"

    # ---- (c) pred all zeros, gt random → high loss ----
    pred_zero = torch.zeros(2, 32000, device=device)
    total_zero, bd_zero = loss_fn(pred_zero, gt)
    print(f"\n[c] pred=zeros, gt=random:")
    print(f"    total={float(total_zero):.4f}, breakdown={bd_zero}")
    assert float(total_zero) > float(total_eq), \
        "zero-vs-random must exceed identical-pair loss"

    # ---- (d) length mismatch (pred 32000, gt 31000) → should crop ----
    pred_long = torch.randn(2, 32000, device=device)
    gt_short = torch.randn(2, 31000, device=device)
    total_mm, bd_mm = loss_fn(pred_long, gt_short)
    print(f"\n[d] pred=(2,32000), gt=(2,31000) → cropped to min:")
    print(f"    total={float(total_mm):.4f}, breakdown={bd_mm}")
    assert math.isfinite(float(total_mm)), "length-mismatch total must be finite"

    # ---- (e) backward / gradient check ----
    pred_grad = torch.randn(2, 32000, device=device, requires_grad=True)
    gt_grad = torch.randn(2, 32000, device=device)
    total_g, _ = loss_fn(pred_grad, gt_grad)
    total_g.backward()
    grad_norm = float(pred_grad.grad.norm().item())
    print(f"\n[e] backward test:")
    print(f"    total={float(total_g):.4f}, |grad|={grad_norm:.4f}")
    assert pred_grad.grad is not None, "gradient must exist"
    assert torch.isfinite(pred_grad.grad).all(), "gradient must be finite everywhere"

    # ---- (f) memory at smoke-test scale ----
    if device.type == "cuda":
        max_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        print(f"\n[f] peak CUDA memory @ smoke scale (B=2, T=32000): {max_mb:.1f} MB")
    else:
        print("\n[f] (CUDA not available, skipping memory probe)")

    # ---- (g) TremorLoss: clean==0, AM>0, am_depth parity, grad finite ----
    sr = 16000
    tt = torch.arange(sr, device=device) / sr
    trem = (1.0 + 0.3 * torch.sin(2 * torch.pi * 50 * tt)) * torch.sin(2 * torch.pi * 150 * tt)
    clean = torch.sin(2 * torch.pi * 150 * tt)

    # am_depth parity with the validated prototype (0.2113 for 0.3-depth 50 Hz AM).
    md = float(am_depth(trem[None]).item())
    print(f"\n[g] TremorLoss block:")
    print(f"    am_depth @50Hz trembly={md:.4f} (prototype ref 0.2113), "
          f"clean={float(am_depth(clean[None]).item()):.4f}")
    assert abs(md - 0.2113) < 1e-3, f"am_depth parity broke: {md} vs 0.2113"

    tl = TremorLoss().to(device)            # default w_harm=0.0 (md50-equivalent)
    l_trem = float(tl(trem[None], clean[None]).item())
    l_clean = float(tl(clean[None], clean[None]).item())
    print(f"    TremorLoss(trembly vs clean)={l_trem:.4f} (>0)  "
          f"TremorLoss(clean vs clean)={l_clean:.6f} (==0)")
    assert l_clean == 0.0, f"clean-vs-clean must be exactly 0, got {l_clean}"
    assert l_trem > 0.0, f"trembly-vs-clean must be > 0, got {l_trem}"
    assert tl.w_harm == 0.0, "production TremorLoss default w_harm must be 0.0"

    ap = trem[None].clone().requires_grad_(True)
    tl(ap, clean[None]).backward()
    assert torch.isfinite(ap.grad).all(), "TremorLoss grad must be finite"
    print(f"    TremorLoss grad finite=True |grad|={ap.grad.abs().mean().item():.3e}")

    # harmonic path stays available
    tl_h = TremorLoss(w_harm=0.5).to(device)
    assert float(tl_h(trem[None], clean[None]).item()) >= l_trem - 1e-9, \
        "harmonic-enabled loss should be >= base"
    print("    harmonic path (w_harm=0.5) available — OK")

    print("\n[smoke] ALL CHECKS PASSED")
