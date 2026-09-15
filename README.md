# DiffPT — a differentiable Pink Trombone

A fully differentiable port of the [Pink Trombone](https://dood.al/pinktrombone/)
articulatory synthesiser, built to sit inside a neural training loop. It keeps the
complete waveguide — nasal branch, turbulence and bursts included — and makes the
gradient reach the vocal-tract geometry.

This is the renderer from *A Differentiable Pink Trombone for Articulatory Inversion
without Articulatory Supervision* (ICASSP 2027, under review).

## Why the original resists differentiation

Four things, and only two of them cost anything:

| | Obstacle | Why it blocks a gradient | What we do |
|---|---|---|---|
| 1 | **The scattering recursion** | The waveguide advances two steps per output sample, so one second at 48 kHz is a chain of 96,000 dependent updates. | Hold the reflections fixed within a control block. The block is then a fixed filter: compute its 512-sample impulse response once and apply it by convolution. Six anchors per frame, cross-faded at the sample rate. |
| 2 | **Turbulence and burst gating** | The injection site is named by an externally supplied touch list, not computed from the tract shape, so no gradient exists. Bursts fire only at exact closure, which a bounded output cannot reach. | Apply the same diameter-dependent gate to every section and let the weighting find the constrictions. Release the burst at a diameter of 0.05. |
| 3 | **The glottal source** | *Appears* piecewise: the LF model splits into an open phase and a return phase. | Not actually a problem. The split is on time within a pitch period; the tract diameters never enter that function. Both branches are evaluated and blended by a mask. |
| 4 | **Velum coupling** | *Appears* to need a discrete decision at the nasal junction. | Not actually a problem. The three-way junction reflections are an algebraic function of areas, differentiable everywhere; the velum is just one coordinate of the diameter vector. |

Obstacles 3 and 4 are exact substitutions. Only 1 and 2 are approximations, and the
paper measures what they cost.

## Layout

```
diff_pt_v2/
  components/
    audio_system.py        the full renderer: source, waveguide, turbulence, bursts
    fast_audio_system.py   length-bucketed wrapper used during training
    waveguide.py           Kelly–Lochbaum tract, 44 oral + 28 nasal sections
    glottis.py             Liljencrants–Fant source
    noise.py               fricative / aspiration filter bank
    transient.py           closure-release transients
  losses/
    audio_cycle_loss.py    multi-resolution STFT, log-RMS envelope, tremor
  tests/
    test_end_to_end_vs_js.py            against the sample-wise reference
    test_v34_kanchor_antialiasing.py    the K-anchor approximation
    verify_dsp_audit.py                 impulse-response truncation audit
    waveguide_samplelevel.py            sample-level reference implementation
```

## Usage

```python
import torch
from diff_pt_v2.components.audio_system import DifferentiableAudioSystem

renderer = DifferentiableAudioSystem()          # 44 oral + 28 nasal, 48 kHz internal, 16 kHz out

vtd  = torch.full((1, T, 45), 0.6)              # velum + 44 oral diameters, per frame
vtd[0, :, 21:25] = 0.15                         # a constriction
voi  = torch.full((1, T), 0.9)                  # voicing
f0   = torch.full((1, T), 120.0)

vtd.requires_grad_(True)
audio = renderer(vtd, voi, f0)                  # (1, T * 320) at 16 kHz
audio.pow(2).mean().backward()                  # the gradient reaches vtd
```

**Set `DIFFPT_DISABLE_COMPILE=1`.** `torch.compile` unrolls the 512-step impulse-response
loop into a graph of 5000+ nodes; compilation took 362 s on CPU and did not finish on a
GPU. The reason is documented at the top of `waveguide.py`.

## Relation to other work

The synthetic corpus used in the paper was generated with the
[Node.js Pink Trombone server](https://github.com/MateoCamara/Pink-Trombone-Node-Server)
of Cámara et al. (GPL-3.0); that code is **not** vendored here, only linked.
The original Pink Trombone is by Neil Thapen.

## Citation

```bibtex
@inproceedings{xu2027diffpt,
  author    = {Zhiyuan Xu},
  title     = {A Differentiable {Pink Trombone} for Articulatory Inversion
               without Articulatory Supervision},
  booktitle = {Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2027}
}
```

## License

MIT. See [LICENSE](LICENSE).
