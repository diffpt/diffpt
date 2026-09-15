#!/usr/bin/env python
"""V34 realistic re-run: real LF glottal source + a real vowel tract (not uniform tube).

Same three tests as test_v34_kanchor_antialiasing.py, but the excitation is the project's
own GlottalSourceGenerator (LF pulse + aspiration) and the base tract is a real steady-vowel
VTD loaded from the local PT corpus, animated by a wobble. This makes the 50 Hz level
comparable to real-speech conditions instead of below-floor.
"""
import os
os.environ.setdefault('DIFFPT_DISABLE_COMPILE', '1')
import sys, json, time
import numpy as np
import torch
from scipy.signal import periodogram, hilbert
import librosa

HERE = os.path.dirname(os.path.abspath(__file__))
REV2 = os.path.abspath(os.path.join(HERE, '..', '..'))
if REV2 not in sys.path:
    sys.path.insert(0, REV2)
from diff_pt_v2.components.waveguide import WaveguideTract            # noqa: E402
from diff_pt_v2.components.audio_system import DifferentiableAudioSystem  # noqa: E402

SR, FRAME, FR, F0, NOR = 48000, 960, 50, 120, 44
torch.manual_seed(0)
DEV = torch.device('cpu')

_sys = DifferentiableAudioSystem(sr_internal=48000, sr_out=16000, frame_rate=50,
                                 n_oral=44, n_nose=28, n_ir_samples=512,
                                 closure_eps=1e-6, seed=31337).to(DEV).eval()


def real_glottal(T):
    f0 = torch.full((1, T), float(F0)); voicing = torch.ones(1, T)
    with torch.no_grad():
        asp, _ = _sys.noise(n_samples=T * FRAME, batch_size=1, device=DEV)
        g = _sys.glottis(f0, voicing, asp)
    return g.to(torch.float32)


def load_vowel_base():
    p = os.path.join(REV2, '..', '..', '..', 'v1', 'py', 'data_phase4_s2', 'worker0_batch0000.npz')
    p = os.path.abspath(p)
    npz = np.load(p)
    keys = [k for k in npz.files if k.startswith('target_')]
    vtd = npz[keys[0]].astype(np.float32)          # (T, 45)
    base = vtd[vtd.shape[0] // 2]                   # a steady mid frame
    return torch.from_numpy(base), p


VOWEL, VOWEL_PATH = load_vowel_base()
print(f"vowel base from {os.path.basename(VOWEL_PATH)}: "
      f"velum={VOWEL[0]:.3f}  oral[min/mean/max]={VOWEL[1:].min():.2f}/{VOWEL[1:].mean():.2f}/{VOWEL[1:].max():.2f}")


def env_spectrum(audio):
    env = np.abs(hilbert(audio.astype(np.float64)))
    env = librosa.resample(env.astype(np.float32), orig_sr=SR, target_sr=1000)
    env = env - env.mean()
    f, pxx = periodogram(env, fs=1000, nfft=8192)
    return f, pxx


def line_above_floor(f, pxx, hz, tol=2.0):
    line = pxx[(f >= hz - tol) & (f <= hz + tol)].mean()
    floor = pxx[(f >= 200) & (f <= 400)].mean()
    return 10 * np.log10(max(line, 1e-20)) - 10 * np.log10(max(floor, 1e-20))


def vowel_vtd(T):
    vtd = VOWEL.view(1, 1, 45).repeat(1, T, 1).clone()
    vtd[..., 0] = 0.01  # force oral (velum closed); loaded sample may be nasalised
    return vtd


def jitter_vtd(T, std=0.15, dims=tuple(range(15, 31))):
    """Per-frame i.i.d. perturbation on mid-oral dims — mimics the model's
    frame-rate-noisy VTD predictions that actually drove the +15 dB tremor."""
    vtd = vowel_vtd(T); g = torch.Generator().manual_seed(7)
    for d in dims:
        vtd[0, :, d] = (vtd[0, :, d] + std * torch.randn(T, generator=g)).clamp(min=0.05)
    return vtd


def wobble_vtd(T, f_d, A=0.6, dim=20):
    vtd = vowel_vtd(T); t = torch.arange(T).float()
    vtd[0, :, dim] = (vtd[0, :, dim] + A * torch.sin(2 * np.pi * f_d * t / FR)).clamp(min=0.05)
    return vtd


def step_vtd(T, step, dim=20, lo=0.4, hi=1.4):
    vtd = vowel_vtd(T); vtd[0, :step, dim] = lo; vtd[0, step:, dim] = hi; return vtd


def run_kanchor(K, vtd, gl):
    wg = WaveguideTract(n_subframe=K).eval()
    with torch.no_grad():
        return wg(vtd, gl).squeeze(0).numpy().astype(np.float32)


def run_hardswitch(vtd, gl):
    wg = WaveguideTract(n_subframe=1).eval()
    with torch.no_grad():
        r = wg._compute_reflections(vtd)
        ir = wg._run_ir_loop(*r)
        return wg._frame_aligned_convolve(gl, ir).squeeze(0).numpy().astype(np.float32)


def db50(audio):
    f, p = env_spectrum(audio); return float(line_above_floor(f, p, 50))


def test1():
    print("\n[Test 1] hard-switch vs cross-fade (real LF + real ORAL vowel)")
    T = 160; gl = real_glottal(T); res = {}
    for name, vtd in (('smooth_5Hz_wobble', wobble_vtd(T, 5.0)),
                      ('frame_rate_jitter', jitter_vtd(T))):
        hard = db50(run_hardswitch(vtd, gl))
        ks = {f'K={K}': round(db50(run_kanchor(K, vtd, gl)), 2) for K in (1, 2, 5, 10)}
        print(f"  [{name}] hard-switch={hard:+.2f} dB   "
              + "  ".join(f"{k}:{v:+.2f}" for k, v in ks.items())
              + f"   (hard - K=5 = {hard - ks['K=5']:+.2f} dB)")
        res[name] = {'hardswitch': round(hard, 2), 'kanchor': ks,
                     'hard_minus_K5': round(hard - ks['K=5'], 2)}
    return res


def test2():
    print("\n[Test 2] does K matter? fast 15 Hz wobble (real LF + real vowel)")
    T = 160; vtd = wobble_vtd(T, 15.0); gl = real_glottal(T); rows = []
    for K in (1, 2, 3, 5, 8, 16):
        d = db50(run_kanchor(K, vtd, gl)); rows.append({'K': K, 'db50': round(d, 2)})
        print(f"  K={K:2d}: 50Hz_above_floor = {d:+6.2f} dB")
    hard = db50(run_hardswitch(vtd, gl))
    spread = max(r['db50'] for r in rows) - min(r['db50'] for r in rows)
    print(f"  (hard-switch: {hard:+.2f} dB)   K spread = {spread:.2f} dB")
    return {'rows': rows, 'hardswitch_db50': round(hard, 2), 'spread_db': round(spread, 2)}


def test3():
    print("\n[Test 3] step transient: hard-switch vs K=5 (real LF + real vowel)")
    T = 60; step = 30; out = {}
    for label, fn in (('hard-switch', lambda v, g: run_hardswitch(v, g)),
                      ('K-anchor K=5', lambda v, g: run_kanchor(5, v, g))):
        a = fn(step_vtd(T, step), real_glottal(T))
        w = 64; rms = np.sqrt(np.convolve(a ** 2, np.ones(w) / w, 'same')); b = step * FRAME
        slope = float(np.max(np.abs(np.diff(rms[b - 200:b + 600])))); out[label] = round(slope, 6)
        print(f"  {label:14s}: peak |d(env)/dt| = {slope:.3e}")
    ratio = out['hard-switch'] / max(out['K-anchor K=5'], 1e-12)
    print(f"  hard-switch / K=5 ratio = {ratio:.2f}")
    return {'peak_slope': out, 'ratio_hard_over_K5': round(ratio, 2)}


if __name__ == '__main__':
    t0 = time.time()
    out = {'vowel_base': {'velum': round(float(VOWEL[0]), 3),
                          'oral_mean': round(float(VOWEL[1:].mean()), 3),
                          'source': os.path.basename(VOWEL_PATH)},
           'test1': test1(), 'test2': test2(), 'test3': test3()}
    out['total_seconds'] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, 'V34_realistic_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nTotal {out['total_seconds']}s -> V34_realistic_results.json")
