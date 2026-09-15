#!/usr/bin/env python
"""V34 — K-anchor anti-aliasing ablation (thesis X.10.2). Local, no training.

KEY ARCHITECTURAL FACT (found while running this): the current waveguide cross-fades
even at n_subframe=1 (K+1=2 anchors at lambda 0=prev, 1=cur, blended across the frame).
So a bare K-sweep barely moves the 50 Hz line. The artefact source is the OLD per-frame-IR
HARD-SWITCH (one cur-r IR per frame, no cross-fade), reconstructed here via
_compute_reflections -> _run_ir_loop -> _frame_aligned_convolve. The meaningful comparison
is therefore HARD-SWITCH vs CROSS-FADE (the K-anchor), not K vs K.

50 Hz artefact measured exactly as analyze_tremor.py: Hilbert envelope -> 1 kHz ->
periodogram -> line[48-52]Hz minus floor[200-400]Hz, in dB.
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
from diff_pt_v2.components.waveguide import WaveguideTract  # noqa: E402

SR, FRAME, FR, F0, NOR = 48000, 960, 50, 120, 44
torch.manual_seed(0)


def impulse_train(n):
    x = np.zeros(n, np.float32); x[::int(SR / F0)] = 1.0; return x


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


def base_vtd(T, dval=1.0):
    vtd = torch.full((1, T, 1 + NOR), float(dval)); vtd[..., 0] = 0.01; return vtd


def wobble_vtd(T, f_d, A=0.7, d0=1.0, dim=20):
    vtd = base_vtd(T)
    t = torch.arange(T).float()
    vtd[0, :, dim] = d0 + A * torch.sin(2 * np.pi * f_d * t / FR)
    return vtd


_gcache = {}
def glottal(T):
    if T not in _gcache:
        _gcache[T] = torch.from_numpy(impulse_train(T * FRAME)).unsqueeze(0)
    return _gcache[T]


def run_kanchor(K, vtd, gl):
    wg = WaveguideTract(n_subframe=K).eval()
    with torch.no_grad():
        return wg(vtd, gl).squeeze(0).numpy().astype(np.float32)


def run_hardswitch(vtd, gl):
    """Old per-frame-IR hard switch: one cur-r IR per frame, no cross-fade."""
    wg = WaveguideTract(n_subframe=1).eval()
    with torch.no_grad():
        r = wg._compute_reflections(vtd)
        ir = wg._run_ir_loop(*r)
        return wg._frame_aligned_convolve(gl, ir).squeeze(0).numpy().astype(np.float32)


def db50(audio):
    f, p = env_spectrum(audio); return float(line_above_floor(f, p, 50))


def test1_hardswitch_vs_crossfade():
    print("\n[Test 1] hard-switch vs cross-fade (5 Hz diameter wobble, +/-0.7 mm)")
    T = 160; vtd = wobble_vtd(T, f_d=5.0); gl = glottal(T)
    hard = db50(run_hardswitch(vtd, gl))
    print(f"  HARD-SWITCH (no cross-fade)   : 50Hz_above_floor = {hard:+6.2f} dB")
    rows = [{'config': 'hard-switch', 'db50': round(hard, 2)}]
    for K in (1, 2, 5, 10):
        d = db50(run_kanchor(K, vtd, gl))
        rows.append({'config': f'K-anchor K={K}', 'db50': round(d, 2)})
        print(f"  K-anchor  K={K:2d}              : 50Hz_above_floor = {d:+6.2f} dB")
    print(f"  => hard-switch minus K-anchor(K=5) = {hard - rows[2]['db50']:+.2f} dB")
    return rows


def test2_does_K_matter():
    print("\n[Test 2] does K (interpolation fineness) matter? fast 15 Hz wobble")
    T = 160; vtd = wobble_vtd(T, f_d=15.0); gl = glottal(T)
    rows = []
    for K in (1, 2, 3, 5, 8, 16):
        d = db50(run_kanchor(K, vtd, gl))
        rows.append({'K': K, 'db50': round(d, 2)})
        print(f"  K={K:2d}: 50Hz_above_floor = {d:+6.2f} dB")
    hard = db50(run_hardswitch(vtd, gl))
    print(f"  (hard-switch on same fast traj: {hard:+.2f} dB)")
    spread = max(r['db50'] for r in rows) - min(r['db50'] for r in rows)
    print(f"  K-anchor spread across K=1..16: {spread:.2f} dB")
    return {'rows': rows, 'hardswitch_db50': round(hard, 2), 'spread_db': round(spread, 2)}


def test3_step():
    print("\n[Test 3] step transient: hard-switch vs K-anchor K=5")
    T = 60; step = 30; out = {}
    for label, fn in (('hard-switch', lambda v, g: run_hardswitch(v, g)),
                      ('K-anchor K=5', lambda v, g: run_kanchor(5, v, g))):
        vtd = base_vtd(T); vtd[0, :step, 20] = 0.4; vtd[0, step:, 20] = 1.4
        a = fn(vtd, glottal(T))
        w = 64; rms = np.sqrt(np.convolve(a ** 2, np.ones(w) / w, 'same')); b = step * FRAME
        slope = float(np.max(np.abs(np.diff(rms[b - 200:b + 600]))))
        out[label] = round(slope, 6)
        print(f"  {label:14s}: peak |d(env)/dt| at step = {slope:.3e}")
    ratio = out['hard-switch'] / max(out['K-anchor K=5'], 1e-12)
    print(f"  hard-switch / K=5 peak-slope ratio = {ratio:.2f}  (>1 => K=5 smoother)")
    return {'peak_slope': out, 'ratio_hard_over_K5': round(ratio, 2),
            'samples_per_subframe_K5': FRAME // 5}


if __name__ == '__main__':
    t0 = time.time()
    out = {'test1_hardswitch_vs_crossfade': test1_hardswitch_vs_crossfade(),
           'test2_does_K_matter': test2_does_K_matter(),
           'test3_step': test3_step(),
           'total_seconds': None}
    out['total_seconds'] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, 'V34_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nTotal {out['total_seconds']}s -> V34_results.json")
