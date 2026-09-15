#!/usr/bin/env python
"""DSP audit for the thesis review (2026-06-16).

Two questions the code reading could not settle, answered by measurement:

  A. 250 Hz line: the K-anchor effective IR is C0-continuous but has slope
     discontinuities at sub-frame boundaries (every spf/K = 192 samples =>
     250 Hz for K=5). Does that leave a residual 250 Hz (=K*50) line?
     -> drive waveguide with real LF source on jitter + wobble VTD; compare
        hard-switch vs K=5 envelope-spectrum lines at 50,100,...,350 Hz.

  B. 512-tap IR truncation: is 10.7 ms enough, esp. for high-Q near-closed
     tracts where ringing is longest? -> compute a long (4096-tap) reference
     IR for open-vowel / alveolar-constriction / near-closure geometries,
     measure the IR energy beyond tap 512, and the log-spectral distance
     between 512-tap and 4096-tap renderings.
"""
import os
os.environ.setdefault('DIFFPT_DISABLE_COMPILE', '1')
import sys, json, time
import numpy as np
import torch
from scipy.signal import periodogram, hilbert, welch
import librosa

HERE = os.path.dirname(os.path.abspath(__file__))
REV2 = os.path.abspath(os.path.join(HERE, '..', '..'))
if REV2 not in sys.path:
    sys.path.insert(0, REV2)
from diff_pt_v2.components.waveguide import WaveguideTract                  # noqa: E402
from diff_pt_v2.components.audio_system import DifferentiableAudioSystem    # noqa: E402

SR, FRAME, FR, F0 = 48000, 960, 50, 120
DEV = torch.device('cpu')
torch.manual_seed(0)

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
    vtd = npz[keys[0]].astype(np.float32)
    base = torch.from_numpy(vtd[vtd.shape[0] // 2])
    base[0] = 0.01  # force oral (velum closed)
    return base


VOWEL = load_vowel_base()
print(f"vowel base: velum={VOWEL[0]:.3f}  oral[min/mean/max]="
      f"{VOWEL[1:].min():.2f}/{VOWEL[1:].mean():.2f}/{VOWEL[1:].max():.2f}")


# ----------------------------------------------------------------------- #
# Test A — 250 Hz (= K*50) line check
# ----------------------------------------------------------------------- #
def env_spectrum(audio):
    env = np.abs(hilbert(audio.astype(np.float64)))
    env = librosa.resample(env.astype(np.float32), orig_sr=SR, target_sr=1000)
    env = env - env.mean()
    f, pxx = periodogram(env, fs=1000, nfft=8192)
    return f, pxx


def line_db(f, pxx, hz, tol=2.0):
    """Power at `hz` above a broadband floor that EXCLUDES all 50-Hz
    multiples (so a 250 Hz line is measured against non-harmonic noise)."""
    line = pxx[(f >= hz - tol) & (f <= hz + tol)].max()
    mask = (f >= 40) & (f <= 520)
    for h in range(50, 551, 50):
        mask &= ~((f >= h - 4) & (f <= h + 4))
    floor = np.median(pxx[mask])
    return 10 * np.log10(max(line, 1e-20)) - 10 * np.log10(max(floor, 1e-20))


def vowel_vtd(T):
    return VOWEL.view(1, 1, 45).repeat(1, T, 1).clone()


def jitter_vtd(T, std=0.15, dims=tuple(range(15, 31))):
    vtd = vowel_vtd(T); g = torch.Generator().manual_seed(7)
    for d in dims:
        vtd[0, :, d] = (vtd[0, :, d] + std * torch.randn(T, generator=g)).clamp(min=0.05)
    return vtd


def wobble_vtd(T, f_d, A=0.6, dim=20):
    vtd = vowel_vtd(T); t = torch.arange(T).float()
    vtd[0, :, dim] = (vtd[0, :, dim] + A * torch.sin(2 * np.pi * f_d * t / FR)).clamp(min=0.05)
    return vtd


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


HARMONICS = [50, 100, 150, 200, 250, 300, 350]


def testA():
    print("\n[Test A] 250 Hz (=K*50) line: hard-switch vs K=5")
    T = 160; gl = real_glottal(T); res = {}
    for name, vtd in (('wobble_10Hz', wobble_vtd(T, 10.0)),
                      ('frame_jitter', jitter_vtd(T))):
        row = {}
        for scheme, audio in (('hard', run_hardswitch(vtd, gl)),
                              ('K5', run_kanchor(5, vtd, gl))):
            f, p = env_spectrum(audio)
            row[scheme] = {str(h): round(float(line_db(f, p, h)), 1) for h in HARMONICS}
        res[name] = row
        print(f"  [{name}]")
        print("     Hz:    " + "  ".join(f"{h:>6d}" for h in HARMONICS))
        print("     hard:  " + "  ".join(f"{row['hard'][str(h)]:>6.1f}" for h in HARMONICS))
        print("     K=5:   " + "  ".join(f"{row['K5'][str(h)]:>6.1f}" for h in HARMONICS))
    return res


# ----------------------------------------------------------------------- #
# Test B — 512-tap IR truncation
# ----------------------------------------------------------------------- #
def ir_for(vtd_frame, n_ir):
    wg = WaveguideTract(n_ir_samples=n_ir, n_subframe=1).eval()
    with torch.no_grad():
        r = wg._compute_reflections(vtd_frame.view(1, 1, 45))
        ir = wg._run_ir_loop(*r)
    return ir[0, 0].numpy().astype(np.float64)


def lsd_db(y1, y2):
    f, P1 = welch(y1, fs=SR, nperseg=2048)
    _, P2 = welch(y2, fs=SR, nperseg=2048)
    band = (f >= 50) & (f <= 8000)
    d = 10 * np.log10(P1[band] + 1e-12) - 10 * np.log10(P2[band] + 1e-12)
    return float(np.sqrt(np.mean(d ** 2)))


def geometries():
    g = {}
    g['open_vowel'] = VOWEL.clone()
    alv = VOWEL.clone(); alv[28:33] = 0.05; g['alveolar_constriction_0.05'] = alv
    clo = VOWEL.clone(); clo[28:33] = 0.02; g['near_closure_0.02'] = clo
    lip = VOWEL.clone(); lip[44] = 0.05; g['lip_near_closed_0.05'] = lip
    return g


def testB():
    print("\n[Test B] 512-tap IR truncation vs 4096-tap reference")
    N_LONG = 4096
    gl = real_glottal(30)[0].numpy().astype(np.float64)
    res = {}
    for name, geo in geometries().items():
        ir = ir_for(geo, N_LONG)
        e = ir ** 2; tot = e.sum()
        cum = np.cumsum(e) / max(tot, 1e-30)
        tail512 = float(1.0 - cum[511])
        tap99 = int(np.searchsorted(cum, 0.99))
        tap999 = int(np.searchsorted(cum, 0.999))
        y512 = np.convolve(gl, ir[:512])[:len(gl)]
        y4096 = np.convolve(gl, ir)[:len(gl)]
        lsd = lsd_db(y512, y4096)
        res[name] = {'tail_energy_beyond_512': tail512, 'tap_99pct': tap99,
                     'tap_999pct': tap999, 'lsd_512_vs_4096_db': round(lsd, 3)}
        print(f"  [{name:28s}] tail>512={tail512*100:6.3f}%  "
              f"99%@{tap99:4d}  99.9%@{tap999:4d} taps  "
              f"LSD(512 vs 4096)={lsd:.3f} dB")
    return res


if __name__ == '__main__':
    t0 = time.time()
    out = {'testA_harmonic_lines_db': testA(), 'testB_ir_truncation': testB(),
           'seconds': None}
    out['seconds'] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, 'verify_dsp_audit_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nTotal {out['seconds']}s -> verify_dsp_audit_results.json")
