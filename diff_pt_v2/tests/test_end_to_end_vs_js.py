#!/usr/bin/env python
"""test_end_to_end_vs_js.py — Compare PyTorch `DifferentiableAudioSystem` output
against JS Pink Trombone oracle audio. This is the end-to-end Phase 4 verification
step that was deferred during the dual-agent rounds.

WHAT THIS DOES
==============
For each sample in `audit/audit_pt/` (already synthesized by JS PT via
audit_gt_resynth.py):
  1. Load the GT inputs (VTD + voicing + f0) from the original npz batch
  2. Feed them into our PyTorch `DifferentiableAudioSystem`
  3. Get audio_py @ 16 kHz
  4. Compare to:
     - audio_oracle (JS PT synth from same GT inputs)  ← the "match this" target
     - audio_gt (recorded/original audio in the npz)   ← the ultimate ground truth

We report THREE pairwise comparisons:
  - audio_py     ↔ audio_oracle  (port fidelity — is our PyTorch port the same as JS?)
  - audio_py     ↔ audio_gt      (model upper bound when feeding GT VTD)
  - audio_oracle ↔ audio_gt      (audit's own number, included as a consistency check)

WHY
===
Run 13a will train with `audio_cycle_loss(audio_pred, audio_gt)`. The pipeline is:
  VTD (predicted) → DifferentiableAudioSystem → audio_pred → loss vs audio_gt.

If our PyTorch port (DifferentiableAudioSystem) doesn't faithfully reproduce JS PT
output, then `audio_pred` will be optimized to match `audio_gt` through a wrong
synthesizer — defeating the purpose. This test asks: when we feed GT VTD into both
synthesizers, do they produce similar audio?

PASS CRITERIA (per design/03_verification_plan.md, loosened to be empirical)
============================================================================
  mel_L1(audio_py, audio_oracle) mean < 1.5      (port "close enough")
  correlation                  mean > 0.70       (similar waveforms)
  |closure_count_py − closure_count_oracle| ≤ 1  (same number of stops detected)

OUTCOMES
========
  - PASS:  PyTorch port faithfully reproduces JS PT. Training will optimize the
           correct target. Go ahead with Run 13a as configured.
  - WARNING: structural drift between PyTorch and JS PT. Training will still
           run but optimize against a slightly different audio than JS oracle.
           Likely causes (in order of suspicion):
              (a) Time-varying convolution OA approximation (per-frame IR vs JS
                  sample-by-sample waveguide state continuity)
              (b) Fricative noise additive-post-tract vs JS in-waveguide injection
              (c) Downsample 48 → 16 kHz filter differences
              (d) Per-frame stateless glottal vs JS continuous-state glottal
           In WARNING case, you still benefit from the new closure-sensitive
           loss, but should re-test after Run 13a finishes — if Run 13a audio
           sounds different from JS oracle, this verification's "WARNING" was
           the cause.

USAGE
=====
From `phase_t1_rev2/src/`:

    python diff_pt_v2/tests/test_end_to_end_vs_js.py \\
        --audit-dir audit/audit_pt \\
        --pt-npz-dir /path/to/pt_corpus_npz \\
        --out-dir diff_pt_v2/tests/results_end_to_end \\
        --device cuda \\
        --resynth

Args:
  --audit-dir    dir produced by audit_gt_resynth.py (has *_oracle.wav)
                 default: audit/audit_pt
  --pt-npz-dir   PT npz batches (for loading GT VTD per sample)
                 required; no default
  --out-dir      where to write comparison.json + (optional) audio_py wavs
                 default: diff_pt_v2/tests/results_end_to_end
  --device       cuda or cpu
  --resynth      also save PyTorch-generated wav for listening
  --max-samples  cap (default: all in audit_dir)

OUTPUT
======
  comparison.json:
    {
      "per_sample": [
        {
          "sample_id": "105773",
          "status": "ok",
          "T_frames": 50,
          "mel_l1_py_vs_oracle": 0.42,
          "mel_l1_py_vs_gt":     0.78,
          "mel_l1_oracle_vs_gt": 0.60,
          "correlation_py_vs_oracle": 0.91,
          "correlation_py_vs_gt":     0.83,
          "closure_count_py":     1,
          "closure_count_oracle": 1,
          "closure_count_gt":     0,
          ...
        }, ...
      ],
      "summary": {
        "mel_l1_py_vs_oracle":      {"mean": 0.45, "median": 0.40, ...},
        "correlation_py_vs_oracle": {"mean": 0.88, ...},
        "closure_count_abs_diff":   {"mean": 0.3, ...},
        "pass_mel": true,
        "pass_correlation": true,
        "pass_closure": true,
        "overall_pass": true,
      }
    }

  audio_py/<sample_id>_py.wav (only if --resynth):
    PyTorch-port-generated audio. Listen and compare to <audit_dir>/<sid>_oracle.wav.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Optional

import numpy as np
import soundfile as sf
import torch

try:
    import librosa
except ImportError:
    print("ERROR: librosa required. Install: pip install librosa", file=sys.stderr)
    sys.exit(1)

# ── Path setup ──────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DIFF_PT_V2 = os.path.dirname(HERE)
REV2_SRC = os.path.dirname(DIFF_PT_V2)
for p in (REV2_SRC, DIFF_PT_V2):
    if p not in sys.path:
        sys.path.insert(0, p)

# Imports from diff_pt_v2 (qualified path)
from diff_pt_v2.components.audio_system import DifferentiableAudioSystem  # noqa: E402


# ── Sample loader ───────────────────────────────────────────────────────────
def load_pt_sample(pt_npz_dir: str, sid: str) -> Optional[tuple]:
    """Find sample_id in any worker*_batch*.npz and return (audio, vtd, f0_hz, voicing).

    All as np.float32 ndarrays.
    audio:   (T_audio,) at 16 kHz
    vtd:     (T_vtd, 45)
    f0_hz:   (T_vtd,) Hz
    voicing: (T_vtd,) [0, 1]
    Returns None if not found.
    """
    sid_int = int(sid)
    meta_files = sorted(glob.glob(os.path.join(pt_npz_dir, "*_meta.json")))
    for mf in meta_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                meta_list = json.load(f)
        except Exception:
            continue
        npz_path = mf.replace("_meta.json", ".npz")
        if not os.path.isfile(npz_path):
            continue
        # quick check: is sid in this batch's meta?
        sids_in_batch = {int(m["sample_id"]) for m in meta_list}
        if sid_int not in sids_in_batch:
            continue
        npz = np.load(npz_path, allow_pickle=False)
        try:
            for local_idx, m in enumerate(meta_list):
                if int(m["sample_id"]) != sid_int:
                    continue
                audio = np.asarray(npz[f"audio_{local_idx}"], dtype=np.float32)
                vtd = np.asarray(npz[f"target_{local_idx}"], dtype=np.float32)
                ctrl_key = f"control_{local_idx}"
                if ctrl_key in npz.files:
                    ctrl = np.asarray(npz[ctrl_key], dtype=np.float32)
                    f0 = ctrl[0]
                    voicing = ctrl[1]
                else:
                    f0 = np.full(vtd.shape[0], 140.0, dtype=np.float32)
                    voicing = np.ones(vtd.shape[0], dtype=np.float32)
                return audio, vtd, f0, voicing
        finally:
            npz.close()
    return None


# ── Audio metrics ───────────────────────────────────────────────────────────
def compute_mel_l1(a: np.ndarray, b: np.ndarray,
                   sr: int = 16000, n_mels: int = 80,
                   n_fft: int = 1024, hop: int = 320) -> float:
    """Log-mel L1 between two audio arrays. Crops to common length."""
    n = min(len(a), len(b))
    if n < n_fft:
        return float("nan")
    a, b = a[:n].astype(np.float32), b[:n].astype(np.float32)
    ma = librosa.feature.melspectrogram(y=a, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop)
    mb = librosa.feature.melspectrogram(y=b, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop)
    la = np.log(np.maximum(ma, 1e-6))
    lb = np.log(np.maximum(mb, 1e-6))
    T = min(la.shape[1], lb.shape[1])
    return float(np.abs(la[:, :T] - lb[:, :T]).mean())


def compute_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Time-domain Pearson correlation. Crops to common length, mean-centers."""
    n = min(len(a), len(b))
    if n < 100:
        return float("nan")
    a = a[:n].astype(np.float64) - a[:n].mean()
    b = b[:n].astype(np.float64) - b[:n].mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def detect_closures(x: np.ndarray, sr: int = 16000, hop: int = 320,
                    db_thresh: float = -40.0, min_frames: int = 3) -> list:
    """Detect silence runs (RMS < db_thresh dB) of duration ≥ min_frames frames.

    Returns a list of run durations in ms.
    """
    n = max(1, len(x) // hop)
    rms = np.array([np.sqrt(np.mean(x[i*hop:(i+1)*hop]**2) + 1e-12) for i in range(n)])
    rms_db = 20.0 * np.log10(rms + 1e-6)
    is_closed = rms_db < db_thresh
    closures = []
    i = 0
    while i < len(is_closed):
        if is_closed[i]:
            j = i
            while j < len(is_closed) and is_closed[j]:
                j += 1
            if j - i >= min_frames:
                closures.append((j - i) * hop / sr * 1000.0)
            i = j
        else:
            i += 1
    return closures


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(
        description="End-to-end PyTorch DifferentiableAudioSystem vs JS PT oracle"
    )
    p.add_argument("--audit-dir", default="audit/audit_pt",
                   help="dir from audit_gt_resynth.py (has *_oracle.wav, audit_metrics.json)")
    p.add_argument("--pt-npz-dir", default=os.path.expanduser(
        ""
    ))
    p.add_argument("--out-dir", default="diff_pt_v2/tests/results_end_to_end")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resynth", action="store_true",
                   help="Also save audio_py wav for each sample (for listening test)")
    p.add_argument("--max-samples", type=int, default=-1)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.resynth:
        os.makedirs(os.path.join(args.out_dir, "audio_py"), exist_ok=True)

    device = torch.device(args.device)
    print(f"[setup] device     = {device}")
    print(f"[setup] audit_dir  = {args.audit_dir}")
    print(f"[setup] pt_npz_dir = {args.pt_npz_dir}")
    print(f"[setup] out_dir    = {args.out_dir}")
    print(f"[setup] resynth    = {args.resynth}")

    # ── Step 1: enumerate oracle wavs ───────────────────────────────────────
    oracle_paths = sorted(glob.glob(os.path.join(args.audit_dir, "*_oracle.wav")))
    if not oracle_paths:
        print(f"\nERROR: no *_oracle.wav under {args.audit_dir}", file=sys.stderr)
        print(f"  → run `python classification/audit_gt_resynth.py` first.", file=sys.stderr)
        return 1
    if args.max_samples > 0:
        oracle_paths = oracle_paths[:args.max_samples]
    print(f"\n[Step 1] Found {len(oracle_paths)} oracle wavs in {args.audit_dir}")

    # ── Step 2: instantiate DifferentiableAudioSystem ───────────────────────
    print(f"\n[Step 2] Loading DifferentiableAudioSystem on {device} ...")
    system = DifferentiableAudioSystem(
        sr_internal=48000, sr_out=16000, frame_rate=50,
        n_oral=44, n_nose=28, n_ir_samples=512,
        closure_eps=1e-6, seed=31337,
    ).to(device).eval()
    print(f"  System ready. Internal sr=48kHz, output sr=16kHz, frame_rate=50")

    # ── Step 3: per-sample comparison ───────────────────────────────────────
    results = []
    print(f"\n[Step 3] Running per-sample comparison ({len(oracle_paths)} samples) ...")
    for i, oracle_path in enumerate(oracle_paths):
        sid = os.path.basename(oracle_path).replace("_oracle.wav", "")
        print(f"\n  [{i+1}/{len(oracle_paths)}] sample {sid}")

        # 3.1 Load GT inputs from npz
        loaded = load_pt_sample(args.pt_npz_dir, sid)
        if loaded is None:
            print(f"    SKIP: sample {sid} not found in {args.pt_npz_dir}")
            results.append({"sample_id": sid, "status": "not_found"})
            continue
        gt_audio, vtd, f0, voicing = loaded
        T_frames = int(vtd.shape[0])

        # 3.2 Load oracle wav (JS PT output from audit)
        oracle_audio, sr = sf.read(oracle_path)
        oracle_audio = oracle_audio.astype(np.float32)
        if sr != 16000:
            print(f"    WARN: oracle sr={sr}, expected 16000 — resampling")
            oracle_audio = librosa.resample(oracle_audio, orig_sr=sr, target_sr=16000)

        # 3.3 Forward through DifferentiableAudioSystem
        try:
            with torch.no_grad():
                vtd_t = torch.from_numpy(vtd).unsqueeze(0).to(device)            # (1, T, 45)
                voicing_t = torch.from_numpy(voicing).unsqueeze(0).to(device)    # (1, T)
                f0_t = torch.from_numpy(f0).unsqueeze(0).to(device).clamp(min=20.0)
                audio_py_t = system(vtd_t, voicing_t, f0_t)                      # (1, T*320)
                audio_py = audio_py_t.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"    ERROR forward: {e}")
            results.append({"sample_id": sid, "status": "forward_error", "error": str(e)})
            continue

        # 3.4 Save PyTorch audio if requested
        if args.resynth:
            sf.write(os.path.join(args.out_dir, "audio_py", f"{sid}_py.wav"),
                     audio_py, 16000)

        # 3.5 Compute metrics
        try:
            mel_l1_py_oracle = compute_mel_l1(audio_py, oracle_audio)
            mel_l1_py_gt = compute_mel_l1(audio_py, gt_audio)
            mel_l1_oracle_gt = compute_mel_l1(oracle_audio, gt_audio)
            corr_py_oracle = compute_correlation(audio_py, oracle_audio)
            corr_py_gt = compute_correlation(audio_py, gt_audio)
            cl_py = detect_closures(audio_py)
            cl_oracle = detect_closures(oracle_audio)
            cl_gt = detect_closures(gt_audio)
        except Exception as e:
            print(f"    ERROR metrics: {e}")
            results.append({"sample_id": sid, "status": "metric_error", "error": str(e)})
            continue

        result = {
            "sample_id": sid,
            "status": "ok",
            "T_frames": T_frames,
            "T_audio_py": int(len(audio_py)),
            "T_audio_oracle": int(len(oracle_audio)),
            "T_audio_gt": int(len(gt_audio)),
            "mel_l1_py_vs_oracle": mel_l1_py_oracle,
            "mel_l1_py_vs_gt":     mel_l1_py_gt,
            "mel_l1_oracle_vs_gt": mel_l1_oracle_gt,
            "correlation_py_vs_oracle": corr_py_oracle,
            "correlation_py_vs_gt":     corr_py_gt,
            "closure_count_py":     len(cl_py),
            "closure_count_oracle": len(cl_oracle),
            "closure_count_gt":     len(cl_gt),
            "longest_closure_ms_py":     float(max(cl_py)) if cl_py else 0.0,
            "longest_closure_ms_oracle": float(max(cl_oracle)) if cl_oracle else 0.0,
            "longest_closure_ms_gt":     float(max(cl_gt)) if cl_gt else 0.0,
            "rms_py":     float(np.sqrt(np.mean(audio_py ** 2))),
            "rms_oracle": float(np.sqrt(np.mean(oracle_audio ** 2))),
            "rms_gt":     float(np.sqrt(np.mean(gt_audio ** 2))),
        }
        results.append(result)
        print(f"    mel_L1:  py↔oracle={mel_l1_py_oracle:.3f}  py↔gt={mel_l1_py_gt:.3f}  oracle↔gt={mel_l1_oracle_gt:.3f}")
        print(f"    corr:    py↔oracle={corr_py_oracle:.3f}  py↔gt={corr_py_gt:.3f}")
        print(f"    closures: py={len(cl_py)}  oracle={len(cl_oracle)}  gt={len(cl_gt)}")
        print(f"    rms:     py={result['rms_py']:.3f}  oracle={result['rms_oracle']:.3f}  gt={result['rms_gt']:.3f}")

    # ── Step 4: aggregate + pass/fail ───────────────────────────────────────
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        print("\n[FAIL] No successful samples — cannot evaluate port fidelity.")
        return 1

    def stats(vals: list) -> dict:
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "std": float(arr.std()),
        }

    mel_l1_po = [r["mel_l1_py_vs_oracle"] for r in ok]
    corr_po = [r["correlation_py_vs_oracle"] for r in ok]
    cl_diff = [abs(r["closure_count_py"] - r["closure_count_oracle"]) for r in ok]

    s_mel = stats(mel_l1_po)
    s_corr = stats(corr_po)
    s_cl_diff = stats(cl_diff)

    pass_mel = s_mel["mean"] < 1.5
    pass_corr = s_corr["mean"] > 0.7
    pass_closure = s_cl_diff["mean"] <= 1.0
    overall_pass = pass_mel and pass_corr and pass_closure

    print("\n" + "=" * 70)
    print(f"AGGREGATE — n={len(ok)} samples")
    print("=" * 70)
    print(f"  mel_L1 py↔oracle:    mean={s_mel['mean']:.3f}  median={s_mel['median']:.3f}"
          f"  range=[{s_mel['min']:.3f}, {s_mel['max']:.3f}]")
    print(f"  corr   py↔oracle:    mean={s_corr['mean']:.3f}  median={s_corr['median']:.3f}"
          f"  range=[{s_corr['min']:.3f}, {s_corr['max']:.3f}]")
    print(f"  closure |Δ|:          mean={s_cl_diff['mean']:.2f}  max={s_cl_diff['max']:.0f}")

    print(f"\nPASS CRITERIA:")
    print(f"  mel_L1 mean < 1.5:    {'PASS' if pass_mel else 'FAIL'}  ({s_mel['mean']:.3f})")
    print(f"  correlation > 0.7:    {'PASS' if pass_corr else 'FAIL'}  ({s_corr['mean']:.3f})")
    print(f"  closure |Δ| ≤ 1:      {'PASS' if pass_closure else 'FAIL'}  ({s_cl_diff['mean']:.2f})")
    print(f"\nOVERALL: {'PASS — PyTorch port faithful to JS PT' if overall_pass else 'WARNING — see header docstring for likely causes'}")

    # ── Step 5: write JSON ──────────────────────────────────────────────────
    out_json = os.path.join(args.out_dir, "comparison.json")
    summary = {
        "n_samples_ok": len(ok),
        "n_samples_total": len(results),
        "mel_l1_py_vs_oracle":      s_mel,
        "correlation_py_vs_oracle": s_corr,
        "closure_count_abs_diff":   s_cl_diff,
        "pass_mel": pass_mel,
        "pass_correlation": pass_corr,
        "pass_closure": pass_closure,
        "overall_pass": overall_pass,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"per_sample": results, "summary": summary}, f, indent=2)
    print(f"\nFull results: {out_json}")
    if args.resynth:
        print(f"PyTorch wavs: {os.path.join(args.out_dir, 'audio_py')}/")
    print("\nNext steps:")
    if overall_pass:
        print("  ✓ Port verified. Launch Run 13a: `bash run_v13a_audio_cycle.sh`")
    else:
        print("  ⚠ Port has drift. You can still launch Run 13a, but be aware that")
        print("    the loss optimizes against a slightly different audio than JS PT.")
        print("    Listen to audio_py/*.wav vs audit_pt/*_oracle.wav to characterize.")
        print("    If audio_py sounds clearly worse, debug the components flagged in")
        print("    the test header docstring before training.")

    return 0 if overall_pass else 2


if __name__ == "__main__":
    sys.exit(main())
