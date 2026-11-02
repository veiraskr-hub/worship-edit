"""목소리 분리 (UVR MDX-Net Voc_FT, onnxruntime + numpy).

audio-separator와 같은 결과(상관 1.0)를 내되 torch 없이 가볍게 돈다.
모델: UVR-MDX-NET-Voc_FT.onnx (github.com/TRvlvr/model_repo releases, md5 d21dc03e4b9ef397b47231f483af6db8)
"""
from __future__ import annotations

import os
import subprocess
import time

import numpy as np

MODEL_URL = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx"
N_FFT, HOP, DIM_F, DIM_T, COMP = 7680, 1024, 3072, 256, 1.021
NB = N_FFT // 2 + 1
TRIM = N_FFT // 2
CH = HOP * (DIM_T - 1)
GEN = CH - 2 * TRIM
SR = 44100
_WIN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)


def model_path(cache_dir: str | None = None) -> str:
    cache_dir = cache_dir or os.path.expanduser("~/Library/Application Support/WorshipEdit/models")
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, "UVR-MDX-NET-Voc_FT.onnx")
    if not os.path.exists(p):
        for cand in (os.path.expanduser("~/Downloads/UVR-MDX-NET-Voc_FT.onnx"),):
            if os.path.exists(cand):
                return cand
        from .media import download
        download(MODEL_URL, p)
    return p


def _stft(x):
    xp = np.pad(x, ((0, 0), (N_FFT // 2, N_FFT // 2)), mode="reflect")
    idx = np.arange(DIM_T)[:, None] * HOP + np.arange(N_FFT)[None, :]
    return np.fft.rfft(xp[:, idx] * _WIN, axis=-1).transpose(0, 2, 1)


def _istft(X):
    fr = np.fft.irfft(X.transpose(0, 2, 1), n=N_FFT, axis=-1) * _WIN
    L = N_FFT + HOP * (DIM_T - 1)
    out = np.zeros((X.shape[0], L), np.float32)
    ws = np.zeros(L, np.float32)
    for k in range(DIM_T):
        out[:, k * HOP:k * HOP + N_FFT] += fr[:, k]
        ws[k * HOP:k * HOP + N_FFT] += _WIN ** 2
    out /= np.maximum(ws, 1e-8)
    return out[:, N_FFT // 2:N_FFT // 2 + CH]


def read_audio(path: str, start: float = 0, dur: float = 0) -> np.ndarray:
    from .media import ffmpeg_exe
    args = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error"]
    if start:
        args += ["-ss", str(start)]
    if dur:
        args += ["-t", str(dur)]
    args += ["-i", path, "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(args, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy()


def _session(model: str):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count() or 4
    providers = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    try:
        return ort.InferenceSession(model, so, providers=providers)
    except Exception:
        return ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])


def separate_vocals(inp: str, out_wav: str, model: str | None = None, progress=None) -> str:
    """전체 파일을 모델 단위(약 5.7초)로 순서대로 처리해 메모리를 적게 쓴다."""
    import soundfile as sf
    model = model or model_path()
    sess = _session(model)
    iname = sess.get_inputs()[0].name
    mix = read_audio(inp)
    n = mix.shape[1]
    pad = GEN - n % GEN
    mp = np.concatenate([np.zeros((2, TRIM), np.float32), mix, np.zeros((2, pad), np.float32), np.zeros((2, TRIM), np.float32)], 1)
    steps = list(range(0, n + pad, GEN))
    t0 = time.time()
    with sf.SoundFile(out_wav, "w", samplerate=SR, channels=1) as f:
        written = 0
        for k, i in enumerate(steps):
            X = _stft(mp[:, i:i + CH])[:, :DIM_F]
            spec = np.stack([X.real, X.imag], 1).reshape(1, 4, DIM_F, DIM_T).astype(np.float32)
            Y = sess.run(None, {iname: spec})[0].reshape(2, 2, DIM_F, DIM_T)
            Yc = np.zeros((2, NB, DIM_T), np.complex64)
            Yc[:, :DIM_F] = Y[:, 0] + 1j * Y[:, 1]
            chunk = _istft(Yc)[:, TRIM:-TRIM] * COMP
            take = min(chunk.shape[1], n - written)
            if take > 0:
                f.write(chunk[:, :take].mean(0))
                written += take
            if progress:
                progress((k + 1) / len(steps), time.time() - t0)
    return out_wav
