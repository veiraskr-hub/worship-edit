"""앱 안 받아쓰기 (whisper.cpp, Mac에서는 Metal 가속).

목소리 트랙을 받아써서 구간(시작, 끝, 글자) 목록을 돌려준다. 결과는 autoalign.Seg 형식.
모델: ggml-large-v3-turbo-q5_0.bin (huggingface.co/ggerganov/whisper.cpp)
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import urllib.request

from .autoalign import Seg

MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin"
MODEL_NAME = "ggml-large-v3-turbo-q5_0.bin"


def model_path(cache_dir: str | None = None) -> str:
    cache_dir = cache_dir or os.path.expanduser("~/Library/Application Support/WorshipEdit/models")
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, MODEL_NAME)
    if os.path.exists(p):
        return p
    # 사용자가 이미 받아 둔 파일이 있으면 그걸 쓴다
    for cand in (os.path.expanduser(f"~/Downloads/{MODEL_NAME}"),):
        if os.path.exists(cand):
            return cand
    tmp = p + ".part"
    urllib.request.urlretrieve(MODEL_URL, tmp)
    os.replace(tmp, p)
    return p


def _wav16k(path: str) -> str:
    fd, out = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path, "-ac", "1", "-ar", "16000", out], check=True)
    return out


def transcribe(vocals_path: str, prompt: str = "", model: str | None = None, progress=None) -> list[Seg]:
    """prompt: 그 주 가사 일부를 주면 노래 인식이 좋아진다."""
    from pywhispercpp.model import Model
    wav = _wav16k(vocals_path)
    try:
        m = Model(model or model_path(), n_threads=os.cpu_count() or 4, print_progress=False, print_realtime=False)
        kw = dict(language="ko", no_context=True, temperature=0.0)
        if prompt:
            kw["initial_prompt"] = prompt[:600]
        if progress:
            kw["progress_callback"] = lambda p: progress(p / 100)
        segs = m.transcribe(wav, **kw)
        return [Seg(s.t0 / 100, s.t1 / 100, s.text.strip()) for s in segs if s.text.strip()]
    finally:
        os.remove(wav)


def write_srt(segs: list[Seg], path: str):
    def f(x):
        ms = int(round(x * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    with open(path, "w", encoding="utf-8") as fp:
        for i, s in enumerate(segs, 1):
            fp.write(f"{i}\n{f(s.start)} --> {f(s.end)}\n{s.text}\n\n")
