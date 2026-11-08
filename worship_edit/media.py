"""ffmpeg 찾기, 길이 재기, XML이 가리키는 원본 녹화본에서 프로젝트 구간 오디오 꺼내기."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import urllib.parse
from fractions import Fraction

from .fcpxml import Project, _t


def ffmpeg_exe() -> str:
    """앱에 들어 있는 ffmpeg(imageio-ffmpeg)를 먼저, 없으면 시스템 ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        p = shutil.which("ffmpeg")
        if p:
            return p
        for cand in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
            if os.path.exists(cand):
                return cand
    raise FileNotFoundError("ffmpeg를 찾지 못했어요")


def duration(path: str) -> float:
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr)
    if not m:
        raise ValueError(f"길이를 읽지 못했어요: {path}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def audio_source(prj: Project) -> tuple[str, float, float, float]:
    """(원본 파일 경로, 원본 안 시작초, 앞에 채울 무음초, 길이초).

    오디오로 쓰는 앵글(srcEnable all/audio)의 asset-clip과 원본 asset을 따라가서
    프로젝트 첫 프레임이 원본 파일의 몇 초인지 계산한다."""
    s = prj.text
    ang = re.search(r'<mc-angle name="[^"]*" angleID="' + re.escape(prj.audio_angle_id) + r'">(.*?)</mc-angle>', s, re.S)
    if not ang:
        raise ValueError("오디오 앵글을 찾지 못했어요")
    clip = re.search(r'<asset-clip ref="([^"]*)"([^>]*)>', ang.group(1)) or re.search(r'<asset-clip ref="([^"]*)"([^>]*)/>', ang.group(1))
    ref, attrs = clip.group(1), clip.group(2)
    get = lambda k, a=attrs: (re.search(rf'\b{k}="([^"]*)"', a) or [None, ""])[1]
    c_off, c_start, c_dur = _t(get("offset")), _t(get("start")), _t(get("duration"))
    asset = re.search(r'<asset id="' + re.escape(ref) + r'"([^>]*)>(.*?)</asset>', s, re.S)
    a_start = _t((re.search(r'\bstart="([^"]*)"', asset.group(1)) or [None, ""])[1])
    src = re.search(r'<media-rep kind="original-media"[^>]*src="([^"]*)"', asset.group(2)).group(1)
    path = urllib.parse.unquote(urllib.parse.urlparse(src).path)
    path = __import__("unicodedata").normalize("NFC", path) if not os.path.exists(path) else path
    mc_t = prj.clip_start                      # 프로젝트 첫 프레임의 멀티캠 시간
    lead_silence = max(Fraction(0), c_off - mc_t)
    src_t = (c_start - a_start) + max(Fraction(0), mc_t - c_off)
    return path, float(src_t), float(lead_silence), prj.length


def extract_project_audio(prj: Project, out_wav: str) -> str:
    path, src_t, pad, dur = audio_source(prj)
    if not os.path.exists(path):
        raise FileNotFoundError(f"원본 녹화본을 찾지 못했어요: {path}\n(외장 드라이브가 연결돼 있는지 확인하거나, 오디오 파일을 직접 넣어 주세요)")
    af = f"adelay={int(pad * 1000)}:all=1," if pad > 0 else ""
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{src_t:.4f}", "-i", path,
                    "-t", f"{dur - pad:.4f}", "-vn", "-ac", "2", "-ar", "44100",
                    "-af", f"{af}apad=whole_dur={dur:.4f}", out_wav], check=True)
    return out_wav


def download(url: str, dest: str, tries: int = 4) -> str:
    """모델 내려받기 (서버가 잠깐 오류를 내면 다시 시도)."""
    import time
    err = None
    for k in range(tries):
        try:
            return _download_once(url, dest)
        except Exception as e:
            err = e
            time.sleep(5 * (k + 1))
    raise err


def _download_once(url: str, dest: str) -> str:
    """맥 기본 curl(시스템 인증서)을 먼저 쓰고, 안 되면 certifi 인증서로 urllib."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    curl = shutil.which("curl") or "/usr/bin/curl"
    if os.path.exists(curl):
        r = subprocess.run([curl, "-fL", "--retry", "3", "-o", tmp, url], capture_output=True)
        if r.returncode == 0 and os.path.getsize(tmp) > 0:
            os.replace(tmp, dest)
            return dest
    import ssl
    import urllib.request
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, context=ctx) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(tmp, dest)
    return dest
