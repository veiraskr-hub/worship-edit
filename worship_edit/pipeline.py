"""전체 자동 흐름 (명령어·앱 화면이 함께 쓴다).

확인 → 목소리 분리 → 받아쓰기 → 가사 자동 정렬 → 곡 구조(가사로 보정) → 규칙 컷 → 타이틀 → XML
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import fcpxml
from .align import VocalTrack, display_spans
from .analyze import Audio, Structure, detect_structure
from .autoalign import Seg, auto_lines, read_srt
from .cuts import CutSettings, make_cuts
from .library import Song


@dataclass
class Job:
    project: str
    audio: str
    songs: list[Song]
    out: str
    name: str = ""
    vocals: str = ""              # 비우면 분리해서 만든다
    srt: str = ""                 # 비우면 Whisper로 받아쓴다
    roles: dict[str, str] = field(default_factory=dict)   # 비우면 이름 규칙으로 자동
    structure: Structure | None = None                    # 비우면 자동 감지
    work_dir: str = ""


def _log(progress, stage, frac=None, msg=""):
    if progress:
        progress(stage, frac, msg)


def refine_structure(st: Structure, lines, length: float) -> Structure:
    """가사가 나온 시간으로 곡 구간을 보정한다: 첫 가사보다 늦게 시작하거나 마지막 가사보다 일찍 끝나면 넓힌다."""
    if not lines:
        return st
    for s in st.songs:
        inside = [l.time for l in lines if s.start - 30 <= l.time <= s.end + 10]
        if not inside:
            continue
        if min(inside) < s.start:
            s.start = max(0.0, min(inside) - 4.0); s.open, s.countin = "keys", None
        if max(inside) > s.end - 2:
            s.end = min(length, max(inside) + 6.0)
    ments, t = [], 0.0
    for s in st.songs:
        a = s.countin if s.countin is not None else s.start
        if a - t > 1:
            ments.append((round(t, 2), round(a, 2)))
        t = s.end
    if length - t > 1:
        ments.append((round(t, 2), round(length, 2)))
    st.ments = ments
    return st


def run(job: Job, progress=None) -> dict:
    work = job.work_dir or os.path.splitext(job.out)[0] + "_작업"
    os.makedirs(work, exist_ok=True)
    rep: dict = {}

    _log(progress, "확인", 0)
    prj = fcpxml.load(job.project)
    from .cli import audio_length
    al = audio_length(job.audio)
    if abs(al - prj.length) > 1.0:
        raise ValueError(f"오디오 길이({al:.1f}초)가 프로젝트({prj.length:.1f}초)와 달라요. 타임라인에서 ⌥X로 선택 구간을 해제하고 다시 내보내 주세요.")
    roles = job.roles or fcpxml.classify_roles(prj.angles)
    unknown = [n for n, r in roles.items() if r == "unknown"]
    if unknown:
        raise ValueError(f"역할을 모르는 앵글이 있어요: {', '.join(unknown)}")
    cam_roles = {n: r for n, r in roles.items() if r in fcpxml.ROLES}

    vocals = job.vocals
    if not vocals:
        from .separate import separate_vocals
        vocals = os.path.join(work, "목소리.wav")
        separate_vocals(job.audio, vocals, progress=lambda f, t: _log(progress, "목소리 분리", f))
    vt = VocalTrack(vocals)

    if job.srt:
        segs = read_srt(job.srt)
    else:
        from .transcribe import transcribe, write_srt
        prompt = " ".join(l for s in job.songs for _, _, l in s.lines)
        segs = transcribe(vocals, prompt=prompt, progress=lambda f: _log(progress, "받아쓰기", f))
        write_srt(segs, os.path.join(work, "받아쓰기.srt"))

    _log(progress, "가사 정렬", 0)
    lines, arep = auto_lines(segs, job.songs, vt)
    rep["lyrics"] = arep

    _log(progress, "곡 구조", 0)
    au = Audio(job.audio)
    st = job.structure or refine_structure(detect_structure(au, vt), lines, prj.length)
    st.chorus = [l.time for l in lines if l.chorus]
    rep["structure"] = st.to_dict()

    _log(progress, "컷", 0)
    cuts, crep = make_cuts(prj, cam_roles, au, st, CutSettings())
    rep["cuts"] = crep

    text = fcpxml.write_cuts(prj, [(c.start, c.end, c.angle) for c in cuts])
    text = fcpxml.write_titles(prj, text, display_spans(lines))
    text = fcpxml.rename(text, job.name or os.path.splitext(os.path.basename(job.out))[0])
    fcpxml.save(text, job.out)
    rep["lines"] = [(round(l.time, 2), l.text, l.chorus, l.how) for l in lines]
    rep["out"] = job.out
    _log(progress, "완료", 1)
    return rep
