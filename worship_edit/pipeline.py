"""전체 자동 흐름 (명령어·앱 화면이 함께 쓴다).

확인 → 오디오(원본에서 꺼내기) → 목소리 분리 → 받아쓰기 → 가사 자동 정렬 → 곡 구조 → 규칙 컷 → 타이틀 → XML

느린 단계(오디오·분리·받아쓰기)는 작업 폴더에 남겨 두고, 검토 화면에서 가사 위치나 곡 구간을
고친 뒤에는 `rebuild()`로 컷·타이틀만 다시 만든다.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from . import fcpxml
from .align import Line, VocalTrack, display_spans
from .analyze import Audio, Structure, detect_structure
from .autoalign import auto_lines, read_srt
from .cuts import CutSettings, make_cuts
from .library import Song


@dataclass
class Job:
    project: str
    songs: list[Song]
    out: str
    audio: str = ""               # 비우면 XML이 가리키는 원본 녹화본에서 꺼낸다
    name: str = ""
    vocals: str = ""              # 비우면 분리해서 만든다
    srt: str = ""                 # 비우면 Whisper로 받아쓴다
    roles: dict[str, str] = field(default_factory=dict)   # 비우면 이름 규칙으로 자동
    structure: Structure | None = None                    # 비우면 자동 감지
    work_dir: str = ""


@dataclass
class Result:
    job: Job
    prj: fcpxml.Project
    roles: dict[str, str]
    audio: str
    vocals: str
    lines: list[Line]
    structure: Structure
    cuts: list
    report: dict


def _log(progress, stage, frac=None, msg=""):
    if progress:
        progress(stage, frac, msg)


def work_dir_for(job: Job) -> str:
    return job.work_dir or os.path.splitext(job.out)[0] + "_작업"


def _lyric_spans(lines, songs):
    """가사 줄이 어느 곡인지(가사 보관함 글자로) 보고 곡별 (처음, 끝) 시간."""
    from .autoalign import norm
    owner = {}
    for i, sg in enumerate(songs):
        for _, _, t in sg.lines:
            owner.setdefault(norm(t), i)
    spans = {}
    for l in lines:
        i = owner.get(norm(l.text))
        if i is not None:
            a, b = spans.get(i, (l.time, l.time))
            spans[i] = (min(a, l.time), max(b, l.time))
    return [spans[i] for i in sorted(spans, key=lambda k: spans[k][0])]


def refine_structure(st: Structure, lines, length: float, songs=None) -> Structure:
    """가사가 나온 시간으로 곡 구간을 보정한다.
    곡 가사를 알면 곡 수와 구간을 가사 기준으로 다시 잡는다 (소리만으로 감지하면 한 곡이 둘로 나뉘거나 끝을 놓칠 수 있음)."""
    spans = _lyric_spans(lines, songs) if (lines and songs) else []
    if len(spans) >= 1:
        from .analyze import Song as Sec
        new = []
        for k, (a, b) in enumerate(spans):
            lo = new[-1].end if new else 0.0
            hi = spans[k + 1][0] - 4.0 if k + 1 < len(spans) else length
            over = [x for x in st.songs if x.end > a - 30 and x.start < b]
            start = min([a - 4.0] + [x.start for x in over if x.start >= lo])
            end = max([b + 6.0] + [x.end for x in over if x.end <= hi])
            start, end = max(lo, start), min(hi, end)
            first = min(over, key=lambda x: x.start) if over else None
            keep = first is not None and abs(first.start - start) < 1.0
            new.append(Sec(round(start, 2), round(end, 2), first.open if keep else "keys",
                           first.countin if keep else None))
        st.songs = new
        return recompute_ments(st, length)
    if lines:
        for s in st.songs:
            inside = [l.time for l in lines if s.start - 30 <= l.time <= s.end + 10]
            if not inside:
                continue
            if min(inside) < s.start:
                s.start = max(0.0, min(inside) - 4.0); s.open, s.countin = "keys", None
            if max(inside) > s.end - 2:
                s.end = min(length, max(inside) + 6.0)
    return recompute_ments(st, length)


def recompute_ments(st: Structure, length: float) -> Structure:
    st.songs.sort(key=lambda s: s.start)
    ments, t = [], 0.0
    for s in st.songs:
        a = s.countin if s.countin is not None else s.start
        if a - t > 1:
            ments.append((round(t, 2), round(a, 2)))
        t = max(t, s.end)
    if length - t > 1:
        ments.append((round(t, 2), round(length, 2)))
    st.ments = ments
    return st


def prepare(job: Job, progress=None) -> tuple[fcpxml.Project, dict[str, str], dict]:
    """XML 읽기와 점검만 (화면에 역할·날짜를 먼저 보여줄 때)."""
    prj = fcpxml.load(job.project)
    roles = job.roles or fcpxml.classify_roles(prj.angles)
    info = {"length": prj.length, "unknown": [n for n, r in roles.items() if r == "unknown"]}
    try:
        from .media import audio_source
        path, src_t, pad, dur = audio_source(prj)
        info["audio_source"] = path
        info["audio_source_found"] = os.path.exists(path)
    except Exception as e:
        info["audio_source"] = ""
        info["audio_source_found"] = False
        info["audio_error"] = str(e)
    return prj, roles, info


def run(job: Job, progress=None) -> Result:
    work = work_dir_for(job)
    os.makedirs(work, exist_ok=True)
    _log(progress, "확인", 0)
    prj, roles, info = prepare(job)
    if info["unknown"]:
        raise ValueError(f"역할을 모르는 앵글이 있어요: {', '.join(info['unknown'])}")

    from .media import duration, extract_project_audio
    audio = job.audio
    if not audio:
        audio = os.path.join(work, "오디오.wav")
        if not os.path.exists(audio):
            _log(progress, "오디오 꺼내기", None)
            extract_project_audio(prj, audio)
    al = duration(audio)
    if abs(al - prj.length) > 1.0:
        raise ValueError(f"오디오 길이({al:.1f}초)가 프로젝트({prj.length:.1f}초)와 달라요. 타임라인에서 ⌥X로 선택 구간을 해제하고 다시 내보내 주세요.")

    vocals = job.vocals or os.path.join(work, "목소리.wav")
    if not os.path.exists(vocals):
        from .separate import separate_vocals
        separate_vocals(audio, vocals, progress=lambda f, t: _log(progress, "목소리 분리", f))
    vt = VocalTrack(vocals)

    srt = job.srt or os.path.join(work, "받아쓰기.srt")
    if os.path.exists(srt):
        segs = read_srt(srt)
    else:
        from .transcribe import transcribe, write_srt
        _log(progress, "받아쓰기", None)
        prompt = " ".join(l for s in job.songs for _, _, l in s.lines)
        segs = transcribe(vocals, prompt=prompt, progress=lambda f: _log(progress, "받아쓰기", f))
        write_srt(segs, srt)

    _log(progress, "가사 정렬", None)
    lines, arep = auto_lines(segs, job.songs, vt)

    _log(progress, "곡 구조", None)
    au = Audio(audio)
    st = job.structure or refine_structure(detect_structure(au, vt), lines, prj.length, job.songs)

    res = Result(job, prj, roles, audio, vocals, lines, st, [], {"lyrics": arep})
    rebuild(res, progress, audio_obj=au)
    return res


def rebuild(res: Result, progress=None, audio_obj: Audio | None = None) -> Result:
    """가사 위치·곡 구간을 고친 뒤 컷과 타이틀만 다시 만든다."""
    job, prj = res.job, res.prj
    _log(progress, "컷", None)
    au = audio_obj or Audio(res.audio)
    res.lines.sort(key=lambda l: l.time)
    res.structure.chorus = [l.time for l in res.lines if l.chorus]
    recompute_ments(res.structure, prj.length)
    cam_roles = {n: r for n, r in res.roles.items() if r in fcpxml.ROLES}
    cuts, crep = make_cuts(prj, cam_roles, au, res.structure, CutSettings())
    res.cuts = cuts
    text = fcpxml.write_cuts(prj, [(c.start, c.end, c.angle) for c in cuts])
    text = fcpxml.write_titles(prj, text, display_spans(res.lines))
    text = fcpxml.rename(text, job.name or os.path.splitext(os.path.basename(job.out))[0])
    fcpxml.save(text, job.out)
    res.report.update({"structure": res.structure.to_dict(), "cuts": crep, "out": job.out})
    try:                                   # 프리미어용 XML + 가사 SRT도 같이
        from .premiere import export
        base = os.path.splitext(job.out)[0]
        res.report["premiere"] = export(job.out, base + "_프리미어.xml", base + "_가사.srt")
        res.report["premiere"]["out"] = base + "_프리미어.xml"
    except Exception as e:
        res.report["premiere"] = {"error": str(e)}
    with open(os.path.join(work_dir_for(job), "결과.json"), "w", encoding="utf-8") as f:
        json.dump({"structure": res.structure.to_dict(),
                   "lines": [(l.time, l.text, l.chorus, l.how) for l in res.lines],
                   "report": {k: v for k, v in res.report.items() if k != "structure"}}, f, ensure_ascii=False, indent=1)
    _log(progress, "완료", 1)
    return res
