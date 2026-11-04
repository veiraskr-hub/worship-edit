"""결과 FCPXML(컷·타이틀이 들어간 것) → 프리미어용 XML(Final Cut Pro 7 XML, xmeml) + 가사 SRT.

프리미어: 파일 → 가져오기 → 이 XML. 컷이 다 들어간 시퀀스가 생긴다.
- 비디오: 컷마다 그 앵글의 원본 파일 구간 (4K 원본은 1080p 화면에 맞게 크기 조절, FCP 확대값 반영)
- 오디오: 컷의 오디오 앵글(실황 녹화본) 구간을 이어 붙인 스테레오
- 자막: 타이틀을 SRT로 (프리미어에서 가져와 캡션 트랙에 올린다)
멀티캠 클립이 아니라 잘린 컷으로 들어간다.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fractions import Fraction
from xml.sax.saxutils import escape

FPS = Fraction(30000, 1001)


def _t(v: str | None) -> Fraction:
    if not v:
        return Fraction(0)
    v = v.rstrip("s")
    if "/" in v:
        a, b = v.split("/")
        return Fraction(int(a), int(b))
    return Fraction(v)


@dataclass
class Asset:
    id: str
    name: str
    src: str
    start: Fraction
    duration: Fraction
    width: int
    height: int
    has_audio: bool


@dataclass
class Piece:            # 앵글 안의 asset-clip 하나 (멀티캠 시간)
    asset: str
    offset: Fraction
    start: Fraction     # 원본 파일 시간(asset.start 기준 아님, 절대값)
    duration: Fraction
    scale: float


@dataclass
class Cut:
    t0: Fraction        # 타임라인(프로젝트) 시간
    t1: Fraction
    video: str          # angleID
    audio: str
    mstart: Fraction    # 이 컷 시작의 멀티캠 시간


def _abs_children(el, base, parent_start, out_cuts, out_titles):
    """FCPXML 시간 규칙: 자식 offset은 부모의 로컬 시간 → 타임라인 = base + (offset - parent_start)."""
    for ch in el:
        tag = ch.tag
        if tag not in ("gap", "spine", "mc-clip", "title", "asset-clip", "video", "clip", "ref-clip", "sync-clip"):
            continue
        off = _t(ch.get("offset"))
        at = base + (off - parent_start)
        if tag == "mc-clip":
            dur = _t(ch.get("duration"))
            v = a = None
            for s in ch.findall("mc-source"):
                en = s.get("srcEnable", "all")
                if en in ("video", "all"):
                    v = s.get("angleID")
                if en in ("audio", "all"):
                    a = s.get("angleID")
            out_cuts.append(Cut(at, at + dur, v, a, _t(ch.get("start"))))
        elif tag == "title":
            dur = _t(ch.get("duration"))
            txt = "".join(x.text or "" for x in ch.iter("text-style")).strip()
            if txt:
                out_titles.append((at, at + dur, txt))
        if tag == "spine":
            _abs_children(ch, at, Fraction(0), out_cuts, out_titles)
        elif tag in ("gap", "clip", "sync-clip"):
            _abs_children(ch, at, _t(ch.get("start")), out_cuts, out_titles)


def read_result(path: str):
    p = os.path.join(path, "Info.fcpxml") if os.path.isdir(path) else path
    root = ET.parse(p).getroot()
    res = root.find("resources")
    fmts = {f.get("id"): f for f in res.findall("format")}
    assets: dict[str, Asset] = {}
    for a in res.findall("asset"):
        rep = a.find("media-rep")
        f = fmts.get(a.get("format"))
        assets[a.get("id")] = Asset(a.get("id"), a.get("name"), rep.get("src") if rep is not None else "",
                                    _t(a.get("start")), _t(a.get("duration")),
                                    int(f.get("width", 1920)) if f is not None else 1920,
                                    int(f.get("height", 1080)) if f is not None else 1080,
                                    a.get("hasAudio") == "1")
    angles: dict[str, tuple[str, list[Piece]]] = {}
    for mc in res.iter("media"):
        for ang in mc.iter("mc-angle"):
            pieces = []
            for ac in ang.findall("asset-clip"):
                asset = assets[ac.get("ref")]
                tr = ac.find("adjust-transform")
                sc = float(tr.get("scale", "1 1").split()[0]) if tr is not None and tr.get("scale") else 1.0
                st = _t(ac.get("start")) if ac.get("start") else asset.start
                pieces.append(Piece(asset.id, _t(ac.get("offset")), st, _t(ac.get("duration")), sc))
            angles[ang.get("angleID")] = (ang.get("name"), pieces)
    seq = root.find(".//sequence")
    cuts, titles = [], []
    _abs_children(seq.find("spine"), Fraction(0), Fraction(0), cuts, titles)
    cuts.sort(key=lambda c: c.t0)
    t0 = cuts[0].t0 if cuts else Fraction(0)       # 프로젝트 시작
    for c in cuts:
        c.t0 -= t0; c.t1 -= t0
    titles = sorted((a - t0, b - t0, s) for a, b, s in titles)
    name = root.find(".//project").get("name", "찬양편집")
    return name, assets, angles, cuts, titles


def _source(angles, angle_id, m: Fraction):
    """멀티캠 시간 m에 그 앵글이 보여주는 (piece, 원본 파일 시간)."""
    _, pieces = angles[angle_id]
    for pc in pieces:
        if pc.offset - Fraction(1, 100) <= m < pc.offset + pc.duration:
            return pc, pc.start + (m - pc.offset)
    return None, None


def _fr(x: Fraction) -> int:
    return int(round(x * FPS))


def _pathurl(src: str) -> str:
    return src.replace("file:///", "file://localhost/", 1)


def write_xmeml(path: str, name: str, assets, angles, cuts, seq_w=1920, seq_h=1080) -> dict:
    total = _fr(cuts[-1].t1)
    files_done: set[str] = set()

    def file_el(a: Asset, with_audio: bool) -> str:
        if a.id in files_done:
            return f'<file id="f{a.id}"/>'
        files_done.add(a.id)
        aud = ('<audio><samplecharacteristics><depth>16</depth><samplerate>48000</samplerate></samplecharacteristics>'
               '<channelcount>2</channelcount></audio>') if a.has_audio else ""
        return (f'<file id="f{a.id}"><name>{escape(os.path.basename(a.src) or a.name)}</name>'
                f'<pathurl>{escape(_pathurl(a.src))}</pathurl>'
                f'<rate><timebase>30</timebase><ntsc>TRUE</ntsc></rate><duration>{_fr(a.duration)}</duration>'
                f'<timecode><rate><timebase>30</timebase><ntsc>TRUE</ntsc></rate><frame>{_fr(a.start)}</frame>'
                f'<displayformat>NDF</displayformat></timecode>'
                f'<media><video><samplecharacteristics><width>{a.width}</width><height>{a.height}</height>'
                f'</samplecharacteristics></video>{aud}</media></file>')

    rate = "<rate><timebase>30</timebase><ntsc>TRUE</ntsc></rate>"
    vitems, aitems = [], []
    problems = []
    # 오디오: 같은 원본이 이어지는 구간은 하나로 합친다
    aud_runs: list[list] = []
    for i, c in enumerate(cuts):
        a_fr, b_fr = _fr(c.t0), _fr(c.t1)
        pc, src_t = _source(angles, c.video, c.mstart)
        if pc is None:
            problems.append(f"{c.t0:.2f}s 영상 앵글 원본 없음"); continue
        a = assets[pc.asset]
        sin = _fr(src_t - a.start)
        fit = min(seq_w / a.width, seq_h / a.height) * 100 * pc.scale
        vitems.append(
            f'<clipitem id="v{i}"><name>{escape(a.name)}</name><enabled>TRUE</enabled>{rate}'
            f'<start>{a_fr}</start><end>{b_fr}</end><in>{sin}</in><out>{sin + (b_fr - a_fr)}</out>'
            f'{file_el(a, False)}'
            f'<filter><effect><name>Basic Motion</name><effectid>basic</effectid><effectcategory>motion</effectcategory>'
            f'<effecttype>motion</effecttype><mediatype>video</mediatype>'
            f'<parameter><parameterid>scale</parameterid><name>Scale</name><value>{fit:.2f}</value></parameter>'
            f'</effect></filter></clipitem>')
        if c.audio:
            apc, asrc = _source(angles, c.audio, c.mstart)
            if apc is None:
                problems.append(f"{c.t0:.2f}s 오디오 앵글 원본 없음")
            else:
                ain = _fr(asrc - assets[apc.asset].start)
                if aud_runs and aud_runs[-1][0] == apc.asset and aud_runs[-1][2] == a_fr \
                        and abs(aud_runs[-1][4] - ain) <= 1:
                    aud_runs[-1][2] = b_fr; aud_runs[-1][4] = ain + (b_fr - a_fr)
                else:
                    aud_runs.append([apc.asset, a_fr, b_fr, ain, ain + (b_fr - a_fr)])
    for ch in (1, 2):
        items = []
        for j, (aid, s, e, i_, o_) in enumerate(aud_runs):
            a = assets[aid]
            items.append(f'<clipitem id="a{ch}_{j}"><name>{escape(a.name)}</name><enabled>TRUE</enabled>{rate}'
                         f'<start>{s}</start><end>{e}</end><in>{i_}</in><out>{o_}</out>{file_el(a, True)}'
                         f'<sourcetrack><mediatype>audio</mediatype><trackindex>{ch}</trackindex></sourcetrack></clipitem>')
        aitems.append("<track>" + "".join(items) + "</track>")
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="4">'
           f'<sequence id="seq1"><name>{escape(name)}</name><duration>{total}</duration>{rate}'
           '<timecode><rate><timebase>30</timebase><ntsc>TRUE</ntsc></rate><string>00:00:00:00</string>'
           '<frame>0</frame><displayformat>NDF</displayformat></timecode>'
           '<media><video><format><samplecharacteristics>'
           f'{rate}<width>{seq_w}</width><height>{seq_h}</height><pixelaspectratio>square</pixelaspectratio>'
           '</samplecharacteristics></format>'
           f'<track>{"".join(vitems)}</track></video>'
           '<audio><numOutputChannels>2</numOutputChannels>'
           f'{"".join(aitems)}</audio></media></sequence></xmeml>\n')
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    return {"video_clips": len(vitems), "audio_runs": len(aud_runs), "frames": total, "problems": problems}


def write_srt(path: str, titles):
    def f(x):
        ms = int(round(float(x) * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    with open(path, "w", encoding="utf-8") as fp:
        for i, (a, b, s) in enumerate(titles, 1):
            fp.write(f"{i}\n{f(a)} --> {f(b)}\n{s}\n\n")


def export(result_fcpxml: str, out_xml: str, out_srt: str | None = None) -> dict:
    name, assets, angles, cuts, titles = read_result(result_fcpxml)
    if not cuts:
        raise ValueError("컷(mc-clip)이 없는 XML이에요. WorshipEdit가 만든 결과 XML을 넣어 주세요.")
    rep = write_xmeml(out_xml, name, assets, angles, cuts)
    if out_srt is not None:
        write_srt(out_srt, titles)
    rep["titles"] = len(titles)
    return rep
