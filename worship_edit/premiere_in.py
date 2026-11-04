"""프리미어 멀티캠 시퀀스 XML(Final Cut Pro 7 XML, xmeml) → 앱이 쓰는 FCPXML 구조로 바꾸기.

프리미어: 멀티캠 클립으로 만든 시퀀스 선택 → 파일 → 내보내기 → Final Cut Pro XML.
- 안쪽 시퀀스(멀티캠 소스)의 비디오 트랙 = 앵글. clipitem start = 멀티캠 시간, in = 원본 시간 (29.97 프레임)
- 바깥 시퀀스의 멀티캠 clipitem in/out = 쓸 구간 (시퀀스에 시작/종료 표시가 있으면 그 구간)
- 오디오 원본 = 카메라 이름 규칙에 안 맞는 앵글 중 가장 긴 것 (실황 녹화본)
"""
from __future__ import annotations

import re
import uuid
import xml.etree.ElementTree as ET
from fractions import Fraction
from xml.sax.saxutils import escape, quoteattr

FR = Fraction(1001, 30000)          # 29.97 한 프레임(초)
CAMERA = re.compile(r"(C\d{4}|GH\d{6}|DJI_)")


def is_xmeml(text: str) -> bool:
    return "<xmeml" in text[:2000]


def _ft(frames: int | Fraction) -> str:
    """프레임 수(29.97) → FCPXML 시간."""
    return f"{int(round(frames * 1001))}/30000s"


def _sec(x: Fraction) -> str:
    return f"{int(round(x * 30000))}/30000s"


def _stem(name: str) -> str:
    return re.sub(r"\.(mp4|mov|mxf|m4v)$", "", name, flags=re.I)


def _rate(el) -> Fraction:
    tb = int(el.findtext("rate/timebase") or 30)
    ntsc = (el.findtext("rate/ntsc") or "TRUE").upper() == "TRUE"
    return Fraction(tb * 1000, 1001) if ntsc else Fraction(tb)


def to_fcpxml(text: str) -> str:
    root = ET.fromstring(text.encode("utf-8") if isinstance(text, str) else text)
    outer = root.find("sequence")
    files = {f.get("id"): f for f in root.iter("file") if f.find("pathurl") is not None}

    # 멀티캠 clipitem: 안에 sequence가 들어 있는 clipitem
    mc = next((c for c in outer.iter("clipitem") if c.find("sequence") is not None), None)
    if mc is None:
        raise ValueError("멀티캠 클립이 들어간 시퀀스가 아니에요. 프리미어에서 멀티캠 클립으로 만든 시퀀스를 내보내 주세요.")
    inner_id = mc.find("sequence").get("id")
    inner = next(s for s in root.iter("sequence") if s.get("id") == inner_id and s.find("media") is not None)

    angles = []        # (이름, file element, 멀티캠 시작 프레임, 원본 in 프레임, 길이 프레임)
    for tr in inner.find("media/video").findall("track"):
        for c in tr.findall("clipitem"):
            fref = c.find("file")
            f = files.get(fref.get("id")) if fref is not None else None
            if f is None:
                continue
            s, e, i = int(c.findtext("start")), int(c.findtext("end")), int(c.findtext("in"))
            angles.append((_stem(c.findtext("name") or f.findtext("name")), f, s, i, e - s))
    if not angles:
        raise ValueError("멀티캠 안에서 카메라 클립을 찾지 못했어요.")
    others = [a for a in angles if not CAMERA.match(a[0])]
    audio = max(others or angles, key=lambda a: a[4])

    # 쓸 구간: 시퀀스 시작/종료 표시 → 없으면 멀티캠 clipitem in/out
    m_in, m_out = int(mc.findtext("in")), int(mc.findtext("out"))
    s_in, s_out = outer.findtext("in"), outer.findtext("out")
    seq_start = int(mc.findtext("start") or 0)
    if s_in and s_out and int(s_in) >= 0 and int(s_out) > int(s_in):
        a, b = int(s_in), int(s_out)
        m_in, m_out = m_in + (a - seq_start), m_in + (b - seq_start)
    length = m_out - m_in

    fmts, assets, mcang = [], [], []
    ang_ids = {}
    for k, (name, f, s, i, d) in enumerate(angles):
        sc = f.find("media/video/samplecharacteristics")
        w, h = (sc.findtext("width"), sc.findtext("height")) if sc is not None else ("1920", "1080")
        rate = _rate(f)
        dur_s = Fraction(int(f.findtext("duration") or d)) / rate
        fmts.append(f'<format id="rf{k}" name="Premiere{k}" frameDuration="1001/30000s" width="{w}" height="{h}"/>')
        assets.append(
            f'<asset id="ra{k}" name={quoteattr(name)} uid="{uuid.uuid4().hex.upper()}" start="0s" duration="{_sec(dur_s)}" '
            f'hasVideo="1" format="rf{k}" hasAudio="1" videoSources="1" audioSources="1" audioChannels="2" audioRate="48000">'
            f'<media-rep kind="original-media" src={quoteattr(f.findtext("pathurl").replace("file://localhost/", "file:///"))}/></asset>')
        aid = uuid.uuid4().hex[:22]
        ang_ids[name] = aid
        gap = f'<gap name="Gap" offset="0s" start="0s" duration="{_ft(s)}"/>' if s > 0 else ""
        mcang.append(f'<mc-angle name={quoteattr(name)} angleID="{aid}">{gap}'
                     f'<asset-clip ref="ra{k}" offset="{_ft(s)}" name={quoteattr(name)} start="{_ft(i)}" duration="{_ft(d)}" '
                     f'format="rf{k}" tcFormat="NDF" audioRole="dialogue"/></mc-angle>')
    video_aid = ang_ids[next((a[0] for a in angles if CAMERA.match(a[0])), angles[0][0])]
    name = outer.findtext("name") or "프리미어 멀티캠"
    G = 108000000          # 3600s (gap 시작, /30000s)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n\n<fcpxml version="1.14">\n    <resources>\n'
        '        <format id="r1" name="FFVideoFormat1080p2997" frameDuration="1001/30000s" width="1920" height="1080" colorSpace="1-1-1 (Rec. 709)"/>\n'
        + "".join(f"        {x}\n" for x in fmts + assets)
        + f'        <media id="r4" name={quoteattr(name)} uid="{uuid.uuid4().hex.upper()}">\n'
        '            <multicam format="r1" tcStart="0s" tcFormat="NDF">\n'
        + "".join(f"                {x}\n" for x in mcang)
        + '            </multicam>\n        </media>\n    </resources>\n    <library>\n'
        f'        <event name="WorshipEdit" uid="{uuid.uuid4().hex.upper()}">\n'
        f'            <project name={quoteattr(name)} uid="{str(uuid.uuid4()).upper()}">\n'
        f'                <sequence format="r1" duration="{_ft(length)}" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">\n'
        '                    <spine>\n'
        f'                        <gap name="Gap" offset="0s" start="{G}/30000s" duration="{length * 1001}/30000s">\n'
        f'                            <mc-clip ref="r4" lane="1" offset="{G}/30000s" name={quoteattr(name)} start="{_ft(m_in)}" duration="{_ft(length)}">\n'
        f'                                <mc-source angleID="{video_aid}" srcEnable="video"/>\n'
        f'                                <mc-source angleID="{ang_ids[audio[0]]}" srcEnable="audio"/>\n'
        '                            </mc-clip>\n'
        '                        </gap>\n                    </spine>\n                </sequence>\n'
        '            </project>\n        </event>\n    </library>\n</fcpxml>\n')
