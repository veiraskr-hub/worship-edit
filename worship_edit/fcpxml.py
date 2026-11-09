"""Final Cut Pro XML(.fcpxmld / .fcpxml) 읽기·쓰기.

프로젝트 안의 멀티캠 클립 하나를 찾아 앵글 정보와 위치를 읽고,
컷(mc-clip 여러 개)과 가사 타이틀을 써 넣는다.
"""
from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from fractions import Fraction
from xml.sax.saxutils import escape

FRAME = 1001            # 29.97fps 한 프레임 = 1001/30000초
FPS = 30000 / 1001


def _t(v: str) -> Fraction:
    return Fraction(v.rstrip("s")) if v else Fraction(0)


@dataclass
class Angle:
    name: str
    angle_id: str
    coverage: list[tuple[float, float]] = field(default_factory=list)  # 멀티캠 시간(초)

    def covers(self, a: float, b: float, margin: float = 0.5) -> bool:
        return any(x + margin <= a and b <= y - margin for x, y in self.coverage)


@dataclass
class Project:
    path: str
    text: str
    angles: dict[str, Angle]
    clip_xml: str            # 원래 mc-clip 블록 (교체 대상)
    clip_ref: str
    clip_lane: str           # ' lane="1"' 또는 ''
    clip_offset: Fraction    # 타임라인 위치
    clip_start: Fraction     # 멀티캠 시간
    duration: Fraction
    audio_angle_id: str
    gap_start: int           # 1/30000초 단위
    gap_duration: int

    @property
    def length(self) -> float:
        return float(self.duration)


def info_path(path: str) -> str:
    """`.fcpxmld` 폴더면 안의 Info.fcpxml 경로를 돌려준다."""
    if os.path.isdir(path):
        p = os.path.join(path, "Info.fcpxml")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{path} 안에 Info.fcpxml이 없어요")
        return p
    return path


def load(path: str) -> Project:
    p = info_path(path)
    s = open(p, encoding="utf-8").read()
    from .premiere_in import is_xmeml, to_fcpxml
    if is_xmeml(s):                       # 프리미어 멀티캠 시퀀스 XML
        s = to_fcpxml(s)
    angles: dict[str, Angle] = {}
    for m in re.finditer(r'<mc-angle name="([^"]*)" angleID="([^"]*)">(.*?)</mc-angle>', s, re.S):
        name, aid, body = m.groups()
        ang = Angle(name, aid)
        for c in re.finditer(r'<asset-clip [^>]*?offset="([^"]*)"[^>]*?duration="([^"]*)"', body):
            o, d = _t(c.group(1)), _t(c.group(2))
            ang.coverage.append((float(o), float(o + d)))
        angles[name] = ang
    clips = list(re.finditer(r'[ ]*<mc-clip ref="([^"]*)"([^>]*)>(.*?)</mc-clip>\n', s, re.S))
    if len(clips) != 1:
        raise ValueError(f"프로젝트에 멀티캠 클립이 {len(clips)}개 있어요. 컷하지 않은 멀티캠 클립 하나만 있는 프로젝트를 내보내 주세요.")
    mc = clips[0]
    attrs, body = mc.group(2), mc.group(3)
    lane = re.search(r'lane="([^"]*)"', attrs)
    aud = re.search(r'angleID="([^"]*)" srcEnable="(?:all|audio)"', body)
    gap = re.search(r'<gap name="Gap" offset="0s" start="(\d+)/30000s" duration="(\d+)/30000s">', s)
    if not gap:
        raise ValueError("프로젝트 기본 스토리라인의 gap을 찾지 못했어요")
    return Project(
        path=p, text=s, angles=angles, clip_xml=mc.group(0), clip_ref=mc.group(1),
        clip_lane=f' lane="{lane.group(1)}"' if lane else "",
        clip_offset=_t(re.search(r'offset="([^"]*)"', attrs).group(1)),
        clip_start=_t(re.search(r'start="([^"]*)"', attrs).group(1)),
        duration=_t(re.search(r'duration="([^"]*)"', attrs).group(1)),
        audio_angle_id=aud.group(1) if aud else "",
        gap_start=int(gap.group(1)), gap_duration=int(gap.group(2)),
    )


# ── 앵글 역할 자동 분류 ─────────────────────────────────────────────
ROLES = ("lead", "singer", "drum", "keys", "guitar")
ROLE_KO = {"lead": "인도자", "singer": "싱어", "drum": "드럼", "keys": "피아노", "guitar": "기타", "singer,keys": "싱어+피아노",
           "audio": "실황 녹화본(오디오)"}


def is_camera_role(r: str) -> bool:
    return bool(r) and all(x in ROLES for x in r.split(","))


def classify_roles(angles: dict[str, Angle]) -> dict[str, str]:
    """카메라 파일 이름 규칙으로 역할을 정한다. 매주 이름만 바뀌고 배치는 같다는 가정."""
    roles: dict[str, str] = {}
    djis = []
    for name, a in angles.items():
        if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}", name):
            roles[name] = "audio"
        elif re.match(r"C9\d{3}", name):
            roles[name] = "lead"
        elif re.match(r"C[0-8]\d{3}", name):
            roles[name] = "singer"
        elif re.match(r"GH\d{2}", name):
            roles[name] = "guitar"
        elif name.startswith("DJI"):
            djis.append(name)
        else:
            roles[name] = "unknown"
    # DJI 두 대: 파일 번호가 작은(먼저 녹화를 시작한) 쪽이 드럼, 나머지가 피아노
    djis.sort(key=lambda n: int(re.search(r"_(\d{4})_D", n).group(1)) if re.search(r"_(\d{4})_D", n) else 9999)
    for i, n in enumerate(djis):
        roles[n] = "drum" if i == 0 else "keys"
    # 날짜 이름 실황본이 없으면(프리미어 등) 이름 규칙에 안 맞는 앵글 중 가장 긴 것을 실황 녹화본으로
    unk = [n for n, r in roles.items() if r == "unknown"]
    if unk and "audio" not in roles.values():
        roles[max(unk, key=lambda n: sum(b - a for a, b in angles[n].coverage))] = "audio"
    return roles


# ── 쓰기 ───────────────────────────────────────────────────────────
def rename(text: str, name: str) -> str:
    return re.sub(r'<project name="[^"]*" uid="[^"]*"',
                  f'<project name="{escape(name)}" uid="{str(uuid.uuid4()).upper()}"', text, count=1)


def write_cuts(prj: Project, cuts: list[tuple[int, int, str]], text: str | None = None) -> str:
    """cuts: (시작 프레임, 끝 프레임, 앵글 이름) — 프로젝트 시작 기준."""
    s = text if text is not None else prj.text
    ind = " " * (len(prj.clip_xml) - len(prj.clip_xml.lstrip(" ")))
    off, st = int(prj.clip_offset * 30000), int(prj.clip_start * 30000)
    blk = []
    for a, b, name in cuts:
        aid = prj.angles[name].angle_id
        blk.append(
            f'{ind}<mc-clip ref="{prj.clip_ref}"{prj.clip_lane} offset="{off + a * FRAME}/30000s" '
            f'name="Untitled Multicam Clip" start="{st + a * FRAME}/30000s" duration="{(b - a) * FRAME}/30000s">\n'
            f'{ind}    <mc-source angleID="{aid}" srcEnable="video"/>\n'
            f'{ind}    <mc-source angleID="{prj.audio_angle_id}" srcEnable="audio"/>\n'
            f'{ind}</mc-clip>\n')
    return s.replace(prj.clip_xml, "".join(blk))


# 글자 면(Face): 흰색 100%, Blur 0 / 외곽선: 검정 20%, Width 0.5 / 그림자: 검정 75%, Blur 7.53, Distance 5, Angle 315
# strokeWidth는 음수로 써야 "면 채우기 + 외곽선"이 된다. 양수면 외곽선만 그려져 글자 속이 비어 검게 보인다.
TITLE_STYLE = {
    "font": "NanumMyeongjo", "fontSize": "50", "fontFace": "Regular", "fontColor": "1 1 1 1",
    "strokeColor": "0 0 0 0.2", "strokeWidth": "-0.5",
    "shadowColor": "0 0 0 0.75", "shadowOffset": "5 315", "shadowBlurRadius": "7.53",
    "alignment": "center",
}
TITLE_POSITION = "0 -450"


def write_titles(prj: Project, text: str, lines: list[tuple[float, float, str]],
                 style: dict | None = None, position: str = TITLE_POSITION, lane: int = 2) -> str:
    """lines: (표시 시작초, 끝초, 가사). gap 안에 Basic Title로 넣는다."""
    st = {**TITLE_STYLE, **(style or {})}
    attrs = " ".join(f'{k}="{escape(v)}"' for k, v in st.items())
    fr = lambda x: int(round(x * FPS))
    out, prev_end = [], -1
    gd = prj.gap_duration // FRAME
    for i, (a_s, b_s, txt) in enumerate(lines, 1):
        a, b = max(fr(a_s), prev_end), min(fr(b_s), gd)
        if b <= a:
            continue
        prev_end = b
        q = escape(txt.replace("\n", " / "), {'"': "&quot;"})
        out.append(
            f'<title ref="rWE_BT" lane="{lane}" offset="{prj.gap_start + a * FRAME}/30000s" name="{q}" start="3600s" duration="{(b - a) * FRAME}/30000s">\n'
            f'    <param name="Position" key="9999/999166631/999166633/1/100/101" value="{position}"/>\n'
            f'    <param name="Alignment" key="9999/999166631/999166633/2/354/999169573/401" value="1 (Center)"/>\n'
            f'    <text><text-style ref="wets{i}">{escape(txt)}</text-style></text>\n'
            f'    <text-style-def id="wets{i}"><text-style {attrs}/></text-style-def>\n'
            f'</title>\n')
    if 'id="rWE_BT"' not in text:
        text = text.replace("</resources>",
                            '    <effect id="rWE_BT" name="Basic Title" uid=".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti"/>\n    </resources>', 1)
    g = re.search(r'<gap name="Gap" offset="0s" start="\d+/30000s" duration="\d+/30000s">', text)
    k = text.index("</gap>", g.end())
    return text[:k] + "".join(out) + text[k:]


def save(text: str, out_path: str) -> str:
    """`.fcpxml` 단일 파일로 저장 (파이널 컷 가져오기용)."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path
