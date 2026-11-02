"""가사 보관함.

곡 하나 = 텍스트 파일 하나 (사용자 Mac에만 저장, 저장소에는 올리지 않음):

    제목: 새힘얻으리
    진행: V V V V P C V V P P C P P P C C T T T T
    후렴: C

    [V]
    새 힘 얻으리 주를 바랄 때
    주를 바랄 때 우리 주를 바랄 때

    [P]
    주님 통치하시네
    소망 구원 주시는

    [C]
    당신은 영원하신 주
    ...

- 파트 이름은 자유(V, V1, V2, C, P, B, T …). `진행`은 실제로 부르는 순서(악보 진행표).
- `후렴`에 적은 파트의 첫 줄이 규칙 B(후렴 첫 소절)의 기준이 된다. 생략하면 C로 시작하는 파트.
- 한 줄 = 화면에 한 번 뜨는 자막 한 줄.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

LIB_DIR = os.path.expanduser("~/Library/Application Support/WorshipEdit/lyrics")


@dataclass
class Song:
    title: str
    sections: dict[str, list[str]]
    order: list[str]
    chorus_parts: list[str] = field(default_factory=list)

    def sequence(self) -> list[tuple[str, int, str]]:
        """진행 순서대로 펼친 (파트, 줄 번호, 가사)."""
        out = []
        for part in self.order:
            for i, line in enumerate(self.sections.get(part, [])):
                out.append((part, i, line))
        return out

    def is_chorus_start(self, part: str, i: int) -> bool:
        return i == 0 and part in self.chorus_parts

    @property
    def lines(self) -> list[tuple[str, int, str]]:
        return [(p, i, l) for p, ls in self.sections.items() for i, l in enumerate(ls)]


def parse(text: str) -> Song:
    title, order, chorus = "", [], []
    sections: dict[str, list[str]] = {}
    cur = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(제목|진행|후렴)\s*[:：]\s*(.*)$", line)
        if m:
            k, v = m.groups()
            if k == "제목":
                title = v.strip()
            elif k == "진행":
                order = v.replace(",", " ").replace("-", " ").split()
            else:
                chorus = v.replace(",", " ").split()
            continue
        m = re.match(r"^\[(.+?)\]$", line)
        if m:
            cur = m.group(1).strip()
            sections.setdefault(cur, [])
            continue
        if cur is not None:
            sections[cur].append(line)
    if not order:
        order = list(sections)
    if not chorus:
        chorus = [p for p in sections if p.upper().startswith("C")]
    return Song(title, sections, order, chorus)


def dump(song: Song) -> str:
    out = [f"제목: {song.title}", f"진행: {' '.join(song.order)}", f"후렴: {' '.join(song.chorus_parts)}", ""]
    for p, ls in song.sections.items():
        out.append(f"[{p}]")
        out += ls
        out.append("")
    return "\n".join(out)


def _key(title: str) -> str:
    return re.sub(r"\s+", "", title)


def load_library(lib_dir: str = LIB_DIR) -> dict[str, Song]:
    songs = {}
    if os.path.isdir(lib_dir):
        for f in sorted(os.listdir(lib_dir)):
            if f.endswith(".txt"):
                s = parse(open(os.path.join(lib_dir, f), encoding="utf-8").read())
                songs[_key(s.title or f[:-4])] = s
    return songs


def save_song(song: Song, lib_dir: str = LIB_DIR) -> str:
    os.makedirs(lib_dir, exist_ok=True)
    p = os.path.join(lib_dir, _key(song.title) + ".txt")
    open(p, "w", encoding="utf-8").write(dump(song))
    return p


INBOX = os.path.expanduser("~/Downloads/WorshipEdit 가사")


def import_files(paths, lib_dir: str = LIB_DIR) -> tuple[list[str], list[str]]:
    """가사 파일(.txt)을 보관함으로 가져온다. 같은 제목이면 새 내용으로 바꾼다. (가져온 제목, 실패한 파일)"""
    ok, bad = [], []
    for p in paths:
        try:
            s = parse(open(p, encoding="utf-8-sig").read())
            if not s.title or not s.sections:
                raise ValueError
            save_song(s, lib_dir); ok.append(s.title)
        except Exception:
            bad.append(os.path.basename(p))
    return ok, bad


def import_inbox(inbox: str = INBOX, lib_dir: str = LIB_DIR) -> list[str]:
    """앱을 켤 때 '다운로드/WorshipEdit 가사' 폴더의 새 파일·바뀐 파일을 자동으로 가져온다."""
    if not os.path.isdir(inbox):
        return []
    todo = []
    for n in sorted(os.listdir(inbox)):
        if not n.endswith(".txt"):
            continue
        src = os.path.join(inbox, n)
        try:
            dst = os.path.join(lib_dir, _key(parse(open(src, encoding="utf-8-sig").read()).title) + ".txt")
        except Exception:
            continue
        if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
            todo.append(src)
    return import_files(todo, lib_dir)[0]


def find(library: dict[str, Song], title: str) -> Song | None:
    return library.get(_key(title))
