"""규칙 기반 멀티캠 앵글 컷 생성.

규칙
 1  같은 앵글 연속 금지 (A·D로 이어지는 인도자 컷은 하나로 합침)
 2  인도자 노래 구간 비중 목표 (기본 35%) — 가중치 배수를 자동 조정
 3  조용한 구간에는 드럼 금지
 4  곡 시작: 드럼 카운트면 드럼(다운비트+4박), 건반 인트로면 피아노(8박)
 5  드럼 예비박 뒤 첫 간주 컷은 기타(8박)
 A  멘트·기도는 인도자 고정
 B  후렴 첫 소절은 인도자 또는 싱어
 C  곡 마지막 40초는 컷 길이 20% 단축
 D  곡 마지막 8초는 인도자 롱테이크
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .analyze import Audio, Structure
from .fcpxml import FPS, Project

DEFAULT_WEIGHTS = {
    "loud": {"lead": 38, "drum": 20, "singer": 20, "guitar": 15, "keys": 15},
    "mid": {"lead": 48, "singer": 20, "keys": 15, "guitar": 15, "drum": 12},
    "quiet": {"lead": 70, "singer": 15, "keys": 15, "guitar": 10},
}
DEFAULT_DUR = {"loud": (2.2, 4.5), "mid": (3.5, 6.5), "quiet": (8.0, 14.0)}


@dataclass
class CutSettings:
    weights: dict = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_WEIGHTS.items()})
    durations: dict = field(default_factory=lambda: dict(DEFAULT_DUR))
    lead_target: float = 0.35
    last_chorus_len: float = 40.0
    last_chorus_factor: float = 0.8
    end_hold: float = 8.0
    seed: int = 1102


@dataclass
class Cut:
    start: float
    end: float
    role: str
    tag: str
    angle: str = ""


def _angle_for(prj: Project, roles: dict[str, str], role: str, a: float, b: float) -> str | None:
    ms, me = float(prj.clip_start) + a, float(prj.clip_start) + b
    for name, r in roles.items():
        if r == role and prj.angles[name].covers(ms, me):
            return name
    return None


def _generate(prj, roles, audio: Audio, st: Structure, cfg: CutSettings, lead_mult: float):
    rnd = random.Random(cfg.seed)
    W = {k: dict(v) for k, v in cfg.weights.items()}
    for k in W:
        W[k]["lead"] = W[k].get("lead", 0) * lead_mult
    beats, bstr = audio.beats(st.songs)
    END = prj.length
    fixed = [[a, b, "lead", "A"] for a, b in st.ments]
    for sg in st.songs:
        fixed.append([sg.end - cfg.end_hold, sg.end, "lead", "D"])
        bb = beats[beats > sg.start + 0.1]
        if sg.open == "drum":
            e1 = float(bb[3]) if len(bb) > 3 else sg.start + 2.5
            fixed.append([sg.countin if sg.countin is not None else sg.start, e1, "drum", "4"])
            i1 = int(np.searchsorted(beats, e1))
            e2 = float(beats[i1 + 8]) if i1 + 8 < len(beats) else e1 + 4
            fixed.append([e1, e2, "guitar", "5"])
        else:
            e1 = float(bb[7]) if len(bb) > 7 else sg.start + 6
            fixed.append([sg.start, e1, "keys", "4"])
    fixed.sort()
    lastch = [(sg.end - cfg.last_chorus_len, sg.end - cfg.end_hold) for sg in st.songs]
    chorus = st.chorus
    cuts: list[Cut] = []

    def add(a, b, r, tag):
        if b - a < 0.05:
            return
        if cuts and cuts[-1].role == r == "lead" and any(x in tag + cuts[-1].tag for x in "AD"):
            cuts[-1].end = b; cuts[-1].tag += "+" + tag; return
        cuts.append(Cut(a, b, r, tag))

    def free(a, b, prev, nextr):
        t = a
        while t < b - 0.05:
            c = audio.level(t)
            lo, hi = cfg.durations[c]
            if any(x <= t < z for x, z in lastch):
                lo, hi = lo * cfg.last_chorus_factor, hi * cfg.last_chorus_factor
            tgt = t + rnd.uniform(lo, hi)
            ch = [x for x in chorus if t + 1.2 < x < tgt + 1.5]
            if ch:
                te = ch[0]
            else:
                m = (beats > tgt - 0.7) & (beats < tgt + 0.7) & (beats > t + 1.5)
                te = float(beats[m][np.argmax(bstr[m] - np.abs(beats[m] - tgt) * 2)]) if m.any() else tgt
            if b - te < 1.8:
                te = b
            te = min(te, b)
            is_ch = any(abs(x - t) < 0.05 for x in chorus)
            pool = {"lead": 1, "singer": 1} if is_ch else W[c]
            opts = [r for r in pool if r != prev and _angle_for(prj, roles, r, t, te)
                    and not (te >= b - 0.01 and r == nextr)]
            if not opts:
                opts = [r for r in W["mid"] if r != prev and _angle_for(prj, roles, r, t, te)] or ["lead"]
            r = rnd.choices(opts, weights=[pool.get(x, 10) for x in opts])[0]
            add(t, te, r, "B" if is_ch else c)
            prev, t = r, te
        return prev

    t, prev = 0.0, None
    for a, b, r, tag in fixed:
        a = max(a, t)
        if a > t + 0.05:
            prev = free(t, a, prev, r)
        if b > a:
            if not _angle_for(prj, roles, r, a, b):
                r = "lead"
            add(a, b, r, tag); prev, t = r, b
    if t < END - 0.05:
        free(t, END, prev, None)
    return cuts


def _to_frames(prj, roles, cuts: list[Cut]) -> list[Cut]:
    N = int(prj.duration * 30000) // 1001
    out: list[Cut] = []
    p = 0
    for c in cuts:
        e = N if c.end >= prj.length - 0.01 else int(round(c.end * FPS))
        if e <= p:
            continue
        if out and out[-1].role == c.role:
            out[-1].end = e; continue
        out.append(Cut(p, e, c.role, c.tag)); p = e
    for c in out:
        c.angle = _angle_for(prj, roles, c.role, c.start / FPS, c.end / FPS) or _angle_for(prj, roles, "lead", c.start / FPS, c.end / FPS)
    return out


def lead_share(cuts: list[Cut], st: Structure) -> float:
    tot = lead = 0.0
    for c in cuts:
        a, b = c.start / FPS, c.end / FPS
        for s in st.songs:
            ov = max(0.0, min(b, s.end) - max(a, s.start))
            tot += ov
            lead += ov if c.role == "lead" else 0
    return lead / tot if tot else 0.0


def make_cuts(prj: Project, roles: dict[str, str], audio: Audio, st: Structure,
              cfg: CutSettings | None = None) -> tuple[list[Cut], dict]:
    """규칙 컷을 만들고 인도자 비중이 목표에 가장 가까운 결과를 고른다. 프레임 단위 Cut 목록 반환."""
    cfg = cfg or CutSettings()
    best = None
    for mult in (0.6, 0.7, 0.8, 0.85, 0.9, 1.0, 1.15, 1.3):
        fr = _to_frames(prj, roles, _generate(prj, roles, audio, st, cfg, mult))
        share = lead_share(fr, st)
        score = abs(share - cfg.lead_target)
        if best is None or score < best[0]:
            best = (score, fr, share, mult)
    _, fr, share, mult = best
    report = {
        "cuts": len(fr),
        "lead_share_songs": round(share, 3),
        "lead_mult": mult,
        "same_angle_in_row": sum(1 for i in range(1, len(fr)) if fr[i].angle == fr[i - 1].angle),
        "rule_cuts": [(round(c.start / FPS, 2), round(c.end / FPS, 2), c.role, c.tag) for c in fr if c.tag not in ("loud", "mid", "quiet")],
    }
    return fr, report
