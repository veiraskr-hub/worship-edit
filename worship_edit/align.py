"""가사 줄 위치 맞추기.

1단계: 기준 시간(anchor)이 적힌 가사 목록을 받아, 목소리 트랙에서 실제로 노래가
       시작되는 순간(쉬었다 시작 / 숨 쉬고 다시 시작)으로 옮긴다.
2단계(예정): Whisper 단어 타임스탬프와 악보 가사를 자동 대조해 anchor를 만든다.
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class Line:
    time: float          # 노래 시작(초)
    text: str
    chorus: bool = False  # 후렴 블록 첫 줄 (규칙 B)
    how: str = ""        # onset | breath | anchor | slide
    end: float | None = None   # 실황 자막 슬라이드처럼 끝 시각을 아는 경우


def read_anchors(path: str) -> list[Line]:
    """한 줄에 `시간|가사[|C]`."""
    out = []
    for raw in open(path, encoding="utf-8").read().strip().splitlines():
        p = raw.split("|")
        out.append(Line(float(p[0]), p[1].strip(), len(p) > 2 and p[2].strip().upper() == "C"))
    return out


class VocalTrack:
    def __init__(self, path: str):
        y, sr = librosa.load(path, sr=16000, mono=True)
        hop = 160
        db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=640, hop_length=hop)[0] + 1e-7)
        self.db = np.convolve(db, np.ones(5) / 5, "same")
        self.t = np.arange(len(self.db)) * hop / sr

    def snap(self, x: float, window: float = 1.4) -> tuple[float, str]:
        t, db = self.t, self.db
        m = np.where((t > x - window) & (t < x + window))[0]
        if len(m) < 3:
            return x, "anchor"
        ref = np.percentile(db[np.where((t > x - 6) & (t < x + 6))[0]], 90)
        act = db[m] > ref - 20
        cand = [t[m[i]] for i in range(1, len(m)) if act[i] and not act[max(0, i - 15):i].any()]
        if cand:
            return float(min(cand, key=lambda v: abs(v - x))), "onset"
        seg = db[m]
        i = int(np.argmin(seg))
        if np.median(seg) - seg[i] >= 8:
            j = i
            while j < len(seg) - 1 and seg[j] < seg[i] + 8:
                j += 1
            return float(t[m[j]]), "breath"
        return x, "anchor"

    def starts_ok(self, times: list[float]) -> int:
        n = 0
        for s in times:
            a = self.db[(self.t >= s) & (self.t < s + 0.5)].mean()
            b = self.db[(self.t >= s - 0.5) & (self.t < s)].mean()
            n += bool(a > -30 and a - b > 3)
        return n


def snap_lines(vocals: VocalTrack, lines: list[Line]) -> list[Line]:
    out = []
    for ln in lines:
        s, how = vocals.snap(ln.time)
        out.append(Line(s, ln.text, ln.chorus, how))
    for i in range(1, len(out)):
        if out[i].time <= out[i - 1].time + 0.4:
            out[i].time = out[i - 1].time + 0.4
    return out


def display_spans(lines: list[Line], lead: float = 0.1, join_gap: float = 30, hold: float = 8.0):
    """자막 표시 구간: 목소리 0.1초 전에 띄우고, 곡 안에서는 다음 줄 직전까지 이어 붙인다."""
    spans = []
    for i, ln in enumerate(lines):
        st = ln.time - lead
        if ln.end is not None:                      # 실황 자막: 슬라이드가 내려간 시각(다음 줄 전까지)
            en = ln.end - lead
            if i + 1 < len(lines):
                en = min(en, lines[i + 1].time - lead)
        elif i + 1 < len(lines) and lines[i + 1].time - ln.time < join_gap:
            en = lines[i + 1].time - lead
        else:
            en = st + hold
        spans.append((st, en, ln.text))
    return spans
