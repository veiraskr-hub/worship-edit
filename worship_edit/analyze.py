"""오디오 분석: 곡·멘트 구간, 곡 시작 방식(드럼 카운트/피아노), 박자, 세기."""
from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np

SR = 22050
HOP = 512


@dataclass
class Song:
    start: float
    end: float
    open: str = "keys"          # "drum" | "keys"
    countin: float | None = None


@dataclass
class Structure:
    songs: list[Song]
    ments: list[tuple[float, float]]
    chorus: list[float] = field(default_factory=list)

    def to_dict(self):
        return {"songs": [s.__dict__ for s in self.songs], "ments": self.ments, "chorus": self.chorus}

    @staticmethod
    def from_dict(d):
        return Structure([Song(**s) for s in d["songs"]], [tuple(m) for m in d["ments"]], list(d.get("chorus", [])))


class Audio:
    """원본 오디오의 음량·박자 정보를 한 번 계산해 둔다."""

    def __init__(self, path: str):
        self.y, _ = librosa.load(path, sr=SR, mono=True)
        self.length = len(self.y) / SR
        rms = librosa.feature.rms(y=self.y, hop_length=HOP)[0]
        self.db = 20 * np.log10(rms + 1e-6)
        self.t = librosa.frames_to_time(np.arange(len(self.db)), sr=SR, hop_length=HOP)
        self.onset = librosa.onset.onset_strength(y=self.y, sr=SR, hop_length=HOP)

    def loud(self, a: float, b: float) -> float:
        m = (self.t >= a) & (self.t < b)
        return float(self.db[m].mean()) if m.any() else -60.0

    def level(self, a: float) -> str:
        v = self.loud(a, a + 3)
        return "loud" if v > -15.2 else ("mid" if v > -24 else "quiet")

    def beats(self, songs: list[Song]) -> tuple[np.ndarray, np.ndarray]:
        bt = []
        for s in songs:
            seg = self.y[int(s.start * SR):int(s.end * SR)]
            _, bf = librosa.beat.beat_track(y=seg, sr=SR, hop_length=HOP)
            bt += (librosa.frames_to_time(bf, sr=SR, hop_length=HOP) + s.start).tolist()
        bt = np.array(sorted(bt))
        strength = np.array([self.onset[min(len(self.onset) - 1, int(b * SR / HOP))] for b in bt])
        return bt, strength

    def hp_profile(self, a: float, b: float, step: float = 0.5):
        """[a,b] 구간의 (시간, 화성 dB, 타악 dB) 목록."""
        a = max(0.0, a)
        seg = self.y[int(a * SR):int(b * SR)]
        H, P = librosa.effects.hpss(seg)
        n = int(SR * step)
        h = librosa.feature.rms(y=H, frame_length=n, hop_length=n)[0]
        p = librosa.feature.rms(y=P, frame_length=n, hop_length=n)[0]
        return [(a + i * step, 20 * np.log10(x + 1e-9), 20 * np.log10(z + 1e-9)) for i, (x, z) in enumerate(zip(h, p))]


def detect_structure(audio: Audio, vocals=None, min_song: float = 40) -> Structure:
    """1초 음량으로 곡 구간을 찾고, 나머지를 멘트·기도로 본다."""
    L = np.array([audio.loud(i, i + 1) for i in range(int(audio.length))])
    act = np.convolve(L > -24, np.ones(3) / 3, "same") > 0.5
    segs, s0 = [], None
    for i, a in enumerate(act):
        if a and s0 is None:
            s0 = i
        if not a and s0 is not None:
            segs.append([s0, i]); s0 = None
    if s0 is not None:
        segs.append([s0, len(act)])
    merged = []
    for a, b in segs:
        # 사이가 짧고, 그 사이에도 음악이 깔려 있으면(완전히 조용하지 않으면) 같은 곡
        if merged and a - merged[-1][1] < 30 and np.percentile(L[merged[-1][1]:a], 20) > -34:
            merged[-1][1] = b
        elif merged and a - merged[-1][1] < 4:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    songs = []
    for a, b in merged:
        if b - a < min_song:
            continue
        start, open_, cin = _find_start(audio, a, vocals=vocals)
        end = _refine_end(audio, b)
        songs.append(Song(round(start, 2), round(end, 2), open_, cin))
    ments, t = [], 0.0
    for s in songs:
        a = s.countin if s.countin is not None else s.start
        if a - t > 1:
            ments.append((round(t, 2), round(a, 2)))
        t = s.end
    if audio.length - t > 1:
        ments.append((round(t, 2), round(audio.length, 2)))
    return Structure(songs, ments)


def _speech(vocals, t: float) -> bool:
    """목소리 트랙에서 t~t+0.5초에 말소리(멘트)가 있는지."""
    if vocals is None:
        return False
    m = (vocals.t >= t) & (vocals.t < t + 0.5)
    return bool(m.any() and vocals.db[m].mean() > -32)


def _find_start(audio: Audio, a: float, back: float = 50, vocals=None) -> tuple[float, str, float | None]:
    """곡 시작 찾기.
    1) 크게 시작하는 지점(a)에서 거꾸로, 건반·패드 같은 화성 소리가 끊김 없이(1초 이하 끊김 허용)
       이어지는 데까지 되짚어 조용한 인트로를 포함한다.
    2) 그 인트로 안에 강한 타악 히트가 있으면 드럼으로 시작하는 곡(그 히트가 시작).
    """
    prof = audio.hp_profile(a - back, a + 3, 0.5)
    idx = next((i for i, (t, _, _) in enumerate(prof) if t >= a), len(prof) - 1)
    i, miss = idx, 0
    while i > 0:
        tt, h, p = prof[i - 1]
        if _speech(vocals, tt):          # 멘트가 들리면 거기까지는 곡이 아님
            break
        if max(h, p) > -42:
            miss = 0
        else:
            miss += 1
            if miss > 2:
                break
        i -= 1
    first = i + miss
    start = prof[first][0] if first < len(prof) else float(a)
    # 시작 4초 안에 강한 타악 히트가 있으면 드럼 카운트로 시작하는 곡
    hits = [(t, h, p) for t, h, p in prof[first:] if p > -38 and p > h + 1 and t <= start + 4]
    if hits:
        t_hit = hits[0][0]
        return t_hit + 0.5, "drum", round(max(start, t_hit - 0.5), 2)
    return start, "keys", None


def _refine_end(audio: Audio, b: float) -> float:
    prof = audio.hp_profile(b - 8, b + 6, 0.5)
    peak = max(max(h, p) for _, h, p in prof)
    end = b
    for t, h, p in prof:
        if max(h, p) > peak - 15:
            end = t + 0.5
    return float(end)


