"""받아쓰기(SRT 또는 Whisper 구간)와 가사 보관함을 대조해 가사 줄 위치를 자동으로 잡는다.

1. 라벨링: 받아쓰기 구간마다 글자를 훑으며 가장 비슷한 가사 줄을 찾아 붙인다 (곡 순서는 앞으로만).
2. 채우기: 붙은 줄 사이에 빠진 줄은 진행 순서로 무엇이 와야 하는지 찾고,
          목소리 트랙의 소절 시작점에 배치한다.
3. 싱크: 각 줄을 실제 목소리 시작 순간으로 옮긴다 (align.snap_lines).
"""
from __future__ import annotations

import difflib
import re
from collections import deque
from dataclasses import dataclass

import numpy as np

from .align import Line, VocalTrack, snap_lines
from .library import Song


def norm(s: str) -> str:
    return re.sub(r"[^가-힣a-zA-Z]", "", s).lower()


@dataclass
class Seg:
    start: float
    end: float
    text: str


def read_srt(path: str) -> list[Seg]:
    def ts(x):
        h, m, r = x.strip().split(":"); s, ms = r.split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    out = []
    for b in re.split(r"\n\s*\n", open(path, encoding="utf-8-sig").read().replace("\r", "").strip()):
        l = b.strip().split("\n")
        if len(l) >= 3 and "-->" in l[1]:
            a, e = l[1].split("-->")
            out.append(Seg(ts(a), ts(e), " ".join(l[2:]).strip()))
    return out


@dataclass
class Hit:
    time: float
    song: int
    text: str        # 가사 원문
    key: str         # 정규화 텍스트
    score: float
    filled: bool = False


def _vocab(songs: list[Song]):
    vocab = []   # (song idx, 원문, 정규화)
    for k, s in enumerate(songs):
        seen = set()
        for _, _, line in s.lines:
            n = norm(line)
            if n and n not in seen:
                seen.add(n); vocab.append((k, line, n))
    return vocab


def _scan(sg: Seg, vocab, allowed: set[int], min_score: float) -> list[Hit]:
    """구간 하나의 글자를 앞에서부터 훑으며 가사 줄을 붙인다."""
    T = norm(sg.text)
    hits, pos = [], 0
    while pos < len(T):
        best = None
        for k, line, n in vocab:
            if k not in allowed:
                continue
            for st in range(pos, min(pos + 6, len(T))):
                for L in (len(n) - 1, len(n), len(n) + 1):
                    cand = T[st:st + L]
                    if L <= 1 or len(cand) < max(2, int(len(n) * 0.6)):
                        continue
                    r = difflib.SequenceMatcher(None, n, cand).ratio() - 0.01 * (st - pos)
                    if best is None or r > best[0]:
                        best = (r, k, line, n, st, L)
        if best and best[0] >= min_score:
            r, k, line, n, st, L = best
            hits.append(Hit(sg.start + (sg.end - sg.start) * st / max(1, len(T)), k, line, n, r))
            pos = st + L
        else:
            pos += 1
    return hits


def _song_of_segments(segs: list[Seg], vocab, n_songs: int) -> list[int]:
    """곡은 순서대로 부른다고 보고, 구간별 1차 라벨과 가장 잘 맞는 곡 경계를 정한다."""
    votes = []
    for sg in segs:
        hs = _scan(sg, vocab, set(range(n_songs)), 0.72)
        v = [0.0] * n_songs
        for h in hs:
            v[h.song] += h.score * len(h.key)
        votes.append(v)
    n = len(segs)
    # dp[i][k] = 구간 0..i-1을 곡 0..k로 나눴을 때 최대 점수
    NEG = -1e18
    dp = [[NEG] * n_songs for _ in range(n + 1)]
    back = [[0] * n_songs for _ in range(n + 1)]
    for k in range(n_songs):
        dp[0][k] = 0.0
    for i in range(1, n + 1):
        for k in range(n_songs):
            stay = dp[i - 1][k]
            prev = max(range(k + 1), key=lambda j: dp[i - 1][j])
            move = dp[i - 1][prev]
            if stay >= move:
                dp[i][k], back[i][k] = stay + votes[i - 1][k], k
            else:
                dp[i][k], back[i][k] = move + votes[i - 1][k], prev
    k = max(range(n_songs), key=lambda j: dp[n][j])
    assign = [0] * n
    for i in range(n, 0, -1):
        assign[i - 1] = k
        k = back[i][k]
    return assign


def _place_in_segment(sg: Seg, hs: list[Hit], vt: VocalTrack | None):
    """한 구간에 여러 줄이 붙었으면, 글자 비율 대신 그 구간 안의 목소리 소절 시작점에 차례로 놓는다."""
    if vt is None or len(hs) < 2:
        return
    ons = _phrase_onsets(vt, sg.start - 1.0, sg.end)
    if len(ons) < len(hs):
        return
    j = 0
    for i, h in enumerate(hs):
        rest = len(hs) - i - 1
        cands = ons[j:len(ons) - rest]
        o = min(cands, key=lambda x: abs(x - h.time))
        j = ons.index(o) + 1
        h.time = o


def _label(segs: list[Seg], songs: list[Song], min_score=0.62, vt: VocalTrack | None = None) -> list[Hit]:
    vocab = _vocab(songs)
    song_of = _song_of_segments(segs, vocab, len(songs))
    hits: list[Hit] = []
    for sg, k in zip(segs, song_of):
        hs = _scan(sg, vocab, {k}, min_score)
        _place_in_segment(sg, hs, vt)
        hits += hs
    # 같은 줄이 4초 안에 두 번 → 인도자가 먼저 불러 준 것으로 보고 뒤의 것을 남긴다
    out: list[Hit] = []
    for h in hits:
        if out and out[-1].key == h.key and h.time - out[-1].time < 4:
            out[-1] = h
        else:
            out.append(h)
    return out


def _graph(song: Song) -> dict[str, list[str]]:
    seq = [norm(l) for _, _, l in song.sequence()]
    g: dict[str, list[str]] = {}
    for a, b in zip(seq, seq[1:]):
        g.setdefault(a, [])
        if b not in g[a]:
            g[a].append(b)
    return g


def _path(g, a: str, b: str, max_len=8) -> list[str] | None:
    """a 다음부터 b 직전까지 와야 하는 줄들 (가장 짧은 경로)."""
    q = deque([(a, [])]); seen = {a}
    while q:
        node, path = q.popleft()
        for nx in g.get(node, []):
            if nx == b:
                return path
            if nx not in seen and len(path) < max_len:
                seen.add(nx); q.append((nx, path + [nx]))
    return None


def _walk(g, a: str, b: str, n_mid: int) -> list[str] | None:
    """a 다음부터 b 직전까지 정확히 n_mid줄이 오는 경로 (반복 구간 허용)."""
    if n_mid < 0:
        return None
    layer = {a: []}
    for _ in range(n_mid):
        nxt = {}
        for node, path in layer.items():
            for x in g.get(node, []):
                if x not in nxt:
                    nxt[x] = path + [x]
        layer = nxt
        if not layer:
            return None
    for node, path in layer.items():
        if b in g.get(node, []):
            return path
    return None


def _phrase_onsets(vt: VocalTrack, a: float, b: float) -> list[float]:
    t, db = vt.t, vt.db
    m = np.where((t >= a) & (t < b))[0]
    if len(m) < 10:
        return []
    ref = np.percentile(db[np.where((t > a - 10) & (t < b + 10))[0]], 90)
    act = db[m] > ref - 20
    return [float(t[m[i]]) for i in range(15, len(m)) if act[i] and not act[i - 15:i].any()]


def _fill(hits: list[Hit], songs: list[Song], vt: VocalTrack) -> list[Hit]:
    texts = {k: {norm(l): l for _, _, l in s.lines} for k, s in enumerate(songs)}
    graphs = [_graph(s) for s in songs]
    out: list[Hit] = []
    for i, h in enumerate(hits):
        out.append(h)
        if i + 1 >= len(hits):
            break
        nx = hits[i + 1]
        if nx.song != h.song:
            continue
        gap = nx.time - h.time
        # 같은 곡 안에서 줄 사이 평균 간격
        diffs = [b.time - a.time for a, b in zip(hits, hits[1:]) if a.song == b.song == h.song and 1 < b.time - a.time < 15]
        step = float(np.median(diffs)) if diffs else 4.0
        want = int(round(gap / step)) - 1          # 시간상 들어갈 줄 수
        miss = None
        if want >= 1:
            for n in sorted(range(max(1, want - 1), min(want + 2, 14)), key=lambda n: abs(n - want)):
                miss = _walk(graphs[h.song], h.key, nx.key, n)
                if miss:
                    break
        if not miss:
            miss = _path(graphs[h.song], h.key, nx.key)
        if not miss or gap < (len(miss) + 1) * step * 0.6:
            continue
        ons = _phrase_onsets(vt, h.time + 1.0, nx.time - 1.0)
        for j, key in enumerate(miss, 1):
            target = h.time + gap * j / (len(miss) + 1)
            near = [o for o in ons if abs(o - target) < step * 0.5]
            if not near:                      # 그 자리에 목소리 소절이 없으면(간주 등) 넣지 않는다
                continue
            tt = min(near, key=lambda o: abs(o - target))
            ons = [o for o in ons if o > tt + 0.5]
            out.append(Hit(tt, h.song, texts[h.song].get(key, key), key, 0.0, filled=True))
    return sorted(out, key=lambda x: x.time)


def _chorus_flags(hits: list[Hit], songs: list[Song]) -> list[bool]:
    flags = []
    for i, h in enumerate(hits):
        s = songs[h.song]
        first = {norm(s.sections[p][0]) for p in s.chorus_parts if s.sections.get(p)}
        chorus_lines = {norm(l) for p in s.chorus_parts for l in s.sections.get(p, [])}
        prev = hits[i - 1] if i else None
        flags.append(h.key in first and (prev is None or prev.song != h.song or prev.key not in chorus_lines))
    return flags


def _fix_lag(hits: list[Hit], vt: VocalTrack, n_songs: int) -> dict:
    """받아쓰기 시간이 실제 노래보다 늦게 찍히는 정도를 곡마다 추정해 당긴다."""
    lags = {}
    for k in range(n_songs):
        d = []
        for h in hits:
            if h.song != k:
                continue
            ons = _phrase_onsets(vt, h.time - 4.0, h.time + 1.5)
            if ons:
                o = min(ons, key=lambda x: abs(x - h.time))
                d.append(h.time - o)
        lag = float(np.clip(np.median(d), 0, 3.5)) if len(d) >= 5 else 0.0
        lags[k] = round(lag, 2)
        for h in hits:
            if h.song == k:
                h.time -= lag
    return lags


def auto_lines(segs: list[Seg], songs: list[Song], vt: VocalTrack) -> tuple[list[Line], dict]:
    hits = _label(segs, songs, vt=vt)
    lags = _fix_lag(hits, vt, len(songs))
    hits = _fill(hits, songs, vt)
    flags = _chorus_flags(hits, songs)
    lines = [Line(h.time, h.text, f) for h, f in zip(hits, flags)]
    snapped = snap_lines(vt, lines)
    rep = {
        "labeled": sum(1 for h in hits if not h.filled),
        "filled": sum(1 for h in hits if h.filled),
        "voice_start_ok": vt.starts_ok([l.time for l in snapped]),
        "per_song": {songs[k].title: sum(1 for h in hits if h.song == k) for k in range(len(songs))},
        "lag": {songs[k].title: lags[k] for k in lags},
    }
    return snapped, rep
