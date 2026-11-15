"""실황 녹화본 하단 가사(예배 자막 슬라이드) 읽기.

실황 화면 아래에 뜨는 가사 슬라이드(보통 두 줄)를 0.5초 간격으로 문자 인식(OCR)해서
"언제 어떤 슬라이드가 떠 있었는지" 목록을 만든다. 가사 글자·순서의 정답으로 쓴다.

- 맥: Apple Vision (라이브 텍스트와 같은 엔진, 한국어)  — pyobjc-framework-Vision
- 그 밖(개발·테스트): tesseract (kor)
- 흰 글자만 남기는 전처리 → 글자 모양이 바뀔 때만 OCR (같은 슬라이드는 다시 읽지 않음)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

# 1920x1080 기준 하단 가사 영역 (x, y, w, h) — 화면 비율로 저장
REGION = (0.01, 0.76, 0.62, 0.185)


@dataclass
class Slide:
    start: float
    end: float
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " / ".join(self.lines)


def norm(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", s)


def similar(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# ── 프레임 ─────────────────────────────────────────────────────────
def frames(video: str, start: float, dur: float, fps: float = 2.0, region=REGION, size=None):
    """(시간, RGB 배열) — 영상 시간이 아니라 start 기준 시간."""
    from .media import ffmpeg_exe
    if size is None:
        size = _probe_size(video)
    W, H = size
    x, y, w, h = (int(region[0] * W), int(region[1] * H), int(region[2] * W), int(region[3] * H))
    w -= w % 2; h -= h % 2
    cmd = [ffmpeg_exe(), "-v", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if dur:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-i", video, "-vf", f"fps={fps},crop={w}:{h}:{x}:{y}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = w * h * 3
    k = 0
    try:
        while True:
            b = p.stdout.read(n)
            if len(b) < n:
                break
            yield k / fps, np.frombuffer(b, np.uint8).reshape(h, w, 3)
            k += 1
    finally:
        p.stdout.close(); p.wait()


def _probe_size(video: str) -> tuple[int, int]:
    from .media import ffmpeg_exe
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", video], capture_output=True, text=True)
    m = re.search(r"Video:.*?(\d{3,5})x(\d{3,5})", r.stderr)
    return (int(m.group(1)), int(m.group(2))) if m else (1920, 1080)


def text_mask(img: np.ndarray) -> np.ndarray:
    """흰 글자만 (크림색 벽·조명은 파란 값이 낮아서 빠짐)."""
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    return (r > 215) & (g > 215) & (b > 215) & (np.abs(r - b) < 28)


# ── OCR ────────────────────────────────────────────────────────────
class OCR:
    def __init__(self, words: list[str] | None = None):
        self.engine = "vision" if sys.platform == "darwin" and self._vision_ok() else "tesseract"
        self.words = sorted({w for w in (words or []) if len(w) >= 2})[:500]   # 가사 보관함 낱말 (Vision 힌트)

    @staticmethod
    def _vision_ok() -> bool:
        try:
            import Vision  # noqa: F401
            import Quartz  # noqa: F401
            return True
        except Exception:
            return False

    def read(self, gray_text_on_white: np.ndarray) -> list[str]:
        if self.engine == "vision":
            return self._vision(gray_text_on_white)[0]
        return self._tesseract(gray_text_on_white)

    def read_best(self, variants: list[np.ndarray]) -> list[str]:
        """같은 장면을 여러 모양(흰 글자만 남긴 것, 원본 색)으로 읽어서 더 확실한 쪽.
        Vision만 확신도를 주므로, tesseract는 첫 번째(흰 글자 마스크)만 읽는다."""
        if self.engine != "vision":
            return self._tesseract(variants[0])
        best, score = [], -1.0
        for v in variants:
            lines, sc = self._vision(v)
            if sc > score:
                best, score = lines, sc
        return best

    def _png(self, a: np.ndarray) -> str:
        from PIL import Image
        fd, p = tempfile.mkstemp(suffix=".png"); os.close(fd)
        Image.fromarray(a).save(p)
        return p

    def _tesseract(self, a) -> list[str]:
        p = self._png(a)
        try:
            out = subprocess.run(["tesseract", p, "-", "-l", "kor", "--psm", "6"], capture_output=True, text=True).stdout
        finally:
            os.remove(p)
        return [l.strip() for l in out.splitlines() if norm(l)]

    def _vision(self, a) -> tuple[list[str], float]:
        """(줄 목록, 점수). 점수 = 줄마다 (확신도 × 한글 글자 수)의 합 — 잡음 줄은 한글이 없어 0에 가깝다."""
        import Quartz
        import Vision
        from Foundation import NSURL
        p = self._png(a)
        try:
            src = Quartz.CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(p), None)
            cg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
            req = Vision.VNRecognizeTextRequest.alloc().init()
            req.setRecognitionLanguages_(["ko-KR", "en-US"])
            req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
            req.setUsesLanguageCorrection_(True)
            h = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
            if self.words:
                try:
                    req.setCustomWords_(self.words)
                except Exception:
                    pass
            h.performRequests_error_([req], None)
            boxes, score = [], 0.0
            for o in req.results() or []:
                c = o.topCandidates_(1)
                if c:
                    b = o.boundingBox()          # 0~1, 아래가 0
                    txt = c[0].string().strip()
                    boxes.append((1 - b.origin.y - b.size.height, b.size.height, b.origin.x, txt))
                    if clean_line(txt):
                        score += float(c[0].confidence()) * len(re.findall(r"[가-힣]", txt))
            return rows(boxes), score
        finally:
            os.remove(p)


def rows(boxes) -> list[str]:
    """글자 상자들 (위 y, 높이, 왼쪽 x, 글자) → 줄 목록.
    한 줄이 두 상자로 갈라져 읽히면(‘못하고’ / ‘세상이 이해’) 높이가 겹치는 상자끼리 묶어 왼쪽부터 잇는다."""
    out: list[list] = []                     # [위, 아래, [(x, 글자)]]
    for y, h, x, s in sorted(boxes):
        if not s:
            continue
        for r in out:
            ov = min(r[1], y + h) - max(r[0], y)
            if ov > 0.5 * min(h, r[1] - r[0]):
                r[0], r[1] = min(r[0], y), max(r[1], y + h); r[2].append((x, s)); break
        else:
            out.append([y, y + h, [(x, s)]])
    out.sort(key=lambda r: r[0])
    return [" ".join(s for _, s in sorted(r[2])) for r in out]


def _prep(mask: np.ndarray) -> np.ndarray:
    """OCR용: 흰 바탕에 검은 글자, 2배 확대."""
    a = np.where(mask, 0, 255).astype(np.uint8)
    return np.kron(a, np.ones((2, 2), np.uint8))


# ── 슬라이드 목록 ──────────────────────────────────────────────────
def read_slides(video: str, start: float = 0, dur: float = 0, fps: float = 2.0, region=REGION,
                progress=None, ocr: OCR | None = None) -> list[Slide]:
    """1) 영상을 훑으며 흰 글자 마스크만 모은다 (빠름)
    2) 늘 흰색인 자리(건반·아이패드·선풍기·조명 같은 배경)를 지운다 — 자막 글자는 슬라이드마다 자리가 바뀌어서 안 지워진다
    3) 글자 모양이 바뀔 때마다, 같은 슬라이드라도 3초마다 한 번 더 읽어서 여러 번 읽은 것으로 투표"""
    ocr = ocr or OCR()
    total = dur or 0
    packed, shape, times, jpgs = [], None, [], {}
    keep_color = ocr.engine == "vision"
    for t, img in frames(video, start, dur, fps, region):
        m = text_mask(img)
        shape = m.shape
        packed.append(np.packbits(m))
        times.append(t)
        if keep_color and m.mean() >= 0.004:
            jpgs[len(packed) - 1] = _jpg(img)
        if progress and total:
            progress(0.4 * min(1.0, t / total))
    if not packed:
        return []
    n_px = shape[0] * shape[1]
    unpack = lambda k: np.unpackbits(packed[k])[:n_px].reshape(shape).astype(bool)
    bgs = background(packed, shape, fps)
    obs: list[tuple[float, list[str]]] = []
    last_mask, last_lines, last_t = None, [], -99.0
    for k, t in enumerate(times):
        m = clean_mask(unpack(k), bgs[k])
        if m.mean() < 0.004:                              # 글자 없음
            lines = []
        elif last_mask is not None and last_lines and _same_mask(m, last_mask) and t - last_t < 3.0:
            lines = last_lines
        else:
            lines = ocr.read_best([_prep(m), _unjpg(jpgs[k])] if k in jpgs else [_prep(m)])
            last_mask, last_t = m, t
        last_lines = lines
        obs.append((t, lines))
        if progress and total:
            progress(0.4 + 0.6 * min(1.0, t / total))
    return consensus(unify(_group(obs, 1.0 / fps)))


def _jpg(img: np.ndarray) -> bytes:
    import io
    from PIL import Image
    b = io.BytesIO(); Image.fromarray(img).save(b, "JPEG", quality=92)
    return b.getvalue()


def _unjpg(b: bytes) -> np.ndarray:
    import io
    from PIL import Image
    return np.array(Image.open(io.BytesIO(b)).convert("RGB"))


def background(packed, shape, fps: float, window: float = 300.0, thr: float = 0.9):
    """프레임마다 배경 마스크: 앞뒤 5분 동안 90% 넘게 흰색이던 자리 (1분 단위로 계산)."""
    from scipy import ndimage as ndi
    n_px = shape[0] * shape[1]
    chunk = max(1, int(60 * fps))
    sums, counts = [], []
    for i in range(0, len(packed), chunk):
        s = np.zeros(shape, np.uint32)
        for p in packed[i:i + chunk]:
            s += np.unpackbits(p)[:n_px].reshape(shape)
        sums.append(s); counts.append(len(packed[i:i + chunk]))
    w = max(1, int(round(window / 60)))
    out_chunks = []
    for c in range(len(sums)):
        a, b = max(0, c - w), min(len(sums), c + w + 1)
        f = sum(sums[a:b]) / max(1, sum(counts[a:b]))
        out_chunks.append(ndi.binary_dilation(f > thr, iterations=1))
    return [out_chunks[k // chunk] for k in range(len(packed))]


def clean_mask(m: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """배경 자리를 지우고, 지우고 남은 부스러기(15픽셀 미만 조각)도 지운다."""
    from scipy import ndimage as ndi
    m = m & ~bg
    lab, n = ndi.label(m)
    if n:
        size = ndi.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, np.nonzero(size >= 15)[0] + 1)
    return m


def _same_mask(a: np.ndarray, b: np.ndarray) -> bool:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return union == 0 or inter / union > 0.80


def clean_line(l: str) -> str:
    """OCR 잡음 정리: 낱자모·기호 지우기, 한글이 거의 없는 줄은 버림."""
    l = re.sub(r"[ㄱ-ㅣ|~^`'\"!·ㆍ_=+*<>{}\[\]]", " ", l)
    l = re.sub(r"\s+", " ", l).strip(" -.,:;")
    syl = len(re.findall(r"[가-힣]", l))
    other = len(re.sub(r"[가-힣\s]", "", l))
    return l if syl >= 2 and syl >= 2 * other else ""


def _clean(lines: list[str]) -> list[str]:
    """잡음 줄 빼기: 한 글자짜리 낱말이 셋 이상 늘어선 줄('승 즈 소')은 건반·배경을 글자로 잘못 읽은 것."""
    out = []
    for c in (clean_line(l) for l in lines):
        ws = c.split()
        if c and not (len(ws) >= 3 and sum(len(w) for w in ws) / len(ws) < 1.6):
            out.append(c)
    return out[:2]


def _key(lines: list[str]) -> str:
    return lines[0] if lines else ""


def _same_slide(a: list[str], b: list[str]) -> bool:
    """첫 줄이 같고, 둘째 줄도 (둘 다 있으면) 비슷해야 같은 슬라이드.
    첫 줄만 같은 다른 슬라이드('주님 뜻대로 살기로 했네 / 주님 뜻대로…' → '… / 뒤돌아서지 않겠네')를 가르기 위해."""
    if similar(_key(a), _key(b)) <= 0.7:
        return False
    if len(a) != len(b):              # 한 줄만 읽힌 장면이 서로 다른 두 슬라이드를 잇지 않게
        return False
    if len(a) >= 2 and similar(a[1], b[1]) < 0.6:
        return False
    return True


def _group(obs, step: float) -> list[Slide]:
    """같은 슬라이드(첫 줄 기준)가 이어지는 구간을 하나로. 0.5~1초짜리 전환(겹침) 장면은 버림.
    한 슬라이드 동안 여러 번 읽은 글자는 가장 많이 나온 것으로."""
    runs: list[list] = []          # [start, end, [lines 후보들]]
    for t, lines in obs:
        lines = _clean(lines)
        if runs and ((not lines and not runs[-1][2]) or
                     (lines and runs[-1][2] and any(_same_slide(lines, c) for c in runs[-1][2][-4:]))):
            runs[-1][1] = t + step
            if lines:
                runs[-1][2].append(lines)
            continue
        runs.append([t, t + step, [lines] if lines else []])
    runs = [r for r in runs if r[2] and r[1] - r[0] >= 1.5]
    merged: list[list] = []
    for r in runs:                 # 전환 장면이 빠지며 둘로 갈라진 같은 슬라이드 다시 합치기
        if merged and _same_slide(vote(r[2]), vote(merged[-1][2])) and r[0] - merged[-1][1] <= 1.5:
            merged[-1][1] = r[1]; merged[-1][2] += r[2]
        else:
            merged.append(r)
    # 3.5초 미만 조각(한 번 잘못 읽어 갈라진 것)은 붙어 있는 이웃 슬라이드와 첫 줄이 같으면 그쪽에 붙인다
    i = 0
    while i < len(merged):
        r = merged[i]
        if r[1] - r[0] < 3.5 and len(merged) > 1:
            v = vote(r[2])
            near = [j for j in (i - 1, i + 1) if 0 <= j < len(merged)
                    and min(abs(merged[j][1] - r[0]), abs(r[1] - merged[j][0])) <= 1.0
                    and any(similar(x, _key(vote(merged[j][2]))) > 0.7 for x in v)]
            if near:
                j = max(near, key=lambda j: merged[j][1] - merged[j][0])
                merged[j][0], merged[j][1] = min(merged[j][0], r[0]), max(merged[j][1], r[1])
                merged.pop(i)
                continue
        i += 1
    return [Slide(a, b, vote(c)) for a, b, c in merged]


def vote(cands: list[list[str]]) -> list[str]:
    """여러 번 읽은 결과에서 줄마다 가장 많이 나온 글자(띄어쓰기 무시하고 셈)."""
    from collections import Counter
    n = Counter(len(c) for c in cands).most_common(1)[0][0]
    out = []
    for i in range(n):
        opts = [c[i] for c in cands if len(c) > i]
        cnt = Counter(norm(o) for o in opts)
        best = cnt.most_common(1)[0][0]
        out.append(next(o for o in opts if norm(o) == best))
    return out


def unify(slides: list[Slide]) -> list[Slide]:
    """같은 슬라이드가 여러 번 나오면(후렴 반복) 모든 출현의 읽기를 모아 글자를 하나로 맞춘다."""
    groups: list[list[Slide]] = []
    for s in slides:
        for g in groups:
            if similar(s.text, g[0].text) > 0.92:
                g.append(s); break
        else:
            groups.append([s])
    for g in groups:
        best = vote([x.lines for x in g])
        for x in g:
            x.lines = list(best)
    return slides


def consensus(slides: list[Slide], min_sim: float = 0.85) -> list[Slide]:
    """같은 가사 줄은 예배 중 여러 슬라이드에 되풀이해 나온다 ('뒤돌아서지 않겠네' 4번).
    글자 수가 같고 거의 같은(85% 이상) 줄끼리 모아, 오래 떠 있던 표기로 통일한다 — 한 번 잘못 읽은 글자를 다른 출현이 고친다."""
    from collections import defaultdict
    weight: dict[str, float] = defaultdict(float)
    for sl in slides:
        for l in sl.lines:
            weight[l] += sl.end - sl.start
    groups: list[list[str]] = []
    for l in sorted(weight, key=lambda x: -weight[x]):
        for g in groups:
            if len(norm(l)) == len(norm(g[0])) and similar(l, g[0]) >= min_sim:
                g.append(l); break
        else:
            groups.append([l])
    best = {}
    for g in groups:
        by = defaultdict(float)
        for l in g:
            by[norm(l)] += weight[l]
        win = max(g, key=lambda l: (by[norm(l)], weight[l]))
        for l in g:
            best[l] = win
    for sl in slides:
        sl.lines = [best.get(l, l) for l in sl.lines]
    return slides


def write_slides(path: str, slides: list[Slide]):
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"start": round(s.start, 2), "end": round(s.end, 2), "lines": s.lines} for s in slides],
                  f, ensure_ascii=False, indent=1)


def load_slides(path: str) -> list[Slide]:
    import json
    return [Slide(d["start"], d["end"], d["lines"]) for d in json.load(open(path, encoding="utf-8"))]


# ── 슬라이드 → 자막 줄 ──────────────────────────────────────────────
def units(slide: Slide) -> list[str]:
    """슬라이드 두 줄 → 자막 단위. 한쪽이 짧으면(5글자 이하) 한 줄로 합친다 ('너와 함께 하려 내가 왔노라')."""
    ls = slide.lines
    if len(ls) == 2:
        n = [len(re.findall(r"[가-힣]", x)) for x in ls]
        if min(n) <= 5:
            return [ls[0] + " " + ls[1]]
    return list(ls)


def _fix_with_library(text: str, lib_lines: list[str], lo: float = 0.5) -> str:
    """OCR이 흐린 글자는 가사 보관함 줄로 고친다 (거의 같으면 실황 자막 글자를 그대로 둔다)."""
    if not lib_lines:
        return text
    n = len(norm(text))
    same_len = [l for l in lib_lines if 0.75 <= len(norm(l)) / max(1, n) <= 1.33]
    if not same_len:
        return text
    best = max(same_len, key=lambda l: similar(text, l))
    r = similar(text, best)
    return best if lo <= r < 0.85 else text


def sung_ratio(slide: Slide, segs) -> float:
    """받아쓰기에 이 슬라이드 글자가 얼마나 들어 있나 (기도·멘트 중 떠 있는 슬라이드 거르기)."""
    if not segs:
        return 1.0
    heard = norm(" ".join(s.text for s in segs if s.end > slide.start - 2 and s.start < slide.end + 2))
    want = norm(slide.text)
    if not want or not heard:
        return 0.0
    m = SequenceMatcher(None, want, heard, autojunk=False)
    return sum(b.size for b in m.get_matching_blocks()) / len(want)


def phrases(vt, gap: float = 0.35, min_dur: float = 0.6, below: float = 22.0) -> list[tuple[float, float]]:
    """목소리 소절 (시작, 길이): 0.35초 이상 쉬었다가 시작해서 0.6초 넘게 이어지는 소리."""
    t, db = vt.t, vt.db
    act = db > np.percentile(db, 90) - below
    step = float(t[1] - t[0])
    g, n = max(1, int(gap / step)), len(t)
    out, i = [], 0
    while i < n:
        if act[i] and not act[max(0, i - g):i].any():
            j = i
            while j < n and act[j:j + int(0.3 / step)].any():
                j += 1
            if t[min(j, n - 1)] - t[i] >= min_dur:
                out.append((float(t[i]), float(t[min(j, n - 1)] - t[i])))
            i = j
        i += 1
    return out


def to_lines(slides: list[Slide], vt, segs=None, songs=None):
    """슬라이드 → Line 목록.
    - 부르지 않은 슬라이드(기도·멘트 중 떠 있음) 빼기: 받아쓰기에 글자가 적고(50% 미만) 길게 이어지는 소절도 없으면
    - 첫 줄: 슬라이드가 뜨기 3초 전부터의 첫 소절 시작
    - 둘째 줄: 슬라이드 한 줄은 보통 두 마디 → 다음 슬라이드까지 간격의 절반 근처 소절 시작"""
    from .align import Line
    lib = [l for s in (songs or []) for _, _, l in s.lines]
    chorus_first = set()
    for s in songs or []:
        for part in getattr(s, "chorus_parts", []) or []:
            if s.sections.get(part):
                chorus_first.add(norm(s.sections[part][0]))
    ph = phrases(vt)
    starts = [a for a, _ in ph]
    sung, skipped = [], []
    for sl in slides:
        longest = max([d for a, d in ph if sl.start - 3 <= a < sl.end], default=0)
        if longest < 2.5 and sung_ratio(sl, segs) < 0.5:
            skipped.append((round(sl.start, 1), sl.text)); continue
        sung.append(sl)
    # 1) 슬라이드 첫 줄
    t1s, prev = [], -10.0
    for sl in sung:
        lo = max(prev + 2.0, sl.start - 3.0)
        c = [a for a in starts if lo <= a < sl.end]
        t = c[0] if c else max(lo, sl.start)
        t1s.append(t); prev = t
    # 2) 둘째 줄 (한 줄 간격 = 다음 슬라이드까지의 절반, 너무 멀면 전체 중앙값)
    gaps = [(b - a) / 2 for a, b in zip(t1s, t1s[1:]) if 8 < b - a < 25]
    med = float(np.median(gaps)) if gaps else 7.5
    out = []
    for k, sl in enumerate(sung):
        us = [_fix_with_library(u, lib) for u in units(sl)]
        t1 = t1s[k]
        nxt = t1s[k + 1] if k + 1 < len(t1s) else t1 + 2 * med
        est = (nxt - t1) / 2 if 8 < nxt - t1 < 25 else med
        times = [t1]
        if len(us) > 1:
            want = t1 + est
            c = [a for a in starts if abs(a - want) <= 2.5 and t1 + 2.5 <= a < nxt - 1.0]
            times.append(min(c, key=lambda a: abs(a - want)) if c else min(want, nxt - 1.0))
        for t, u in zip(times, us):
            out.append(Line(round(t, 2), u, norm(u) in chorus_first, "slide"))
    return out, {"slides": len(slides), "sung_slides": len(sung), "skipped": skipped, "lines": len(out),
                 "line_spacing": round(med, 2)}


def song_groups(lines, slides_text_gap: float = 30.0) -> list[tuple[float, float]]:
    """가사 보관함 없이 곡 구간 추정: 시간상 이어지거나(30초 이내) 같은 가사가 다시 나오면 같은 곡."""
    if not lines:
        return []
    parent = list(range(len(lines)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(1, len(lines)):
        if lines[i].time - lines[i - 1].time < slides_text_gap:
            parent[find(i)] = find(i - 1)
    first = {}
    for i, l in enumerate(lines):
        k = norm(l.text)
        if k in first:
            parent[find(i)] = find(first[k])
        else:
            first[k] = i
    groups = {}
    for i, l in enumerate(lines):
        e = l.end if getattr(l, "end", None) is not None else l.time
        a, b = groups.get(find(i), (l.time, e))
        groups[find(i)] = (min(a, l.time), max(b, e))
    spans = sorted(groups.values())
    merged = []                       # 겹치는 구간은 하나로
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def _first_sung(sl, ph):
    sing = [a for a, d in ph if sl.start - 1.0 <= a < sl.start + 8.0 and d >= 1.0]
    return sing[0] if sing else sl.start


def _split_lines(pend, ph, lead_in=0.0):
    """슬라이드 두 줄 → 한 줄씩. 한 줄은 보통 두 마디라서 둘째 줄은
    '이 슬라이드를 부르기 시작한 때'와 '다음 슬라이드를 부르기 시작한 때'의 가운데 무렵 소절 시작."""
    from .align import Line
    starts = [_first_sung(sl, ph) for sl, _, _, _ in pend]
    out = []
    for k, (sl, us, ch, end) in enumerate(pend):
        if len(us) == 1:
            out.append(Line(round(sl.start - lead_in, 2), us[0], ch, "slide", round(end, 2)))
            continue
        t1 = starts[k]
        nxt = starts[k + 1] if k + 1 < len(pend) and starts[k + 1] - t1 < 30 else end
        n1, n2 = (len(re.findall(r"[가-힣]", u)) for u in us)
        want = t1 + (nxt - t1) * min(0.7, max(0.4, n1 / max(1, n1 + n2)))
        cand = [a for a, d in ph if t1 + 2.5 <= a < nxt - 1.5 and abs(a - want) <= 2.5 and d >= 0.8]
        t2 = min(cand, key=lambda a: abs(a - want)) if cand else want
        out.append(Line(round(sl.start - lead_in, 2), us[0], ch, "slide", round(t2, 2)))
        out.append(Line(round(t2 - lead_in, 2), us[1], False, "slide", round(end, 2)))
    return out


def _unsung(slides: list[Slide], i: int, ph, segs) -> bool:
    """기도·멘트 중에 떠 있던 (부르지 않은) 슬라이드인가.
    - 곡 중간(앞뒤 슬라이드가 3초 안에 바로 이어짐)이면 부른 것으로 본다: 받아쓰기가 노래를 놓치거나
      소절이 슬라이드 전부터 길게 이어져도 빠뜨리지 않게 (11/15 둘째 곡에서 9장이 빠졌던 문제)
    - 묶음 처음·끝 슬라이드는: 안에서 시작하는 긴 소절(2.5초+)이 없고 받아쓰기에도 글자가 절반 미만이면 뺀다.
      단, 앞 슬라이드부터 5초 넘게 이어 부르던 소리가 2초 넘게 걸쳐 있으면 부른 것 (곡 마지막 슬라이드)"""
    sl = slides[i]
    prev_gap = sl.start - slides[i - 1].end if i > 0 else 99.0
    next_gap = slides[i + 1].start - sl.end if i + 1 < len(slides) else 99.0
    if prev_gap < 3.0 and next_gap < 3.0:
        return False
    if any(a <= sl.start - 5.0 and a + d >= sl.start + 2.0 for a, d in ph):
        return False
    longest = max([d for a, d in ph if sl.start + 0.5 <= a < sl.end], default=0)
    return longest < 2.5 and sung_ratio(sl, segs) < 0.5


def slide_lines(slides: list[Slide], vt, segs=None, songs=None, lead_in: float = 0.0, one_line: bool = True,
                fix_songs=None):
    """실황 자막을 그대로: 슬라이드 하나 = 자막 하나(두 줄), 뜬 시각·내려간 시각 그대로.
    부르지 않은 슬라이드는 빼고, 노래가 끝났는데 오래 떠 있는 슬라이드(기도 중)는 노래 끝 2초 뒤에 내린다."""
    from .align import Line
    lib = [l for s in (songs or fix_songs or []) for _, _, l in s.lines]   # 글자 고치기: 이번 주 곡, 없으면 보관함 전체
    lo = 0.5 if songs else 0.7                     # 보관함 전체와 견줄 땐 더 비슷해야 바꾼다
    chorus_first = []
    for s in songs or []:
        for part in getattr(s, "chorus_parts", []) or []:
            if s.sections.get(part):
                chorus_first.append(s.sections[part][0])
    ph = phrases(vt)
    out, skipped, pend = [], [], []
    for i, sl in enumerate(slides):
        inside = [(a, d) for a, d in ph if sl.start + 0.5 <= a < sl.end]
        if _unsung(slides, i, ph, segs):
            skipped.append((round(sl.start, 1), sl.text)); continue
        end = sl.end
        sung_end = max([a + d for a, d in inside if d >= 3.5], default=None)
        if sung_end is not None and end - sung_end > 15:      # 노래 끝난 뒤 기도·멘트 동안 떠 있던 슬라이드
            end = sung_end + 2.0
        ch = bool(sl.lines) and any(similar(sl.lines[0], c) > 0.8 or similar(sl.lines[0] + sl.lines[-1], c) > 0.8
                                    for c in chorus_first)
        if not one_line:
            text = "\n".join(_fix_with_library(x, lib, lo) for x in sl.lines)
            out.append(Line(round(sl.start - lead_in, 2), text, ch, "slide", round(end, 2)))
            continue
        # 한 줄씩: 첫 줄은 슬라이드가 뜰 때, 둘째 줄은 나중에 (다음 슬라이드를 알아야 해서 아래에서)
        us = [_fix_with_library(u, lib, lo) for u in units(sl)]
        pend.append((sl, us, ch, end))
    if one_line:
        out = _split_lines(pend, ph, lead_in)
    return out, {"slides": len(slides), "sung_slides": len(out), "skipped": skipped}
