"""실황 자막 읽기 단위 점검 (가사 대신 지어낸 문장만 쓴다)."""
import numpy as np

from worship_edit import slides as S


def test_rows_joins_split_box_left_to_right():
    # 한 줄이 두 상자로 갈라지고 오른쪽 상자가 살짝 위에 있어도 한 줄로, 왼쪽부터
    boxes = [(0.50, 0.20, 0.40, "넘어갑니다"), (0.52, 0.20, 0.05, "바람이 산을"), (0.10, 0.20, 0.05, "첫째 줄 글자")]
    assert S.rows(boxes) == ["첫째 줄 글자", "바람이 산을 넘어갑니다"]


def test_same_slide_needs_second_line():
    a = ["푸른 하늘 아래 걷네", "푸른 하늘 아래 걷네"]
    b = ["푸른 하늘 아래 걷네", "멈추지 않고 가겠네"]
    assert not S._same_slide(a, b)
    assert S._same_slide(b, ["푸른 하늘 아래 걷네", "멈추지 않고 가곘네"])
    assert not S._same_slide(b, ["푸른 하늘 아래 걷네"])


def test_clean_drops_noise_line():
    assert S._clean(["승 즈 소", "푸른 하늘 아래 걷네"]) == ["푸른 하늘 아래 걷네"]


def test_background_removed_text_kept():
    h, w = 40, 200
    frames = []
    for k in range(240):                       # 2분 (2fps)
        m = np.zeros((h, w), bool)
        m[5:35, 0:10] = True                   # 늘 흰 물건 (건반·아이패드)
        x = 20 + (k // 20) * 12                # 슬라이드마다 다른 자리의 글자
        m[10:30, x:x + 8] = True
        frames.append(np.packbits(m))
    bg = S.background(frames, (h, w), fps=2.0)
    m = np.unpackbits(frames[0])[: h * w].reshape(h, w).astype(bool)
    c = S.clean_mask(m, bg[0])
    assert not c[:, 0:10].any()
    assert c[10:30, 20:28].all()


def test_consensus_fixes_one_bad_read():
    sl = [S.Slide(0, 10, ["멈추지 않고 가겠네"]), S.Slide(20, 30, ["멈추지 않고 가겠네"]),
          S.Slide(40, 45, ["멈추지 않코 가겠네"]), S.Slide(50, 60, ["멈추지 않고 가겠어"])]
    S.consensus(sl)
    assert [s.lines[0] for s in sl[:3]] == ["멈추지 않고 가겠네"] * 3


def test_unsung_mid_run_is_kept():
    sl = [S.Slide(0, 10, ["가"]), S.Slide(10.5, 18, ["나"]), S.Slide(18.5, 25, ["다"]), S.Slide(60, 70, ["라"])]
    ph = [(1.0, 3.0), (19.0, 3.0)]            # 둘째 슬라이드 동안 소절 감지 못함
    assert not S._unsung(sl, 1, ph, None)
