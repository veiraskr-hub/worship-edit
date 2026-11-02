"""WorshipEdit Mac 앱 화면 (PySide6).

1 시작: XML(.fcpxmld) 끌어다 놓기
2 준비: 앵글 역할 확인, 곡 고르기(가사 보관함), 오디오 원본 확인
3 진행: 단계별 진행 표시
4 검토: 타임라인 + 가사 줄 목록(시간·가사·후렴 수정, 듣기) + 곡 구간 → 다시 만들기 / 저장 / 파이널 컷에서 열기
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import traceback

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from . import fcpxml
from .align import Line
from .analyze import Song as Section
from .fcpxml import FPS, ROLE_KO
from .library import INBOX, LIB_DIR, Song, dump, import_files, import_inbox, load_library, parse, save_song
from .pipeline import Job, Result, prepare, rebuild, run

APP = "WorshipEdit"
ROLE_COLOR = {"lead": "#e4572e", "singer": "#f3a712", "drum": "#29335c", "keys": "#669bbc", "guitar": "#7bae37"}
ROLE_CHOICES = ["lead", "singer", "drum", "keys", "guitar", "audio", "unknown"]


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def parse_mmss(s: str) -> float:
    s = s.strip()
    if ":" in s:
        m, sec = s.split(":", 1)
        return int(m) * 60 + float(sec)
    return float(s)


# ── 가사 보관함 편집 ─────────────────────────────────────────────────
TEMPLATE = """제목:
진행: V C V C
후렴: C

[V]
첫째 줄
둘째 줄

[C]
후렴 첫 줄
"""


class SongEditor(QDialog):
    def __init__(self, parent=None, song: Song | None = None):
        super().__init__(parent)
        self.setWindowTitle("곡 가사 편집")
        self.resize(560, 640)
        lay = QVBoxLayout(self)
        tip = QLabel("한 줄 = 화면에 한 번 뜨는 자막 한 줄. ‘진행’은 악보 진행표 순서(파트 이름을 띄어 쓰기), "
                     "‘후렴’은 후렴 첫 소절 규칙에 쓸 파트예요.")
        tip.setWordWrap(True)
        lay.addWidget(tip)
        self.edit = QPlainTextEdit(dump(song) if song else TEMPLATE)
        self.edit.setFont(QFont("Menlo", 13))
        lay.addWidget(self.edit)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.song: Song | None = None

    def _save(self):
        s = parse(self.edit.toPlainText())
        if not s.title or not s.sections:
            QMessageBox.warning(self, APP, "제목과 가사 파트([V] 같은 줄 아래 가사)를 넣어 주세요.")
            return
        missing = [p for p in s.order if p not in s.sections]
        if missing:
            QMessageBox.warning(self, APP, f"진행에 있는 파트가 가사에 없어요: {', '.join(missing)}")
            return
        save_song(s)
        self.song = s
        self.accept()


# ── 시작 화면 ───────────────────────────────────────────────────────
class DropPage(QWidget):
    dropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        lay = QVBoxLayout(self)
        lay.addStretch()
        t = QLabel("파이널 컷에서 내보낸 XML(.fcpxmld)을\n여기에 끌어다 놓으세요")
        t.setAlignment(Qt.AlignCenter)
        t.setFont(QFont("", 20))
        lay.addWidget(t)
        sub = QLabel("파일 → XML 내보내기 · 컷하지 않은 멀티캠 클립 하나가 올라간 프로젝트")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color: gray")
        lay.addWidget(sub)
        b = QPushButton("파일 고르기…")
        b.clicked.connect(self._pick)
        lay.addWidget(b, alignment=Qt.AlignCenter)
        lay.addStretch()
        self.setStyleSheet("DropPage { border: 2px dashed #999; border-radius: 12px; }")

    def _pick(self):
        p = QFileDialog.getExistingDirectory(self, "XML 폴더(.fcpxmld) 고르기", os.path.expanduser("~/Downloads"))
        if not p:
            p, _ = QFileDialog.getOpenFileName(self, "XML 고르기", os.path.expanduser("~/Downloads"), "FCPXML (*.fcpxml *.fcpxmld)")
        if p:
            self.dropped.emit(p)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile().rstrip("/")
            if p.endswith((".fcpxmld", ".fcpxml")):
                self.dropped.emit(p)
                return
        QMessageBox.warning(self, APP, ".fcpxmld 또는 .fcpxml 파일을 넣어 주세요.")


# ── 준비 화면 ───────────────────────────────────────────────────────
class SetupPage(QWidget):
    start = Signal()

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        self.title = QLabel()
        self.title.setFont(QFont("", 16, QFont.Bold))
        lay.addWidget(self.title)
        self.audio_info = QLabel()
        self.audio_info.setWordWrap(True)
        lay.addWidget(self.audio_info)
        row = QHBoxLayout()
        self.audio_btn = QPushButton("오디오 파일 직접 넣기…")
        self.audio_btn.clicked.connect(self._pick_audio)
        row.addWidget(self.audio_btn)
        row.addStretch()
        lay.addLayout(row)

        split = QSplitter()
        # 앵글 역할
        left = QWidget(); ll = QVBoxLayout(left)
        ll.addWidget(QLabel("카메라 앵글 역할 (자동 분류 — 다르면 바꿔 주세요)"))
        self.roles = QTableWidget(0, 2)
        self.roles.setHorizontalHeaderLabels(["앵글", "역할"])
        self.roles.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        ll.addWidget(self.roles)
        split.addWidget(left)
        # 곡 선택
        right = QWidget(); rl = QVBoxLayout(right)
        rl.addWidget(QLabel("이번 주 부른 곡 (부른 순서대로)"))
        self.chosen = QListWidget()
        self.chosen.setDragDropMode(QAbstractItemView.InternalMove)
        rl.addWidget(self.chosen)
        rb = QHBoxLayout()
        for text, fn in (("위로", lambda: self._move(-1)), ("아래로", lambda: self._move(1)), ("빼기", self._remove)):
            b = QPushButton(text); b.clicked.connect(fn); rb.addWidget(b)
        rl.addLayout(rb)
        rl.addWidget(QLabel("가사 보관함"))
        self.search = QLineEdit(); self.search.setPlaceholderText("곡 제목 검색")
        self.search.textChanged.connect(self._fill_library)
        rl.addWidget(self.search)
        self.library = QListWidget()
        self.library.itemDoubleClicked.connect(self._add_song)
        rl.addWidget(self.library)
        lb = QHBoxLayout()
        for text, fn in (("추가 ↑", self._add_selected), ("새 곡…", self._new_song), ("가사 편집…", self._edit_song),
                         ("가사 파일 가져오기…", self._import)):
            b = QPushButton(text); b.clicked.connect(fn); lb.addWidget(b)
        rl.addLayout(lb)
        split.addWidget(right)
        lay.addWidget(split, 1)

        go = QPushButton("시작")
        go.setFont(QFont("", 15, QFont.Bold))
        go.clicked.connect(self.start.emit)
        lay.addWidget(go)
        self.audio_path = ""
        self.lib: dict[str, Song] = {}
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        ps = [u.toLocalFile() for u in e.mimeData().urls() if u.toLocalFile().endswith(".txt")]
        if ps:
            self.import_paths(ps)

    def load(self, project: str, prj, roles, info):
        self.project = project
        name = os.path.basename(project)
        date = next((re.match(r"(\d{4}-\d{2}-\d{2})", n).group(1) for n in prj.angles if re.match(r"\d{4}-\d{2}-\d{2} ", n)), "")
        self.title.setText(f"{name}   ·   {date}   ·   {mmss(prj.length)}")
        if info.get("audio_source_found"):
            self.audio_info.setText(f"오디오: 원본 녹화본에서 자동으로 꺼내요 ({os.path.basename(info['audio_source'])})")
        else:
            self.audio_info.setText("⚠️ 원본 녹화본을 찾지 못했어요. 외장 드라이브를 연결하거나, 파이널 컷에서 오디오만 내보낸 파일을 넣어 주세요.\n"
                                    + (info.get("audio_source") or info.get("audio_error", "")))
        self.roles.setRowCount(0)
        for n, r in roles.items():
            i = self.roles.rowCount(); self.roles.insertRow(i)
            self.roles.setItem(i, 0, QTableWidgetItem(n))
            cb = QComboBox()
            for c in ROLE_CHOICES:
                cb.addItem(ROLE_KO.get(c, "모름" if c == "unknown" else c), c)
            cb.setCurrentIndex(ROLE_CHOICES.index(r))
            self.roles.setCellWidget(i, 1, cb)
        self._fill_library()

    def _fill_library(self):
        self.lib = load_library()
        q = self.search.text().replace(" ", "")
        self.library.clear()
        for k, s in self.lib.items():
            if q in k:
                self.library.addItem(s.title)

    def _add_song(self, item: QListWidgetItem):
        if not self.chosen.findItems(item.text(), Qt.MatchExactly):
            self.chosen.addItem(item.text())

    def _add_selected(self):
        for it in self.library.selectedItems():
            self._add_song(it)

    def _new_song(self):
        d = SongEditor(self)
        if d.exec() and d.song:
            self._fill_library()
            self.chosen.addItem(d.song.title)

    def _edit_song(self):
        it = self.library.currentItem() or self.chosen.currentItem()
        if not it:
            return
        s = self.lib.get(it.text().replace(" ", ""))
        d = SongEditor(self, s)
        if d.exec():
            self._fill_library()

    def _import(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "가사 파일(.txt) 가져오기", INBOX if os.path.isdir(INBOX) else os.path.expanduser("~/Downloads"), "가사 (*.txt)")
        if ps:
            self.import_paths(ps)

    def import_paths(self, ps):
        ok, bad = import_files(ps)
        self._fill_library()
        msg = f"{len(ok)}곡을 가사 보관함에 넣었어요." + (f"\n형식이 맞지 않아 못 넣은 파일: {', '.join(bad)}" if bad else "")
        QMessageBox.information(self, APP, msg)

    def _move(self, d):
        r = self.chosen.currentRow()
        if r < 0 or not (0 <= r + d < self.chosen.count()):
            return
        it = self.chosen.takeItem(r)
        self.chosen.insertItem(r + d, it)
        self.chosen.setCurrentRow(r + d)

    def _remove(self):
        r = self.chosen.currentRow()
        if r >= 0:
            self.chosen.takeItem(r)

    def _pick_audio(self):
        p, _ = QFileDialog.getOpenFileName(self, "오디오 파일", os.path.expanduser("~/Downloads"), "오디오 (*.mp3 *.wav *.m4a *.aif *.aiff)")
        if p:
            self.audio_path = p
            self.audio_info.setText(f"오디오: {os.path.basename(p)}")

    def roles_value(self) -> dict[str, str]:
        return {self.roles.item(i, 0).text(): self.roles.cellWidget(i, 1).currentData() for i in range(self.roles.rowCount())}

    def songs_value(self) -> list[Song]:
        return [self.lib[self.chosen.item(i).text().replace(" ", "")] for i in range(self.chosen.count())]


# ── 진행 화면 ───────────────────────────────────────────────────────
class Worker(QObject):
    progress = Signal(str, float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, job: Job):
        super().__init__()
        self.job = job

    def run(self):
        try:
            res = run(self.job, progress=lambda st, f, m: self.progress.emit(st, -1.0 if f is None else float(f), m or ""))
            self.done.emit(res)
        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class ProgressPage(QWidget):
    STAGES = ["확인", "오디오 꺼내기", "목소리 분리", "받아쓰기", "가사 정렬", "곡 구조", "컷", "완료"]

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.addStretch()
        self.label = QLabel("준비 중…")
        self.label.setFont(QFont("", 16))
        self.label.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.label)
        self.bar = QProgressBar()
        lay.addWidget(self.bar)
        self.steps = QLabel(" → ".join(self.STAGES))
        self.steps.setAlignment(Qt.AlignCenter)
        self.steps.setStyleSheet("color: gray")
        lay.addWidget(self.steps)
        lay.addStretch()

    def update(self, stage, frac, msg):
        self.label.setText(stage + (f" — {msg}" if msg else ""))
        if frac < 0:
            self.bar.setRange(0, 0)
        else:
            self.bar.setRange(0, 1000); self.bar.setValue(int(frac * 1000))
        self.steps.setText(" → ".join(f"<b>{s}</b>" if s == stage else s for s in self.STAGES))


# ── 검토 화면 ───────────────────────────────────────────────────────
class Timeline(QWidget):
    seek = Signal(float)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(120)
        self.res: Result | None = None
        self.sel = None

    def set_result(self, res: Result):
        self.res = res
        self.update()

    def _x(self, t):
        return 10 + (self.width() - 20) * t / max(1.0, self.res.prj.length)

    def paintEvent(self, e):
        if not self.res:
            return
        p = QPainter(self)
        r = self.res
        h = self.height()
        for s in r.structure.songs:                         # 곡 구간
            p.fillRect(int(self._x(s.start)), 4, int(self._x(s.end) - self._x(s.start)), 22, QColor("#cfe3f3"))
        for a, b in r.structure.ments:                       # 멘트
            p.fillRect(int(self._x(a)), 4, int(self._x(b) - self._x(a)), 22, QColor("#e5e5e5"))
        for c in r.cuts:                                     # 컷 (역할 색)
            x0, x1 = self._x(c.start / FPS), self._x(c.end / FPS)
            p.fillRect(int(x0), 32, max(1, int(x1 - x0)), 26, QColor(ROLE_COLOR.get(c.role, "#888")))
        for ln in r.lines:                                   # 가사 줄
            x = int(self._x(ln.time))
            col = QColor("#c0392b") if ln.how in ("anchor", "") else QColor("#2e7d32")
            p.setPen(QPen(col, 2 if ln is self.sel else 1))
            p.drawLine(x, 64, x, h - 18)
        p.setPen(QColor("#555"))
        for m in range(0, int(r.prj.length // 60) + 1):
            x = int(self._x(m * 60))
            p.drawText(x, h - 4, f"{m}분")
        p.end()

    def mousePressEvent(self, e):
        if self.res:
            t = (e.position().x() - 10) / (self.width() - 20) * self.res.prj.length
            self.seek.emit(max(0.0, t))


class ReviewPage(QWidget):
    rebuild_req = Signal()
    restart = Signal()

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        legend = QLabel("  ".join(f'<span style="color:{c}">■</span> {ROLE_KO[r]}' for r, c in ROLE_COLOR.items())
                        + '   가사: <span style="color:#2e7d32">|</span> 목소리에 맞춤  <span style="color:#c0392b">|</span> 확인 필요')
        lay.addWidget(legend)
        self.timeline = Timeline()
        self.timeline.seek.connect(self._seek)
        lay.addWidget(self.timeline)

        split = QSplitter()
        self.lines = QTableWidget(0, 4)
        self.lines.setHorizontalHeaderLabels(["시간", "가사", "후렴 첫줄", "상태"])
        self.lines.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.lines.cellDoubleClicked.connect(self._play_row)
        self.lines.itemSelectionChanged.connect(self._select)
        lw = QWidget(); ll = QVBoxLayout(lw)
        ll.addWidget(QLabel("가사 줄 — 시간·가사를 고칠 수 있어요. 줄을 더블클릭하면 그 부분부터 들어요."))
        ll.addWidget(self.lines)
        lb = QHBoxLayout()
        for text, fn in (("줄 추가", self._add_line), ("줄 삭제", self._del_line), ("−0.2초", lambda: self._nudge(-0.2)), ("+0.2초", lambda: self._nudge(0.2))):
            b = QPushButton(text); b.clicked.connect(fn); lb.addWidget(b)
        ll.addLayout(lb)
        split.addWidget(lw)

        sw = QWidget(); sl = QVBoxLayout(sw)
        sl.addWidget(QLabel("곡 구간"))
        self.songs = QTableWidget(0, 4)
        self.songs.setHorizontalHeaderLabels(["시작", "끝", "시작 악기", "카운트"])
        sl.addWidget(self.songs)
        sb = QHBoxLayout()
        for text, fn in (("곡 추가", self._add_song), ("곡 삭제", self._del_song)):
            b = QPushButton(text); b.clicked.connect(fn); sb.addWidget(b)
        sl.addLayout(sb)
        split.addWidget(sw)
        split.setSizes([700, 320])
        lay.addWidget(split, 1)

        bl = QHBoxLayout()
        self.rb = QPushButton("고친 내용으로 다시 만들기")
        self.rb.clicked.connect(self.rebuild_req.emit)
        bl.addWidget(self.rb)
        b = QPushButton("파이널 컷에서 열기"); b.clicked.connect(self._open_fcp); bl.addWidget(b)
        b = QPushButton("Finder에서 보기"); b.clicked.connect(self._reveal); bl.addWidget(b)
        b = QPushButton("처음으로"); b.clicked.connect(self.restart.emit); bl.addWidget(b)
        lay.addLayout(bl)
        self.res: Result | None = None
        self.player = None

    # 표시
    def show_result(self, res: Result):
        self.res = res
        rep = res.report
        ly, ct = rep.get("lyrics", {}), rep.get("cuts", {})
        need = sum(1 for l in res.lines if l.how in ("anchor", ""))
        self.summary.setText(
            f"<b>완료</b> — 컷 {ct.get('cuts')}개 · 노래 구간 인도자 {ct.get('lead_share_songs', 0) * 100:.0f}% · "
            f"가사 {len(res.lines)}줄 (목소리 시작에 맞춤 {ly.get('voice_start_ok', '?')}줄, 확인 필요 {need}줄)<br>"
            f"저장: {res.job.out}")
        self.timeline.set_result(res)
        self.lines.blockSignals(True)
        self.lines.setRowCount(0)
        for ln in res.lines:
            self._append_row(ln)
        self.lines.blockSignals(False)
        self.songs.setRowCount(0)
        for s in res.structure.songs:
            i = self.songs.rowCount(); self.songs.insertRow(i)
            self.songs.setItem(i, 0, QTableWidgetItem(mmss(s.start)))
            self.songs.setItem(i, 1, QTableWidgetItem(mmss(s.end)))
            cb = QComboBox(); cb.addItem("피아노", "keys"); cb.addItem("드럼", "drum")
            cb.setCurrentIndex(1 if s.open == "drum" else 0)
            self.songs.setCellWidget(i, 2, cb)
            self.songs.setItem(i, 3, QTableWidgetItem("" if s.countin is None else mmss(s.countin)))

    def _append_row(self, ln: Line, at: int | None = None):
        i = self.lines.rowCount() if at is None else at
        self.lines.insertRow(i)
        self.lines.setItem(i, 0, QTableWidgetItem(mmss(ln.time)))
        self.lines.setItem(i, 1, QTableWidgetItem(ln.text))
        ch = QTableWidgetItem(); ch.setFlags(ch.flags() | Qt.ItemIsUserCheckable)
        ch.setCheckState(Qt.Checked if ln.chorus else Qt.Unchecked)
        self.lines.setItem(i, 2, ch)
        st = QTableWidgetItem({"onset": "맞춤", "breath": "맞춤(숨)", "anchor": "확인 필요", "": "확인 필요"}.get(ln.how, ln.how))
        st.setFlags(st.flags() & ~Qt.ItemIsEditable)
        if ln.how in ("anchor", ""):
            st.setForeground(QColor("#c0392b"))
        self.lines.setItem(i, 3, st)

    # 편집 → Result 반영
    def collect(self) -> bool:
        lines = []
        try:
            for i in range(self.lines.rowCount()):
                t = parse_mmss(self.lines.item(i, 0).text())
                txt = self.lines.item(i, 1).text().strip()
                if not txt:
                    continue
                how = {"맞춤": "onset", "맞춤(숨)": "breath"}.get(self.lines.item(i, 3).text(), "edited")
                lines.append(Line(t, txt, self.lines.item(i, 2).checkState() == Qt.Checked, how))
            songs = []
            for i in range(self.songs.rowCount()):
                cin = self.songs.item(i, 3).text().strip() if self.songs.item(i, 3) else ""
                songs.append(Section(parse_mmss(self.songs.item(i, 0).text()), parse_mmss(self.songs.item(i, 1).text()),
                                     self.songs.cellWidget(i, 2).currentData(), parse_mmss(cin) if cin else None))
        except Exception:
            QMessageBox.warning(self, APP, "시간은 3:25.40 또는 205.4 처럼 적어 주세요.")
            return False
        for s in songs:
            if s.open == "drum" and s.countin is None:
                s.countin = max(0.0, s.start - 1.0)
        self.res.lines = lines
        self.res.structure.songs = songs
        return True

    def _add_line(self):
        r = max(0, self.lines.currentRow())
        t = parse_mmss(self.lines.item(r, 0).text()) + 2 if self.lines.rowCount() else 0
        self._append_row(Line(t, "", False, "edited"), r + 1)

    def _del_line(self):
        for r in sorted({i.row() for i in self.lines.selectedIndexes()}, reverse=True):
            self.lines.removeRow(r)

    def _nudge(self, d):
        for r in sorted({i.row() for i in self.lines.selectedIndexes()}):
            it = self.lines.item(r, 0)
            it.setText(mmss(max(0.0, parse_mmss(it.text()) + d)))

    def _add_song(self):
        i = self.songs.rowCount(); self.songs.insertRow(i)
        for c, v in ((0, "0:00.00"), (1, "0:00.00"), (3, "")):
            self.songs.setItem(i, c, QTableWidgetItem(v))
        cb = QComboBox(); cb.addItem("피아노", "keys"); cb.addItem("드럼", "drum")
        self.songs.setCellWidget(i, 2, cb)

    def _del_song(self):
        r = self.songs.currentRow()
        if r >= 0:
            self.songs.removeRow(r)

    def _select(self):
        r = self.lines.currentRow()
        if self.res and 0 <= r < len(self.res.lines):
            self.timeline.sel = self.res.lines[r]
            self.timeline.update()

    # 듣기
    def _player(self):
        if self.player is None:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
            self.player = QMediaPlayer(self); self.out = QAudioOutput(self)
            self.player.setAudioOutput(self.out)
            self.player.setSource(QUrl.fromLocalFile(self.res.audio))
        return self.player

    def _seek(self, t):
        try:
            pl = self._player(); pl.setPosition(int(max(0, t - 1) * 1000)); pl.play()
        except Exception:
            pass

    def _play_row(self, r, c):
        if c == 0 or c == 1:
            try:
                self._seek(parse_mmss(self.lines.item(r, 0).text()))
            except Exception:
                pass

    def _open_fcp(self):
        subprocess.run(["open", "-a", "Final Cut Pro", self.res.job.out])

    def _reveal(self):
        subprocess.run(["open", "-R", self.res.job.out])


# ── 메인 창 ─────────────────────────────────────────────────────────
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WorshipEdit — 찬양 실황 컷·가사 자동 편집")
        self.resize(1200, 780)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.drop, self.setup, self.prog, self.review = DropPage(), SetupPage(), ProgressPage(), ReviewPage()
        for w in (self.drop, self.setup, self.prog, self.review):
            self.stack.addWidget(w)
        self.drop.dropped.connect(self.open_project)
        self.setup.start.connect(self.start)
        self.review.rebuild_req.connect(self.do_rebuild)
        self.review.restart.connect(lambda: self.stack.setCurrentWidget(self.drop))
        self.thread = None

    def open_project(self, path: str):
        try:
            prj, roles, info = prepare(Job(path, [], ""))
        except Exception as e:
            QMessageBox.critical(self, APP, str(e))
            return
        self.project = path
        self.setup.load(path, prj, roles, info)
        self.stack.setCurrentWidget(self.setup)

    def start(self):
        songs = self.setup.songs_value()
        if not songs:
            QMessageBox.warning(self, APP, "이번 주 부른 곡을 가사 보관함에서 골라 주세요.")
            return
        roles = self.setup.roles_value()
        if any(r == "unknown" for r in roles.values()):
            QMessageBox.warning(self, APP, "역할이 ‘모름’인 앵글이 있어요. 역할을 정해 주세요.")
            return
        base = os.path.splitext(self.project.rstrip("/"))[0]
        out = base + "_자동편집.fcpxml"
        job = Job(self.project, songs, out, audio=self.setup.audio_path, roles=roles,
                  name=os.path.basename(base) + "_자동편집")
        self.stack.setCurrentWidget(self.prog)
        self.thread = QThread()
        self.worker = Worker(job)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.prog.update)
        self.worker.done.connect(self.finished)
        self.worker.failed.connect(self.failed)
        self.worker.done.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.start()

    def finished(self, res: Result):
        self.review.show_result(res)
        self.stack.setCurrentWidget(self.review)

    def failed(self, msg: str):
        QMessageBox.critical(self, APP, msg[:2000])
        self.stack.setCurrentWidget(self.setup)

    def do_rebuild(self):
        res = self.review.res
        if not res or not self.review.collect():
            return
        try:
            rebuild(res)
        except Exception as e:
            QMessageBox.critical(self, APP, str(e))
            return
        self.review.show_result(res)
        QMessageBox.information(self, APP, "다시 만들었어요. 파이널 컷에서 열어 확인해 주세요.")


def selftest() -> int:
    """빌드된 앱 점검: 무거운 모듈이 다 들어갔는지 확인 (GitHub 빌드에서 실행)."""
    import librosa, numpy, onnxruntime, soundfile  # noqa: F401
    from .media import ffmpeg_exe
    ok = {"ffmpeg": os.path.exists(ffmpeg_exe()), "onnx": onnxruntime.get_available_providers()}
    try:
        import pywhispercpp.model  # noqa: F401
        ok["whisper"] = True
    except Exception as e:
        ok["whisper"] = f"없음: {e}"
    app = QApplication.instance() or QApplication(sys.argv)
    Main()
    print("selftest", ok)
    return 0 if ok["ffmpeg"] and ok["whisper"] is True else 1


def main():
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    app = QApplication(sys.argv)
    app.setApplicationName(APP)
    os.makedirs(LIB_DIR, exist_ok=True)
    try:
        import_inbox()
    except Exception:
        pass
    w = Main()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
