from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Callable

from PySide6.QtCore import Qt, Signal, QTimer, QUrl, QEvent
from PySide6.QtGui import QTextCursor, QTextCharFormat, QColor
from PySide6.QtWidgets import (
    QTextEdit, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit as QTextEditWidget,
    QPushButton, QToolButton, QSlider, QStyle, QWidget, QScrollArea, QFormLayout, QMessageBox
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput


_TS_BRACKET_RE = re.compile(
    # Support both '-' and '–' between timestamps, and allow ',' as decimal separator.
    r"\[(?P<a>[0-9:.,]+)\s*[-–]\s*(?P<b>[0-9:.,]+)\]"
)

# Speaker label: allow Unicode letters (\w), spaces and a few common punctuation marks.
# We exclude ':' and brackets to avoid consuming timestamps.
_SPK_PREFIX_RE = re.compile(
    r"^(?P<spk>[^:\[\]]{1,64}):\s*(?P<rest>.*)$"
)

_SPK_AFTER_TS_RE = re.compile(
    r"^\s*(?P<spk>[^:\[\]]{1,64}):\s*(?P<txt>.*)$"
)


def _parse_time_to_seconds(s: str) -> Optional[float]:
    s = (s or "").strip()
    s = s.replace(',', '.')
    if not s:
        return None
    # float seconds format
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        try:
            return float(s)
        except Exception:
            return None
    # HH:MM:SS(.ms)
    m = re.fullmatch(r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<sec>\d{2})(?:\.(?P<ms>\d{1,3}))?", s)
    if not m:
        return None
    h = int(m.group("h"))
    mi = int(m.group("m"))
    sec = int(m.group("sec"))
    ms_s = m.group("ms") or "0"
    ms = int(ms_s.ljust(3, "0")[:3])
    return float(h * 3600 + mi * 60 + sec) + ms / 1000.0


@dataclass
class Segment:
    block_number: int
    ts_bracket: str
    start_s: float
    end_s: float
    speaker: str  # may be "" if not present
    text: str
    speaker_position: str  # "after_ts" or "before_ts" or "none"


def parse_segment_line(line: str, block_number: int = 0) -> Optional[Segment]:
    """
    Supported common formats:
      1) [00:00:01.230 - 00:00:03.900] some text
      2) [12.34-15.67] SPEAKER_00: some text
      3) SPEAKER_00: [00:00:01.230 - 00:00:03.900] some text
    """
    raw = (line or "").rstrip("\n")
    if not raw.strip():
        return None

    speaker = ""
    speaker_position = "none"
    rest = raw.strip()

    # Detect speaker before timestamp (format 3)
    m0 = _SPK_PREFIX_RE.match(rest)
    if m0 and _TS_BRACKET_RE.search(m0.group("rest") or ""):
        speaker = (m0.group("spk") or "").strip()
        speaker_position = "before_ts"
        rest = (m0.group("rest") or "").strip()

    m = _TS_BRACKET_RE.search(rest)
    if not m:
        return None

    ts_bracket = m.group(0)
    a = _parse_time_to_seconds(m.group("a"))
    b = _parse_time_to_seconds(m.group("b"))
    if a is None or b is None:
        return None

    after = (rest[m.end():] or "").strip()
    before = (rest[:m.start()] or "").strip()

    # If speaker not set before timestamp, try speaker after timestamp (format 2)
    if not speaker and after:
        m1 = _SPK_AFTER_TS_RE.match(after)
        if m1:
            speaker = (m1.group("spk") or "").strip()
            after = (m1.group("txt") or "").strip()
            speaker_position = "after_ts"

    # Plain text case (format 1)
    text = after.strip() if after is not None else ""

    # If we still have text empty, but there is content before timestamp, keep it
    # (rare case) – don't drop information.
    if not text and before and speaker_position == "none":
        text = before

    if b < a:
        a, b = b, a

    return Segment(
        block_number=block_number,
        ts_bracket=ts_bracket,
        start_s=float(a),
        end_s=float(b),
        speaker=speaker,
        text=text,
        speaker_position=speaker_position,
    )



class SegmentTextEdit(QTextEdit):
    segmentHovered = Signal(object)        # Segment | None
    segmentDoubleClicked = Signal(object)  # Segment

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self._hover_block: Optional[int] = None
        self._hover_seg: Optional[Segment] = None

        # Base selections are used for features like "color by speaker".
        # Hover selection is layered on top.
        self._base_selections: list[QTextEdit.ExtraSelection] = []

        self._color_by_speaker: bool = False
        self._speaker_color_map: dict[str, object] = {}

        # Recompute speaker colors when text changes (only if enabled)
        self.textChanged.connect(self._maybe_recompute_speaker_coloring)

    # ---------- Speaker coloring (diarization view) ----------
    def enable_speaker_coloring(self, enabled: bool) -> None:
        self._color_by_speaker = bool(enabled)
        self._recompute_speaker_coloring()

    def _maybe_recompute_speaker_coloring(self) -> None:
        if self._color_by_speaker:
            self._recompute_speaker_coloring()

    def _blend(self, c1, c2, alpha: float):
        a = max(0.0, min(1.0, float(alpha)))
        r = int(round(c1.red() * (1 - a) + c2.red() * a))
        g = int(round(c1.green() * (1 - a) + c2.green() * a))
        b = int(round(c1.blue() * (1 - a) + c2.blue() * a))
        return QColor(r, g, b)

    def _rel_luminance(self, c: QColor) -> float:
        """Relative luminance for sRGB (0..1)."""
        r = c.red() / 255.0
        g = c.green() / 255.0
        b = c.blue() / 255.0

        def _lin(u: float) -> float:
            return u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4

        return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)

    def _best_text_for_bg(self, bg: QColor) -> QColor:
        """Pick black/white text for best contrast on the given background."""
        lum = self._rel_luminance(bg)

        def _contrast(l1: float, l2: float) -> float:
            hi = max(l1, l2)
            lo = min(l1, l2)
            return (hi + 0.05) / (lo + 0.05)

        c_black = _contrast(lum, 0.0)
        c_white = _contrast(lum, 1.0)
        return QColor(0, 0, 0) if c_black >= c_white else QColor(255, 255, 255)

    def _speaker_tint(self, index: int):
        """Return a subtle per-speaker background that stays readable in light/dark themes."""
        base = self.palette().base().color()
        base_v = base.toHsv().value()
        is_dark = base_v < 128

        hi = self.palette().highlight().color().toHsv()
        hue = hi.hue() if hi.hue() >= 0 else 210
        h = (hue + (index * 37)) % 360
        sat = max(35, min(110, hi.saturation() if hi.saturation() >= 0 else 85))

        # Keep tints near the current base so dark themes don't produce "bright" blocks.
        if is_dark:
            val = min(255, base_v + 28)
            alpha = 0.22
        else:
            val = max(0, base_v - 14)
            alpha = 0.14

        tint = QColor.fromHsv(int(h), int(sat), int(val))
        return self._blend(base, tint, alpha)

    def _recompute_speaker_coloring(self) -> None:
        # Build stable mapping: speaker -> tint
        speakers: list[str] = []
        for i, line in enumerate((self.toPlainText() or "").splitlines()):
            seg = parse_segment_line(line, i)
            if seg and seg.speaker:
                spk = seg.speaker.strip()
                if spk and spk not in speakers:
                    speakers.append(spk)

        self._speaker_color_map = {spk: self._speaker_tint(idx) for idx, spk in enumerate(speakers)}

        # Prepare base selections (one per block)
        sels: list[QTextEdit.ExtraSelection] = []
        doc = self.document()
        for bn in range(doc.blockCount()):
            block = doc.findBlockByNumber(bn)
            if not block.isValid():
                continue
            line = block.text()
            seg = parse_segment_line(line, bn)
            if not seg or not seg.speaker:
                continue
            color = self._speaker_color_map.get(seg.speaker.strip())
            if color is None:
                continue
            cur = QTextCursor(block)
            cur.select(QTextCursor.BlockUnderCursor)
            fmt = QTextCharFormat()
            fmt.setBackground(color)
            fmt.setForeground(self._best_text_for_bg(color))
            sel = QTextEdit.ExtraSelection()
            sel.cursor = cur
            sel.format = fmt
            sels.append(sel)

        self._base_selections = sels
        self._apply_selections()

    # ---------- Hover highlight layering ----------
    def _apply_selections(self) -> None:
        # Apply base selections plus hovered selection (if any)
        sels = list(self._base_selections)
        if self._hover_block is not None:
            doc = self.document()
            block = doc.findBlockByNumber(self._hover_block)
            if block.isValid():
                cur = QTextCursor(block)
                cur.select(QTextCursor.BlockUnderCursor)
                fmt = QTextCharFormat()
                hover_brush = self.palette().alternateBase()
                fmt.setBackground(hover_brush)
                try:
                    bg = hover_brush.color()
                except Exception:
                    bg = self.palette().alternateBase().color()
                fmt.setForeground(self._best_text_for_bg(bg))
                hover_sel = QTextEdit.ExtraSelection()
                hover_sel.cursor = cur
                hover_sel.format = fmt
                sels.append(hover_sel)
        self.setExtraSelections(sels)

    def changeEvent(self, event) -> None:  # type: ignore[override]
        # When theme/palette changes, recompute selections so text stays readable
        # (especially important when switching to a dark skin).
        try:
            et = event.type()
        except Exception:
            et = None

        if et in (QEvent.PaletteChange, QEvent.StyleChange):
            if self._color_by_speaker:
                self._recompute_speaker_coloring()
            else:
                self._apply_selections()

        return super().changeEvent(event)

    def _highlight_block(self, block_number: Optional[int]) -> None:
        self._hover_block = block_number
        self._apply_selections()

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._hover_block = None
        self._hover_seg = None
        self._apply_selections()
        self.segmentHovered.emit(None)
        return super().leaveEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        cur = self.cursorForPosition(event.pos())
        block = cur.block()
        bn = block.blockNumber()
        line = block.text()
        seg = parse_segment_line(line, bn)
        if seg is None:
            if self._hover_block is not None:
                self._hover_block = None
                self._hover_seg = None
                self._apply_selections()
                self.segmentHovered.emit(None)
            return super().mouseMoveEvent(event)

        if self._hover_block != bn:
            self._hover_block = bn
            self._hover_seg = seg
            self._apply_selections()
            self.segmentHovered.emit(seg)

        return super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        cur = self.cursorForPosition(event.pos())
        block = cur.block()
        bn = block.blockNumber()
        seg = parse_segment_line(block.text(), bn)
        if seg is not None:
            self.segmentDoubleClicked.emit(seg)
        return super().mouseDoubleClickEvent(event)

    def replace_block_text(self, block_number: int, new_line: str) -> None:
        """Replace exactly one QTextDocument block without removing the paragraph separator.

        Using QTextCursor.BlockUnderCursor may include the paragraph separator, which can merge
        the edited line with the next one and break segment->audio mapping.
        """
        doc = self.document()
        block = doc.findBlockByNumber(block_number)
        if not block.isValid():
            return

        # Keep segments one-line: newlines would create extra blocks and break playback mapping.
        safe_line = (new_line or "").replace("\r", "").replace("\n", " ").strip()

        cur = QTextCursor(doc)
        cur.setPosition(block.position())
        cur.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)  # exclude separator
        cur.insertText(safe_line)



class SegmentEditDialog(QDialog):

    """
    Popup editor with basic audio controls and inline correction of the selected segment.
    """
    def __init__(
        self,
        parent: QWidget,
        audio_path: str,
        seg: Segment,
        current_line: str,
        speaker_name_map: Dict[str, str] | None = None,
        t: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.t = t or (lambda k: k)
        self.setWindowTitle(self.t("seg_edit_title"))
        self.setModal(True)
        self.resize(720, 420)

        self.audio_path = audio_path
        self.seg = seg
        self.current_line = current_line
        self.speaker_name_map = speaker_name_map or {}

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        if self.audio_path:
            self.player.setSource(QUrl.fromLocalFile(self.audio_path))

        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._sync_slider)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Segment meta
        meta = QLabel(self)
        spk_disp = self.seg.speaker or self.t("seg_no_speaker")
        if self.seg.speaker and self.seg.speaker in self.speaker_name_map:
            spk_disp = f"{spk_disp}  →  {self.speaker_name_map[self.seg.speaker]}"
        meta.setText(self.t("seg_meta").format(ts=self.seg.ts_bracket, spk=spk_disp))
        meta.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(meta)

        # Controls
        controls = QHBoxLayout()
        self.btn_back = QToolButton(self)
        self.btn_back.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekBackward))
        self.btn_play = QToolButton(self)
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_pause = QToolButton(self)
        self.btn_pause.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.btn_stop = QToolButton(self)
        self.btn_stop.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.btn_fwd = QToolButton(self)
        self.btn_fwd.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekForward))

        self.btn_back.clicked.connect(lambda: self._seek_rel(-2000))
        self.btn_fwd.clicked.connect(lambda: self._seek_rel(+2000))
        self.btn_play.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self.player.pause)
        self.btn_stop.clicked.connect(self.stop)

        controls.addWidget(self.btn_back)
        controls.addWidget(self.btn_play)
        controls.addWidget(self.btn_pause)
        controls.addWidget(self.btn_stop)
        controls.addWidget(self.btn_fwd)
        controls.addStretch(1)

        self.pos_label = QLabel(self)
        self.pos_label.setText("00:00 / 00:00")
        controls.addWidget(self.pos_label)

        root.addLayout(controls)

        # Slider within the fragment
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, max(1, int((self.seg.end_s - self.seg.start_s) * 1000)))
        self.slider.valueChanged.connect(self._on_slider)
        root.addWidget(self.slider)

        # Edit fields
        form = QFormLayout()
        self.spk_edit = QLineEdit(self)
        self.spk_edit.setText(self.seg.speaker or "")
        self.txt_edit = QTextEditWidget(self)
        self.txt_edit.setPlainText(self.seg.text or "")
        form.addRow(self.t("seg_speaker_label"), self.spk_edit)
        form.addRow("Tekst:", self.txt_edit)
        root.addLayout(form)

        # Buttons
        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_cancel = QPushButton(self.t("btn_cancel"), self)
        self.btn_apply = QPushButton(self.t("btn_apply"), self)
        self.btn_apply.setDefault(True)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_apply.clicked.connect(self.accept)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_apply)
        root.addLayout(btns)

        # Start positioned at segment start
        self.stop()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            self.player.stop()
        except Exception:
            pass
        return super().closeEvent(event)

    def _fmt_mmss(self, ms: int) -> str:
        s = max(0, ms // 1000)
        m, s = divmod(s, 60)
        return f"{m:02d}:{s:02d}"

    def _sync_slider(self) -> None:
        pos = int(self.player.position())
        start_ms = int(self.seg.start_s * 1000)
        end_ms = int(self.seg.end_s * 1000)
        if pos < start_ms:
            pos = start_ms
        if pos > end_ms:
            pos = end_ms

        rel = pos - start_ms
        self.slider.blockSignals(True)
        self.slider.setValue(rel)
        self.slider.blockSignals(False)
        self.pos_label.setText(f"{self._fmt_mmss(rel)} / {self._fmt_mmss(end_ms - start_ms)}")

        # auto-stop at end of segment
        if pos >= end_ms and self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()

    def _on_slider(self, value: int) -> None:
        start_ms = int(self.seg.start_s * 1000)
        self.player.setPosition(start_ms + int(value))

    def _seek_rel(self, delta_ms: int) -> None:
        self.player.setPosition(max(0, self.player.position() + int(delta_ms)))

    def play(self) -> None:
        # Ensure we start within fragment window
        start_ms = int(self.seg.start_s * 1000)
        end_ms = int(self.seg.end_s * 1000)
        pos = int(self.player.position())
        if pos < start_ms or pos > end_ms:
            self.player.setPosition(start_ms)
        self.player.play()
        self._timer.start()

    def stop(self) -> None:
        try:
            self.player.stop()
        except Exception:
            pass
        start_ms = int(self.seg.start_s * 1000)
        self.player.setPosition(start_ms)
        self._timer.start()
    def build_new_line(self) -> str:
        speaker = (self.spk_edit.text() or "").strip()
        text = (self.txt_edit.toPlainText() or "").strip()
        # Keep it as a single line to preserve the 1-line-per-segment convention.
        text = re.sub(r"\s*\n\s*", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        ts = self.seg.ts_bracket

        # Reconstruct respecting speaker position
        if self.seg.speaker_position == "before_ts":
            if speaker:
                return f"{speaker}: {ts} {text}".rstrip()
            return f"{ts} {text}".rstrip()

        if self.seg.speaker_position == "after_ts":
            if speaker:
                return f"{ts} {speaker}: {text}".rstrip()
            return f"{ts} {text}".rstrip()

        # none
        if speaker:
            # If previously no speaker, put it after timestamp for consistency
            return f"{ts} {speaker}: {text}".rstrip()
        return f"{ts} {text}".rstrip()

    def reject(self) -> None:  # type: ignore[override]
        try:
            self._timer.stop()
        except Exception:
            pass
        try:
            self.player.stop()
        except Exception:
            pass
        return super().reject()

    def accept(self) -> None:  # type: ignore[override]
        try:
            self._timer.stop()
        except Exception:
            pass
        try:
            self.player.stop()
        except Exception:
            pass
        return super().accept()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            self._timer.stop()
        except Exception:
            pass
        try:
            self.player.stop()
        except Exception:
            pass
        return super().closeEvent(event)




class SpeakerNamesPanel(QWidget):
    mappingApplied = Signal(object)  # Dict[str,str]

    """
    Shared panel placed under transcription/diarization panes.
    Allows naming/renaming speakers across both views at once.
    """
    def __init__(self, parent: QWidget, left: SegmentTextEdit, right: SegmentTextEdit, t: Callable[[str], str] | None = None) -> None:
        super().__init__(parent)
        self.t = t or (lambda k: k)
        self.left = left
        self.right = right

        self._edit_fields: Dict[str, QLineEdit] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Title
        top = QHBoxLayout()
        self.title = QLabel(self.t("speaker_panel_title"), self)
        top.addWidget(self.title)
        top.addStretch(1)
        root.addLayout(top)

        self.hint = QLabel(self.t("speaker_names_hint"), self)
        self.hint.setWordWrap(True)
        root.addWidget(self.hint)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.inner = QWidget(self.scroll)
        self.form = QFormLayout(self.inner)
        self.form.setContentsMargins(4, 4, 4, 4)
        self.form.setSpacing(6)
        self.scroll.setWidget(self.inner)
        root.addWidget(self.scroll, 1)

        # Control buttons under the speaker list (more standard UX)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.btn_refresh = QPushButton(self.t("speaker_panel_refresh"), self)
        self.btn_apply = QPushButton(self.t("btn_apply"), self)
        bottom.addWidget(self.btn_refresh)
        bottom.addWidget(self.btn_apply)
        root.addLayout(bottom)


        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_apply.clicked.connect(self.apply)

        self.refresh()


    def set_translator(self, t: Callable[[str], str]) -> None:
        self.t = t
        self.retranslate()
        # rebuild rows so empty-state message and hint match the new language
        self.refresh()

    def retranslate(self) -> None:
        self.title.setText(self.t("speaker_panel_title"))
        self.btn_refresh.setText(self.t("speaker_panel_refresh"))
        self.btn_apply.setText(self.t("btn_apply"))
        self.hint.setText(self.t("speaker_names_hint"))

    def _collect_speakers_from_text(self, text: str) -> List[str]:
        speakers: set[str] = set()
        for i, line in enumerate((text or "").splitlines()):
            seg = parse_segment_line(line, i)
            if seg and seg.speaker:
                speakers.add(seg.speaker)
            else:
                # also support lines without timestamps: "SPEAKER_00: text"
                m = _SPK_PREFIX_RE.match(line.strip())
                if m:
                    speakers.add((m.group("spk") or "").strip())
        return sorted(speakers)

    def refresh(self) -> None:
        # clear current fields
        while self.form.rowCount():
            self.form.removeRow(0)
        self._edit_fields.clear()

        speakers = set(self._collect_speakers_from_text(self.left.toPlainText()))
        speakers |= set(self._collect_speakers_from_text(self.right.toPlainText()))
        speakers_list = sorted(speakers)

        if not speakers_list:
            self.form.addRow(QLabel(self.t("speaker_panel_no_speakers")), QLabel(""))
            return

        for spk in speakers_list:
            le = QLineEdit(self)
            le.setPlaceholderText("np. Jan / Justyna / Tomek")
            self._edit_fields[spk] = le
            self.form.addRow(QLabel(spk, self), le)

    def apply(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for old, le in self._edit_fields.items():
            new = (le.text() or "").strip()
            if new and new != old:
                mapping[old] = new

        if not mapping:
            QMessageBox.information(self, self.t("speaker_panel_title"), self.t("speaker_panel_no_changes"))
            return {}

        def _apply_to_text(text: str) -> str:
            out = text
            for old, new in mapping.items():
                # Replace only speaker labels (line-start or after timestamp)
                out = re.sub(
                    rf"(^\s*){re.escape(old)}(?=:)",
                    lambda m: (m.group(1) or "") + new,
                    out,
                    flags=re.MULTILINE,
                )
                out = re.sub(
                    rf"(\]\s*){re.escape(old)}(?=:)",
                    lambda m: (m.group(1) or "") + new,
                    out,
                )
            return out

        self.left.setPlainText(_apply_to_text(self.left.toPlainText()))
        self.right.setPlainText(_apply_to_text(self.right.toPlainText()))
        self.refresh()
        self.mappingApplied.emit(mapping)
        return mapping
