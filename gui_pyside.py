from __future__ import annotations

import datetime
import os
import re
import wave
from zoneinfo import ZoneInfo
from typing import Optional

from PySide6.QtCore import Qt, Slot, QUrl, QTimer
from PySide6.QtWidgets import (
    QWidget, QMainWindow, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QPushButton,
    QTextEdit, QTextBrowser, QComboBox, QSpinBox, QStatusBar, QMenuBar, QMenu,
    QTabWidget, QGroupBox, QFormLayout, QLineEdit, QDialog
)

from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from ui.report_dialog import ReportDialog
from backend.tasks import BackgroundTask, TaskRunner
from backend.legacy_adapter import (
    whisper_transcribe_safe,
    diarize_text_simple,
    diarize_voice_whisper_pyannote_safe,
)
from backend.settings import Settings, APP_NAME, APP_VERSION, AUTHOR_EMAIL
from backend.settings_store import load_settings, save_settings

from generators import generate_txt_report, generate_pdf_report, generate_html_report

from ui.theme import THEMES, apply_theme
from ui.i18n import tr
from ui.segments import SegmentTextEdit, SegmentEditDialog, SpeakerNamesPanel, Segment, parse_segment_line


class MainWindow(QMainWindow):
    def __init__(self, app=None) -> None:
        super().__init__()
        self._app = app  # QApplication (optional, for theme apply)
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1250, 820)

        self.settings: Settings = load_settings()
        self.task_runner = TaskRunner()
        self.audio_path: Optional[str] = None

        self._build_ui()
        self._init_audio_preview()
        self._apply_settings()
        self._update_status()
        self.log("Application started.")


    # ---------- Audio preview / segment playback (hover + popup editor) ----------
    def _init_audio_preview(self) -> None:
        """Initialize a lightweight player used for hover-preview of segments."""
        self.preview_player = QMediaPlayer(self)
        self.preview_audio_output = QAudioOutput(self)
        self.preview_player.setAudioOutput(self.preview_audio_output)

        # Stop timer for hover preview
        self._preview_stop_timer = QTimer(self)
        self._preview_stop_timer.setSingleShot(True)
        self._preview_stop_timer.timeout.connect(self.preview_player.stop)

        # Debounce hover events (avoid accidental starts when moving mouse quickly)
        self._hover_debounce = QTimer(self)
        self._hover_debounce.setSingleShot(True)
        self._hover_debounce.setInterval(180)
        self._hover_debounce.timeout.connect(self._play_pending_hover)
        self._pending_hover_seg: Segment | None = None

        self._set_preview_source()

    def _set_preview_source(self) -> None:
        try:
            if self.audio_path:
                self.preview_player.setSource(QUrl.fromLocalFile(self.audio_path))
            else:
                self.preview_player.setSource(QUrl())
        except Exception:
            # If multimedia backend is missing, we keep UI running; user will see no preview.
            pass

    def _play_pending_hover(self) -> None:
        seg = self._pending_hover_seg
        self._pending_hover_seg = None
        if not seg or not self.audio_path:
            return

        # Play a short preview (up to 2.5s) from the segment start
        start_ms = int(max(0.0, seg.start_s) * 1000)
        max_len_ms = 2500
        seg_len_ms = int(max(0.0, (seg.end_s - seg.start_s)) * 1000)
        play_len_ms = max(250, min(max_len_ms, seg_len_ms if seg_len_ms > 0 else max_len_ms))

        try:
            self.preview_player.setPosition(start_ms)
            self.preview_player.play()
            self._preview_stop_timer.start(play_len_ms)
        except Exception:
            pass

    def _on_segment_hovered(self, seg: Segment | None, editor: SegmentTextEdit) -> None:
        # SegmentTextEdit already highlights the hovered line.
        if seg is None:
            self._hover_debounce.stop()
            self._pending_hover_seg = None
            try:
                self.preview_player.stop()
            except Exception:
                pass
            return

        if not self.audio_path:
            return

        self._pending_hover_seg = seg
        self._hover_debounce.start()

    def _on_segment_double_clicked(self, seg: Segment, editor: SegmentTextEdit) -> None:
        if not self.audio_path:
            QMessageBox.information(self, "Audio", "Najpierw wczytaj plik audio, aby odsłuchiwać fragmenty.")
            return

        block = editor.document().findBlockByNumber(seg.block_number)
        current_line = block.text() if block.isValid() else ""

        dlg = SegmentEditDialog(
            parent=self,
            audio_path=self.audio_path,
            seg=seg,
            current_line=current_line,
            speaker_name_map=getattr(self, "speaker_name_map", {}),
            t=self.t,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        new_line = dlg.build_new_line()

        editor.replace_block_text(seg.block_number, new_line)

        try:
            self.speaker_panel.refresh()
        except Exception:
            pass

    def _on_speaker_mapping_applied(self, mapping: object) -> None:
        """Keep a local copy of the last mapping (used by the popup meta label)."""
        if not isinstance(mapping, dict):
            return
        if not hasattr(self, "speaker_name_map") or self.speaker_name_map is None:
            self.speaker_name_map = {}
        # store mapping (old -> new). This is mostly informational.
        try:
            self.speaker_name_map.update({str(k): str(v) for k, v in mapping.items()})
        except Exception:
            pass
        try:
            self.log(f"Speaker mapping applied: {mapping}")
        except Exception:
            pass


    def _render_info_markdown(self) -> None:
        """
        Load ui/Info_{lang}.md or ui/Info.md and render it in the Info tab (Markdown).
        Placeholders supported: {{APP_NAME}}, {{APP_VERSION}}.
        Relative links/images work if they are inside ./ui (e.g. assets/logo.png).
        """
        lang = (self.settings.ui_language or "pl").strip() or "pl"

        root_dir = os.path.dirname(os.path.abspath(__file__))
        ui_dir = os.path.join(root_dir, "ui")

        candidates = [
            os.path.join(ui_dir, f"Info_{lang}.md"),
            os.path.join(ui_dir, "Info.md"),
        ]

        info_path = next((p for p in candidates if os.path.isfile(p)), None)

        if not info_path:
            # Built-in fallback (Markdown)
            md = (
                f"# {{APP_NAME}}\n\n"
                f"**Version:** {{APP_VERSION}}\n\n"
                "**Author:** pawlict\n\n"
                "**License:** AISTATElight License v1.2 (Source-Available) — AS IS\n\n"
                "## Notes\n"
                "- Whisper models auto-download on first use.\n"
                "- Voice diarization uses pyannote and requires a HF token + accepted model terms.\n"
            )
        else:
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    md = f.read()
            except Exception as e:
                self.info_text.setPlainText(f"Failed to load Info markdown:\n{e}")
                return

        md = md.replace("{{APP_NAME}}", APP_NAME).replace("{{APP_VERSION}}", APP_VERSION)

        # Base URL so relative images/links resolve from ./ui
        try:
            self.info_text.document().setBaseUrl(QUrl.fromLocalFile(ui_dir + os.sep))
        except Exception:
            pass

        try:
            self.info_text.setMarkdown(md)
        except Exception:
            # fallback if setMarkdown is unavailable
            self.info_text.setPlainText(md)

    # ---------- i18n helpers ----------
    def t(self, key: str) -> str:
        lang = (self.settings.ui_language or "pl").strip() or "pl"
        return tr(lang, key)

    # ---------- UI ----------
    def _build_ui(self) -> None:
        self._build_menu()  # File menu + File tab

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.tabs = QTabWidget(self)
        root.addWidget(self.tabs, 1)
        # --- File tab (aligned with other tabs) ---
        self.file_tab = QWidget(self)
        file_layout = QVBoxLayout(self.file_tab)
        file_layout.setContentsMargins(8, 8, 8, 8)
        file_layout.setSpacing(8)

        self.gb_file = QGroupBox(self.file_tab)
        fl_file = QFormLayout(self.gb_file)

        self.btn_open_transcript = QPushButton(self.file_tab)
        self.btn_open_transcript.clicked.connect(self.on_open_text)
        fl_file.addRow(self.btn_open_transcript)

        self.btn_save_input = QPushButton(self.file_tab)
        self.btn_save_input.clicked.connect(self.on_save_input)
        fl_file.addRow(self.btn_save_input)

        self.btn_save_output = QPushButton(self.file_tab)
        self.btn_save_output.clicked.connect(self.on_save_output)
        fl_file.addRow(self.btn_save_output)

        self.btn_save_logs2 = QPushButton(self.file_tab)
        self.btn_save_logs2.clicked.connect(self.on_save_logs)
        fl_file.addRow(self.btn_save_logs2)

        self.btn_quit = QPushButton(self.file_tab)
        self.btn_quit.clicked.connect(self.close)
        fl_file.addRow(self.btn_quit)

        file_layout.addWidget(self.gb_file)
        file_layout.addStretch(1)

        self.tabs.addTab(self.file_tab, self.t("tab_file"))

        # --- Home tab ---
        self.home = QWidget(self)
        home_layout = QVBoxLayout(self.home)
        home_layout.setContentsMargins(8, 8, 8, 8)
        home_layout.setSpacing(8)

        # Ribbon row (Word-like groups)
        ribbon = QHBoxLayout()
        ribbon.setSpacing(10)
        home_layout.addLayout(ribbon)

        # Audio group
        self.gb_audio = QGroupBox(self.home)
        fl_audio = QFormLayout(self.gb_audio)
        self.lang_combo = QComboBox(self.home)
        self.lang_combo.addItems(["auto", "pl", "en", "de", "fr", "es", "ru", "uk", "ja"])
        self.lang_combo.setEditable(True)
        self.btn_load_audio = QPushButton(self.home)
        self.btn_load_audio.clicked.connect(self.on_load_audio_clicked)
        self.lbl_audio = QLabel(self.home)
        self.lbl_audio.setTextInteractionFlags(Qt.TextSelectableByMouse)
        fl_audio.addRow(self._mk_lbl("audio_lang"), self.lang_combo)
        fl_audio.addRow(self.btn_load_audio, self.lbl_audio)
        ribbon.addWidget(self.gb_audio, 2)

        # Whisper group
        self.gb_trans = QGroupBox(self.home)
        fl_trans = QFormLayout(self.gb_trans)
        self.whisper_model_combo = QComboBox(self.home)
        self.whisper_model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.btn_transcribe = QPushButton(self.home)
        self.btn_transcribe.clicked.connect(self.on_transcribe_clicked)
        fl_trans.addRow(self._mk_lbl("whisper_model"), self.whisper_model_combo)
        fl_trans.addRow(self.btn_transcribe)
        ribbon.addWidget(self.gb_trans, 2)

        # Voice diarization group (pyannote)
        self.gb_voice = QGroupBox(self.home)
        vbox_voice = QVBoxLayout(self.gb_voice)
        self.btn_voice_diar = QPushButton(self.home)
        self.btn_voice_diar.clicked.connect(self.on_voice_diarize_clicked)
        self.lbl_hf_status = QLabel(self.home)
        self.lbl_hf_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        vbox_voice.addWidget(self.btn_voice_diar)
        vbox_voice.addWidget(self.lbl_hf_status)
        vbox_voice.addStretch(1)
        ribbon.addWidget(self.gb_voice, 2)

        # Text diarization group
        self.gb_text = QGroupBox(self.home)
        fl_text = QFormLayout(self.gb_text)
        self.method_combo = QComboBox(self.home)
        self.method_combo.addItems([
            self.t('method_alt_lines'),
            self.t('method_block_lines'),
            self.t('method_alt_sent'),
            self.t('method_block_sent'),
            self.t('method_merge_alt'),
            self.t('method_paragraph'),
            self.t('method_keep'),
        ])
        fl_text.addRow(self._mk_lbl('text_method'), self.method_combo)
        self.speakers_spin = QSpinBox(self.home)
        self.speakers_spin.setRange(1, 20)
        self.speakers_spin.setValue(2)
        self.btn_diarize = QPushButton(self.home)
        self.btn_diarize.clicked.connect(self.on_diarize_clicked)
        fl_text.addRow(self._mk_lbl("speakers"), self.speakers_spin)
        fl_text.addRow(self.btn_diarize)
        ribbon.addWidget(self.gb_text, 2)

        # Content splitters (input/output + logs resizable)
        self.splitter_v = QSplitter(Qt.Vertical, self.home)

        self.splitter_h = QSplitter(Qt.Horizontal, self.home)

        left = QWidget(self.home)
        left_layout = QVBoxLayout(left)
        self.lbl_input = QLabel(self.home)
        self.input_text = SegmentTextEdit(self.home)
        left_layout.addWidget(self.lbl_input)
        left_layout.addWidget(self.input_text, 1)

        # Report saving (under transcription result)
        self.btn_save_trans_report = QPushButton(self.home)
        self.btn_save_trans_report.clicked.connect(self.on_save_transcription_report)
        self.btn_save_trans_report.setEnabled(False)
        btn_row_trans_report = QHBoxLayout()
        btn_row_trans_report.addStretch(1)
        btn_row_trans_report.addWidget(self.btn_save_trans_report)
        left_layout.addLayout(btn_row_trans_report)
        self.splitter_h.addWidget(left)

        right = QWidget(self.home)
        right_layout = QVBoxLayout(right)
        self.lbl_output = QLabel(self.home)
        self.output_text = SegmentTextEdit(self.home)
        self.output_text.setReadOnly(True)
        # Subtle background tint per speaker in diarization view
        try:
            self.output_text.enable_speaker_coloring(True)
        except Exception:
            pass

        # Segment hover preview + popup editor
        self.input_text.segmentHovered.connect(lambda seg: self._on_segment_hovered(seg, self.input_text))
        self.output_text.segmentHovered.connect(lambda seg: self._on_segment_hovered(seg, self.output_text))
        self.input_text.segmentDoubleClicked.connect(lambda seg: self._on_segment_double_clicked(seg, self.input_text))
        self.output_text.segmentDoubleClicked.connect(lambda seg: self._on_segment_double_clicked(seg, self.output_text))

        right_layout.addWidget(self.lbl_output)
        right_layout.addWidget(self.output_text, 1)

        # Report saving (under diarization output)
        self.btn_save_report = QPushButton(self.home)
        self.btn_save_report.clicked.connect(self.on_save_diarization_report)
        self.btn_save_report.setEnabled(False)
        btn_row_report = QHBoxLayout()
        btn_row_report.addStretch(1)
        btn_row_report.addWidget(self.btn_save_report)
        right_layout.addLayout(btn_row_report)
        self.splitter_h.addWidget(right)

        self.splitter_h.setStretchFactor(0, 1)
        self.splitter_h.setStretchFactor(1, 1)
        self.splitter_h.setSizes([600, 650])

        self.splitter_v.addWidget(self.splitter_h)

        # Speaker naming / renaming (shared for transcription + diarization)
        self.speaker_name_map: dict[str, str] = {}
        self.speaker_panel = SpeakerNamesPanel(self.home, self.input_text, self.output_text, t=self.t)
        self.speaker_panel.mappingApplied.connect(self._on_speaker_mapping_applied)
        self.splitter_v.addWidget(self.speaker_panel)


        # Logs panel + buttons
        logs_panel = QWidget(self.home)
        logs_layout = QVBoxLayout(logs_panel)
        logs_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_logs = QLabel(self.home)
        self.log_box = QTextEdit(self.home)
        self.log_box.setReadOnly(True)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_clear_logs = QPushButton(self.home)
        self.btn_clear_logs.clicked.connect(self.on_clear_logs)
        self.btn_save_logs = QPushButton(self.home)
        self.btn_save_logs.clicked.connect(self.on_save_logs)
        btns.addWidget(self.btn_clear_logs)
        btns.addWidget(self.btn_save_logs)

        logs_layout.addWidget(self.lbl_logs)
        logs_layout.addWidget(self.log_box, 1)
        logs_layout.addLayout(btns)

        self.splitter_v.addWidget(logs_panel)
        self.splitter_v.setStretchFactor(0, 4)
        self.splitter_v.setStretchFactor(1, 1)
        self.splitter_v.setSizes([540, 120, 200])

        home_layout.addWidget(self.splitter_v, 1)

        self.tabs.addTab(self.home, "")

        # --- Settings tab ---
        self.settings_page = QWidget(self)
        sl = QVBoxLayout(self.settings_page)
        sl.setContentsMargins(12, 12, 12, 12)
        sl.setSpacing(10)

        self.gb_settings = QGroupBox(self.settings_page)
        form = QFormLayout(self.gb_settings)

        self.hf_token_edit = QLineEdit(self.settings_page)
        self.hf_token_edit.setEchoMode(QLineEdit.Password)
        self.hf_token_edit.textChanged.connect(self.on_hf_token_changed)

        self.theme_combo = QComboBox(self.settings_page)
        self.theme_combo.addItems(THEMES)
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)

        self.ui_lang_combo = QComboBox(self.settings_page)
        self.ui_lang_combo.addItem(self.t("ui_pl"), "pl")
        self.ui_lang_combo.addItem(self.t("ui_en"), "en")
        self.ui_lang_combo.currentIndexChanged.connect(self.on_ui_lang_changed)

        form.addRow(self._mk_lbl("settings_hf"), self.hf_token_edit)
        form.addRow(self._mk_lbl("settings_theme"), self.theme_combo)
        form.addRow(self._mk_lbl("settings_lang"), self.ui_lang_combo)

        sl.addWidget(self.gb_settings)
        sl.addStretch(1)
        self.tabs.addTab(self.settings_page, "")

        # --- Info tab ---
        self.info_page = QWidget(self)
        il = QVBoxLayout(self.info_page)
        il.setContentsMargins(12, 12, 12, 12)
        self.info_text = QTextBrowser(self.info_page)
        self.info_text.setOpenExternalLinks(True)
        self.info_text.setReadOnly(True)
        il.addWidget(self.info_text)
        self.tabs.addTab(self.info_page, "")

        # STATUS BAR
        status = QStatusBar(self)
        self.setStatusBar(status)
        self.status_label = QLabel("", self)
        status.addWidget(self.status_label)

        # Texts
        self._refresh_texts()
        self.tabs.setCurrentWidget(self.home)

    def _mk_lbl(self, key: str) -> QLabel:
        lbl = QLabel(self.t(key), self)
        return lbl

    def _refresh_texts(self) -> None:
        # Tabs
        self.tabs.setTabText(0, self.t("tab_file"))
        self.tabs.setTabText(1, self.t("tab_home"))
        self.tabs.setTabText(2, self.t("tab_settings"))
        self.tabs.setTabText(3, self.t("tab_info"))

        # Group titles
        self.gb_file.setTitle(self.t("grp_file"))
        self.btn_open_transcript.setText(self.t("btn_open_transcript"))
        self.btn_save_input.setText(self.t("btn_save_input"))
        self.btn_save_output.setText(self.t("btn_save_output"))
        self.btn_save_logs2.setText(self.t("btn_save_logs"))
        self.btn_quit.setText(self.t("btn_quit"))

        self.gb_audio.setTitle(self.t("grp_audio"))
        self.gb_trans.setTitle(self.t("grp_trans"))
        self.gb_text.setTitle(self.t("grp_text_diar"))
        self.gb_voice.setTitle(self.t("grp_voice_diar"))

        # Buttons/labels
        self.btn_load_audio.setText(self.t("load_audio"))
        self.btn_transcribe.setText(self.t("btn_transcribe"))
        self.btn_voice_diar.setText(self.t("btn_voice_diar"))
        self.btn_diarize.setText(self.t("btn_text_diar"))

        self.lbl_input.setText(self.t("input_label"))
        self.lbl_output.setText(self.t("output_label"))
        self.lbl_logs.setText(self.t("logs"))
        self.btn_clear_logs.setText(self.t("btn_clear"))
        self.btn_save_logs.setText(self.t("btn_save"))
        try:
            self.btn_save_trans_report.setText(self.t("btn_save_report"))
        except Exception:
            pass
        try:
            self.btn_save_report.setText(self.t("btn_save_report"))
        except Exception:
            pass

        # Settings
        self.gb_settings.setTitle(self.t("tab_settings"))
        self.ui_lang_combo.setItemText(0, self.t("ui_pl"))
        self.ui_lang_combo.setItemText(1, self.t("ui_en"))

        # Speaker panel (speaker names) uses its own widgets — retranslate them too
        try:
            self.speaker_panel.set_translator(self.t)
        except Exception:
            pass

        # Info
        self._render_info_markdown()

        # Audio label fallback
        if not self.audio_path:
            self.lbl_audio.setText(self.t("no_audio"))

        # Menu retranslate
        try:
            self._file_menu.setTitle(self.t("menu_file"))
            self._act_open_text.setText(self.t("btn_open_transcript"))
            self._act_save_transcript.setText(self.t("btn_save_input"))
            self._act_save_diar_report.setText(self.t("act_save_report"))
            self._act_save_logs.setText(self.t("btn_save_logs"))
            self._act_quit.setText(self.t("btn_quit"))
        except Exception:
            pass

        self._update_hf_status()

    def _apply_settings(self) -> None:
        # apply widget state from settings
        self.lang_combo.setCurrentText(self.settings.default_language or "auto")
        self.whisper_model_combo.setCurrentText(self.settings.whisper_model or "base")
        self.hf_token_edit.setText(self.settings.hf_token or "")
        # theme
        theme = self.settings.theme or "Fusion Light (Blue)"
        if theme in THEMES:
            self.theme_combo.setCurrentText(theme)
        else:
            self.theme_combo.setCurrentText("Fusion Light (Blue)")
        # ui language
        lang = (self.settings.ui_language or "pl").strip() or "pl"
        idx = self.ui_lang_combo.findData(lang)
        if idx >= 0:
            self.ui_lang_combo.setCurrentIndex(idx)

        if self._app is not None:
            apply_theme(self._app, self.theme_combo.currentText())

    def _update_status(self) -> None:
        model = self.whisper_model_combo.currentText()
        lang = self.lang_combo.currentText()
        self.status_label.setText(f"Whisper: {model} | Lang: {lang}")

    def _update_hf_status(self) -> None:
        tok = (self.settings.hf_token or "").strip()
        if tok:
            masked = tok[:4] + "…" + tok[-4:] if len(tok) > 8 else "*" * len(tok)
            self.lbl_hf_status.setText(f"HF: OK ({masked})")
        else:
            self.lbl_hf_status.setText("HF: MISSING")

    # ---------- Menu ----------
    def _build_menu(self) -> None:
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)

        # Keep references so we can retranslate on UI language change.
        self._file_menu = QMenu(self.t("menu_file"), self)
        menubar.addMenu(self._file_menu)

        self._act_open_text = self._file_menu.addAction(self.t("btn_open_transcript"))
        self._act_open_text.triggered.connect(self.on_open_text)

        self._act_save_transcript = self._file_menu.addAction(self.t("btn_save_input"))
        self._act_save_transcript.triggered.connect(self.on_save_input)

        self._act_save_diar_report = self._file_menu.addAction(self.t("act_save_report"))
        self._act_save_diar_report.triggered.connect(self.on_save_diarization_report)

        self._act_save_logs = self._file_menu.addAction(self.t("btn_save_logs"))
        self._act_save_logs.triggered.connect(self.on_save_logs)

        self._file_menu.addSeparator()
        self._act_quit = self._file_menu.addAction(self.t("btn_quit"))
        self._act_quit.triggered.connect(self.close)

    # ---------- Logging ----------
    def log(self, msg: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{ts}] {msg}")

    @Slot()
    def on_clear_logs(self) -> None:
        self.log_box.clear()

    @Slot()
    def on_save_logs(self) -> None:
        text = self.log_box.toPlainText()
        if not text.strip():
            QMessageBox.information(self, "No logs", "Nothing to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save logs", "aistate_logs.txt", "Text (*.txt);;All files (*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self.log(f"Logs saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    # ---------- Settings handlers ----------
    @Slot()
    def on_hf_token_changed(self) -> None:
        self.settings.hf_token = self.hf_token_edit.text().strip()
        save_settings(self.settings)
        self._update_hf_status()

    @Slot(str)
    def on_theme_changed(self, theme: str) -> None:
        self.settings.theme = theme
        save_settings(self.settings)
        if self._app is not None:
            apply_theme(self._app, theme)
        self.log(f"Theme set: {theme}")

    @Slot()
    def on_ui_lang_changed(self) -> None:
        lang = self.ui_lang_combo.currentData()
        self.settings.ui_language = lang
        save_settings(self.settings)
        self._refresh_texts()
        self.log(f"UI language set: {lang}")

    # ---------- File menu ----------
    @Slot()
    def on_open_text(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open transcript",
            "", "Text (*.txt);;All files (*)"
        )
        if not path:
            return
        try:
            content = open(path, "r", encoding="utf-8").read()
        except Exception as e:
            QMessageBox.critical(self, "Read error", str(e))
            return
        self.input_text.setPlainText(content)
        try:
            self.btn_save_trans_report.setEnabled(bool((content or "").strip()))
        except Exception:
            pass
        self.log(f"Loaded transcript: {path}")

    @Slot()
    def on_save_input(self) -> None:
        # Save transcription using the same report templates (TXT/PDF/HTML)
        # Default preselect: TXT (user can add PDF/HTML in the dialog)
        self.on_save_transcription_report(preselect={"txt"})

    @Slot()
    def on_save_output(self, checked: bool = False) -> None:
        # Save diarization as report in selected formats
        self.on_save_diarization_report(preselect={'txt'})

    def on_save_output_raw(self) -> None:
        text = self.output_text.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "No data", "Output is empty.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save diarization", "", "Text (*.txt);;All files (*)")
        if not path:
            return
        try:
            open(path, "w", encoding="utf-8").write(text)
            self.log(f"Saved diarization: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def on_load_audio_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select audio file",
            "", "Audio (*.wav *.mp3 *.m4a *.flac *.ogg *.opus);;All files (*)"
        )
        if not path:
            return
        self.audio_path = path
        self.lbl_audio.setText(path)
        self._set_preview_source()
        self.log(f"Audio loaded: {path}")

    @Slot()
    def on_transcribe_clicked(self) -> None:
        if not self.audio_path:
            QMessageBox.warning(self, "No audio", "Select an audio file first.")
            return

        model = self.whisper_model_combo.currentText()
        lang_raw = self.lang_combo.currentText().strip()
        language = "auto" if not lang_raw else lang_raw

        # persist choices
        self.settings.whisper_model = model
        self.settings.default_language = language
        save_settings(self.settings)

        self.log(f"Transcribe: Whisper '{model}', lang='{language}'")

        task = BackgroundTask(
            whisper_transcribe_safe,
            self.audio_path,
            model,
            language,
        )
        task.signals.progress.connect(self.on_task_progress)
        task.signals.log.connect(self.log)
        task.signals.error.connect(self.on_task_error)
        task.signals.finished.connect(self.on_transcribe_finished)

        self.task_runner.submit(task)

    def on_transcribe_finished(self, result: object) -> None:
        if not isinstance(result, dict) or result.get("kind") != "transcript":
            QMessageBox.warning(self, "Error", "Invalid transcription result.")
            return
        text_ts = result.get("text_ts") or result.get("text", "")
        self.input_text.setPlainText(text_ts)
        try:
            self.btn_save_trans_report.setEnabled(bool((text_ts or "").strip()))
        except Exception:
            pass
        # Auto-prompt report dialog after successful transcription
        if bool((text_ts or "").strip()) and bool(result.get("ok", True)):
            QTimer.singleShot(0, lambda: self.on_save_transcription_report())
        try:
            self.speaker_panel.refresh()
        except Exception:
            pass
        self.log("Transcription finished -> pasted into input pane.")

    def on_save_transcription_report(self, preselect: Optional[set[str]] = None) -> None:
        """Open modal report dialog and generate TXT/PDF/HTML reports for transcription result."""
        if isinstance(preselect, bool):
            preselect = None
        elif preselect is not None and not isinstance(preselect, set):
            try:
                preselect = set(preselect)  # type: ignore[arg-type]
            except TypeError:
                preselect = None

        transcript = (self.input_text.toPlainText() or "").strip()
        logs = (self.log_box.toPlainText() or "").strip()

        if not transcript and not logs:
            QMessageBox.warning(self, self.t("dlg_report_title"), self.t("msg_no_data_report"))
            return

        default_dir = os.path.dirname(self.audio_path) if self.audio_path else os.getcwd()
        default_base = "wynik"
        if self.audio_path:
            default_base = os.path.splitext(os.path.basename(self.audio_path))[0]

        dlg = ReportDialog(self, t=self.t, default_dir=default_dir, default_base=default_base, preselect=preselect)
        if dlg.exec() != QDialog.Accepted:
            return
        res = dlg.result_data()
        if not res:
            return

        export_formats = sorted(list(res.formats))
        data = self._collect_transcription_report_data(export_formats=export_formats, include_logs=res.include_logs)

        ts_file = datetime.datetime.now(ZoneInfo("Europe/Warsaw")).strftime("%Y%m%d_%H%M%S")
        base = res.base_name or "wynik"

        saved = []
        errors = []

        for fmt in export_formats:
            out_path = os.path.join(res.output_dir, f"raport_{base}_{ts_file}.{fmt}")
            try:
                if fmt == "txt":
                    generate_txt_report(data, logs=res.include_logs, output_path=out_path)
                elif fmt == "html":
                    generate_html_report(data, logs=res.include_logs, output_path=out_path)
                elif fmt == "pdf":
                    generate_pdf_report(data, logs=res.include_logs, output_path=out_path)
                else:
                    continue
                saved.append(out_path)
            except ModuleNotFoundError as e:
                errors.append(str(e))
            except Exception as e:
                errors.append(str(e))

        if saved:
            for p in saved:
                self.log(f"Report saved: {p}")
            QMessageBox.information(self, self.t("dlg_report_title"), "OK:\n" + "\n".join(saved))

        if errors:
            QMessageBox.warning(self, self.t("dlg_report_title"), "Errors:\n" + "\n".join(errors))

    @Slot()
    def on_diarize_clicked(self) -> None:
        # KEEP: working diarization in output window
        text = self.input_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No text", "Paste or generate transcript first.")
            return
        speakers = int(self.speakers_spin.value())
        self.log(f"Text diarization: alternate, speakers={speakers}")

        task = BackgroundTask(
            diarize_text_simple,
            text,
            speakers,
            "naprzemienna",
        )
        task.signals.progress.connect(self.on_task_progress)
        task.signals.log.connect(self.log)
        task.signals.error.connect(self.on_task_error)
        task.signals.finished.connect(self.on_diarize_finished)

        self.task_runner.submit(task)

    def on_diarize_finished(self, result: object) -> None:
        if not isinstance(result, dict) or result.get("kind") != "diarized_text":
            QMessageBox.warning(self, "Error", "Invalid diarization result.")
            return
        text = result.get("text", "")
        self.output_text.setPlainText(text)
        try:
            self.btn_save_report.setEnabled(True)
        except Exception:
            pass
        # Auto-prompt report dialog after successful diarization
        QTimer.singleShot(0, lambda: self.on_save_diarization_report())
        try:
            self.speaker_panel.refresh()
        except Exception:
            pass
        self.log("Text diarization finished -> pasted into output pane.")

    # New: voice diarization (pyannote)
    @Slot()
    def on_voice_diarize_clicked(self) -> None:
        if not self.audio_path:
            QMessageBox.warning(self, "No audio", "Select an audio file first.")
            return
        if not (self.settings.hf_token or "").strip():
            QMessageBox.information(self, self.t("msg_missing_hf_title"), self.t("msg_missing_hf_body"))
            self.log("HF token missing -> voice diarization cancelled.")
            return

        model = self.whisper_model_combo.currentText()
        lang_raw = self.lang_combo.currentText().strip()
        language = "auto" if not lang_raw else lang_raw

        self.settings.whisper_model = model
        self.settings.default_language = language
        save_settings(self.settings)

        self.log(f"Voice diarization: Whisper '{model}', lang='{language}' + pyannote")

        task = BackgroundTask(
            diarize_voice_whisper_pyannote_safe,
            self.audio_path,
            model,
            language,
            self.settings.hf_token,
        )
        task.signals.progress.connect(self.on_task_progress)
        task.signals.log.connect(self.log)
        task.signals.error.connect(self.on_task_error)
        task.signals.finished.connect(self.on_voice_diarize_finished)

        self.task_runner.submit(task)

    def on_voice_diarize_finished(self, result: object) -> None:
        if not isinstance(result, dict) or result.get("kind") != "diarized_voice":
            QMessageBox.warning(self, "Error", "Invalid voice diarization result.")
            return
        text = result.get("text", "") or ""
        self.output_text.setPlainText(text)
        try:
            self.btn_save_report.setEnabled(True)
        except Exception:
            pass
        # Auto-prompt report dialog after successful diarization
        QTimer.singleShot(0, lambda: self.on_save_diarization_report())
        try:
            self.speaker_panel.refresh()
        except Exception:
            pass
        self.log("Voice diarization completed -> pasted into output pane.")


    def on_save_diarization_report(self, preselect: Optional[set[str]] = None) -> None:
        """
        Open modal report dialog and generate TXT/PDF/HTML reports.
        """
        # Some Qt signals (e.g., QAction.triggered / QPushButton.clicked) pass a boolean 'checked'.
        # If this slot is connected directly, that bool may land in 'preselect' -> normalize it.
        if isinstance(preselect, bool):
            preselect = None
        elif preselect is not None and not isinstance(preselect, set):
            try:
                preselect = set(preselect)  # type: ignore[arg-type]
            except TypeError:
                preselect = None

        transcript = (self.input_text.toPlainText() or "").strip()
        diarized = (self.output_text.toPlainText() or "").strip()
        # Logs are displayed in the GUI log pane (self.log_box)
        logs = (self.log_box.toPlainText() or "").strip()

        if not transcript and not diarized and not logs:
            QMessageBox.warning(self, self.t("dlg_report_title"), self.t("msg_no_data_report"))
            return

        # default folder: beside audio or cwd
        default_dir = os.path.dirname(self.audio_path) if self.audio_path else os.getcwd()

        # Default base name = input audio stem (user can edit it in the dialog)
        default_base = "wynik"
        if self.audio_path:
            default_base = os.path.splitext(os.path.basename(self.audio_path))[0]

        dlg = ReportDialog(self, t=self.t, default_dir=default_dir, default_base=default_base, preselect=preselect)
        if dlg.exec() != QDialog.Accepted:
            return
        res = dlg.result_data()
        if not res:
            return

        # Build data dict (matches requested structure; best-effort)
        export_formats = sorted(list(res.formats))
        data = self._collect_report_data(export_formats=export_formats, include_logs=res.include_logs)

        # Generate files
        ts_file = datetime.datetime.now(ZoneInfo("Europe/Warsaw")).strftime("%Y%m%d_%H%M%S")
        base = res.base_name or "wynik"

        saved = []
        errors = []

        for fmt in export_formats:
            out_path = os.path.join(res.output_dir, f"raport_{base}_{ts_file}.{fmt}")
            try:
                if fmt == "txt":
                    generate_txt_report(data, logs=res.include_logs, output_path=out_path)
                elif fmt == "html":
                    generate_html_report(data, logs=res.include_logs, output_path=out_path)
                elif fmt == "pdf":
                    generate_pdf_report(data, logs=res.include_logs, output_path=out_path)
                else:
                    continue
                saved.append(out_path)
            except ModuleNotFoundError as e:
                errors.append(str(e))
            except Exception as e:
                errors.append(str(e))

        if saved:
            for p in saved:
                self.log(f"Report saved: {p}")
            QMessageBox.information(self, self.t("dlg_report_title"), "OK:\n" + "\n".join(saved))

        if errors:
            QMessageBox.warning(self, self.t("dlg_report_title"), "Errors:\n" + "\n".join(errors))

    def _collect_report_data(self, *, export_formats: list[str], include_logs: bool) -> dict:
        now = datetime.datetime.now(ZoneInfo("Europe/Warsaw"))
        dt_str = now.strftime("%Y-%m-%d %H:%M ") + (now.tzname() or "CET")

        audio_file = os.path.basename(self.audio_path) if self.audio_path else ""
        audio_duration = ""
        audio_specs = ""
        if self.audio_path and os.path.exists(self.audio_path):
            audio_duration, audio_specs = self._probe_audio(self.audio_path)

        diar_lines = [ln.strip() for ln in (self.output_text.toPlainText() or "").splitlines() if ln.strip()]
        raw_lines = [ln.rstrip() for ln in (self.input_text.toPlainText() or "").splitlines() if ln.strip()]

        seg_stats = self._compute_segment_stats(diar_lines)
        non_verbal = self._extract_nonverbal(diar_lines)

        py_model = self._extract_pyannote_model(self.log_box.toPlainText() or "")

        data = {
            "program_name": APP_NAME,
            "version": APP_VERSION,
            "author": AUTHOR_EMAIL,
            "datetime": dt_str,
            "audio_file": audio_file,
            "audio_file_path": self.audio_path or "",
            "audio_duration": audio_duration or "",
            "audio_specs": audio_specs or "",
            "whisper_model": getattr(self.settings, "whisper_model", "") or (getattr(self, "whisper_model_combo", None).currentText() if getattr(self, "whisper_model_combo", None) else ""),
            "language": getattr(self.settings, "default_language", "") or "auto",
            "pyannote_model": py_model or "",
            "speakers_count": seg_stats["speakers_count"],
            "segments_count": seg_stats["segments_count"],
            "speaker_times": seg_stats["speaker_times"],
            "transcript": diar_lines,
            "raw_transcript": raw_lines,
            "non_verbal": non_verbal,
            "export_formats": export_formats,
            "logs": (self.log_box.toPlainText() or "") if include_logs else "",
            "ui_language": getattr(self.settings, "ui_language", "") or "",
            "theme": getattr(self.settings, "theme", "") or "",
            "speaker_name_map": getattr(self, "speaker_name_map", {}) or {},
        }
        return data

    def _collect_transcription_report_data(self, *, export_formats: list[str], include_logs: bool) -> dict:
        """Build report payload for transcription-only output (uses the same templates)."""
        now = datetime.datetime.now(ZoneInfo("Europe/Warsaw"))
        dt_str = now.strftime("%Y-%m-%d %H:%M ") + (now.tzname() or "CET")

        audio_file = os.path.basename(self.audio_path) if self.audio_path else ""
        audio_duration = ""
        audio_specs = ""
        if self.audio_path and os.path.exists(self.audio_path):
            audio_duration, audio_specs = self._probe_audio(self.audio_path)

        raw_lines = [ln.rstrip() for ln in (self.input_text.toPlainText() or "").splitlines() if ln.strip()]
        non_verbal = self._extract_nonverbal(raw_lines)

        data = {
            "program_name": APP_NAME,
            "version": APP_VERSION,
            "author": AUTHOR_EMAIL,
            "datetime": dt_str,
            "audio_file": audio_file,
            "audio_file_path": self.audio_path or "",
            "audio_duration": audio_duration or "",
            "audio_specs": audio_specs or "",
            "whisper_model": getattr(self.settings, "whisper_model", "") or (getattr(self, "whisper_model_combo", None).currentText() if getattr(self, "whisper_model_combo", None) else ""),
            "language": getattr(self.settings, "default_language", "") or "auto",
            "pyannote_model": "",
            "speakers_count": "",
            "segments_count": len(raw_lines),
            "speaker_times": {},
            "transcript": raw_lines,
            "raw_transcript": raw_lines,
            "non_verbal": non_verbal,
            "export_formats": export_formats,
            "logs": (self.log_box.toPlainText() or "") if include_logs else "",
            "ui_language": getattr(self.settings, "ui_language", "") or "",
            "theme": getattr(self.settings, "theme", "") or "",
            "speaker_name_map": getattr(self, "speaker_name_map", {}) or {},
            "section_title": self.t("section_title_transcription"),
        }
        return data

    def _probe_audio(self, path: str) -> tuple[str, str]:
        """Best-effort WAV metadata using stdlib wave; falls back to size only."""
        size_b = 0
        try:
            size_b = os.path.getsize(path)
        except Exception:
            pass
        size_mb = ""
        if size_b:
            size_mb = f"{int(round(size_b / (1024*1024)))}MB"

        # Defaults
        duration_str = ""
        specs = size_mb

        try:
            with wave.open(path, "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                nframes = wf.getnframes()
                seconds = (nframes / float(sr)) if sr else 0.0
                h = int(seconds // 3600)
                m = int((seconds % 3600) // 60)
                s = int(seconds % 60)
                duration_str = f"{h:02d}:{m:02d}:{s:02d}"
                khz = sr / 1000.0
                ch_txt = "stereo" if ch == 2 else ("mono" if ch == 1 else f"{ch}ch")
                specs_parts = [f"{khz:.1f}kHz", ch_txt]
                if size_mb:
                    specs_parts.append(size_mb)
                specs = ", ".join(specs_parts)
        except Exception:
            # Not a WAV or can't be parsed; keep size only
            duration_str = ""
            specs = size_mb or ""

        return duration_str, specs

    def _compute_segment_stats(self, lines: list[str]) -> dict:
        """Parse diarized lines like: [00:01.23–00:05.67] SPEAKER_00: ..."""
        ts_re = re.compile(r"^\[(?P<a>[0-9:.,]+)\s*[-–]\s*(?P<b>[0-9:.,]+)\]\s*(?P<rest>.*)$")
        spk_re = re.compile(r"^(?P<spk>[^:\[\]]{1,64}):\s*(?P<txt>.*)$")

        def parse_ts(x: str) -> float:
            x = x.replace(",", ".")
            parts = x.split(":")
            try:
                if len(parts) == 3:
                    h = float(parts[0]); m = float(parts[1]); s = float(parts[2])
                    return h*3600 + m*60 + s
                if len(parts) == 2:
                    m = float(parts[0]); s = float(parts[1])
                    return m*60 + s
                return float(parts[0])
            except Exception:
                return 0.0

        segs = []
        for ln in lines:
            m = ts_re.match(ln.strip())
            if not m:
                continue
            rest = m.group("rest").strip()
            sm = spk_re.match(rest)
            if not sm:
                continue
            a = parse_ts(m.group("a"))
            b = parse_ts(m.group("b"))
            dur = max(0.0, b - a)
            spk = sm.group("spk").strip()
            segs.append((spk, dur))

        speakers = sorted(set([s for s, _ in segs]))
        totals = {}
        for spk, dur in segs:
            totals[spk] = totals.get(spk, 0.0) + dur

        total_dur = sum(totals.values()) or 0.0
        perc = {}
        if total_dur > 0:
            for spk in speakers:
                perc[spk] = f"{int(round(100.0 * totals.get(spk, 0.0) / total_dur))}%"
        else:
            for spk in speakers:
                perc[spk] = "0%"

        return {
            "speakers_count": len(speakers),
            "segments_count": len(segs),
            "speaker_times": perc,
        }

    def _extract_nonverbal(self, lines: list[str]) -> list[str]:
        # Detect bracketed tags like [APLAUZ], [HAŁAS ULICY], [MUZYKA W TLE]
        tag_re = re.compile(r"\[(?!\d)(?P<tag>[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż \-_]{2,64})\]")
        counts = {}
        for ln in lines:
            for m in tag_re.finditer(ln):
                tag = m.group("tag").strip()
                if not tag:
                    continue
                # Ignore if it looks like a timestamp part
                if ":" in tag and any(c.isdigit() for c in tag):
                    continue
                counts[tag] = counts.get(tag, 0) + 1
        out = []
        for tag, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            out.append(f"{tag} x{c}")
        return out

    def _extract_pyannote_model(self, logs: str) -> str:
        m = re.search(r"pipeline loaded OK:\s*([^\s]+)", logs or "")
        if m:
            return m.group(1).strip()
        # fallback (best-effort)
        m2 = re.search(r"trying pipeline '([^']+)'", logs or "")
        if m2:
            return m2.group(1).strip()
        return ""

    # ---------- Task events ----------
    def on_task_progress(self, value: int) -> None:
        self.status_label.setText(f"Progress: {value}%")

    def on_task_error(self, tb: str) -> None:
        self.log("ERROR:")
        for line in tb.splitlines():
            self.log(line)
        QMessageBox.critical(self, "Error", "Task failed. See logs.")
