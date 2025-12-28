from __future__ import annotations

import datetime
import os
import re
import wave
from pathlib import Path
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
from ui.segments import SegmentTextEdit, SegmentEditDialog, SpeakerNamesPanel, Segment


class MainWindow(QMainWindow):
    """Main application window with transcription and diarization functionality."""
    
    # Constants for better maintainability
    DEFAULT_WINDOW_SIZE = (1250, 820)
    PREVIEW_MAX_DURATION_MS = 2500
    PREVIEW_MIN_DURATION_MS = 250
    HOVER_DEBOUNCE_MS = 180
    SPLITTER_SIZES_H = [600, 650]
    SPLITTER_SIZES_V = [540, 120, 200]
    
    SUPPORTED_AUDIO_FORMATS = "Audio (*.wav *.mp3 *.m4a *.flac *.ogg *.opus);;All files (*)"
    SUPPORTED_TEXT_FORMATS = "Text (*.txt);;All files (*)"

    def __init__(self, app=None) -> None:
        super().__init__()
        self._app = app
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(*self.DEFAULT_WINDOW_SIZE)

        # Initialize state
        self.settings: Settings = load_settings()
        self.task_runner = TaskRunner()
        self.audio_path: Optional[str] = None
        self.speaker_name_map: dict[str, str] = {}

        # Build UI and apply configuration
        self._build_ui()
        self._init_audio_preview()
        self._apply_settings()
        self._update_status()
        self.log("Application started.")

    # ========== Audio Preview Methods ==========
    
    def _init_audio_preview(self) -> None:
        """Initialize audio player for segment hover preview."""
        self.preview_player = QMediaPlayer(self)
        self.preview_audio_output = QAudioOutput(self)
        self.preview_player.setAudioOutput(self.preview_audio_output)

        # Timer to stop preview playback
        self._preview_stop_timer = QTimer(self)
        self._preview_stop_timer.setSingleShot(True)
        self._preview_stop_timer.timeout.connect(self._stop_preview_player)

        # Debounce timer for hover events
        self._hover_debounce = QTimer(self)
        self._hover_debounce.setSingleShot(True)
        self._hover_debounce.setInterval(self.HOVER_DEBOUNCE_MS)
        self._hover_debounce.timeout.connect(self._play_pending_hover)
        self._pending_hover_seg: Optional[Segment] = None

        self._set_preview_source()

    def _set_preview_source(self) -> None:
        """Set audio source for preview player."""
        try:
            if self.audio_path:
                self.preview_player.setSource(QUrl.fromLocalFile(self.audio_path))
            else:
                self.preview_player.setSource(QUrl())
        except Exception as e:
            self.log(f"Warning: Could not set preview source: {e}")

    def _stop_preview_player(self) -> None:
        """Safely stop the preview player."""
        try:
            self.preview_player.stop()
        except Exception as e:
            self.log(f"Warning: Could not stop preview: {e}")

    def _play_pending_hover(self) -> None:
        """Play audio preview for hovered segment."""
        seg = self._pending_hover_seg
        self._pending_hover_seg = None
        
        if not seg or not self.audio_path:
            return

        # Calculate playback duration (max 2.5s)
        start_ms = int(max(0.0, seg.start_s) * 1000)
        seg_duration_ms = int(max(0.0, (seg.end_s - seg.start_s)) * 1000)
        play_duration_ms = max(
            self.PREVIEW_MIN_DURATION_MS,
            min(self.PREVIEW_MAX_DURATION_MS, seg_duration_ms or self.PREVIEW_MAX_DURATION_MS)
        )

        try:
            self.preview_player.setPosition(start_ms)
            self.preview_player.play()
            self._preview_stop_timer.start(play_duration_ms)
        except Exception as e:
            self.log(f"Warning: Could not play preview: {e}")

    def _on_segment_hovered(self, seg: Optional[Segment], editor: SegmentTextEdit) -> None:
        """Handle segment hover event."""
        if seg is None:
            self._hover_debounce.stop()
            self._pending_hover_seg = None
            self._stop_preview_player()
            return

        if not self.audio_path:
            return

        self._pending_hover_seg = seg
        self._hover_debounce.start()

    def _on_segment_double_clicked(self, seg: Segment, editor: SegmentTextEdit) -> None:
        """Handle segment double-click to open edit dialog."""
        if not self.audio_path:
            QMessageBox.information(
                self, 
                "Audio", 
                "Najpierw wczytaj plik audio, aby odsłuchiwać fragmenty."
            )
            return

        block = editor.document().findBlockByNumber(seg.block_number)
        current_line = block.text() if block.isValid() else ""

        dlg = SegmentEditDialog(
            parent=self,
            audio_path=self.audio_path,
            seg=seg,
            current_line=current_line,
            speaker_name_map=self.speaker_name_map,
            t=self.t,
        )
        
        if dlg.exec() != QDialog.Accepted:
            return
            
        new_line = dlg.build_new_line()
        editor.replace_block_text(seg.block_number, new_line)

        # Refresh speaker panel
        try:
            self.speaker_panel.refresh()
        except Exception as e:
            self.log(f"Warning: Could not refresh speaker panel: {e}")

    def _on_speaker_mapping_applied(self, mapping: dict) -> None:
        """Update speaker name mapping when applied."""
        if not isinstance(mapping, dict):
            return
            
        try:
            self.speaker_name_map.update({str(k): str(v) for k, v in mapping.items()})
            self.log(f"Speaker mapping applied: {mapping}")
        except Exception as e:
            self.log(f"Warning: Could not apply speaker mapping: {e}")

    # ========== UI Building Methods ==========

    def _build_ui(self) -> None:
        """Build the main user interface."""
        self._build_menu()

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.tabs = QTabWidget(self)
        root.addWidget(self.tabs, 1)

        # Build all tabs
        self._build_file_tab()
        self._build_home_tab()
        self._build_settings_tab()
        self._build_info_tab()

        # Status bar
        status = QStatusBar(self)
        self.setStatusBar(status)
        self.status_label = QLabel("", self)
        status.addWidget(self.status_label)

        # Apply translations
        self._refresh_texts()
        self.tabs.setCurrentWidget(self.home)

    def _build_file_tab(self) -> None:
        """Build the File tab."""
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

        self.tabs.addTab(self.file_tab, "")

    def _build_home_tab(self) -> None:
        """Build the Home tab with main functionality."""
        self.home = QWidget(self)
        home_layout = QVBoxLayout(self.home)
        home_layout.setContentsMargins(8, 8, 8, 8)
        home_layout.setSpacing(8)

        # Ribbon with control groups
        ribbon = QHBoxLayout()
        ribbon.setSpacing(10)
        home_layout.addLayout(ribbon)

        self._build_audio_group(ribbon)
        self._build_whisper_group(ribbon)
        self._build_voice_diarization_group(ribbon)
        self._build_text_diarization_group(ribbon)

        # Content area
        self._build_content_area(home_layout)

        self.tabs.addTab(self.home, "")

    def _build_audio_group(self, parent_layout: QHBoxLayout) -> None:
        """Build audio controls group."""
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
        
        parent_layout.addWidget(self.gb_audio, 2)

    def _build_whisper_group(self, parent_layout: QHBoxLayout) -> None:
        """Build Whisper transcription group."""
        self.gb_trans = QGroupBox(self.home)
        fl_trans = QFormLayout(self.gb_trans)
        
        self.whisper_model_combo = QComboBox(self.home)
        self.whisper_model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        
        self.btn_transcribe = QPushButton(self.home)
        self.btn_transcribe.clicked.connect(self.on_transcribe_clicked)
        
        fl_trans.addRow(self._mk_lbl("whisper_model"), self.whisper_model_combo)
        fl_trans.addRow(self.btn_transcribe)
        
        parent_layout.addWidget(self.gb_trans, 2)

    def _build_voice_diarization_group(self, parent_layout: QHBoxLayout) -> None:
        """Build voice diarization group."""
        self.gb_voice = QGroupBox(self.home)
        vbox_voice = QVBoxLayout(self.gb_voice)
        
        self.btn_voice_diar = QPushButton(self.home)
        self.btn_voice_diar.clicked.connect(self.on_voice_diarize_clicked)
        
        self.lbl_hf_status = QLabel(self.home)
        self.lbl_hf_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        vbox_voice.addWidget(self.btn_voice_diar)
        vbox_voice.addWidget(self.lbl_hf_status)
        vbox_voice.addStretch(1)
        
        parent_layout.addWidget(self.gb_voice, 2)

    def _build_text_diarization_group(self, parent_layout: QHBoxLayout) -> None:
        """Build text diarization group."""
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
        
        self.speakers_spin = QSpinBox(self.home)
        self.speakers_spin.setRange(1, 20)
        self.speakers_spin.setValue(2)
        
        self.btn_diarize = QPushButton(self.home)
        self.btn_diarize.clicked.connect(self.on_diarize_clicked)
        
        fl_text.addRow(self._mk_lbl('text_method'), self.method_combo)
        fl_text.addRow(self._mk_lbl("speakers"), self.speakers_spin)
        fl_text.addRow(self.btn_diarize)
        
        parent_layout.addWidget(self.gb_text, 2)

    def _build_content_area(self, parent_layout: QVBoxLayout) -> None:
        """Build main content area with input/output panels."""
        self.splitter_v = QSplitter(Qt.Vertical, self.home)
        self.splitter_h = QSplitter(Qt.Horizontal, self.home)

        # Input panel
        self._build_input_panel()
        
        # Output panel
        self._build_output_panel()

        self.splitter_h.setStretchFactor(0, 1)
        self.splitter_h.setStretchFactor(1, 1)
        self.splitter_h.setSizes(self.SPLITTER_SIZES_H)

        self.splitter_v.addWidget(self.splitter_h)

        # Speaker naming panel
        self.speaker_panel = SpeakerNamesPanel(
            self.home, 
            self.input_text, 
            self.output_text, 
            t=self.t
        )
        self.speaker_panel.mappingApplied.connect(self._on_speaker_mapping_applied)
        self.splitter_v.addWidget(self.speaker_panel)

        # Logs panel
        self._build_logs_panel()

        self.splitter_v.setStretchFactor(0, 4)
        self.splitter_v.setStretchFactor(1, 1)
        self.splitter_v.setSizes(self.SPLITTER_SIZES_V)

        parent_layout.addWidget(self.splitter_v, 1)

    def _build_input_panel(self) -> None:
        """Build input text panel."""
        left = QWidget(self.home)
        left_layout = QVBoxLayout(left)
        
        self.lbl_input = QLabel(self.home)
        self.input_text = SegmentTextEdit(self.home)
        
        # Connect signals
        self.input_text.segmentHovered.connect(
            lambda seg: self._on_segment_hovered(seg, self.input_text)
        )
        self.input_text.segmentDoubleClicked.connect(
            lambda seg: self._on_segment_double_clicked(seg, self.input_text)
        )
        
        left_layout.addWidget(self.lbl_input)
        left_layout.addWidget(self.input_text, 1)
        
        self.splitter_h.addWidget(left)

    def _build_output_panel(self) -> None:
        """Build output text panel."""
        right = QWidget(self.home)
        right_layout = QVBoxLayout(right)
        
        self.lbl_output = QLabel(self.home)
        self.output_text = SegmentTextEdit(self.home)
        self.output_text.setReadOnly(True)
        
        # Enable speaker coloring
        try:
            self.output_text.enable_speaker_coloring(True)
        except Exception as e:
            self.log(f"Warning: Could not enable speaker coloring: {e}")

        # Connect signals
        self.output_text.segmentHovered.connect(
            lambda seg: self._on_segment_hovered(seg, self.output_text)
        )
        self.output_text.segmentDoubleClicked.connect(
            lambda seg: self._on_segment_double_clicked(seg, self.output_text)
        )
        
        right_layout.addWidget(self.lbl_output)
        right_layout.addWidget(self.output_text, 1)

        # Save report button
        self.btn_save_report = QPushButton(self.home)
        self.btn_save_report.clicked.connect(self.on_save_diarization_report)
        self.btn_save_report.setEnabled(False)
        
        btn_row_report = QHBoxLayout()
        btn_row_report.addStretch(1)
        btn_row_report.addWidget(self.btn_save_report)
        right_layout.addLayout(btn_row_report)
        
        self.splitter_h.addWidget(right)

    def _build_logs_panel(self) -> None:
        """Build logs panel."""
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

    def _build_settings_tab(self) -> None:
        """Build Settings tab."""
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

    def _build_info_tab(self) -> None:
        """Build Info tab."""
        self.info_page = QWidget(self)
        il = QVBoxLayout(self.info_page)
        il.setContentsMargins(12, 12, 12, 12)
        
        self.info_text = QTextBrowser(self.info_page)
        self.info_text.setOpenExternalLinks(True)
        self.info_text.setReadOnly(True)
        
        il.addWidget(self.info_text)
        self.tabs.addTab(self.info_page, "")

    def _build_menu(self) -> None:
        """Build application menu bar."""
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)

        file_menu = QMenu(self.t("menu_file"), self)
        menubar.addMenu(file_menu)

        act_open_text = file_menu.addAction(self.t("btn_open_transcript"))
        act_open_text.triggered.connect(self.on_open_text)

        act_save_input = file_menu.addAction(self.t("btn_save_input"))
        act_save_input.triggered.connect(self.on_save_input)

        act_save_output = file_menu.addAction(self.t("act_save_report"))
        act_save_output.triggered.connect(self.on_save_diarization_report)

        act_save_logs = file_menu.addAction("Save logs…")
        act_save_logs.triggered.connect(self.on_save_logs)

        file_menu.addSeparator()
        
        act_quit = file_menu.addAction(self.t("btn_quit"))
        act_quit.triggered.connect(self.close)

    # ========== Translation & Info Methods ==========

    def t(self, key: str) -> str:
        """Translate key using current UI language."""
        lang = (self.settings.ui_language or "pl").strip() or "pl"
        return tr(lang, key)

    def _mk_lbl(self, key: str) -> QLabel:
        """Create translated label."""
        return QLabel(self.t(key), self)

    def _refresh_texts(self) -> None:
        """Refresh all UI texts with current translations."""
        # Tab titles
        self.tabs.setTabText(0, self.t("tab_file"))
        self.tabs.setTabText(1, self.t("tab_home"))
        self.tabs.setTabText(2, self.t("tab_settings"))
        self.tabs.setTabText(3, self.t("tab_info"))

        # File tab
        self.gb_file.setTitle(self.t("grp_file"))
        self.btn_open_transcript.setText(self.t("btn_open_transcript"))
        self.btn_save_input.setText(self.t("btn_save_input"))
        self.btn_save_output.setText(self.t("btn_save_output"))
        self.btn_save_logs2.setText(self.t("btn_save_logs"))
        self.btn_quit.setText(self.t("btn_quit"))

        # Home tab groups
        self.gb_audio.setTitle(self.t("grp_audio"))
        self.gb_trans.setTitle(self.t("grp_trans"))
        self.gb_text.setTitle(self.t("grp_text_diar"))
        self.gb_voice.setTitle(self.t("grp_voice_diar"))

        # Buttons
        self.btn_load_audio.setText(self.t("load_audio"))
        self.btn_transcribe.setText(self.t("btn_transcribe"))
        self.btn_voice_diar.setText(self.t("btn_voice_diar"))
        self.btn_diarize.setText(self.t("btn_text_diar"))

        # Labels
        self.lbl_input.setText(self.t("input_label"))
        self.lbl_output.setText(self.t("output_label"))
        self.lbl_logs.setText(self.t("logs"))
        self.btn_clear_logs.setText(self.t("btn_clear"))
        self.btn_save_logs.setText(self.t("btn_save"))
        
        try:
            self.btn_save_report.setText(self.t("btn_save_report"))
        except Exception:
            pass

        # Settings tab
        self.gb_settings.setTitle(self.t("tab_settings"))
        self.ui_lang_combo.setItemText(0, self.t("ui_pl"))
        self.ui_lang_combo.setItemText(1, self.t("ui_en"))

        # Speaker panel
        try:
            self.speaker_panel.set_translator(self.t)
        except Exception as e:
            self.log(f"Warning: Could not set speaker panel translator: {e}")

        # Info tab
        self._render_info_markdown()

        # Audio label
        if not self.audio_path:
            self.lbl_audio.setText(self.t("no_audio"))

        self._update_hf_status()

    def _render_info_markdown(self) -> None:
        """Load and render Info markdown file."""
        lang = (self.settings.ui_language or "pl").strip() or "pl"
        
        root_dir = Path(__file__).parent
        ui_dir = root_dir / "ui"
        
        # Try language-specific file first, then fallback
        candidates = [
            ui_dir / f"Info_{lang}.md",
            ui_dir / "Info.md",
        ]
        
        info_path = next((p for p in candidates if p.is_file()), None)

        if not info_path:
            # Built-in fallback
            md = (
                f"# {APP_NAME}\n\n"
                f"**Version:** {APP_VERSION}\n\n"
                "**Author:** pawlict\n\n"
                "**License:** AISTATElight License v1.2 (Source-Available) – AS IS\n\n"
                "## Notes\n"
                "- Whisper models auto-download on first use.\n"
                "- Voice diarization uses pyannote and requires a HF token + accepted model terms.\n"
            )
        else:
            try:
                md = info_path.read_text(encoding="utf-8")
            except Exception as e:
                self.info_text.setPlainText(f"Failed to load Info markdown:\n{e}")
                return

        # Replace placeholders
        md = md.replace("{{APP_NAME}}", APP_NAME).replace("{{APP_VERSION}}", APP_VERSION)

        # Set base URL for relative links
        try:
            self.info_text.document().setBaseUrl(QUrl.fromLocalFile(str(ui_dir) + os.sep))
        except Exception as e:
            self.log(f"Warning: Could not set base URL: {e}")

        # Render markdown
        try:
            self.info_text.setMarkdown(md)
        except Exception:
            self.info_text.setPlainText(md)

    # ========== Settings Methods ==========

    def _apply_settings(self) -> None:
        """Apply settings to UI widgets."""
        self.lang_combo.setCurrentText(self.settings.default_language or "auto")
        self.whisper_model_combo.setCurrentText(self.settings.whisper_model or "base")
        self.hf_token_edit.setText(self.settings.hf_token or "")
        
        # Theme
        theme = self.settings.theme or "Fusion Light (Blue)"
        if theme in THEMES:
            self.theme_combo.setCurrentText(theme)
        else:
            self.theme_combo.setCurrentText("Fusion Light (Blue)")
        
        # UI language
        lang = (self.settings.ui_language or "pl").strip() or "pl"
        idx = self.ui_lang_combo.findData(lang)
        if idx >= 0:
            self.ui_lang_combo.setCurrentIndex(idx)

        # Apply theme
        if self._app is not None:
            apply_theme(self._app, self.theme_combo.currentText())

    def _update_status(self) -> None:
        """Update status bar with current settings."""
        model = self.whisper_model_combo.currentText()
        lang = self.lang_combo.currentText()
        self.status_label.setText(f"Whisper: {model} | Lang: {lang}")

    def _update_hf_status(self) -> None:
        """Update HuggingFace token status display."""
        tok = (self.settings.hf_token or "").strip()
        if tok:
            masked = tok[:4] + "…" + tok[-4:] if len(tok) > 8 else "*" * len(tok)
            self.lbl_hf_status.setText(f"HF: OK ({masked})")
        else:
            self.lbl_hf_status.setText("HF: MISSING")

    # ========== Logging Methods ==========

    def log(self, msg: str) -> None:
        """Add timestamped message to log."""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{ts}] {msg}")

    @Slot()
    def on_clear_logs(self) -> None:
        """Clear log display."""
        self.log_box.clear()

    @Slot()
    def on_save_logs(self) -> None:
        """Save logs to file."""
        text = self.log_box.toPlainText()
        if not text.strip():
            QMessageBox.information(self, "No logs", "Nothing to save.")
            return
            
        path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save logs", 
            "aistate_logs.txt", 
            self.SUPPORTED_TEXT_FORMATS
        )
        if not path:
            return
            
        try:
            Path(path).write_text(text, encoding="utf-8")
            self.log(f"Logs saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))
            self.log(f"Error saving logs: {e}")

    # ========== Settings Handlers ==========

    @Slot()
    def on_hf_token_changed(self) -> None:
        """Handle HuggingFace token change."""
        self.settings.hf_token = self.hf_token_edit.text().strip()
        save_settings(self.settings)
        self._update_hf_status()

    @Slot(str)
    def on_theme_changed(self, theme: str) -> None:
        """Handle theme change."""
        self.settings.theme = theme
        save_settings(self.settings)
        if self._app is not None:
            apply_theme(self._app, theme)
        self.log(f"Theme set: {theme}")

    @Slot()
    def on_ui_lang_changed(self) -> None:
        """Handle UI language change."""
        lang = self.ui_lang_combo.currentData()
        self.settings.ui_language = lang
        save_settings(self.settings)
        self._refresh_texts()
        self.log(f"UI language set: {lang}")

    # ========== File Operations ==========

    @Slot()
    def on_open_text(self) -> None:
        """Open transcript file."""
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open transcript",
            "", 
            self.SUPPORTED_TEXT_FORMATS
        )
        if not path:
            return
            
        try:
            content = Path(path).read_text(encoding="utf-8")
            self.input_text.setPlainText(content)
            self.log(f"Loaded transcript: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Read error", str(e))
            self.log(f"Error loading transcript: {e}")

    @Slot()
    def on_save_input(self) -> None:
        """Save input text to file."""
        text = self.input_text.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "No data", "Input is empty.")
            return
            
        path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save input", 
            "", 
            self.SUPPORTED_TEXT_FORMATS
        )
        if not path:
            return
            
        try:
            Path(path).write_text(text, encoding="utf-8")
            self.log(f"Saved input: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))
            self.log(f"Error saving input: {e}")

    @Slot()
    def on_save_output(self, checked: bool = False) -> None:
        """Save output as report."""
        self.on_save_diarization_report(preselect={'txt'})

    @Slot()
    def on_load_audio_clicked(self) -> None:
        """Load audio file."""
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select audio file",
            "", 
            self.SUPPORTED_AUDIO_FORMATS
        )
        if not path:
            return
            
        self.audio_path = path
        self.lbl_audio.setText(path)
        self._set_preview_source()
        self.log(f"Audio loaded: {path}")

    # ========== Processing Methods ==========

    @Slot()
    def on_transcribe_clicked(self) -> None:
        """Start Whisper transcription."""
        if not self.audio_path:
            QMessageBox.warning(self, "No audio", "Select an audio file first.")
            return

        model = self.whisper_model_combo.currentText()
        lang_raw = self.lang_combo.currentText().strip()
        language = "auto" if not lang_raw else lang_raw

        # Save settings
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
        """Handle transcription completion."""
        if not isinstance(result, dict) or result.get("kind") != "transcript":
            QMessageBox.warning(self, "Error", "Invalid transcription result.")
            return
            
        text_ts = result.get("text_ts") or result.get("text", "")
        self.input_text.setPlainText(text_ts)
        
        try:
            self.speaker_panel.refresh()
        except Exception as e:
            self.log(f"Warning: Could not refresh speaker panel: {e}")
            
        self.log("Transcription finished -> pasted into input pane.")

    @Slot()
    def on_diarize_clicked(self) -> None:
        """Start text-based diarization."""
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
        """Handle text diarization completion."""
        if not isinstance(result, dict) or result.get("kind") != "diarized_text":
            QMessageBox.warning(self, "Error", "Invalid diarization result.")
            return
            
        text = result.get("text", "")
        self.output_text.setPlainText(text)
        
        try:
            self.btn_save_report.setEnabled(True)
        except Exception as e:
            self.log(f"Warning: Could not enable save button: {e}")
            
        # Auto-prompt report dialog
        QTimer.singleShot(0, lambda: self.on_save_diarization_report())
        
        try:
            self.speaker_panel.refresh()
        except Exception as e:
            self.log(f"Warning: Could not refresh speaker panel: {e}")
            
        self.log("Text diarization finished -> pasted into output pane.")

    @Slot()
    def on_voice_diarize_clicked(self) -> None:
        """Start voice-based diarization with pyannote."""
        if not self.audio_path:
            QMessageBox.warning(self, "No audio", "Select an audio file first.")
            return
            
        if not (self.settings.hf_token or "").strip():
            QMessageBox.information(
                self, 
                self.t("msg_missing_hf_title"), 
                self.t("msg_missing_hf_body")
            )
            self.log("HF token missing -> voice diarization cancelled.")
            return

        model = self.whisper_model_combo.currentText()
        lang_raw = self.lang_combo.currentText().strip()
        language = "auto" if not lang_raw else lang_raw

        # Save settings
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
        """Handle voice diarization completion."""
        if not isinstance(result, dict) or result.get("kind") != "diarized_voice":
            QMessageBox.warning(self, "Error", "Invalid voice diarization result.")
            return
            
        text = result.get("text", "") or ""
        self.output_text.setPlainText(text)
        
        try:
            self.btn_save_report.setEnabled(True)
        except Exception as e:
            self.log(f"Warning: Could not enable save button: {e}")
            
        # Auto-prompt report dialog
        QTimer.singleShot(0, lambda: self.on_save_diarization_report())
        
        try:
            self.speaker_panel.refresh()
        except Exception as e:
            self.log(f"Warning: Could not refresh speaker panel: {e}")
            
        self.log("Voice diarization completed -> pasted into output pane.")

    # ========== Report Generation ==========

    def on_save_diarization_report(self, preselect: Optional[set[str]] = None) -> None:
        """Generate and save diarization report."""
        # Normalize preselect parameter (handle Qt signal boolean)
        if isinstance(preselect, bool):
            preselect = None
        elif preselect is not None and not isinstance(preselect, set):
            try:
                preselect = set(preselect)
            except TypeError:
                preselect = None

        transcript = (self.input_text.toPlainText() or "").strip()
        diarized = (self.output_text.toPlainText() or "").strip()
        logs = (self.log_box.toPlainText() or "").strip()

        if not transcript and not diarized and not logs:
            QMessageBox.warning(
                self, 
                self.t("dlg_report_title"), 
                self.t("msg_no_data_report")
            )
            return

        # Determine default directory and base name
        default_dir = Path(self.audio_path).parent if self.audio_path else Path.cwd()
        default_base = "wynik"
        if self.audio_path:
            default_base = Path(self.audio_path).stem

        dlg = ReportDialog(
            self, 
            t=self.t, 
            default_dir=str(default_dir), 
            default_base=default_base, 
            preselect=preselect
        )
        
        if dlg.exec() != QDialog.Accepted:
            return
            
        res = dlg.result_data()
        if not res:
            return

        # Collect report data
        export_formats = sorted(list(res.formats))
        data = self._collect_report_data(
            export_formats=export_formats, 
            include_logs=res.include_logs
        )

        # Generate timestamp
        ts_file = datetime.datetime.now(ZoneInfo("Europe/Warsaw")).strftime("%Y%m%d_%H%M%S")
        base = res.base_name or "wynik"

        saved = []
        errors = []

        # Generate each format
        for fmt in export_formats:
            out_path = Path(res.output_dir) / f"raport_{base}_{ts_file}.{fmt}"
            try:
                if fmt == "txt":
                    generate_txt_report(data, logs=res.include_logs, output_path=str(out_path))
                elif fmt == "html":
                    generate_html_report(data, logs=res.include_logs, output_path=str(out_path))
                elif fmt == "pdf":
                    generate_pdf_report(data, logs=res.include_logs, output_path=str(out_path))
                else:
                    continue
                saved.append(str(out_path))
            except ModuleNotFoundError as e:
                errors.append(str(e))
                self.log(f"Module error generating {fmt}: {e}")
            except Exception as e:
                errors.append(str(e))
                self.log(f"Error generating {fmt}: {e}")

        # Show results
        if saved:
            for p in saved:
                self.log(f"Report saved: {p}")
            QMessageBox.information(
                self, 
                self.t("dlg_report_title"), 
                "OK:\n" + "\n".join(saved)
            )

        if errors:
            QMessageBox.warning(
                self, 
                self.t("dlg_report_title"), 
                "Errors:\n" + "\n".join(errors)
            )

    def _collect_report_data(
        self, 
        *, 
        export_formats: list[str], 
        include_logs: bool
    ) -> dict:
        """Collect all data needed for report generation."""
        now = datetime.datetime.now(ZoneInfo("Europe/Warsaw"))
        dt_str = now.strftime("%Y-%m-%d %H:%M ") + (now.tzname() or "CET")

        audio_file = Path(self.audio_path).name if self.audio_path else ""
        audio_duration = ""
        audio_specs = ""
        
        if self.audio_path and Path(self.audio_path).exists():
            audio_duration, audio_specs = self._probe_audio(self.audio_path)

        diar_lines = [
            ln.strip() 
            for ln in (self.output_text.toPlainText() or "").splitlines() 
            if ln.strip()
        ]
        raw_lines = [
            ln.rstrip() 
            for ln in (self.input_text.toPlainText() or "").splitlines() 
            if ln.strip()
        ]

        seg_stats = self._compute_segment_stats(diar_lines)
        non_verbal = self._extract_nonverbal(diar_lines)
        py_model = self._extract_pyannote_model(self.log_box.toPlainText() or "")

        return {
            "program_name": APP_NAME,
            "version": APP_VERSION,
            "author": AUTHOR_EMAIL,
            "datetime": dt_str,
            "audio_file": audio_file,
            "audio_file_path": self.audio_path or "",
            "audio_duration": audio_duration,
            "audio_specs": audio_specs,
            "whisper_model": self.settings.whisper_model or self.whisper_model_combo.currentText(),
            "language": self.settings.default_language or "auto",
            "pyannote_model": py_model,
            "speakers_count": seg_stats["speakers_count"],
            "segments_count": seg_stats["segments_count"],
            "speaker_times": seg_stats["speaker_times"],
            "transcript": diar_lines,
            "raw_transcript": raw_lines,
            "non_verbal": non_verbal,
            "export_formats": export_formats,
            "logs": (self.log_box.toPlainText() or "") if include_logs else "",
            "ui_language": self.settings.ui_language or "",
            "theme": self.settings.theme or "",
            "speaker_name_map": self.speaker_name_map,
        }

    # ========== Audio Analysis Methods ==========

    def _probe_audio(self, path: str) -> tuple[str, str]:
        """Extract audio file metadata."""
        # Get file size
        size_b = 0
        try:
            size_b = Path(path).stat().st_size
        except Exception:
            pass
            
        size_mb = f"{int(round(size_b / (1024*1024)))}MB" if size_b else ""

        duration_str = ""
        specs = size_mb

        # Try to read WAV metadata
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
            # Not a WAV or cannot be parsed
            pass

        return duration_str, specs

    def _compute_segment_stats(self, lines: list[str]) -> dict:
        """Parse diarized segments and compute statistics."""
        # Regex patterns
        ts_re = re.compile(
            r"^\[(?P<a>[0-9:.,]+)\s*[-–]\s*(?P<b>[0-9:.,]+)\]\s*(?P<rest>.*)$"
        )
        spk_re = re.compile(r"^(?P<spk>[^:\[\]]{1,64}):\s*(?P<txt>.*)$")

        def parse_ts(x: str) -> float:
            """Parse timestamp string to seconds."""
            x = x.replace(",", ".")
            parts = x.split(":")
            try:
                if len(parts) == 3:
                    h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
                    return h * 3600 + m * 60 + s
                if len(parts) == 2:
                    m, s = float(parts[0]), float(parts[1])
                    return m * 60 + s
                return float(parts[0])
            except (ValueError, IndexError):
                return 0.0

        # Parse segments
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

        # Calculate statistics
        speakers = sorted(set(s for s, _ in segs))
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
        """Extract non-verbal annotations from transcript."""
        # Match bracketed tags like [APLAUZ], [HAŁAS ULICY]
        tag_re = re.compile(r"\[(?!\d)(?P<tag>[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż \-_]{2,64})\]")
        counts = {}
        
        for ln in lines:
            for m in tag_re.finditer(ln):
                tag = m.group("tag").strip()
                if not tag:
                    continue
                    
                # Ignore timestamp-like patterns
                if ":" in tag and any(c.isdigit() for c in tag):
                    continue
                    
                counts[tag] = counts.get(tag, 0) + 1

        # Format output
        return [
            f"{tag} x{count}" 
            for tag, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        ]

    def _extract_pyannote_model(self, logs: str) -> str:
        """Extract pyannote model name from logs."""
        m = re.search(r"pipeline loaded OK:\s*([^\s]+)", logs)
        if m:
            return m.group(1).strip()
            
        # Fallback
        m2 = re.search(r"trying pipeline '([^']+)'", logs)
        if m2:
            return m2.group(1).strip()
            
        return ""

    # ========== Task Event Handlers ==========

    def on_task_progress(self, value: int) -> None:
        """Handle task progress update."""
        self.status_label.setText(f"Progress: {value}%")

    def on_task_error(self, tb: str) -> None:
        """Handle task error."""
        self.log("ERROR:")
        for line in tb.splitlines():
            self.log(line)
        QMessageBox.critical(self, "Error", "Task failed. See logs.")
