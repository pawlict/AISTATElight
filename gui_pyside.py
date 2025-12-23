from __future__ import annotations

import datetime
import os
from typing import Optional

from PySide6.QtCore import Qt, Slot, QUrl
from PySide6.QtWidgets import (
    QWidget, QMainWindow, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QPushButton,
    QTextEdit, QTextBrowser, QComboBox, QSpinBox, QStatusBar, QMenuBar, QMenu,
    QTabWidget, QGroupBox, QFormLayout, QLineEdit
)

from backend.tasks import BackgroundTask, TaskRunner
from backend.legacy_adapter import (
    whisper_transcribe_safe,
    diarize_text_simple,
    diarize_voice_whisper_pyannote_safe,
)
from backend.settings import Settings
from backend.settings_store import load_settings, save_settings

from ui.theme import THEMES, apply_theme
from ui.i18n import tr


APP_NAME = "AI.S.T.A.T.E Light"
APP_VERSION = "v 1.0.0"


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
        self._apply_settings()
        self._update_status()
        self.log("Application started.")


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
                "**License:** MIT (AS IS)\n\n"
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
        # self._build_menu()  # replaced by File tab

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
        self.input_text = QTextEdit(self.home)
        left_layout.addWidget(self.lbl_input)
        left_layout.addWidget(self.input_text, 1)
        self.splitter_h.addWidget(left)

        right = QWidget(self.home)
        right_layout = QVBoxLayout(right)
        self.lbl_output = QLabel(self.home)
        self.output_text = QTextEdit(self.home)
        self.output_text.setReadOnly(True)
        right_layout.addWidget(self.lbl_output)
        right_layout.addWidget(self.output_text, 1)
        self.splitter_h.addWidget(right)

        self.splitter_h.setStretchFactor(0, 1)
        self.splitter_h.setStretchFactor(1, 1)
        self.splitter_h.setSizes([600, 650])

        self.splitter_v.addWidget(self.splitter_h)

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
        self.splitter_v.setSizes([620, 180])

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

        # Settings
        self.gb_settings.setTitle(self.t("tab_settings"))
        self.ui_lang_combo.setItemText(0, self.t("ui_pl"))
        self.ui_lang_combo.setItemText(1, self.t("ui_en"))

        # Info
        self._render_info_markdown()

        # Audio label fallback
        if not self.audio_path:
            self.lbl_audio.setText(self.t("no_audio"))

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

        file_menu = QMenu("File", self)
        menubar.addMenu(file_menu)

        act_open_text = file_menu.addAction("Open transcript…")
        act_open_text.triggered.connect(self.on_open_text)

        act_save_input = file_menu.addAction("Save input…")
        act_save_input.triggered.connect(self.on_save_input)

        act_save_output = file_menu.addAction("Save diarization…")
        act_save_output.triggered.connect(self.on_save_output)

        act_save_logs = file_menu.addAction("Save logs…")
        act_save_logs.triggered.connect(self.on_save_logs)

        file_menu.addSeparator()
        act_quit = file_menu.addAction("Quit")
        act_quit.triggered.connect(self.close)

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
        self.log(f"Loaded transcript: {path}")

    @Slot()
    def on_save_input(self) -> None:
        text = self.input_text.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "No data", "Input is empty.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save input", "", "Text (*.txt);;All files (*)")
        if not path:
            return
        try:
            open(path, "w", encoding="utf-8").write(text)
            self.log(f"Saved input: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    @Slot()
    def on_save_output(self) -> None:
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

    # ---------- Actions (existing logic preserved) ----------
    @Slot()
    def on_load_audio_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select audio file",
            "", "Audio (*.wav *.mp3 *.m4a *.flac *.ogg *.opus);;All files (*)"
        )
        if not path:
            return
        self.audio_path = path
        self.lbl_audio.setText(path)
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
        self.log("Transcription finished -> pasted into input pane.")

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
        self.log("Voice diarization completed -> pasted into output pane.")

    # ---------- Task events ----------
    def on_task_progress(self, value: int) -> None:
        self.status_label.setText(f"Progress: {value}%")

    def on_task_error(self, tb: str) -> None:
        self.log("ERROR:")
        for line in tb.splitlines():
            self.log(line)
        QMessageBox.critical(self, "Error", "Task failed. See logs.")
