from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Set

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QCheckBox, QPushButton, QLineEdit, QFileDialog, QLabel, QMessageBox
)

@dataclass
class ReportDialogResult:
    include_logs: bool
    formats: Set[str]          # {"txt","pdf","html"}
    output_dir: str
    base_name: str             # base name used for generated files (without extension)
    base_name: str             # base name used for generated files (without extension)

class ReportDialog(QDialog):
    """
    Modal dialog to choose report formats (multi-select), include logs,
    and output directory.
    """
    def __init__(
        self,
        parent=None,
        *,
        t: Callable[[str], str],
        default_dir: str,
        default_base: str,
        preselect: Optional[Set[str]] = None,
    ) -> None:
        super().__init__(parent)
        self._t = t
        self.setWindowTitle(t("dlg_report_title"))
        self.setModal(True)

        # Qt can pass a bool when connecting signals directly (triggered(bool), clicked(bool)).
        # Normalize 'preselect' so membership checks work reliably.
        if isinstance(preselect, bool):
            preselect = None
        elif preselect is not None and not isinstance(preselect, set):
            try:
                preselect = set(preselect)
            except TypeError:
                preselect = None

        if preselect is None:
            preselect = {"pdf"}

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Options
        gb_opts = QGroupBox(t("dlg_report_options"), self)
        fl = QFormLayout(gb_opts)

        self.cb_logs = QCheckBox(t("dlg_include_logs"), gb_opts)
        self.cb_logs.setChecked(False)
        fl.addRow(self.cb_logs)

        self.cb_txt = QCheckBox("TXT", gb_opts)
        self.cb_pdf = QCheckBox("PDF", gb_opts)
        self.cb_html = QCheckBox("HTML", gb_opts)

        self.cb_txt.setChecked("txt" in preselect)
        self.cb_pdf.setChecked("pdf" in preselect)
        self.cb_html.setChecked("html" in preselect)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(self.cb_txt)
        fmt_row.addWidget(self.cb_pdf)
        fmt_row.addWidget(self.cb_html)
        fmt_row.addStretch(1)
        fl.addRow(QLabel(t("dlg_formats"), gb_opts), fmt_row)

        root.addWidget(gb_opts)

        # Output directory
        gb_out = QGroupBox(t("dlg_output"), self)
        out_layout = QVBoxLayout(gb_out)

        # Base filename (without extension)
        name_row = QHBoxLayout()
        self.name_edit = QLineEdit(gb_out)
        self.name_edit.setText((default_base or "").strip())
        self.name_edit.setPlaceholderText(t("dlg_filename_placeholder"))
        name_row.addWidget(QLabel(t("dlg_filename"), gb_out))
        name_row.addWidget(self.name_edit, 1)
        out_layout.addLayout(name_row)

        # Output folder
        out_row = QHBoxLayout()
        self.path_edit = QLineEdit(gb_out)
        self.path_edit.setText(default_dir or "")
        self.path_edit.setPlaceholderText(t("dlg_output_placeholder"))
        btn_browse = QPushButton(t("dlg_browse"), gb_out)
        btn_browse.clicked.connect(self._browse_dir)
        out_row.addWidget(self.path_edit, 1)
        out_row.addWidget(btn_browse)
        out_layout.addLayout(out_row)
        root.addWidget(gb_out)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_cancel = QPushButton(t("btn_cancel"), self)
        self.btn_generate = QPushButton(t("dlg_generate"), self)
        self.btn_generate.setDefault(True)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_generate.clicked.connect(self._on_generate)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_generate)
        root.addLayout(btn_row)

        self._result: Optional[ReportDialogResult] = None

    def _browse_dir(self) -> None:
        start = self.path_edit.text().strip() or os.getcwd()
        path = QFileDialog.getExistingDirectory(self, self._t("dlg_choose_folder"), start)
        if path:
            self.path_edit.setText(path)

    def _on_generate(self) -> None:
        base_name = self._sanitize_base_name(self.name_edit.text())
        if not base_name:
            QMessageBox.warning(self, self._t("dlg_report_title"), self._t("msg_no_filename"))
            return

        fmts = set()
        if self.cb_txt.isChecked():
            fmts.add("txt")
        if self.cb_pdf.isChecked():
            fmts.add("pdf")
        if self.cb_html.isChecked():
            fmts.add("html")

        if not fmts:
            QMessageBox.warning(self, self._t("dlg_report_title"), self._t("msg_no_format"))
            return

        out_dir = self.path_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, self._t("dlg_report_title"), self._t("msg_no_output"))
            return

        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            QMessageBox.warning(self, self._t("dlg_report_title"), self._t("msg_bad_output"))
            return

        self._result = ReportDialogResult(
            include_logs=self.cb_logs.isChecked(),
            formats=fmts,
            output_dir=out_dir,
            base_name=base_name,
        )
        self.accept()

    def _sanitize_base_name(self, name: str) -> str:
        """Return a filesystem-safe base name (no extension)."""
        name = (name or "").strip()
        if not name:
            return ""

        # Strip extension if user typed it.
        name = os.path.splitext(name)[0]

        # Replace characters that are invalid on Windows and problematic on Linux.
        for ch in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
            name = name.replace(ch, "_")

        # Avoid trailing dots/spaces (Windows) and empty result.
        name = name.strip(" .\t\n\r")
        return name

    def result_data(self) -> Optional[ReportDialogResult]:
        return self._result