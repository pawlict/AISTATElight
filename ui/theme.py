# ui/theme.py
# Modern, contrast-safe themes for PySide6/Qt (Fusion-based).
# Drop-in replacement for the project: provides THEMES + apply_theme().

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


# --------------------------------------------------------------------------------------
# Public API expected by the app
# --------------------------------------------------------------------------------------

THEMES = [
    "Fusion Dark (Nord)",
    "Fusion Dark (Dracula)",
    "Fusion Dark (Graphite)",
    "Fusion Dark (Fluent 11)",
    "Fusion Light (Solarized)",
    "Fusion Light (Paper)",
    "Fusion Light (Clean)",
    "Fusion Light (Blue)",
    "Fusion Light (Fluent 11)",
]


def apply_theme(app: QApplication, theme: str) -> None:
    """
    Apply a theme safely:
      - Forces Fusion style
      - Applies a matching QPalette (ensures readable text)
      - Applies QSS (gives the "effectful" look)
    """
    theme = (theme or "").strip()
    spec = _THEME_SPECS.get(theme, _THEME_SPECS["Fusion Dark (Nord)"])

    app.setStyle("Fusion")
    _apply_palette(app, spec.palette)
    app.setStyleSheet(_build_qss(spec))


def theme_stylesheet(theme: str) -> str:
    """Backwards-compatible helper if your code calls theme_stylesheet()."""
    theme = (theme or "").strip()
    spec = _THEME_SPECS.get(theme, _THEME_SPECS["Fusion Dark (Nord)"])
    return _build_qss(spec)


# --------------------------------------------------------------------------------------
# Theme specs
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PaletteSpec:
    mode: str  # "dark" | "light"
    window: str
    base: str
    alt_base: str
    text: str
    disabled_text: str
    button: str
    button_text: str
    highlight: str
    highlighted_text: str
    link: str
    tooltip_base: str
    tooltip_text: str
    border: str


@dataclass(frozen=True)
class ThemeSpec:
    name: str
    palette: PaletteSpec
    accent: str          # main accent color
    accent_2: str        # secondary accent
    radius: int          # corner radius for modern look
    font_size_px: int    # base font size
    dense: bool = False  # denser spacing
    fluent: bool = False # apply "Fluent/Win11-like" controls


# Dark palettes
_NORD = PaletteSpec(
    mode="dark",
    window="#2E3440",
    base="#2B303B",
    alt_base="#3B4252",
    text="#ECEFF4",
    disabled_text="#6C768A",
    button="#3B4252",
    button_text="#ECEFF4",
    highlight="#88C0D0",
    highlighted_text="#1F232A",
    link="#81A1C1",
    tooltip_base="#3B4252",
    tooltip_text="#ECEFF4",
    border="#4C566A",
)

_DRACULA = PaletteSpec(
    mode="dark",
    window="#282A36",
    base="#21222C",
    alt_base="#343746",
    text="#F8F8F2",
    disabled_text="#6A6D86",
    button="#343746",
    button_text="#F8F8F2",
    highlight="#BD93F9",
    highlighted_text="#1E1F29",
    link="#8BE9FD",
    tooltip_base="#343746",
    tooltip_text="#F8F8F2",
    border="#44475A",
)

_GRAPHITE = PaletteSpec(
    mode="dark",
    window="#1E1F22",
    base="#17181A",
    alt_base="#25262A",
    text="#E6E6E6",
    disabled_text="#7B7D86",
    button="#25262A",
    button_text="#E6E6E6",
    highlight="#4EA1FF",
    highlighted_text="#0D1117",
    link="#4EA1FF",
    tooltip_base="#25262A",
    tooltip_text="#E6E6E6",
    border="#35363C",
)

# Light palettes (high contrast, no "white-on-white")
_SOLARIZED_LIGHT = PaletteSpec(
    mode="light",
    window="#FDF6E3",
    base="#FFFFFF",
    alt_base="#F3EAD4",
    text="#073642",
    disabled_text="#93A1A1",
    button="#F3EAD4",
    button_text="#073642",
    highlight="#268BD2",
    highlighted_text="#FFFFFF",
    link="#268BD2",
    tooltip_base="#EEE8D5",
    tooltip_text="#073642",
    border="#D8CFAF",
)

_PAPER = PaletteSpec(
    mode="light",
    window="#F5F6F8",
    base="#FFFFFF",
    alt_base="#EEF1F5",
    text="#1E2329",
    disabled_text="#8A93A3",
    button="#EEF1F5",
    button_text="#1E2329",
    highlight="#2B7FFF",
    highlighted_text="#FFFFFF",
    link="#2B7FFF",
    tooltip_base="#FFFFFF",
    tooltip_text="#1E2329",
    border="#D5D9E0",
)

_CLEAN_BLUE = PaletteSpec(
    mode="light",
    window="#EEF5FF",
    base="#FFFFFF",
    alt_base="#E4EFFD",
    text="#101827",
    disabled_text="#7B8499",
    button="#E4EFFD",
    button_text="#101827",
    highlight="#2B7FFF",
    highlighted_text="#FFFFFF",
    link="#2B7FFF",
    tooltip_base="#FFFFFF",
    tooltip_text="#101827",
    border="#C9D6F2",
)

_CLEAN = PaletteSpec(
    mode="light",
    window="#F7F8FA",
    base="#FFFFFF",
    alt_base="#EFF2F6",
    text="#141A22",
    disabled_text="#8A93A3",
    button="#EFF2F6",
    button_text="#141A22",
    highlight="#3A86FF",
    highlighted_text="#FFFFFF",
    link="#3A86FF",
    tooltip_base="#FFFFFF",
    tooltip_text="#141A22",
    border="#D5D9E0",
)


_THEME_SPECS: Dict[str, ThemeSpec] = {
    "Fusion Dark (Nord)": ThemeSpec(
        name="Fusion Dark (Nord)",
        palette=_NORD,
        accent="#88C0D0",
        accent_2="#A3BE8C",
        radius=10,
        font_size_px=13,
        fluent=False,
    ),
    "Fusion Dark (Dracula)": ThemeSpec(
        name="Fusion Dark (Dracula)",
        palette=_DRACULA,
        accent="#BD93F9",
        accent_2="#50FA7B",
        radius=10,
        font_size_px=13,
        fluent=False,
    ),
    "Fusion Dark (Graphite)": ThemeSpec(
        name="Fusion Dark (Graphite)",
        palette=_GRAPHITE,
        accent="#4EA1FF",
        accent_2="#36D399",
        radius=10,
        font_size_px=13,
        fluent=True,
    ),
    "Fusion Dark (Fluent 11)": ThemeSpec(
        name="Fusion Dark (Fluent 11)",
        palette=_GRAPHITE,
        accent="#4EA1FF",
        accent_2="#9B8CFF",
        radius=12,
        font_size_px=13,
        fluent=True,
    ),
    "Fusion Light (Solarized)": ThemeSpec(
        name="Fusion Light (Solarized)",
        palette=_SOLARIZED_LIGHT,
        accent="#268BD2",
        accent_2="#2AA198",
        radius=10,
        font_size_px=13,
        fluent=False,
    ),
    "Fusion Light (Paper)": ThemeSpec(
        name="Fusion Light (Paper)",
        palette=_PAPER,
        accent="#2B7FFF",
        accent_2="#00B894",
        radius=10,
        font_size_px=13,
        fluent=False,
    ),
    "Fusion Light (Clean)": ThemeSpec(
        name="Fusion Light (Clean)",
        palette=_CLEAN,
        accent="#3A86FF",
        accent_2="#00B894",
        radius=10,
        font_size_px=13,
        fluent=False,
    ),
    "Fusion Light (Blue)": ThemeSpec(
        name="Fusion Light (Blue)",
        palette=_CLEAN_BLUE,
        accent="#2B7FFF",
        accent_2="#00B894",
        radius=10,
        font_size_px=13,
        fluent=False,
    ),
    "Fusion Light (Fluent 11)": ThemeSpec(
        name="Fusion Light (Fluent 11)",
        palette=_PAPER,
        accent="#2B7FFF",
        accent_2="#7C5CFF",
        radius=12,
        font_size_px=13,
        fluent=True,
    ),
}


# --------------------------------------------------------------------------------------
# Palette application (the key fix for "letters invisible")
# --------------------------------------------------------------------------------------

def _apply_palette(app: QApplication, p: PaletteSpec) -> None:
    pal = QPalette()

    window = QColor(p.window)
    base = QColor(p.base)
    alt = QColor(p.alt_base)
    text = QColor(p.text)
    dis = QColor(p.disabled_text)
    button = QColor(p.button)
    btn_text = QColor(p.button_text)
    hi = QColor(p.highlight)
    hi_text = QColor(p.highlighted_text)
    link = QColor(p.link)
    tip_base = QColor(p.tooltip_base)
    tip_text = QColor(p.tooltip_text)

    pal.setColor(QPalette.Window, window)
    pal.setColor(QPalette.WindowText, text)

    pal.setColor(QPalette.Base, base)
    pal.setColor(QPalette.AlternateBase, alt)
    pal.setColor(QPalette.Text, text)

    pal.setColor(QPalette.Button, button)
    pal.setColor(QPalette.ButtonText, btn_text)

    pal.setColor(QPalette.Highlight, hi)
    pal.setColor(QPalette.HighlightedText, hi_text)

    pal.setColor(QPalette.Link, link)

    pal.setColor(QPalette.ToolTipBase, tip_base)
    pal.setColor(QPalette.ToolTipText, tip_text)

    # Disabled
    pal.setColor(QPalette.Disabled, QPalette.WindowText, dis)
    pal.setColor(QPalette.Disabled, QPalette.Text, dis)
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, dis)

    app.setPalette(pal)


# --------------------------------------------------------------------------------------
# QSS builder (more "effectful" look, Fluent-like option)
# --------------------------------------------------------------------------------------

def _build_qss(t: ThemeSpec) -> str:
    p = t.palette
    radius = t.radius
    accent = t.accent
    accent2 = t.accent_2

    # spacing
    pad_y = 7 if not t.dense else 5
    pad_x = 10 if not t.dense else 8

    # A subtle "Fluent-ish" appearance: rounded controls, softer borders, better focus
    # Note: Qt can't do real Windows 11 acrylic; we emulate with clean shapes/colors.
    fluent_extra = ""
    if t.fluent:
        fluent_extra = f"""
/* Fluent-like details */
QToolBar {{
    background: {p.window};
    border: none;
    spacing: 6px;
    padding: 6px;
}}
QToolButton {{
    padding: 6px 10px;
    border-radius: {radius}px;
}}
QToolButton:hover {{
    background: {p.alt_base};
}}
QToolButton:checked {{
    background: {accent};
    color: {p.highlighted_text};
}}
QMenuBar {{
    background: {p.window};
    padding: 4px 6px;
}}
QMenuBar::item {{
    padding: 6px 10px;
    border-radius: {radius}px;
}}
QMenuBar::item:selected {{
    background: {p.alt_base};
}}
"""

    return f"""
/* Global */
* {{
    font-size: {t.font_size_px}px;
}}
QWidget {{
    background: {p.window};
    color: {p.text};
}}
QGroupBox {{
    border: 1px solid {p.border};
    border-radius: {radius}px;
    margin-top: 10px;
    padding: 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: {p.text};
}}
QLabel {{
    color: {p.text};
}}

/* Inputs */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit, QDateTimeEdit, QComboBox {{
    background: {p.base};
    color: {p.text};
    border: 1px solid {p.border};
    border-radius: {radius}px;
    padding: {pad_y}px {pad_x}px;
    selection-background-color: {p.highlight};
    selection-color: {p.highlighted_text};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 2px solid {accent};
    padding: {max(0, pad_y-1)}px {max(0, pad_x-1)}px;
}}

/* Combo */
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    background: {p.base};
    color: {p.text};
    border: 1px solid {p.border};
    selection-background-color: {p.highlight};
    selection-color: {p.highlighted_text};
    outline: 0;
}}

/* Buttons */
QPushButton {{
    background: {p.button};
    color: {p.button_text};
    border: 1px solid {p.border};
    border-radius: {radius}px;
    padding: {pad_y}px {pad_x}px;
}}
QPushButton:hover {{
    border-color: {accent2};
}}
QPushButton:pressed {{
    background: {p.alt_base};
}}
QPushButton:disabled {{
    color: {p.disabled_text};
    border-color: {p.border};
}}
QPushButton#primary, QPushButton[primary="true"] {{
    background: {accent};
    color: {p.highlighted_text};
    border: 1px solid {accent};
}}
QPushButton#primary:hover, QPushButton[primary="true"]:hover {{
    background: {accent2};
    border-color: {accent2};
}}

/* Tabs */
QTabWidget::pane {{
    border: 1px solid {p.border};
    border-radius: {radius}px;
    top: -1px;
}}
QTabBar::tab {{
    background: {p.window};
    color: {p.text};
    border: 1px solid {p.border};
    border-bottom: none;
    padding: 8px 12px;
    border-top-left-radius: {radius}px;
    border-top-right-radius: {radius}px;
    margin-right: 4px;
}}
QTabBar::tab:selected {{
    background: {p.base};
    border-color: {accent};
}}
QTabBar::tab:hover {{
    border-color: {accent2};
}}

/* Tables */
QTableView {{
    background: {p.base};
    alternate-background-color: {p.alt_base};
    gridline-color: {p.border};
    color: {p.text};
    border: 1px solid {p.border};
    border-radius: {radius}px;
}}
QHeaderView::section {{
    background: {p.alt_base};
    color: {p.text};
    border: none;
    padding: 6px 8px;
}}
QTableView::item:selected {{
    background: {p.highlight};
    color: {p.highlighted_text};
}}

/* Scrollbars (clean) */
QScrollBar:vertical {{
    background: transparent;
    width: 12px;
    margin: 6px 4px 6px 4px;
}}
QScrollBar::handle:vertical {{
    background: {p.border};
    border-radius: 6px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {accent};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 12px;
    margin: 4px 6px 4px 6px;
}}
QScrollBar::handle:horizontal {{
    background: {p.border};
    border-radius: 6px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {accent};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* Menus */
QMenu {{
    background: {p.base};
    color: {p.text};
    border: 1px solid {p.border};
    border-radius: {radius}px;
    padding: 6px;
}}
QMenu::item {{
    padding: 6px 10px;
    border-radius: {radius}px;
}}
QMenu::item:selected {{
    background: {p.highlight};
    color: {p.highlighted_text};
}}

/* Tooltips */
QToolTip {{
    background: {p.tooltip_base};
    color: {p.tooltip_text};
    border: 1px solid {p.border};
    padding: 6px;
    border-radius: {radius}px;
}}

{fluent_extra}
"""


# --------------------------------------------------------------------------------------
# Optional helper
# --------------------------------------------------------------------------------------

def default_theme() -> str:
    """If you need a consistent default, call this."""
    return "Fusion Light (Fluent 11)"
