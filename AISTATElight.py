from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import QApplication, QSplashScreen
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPixmap, QGuiApplication

from gui_pyside import MainWindow


SPLASH_MS = 3000
SPLASH_SCALE = 0.60  # <-- OPCJA 1: procent ekranu (np. 0.35 / 0.50 / 0.75)


def _find_logo_path() -> str | None:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "ui", "assets", "logo.png"),
        os.path.join(here, "ui", "assets", "logo.jpg"),
        os.path.join(here, "ui", "assets", "logo.jpeg"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _scale_to_screen_percent(pix: QPixmap, screen_w: int, screen_h: int, scale: float) -> QPixmap:
    """Scale pixmap to a % of screen size (contain, no crop)."""
    if pix.isNull() or screen_w <= 0 or screen_h <= 0:
        return pix

    scale = max(0.05, min(scale, 1.0))
    target_w = int(screen_w * scale)
    target_h = int(screen_h * scale)

    return pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def main() -> None:
    app = QApplication(sys.argv)

    splash = None
    logo_path = _find_logo_path()
    if logo_path:
        pix = QPixmap(logo_path)
        if not pix.isNull():
            screen = QGuiApplication.primaryScreen()
            geom = screen.geometry() if screen else None

            if geom:
                pix = _scale_to_screen_percent(pix, geom.width(), geom.height(), SPLASH_SCALE)

            splash = QSplashScreen(pix)
            splash.setWindowFlag(Qt.FramelessWindowHint, True)

            # Wyśrodkuj na ekranie
            if geom:
                splash.resize(pix.size())
                splash.move(
                    geom.x() + (geom.width() - pix.width()) // 2,
                    geom.y() + (geom.height() - pix.height()) // 2,
                )

            splash.show()
            app.processEvents()

    win = MainWindow(app=app)

    def show_main():
        win.show()
        if splash is not None:
            splash.finish(win)

    if splash is not None:
        QTimer.singleShot(SPLASH_MS, show_main)
    else:
        show_main()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
