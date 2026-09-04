import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QAction, QApplication, QDialog, QPushButton

import gui
from test_gui_phase1_cleanup import DummySystem
from test_gui_phase2_layout import NearbySystem, _metadata, _messages


DECORATIVE_MARKERS = (
    "🔍", "📁", "🔧", "📊", "📋", "📄", "📝", "➕", "➖", "⚙", "▶",
    "⏪", "⏩", "⬅", "➡", "✅", "❌", "🗑", "🔄", "📷", "📹", "🎬",
    "📈", "💬", "🌐", "🚀", "⚡", "🎯", "📦", "💾", "🖼", "📏", "💡",
    "🔎", "✕", "👁", "⬇", "🛑", "🔗", "📤", "🎉",
)


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def main_window(qt_app):
    window = gui.MainWindow(DummySystem())
    yield window
    window.close()


def test_phase21_active_controls_and_actions_are_text_only(main_window):
    # Spin boxes create internal Qt tool buttons for their standard arrows. The
    # application-owned command buttons and actions must not assign icons.
    for button in main_window.findChildren(QPushButton):
        assert button.icon().isNull(), button.text()
        assert not any(marker in button.text() for marker in DECORATIVE_MARKERS)

    for action in main_window.findChildren(QAction):
        assert action.icon().isNull(), action.text()
        assert not any(marker in action.text() for marker in DECORATIVE_MARKERS)


def test_phase21_static_icon_audit_is_clean():
    source = Path("gui.py").read_text(encoding="utf-8-sig")

    forbidden_apis = ("QIcon", "setIcon(", "standardIcon", "QStyle.SP_", "setWindowIcon")
    assert all(token not in source for token in forbidden_apis)
    assert not any(marker in source for marker in DECORATIVE_MARKERS)


def test_phase21_blue_theme_and_primary_actions(main_window):
    stylesheet = main_window.styleSheet()
    for color in ("#1F6FEB", "#EAF3FF", "#F7F9FC", "#FFFFFF"):
        assert color in stylesheet

    assert main_window.theme.hover_color == "#1858B8"
    assert main_window.theme.pressed_color == "#164A9A"

    for button in (
        main_window.search_button,
        main_window.build_button,
        main_window.smart_load_button,
        main_window.results_widget.add_csv_btn,
        main_window.results_widget.export_csv_btn,
    ):
        assert "#1F6FEB" in button.styleSheet()
        assert button.icon().isNull()


def test_phase21_header_uses_balanced_brand_logo(main_window):
    assert hasattr(main_window, "header_logo_label")
    assert main_window.header_logo_label.width() == 52
    assert main_window.header_logo_label.height() == 52
    assert main_window.header_logo_label.pixmap() is not None
    assert not main_window.header_logo_label.pixmap().isNull()
    assert main_window.header_logo_label.pixmap().width() <= 44
    assert main_window.header_logo_label.pixmap().height() <= 44


def test_phase21_zoom_controls_are_text_only_and_connected(main_window):
    preview = main_window.results_widget.preview_widget
    assert preview.zoom_out_btn.text() == "-"
    assert preview.zoom_in_btn.text() == "+"
    assert preview.fit_btn.text() == "Fit"
    assert preview.zoom_out_btn.icon().isNull()
    assert preview.zoom_in_btn.icon().isNull()

    assert preview.zoom_in_btn.receivers(preview.zoom_in_btn.clicked) > 0
    assert preview.zoom_out_btn.receivers(preview.zoom_out_btn.clicked) > 0


def test_phase21_nearby_actions_are_text_only_and_connected(qt_app, monkeypatch):
    _messages(monkeypatch)
    monkeypatch.setattr(QDialog, "exec_", lambda self: None)
    monkeypatch.setattr(gui.ResultDisplayWidget, "_load_thumbnail_lazy", lambda self, label: None)
    monkeypatch.setattr(gui.ResultDisplayWidget, "_get_memory_usage_info", lambda self: "RAM: test")

    widget = gui.ResultDisplayWidget(system=NearbySystem())
    try:
        widget.show_surrounding_frames(_metadata(local_idx=87, frame_id=4350))
        buttons = {button.text(): button for button in widget.current_dialog.findChildren(QPushButton)}

        expected = {
            "Load Before", "Load After", "Select All", "Clear Selection",
            "Add Selected to CSV", "Close",
        }
        assert expected.issubset(buttons)
        assert all(button.icon().isNull() for button in buttons.values())
        assert buttons["Load Before"].receivers(buttons["Load Before"].clicked) > 0
        assert buttons["Load After"].receivers(buttons["Load After"].clicked) > 0
        assert buttons["Add Selected to CSV"].receivers(buttons["Add Selected to CSV"].clicked) > 0

        widget.frame_checkboxes[0].setChecked(True)
        assert "#1F6FEB" in widget.frame_checkboxes[0].frame_widget.styleSheet()
        assert "#EAF3FF" in widget.frame_checkboxes[0].frame_widget.styleSheet()
    finally:
        widget.close()


def test_phase21_phase1_removals_and_runtime_ownership_remain_intact(main_window):
    removed_attrs = (
        "mode_combo", "explanations_checkbox", "hide_displayed_checkbox",
        "multi_scene_checkbox", "vietnamese_processing_checkbox",
        "trake_processing_checkbox", "stats_table",
    )
    assert all(not hasattr(main_window, attr) for attr in removed_attrs)

    source = Path("gui.py").read_text(encoding="utf-8-sig")
    assert 'QAction("Nearby Frames", self)' in source
    assert "runtime_system = get_runtime_system(self.system)" in source
    assert "unified_builder = add_unified_index_support(runtime_system)" in source
