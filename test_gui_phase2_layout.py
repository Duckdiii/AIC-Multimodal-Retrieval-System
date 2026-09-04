import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtWidgets import QApplication

import gui
from core import KeyframeMetadata, SearchResult
from test_gui_phase1_cleanup import DummySystem, CapturingWorker


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def main_window(qt_app):
    window = gui.MainWindow(DummySystem())
    yield window
    window.close()


def _metadata(video="L26_V041", local_idx=87, frame_id=4368):
    metadata = KeyframeMetadata(
        folder_name=video,
        image_name=f"{local_idx:03d}.jpg",
        frame_id=frame_id,
        file_path=f"{video}/{local_idx:03d}.jpg",
    )
    metadata.local_keyframe_idx = local_idx
    return metadata


def _messages(monkeypatch):
    messages = []
    monkeypatch.setattr(gui.QMessageBox, "information", lambda *args, **kwargs: messages.append(("info", args)))
    monkeypatch.setattr(gui.QMessageBox, "warning", lambda *args, **kwargs: messages.append(("warning", args)))
    monkeypatch.setattr(gui.QMessageBox, "critical", lambda *args, **kwargs: messages.append(("critical", args)))
    monkeypatch.setattr(gui.QMessageBox, "question", lambda *args, **kwargs: gui.QMessageBox.Yes)
    return messages


def test_phase2_search_result_gallery_cards_select_and_show_real_frame(main_window):
    result = SearchResult(
        metadata=_metadata(frame_id=4368),
        similarity_score=0.314,
        ranking_score=0.875,
        rank=1,
    )

    main_window.results_widget.display_results([result])

    assert len(main_window.results_widget.result_cards) == 1
    card = main_window.results_widget.result_cards[0]
    labels = " ".join(label.text() for label in card.findChildren(gui.QLabel))
    assert "L26_V041" in labels
    assert "Frame: 4368" in labels
    assert "Sim: 0.314" in labels

    main_window.results_widget._on_thumbnail_clicked(0)
    info = main_window.results_widget.metadata_text.toPlainText()
    assert "Video ID: L26_V041" in info
    assert "Keyframe: 87" in info
    assert "Frame ID: 4368" in info
    assert "Similarity: 0.314" in info
    assert "Ranking: 0.875" in info


def test_phase2_right_click_nearby_handler_remains_available(main_window, monkeypatch):
    result = SearchResult(metadata=_metadata(), similarity_score=0.31, rank=1)
    main_window.results_widget.display_results([result])
    thumbnail = main_window.results_widget.result_cards[0].findChild(gui.ClickableThumbnail)

    called = {}
    monkeypatch.setattr(main_window.results_widget, "show_surrounding_frames", lambda metadata: called.setdefault("metadata", metadata))

    thumbnail._handle_show_surrounding_frames()

    assert called["metadata"].frame_id == 4368
    source = Path("gui.py").read_text(encoding="utf-8-sig")
    assert "Nearby Frames" in source


class NearbySystem(DummySystem):
    def __init__(self):
        super().__init__()
        self.calls = []

    def get_asymmetric_surrounding_frames(self, folder_name, image_name, before_window, after_window):
        self.calls.append((before_window, after_window))
        center = int(Path(image_name).stem)
        frames = []
        for local_idx in range(center - before_window, center + after_window + 1):
            if local_idx <= 0:
                continue
            frames.append(_metadata(folder_name, local_idx, local_idx * 50))
        frames.append(_metadata(folder_name, center, center * 50))
        return frames


@pytest.fixture
def nearby_widget(qt_app, monkeypatch):
    _messages(monkeypatch)
    monkeypatch.setattr(gui.QDialog, "exec_", lambda self: None)
    monkeypatch.setattr(gui.ResultDisplayWidget, "_load_thumbnail_lazy", lambda self, label: None)
    monkeypatch.setattr(gui.ResultDisplayWidget, "_get_memory_usage_info", lambda self: "RAM: test")
    widget = gui.ResultDisplayWidget(system=NearbySystem())
    yield widget
    widget.close()


def _nearby_frame_ids(widget):
    return [frame.frame_id for frame in widget.frame_data]


def test_phase2_nearby_initial_load_order_and_current_highlight(nearby_widget):
    current = _metadata(local_idx=87, frame_id=4350)
    nearby_widget.show_surrounding_frames(current)

    ids = _nearby_frame_ids(nearby_widget)
    assert ids == sorted(set(ids))
    assert 4350 in ids
    highlighted = [
        nearby_widget.scroll_layout.itemAt(i).widget()
        for i in range(nearby_widget.scroll_layout.count())
        if nearby_widget.scroll_layout.itemAt(i).widget().property("is_current_frame")
    ]
    assert len(highlighted) == 1


def test_phase2_load_before_and_after_are_iterative_no_duplicates(nearby_widget):
    nearby_widget.show_surrounding_frames(_metadata(local_idx=87, frame_id=4350))
    initial_ids = _nearby_frame_ids(nearby_widget)

    nearby_widget._load_more_frames("before", 10)
    after_before_once = _nearby_frame_ids(nearby_widget)
    nearby_widget._load_more_frames("before", 10)
    after_before_twice = _nearby_frame_ids(nearby_widget)
    nearby_widget._load_more_frames("after", 10)
    after_after_once = _nearby_frame_ids(nearby_widget)
    nearby_widget._load_more_frames("after", 10)
    after_after_twice = _nearby_frame_ids(nearby_widget)

    assert min(after_before_once) < min(initial_ids)
    assert min(after_before_twice) < min(after_before_once)
    assert max(after_after_once) > max(after_before_twice)
    assert max(after_after_twice) > max(after_after_once)
    assert after_after_twice == sorted(set(after_after_twice))


def test_phase2_nearby_csv_uses_real_frame_id(nearby_widget, monkeypatch):
    _messages(monkeypatch)
    nearby_widget.show_surrounding_frames(_metadata(local_idx=87, frame_id=4350))

    nearby_widget.frame_checkboxes[0].setChecked(True)
    nearby_widget.frame_checkboxes[-1].setChecked(True)
    nearby_widget._add_selected_frames_to_csv(SimpleNamespace(close=lambda: None))

    exported_frame_ids = [result.metadata.frame_id for result in nearby_widget.csv_results]
    assert exported_frame_ids == [min(exported_frame_ids), max(exported_frame_ids)]
    assert all(frame_id % 50 == 0 for frame_id in exported_frame_ids)


def test_phase2_main_csv_export_uses_real_frame_id(main_window, monkeypatch, tmp_path):
    _messages(monkeypatch)
    monkeypatch.setattr(gui.os, "getcwd", lambda: str(tmp_path))

    result = SearchResult(metadata=_metadata(local_idx=87, frame_id=4368), similarity_score=0.314, rank=1)
    main_window.results_widget.display_results([result])
    main_window.results_widget.result_checkboxes[0].setChecked(True)
    main_window.results_widget._add_to_csv()
    main_window.results_widget._export_csv_list()

    csv_files = list((tmp_path / "result").glob("submission_*.csv"))
    assert len(csv_files) == 1
    assert csv_files[0].read_text(encoding="utf-8") == "L26_V041,4368"


def test_phase2_zoom_buttons_do_not_crash(main_window):
    main_window.results_widget.preview_widget.zoom_in()
    main_window.results_widget.preview_widget.zoom_out()
    assert main_window.results_widget.preview_widget.scale_factor >= 0.1
    assert main_window.results_widget.preview_widget.zoom_label.text().endswith("%")


def test_phase2_search_and_grouping_defaults_still_match_phase1(main_window, monkeypatch):
    CapturingWorker.instances.clear()
    monkeypatch.setattr(gui, "WorkerThread", CapturingWorker)

    main_window.video_grouping_checkbox.setChecked(True)
    main_window.max_per_video_spinbox.setValue(3)
    main_window.diversity_threshold_spinbox.setValue(0.5)
    main_window.grouping_strategy_combo.setCurrentText("weighted")
    main_window.temporal_checkbox.setChecked(False)

    main_window._perform_search("competition query")

    options = CapturingWorker.instances[-1].args[1]
    assert options.mode == "clip_only"
    assert options.enable_video_grouping is True
    assert options.max_results_per_video == 3
    assert options.diversity_threshold == 0.5
    assert options.video_grouping_strategy == "weighted"
    assert options.include_temporal_context is False
