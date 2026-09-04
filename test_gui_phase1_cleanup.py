import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

import gui
from core import KeyframeMetadata, SearchResult


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class DummyConfig:
    config = {"network": {}}
    network = {}

    def get(self, _key, default=None):
        return default


class DummySystem:
    def __init__(self):
        self.logger = DummyLogger()
        self.config = DummyConfig()
        self.status = SimpleNamespace(is_ready=True, index_loaded=True)
        self.faiss_retriever = SimpleNamespace(
            index=None,
            dimension=512,
            index_clip_model="openai/clip-vit-base-patch32",
            index_build_source="btc_clip_features",
        )
        self.active_query_encoder = SimpleNamespace(model_path="openai/clip-vit-base-patch32")
        self.received_options = None
        self.search_results = []

    def search(self, _query, options):
        self.received_options = options
        return self.search_results

    def _mark_unified_ready(self):
        self.status.is_ready = True
        self.status.index_loaded = True


class DummySignal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)


class CapturingWorker:
    instances = []

    def __init__(self, fn, *args):
        self.fn = fn
        self.args = args
        self.progress_updated = DummySignal()
        self.result_ready = DummySignal()
        self.error_occurred = DummySignal()
        CapturingWorker.instances.append(self)

    def start(self):
        pass

    def isRunning(self):
        return False

    def cancel(self):
        pass

    def wait(self, *_args):
        return True

    def terminate(self):
        pass


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def main_window(qt_app):
    window = gui.MainWindow(DummySystem())
    yield window
    window.close()


def _tab_texts(tab_widget):
    return [tab_widget.tabText(i) for i in range(tab_widget.count())]


def test_gui_phase1_startup_removed_components(main_window):
    top_tabs = " ".join(_tab_texts(main_window.tab_widget))
    result_tabs = " ".join(_tab_texts(main_window.results_widget.csv_tab_widget))

    assert "Search Results" in top_tabs
    assert "System Info" in top_tabs
    assert "Chat Interface" not in top_tabs
    assert "Network" not in top_tabs
    assert "Image Info" in result_tabs
    assert "CSV List" in result_tabs
    assert "Search Index" not in result_tabs

    removed_attrs = [
        "mode_combo",
        "explanations_checkbox",
        "hide_displayed_checkbox",
        "multi_scene_checkbox",
        "vietnamese_processing_checkbox",
        "trake_processing_checkbox",
        "stats_table",
        "filter_button",
        "optimize_button",
        "cleanup_button",
    ]
    for attr in removed_attrs:
        assert not hasattr(main_window, attr)

    assert main_window.chat_widget is None
    assert main_window.network_widget is None
    assert main_window.network_client is None


def test_gui_phase1_fixed_clip_only_mode_and_defaults(main_window, monkeypatch):
    CapturingWorker.instances.clear()
    monkeypatch.setattr(gui, "WorkerThread", CapturingWorker)

    main_window.video_grouping_checkbox.setChecked(True)
    main_window.max_per_video_spinbox.setValue(3)
    main_window.diversity_threshold_spinbox.setValue(0.5)
    main_window.grouping_strategy_combo.setCurrentText("weighted")
    main_window.temporal_checkbox.setChecked(False)

    main_window._perform_search("test query")

    worker = CapturingWorker.instances[-1]
    options = worker.args[1]

    assert options.mode == "clip_only"
    assert options.include_explanations is False
    assert options.include_temporal_context is False
    assert options.enable_multi_scene_parsing is False
    assert options.enable_vietnamese_processing is False
    assert options.enable_trake_processing is False
    assert options.enable_video_grouping is True
    assert options.max_results_per_video == 3
    assert options.diversity_threshold == 0.5
    assert options.video_grouping_strategy == "weighted"


def test_gui_phase1_fallback_search_options_are_clip_only(main_window, monkeypatch):
    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "system_v4_integration":
            raise ImportError("forced fallback")
        return real_import(name, *args, **kwargs)

    CapturingWorker.instances.clear()
    monkeypatch.setattr(gui, "WorkerThread", CapturingWorker)
    monkeypatch.setattr("builtins.__import__", guarded_import)

    main_window._perform_search("fallback query")

    options = CapturingWorker.instances[-1].args[1]
    assert options.mode == "clip_only"
    assert options.include_explanations is False


def test_gui_phase1_search_completion_no_removed_widget_access(main_window):
    main_window._on_search_completed_with_translation(([], None), "missing_progress")
    assert main_window.current_results == []


def test_gui_phase1_public_search_path_without_chat_or_network(main_window):
    metadata = KeyframeMetadata(
        folder_name="L26_V041",
        image_name="087",
        frame_id=4368,
        file_path="missing.jpg",
    )
    result = SearchResult(metadata=metadata, similarity_score=0.33, rank=1)
    main_window.system.search_results = [result]

    options = gui.SearchOptions(mode="clip_only", limit=1, cache_results=False)
    results, translation_info = main_window._search_with_translation_info("grapes", options)
    main_window._on_search_completed_with_translation((results, translation_info), "missing_progress")

    assert results == [result]
    assert main_window.current_results == [result]
    assert main_window.system.received_options.mode == "clip_only"


def test_gui_phase1_result_display_csv_and_nearby_wiring(main_window, monkeypatch):
    messages = []
    monkeypatch.setattr(gui.QMessageBox, "information", lambda *args, **kwargs: messages.append(args))
    monkeypatch.setattr(gui.QMessageBox, "warning", lambda *args, **kwargs: messages.append(args))
    monkeypatch.setattr(gui.QMessageBox, "critical", lambda *args, **kwargs: messages.append(args))

    metadata = KeyframeMetadata(
        folder_name="L26_V041",
        image_name="087",
        frame_id=4368,
        file_path="missing.jpg",
    )
    nearby = KeyframeMetadata(
        folder_name="L26_V041",
        image_name="088",
        frame_id=4382,
        file_path="nearby.jpg",
    )
    result = SearchResult(metadata=metadata, similarity_score=0.33, ranking_score=1.2, rank=1)

    main_window.system.get_surrounding_frames = lambda *_args, **_kwargs: [nearby]
    main_window.results_widget.system = main_window.system
    main_window.results_widget.csv_results = []

    main_window.results_widget._add_csv_list_item(result)
    main_window.results_widget.csv_results.append(result)
    main_window.results_widget._update_csv_tab_title()

    assert main_window.results_widget.csv_results_list.rowCount() == 1
    assert "CSV List (1)" in main_window.results_widget.csv_tab_widget.tabText(1)
    assert result.metadata.frame_id == 4368

    main_window.results_widget.add_surrounding_frames_to_csv(metadata)

    assert len(main_window.results_widget.csv_results) == 2
    added = main_window.results_widget.csv_results[-1]
    assert added.metadata.folder_name == "L26_V041"
    assert added.metadata.frame_id == 4382


def test_gui_phase1_system_info_without_stats_table(main_window):
    assert not hasattr(main_window, "stats_table")
    main_window._update_system_info()
    main_window._refresh_stats()
    assert main_window.system_status_label.text() == "Ready"
    assert main_window.index_status_label.text() == "Loaded"


def test_gui_phase1_menu_actions_removed(main_window):
    action_texts = []
    for menu_action in main_window.menuBar().actions():
        menu = menu_action.menu()
        if menu:
            action_texts.extend(action.text() for action in menu.actions())

    combined = " ".join(action_texts)
    assert "Build System" in combined
    assert "Smart Load" in combined
    assert "Optimize System" not in combined
    assert "Cleanup System" not in combined
    assert "Refresh Statistics" not in combined
    assert "Refresh" not in combined


def test_gui_phase1_smart_load_and_build_keep_runtime_owner_calls():
    source = open("gui.py", encoding="utf-8-sig").read()
    assert "runtime_system = get_runtime_system(self.system)" in source
    assert "unified_builder = add_unified_index_support(runtime_system)" in source
    assert "add_unified_index_support(get_runtime_system(self.system))" in source


def test_gui_phase1_unified_load_targets_runtime_base(main_window, monkeypatch):
    import unified_builder

    base = DummySystem()
    wrapper = SimpleNamespace(base_system=base)
    calls = {}

    def fake_get_runtime_system(system):
        calls["get_arg"] = system
        return system.base_system

    def fake_add_unified_index_support(system):
        calls["add_arg"] = system
        return SimpleNamespace(load_unified_index_fast=lambda _path: {"total_vectors": 1})

    CapturingWorker.instances.clear()
    main_window.system = wrapper
    monkeypatch.setattr(gui, "WorkerThread", CapturingWorker)
    monkeypatch.setattr(unified_builder, "get_runtime_system", fake_get_runtime_system)
    monkeypatch.setattr(unified_builder, "add_unified_index_support", fake_add_unified_index_support)

    main_window._load_unified_index("synthetic_btc_clip32.rvdb")

    assert calls["get_arg"] is wrapper
    assert calls["add_arg"] is base
    assert CapturingWorker.instances
