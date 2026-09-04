from dataclasses import dataclass
from contextlib import contextmanager

import pytest
import numpy as np

from core import FAISSRetriever, KeyframeMetadata, SearchResult, resolve_frame_id_for_keyframe
from competition_csv_generator import (
    CompetitionCSVGenerator,
    CompetitionQuery,
    QueryType,
)
from unified_builder import UnifiedBuilderIntegration, add_unified_index_support, get_runtime_system
from unified_index import UnifiedIndex
from system import EnhancedRetrievalSystem, SearchOptions, SystemStatus
from system_v4_integration import EnhancedRetrievalSystemV4, EnhancedSearchOptions


@dataclass
class DummyMetadata:
    folder_name: str
    image_name: str
    frame_id: int


@dataclass
class DummySearchResult:
    metadata: DummyMetadata
    similarity_score: float = 0.95


class DummyLogger:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.infos = []

    def error(self, message, **kwargs):
        self.errors.append((message, kwargs))

    def info(self, message, **kwargs):
        self.infos.append((message, kwargs))

    def warning(self, message, **kwargs):
        self.warnings.append((message, kwargs))

    def debug(self, *args, **kwargs):
        pass


class DummyConfig:
    def get(self, _key, default=None):
        return default


class DummyPerfMonitor:
    @contextmanager
    def timer(self, *_args, **_kwargs):
        yield None


class DummyCache:
    def get_cached_results(self, *_args, **_kwargs):
        return None

    def cache_query_results(self, *_args, **_kwargs):
        pass


class DummyMetadataManager:
    def __init__(self):
        self.temporal_index = {}

    def get_temporal_neighbors(self, *_args, **_kwargs):
        return []


class DummyLargeProcessor:
    model_path = "openai/clip-vit-large-patch14"
    model = type("Model", (), {"config": type("Config", (), {"projection_dim": 768})()})()

    def __init__(self):
        self.last_shape = None

    def encode_text(self, *_args, **_kwargs):
        vector = np.zeros((1, 768), dtype=np.float32)
        vector[0, 0] = 1.0
        self.last_shape = vector.shape
        return vector


class DummyB32Processor:
    instances = []

    def __init__(self, model_path, *_args, **_kwargs):
        self.model_path = model_path
        self.model = type("Model", (), {"config": type("Config", (), {"projection_dim": 512})()})()
        self.last_shape = None
        DummyB32Processor.instances.append(self)

    def encode_text(self, *_args, **_kwargs):
        vector = np.zeros((1, 512), dtype=np.float32)
        vector[0, 0] = 1.0
        self.last_shape = vector.shape
        return vector


def _make_base_system_for_search():
    system = EnhancedRetrievalSystem.__new__(EnhancedRetrievalSystem)
    system.verbose = False
    system.enable_validation = False
    system.status = SystemStatus(is_initialized=True, is_ready=True, index_loaded=True)
    system.config = DummyConfig()
    system.logger = DummyLogger()
    system.perf_monitor = DummyPerfMonitor()
    system.cache = DummyCache()
    system.query_translator = None
    system.clip_processor = DummyLargeProcessor()
    system.active_query_encoder = system.clip_processor
    system.faiss_retriever = FAISSRetriever(system.config, system.logger, cache=object())
    system.metadata_manager = DummyMetadataManager()
    system.llm_processor = None
    return system


def _create_synthetic_btc_index(tmp_path, rows=2):
    features_dir = tmp_path / "clip_features" / "clip-features-32"
    keyframe_dir = tmp_path / "Keyframes" / "Keyframes_L21" / "L21_V001"
    map_dir = tmp_path / "map-keyframes"
    features_dir.mkdir(parents=True)
    keyframe_dir.mkdir(parents=True)
    map_dir.mkdir(parents=True)

    features = np.zeros((rows, 512), dtype=np.float32)
    for i in range(rows):
        features[i, i % 512] = 1.0
        (keyframe_dir / f"{i + 1:03d}.jpg").write_bytes(b"fake")
    np.save(features_dir / "L21_V001.npy", features)

    lines = ["n,pts_time,fps,frame_idx"]
    for i in range(rows):
        lines.append(f"{i + 1},{float(i):.1f},30.0,{i * 90}")
    (map_dir / "L21_V001.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    output_file = tmp_path / "btc.rvdb"
    UnifiedIndex(logger=DummyLogger()).create_unified_index_from_btc_features(
        str(features_dir),
        str(tmp_path / "Keyframes"),
        str(map_dir),
        str(output_file),
        ["L21_V001"],
    )
    return output_file


def _competition_config():
    return {
        "competition": {
            "enabled": True,
            "strict_frame_mapping": True,
        },
        "retrieval": {
            "allow_filename_frame_id_fallback": True,
        },
    }


def test_competition_mapping_success_exports_real_frame_idx(tmp_path):
    frame_id = resolve_frame_id_for_keyframe(
        {"000123": 3876},
        "L21_V001",
        "000123",
        _competition_config(),
        DummyLogger(),
        "map-keyframes-aic25-b1/L21_V001.csv",
    )

    assert frame_id == 3876

    result = DummySearchResult(DummyMetadata("L21_V001", "000123", frame_id))
    generator = CompetitionCSVGenerator(output_dir=str(tmp_path))
    competition_results = generator.convert_search_results_to_competition_format(
        [result],
        query_id="Q001",
        query_type=QueryType.KIS,
    )

    query = CompetitionQuery(
        query_id="Q001",
        query_text="find keyframe",
        query_type=QueryType.KIS,
        results=competition_results,
        processing_time=0.0,
        v4_enhancements_used=[],
    )
    csv_path = generator.generate_csv_file(query, filename="submission.csv")

    assert competition_results[0].frame_id == "3876"
    assert open(csv_path, encoding="utf-8").read() == "L21_V001,3876\n"


def test_competition_mapping_fail_logs_error_and_does_not_export_filename(tmp_path):
    logger = DummyLogger()
    frame_id = resolve_frame_id_for_keyframe(
        {},
        "L21_V001",
        "000999",
        _competition_config(),
        logger,
        "map-keyframes-aic25-b1/L21_V001.csv",
    )

    assert frame_id == -1
    assert logger.errors
    assert logger.errors[0][0] == "MAPPING_ERROR"
    assert logger.errors[0][1]["folder_name"] == "L21_V001"
    assert logger.errors[0][1]["image_name"] == "000999"

    result = DummySearchResult(DummyMetadata("L21_V001", "000999", frame_id))
    generator = CompetitionCSVGenerator(output_dir=str(tmp_path))
    competition_results = generator.convert_search_results_to_competition_format(
        [result],
        query_id="Q002",
        query_type=QueryType.KIS,
    )

    query = CompetitionQuery(
        query_id="Q002",
        query_text="find keyframe",
        query_type=QueryType.KIS,
        results=competition_results,
        processing_time=0.0,
        v4_enhancements_used=[],
    )
    csv_path = generator.generate_csv_file(query, filename="submission_fail.csv")

    assert competition_results == []
    assert "999" not in open(csv_path, encoding="utf-8").read()


def test_unified_index_uses_mapping_for_real_frame_idx():
    unified_index = UnifiedIndex(logger=DummyLogger())

    frame_id = unified_index._resolve_frame_id(
        "L21_V001",
        "000123.jpg",
        {"L21_V001": {"000123": 3876}},
    )

    assert frame_id == 3876


def test_unified_index_mapping_fail_is_strict_when_csv_mapping_exists():
    logger = DummyLogger()
    unified_index = UnifiedIndex(logger=logger)

    frame_id = unified_index._resolve_frame_id(
        "L21_V001",
        "000999.jpg",
        {"L21_V001": {"000123": 3876}},
    )

    assert frame_id is None
    assert logger.errors
    assert logger.errors[0][0].startswith("MAPPING_ERROR")


def test_unified_scan_fast_mode_does_not_content_hash(tmp_path, monkeypatch):
    image_dir = tmp_path / "Keyframes_L21" / "L21_V001"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "001.jpg"
    image_path.write_bytes(b"fake-jpg-bytes")

    unified_index = UnifiedIndex(logger=DummyLogger())

    def fail_if_called(_file_path):
        raise AssertionError("content hash should not be used for fresh fast scan")

    monkeypatch.setattr(unified_index, "_calculate_file_hash", fail_if_called)
    inventory = unified_index._scan_files(str(tmp_path), hash_files=False)

    assert list(inventory) == ["Keyframes_L21\\L21_V001\\001.jpg"] or list(inventory) == ["Keyframes_L21/L21_V001/001.jpg"]
    assert inventory[list(inventory)[0]]["hash"].startswith("stat:")


def test_unified_scan_hash_mode_keeps_content_hash_for_resume(tmp_path, monkeypatch):
    image_dir = tmp_path / "Keyframes_L21" / "L21_V001"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "001.jpg"
    image_path.write_bytes(b"fake-jpg-bytes")

    unified_index = UnifiedIndex(logger=DummyLogger())

    monkeypatch.setattr(unified_index, "_calculate_file_hash", lambda _file_path: "sha256-prefix")
    inventory = unified_index._scan_files(str(tmp_path), hash_files=True)

    assert inventory[list(inventory)[0]]["hash"] == "sha256-prefix"


def test_btc_clip_features_build_uses_strict_csv_frame_idx(tmp_path):
    features_dir = tmp_path / "clip_features" / "clip-features-32"
    keyframe_dir = tmp_path / "Keyframes" / "Keyframes_L21" / "L21_V001"
    map_dir = tmp_path / "map-keyframes"
    features_dir.mkdir(parents=True)
    keyframe_dir.mkdir(parents=True)
    map_dir.mkdir(parents=True)

    (keyframe_dir / "001.jpg").write_bytes(b"fake")
    (keyframe_dir / "002.jpg").write_bytes(b"fake")
    np.save(features_dir / "L21_V001.npy", np.ones((2, 512), dtype=np.float32))
    (map_dir / "L21_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,30.0,0\n2,3.0,30.0,90\n",
        encoding="utf-8",
    )

    output_file = tmp_path / "btc.rvdb"
    unified_index = UnifiedIndex(logger=DummyLogger())
    audit_rows = unified_index.audit_btc_clip_features(
        str(features_dir),
        str(tmp_path / "Keyframes"),
        str(map_dir),
        ["L21_V001"],
    )

    assert audit_rows[0]["npy_rows"] == 2
    assert audit_rows[0]["mapped_rows"] == 2
    assert audit_rows[0]["missing_mapping"] == []
    assert audit_rows[0]["last_frame_id"] == 90

    stats = unified_index.create_unified_index_from_btc_features(
        str(features_dir),
        str(tmp_path / "Keyframes"),
        str(map_dir),
        str(output_file),
        ["L21_V001"],
    )

    loaded = UnifiedIndex(logger=DummyLogger())
    loaded.load_unified_index(str(output_file))

    assert stats["processed_files"] == 2
    assert loaded.faiss_index.d == 512
    assert loaded.metadata_list[0]["image_name"] == "001"
    assert loaded.metadata_list[0]["frame_id"] == 0
    assert loaded.metadata_list[1]["image_name"] == "002"
    assert loaded.metadata_list[1]["frame_id"] == 90
    assert len(loaded.metadata_list[0]["clip_features"]) == 512


def test_btc_unified_load_sets_512d_query_encoder_and_faiss_accepts(tmp_path, monkeypatch):
    features_dir = tmp_path / "clip_features" / "clip-features-32"
    keyframe_dir = tmp_path / "Keyframes" / "Keyframes_L21" / "L21_V001"
    map_dir = tmp_path / "map-keyframes"
    features_dir.mkdir(parents=True)
    keyframe_dir.mkdir(parents=True)
    map_dir.mkdir(parents=True)

    (keyframe_dir / "001.jpg").write_bytes(b"fake")
    (keyframe_dir / "002.jpg").write_bytes(b"fake")
    np.save(features_dir / "L21_V001.npy", np.eye(2, 512, dtype=np.float32))
    (map_dir / "L21_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,30.0,0\n2,3.0,30.0,90\n",
        encoding="utf-8",
    )

    output_file = tmp_path / "btc.rvdb"
    UnifiedIndex(logger=DummyLogger()).create_unified_index_from_btc_features(
        str(features_dir),
        str(tmp_path / "Keyframes"),
        str(map_dir),
        str(output_file),
        ["L21_V001"],
    )

    class OldLargeProcessor:
        model_path = "openai/clip-vit-large-patch14"
        model = type("Model", (), {"config": type("Config", (), {"projection_dim": 768})()})()

        def encode_text(self, *_args, **_kwargs):
            return np.ones((1, 768), dtype=np.float32)

    class DummyB32Processor:
        def __init__(self, model_path, *_args, **_kwargs):
            self.model_path = model_path
            self.model = type("Model", (), {"config": type("Config", (), {"projection_dim": 512})()})()

        def encode_text(self, *_args, **_kwargs):
            vector = np.zeros((1, 512), dtype=np.float32)
            vector[0, 0] = 1.0
            return vector

    class DummyMetadataManager:
        def __init__(self):
            self.temporal_index = {}

    class DummySystem:
        def __init__(self):
            self.config = DummyConfig()
            self.logger = DummyLogger()
            self.clip_processor = OldLargeProcessor()
            self.faiss_retriever = FAISSRetriever(self.config, self.logger, cache=object())
            self.metadata_manager = DummyMetadataManager()

    import core

    monkeypatch.setattr(core, "CLIPFeatureExtractor", DummyB32Processor)

    system = DummySystem()
    builder = UnifiedBuilderIntegration(system, system.logger)
    stats = builder.load_unified_index_fast(str(output_file))
    query_features = system.active_query_encoder.encode_text("motorcycle", validate_input=False)
    results = system.faiss_retriever.search(query_features, 1)

    assert stats["index_info"]["clip_model"] == "openai/clip-vit-base-patch32"
    assert system.clip_processor.model_path == "openai/clip-vit-base-patch32"
    assert query_features.shape == (1, 512)
    assert system.faiss_retriever.dimension == 512
    assert results


def test_public_clip_only_search_uses_active_btc_query_encoder(tmp_path, monkeypatch):
    features_dir = tmp_path / "clip_features" / "clip-features-32"
    keyframe_dir = tmp_path / "Keyframes" / "Keyframes_L21" / "L21_V001"
    map_dir = tmp_path / "map-keyframes"
    features_dir.mkdir(parents=True)
    keyframe_dir.mkdir(parents=True)
    map_dir.mkdir(parents=True)

    (keyframe_dir / "001.jpg").write_bytes(b"fake")
    (keyframe_dir / "002.jpg").write_bytes(b"fake")
    features = np.zeros((2, 512), dtype=np.float32)
    features[0, 0] = 1.0
    features[1, 1] = 1.0
    np.save(features_dir / "L21_V001.npy", features)
    (map_dir / "L21_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,30.0,0\n2,3.0,30.0,90\n",
        encoding="utf-8",
    )

    output_file = tmp_path / "btc.rvdb"
    UnifiedIndex(logger=DummyLogger()).create_unified_index_from_btc_features(
        str(features_dir),
        str(tmp_path / "Keyframes"),
        str(map_dir),
        str(output_file),
        ["L21_V001"],
    )

    class OldLargeProcessor:
        model_path = "openai/clip-vit-large-patch14"
        model = type("Model", (), {"config": type("Config", (), {"projection_dim": 768})()})()

        def encode_text(self, *_args, **_kwargs):
            return np.ones((1, 768), dtype=np.float32)

    class DummyB32Processor:
        instances = []

        def __init__(self, model_path, *_args, **_kwargs):
            self.model_path = model_path
            self.model = type("Model", (), {"config": type("Config", (), {"projection_dim": 512})()})()
            self.last_shape = None
            DummyB32Processor.instances.append(self)

        def encode_text(self, *_args, **_kwargs):
            vector = np.zeros((1, 512), dtype=np.float32)
            vector[0, 0] = 1.0
            self.last_shape = vector.shape
            return vector

    class DummyMetadataManager:
        def __init__(self):
            self.temporal_index = {}

    import core

    monkeypatch.setattr(core, "CLIPFeatureExtractor", DummyB32Processor)

    system = EnhancedRetrievalSystem.__new__(EnhancedRetrievalSystem)
    system.verbose = False
    system.enable_validation = False
    system.status = SystemStatus(is_initialized=True, is_ready=True, index_loaded=True)
    system.config = DummyConfig()
    system.logger = DummyLogger()
    system.perf_monitor = DummyPerfMonitor()
    system.cache = DummyCache()
    system.query_translator = None
    system.clip_processor = OldLargeProcessor()
    system.active_query_encoder = system.clip_processor
    system.faiss_retriever = FAISSRetriever(system.config, system.logger, cache=object())
    system.metadata_manager = DummyMetadataManager()
    system.llm_processor = None

    builder = UnifiedBuilderIntegration(system, system.logger)
    system.unified_builder = builder
    builder.load_unified_index_fast(str(output_file))

    options = SearchOptions(mode="clip_only", limit=1, cache_results=False, include_temporal_context=False)
    results = system.search("a man riding a motorcycle on the street", options)

    assert system.active_query_encoder.model_path == "openai/clip-vit-base-patch32"
    assert DummyB32Processor.instances[-1].last_shape == (1, 512)
    assert system.faiss_retriever.dimension == 512
    assert results


def test_public_hybrid_search_uses_active_btc_query_encoder(tmp_path, monkeypatch):
    features_dir = tmp_path / "clip_features" / "clip-features-32"
    keyframe_dir = tmp_path / "Keyframes" / "Keyframes_L21" / "L21_V001"
    map_dir = tmp_path / "map-keyframes"
    features_dir.mkdir(parents=True)
    keyframe_dir.mkdir(parents=True)
    map_dir.mkdir(parents=True)

    (keyframe_dir / "001.jpg").write_bytes(b"fake")
    np.save(features_dir / "L21_V001.npy", np.eye(1, 512, dtype=np.float32))
    (map_dir / "L21_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,30.0,0\n",
        encoding="utf-8",
    )

    output_file = tmp_path / "btc.rvdb"
    UnifiedIndex(logger=DummyLogger()).create_unified_index_from_btc_features(
        str(features_dir),
        str(tmp_path / "Keyframes"),
        str(map_dir),
        str(output_file),
        ["L21_V001"],
    )

    class DummyB32Processor:
        instances = []

        def __init__(self, model_path, *_args, **_kwargs):
            self.model_path = model_path
            self.model = type("Model", (), {"config": type("Config", (), {"projection_dim": 512})()})()
            self.last_shape = None
            DummyB32Processor.instances.append(self)

        def encode_text(self, *_args, **_kwargs):
            vector = np.zeros((1, 512), dtype=np.float32)
            vector[0, 0] = 1.0
            self.last_shape = vector.shape
            return vector

    class DummyMetadataManager:
        def __init__(self):
            self.temporal_index = {}

    import core

    monkeypatch.setattr(core, "CLIPFeatureExtractor", DummyB32Processor)

    system = EnhancedRetrievalSystem.__new__(EnhancedRetrievalSystem)
    system.verbose = False
    system.enable_validation = False
    system.status = SystemStatus(is_initialized=True, is_ready=True, index_loaded=True)
    system.config = DummyConfig()
    system.logger = DummyLogger()
    system.perf_monitor = DummyPerfMonitor()
    system.cache = DummyCache()
    system.query_translator = None
    system.clip_processor = type("OldLarge", (), {
        "model_path": "openai/clip-vit-large-patch14",
        "model": type("Model", (), {"config": type("Config", (), {"projection_dim": 768})()})(),
        "encode_text": lambda self, *_args, **_kwargs: np.ones((1, 768), dtype=np.float32),
    })()
    system.active_query_encoder = system.clip_processor
    system.faiss_retriever = FAISSRetriever(system.config, system.logger, cache=object())
    system.metadata_manager = DummyMetadataManager()
    system.llm_processor = None

    builder = UnifiedBuilderIntegration(system, system.logger)
    system.unified_builder = builder
    builder.load_unified_index_fast(str(output_file))

    options = SearchOptions(mode="hybrid", limit=1, cache_results=False, include_temporal_context=False)
    results = system.search("a man riding a motorcycle on the street", options)

    assert system.active_query_encoder.model_path == "openai/clip-vit-base-patch32"
    assert DummyB32Processor.instances[-1].last_shape == (1, 512)
    assert system.faiss_retriever.dimension == 512
    assert results


def test_wrapper_lifecycle_smart_load_targets_base_clip_only(tmp_path, monkeypatch):
    import core

    DummyB32Processor.instances.clear()
    monkeypatch.setattr(core, "CLIPFeatureExtractor", DummyB32Processor)
    output_file = _create_synthetic_btc_index(tmp_path, rows=2)

    base = _make_base_system_for_search()
    wrapper = EnhancedRetrievalSystemV4(base_system=base, enable_v4_features=False, debug=False)

    builder = add_unified_index_support(wrapper)
    stats = builder.load_unified_index_fast(str(output_file))

    assert get_runtime_system(wrapper) is base
    assert builder.system is base
    assert base.unified_builder is builder
    assert "unified_builder" not in wrapper.__dict__
    assert base.active_query_encoder.model_path == "openai/clip-vit-base-patch32"
    assert base._encode_query_text("motorcycle", "test_wrapper_lifecycle").shape == (1, 512)
    assert base.faiss_retriever.dimension == 512
    assert base.faiss_retriever.index_clip_model == "openai/clip-vit-base-patch32"
    assert stats["index_info"]["build_source"] == "btc_clip_features"

    options = EnhancedSearchOptions(mode="clip_only", limit=1, cache_results=False, include_temporal_context=False)
    results = wrapper.search("a man riding a motorcycle on the street", options)

    assert DummyB32Processor.instances[-1].last_shape == (1, 512)
    assert results


def test_wrapper_lifecycle_hybrid_uses_base_btc_encoder(tmp_path, monkeypatch):
    import core

    DummyB32Processor.instances.clear()
    monkeypatch.setattr(core, "CLIPFeatureExtractor", DummyB32Processor)
    output_file = _create_synthetic_btc_index(tmp_path, rows=2)

    base = _make_base_system_for_search()
    wrapper = EnhancedRetrievalSystemV4(base_system=base, enable_v4_features=False, debug=False)

    add_unified_index_support(wrapper).load_unified_index_fast(str(output_file))

    options = EnhancedSearchOptions(mode="hybrid", limit=1, cache_results=False, include_temporal_context=False)
    results = wrapper.search("a man riding a motorcycle on the street", options)

    assert base.active_query_encoder.model_path == "openai/clip-vit-base-patch32"
    assert DummyB32Processor.instances[-1].last_shape == (1, 512)
    assert base.faiss_retriever.dimension == 512
    assert results


def test_wrapper_lifecycle_stale_legacy_clip_processor_does_not_override_active_btc_encoder(tmp_path, monkeypatch):
    import core

    DummyB32Processor.instances.clear()
    monkeypatch.setattr(core, "CLIPFeatureExtractor", DummyB32Processor)
    output_file = _create_synthetic_btc_index(tmp_path, rows=2)

    base = _make_base_system_for_search()
    wrapper = EnhancedRetrievalSystemV4(base_system=base, enable_v4_features=False, debug=False)

    add_unified_index_support(wrapper).load_unified_index_fast(str(output_file))
    b32_encoder = base.active_query_encoder
    base.clip_processor = DummyLargeProcessor()

    options = EnhancedSearchOptions(mode="clip_only", limit=1, cache_results=False, include_temporal_context=False)
    results = wrapper.search("a man riding a motorcycle on the street", options)

    assert base.active_query_encoder is b32_encoder
    assert base.clip_processor.model_path == "openai/clip-vit-large-patch14"
    assert b32_encoder.last_shape == (1, 512)
    assert results


def test_wrapper_lifecycle_btc_encoder_load_failure_fails_closed(tmp_path, monkeypatch):
    import core

    output_file = _create_synthetic_btc_index(tmp_path, rows=1)

    class FailingB32Processor:
        def __init__(self, model_path, *_args, **_kwargs):
            if model_path == "openai/clip-vit-base-patch32":
                raise RuntimeError("B32 unavailable")
            self.model_path = model_path

    monkeypatch.setattr(core, "CLIPFeatureExtractor", FailingB32Processor)

    base = _make_base_system_for_search()
    wrapper = EnhancedRetrievalSystemV4(base_system=base, enable_v4_features=False, debug=False)
    builder = add_unified_index_support(wrapper)

    with pytest.raises(RuntimeError, match="CLIP model initialization failed|B32 unavailable"):
        builder.load_unified_index_fast(str(output_file))

    assert base.active_query_encoder.model_path == "openai/clip-vit-large-patch14"
    assert base.clip_processor.model_path == "openai/clip-vit-large-patch14"


def test_wrapper_does_not_keep_conflicting_retrieval_shadow_state(tmp_path, monkeypatch):
    import core

    DummyB32Processor.instances.clear()
    monkeypatch.setattr(core, "CLIPFeatureExtractor", DummyB32Processor)
    output_file = _create_synthetic_btc_index(tmp_path, rows=1)

    base = _make_base_system_for_search()
    wrapper = EnhancedRetrievalSystemV4(base_system=base, enable_v4_features=False, debug=False)
    add_unified_index_support(wrapper).load_unified_index_fast(str(output_file))

    shadow_keys = {"active_query_encoder", "clip_processor", "unified_builder"} & set(wrapper.__dict__)
    assert shadow_keys == set()
    assert wrapper.active_query_encoder is base.active_query_encoder
    assert wrapper.clip_processor is base.clip_processor
    assert wrapper.unified_builder is base.unified_builder


def test_wrapper_legacy_l14_regression_still_searches_768d_index():
    base = _make_base_system_for_search()
    metadata = [
        KeyframeMetadata(
            folder_name="L21_V001",
            image_name="001.jpg",
            frame_id=0,
            file_path="L21_V001/001.jpg",
            clip_features=np.array([1.0] + [0.0] * 767, dtype=np.float32),
        )
    ]
    features = np.array([[1.0] + [0.0] * 767], dtype=np.float32)
    base.faiss_retriever.build_index(features, metadata, validate_consistency=False)
    wrapper = EnhancedRetrievalSystemV4(base_system=base, enable_v4_features=False, debug=False)

    options = EnhancedSearchOptions(mode="clip_only", limit=1, cache_results=False, include_temporal_context=False)
    results = wrapper.search("legacy motorcycle query", options)

    assert base.active_query_encoder.model_path == "openai/clip-vit-large-patch14"
    assert base.clip_processor.last_shape == (1, 768)
    assert base.faiss_retriever.dimension == 768
    assert results


def test_unified_similarity_preserved_while_ranking_score_is_boosted(tmp_path, monkeypatch):
    import core

    DummyB32Processor.instances.clear()
    monkeypatch.setattr(core, "CLIPFeatureExtractor", DummyB32Processor)
    output_file = _create_synthetic_btc_index(tmp_path, rows=2)

    base = _make_base_system_for_search()
    base.enable_validation = True
    wrapper = EnhancedRetrievalSystemV4(base_system=base, enable_v4_features=False, debug=False)
    add_unified_index_support(wrapper).load_unified_index_fast(str(output_file))

    options = EnhancedSearchOptions(mode="clip_only", limit=1, cache_results=False, include_temporal_context=False)
    results = wrapper.search("a man riding a motorcycle on the street", options)

    assert results
    result = results[0]
    assert result.similarity_score == 1.0
    assert result.ranking_score > result.similarity_score
    assert result.ranking_score > 1.0
    assert not any("Invalid similarity score" in message for message, _ in base.logger.warnings)


def test_consensus_boost_targets_ranking_score_not_similarity(tmp_path, monkeypatch):
    import core

    DummyB32Processor.instances.clear()
    monkeypatch.setattr(core, "CLIPFeatureExtractor", DummyB32Processor)
    output_file = _create_synthetic_btc_index(tmp_path, rows=1)

    base = _make_base_system_for_search()
    base.enable_validation = True
    add_unified_index_support(base).load_unified_index_fast(str(output_file))

    options = SearchOptions(mode="clip_only", limit=1, cache_results=False, include_temporal_context=False)
    results = base.search("a man riding a motorcycle on the street", options)

    assert results[0].similarity_score == 1.0
    assert results[0].ranking_score > 1.0
    assert not any("Invalid similarity score" in message for message, _ in base.logger.warnings)


def test_unified_ordering_uses_ranking_score_without_mutating_similarity():
    system = _make_base_system_for_search()
    candidates = [
        {
            "index": 1,
            "similarity_score": 0.95,
            "ranking_score": 0.2,
            "query_matches": 1,
        },
        {
            "index": 2,
            "similarity_score": 0.31,
            "ranking_score": 1.4,
            "query_matches": 1,
        },
    ]

    ranked = system._apply_advanced_ranking(candidates, "query", [np.zeros((1, 512), dtype=np.float32)])
    selected = system._select_top_quality_results(ranked, 2)

    assert selected[0]["index"] == 2
    assert selected[0]["similarity_score"] == 0.31
    assert selected[0]["ranking_score"] > selected[1]["ranking_score"]


def test_quality_metrics_report_similarity_and_ranking_separately():
    system = _make_base_system_for_search()
    metadata = KeyframeMetadata(folder_name="L21_V001", image_name="001", frame_id=0, file_path="x")
    results = [
        SearchResult(metadata=metadata, similarity_score=0.31, ranking_score=1.1, rank=1),
        SearchResult(metadata=metadata, similarity_score=0.33, ranking_score=1.2, rank=2),
        SearchResult(metadata=metadata, similarity_score=0.35, ranking_score=1.3, rank=3),
    ]

    system._log_search_quality_metrics(results, "query", 0.01)
    info_messages = [message for message, _ in system.logger.infos]

    assert any("Similarity: Avg=0.330" in message for message in info_messages)
    assert any("Ranking: Avg=1.200" in message for message in info_messages)


def test_public_clip_only_real_path_has_valid_similarity_and_ranking(tmp_path, monkeypatch):
    import core

    DummyB32Processor.instances.clear()
    monkeypatch.setattr(core, "CLIPFeatureExtractor", DummyB32Processor)
    output_file = _create_synthetic_btc_index(tmp_path, rows=2)

    base = _make_base_system_for_search()
    base.enable_validation = True
    wrapper = EnhancedRetrievalSystemV4(base_system=base, enable_v4_features=False, debug=False)
    add_unified_index_support(wrapper).load_unified_index_fast(str(output_file))

    options = EnhancedSearchOptions(mode="clip_only", limit=2, cache_results=False, include_temporal_context=False)
    results = wrapper.search("a man riding a motorcycle on the street", options)

    assert results
    assert all(0.0 <= r.similarity_score <= 1.0 for r in results)
    assert all(getattr(r, "ranking_score", None) is not None for r in results)
    assert not any("Invalid similarity score" in message for message, _ in base.logger.warnings)


def test_public_hybrid_real_path_has_valid_similarity_and_ranking(tmp_path, monkeypatch):
    import core

    DummyB32Processor.instances.clear()
    monkeypatch.setattr(core, "CLIPFeatureExtractor", DummyB32Processor)
    output_file = _create_synthetic_btc_index(tmp_path, rows=2)

    base = _make_base_system_for_search()
    base.enable_validation = True
    wrapper = EnhancedRetrievalSystemV4(base_system=base, enable_v4_features=False, debug=False)
    add_unified_index_support(wrapper).load_unified_index_fast(str(output_file))

    options = EnhancedSearchOptions(mode="hybrid", limit=2, cache_results=False, include_temporal_context=False)
    results = wrapper.search("a man riding a motorcycle on the street", options)

    assert results
    assert all(0.0 <= r.similarity_score <= 1.0 for r in results)
    assert all(getattr(r, "ranking_score", None) is not None for r in results)
    assert not any("Invalid similarity score" in message for message, _ in base.logger.warnings)


def test_legacy_search_result_defaults_ranking_score_to_similarity():
    metadata = KeyframeMetadata(
        folder_name="L21_V001",
        image_name="001.jpg",
        frame_id=0,
        file_path="L21_V001/001.jpg",
        clip_features=np.array([1.0] + [0.0] * 767, dtype=np.float32),
    )
    result = SearchResult(metadata=metadata, similarity_score=0.42, rank=1)

    assert result.ranking_score == result.similarity_score


def test_competition_export_ignores_ranking_score_and_keeps_frame_id(tmp_path):
    metadata = DummyMetadata(folder_name="L21_V001", image_name="000123", frame_id=3876)
    result = DummySearchResult(metadata=metadata, similarity_score=0.33)
    result.ranking_score = 1.25

    generator = CompetitionCSVGenerator(output_dir=str(tmp_path))
    competition_results = generator.convert_search_results_to_competition_format(
        [result],
        query_id="Q_SCORE",
        query_type=QueryType.KIS,
    )
    query = CompetitionQuery(
        query_id="Q_SCORE",
        query_text="motorcycle",
        query_type=QueryType.KIS,
        results=competition_results,
        processing_time=0.0,
        v4_enhancements_used=[],
    )
    csv_path = generator.generate_csv_file(query, filename="submission_score.csv")

    assert open(csv_path, encoding="utf-8").read() == "L21_V001,3876\n"
