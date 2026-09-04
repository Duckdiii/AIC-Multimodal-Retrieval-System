"""
Enhanced Retrieval System - Unified System Wrapper (Cleaned & Optimized)
=========================================================================

Main orchestrator class that provides a single, unified interface for the entire system.
This is the primary entry point for all functionality including search, indexing, 
GUI launching, and server operations.

Enhanced with robust validation, system health monitoring, and recovery mechanisms.
Optimized version with OpenAI-only integration.

Author: Enhanced Retrieval System
Version: 2.1
"""

import os
import json
import time
import threading
import shutil
import re
import random
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import traceback

import numpy as np
from datetime import datetime

# Import our custom modules
from core import (
    FAISSRetriever, CLIPFeatureExtractor, LLMProcessor, 
    MetadataManager, TemporalAnalyzer,
    KeyframeMetadata, SearchResult, DataConsistencyValidator,
    UniversalQueryTranslator, QueryStructure
)


@dataclass
class SystemStatus:
    """Enhanced system status information with detailed health metrics"""
    is_initialized: bool = False
    is_ready: bool = False
    index_loaded: bool = False
    components_loaded: Dict[str, bool] = None
    last_error: Optional[str] = None
    system_stats: Dict[str, Any] = None
    health_status: Dict[str, Any] = None
    consistency_check: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.components_loaded is None:
            self.components_loaded = {
                "config": False,
                "logger": False,
                "clip_processor": False,
                "faiss_retriever": False,
                "llm_processor": False,
                "metadata_manager": False,
                "temporal_analyzer": False,
                "query_translator": False
            }
        if self.system_stats is None:
            self.system_stats = {}
        if self.health_status is None:
            self.health_status = {}
        if self.consistency_check is None:
            self.consistency_check = {}


@dataclass  
class SearchOptions:
    """Enhanced search configuration options with validation"""
    mode: str = "hybrid"  # clip_only, llm_enhanced, hybrid
    limit: int = 50
    include_temporal_context: bool = True
    include_explanations: bool = False
    similarity_threshold: float = 0.0
    enable_reranking: bool = True
    cache_results: bool = True
    validate_results: bool = True
    
    def validate(self) -> None:
        """Validate search options with detailed error messages"""
        if self.mode not in ["clip_only", "llm_enhanced", "hybrid"]:
            raise ValueError(f"Invalid search mode: {self.mode}. Must be one of: clip_only, llm_enhanced, hybrid")
        if self.limit <= 0:
            raise ValueError(f"Search limit must be positive, got: {self.limit}")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError(f"Similarity threshold must be between 0 and 1, got: {self.similarity_threshold}")
        if self.limit > 1000:
            raise ValueError(f"Search limit too high ({self.limit}), maximum is 1000")


class SystemHealthMonitor:
    """
    🏥 System Health Monitoring and Diagnostics
    
    Monitors system health, detects issues, and provides recovery recommendations.
    """
    
    def __init__(self, logger=None):
        self.logger = logger
        self.validator = DataConsistencyValidator(logger)
    
    def comprehensive_health_check(self, system) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        health_report = {
            "overall_health": "healthy",
            "timestamp": time.time(),
            "components": {},
            "issues": [],
            "recommendations": [],
            "stats": {},
            "recovery_options": []
        }
        
        try:
            # Check each component
            health_report["components"]["config"] = self._check_config_health(system)
            health_report["components"]["faiss"] = self._check_faiss_health(system)
            health_report["components"]["metadata"] = self._check_metadata_health(system)
            health_report["components"]["clip"] = self._check_clip_health(system)
            health_report["components"]["llm"] = self._check_llm_health(system)
            
            # Check overall consistency
            health_report["consistency"] = self._check_system_consistency(system)
            
            # Aggregate health status
            all_healthy = all(
                comp.get("is_healthy", False) 
                for comp in health_report["components"].values()
            )
            
            consistency_healthy = health_report["consistency"].get("is_consistent", False)
            
            if not all_healthy or not consistency_healthy:
                health_report["overall_health"] = "unhealthy"
                
                # Collect all issues
                for comp_name, comp_health in health_report["components"].items():
                    health_report["issues"].extend(
                        [f"{comp_name}: {issue}" for issue in comp_health.get("issues", [])]
                    )
                    health_report["recommendations"].extend(comp_health.get("recommendations", []))
                
                health_report["issues"].extend(health_report["consistency"].get("issues", []))
                health_report["recommendations"].extend(health_report["consistency"].get("recommendations", []))
                
                # Add recovery options
                health_report["recovery_options"] = self._generate_recovery_options(health_report)
            
            # Generate stats
            health_report["stats"] = self._generate_health_stats(system, health_report)
            
        except Exception as e:
            health_report["overall_health"] = "error"
            health_report["issues"].append(f"Health check failed: {str(e)}")
            if self.logger:
                self.logger.error(f"Health check failed", error=str(e), exc_info=True)
        
        return health_report
    
    def _check_config_health(self, system) -> Dict[str, Any]:
        """Check configuration health"""
        health = {"is_healthy": True, "issues": [], "recommendations": []}
        
        try:
            if not system.config:
                health["is_healthy"] = False
                health["issues"].append("No configuration loaded")
                health["recommendations"].append("Initialize configuration")
                return health
            
            # Check required paths
            required_paths = ["keyframes", "models", "cache", "exports", "logs", "index", "metadata"]
            for path_key in required_paths:
                path_value = system.config.get(f"paths.{path_key}")
                if not path_value:
                    health["issues"].append(f"Missing path configuration: {path_key}")
                    health["recommendations"].append(f"Configure paths.{path_key} in config")
            
            # Check system settings
            if not system.config.get("system.name"):
                health["issues"].append("System name not configured")
            
        except Exception as e:
            health["is_healthy"] = False
            health["issues"].append(f"Config check failed: {str(e)}")
        
        if health["issues"]:
            health["is_healthy"] = False
        
        return health
    
    def _check_faiss_health(self, system) -> Dict[str, Any]:
        """Check FAISS retriever health (includes unified index support)"""
        # Check if unified index is available first
        has_unified_index = (hasattr(system, 'unified_builder') and 
                           system.unified_builder and 
                           system.unified_builder.unified_index and 
                           system.unified_builder.unified_index.is_loaded)
        
        if has_unified_index:
            return {
                "is_healthy": True,
                "issues": [],
                "recommendations": [],
                "index_type": "Unified Index (.rvdb)",
                "status": "Loaded and ready"
            }
        
        # Check if Colab extracted index is loaded
        if system.status.is_ready and system.faiss_retriever and system.faiss_retriever.index is not None:
            return {
                "is_healthy": True,
                "issues": [],
                "recommendations": [],
                "index_type": "Colab Extracted FAISS",
                "status": f"Loaded and ready ({system.faiss_retriever.index.ntotal} vectors)"
            }
        
        # Fallback to traditional FAISS check
        if not system.faiss_retriever:
            return {
                "is_healthy": False,
                "issues": ["No FAISS index loaded"],
                "recommendations": ["Build or load a FAISS index"]
            }
        
        return system.faiss_retriever.get_system_health()
    
    def _check_metadata_health(self, system) -> Dict[str, Any]:
        """Check metadata manager health (includes unified index support)"""
        # Check if unified index is available first
        has_unified_index = (hasattr(system, 'unified_builder') and 
                           system.unified_builder and 
                           system.unified_builder.unified_index and 
                           system.unified_builder.unified_index.is_loaded)
        
        if has_unified_index:
            metadata_count = len(system.unified_builder.unified_index.metadata_list) if system.unified_builder.unified_index.metadata_list else 0
            return {
                "is_healthy": True,
                "issues": [],
                "recommendations": [],
                "metadata_source": "Unified Index",
                "metadata_count": metadata_count,
                "status": "Loaded and ready"
            }
        
        # Check if Colab extracted metadata is loaded
        if system.status.is_ready and system.faiss_retriever and len(system.faiss_retriever.id_to_metadata) > 0:
            return {
                "is_healthy": True,
                "issues": [],
                "recommendations": [],
                "metadata_source": "Colab Mapping",
                "metadata_count": len(system.faiss_retriever.id_to_metadata),
                "status": "Loaded and ready"
            }
        
        # Fallback to traditional metadata manager check
        if not system.metadata_manager:
            return {
                "is_healthy": False,
                "issues": ["No metadata loaded"],
                "recommendations": ["Build or load a system with metadata"]
            }
        
        return system.metadata_manager.get_health_status()
    
    def _check_clip_health(self, system) -> Dict[str, Any]:
        """Check CLIP processor health"""
        health = {"is_healthy": True, "issues": [], "recommendations": []}
        
        try:
            if not system.clip_processor:
                health["is_healthy"] = False
                health["issues"].append("CLIP processor not initialized")
                health["recommendations"].append("Initialize CLIP processor")
                return health
            
            if not system.clip_processor.model:
                health["is_healthy"] = False
                health["issues"].append("CLIP model not loaded")
                health["recommendations"].append("Load CLIP model")
            
            # Test basic functionality
            try:
                test_features = system.clip_processor.encode_text(["test"], validate_input=False)
                if test_features is None or len(test_features) == 0:
                    health["issues"].append("CLIP text encoding not working")
                    health["recommendations"].append("Restart CLIP processor")
            except Exception as e:
                health["issues"].append(f"CLIP functionality test failed: {str(e)}")
                health["recommendations"].append("Check CLIP model and dependencies")
            
        except Exception as e:
            health["is_healthy"] = False
            health["issues"].append(f"CLIP health check failed: {str(e)}")
        
        if health["issues"]:
            health["is_healthy"] = False
        
        return health
    
    def _check_llm_health(self, system) -> Dict[str, Any]:
        """Check LLM processor health (OpenAI only)"""
        health = {"is_healthy": True, "issues": [], "recommendations": []}
        
        try:
            if not system.llm_processor:
                health["is_healthy"] = False
                health["issues"].append("LLM processor not initialized")
                health["recommendations"].append("Initialize LLM processor")
                return health
            
            # Check OpenAI integration - prefer agent over client
            if hasattr(system.llm_processor, 'conversational_agent') and system.llm_processor.conversational_agent:
                health["recommendations"].append("OpenAI GPT-4 agent active")
                health["is_healthy"] = True
            elif hasattr(system.llm_processor, 'openai_client') and system.llm_processor.openai_client:
                health["recommendations"].append("OpenAI client initialized")  
                health["is_healthy"] = True
            else:
                health["issues"].append("OpenAI client not available")
                health["recommendations"].append("Check OpenAI API key configuration")
                health["is_healthy"] = False
            
        except Exception as e:
            health["is_healthy"] = False
            health["issues"].append(f"LLM health check failed: {str(e)}")
        
        if health["issues"]:
            health["is_healthy"] = False
        
        return health
    
    def _check_system_consistency(self, system) -> Dict[str, Any]:
        """Check overall system consistency"""
        if not system.faiss_retriever or not system.metadata_manager:
            return {
                "is_consistent": False,
                "issues": ["Core components not initialized"],
                "recommendations": ["Initialize all core components"]
            }
        
        return self.validator.validate_index_metadata_consistency(
            system.faiss_retriever.index,
            system.faiss_retriever.id_to_metadata
        )
    
    def _generate_recovery_options(self, health_report: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recovery options based on health issues"""
        recovery_options = []
        
        issues = health_report.get("issues", [])
        
        # Check for common patterns
        if any("metadata" in issue.lower() for issue in issues):
            recovery_options.append({
                "action": "rebuild_metadata",
                "description": "Rebuild metadata from keyframes folder",
                "command": "system.build_system(keyframes_folder, force_rebuild=True)"
            })
        
        if any("index" in issue.lower() for issue in issues):
            recovery_options.append({
                "action": "rebuild_index",
                "description": "Rebuild FAISS index from scratch",
                "command": "system.build_system(keyframes_folder, force_rebuild=True)"
            })
        
        if any("inconsistent" in issue.lower() for issue in issues):
            recovery_options.append({
                "action": "full_rebuild",
                "description": "Complete system rebuild (recommended)",
                "command": "system.build_system(keyframes_folder, force_rebuild=True)"
            })
        
        if any("clip" in issue.lower() for issue in issues):
            recovery_options.append({
                "action": "restart_clip",
                "description": "Restart CLIP processor",
                "command": "system._initialize_ai_components()"
            })
        
        return recovery_options
    
    def _generate_health_stats(self, system, health_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate health statistics"""
        stats = {}
        
        try:
            # Component status
            components = health_report.get("components", {})
            healthy_components = sum(1 for comp in components.values() if comp.get("is_healthy", False))
            total_components = len(components)
            
            stats["component_health_rate"] = healthy_components / total_components if total_components > 0 else 0
            stats["healthy_components"] = healthy_components
            stats["total_components"] = total_components
            
            # System stats
            if system.faiss_retriever and system.faiss_retriever.index:
                stats["index_size"] = system.faiss_retriever.index.ntotal
                stats["metadata_count"] = len(system.faiss_retriever.id_to_metadata)
                stats["index_metadata_ratio"] = (
                    stats["metadata_count"] / stats["index_size"] 
                    if stats["index_size"] > 0 else 0
                )
            
            # Issue count
            stats["total_issues"] = len(health_report.get("issues", []))
            stats["recovery_options"] = len(health_report.get("recovery_options", []))
            
        except Exception as e:
            stats["error"] = str(e)
        
        return stats


class EnhancedRetrievalSystem:
    """
    🎯 MAIN SYSTEM CLASS - Unified Interface with OpenAI Integration
    
    Single entry point for all Enhanced Retrieval System functionality.
    Orchestrates all components with comprehensive validation and health monitoring.
    
    Enhanced Features:
    - Comprehensive system validation
    - Health monitoring and diagnostics
    - Robust error handling and recovery
    - Performance optimization
    - Detailed progress tracking
    - OpenAI GPT-4 integration only
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 auto_initialize: bool = True,
                 verbose: bool = True,
                 enable_validation: bool = True):
        """
        Initialize Enhanced Retrieval System with robust validation
        
        Args:
            config_path: Path to configuration file
            auto_initialize: Whether to automatically initialize components
            verbose: Enable verbose logging
            enable_validation: Enable comprehensive validation
        """
        # Core attributes
        self.verbose = verbose
        self.enable_validation = enable_validation
        self.status = SystemStatus()
        self._lock = threading.RLock()
        self._callbacks = {}  # Event callbacks
        
        # Initialize base components
        self.config = None
        self.logger = None
        self.file_manager = None
        self.data_processor = None
        self.cache = None
        self.perf_monitor = None
        
        # Initialize AI components
        self.clip_processor = None
        self.active_query_encoder = None
        self.faiss_retriever = None
        self.llm_processor = None
        self.metadata_manager = None
        self.temporal_analyzer = None
        self.query_translator = None
        
        # System health monitoring
        self.health_monitor = None
        
        # System state
        self.keyframe_data = {}
        self.feature_matrix = None
        self.metadata_list = []
        self.system_metrics = {}
        
        # Initialize system
        try:
            self._initialize_base_components(config_path)
            
            if auto_initialize:
                self._initialize_ai_components()
                
            self.status.is_initialized = True
            
            # Auto-load Colab extracted dataset if available
            extracted_path = Path("data/extracted")
            if (extracted_path / "clip_index" / "clip_faiss.index").exists():
                self.load_colab_extracted(str(extracted_path))
                
            self._perform_initial_health_check()
            
            self._log_info("Enhanced Retrieval System initialized successfully")
            
        except Exception as e:
            self.status.last_error = str(e)
            self._log_error(f"System initialization failed: {e}", exc_info=True)
            raise
    
    # =================== CORE SYSTEM METHODS ===================
    
    def load_colab_extracted(self, extracted_path: str = "data/extracted") -> bool:
        """
        Load precomputed index and metadata directly from Colab extracted folder
        """
        from aic_core.retrieval.faiss_backend import FaissUnifiedRetriever
        from core import KeyframeMetadata

        extracted_dir = Path(extracted_path)
        index_file = extracted_dir / "clip_index" / "clip_faiss.index"
        if not index_file.exists():
            return False

        try:
            self._log_info(f"Auto-loading Colab extracted dataset from {extracted_path}...")
            retriever = FaissUnifiedRetriever.from_colab_extracted(str(extracted_dir))

            self.faiss_retriever.index = retriever.index
            self.faiss_retriever.dimension = retriever.index.d
            self.faiss_retriever.is_trained = True
            self.faiss_retriever.index_clip_model = "openai/clip-vit-base-patch32"
            self.faiss_retriever.id_to_metadata.clear()

            metadata_list = []
            for idx, meta in enumerate(retriever.metadata):
                v_name = meta["video_name"]
                f_id = meta["frame_id"]
                l_idx = meta.get("local_frame_idx", 0)
                img_name = f"{l_idx+1:03d}.jpg"
                kf_path = str(Path("keyframes") / v_name / img_name)
                kf_meta = KeyframeMetadata(
                    folder_name=v_name,
                    image_name=img_name,
                    frame_id=int(f_id),
                    file_path=kf_path,
                )
                self.faiss_retriever.id_to_metadata[idx] = kf_meta
                metadata_list.append(kf_meta)

            # Populate metadata_manager so temporal operations (Nearby Frames, etc.) work flawlessly
            from collections import defaultdict
            if not self.metadata_manager:
                from core import MetadataManager
                self.metadata_manager = MetadataManager(self.config, self.logger)

            folder_groups = defaultdict(list)
            for kf_meta in metadata_list:
                folder_groups[kf_meta.folder_name].append(kf_meta)
                if kf_meta.folder_name not in self.metadata_manager.metadata_db:
                    self.metadata_manager.metadata_db[kf_meta.folder_name] = {}
                self.metadata_manager.metadata_db[kf_meta.folder_name][kf_meta.image_name] = kf_meta
                self.metadata_manager.metadata_db[kf_meta.folder_name][Path(kf_meta.image_name).stem] = kf_meta
                self.metadata_manager.metadata_db[kf_meta.folder_name][str(kf_meta.frame_id)] = kf_meta

            for folder_name, kf_items in folder_groups.items():
                kf_items.sort(key=lambda x: x.frame_id)
                self.metadata_manager.temporal_index[folder_name] = kf_items

            self.metadata_list = metadata_list
            self.active_query_encoder = self.clip_processor
            self.status.is_ready = True
            self.status.index_loaded = True
            self._log_info(f"Colab extracted dataset loaded successfully: {len(self.faiss_retriever.id_to_metadata)} vectors ready across {len(folder_groups)} video folders.")
            return True
        except Exception as e:
            self._log_error(f"Failed to auto-load Colab extracted dataset: {e}", exc_info=True)
            return False

    
    def build_system_with_map(self, 
                             keyframe_folder: str,
                             map_folder: str,
                             force_rebuild: bool = False,
                             validate_inputs: bool = True,
                             progress_callback: Optional[Callable] = None) -> bool:
        """
        Build system with explicit map folder (for GUI)
        
        Args:
            keyframe_folder: Path to keyframes folder
            map_folder: Path to map CSV files folder
            force_rebuild: Force rebuild even if index exists
            validate_inputs: Enable input validation
            progress_callback: Progress callback function
            
        Returns:
            bool: True if successful
        """
        # Store map folder path temporarily for the build process
        original_map_path = os.environ.get('MAP_FOLDER_PATH', '')
        os.environ['MAP_FOLDER_PATH'] = map_folder
        
        try:
            self._log_info(f"Building system with map folder", 
                          keyframes=keyframe_folder, 
                          map_folder=map_folder)
            
            # Call original build_system method
            return self.build_system(keyframe_folder, force_rebuild, True, progress_callback, validate_inputs)
        finally:
            # Restore original map path
            if original_map_path:
                os.environ['MAP_FOLDER_PATH'] = original_map_path
            else:
                os.environ.pop('MAP_FOLDER_PATH', None)
    
    def build_system(self, 
                    keyframe_folder: str, 
                    force_rebuild: bool = False,
                    save_index: bool = True,
                    progress_callback: Optional[Callable] = None,
                    validate_inputs: bool = None) -> bool:
        """
        Build entire system from keyframe folder with comprehensive validation
        
        Args:
            keyframe_folder: Path to keyframes directory
            force_rebuild: Force rebuild even if index exists
            save_index: Save index after building
            progress_callback: Progress callback function(step, total, message)
            validate_inputs: Whether to validate inputs (defaults to system setting)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.status.is_initialized:
            raise RuntimeError("System not initialized. Call __init__ first.")
        
        validate_inputs = validate_inputs if validate_inputs is not None else self.enable_validation
        
        try:
            with self.perf_monitor.timer("build_system") as timer_id:
                self._log_info(f"Building system from keyframes", folder=keyframe_folder)
                
                # Validate input folder
                if validate_inputs:
                    self._validate_keyframes_folder(keyframe_folder)
                
                # Check if index already exists
                index_path = self.config.get("paths.index", "index/")
                if not force_rebuild and self._index_exists(index_path):
                    self._log_info("Index already exists, loading instead of rebuilding")
                    if self.load_system(index_path):
                        # Verify loaded system health
                        health = self.get_system_health()
                        if health["overall_health"] == "healthy":
                            return True
                        else:
                            self._log_warning("Loaded system is unhealthy, forcing rebuild")
                            force_rebuild = True
                    else:
                        force_rebuild = True
                
                total_steps = 6
                step = 0
                
                # Step 1: Pre-build validation
                step += 1
                self._update_progress(progress_callback, step, total_steps, "Pre-build validation...")
                if validate_inputs:
                    self._pre_build_validation(keyframe_folder)
                
                # Step 2: Scan keyframes
                step += 1
                self._update_progress(progress_callback, step, total_steps, "Scanning keyframes...")
                self.keyframe_data = self.file_manager.scan_keyframes(keyframe_folder)
                
                if not self.keyframe_data:
                    raise ValueError("No keyframes found in the specified folder")
                
                # Step 3: Extract features
                step += 1  
                self._update_progress(progress_callback, step, total_steps, "Extracting CLIP features...")
                self.feature_matrix, self.metadata_list = self.clip_processor.extract_features_batch(keyframe_folder)
                
                if len(self.feature_matrix) == 0:
                    raise ValueError("No features extracted from keyframes")
                
                # Step 4: Build metadata
                step += 1
                self._update_progress(progress_callback, step, total_steps, "Building metadata...")
                self.metadata_manager.build_metadata(keyframe_folder, self.feature_matrix, validate_inputs)
                
                # Step 5: Build FAISS index
                step += 1
                self._update_progress(progress_callback, step, total_steps, "Building FAISS index...")
                self.faiss_retriever.build_index(
                    self.feature_matrix, 
                    self.metadata_list,
                    validate_consistency=validate_inputs
                )
                
                # UPDATE STATUS BEFORE SAVE/VALIDATION
                self.status.is_ready = True
                self.status.index_loaded = True
                self._update_system_stats()
                
                # Step 6: Save and validate system
                if save_index:
                    step += 1
                    self._update_progress(progress_callback, step, total_steps, "Saving system...")
                    self._save_system(index_path)
                    
                    # Store index path for later use (e.g., portable export)
                    self.index_path = str(index_path)
                    
                    # Post-save validation (SIMPLIFIED)
                    if validate_inputs:
                        try:
                            health = self.get_system_health()
                            if health["overall_health"] != "healthy":
                                self._log_warning(f"System built but health check found issues: {health['issues'][:3]}")  # Only show first 3
                        except Exception as e:
                            self._log_warning(f"Post-build health check failed: {e}")
                
                # Final health check
                health = self.get_system_health()
                
                duration = self.perf_monitor.end_timer(timer_id)
                self._log_info(f"System built successfully", 
                             duration=duration, 
                             keyframes=len(self.metadata_list),
                             folders=len(self.keyframe_data),
                             health=health["overall_health"])
                
                return True
                
        except Exception as e:
            self.status.last_error = str(e)
            self._log_error(f"System building failed: {e}", exc_info=True)
            
            # Cleanup partial build
            self._cleanup_partial_build()
            return False
    
    def load_system(self, index_path: Optional[str] = None, validate_after_load: bool = None) -> bool:
        """
        Load pre-built system from disk with comprehensive validation
        
        Args:
            index_path: Path to index directory
            validate_after_load: Whether to validate after loading
            
        Returns:
            True if successful, False otherwise
        """
        if not self.status.is_initialized:
            raise RuntimeError("System not initialized")
        
        index_path = index_path or self.config.get("paths.index", "index/")
        validate_after_load = validate_after_load if validate_after_load is not None else self.enable_validation
        
        try:
            with self.perf_monitor.timer("load_system"):
                self._log_info(f"Loading system from index", path=index_path)
                
                # Pre-load validation
                if validate_after_load:
                    self._pre_load_validation(index_path)
                
                # Load FAISS index
                self._log_info("Loading FAISS index...")
                self.faiss_retriever.load_index(index_path, validate_after_load)
                
                # Load metadata
                metadata_path = Path(index_path) / "metadata.json"
                if metadata_path.exists():
                    self._log_info("Loading metadata manager...")
                    self.metadata_manager.load_metadata(metadata_path)
                else:
                    if validate_after_load:
                        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
                    else:
                        self._log_warning("Metadata file not found, system will work with limited functionality")
                
                # Load system info
                self._load_system_info(index_path)
                
                # Post-load validation
                if validate_after_load:
                    validation_result = self._post_load_validation()
                    if not validation_result["is_valid"]:
                        raise RuntimeError(f"System validation failed: {validation_result['errors']}")
                
                # Store index path for later use
                self.index_path = str(index_path)
                
                # CRITICAL: Ensure temporal index is properly loaded/rebuilt
                self._ensure_temporal_index_loaded()
                
                # Update status
                self.status.is_ready = True
                self.status.index_loaded = True
                self._update_system_stats()
                
                # Health check
                health = self.get_system_health()
                if health["overall_health"] != "healthy" and validate_after_load:
                    self._log_error(f"Loaded system is unhealthy: {health}")
                    raise RuntimeError(f"System health check failed: {health['issues']}")
                
                index_size = self.faiss_retriever.index.ntotal if self.faiss_retriever.index else 0
                self._log_info(f"System loaded successfully", 
                             index_size=index_size,
                             health=health["overall_health"])
                
                return True
                
        except Exception as e:
            self.status.last_error = str(e)
            self._log_error(f"System loading failed: {e}", exc_info=True)
            
            # Clear inconsistent state
            self._clear_system_state()
            return False
    
    # =================== SEARCH INTERFACE ===================
    
    def search(self, 
              query: str, 
              options: Optional[SearchOptions] = None) -> List[SearchResult]:
        """
        Main search interface with comprehensive validation and error handling
        
        Args:
            query: Search query text
            options: Search configuration options
            
        Returns:
            List of search results with metadata and context
        """
        if not self.status.is_ready:
            raise RuntimeError("System not ready. Build or load system first.")
        
        options = options or SearchOptions()
        options.validate()
        
        # Pre-search validation
        if self.enable_validation:
            search_validation = self._validate_search_readiness()
            if not search_validation["is_ready"]:
                raise RuntimeError(f"System not ready for search: {search_validation['issues']}")
        
        try:
            with self.perf_monitor.timer("search_query", query_length=len(query)):
                search_start_time = time.time()
                self._log_debug(f"Searching with query", query=query, mode=options.mode)
                
                # Check cache first
                if options.cache_results:
                    cached_results = self.cache.get_cached_results(query, ttl=self.config.get("llm.cache_ttl", 3600))
                    if cached_results:
                        self._log_debug("Returning cached results", count=len(cached_results))
                        return cached_results[:options.limit]
                
                # Translate and optimize query if translator is available
                translated_query_struct = None
                search_query = query
                
                if self.query_translator:
                    try:
                        translated_query_struct = self.query_translator.translate_query(query)
                        
                        # Use translated CLIP prompt for better results
                        if translated_query_struct.clip_prompt and translated_query_struct.confidence > 0.3:
                            search_query = translated_query_struct.clip_prompt
                            self._log_debug(f"Query translated", 
                                          original=query,
                                          translated=search_query,
                                          language=translated_query_struct.detected_language,
                                          confidence=translated_query_struct.confidence)
                    except Exception as e:
                        self._log_warning(f"Query translation failed, using original: {e}")
                
                # Enhanced query optimization for better CLIP understanding
                search_query = self._optimize_query_for_clip(search_query)
                
                # Execute search based on mode
                if options.mode == "clip_only":
                    results = self._search_clip_only(search_query, options, translated_query_struct)
                elif options.mode == "llm_enhanced":
                    results = self._search_llm_enhanced(search_query, options, translated_query_struct)
                elif options.mode == "hybrid":
                    results = self._search_hybrid(search_query, options, translated_query_struct)
                else:
                    raise ValueError(f"Unknown search mode: {options.mode}")
                
                # Post-process results
                results = self._post_process_results(results, query, options)
                
                # Validate results if requested
                if options.validate_results and self.enable_validation:
                    # Check if unified index was used (skip file validation for unified index)
                    is_using_unified_index = (hasattr(self, 'unified_builder') and 
                                             self.unified_builder and 
                                             self.unified_builder.unified_index and 
                                             self.unified_builder.unified_index.is_loaded)
                    results = self._validate_search_results(results, query, skip_file_validation=is_using_unified_index)
                
                # Cache results
                if options.cache_results and results:
                    self.cache.cache_query_results(query, results)
                
                # Log quality metrics
                search_time = time.time() - search_start_time
                self._log_search_quality_metrics(results, query, search_time)
                
                self._log_info(f"Search completed", 
                             query=self._safe_log_string(query), 
                             mode=options.mode,
                             results_count=len(results))
                
                return results
                
        except Exception as e:
            self.status.last_error = str(e)
            self._log_error(f"Search failed: {e}", query=self._safe_log_string(query), exc_info=True)
            
            # Return empty results instead of crashing
            return []
    
    def search_by_image(self, 
                       folder_name: str, 
                       image_name: str, 
                       options: Optional[SearchOptions] = None) -> List[SearchResult]:
        """
        Search for similar images using an existing keyframe as query with validation
        
        Args:
            folder_name: Source folder name
            image_name: Source image name
            options: Search options
            
        Returns:
            List of similar search results
        """
        if not self.status.is_ready:
            raise RuntimeError("System not ready")
        
        options = options or SearchOptions()
        
        try:
            with self.perf_monitor.timer("search_by_image"):
                # Validate input parameters
                if not folder_name or not image_name:
                    raise ValueError("Folder name and image name must be provided")
                
                # Get metadata for the query image
                query_metadata = self.metadata_manager.get_metadata(folder_name, image_name)
                if not query_metadata:
                    self._log_warning("Query image not found", folder=folder_name, image=image_name)
                    return []
                
                if query_metadata.clip_features is None:
                    self._log_warning("Query image has no features", folder=folder_name, image=image_name)
                    return []
                
                # Search using FAISS
                results = self.faiss_retriever.search(
                    query_metadata.clip_features, 
                    options.limit,
                    validate_results=options.validate_results
                )
                
                # Add temporal context if requested
                if options.include_temporal_context:
                    # Check if unified index is available for temporal context
                    has_unified_index = (hasattr(self, 'unified_builder') and 
                                       self.unified_builder and 
                                       self.unified_builder.unified_index and 
                                       self.unified_builder.unified_index.is_loaded)
                    
                    for result in results:
                        if has_unified_index:
                            # Use unified index temporal context (no warnings)
                            result.temporal_context = []  # Skip temporal for unified for now
                        else:
                            # Use legacy temporal context
                            temporal_neighbors = self.metadata_manager.get_temporal_neighbors(
                                result.metadata.folder_name, 
                                result.metadata.image_name,
                                window=self.config.get("retrieval.temporal_window", 3)
                            )
                            # Convert neighbors to SearchResult objects
                            result.temporal_context = [
                                SearchResult(neighbor, 0.0, 0) for neighbor in temporal_neighbors
                            ]
                
                self._log_info(f"Image-based search completed",
                             query_folder=folder_name,
                             query_image=image_name,
                             results_count=len(results))
                
                return results
                
        except Exception as e:
            self._log_error(f"Image-based search failed: {e}", exc_info=True)
            return []
    
    def chat_search(self, 
                   question: str, 
                   context_frames: Optional[List[KeyframeMetadata]] = None) -> Dict[str, Any]:
        """
        LLM-powered conversational search with contextual understanding and validation
        
        Args:
            question: Natural language question
            context_frames: Optional context frames for the conversation
            
        Returns:
            Dictionary with response and related frames
        """
        if not self.status.is_ready:
            raise RuntimeError("System not ready")
        
        try:
            with self.perf_monitor.timer("chat_search"):
                self._log_debug(f"Chat search with question", question=self._safe_log_string(question))
                
                # Validate question
                if not question or not isinstance(question, str):
                    raise ValueError("Question must be a non-empty string")
                
                question = question.strip()
                if len(question) > 1000:
                    question = question[:1000]
                    self._log_warning("Question truncated to 1000 characters")
                
                # Use LLM to understand and expand the question
                expanded_queries = self.llm_processor.expand_query(question)
                
                # Search for relevant frames
                all_results = []
                for query in expanded_queries[:3]:  # Limit to top 3 expansions
                    search_options = SearchOptions(mode="hybrid", limit=20)
                    results = self.search(query, search_options)
                    all_results.extend(results)
                
                # Merge and deduplicate results
                unique_results = self.data_processor.merge_results([all_results])
                top_results = unique_results[:10]
                
                # Generate contextual response
                context_metadata = context_frames or [r.metadata for r in top_results]
                llm_response = self.llm_processor.chat_about_frames(question, context_metadata)
                
                response = {
                    "answer": llm_response,
                    "related_frames": top_results,
                    "expanded_queries": expanded_queries,
                    "context_used": len(context_metadata)
                }
                
                self._log_info(f"Chat search completed",
                             question_length=len(question),
                             related_frames=len(top_results))
                
                return response
                
        except Exception as e:
            self._log_error(f"Chat search failed: {e}", exc_info=True)
            return {
                "answer": "I'm sorry, I couldn't process your question right now.",
                "related_frames": [],
                "expanded_queries": [question],
                "context_used": 0,
                "error": str(e)
            }
    
    def chat(self, message: str, selected_frames: Optional[List[str]] = None, user_id: str = "default_user") -> Dict[str, Any]:
        """
        Direct chat interface using OpenAI and integrated search
        
        Args:
            message: User's message
            selected_frames: Optional list of selected frame identifiers for vision analysis
            user_id: User identifier for session management
            
        Returns:
            Dictionary containing response and search results if applicable
        """
        try:
            # Check for simple greetings first - handle without full processing
            message_lower = message.lower().strip()
            simple_greetings = ['hi', 'hello', 'xin chào', 'chào', 'chào bạn', 'xin chao', 'hey']
            
            if message_lower in simple_greetings:
                return {
                    'response_content': "Xin chào! Tôi là trợ lý AI cho hệ thống tìm kiếm keyframes video. Bạn có thể hỏi tôi về hệ thống hoặc yêu cầu tìm kiếm những cảnh cụ thể trong video.",
                    'search_performed': False,
                    'search_results': [],
                    'vision_analysis': False
                }
            
            # Use the structured chat_with_user method (OpenAI only)
            if hasattr(self.llm_processor, 'chat_with_user'):
                # Let chat_with_user handle image processing to avoid duplication
                if selected_frames:
                    self._log_info(f"Passing {len(selected_frames)} selected frames to chat interface")
                
                # Call the structured chat method (OpenAI)
                result = self.llm_processor.chat_with_user(
                    message=message,
                    images=None,  # Let chat_with_user extract from selected_frames
                    selected_frames=selected_frames
                )
                
                # Return the structured result
                return result
                
            else:
                # Fallback to conversational agent if new method not available
                if hasattr(self.llm_processor, 'conversational_agent') and self.llm_processor.conversational_agent:
                    response = self.llm_processor.conversational_agent.run(message, user_id=user_id)
                    content = response.content if hasattr(response, 'content') else str(response)
                    return {
                        'response_content': content,
                        'search_performed': False,
                        'search_results': [],
                        'vision_analysis': False
                    }
                else:
                    # Ultimate fallback
                    result = self.chat_search(message)
                    return {
                        'response_content': result.get("answer", "Sorry, I couldn't process your message."),
                        'search_performed': bool(result.get("related_frames", [])),
                        'search_results': result.get("related_frames", []),
                        'vision_analysis': False
                    }
                    
        except Exception as e:
            self._log_error(f"Chat failed: {e}", exc_info=True)
            return {
                'response_content': f"Sorry, I encountered an error: {str(e)}",
                'search_performed': False,
                'search_results': [],
                'vision_analysis': False
            }
    
    # =================== SYSTEM MANAGEMENT ===================
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health information with detailed diagnostics
        
        Returns:
            Dictionary of system health information
        """
        try:
            if not self.health_monitor:
                self.health_monitor = SystemHealthMonitor(self.logger)
            
            return self.health_monitor.comprehensive_health_check(self)
            
        except Exception as e:
            self._log_error(f"Health check failed: {e}")
            return {
                "overall_health": "error",
                "timestamp": time.time(),
                "error": str(e),
                "issues": ["Health check system failed"],
                "recommendations": ["Restart system or check logs"]
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics with health information
        
        Returns:
            Dictionary of system statistics
        """
        try:
            stats = {
                "system_status": asdict(self.status),
                "performance": self.perf_monitor.get_stats(),
                "cache_stats": {
                    "memory_usage": self.cache.current_memory,
                    "cache_hits": len(self.cache.memory_cache)
                },
                "index_stats": {},
                "metadata_stats": {},
                "health": self.get_system_health()
            }
            
            # Index statistics
            if self.faiss_retriever and self.faiss_retriever.index:
                stats["index_stats"] = {
                    "index_size": self.faiss_retriever.index.ntotal,
                    "dimension": self.faiss_retriever.dimension,
                    "index_type": self.faiss_retriever.index_type,
                    "is_trained": self.faiss_retriever.is_trained,
                    "gpu_enabled": self.faiss_retriever.use_gpu
                }
            
            # Metadata statistics
            if self.metadata_manager and self.metadata_manager.metadata_db:
                total_metadata = sum(len(folder) for folder in self.metadata_manager.metadata_db.values())
                stats["metadata_stats"] = {
                    "total_keyframes": total_metadata,
                    "folders": len(self.metadata_manager.metadata_db),
                    "has_temporal_index": len(self.metadata_manager.temporal_index) > 0,
                    "similarity_graph_size": len(self.metadata_manager.similarity_graph)
                }
            
            # System metrics
            stats.update(self.system_metrics)
            
            return stats
            
        except Exception as e:
            self._log_error(f"Failed to get system stats: {e}")
            return {"error": str(e)}
    
    def optimize_index(self) -> bool:
        """
        Optimize FAISS index for better performance
        
        Returns:
            True if optimization successful
        """
        if not self.status.index_loaded:
            self._log_warning("No index loaded for optimization")
            return False
        
        try:
            with self.perf_monitor.timer("optimize_index"):
                # Check system health before optimization
                health = self.get_system_health()
                if health["overall_health"] != "healthy":
                    self._log_warning(f"System unhealthy, optimization may not be effective: {health['issues']}")
                
                optimization_results = {
                    "index_validated": False,
                    "memory_optimized": False,
                    "gpu_optimized": False
                }
                
                # Validate index integrity
                try:
                    validation = self.faiss_retriever.validator.validate_index_metadata_consistency(
                        self.faiss_retriever.index,
                        self.faiss_retriever.id_to_metadata
                    )
                    optimization_results["index_validated"] = validation["is_consistent"]
                except Exception as e:
                    self._log_error(f"Index validation during optimization failed: {e}")
                
                # Memory optimization (cleanup cache)
                try:
                    self.cache.clear_cache(older_than_days=1)
                    optimization_results["memory_optimized"] = True
                except Exception as e:
                    self._log_error(f"Memory optimization failed: {e}")
                
                # GPU optimization check
                if self.faiss_retriever.use_gpu:
                    optimization_results["gpu_optimized"] = True
                
                success = any(optimization_results.values())
                
                self._log_info("Index optimization completed", 
                             success=success,
                             results=optimization_results)
                return success
                
        except Exception as e:
            self._log_error(f"Index optimization failed: {e}")
            return False
    
    def cleanup(self) -> None:
        """Cleanup temporary files and cache with enhanced error handling"""
        try:
            with self.perf_monitor.timer("system_cleanup"):
                cleanup_results = {
                    "cache_cleaned": 0,
                    "temp_files_cleaned": 0,
                    "errors": []
                }
                
                # Cleanup cache
                try:
                    removed_cache = self.cache.clear_cache(older_than_days=7)
                    cleanup_results["cache_cleaned"] = removed_cache
                except Exception as e:
                    cleanup_results["errors"].append(f"Cache cleanup failed: {e}")
                
                # Cleanup temporary files
                try:
                    removed_temp = self.file_manager.cleanup_temp_files()
                    cleanup_results["temp_files_cleaned"] = removed_temp
                except Exception as e:
                    cleanup_results["errors"].append(f"Temp file cleanup failed: {e}")
                
                # Cleanup old log files
                try:
                    self._cleanup_old_logs()
                except Exception as e:
                    cleanup_results["errors"].append(f"Log cleanup failed: {e}")
                
                self._log_info(f"System cleanup completed",
                             **cleanup_results)
                
        except Exception as e:
            self._log_error(f"Cleanup failed: {e}")
    
    # =================== VALIDATION METHODS ===================
    
    def _validate_keyframes_folder(self, keyframe_folder: str) -> None:
        """Validate keyframes folder structure"""
        if not os.path.exists(keyframe_folder):
            raise FileNotFoundError(f"Keyframes folder not found: {keyframe_folder}")
        
        validator = DataConsistencyValidator(self.logger)
        validation = validator.validate_keyframes_folder(keyframe_folder)
        
        if not validation["is_valid"]:
            raise ValueError(f"Invalid keyframes folder: {validation['issues']}")
        
        # Check minimum requirements
        stats = validation["stats"]
        if stats["total_images"] < 10:
            raise ValueError(f"Insufficient images in keyframes folder: {stats['total_images']} (minimum: 10)")
        
        if stats["total_folders"] == 0:
            raise ValueError("No valid folders found in keyframes directory")
    
    def _pre_build_validation(self, keyframe_folder: str) -> None:
        """Pre-build validation checks"""
        # Check disk space
        try:
            import shutil
            _, _, free_space = shutil.disk_usage(keyframe_folder)
            required_space = 1024 * 1024 * 1024  # 1GB minimum
            
            if free_space < required_space:
                self._log_warning(f"Low disk space: {free_space / 1024**3:.1f}GB available")
        except Exception:
            pass  # Non-critical
        
        # Check component readiness
        if not self.clip_processor:
            raise RuntimeError("CLIP processor not initialized")
        
        if not self.faiss_retriever:
            raise RuntimeError("FAISS retriever not initialized")
        
        if not self.metadata_manager:
            raise RuntimeError("Metadata manager not initialized")
    
    def _post_build_validation(self) -> None:
        """Post-build validation checks - SIMPLIFIED"""
        try:
            health = self.get_system_health()
            
            if health["overall_health"] != "healthy":
                issues_str = "; ".join(health.get("issues", [])[:3])  # Limit to 3 issues
                self._log_warning(f"Post-build validation found issues: {issues_str}")
                # Don't raise exception, just log warning
            
            # SIMPLIFIED test - just check if components are ready
            if not self.faiss_retriever or not self.faiss_retriever.index:
                raise RuntimeError("FAISS index not available after build")
                
            if len(self.faiss_retriever.id_to_metadata) == 0:
                raise RuntimeError("No metadata available after build")
            
            # Skip search test to avoid circular dependency
            self._log_info("Post-build validation completed")
            
        except Exception as e:
            self._log_error(f"Post-build validation failed: {e}")
            # Don't raise exception to avoid build failure
    
    def _pre_load_validation(self, index_path: str) -> None:
        """Pre-load validation checks"""
        index_path = Path(index_path)
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index directory not found: {index_path}")
        
        required_files = ["index.faiss", "metadata.json"]
        missing_files = []
        
        for file_name in required_files:
            file_path = index_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")
        
        # Check file sizes
        faiss_file = index_path / "index.faiss"
        if faiss_file.stat().st_size < 1000:  # Very small file
            raise ValueError(f"FAISS index file appears corrupted (size: {faiss_file.stat().st_size} bytes)")
    
    def _post_load_validation(self) -> Dict[str, Any]:
        """Post-load validation checks"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check basic functionality
            if not self.faiss_retriever.index:
                validation_result["errors"].append("FAISS index not loaded")
            
            if len(self.faiss_retriever.id_to_metadata) == 0:
                validation_result["errors"].append("No metadata loaded")
            
            # Check consistency
            health = self.get_system_health()
            if health["overall_health"] != "healthy":
                validation_result["errors"].extend(health.get("issues", []))
            
            # Test basic operations
            try:
                # Test vector search capability
                if self.faiss_retriever.index and self.faiss_retriever.index.ntotal > 0:
                    test_vector = np.random.random((1, self.faiss_retriever.dimension)).astype(np.float32)
                    _, _ = self.faiss_retriever.index.search(test_vector, 1)
            except Exception as e:
                validation_result["errors"].append(f"Index search test failed: {e}")
            
        except Exception as e:
            validation_result["errors"].append(f"Post-load validation error: {e}")
        
        validation_result["is_valid"] = len(validation_result["errors"]) == 0
        return validation_result
    
    def _validate_search_readiness(self) -> Dict[str, Any]:
        """Validate system readiness for search operations"""
        readiness = {
            "is_ready": True,
            "issues": []
        }
        
        # Check if unified index is available first
        has_unified_index = (hasattr(self, 'unified_builder') and 
                           self.unified_builder and 
                           self.unified_builder.unified_index and 
                           self.unified_builder.unified_index.is_loaded)
        
        # Also check if unified system is ready (additional check)
        unified_system_ready = (hasattr(self, '_unified_system_ready') and 
                               getattr(self, '_unified_system_ready', False))
        
        # Check if remote index is available
        has_remote_index = (hasattr(self, 'remote_indexes') and 
                          self.remote_indexes and 
                          len(self.remote_indexes) > 0)
        
        # System is ready if any of these are true
        system_ready = has_unified_index or unified_system_ready or has_remote_index
        
        # Check traditional FAISS retriever only if no other index is available
        if not system_ready:
            if not self.faiss_retriever or not self.faiss_retriever.index:
                readiness["issues"].append("FAISS index not available")
            
            if not self.faiss_retriever or len(self.faiss_retriever.id_to_metadata) == 0:
                readiness["issues"].append("No metadata available for search results")
        
        if not self.clip_processor:
            readiness["issues"].append("CLIP processor not available")
        
        readiness["is_ready"] = len(readiness["issues"]) == 0
        return readiness
    
    def _validate_search_results(self, results: List[SearchResult], query: str, skip_file_validation: bool = False) -> List[SearchResult]:
        """Validate and filter search results"""
        validated_results = []
        
        for result in results:
            try:
                # Validate metadata
                result.metadata._validate()
                
                # Check file existence (skip for unified index since thumbnails are embedded)
                if not skip_file_validation:
                    if not result.metadata.validate_file_exists():
                        self._log_warning(f"Result file not found: {result.metadata.file_path}")
                        continue
                
                # Validate similarity score - should be in range [0, 1]
                if not (0 <= result.similarity_score <= 1):
                    self._log_warning(f"Invalid similarity score: {result.similarity_score}, clamping to valid range")
                    result.similarity_score = max(0.0, min(1.0, result.similarity_score))

                if getattr(result, 'ranking_score', None) is None:
                    result.ranking_score = result.similarity_score
                
                validated_results.append(result)
                
            except Exception as e:
                self._log_warning(f"Invalid search result: {e}")
                continue
        
        return validated_results

    def _get_ranking_score(self, result: SearchResult) -> float:
        """Return ranking score when available, otherwise semantic similarity."""
        ranking_score = getattr(result, 'ranking_score', None)
        if ranking_score is None:
            return float(getattr(result, 'similarity_score', 0.0))
        return float(ranking_score)
    
    # =================== INTERNAL HELPER METHODS ===================
    
    def _initialize_base_components(self, config_path: Optional[str]) -> None:
        """Initialize base system components with validation"""
        from utils import Config, Logger, FileManager, DataProcessor, CacheManager, PerformanceMonitor
        
        # Initialize configuration
        self.config = Config(config_path)
        self.status.components_loaded["config"] = True
        
        # Initialize logger
        self.logger = Logger(
            log_level=self.config.get("system.debug", False) and "DEBUG" or "INFO",
            log_dir=self.config.get("paths.logs", "logs/"),
            config=self.config
        )
        self.status.components_loaded["logger"] = True
        
        # Initialize other base components
        self.file_manager = FileManager(self.logger)
        self.data_processor = DataProcessor(self.logger)
        self.cache = CacheManager(
            cache_dir=self.config.get("paths.cache", ".cache/"),
            max_memory_mb=self.config.get("performance.cache_memory_mb", 512),
            config=self.config
        )
        self.perf_monitor = PerformanceMonitor(self.logger, self.config)
        
        # Initialize health monitor
        self.health_monitor = SystemHealthMonitor(self.logger)
    
    def _initialize_ai_components(self) -> None:
        """Initialize AI components with enhanced error handling (OpenAI only)"""
        try:
            # Initialize CLIP processor
            model_path = self.config.get("paths.models", "models/") + "clip_model/"
            if not os.path.exists(model_path):
                # Check environment variable first, then fallback
                model_path = self.config.get("clip.model_name", "openai/clip-vit-base-patch32")
            
            self.clip_processor = CLIPFeatureExtractor(model_path, self.config, self.logger)
            self.active_query_encoder = self.clip_processor
            self.status.components_loaded["clip_processor"] = True
            
            # Initialize FAISS retriever
            self.faiss_retriever = FAISSRetriever(self.config, self.logger, self.cache)
            self.status.components_loaded["faiss_retriever"] = True
            
            # Initialize LLM processor (OpenAI only)
            self.llm_processor = LLMProcessor(self.config, self.logger, self.cache, retrieval_system=self)
            self.status.components_loaded["llm_processor"] = True
            
            # Initialize metadata manager
            self.metadata_manager = MetadataManager(self.config, self.logger)
            self.status.components_loaded["metadata_manager"] = True
            
            # Initialize temporal analyzer
            self.temporal_analyzer = TemporalAnalyzer(self.config, self.logger)
            self.status.components_loaded["temporal_analyzer"] = True
            
            # Initialize universal query translator
            self.query_translator = UniversalQueryTranslator(
                self.config, self.logger, self.cache
            )
            self.status.components_loaded["query_translator"] = True
            
        except Exception as e:
            self._log_error(f"AI components initialization failed: {e}")
            raise
    
    def _ensure_components_loaded(self) -> None:
        """Ensure all AI components are loaded, initialize if needed"""
        try:
            # Check if components are already loaded
            required_components = ['clip_processor', 'faiss_retriever', 'llm_processor', 'metadata_manager', 'temporal_analyzer']
            
            for component in required_components:
                if not self.status.components_loaded.get(component, False):
                    self._log_info(f"🔄 Component {component} not loaded, initializing AI components...")
                    self._initialize_ai_components()
                    break
                    
            # Verify all required components exist as attributes
            if not hasattr(self, 'clip_processor') or not hasattr(self, 'faiss_retriever'):
                self._log_info("🔄 Required components missing, initializing AI components...")
                self._initialize_ai_components()
                
        except Exception as e:
            self._log_error(f"Component loading verification failed: {e}")
            # Try to initialize components anyway
            self._initialize_ai_components()
    
    def _ensure_temporal_index_loaded(self) -> None:
        """Ensure temporal index is properly loaded, rebuild if empty or missing"""
        try:
            if not hasattr(self.faiss_retriever, 'temporal_index') or not self.faiss_retriever.temporal_index:
                self._log_info("🔄 Temporal index is empty, rebuilding from loaded metadata...")
                
                # Get metadata from FAISS retriever
                if hasattr(self.faiss_retriever, 'id_to_metadata') and self.faiss_retriever.id_to_metadata:
                    # Group metadata by folder
                    folder_groups = {}
                    for idx, metadata in self.faiss_retriever.id_to_metadata.items():
                        # Use folder_name from metadata directly if available
                        folder_name = metadata.folder_name if hasattr(metadata, 'folder_name') and metadata.folder_name else Path(metadata.file_path).parent.name
                        if folder_name not in folder_groups:
                            folder_groups[folder_name] = []
                        folder_groups[folder_name].append(metadata)
                    
                    # Rebuild temporal index for each folder
                    self.faiss_retriever.temporal_index = {}
                    for folder_name, folder_metadata in folder_groups.items():
                        valid_frames = []
                        for meta in folder_metadata:
                            if isinstance(meta.frame_id, int) and meta.frame_id >= 0:
                                valid_frames.append((meta.frame_id, meta.image_name))
                            else:
                                self._log_warning(f"Invalid frame ID for {meta.folder_name}/{meta.image_name}: {meta.frame_id}")
                        
                        valid_frames.sort(key=lambda x: x[0])
                        self.faiss_retriever.temporal_index[folder_name] = valid_frames
                        self._log_info(f"   - {folder_name}: {len(valid_frames)} frames")
                    
                    self._log_info(f"✅ Temporal index rebuilt for {len(folder_groups)} folders: {list(folder_groups.keys())}")
                    self._log_info(f"🔍 Final temporal_index keys: {list(self.faiss_retriever.temporal_index.keys())}")
                else:
                    self._log_warning("⚠️ No metadata available to rebuild temporal index")
            else:
                self._log_info(f"📊 Temporal index loaded: {len(self.faiss_retriever.temporal_index)} folders")
                
        except Exception as e:
            self._log_error(f"Failed to ensure temporal index: {e}")
    
    def _perform_initial_health_check(self) -> None:
        """Perform initial health check after initialization"""
        try:
            health = self.get_system_health()
            self.status.health_status = health
            
            if health["overall_health"] != "healthy":
                self._log_warning(f"Initial health check found issues: {health['issues']}")
            
        except Exception as e:
            self._log_error(f"Initial health check failed: {e}")
    
    def _save_system(self, index_path: str) -> None:
        """Save complete system to disk with validation"""
        index_path = Path(index_path)
        index_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save FAISS index
            self.faiss_retriever.save_index(index_path, validate_before_save=self.enable_validation)
            
            # Save metadata
            metadata_path = index_path / "metadata.json"
            self.metadata_manager.save_metadata(metadata_path)
            
            # Save system info
            system_info = {
                "version": "2.1",
                "created_at": time.time(),
                "build_stats": {
                    "keyframes_processed": len(self.metadata_list),
                    "folders_processed": len(self.keyframe_data),
                    "features_extracted": len(self.feature_matrix) if self.feature_matrix is not None else 0
                },
                "metrics": self.system_metrics,
                "health": self.get_system_health(),
                "config": dict(self.config.config)
            }
            
            system_info_path = index_path / "system_info.json"
            with open(system_info_path, 'w', encoding='utf-8') as f:
                json.dump(system_info, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            self._log_error(f"System save failed: {e}")
            raise
    
    def _load_system_info(self, index_path: str) -> None:
        """Load system info from disk"""
        try:
            system_info_path = Path(index_path) / "system_info.json"
            if system_info_path.exists():
                with open(system_info_path, 'r', encoding='utf-8') as f:
                    system_info = json.load(f)
                    self.system_metrics.update(system_info.get("metrics", {}))
                    
                    # Check version compatibility
                    saved_version = system_info.get("version", "1.0")
                    if saved_version != "2.1":
                        self._log_warning(f"Loading system from different version: {saved_version}")
        except Exception as e:
            self._log_warning(f"Failed to load system info: {e}")
    
    def _cleanup_partial_build(self) -> None:
        """Cleanup after failed build"""
        try:
            self.feature_matrix = None
            self.metadata_list = []
            self.keyframe_data = {}
            
            if self.faiss_retriever:
                self.faiss_retriever._clear_index_data()
            
            self.status.is_ready = False
            self.status.index_loaded = False
            
        except Exception as e:
            self._log_error(f"Cleanup after failed build error: {e}")
    
    def _clear_system_state(self) -> None:
        """Clear system state after failed load"""
        try:
            self.status.is_ready = False
            self.status.index_loaded = False
            
            if self.faiss_retriever:
                self.faiss_retriever._clear_index_data()
            
            if self.metadata_manager:
                self.metadata_manager.metadata_db.clear()
                self.metadata_manager.frame_mappings.clear()
                self.metadata_manager.temporal_index.clear()
            
        except Exception as e:
            self._log_error(f"Clear system state error: {e}")
    
    def _cleanup_old_logs(self) -> None:
        """Cleanup old log files"""
        try:
            logs_dir = Path(self.config.get("paths.logs", "logs/"))
            if logs_dir.exists():
                cutoff_time = time.time() - (30 * 24 * 3600)  # 30 days
                
                for log_file in logs_dir.glob("*.log"):
                    if log_file.stat().st_mtime < cutoff_time:
                        log_file.unlink()
                        
        except Exception as e:
            self._log_error(f"Log cleanup failed: {e}")
    
    def _index_exists(self, index_path: str) -> bool:
        """Check if index exists with validation"""
        index_path = Path(index_path)
        required_files = ["index.faiss", "metadata.json"]
        
        return all((index_path / file_name).exists() for file_name in required_files)
    
    def _update_system_stats(self) -> None:
        """Update system statistics"""
        try:
            if self.faiss_retriever and self.faiss_retriever.index:
                self.system_metrics.update({
                    "index_size": self.faiss_retriever.index.ntotal,
                    "feature_dimension": self.faiss_retriever.dimension,
                    "metadata_count": len(self.faiss_retriever.id_to_metadata),
                    "last_updated": time.time()
                })
        except Exception as e:
            self._log_error(f"System stats update failed: {e}")
    
    def _update_progress(self, 
                        callback: Optional[Callable], 
                        step: int, 
                        total: int, 
                        message: str) -> None:
        """Update progress callback with error handling"""
        try:
            if callback:
                callback(step, total, message)
        except Exception as e:
            self._log_warning(f"Progress callback failed: {e}")
        
        if self.verbose:
            self._log_info(f"Progress: {step}/{total} - {message}")
    
    def _safe_log_string(self, text: str) -> str:
        """Create safe string for logging (handle Unicode)"""
        try:
            if isinstance(text, str) and len(text) > 100:
                return text[:100] + "..."
            return text
        except Exception:
            return "[unsafe_string]"
    
    # =================== SEARCH IMPLEMENTATION METHODS ===================
    
    def _optimize_query_for_clip(self, query: str) -> str:
        """
        Enhanced query optimization for better CLIP understanding and accuracy
        
        CLIP performs better with:
        - Visual descriptive terms
        - Complete descriptive phrases
        - Specific object/scene descriptions
        - Contextual visual information
        """
        if not query or len(query.strip()) == 0:
            return query
        
        # Clean and normalize query
        query = query.strip()
        original_query = query
        
        # Advanced visual context prefixes for better CLIP understanding
        visual_prefixes = {
            # People and actions (enhanced)
            r'\b(người đàn ông|man|male person)\b': 'a photo of a man',
            r'\b(người phụ nữ|woman|female person)\b': 'a photo of a woman', 
            r'\b(nhiều người|several people|group)\b': 'a photo of many people',
            r'\b(người|person|people)\b': 'a photo of people',
            r'\b(trẻ em|children|child|kid)\b': 'a photo of children',
            r'\b(gia đình|family)\b': 'a photo of a family',
            
            # Actions and activities (enhanced)
            r'\b(đang sửa|repairing|fixing|maintaining)\b': 'someone repairing or fixing',
            r'\b(đang làm việc|working|work)\b': 'person working',
            r'\b(nói chuyện|talking|speaking|conversation)\b': 'people talking or having conversation',
            r'\b(ngồi|sitting|seated)\b': 'person sitting',
            r'\b(đứng|standing|stand)\b': 'person standing',
            r'\b(đi bộ|walking|walk)\b': 'person walking',
            r'\b(chạy|running|run)\b': 'person running',
            r'\b(ăn|eating|eat)\b': 'person eating',
            r'\b(uống|drinking|drink)\b': 'person drinking',
            r'\b(ngủ|sleeping|sleep)\b': 'person sleeping',
            r'\b(đọc|reading|read)\b': 'person reading',
            r'\b(viết|writing|write)\b': 'person writing',
            
            # Objects and vehicles (enhanced)
            r'\b(xe đạp|bicycle|bike|cycle)\b': 'a photo of a bicycle or bike',
            r'\b(xe oto|xe hơi|car|automobile|ô tô)\b': 'a photo of a car or automobile',
            r'\b(xe máy|motorcycle|motorbike)\b': 'a photo of a motorcycle',
            r'\b(xe buýt|bus)\b': 'a photo of a bus',
            r'\b(xe tải|truck)\b': 'a photo of a truck',
            r'\b(máy bay|airplane|aircraft)\b': 'a photo of an airplane',
            r'\b(tàu|train|ship|boat)\b': 'a photo of a train or ship',
            
            # Buildings and structures (enhanced)
            r'\b(nhà|house|home)\b': 'a photo of a house or home',
            r'\b(tòa nhà|building|office)\b': 'a photo of a building',
            r'\b(trường học|school)\b': 'a photo of a school',
            r'\b(bệnh viện|hospital)\b': 'a photo of a hospital',
            r'\b(cửa hàng|shop|store)\b': 'a photo of a shop or store',
            r'\b(nhà hàng|restaurant)\b': 'a photo of a restaurant',
            r'\b(khách sạn|hotel)\b': 'a photo of a hotel',
            
            # Nature and environment (enhanced)
            r'\b(cây xanh|cây|tree|trees)\b': 'a photo of trees or green vegetation',
            r'\b(hoa|flower|flowers)\b': 'a photo of flowers',
            r'\b(cỏ|grass|lawn)\b': 'a photo of grass or lawn',
            r'\b(rừng|forest|woods)\b': 'a photo of forest or woods',
            r'\b(núi|mountain|hill)\b': 'a photo of mountains or hills',
            r'\b(biển|sea|ocean)\b': 'a photo of sea or ocean',
            r'\b(sông|river|stream)\b': 'a photo of river or stream',
            r'\b(hồ|lake|pond)\b': 'a photo of lake or pond',
            
            # Locations and settings (enhanced)
            r'\b(công viên|park|garden)\b': 'a photo of a park or garden',
            r'\b(đường phố|street|road)\b': 'a photo of street or road',
            r'\b(quảng trường|square|plaza)\b': 'a photo of a square or plaza',
            r'\b(chợ|market|marketplace)\b': 'a photo of a market',
            r'\b(bãi biển|beach|shore)\b': 'a photo of a beach',
            r'\b(sân bay|airport)\b': 'a photo of an airport',
            
            # Indoor/outdoor context (enhanced)
            r'\b(trong nhà|indoor|inside|interior)\b': 'indoor scene showing',
            r'\b(ngoài trời|outdoor|outside|exterior)\b': 'outdoor scene showing',
            r'\b(trong phòng|in room|room)\b': 'indoor room scene with',
            r'\b(trên đường|on street|street)\b': 'street scene with',
            
            # Time and lighting (enhanced)
            r'\b(ban ngày|daytime|day|sáng|morning)\b': 'bright daytime photo of',
            r'\b(ban đêm|nighttime|night|tối|evening)\b': 'nighttime or evening photo of',
            r'\b(hoàng hôn|sunset|dusk)\b': 'sunset photo showing',
            r'\b(bình minh|sunrise|dawn)\b': 'sunrise photo showing',
            r'\b(trưa|noon|midday)\b': 'midday photo of',
            
            # Colors (enhanced)
            r'\b(màu đỏ|red|đỏ)\b': 'red colored',
            r'\b(màu xanh|blue|xanh dương)\b': 'blue colored',
            r'\b(màu vàng|yellow|vàng)\b': 'yellow colored',
            r'\b(màu xanh lá|green|xanh lá cây)\b': 'green colored',
            r'\b(màu trắng|white|trắng)\b': 'white colored',
            r'\b(màu đen|black|đen)\b': 'black colored',
        }
        
        # Apply visual context enhancement with priority system
        import re  # Ensure re module is available in function scope
        best_match = None
        best_priority = 0
        
        for pattern, prefix in visual_prefixes.items():
            if re.search(pattern, query, re.IGNORECASE):
                # Calculate priority based on specificity
                priority = len(pattern) + (10 if 'photo of' in prefix else 0)
                if priority > best_priority:
                    best_match = prefix
                    best_priority = priority
        
        # Apply best matching visual prefix
        if best_match and not query.lower().startswith(('a photo', 'an image', 'a picture', 'a scene')):
            query = f"{best_match}, {query}"
        
        # Additional semantic enhancements
        if 'hình ảnh' in query.lower() and not any(x in query.lower() for x in ['photo', 'image', 'picture']):
            query = query.replace('hình ảnh', 'a photo showing')
            
        # Enhance specific combinations
        enhancement_patterns = {
            r'\b(người.*đang.*)\b': lambda m: f"a detailed photo of {m.group(1)}",
            r'\b(xe.*màu.*)\b': lambda m: f"a clear photo of {m.group(1)}",
            r'\b(.*trong.*nhà.*)\b': lambda m: f"an indoor photo showing {m.group(1)}",
            r'\b(.*ngoài.*trời.*)\b': lambda m: f"an outdoor photo showing {m.group(1)}",
        }
        
        for pattern, replacement in enhancement_patterns.items():
            if re.search(pattern, original_query, re.IGNORECASE):
                enhanced = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                if enhanced != query:
                    query = enhanced
                    break
        
        return query
        import re
        original_query = query.lower()
        enhanced_query = query
        
        for pattern, replacement in visual_prefixes.items():
            if re.search(pattern, original_query, re.IGNORECASE):
                # If query is very short, replace completely
                if len(query.split()) <= 2:
                    enhanced_query = replacement
                else:
                    # For longer queries, enhance with visual context
                    enhanced_query = f"{replacement}, {query}"
                break
        
        # Add "a photo of" prefix if query doesn't have visual context
        if enhanced_query == query and not any(prefix in query.lower() for prefix in 
                                               ['photo', 'image', 'picture', 'scene', 'showing']):
            if len(query.split()) <= 3:
                enhanced_query = f"a photo of {query}"
        
        # Clean up redundant phrases
        enhanced_query = re.sub(r'\ba photo of a photo of\b', 'a photo of', enhanced_query, flags=re.IGNORECASE)
        enhanced_query = re.sub(r'\bphoto.*photo\b', 'photo', enhanced_query, flags=re.IGNORECASE)
        
        # Log query optimization
        if enhanced_query != query:
            self._log_debug(f"Query optimized for CLIP", original=query, optimized=enhanced_query)
        
        return enhanced_query
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """
        Generate multiple query variations to improve search recall
        
        Creates variations with:
        - Original query
        - Synonyms and alternative phrasings
        - Different visual contexts
        - Simplified versions
        """
        variations = [query]  # Always include original query first
        
        query_lower = query.lower()
        
        # Enhanced semantic variations for large datasets (150GB+)
        enhanced_semantic_variations = {
            # People variations - Vietnamese-English enhanced
            'person': ['people', 'man', 'woman', 'individual', 'human', 'someone'],
            'people': ['person', 'group', 'crowd', 'individuals', 'humans', 'many people'],
            'man': ['person', 'male', 'guy', 'gentleman', 'người đàn ông'],
            'woman': ['person', 'female', 'lady', 'girl', 'người phụ nữ'],
            'child': ['kid', 'baby', 'toddler', 'young person', 'trẻ em'],
            'người': ['person', 'people', 'individual', 'human'],
            'người đàn ông': ['man', 'male person', 'gentleman', 'guy'],
            'người phụ nữ': ['woman', 'female person', 'lady', 'girl'],
            'nhiều người': ['many people', 'group of people', 'crowd', 'several people'],
            
            # Action variations - Vietnamese-English enhanced
            'talking': ['speaking', 'conversation', 'discussion', 'chat', 'nói chuyện'],
            'walking': ['moving', 'strolling', 'going', 'đi bộ'],
            'sitting': ['seated', 'resting', 'ngồi'],
            'standing': ['upright', 'vertical', 'đứng'],
            'running': ['jogging', 'racing', 'sprinting', 'chạy'],
            'working': ['làm việc', 'job', 'labor', 'activity'],
            'đang': ['is', 'are', 'currently', 'in the process of'],
            'sửa': ['repairing', 'fixing', 'working on', 'maintaining'],
            'nói chuyện': ['talking', 'speaking', 'conversation', 'discussing'],
            
            # Object variations - Vietnamese enhanced
            'car': ['vehicle', 'automobile', 'auto', 'xe hơi', 'ô tô'],
            'bike': ['bicycle', 'cycle', 'xe đạp'],
            'xe đạp': ['bicycle', 'bike', 'cycle'],
            'xe': ['vehicle', 'car', 'transport', 'automobile'],
            'ô tô': ['car', 'automobile', 'vehicle'],
            'house': ['building', 'home', 'structure', 'nhà'],
            'tree': ['plant', 'vegetation', 'foliage', 'cây'],
            'water': ['lake', 'river', 'ocean', 'sea', 'nước'],
            'road': ['street', 'path', 'highway', 'đường'],
            
            # Scene variations - Vietnamese enhanced
            'outdoor': ['outside', 'exterior', 'open air', 'ngoài trời'],
            'indoor': ['inside', 'interior', 'enclosed', 'trong nhà'],
            'daytime': ['day', 'daylight', 'bright', 'ban ngày'],
            'nighttime': ['night', 'dark', 'evening', 'ban đêm'],
            'ngoài trời': ['outdoor', 'outside', 'exterior'],
            'trong nhà': ['indoor', 'inside', 'interior'],
            'hình ảnh': ['image', 'photo', 'picture', 'visual'],
        }
        
        # Create variations with enhanced synonyms (more for large datasets)
        for original, synonyms in enhanced_semantic_variations.items():
            if original in query_lower:
                for synonym in synonyms[:3]:  # Use more synonyms for large datasets
                    variant = query.replace(original, synonym)
                    if variant != query and variant not in variations:
                        variations.append(variant)
        
        # Generate simplified versions (remove modifiers)
        words = query.split()
        if len(words) > 3:
            # Keep main nouns and verbs, remove adjectives
            import re
            simplified = re.sub(r'\b(very|quite|really|extremely|highly|beautiful|nice|good|bad)\b', '', query, flags=re.IGNORECASE)
            simplified = re.sub(r'\s+', ' ', simplified).strip()
            if simplified != query and simplified not in variations:
                variations.append(simplified)
        
        # Enhanced visual context variations for large datasets
        visual_contexts = [
            f"a photo of {query}",
            f"an image showing {query}",
            f"a picture of {query}",
            f"a scene with {query}",
            f"visual content of {query}",
        ]
        
        for context in visual_contexts:
            if context not in variations:
                variations.append(context)
        
        # Generate broader context variations for better recall
        words = query.split()
        if len(words) <= 2:
            variations.append(f"{query} in a photo")
            variations.append(f"showing {query}")
            variations.append(f"depicts {query}")
        
        # Add individual significant words for large dataset recall
        significant_words = [w for w in words if len(w) > 2]
        if len(significant_words) > 1:
            for word in significant_words[:4]:  # More individual words for large datasets
                if word not in variations:
                    variations.append(word)
        
        # Limit total variations for large datasets (increased from 5 to 8)
        final_variations = variations[:8]
        
        if len(final_variations) > 1:
            self._log_debug(f"Generated {len(final_variations)} query variations", 
                          original=query, 
                          variations=final_variations[1:])
        
        return final_variations
    
    def _get_unified_build_info(self) -> Dict[str, Any]:
        """Return metadata for the active unified index, if one is loaded."""
        try:
            if (hasattr(self, 'unified_builder') and self.unified_builder and
                    self.unified_builder.unified_index and
                    self.unified_builder.unified_index.is_loaded):
                return self.unified_builder.unified_index._load_build_metadata()
        except Exception:
            pass
        return {}

    def _get_index_clip_model(self) -> Optional[str]:
        """Return the CLIP model declared by the active index."""
        build_info = self._get_unified_build_info()
        return (
            build_info.get('clip_model')
            or getattr(self.faiss_retriever, 'index_clip_model', None)
            or getattr(self, 'active_index_clip_model', None)
        )

    def _get_active_query_encoder_model(self) -> str:
        """Return the model path/name for the encoder currently used for text queries."""
        encoder = self.active_query_encoder or self.clip_processor
        return str(getattr(encoder, 'model_path', 'unknown'))

    def _get_active_query_encoder(self):
        """Return the authoritative encoder for all runtime text query embeddings."""
        return self.active_query_encoder or self.clip_processor

    def _encode_query_text(self, text: str, context: str) -> np.ndarray:
        """Encode runtime search text with the active index-compatible query encoder."""
        encoder = self._get_active_query_encoder()
        if encoder is None:
            raise RuntimeError("No active query encoder available")

        query_features = encoder.encode_text(text, validate_input=self.enable_validation)
        self._validate_query_features_against_active_index(query_features, context)
        return query_features

    def _is_btc_unified_index_active(self) -> bool:
        """True when the loaded unified index was built from BTC CLIP features."""
        build_info = self._get_unified_build_info()
        return (
            build_info.get('build_source') == 'btc_clip_features'
            or getattr(self.faiss_retriever, 'index_build_source', None) == 'btc_clip_features'
        )

    def _validate_query_features_against_active_index(self, query_features: np.ndarray, context: str) -> None:
        """Fail closed if query embedding dimension/model does not match the active FAISS index."""
        query_dim = int(query_features.shape[-1])
        index_dim = getattr(self.faiss_retriever, 'dimension', None)
        active_query_encoder = self._get_active_query_encoder_model()
        index_clip_model = self._get_index_clip_model()

        self._log_info(
            "CLIP query/index validation",
            context=context,
            active_query_encoder=active_query_encoder,
            query_embedding_shape=tuple(query_features.shape),
            index_clip_model=index_clip_model,
            faiss_dimension=index_dim,
        )

        if index_dim is not None and query_dim != int(index_dim):
            raise ValueError(
                "CLIP_QUERY_INDEX_DIMENSION_MISMATCH | "
                f"context={context} | "
                f"active_query_encoder={active_query_encoder} | "
                f"query_dimension={query_dim} | "
                f"index_clip_model={index_clip_model} | "
                f"index_dimension={index_dim}"
            )

        if index_clip_model and active_query_encoder != index_clip_model:
            raise ValueError(
                "CLIP_QUERY_INDEX_MODEL_MISMATCH | "
                f"context={context} | "
                f"active_query_encoder={active_query_encoder} | "
                f"query_dimension={query_dim} | "
                f"index_clip_model={index_clip_model} | "
                f"index_dimension={index_dim}"
            )

    def _search_clip_only(self, query: str, options: SearchOptions, query_struct: Optional[QueryStructure] = None) -> List[SearchResult]:
        """CLIP-only search implementation"""
        # Check if unified index is available and use it instead
        if (hasattr(self, 'unified_builder') and self.unified_builder and
                self.unified_builder.unified_index and
                self.unified_builder.unified_index.is_loaded):
            return self._search_with_unified_index(query, options)
        
        # Check if remote index is available
        if hasattr(self, 'remote_indexes') and self.remote_indexes and len(self.remote_indexes) > 0:
            return self._search_with_remote_index(query, options)
        
        # Fallback to legacy FAISS search
        # Encode query text
        query_features = self._encode_query_text(query, "clip_only_legacy")
        
        # Search with FAISS
        results = self.faiss_retriever.search(
            query_features, 
            options.limit,
            validate_results=options.validate_results
        )
        
        # Filter by similarity threshold
        if options.similarity_threshold > 0:
            results = [r for r in results if r.similarity_score >= options.similarity_threshold]
        
        return results
    
    def _search_with_remote_index(self, query: str, options: SearchOptions) -> List[SearchResult]:
        """Search using remote index system"""
        try:
            # Encode query text using CLIP
            query_features = self._encode_query_text(query, "remote_index_query")
            
            # Search across all connected remote indexes
            all_results = []
            
            for remote_info in self.remote_indexes:
                try:
                    remote_index = remote_info['instance']
                    
                    # Perform search on remote index
                    remote_results = remote_index.search(query_features, options.limit)
                    
                    # Convert remote results to SearchResult objects
                    for result in remote_results:
                        search_result = SearchResult(
                            similarity_score=result.get('similarity_score', 0.0),
                            ranking_score=result.get('ranking_score', result.get('similarity_score', 0.0)),
                            metadata=self._convert_remote_metadata(result.get('metadata', {})),
                            rank=result.get('rank', 0)
                        )
                        all_results.append(search_result)
                        
                except Exception as e:
                    self._log_warning(f"Remote search failed for {remote_info['name']}: {e}")
                    continue
            
            # Sort by similarity score and apply limit
            all_results.sort(key=self._get_ranking_score, reverse=True)
            results = all_results[:options.limit]
            
            # Filter by similarity threshold
            if options.similarity_threshold > 0:
                results = [r for r in results if r.similarity_score >= options.similarity_threshold]
            
            self._log_info(f"🌐 Remote search completed: {len(results)} results from {len(self.remote_indexes)} remote indexes")
            return results
            
        except Exception as e:
            self._log_error(f"Remote search failed: {e}")
            return []
    
    def _convert_remote_metadata(self, remote_metadata: Dict) -> 'KeyframeMetadata':
        """Convert remote metadata to KeyframeMetadata object"""
        try:
            # Create KeyframeMetadata from remote metadata
            return KeyframeMetadata(
                folder_name=remote_metadata.get('folder_name', 'remote'),
                image_name=remote_metadata.get('image_name', 'unknown'),
                frame_id=remote_metadata.get('frame_id', 0),
                file_path=remote_metadata.get('file_path', ''),
                # Add other fields as needed
            )
        except Exception as e:
            # Fallback to minimal metadata
            return KeyframeMetadata(
                folder_name='remote',
                image_name='unknown',
                frame_id=0,
                file_path=''
            )
    
    def _rerank_results_with_context(self, 
                                   results: List[SearchResult], 
                                   query: str, 
                                   options: SearchOptions) -> List[SearchResult]:
        """Advanced re-ranking with contextual scoring"""
        if len(results) <= 1:
            return results
        
        try:
            # 1. Calculate quality scores for each result
            scored_results = []
            query_lower = query.lower()
            
            for result in results:
                quality_score = self._calculate_result_quality_score(result, query_lower)
                scored_results.append((result, quality_score))
            
            # 2. Sort by combined score (similarity + quality)
            def combined_score(item):
                result, quality = item
                # Weighted combination of ranking signal and semantic quality
                ranking_weight = 0.7
                quality_weight = 0.3
                return (self._get_ranking_score(result) * ranking_weight + quality * quality_weight)
            
            scored_results.sort(key=combined_score, reverse=True)
            for result, quality in scored_results:
                result.ranking_score = combined_score((result, quality))
            
            # 3. Extract reranked results
            reranked_results = [result for result, _ in scored_results]
            
            # 4. Diversity filtering - remove near-duplicates
            if len(reranked_results) > 1:
                reranked_results = self._apply_diversity_filter(reranked_results)
            
            self._log_info(f"Re-ranked {len(results)} results using contextual scoring")
            return reranked_results
            
        except Exception as e:
            self._log_warning(f"Re-ranking failed, using original order: {e}")
            return results
    
    def _calculate_result_quality_score(self, result: SearchResult, query_lower: str) -> float:
        """Calculate quality score for a result based on multiple factors"""
        score = 0.0
        
        try:
            # Factor 1: Metadata completeness (0.0-0.3)
            metadata_score = 0.0
            if hasattr(result.metadata, 'folder_name') and result.metadata.folder_name:
                metadata_score += 0.1
            if hasattr(result.metadata, 'image_name') and result.metadata.image_name:
                metadata_score += 0.1
            if hasattr(result.metadata, 'timestamp') and result.metadata.timestamp:
                metadata_score += 0.1
            
            # Factor 2: Filename relevance (0.0-0.3)
            filename_score = 0.0
            if hasattr(result.metadata, 'image_name') and result.metadata.image_name:
                filename_lower = result.metadata.image_name.lower()
                # Check for query terms in filename
                query_words = query_lower.split()
                for word in query_words:
                    if len(word) > 2 and word in filename_lower:
                        filename_score += 0.1
                        break
                filename_score = min(filename_score, 0.3)
            
            # Factor 3: Folder relevance (0.0-0.2)
            folder_score = 0.0
            if hasattr(result.metadata, 'folder_name') and result.metadata.folder_name:
                folder_lower = result.metadata.folder_name.lower()
                query_words = query_lower.split()
                for word in query_words:
                    if len(word) > 2 and word in folder_lower:
                        folder_score += 0.1
                        break
                folder_score = min(folder_score, 0.2)
            
            # Factor 4: Similarity distribution bonus (0.0-0.2)
            # Higher similarity scores get bonus points
            similarity_bonus = 0.0
            if result.similarity_score > 0.8:
                similarity_bonus = 0.2
            elif result.similarity_score > 0.6:
                similarity_bonus = 0.1
            elif result.similarity_score > 0.4:
                similarity_bonus = 0.05
            
            score = metadata_score + filename_score + folder_score + similarity_bonus
            
        except Exception as e:
            self._log_warning(f"Quality score calculation failed: {e}")
            score = 0.0
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _apply_diversity_filter(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove near-duplicate results to improve diversity - minimal filtering for maximum recall"""
        if len(results) <= 1:
            return results
        
        try:
            filtered_results = [results[0]]  # Always keep the top result
            
            # Very minimal filtering - only remove exact duplicates
            seen_files = set()
            seen_files.add(f"{results[0].metadata.folder_name}|{results[0].metadata.image_name}")
            
            for candidate in results[1:]:
                # Create unique identifier for file
                file_id = f"{candidate.metadata.folder_name}|{candidate.metadata.image_name}"
                
                # Only filter exact duplicates (same folder + same filename)
                if file_id not in seen_files:
                    filtered_results.append(candidate)
                    seen_files.add(file_id)
            
            self._log_info(f"Diversity filter: {len(results)} -> {len(filtered_results)} results")
            return filtered_results
            
        except Exception as e:
            self._log_warning(f"Diversity filtering failed: {e}")
            return results
    
    def _are_filenames_very_similar(self, name1: str, name2: str) -> bool:
        """Check if two filenames are VERY similar (more strict than original)"""
        if not name1 or not name2:
            return False
        
        # Remove extensions for comparison
        base1 = name1.rsplit('.', 1)[0].lower()
        base2 = name2.rsplit('.', 1)[0].lower()
        
        # Check for exact match
        if base1 == base2:
            return True
        
        # Check for very similar patterns with same length
        if len(base1) == len(base2):
            import re
            # Remove numbers for pattern matching
            pattern1 = re.sub(r'\d+', 'N', base1)
            pattern2 = re.sub(r'\d+', 'N', base2)
            
            # Only mark as similar if patterns match AND differ by only 1-2 characters
            if pattern1 == pattern2:
                differences = sum(c1 != c2 for c1, c2 in zip(base1, base2))
                return differences <= 2
        
        return False
    
    def _are_filenames_similar(self, name1: str, name2: str) -> bool:
        """Check if two filenames are very similar (potential duplicates)"""
        if not name1 or not name2:
            return False
        
        # Remove extensions for comparison
        base1 = name1.rsplit('.', 1)[0].lower()
        base2 = name2.rsplit('.', 1)[0].lower()
        
        # Check for exact match
        if base1 == base2:
            return True
        
        # Check for similar patterns (e.g., img001.jpg vs img002.jpg)
        import re
        # Remove numbers for pattern matching
        pattern1 = re.sub(r'\d+', 'N', base1)
        pattern2 = re.sub(r'\d+', 'N', base2)
        
        return pattern1 == pattern2
    
    def _log_search_quality_metrics(self, results: List[SearchResult], query: str, search_time: float):
        """Log quality metrics for search performance monitoring"""
        try:
            if not results:
                self._log_info(f"🔍 Search Quality: Query='{query}' | Results=0 | Time={search_time:.3f}s")
                return
            
            # Calculate quality metrics
            avg_similarity = sum(r.similarity_score for r in results) / len(results)
            max_similarity = max(r.similarity_score for r in results)
            min_similarity = min(r.similarity_score for r in results)
            ranking_scores = [self._get_ranking_score(r) for r in results]
            avg_ranking = sum(ranking_scores) / len(ranking_scores)
            max_ranking = max(ranking_scores)
            min_ranking = min(ranking_scores)
            
            # Distribution analysis
            high_quality_count = sum(1 for r in results if r.similarity_score > 0.7)
            medium_quality_count = sum(1 for r in results if 0.4 <= r.similarity_score <= 0.7)
            low_quality_count = sum(1 for r in results if r.similarity_score < 0.4)
            
            # Folder diversity
            unique_folders = len(set(r.metadata.folder_name for r in results 
                                  if hasattr(r.metadata, 'folder_name') and r.metadata.folder_name))
            
            self._log_info(f"🔍 Search Quality Metrics:")
            self._log_info(f"   Query: '{query}' | Time: {search_time:.3f}s")
            self._log_info(f"   Results: {len(results)} | Folders: {unique_folders}")
            self._log_info(f"   Similarity: Avg={avg_similarity:.3f} | Max={max_similarity:.3f} | Min={min_similarity:.3f}")
            self._log_info(f"   Ranking: Avg={avg_ranking:.3f} | Max={max_ranking:.3f} | Min={min_ranking:.3f}")
            self._log_info(f"   Quality Distribution: High={high_quality_count} | Medium={medium_quality_count} | Low={low_quality_count}")
            
        except Exception as e:
            self._log_warning(f"Quality metrics logging failed: {e}")
    
    def _search_with_unified_index(self, query: str, options: SearchOptions) -> List[SearchResult]:
        """Enhanced search using unified index with advanced ranking algorithms"""
        try:
            # Generate multiple query variations for better recall
            query_variations = self._generate_query_variations(query)
            
            # Advanced search limit calculation based on dataset size
            dataset_size = getattr(self.unified_builder.unified_index, 'total_items', 0)
            if dataset_size > 1500000:  # 150GB+ datasets
                search_multiplier = 6  # Even higher multiplier for very large datasets
            elif dataset_size > 1000000:  # 100GB+ datasets  
                search_multiplier = 5
            elif dataset_size > 500000:  # 50GB+ datasets
                search_multiplier = 4
            else:
                search_multiplier = 3
            
            search_limit = min(options.limit * search_multiplier, 1000)  # Increased maximum
            
            all_results = {}  # Use dict to deduplicate by index
            query_embeddings = []  # Store embeddings for ensemble scoring
            
            # Multi-query processing with enhanced scoring
            for i, query_variant in enumerate(query_variations):
                # Encode query text using CLIP
                query_features = self._encode_query_text(query_variant, f"unified_clip_query[{i}]")
                query_embeddings.append(query_features)
                
                # Dynamic limit per query variation
                if i == 0:  # Primary query gets highest limit
                    variant_limit = search_limit
                elif i < 3:  # Top 3 variations get high limit
                    variant_limit = min(search_limit // 2, 500)
                else:  # Other variations get moderate limit
                    variant_limit = min(search_limit // 3, 300)
                
                # Use unified search with enhanced parameters
                unified_results = self.unified_builder.search_unified_fast(
                    query_features, 
                    k=variant_limit,
                    similarity_threshold=options.similarity_threshold
                )
                
                # Advanced result merging with multiple scoring factors
                for result in unified_results:
                    index_key = result['index']
                    base_score = float(result['similarity_score'])
                    incoming_ranking_score = base_score
                    
                    # Advanced score adjustment with multiple factors
                    if i == 0:
                        # Primary query gets highest weight
                        score_multiplier = 1.3  # 30% bonus
                        query_weight = 1.0
                    elif i == 1:
                        # Main translation gets high weight
                        score_multiplier = 1.15  # 15% bonus
                        query_weight = 0.9
                    elif i < 4:
                        # Important variations get moderate weight
                        score_multiplier = 1.0
                        query_weight = 0.8
                    else:
                        # Additional variations get lower weight
                        score_multiplier = 0.9
                        query_weight = 0.7
                    
                    # Calculate ensemble score if we have multiple matches
                    if index_key in all_results:
                        # Preserve semantic similarity as the best true cosine match.
                        all_results[index_key]['similarity_score'] = max(
                            float(all_results[index_key]['similarity_score']),
                            base_score
                        )

                        # Ensemble scoring: combine ranking signals only.
                        prev_score = all_results[index_key].get(
                            'ranking_score',
                            all_results[index_key]['similarity_score']
                        )
                        prev_weight = all_results[index_key].get('total_weight', 1.0)
                        
                        # Weighted average with bias toward higher scores
                        ensemble_score = (prev_score * prev_weight + incoming_ranking_score * score_multiplier * query_weight) / (prev_weight + query_weight)
                        
                        # Boost score for multiple query matches (consensus bonus)
                        consensus_bonus = min(0.1 * (i + 1), 0.3)  # Up to 30% bonus
                        final_score = ensemble_score * (1 + consensus_bonus)
                        
                        # Update if this gives better score
                        if final_score > all_results[index_key].get('ranking_score', all_results[index_key]['similarity_score']):
                            result['similarity_score'] = all_results[index_key]['similarity_score']
                            result['ranking_score'] = final_score
                            result['ensemble_score'] = ensemble_score
                            result['consensus_bonus'] = consensus_bonus
                            result['query_matches'] = all_results[index_key].get('query_matches', 1) + 1
                            result['total_weight'] = prev_weight + query_weight
                            all_results[index_key] = result
                    else:
                        # First match for this index
                        adjusted_score = incoming_ranking_score * score_multiplier
                        result['similarity_score'] = base_score
                        result['ranking_score'] = adjusted_score
                        result['query_variant'] = i
                        result['original_score'] = base_score
                        result['score_multiplier'] = score_multiplier
                        result['query_matches'] = 1
                        result['total_weight'] = query_weight
                        all_results[index_key] = result
            
            # Advanced result ranking and filtering
            all_results_list = list(all_results.values())
            
            # Apply advanced ranking algorithms
            ranked_results = self._apply_advanced_ranking(all_results_list, query, query_embeddings)
            
            # Apply diversity filtering with minimal impact
            if len(ranked_results) > options.limit * 2:
                filtered_results = self._apply_minimal_diversity_filter(ranked_results, options.limit * 1.5)
            else:
                filtered_results = ranked_results
            
            # Final selection with quality-based filtering
            top_results = self._select_top_quality_results(filtered_results, options.limit)
            
            # Convert unified results to SearchResult format
            search_results = []
            for i, result in enumerate(top_results):
                # Handle metadata conversion for remote indexes
                metadata = result['metadata']
                if isinstance(metadata, dict):
                    # Convert dictionary to KeyframeMetadata object
                    from core import KeyframeMetadata
                    
                    # Filter metadata to only include valid KeyframeMetadata fields
                    valid_fields = {
                        'folder_name', 'image_name', 'frame_id', 'file_path',
                        'sequence_position', 'total_frames', 'neighboring_frames', 'scene_boundaries',
                        'clip_features', 'llm_description', 'detected_objects', 'scene_tags', 
                        'confidence_score', 'similar_frames', 'transition_frames'
                    }
                    filtered_metadata = {k: v for k, v in metadata.items() if k in valid_fields}
                    metadata = KeyframeMetadata.from_dict(filtered_metadata)
                
                search_result = SearchResult(
                    metadata=metadata,
                    similarity_score=result['similarity_score'],
                    ranking_score=result.get('ranking_score', result.get('final_score', result['similarity_score'])),
                    rank=i  # Re-rank based on final ordering
                )
                # Store unified index frame index for thumbnail access
                search_result.unified_index = result.get('index', result['rank'])
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            if self._is_btc_unified_index_active():
                self._log_error(f"BTC unified search failed; refusing legacy fallback across CLIP model spaces: {e}")
                raise

            self._log_error(f"Unified search failed, falling back to legacy: {e}")
            # Fallback to legacy search
            query_features = self._encode_query_text(query, "unified_fallback_legacy")
            return self.faiss_retriever.search(
                query_features, 
                options.limit,
                validate_results=options.validate_results
            )
    
    def _search_llm_enhanced(self, query: str, options: SearchOptions, query_struct: Optional[QueryStructure] = None) -> List[SearchResult]:
        """LLM-enhanced search implementation (OpenAI only)"""
        # Check if unified index is available and use it instead
        if (hasattr(self, 'unified_builder') and self.unified_builder and
                self.unified_builder.unified_index and
                self.unified_builder.unified_index.is_loaded):
            return self._search_with_unified_index(query, options)
        
        # Check if remote index is available
        if hasattr(self, 'remote_indexes') and self.remote_indexes and len(self.remote_indexes) > 0:
            return self._search_with_remote_index(query, options)
        
        # Fallback to legacy LLM-enhanced search
        # Expand query using LLM
        expanded_queries = self.llm_processor.expand_query(query)
        
        # Search with each expanded query
        all_results = []
        queries_per_expansion = max(1, options.limit // len(expanded_queries))
        
        for expanded_query in expanded_queries:
            query_features = self._encode_query_text(expanded_query, "llm_enhanced_legacy")
            results = self.faiss_retriever.search(
                query_features, 
                queries_per_expansion,
                validate_results=options.validate_results
            )
            all_results.extend(results)
        
        # Merge and deduplicate
        unique_results = self.data_processor.merge_results([all_results])
        
        # Re-rank with LLM if enabled (OpenAI)
        if options.enable_reranking and len(unique_results) > 1:
            unique_results = self.llm_processor.rank_results(unique_results, query, options.limit)
        
        return unique_results[:options.limit]
    
    def _search_hybrid(self, query: str, options: SearchOptions, query_struct: Optional[QueryStructure] = None) -> List[SearchResult]:
        """Hybrid search combining CLIP and LLM (OpenAI)"""
        # Check if unified index is available and use it instead
        if (hasattr(self, 'unified_builder') and self.unified_builder and
                self.unified_builder.unified_index and
                self.unified_builder.unified_index.is_loaded):
            self._log_info(f"🔄 Using unified index for hybrid search: {query}")
            return self._search_with_unified_index(query, options)
        
        # Check if remote index is available
        if hasattr(self, 'remote_indexes') and self.remote_indexes and len(self.remote_indexes) > 0:
            return self._search_with_remote_index(query, options)
        
        # Fallback to legacy hybrid search
        # Start with CLIP search
        clip_results = self._search_clip_only(query, options, query_struct)
        
        # Enhance with LLM if we have reasonable results
        if len(clip_results) >= 5:
            enhanced_options = SearchOptions(
                mode="llm_enhanced",
                limit=options.limit,
                enable_reranking=options.enable_reranking,
                validate_results=options.validate_results
            )
            llm_results = self._search_llm_enhanced(query, enhanced_options, query_struct)
            
            # Combine results with weighted scoring
            combined_results = self._combine_search_results(clip_results, llm_results, weights=(0.7, 0.3))
            return combined_results[:options.limit]
        
        return clip_results
    
    def _combine_search_results(self, 
                               results1: List[SearchResult], 
                               results2: List[SearchResult],
                               weights: Tuple[float, float] = (0.5, 0.5)) -> List[SearchResult]:
        """Combine two sets of search results with weighted scoring"""
        combined_scores = {}
        
        # Score first set
        for result in results1:
            key = f"{result.metadata.folder_name}_{result.metadata.image_name}"
            combined_scores[key] = {
                'result': result,
                'score1': self._get_ranking_score(result) * weights[0],
                'score2': 0.0
            }
        
        # Score second set
        for result in results2:
            key = f"{result.metadata.folder_name}_{result.metadata.image_name}"
            if key in combined_scores:
                combined_scores[key]['score2'] = self._get_ranking_score(result) * weights[1]
                combined_scores[key]['result'].similarity_score = max(
                    combined_scores[key]['result'].similarity_score,
                    result.similarity_score
                )
            else:
                combined_scores[key] = {
                    'result': result,
                    'score1': 0.0,
                    'score2': self._get_ranking_score(result) * weights[1]
                }
        
        # Calculate combined scores and sort
        for key, data in combined_scores.items():
            total_score = data['score1'] + data['score2']
            data['result'].ranking_score = total_score
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.values(), 
                              key=lambda x: self._get_ranking_score(x['result']), 
                              reverse=True)
        
        return [item['result'] for item in sorted_results]
    
    def _post_process_results(self, 
                            results: List[SearchResult], 
                            query: str, 
                            options: SearchOptions) -> List[SearchResult]:
        """Post-process search results with enhanced re-ranking and validation"""
        if not results:
            return results
        
        # Enhanced re-ranking for better accuracy
        results = self._rerank_results_with_context(results, query, options)
        
        # Add temporal context if requested
        if options.include_temporal_context:
            # Check if unified index is available for temporal context
            has_unified_index = (hasattr(self, 'unified_builder') and 
                               self.unified_builder and 
                               self.unified_builder.unified_index and 
                               self.unified_builder.unified_index.is_loaded)
            
            for result in results:
                try:
                    # Check if this is a remote result (skip temporal context for remote results)
                    is_remote_result = (hasattr(self, 'remote_indexes') and 
                                      self.remote_indexes and 
                                      len(self.remote_indexes) > 0)
                    
                    if has_unified_index or is_remote_result:
                        # Skip temporal context for unified index or remote results (no warnings)
                        result.temporal_context = []
                    else:
                        # Use legacy temporal context for local FAISS results only
                        temporal_neighbors = self.metadata_manager.get_temporal_neighbors(
                            result.metadata.folder_name,
                            result.metadata.image_name,
                            window=self.config.get("retrieval.temporal_window", 3)
                        )
                        result.temporal_context = [
                            SearchResult(neighbor, 0.0, 0) for neighbor in temporal_neighbors
                        ]
                except Exception as e:
                    self._log_warning(f"Failed to get temporal context: {e}")
                    result.temporal_context = []
        
        # Add explanations if requested (OpenAI)
        if options.include_explanations:
            try:
                results = self.llm_processor.explain_results(results, query)
            except Exception as e:
                self._log_warning(f"Failed to generate explanations: {e}")
        
        # Final ranking update
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    # =================== DATA EXPORT METHODS ===================
    
    def export_data(self, 
                   results: List[SearchResult], 
                   format: str = "csv",
                   output_path: Optional[str] = None,
                   include_metadata: bool = True) -> str:
        """
        Export search results in various formats with enhanced error handling
        
        Args:
            results: Search results to export
            format: Export format (csv, json, xlsx)
            output_path: Output file path
            include_metadata: Include full metadata
            
        Returns:
            Path to exported file
        """
        try:
            with self.perf_monitor.timer("export_data"):
                if not output_path:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    export_dir = self.config.get("paths.exports", "exports/")
                    output_path = os.path.join(export_dir, f"search_results_{timestamp}.{format}")
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                if format.lower() == "csv":
                    self._export_csv(results, output_path, include_metadata)
                elif format.lower() == "json":
                    self._export_json(results, output_path, include_metadata)
                elif format.lower() == "xlsx":
                    self._export_xlsx(results, output_path, include_metadata)
                else:
                    raise ValueError(f"Unsupported export format: {format}")
                
                self._log_info(f"Data exported successfully", 
                             format=format,
                             path=output_path,
                             results_count=len(results))
                
                return output_path
                
        except Exception as e:
            self._log_error(f"Export failed: {e}", exc_info=True)
            raise
    
    def _export_csv(self, results: List[SearchResult], output_path: str, include_metadata: bool):
        """Export results to CSV format"""
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ["Rank", "Folder", "Image", "FrameID", "Similarity", "RankingScore", "File_Path"]
            if include_metadata:
                header.extend(["Scene_Tags", "Detected_Objects", "Explanation"])
            writer.writerow(header)
            
            # Write data
            for result in results:
                row = [
                    result.rank,
                    result.metadata.folder_name,
                    result.metadata.image_name,
                    result.metadata.frame_id,
                    f"{result.similarity_score:.4f}",
                    f"{self._get_ranking_score(result):.4f}",
                    result.metadata.file_path
                ]
                
                if include_metadata:
                    row.extend([
                        "; ".join(result.metadata.scene_tags) if result.metadata.scene_tags else "",
                        "; ".join(result.metadata.detected_objects) if result.metadata.detected_objects else "",
                        result.explanation or ""
                    ])
                
                writer.writerow(row)
    
    def _export_json(self, results: List[SearchResult], output_path: str, include_metadata: bool):
        """Export results to JSON format"""
        export_data = {
            "export_timestamp": time.time(),
            "total_results": len(results),
            "results": []
        }
        
        for result in results:
            result_data = {
                "rank": result.rank,
                "similarity_score": result.similarity_score,
                "ranking_score": self._get_ranking_score(result),
                "metadata": result.metadata.to_dict() if include_metadata else {
                    "folder_name": result.metadata.folder_name,
                    "image_name": result.metadata.image_name,
                    "frame_id": result.metadata.frame_id,
                    "file_path": result.metadata.file_path
                }
            }
            
            if result.explanation:
                result_data["explanation"] = result.explanation
                
            if result.temporal_context:
                result_data["temporal_context"] = [
                    ctx.metadata.to_dict() if include_metadata else {
                        "folder_name": ctx.metadata.folder_name,
                        "image_name": ctx.metadata.image_name,
                        "frame_id": ctx.metadata.frame_id
                    }
                    for ctx in result.temporal_context
                ]
            
            export_data["results"].append(result_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def _export_xlsx(self, results: List[SearchResult], output_path: str, include_metadata: bool):
        """Export results to Excel format"""
        try:
            import pandas as pd
            
            # Prepare data
            data = []
            for result in results:
                row = {
                    "Rank": result.rank,
                    "Folder": result.metadata.folder_name,
                    "Image": result.metadata.image_name,
                    "FrameID": result.metadata.frame_id,
                    "Similarity": result.similarity_score,
                    "RankingScore": self._get_ranking_score(result),
                    "File_Path": result.metadata.file_path
                }
                
                if include_metadata:
                    row.update({
                        "Scene_Tags": "; ".join(result.metadata.scene_tags) if result.metadata.scene_tags else "",
                        "Detected_Objects": "; ".join(result.metadata.detected_objects) if result.metadata.detected_objects else "",
                        "Explanation": result.explanation or ""
                    })
                
                data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(data)
            df.to_excel(output_path, index=False)
            
        except ImportError:
            raise ImportError("pandas and openpyxl required for Excel export")
    
    # =================== PORTABLE INDEX SYSTEM ===================
    
    def export_portable_index(self, 
                             output_path: str, 
                             include_keyframes: bool = False,
                             compress: bool = True) -> Dict[str, Any]:
        """
        🚀 Export current index as a portable package
        
        Creates a portable index package that can be moved between machines
        while maintaining full functionality.
        
        Args:
            output_path: Directory to create the portable package
            include_keyframes: Whether to include keyframes in package (increases size)
            compress: Whether to compress the package (experimental)
            
        Returns:
            Export status and information
        """
        if not self.status.is_ready:
            raise RuntimeError("System not ready. Build or load system first.")
        
        try:
            from core import PortableIndex
            
            # Initialize portable index manager
            portable = PortableIndex(logger=self.logger)
            
            # Create output directory
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            self._log_info(f"🚀 Creating portable index at: {output_path}")
            
            # 1. Export FAISS index
            faiss_dest = output_path / "index.faiss"
            
            # Check if we have a saved index path
            if hasattr(self, 'index_path') and self.index_path:
                faiss_source = Path(self.index_path) / "index.faiss"
                if faiss_source.exists():
                    shutil.copy2(faiss_source, faiss_dest)
                    self._log_info("✅ FAISS index copied from saved location")
                else:
                    # Save current index if saved location doesn't exist
                    self.faiss_retriever.save_index(str(faiss_dest))
                    self._log_info("✅ FAISS index exported directly")
            else:
                # No saved index path, export current index directly
                self.faiss_retriever.save_index(str(faiss_dest))
                self._log_info("✅ FAISS index exported directly")
            
            # 2. Convert metadata to portable format
            metadata_list = list(self.faiss_retriever.id_to_metadata.values())
            portable_metadata = portable.create_portable_metadata(metadata_list)
            
            # Save portable metadata
            metadata_dest = output_path / "metadata.json"
            with open(metadata_dest, 'w', encoding='utf-8') as f:
                json.dump(portable_metadata, f, indent=2, ensure_ascii=False)
            
            self._log_info(f"✅ Exported {len(portable_metadata)} metadata entries")
            
            # 3. Create manifest
            keyframes_structure = None
            if hasattr(self, 'keyframes_path') and self.keyframes_path:
                try:
                    keyframes_structure = self._analyze_keyframes_structure(self.keyframes_path)
                except Exception as e:
                    self._log_warning(f"Could not analyze keyframes structure: {e}")
            
            manifest = portable.create_manifest(
                metadata_count=len(portable_metadata),
                keyframes_structure=keyframes_structure
            )
            
            # Add export info
            manifest['export_info'] = {
                'exported_from': str(Path.cwd()),
                'original_keyframes_path': getattr(self, 'keyframes_path', None),
                'include_keyframes': include_keyframes,
                'compressed': compress
            }
            
            # Save manifest
            manifest_dest = output_path / "manifest.json"
            with open(manifest_dest, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            
            # 4. Optionally include keyframes
            if include_keyframes and hasattr(self, 'keyframes_path') and self.keyframes_path:
                keyframes_source = Path(self.keyframes_path)
                keyframes_dest = output_path / "keyframes"
                
                if keyframes_source.exists():
                    self._log_info("📁 Copying keyframes folder...")
                    shutil.copytree(keyframes_source, keyframes_dest, dirs_exist_ok=True)
                    self._log_info("✅ Keyframes included in package")
                else:
                    self._log_warning(f"Keyframes source not found: {keyframes_source}")
            
            # 5. Copy configuration
            config_dest = output_path / "index_config.json"
            config_data = {
                'system_version': '2.1',
                'clip_model': self.config.get('clip', {}).get('model_name', 'unknown'),
                'faiss_config': self.config.get('retrieval', {}),
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(config_dest, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            # 6. Create README
            readme_dest = output_path / "README.md"
            readme_content = f"""# Enhanced Retrieval System - Portable Index

## Package Information
- **Version**: {manifest['version']}
- **Created**: {manifest['created_at']}
- **Metadata entries**: {len(portable_metadata)}
- **Includes keyframes**: {include_keyframes}

## Usage
1. Copy this package to target machine
2. Ensure keyframes folder is available (if not included)
3. Load using: `system.load_portable_index('package_path', 'keyframes_path')`

## Requirements
- Python >= 3.8
- Enhanced Retrieval System v2.0+
- FAISS >= 1.7.0
- NumPy >= 1.19.0

## Files
- `index.faiss`: FAISS similarity index
- `metadata.json`: Portable metadata with relative paths
- `manifest.json`: Package information and validation
- `index_config.json`: System configuration
- `keyframes/`: Original keyframes (if included)
"""
            
            with open(readme_dest, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            # Calculate package size
            total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            
            result = {
                'success': True,
                'package_path': str(output_path),
                'size_mb': round(size_mb, 2),
                'metadata_count': len(portable_metadata),
                'includes_keyframes': include_keyframes,
                'files_created': len(list(output_path.rglob('*'))),
                'manifest': manifest
            }
            
            self._log_info(f"🎉 Portable index exported successfully!")
            self._log_info(f"📦 Package size: {size_mb:.2f} MB")
            self._log_info(f"📍 Location: {output_path}")
            
            return result
            
        except Exception as e:
            self._log_error(f"Export failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to export portable index: {e}")
    
    def load_portable_index(self, 
                           package_path: str, 
                           keyframes_path: str = None,
                           auto_detect_keyframes: bool = True) -> bool:
        """
        🚀 Load from portable index package
        
        Loads a portable index package created by export_portable_index.
        Automatically resolves paths and validates compatibility.
        
        Args:
            package_path: Path to portable package directory
            keyframes_path: Path to keyframes folder (auto-detected if None)
            auto_detect_keyframes: Try to auto-detect keyframes location
            
        Returns:
            True if loaded successfully
        """
        try:
            from core import PortableIndex
            
            package_path = Path(package_path)
            
            # Initialize portable index manager
            portable = PortableIndex(logger=self.logger)
            
            # Validate package
            self._log_info(f"🔍 Validating portable package: {package_path}")
            validation = portable.validate_portable_package(str(package_path))
            
            if not validation['is_valid']:
                raise ValueError(f"Invalid portable package: {validation['issues']}")
            
            self._log_info(f"✅ Package validation passed (v{validation['version']})")
            
            # Load manifest
            manifest_file = package_path / "manifest.json"
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            
            # Auto-detect or validate keyframes path
            if not keyframes_path and auto_detect_keyframes:
                keyframes_path = self._auto_detect_keyframes(package_path, manifest)
            
            if not keyframes_path:
                # Check if keyframes are included in package
                package_keyframes = package_path / "keyframes"
                if package_keyframes.exists():
                    keyframes_path = str(package_keyframes)
                    self._log_info("✅ Using keyframes included in package")
                else:
                    raise ValueError("Keyframes path not provided and could not be auto-detected")
            
            # Validate keyframes structure
            if not self._validate_keyframes_compatibility(keyframes_path, manifest):
                self._log_warning("⚠️ Keyframes structure may not match package")
            
            # Load portable metadata
            metadata_file = package_path / "metadata.json"
            with open(metadata_file, 'r', encoding='utf-8') as f:
                portable_metadata = json.load(f)
            
            # Resolve paths
            self._log_info(f"🔄 Resolving paths for {len(portable_metadata)} entries...")
            resolved_metadata = portable.resolve_portable_paths(portable_metadata, keyframes_path)
            
            # Load FAISS index
            # Initialize components if needed
            self._ensure_components_loaded()
            
            # Load only the FAISS index file directly (skip metadata validation)
            faiss_file = package_path / "index.faiss"
            import faiss
            self.faiss_retriever.index = faiss.read_index(str(faiss_file))
            
            # CRITICAL: Mark the index as trained and set dimension after loading
            self.faiss_retriever.is_trained = True
            self.faiss_retriever.dimension = self.faiss_retriever.index.d  # Set dimension from loaded index
            self._log_info(f"✅ FAISS index loaded and marked as trained (dimension: {self.faiss_retriever.dimension})")
            
            if self.config.get("retrieval.enable_gpu", False):
                try:
                    # Check if FAISS GPU support is available
                    if hasattr(faiss, 'StandardGpuResources'):
                        res = faiss.StandardGpuResources()
                        self.faiss_retriever.index = faiss.index_cpu_to_gpu(res, 0, self.faiss_retriever.index)
                        self._log_info("🚀 FAISS index moved to GPU")
                    else:
                        self._log_info("💡 FAISS GPU support not available, using CPU (this is normal for CPU-only FAISS)")
                except Exception as e:
                    self._log_info(f"💡 GPU not available, using CPU: {e}")
            else:
                self._log_info("💻 Using CPU for FAISS index (GPU disabled in config)")
            
            # Update metadata
            self.faiss_retriever.id_to_metadata.clear()
            for i, metadata in enumerate(resolved_metadata):
                self.faiss_retriever.id_to_metadata[i] = metadata
            
            # CRITICAL: Rebuild temporal index from loaded metadata
            if hasattr(self, 'temporal_analyzer') and self.temporal_analyzer:
                self._log_info("🔄 Rebuilding temporal index from portable metadata...")
                
                # Clear existing temporal index
                self.faiss_retriever.temporal_index = {}
                
                # Group metadata by folder and rebuild temporal index directly
                folder_groups = {}
                for metadata in resolved_metadata:
                    folder_name = Path(metadata.file_path).parent.name
                    if folder_name not in folder_groups:
                        folder_groups[folder_name] = []
                    folder_groups[folder_name].append(metadata)
                
                # Rebuild temporal index manually (avoid calling private methods)
                for folder_name, folder_metadata in folder_groups.items():
                    # Sort by frame ID
                    valid_frames = []
                    for meta in folder_metadata:
                        if isinstance(meta.frame_id, int) and meta.frame_id >= 0:
                            valid_frames.append((meta.frame_id, meta.image_name))
                        else:
                            self._log_warning(f"Invalid frame ID for {meta.folder_name}/{meta.image_name}: {meta.frame_id}")
                    
                    valid_frames.sort(key=lambda x: x[0])
                    self.faiss_retriever.temporal_index[folder_name] = valid_frames
                    self._log_info(f"   - {folder_name}: {len(valid_frames)} frames")
                
                self._log_info(f"✅ Temporal index rebuilt for {len(folder_groups)} folders: {list(folder_groups.keys())}")
                self._log_info(f"🔍 Final temporal_index keys: {list(self.faiss_retriever.temporal_index.keys())}")
            else:
                self._log_warning("⚠️ Temporal analyzer not available, temporal features disabled")
            
            # Store paths
            self.keyframes_path = keyframes_path
            self.index_path = str(package_path)
            
            # Update status
            self.status.is_ready = True
            self.status.index_loaded = True
            
            self._log_info(f"🎉 Portable index loaded successfully!")
            self._log_info(f"📊 Loaded {len(resolved_metadata)} entries")
            self._log_info(f"📁 Keyframes: {keyframes_path}")
            
            return True
            
        except Exception as e:
            self._log_error(f"Load portable index failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load portable index: {e}")
    
    def _analyze_keyframes_structure(self, keyframes_path: str) -> Dict[str, Any]:
        """Analyze keyframes folder structure for compatibility checking"""
        try:
            keyframes_path = Path(keyframes_path)
            structure = {
                'total_folders': 0,
                'total_files': 0,
                'folder_names': [],
                'file_extensions': set(),
                'sample_structure': {}
            }
            
            for folder in keyframes_path.iterdir():
                if folder.is_dir():
                    structure['total_folders'] += 1
                    structure['folder_names'].append(folder.name)
                    
                    # Sample first few files from this folder
                    files = [f.name for f in folder.iterdir() if f.is_file()]
                    structure['total_files'] += len(files)
                    
                    for file_name in files:
                        ext = Path(file_name).suffix.lower()
                        structure['file_extensions'].add(ext)
                    
                    # Store sample structure for first 3 folders
                    if len(structure['sample_structure']) < 3:
                        structure['sample_structure'][folder.name] = files[:5]  # First 5 files
            
            structure['file_extensions'] = list(structure['file_extensions'])
            return structure
            
        except Exception as e:
            self._log_warning(f"Could not analyze keyframes structure: {e}")
            return {}
    
    def _auto_detect_keyframes(self, package_path: Path, manifest: Dict) -> Optional[str]:
        """Auto-detect keyframes folder location"""
        self._log_info("🔍 Auto-detecting keyframes location...")
        
        # Get expected structure from manifest
        expected_structure = manifest.get('keyframes_structure', {})
        
        # Possible locations to check
        search_paths = [
            package_path / "keyframes",  # Included in package
            package_path.parent / "keyframes",  # Adjacent to package
            package_path / "../keyframes",  # Parent directory
            Path.cwd() / "keyframes",  # Current working directory
        ]
        
        # Also check export info if available
        export_info = manifest.get('export_info', {})
        if export_info.get('original_keyframes_path'):
            search_paths.insert(0, Path(export_info['original_keyframes_path']))
        
        for search_path in search_paths:
            try:
                search_path = search_path.resolve()
                if search_path.exists() and search_path.is_dir():
                    # Check if structure matches
                    if self._validate_keyframes_compatibility(str(search_path), manifest):
                        self._log_info(f"✅ Auto-detected keyframes: {search_path}")
                        return str(search_path)
            except Exception:
                continue
        
        self._log_warning("❌ Could not auto-detect keyframes location")
        return None
    
    def _validate_keyframes_compatibility(self, keyframes_path: str, manifest: Dict) -> bool:
        """Validate if keyframes folder is compatible with package"""
        try:
            current_structure = self._analyze_keyframes_structure(keyframes_path)
            expected_structure = manifest.get('keyframes_structure', {})
            
            if not expected_structure:
                return True  # No structure info available, assume compatible
            
            # Check basic compatibility
            if current_structure.get('total_folders', 0) != expected_structure.get('total_folders', 0):
                return False
            
            # Check if folder names match (at least 80%)
            expected_folders = set(expected_structure.get('folder_names', []))
            current_folders = set(current_structure.get('folder_names', []))
            
            if expected_folders:
                match_ratio = len(expected_folders & current_folders) / len(expected_folders)
                return match_ratio >= 0.8
            
            return True
            
        except Exception:
            return False
    
    # =================== APP LAUNCHER METHODS ===================
    
    def start_gui(self, **kwargs) -> None:
        """Launch GUI application with enhanced error handling"""
        try:
            from gui import MainWindow
            from PyQt5.QtWidgets import QApplication
            import sys
            
            app = QApplication(sys.argv if not QApplication.instance() else [])
            window = MainWindow(self, **kwargs)
            window.show()
            
            self._log_info("GUI application started")
            app.exec_()
            
        except ImportError as e:
            self._log_error(f"GUI dependencies not available: {e}")
            raise RuntimeError("GUI components not available. Install PyQt5 and gui module.")
        except Exception as e:
            self._log_error(f"GUI startup failed: {e}")
            raise
    
    def start_server(self, 
                    host: str = None, 
                    port: int = None,
                    **kwargs) -> None:
        """Start SocketIO server with enhanced error handling"""
        try:
            from api import SocketIOServer
            
            host = host or self.config.get("api.host", "localhost")
            port = port or self.config.get("api.port", 5000)
            
            server = SocketIOServer(self, **kwargs)
            
            self._log_info(f"Starting server", host=host, port=port)
            server.run(host, port)
            
        except ImportError as e:
            self._log_error(f"Server dependencies not available: {e}")
            raise RuntimeError("Server components not available. Install api module.")
        except Exception as e:
            self._log_error(f"Server startup failed: {e}")
            raise
    
    # =================== CALLBACK SYSTEM ===================
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for system events"""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
    
    def _trigger_callback(self, event: str, *args, **kwargs) -> None:
        """Trigger registered callbacks"""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    self._log_warning(f"Callback failed for event {event}: {e}")
    
    # =================== CONTEXT MANAGERS ===================
    
    @contextmanager
    def performance_context(self, operation: str):
        """Context manager for performance monitoring"""
        with self.perf_monitor.timer(operation) as timer_id:
            yield timer_id
    
    # =================== LOGGING HELPERS ===================
    
    def _log_info(self, message: str, **kwargs) -> None:
        if self.logger:
            self.logger.info(message, **kwargs)
        elif self.verbose:
            print(f"INFO: {message}")
    
    def _log_warning(self, message: str, **kwargs) -> None:
        if self.logger:
            self.logger.warning(message, **kwargs)
        elif self.verbose:
            print(f"WARNING: {message}")
    
    def _log_error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        if self.logger:
            self.logger.error(message, exc_info=exc_info, **kwargs)
        elif self.verbose:
            print(f"ERROR: {message}")
            if exc_info:
                traceback.print_exc()
    
    def _apply_advanced_ranking(self, results: List[Dict], query: str, query_embeddings: List) -> List[Dict]:
        """Apply advanced ranking algorithms to improve result quality"""
        if not results:
            return results
        
        # Calculate additional ranking features
        for result in results:
            ranking_score = float(result.get('ranking_score', result.get('similarity_score', 0.0)))
            similarity_score = float(result.get('similarity_score', 0.0))

            # Feature 1: Query-result semantic consistency
            if len(query_embeddings) > 1:
                # Calculate consistency across multiple query variations
                consistencies = []
                for emb in query_embeddings:
                    # Simulate consistency score for ranking only.
                    consistency = ranking_score * (0.9 + 0.2 * random.random())  # Simplified
                    consistencies.append(consistency)
                result['semantic_consistency'] = np.mean(consistencies)
                result['consistency_variance'] = np.var(consistencies)
            else:
                result['semantic_consistency'] = ranking_score
                result['consistency_variance'] = 0.0
            
            # Feature 2: Multi-query consensus score
            consensus_score = result.get('query_matches', 1) / len(query_embeddings)
            result['consensus_score'] = consensus_score
            
            # Feature 3: Score stability (lower variance = more stable)
            stability_bonus = 1.0 - min(result['consistency_variance'] * 10, 0.3)  # Up to 30% penalty for high variance
            result['stability_score'] = stability_bonus
            
            # Feature 4: Quality confidence (based on original score distribution)
            confidence = min(similarity_score / 0.8, 1.0)  # Normalize by expected high cosine
            result['quality_confidence'] = confidence
            
            # Composite ranking score
            weights = {
                'similarity': 0.4,      # Base similarity score
                'consensus': 0.25,      # Multi-query agreement
                'semantic': 0.2,        # Semantic consistency
                'stability': 0.1,       # Score stability
                'confidence': 0.05      # Quality confidence
            }
            
            composite_score = (
                weights['similarity'] * ranking_score +
                weights['consensus'] * result['consensus_score'] +
                weights['semantic'] * result['semantic_consistency'] +
                weights['stability'] * result['stability_score'] +
                weights['confidence'] * result['quality_confidence']
            )
            
            result['composite_score'] = composite_score
            result['ranking_features'] = {
                'similarity': similarity_score,
                'ranking': ranking_score,
                'consensus': result['consensus_score'],
                'semantic': result['semantic_consistency'],
                'stability': result['stability_score'],
                'confidence': result['quality_confidence']
            }
        
        # Sort by composite score
        ranked_results = sorted(results, key=lambda x: x['composite_score'], reverse=True)
        return ranked_results
    
    def _apply_minimal_diversity_filter(self, results: List[Dict], target_count: int) -> List[Dict]:
        """Apply minimal diversity filtering to reduce redundancy while preserving quality"""
        if len(results) <= target_count:
            return results
        
        filtered_results = []
        seen_similarity_groups = set()
        similarity_threshold = 0.95  # Very high threshold for minimal filtering
        
        for result in results:
            # Create similarity signature based on high-level features
            sim_signature = (
                round(result.get('ranking_score', result['similarity_score']) * 100),  # Rounded ranking signal
                result.get('query_variant', 0)  # Query variation used
            )
            
            # Check if this is too similar to already selected results
            is_too_similar = False
            for seen_sig in seen_similarity_groups:
                sig_diff = abs(sim_signature[0] - seen_sig[0])
                if sig_diff < 5 and sim_signature[1] == seen_sig[1]:  # Very strict similarity
                    is_too_similar = True
                    break
            
            if not is_too_similar or len(filtered_results) < target_count * 0.7:  # Always keep top 70%
                filtered_results.append(result)
                seen_similarity_groups.add(sim_signature)
            
            # Stop when we have enough results
            if len(filtered_results) >= target_count:
                break
        
        return filtered_results
    
    def _select_top_quality_results(self, results: List[Dict], limit: int) -> List[Dict]:
        """Select top quality results with final filtering and ranking"""
        if len(results) <= limit:
            return results
        
        # Apply quality threshold filter
        min_quality_score = 0.1  # Very low threshold for 150GB datasets
        quality_filtered = [r for r in results if r.get('composite_score', 0) >= min_quality_score]
        
        # If we filtered too aggressively, take top results regardless of threshold
        if len(quality_filtered) < limit * 0.8:
            quality_filtered = results
        
        # Final ranking with boost for high-consensus results
        for result in quality_filtered:
            consensus_boost = min(result.get('query_matches', 1) * 0.05, 0.2)  # Up to 20% boost
            result['final_score'] = result['composite_score'] * (1 + consensus_boost)
            result['ranking_score'] = result['final_score']
        
        # Sort by final score and return top results
        final_ranked = sorted(quality_filtered, key=lambda x: x['final_score'], reverse=True)
        return final_ranked[:limit]

    def get_surrounding_frames(self, folder_name: str, image_name: str, window: int = 10) -> List[KeyframeMetadata]:
        """
        Lấy 10 frame trước và sau (cùng tên video) của một frame được chọn.
        
        Args:
            folder_name: Tên thư mục video
            image_name: Tên frame được chọn
            window: Số lượng frame trước/sau (mặc định 10)
            
        Returns:
            List[KeyframeMetadata]: Danh sách frame lân cận (không bao gồm frame gốc)
        """
        if not self.status.is_ready:
            raise RuntimeError("System not ready. Build or load system first.")
        
        surrounding_frames = []
        if self.metadata_manager:
            try:
                surrounding_frames = self.metadata_manager.get_temporal_neighbors(
                    folder_name=folder_name,
                    image_name=image_name,
                    window=window
                )
            except Exception as e:
                self.logger.error(f"Error getting surrounding frames from metadata_manager: {e}")
        
        # Fallback to direct metadata_list lookup if metadata_manager is empty
        if not surrounding_frames and hasattr(self, 'metadata_list') and self.metadata_list:
            folder_frames = [m for m in self.metadata_list if m.folder_name == folder_name]
            if folder_frames:
                folder_frames.sort(key=lambda x: x.frame_id)
                curr_pos = None
                for idx, m in enumerate(folder_frames):
                    if (m.image_name == image_name or 
                        Path(str(m.image_name)).stem == Path(str(image_name)).stem or
                        str(m.frame_id) == str(image_name)):
                        curr_pos = idx
                        break
                if curr_pos is not None:
                    start = max(0, curr_pos - window)
                    end = min(len(folder_frames), curr_pos + window + 1)
                    surrounding_frames = [folder_frames[i] for i in range(start, end) if i != curr_pos]
            
        return surrounding_frames
    
    def get_asymmetric_surrounding_frames(self, folder_name: str, image_name: str, before_window: int = 10, after_window: int = 10) -> List[KeyframeMetadata]:
        """
        Lấy frame trước và sau với số lượng khác nhau cho mỗi chiều.
        
        Args:
            folder_name: Tên thư mục video
            image_name: Tên frame được chọn
            before_window: Số lượng frame trước
            after_window: Số lượng frame sau
            
        Returns:
            List[KeyframeMetadata]: Danh sách frame lân cận (không bao gồm frame gốc)
        """
        if not self.status.is_ready:
            raise RuntimeError("System not ready. Build or load system first.")

        surrounding_frames = []
        if self.metadata_manager:
            try:
                surrounding_frames = self.metadata_manager.get_asymmetric_temporal_neighbors(
                    folder_name=folder_name,
                    image_name=image_name,
                    before_window=before_window,
                    after_window=after_window
                )
            except Exception as e:
                self.logger.error(f"Error getting asymmetric surrounding frames from metadata_manager: {e}")

        # Fallback to direct metadata_list lookup if metadata_manager is empty
        if not surrounding_frames and hasattr(self, 'metadata_list') and self.metadata_list:
            folder_frames = [m for m in self.metadata_list if m.folder_name == folder_name]
            if folder_frames:
                folder_frames.sort(key=lambda x: x.frame_id)
                curr_pos = None
                for idx, m in enumerate(folder_frames):
                    if (m.image_name == image_name or 
                        Path(str(m.image_name)).stem == Path(str(image_name)).stem or
                        str(m.frame_id) == str(image_name)):
                        curr_pos = idx
                        break
                if curr_pos is not None:
                    start = max(0, curr_pos - before_window)
                    end = min(len(folder_frames), curr_pos + after_window + 1)
                    surrounding_frames = [folder_frames[i] for i in range(start, end) if i != curr_pos]
            
        return surrounding_frames

    def export_selected_frames_to_csv(self, frames: List[KeyframeMetadata], output_path: str = None) -> str:
        """
        Xuất danh sách frame đã chọn ra file CSV.
        
        Args:
            frames: Danh sách KeyframeMetadata để xuất
            output_path: Đường dẫn file CSV (optional)
            
        Returns:
            str: Đường dẫn file CSV đã tạo
        """
        if not frames:
            raise ValueError("No frames provided for export")
        
        # Tạo tên file nếu không có
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"selected_frames_{timestamp}.csv"
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Header
                writer.writerow([
                    'Frame ID', 'Video Name', 'Image Name', 'File Path',
                    'Sequence Position', 'Total Frames', 'Confidence Score'
                ])
                
                # Data
                for frame in frames:
                    writer.writerow([
                        frame.frame_id,
                        frame.folder_name,
                        frame.image_name,
                        frame.file_path,
                        frame.sequence_position,
                        frame.total_frames,
                        frame.confidence_score
                    ])
            
            self.logger.info(f"Exported {len(frames)} frames to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error exporting frames to CSV: {e}")
            raise

    def get_frame_context_with_export(self, folder_name: str, image_name: str, window: int = 10) -> Dict[str, Any]:
        """
        Lấy frame context và chuẩn bị dữ liệu cho export.
        
        Args:
            folder_name: Tên thư mục video
            image_name: Tên frame được chọn  
            window: Số lượng frame trước/sau
            
        Returns:
            Dict chứa frame gốc, surrounding frames và thống kê
        """
        try:
            # Lấy frame gốc từ metadata manager
            original_frame = None
            if self.metadata_manager:
                original_frame = self.metadata_manager.get_metadata(folder_name, image_name)
            
            # Lấy surrounding frames
            surrounding_frames = self.get_surrounding_frames(folder_name, image_name, window)
            
            return {
                'original_frame': original_frame,
                'surrounding_frames': surrounding_frames,
                'total_surrounding': len(surrounding_frames),
                'window_size': window,
                'video_name': folder_name
            }
            
        except Exception as e:
            self.logger.error(f"Error getting frame context: {e}")
            return {
                'original_frame': None,
                'surrounding_frames': [],
                'total_surrounding': 0,
                'window_size': window,
                'video_name': folder_name,
                'error': str(e)
            }

    def _log_debug(self, message: str, **kwargs) -> None:
        if self.logger:
            self.logger.debug(message, **kwargs)


# ============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS (maintaining compatibility)
# ============================================================================

def create_system(config_path: str = None, **kwargs) -> EnhancedRetrievalSystem:
    """Create Enhanced Retrieval System instance"""
    return EnhancedRetrievalSystem(config_path=config_path, **kwargs)

def quick_search(keyframe_folder: str, query: str, **kwargs) -> List[SearchResult]:
    """Quick search without saving index"""
    system = EnhancedRetrievalSystem(auto_initialize=True)
    system.build_system(keyframe_folder, save_index=False)
    return system.search(query, **kwargs)


if __name__ == "__main__":
    # Example usage and testing
    print("🎯 Enhanced Retrieval System - Main System v2.1 (Cleaned & Optimized)")
    print("=" * 60)
    
    try:
        # Create system instance with validation
        system = EnhancedRetrievalSystem(verbose=True, enable_validation=True)
        
        # Test system status
        status = system.status
        print(f"✅ System initialized: {status.is_initialized}")
        print(f"📊 Components loaded: {sum(status.components_loaded.values())}/{len(status.components_loaded)}")
        
        # Test health monitoring
        health = system.get_system_health()
        print(f"🏥 System health: {health['overall_health']}")
        if health['issues']:
            print(f"⚠️ Health issues: {health['issues'][:3]}")  # Show first 3 issues
        
        # Test system stats
        stats = system.get_system_stats()
        print(f"📈 Performance stats available: {'performance' in stats}")
        
        # Test search options validation
        try:
            options = SearchOptions(mode="hybrid", limit=10)
            options.validate()
            print(f"✅ Search options validated: {options.mode}")
        except Exception as e:
            print(f"❌ Search options validation failed: {e}")
        
        # Test invalid search options
        try:
            bad_options = SearchOptions(mode="invalid_mode", limit=-5)
            bad_options.validate()
        except ValueError as e:
            print(f"✅ Invalid options correctly rejected: {str(e)[:50]}...")
        
        print("\n✅ Enhanced system wrapper tested successfully!")
        print("\n🚀 Optimized Features:")
        print("  • OpenAI GPT-4 integration only")
        print("  • Comprehensive health monitoring")
        print("  • Robust validation at all levels")
        print("  • Enhanced error handling and recovery")
        print("  • Detailed system diagnostics")
        print("  • Backward compatibility maintained")
        print("  • Cleaned and organized codebase")
        
        print("\n📋 To use the optimized system:")
        print("1. system.get_system_health() - Check system health")
        print("2. system.build_system('keyframes/', validate_inputs=True)")
        print("3. system.search('query', SearchOptions(validate_results=True))")
        print("4. system.start_gui() - Launch with health monitoring")
        print("5. system.chat('message') - Direct OpenAI chat interface")
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        traceback.print_exc()

# Add method to EnhancedRetrievalSystem class
def _mark_unified_ready(self):
    """Mark system as ready when unified index is loaded"""
    self.status.is_ready = True
    self.status.index_loaded = True
    self._unified_system_ready = True  # Add flag for search validation
    self._update_system_stats()
    if self.logger:
        self.logger.info("System marked as ready for unified search")

# Monkey patch the method to the class
EnhancedRetrievalSystem._mark_unified_ready = _mark_unified_ready
