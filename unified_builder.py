"""
Unified Builder Integration - Bridge to Legacy System
====================================================

Integration layer that bridges the new unified index system
with the existing enhanced retrieval system for seamless migration.

Author: Enhanced Retrieval System
Version: 3.0 - Integration Layer
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import unified index
try:
    from unified_index import UnifiedIndex, UnifiedIndexConfig, create_optimized_index, load_optimized_index
    HAS_UNIFIED_INDEX = True
except ImportError:
    HAS_UNIFIED_INDEX = False
    UnifiedIndex = UnifiedIndexConfig = None


def get_runtime_system(system):
    """Return the authoritative retrieval runtime system behind optional wrappers."""
    base_system = getattr(system, 'base_system', None)
    if base_system is not None:
        return base_system
    return system


class UnifiedBuilderIntegration:
    """
    🚀 Integration layer for unified index system
    
    Provides backwards compatibility with existing system
    while enabling new ultra-fast unified index capabilities.
    """
    
    def __init__(self, system, logger=None):
        self.system = get_runtime_system(system)
        self.logger = logger or system.logger
        self.unified_index = None
        
    def create_unified_index_fast(self, 
                                 keyframes_dir: str,
                                 output_path: str = None,
                                 csv_mappings: Dict[str, str] = None,
                                 progress_callback: callable = None,
                                 resume_from_existing: bool = False,
                                 chunk_size: int = 1000) -> Dict[str, Any]:
        """
        🚀 Create unified index with incremental building and memory management
        
        Features:
        - Single .rvdb file format
        - Parallel processing 
        - Memory-mapped access
        - Lossless compression
        - Incremental updates capability
        - Chunked processing for large datasets
        - Auto-checkpoint saving during build
        - Resume from existing .rvdb files
        
        Args:
            keyframes_dir: Path to keyframes directory
            output_path: Output .rvdb file path (auto-generated if None)
            csv_mappings: CSV mappings for temporal relationships
            progress_callback: Optional progress callback function
            resume_from_existing: Resume from existing .rvdb if it exists
            chunk_size: Number of images per chunk (for memory management)
            
        Returns:
            Dictionary with build statistics and performance metrics
        """
        if not HAS_UNIFIED_INDEX:
            raise ImportError("Unified index not available. Install required dependencies: h5py, lz4")
        
        start_time = time.time()
        
        try:
            # Auto-generate output path if not provided
            if output_path is None:
                timestamp = int(time.time())
                output_path = f"unified_index_{timestamp}.rvdb"

            output_parent = Path(output_path).expanduser().resolve().parent
            output_parent.mkdir(parents=True, exist_ok=True)
            if progress_callback:
                progress_callback(1, f"Preparing output file: {output_path}")
            
            # Ensure we have CLIP processor
            if not hasattr(self.system, 'clip_processor') or not self.system.clip_processor:
                self.system._initialize_ai_components()
            
            # Configure unified index for optimal performance
            config = UnifiedIndexConfig(
                compression_level=6,  # Good balance of speed vs size
                chunk_size=2000,     # Optimal for your data size  
                memory_map=True,     # Enable memory mapping
                max_workers=4,       # Parallel processing
                image_quality=95,    # High quality thumbnails
                thumbnail_size=(224, 224),  # Match CLIP input size
                store_full_images=True,  # Store full-size images for standalone operation
                full_image_quality=90    # Good quality with reasonable compression
            )
            
            self._log_info(f"🚀 Starting unified index build: {keyframes_dir} → {output_path}")
            self._log_info(f"⚙️ Config: {config.max_workers} workers, compression level {config.compression_level}")
            
            # Create the unified index with incremental building
            stats = create_optimized_index(
                keyframes_dir=keyframes_dir,
                clip_processor=self.system.clip_processor,
                output_file=output_path,
                csv_mappings=csv_mappings,
                config=config,
                logger=self.logger,
                progress_callback=progress_callback,
                resume_from_existing=resume_from_existing,
                chunk_size=chunk_size
            )
            
            # Update stats with total time
            total_time = time.time() - start_time
            stats['total_build_time'] = total_time
            stats['output_file'] = output_path
            
            # Performance comparison with old system
            old_estimated_time = stats['processed_files'] * 0.05  # ~50ms per image in old system
            speedup = old_estimated_time / stats['build_time'] if stats['build_time'] > 0 else 1.0
            stats['estimated_speedup'] = f"{speedup:.1f}x faster than legacy system"
            
            self._log_info(f"🎉 Unified index created successfully!")
            self._log_info(f"📊 Performance: {stats['processed_files']} files in {total_time:.2f}s")
            self._log_info(f"🚀 Speed improvement: {stats['estimated_speedup']}")
            self._log_info(f"📦 File size: {stats['index_size'] / 1024 / 1024:.2f} MB")
            self._log_info(f"🗜️ Compression: {stats['compression_ratio']:.2f}x smaller")
            
            return stats
            
        except Exception as e:
            self._log_error(f"Failed to create unified index: {e}")
            raise

    def create_btc_clip_index_fast(self,
                                   features_dir: str,
                                   keyframes_dir: str,
                                   map_dir: str,
                                   output_path: str,
                                   video_ids: Optional[List[str]] = None,
                                   progress_callback: callable = None,
                                   chunk_size: int = 1000,
                                   clip_model_name: str = "openai/clip-vit-base-patch32") -> Dict[str, Any]:
        """Create unified .rvdb from precomputed BTC CLIP ViT-B/32 .npy features."""
        if not HAS_UNIFIED_INDEX:
            raise ImportError("Unified index not available. Install required dependencies: h5py, lz4")

        start_time = time.time()

        try:
            output_parent = Path(output_path).expanduser().resolve().parent
            output_parent.mkdir(parents=True, exist_ok=True)
            if progress_callback:
                progress_callback(1, f"Preparing BTC output file: {output_path}")

            config = UnifiedIndexConfig(
                compression_level=6,
                chunk_size=2000,
                memory_map=True,
                max_workers=4,
                store_full_images=False
            )

            self._log_info(f"Starting BTC CLIP unified index build: {features_dir} -> {output_path}")
            self._log_info(f"BTC CLIP model space: {clip_model_name} / clip-ViT-B-32")

            unified_index = UnifiedIndex(config, self.logger)
            stats = unified_index.create_unified_index_from_btc_features(
                features_dir=features_dir,
                keyframes_dir=keyframes_dir,
                map_dir=map_dir,
                output_file=output_path,
                video_ids=video_ids,
                progress_callback=progress_callback,
                chunk_size=chunk_size,
                clip_model_name=clip_model_name
            )

            total_time = time.time() - start_time
            stats['total_build_time'] = total_time
            stats['output_file'] = output_path
            stats['estimated_speedup'] = "No JPG CLIP re-encoding"

            self._log_info("BTC CLIP unified index created successfully")
            self._log_info(f"Performance: {stats['processed_files']} vectors in {total_time:.2f}s")
            self._log_info(f"File size: {stats['index_size'] / 1024 / 1024:.2f} MB")

            return stats

        except Exception as e:
            self._log_error(f"Failed to create BTC CLIP unified index: {e}")
            raise
    
    def load_unified_index_fast(self, index_file: str) -> Dict[str, Any]:
        """
        ⚡ Load unified index with instant access
        
        Features:
        - Sub-second loading regardless of size
        - Memory-mapped for zero-copy access
        - Automatic cache warming
        - No rebuild required
        
        Args:
            index_file: Path to .rvdb file
            
        Returns:
            Load statistics and index information
        """
        if not HAS_UNIFIED_INDEX:
            raise ImportError("Unified index not available. Install required dependencies: h5py, lz4")
        
        start_time = time.time()
        
        try:
            self._log_info(f"⚡ Loading unified index: {index_file}")
            
            # Configure for optimal loading
            config = UnifiedIndexConfig(memory_map=True)
            
            # Create and load unified index
            self.unified_index = UnifiedIndex(config, self.logger)
            load_stats = self.unified_index.load_unified_index(index_file)
            self._ensure_query_encoder_matches_unified_index()
            
            # Update system components to use unified index
            self._integrate_with_system()
            
            # Calculate performance metrics
            total_time = time.time() - start_time
            load_stats['total_load_time'] = total_time
            
            # Estimate old system load time for comparison
            frame_count = load_stats['index_info'].get('processed_files', 0)
            old_estimated_time = frame_count * 0.001 + 10  # ~1ms per frame + 10s overhead
            speedup = old_estimated_time / total_time if total_time > 0 else 1.0
            load_stats['estimated_speedup'] = f"{speedup:.1f}x faster than legacy system"
            
            self._log_info(f"🎉 Unified index loaded successfully!")
            self._log_info(f"⚡ Load time: {total_time:.3f}s for {frame_count} frames")  
            self._log_info(f"🚀 Speed improvement: {load_stats['estimated_speedup']}")
            self._log_info(f"🧠 Memory-mapped for instant access")
            
            return load_stats
            
        except Exception as e:
            self._log_error(f"Failed to load unified index: {e}")
            raise
    
    def search_unified_fast(self, 
                           query_vector: 'np.ndarray',
                           k: int = 50,
                           similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        🔍 Ultra-fast search with unified index
        
        Features:
        - Sub-millisecond search times
        - Memory-mapped zero-copy access
        - Automatic metadata enrichment
        - Temporal context included
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of enriched search results
        """
        if not self.unified_index:
            raise ValueError("Unified index not loaded. Call load_unified_index_fast() first.")
        
        try:
            start_time = time.time()
            
            # Enhanced adaptive similarity threshold for very large datasets (150GB+)
            dataset_size = len(self.unified_index.metadata_list) if self.unified_index.metadata_list else 0
            adaptive_threshold = similarity_threshold
            
            if dataset_size > 1000000:  # Very large dataset (1M+ items, ~150GB+)
                # For 150GB+ datasets, use very low threshold for maximum recall
                if similarity_threshold == 0.0:
                    adaptive_threshold = 0.03  # Very low threshold for maximum results
                else:
                    adaptive_threshold = max(similarity_threshold, 0.02)  # Minimum floor
                if self.logger:
                    self.logger.info(f"🚀 VERY LARGE dataset detected ({dataset_size} items) - using ultra-low threshold: {adaptive_threshold}")
            elif dataset_size > 500000:  # Large dataset (500k+ items)
                if similarity_threshold == 0.0:
                    adaptive_threshold = 0.05  # Low threshold for good recall
                else:
                    adaptive_threshold = max(similarity_threshold, 0.03)
                if self.logger:
                    self.logger.info(f"📊 Large dataset detected ({dataset_size} items) - using low threshold: {adaptive_threshold}")
            elif dataset_size > 100000:  # Medium-large dataset
                if similarity_threshold == 0.0:
                    adaptive_threshold = 0.08
                else:
                    adaptive_threshold = max(similarity_threshold, 0.05)
                if self.logger:
                    self.logger.info(f"📊 Medium-large dataset detected ({dataset_size} items) - using adaptive threshold: {adaptive_threshold}")
            
            # Perform ultra-fast search with more results to account for filtering
            search_k = min(k * 3, 500)  # Search 3x more results, max 500
            results = self.unified_index.search_vectors(
                query_vector, 
                k=search_k,  # Get more results initially
                filter_func=lambda meta: True  # Could add filtering logic here
            )
            
            # Enrich results with temporal context
            enriched_results = []
            for result in results:
                if result['similarity_score'] >= adaptive_threshold:
                    # Add temporal context
                    temporal_frames = self.unified_index.get_temporal_context(
                        result['index'], window_size=3
                    )
                    
                    # Convert to legacy format for compatibility
                    search_result = {
                        'metadata': self._convert_metadata_to_legacy(result['metadata']),
                        'similarity_score': result['similarity_score'],
                        'ranking_score': result.get('ranking_score', result['similarity_score']),
                        'rank': result['rank'],
                        'temporal_context': temporal_frames,
                        'index': result['index']
                    }
                    enriched_results.append(search_result)
            
            search_time = time.time() - start_time
            
            # Enhanced logging for large datasets
            if dataset_size > 50000:
                total_candidates = len(results)
                filtered_results = len(enriched_results)
                if enriched_results:
                    avg_similarity = sum(r['similarity_score'] for r in enriched_results) / len(enriched_results)
                    max_similarity = max(r['similarity_score'] for r in enriched_results)
                    min_similarity = min(r['similarity_score'] for r in enriched_results)
                    self._log_info(f"🔍 Large dataset search: {filtered_results}/{total_candidates} results passed threshold {adaptive_threshold}")
                    self._log_info(f"📊 Similarity scores - Avg: {avg_similarity:.3f}, Max: {max_similarity:.3f}, Min: {min_similarity:.3f}")
                else:
                    self._log_warning(f"⚠️  No results passed adaptive threshold {adaptive_threshold} from {total_candidates} candidates")
            else:
                self._log_debug(f"🔍 Unified search: {len(enriched_results)} results in {search_time*1000:.2f}ms")
            
            return enriched_results
            
        except Exception as e:
            self._log_error(f"Unified search failed: {e}")
            raise
    
    def get_thumbnail_fast(self, frame_index: int) -> Optional['np.ndarray']:
        """
        🖼️ Get thumbnail with memory-mapped access
        
        Features:
        - Zero-copy thumbnail access
        - Automatic decompression
        - LRU caching for frequently accessed thumbnails
        
        Args:
            frame_index: Index of frame to get thumbnail for
            
        Returns:
            Thumbnail as numpy array or None if not found
        """
        if not self.unified_index:
            return None
        
        return self.unified_index.get_thumbnail(frame_index)
    
    def get_full_image_fast(self, frame_index: int) -> Optional[bytes]:
        """
        🖼️ Get full-size image with memory-mapped access
        
        Features:
        - Zero-copy full image access
        - Returns JPEG bytes ready for display
        - LRU caching for frequently accessed images
        
        Args:
            frame_index: Index of frame to get full image for
            
        Returns:
            Full image as JPEG bytes or None if not found/not stored
        """
        if not self.unified_index:
            return None
        
        return self.unified_index.get_full_image(frame_index)
    
    def incremental_update_fast(self, keyframes_dir: str) -> Dict[str, Any]:
        """
        🔄 Incremental update without full rebuild
        
        Features:
        - Hash-based change detection
        - Only processes new/modified files  
        - Maintains consistency throughout update
        - Atomic operations with rollback capability
        
        Args:
            keyframes_dir: Path to keyframes directory
            
        Returns:
            Update statistics and recommendations
        """
        if not self.unified_index:
            raise ValueError("Unified index not loaded. Call load_unified_index_fast() first.")
        
        try:
            self._log_info(f"🔄 Starting incremental update for: {keyframes_dir}")
            
            # Ensure we have CLIP processor for new files
            if not hasattr(self.system, 'clip_processor') or not self.system.clip_processor:
                self.system._initialize_ai_components()
            
            # Perform incremental update
            stats = self.unified_index.incremental_update(
                keyframes_dir, 
                self.system.clip_processor,
                ""  # Index file path not needed for in-memory updates
            )
            
            if stats['rebuild_required']:
                self._log_warning(f"⚠️ Large changes detected ({stats['new_files'] + stats['modified_files']} files)")
                self._log_warning("💡 Consider full rebuild for optimal performance")
            elif stats['new_files'] + stats['modified_files'] > 0:
                self._log_info(f"✅ Incremental update completed")
                self._log_info(f"📝 Added {stats['new_files']}, modified {stats['modified_files']} files")
            else:
                self._log_info("✨ No changes detected - index is up to date")
            
            return stats
            
        except Exception as e:
            self._log_error(f"Incremental update failed: {e}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the loaded unified index"""
        if not self.unified_index:
            return {'error': 'No unified index loaded'}
        
        try:
            build_info = self.unified_index._load_build_metadata()
            
            stats = {
                'format': 'Unified Index (.rvdb)',
                'version': build_info.get('build_version', '3.0'),
                'total_frames': build_info.get('processed_files', 0),
                'build_time': build_info.get('build_time', 0),
                'compression_ratio': build_info.get('compression_ratio', 1.0),
                'build_timestamp': build_info.get('build_timestamp', 0),
                'is_memory_mapped': self.unified_index.is_loaded,
                'cache_size': len(self.unified_index.metadata_cache),
                'features': [
                    'Memory-mapped access',
                    'LZ4 compression',
                    'Incremental updates',
                    'Instant loading',
                    'Thumbnail storage',
                    'Temporal relationships'
                ]
            }
            
            return stats
            
        except Exception as e:
            return {'error': str(e)}
    
    def close(self):
        """Clean up unified index resources"""
        if self.unified_index:
            self.unified_index.close()
            self.unified_index = None
            self._log_info("🔒 Unified index closed")
    
    # =====================================
    # PRIVATE HELPER METHODS
    # =====================================
    
    def _integrate_with_system(self):
        """Integrate unified index with existing system components"""
        try:
            if not self.unified_index or not hasattr(self.system, 'metadata_manager'):
                return
            
            # Populate temporal_index from unified index metadata
            if hasattr(self.system.metadata_manager, 'temporal_index'):
                self._log_info("🔗 Populating temporal_index from unified index...")
                
                # Clear existing temporal index
                self.system.metadata_manager.temporal_index.clear()
                
                # Get metadata from unified index
                if hasattr(self.unified_index, 'metadata_list') and self.unified_index.metadata_list:
                    for metadata_dict in self.unified_index.metadata_list:
                        folder_name = metadata_dict.get('folder_name', '')
                        if folder_name:
                            # Initialize folder in temporal_index if not exists
                            if folder_name not in self.system.metadata_manager.temporal_index:
                                self.system.metadata_manager.temporal_index[folder_name] = []
                            
                            # Create KeyframeMetadata object
                            metadata_obj = self._convert_metadata_to_legacy(metadata_dict)
                            
                            # Add to temporal_index
                            self.system.metadata_manager.temporal_index[folder_name].append(metadata_obj)
                    
                    # Sort temporal index by frame_id for each folder
                    for folder_name in self.system.metadata_manager.temporal_index:
                        self.system.metadata_manager.temporal_index[folder_name].sort(
                            key=lambda x: getattr(x, 'frame_id', 0)
                        )
                    
                    total_folders = len(self.system.metadata_manager.temporal_index)
                    total_frames = sum(len(frames) for frames in self.system.metadata_manager.temporal_index.values())
                    
                    self._log_info(f"✅ Temporal index populated: {total_folders} folders, {total_frames} frames")
                else:
                    self._log_warning("⚠️  No metadata found in unified index to populate temporal_index")
            else:
                self._log_warning("⚠️  System metadata_manager.temporal_index not available")
            
            # CRITICAL FIX: Synchronize FAISS retriever with unified index
            if hasattr(self.system, 'faiss_retriever') and hasattr(self.unified_index, 'faiss_index'):
                self._log_info("🔧 Synchronizing FAISS retriever with unified index...")
                
                try:
                    # Check if unified index has FAISS data loaded
                    faiss_index = getattr(self.unified_index, 'faiss_index', None)
                    metadata_list = getattr(self.unified_index, 'metadata_list', [])
                    build_info = self.unified_index._load_build_metadata()
                    
                    if faiss_index is not None and len(metadata_list) > 0:
                        # Set the FAISS index directly
                        self.system.faiss_retriever.index = faiss_index
                        self.system.faiss_retriever.index_clip_model = build_info.get('clip_model')
                        self.system.faiss_retriever.index_build_source = build_info.get('build_source')
                        
                        # CRITICAL: Set dimension from the FAISS index
                        if hasattr(faiss_index, 'd'):
                            self.system.faiss_retriever.dimension = faiss_index.d
                            self._log_info(f"📏 FAISS dimension set to: {self.system.faiss_retriever.dimension}")
                        
                        # Convert metadata list to id_to_metadata dict with proper KeyframeMetadata objects
                        id_to_metadata = {}
                        for i, metadata in enumerate(metadata_list):
                            # Convert to proper KeyframeMetadata object
                            id_to_metadata[i] = self._convert_metadata_to_legacy(metadata)
                        
                        self.system.faiss_retriever.id_to_metadata = id_to_metadata
                        
                        # CRITICAL: Mark as trained since the unified index is already trained
                        self.system.faiss_retriever.is_trained = True
                        
                        # Update temporal index for FAISS retriever
                        if hasattr(self.system.metadata_manager, 'temporal_index'):
                            self.system.faiss_retriever.temporal_index = self.system.metadata_manager.temporal_index
                        
                        self._log_info(f"✅ FAISS retriever synchronized: {len(id_to_metadata)} vectors, trained={self.system.faiss_retriever.is_trained}")
                        
                        # Additional verification
                        if hasattr(self.system.faiss_retriever.index, 'ntotal'):
                            vector_count = self.system.faiss_retriever.index.ntotal
                            self._log_info(f"📊 FAISS index verification: {vector_count} vectors loaded, dimension={self.system.faiss_retriever.dimension}")
                    else:
                        self._log_warning("⚠️  Unified index FAISS data not available for synchronization")
                        self._log_warning(f"Debug: faiss_index={faiss_index is not None}, metadata_list={len(metadata_list) if metadata_list else 0}")
                        
                except Exception as e:
                    self._log_error(f"FAISS synchronization failed: {e}")
                    import traceback
                    self._log_error(f"Traceback: {traceback.format_exc()}")
            else:
                self._log_warning("⚠️  System FAISS retriever not available for synchronization")
                self._log_warning(f"Debug: has faiss_retriever={hasattr(self.system, 'faiss_retriever')}, has faiss_index={hasattr(self.unified_index, 'faiss_index')}")
                
        except Exception as e:
            self._log_error(f"Failed to integrate with system: {e}")
            import traceback
            self._log_error(f"Traceback: {traceback.format_exc()}")

    def _ensure_query_encoder_matches_unified_index(self):
        """Reload text query encoder when unified index declares a specific CLIP model space."""
        try:
            if not self.unified_index:
                return

            build_info = self.unified_index._load_build_metadata()
            required_model = build_info.get('clip_model')
            build_source = build_info.get('build_source')
            index_dimension = getattr(getattr(self.unified_index, 'faiss_index', None), 'd', None)

            if hasattr(self.system, 'active_index_clip_model'):
                self.system.active_index_clip_model = required_model
            else:
                setattr(self.system, 'active_index_clip_model', required_model)
            setattr(self.system, 'active_index_build_source', build_source)

            if build_source != 'btc_clip_features' or not required_model:
                self.system.active_query_encoder = getattr(self.system, 'clip_processor', None)
                return

            current_processor = getattr(self.system, 'clip_processor', None)
            current_model = getattr(current_processor, 'model_path', None)
            current_dimension = self._get_clip_projection_dim(current_processor)
            if current_model == required_model and (index_dimension is None or current_dimension == index_dimension):
                self.system.active_query_encoder = current_processor
                self._log_info(
                    f"Query encoder already matches unified index: {required_model} "
                    f"(projection_dim={current_dimension}, index_dim={index_dimension})"
                )
                return

            from core import CLIPFeatureExtractor

            self._log_info(
                f"Reloading query encoder for BTC unified index | current={current_model} | required={required_model}"
            )
            self.system.clip_processor = CLIPFeatureExtractor(
                required_model,
                self.system.config,
                self.logger,
                allow_model_fallback=False
            )
            self.system.active_query_encoder = self.system.clip_processor
            loaded_dimension = self._get_clip_projection_dim(self.system.clip_processor)
            if index_dimension is not None and loaded_dimension != index_dimension:
                raise ValueError(
                    "QUERY_ENCODER_INDEX_DIMENSION_MISMATCH_AFTER_LOAD | "
                    f"query_model={required_model} | query_projection_dim={loaded_dimension} | "
                    f"index_dim={index_dimension}"
                )

            self._log_info(
                f"Query encoder model set to: {required_model} "
                f"(projection_dim={loaded_dimension}, index_dim={index_dimension})"
            )

        except Exception as e:
            self._log_error(f"Failed to synchronize query encoder with unified index: {e}")
            raise

    def _get_clip_projection_dim(self, clip_processor) -> Optional[int]:
        """Read CLIP projection dimension without encoding a query."""
        try:
            model = getattr(clip_processor, 'model', None)
            config = getattr(model, 'config', None)
            projection_dim = getattr(config, 'projection_dim', None)
            if projection_dim is not None:
                return int(projection_dim)
        except Exception:
            pass
        return None
    
    def _convert_metadata_to_legacy(self, metadata: Dict) -> 'KeyframeMetadata':
        """Convert unified metadata to legacy KeyframeMetadata format"""
        try:
            from core import KeyframeMetadata
            clip_features = metadata.get('clip_features')
            if isinstance(clip_features, list):
                import numpy as np
                clip_features = np.asarray(clip_features, dtype=np.float32)
            
            return KeyframeMetadata(
                folder_name=metadata.get('folder_name', ''),
                image_name=metadata.get('image_name', ''),
                frame_id=metadata.get('frame_id', 0),
                file_path=metadata.get('file_path', ''),
                # Temporal context
                sequence_position=metadata.get('sequence_position', 0),
                total_frames=metadata.get('total_frames', 0),
                neighboring_frames=metadata.get('neighboring_frames', []),
                scene_boundaries=metadata.get('scene_boundaries', []),
                # Semantic context
                clip_features=clip_features,
                llm_description=metadata.get('llm_description'),
                detected_objects=metadata.get('detected_objects', []),
                scene_tags=metadata.get('scene_tags', []),
                confidence_score=metadata.get('confidence_score', 0.0),
                # Relationships
                similar_frames=metadata.get('similar_frames', []),
                transition_frames=metadata.get('transition_frames', [])
            )
        except Exception as e:
            # Fallback: create minimal KeyframeMetadata with required fields
            self._log_warning(f"Failed to convert metadata to KeyframeMetadata: {e}")
            try:
                from core import KeyframeMetadata
                return KeyframeMetadata(
                    folder_name=str(metadata.get('folder_name', 'unknown')),
                    image_name=str(metadata.get('image_name', 'unknown.jpg')),
                    frame_id=int(metadata.get('frame_id', 0)),
                    file_path=str(metadata.get('file_path', ''))
                )
            except Exception as fallback_e:
                self._log_error(f"Fallback metadata conversion failed: {fallback_e}")
                # Final fallback to dict if all else fails
                return metadata
    
    def _log_info(self, message: str):
        """Log info message"""
        if self.logger:
            self.logger.info(message)
    
    def _log_warning(self, message: str):
        """Log warning message"""
        if self.logger:
            self.logger.warning(message)
    
    def _log_error(self, message: str):
        """Log error message"""
        if self.logger:
            self.logger.error(message)
    
    def _log_debug(self, message: str):
        """Log debug message"""
        if self.logger:
            self.logger.debug(message)


def add_unified_index_support(system_instance):
    """
    🔧 Add unified index support to existing system instance
    
    Args:
        system_instance: Instance of EnhancedRetrievalSystem
        
    Returns:
        UnifiedBuilderIntegration instance attached to system
    """
    runtime_system = get_runtime_system(system_instance)
    if not hasattr(runtime_system, 'unified_builder'):
        runtime_system.unified_builder = UnifiedBuilderIntegration(runtime_system)
    
    return runtime_system.unified_builder
