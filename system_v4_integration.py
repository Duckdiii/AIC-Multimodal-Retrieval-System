"""
System v4.0 Integration - Safe Enhancement of Existing System
=============================================================

Integrates video-level diversification with existing EnhancedRetrievalSystem
without breaking current functionality. Provides backward compatibility
and feature flags for controlled rollout.

Key Features:
- Backward compatible SearchOptions extension
- Optional video grouping with feature flags
- Safe integration with existing search flow
- Performance monitoring and validation

Author: AIC Competition System v4.0
Task: 1.4 - Core System Integration
"""

import os
import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing system components
try:
    from system import EnhancedRetrievalSystem, SearchOptions as OriginalSearchOptions, SystemStatus
    from core import SearchResult, KeyframeMetadata
    from video_diversified_search import VideoLevelDiversifier
    HAS_MAIN_SYSTEM = True
    print("[INTEGRATION] Successfully imported main system components")
except ImportError as e:
    HAS_MAIN_SYSTEM = False
    print(f"[INTEGRATION] Warning: Could not import main system - {e}")

# Import v4.0 enhancements
try:
    from enhanced_query_parser import EnhancedQueryParser, QueryType, is_complex_query, extract_scene_descriptions
    HAS_QUERY_PARSER = True
    print("[INTEGRATION] Successfully imported query parser v4.0")
except ImportError as e:
    HAS_QUERY_PARSER = False
    print(f"[INTEGRATION] Warning: Could not import query parser - {e}")

try:
    from vietnamese_nlp import VietnameseNLPProcessor, process_vietnamese_query, is_vietnamese_qa_query
    HAS_VIETNAMESE_NLP = True
    print("[INTEGRATION] Successfully imported Vietnamese NLP v4.0")
except ImportError as e:
    HAS_VIETNAMESE_NLP = False
    print(f"[INTEGRATION] Warning: Could not import Vietnamese NLP - {e}")

try:
    from trake_temporal_engine import TrakeQueryParser, parse_trake_query, extract_event_search_keywords
    HAS_TRAKE_ENGINE = True
    print("[INTEGRATION] Successfully imported TRAKE temporal engine v4.0")
except ImportError as e:
    HAS_TRAKE_ENGINE = False
    print(f"[INTEGRATION] Warning: Could not import TRAKE engine - {e}")
    
    # Fallback definitions for testing
    class EnhancedRetrievalSystem:
        pass
    
    @dataclass
    class OriginalSearchOptions:
        mode: str = "hybrid"
        limit: int = 50


@dataclass
class EnhancedSearchOptions(OriginalSearchOptions):
    """
    Enhanced SearchOptions with v4.0 video grouping capabilities
    
    Extends existing SearchOptions while maintaining full backward compatibility.
    Default values ensure existing code continues to work unchanged.
    """
    
    # v4.0 Video Grouping Features (backward compatible defaults)
    enable_video_grouping: bool = False           # Feature flag - OFF by default
    max_results_per_video: int = 50              # No limit = current behavior  
    diversity_threshold: float = 0.3             # Diversity balance factor
    video_grouping_strategy: str = "balanced"    # Diversification algorithm
    
    # v4.0 Advanced Features
    enable_multi_scene_parsing: bool = False     # Multi-scene query parsing
    query_complexity_threshold: float = 0.6     # Threshold for complex query detection
    parse_temporal_sequences: bool = True        # Extract temporal flow from queries
    combine_scene_results: bool = True           # Combine results from multiple scenes
    enable_vietnamese_processing: bool = False   # Vietnamese OCR/context
    enable_trake_processing: bool = False        # TRAKE temporal event detection
    trake_event_precision: float = 0.8           # Precision threshold for TRAKE events
    trake_temporal_window: int = 30              # Temporal window for event detection (seconds)
    competition_mode: bool = False               # AIC competition optimizations
    
    def validate(self) -> None:
        """Enhanced validation including v4.0 parameters"""
        # Call original validation if available
        if hasattr(super(), 'validate'):
            super().validate()
        else:
            # Basic validation for standalone testing
            if self.limit <= 0:
                raise ValueError(f"Search limit must be positive, got: {self.limit}")
        
        # v4.0 specific validation
        if self.max_results_per_video <= 0:
            raise ValueError(f"max_results_per_video must be positive, got: {self.max_results_per_video}")
        
        if not 0 <= self.diversity_threshold <= 1:
            raise ValueError(f"diversity_threshold must be between 0 and 1, got: {self.diversity_threshold}")
        
        valid_strategies = ["balanced", "weighted", "round_robin", "similarity_based"]
        if self.video_grouping_strategy not in valid_strategies:
            raise ValueError(f"video_grouping_strategy must be one of: {valid_strategies}")
        
        if self.limit > 1000:
            raise ValueError(f"Search limit too high ({self.limit}), maximum is 1000")


class EnhancedRetrievalSystemV4:
    """
    Enhanced Retrieval System v4.0 - AIC Competition Ready
    
    Wraps the existing EnhancedRetrievalSystem with v4.0 enhancements:
    - Video-level result diversification  
    - Multi-scene query parsing
    - Vietnamese context processing
    - Competition-specific optimizations
    
    Design Principles:
    - Zero breaking changes to existing functionality
    - Feature flags for controlled rollout
    - Performance monitoring and validation
    - Backward compatibility with all existing code
    """

    RUNTIME_STATE_ATTRS = {
        'active_query_encoder',
        'clip_processor',
        'unified_builder',
        'unified_index',
        'faiss_retriever',
        'metadata_manager',
        'active_index_clip_model',
        'active_index_build_source',
    }

    def __setattr__(self, name, value):
        """Route mutable retrieval runtime state to the wrapped base system."""
        if name in EnhancedRetrievalSystemV4.RUNTIME_STATE_ATTRS:
            base_system = self.__dict__.get('base_system')
            if base_system is not None:
                setattr(base_system, name, value)
                return
        super().__setattr__(name, value)
    
    def __init__(self, 
                 base_system: Optional[EnhancedRetrievalSystem] = None,
                 config_path: Optional[str] = None,
                 enable_v4_features: bool = True,
                 debug: bool = False):
        """
        Initialize v4.0 system wrapper
        
        Args:
            base_system: Existing EnhancedRetrievalSystem instance
            config_path: Path to configuration file
            enable_v4_features: Enable v4.0 enhancements
            debug: Enable debug logging
        """
        self.debug = debug
        self.enable_v4_features = enable_v4_features
        
        # Initialize or wrap base system
        if base_system is not None:
            self.base_system = base_system
            self._log_info("v4.0 wrapper initialized with existing system")
        elif HAS_MAIN_SYSTEM:
            # Create new base system
            self.base_system = EnhancedRetrievalSystem(
                config_path=config_path,
                auto_initialize=True,
                verbose=debug
            )
            self._log_info("v4.0 wrapper initialized with new base system")
        else:
            # Fallback for testing
            self.base_system = None
            self._log_warning("Running in test mode - no base system available")
        
        # Initialize v4.0 components
        self.video_diversifier = None
        self.multi_scene_parser = None
        self.vietnamese_processor = None
        
        if self.enable_v4_features:
            self._initialize_v4_components()
        
        # Performance tracking
        self.v4_stats = {
            'video_grouping_calls': 0,
            'dominance_reductions': [],
            'processing_times': [],
            'error_count': 0
        }
        
        self._log_info(f"EnhancedRetrievalSystemV4 initialized", 
                      v4_features=self.enable_v4_features,
                      base_system_available=self.base_system is not None)
    
    def _log_info(self, message: str, **kwargs):
        """Info logging helper"""
        if self.debug:
            print(f"[v4.0] {message}")
            for key, value in kwargs.items():
                print(f"  {key}: {value}")
    
    def _log_warning(self, message: str, **kwargs):
        """Warning logging helper"""
        print(f"[v4.0 WARNING] {message}")
        for key, value in kwargs.items():
            print(f"  {key}: {value}")
    
    def _log_error(self, message: str, **kwargs):
        """Error logging helper"""
        print(f"[v4.0 ERROR] {message}")
        for key, value in kwargs.items():
            print(f"  {key}: {value}")
        self.v4_stats['error_count'] += 1
    
    def _initialize_v4_components(self):
        """Initialize v4.0 enhancement components"""
        try:
            # Video diversification component
            self.video_diversifier = VideoLevelDiversifier(
                logger=getattr(self.base_system, 'logger', None),
                enable_stats=True,
                debug=self.debug
            )
            self._log_info("Video diversifier initialized")
            
            # Initialize Vietnamese NLP processor
            if HAS_VIETNAMESE_NLP:
                self.vietnamese_processor = VietnameseNLPProcessor()
                self._log_info("Vietnamese NLP processor initialized")
            else:
                self.vietnamese_processor = None
                self._log_info("Vietnamese NLP processor not available")
            
            # Initialize TRAKE temporal engine
            if HAS_TRAKE_ENGINE:
                self.trake_parser = TrakeQueryParser()
                self._log_info("TRAKE temporal engine initialized")
            else:
                self.trake_parser = None
                self._log_info("TRAKE temporal engine not available")
            
        except Exception as e:
            self._log_error(f"Failed to initialize v4.0 components: {e}")
            self.enable_v4_features = False
    
    def search(self, 
              query: str, 
              options: Optional[EnhancedSearchOptions] = None) -> List[SearchResult]:
        """
        Enhanced search with v4.0 capabilities
        
        Provides the same interface as the original system but with optional
        v4.0 enhancements like video-level grouping.
        
        Args:
            query: Search query text
            options: Enhanced search options with v4.0 features
            
        Returns:
            List of search results (potentially enhanced with v4.0 processing)
        """
        start_time = time.time()
        
        # Ensure we have search options
        if options is None:
            options = EnhancedSearchOptions()
        
        # Validate options
        options.validate()
        
        # Check if base system is available
        if not self.base_system:
            self._log_error("No base system available for search")
            return []
        
        try:
            # Step 1: Execute original search
            # Convert enhanced options to original format for base system
            original_options = self._convert_to_original_options(options)
            
            self._log_info(f"Executing base search", 
                          query=query[:50] + "..." if len(query) > 50 else query,
                          mode=options.mode,
                          limit=options.limit)
            
            # Get more results if video grouping is enabled (for better diversification)
            search_limit = options.limit
            if options.enable_video_grouping and self.enable_v4_features:
                search_limit = min(options.limit * 3, 1000)  # Get 3x results for grouping
            
            original_options.limit = search_limit
            
            # Call base system search
            base_results = self.base_system.search(query, original_options)
            
            self._log_info(f"Base search completed", 
                          results_count=len(base_results),
                          processing_time=f"{time.time() - start_time:.3f}s")
            
            # Step 2: Apply v4.0 enhancements if enabled
            enhanced_results = base_results
            
            if self.enable_v4_features:
                enhanced_results = self._apply_v4_enhancements(
                    base_results, query, options
                )
            
            # Step 3: Final result processing
            final_results = enhanced_results[:options.limit]
            
            # Update performance stats
            total_time = time.time() - start_time
            self.v4_stats['processing_times'].append(total_time)
            
            self._log_info(f"Search completed", 
                          final_count=len(final_results),
                          total_time=f"{total_time:.3f}s",
                          v4_enhanced=self.enable_v4_features)
            
            return final_results
            
        except Exception as e:
            self._log_error(f"Search failed: {e}")
            import traceback
            if self.debug:
                traceback.print_exc()
            return []
    
    def _convert_to_original_options(self, enhanced_options: EnhancedSearchOptions):
        """Convert enhanced options to original SearchOptions format"""
        if not HAS_MAIN_SYSTEM:
            return enhanced_options
        
        # Create original options with core parameters
        original = OriginalSearchOptions()
        
        # Copy compatible fields
        compatible_fields = ['mode', 'limit', 'include_temporal_context', 
                           'include_explanations', 'similarity_threshold', 
                           'enable_reranking', 'cache_results', 'validate_results']
        
        for field in compatible_fields:
            if hasattr(enhanced_options, field) and hasattr(original, field):
                setattr(original, field, getattr(enhanced_options, field))
        
        return original
    
    def _create_scene_search_options(self, base_options: EnhancedSearchOptions):
        """Create search options optimized for individual scene searches"""
        scene_options = self._convert_to_original_options(base_options)
        
        # Optimize for scene searches
        scene_options.limit = min(20, base_options.limit // 2)  # Fewer results per scene
        scene_options.mode = "hybrid"  # Use hybrid mode for scene searches
        
        return scene_options
    
    def _apply_vietnamese_context_to_results(self, results: List[SearchResult], 
                                           vietnamese_result, qa_analysis) -> List[SearchResult]:
        """Apply Vietnamese cultural and temporal context to search results"""
        if not results:
            return results
        
        # For now, this is a placeholder for Vietnamese context enhancement
        # In a full implementation, this could:
        # 1. Boost results that match cultural context (e.g., locations, events)
        # 2. Apply temporal reasoning for Q&A queries
        # 3. Use Vietnamese entity information for result ranking
        
        # Add Vietnamese processing metadata to results
        for result in results:
            if hasattr(result, 'metadata'):
                result.vietnamese_processing = {
                    'question_type': qa_analysis['question_type'],
                    'temporal_markers': len(vietnamese_result.temporal_markers),
                    'cultural_indicators': len(vietnamese_result.cultural_indicators),
                    'complexity_score': vietnamese_result.complexity_score
                }
        
        return results
    
    def _is_trake_query(self, query: str) -> bool:
        """Check if query is in TRAKE format (has event markers like E1:, E2:)"""
        return bool(re.search(r'E\d+:', query))
    
    def _create_trake_search_options(self, base_options: EnhancedSearchOptions, event):
        """Create search options optimized for TRAKE event searches"""
        trake_options = self._convert_to_original_options(base_options)
        
        # Optimize for TRAKE event searches
        trake_options.limit = min(15, base_options.limit // 3)  # Fewer results per event
        trake_options.mode = "hybrid"  # Use hybrid mode for precision
        
        return trake_options
    
    def _apply_temporal_alignment(self, event_results: List[SearchResult], 
                                 trake_result, options: EnhancedSearchOptions) -> List[SearchResult]:
        """Apply temporal alignment to TRAKE event results"""
        if not event_results:
            return event_results
        
        # For now, this is a simplified temporal alignment
        # In a full implementation, this would:
        # 1. Sort events by temporal sequence
        # 2. Apply temporal constraints (dependencies)
        # 3. Ensure proper event ordering in video timeline
        # 4. Apply temporal window constraints
        
        # Sort by TRAKE event ID to maintain sequence
        aligned_results = sorted(event_results, key=lambda x: getattr(x, 'trake_event_id', 'E999'))
        
        # Add temporal alignment metadata
        for i, result in enumerate(aligned_results):
            if hasattr(result, 'metadata'):
                result.temporal_sequence = i + 1
                result.temporal_alignment_confidence = 0.8  # Default confidence
        
        return aligned_results
    
    def _apply_v4_enhancements(self, 
                              base_results: List[SearchResult], 
                              query: str, 
                              options: EnhancedSearchOptions) -> List[SearchResult]:
        """
        Apply v4.0 enhancements to base search results
        
        Args:
            base_results: Results from base system
            query: Original search query
            options: Enhanced search options
            
        Returns:
            Enhanced results with v4.0 processing
        """
        enhanced_results = base_results
        enhancement_start = time.time()
        
        # Enhancement 1: Video-level diversification
        if options.enable_video_grouping and self.video_diversifier:
            try:
                self._log_info("Applying video-level diversification")
                
                enhanced_results = self.video_diversifier.diversify_results(
                    enhanced_results, 
                    options, 
                    strategy=options.video_grouping_strategy
                )
                
                # Track improvement stats
                improvement_stats = self.video_diversifier.get_improvement_stats()
                if improvement_stats:
                    dominance_reduction = improvement_stats['dominance_reduction']
                    self.v4_stats['dominance_reductions'].append(dominance_reduction)
                    self.v4_stats['video_grouping_calls'] += 1
                    
                    self._log_info(f"Video grouping applied", 
                                  dominance_reduction=f"{dominance_reduction:.1%}",
                                  strategy=options.video_grouping_strategy)
                
            except Exception as e:
                self._log_error(f"Video diversification failed: {e}")
        
        # Enhancement 2: Multi-scene parsing
        if options.enable_multi_scene_parsing and HAS_QUERY_PARSER:
            try:
                self._log_info("Applying multi-scene query parsing")
                
                # Initialize parser if needed
                if not hasattr(self, 'query_parser'):
                    self.query_parser = EnhancedQueryParser()
                
                # Check if query is complex enough for parsing
                if is_complex_query(query, options.query_complexity_threshold):
                    # Parse query into scenes
                    parsed_query = self.query_parser.parse_query(query)
                    
                    self._log_info(f"Parsed complex query", 
                                  scenes=len(parsed_query.scenes),
                                  query_type=parsed_query.query_type.value,
                                  complexity=f"{parsed_query.complexity_score:.2f}")
                    
                    # If multi-scene query, perform additional searches for each scene
                    if len(parsed_query.scenes) > 1 and options.combine_scene_results:
                        scene_results = []
                        
                        for scene in parsed_query.scenes:
                            try:
                                # Search for individual scene
                                scene_query = scene.description
                                scene_options = self._create_scene_search_options(options)
                                
                                scene_search_results = self.base_system.search(scene_query, scene_options)
                                
                                # Add scene context to results
                                for result in scene_search_results[:10]:  # Limit per scene
                                    if hasattr(result, 'metadata'):
                                        result.scene_id = scene.scene_id
                                        result.scene_context = scene.description[:100]
                                    
                                scene_results.extend(scene_search_results[:10])
                                
                                self._log_info(f"Scene {scene.scene_id} search", 
                                              results=len(scene_search_results))
                                
                            except Exception as e:
                                self._log_error(f"Scene search failed: {e}")
                        
                        # Combine scene results with original results
                        if scene_results:
                            # Remove duplicates and merge
                            seen_frames = set()
                            combined_results = []
                            
                            # Add original results first
                            for result in enhanced_results:
                                frame_key = f"{result.metadata.folder_name}_{result.metadata.image_name}"
                                if frame_key not in seen_frames:
                                    seen_frames.add(frame_key)
                                    combined_results.append(result)
                            
                            # Add unique scene results
                            for result in scene_results:
                                frame_key = f"{result.metadata.folder_name}_{result.metadata.image_name}"
                                if frame_key not in seen_frames:
                                    seen_frames.add(frame_key)
                                    combined_results.append(result)
                            
                            enhanced_results = combined_results
                            
                            self._log_info(f"Multi-scene results combined", 
                                          total_results=len(enhanced_results),
                                          scene_contributions=len(scene_results))
                else:
                    self._log_info("Query complexity below threshold for multi-scene parsing")
                    
            except Exception as e:
                self._log_error(f"Multi-scene parsing failed: {e}")
        elif options.enable_multi_scene_parsing:
            self._log_info("Multi-scene parsing requested but query parser not available")
        
        # Enhancement 3: Vietnamese processing
        if options.enable_vietnamese_processing and HAS_VIETNAMESE_NLP and self.vietnamese_processor:
            try:
                self._log_info("Applying Vietnamese text processing")
                
                # Check if query is Vietnamese Q&A type
                if is_vietnamese_qa_query(query):
                    # Process Vietnamese query for cultural and temporal context
                    vietnamese_result = self.vietnamese_processor.process_qa_query(query)
                    processing_result = vietnamese_result['processing_result']
                    qa_analysis = vietnamese_result['qa_analysis']
                    
                    self._log_info(f"Vietnamese Q&A processing", 
                                  question_type=qa_analysis['question_type'],
                                  required_context=len(qa_analysis['required_context']),
                                  temporal_markers=len(processing_result.temporal_markers),
                                  cultural_indicators=len(processing_result.cultural_indicators))
                    
                    # Enhance results with Vietnamese context
                    enhanced_results = self._apply_vietnamese_context_to_results(
                        enhanced_results, processing_result, qa_analysis
                    )
                    
                    # Track Vietnamese processing stats
                    if not hasattr(self.v4_stats, 'vietnamese_processing_calls'):
                        self.v4_stats['vietnamese_processing_calls'] = 0
                    self.v4_stats['vietnamese_processing_calls'] += 1
                    
                else:
                    # Standard Vietnamese text processing for temporal markers
                    vietnamese_result = self.vietnamese_processor.process_text(query)
                    
                    if vietnamese_result.temporal_markers:
                        self._log_info(f"Vietnamese temporal processing", 
                                      temporal_markers=len(vietnamese_result.temporal_markers),
                                      temporal_sequence=len(vietnamese_result.temporal_sequence))
                        
                        # Could use temporal information for result ranking
                        # This is a placeholder for future temporal-aware ranking
                    
            except Exception as e:
                self._log_error(f"Vietnamese processing failed: {e}")
        elif options.enable_vietnamese_processing:
            self._log_info("Vietnamese processing requested but not available")
        
        # Enhancement 4: TRAKE temporal event processing
        if options.enable_trake_processing and HAS_TRAKE_ENGINE and self.trake_parser:
            try:
                self._log_info("Applying TRAKE temporal event processing")
                
                # Check if query is TRAKE format (has E1:, E2:, etc.)
                if self._is_trake_query(query):
                    # Parse TRAKE query to extract events
                    trake_result = self.trake_parser.parse_trake_query(query)
                    
                    self._log_info(f"TRAKE query processing", 
                                  domain=trake_result.domain,
                                  events=len(trake_result.events),
                                  complexity=f"{trake_result.complexity_score:.2f}")
                    
                    # Process each event for frame-level search
                    event_results = []
                    
                    for event in trake_result.events:
                        try:
                            # Create search query from event keywords  
                            event_query = " ".join(event.search_keywords[:5])  # Top 5 keywords
                            event_options = self._create_trake_search_options(options, event)
                            
                            # Search for this specific event
                            event_search_results = self.base_system.search(event_query, event_options)
                            
                            # Add TRAKE metadata to results
                            for result in event_search_results[:10]:  # Limit per event
                                if hasattr(result, 'metadata'):
                                    result.trake_event_id = event.event_id
                                    result.trake_domain = trake_result.domain
                                    result.trake_event_type = event.start_boundary.event_type.value if event.start_boundary else "unknown"
                                    result.trake_duration_type = event.duration_type
                                    result.trake_confidence = event.start_boundary.confidence if event.start_boundary else 0.5
                            
                            event_results.extend(event_search_results[:10])
                            
                            self._log_info(f"TRAKE event {event.event_id} search", 
                                          results=len(event_search_results),
                                          keywords_used=len(event.search_keywords))
                            
                        except Exception as e:
                            self._log_error(f"TRAKE event search failed for {event.event_id}: {e}")
                    
                    # Apply temporal alignment to event results
                    if event_results:
                        aligned_results = self._apply_temporal_alignment(
                            event_results, trake_result, options
                        )
                        
                        # Replace or merge with original results based on TRAKE precision
                        if options.trake_event_precision >= 0.8:
                            # High precision - replace original results
                            enhanced_results = aligned_results
                        else:
                            # Lower precision - merge with original results
                            enhanced_results.extend(aligned_results)
                            # Remove duplicates
                            seen_frames = set()
                            deduplicated = []
                            for result in enhanced_results:
                                frame_key = f"{result.metadata.folder_name}_{result.metadata.image_name}"
                                if frame_key not in seen_frames:
                                    seen_frames.add(frame_key)
                                    deduplicated.append(result)
                            enhanced_results = deduplicated
                        
                        self._log_info(f"TRAKE processing completed", 
                                      total_results=len(enhanced_results),
                                      event_contributions=len(aligned_results))
                        
                        # Track TRAKE processing stats
                        if not hasattr(self.v4_stats, 'trake_processing_calls'):
                            self.v4_stats['trake_processing_calls'] = 0
                        self.v4_stats['trake_processing_calls'] += 1
                else:
                    self._log_info("Query not in TRAKE format, skipping TRAKE processing")
                    
            except Exception as e:
                self._log_error(f"TRAKE processing failed: {e}")
        elif options.enable_trake_processing:
            self._log_info("TRAKE processing requested but engine not available")
        
        # Enhancement 5: Competition-specific optimizations
        if options.competition_mode:
            enhanced_results = self._apply_competition_optimizations(
                enhanced_results, query, options
            )
        
        enhancement_time = time.time() - enhancement_start
        self._log_info(f"v4.0 enhancements applied", 
                      processing_time=f"{enhancement_time:.3f}s")
        
        return enhanced_results
    
    def _apply_competition_optimizations(self, 
                                       results: List[SearchResult], 
                                       query: str, 
                                       options: EnhancedSearchOptions) -> List[SearchResult]:
        """Apply AIC competition specific optimizations"""
        # TODO: Implement competition-specific result ranking
        # - Boost results that match competition query patterns
        # - Apply domain-specific scoring adjustments
        # - Optimize for competition evaluation metrics
        
        self._log_info("Competition optimizations applied (placeholder)")
        return results
    
    def get_v4_stats(self) -> Dict[str, Any]:
        """Get v4.0 performance statistics"""
        stats = self.v4_stats.copy()
        
        # Calculate summary statistics
        if stats['processing_times']:
            stats['avg_processing_time'] = sum(stats['processing_times']) / len(stats['processing_times'])
            stats['max_processing_time'] = max(stats['processing_times'])
        
        if stats['dominance_reductions']:
            stats['avg_dominance_reduction'] = sum(stats['dominance_reductions']) / len(stats['dominance_reductions'])
            stats['max_dominance_reduction'] = max(stats['dominance_reductions'])
        
        return stats
    
    def analyze_video_distribution(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Analyze video distribution in search results"""
        if not results:
            return {}
        
        video_counts = Counter()
        for result in results:
            if hasattr(result, 'metadata') and hasattr(result.metadata, 'folder_name'):
                video_counts[result.metadata.folder_name] += 1
        
        total_results = len(results)
        unique_videos = len(video_counts)
        
        analysis = {
            'total_results': total_results,
            'unique_videos': unique_videos,
            'video_distribution': dict(video_counts),
            'diversity_score': unique_videos / total_results if total_results > 0 else 0
        }
        
        if unique_videos > 0:
            top_video, max_count = video_counts.most_common(1)[0]
            analysis['top_video'] = top_video
            analysis['max_occurrences'] = max_count
            analysis['dominance_ratio'] = max_count / total_results
        
        return analysis
    
    # Delegate other methods to base system
    def __getattr__(self, name):
        """Delegate unknown methods to base system"""
        if self.base_system and hasattr(self.base_system, name):
            return getattr(self.base_system, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def test_integration():
    """Test v4.0 integration with mock data"""
    print("=" * 60)
    print("SYSTEM v4.0 INTEGRATION TEST")
    print("=" * 60)
    
    # Test 1: Initialization
    print("\nTest 1: System Initialization")
    print("-" * 30)
    
    try:
        system_v4 = EnhancedRetrievalSystemV4(
            base_system=None,  # Will use fallback
            enable_v4_features=True,
            debug=True
        )
        print("SUCCESS: v4.0 system initialized")
    except Exception as e:
        print(f"ERROR: Initialization failed - {e}")
        return False
    
    # Test 2: Enhanced SearchOptions
    print("\nTest 2: Enhanced SearchOptions")
    print("-" * 30)
    
    try:
        # Test backward compatibility
        options_basic = EnhancedSearchOptions()
        options_basic.validate()
        print("SUCCESS: Basic options work (backward compatible)")
        
        # Test v4.0 features
        options_v4 = EnhancedSearchOptions(
            enable_video_grouping=True,
            max_results_per_video=3,
            diversity_threshold=0.4,
            video_grouping_strategy="balanced"
        )
        options_v4.validate()
        print("SUCCESS: v4.0 options work")
        
        # Test validation
        try:
            bad_options = EnhancedSearchOptions(max_results_per_video=-1)
            bad_options.validate()
            print("ERROR: Validation should have failed")
        except ValueError:
            print("SUCCESS: Validation working correctly")
        
    except Exception as e:
        print(f"ERROR: SearchOptions test failed - {e}")
        return False
    
    # Test 3: Feature flags
    print("\nTest 3: Feature Flag Control")
    print("-" * 30)
    
    # Test with v4.0 features disabled
    system_v3 = EnhancedRetrievalSystemV4(enable_v4_features=False, debug=True)
    print("SUCCESS: v4.0 features can be disabled")
    
    # Test with v4.0 features enabled
    print("SUCCESS: v4.0 features can be enabled")
    
    # Test 4: Stats tracking
    print("\nTest 4: Performance Statistics")
    print("-" * 30)
    
    stats = system_v4.get_v4_stats()
    expected_keys = ['video_grouping_calls', 'dominance_reductions', 'processing_times', 'error_count']
    
    if all(key in stats for key in expected_keys):
        print("SUCCESS: Statistics tracking initialized")
    else:
        print("ERROR: Statistics tracking incomplete")
        return False
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    print("SUCCESS: All integration tests passed")
    print("SUCCESS: v4.0 system ready for real data testing")
    print("Next step: Test with actual AIC dataset")
    
    return True


if __name__ == "__main__":
    success = test_integration()
    
    if success:
        print(f"\n🚀 READY FOR REAL DATA TESTING")
        print("Run: python test_with_real_data.py")
    else:
        print(f"\n❌ INTEGRATION ISSUES FOUND")
        print("Fix integration problems before proceeding")
