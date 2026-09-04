"""
Video-Level Diversified Search - AIC Competition System v4.0
============================================================

Enhances existing search with video-level result grouping to solve dominance problem.
Designed for safe integration with current system without breaking changes.

Key Features:
- Reduces single-video dominance (target: >50% reduction)
- Backward compatible with existing SearchOptions
- Maintains performance with large datasets (1.6M frames)
- Pluggable architecture for easy integration

Author: AIC Competition System v4.0
Task: 1.1 - Video-Level Result Grouping (HIGH PRIORITY)
"""

import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
import numpy as np

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing system components (maintain compatibility)
try:
    from core import SearchResult, KeyframeMetadata
    from system import SearchOptions
    HAS_CORE_SYSTEM = True
except ImportError:
    # Fallback for standalone testing
    HAS_CORE_SYSTEM = False
    
    @dataclass
    class SearchResult:
        metadata: Any
        similarity_score: float = 0.0
        
    @dataclass
    class SearchOptions:
        limit: int = 50
        enable_video_grouping: bool = False
        max_results_per_video: int = 5
        diversity_threshold: float = 0.3
        mode: str = "hybrid"


@dataclass
class VideoGroupStats:
    """Statistics for video-level grouping analysis"""
    total_results: int
    unique_videos: int
    max_occurrences: int
    top_video_name: str
    dominance_ratio: float
    diversity_score: float
    
    
class VideoLevelDiversifier:
    """
    🎯 Video-Level Result Diversifier
    
    Enhances existing FAISS search results with video-level grouping
    to prevent single-video dominance and improve result diversity.
    
    Design Principles:
    - Non-breaking: Works with existing SearchResult format
    - Configurable: Multiple diversification strategies
    - Performance: Efficient with large result sets
    - Measurable: Provides detailed statistics
    """
    
    def __init__(self, 
                 logger=None, 
                 enable_stats: bool = True,
                 debug: bool = False):
        """
        Initialize video diversifier
        
        Args:
            logger: Optional logger instance
            enable_stats: Whether to calculate detailed statistics
            debug: Enable debug logging
        """
        self.logger = logger
        self.enable_stats = enable_stats
        self.debug = debug
        
        # Statistics tracking
        self.last_stats = None
        self.improvement_history = []
        
        # Diversification strategies
        self.strategies = {
            'balanced': self._balanced_diversification,
            'weighted': self._weighted_diversification,
            'round_robin': self._round_robin_diversification,
            'similarity_based': self._similarity_based_diversification
        }
        
        if self.debug:
            self._log_debug("VideoLevelDiversifier initialized", 
                          strategies=list(self.strategies.keys()))
    
    def _log_debug(self, message: str, **kwargs):
        """Debug logging helper"""
        if self.debug:
            print(f"[VIDEO_DIVERSIFIER] {message}")
            for key, value in kwargs.items():
                print(f"  {key}: {value}")
    
    def _log_info(self, message: str, **kwargs):
        """Info logging helper"""  
        if self.logger:
            self.logger.info(message, **kwargs)
        elif self.debug:
            print(f"[INFO] {message}")

    def _get_ranking_score(self, result: SearchResult) -> float:
        """Use ranking score for ordering, falling back to semantic similarity."""
        ranking_score = getattr(result, 'ranking_score', None)
        if ranking_score is None:
            return float(getattr(result, 'similarity_score', 0.0))
        return float(ranking_score)
    
    def analyze_current_results(self, results: List[SearchResult]) -> VideoGroupStats:
        """
        Analyze video distribution in current results
        
        Args:
            results: List of search results from existing system
            
        Returns:
            VideoGroupStats with detailed analysis
        """
        if not results:
            return VideoGroupStats(0, 0, 0, "", 0.0, 0.0)
        
        # Count video occurrences
        video_counts = Counter()
        for result in results:
            if hasattr(result, 'metadata') and hasattr(result.metadata, 'folder_name'):
                video_name = result.metadata.folder_name
            elif hasattr(result, 'folder_name'):
                video_name = result.folder_name
            else:
                video_name = "unknown"
            
            video_counts[video_name] += 1
        
        # Calculate statistics
        total_results = len(results)
        unique_videos = len(video_counts)
        
        if unique_videos > 0:
            top_video, max_occurrences = video_counts.most_common(1)[0]
            dominance_ratio = max_occurrences / total_results
            
            # Diversity score: higher is better (closer to uniform distribution)
            expected_per_video = total_results / unique_videos
            variance = sum((count - expected_per_video) ** 2 for count in video_counts.values())
            diversity_score = 1.0 / (1.0 + variance / expected_per_video)
            
        else:
            top_video, max_occurrences = "none", 0
            dominance_ratio = 0.0
            diversity_score = 0.0
        
        return VideoGroupStats(
            total_results=total_results,
            unique_videos=unique_videos,
            max_occurrences=max_occurrences,
            top_video_name=top_video,
            dominance_ratio=dominance_ratio,
            diversity_score=diversity_score
        )
    
    def diversify_results(self, 
                         raw_results: List[SearchResult], 
                         options: SearchOptions,
                         strategy: str = "balanced") -> List[SearchResult]:
        """
        Apply video-level diversification to search results
        
        Args:
            raw_results: Results from existing FAISS search
            options: Enhanced SearchOptions with video grouping parameters
            strategy: Diversification strategy to use
            
        Returns:
            Diversified results with improved video balance
        """
        start_time = time.time()
        
        # Check if video grouping is enabled
        if not getattr(options, 'enable_video_grouping', False):
            self._log_debug("Video grouping disabled, returning original results")
            return raw_results[:options.limit]
        
        if not raw_results:
            self._log_debug("No results to diversify")
            return []
        
        # Analyze current distribution (before)
        if self.enable_stats:
            before_stats = self.analyze_current_results(raw_results)
            self._log_info(f"Before diversification: {before_stats.unique_videos} videos, "
                          f"max dominance: {before_stats.max_occurrences}/{before_stats.total_results} "
                          f"({before_stats.dominance_ratio:.1%})")
        
        # Apply diversification strategy
        if strategy not in self.strategies:
            self._log_debug(f"Unknown strategy '{strategy}', using 'balanced'")
            strategy = "balanced"
        
        diversified_results = self.strategies[strategy](raw_results, options)
        
        # Analyze results (after)
        if self.enable_stats:
            after_stats = self.analyze_current_results(diversified_results)
            
            # Calculate improvement
            dominance_reduction = before_stats.dominance_ratio - after_stats.dominance_ratio
            diversity_improvement = after_stats.diversity_score - before_stats.diversity_score
            
            self.last_stats = {
                'before': before_stats,
                'after': after_stats,
                'dominance_reduction': dominance_reduction,
                'diversity_improvement': diversity_improvement,
                'processing_time': time.time() - start_time,
                'strategy_used': strategy
            }
            
            self._log_info(f"After diversification: {after_stats.unique_videos} videos, "
                          f"max dominance: {after_stats.max_occurrences}/{after_stats.total_results} "
                          f"({after_stats.dominance_ratio:.1%})")
            self._log_info(f"Improvement: {dominance_reduction:.1%} dominance reduction, "
                          f"{diversity_improvement:.3f} diversity gain")
            
            # Track improvement history
            self.improvement_history.append({
                'timestamp': time.time(),
                'dominance_reduction': dominance_reduction,
                'diversity_improvement': diversity_improvement
            })
        
        return diversified_results
    
    def _balanced_diversification(self, 
                                 results: List[SearchResult], 
                                 options: SearchOptions) -> List[SearchResult]:
        """
        Balanced diversification: Limit max results per video
        
        Strategy: Ensures no single video dominates by capping occurrences
        Target: Achieve >50% dominance reduction
        """
        max_per_video = getattr(options, 'max_results_per_video', 5)
        target_count = options.limit
        
        # Group results by video
        video_groups = defaultdict(list)
        for result in results:
            video_name = self._get_video_name(result)
            video_groups[video_name].append(result)
        
        # Sort videos by their best ranking score
        sorted_videos = sorted(video_groups.items(), 
                              key=lambda x: max(self._get_ranking_score(r) for r in x[1]), 
                              reverse=True)
        
        # Apply balanced selection
        diversified = []
        video_used_counts = defaultdict(int)
        
        # Round-robin through videos, respecting per-video limits
        round_num = 0
        while len(diversified) < target_count and any(len(group) > round_num for _, group in sorted_videos):
            for video_name, group in sorted_videos:
                if len(diversified) >= target_count:
                    break
                
                if (round_num < len(group) and 
                    video_used_counts[video_name] < max_per_video):
                    
                    diversified.append(group[round_num])
                    video_used_counts[video_name] += 1
            
            round_num += 1
        
        return diversified
    
    def _weighted_diversification(self,
                                 results: List[SearchResult], 
                                 options: SearchOptions) -> List[SearchResult]:
        """
        Weighted diversification: Balance similarity scores with diversity
        
        Strategy: Use weighted selection favoring both quality and diversity
        """
        diversity_weight = getattr(options, 'diversity_threshold', 0.3)
        target_count = options.limit
        
        # Group and analyze videos
        video_groups = defaultdict(list)
        for result in results:
            video_name = self._get_video_name(result)
            video_groups[video_name].append(result)
        
        # Calculate video weights (higher = better diversity value)
        video_weights = {}
        total_videos = len(video_groups)
        
        for video_name, group in video_groups.items():
            # Base weight: average ranking score of top results
            avg_similarity = np.mean([self._get_ranking_score(r) for r in group[:3]])
            
            # Diversity bonus: fewer results = higher bonus
            diversity_bonus = 1.0 / (1.0 + len(group) / 10.0)
            
            # Combined weight
            video_weights[video_name] = (
                (1 - diversity_weight) * avg_similarity + 
                diversity_weight * diversity_bonus
            )
        
        # Select results based on weighted priority
        diversified = []
        used_videos = set()
        
        # First pass: One result from each video (diversity)
        sorted_by_weight = sorted(video_groups.items(), 
                                 key=lambda x: video_weights[x[0]], 
                                 reverse=True)
        
        for video_name, group in sorted_by_weight:
            if len(diversified) < target_count:
                diversified.append(group[0])  # Best result from this video
                used_videos.add(video_name)
        
        # Second pass: Fill remaining slots with best remaining results
        remaining_results = []
        for video_name, group in video_groups.items():
            remaining_results.extend(group[1:])  # Skip first (already used)
        
        remaining_results.sort(key=self._get_ranking_score, reverse=True)
        
        for result in remaining_results:
            if len(diversified) >= target_count:
                break
            diversified.append(result)
        
        return diversified
    
    def _round_robin_diversification(self,
                                   results: List[SearchResult], 
                                   options: SearchOptions) -> List[SearchResult]:
        """
        Round-robin diversification: Cycle through videos evenly
        
        Strategy: Maximum diversity by rotating through all available videos
        """
        target_count = options.limit
        
        # Group by video
        video_groups = defaultdict(list)
        for result in results:
            video_name = self._get_video_name(result)
            video_groups[video_name].append(result)
        
        # Sort videos by their best result quality
        sorted_videos = sorted(video_groups.items(),
                              key=lambda x: max(self._get_ranking_score(r) for r in x[1]),
                              reverse=True)
        
        # Round-robin selection
        diversified = []
        max_rounds = max(len(group) for _, group in sorted_videos) if sorted_videos else 0
        
        for round_num in range(max_rounds):
            for video_name, group in sorted_videos:
                if len(diversified) >= target_count:
                    break
                
                if round_num < len(group):
                    diversified.append(group[round_num])
            
            if len(diversified) >= target_count:
                break
        
        return diversified
    
    def _similarity_based_diversification(self,
                                        results: List[SearchResult], 
                                        options: SearchOptions) -> List[SearchResult]:
        """
        Similarity-based diversification: Balance quality with video diversity
        
        Strategy: Prioritize high-quality results while maintaining video diversity
        """
        similarity_threshold = getattr(options, 'diversity_threshold', 0.7)
        target_count = options.limit
        
        # Separate high-quality and diverse results
        high_quality = [r for r in results if r.similarity_score >= similarity_threshold]
        diverse_pool = [r for r in results if r.similarity_score < similarity_threshold]
        
        # Group both pools by video
        hq_video_groups = defaultdict(list)
        diverse_video_groups = defaultdict(list)
        
        for result in high_quality:
            video_name = self._get_video_name(result)
            hq_video_groups[video_name].append(result)
        
        for result in diverse_pool:
            video_name = self._get_video_name(result)
            diverse_video_groups[video_name].append(result)
        
        # Selection strategy: 70% from high-quality (limited per video), 30% from diverse pool
        diversified = []
        
        # Phase 1: High-quality results (limited per video to prevent dominance)
        hq_limit = int(target_count * 0.7)
        max_hq_per_video = max(1, hq_limit // len(hq_video_groups)) if hq_video_groups else 1
        
        for video_name, group in hq_video_groups.items():
            take_count = min(len(group), max_hq_per_video, hq_limit - len(diversified))
            diversified.extend(group[:take_count])
            
            if len(diversified) >= hq_limit:
                break
        
        # Phase 2: Fill remaining with diverse results
        remaining_needed = target_count - len(diversified)
        if remaining_needed > 0:
            # Prioritize videos not yet represented
            used_videos = set(self._get_video_name(r) for r in diversified)
            
            # First, add results from unused videos
            for video_name, group in diverse_video_groups.items():
                if video_name not in used_videos and remaining_needed > 0:
                    diversified.append(group[0])
                    remaining_needed -= 1
                    used_videos.add(video_name)
            
            # Then, add remaining best results
            if remaining_needed > 0:
                remaining_results = []
                for group in diverse_video_groups.values():
                    remaining_results.extend(group)
                
                remaining_results.sort(key=self._get_ranking_score, reverse=True)
                diversified.extend(remaining_results[:remaining_needed])
        
        return diversified
    
    def _get_video_name(self, result: SearchResult) -> str:
        """Extract video name from search result"""
        if hasattr(result, 'metadata') and hasattr(result.metadata, 'folder_name'):
            return result.metadata.folder_name
        elif hasattr(result, 'folder_name'):
            return result.folder_name
        else:
            return "unknown"
    
    def get_improvement_stats(self) -> Optional[Dict[str, Any]]:
        """Get latest improvement statistics"""
        return self.last_stats
    
    def get_improvement_trend(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent improvement history"""
        return self.improvement_history[-limit:] if self.improvement_history else []
    
    def test_strategy_performance(self, 
                                 results: List[SearchResult],
                                 options: SearchOptions) -> Dict[str, Any]:
        """
        Test all diversification strategies and compare performance
        
        Args:
            results: Test results to diversify
            options: Search options to use
            
        Returns:
            Performance comparison of all strategies
        """
        if not results:
            return {}
        
        original_stats = self.analyze_current_results(results)
        strategy_performance = {}
        
        for strategy_name in self.strategies.keys():
            try:
                # Test strategy
                diversified = self.strategies[strategy_name](results, options)
                diversified_stats = self.analyze_current_results(diversified)
                
                # Calculate metrics
                dominance_reduction = original_stats.dominance_ratio - diversified_stats.dominance_ratio
                diversity_improvement = diversified_stats.diversity_score - original_stats.diversity_score
                
                strategy_performance[strategy_name] = {
                    'dominance_reduction': dominance_reduction,
                    'diversity_improvement': diversity_improvement,
                    'final_unique_videos': diversified_stats.unique_videos,
                    'final_max_occurrences': diversified_stats.max_occurrences,
                    'final_diversity_score': diversified_stats.diversity_score,
                    'success': dominance_reduction > 0.3  # Target: >30% reduction
                }
                
            except Exception as e:
                strategy_performance[strategy_name] = {
                    'error': str(e),
                    'success': False
                }
        
        return strategy_performance


# ============================================================================
# ENHANCED SEARCHOPTIONS EXTENSION
# ============================================================================

def extend_search_options():
    """
    Safely extend existing SearchOptions with v4.0 video grouping parameters
    """
    if not HAS_CORE_SYSTEM:
        return None
    
    # Import the original class
    from system import SearchOptions as OriginalSearchOptions
    
    # Create enhanced version with additional fields
    @dataclass
    class EnhancedSearchOptions(OriginalSearchOptions):
        """Enhanced SearchOptions with video-level grouping support"""
        
        # NEW v4.0 fields (backward compatible defaults)
        enable_video_grouping: bool = False
        max_results_per_video: int = 5
        diversity_threshold: float = 0.3
        video_grouping_strategy: str = "balanced"
        
        def validate(self) -> None:
            """Enhanced validation including v4.0 parameters"""
            # Call original validation
            super().validate()
            
            # Additional v4.0 validation
            if self.max_results_per_video <= 0:
                raise ValueError(f"max_results_per_video must be positive, got: {self.max_results_per_video}")
            
            if not 0 <= self.diversity_threshold <= 1:
                raise ValueError(f"diversity_threshold must be between 0 and 1, got: {self.diversity_threshold}")
            
            valid_strategies = ["balanced", "weighted", "round_robin", "similarity_based"]
            if self.video_grouping_strategy not in valid_strategies:
                raise ValueError(f"video_grouping_strategy must be one of: {valid_strategies}")
    
    return EnhancedSearchOptions


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def create_mock_results(num_results: int = 50, num_videos: int = 10) -> List[SearchResult]:
    """Create mock SearchResult objects for testing"""
    
    # Create realistic video names (like actual dataset)
    video_names = [f"L{21 + i//30}_V{str(i%30).zfill(3)}" for i in range(num_videos)]
    
    results = []
    
    # Simulate dominance problem: 70% results from top 3 videos
    dominant_videos = video_names[:3]
    other_videos = video_names[3:]
    
    for i in range(num_results):
        # 70% chance for dominant videos (simulating current problem)
        if i < int(num_results * 0.7) and dominant_videos:
            video = random.choice(dominant_videos)
        else:
            video = random.choice(other_videos) if other_videos else random.choice(video_names)
        
        # Create mock metadata
        metadata = type('MockMetadata', (), {})()
        metadata.folder_name = video
        metadata.frame_id = random.randint(100, 30000)
        
        # Create search result with realistic similarity scores
        result = SearchResult(
            metadata=metadata,
            similarity_score=random.uniform(0.6, 0.95),
            rank=i,  # Required field
            query_relevance=random.uniform(0.5, 0.9)
        )
        
        results.append(result)
    
    # Sort by similarity score (like real search would)
    results.sort(key=lambda x: x.similarity_score, reverse=True)
    
    return results


def test_video_diversification():
    """Test video diversification with mock data"""
    print("=" * 60)
    print("VIDEO DIVERSIFICATION TEST")
    print("=" * 60)
    
    # Create test data
    mock_results = create_mock_results(50, 15)
    
    # Create test options
    options = SearchOptions()
    options.enable_video_grouping = True
    options.max_results_per_video = 4
    options.limit = 50
    options.diversity_threshold = 0.3
    
    # Initialize diversifier
    diversifier = VideoLevelDiversifier(debug=True, enable_stats=True)
    
    print(f"\nTest Data: {len(mock_results)} results from {len(set(r.metadata.folder_name for r in mock_results))} videos")
    
    # Analyze original distribution
    original_stats = diversifier.analyze_current_results(mock_results)
    print(f"\nOriginal Distribution:")
    print(f"  Unique videos: {original_stats.unique_videos}")
    print(f"  Max occurrences: {original_stats.max_occurrences} ({original_stats.dominance_ratio:.1%})")
    print(f"  Top video: {original_stats.top_video_name}")
    print(f"  Diversity score: {original_stats.diversity_score:.3f}")
    
    # Test all strategies
    print(f"\nTesting diversification strategies:")
    performance = diversifier.test_strategy_performance(mock_results, options)
    
    for strategy, metrics in performance.items():
        if 'error' in metrics:
            print(f"  {strategy}: ERROR - {metrics['error']}")
        else:
            success_mark = "PASS" if metrics['success'] else "FAIL"
            print(f"  {strategy}: {success_mark}")
            print(f"    Dominance reduction: {metrics['dominance_reduction']:.1%}")
            print(f"    Diversity improvement: {metrics['diversity_improvement']:.3f}")
            print(f"    Final unique videos: {metrics['final_unique_videos']}")
            print(f"    Final max occurrences: {metrics['final_max_occurrences']}")
    
    # Test best strategy in detail
    best_strategy = "balanced"
    diversified_results = diversifier.diversify_results(mock_results, options, best_strategy)
    
    print(f"\nDetailed Results with '{best_strategy}' strategy:")
    final_stats = diversifier.get_improvement_stats()
    
    if final_stats:
        print(f"  Before: {final_stats['before'].unique_videos} videos, max {final_stats['before'].max_occurrences} occurrences")
        print(f"  After: {final_stats['after'].unique_videos} videos, max {final_stats['after'].max_occurrences} occurrences")
        print(f"  Dominance reduction: {final_stats['dominance_reduction']:.1%}")
        print(f"  Diversity improvement: {final_stats['diversity_improvement']:.3f}")
        print(f"  Processing time: {final_stats['processing_time']:.3f}s")
        
        # Success criteria
        target_reduction = 0.5  # 50% dominance reduction
        success = final_stats['dominance_reduction'] >= target_reduction
        print(f"\nTarget Achievement: {'SUCCESS' if success else 'NEEDS IMPROVEMENT'}")
        print(f"   Target: >={target_reduction:.0%} dominance reduction")
        print(f"   Achieved: {final_stats['dominance_reduction']:.1%}")
    
    return diversified_results, final_stats


if __name__ == "__main__":
    # Run comprehensive test
    diversified_results, stats = test_video_diversification()
    
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if stats and stats['dominance_reduction'] >= 0.5:
        print("SUCCESS: Video-level diversification is working!")
        print(f"SUCCESS: Achieved {stats['dominance_reduction']:.1%} dominance reduction")
        print("SUCCESS: Ready for integration with main system")
        
        # Show sample results
        print(f"\nSample diversified results:")
        video_counts = Counter()
        for i, result in enumerate(diversified_results[:10]):
            video_name = result.metadata.folder_name
            video_counts[video_name] += 1
            print(f"  {i+1}. {video_name} (similarity: {result.similarity_score:.3f})")
        
        print(f"\nVideo distribution in top 10:")
        for video, count in video_counts.most_common():
            print(f"  {video}: {count} results")
        
    else:
        print("NEEDS WORK: Video-level diversification needs refinement")
        if stats:
            print(f"NEEDS WORK: Only achieved {stats['dominance_reduction']:.1%} dominance reduction")
        print("NEEDS WORK: Requires algorithm improvement before integration")
    
    print(f"\nNext step: Integrate with EnhancedRetrievalSystem")
