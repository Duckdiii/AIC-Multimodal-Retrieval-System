"""
Batch Query Processor - AIC Competition Automated Processing
===========================================================

Processes multiple queries automatically for AIC competition.
Handles batch processing with v4.0 enhancements and generates submission files.

Key Features:
- Automated batch processing of query files
- v4.0 enhancement selection per query type
- Progress tracking and performance monitoring
- Error handling and recovery
- Competition CSV generation
- Performance optimization for large query sets

Author: AIC Competition System v4.0
Task: 3.2 - Batch Query Processor
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryBatch:
    """Batch of queries to process"""
    batch_id: str
    queries: List[Tuple[str, str]]  # (query_id, query_text)
    query_type: str  # "KIS", "QA", "TRAKE", "MIXED"
    processing_config: Dict[str, Any]
    output_dir: str

@dataclass
class BatchProcessingResult:
    """Result from batch processing"""
    batch_id: str
    processed_queries: int
    successful_queries: int
    failed_queries: int
    total_results: int
    processing_time: float
    csv_files_generated: List[str]
    errors: List[Dict[str, Any]]
    performance_stats: Dict[str, Any]

class BatchQueryProcessor:
    """Main batch processing engine for AIC competition"""
    
    def __init__(self, system=None, max_workers: int = 3):
        self.system = system
        self.max_workers = max_workers
        
        # Processing configurations for different query types
        self.processing_configs = {
            'KIS': {
                'enable_video_grouping': True,
                'max_results_per_video': 4,
                'enable_multi_scene_parsing': True,
                'enable_vietnamese_processing': True,
                'enable_trake_processing': False,
                'query_complexity_threshold': 0.6,
                'limit': 50
            },
            'QA': {
                'enable_video_grouping': True,
                'max_results_per_video': 6,
                'enable_multi_scene_parsing': False,
                'enable_vietnamese_processing': True,
                'enable_trake_processing': False,
                'query_complexity_threshold': 0.4,
                'limit': 30
            },
            'TRAKE': {
                'enable_video_grouping': True,
                'max_results_per_video': 8,
                'enable_multi_scene_parsing': False,
                'enable_vietnamese_processing': True,
                'enable_trake_processing': True,
                'trake_event_precision': 0.9,
                'trake_temporal_window': 30,
                'limit': 100
            }
        }
        
        # Performance tracking
        self.performance_stats = {
            'queries_processed': 0,
            'total_processing_time': 0.0,
            'average_query_time': 0.0,
            'fastest_query_time': float('inf'),
            'slowest_query_time': 0.0,
            'errors_encountered': 0
        }
    
    def initialize_system(self, config_path: Optional[str] = None):
        """Initialize the v4.0 system for batch processing"""
        if self.system is None:
            try:
                from system_v4_integration import EnhancedRetrievalSystemV4
                
                self.system = EnhancedRetrievalSystemV4(
                    config_path=config_path,
                    enable_v4_features=True,
                    debug=False  # Disable debug for batch processing
                )
                
                logger.info("Batch processing system initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize system: {e}")
                return False
        
        return True
    
    def load_queries_from_directory(self, queries_dir: str) -> Dict[str, List[Tuple[str, str]]]:
        """Load queries from directory organized by type"""
        queries_by_type = {
            'KIS': [],
            'QA': [],
            'TRAKE': []
        }
        
        queries_path = Path(queries_dir)
        
        if not queries_path.exists():
            logger.error(f"Queries directory not found: {queries_dir}")
            return queries_by_type
        
        # Load query files
        for query_file in queries_path.glob("*.txt"):
            try:
                with open(query_file, 'r', encoding='utf-8') as f:
                    query_text = f.read().strip()
                
                # Extract query ID and type from filename
                query_id = query_file.stem
                query_type = self._detect_query_type_from_filename(query_file.name, query_text)
                
                queries_by_type[query_type].append((query_id, query_text))
                
                logger.debug(f"Loaded query {query_id} as type {query_type}")
                
            except Exception as e:
                logger.warning(f"Failed to load query from {query_file}: {e}")
        
        # Log statistics
        total_queries = sum(len(queries) for queries in queries_by_type.values())
        logger.info(f"Loaded queries: {total_queries} total")
        for query_type, queries in queries_by_type.items():
            logger.info(f"  {query_type}: {len(queries)} queries")
        
        return queries_by_type
    
    def _detect_query_type_from_filename(self, filename: str, query_text: str) -> str:
        """Detect query type from filename and content"""
        filename_lower = filename.lower()
        
        # Check filename patterns first
        if 'trake' in filename_lower:
            return 'TRAKE'
        elif 'qa' in filename_lower or 'q&a' in filename_lower:
            return 'QA'
        elif 'kis' in filename_lower:
            return 'KIS'
        
        # Check content patterns
        query_lower = query_text.lower()
        
        # TRAKE patterns
        if re.search(r'E\d+:', query_text):
            return 'TRAKE'
        
        # Q&A patterns
        qa_patterns = [r'\?', r'ai\b', r'gì\b', r'đâu\b', r'như thế nào', r'tại sao']
        if any(re.search(pattern, query_lower) for pattern in qa_patterns):
            return 'QA'
        
        # Default to KIS
        return 'KIS'
    
    def create_processing_options(self, query_type: str, custom_config: Optional[Dict] = None):
        """Create processing options for specific query type"""
        try:
            from system_v4_integration import EnhancedSearchOptions
            
            # Get base config for query type
            base_config = self.processing_configs.get(query_type, self.processing_configs['KIS'])
            
            # Merge with custom config if provided
            if custom_config:
                config = {**base_config, **custom_config}
            else:
                config = base_config.copy()
            
            # Create options object
            options = EnhancedSearchOptions(**config)
            return options
            
        except Exception as e:
            logger.error(f"Failed to create processing options: {e}")
            return None
    
    def process_single_query(
        self, 
        query_id: str, 
        query_text: str, 
        query_type: str,
        options = None
    ) -> Tuple[bool, List[Any], float, Dict[str, Any]]:
        """Process a single query and return results"""
        
        start_time = time.time()
        
        try:
            # Create options if not provided
            if options is None:
                options = self.create_processing_options(query_type)
                if options is None:
                    return False, [], 0.0, {'error': 'Failed to create processing options'}
            
            # Execute search
            results = self.system.search(query_text, options)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Gather processing metadata
            metadata = {
                'query_id': query_id,
                'query_type': query_type,
                'results_count': len(results),
                'processing_time': processing_time,
                'v4_enhancements_used': self._extract_enhancements_used(options)
            }
            
            # Update performance stats
            self._update_performance_stats(processing_time, True)
            
            logger.debug(f"Query {query_id} processed: {len(results)} results in {processing_time:.3f}s")
            
            return True, results, processing_time, metadata
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_info = {
                'error': str(e),
                'query_id': query_id,
                'query_type': query_type,
                'processing_time': processing_time
            }
            
            self._update_performance_stats(processing_time, False)
            
            logger.error(f"Query {query_id} failed: {e}")
            return False, [], processing_time, error_info
    
    def _extract_enhancements_used(self, options) -> List[str]:
        """Extract which v4.0 enhancements were used"""
        enhancements = []
        
        if hasattr(options, 'enable_video_grouping') and options.enable_video_grouping:
            enhancements.append('video_grouping')
        
        if hasattr(options, 'enable_multi_scene_parsing') and options.enable_multi_scene_parsing:
            enhancements.append('multi_scene_parsing')
        
        if hasattr(options, 'enable_vietnamese_processing') and options.enable_vietnamese_processing:
            enhancements.append('vietnamese_processing')
        
        if hasattr(options, 'enable_trake_processing') and options.enable_trake_processing:
            enhancements.append('trake_processing')
        
        return enhancements
    
    def _update_performance_stats(self, processing_time: float, success: bool):
        """Update performance statistics"""
        self.performance_stats['queries_processed'] += 1
        self.performance_stats['total_processing_time'] += processing_time
        
        if success:
            self.performance_stats['fastest_query_time'] = min(
                self.performance_stats['fastest_query_time'], processing_time
            )
            self.performance_stats['slowest_query_time'] = max(
                self.performance_stats['slowest_query_time'], processing_time
            )
        else:
            self.performance_stats['errors_encountered'] += 1
        
        # Update average
        if self.performance_stats['queries_processed'] > 0:
            self.performance_stats['average_query_time'] = (
                self.performance_stats['total_processing_time'] / 
                self.performance_stats['queries_processed']
            )
    
    def process_batch(
        self, 
        query_batch: QueryBatch,
        parallel_processing: bool = True
    ) -> BatchProcessingResult:
        """Process a batch of queries"""
        
        logger.info(f"Starting batch processing: {query_batch.batch_id}")
        logger.info(f"Queries to process: {len(query_batch.queries)}")
        
        batch_start_time = time.time()
        
        # Results storage
        all_results = []
        successful_queries = 0
        failed_queries = 0
        errors = []
        query_metadata = []
        
        # Create processing options
        options = self.create_processing_options(
            query_batch.query_type, 
            query_batch.processing_config
        )
        
        if parallel_processing and len(query_batch.queries) > 1:
            # Parallel processing
            results = self._process_batch_parallel(query_batch, options)
        else:
            # Sequential processing
            results = self._process_batch_sequential(query_batch, options)
        
        # Process results
        for success, query_results, processing_time, metadata in results:
            if success:
                successful_queries += 1
                all_results.extend(query_results)
                query_metadata.append(metadata)
            else:
                failed_queries += 1
                errors.append(metadata)
        
        # Generate CSV files
        csv_files = self._generate_csv_files(query_batch, all_results, query_metadata)
        
        batch_processing_time = time.time() - batch_start_time
        
        # Create result object
        result = BatchProcessingResult(
            batch_id=query_batch.batch_id,
            processed_queries=len(query_batch.queries),
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            total_results=len(all_results),
            processing_time=batch_processing_time,
            csv_files_generated=csv_files,
            errors=errors,
            performance_stats=self.performance_stats.copy()
        )
        
        logger.info(f"Batch processing completed: {query_batch.batch_id}")
        logger.info(f"Success rate: {successful_queries}/{len(query_batch.queries)} ({successful_queries/len(query_batch.queries):.1%})")
        logger.info(f"Total processing time: {batch_processing_time:.2f}s")
        
        return result
    
    def _process_batch_parallel(self, query_batch: QueryBatch, options) -> List[Tuple]:
        """Process queries in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all queries
            futures = []
            for query_id, query_text in query_batch.queries:
                future = executor.submit(
                    self.process_single_query, 
                    query_id, query_text, query_batch.query_type, options
                )
                futures.append(future)
            
            # Collect results
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per query
                    results.append(result)
                    
                    # Progress logging
                    if (i + 1) % 5 == 0:
                        logger.info(f"Processed {i + 1}/{len(query_batch.queries)} queries")
                        
                except Exception as e:
                    logger.error(f"Query processing failed: {e}")
                    query_id = query_batch.queries[i][0] if i < len(query_batch.queries) else f"unknown_{i}"
                    results.append((False, [], 0.0, {'error': str(e), 'query_id': query_id}))
        
        return results
    
    def _process_batch_sequential(self, query_batch: QueryBatch, options) -> List[Tuple]:
        """Process queries sequentially"""
        results = []
        
        for i, (query_id, query_text) in enumerate(query_batch.queries):
            result = self.process_single_query(query_id, query_text, query_batch.query_type, options)
            results.append(result)
            
            # Progress logging
            if (i + 1) % 5 == 0:
                logger.info(f"Processed {i + 1}/{len(query_batch.queries)} queries")
        
        return results
    
    def _generate_csv_files(self, query_batch: QueryBatch, all_results: List[Any], 
                           query_metadata: List[Dict]) -> List[str]:
        """Generate CSV files for batch results"""
        try:
            from competition_csv_generator import CompetitionCSVGenerator, create_competition_query_from_search
            
            generator = CompetitionCSVGenerator(query_batch.output_dir)
            competition_queries = []
            csv_files = []
            
            # Group results by query
            results_by_query = {}
            for metadata in query_metadata:
                query_id = metadata['query_id']
                if query_id not in results_by_query:
                    results_by_query[query_id] = {
                        'metadata': metadata,
                        'results': []
                    }
            
            # This is a simplified approach - in practice, you'd need to track
            # which results belong to which query during processing
            
            # Generate batch CSV with all results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_csv_filename = f"batch_{query_batch.batch_id}_{timestamp}.csv"
            
            # For now, create a simple CSV with all results
            csv_path = Path(query_batch.output_dir) / batch_csv_filename
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                import csv
                writer = csv.writer(csvfile)
                
                for result in all_results[:1000]:  # Limit results
                    try:
                        if hasattr(result, 'metadata'):
                            video_name = getattr(result.metadata, 'folder_name', 'unknown')
                            frame_id = getattr(result.metadata, 'image_name', '000000.jpg').split('.')[0]
                        else:
                            video_name = 'unknown'
                            frame_id = '000000'
                        
                        writer.writerow([video_name, frame_id])
                    except Exception as e:
                        logger.warning(f"Failed to write result to CSV: {e}")
            
            csv_files.append(str(csv_path))
            logger.info(f"Generated batch CSV: {csv_path}")
            
            return csv_files
            
        except Exception as e:
            logger.error(f"Failed to generate CSV files: {e}")
            return []
    
    def process_competition_queries(
        self, 
        queries_dir: str, 
        output_dir: str = "competition_results",
        parallel_processing: bool = True
    ) -> Dict[str, BatchProcessingResult]:
        """Process all competition queries by type"""
        
        logger.info("Starting competition query processing")
        
        # Ensure output directory exists
        Path(output_dir).mkdir(exist_ok=True)
        
        # Load queries by type
        queries_by_type = self.load_queries_from_directory(queries_dir)
        
        # Process each query type
        batch_results = {}
        
        for query_type, queries in queries_by_type.items():
            if not queries:
                logger.info(f"No {query_type} queries found, skipping")
                continue
            
            logger.info(f"Processing {len(queries)} {query_type} queries")
            
            # Create batch
            batch = QueryBatch(
                batch_id=f"{query_type.lower()}_batch",
                queries=queries,
                query_type=query_type,
                processing_config={},  # Use default configs
                output_dir=output_dir
            )
            
            # Process batch
            result = self.process_batch(batch, parallel_processing)
            batch_results[query_type] = result
        
        # Generate overall summary
        self._generate_processing_summary(batch_results, output_dir)
        
        logger.info("Competition query processing completed")
        return batch_results
    
    def _generate_processing_summary(
        self, 
        batch_results: Dict[str, BatchProcessingResult], 
        output_dir: str
    ):
        """Generate processing summary report"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'overall_stats': {
                'total_queries': sum(r.processed_queries for r in batch_results.values()),
                'successful_queries': sum(r.successful_queries for r in batch_results.values()),
                'failed_queries': sum(r.failed_queries for r in batch_results.values()),
                'total_results': sum(r.total_results for r in batch_results.values()),
                'total_processing_time': sum(r.processing_time for r in batch_results.values())
            },
            'batch_results': {
                query_type: {
                    'processed': result.processed_queries,
                    'successful': result.successful_queries,
                    'failed': result.failed_queries,
                    'results': result.total_results,
                    'processing_time': result.processing_time,
                    'csv_files': result.csv_files_generated
                }
                for query_type, result in batch_results.items()
            },
            'performance_stats': self.performance_stats
        }
        
        summary_path = Path(output_dir) / "processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processing summary saved: {summary_path}")

# Testing and utility functions
def test_batch_processor():
    """Test batch processor with mock system"""
    
    # Create mock queries
    mock_queries_by_type = {
        'KIS': [
            ('KIS001', 'Tìm đoạn video nấu ăn với nấm'),
            ('KIS002', 'Video có cảnh cắt rau củ')
        ],
        'QA': [
            ('QA001', 'Địa điểm nào ở Khánh Hòa?')
        ],
        'TRAKE': [
            ('TRAKE001', 'E1: Cut mushroom. E2: Add to pan.')
        ]
    }
    
    # Create processor (without actual system for testing)
    processor = BatchQueryProcessor(max_workers=2)
    
    print("Testing Batch Query Processor")
    print("=" * 50)
    
    for query_type, queries in mock_queries_by_type.items():
        print(f"\nTesting {query_type} queries:")
        
        # Create batch
        batch = QueryBatch(
            batch_id=f"test_{query_type.lower()}",
            queries=queries,
            query_type=query_type,
            processing_config={},
            output_dir="test_batch_output"
        )
        
        print(f"  Batch created with {len(queries)} queries")
        print(f"  Query type: {query_type}")
        
        # Test configuration creation
        options = processor.create_processing_options(query_type)
        if options:
            print(f"  Processing options created successfully")
            enhancements = processor._extract_enhancements_used(options)
            print(f"  Enhancements: {enhancements}")
        else:
            print(f"  Failed to create processing options")

if __name__ == "__main__":
    import re
    test_batch_processor()