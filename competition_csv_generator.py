"""
Competition CSV Generator - AIC Competition Submission Format
============================================================

Generates competition-compliant CSV files for AIC Video Retrieval Competition.
Handles KIS, Q&A, and TRAKE query types with proper formatting and validation.

Key Features:
- Competition-compliant CSV generation
- UTF-8 encoding with proper format
- Batch processing for multiple queries
- Query type specific handling (KIS, Q&A, TRAKE)
- Validation and error checking
- Submission package generation

Author: AIC Competition System v4.0
Task: 3.1 - Competition CSV Generator
"""

import os
import csv
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    KIS = "KIS"  # Keyframe-based Image Search
    QA = "Q&A"   # Question & Answer
    TRAKE = "TRAKE"  # Temporal Retrieval and Alignment of Key Events

@dataclass
class CompetitionResult:
    """Single competition result entry"""
    query_id: str
    query_type: QueryType
    video_name: str
    frame_id: str
    rank: int
    confidence_score: float
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class CompetitionQuery:
    """Competition query with results"""
    query_id: str
    query_text: str
    query_type: QueryType
    results: List[CompetitionResult]
    processing_time: float
    v4_enhancements_used: List[str]

class CompetitionCSVGenerator:
    """Main CSV generator for AIC competition submissions"""
    
    def __init__(self, output_dir: str = "competition_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Competition format specifications
        self.csv_format = {
            'encoding': 'utf-8',
            'delimiter': ',',
            'quotechar': '"',
            'quoting': csv.QUOTE_MINIMAL,
            'lineterminator': '\n'
        }
        
        # Maximum results per query (competition limit)
        self.max_results_per_query = 1000
        
        # Query type patterns for auto-detection
        self.query_type_patterns = {
            QueryType.TRAKE: [r'E\d+:', r'khoảnh khắc.*đầu tiên', r'khoảnh khắc.*cuối cùng'],
            QueryType.QA: [r'\?', r'ai\b', r'gì\b', r'đâu\b', r'như thế nào', r'tại sao'],
            QueryType.KIS: [r'tìm.*video', r'đoạn video', r'cảnh.*trong']
        }
        
    def detect_query_type(self, query_text: str) -> QueryType:
        """Auto-detect query type from text"""
        query_lower = query_text.lower()
        
        # Check for TRAKE format first (most specific)
        for pattern in self.query_type_patterns[QueryType.TRAKE]:
            if re.search(pattern, query_lower):
                return QueryType.TRAKE
        
        # Check for Q&A patterns
        for pattern in self.query_type_patterns[QueryType.QA]:
            if re.search(pattern, query_lower):
                return QueryType.QA
        
        # Default to KIS
        return QueryType.KIS
    
    def convert_search_results_to_competition_format(
        self, 
        search_results: List[Any], 
        query_id: str, 
        query_type: QueryType
    ) -> List[CompetitionResult]:
        """Convert system search results to competition format"""
        competition_results = []
        
        for rank, result in enumerate(search_results[:self.max_results_per_query], 1):
            try:
                # Extract video name and frame ID from metadata
                video_name = self._extract_video_name(result)
                frame_id = self._extract_frame_id(result)
                
                # Get confidence score
                confidence = self._extract_confidence(result)
                
                # Create competition result
                comp_result = CompetitionResult(
                    query_id=query_id,
                    query_type=query_type,
                    video_name=video_name,
                    frame_id=frame_id,
                    rank=rank,
                    confidence_score=confidence,
                    metadata=self._extract_additional_metadata(result)
                )
                
                competition_results.append(comp_result)
                
            except Exception as e:
                logger.error(f"Failed to convert result {rank} for query {query_id}: {e}")
                continue
        
        return competition_results
    
    def _extract_video_name(self, result: Any) -> str:
        """Extract video name from search result"""
        if hasattr(result, 'metadata'):
            if hasattr(result.metadata, 'folder_name'):
                return result.metadata.folder_name
            elif hasattr(result.metadata, 'video_name'):
                return result.metadata.video_name
        
        # Fallback extraction
        if hasattr(result, 'folder_name'):
            return result.folder_name
        elif hasattr(result, 'video_name'):
            return result.video_name
        
        return "unknown_video"
    
    def _extract_frame_id(self, result: Any) -> str:
        """Extract the real video frame index from search result metadata."""
        if hasattr(result, 'metadata'):
            frame_id = getattr(result.metadata, 'frame_id', None)
            if self._is_valid_frame_id(frame_id):
                return str(int(frame_id))

            video_name = getattr(result.metadata, 'folder_name', 'unknown_video')
            image_name = getattr(result.metadata, 'image_name', 'unknown_keyframe')
            raise ValueError(
                "MAPPING_ERROR: missing real frame_id for competition export; "
                f"video_id={video_name}; image_name={image_name}; frame_id={frame_id}"
            )
        
        frame_id = getattr(result, 'frame_id', None)
        if self._is_valid_frame_id(frame_id):
            return str(int(frame_id))

        video_name = getattr(result, 'folder_name', getattr(result, 'video_name', 'unknown_video'))
        image_name = getattr(result, 'image_name', 'unknown_keyframe')
        raise ValueError(
            "MAPPING_ERROR: missing real frame_id for competition export; "
            f"video_id={video_name}; image_name={image_name}; frame_id={frame_id}"
        )

    def _is_valid_frame_id(self, frame_id: Any) -> bool:
        """Return True only for non-negative integer frame ids."""
        if frame_id is None:
            return False
        try:
            return int(frame_id) >= 0
        except (TypeError, ValueError):
            return False
    
    def _extract_confidence(self, result: Any) -> float:
        """Extract confidence score from search result"""
        if hasattr(result, 'similarity_score'):
            return float(result.similarity_score)
        elif hasattr(result, 'confidence'):
            return float(result.confidence)
        elif hasattr(result, 'score'):
            return float(result.score)
        
        return 0.5  # Default confidence
    
    def _extract_additional_metadata(self, result: Any) -> Dict[str, Any]:
        """Extract additional metadata for analysis"""
        metadata = {}
        
        # v4.0 enhancements metadata
        if hasattr(result, 'trake_event_id'):
            metadata['trake_event_id'] = result.trake_event_id
            metadata['trake_domain'] = getattr(result, 'trake_domain', 'unknown')
            metadata['trake_event_type'] = getattr(result, 'trake_event_type', 'unknown')
        
        if hasattr(result, 'scene_id'):
            metadata['scene_id'] = result.scene_id
            metadata['scene_context'] = getattr(result, 'scene_context', '')
        
        if hasattr(result, 'vietnamese_processing'):
            metadata['vietnamese_processing'] = result.vietnamese_processing
        
        return metadata
    
    def generate_csv_file(
        self, 
        competition_query: CompetitionQuery, 
        filename: Optional[str] = None
    ) -> str:
        """Generate CSV file for a single query"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{competition_query.query_id}_{competition_query.query_type.value}_{timestamp}.csv"
        
        csv_path = self.output_dir / filename
        
        # Write CSV file
        with open(csv_path, 'w', newline='', encoding=self.csv_format['encoding']) as csvfile:
            writer = csv.writer(
                csvfile,
                delimiter=self.csv_format['delimiter'],
                quotechar=self.csv_format['quotechar'],
                quoting=self.csv_format['quoting']
            )
            
            # Write header (optional - check competition requirements)
            # writer.writerow(['video_name', 'frame_id'])
            
            # Write results
            for result in competition_query.results:
                row = [result.video_name, result.frame_id]
                writer.writerow(row)
        
        logger.info(f"Generated CSV file: {csv_path} with {len(competition_query.results)} results")
        return str(csv_path)
    
    def generate_batch_csv(
        self, 
        competition_queries: List[CompetitionQuery],
        output_filename: str = "batch_submission.csv"
    ) -> str:
        """Generate single CSV file for multiple queries"""
        
        csv_path = self.output_dir / output_filename
        
        with open(csv_path, 'w', newline='', encoding=self.csv_format['encoding']) as csvfile:
            writer = csv.writer(
                csvfile,
                delimiter=self.csv_format['delimiter'],
                quotechar=self.csv_format['quotechar'],
                quoting=self.csv_format['quoting']
            )
            
            # Write all results from all queries
            total_results = 0
            for query in competition_queries:
                for result in query.results:
                    row = [result.video_name, result.frame_id]
                    writer.writerow(row)
                    total_results += 1
        
        logger.info(f"Generated batch CSV: {csv_path} with {total_results} total results from {len(competition_queries)} queries")
        return str(csv_path)
    
    def validate_csv_format(self, csv_path: str) -> Dict[str, Any]:
        """Validate CSV file format for competition compliance"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                # Check encoding
                content = csvfile.read()
                
                # Reset file pointer
                csvfile.seek(0)
                
                # Parse CSV
                reader = csv.reader(csvfile)
                rows = list(reader)
                
                # Validation checks
                validation_result['stats']['total_rows'] = len(rows)
                
                if len(rows) == 0:
                    validation_result['valid'] = False
                    validation_result['errors'].append("CSV file is empty")
                    return validation_result
                
                # Check row format
                expected_columns = 2  # video_name, frame_id
                invalid_rows = []
                
                for i, row in enumerate(rows):
                    if len(row) != expected_columns:
                        invalid_rows.append(i + 1)
                
                if invalid_rows:
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"Invalid row format at lines: {invalid_rows[:10]}")
                
                # Check for empty values
                empty_values = []
                for i, row in enumerate(rows):
                    if len(row) >= 2 and (not row[0].strip() or not row[1].strip()):
                        empty_values.append(i + 1)
                
                if empty_values:
                    validation_result['warnings'].append(f"Empty values found at lines: {empty_values[:10]}")
                
                # Statistics
                validation_result['stats']['unique_videos'] = len(set(row[0] for row in rows if len(row) >= 1))
                validation_result['stats']['unique_frames'] = len(set(f"{row[0]}_{row[1]}" for row in rows if len(row) >= 2))
                
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Failed to validate CSV: {e}")
        
        return validation_result
    
    def generate_submission_package(
        self, 
        competition_queries: List[CompetitionQuery],
        package_name: str = "aic_submission"
    ) -> str:
        """Generate complete submission package"""
        
        # Create submission directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_dir = self.output_dir / f"{package_name}_{timestamp}"
        package_dir.mkdir(exist_ok=True)
        
        # Generate individual CSV files for each query
        csv_files = []
        for query in competition_queries:
            filename = f"{query.query_id}_{query.query_type.value}.csv"
            csv_path = self.generate_csv_file(query, filename)
            # Move to package directory
            package_csv_path = package_dir / filename
            os.rename(csv_path, package_csv_path)
            csv_files.append(package_csv_path)
        
        # Generate batch CSV
        batch_csv_path = package_dir / "batch_submission.csv"
        self.generate_batch_csv(competition_queries, batch_csv_path.name)
        os.rename(self.output_dir / "batch_submission.csv", batch_csv_path)
        
        # Generate submission metadata
        metadata = {
            'submission_info': {
                'timestamp': timestamp,
                'total_queries': len(competition_queries),
                'query_types': list(set(q.query_type.value for q in competition_queries)),
                'total_results': sum(len(q.results) for q in competition_queries),
                'system_version': 'v4.0',
                'enhancements_used': list(set(
                    enhancement 
                    for query in competition_queries 
                    for enhancement in query.v4_enhancements_used
                ))
            },
            'files': {
                'individual_csv_files': [f.name for f in csv_files],
                'batch_csv_file': batch_csv_path.name,
                'validation_report': 'validation_report.json'
            },
            'statistics': self._generate_submission_statistics(competition_queries)
        }
        
        # Save metadata
        metadata_path = package_dir / "submission_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Generate validation report
        validation_report = {}
        for csv_file in csv_files + [batch_csv_path]:
            validation_report[csv_file.name] = self.validate_csv_format(str(csv_file))
        
        validation_path = package_dir / "validation_report.json"
        with open(validation_path, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Generated submission package: {package_dir}")
        return str(package_dir)
    
    def _generate_submission_statistics(
        self, 
        competition_queries: List[CompetitionQuery]
    ) -> Dict[str, Any]:
        """Generate statistics for submission"""
        
        stats = {
            'query_type_distribution': {},
            'processing_time_stats': {},
            'result_count_stats': {},
            'enhancement_usage': {}
        }
        
        # Query type distribution
        for query_type in QueryType:
            count = sum(1 for q in competition_queries if q.query_type == query_type)
            stats['query_type_distribution'][query_type.value] = count
        
        # Processing time statistics
        processing_times = [q.processing_time for q in competition_queries]
        if processing_times:
            stats['processing_time_stats'] = {
                'average': sum(processing_times) / len(processing_times),
                'min': min(processing_times),
                'max': max(processing_times),
                'total': sum(processing_times)
            }
        
        # Result count statistics
        result_counts = [len(q.results) for q in competition_queries]
        if result_counts:
            stats['result_count_stats'] = {
                'average': sum(result_counts) / len(result_counts),
                'min': min(result_counts),
                'max': max(result_counts),
                'total': sum(result_counts)
            }
        
        # Enhancement usage
        all_enhancements = []
        for query in competition_queries:
            all_enhancements.extend(query.v4_enhancements_used)
        
        from collections import Counter
        enhancement_counts = Counter(all_enhancements)
        stats['enhancement_usage'] = dict(enhancement_counts)
        
        return stats

# Utility functions for integration
def create_competition_query_from_search(
    query_id: str,
    query_text: str,
    search_results: List[Any],
    processing_time: float = 0.0,
    v4_enhancements: List[str] = None
) -> CompetitionQuery:
    """Create CompetitionQuery from search results"""
    
    generator = CompetitionCSVGenerator()
    query_type = generator.detect_query_type(query_text)
    
    competition_results = generator.convert_search_results_to_competition_format(
        search_results, query_id, query_type
    )
    
    return CompetitionQuery(
        query_id=query_id,
        query_text=query_text,
        query_type=query_type,
        results=competition_results,
        processing_time=processing_time,
        v4_enhancements_used=v4_enhancements or []
    )

def generate_competition_csv(
    query_id: str,
    query_text: str,
    search_results: List[Any],
    output_dir: str = "competition_output"
) -> str:
    """Convenience function to generate CSV from search results"""
    
    generator = CompetitionCSVGenerator(output_dir)
    competition_query = create_competition_query_from_search(
        query_id, query_text, search_results
    )
    
    return generator.generate_csv_file(competition_query)

# Testing function
def test_csv_generator():
    """Test CSV generator with mock data"""
    
    # Create mock search results
    mock_results = []
    for i in range(10):
        mock_result = type('MockResult', (), {
            'metadata': type('MockMetadata', (), {
                'folder_name': f'video_{i % 3:03d}',
                'image_name': f'frame_{i:06d}.jpg'
            })(),
            'similarity_score': 0.9 - i * 0.05
        })()
        mock_results.append(mock_result)
    
    # Test different query types
    test_queries = [
        ("Q001", "Tìm đoạn video nấu ăn với nấm", "KIS"),
        ("Q002", "Địa điểm nào ở Khánh Hòa?", "Q&A"),
        ("Q003", "E1: Cut mushroom. E2: Add to pan.", "TRAKE")
    ]
    
    generator = CompetitionCSVGenerator("test_output")
    competition_queries = []
    
    for query_id, query_text, expected_type in test_queries:
        print(f"\nTesting {query_id} - {expected_type}")
        
        # Create competition query
        competition_query = create_competition_query_from_search(
            query_id, query_text, mock_results, processing_time=1.5,
            v4_enhancements=['video_grouping', 'multi_scene_parsing']
        )
        
        print(f"  Detected type: {competition_query.query_type.value}")
        print(f"  Results: {len(competition_query.results)}")
        
        # Generate CSV
        csv_path = generator.generate_csv_file(competition_query)
        print(f"  CSV generated: {csv_path}")
        
        # Validate CSV
        validation = generator.validate_csv_format(csv_path)
        print(f"  Validation: {'PASS' if validation['valid'] else 'FAIL'}")
        if validation['errors']:
            print(f"  Errors: {validation['errors']}")
        
        competition_queries.append(competition_query)
    
    # Test batch generation
    print(f"\nTesting batch submission generation...")
    package_path = generator.generate_submission_package(
        competition_queries, "test_submission"
    )
    print(f"Submission package: {package_path}")

if __name__ == "__main__":
    import re
    test_csv_generator()
