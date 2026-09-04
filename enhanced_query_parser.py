"""
Enhanced Query Parser for AIC Competition System v4.0
======================================================

Multi-scene query parser for complex Vietnamese KIS queries.
Handles temporal sequences, scene decomposition, and contextual understanding.

Key Features:
- LLM-based scene decomposition 
- Vietnamese temporal marker detection
- Multi-scene query breaking
- Context preservation across scenes

Author: AIC Competition System v4.0
Task: 1.2 - LLM Integration for Scene Decomposition
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    SIMPLE = "simple"
    MULTI_SCENE = "multi_scene"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    COMPLEX_NARRATIVE = "complex_narrative"

@dataclass
class SceneComponent:
    """Individual scene within a complex query"""
    scene_id: int
    description: str
    temporal_markers: List[str]
    key_objects: List[str]
    actions: List[str]
    spatial_context: str
    temporal_order: int
    confidence: float

@dataclass
class ParsedQuery:
    """Complete parsed query structure"""
    original_query: str
    query_type: QueryType
    language: str
    scenes: List[SceneComponent]
    temporal_flow: List[str]
    main_subject: str
    overall_context: str
    complexity_score: float
    parsing_confidence: float

class VietnameseTemporalParser:
    """Vietnamese temporal marker detection and processing"""
    
    def __init__(self):
        # Vietnamese temporal markers
        self.temporal_markers = {
            'start': ['bắt đầu', 'khởi đầu', 'lúc đầu', 'ban đầu', 'trước tiên', 'đầu tiên'],
            'continuation': ['sau đó', 'tiếp theo', 'kế tiếp', 'rồi', 'thì', 'và', 'tiếp theo đó'],
            'sequence': ['bước đầu tiên', 'bước tiếp theo', 'bước cuối', 'lần lượt', 'theo thứ tự'],
            'transition': ['trong khi đó', 'cùng lúc', 'đồng thời', 'trong lúc', 'trong cảnh'],
            'conclusion': ['cuối cùng', 'kết thúc', 'hoàn thành', 'hoàn tất', 'đến đích', 'kết quả']
        }
        
        # Scene transition patterns
        self.scene_patterns = [
            r'phân cảnh (\w+)',  # "phân cảnh bắt đầu"
            r'cảnh (\w+)',       # "cảnh tiếp theo"
            r'trong cảnh',       # "trong cảnh có"
            r'phân cảnh tiếp theo',
            r'bước (\w+)',       # "bước đầu tiên"
        ]
        
    def detect_temporal_markers(self, text: str) -> List[Tuple[str, str, int]]:
        """Detect temporal markers in Vietnamese text
        
        Returns:
            List of (marker_text, marker_type, position)
        """
        markers = []
        text_lower = text.lower()
        
        for marker_type, marker_list in self.temporal_markers.items():
            for marker in marker_list:
                pattern = r'\b' + re.escape(marker) + r'\b'
                for match in re.finditer(pattern, text_lower):
                    markers.append((marker, marker_type, match.start()))
        
        # Sort by position in text
        return sorted(markers, key=lambda x: x[2])
    
    def extract_scene_boundaries(self, text: str) -> List[Tuple[int, int, str]]:
        """Extract scene boundaries using Vietnamese patterns
        
        Returns:
            List of (start_pos, end_pos, scene_type)
        """
        boundaries = []
        
        for pattern in self.scene_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                boundaries.append((match.start(), match.end(), match.group()))
        
        # Also look for sentence boundaries as potential scene breaks
        sentences = re.split(r'[.!?]\s+', text)
        pos = 0
        for i, sentence in enumerate(sentences):
            if any(marker in sentence.lower() for marker_list in self.temporal_markers.values() 
                   for marker in marker_list):
                boundaries.append((pos, pos + len(sentence), f'sentence_{i}'))
            pos += len(sentence) + 2  # +2 for punctuation and space
        
        return sorted(boundaries, key=lambda x: x[0])

class LLMQueryDecomposer:
    """LLM-based query decomposition using OpenAI API"""
    
    def __init__(self):
        self.client = None
        try:
            import openai
            # Try to initialize OpenAI client
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
            else:
                logger.warning("OpenAI API key not found. LLM decomposition will be limited.")
        except ImportError:
            logger.warning("OpenAI package not installed. Using rule-based parsing only.")
    
    def decompose_query(self, query: str) -> Dict[str, Any]:
        """Decompose complex query into scenes using LLM"""
        
        if not self.client:
            return self._fallback_decomposition(query)
        
        try:
            prompt = f"""
Analyze this Vietnamese video search query and break it down into distinct scenes or components.

Query: "{query}"

Please provide a JSON response with:
1. query_type: "simple", "multi_scene", "temporal_sequence", or "complex_narrative"
2. scenes: array of scene objects with:
   - scene_id: number
   - description: clear description of the scene
   - key_objects: array of important objects/people
   - actions: array of main actions
   - temporal_order: sequence number (if applicable)
3. temporal_flow: array of temporal transitions
4. main_subject: the primary focus
5. complexity_score: 0.0-1.0

Focus on identifying:
- Scene transitions (phân cảnh, cảnh tiếp theo)
- Temporal markers (bắt đầu, sau đó, cuối cùng)
- Object descriptions and actions
- Spatial and temporal relationships

Respond only with valid JSON.
"""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a Vietnamese video analysis expert. Analyze queries for multi-scene video search."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"LLM decomposition failed: {e}")
            return self._fallback_decomposition(query)
    
    def _fallback_decomposition(self, query: str) -> Dict[str, Any]:
        """Rule-based fallback decomposition when LLM is unavailable"""
        
        # Simple rule-based parsing
        scenes = []
        
        # Look for scene indicators
        scene_indicators = ['phân cảnh', 'cảnh', 'bước', 'sau đó', 'tiếp theo']
        
        if any(indicator in query.lower() for indicator in scene_indicators):
            # Split by common Vietnamese scene transitions
            parts = re.split(r'(?:phân cảnh|cảnh|sau đó|tiếp theo|bước)', query, flags=re.IGNORECASE)
            
            for i, part in enumerate(parts):
                if part.strip():
                    scenes.append({
                        'scene_id': i,
                        'description': part.strip(),
                        'key_objects': self._extract_objects(part),
                        'actions': self._extract_actions(part),
                        'temporal_order': i
                    })
            
            query_type = "multi_scene" if len(scenes) > 1 else "simple"
        else:
            # Single scene
            scenes = [{
                'scene_id': 0,
                'description': query,
                'key_objects': self._extract_objects(query),
                'actions': self._extract_actions(query),
                'temporal_order': 0
            }]
            query_type = "simple"
        
        return {
            'query_type': query_type,
            'scenes': scenes,
            'temporal_flow': ['sequential'] if len(scenes) > 1 else [],
            'main_subject': self._extract_main_subject(query),
            'complexity_score': min(0.8, len(scenes) * 0.3)
        }
    
    def _extract_objects(self, text: str) -> List[str]:
        """Extract key objects from Vietnamese text"""
        # Common object patterns in Vietnamese
        object_patterns = [
            r'một (\w+)',      # "một chiếc đĩa"
            r'chiếc (\w+)',    # "chiếc bánh"
            r'(\w+) màu (\w+)', # "áo màu xanh"
            r'vận động viên',
            r'đầu bếp',
            r'người (\w+)',
        ]
        
        objects = []
        for pattern in object_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Handle both string and tuple matches
                for match in matches:
                    if isinstance(match, tuple):
                        objects.extend(match)
                    else:
                        objects.append(match)
        
        return list(set(objects))[:10]  # Limit to 10 objects
    
    def _extract_actions(self, text: str) -> List[str]:
        """Extract key actions from Vietnamese text"""
        # Common action patterns
        action_patterns = [
            r'đang (\w+)',     # "đang vượt"
            r'bắt đầu (\w+)',  # "bắt đầu trang trí"
            r'(\w+) lên',      # "đặt lên"
            r'(\w+ing)',       # English verbs
        ]
        
        actions = []
        for pattern in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            actions.extend(matches)
        
        return list(set(actions))[:10]  # Limit to 10 actions
    
    def _extract_main_subject(self, text: str) -> str:
        """Extract main subject of the query"""
        # Look for common subjects
        subjects = ['vận động viên', 'đầu bếp', 'người', 'bánh', 'xe', 'video']
        
        for subject in subjects:
            if subject in text.lower():
                return subject
        
        # Return first noun-like word
        words = text.split()
        for word in words:
            if len(word) > 3:
                return word
        
        return "unknown"

class EnhancedQueryParser:
    """Main query parser combining temporal analysis and LLM decomposition"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.temporal_parser = VietnameseTemporalParser()
        self.llm_decomposer = LLMQueryDecomposer()
        
        # Load configuration if provided
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
    
    def parse_query(self, query: str, query_id: Optional[str] = None) -> ParsedQuery:
        """Parse a query into structured components
        
        Args:
            query: The Vietnamese query text
            query_id: Optional identifier for the query
            
        Returns:
            ParsedQuery object with structured information
        """
        logger.info(f"Parsing query: {query[:100]}...")
        
        # Step 1: Detect temporal markers
        temporal_markers = self.temporal_parser.detect_temporal_markers(query)
        logger.debug(f"Found temporal markers: {temporal_markers}")
        
        # Step 2: Extract scene boundaries
        scene_boundaries = self.temporal_parser.extract_scene_boundaries(query)
        logger.debug(f"Found scene boundaries: {scene_boundaries}")
        
        # Step 3: LLM-based decomposition
        llm_result = self.llm_decomposer.decompose_query(query)
        logger.debug(f"LLM decomposition: {llm_result}")
        
        # Step 4: Combine results
        scenes = []
        for i, scene_data in enumerate(llm_result.get('scenes', [])):
            scene = SceneComponent(
                scene_id=scene_data.get('scene_id', i),
                description=scene_data.get('description', ''),
                temporal_markers=[marker[0] for marker in temporal_markers],
                key_objects=scene_data.get('key_objects', []),
                actions=scene_data.get('actions', []),
                spatial_context=self._extract_spatial_context(scene_data.get('description', '')),
                temporal_order=scene_data.get('temporal_order', i),
                confidence=0.8  # Default confidence
            )
            scenes.append(scene)
        
        # Determine query type
        query_type_str = llm_result.get('query_type', 'simple')
        try:
            query_type = QueryType(query_type_str)
        except ValueError:
            query_type = QueryType.SIMPLE
        
        # Create parsed query object
        parsed_query = ParsedQuery(
            original_query=query,
            query_type=query_type,
            language='vietnamese',
            scenes=scenes,
            temporal_flow=llm_result.get('temporal_flow', []),
            main_subject=llm_result.get('main_subject', 'unknown'),
            overall_context=self._extract_overall_context(query),
            complexity_score=llm_result.get('complexity_score', 0.5),
            parsing_confidence=0.85  # Default confidence
        )
        
        logger.info(f"Parsed query into {len(scenes)} scenes, type: {query_type.value}, complexity: {parsed_query.complexity_score:.2f}")
        return parsed_query
    
    def _extract_spatial_context(self, text: str) -> str:
        """Extract spatial context from scene description"""
        spatial_indicators = ['trên cao', 'bên cạnh', 'phía sau', 'ở giữa', 'hai bên', 'xung quanh']
        
        for indicator in spatial_indicators:
            if indicator in text.lower():
                return indicator
        
        return "unknown"
    
    def _extract_overall_context(self, query: str) -> str:
        """Extract overall context of the query"""
        context_keywords = {
            'cooking': ['bánh', 'đầu bếp', 'trang trí', 'chocolate', 'chuối'],
            'sports': ['xe đạp', 'vận động viên', 'đua', 'vượt', 'dẫn đầu'],
            'charity': ['từ thiện', 'bệnh viện', 'trao quà', 'trẻ em'],
            'general': []
        }
        
        query_lower = query.lower()
        for context, keywords in context_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return context
        
        return "general"
    
    def batch_parse_queries(self, queries: List[str]) -> List[ParsedQuery]:
        """Parse multiple queries in batch"""
        results = []
        
        for i, query in enumerate(queries):
            try:
                result = self.parse_query(query, query_id=f"batch_{i}")
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to parse query {i}: {e}")
                # Create minimal parsed query for failed cases
                results.append(ParsedQuery(
                    original_query=query,
                    query_type=QueryType.SIMPLE,
                    language='vietnamese',
                    scenes=[SceneComponent(0, query, [], [], [], "unknown", 0, 0.5)],
                    temporal_flow=[],
                    main_subject="unknown",
                    overall_context="general",
                    complexity_score=0.5,
                    parsing_confidence=0.3
                ))
        
        return results
    
    def export_parsed_queries(self, parsed_queries: List[ParsedQuery], output_path: str):
        """Export parsed queries to JSON file"""
        data = [asdict(pq) for pq in parsed_queries]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported {len(parsed_queries)} parsed queries to {output_path}")

# Utility functions for integration with existing system
def parse_single_query(query: str, config_path: Optional[str] = None) -> ParsedQuery:
    """Convenience function to parse a single query"""
    parser = EnhancedQueryParser(config_path)
    return parser.parse_query(query)

def is_complex_query(query: str, threshold: float = 0.6) -> bool:
    """Determine if a query is complex enough to need multi-scene parsing"""
    parsed = parse_single_query(query)
    return parsed.complexity_score >= threshold

def extract_scene_descriptions(parsed_query: ParsedQuery) -> List[str]:
    """Extract scene descriptions for individual searches"""
    return [scene.description for scene in parsed_query.scenes]

# Testing and validation functions
def test_parser_with_sample_queries():
    """Test parser with sample queries from the dataset"""
    
    sample_queries = [
        "Đoạn video mô tả cảnh trang trí bánh rán. Phân cảnh bắt đầu là một chiếc đĩa sứ màu trắng.",
        "Tìm một đoạn video đua xe đạp, góc quay từ flycam trên cao.",
        "Đoạn video trong buổi trao quà từ thiện diễn ra tại 1 bệnh viện trong dịp Xuân 2024."
    ]
    
    parser = EnhancedQueryParser()
    
    for i, query in enumerate(sample_queries):
        print(f"\n--- Testing Query {i+1} ---")
        print(f"Original: {query}")
        
        result = parser.parse_query(query)
        
        print(f"Type: {result.query_type.value}")
        print(f"Scenes: {len(result.scenes)}")
        print(f"Complexity: {result.complexity_score:.2f}")
        print(f"Main Subject: {result.main_subject}")
        print(f"Context: {result.overall_context}")
        
        for scene in result.scenes:
            print(f"  Scene {scene.scene_id}: {scene.description[:50]}...")
            print(f"    Objects: {scene.key_objects}")
            print(f"    Actions: {scene.actions}")

if __name__ == "__main__":
    # Run tests if executed directly
    test_parser_with_sample_queries()