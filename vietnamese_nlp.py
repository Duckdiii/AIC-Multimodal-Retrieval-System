"""
Vietnamese Natural Language Processing Module
===========================================

Advanced Vietnamese text processing for AIC Competition System v4.0
Focuses on temporal markers, cultural context, and Q&A query understanding.

Key Features:
- Vietnamese temporal marker detection
- Cultural context extraction  
- Location and entity recognition
- Q&A query processing
- OCR text post-processing

Author: AIC Competition System v4.0
Task: 1.3 - Vietnamese Language Processing
"""

import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemporalMarkerType(Enum):
    START = "start"
    CONTINUATION = "continuation" 
    SEQUENCE = "sequence"
    TRANSITION = "transition"
    CONCLUSION = "conclusion"
    DURATION = "duration"
    FREQUENCY = "frequency"

@dataclass
class TemporalMarker:
    """Vietnamese temporal marker with context"""
    text: str
    marker_type: TemporalMarkerType
    position: int
    confidence: float
    context: str = ""

@dataclass
class VietnameseEntity:
    """Vietnamese entity (location, person, object, etc.)"""
    text: str
    entity_type: str
    confidence: float
    cultural_context: Optional[str] = None

@dataclass
class VietnameseProcessingResult:
    """Complete Vietnamese text processing result"""
    original_text: str
    temporal_markers: List[TemporalMarker]
    entities: List[VietnameseEntity]
    cultural_indicators: List[str]
    temporal_sequence: List[str]
    complexity_score: float
    processing_confidence: float

class VietnameseTemporalProcessor:
    """Advanced Vietnamese temporal marker processing"""
    
    def __init__(self):
        # Comprehensive Vietnamese temporal markers
        self.temporal_patterns = {
            TemporalMarkerType.START: [
                # Beginning indicators
                r'\bbắt đầu\b', r'\bkhởi đầu\b', r'\blúc đầu\b', r'\bban đầu\b',
                r'\btrước tiên\b', r'\bđầu tiên\b', r'\btrước hết\b', r'\bđầu\b',
                r'\bmở đầu\b', r'\bkhai mạc\b', r'\bkhởi phát\b'
            ],
            TemporalMarkerType.CONTINUATION: [
                # Continuation indicators  
                r'\bsau đó\b', r'\btiếp theo\b', r'\bkế tiếp\b', r'\brồi\b', r'\bthì\b',
                r'\btiếp tục\b', r'\btừ đó\b', r'\bkhi đó\b', r'\blúc đó\b',
                r'\btiếp theo đó\b', r'\bsau khi\b', r'\bkhi\b'
            ],
            TemporalMarkerType.SEQUENCE: [
                # Sequential indicators
                r'\bbước đầu tiên\b', r'\bbước tiếp theo\b', r'\bbước cuối\b', r'\bbước sau\b',
                r'\blần lượt\b', r'\btheo thứ tự\b', r'\bthứ nhất\b', r'\bthứ hai\b', r'\bthứ ba\b',
                r'\bphân cảnh\b', r'\bcảnh\b', r'\bgiai đoạn\b', r'\bphần\b'
            ],
            TemporalMarkerType.TRANSITION: [
                # Transition indicators
                r'\btrong khi đó\b', r'\bcùng lúc\b', r'\bđồng thời\b', r'\btrong lúc\b',
                r'\btrong cảnh\b', r'\btại thời điểm\b', r'\bkhi mà\b', r'\bđang khi\b',
                r'\bin the meantime\b', r'\bmeanwhile\b'
            ],
            TemporalMarkerType.CONCLUSION: [
                # Ending indicators
                r'\bcuối cùng\b', r'\bkết thúc\b', r'\bhoàn thành\b', r'\bhoàn tất\b',
                r'\bđến đích\b', r'\bkết quả\b', r'\bsau cùng\b', r'\bchung cuộc\b',
                r'\bkết cục\b', r'\btối hậu\b', r'\bfinally\b'
            ],
            TemporalMarkerType.DURATION: [
                # Duration indicators
                r'\btrong suốt\b', r'\bsuốt\b', r'\bkéo dài\b', r'\bmất\b.*\b(phút|giờ|ngày)\b',
                r'\bdiễn ra\b.*\b(phút|giờ|ngày)\b', r'\btừ.*đến\b', r'\btừ.*cho đến\b'
            ],
            TemporalMarkerType.FREQUENCY: [
                # Frequency indicators  
                r'\blần\b', r'\blần lữợt\b', r'\bmỗi\b', r'\bhàng\b.*\b(ngày|tuần|tháng)\b',
                r'\bthường xuyên\b', r'\bliên tục\b', r'\bthỉnh thoảng\b'
            ]
        }
        
        # Vietnamese cultural context indicators
        self.cultural_markers = {
            'festivals': ['tết', 'xuân', 'trung thu', 'tết nguyên đán', 'lễ hội', 'kỷ niệm'],
            'food': ['bánh', 'phở', 'bún', 'cháo', 'cơm', 'nước mắm', 'chả cá', 'bánh mì'],
            'locations': ['hà nội', 'sài gòn', 'tp hồ chí minh', 'đà nẵng', 'huế', 'khánh hòa', 'nha trang'],
            'traditions': ['áo dài', 'nón lá', 'đình làng', 'chùa', 'đền', 'lăng'],
            'social': ['gia đình', 'cộng đồng', 'từ thiện', 'tình nguyện', 'xã hội'],
            'medical': ['bệnh viện', 'y tế', 'điều trị', 'chăm sóc', 'sức khỏe', 'bác sĩ']
        }
        
        # Location database (simplified - can be expanded)
        self.location_database = {
            'khánh hòa': {
                'type': 'province',
                'region': 'south_central',
                'famous_for': ['nha trang', 'cam ranh', 'beaches', 'seafood'],
                'cultural_significance': 'coastal tourism center'
            },
            'nha trang': {
                'type': 'city', 
                'province': 'khánh hòa',
                'famous_for': ['beaches', 'vinpearl', 'mud baths', 'seafood'],
                'cultural_significance': 'major tourist destination'
            },
            'hà nội': {
                'type': 'capital_city',
                'region': 'north',
                'famous_for': ['old quarter', 'temples', 'history', 'phở'],
                'cultural_significance': 'political and cultural center'
            },
            'hồ chí minh': {
                'type': 'major_city',
                'region': 'south', 
                'famous_for': ['business', 'motorbikes', 'food', 'nightlife'],
                'cultural_significance': 'economic center'
            }
        }
    
    def extract_temporal_markers(self, text: str) -> List[TemporalMarker]:
        """Extract Vietnamese temporal markers from text"""
        markers = []
        text_lower = text.lower()
        
        for marker_type, patterns in self.temporal_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text_lower):
                    # Extract context around the marker
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 20)
                    context = text[start:end].strip()
                    
                    marker = TemporalMarker(
                        text=match.group(),
                        marker_type=marker_type,
                        position=match.start(),
                        confidence=0.8,  # Base confidence
                        context=context
                    )
                    markers.append(marker)
        
        # Sort by position and remove duplicates
        markers = sorted(markers, key=lambda x: x.position)
        return self._deduplicate_markers(markers)
    
    def _deduplicate_markers(self, markers: List[TemporalMarker]) -> List[TemporalMarker]:
        """Remove overlapping or duplicate markers"""
        if not markers:
            return markers
            
        deduplicated = [markers[0]]
        
        for marker in markers[1:]:
            # Check if this marker overlaps with the last added one
            last_marker = deduplicated[-1]
            if marker.position > last_marker.position + len(last_marker.text):
                deduplicated.append(marker)
        
        return deduplicated
    
    def extract_temporal_sequence(self, text: str) -> List[str]:
        """Extract temporal sequence from Vietnamese text"""
        markers = self.extract_temporal_markers(text)
        
        # Group text by temporal markers
        sequence = []
        sentences = re.split(r'[.!?]\s+', text)
        
        for sentence in sentences:
            sentence_markers = [m for m in markers if m.text.lower() in sentence.lower()]
            
            if sentence_markers:
                # Sort by marker type priority
                marker_priority = {
                    TemporalMarkerType.START: 1,
                    TemporalMarkerType.SEQUENCE: 2, 
                    TemporalMarkerType.CONTINUATION: 3,
                    TemporalMarkerType.TRANSITION: 4,
                    TemporalMarkerType.CONCLUSION: 5
                }
                
                sentence_markers.sort(key=lambda x: marker_priority.get(x.marker_type, 6))
                primary_marker = sentence_markers[0]
                
                sequence.append({
                    'text': sentence.strip(),
                    'marker_type': primary_marker.marker_type.value,
                    'marker_text': primary_marker.text
                })
            else:
                # Sentence without explicit markers
                sequence.append({
                    'text': sentence.strip(),
                    'marker_type': 'implicit',
                    'marker_text': None
                })
        
        return sequence

class VietnameseEntityExtractor:
    """Extract Vietnamese entities and cultural context"""
    
    def __init__(self, location_db: Dict[str, Any]):
        self.location_db = location_db
        
        # Vietnamese entity patterns
        self.entity_patterns = {
            'person': [
                r'\b(anh|chị|ông|bà|cô|chú|bác)\s+\w+\b',
                r'\b(thầy|cô)\s+\w+\b',
                r'\b(bác sĩ|tiến sĩ|giáo sư)\s+\w+\b',
                r'\b\w+\s+(anh|chị)\b'
            ],
            'location': [
                r'\b(tại|ở|trong|tới|đến)\s+(\w+(?:\s+\w+)?)\b',
                r'\b(thành phố|tỉnh|quận|huyện|phường|xã)\s+\w+\b',
                r'\b(đường|phố|số)\s+\w+\b'
            ],
            'organization': [
                r'\b(bệnh viện|trường|công ty|tổ chức)\s+\w+\b',
                r'\b(ban|ủy ban|hội|đoàn)\s+\w+\b'
            ],
            'food': [
                r'\b(bánh|phở|bún|cháo|cơm)\s+\w*\b',
                r'\b(món|đặc sản)\s+\w+\b'
            ],
            'event': [
                r'\b(lễ hội|sự kiện|chương trình)\s+\w+\b',
                r'\b(dịp|nhân)\s+\w+\b'
            ]
        }
    
    def extract_entities(self, text: str) -> List[VietnameseEntity]:
        """Extract Vietnamese entities from text"""
        entities = []
        text_lower = text.lower()
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity_text = match.group().strip()
                    
                    # Get cultural context if it's a location
                    cultural_context = None
                    if entity_type == 'location':
                        cultural_context = self._get_location_context(entity_text)
                    
                    entity = VietnameseEntity(
                        text=entity_text,
                        entity_type=entity_type,
                        confidence=0.7,  # Base confidence
                        cultural_context=cultural_context
                    )
                    entities.append(entity)
        
        return self._deduplicate_entities(entities)
    
    def _deduplicate_entities(self, entities: List[VietnameseEntity]) -> List[VietnameseEntity]:
        """Remove duplicate entities"""
        seen = set()
        deduplicated = []
        
        for entity in entities:
            key = (entity.text.lower(), entity.entity_type)
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
        
        return deduplicated
    
    def _get_location_context(self, location_text: str) -> Optional[str]:
        """Get cultural context for a location"""
        location_lower = location_text.lower()
        
        for location_name, info in self.location_db.items():
            if location_name in location_lower or location_lower in location_name:
                return info.get('cultural_significance', 'Vietnamese location')
        
        return None

class VietnameseNLPProcessor:
    """Main Vietnamese NLP processing engine"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.temporal_processor = VietnameseTemporalProcessor()
        self.entity_extractor = VietnameseEntityExtractor(
            self.temporal_processor.location_database
        )
        
        # Load additional configuration if provided
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
    
    def process_text(self, text: str) -> VietnameseProcessingResult:
        """Complete Vietnamese text processing"""
        logger.info(f"Processing Vietnamese text: {text[:100]}...")
        
        # Step 1: Extract temporal markers
        temporal_markers = self.temporal_processor.extract_temporal_markers(text)
        logger.debug(f"Found {len(temporal_markers)} temporal markers")
        
        # Step 2: Extract temporal sequence
        temporal_sequence = self.temporal_processor.extract_temporal_sequence(text)
        logger.debug(f"Extracted temporal sequence with {len(temporal_sequence)} parts")
        
        # Step 3: Extract entities
        entities = self.entity_extractor.extract_entities(text)
        logger.debug(f"Found {len(entities)} entities")
        
        # Step 4: Identify cultural indicators
        cultural_indicators = self._extract_cultural_indicators(text)
        logger.debug(f"Found {len(cultural_indicators)} cultural indicators")
        
        # Step 5: Calculate complexity score
        complexity_score = self._calculate_complexity(
            text, temporal_markers, entities, temporal_sequence
        )
        
        # Step 6: Calculate processing confidence
        processing_confidence = self._calculate_processing_confidence(
            temporal_markers, entities, cultural_indicators
        )
        
        result = VietnameseProcessingResult(
            original_text=text,
            temporal_markers=temporal_markers,
            entities=entities,
            cultural_indicators=cultural_indicators,
            temporal_sequence=[part['text'] for part in temporal_sequence],
            complexity_score=complexity_score,
            processing_confidence=processing_confidence
        )
        
        logger.info(f"Vietnamese processing completed: complexity={complexity_score:.2f}, confidence={processing_confidence:.2f}")
        return result
    
    def _extract_cultural_indicators(self, text: str) -> List[str]:
        """Extract Vietnamese cultural context indicators"""
        indicators = []
        text_lower = text.lower()
        
        for category, markers in self.temporal_processor.cultural_markers.items():
            for marker in markers:
                if marker in text_lower:
                    indicators.append(f"{category}:{marker}")
        
        return indicators
    
    def _calculate_complexity(self, text: str, temporal_markers: List[TemporalMarker], 
                            entities: List[VietnameseEntity], temporal_sequence: List[str]) -> float:
        """Calculate text complexity score"""
        base_score = 0.0
        
        # Text length factor
        base_score += min(len(text) / 500, 0.3)
        
        # Temporal markers factor
        base_score += min(len(temporal_markers) * 0.1, 0.3)
        
        # Entity complexity factor
        base_score += min(len(entities) * 0.05, 0.2)
        
        # Sequence complexity factor
        base_score += min(len(temporal_sequence) * 0.02, 0.2)
        
        return min(base_score, 1.0)
    
    def _calculate_processing_confidence(self, temporal_markers: List[TemporalMarker],
                                       entities: List[VietnameseEntity], 
                                       cultural_indicators: List[str]) -> float:
        """Calculate processing confidence"""
        base_confidence = 0.5
        
        # Temporal markers boost confidence
        if temporal_markers:
            avg_temporal_confidence = sum(m.confidence for m in temporal_markers) / len(temporal_markers)
            base_confidence += avg_temporal_confidence * 0.2
        
        # Entity recognition boost confidence  
        if entities:
            avg_entity_confidence = sum(e.confidence for e in entities) / len(entities)
            base_confidence += avg_entity_confidence * 0.2
        
        # Cultural indicators boost confidence
        base_confidence += min(len(cultural_indicators) * 0.02, 0.1)
        
        return min(base_confidence, 1.0)
    
    def process_qa_query(self, query: str) -> Dict[str, Any]:
        """Specialized processing for Q&A queries"""
        result = self.process_text(query)
        
        # Q&A specific analysis
        qa_analysis = {
            'question_type': self._identify_question_type(query),
            'required_context': self._identify_required_context(result),
            'temporal_focus': self._identify_temporal_focus(result.temporal_markers),
            'cultural_requirements': self._identify_cultural_requirements(result.entities)
        }
        
        return {
            'processing_result': result,
            'qa_analysis': qa_analysis
        }
    
    def _identify_question_type(self, query: str) -> str:
        """Identify the type of Q&A question"""
        question_patterns = {
            'what': r'\b(gì|cái gì|là gì|cái nào)\b',
            'where': r'\b(đâu|ở đâu|tại đâu|chỗ nào)\b', 
            'when': r'\b(khi nào|lúc nào|bao giờ)\b',
            'who': r'\b(ai|người nào|ai đó)\b',
            'how': r'\b(làm sao|như thế nào|bằng cách nào)\b',
            'why': r'\b(tại sao|vì sao|lý do gì)\b'
        }
        
        query_lower = query.lower()
        for q_type, pattern in question_patterns.items():
            if re.search(pattern, query_lower):
                return q_type
        
        return 'general'
    
    def _identify_required_context(self, result: VietnameseProcessingResult) -> List[str]:
        """Identify what context is required to answer the question"""
        required_context = []
        
        # Based on entities
        for entity in result.entities:
            if entity.entity_type == 'location':
                required_context.append('geographic_context')
            elif entity.entity_type == 'person':
                required_context.append('person_context')
            elif entity.entity_type == 'event':
                required_context.append('event_context')
        
        # Based on temporal markers
        if result.temporal_markers:
            required_context.append('temporal_context')
        
        # Based on cultural indicators
        if result.cultural_indicators:
            required_context.append('cultural_context')
        
        return list(set(required_context))
    
    def _identify_temporal_focus(self, temporal_markers: List[TemporalMarker]) -> str:
        """Identify the temporal focus of the query"""
        if not temporal_markers:
            return 'none'
        
        marker_types = [m.marker_type for m in temporal_markers]
        
        if TemporalMarkerType.SEQUENCE in marker_types:
            return 'sequential'
        elif TemporalMarkerType.DURATION in marker_types:
            return 'duration'
        elif TemporalMarkerType.START in marker_types or TemporalMarkerType.CONCLUSION in marker_types:
            return 'boundary'
        else:
            return 'general_temporal'
    
    def _identify_cultural_requirements(self, entities: List[VietnameseEntity]) -> List[str]:
        """Identify cultural context requirements"""
        requirements = []
        
        for entity in entities:
            if entity.cultural_context:
                requirements.append(entity.cultural_context)
        
        return list(set(requirements))

# Utility functions for integration
def process_vietnamese_query(query: str, config_path: Optional[str] = None) -> VietnameseProcessingResult:
    """Convenience function to process a Vietnamese query"""
    processor = VietnameseNLPProcessor(config_path)
    return processor.process_text(query)

def extract_vietnamese_temporal_sequence(text: str) -> List[str]:
    """Extract temporal sequence from Vietnamese text"""
    processor = VietnameseNLPProcessor()
    result = processor.process_text(text)
    return result.temporal_sequence

def is_vietnamese_qa_query(query: str) -> bool:
    """Determine if a query is a Vietnamese Q&A query"""
    processor = VietnameseNLPProcessor()
    qa_result = processor.process_qa_query(query)
    
    # Q&A queries typically have question words and cultural context
    has_question_pattern = qa_result['qa_analysis']['question_type'] != 'general'
    has_cultural_context = len(qa_result['processing_result'].cultural_indicators) > 0
    
    return has_question_pattern or has_cultural_context

# Testing function
def test_vietnamese_nlp():
    """Test Vietnamese NLP processing with sample queries"""
    
    sample_texts = [
        "Đoạn video trong buổi trao quà từ thiện diễn ra tại 1 bệnh viện trong dịp Xuân 2024. Phân cảnh bắt đầu là hai người đàn ông đứng hai bên. Sau đó các em nhỏ được trao bảng tượng trưng.",
        "Tìm địa điểm nào ở Khánh Hòa có cảnh biển đẹp?",
        "Bánh rán được trang trí như thế nào? Bắt đầu là rưới chocolate, sau đó đặt các lát chuối lên trên."
    ]
    
    processor = VietnameseNLPProcessor()
    
    for i, text in enumerate(sample_texts):
        print(f"\n--- Test {i+1} ---")
        print(f"Input: {text}")
        
        result = processor.process_text(text)
        
        print(f"Temporal markers: {len(result.temporal_markers)}")
        for marker in result.temporal_markers[:3]:  # Show first 3
            print(f"  - {marker.text} ({marker.marker_type.value})")
        
        print(f"Entities: {len(result.entities)}")
        for entity in result.entities[:3]:  # Show first 3
            print(f"  - {entity.text} ({entity.entity_type})")
        
        print(f"Cultural indicators: {len(result.cultural_indicators)}")
        for indicator in result.cultural_indicators[:3]:  # Show first 3
            print(f"  - {indicator}")
        
        print(f"Complexity: {result.complexity_score:.2f}")
        print(f"Confidence: {result.processing_confidence:.2f}")

if __name__ == "__main__":
    test_vietnamese_nlp()