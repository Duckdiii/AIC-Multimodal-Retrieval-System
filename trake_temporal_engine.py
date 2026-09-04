"""
TRAKE Temporal Engine - Event Boundary Detection System
======================================================

Advanced temporal event detection for AIC Competition TRAKE queries.
Detects precise start/end moments of cooking, manufacturing, and performance events.

Key Features:
- Event boundary detection for cooking processes
- Manufacturing step identification  
- Performance/cultural event timing
- Sequential alignment and temporal ordering
- Frame-level precision for competition submission

Author: AIC Competition System v4.0
Task: 2.1 - TRAKE Temporal Engine Development
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

class EventType(Enum):
    COOKING_START = "cooking_start"
    COOKING_PROCESS = "cooking_process" 
    COOKING_END = "cooking_end"
    MANUFACTURING_START = "manufacturing_start"
    MANUFACTURING_PROCESS = "manufacturing_process"
    MANUFACTURING_END = "manufacturing_end"
    PERFORMANCE_START = "performance_start"
    PERFORMANCE_ACTION = "performance_action"
    PERFORMANCE_END = "performance_end"
    CONTACT_EVENT = "contact_event"
    TRANSITION_EVENT = "transition_event"

class EventBoundaryType(Enum):
    START_BOUNDARY = "start"
    END_BOUNDARY = "end"
    INSTANT_EVENT = "instant"
    DURATION_EVENT = "duration"

@dataclass
class EventBoundary:
    """Represents a temporal event boundary"""
    event_id: str
    event_type: EventType
    boundary_type: EventBoundaryType
    description: str
    temporal_markers: List[str]
    key_objects: List[str]
    actions: List[str]
    visual_cues: List[str]
    confidence: float
    
@dataclass
class TrakeEvent:
    """Complete TRAKE event specification"""
    event_id: str
    description: str
    start_boundary: Optional[EventBoundary]
    end_boundary: Optional[EventBoundary]
    duration_type: str  # "instant", "short", "extended"
    temporal_sequence: int
    dependencies: List[str]  # Event IDs this depends on
    visual_requirements: List[str]
    search_keywords: List[str]
    
@dataclass
class TrakeQuery:
    """Parsed TRAKE query with all events"""
    query_text: str
    domain: str  # "cooking", "manufacturing", "performance"
    events: List[TrakeEvent]
    temporal_relationships: List[Tuple[str, str, str]]  # (event1_id, relationship, event2_id)
    overall_context: str
    complexity_score: float

class CookingEventDetector:
    """Specialized detector for cooking events"""
    
    def __init__(self):
        # Cooking-specific patterns
        self.cooking_patterns = {
            # Start events
            'ingredient_addition': [
                r'bột được bỏ vào',
                r'thêm.*vào',
                r'cho.*vào', 
                r'đổ.*vào',
                r'bắt đầu cho'
            ],
            'cooking_start': [
                r'bắt đầu.*nấu',
                r'đặt lên bếp',
                r'mở lửa',
                r'lửa bắt đầu xuất hiện',
                r'chảo nóng'
            ],
            'preparation': [
                r'cắt.*đầu tiên',
                r'sơ chế',
                r'thái.*đầu tiên',
                r'bắt đầu cắt'
            ],
            
            # Process events
            'contact_with_oil': [
                r'tiếp xúc với dầu',
                r'cho vào dầu',
                r'thả vào chảo',
                r'rơi vào dầu'
            ],
            'cooking_process': [
                r'đang nấu',
                r'trong chảo',
                r'trên chảo',
                r'chiên.*trong'
            ],
            
            # End events
            'removal': [
                r'rời khỏi chảo',
                r'vớt ra',
                r'lấy ra khỏi',
                r'nằm.*trên dĩa',
                r'hoàn toàn trên dĩa'
            ],
            'completion': [
                r'hoàn thành',
                r'xong',
                r'cuối cùng.*trên',
                r'kết thúc'
            ]
        }
        
        # Key objects for cooking domain
        self.cooking_objects = [
            'bột', 'miến', 'măng tây', 'nấm', 'củ năng', 'đậu hủ',
            'chảo', 'tô', 'dĩa', 'dầu', 'lửa', 'bếp'
        ]
        
        # Actions for cooking domain
        self.cooking_actions = [
            'bỏ vào', 'cắt', 'thái', 'cho vào', 'đặt lên', 'mở lửa',
            'tiếp xúc', 'rời khỏi', 'nấu', 'chiên', 'vớt ra'
        ]

    def detect_cooking_events(self, event_description: str) -> EventBoundary:
        """Detect cooking event boundaries from description"""
        desc_lower = event_description.lower()
        
        # Determine event type
        event_type = EventType.COOKING_PROCESS  # Default
        boundary_type = EventBoundaryType.INSTANT_EVENT  # Default
        
        # Analyze for specific cooking patterns
        visual_cues = []
        detected_actions = []
        detected_objects = []
        temporal_markers = []
        
        # Extract temporal markers
        if 'khoảnh khắc đầu tiên' in desc_lower:
            temporal_markers.append('first_moment')
            boundary_type = EventBoundaryType.START_BOUNDARY
        elif 'khoảnh khắc cuối cùng' in desc_lower:
            temporal_markers.append('last_moment')
            boundary_type = EventBoundaryType.END_BOUNDARY
        elif 'khoảnh khắc' in desc_lower:
            temporal_markers.append('specific_moment')
            boundary_type = EventBoundaryType.INSTANT_EVENT
            
        # Detect cooking event type
        for pattern_type, patterns in self.cooking_patterns.items():
            for pattern in patterns:
                if re.search(pattern, desc_lower):
                    if 'start' in pattern_type or 'preparation' in pattern_type:
                        event_type = EventType.COOKING_START
                    elif 'end' in pattern_type or 'removal' in pattern_type or 'completion' in pattern_type:
                        event_type = EventType.COOKING_END
                    else:
                        event_type = EventType.COOKING_PROCESS
                    
                    visual_cues.append(f"detect_{pattern_type}")
                    break
        
        # Extract objects and actions
        for obj in self.cooking_objects:
            if obj in desc_lower:
                detected_objects.append(obj)
                
        for action in self.cooking_actions:
            if action in desc_lower:
                detected_actions.append(action)
        
        # Add specific visual cues based on description
        if 'dầu' in desc_lower:
            visual_cues.append('oil_presence')
        if 'chảo' in desc_lower:
            visual_cues.append('pan_visible')
        if 'lửa' in desc_lower:
            visual_cues.append('fire_visible')
        if 'dĩa' in desc_lower:
            visual_cues.append('plate_visible')
            
        # Calculate confidence
        confidence = 0.7  # Base confidence
        if temporal_markers:
            confidence += 0.1
        if detected_objects:
            confidence += 0.1
        if detected_actions:
            confidence += 0.1
            
        confidence = min(confidence, 1.0)
        
        return EventBoundary(
            event_id="",  # Will be set by caller
            event_type=event_type,
            boundary_type=boundary_type,
            description=event_description,
            temporal_markers=temporal_markers,
            key_objects=detected_objects,
            actions=detected_actions,
            visual_cues=visual_cues,
            confidence=confidence
        )

class PerformanceEventDetector:
    """Specialized detector for performance/cultural events"""
    
    def __init__(self):
        # Performance-specific patterns
        self.performance_patterns = {
            'movement_start': [
                r'bắt đầu.*xoay',
                r'bắt đầu.*quay',
                r'khởi động',
                r'cử động đầu tiên'
            ],
            'contact_ground': [
                r'chạm đất',
                r'tiếp đất', 
                r'chân.*đất',
                r'hoàn toàn.*đất'
            ],
            'greeting': [
                r'chào.*ban giám khảo',
                r'cúi chào',
                r'chào.*rồng',
                r'kết thúc.*chào'
            ],
            'spinning': [
                r'quay vòng',
                r'xoay vòng',
                r'vòng trên cột',
                r'quay.*chân trước'
            ]
        }
        
        # Performance objects
        self.performance_objects = [
            'lân', 'rồng', 'cột', 'chân', 'đầu', 'ban giám khảo', 'người biểu diễn'
        ]
        
        # Performance actions  
        self.performance_actions = [
            'quay vòng', 'xoay', 'chạm đất', 'tiếp đất', 'chào', 'cử động', 'tiến lại'
        ]

    def detect_performance_events(self, event_description: str) -> EventBoundary:
        """Detect performance event boundaries"""
        desc_lower = event_description.lower()
        
        event_type = EventType.PERFORMANCE_ACTION  # Default
        boundary_type = EventBoundaryType.INSTANT_EVENT
        
        visual_cues = []
        detected_actions = []
        detected_objects = []
        temporal_markers = []
        
        # Extract temporal markers
        if 'khoảnh khắc đầu tiên' in desc_lower:
            temporal_markers.append('first_moment')
            boundary_type = EventBoundaryType.START_BOUNDARY
        elif 'bắt đầu' in desc_lower:
            temporal_markers.append('start_action')
            event_type = EventType.PERFORMANCE_START
            boundary_type = EventBoundaryType.START_BOUNDARY
        elif 'cuối' in desc_lower or 'kết thúc' in desc_lower:
            temporal_markers.append('end_action')
            event_type = EventType.PERFORMANCE_END
            boundary_type = EventBoundaryType.END_BOUNDARY
            
        # Detect specific performance patterns
        for pattern_type, patterns in self.performance_patterns.items():
            for pattern in patterns:
                if re.search(pattern, desc_lower):
                    visual_cues.append(f"detect_{pattern_type}")
                    break
        
        # Extract objects and actions
        for obj in self.performance_objects:
            if obj in desc_lower:
                detected_objects.append(obj)
                
        for action in self.performance_actions:
            if action in desc_lower:
                detected_actions.append(action)
        
        # Add specific visual cues
        if 'lân' in desc_lower:
            visual_cues.append('lion_dance_visible')
        if 'rồng' in desc_lower:
            visual_cues.append('dragon_visible')
        if 'cột' in desc_lower:
            visual_cues.append('pole_visible')
        if 'ban giám khảo' in desc_lower:
            visual_cues.append('judges_visible')
            
        confidence = 0.8  # Higher base confidence for performance events
        if temporal_markers:
            confidence += 0.1
        if detected_objects:
            confidence += 0.05
        if detected_actions:
            confidence += 0.05
            
        confidence = min(confidence, 1.0)
        
        return EventBoundary(
            event_id="",  # Will be set by caller
            event_type=event_type,
            boundary_type=boundary_type,
            description=event_description,
            temporal_markers=temporal_markers,
            key_objects=detected_objects,
            actions=detected_actions,
            visual_cues=visual_cues,
            confidence=confidence
        )

class TrakeQueryParser:
    """Main TRAKE query parser and event detector"""
    
    def __init__(self):
        self.cooking_detector = CookingEventDetector()
        self.performance_detector = PerformanceEventDetector()
        
        # Domain detection patterns
        self.domain_patterns = {
            'cooking': [
                r'nấu ăn', r'món ăn', r'đầu bếp', r'chảo', r'bếp', r'dầu',
                r'cắt.*nấm', r'cắt.*củ', r'sơ chế', r'chiên'
            ],
            'performance': [
                r'múa lân', r'biểu diễn', r'lân.*màu', r'quay vòng',
                r'ban giám khảo', r'rồng', r'chào'
            ],
            'manufacturing': [
                r'sản xuất', r'gia công', r'lắp ráp', r'chế tạo',
                r'máy móc', r'công nghệ'
            ]
        }

    def parse_trake_query(self, query_text: str) -> TrakeQuery:
        """Parse a complete TRAKE query"""
        logger.info(f"Parsing TRAKE query: {query_text[:100]}...")
        
        # Detect domain
        domain = self._detect_domain(query_text)
        logger.debug(f"Detected domain: {domain}")
        
        # Extract individual events
        events = self._extract_events(query_text, domain)
        logger.debug(f"Extracted {len(events)} events")
        
        # Analyze temporal relationships
        temporal_relationships = self._extract_temporal_relationships(events)
        
        # Extract overall context
        overall_context = self._extract_overall_context(query_text, domain)
        
        # Calculate complexity
        complexity_score = self._calculate_complexity(events, temporal_relationships)
        
        trake_query = TrakeQuery(
            query_text=query_text,
            domain=domain,
            events=events,
            temporal_relationships=temporal_relationships,
            overall_context=overall_context,
            complexity_score=complexity_score
        )
        
        logger.info(f"TRAKE query parsed: domain={domain}, events={len(events)}, complexity={complexity_score:.2f}")
        return trake_query
    
    def _detect_domain(self, query_text: str) -> str:
        """Detect the domain of the TRAKE query"""
        query_lower = query_text.lower()
        
        domain_scores = {}
        for domain, patterns in self.domain_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            domain_scores[domain] = score
        
        # Return domain with highest score, default to cooking
        best_domain = max(domain_scores, key=domain_scores.get)
        return best_domain if domain_scores[best_domain] > 0 else 'cooking'
    
    def _extract_events(self, query_text: str, domain: str) -> List[TrakeEvent]:
        """Extract individual events from TRAKE query"""
        events = []
        
        # Split query into event lines
        lines = query_text.split('\n')
        event_pattern = r'^E\d+:\s*(.+)$'
        
        for line in lines:
            line = line.strip()
            match = re.match(event_pattern, line)
            if match:
                event_description = match.group(1)
                event_id = re.match(r'^(E\d+)', line).group(1)
                
                # Detect event boundary based on domain
                if domain == 'cooking':
                    boundary = self.cooking_detector.detect_cooking_events(event_description)
                elif domain == 'performance':
                    boundary = self.performance_detector.detect_performance_events(event_description)
                else:
                    # Default/generic event detection
                    boundary = self._detect_generic_event(event_description)
                
                boundary.event_id = event_id
                
                # Create TRAKE event
                trake_event = TrakeEvent(
                    event_id=event_id,
                    description=event_description,
                    start_boundary=boundary if boundary.boundary_type == EventBoundaryType.START_BOUNDARY else None,
                    end_boundary=boundary if boundary.boundary_type == EventBoundaryType.END_BOUNDARY else None,
                    duration_type=self._determine_duration_type(boundary),
                    temporal_sequence=len(events) + 1,
                    dependencies=self._extract_dependencies(event_description, events),
                    visual_requirements=boundary.visual_cues,
                    search_keywords=self._generate_search_keywords(boundary)
                )
                
                # For instant events, set both boundaries to the same
                if boundary.boundary_type == EventBoundaryType.INSTANT_EVENT:
                    trake_event.start_boundary = boundary
                    trake_event.end_boundary = boundary
                
                events.append(trake_event)
        
        return events
    
    def _detect_generic_event(self, description: str) -> EventBoundary:
        """Generic event detection for unknown domains"""
        desc_lower = description.lower()
        
        boundary_type = EventBoundaryType.INSTANT_EVENT
        event_type = EventType.TRANSITION_EVENT
        
        temporal_markers = []
        if 'khoảnh khắc đầu tiên' in desc_lower:
            temporal_markers.append('first_moment')
            boundary_type = EventBoundaryType.START_BOUNDARY
        elif 'khoảnh khắc cuối' in desc_lower:
            temporal_markers.append('last_moment') 
            boundary_type = EventBoundaryType.END_BOUNDARY
        elif 'khoảnh khắc' in desc_lower:
            temporal_markers.append('specific_moment')
        
        return EventBoundary(
            event_id="",
            event_type=event_type,
            boundary_type=boundary_type,
            description=description,
            temporal_markers=temporal_markers,
            key_objects=[],
            actions=[],
            visual_cues=[],
            confidence=0.6
        )
    
    def _determine_duration_type(self, boundary: EventBoundary) -> str:
        """Determine if event is instant, short, or extended duration"""
        if boundary.boundary_type == EventBoundaryType.INSTANT_EVENT:
            return "instant"
        elif 'đầu tiên' in boundary.description or 'cuối cùng' in boundary.description:
            return "instant"  # Specific moments are instant
        elif any(action in boundary.description.lower() for action in ['cắt', 'cho vào', 'chạm đất']):
            return "short"  # Quick actions
        else:
            return "extended"  # Process events
    
    def _extract_dependencies(self, description: str, existing_events: List[TrakeEvent]) -> List[str]:
        """Extract event dependencies"""
        dependencies = []
        desc_lower = description.lower()
        
        # Look for sequential indicators
        if 'sau đó' in desc_lower or 'tiếp theo' in desc_lower:
            # This event depends on previous events
            if existing_events:
                dependencies.append(existing_events[-1].event_id)
        
        # Look for explicit references to other events
        for event in existing_events:
            event_objects = event.start_boundary.key_objects if event.start_boundary else []
            for obj in event_objects:
                if obj in desc_lower:
                    dependencies.append(event.event_id)
                    break
        
        return dependencies
    
    def _generate_search_keywords(self, boundary: EventBoundary) -> List[str]:
        """Generate keywords for video search"""
        keywords = []
        
        # Add objects as keywords
        keywords.extend(boundary.key_objects)
        
        # Add actions as keywords  
        keywords.extend(boundary.actions)
        
        # Add visual cues as keywords
        for cue in boundary.visual_cues:
            if 'detect_' in cue:
                keywords.append(cue.replace('detect_', ''))
        
        # Add temporal context
        if 'first_moment' in boundary.temporal_markers:
            keywords.append('beginning')
        if 'last_moment' in boundary.temporal_markers:
            keywords.append('ending')
            
        return list(set(keywords))  # Remove duplicates
    
    def _extract_temporal_relationships(self, events: List[TrakeEvent]) -> List[Tuple[str, str, str]]:
        """Extract temporal relationships between events"""
        relationships = []
        
        # Sequential relationships based on event order
        for i in range(len(events) - 1):
            current_event = events[i]
            next_event = events[i + 1]
            relationships.append((current_event.event_id, "before", next_event.event_id))
        
        # Dependency relationships
        for event in events:
            for dep_id in event.dependencies:
                relationships.append((dep_id, "prerequisite_for", event.event_id))
        
        return relationships
    
    def _extract_overall_context(self, query_text: str, domain: str) -> str:
        """Extract overall context of the TRAKE query"""
        query_lower = query_text.lower()
        
        context_parts = []
        
        # Add domain context
        context_parts.append(f"{domain}_sequence")
        
        # Add specific context based on content
        if 'nấu ăn' in query_lower:
            context_parts.append('cooking_process')
        if 'múa lân' in query_lower:
            context_parts.append('lion_dance_performance')
        if 'sơ chế' in query_lower:
            context_parts.append('food_preparation')
        
        return "_".join(context_parts)
    
    def _calculate_complexity(self, events: List[TrakeEvent], 
                            temporal_relationships: List[Tuple[str, str, str]]) -> float:
        """Calculate query complexity"""
        base_score = 0.0
        
        # Event count factor
        base_score += min(len(events) * 0.15, 0.6)  # Max 0.6 for many events
        
        # Temporal relationship complexity
        base_score += min(len(temporal_relationships) * 0.1, 0.3)  # Max 0.3 for relationships
        
        # Event type diversity
        event_types = set()
        for event in events:
            if event.start_boundary:
                event_types.add(event.start_boundary.event_type)
        base_score += min(len(event_types) * 0.05, 0.1)  # Max 0.1 for diversity
        
        return min(base_score, 1.0)

# Utility functions
def parse_trake_query(query_text: str) -> TrakeQuery:
    """Convenience function to parse a TRAKE query"""
    parser = TrakeQueryParser()
    return parser.parse_trake_query(query_text)

def extract_event_search_keywords(trake_query: TrakeQuery) -> Dict[str, List[str]]:
    """Extract search keywords for each event"""
    keywords_by_event = {}
    
    for event in trake_query.events:
        keywords_by_event[event.event_id] = event.search_keywords
    
    return keywords_by_event

# Testing function
def test_trake_parser():
    """Test TRAKE parser with sample queries"""
    
    # Sample TRAKE queries
    sample_queries = [
        """Đoạn video nấu ăn một món ăn về nấm, gồm các khoảnh khắc sơ chế:
E1: Khoảnh khắc đầu tiên thấy cắt nấm.
E2: Khoảnh khắc đầu tiên cắt củ năng.
E3: Khoảnh khắc đầu tiên cắt đậu hủ.
E4: Khoảnh khắc chảo đặt lên bếp, đầu bếp mở lửa và thấy lửa bắt đầu xuất hiện.""",

        """E1: Khoảnh khắc đầu tiên bột được bỏ vào tô măng tây.
E2: Khoảnh khắc đầu tiên thấy miến măng tây đầu tiên tiếp xúc với dầu trong chảo.
E3: Khoảnh khắc miếng măng tây đầu tiên rời khỏi chảo dầu.
E4: Khoảng khắc miếng măng tây cuối cùng rời chảo dầu và nằm hoàn toàn trên dĩa.""",

        """Đoạn video múa lân một con lân màu vàng đen trắng, tìm các sự kiện sau:
E1: Lân quay vòng trên cột số 4 bằng 2 chân trước rồi tiếp đất. Khoảnh khắc đầu tiên mà lân bắt đầu xoay vòng.
E2: Khoảnh khắc 4 chân hoàn toàn chạm đất đầu tiên.
E3: Khoảnh khắc đầu tiên 2 người biểu diễn lân cuối chào ban giám khảo.
E4: Sau đó lân tiến lại chào một con rồng. Khoảnh khắc đầu tiên con rồng cử động đầu."""
    ]
    
    parser = TrakeQueryParser()
    
    for i, query in enumerate(sample_queries):
        print(f"\n--- TRAKE Query {i+1} ---")
        print(f"Query length: {len(query)} characters")
        
        result = parser.parse_trake_query(query)
        
        print(f"Domain: {result.domain}")
        print(f"Events: {len(result.events)}")
        print(f"Temporal relationships: {len(result.temporal_relationships)}")
        print(f"Complexity: {result.complexity_score:.2f}")
        print(f"Context: {result.overall_context}")
        
        print("Events detected:")
        for event in result.events:
            boundary = event.start_boundary or event.end_boundary
            print(f"  {event.event_id}: {boundary.event_type.value} ({boundary.boundary_type.value})")
            print(f"    Objects: {boundary.key_objects}")
            print(f"    Actions: {boundary.actions}")
            print(f"    Visual cues: {boundary.visual_cues}")
            print(f"    Keywords: {event.search_keywords}")

if __name__ == "__main__":
    test_trake_parser()